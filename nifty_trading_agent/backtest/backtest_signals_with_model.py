#!/usr/bin/env python3
"""
Backtesting Script for Nifty Trading Agent with ML Model
Simulates historical trading using ML model predictions
"""

import sys
import os
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.logging_utils import setup_logging, get_logger
from utils.duckdb_tools import load_ohlcv, store_backtest_results
from utils.io_utils import load_yaml_config

logger = get_logger(__name__)

@dataclass
class Trade:
    """Represents a completed trade"""
    symbol: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    pnl_pct: float
    holding_days: int
    conviction: float
    outcome: str  # 'WIN', 'LOSS', 'STOP_LOSS', 'TARGET'

@dataclass
class Position:
    """Represents an open position"""
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    quantity: int
    stop_loss: float
    target_price: float
    conviction: float

class MLBacktester:
    """
    Backtests ML-based trading signals with realistic portfolio simulation
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize backtester

        Args:
            config_path: Path to configuration file
        """
        self.config = load_yaml_config(config_path)
        self.model_params = self.config.get('model_params', {})
        self.risk_settings = self.config.get('risk_settings', {})

        # Backtest parameters
        self.initial_capital = self.config.get('backtest_settings', {}).get('initial_capital', 1000000)
        self.conviction_threshold = self.model_params.get('conviction_threshold', 0.8)
        self.max_positions = self.risk_settings.get('max_open_positions', 5)
        self.stop_loss_pct = self.risk_settings.get('stop_loss_atr_multiplier', 1.5) * 0.05  # Rough stop loss

        # Load model
        self.model, self.metadata = self._load_model()

        # Portfolio state
        self.capital = self.initial_capital
        self.positions: Dict[str, Position] = {}
        self.completed_trades: List[Trade] = []
        self.equity_curve: List[Dict[str, Any]] = []

    def _load_model(self) -> Tuple[Any, Dict[str, Any]]:
        """Load the latest trained model"""
        model_dir = Path(self.model_params.get('model_save_path', 'models/artifacts/'))

        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory {model_dir} not found")

        model_files = list(model_dir.glob("model_*.pkl"))
        if not model_files:
            raise FileNotFoundError(f"No model files found in {model_dir}")

        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)

        logger.info(f"Loading model from {latest_model}")

        with open(latest_model, 'rb') as f:
            model_data = pickle.load(f)

        return model_data['model'], model_data['metadata']

    def run_backtest(self, start_date: str = "2024-01-01",
                    end_date: str = "2024-12-31") -> Dict[str, Any]:
        """
        Run the complete backtest simulation

        Args:
            start_date: Backtest start date
            end_date: Backtest end date

        Returns:
            Backtest results dictionary
        """
        logger.info(f"Starting ML-based backtest from {start_date} to {end_date}")

        # Convert dates
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        # Get all trading days in the period
        trading_days = pd.date_range(start=start_dt, end=end_dt, freq='B')  # Business days

        logger.info(f"Simulating {len(trading_days)} trading days")

        # Simulate each trading day
        for current_date in trading_days:
            try:
                self._process_trading_day(current_date)
            except Exception as e:
                logger.warning(f"Error processing {current_date}: {e}")
                continue

        # Final portfolio liquidation (if any open positions)
        self._close_all_positions(end_dt)

        # Calculate final results
        results = self._calculate_backtest_results()

        logger.info("Backtest simulation completed")
        return results

    def _process_trading_day(self, current_date: pd.Timestamp):
        """
        Process one trading day of the backtest

        Args:
            current_date: Current trading date
        """
        # 1. Check for position exits (stop losses, targets)
        self._check_position_exits(current_date)

        # 2. Generate signals for the day (if we have capacity)
        if len(self.positions) < self.max_positions:
            self._generate_signals_for_date(current_date)

        # 3. Record equity for the day
        self._record_equity(current_date)

    def _check_position_exits(self, current_date: pd.Timestamp):
        """
        Check if any positions need to be closed

        Args:
            current_date: Current date
        """
        positions_to_close = []

        for symbol, position in self.positions.items():
            # Get current price for the symbol
            try:
                ohlcv_data = load_ohlcv([symbol], current_date.strftime('%Y-%m-%d'),
                                       current_date.strftime('%Y-%m-%d'))

                if ohlcv_data.empty:
                    continue

                current_price = ohlcv_data['close'].iloc[0]

                # Check stop loss
                if current_price <= position.stop_loss:
                    self._close_position(symbol, current_date, current_price, 'STOP_LOSS')
                    positions_to_close.append(symbol)
                    continue

                # Check target
                if current_price >= position.target_price:
                    self._close_position(symbol, current_date, current_price, 'TARGET')
                    positions_to_close.append(symbol)
                    continue

            except Exception as e:
                logger.debug(f"Error checking position exit for {symbol}: {e}")
                continue

        # Remove closed positions
        for symbol in positions_to_close:
            if symbol in self.positions:
                del self.positions[symbol]

    def _generate_signals_for_date(self, current_date: pd.Timestamp):
        """
        Generate trading signals for the given date

        Args:
            current_date: Date to generate signals for
        """
        # Get universe symbols
        universe = self.config.get('universe', {}).get('tickers', [])

        # Limit to first 10 for demo (avoid rate limits)
        symbols_to_check = universe[:10]

        for symbol in symbols_to_check:
            try:
                # Build feature vector for this symbol and date
                features = self._build_features_for_symbol(symbol, current_date)

                if features is None:
                    continue

                # Get model prediction
                features_df = pd.DataFrame([features])
                prediction_prob = self.model.predict_proba(features_df)[0][1]

                # Apply conviction threshold
                if prediction_prob >= self.conviction_threshold:
                    # Create position
                    self._open_position(symbol, current_date, prediction_prob, features)

            except Exception as e:
                logger.debug(f"Error generating signal for {symbol}: {e}")
                continue

    def _build_features_for_symbol(self, symbol: str, date: pd.Timestamp) -> Optional[Dict[str, float]]:
        """
        Build feature vector for a symbol on a specific date

        Args:
            symbol: Stock symbol
            date: Date for features

        Returns:
            Feature dictionary or None if insufficient data
        """
        # This is a simplified version - in practice, you'd use the full feature engineering pipeline
        # For now, we'll use basic price-based features

        try:
            # Get historical data (last 60 days)
            start_date = date - timedelta(days=60)
            ohlcv_data = load_ohlcv([symbol], start_date.strftime('%Y-%m-%d'),
                                   date.strftime('%Y-%m-%d'))

            if len(ohlcv_data) < 30:  # Need minimum history
                return None

            # Calculate basic technical features
            close_prices = ohlcv_data['close'].values

            # Simple moving averages
            features = {}
            features['r_1d'] = (close_prices[-1] - close_prices[-2]) / close_prices[-2] if len(close_prices) > 1 else 0
            features['r_5d'] = (close_prices[-1] - close_prices[-6]) / close_prices[-6] if len(close_prices) > 5 else 0
            features['r_10d'] = (close_prices[-1] - close_prices[-11]) / close_prices[-11] if len(close_prices) > 10 else 0
            features['r_20d'] = (close_prices[-1] - close_prices[-21]) / close_prices[-21] if len(close_prices) > 20 else 0

            # Moving averages
            features['ma_5'] = close_prices[-5:].mean() if len(close_prices) >= 5 else close_prices.mean()
            features['ma_10'] = close_prices[-10:].mean() if len(close_prices) >= 10 else close_prices.mean()
            features['ma_20'] = close_prices[-20:].mean() if len(close_prices) >= 20 else close_prices.mean()
            features['ma_50'] = close_prices[-50:].mean() if len(close_prices) >= 50 else close_prices.mean()

            # RSI approximation
            features['rsi_14'] = self._calculate_rsi(close_prices, 14)

            # MACD approximation
            features['macd'] = features['ma_5'] - features['ma_10']

            # Bollinger Band width
            std_20 = close_prices[-20:].std() if len(close_prices) >= 20 else close_prices.std()
            features['bb_width'] = (2 * std_20) / features['ma_20'] if features['ma_20'] != 0 else 0

            # Fill other features with defaults (simplified)
            features['atr_14'] = std_20 * 0.1  # Rough approximation
            features['vol_zscore_20'] = 0.0
            features['quarterly_eps_growth'] = 0.0
            features['quarterly_rev_growth'] = 0.0
            features['profit_margin'] = 0.0
            features['pe_ratio'] = 15.0  # Default
            features['pb_ratio'] = 2.0   # Default
            features['sentiment_short'] = 0.0
            features['sentiment_medium'] = 0.0
            features['recent_earnings_flag'] = 0.0
            features['volume_gap_flag'] = 0.0

            return features

        except Exception as e:
            logger.debug(f"Error building features for {symbol}: {e}")
            return None

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50.0  # Neutral

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _open_position(self, symbol: str, date: pd.Timestamp,
                      conviction: float, features: Dict[str, float]):
        """
        Open a new position

        Args:
            symbol: Stock symbol
            date: Entry date
            conviction: Model conviction score
            features: Feature dictionary (contains current price)
        """
        # Get current price (approximated from features)
        current_price = features.get('ma_5', 100)  # Rough approximation

        # Calculate position size (5% of capital)
        position_value = self.capital * 0.05
        quantity = int(position_value / current_price)

        if quantity <= 0:
            return

        # Calculate stop loss and target
        stop_loss = current_price * (1 - self.stop_loss_pct)
        target_price = current_price * 1.10  # +10% target

        # Create position
        position = Position(
            symbol=symbol,
            entry_date=date,
            entry_price=current_price,
            quantity=quantity,
            stop_loss=stop_loss,
            target_price=target_price,
            conviction=conviction
        )

        self.positions[symbol] = position

        # Update capital
        self.capital -= (quantity * current_price)

        logger.info(f"Opened position: {symbol} x{quantity} @ ₹{current_price:.2f}")

    def _close_position(self, symbol: str, date: pd.Timestamp,
                       exit_price: float, outcome: str):
        """
        Close an existing position

        Args:
            symbol: Stock symbol
            date: Exit date
            exit_price: Exit price
            outcome: Trade outcome
        """
        if symbol not in self.positions:
            return

        position = self.positions[symbol]

        # Calculate P&L
        pnl = (exit_price - position.entry_price) * position.quantity
        pnl_pct = (exit_price - position.entry_price) / position.entry_price

        # Create trade record
        trade = Trade(
            symbol=symbol,
            entry_date=position.entry_date,
            exit_date=date,
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=position.quantity,
            pnl=pnl,
            pnl_pct=pnl_pct,
            holding_days=(date - position.entry_date).days,
            conviction=position.conviction,
            outcome=outcome
        )

        self.completed_trades.append(trade)

        # Update capital
        self.capital += (position.quantity * exit_price)

        logger.info(f"Closed position: {symbol} P&L: ₹{pnl:.2f} ({pnl_pct:.1%}) - {outcome}")

    def _close_all_positions(self, date: pd.Timestamp):
        """Close all remaining positions at the end of backtest"""
        for symbol in list(self.positions.keys()):
            try:
                # Get closing price
                ohlcv_data = load_ohlcv([symbol], date.strftime('%Y-%m-%d'),
                                       date.strftime('%Y-%m-%d'))

                if not ohlcv_data.empty:
                    close_price = ohlcv_data['close'].iloc[0]
                    self._close_position(symbol, date, close_price, 'END_OF_PERIOD')
                else:
                    # Use entry price if no close data
                    position = self.positions[symbol]
                    self._close_position(symbol, date, position.entry_price, 'END_OF_PERIOD')

            except Exception as e:
                logger.warning(f"Error closing position {symbol}: {e}")
                continue

    def _record_equity(self, date: pd.Timestamp):
        """Record equity curve point"""
        # Calculate current portfolio value
        portfolio_value = self.capital

        # Add value of open positions (marked to market)
        for symbol, position in self.positions.items():
            try:
                # Get current price
                ohlcv_data = load_ohlcv([symbol], date.strftime('%Y-%m-%d'),
                                       date.strftime('%Y-%m-%d'))

                if not ohlcv_data.empty:
                    current_price = ohlcv_data['close'].iloc[0]
                    portfolio_value += (position.quantity * current_price)

            except Exception as e:
                # Keep entry price if error
                portfolio_value += (position.quantity * position.entry_price)

        equity_point = {
            'date': date,
            'equity': portfolio_value,
            'capital': self.capital,
            'positions': len(self.positions),
            'trades': len(self.completed_trades)
        }

        self.equity_curve.append(equity_point)

    def _calculate_backtest_results(self) -> Dict[str, Any]:
        """Calculate comprehensive backtest results"""
        if not self.equity_curve:
            return {'error': 'No equity data'}

        # Basic metrics
        final_equity = self.equity_curve[-1]['equity']
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        total_trades = len(self.completed_trades)

        # Win rate
        winning_trades = [t for t in self.completed_trades if t.pnl > 0]
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0

        # Average P&L
        avg_pnl = np.mean([t.pnl for t in self.completed_trades]) if self.completed_trades else 0
        avg_pnl_pct = np.mean([t.pnl_pct for t in self.completed_trades]) if self.completed_trades else 0

        # Max drawdown
        equity_values = [point['equity'] for point in self.equity_curve]
        peak = equity_values[0]
        max_drawdown = 0

        for equity in equity_values:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak
            max_drawdown = max(max_drawdown, drawdown)

        # Sharpe ratio (simplified, using daily returns)
        if len(self.equity_curve) > 1:
            daily_returns = []
            for i in range(1, len(self.equity_curve)):
                ret = (self.equity_curve[i]['equity'] - self.equity_curve[i-1]['equity']) / self.equity_curve[i-1]['equity']
                daily_returns.append(ret)

            if daily_returns:
                avg_daily_return = np.mean(daily_returns)
                std_daily_return = np.std(daily_returns)
                sharpe_ratio = avg_daily_return / std_daily_return * np.sqrt(252) if std_daily_return > 0 else 0
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0

        # Conviction analysis
        conviction_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        conviction_analysis = {}

        for threshold in conviction_thresholds:
            threshold_trades = [t for t in self.completed_trades if t.conviction >= threshold]
            if threshold_trades:
                threshold_win_rate = len([t for t in threshold_trades if t.pnl > 0]) / len(threshold_trades)
                threshold_avg_return = np.mean([t.pnl_pct for t in threshold_trades])
                conviction_analysis[f'{threshold:.1f}'] = {
                    'trades': len(threshold_trades),
                    'win_rate': threshold_win_rate,
                    'avg_return_pct': threshold_avg_return
                }

        results = {
            'summary': {
                'initial_capital': self.initial_capital,
                'final_equity': final_equity,
                'total_return_pct': total_return * 100,
                'total_trades': total_trades,
                'win_rate_pct': win_rate * 100,
                'avg_trade_pnl': avg_pnl,
                'avg_trade_pnl_pct': avg_pnl_pct * 100,
                'max_drawdown_pct': max_drawdown * 100,
                'sharpe_ratio': sharpe_ratio
            },
            'equity_curve': self.equity_curve,
            'trades': [
                {
                    'symbol': t.symbol,
                    'entry_date': t.entry_date.isoformat(),
                    'exit_date': t.exit_date.isoformat(),
                    'entry_price': t.entry_price,
                    'exit_price': t.exit_price,
                    'quantity': t.quantity,
                    'pnl': t.pnl,
                    'pnl_pct': t.pnl_pct,
                    'holding_days': t.holding_days,
                    'conviction': t.conviction,
                    'outcome': t.outcome
                }
                for t in self.completed_trades
            ],
            'conviction_analysis': conviction_analysis,
            'model_info': {
                'version': self.metadata.get('model_version', 'unknown'),
                'features_used': len(self.metadata.get('feature_names', []))
            }
        }

        return results

    def save_results(self, results: Dict[str, Any], run_id: str = None) -> str:
        """
        Save backtest results to files and database

        Args:
            results: Backtest results
            run_id: Optional run identifier

        Returns:
            Path to saved report
        """
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save equity curve to DuckDB
        if 'equity_curve' in results:
            equity_df = pd.DataFrame(results['equity_curve'])
            equity_df['strategy_id'] = 'ml_model_signals'
            equity_df['run_id'] = run_id

            try:
                store_backtest_results('ml_model_signals', run_id, equity_df)
                logger.info("Equity curve saved to DuckDB")
            except Exception as e:
                logger.warning(f"Failed to save equity curve to DuckDB: {e}")

        # Save detailed results to JSON
        results_file = f"reports/backtest_ml_signals_{run_id}.json"

        # Convert numpy types and pandas objects to native Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, 'isoformat'):  # pandas Timestamp, datetime
                return obj.isoformat()
            return obj

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=convert_types)

        logger.info(f"Backtest results saved to {results_file}")
        return results_file

def print_backtest_summary(results: Dict[str, Any]):
    """Print backtest summary to console"""
    print("\n" + "="*80)
    print("📊 ML-BASED BACKTEST RESULTS")
    print("="*80)

    summary = results.get('summary', {})

    print("\n💰 Performance Summary:")
    print(f"   Initial Capital: ₹{summary.get('initial_capital', 0):,.0f}")
    print(f"   Final Equity: ₹{summary.get('final_equity', 0):,.0f}")
    print(f"   Total Return: {summary.get('total_return_pct', 0):+.1f}%")
    print(f"   Sharpe Ratio: {summary.get('sharpe_ratio', 0):.2f}")
    print(f"   Max Drawdown: {summary.get('max_drawdown_pct', 0):.1f}%")

    print("\n📈 Trading Statistics:")
    print(f"   Total Trades: {summary.get('total_trades', 0)}")
    print(f"   Win Rate: {summary.get('win_rate_pct', 0):.1f}%")
    print(f"   Avg Trade P&L: ₹{summary.get('avg_trade_pnl', 0):,.0f}")
    print(f"   Avg Trade Return: {summary.get('avg_trade_pnl_pct', 0):+.1f}%")

    # Conviction analysis
    conviction_analysis = results.get('conviction_analysis', {})
    if conviction_analysis:
        print("\n🎯 Performance by Conviction Threshold:")
        print("   Threshold | Trades | Win Rate | Avg Return")
        print("   ----------|--------|----------|------------")

        for threshold, metrics in conviction_analysis.items():
            print(f"   {threshold:>9} | {metrics['trades']:>6} | {metrics['win_rate']:.1%} | {metrics['avg_return_pct']:+.1f}%")

def main():
    """Main backtest function"""
    print("🚀 NIFTY TRADING AGENT - ML SIGNAL BACKTEST")
    print("=" * 50)

    # Setup logging
    setup_logging({
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file_path': 'logs/backtest_ml.log',
        'max_file_size_mb': 50,
        'backup_count': 3
    })

    try:
        # Initialize backtester
        backtester = MLBacktester()

        # Run backtest (using test period)
        config = load_yaml_config("config/config.yaml")
        test_start = config.get('model_params', {}).get('test_start_date', '2024-07-01')
        test_end = config.get('model_params', {}).get('test_end_date', '2024-12-31')

        logger.info(f"Running backtest from {test_start} to {test_end}")

        results = backtester.run_backtest(test_start, test_end)

        # Print summary
        print_backtest_summary(results)

        # Save results
        results_file = backtester.save_results(results)

        print(f"\n💾 Results saved to: {results_file}")

        print("\n✅ ML signal backtest completed successfully!")
        print("\nNext steps:")
        print("1. Review backtest results")
        print("2. Adjust conviction thresholds if needed")
        print("3. Run interactive mode with ML model")

        return 0

    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        print(f"❌ Backtest failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
