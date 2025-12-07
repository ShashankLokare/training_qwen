"""
Daily Pipeline for Nifty Trading Agent
Orchestrates the daily analysis workflow with user preferences
"""

import logging
import sys
import pickle
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
import pandas as pd

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.logging_utils import get_logger
from utils.io_utils import load_yaml_config
from utils.duckdb_tools import load_ohlcv
from data_providers.market_data_provider import MarketDataProvider
from signals.strategies import StrategyFactory

logger = get_logger(__name__)

class DailyPipeline:
    """
    Orchestrates the daily trading analysis pipeline

    This is a simplified version that demonstrates the core functionality
    with market data fetching and basic analysis.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the daily pipeline

        Args:
            config_path: Path to configuration file
        """
        self.config = load_yaml_config(config_path)
        self.market_provider = MarketDataProvider(
            cache_dir=self.config.get('data_settings', {}).get('price_data_path', 'data/price_data'),
            cache_expiry_hours=self.config.get('data_settings', {}).get('cache_expiry_hours', 24)
        )

        # Load ML model if available
        self.model, self.metadata = self._load_ml_model()

        logger.info("Daily Pipeline initialized")

    def _load_ml_model(self) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
        """
        Load the latest trained ML model if available

        Returns:
            Tuple of (model, metadata) or (None, None) if no model found
        """
        try:
            model_dir = Path(self.config.get('model_params', {}).get('model_save_path', 'models/artifacts/'))

            if not model_dir.exists():
                logger.info("Model directory not found, using heuristic signals")
                return None, None

            # Find latest model file
            model_files = list(model_dir.glob("model_*.pkl"))
            if not model_files:
                logger.info("No model files found, using heuristic signals")
                return None, None

            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)

            logger.info(f"Loading ML model from {latest_model}")

            with open(latest_model, 'rb') as f:
                model_data = pickle.load(f)

            logger.info(f"ML model loaded: {model_data['metadata'].get('model_version', 'unknown')}")
            return model_data['model'], model_data['metadata']

        except Exception as e:
            logger.warning(f"Could not load ML model: {e}, using heuristic signals")
            return None, None

    def run_daily_analysis(self) -> Dict[str, Any]:
        """
        Run the complete daily analysis pipeline

        Returns:
            Dictionary containing analysis results
        """
        logger.info("Starting daily analysis pipeline")

        try:
            # Step 1: Get Nifty50 summary
            nifty_summary = self._get_nifty_summary()

            # Step 2: Analyze universe stocks
            universe_analysis = self._analyze_universe()

            # Step 3: Generate basic signals (simplified)
            signals = self._generate_basic_signals(universe_analysis)

            # Step 4: Prepare final report
            report = self._prepare_report(nifty_summary, universe_analysis, signals)

            logger.info("Daily analysis pipeline completed successfully")
            return report

        except Exception as e:
            logger.error(f"Error in daily analysis pipeline: {e}", exc_info=True)
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def _get_nifty_summary(self) -> Dict[str, Any]:
        """
        Get Nifty50 market summary

        Returns:
            Nifty50 summary data
        """
        logger.info("Fetching Nifty50 summary")

        try:
            # Get last 30 days of Nifty data for analysis
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=30)

            nifty_data = self.market_provider.get_index_data(
                index_symbol="^NSEI",
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat()
            )

            if nifty_data.empty:
                return {'error': 'Unable to fetch Nifty50 data'}

            # Calculate summary metrics
            current_level = nifty_data['Close'].iloc[-1]
            previous_close = nifty_data['Close'].iloc[-2] if len(nifty_data) > 1 else current_level
            daily_change = ((current_level - previous_close) / previous_close) * 100

            period_high = nifty_data['High'].max()
            period_low = nifty_data['Low'].min()
            period_return = ((current_level - nifty_data['Open'].iloc[0]) / nifty_data['Open'].iloc[0]) * 100

            return {
                'current_level': current_level,
                'daily_change_pct': daily_change,
                'period_high': period_high,
                'period_low': period_low,
                'period_return_pct': period_return,
                'data_points': len(nifty_data),
                'last_updated': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting Nifty summary: {e}")
            return {'error': str(e)}

    def _analyze_universe(self) -> Dict[str, Any]:
        """
        Analyze the stock universe

        Returns:
            Universe analysis results
        """
        logger.info("Analyzing stock universe")

        universe = self.config.get('universe', {}).get('tickers', [])
        if not universe:
            return {'error': 'No stocks in universe'}

        # Analyze first 5 stocks for demo (to avoid rate limits)
        analysis_stocks = universe[:5]
        logger.info(f"Analyzing {len(analysis_stocks)} stocks from universe")

        stock_summaries = []
        successful_fetches = 0

        for symbol in analysis_stocks:
            try:
                # Get recent data (last 10 days)
                end_date = datetime.now().date()
                start_date = end_date - timedelta(days=10)

                data = self.market_provider.get_ohlcv(
                    symbol=symbol,
                    start_date=start_date.isoformat(),
                    end_date=end_date.isoformat()
                )

                if not data.empty:
                    # Calculate basic metrics
                    current_price = data['Close'].iloc[-1]
                    prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
                    daily_change = ((current_price - prev_price) / prev_price) * 100

                    # Calculate 5-day return
                    if len(data) >= 5:
                        price_5d_ago = data['Close'].iloc[0]
                        return_5d = ((current_price - price_5d_ago) / price_5d_ago) * 100
                    else:
                        return_5d = 0.0

                    stock_summary = {
                        'symbol': symbol,
                        'current_price': current_price,
                        'daily_change_pct': daily_change,
                        'return_5d_pct': return_5d,
                        'volume': data['Volume'].iloc[-1] if 'Volume' in data.columns else 0,
                        'data_points': len(data)
                    }

                    stock_summaries.append(stock_summary)
                    successful_fetches += 1

            except Exception as e:
                logger.warning(f"Error analyzing {symbol}: {e}")
                continue

        return {
            'total_stocks_analyzed': len(analysis_stocks),
            'successful_fetches': successful_fetches,
            'stock_summaries': stock_summaries,
            'analysis_timestamp': datetime.now().isoformat()
        }

    def _generate_basic_signals(self, universe_analysis: Dict[str, Any]) -> list:
        """
        Generate trading signals using ML model or fallback to heuristics

        Args:
            universe_analysis: Results from universe analysis

        Returns:
            List of trading signals
        """
        logger.info("Generating trading signals")

        signals = []
        stock_summaries = universe_analysis.get('stock_summaries', [])

        # Use ML model if available, otherwise fall back to heuristics
        if self.model is not None:
            signals = self._generate_ml_signals(stock_summaries)
        else:
            logger.info("No ML model available, using heuristic signals")
            signals = self._generate_heuristic_signals(stock_summaries)

        logger.info(f"Generated {len(signals)} trading signals")
        return signals

    def _generate_ml_signals(self, stock_summaries: List[Dict[str, Any]]) -> list:
        """
        Generate signals using ML model predictions

        Args:
            stock_summaries: Stock analysis summaries

        Returns:
            List of ML-based trading signals
        """
        logger.info("Generating ML-based signals")

        signals = []
        conviction_threshold = self.config.get('signal_thresholds', {}).get('min_conviction', 0.8)

        for stock in stock_summaries:
            try:
                symbol = stock['symbol']

                # Build feature vector for this symbol
                features = self._build_features_for_symbol(symbol)

                if features is None:
                    continue

                # Get ML model prediction
                features_df = pd.DataFrame([features])
                prediction_prob = self.model.predict_proba(features_df)[0][1]

                # Apply conviction threshold
                if prediction_prob >= conviction_threshold:
                    # Generate signal based on prediction
                    signal = self._create_signal_from_prediction(symbol, stock, prediction_prob)
                    signals.append(signal)

                    # Limit to 5 signals max
                    if len(signals) >= 5:
                        break

            except Exception as e:
                logger.debug(f"Error generating ML signal for {stock.get('symbol', 'unknown')}: {e}")
                continue

        return signals

    def _build_features_for_symbol(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        Build feature vector for a symbol (simplified version)

        Args:
            symbol: Stock symbol

        Returns:
            Feature dictionary or None if insufficient data
        """
        try:
            # Get historical data (last 60 days)
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=60)

            ohlcv_data = load_ohlcv([symbol], start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

            if len(ohlcv_data) < 30:
                return None

            # Calculate basic technical features (simplified)
            close_prices = ohlcv_data['close'].values

            features = {}
            # Simple returns
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

            # Fill other features with defaults
            features['atr_14'] = std_20 * 0.1
            features['vol_zscore_20'] = 0.0
            features['quarterly_eps_growth'] = 0.0
            features['quarterly_rev_growth'] = 0.0
            features['profit_margin'] = 0.0
            features['pe_ratio'] = 15.0
            features['pb_ratio'] = 2.0
            features['sentiment_short'] = 0.0
            features['sentiment_medium'] = 0.0
            features['recent_earnings_flag'] = 0.0
            features['volume_gap_flag'] = 0.0

            return features

        except Exception as e:
            logger.debug(f"Error building features for {symbol}: {e}")
            return None

    def _create_signal_from_prediction(self, symbol: str, stock: Dict[str, Any],
                                     conviction: float) -> Dict[str, Any]:
        """
        Create a trading signal from ML prediction

        Args:
            symbol: Stock symbol
            stock: Stock summary data
            conviction: ML model conviction score

        Returns:
            Trading signal dictionary
        """
        current_price = stock['current_price']

        # Calculate entry range (±1% around current price)
        entry_low = current_price * 0.99
        entry_high = current_price * 1.01

        # Target: +10% from current price
        target_price = current_price * 1.10

        # Stop loss: -5% from current price
        stop_loss = current_price * 0.95

        # Position size: 5% of capital
        position_size_pct = self.config.get('risk_settings', {}).get('max_position_pct', 0.05)

        signal = {
            'symbol': symbol,
            'entry_range': [entry_low, entry_high],
            'target_price': target_price,
            'stop_loss': stop_loss,
            'position_size_pct': position_size_pct,
            'conviction': conviction,
            'expected_return_pct': 10.0,
            'notes': f'ML prediction with {conviction:.2f} confidence',
            'signal_timestamp': datetime.now().isoformat(),
            'model_version': self.metadata.get('model_version', 'unknown') if self.metadata else 'unknown'
        }

        return signal

    def _generate_heuristic_signals(self, stock_summaries: List[Dict[str, Any]]) -> list:
        """
        Generate signals using heuristic rules (fallback when no ML model)

        Args:
            stock_summaries: Stock analysis summaries

        Returns:
            List of heuristic trading signals
        """
        logger.info("Generating heuristic-based signals")

        signals = []

        for stock in stock_summaries:
            try:
                symbol = stock['symbol']
                current_price = stock['current_price']
                return_5d = stock['return_5d_pct']

                # Simple signal generation logic
                # Signal if 5-day return is between 2-5% (consistent growth)
                if 2.0 <= return_5d <= 5.0:
                    # Calculate entry range (±1% around current price)
                    entry_low = current_price * 0.99
                    entry_high = current_price * 1.01

                    # Target: +10% from current price
                    target_price = current_price * 1.10

                    # Stop loss: -5% from current price
                    stop_loss = current_price * 0.95

                    # Position size: 5% of capital
                    position_size_pct = self.config.get('risk_settings', {}).get('max_position_pct', 0.05)

                    signal = {
                        'symbol': symbol,
                        'entry_range': [entry_low, entry_high],
                        'target_price': target_price,
                        'stop_loss': stop_loss,
                        'position_size_pct': position_size_pct,
                        'conviction': 0.85,  # Fixed conviction for heuristics
                        'expected_return_pct': 10.0,
                        'notes': f'Heuristic: Consistent 5-day growth of {return_5d:.1f}%',
                        'signal_timestamp': datetime.now().isoformat(),
                        'model_version': 'heuristic'
                    }

                    signals.append(signal)

                    # Limit to 5 signals max
                    if len(signals) >= 5:
                        break

            except Exception as e:
                logger.warning(f"Error generating heuristic signal for {stock.get('symbol', 'unknown')}: {e}")
                continue

        return signals

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

    def _prepare_report(self, nifty_summary: Dict, universe_analysis: Dict, signals: list) -> Dict[str, Any]:
        """
        Prepare the final analysis report

        Args:
            nifty_summary: Nifty50 summary
            universe_analysis: Universe analysis results
            signals: Generated signals

        Returns:
            Complete analysis report
        """
        logger.info("Preparing final analysis report")

        # Calculate basic market sentiment
        if signals:
            sentiment = "BULLISH"
            confidence = "HIGH"
        else:
            sentiment = "NEUTRAL"
            confidence = "LOW"

        market_sentiment = f"{sentiment} (Confidence: {confidence}, {len(signals)} signals generated)"

        # Generate recommendations
        recommendations = []
        if signals:
            top_signals = signals[:3]
            symbols = [s['symbol'] for s in top_signals]
            recommendations.append(f"Consider monitoring: {', '.join(symbols)}")
            recommendations.append("Signals generated based on consistent 2-5% growth criteria")
        else:
            recommendations.append("No strong signals generated - maintain defensive position")

        recommendations.extend([
            "Always perform thorough due diligence before trading",
            "Consider consulting with a financial advisor",
            "This is for educational purposes only - not financial advice"
        ])

        report = {
            'report_title': 'NSE Nifty Daily Market Analysis Report',
            'generated_at': datetime.now().isoformat(),
            'nifty50_summary': nifty_summary,
            'universe_analysis': universe_analysis,
            'trading_signals': signals,
            'market_sentiment': market_sentiment,
            'recommendations': recommendations,
            'config_used': self.config.get('universe', {}).get('tickers', [])[:5],  # Show analyzed stocks
            'disclaimer': 'This analysis is for educational purposes only. Not financial advice.'
        }

        return report
