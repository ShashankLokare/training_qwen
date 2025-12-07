#!/usr/bin/env python3
"""
Monte Carlo Full History Analysis (2022-2024)
Comprehensive stress testing report for the complete trading period
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.logging_utils import setup_logging, get_logger
from utils.db_duckdb import execute_query

logger = get_logger(__name__)

class FullHistoryMonteCarloAnalyzer:
    """
    Comprehensive Monte Carlo analysis for 2023-2024 full history
    """

    def __init__(self, num_scenarios: int = 5000):
        """Initialize analyzer"""
        logger.info(f"FullHistoryMonteCarloAnalyzer initialized with {num_scenarios} scenarios")
        self.num_scenarios = num_scenarios

    def run_full_history_analysis(self) -> Dict[str, Any]:
        """
        Run complete Monte Carlo analysis for 2022-2024

        Returns:
            Comprehensive analysis results
        """
        logger.info("🔬 Running full-history Monte Carlo analysis (2022-2024)...")

        # Load historical market data for 2023-2024
        market_data = self._load_market_data()

        # Generate synthetic trading strategy returns
        strategy_returns = self._generate_strategy_returns(market_data)

        # Run Monte Carlo simulation
        simulation_results = self._run_monte_carlo_simulation(strategy_returns)

        # Calculate comprehensive risk metrics
        risk_metrics = self._calculate_risk_metrics(simulation_results)

        # Generate comparative analysis
        comparative_analysis = self._generate_comparative_analysis(market_data, simulation_results)

        # Create final report
        report = {
            'analysis_period': '2022-2024',
            'total_scenarios': self.num_scenarios,
            'market_data_summary': self._summarize_market_data(market_data),
            'strategy_performance': self._analyze_strategy_performance(strategy_returns),
            'risk_metrics': risk_metrics,
            'simulation_summary': {
                'total_scenarios': len(simulation_results),
                'avg_final_value': simulation_results['final_value'].mean(),
                'avg_total_return': simulation_results['total_return_pct'].mean(),
                'avg_volatility': simulation_results['volatility_pct'].mean(),
                'avg_win_rate': simulation_results['win_rate_pct'].mean()
            },
            'comparative_analysis': comparative_analysis,
            'recommendations': self._generate_recommendations(risk_metrics, comparative_analysis),
            'generated_at': datetime.now().isoformat()
        }

        return report

    def _load_market_data(self) -> pd.DataFrame:
        """Load comprehensive market data for 2022-2024"""
        logger.info("Loading market data for 2022-2024...")

        query = """
            SELECT
                date,
                symbol,
                close,
                volume
            FROM ohlcv_nifty
            WHERE date BETWEEN '2022-01-01' AND '2024-12-31'
            AND symbol IN ('RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS')
            ORDER BY date, symbol
        """

        result = execute_query(query)
        if not result:
            logger.error("No market data found")
            return pd.DataFrame()

        df = pd.DataFrame(result, columns=['date', 'symbol', 'close', 'volume'])
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['date', 'symbol'])

        logger.info(f"Loaded {len(df)} market data points for {df['symbol'].nunique()} stocks")
        return df

    def _generate_strategy_returns(self, market_data: pd.DataFrame) -> np.ndarray:
        """Generate synthetic trading strategy returns based on market conditions"""
        logger.info("Generating synthetic strategy returns...")

        if market_data.empty:
            return np.array([])

        # Calculate daily market returns for major stocks
        market_returns = []
        symbols = market_data['symbol'].unique()

        for symbol in symbols:
            symbol_data = market_data[market_data['symbol'] == symbol].copy()
            symbol_data = symbol_data.sort_values('date')
            symbol_data['daily_return'] = symbol_data['close'].pct_change()

            # Use only valid returns
            valid_returns = symbol_data['daily_return'].dropna().values
            if len(valid_returns) > 0:
                market_returns.extend(valid_returns)

        # Convert to numpy array and clean
        market_returns = np.array(market_returns)
        market_returns = market_returns[~np.isnan(market_returns)]
        market_returns = market_returns[np.abs(market_returns) < 0.20]  # Remove extreme outliers

        logger.info(f"Generated {len(market_returns)} daily market returns for strategy simulation")
        return market_returns

    def _run_monte_carlo_simulation(self, strategy_returns: np.ndarray) -> pd.DataFrame:
        """Run Monte Carlo simulation with 5,000 scenarios"""
        logger.info(f"Running {self.num_scenarios} Monte Carlo scenarios...")

        if len(strategy_returns) == 0:
            return pd.DataFrame()

        # Simulation parameters
        initial_capital = 100000.0
        max_positions = 5
        position_size = 0.05  # 5% per position
        stop_loss = 0.05
        take_profit = 0.10

        simulation_results = []

        for scenario in range(self.num_scenarios):
            if scenario % 500 == 0:
                logger.info(f"Running scenario {scenario + 1}/{self.num_scenarios}")

            # Generate random return sequence
            scenario_returns = np.random.choice(strategy_returns, size=252, replace=True)  # ~1 year

            # Simulate portfolio with risk management
            portfolio_value = initial_capital
            positions = 0
            scenario_pnl = []

            for daily_return in scenario_returns:
                # Apply position sizing and risk management
                if positions < max_positions and np.random.random() > 0.7:  # 30% chance of taking trade
                    # Enter position
                    positions += 1
                    position_value = portfolio_value * position_size

                # Simulate P&L with risk management
                if positions > 0:
                    # Apply stop loss/take profit logic
                    adjusted_return = daily_return

                    # Random stop loss hits (5% probability)
                    if np.random.random() < 0.05:
                        adjusted_return = -stop_loss
                        positions -= 1
                    # Random take profit hits (3% probability)
                    elif np.random.random() < 0.03:
                        adjusted_return = take_profit
                        positions -= 1

                    position_pnl = position_value * adjusted_return
                    portfolio_value += position_pnl
                    scenario_pnl.append(adjusted_return)

            # Calculate scenario metrics
            final_value = portfolio_value
            total_return = (final_value - initial_capital) / initial_capital * 100

            # Risk metrics
            if scenario_pnl:
                volatility = np.std(scenario_pnl) * np.sqrt(252) * 100
                win_rate = (np.array(scenario_pnl) > 0).mean() * 100

                # Drawdown calculation
                cumulative = np.cumprod(1 + np.array(scenario_pnl))
                rolling_max = np.maximum.accumulate(cumulative)
                drawdowns = (cumulative - rolling_max) / rolling_max
                max_drawdown = np.min(drawdowns) * 100 if len(drawdowns) > 0 else 0
            else:
                volatility = 0
                win_rate = 0
                max_drawdown = 0

            simulation_results.append({
                'scenario': scenario + 1,
                'final_value': final_value,
                'total_return_pct': total_return,
                'volatility_pct': volatility,
                'max_drawdown_pct': max_drawdown,
                'win_rate_pct': win_rate,
                'total_trades': len(scenario_pnl)
            })

        return pd.DataFrame(simulation_results)

    def _calculate_risk_metrics(self, simulation_results: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive risk metrics"""
        logger.info("Calculating comprehensive risk metrics...")

        if simulation_results.empty:
            return {}

        final_values = simulation_results['final_value'].values
        total_returns = simulation_results['total_return_pct'].values

        metrics = {}

        # Return statistics
        metrics['mean_return'] = np.mean(total_returns)
        metrics['median_return'] = np.median(total_returns)
        metrics['std_return'] = np.std(total_returns)
        metrics['min_return'] = np.min(total_returns)
        metrics['max_return'] = np.max(total_returns)

        # Value-at-Risk (VaR)
        metrics['var_95'] = np.percentile(total_returns, 5)
        metrics['var_99'] = np.percentile(total_returns, 1)
        metrics['var_999'] = np.percentile(total_returns, 0.1)

        # Expected Shortfall (ES)
        var_95_mask = total_returns <= metrics['var_95']
        metrics['expected_shortfall_95'] = np.mean(total_returns[var_95_mask]) if var_95_mask.sum() > 0 else metrics['var_95']

        # Probability analysis
        metrics['prob_loss'] = (total_returns < 0).mean() * 100
        metrics['prob_loss_10pct'] = (total_returns < -10).mean() * 100
        metrics['prob_loss_20pct'] = (total_returns < -20).mean() * 100
        metrics['prob_gain_10pct'] = (total_returns > 10).mean() * 100
        metrics['prob_gain_20pct'] = (total_returns > 20).mean() * 100

        # Portfolio value ranges
        metrics['initial_value'] = 100000.0
        metrics['mean_final_value'] = np.mean(final_values)
        metrics['median_final_value'] = np.median(final_values)
        metrics['p5_final_value'] = np.percentile(final_values, 5)
        metrics['p95_final_value'] = np.percentile(final_values, 95)

        # Risk-adjusted metrics
        metrics['sharpe_ratio_mean'] = simulation_results['total_return_pct'].mean() / simulation_results['total_return_pct'].std() if simulation_results['total_return_pct'].std() > 0 else 0

        return metrics

    def _summarize_market_data(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Summarize market data characteristics"""
        if market_data.empty:
            return {}

        summary = {
            'date_range': f"{market_data['date'].min()} to {market_data['date'].max()}",
            'total_trading_days': market_data['date'].nunique(),
            'stocks_covered': market_data['symbol'].nunique(),
            'total_data_points': len(market_data)
        }

        # Calculate average daily returns
        daily_returns = []
        for symbol in market_data['symbol'].unique():
            symbol_data = market_data[market_data['symbol'] == symbol].copy()
            symbol_data = symbol_data.sort_values('date')
            returns = symbol_data['close'].pct_change().dropna().values
            daily_returns.extend(returns)

        if daily_returns:
            summary['avg_daily_return'] = np.mean(daily_returns) * 100
            summary['market_volatility'] = np.std(daily_returns) * np.sqrt(252) * 100

        return summary

    def _analyze_strategy_performance(self, strategy_returns: np.ndarray) -> Dict[str, Any]:
        """Analyze strategy return characteristics"""
        if len(strategy_returns) == 0:
            return {}

        analysis = {
            'total_return_observations': len(strategy_returns),
            'mean_daily_return': np.mean(strategy_returns) * 100,
            'median_daily_return': np.median(strategy_returns) * 100,
            'std_daily_return': np.std(strategy_returns) * 100,
            'skewness': pd.Series(strategy_returns).skew(),
            'kurtosis': pd.Series(strategy_returns).kurtosis(),
            'positive_return_days': (strategy_returns > 0).sum(),
            'negative_return_days': (strategy_returns < 0).sum(),
            'win_rate': (strategy_returns > 0).mean() * 100
        }

        # Best/worst days
        analysis['best_day'] = np.max(strategy_returns) * 100
        analysis['worst_day'] = np.min(strategy_returns) * 100

        # Consecutive statistics
        consecutive_wins = 0
        max_consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_losses = 0

        for ret in strategy_returns:
            if ret > 0:
                consecutive_wins += 1
                consecutive_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
            elif ret < 0:
                consecutive_losses += 1
                consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)

        analysis['max_consecutive_wins'] = max_consecutive_wins
        analysis['max_consecutive_losses'] = max_consecutive_losses

        return analysis

    def _generate_comparative_analysis(self, market_data: pd.DataFrame, simulation_results: pd.DataFrame) -> Dict[str, Any]:
        """Generate comparative analysis vs market benchmarks"""
        logger.info("Generating comparative analysis...")

        comparative = {}

        # Market benchmark (simplified Nifty 50 proxy)
        if not market_data.empty:
            # Calculate market returns for comparison
            market_returns = []
            for symbol in market_data['symbol'].unique()[:3]:  # Use first 3 stocks as proxy
                symbol_data = market_data[market_data['symbol'] == symbol].copy()
                symbol_data = symbol_data.sort_values('date')
                returns = symbol_data['close'].pct_change().dropna().values
                market_returns.extend(returns)

            if market_returns:
                market_return_avg = np.mean(market_returns) * 252 * 100  # Annualized
                market_volatility = np.std(market_returns) * np.sqrt(252) * 100

                strategy_return_avg = simulation_results['total_return_pct'].mean()
                strategy_volatility = simulation_results['volatility_pct'].mean()

                comparative['market_vs_strategy'] = {
                    'market_annual_return': market_return_avg,
                    'market_volatility': market_volatility,
                    'strategy_annual_return': strategy_return_avg,
                    'strategy_volatility': strategy_volatility,
                    'outperformance': strategy_return_avg - market_return_avg,
                    'volatility_adjusted_outperformance': (strategy_return_avg / strategy_volatility) - (market_return_avg / market_volatility)
                }

        # Risk parity analysis
        if not simulation_results.empty:
            # Calculate Sharpe ratios
            sharpe_ratios = []
            for _, row in simulation_results.iterrows():
                if row['volatility_pct'] > 0:
                    sharpe = row['total_return_pct'] / row['volatility_pct']
                    sharpe_ratios.append(sharpe)

            comparative['risk_analysis'] = {
                'sharpe_ratio_distribution': {
                    'mean': np.mean(sharpe_ratios),
                    'std': np.std(sharpe_ratios),
                    'min': np.min(sharpe_ratios),
                    'max': np.max(sharpe_ratios),
                    'percentile_25': np.percentile(sharpe_ratios, 25),
                    'percentile_75': np.percentile(sharpe_ratios, 75)
                }
            }

        return comparative

    def _generate_recommendations(self, risk_metrics: Dict[str, float], comparative_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate investment recommendations based on analysis"""
        logger.info("Generating investment recommendations...")

        recommendations = {
            'overall_assessment': '',
            'risk_level': '',
            'recommended_position_size': '',
            'monitoring_requirements': [],
            'improvement_suggestions': []
        }

        # Risk assessment
        var_95 = risk_metrics.get('var_95', 0)
        prob_loss_10 = risk_metrics.get('prob_loss_10pct', 0)
        mean_return = risk_metrics.get('mean_return', 0)

        if var_95 > -15 and prob_loss_10 < 20 and mean_return > 5:
            recommendations['overall_assessment'] = "STRONG - Strategy shows robust performance with controlled risk"
            recommendations['risk_level'] = "MODERATE"
            recommendations['recommended_position_size'] = "5-10% of portfolio"
        elif var_95 > -25 and prob_loss_10 < 30:
            recommendations['overall_assessment'] = "MODERATE - Acceptable performance with some risk concerns"
            recommendations['risk_level'] = "MODERATE"
            recommendations['recommended_position_size'] = "2-5% of portfolio"
        else:
            recommendations['overall_assessment'] = "CAUTION - Strategy requires significant improvements"
            recommendations['risk_level'] = "HIGH"
            recommendations['recommended_position_size'] = "1-2% of portfolio"

        # Monitoring requirements
        recommendations['monitoring_requirements'] = [
            "Daily P&L tracking and risk limit monitoring",
            "Weekly performance review against benchmarks",
            "Monthly stress testing with updated market conditions",
            "Quarterly model retraining and validation",
            "Real-time position size and exposure limits"
        ]

        # Improvement suggestions
        recommendations['improvement_suggestions'] = [
            "Implement dynamic position sizing based on volatility",
            "Add market regime detection for adaptive strategies",
            "Incorporate transaction cost modeling",
            "Enhance stop-loss mechanisms with trailing stops",
            "Add fundamental filters to improve signal quality"
        ]

        return recommendations

    def print_comprehensive_report(self, report: Dict[str, Any]):
        """Print comprehensive Monte Carlo analysis report"""
        print("\n" + "="*80)
        print("📊 FULL-HISTORY MONTE CARLO ANALYSIS REPORT (2022-2024)")
        print("="*80)

        # Market data summary
        market = report.get('market_data_summary', {})
        if market:
            print("\n📈 MARKET DATA SUMMARY:")
            print(f"  Period: {market.get('date_range', 'N/A')}")
            print(f"  Trading Days: {market.get('total_trading_days', 0)}")
            print(f"  Stocks Covered: {market.get('stocks_covered', 0)}")
            print(f".2f")
            print(f".1f")

        # Strategy performance
        strategy = report.get('strategy_performance', {})
        if strategy:
            print("\n🎯 STRATEGY PERFORMANCE:")
            print(f"  Total Observations: {strategy.get('total_return_observations', 0):,}")
            print(f".3f")
            print(f".1f")
            print(f"  Win Rate: {strategy.get('win_rate', 0):.1f}%")
            print(f"  Best Day: {strategy.get('best_day', 0):.2f}%")
            print(f"  Worst Day: {strategy.get('worst_day', 0):.2f}%")
            print(f"  Max Consecutive Wins: {strategy.get('max_consecutive_wins', 0)}")
            print(f"  Max Consecutive Losses: {strategy.get('max_consecutive_losses', 0)}")

        # Risk metrics
        risk = report.get('risk_metrics', {})
        if risk:
            print("\n🎲 RISK METRICS (5,000 Scenarios):")
            print(f".2f")
            print(f".2f")
            print(f".2f")
            print(f".2f")
            print(f".2f")
            print(f".1f")
            print(f".1f")
            print(f".1f")
            print(f".1f")
            print(f".1f")

            print("\n💰 PORTFOLIO VALUE RANGE:")
            print(f"  5th Percentile: ${risk.get('p5_final_value', 0):,.0f}")
            print(f"  Median: ${risk.get('median_final_value', 0):,.0f}")
            print(f"  95th Percentile: ${risk.get('p95_final_value', 0):,.0f}")

        # Comparative analysis
        comparative = report.get('comparative_analysis', {})
        if 'market_vs_strategy' in comparative:
            mvs = comparative['market_vs_strategy']
            print("\n📊 MARKET COMPARISON:")
            print(f".1f")
            print(f".1f")
            print(f".1f")
            print(f".1f")
            print(f".2f")
        # Recommendations
        rec = report.get('recommendations', {})
        if rec:
            print("\n🎯 RECOMMENDATIONS:")
            print(f"  Overall Assessment: {rec.get('overall_assessment', 'N/A')}")
            print(f"  Risk Level: {rec.get('risk_level', 'N/A')}")
            print(f"  Recommended Position Size: {rec.get('recommended_position_size', 'N/A')}")

            print("\n📋 MONITORING REQUIREMENTS:")
            for req in rec.get('monitoring_requirements', []):
                print(f"  • {req}")

            print("\n💡 IMPROVEMENT SUGGESTIONS:")
            for sug in rec.get('improvement_suggestions', []):
                print(f"  • {sug}")

        print("\n" + "="*80)
        print(f"Report Generated: {report.get('generated_at', 'N/A')}")
        print("="*80)

def main():
    """Main analysis function"""
    print("🔬 NIFTY TRADING AGENT - FULL-HISTORY MONTE CARLO ANALYSIS")
    print("="*65)

    # Setup logging
    setup_logging({
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file_path': 'logs/full_history_monte_carlo.log',
        'max_file_size_mb': 50,
        'backup_count': 3
    })

    try:
        # Initialize analyzer
        analyzer = FullHistoryMonteCarloAnalyzer(num_scenarios=5000)

        # Run comprehensive analysis
        report = analyzer.run_full_history_analysis()

        if not report:
            logger.error("Analysis failed")
            return 1

        # Print comprehensive report
        analyzer.print_comprehensive_report(report)

        # Save report
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)

        report_path = reports_dir / f"full_history_monte_carlo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(report_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj

            json.dump(report, f, indent=2, default=convert_numpy)

        print(f"\n✅ Report saved to: {report_path}")
        print("\n🎯 Analysis Complete - Full 2023-2024 Monte Carlo Report Generated!")

        return 0

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        print(f"❌ Analysis failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
