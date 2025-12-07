#!/usr/bin/env python3
"""
Monte Carlo Stress Testing for Nifty Trading Agent v2
1000+ scenario analysis with VaR and Expected Shortfall calculations
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.logging_utils import get_logger

logger = get_logger(__name__)

class MonteCarloStressTester:
    """
    Monte Carlo simulation for stress testing trading strategies
    """

    def __init__(self, n_simulations: int = 1000, confidence_level: float = 0.95):
        self.n_simulations = n_simulations
        self.confidence_level = confidence_level

    def run_stress_test(self, historical_returns: pd.Series,
                       strategy_function, **strategy_params) -> Dict[str, Any]:
        """
        Run Monte Carlo stress test on trading strategy

        Args:
            historical_returns: Historical daily returns series
            strategy_function: Function that takes returns and returns pnl
            **strategy_params: Parameters to pass to strategy function
        """
        print(f"🎲 RUNNING MONTE CARLO STRESS TEST ({self.n_simulations} simulations)")
        print("=" * 60)

        # Bootstrap historical returns to create scenarios
        scenarios = self._generate_bootstrapped_scenarios(historical_returns)

        # Run strategy on each scenario
        simulation_results = []

        for i in range(self.n_simulations):
            if (i + 1) % 100 == 0:
                print(f"   Completed {i + 1}/{self.n_simulations} simulations...")

            try:
                scenario_returns = scenarios[i]
                pnl = strategy_function(scenario_returns, **strategy_params)
                simulation_results.append(pnl)
            except Exception as e:
                logger.debug(f"Simulation {i+1} failed: {e}")
                continue

        # Analyze results
        analysis = self._analyze_simulation_results(simulation_results)

        # Generate report
        self._generate_stress_test_report(analysis, simulation_results)

        return analysis

    def _generate_bootstrapped_scenarios(self, historical_returns: pd.Series) -> List[np.ndarray]:
        """Generate bootstrapped return scenarios"""
        n_days = len(historical_returns)
        scenarios = []

        for _ in range(self.n_simulations):
            # Bootstrap with replacement
            bootstrapped_returns = np.random.choice(
                historical_returns.values,
                size=n_days,
                replace=True
            )
            scenarios.append(bootstrapped_returns)

        return scenarios

    def _analyze_simulation_results(self, results: List[float]) -> Dict[str, Any]:
        """Analyze Monte Carlo simulation results"""
        if not results:
            return {'error': 'No simulation results'}

        results_array = np.array(results)

        analysis = {
            'n_simulations': len(results),
            'mean_pnl': float(np.mean(results_array)),
            'std_pnl': float(np.std(results_array)),
            'min_pnl': float(np.min(results_array)),
            'max_pnl': float(np.max(results_array)),
            'median_pnl': float(np.median(results_array)),

            # Risk metrics
            'sharpe_ratio': float(self._calculate_sharpe_ratio(results_array)),
            'sortino_ratio': float(self._calculate_sortino_ratio(results_array)),
            'value_at_risk_95': float(np.percentile(results_array, 5)),
            'value_at_risk_99': float(np.percentile(results_array, 1)),
            'expected_shortfall_95': float(self._calculate_expected_shortfall(results_array, 95)),
            'expected_shortfall_99': float(self._calculate_expected_shortfall(results_array, 99)),
            'maximum_drawdown': float(self._calculate_max_drawdown(results_array)),

            # Probability metrics
            'prob_profit': float(np.mean(results_array > 0)),
            'prob_loss_greater_10pct': float(np.mean(results_array < -0.10)),
            'prob_loss_greater_20pct': float(np.mean(results_array < -0.20)),

            # Confidence intervals
            'pnl_confidence_interval': {
                'lower': float(np.percentile(results_array, (1-self.confidence_level)/2 * 100)),
                'upper': float(np.percentile(results_array, (1+self.confidence_level)/2 * 100))
            }
        }

        return analysis

    def _calculate_sharpe_ratio(self, pnls: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(pnls) == 0 or np.std(pnls) == 0:
            return 0.0

        # Annualized Sharpe ratio (assuming daily PnLs)
        annualized_return = np.mean(pnls) * 252
        annualized_volatility = np.std(pnls) * np.sqrt(252)

        if annualized_volatility == 0:
            return 0.0

        return (annualized_return - risk_free_rate) / annualized_volatility

    def _calculate_sortino_ratio(self, pnls: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (downside deviation only)"""
        if len(pnls) == 0:
            return 0.0

        # Calculate downside deviation (only negative returns)
        downside_returns = pnls[pnls < 0]
        if len(downside_returns) == 0:
            return float('inf')  # No downside risk

        downside_deviation = np.std(downside_returns)

        if downside_deviation == 0:
            return float('inf')

        # Annualized Sortino ratio
        annualized_return = np.mean(pnls) * 252
        annualized_downside_deviation = downside_deviation * np.sqrt(252)

        return (annualized_return - risk_free_rate) / annualized_downside_deviation

    def _calculate_expected_shortfall(self, pnls: np.ndarray, confidence_level: int) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        if len(pnls) == 0:
            return 0.0

        # Find VaR threshold
        var_threshold = np.percentile(pnls, 100 - confidence_level)
        tail_losses = pnls[pnls <= var_threshold]

        if len(tail_losses) == 0:
            return var_threshold

        return np.mean(tail_losses)

    def _calculate_max_drawdown(self, pnls: np.ndarray) -> float:
        """Calculate maximum drawdown from peak to trough"""
        if len(pnls) == 0:
            return 0.0

        # Convert to cumulative returns
        cumulative = np.cumprod(1 + pnls)

        # Calculate drawdowns
        peak = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - peak) / peak

        return np.min(drawdowns)  # Most negative drawdown

    def _generate_stress_test_report(self, analysis: Dict[str, Any],
                                   raw_results: List[float]):
        """Generate comprehensive stress test report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = Path("reports/stress_tests/") / f"monte_carlo_stress_test_{timestamp}.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MONTE CARLO STRESS TEST REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write("SIMULATION PARAMETERS\n")
            f.write("-" * 22 + "\n")
            f.write(f"Number of simulations: {analysis.get('n_simulations', 0)}\n")
            f.write(f"Confidence level: {self.confidence_level:.1%}\n\n")

            f.write("PERFORMANCE STATISTICS\n")
            f.write("-" * 23 + "\n")
            f.write(f"Mean PnL: {analysis.get('mean_pnl', 0):.2%}\n")
            f.write(f"Standard Deviation: {analysis.get('std_pnl', 0):.2%}\n")
            f.write(f"Median PnL: {analysis.get('median_pnl', 0):.2%}\n")
            f.write(f"Best Case: {analysis.get('max_pnl', 0):.2%}\n")
            f.write(f"Worst Case: {analysis.get('min_pnl', 0):.2%}\n\n")

            f.write("RISK METRICS\n")
            f.write("-" * 13 + "\n")
            f.write(f"Sharpe Ratio: {analysis.get('sharpe_ratio', 0):.2f}\n")
            f.write(f"Sortino Ratio: {analysis.get('sortino_ratio', 0):.2f}\n")
            f.write(f"Maximum Drawdown: {analysis.get('maximum_drawdown', 0):.2%}\n\n")

            f.write("VALUE AT RISK (VaR)\n")
            f.write("-" * 19 + "\n")
            f.write(f"VaR 95%: {analysis.get('value_at_risk_95', 0):.2%}\n")
            f.write(f"VaR 99%: {analysis.get('value_at_risk_99', 0):.2%}\n")
            f.write(f"Expected Shortfall 95%: {analysis.get('expected_shortfall_95', 0):.2%}\n")
            f.write(f"Expected Shortfall 99%: {analysis.get('expected_shortfall_99', 0):.2%}\n\n")

            ci = analysis.get('pnl_confidence_interval', {})
            f.write("CONFIDENCE INTERVALS\n")
            f.write("-" * 21 + "\n")
            f.write(f"PnL {self.confidence_level:.0%} CI: [{ci.get('lower', 0):.2%}, {ci.get('upper', 0):.2%}]\n\n")

            f.write("PROBABILITY ANALYSIS\n")
            f.write("-" * 21 + "\n")
            f.write(f"Probability of Profit: {analysis.get('prob_profit', 0):.1%}\n")
            f.write(f"Probability of >10% Loss: {analysis.get('prob_loss_greater_10pct', 0):.1%}\n")
            f.write(f"Probability of >20% Loss: {analysis.get('prob_loss_greater_20pct', 0):.1%}\n\n")

            f.write("RISK ASSESSMENT\n")
            f.write("-" * 16 + "\n")

            sharpe = analysis.get('sharpe_ratio', 0)
            max_dd = analysis.get('maximum_drawdown', 0)
            prob_profit = analysis.get('prob_profit', 0)

            if sharpe > 2.0 and abs(max_dd) < 0.15 and prob_profit > 0.65:
                risk_assessment = "EXCELLENT: Low risk, high reward profile"
            elif sharpe > 1.5 and abs(max_dd) < 0.20 and prob_profit > 0.60:
                risk_assessment = "GOOD: Reasonable risk-adjusted returns"
            elif sharpe > 1.0 and abs(max_dd) < 0.25 and prob_profit > 0.55:
                risk_assessment = "FAIR: Acceptable risk profile"
            else:
                risk_assessment = "POOR: High risk, low reward profile"

            f.write(f"Overall Assessment: {risk_assessment}\n\n")

            f.write("RECOMMENDATIONS\n")
            f.write("-" * 15 + "\n")

            if analysis.get('prob_loss_greater_20pct', 0) > 0.05:
                f.write("• Implement stricter risk limits - high probability of large losses\n")

            if abs(analysis.get('maximum_drawdown', 0)) > 0.20:
                f.write("• Add drawdown controls and position size limits\n")

            if analysis.get('sharpe_ratio', 0) < 1.0:
                f.write("• Improve signal quality or reduce position sizes\n")

            if analysis.get('prob_profit', 0) < 0.60:
                f.write("• Refine entry/exit criteria for better win rate\n")

        print(f"📄 Monte Carlo stress test report saved to: {report_path}")

        # Also save summary statistics as JSON
        json_path = report_path.with_suffix('.json')
        import json

        with open(json_path, 'w') as f:
            json.dump({
                'analysis': analysis,
                'raw_results_summary': {
                    'count': len(raw_results),
                    'mean': np.mean(raw_results),
                    'std': np.std(raw_results),
                    'quartiles': {
                        '25%': np.percentile(raw_results, 25),
                        '50%': np.percentile(raw_results, 50),
                        '75%': np.percentile(raw_results, 75)
                    }
                },
                'generated_at': datetime.now().isoformat()
            }, f, indent=2, default=str)

def run_trading_strategy_simulation(daily_returns: np.ndarray,
                                  signal_threshold: float = 0.6,
                                  max_position_size: float = 0.05,
                                  stop_loss: float = 0.05) -> float:
    """
    Example trading strategy for Monte Carlo testing

    Args:
        daily_returns: Array of daily returns
        signal_threshold: Minimum probability for signal
        max_position_size: Maximum position size
        stop_loss: Stop loss percentage

    Returns:
        Total PnL for the period
    """
    capital = 1.0  # Start with $1
    position = 0.0
    entry_price = 0.0

    for daily_return in daily_returns:
        # Simulate getting a random signal (in real strategy, this would be model predictions)
        signal_prob = np.random.beta(2, 2)  # Simulate realistic signal distribution

        # Check if we should enter a position
        if position == 0 and signal_prob >= signal_threshold:
            # Enter long position
            position_size = min(max_position_size * capital, capital * 0.1)
            position = position_size
            entry_price = capital
            capital -= position_size

        # Manage existing position
        elif position > 0:
            # Check stop loss
            if capital < entry_price * (1 - stop_loss):
                # Close position at loss
                capital += position
                position = 0
                entry_price = 0
            else:
                # Update position value with daily return
                position_value = position * (1 + daily_return)
                unrealized_pnl = position_value - position
                position = position_value

        # Update capital with daily return (uninvested portion)
        uninvested_capital = capital
        capital = uninvested_capital * (1 + daily_return * 0.02)  # 2% of uninvested capital in cash-like instrument

    # Close any remaining position
    if position > 0:
        capital += position

    # Return total return
    return capital - 1.0

def example_monte_carlo_test():
    """Example of how to run Monte Carlo stress test"""

    # Generate synthetic historical returns (replace with real data)
    np.random.seed(42)
    historical_returns = np.random.normal(0.0005, 0.02, 1000)  # 1000 days of returns

    # Initialize stress tester
    tester = MonteCarloStressTester(n_simulations=500)  # Reduced for example

    # Run stress test
    results = tester.run_stress_test(
        historical_returns=pd.Series(historical_returns),
        strategy_function=run_trading_strategy_simulation,
        signal_threshold=0.6,
        max_position_size=0.05,
        stop_loss=0.05
    )

    print("\n🎯 STRESS TEST RESULTS SUMMARY")
    print("=" * 35)
    print(f"Mean Return: {results.get('mean_pnl', 0):.2%}")
    print(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
    print(f"VaR 95%: {results.get('value_at_risk_95', 0):.2%}")
    print(f"Max Drawdown: {results.get('maximum_drawdown', 0):.2%}")
    print(f"Probability of Profit: {results.get('prob_profit', 0):.1%}")

    return results

def main():
    """Main Monte Carlo stress test function"""
    print("🎲 NIFTY TRADING AGENT - MONTE CARLO STRESS TESTING")
    print("=" * 58)

    try:
        # Run example test
        results = example_monte_carlo_test()

        print("\n✅ Monte Carlo stress testing completed!")
        print("   (This was an example using synthetic data)")
        print("   For real testing, replace with actual strategy and historical returns")

        return 0

    except Exception as e:
        print(f"❌ Monte Carlo stress testing failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
