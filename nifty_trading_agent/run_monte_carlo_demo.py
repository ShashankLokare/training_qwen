#!/usr/bin/env python3
"""
Run Monte Carlo Full History Analysis Demo
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from backtest.monte_carlo_full_history import FullHistoryMonteCarloAnalyzer
import json
from pathlib import Path

def main():
    print('🔬 GENERATING FULL-HISTORY MONTE CARLO REPORT (2022-2024)')
    print('='*65)

    try:
        # Initialize analyzer with 5,000 scenarios as requested
        analyzer = FullHistoryMonteCarloAnalyzer(num_scenarios=5000)  # 5,000 scenarios

        # Run comprehensive analysis
        report = analyzer.run_full_history_analysis()

        if report:
            print('\n📊 ANALYSIS RESULTS SUMMARY:')
            print('=' * 40)

            # Market data summary
            market = report.get('market_data_summary', {})
            if market:
                print(f'Market Period: {market.get("date_range", "N/A")}')
                print(f'Trading Days: {market.get("total_trading_days", 0)}')
                print(f'Stocks Covered: {market.get("stocks_covered", 0)}')

            # Strategy performance
            strategy = report.get('strategy_performance', {})
            if strategy:
                print(f'\nStrategy Observations: {strategy.get("total_return_observations", 0):,}')
                print('.3f')
                print(f'Win Rate: {strategy.get("win_rate", 0):.1f}%')

            # Risk metrics
            risk = report.get('risk_metrics', {})
            if risk:
                print(f'\nRISK METRICS (5,000 Scenarios):')
                print('.2f')
                print('.2f')
                print('.1f')
                print('.2f')

                print(f'\nPORTFOLIO VALUE RANGE:')
                print(f'5th Percentile: ${risk.get("p5_final_value", 0):,.0f}')
                print(f'Median: ${risk.get("median_final_value", 0):,.0f}')
                print(f'95th Percentile: ${risk.get("p95_final_value", 0):,.0f}')

            # Recommendations
            rec = report.get('recommendations', {})
            if rec:
                print(f'\nRECOMMENDATIONS:')
                print(f'Assessment: {rec.get("overall_assessment", "N/A")}')
                print(f'Risk Level: {rec.get("risk_level", "N/A")}')
                print(f'Recommended Position Size: {rec.get("recommended_position_size", "N/A")}')

            print(f'\n✅ Report Generated Successfully!')
            print(f'Total Scenarios: {report.get("total_scenarios", 0)}')
            print(f'Generated At: {report.get("generated_at", "N/A")}')

            # Save summary
            summary = {
                'market_summary': market,
                'strategy_performance': strategy,
                'risk_metrics': risk,
                'recommendations': rec,
                'generated_at': report.get('generated_at')
            }

            reports_dir = Path('reports')
            reports_dir.mkdir(exist_ok=True)

            with open(reports_dir / 'full_history_monte_carlo_summary.json', 'w') as f:
                json.dump(summary, f, indent=2, default=str)

            print(f'\n📄 Summary saved to: reports/full_history_monte_carlo_summary.json')

        else:
            print('❌ Report generation failed')

    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
