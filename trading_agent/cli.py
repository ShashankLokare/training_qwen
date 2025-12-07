#!/usr/bin/env python3
"""
Command Line Interface for Indian Trading Agent
"""

import argparse
import json
import sys
import os
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trading_agent import IndianTradingAgent

def print_nifty_summary(summary: dict):
    """Print Nifty50 summary in a readable format"""
    print("\n" + "="*60)
    print("NIFTY50 MARKET SUMMARY")
    print("="*60)

    if 'error' in summary:
        print(f"❌ Error: {summary['error']}")
        if 'current_level' in summary and summary['current_level']:
            print(".2f")
        return

    print(".2f")
    print(f"Daily Change: {summary['daily_change_pct']:+.2f}%")
    print(".2f")
    print(".2f")
    print(".2f")
    print(",.0f")
    print(f"Data Points: {summary['data_points']}")
    print(f"Analysis Period: {summary['analysis_period_days']} days")
    print(f"Last Updated: {summary['last_updated']}")

def print_consistent_gainers(analysis: dict):
    """Print consistent gainers analysis"""
    print("\n" + "="*60)
    print("CONSISTENT GAINERS ANALYSIS")
    print("="*60)

    criteria = analysis.get('analysis_criteria', {})
    print(f"Analysis Criteria: {criteria.get('min_daily_gain_pct')}-{criteria.get('max_daily_gain_pct')}% daily gains over {criteria.get('analysis_period_days')} days")
    print(f"Stocks Analyzed: {criteria.get('stocks_analyzed')}")
    print(f"Consistent Gainers Found: {analysis.get('consistent_gainers_count', 0)}")

    top_performers = analysis.get('top_performers', [])
    if top_performers:
        print(f"\n🏆 TOP {len(top_performers)} PERFORMERS:")
        for i, stock in enumerate(top_performers, 1):
            print(f"{i:2d}. {stock['symbol']:<15} "
                  ".2f"
                  ".1f"
                  ".2f")

def print_market_sentiment(sentiment: str, recommendations: list):
    """Print market sentiment and recommendations"""
    print("\n" + "="*60)
    print("MARKET SENTIMENT & RECOMMENDATIONS")
    print("="*60)

    print(f"📊 Sentiment: {sentiment}")

    print("\n💡 RECOMMENDATIONS:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")

def main():
    parser = argparse.ArgumentParser(description="Indian Market Trading Agent")
    parser.add_argument('--nifty', action='store_true', help='Get Nifty50 summary')
    parser.add_argument('--gainers', action='store_true', help='Find consistent gainers')
    parser.add_argument('--momentum', action='store_true', help='Get momentum ranking')
    parser.add_argument('--comprehensive', action='store_true', help='Run comprehensive analysis')
    parser.add_argument('--days', type=int, default=5, help='Analysis period in days (default: 5)')
    parser.add_argument('--min-gain', type=float, default=2.0, help='Minimum daily gain % (default: 2.0)')
    parser.add_argument('--max-gain', type=float, default=5.0, help='Maximum daily gain % (default: 5.0)')
    parser.add_argument('--output', choices=['console', 'json'], default='console', help='Output format')
    parser.add_argument('--max-stocks', type=int, default=50, help='Maximum stocks to analyze')

    args = parser.parse_args()

    # Initialize the trading agent
    agent = IndianTradingAgent()

    try:
        if args.comprehensive:
            # Run comprehensive analysis
            print("🔄 Running comprehensive market analysis...")
            result = agent.get_comprehensive_market_analysis(days=args.days)

            if args.output == 'json':
                print(json.dumps(result, indent=2, default=str))
            else:
                print_nifty_summary(result['nifty50_summary'])
                print_consistent_gainers(result['consistent_gainers'])
                print_market_sentiment(result['market_sentiment'], result['recommendations'])

        elif args.nifty:
            # Nifty50 summary only
            result = agent.get_nifty50_summary(days=args.days)
            if args.output == 'json':
                print(json.dumps(result, indent=2, default=str))
            else:
                print_nifty_summary(result)

        elif args.gainers:
            # Consistent gainers only
            result = agent.find_consistent_gainers(
                min_gain=args.min_gain,
                max_gain=args.max_gain,
                days=args.days,
                max_stocks=args.max_stocks
            )
            if args.output == 'json':
                print(json.dumps(result, indent=2, default=str))
            else:
                print_consistent_gainers(result)

        elif args.momentum:
            # Momentum ranking only
            result = agent.get_market_momentum_ranking(days=args.days)
            if args.output == 'json':
                print(json.dumps(result, indent=2, default=str))
            else:
                print(f"\nMomentum ranking for {result['stocks_analyzed']} stocks over {result['analysis_period_days']} days:")
                for i, stock in enumerate(result['top_momentum_stocks'][:10], 1):
                    print(f"{i:2d}. {stock['symbol']:<15} Score: {stock['momentum_score']:.2f} "
                          ".2f")

        else:
            # Default: run daily analysis
            print("🔄 Running daily market analysis...")
            result = agent.run_daily_analysis()

            if args.output == 'json':
                print(json.dumps(result, indent=2, default=str))
            else:
                print_nifty_summary(result['nifty50_summary'])
                print_consistent_gainers(result['consistent_gainers'])
                print_market_sentiment(result['market_sentiment'], result['recommendations'])

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
