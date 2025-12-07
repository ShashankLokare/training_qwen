#!/usr/bin/env python3
"""
Demo script showing how to use the Indian Trading Agent
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trading_agent import IndianTradingAgent
import json

def main():
    print("🚀 Indian Market Trading Agent Demo")
    print("=" * 50)

    # Initialize the trading agent
    agent = IndianTradingAgent()

    # Example 1: Get Nifty50 summary
    print("\n📊 Example 1: Nifty50 Summary")
    nifty_summary = agent.get_nifty50_summary(days=7)
    print(f"Current Nifty50 Level: {nifty_summary.get('current_level', 'N/A')}")
    print(f"Daily Change: {nifty_summary.get('daily_change_pct', 0):+.2f}%")

    # Example 2: Find consistent gainers
    print("\n🎯 Example 2: Finding Consistent Gainers (2-5% daily gains over 5 days)")
    gainers = agent.find_consistent_gainers(min_gain=2.0, max_gain=5.0, days=5, max_stocks=20)
    print(f"Found {gainers.get('consistent_gainers_count', 0)} consistent gainers")

    top_performers = gainers.get('top_performers', [])
    if top_performers:
        print("Top 3 performers:")
        for i, stock in enumerate(top_performers[:3], 1):
            print(f"{i}. {stock['symbol']}: ₹{stock['current_price']:.2f} "
                  f"({stock['daily_change_pct']:+.2f}%)")

    # Example 3: Market momentum ranking
    print("\n📈 Example 3: Market Momentum Ranking")
    momentum = agent.get_market_momentum_ranking(days=5)
    top_momentum = momentum.get('top_momentum_stocks', [])
    if top_momentum:
        print("Top momentum stocks:")
        for i, stock in enumerate(top_momentum[:5], 1):
            print(f"{i}. {stock['symbol']}: Score {stock['momentum_score']:.2f}")

    # Example 4: Comprehensive analysis
    print("\n🔍 Example 4: Comprehensive Market Analysis")
    analysis = agent.get_comprehensive_market_analysis(days=5)
    print(f"Market Sentiment: {analysis.get('market_sentiment', 'Unknown')}")

    recommendations = analysis.get('recommendations', [])
    if recommendations:
        print("Key Recommendations:")
        for rec in recommendations[:2]:
            print(f"• {rec}")

    print("\n✨ Demo completed! Use the CLI interface for more detailed analysis.")
    print("Run: python -m trading_agent.cli --help")

if __name__ == "__main__":
    main()
