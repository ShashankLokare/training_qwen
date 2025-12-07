#!/usr/bin/env python3
"""
Interactive Main Script for Nifty Trading Agent
Allows users to configure their trading preferences interactively
"""

import sys
import logging
from datetime import datetime
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.user_interface import get_user_preferences_interactive
from utils.logging_utils import setup_logging
from pipeline.daily_pipeline import DailyPipeline
from utils.io_utils import ensure_directories

def main():
    """Main interactive execution function"""
    print("🚀 NSE NIFTY TRADING AGENT - INTERACTIVE MODE")
    print("=" * 60)
    print("Welcome to the interactive trading agent!")
    print("This tool will guide you through setting up your trading preferences.\n")

    # Setup logging
    log_config = {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file_path': 'logs/interactive_session.log',
        'max_file_size_mb': 50,
        'backup_count': 3
    }

    setup_logging(log_config)
    logger = logging.getLogger(__name__)

    try:
        # Ensure required directories exist
        ensure_directories([
            'logs',
            'reports',
            'data/price_data',
            'data/fundamentals_data',
            'models'
        ])

        # Get user preferences interactively
        print("📝 Let's configure your trading analysis preferences:")
        user_preferences = get_user_preferences_interactive()

        logger.info("User preferences collected successfully")
        logger.info(f"Selected index: {user_preferences['index']['name']}")
        logger.info(f"Number of stocks: {user_preferences['num_stocks']}")
        logger.info(f"Strategy: {user_preferences['strategy']['name']}")

        # Create a custom configuration based on user preferences
        custom_config = create_custom_config(user_preferences)

        # Run the analysis with user preferences
        print("\n🔄 Running analysis with your preferences...")
        logger.info("Starting analysis pipeline with custom configuration")

        pipeline = DailyPipeline()
        # Override the default config with custom config
        pipeline.config = custom_config

        results = pipeline.run_daily_analysis()

        # Display results in a user-friendly format
        display_results(results, user_preferences)

        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"reports/interactive_analysis_{timestamp}.json"

        import json
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n📄 Detailed results saved to: {results_file}")

        # Save signals if any
        if 'trading_signals' in results and results['trading_signals']:
            import pandas as pd
            signals_df = pd.DataFrame(results['trading_signals'])
            csv_file = f"reports/signals_{timestamp}.csv"
            signals_df.to_csv(csv_file, index=False)
            print(f"📈 Trading signals saved to: {csv_file}")

        logger.info("Interactive session completed successfully")
        print("\n✅ Analysis completed successfully!")
        print("\n💡 Tip: You can run this again anytime with different preferences!")

    except KeyboardInterrupt:
        print("\n\n⚠️  Session interrupted by user.")
        logger.info("Session interrupted by user")
    except Exception as e:
        logger.error(f"Error during interactive session: {e}", exc_info=True)
        print(f"\n❌ Error during analysis: {e}")
        print("Please check the logs for more details.")
        sys.exit(1)

def create_custom_config(user_preferences: dict) -> dict:
    """
    Create a custom configuration based on user preferences

    Args:
        user_preferences: User-selected preferences

    Returns:
        Custom configuration dictionary
    """
    # Load base config
    from utils.io_utils import load_yaml_config
    base_config = load_yaml_config("config/config.yaml")

    # Override with user preferences
    custom_config = base_config.copy()

    # Update universe based on selected index
    selected_index = user_preferences['index']
    if selected_index['symbol'] == '^NSEI':
        # Nifty 50 - use first N stocks
        custom_config['universe']['tickers'] = base_config['universe']['tickers'][:user_preferences['num_stocks']]
    elif selected_index['symbol'] == '^NSMIDCP':
        # Nifty Next 50 - use different stocks (simulated)
        next50_stocks = [
            "ADANIPORTS.NS", "DIVISLAB.NS", "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS",
            "MARICO.NS", "MCDOWELL-N.NS", "COLPAL.NS", "BERGEPAINT.NS", "PIDILITIND.NS"
        ]
        custom_config['universe']['tickers'] = next50_stocks[:user_preferences['num_stocks']]
    elif selected_index['symbol'] == '^NSEBANK':
        # Bank Nifty
        bank_stocks = [
            "HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS", "SBIN.NS",
            "INDUSINDBK.NS", "BANDHANBNK.NS", "IDFCFIRSTB.NS", "FEDERALBNK.NS", "RBLBANK.NS"
        ]
        custom_config['universe']['tickers'] = bank_stocks[:user_preferences['num_stocks']]
    else:
        # IT Nifty or default
        it_stocks = [
            "TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS",
            "LTIM.NS", "COFORGE.NS", "MPHASIS.NS", "PERSISTENT.NS", "MINDTREE.NS"
        ]
        custom_config['universe']['tickers'] = it_stocks[:user_preferences['num_stocks']]

    # Update signal thresholds
    custom_config['signal_thresholds']['min_predicted_return_pct'] = user_preferences['profitability_pct']
    custom_config['signal_thresholds']['min_conviction'] = user_preferences['conviction_threshold']

    # Update risk settings
    custom_config['risk_settings'].update(user_preferences['risk_params'])

    return custom_config

def display_results(results: dict, user_preferences: dict) -> None:
    """
    Display analysis results in a user-friendly format

    Args:
        results: Analysis results
        user_preferences: User preferences for context
    """
    print("\n" + "="*80)
    print("📊 ANALYSIS RESULTS")
    print("="*80)

    # User preferences summary
    print(f"📈 Index: {user_preferences['index']['name']}")
    print(f"📊 Stocks Analyzed: {user_preferences['num_stocks']}")
    print(f"🎯 Strategy: {user_preferences['strategy']['name']}")
    print(f"💰 Target Return: {user_preferences['profitability_pct']}%")
    print(f"📅 Data Period: {user_preferences['data_days']} days")
    print(f"🎚️ Conviction Threshold: {user_preferences['conviction_threshold']:.1f}")

    # Show model information if available
    if 'model_version' in results and results['model_version'] != 'heuristic':
        print(f"🤖 Model Version: {results['model_version']}")
    elif any(s.get('model_version') == 'heuristic' for s in results.get('trading_signals', [])):
        print("🤖 Signal Type: Heuristic (No ML model available)")

    print()

    # Market summary
    if 'nifty50_summary' in results:
        nifty = results['nifty50_summary']
        if 'error' not in nifty:
            print("📊 MARKET SUMMARY:")
            if 'current_level' in nifty:
                print(f"   Current Level: ₹{nifty['current_level']:.2f}")
            if 'daily_change_pct' in nifty:
                print(f"   Daily Change: {nifty['daily_change_pct']:+.2f}%")
            print(f"   Period Return: {nifty['period_return_pct']:.2f}%")
            print(f"   Data Points: {nifty['data_points']}")
            print()

    # Universe analysis
    if 'universe_analysis' in results:
        universe = results['universe_analysis']
        if 'error' not in universe:
            print("📈 UNIVERSE ANALYSIS:")
            print(f"   Total Stocks Analyzed: {universe['total_stocks_analyzed']}")
            print(f"   Successful Data Fetches: {universe['successful_fetches']}")

            # Show top performers
            stock_summaries = universe.get('stock_summaries', [])
            if stock_summaries:
                print("   Top Performers (5-day returns):")
                # Sort by 5-day return
                sorted_stocks = sorted(stock_summaries, key=lambda x: x['return_5d_pct'], reverse=True)
                for i, stock in enumerate(sorted_stocks[:3], 1):
                    print(f"     {i}. {stock['symbol']}: ₹{stock['current_price']:.2f} "
                          f"({stock['return_5d_pct']:+.1f}% 5d)")
            print()

    # Trading signals
    signals = results.get('trading_signals', [])
    if signals:
        print("🎯 TRADING SIGNALS GENERATED:")
        print("-" * 50)

        for i, signal in enumerate(signals, 1):
            entry_low, entry_high = signal['entry_range']
            target_price = signal['target_price']
            stop_loss = signal['stop_loss']
            position_size_pct = signal.get('position_size_pct', 0.05)  # Default 5%

            # Calculate position size (assuming $250K capital for demo)
            assumed_capital = 250000  # ₹2.5 lakhs
            position_size = assumed_capital * position_size_pct

            print(f"{i}. {signal['symbol']}")
            print(f"   Entry: ₹{entry_low:.2f} - ₹{entry_high:.2f}")
            print(f"   Target: ₹{target_price:.2f} (+{signal.get('expected_return_pct', 10.0):.1f}%)")
            print(f"   Stop Loss: ₹{stop_loss:.2f}")
            print(f"   Position: ₹{position_size:,.0f} ({position_size_pct:.1f}% of capital)")
            print(f"   Conviction: {signal['conviction']:.2f}")
            print(f"   Notes: {signal['notes']}")
            print()

        print(f"✅ Total Signals: {len(signals)}")
    else:
        print("🎯 TRADING SIGNALS:")
        print("   No signals generated based on current criteria.")
        print("   This could be due to:")
        print("   - No stocks meeting the return criteria")
        print("   - Market conditions not favorable")
        print("   - Try adjusting your parameters")
        print()

    # Market sentiment
    sentiment = results.get('market_sentiment', 'Unknown')
    print(f"📊 MARKET SENTIMENT: {sentiment}")

    # Recommendations
    recommendations = results.get('recommendations', [])
    if recommendations:
        print("\n💡 RECOMMENDATIONS:")
        for rec in recommendations:
            print(f"   • {rec}")

    print("\n⚠️  IMPORTANT DISCLAIMER:")
    print("   This analysis is for educational purposes only.")
    print("   Not financial advice. Always do your own research.")
    print("   Past performance does not guarantee future results.")

if __name__ == "__main__":
    main()
