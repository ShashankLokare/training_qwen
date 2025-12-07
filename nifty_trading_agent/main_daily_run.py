#!/usr/bin/env python3
"""
Main entry point for the Nifty Trading Agent daily run.
This script orchestrates the entire daily trading analysis pipeline.
"""

import sys
import logging
from datetime import datetime
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pipeline.daily_pipeline import DailyPipeline
from utils.logging_utils import setup_logging
from utils.io_utils import ensure_directories

def main():
    """Main execution function"""
    print("🚀 Nifty Trading Agent - Daily Analysis Run")
    print("=" * 60)

    # Setup logging
    log_config = {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file_path': 'logs/daily_run.log',
        'max_file_size_mb': 100,
        'backup_count': 5
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

        logger.info("Starting daily trading analysis pipeline")

        # Initialize and run the daily pipeline
        pipeline = DailyPipeline()
        results = pipeline.run_daily_analysis()

        # Print summary to console
        print("\n" + "="*60)
        print("📊 DAILY ANALYSIS SUMMARY")
        print("="*60)

        if 'nifty_summary' in results:
            nifty = results['nifty_summary']
            if 'current_level' in nifty:
                print(f"Current Nifty Level: ₹{nifty['current_level']:.2f}")
                if 'daily_change_pct' in nifty:
                    print(f"Daily Change: {nifty['daily_change_pct']:+.2f}%")

        if 'consistent_gainers' in results:
            gainers = results['consistent_gainers']
            count = gainers.get('consistent_gainers_count', 0)
            print(f"Consistent Gainers Found: {count}")

            if count > 0:
                top_performers = gainers.get('top_performers', [])
                print("Top Performers:")
                for i, stock in enumerate(top_performers[:3], 1):
                    print(f"  {i}. {stock['symbol']}: ₹{stock['current_price']:.2f}")

        if 'market_sentiment' in results:
            print(f"Market Sentiment: {results['market_sentiment']}")

        # Save detailed results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"reports/daily_analysis_{timestamp}.json"

        import json
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n📄 Detailed results saved to: {results_file}")

        # Save signals to CSV if any
        if 'consistent_gainers' in results:
            gainers = results['consistent_gainers']
            if gainers.get('consistent_gainers_count', 0) > 0:
                import pandas as pd
                signals = gainers.get('all_consistent_gainers', [])
                if signals:
                    df = pd.DataFrame(signals)
                    csv_file = f"reports/daily_signals_{timestamp}.csv"
                    df.to_csv(csv_file, index=False)
                    print(f"📈 Trading signals saved to: {csv_file}")

        logger.info("Daily analysis pipeline completed successfully")
        print("\n✅ Daily analysis completed successfully!")

    except Exception as e:
        logger.error(f"Error during daily analysis: {e}", exc_info=True)
        print(f"\n❌ Error during analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
