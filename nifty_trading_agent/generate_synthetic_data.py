#!/usr/bin/env python3
"""
Generate Synthetic Historical Data for NSE Nifty Trading Agent
Creates realistic OHLCV data for demonstration and testing
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.logging_utils import setup_logging, get_logger
from utils.duckdb_tools import store_ohlcv

logger = get_logger(__name__)

class SyntheticDataGenerator:
    """
    Generates realistic synthetic OHLCV data for NSE stocks
    """

    def __init__(self, seed: int = 42):
        """
        Initialize with random seed for reproducibility

        Args:
            seed: Random seed for reproducible data
        """
        np.random.seed(seed)
        self.seed = seed

        # Base prices for different stocks (approximating current market prices)
        self.base_prices = {
            "RELIANCE.NS": 2500,
            "TCS.NS": 3200,
            "HDFCBANK.NS": 1400,
            "ICICIBANK.NS": 1400,
            "INFY.NS": 1600,
            "HINDUNILVR.NS": 2800,
            "ITC.NS": 450,
            "KOTAKBANK.NS": 1800,
            "LT.NS": 3500,
            "AXISBANK.NS": 1200,
            "BAJFINANCE.NS": 7200,
            "MARUTI.NS": 12000,
            "BAJAJ-AUTO.NS": 9200,
            "WIPRO.NS": 500,
            "HCLTECH.NS": 1600,
            "ADANIPORTS.NS": 1500,
            "DIVISLAB.NS": 4500,
            "SUNPHARMA.NS": 1800,
            "DRREDDY.NS": 6500,
            "CIPLA.NS": 1600
        }

        # Market volatility parameters
        self.market_volatility = 0.02  # 2% daily volatility
        self.stock_specific_volatility = 0.015  # 1.5% stock-specific volatility

    def generate_ohlcv_data(self, symbols: list, start_date: str,
                           end_date: str, trading_days_only: bool = True) -> pd.DataFrame:
        """
        Generate synthetic OHLCV data for specified symbols and date range

        Args:
            symbols: List of stock symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            trading_days_only: Generate only trading days (skip weekends)

        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Generating synthetic OHLCV data for {len(symbols)} symbols from {start_date} to {end_date}")

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        if trading_days_only:
            # Generate business days only
            date_range = pd.date_range(start=start_dt, end=end_dt, freq='B')
        else:
            date_range = pd.date_range(start=start_dt, end=end_dt, freq='D')

        all_data = []

        for symbol in symbols:
            logger.info(f"Generating data for {symbol}")
            symbol_data = self._generate_symbol_data(symbol, date_range)
            all_data.extend(symbol_data)

        df = pd.DataFrame(all_data)
        df = df.sort_values(['symbol', 'date']).reset_index(drop=True)

        logger.info(f"Generated {len(df)} total OHLCV records")
        return df

    def _generate_symbol_data(self, symbol: str, date_range: pd.DatetimeIndex) -> list:
        """
        Generate OHLCV data for a single symbol

        Args:
            symbol: Stock symbol
            date_range: Date range for data generation

        Returns:
            List of OHLCV records
        """
        base_price = self.base_prices.get(symbol, 1000)
        current_price = base_price

        data = []

        for date in date_range:
            # Generate daily price movement
            market_return = np.random.normal(0, self.market_volatility)
            stock_return = np.random.normal(0, self.stock_specific_volatility)
            total_return = market_return + stock_return

            # Apply return to get new price
            new_price = current_price * (1 + total_return)

            # Generate OHLC with realistic intraday volatility
            intraday_vol = abs(total_return) * 0.5 + 0.005  # Base intraday volatility

            high = new_price * (1 + np.random.uniform(0, intraday_vol))
            low = new_price * (1 - np.random.uniform(0, intraday_vol))
            open_price = current_price * (1 + np.random.normal(0, intraday_vol * 0.5))

            # Ensure OHLC relationships are correct
            high = max(high, open_price, new_price)
            low = min(low, open_price, new_price)

            # Generate volume (realistic ranges)
            base_volume = {
                "RELIANCE.NS": 8000000,
                "TCS.NS": 2500000,
                "HDFCBANK.NS": 18000000,
                "ICICIBANK.NS": 25000000,
                "INFY.NS": 6000000,
            }.get(symbol, 1000000)

            volume_variation = np.random.uniform(0.5, 1.5)
            volume = int(base_volume * volume_variation)

            record = {
                'symbol': symbol,
                'date': date.date(),
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(new_price, 2),
                'volume': volume
            }

            data.append(record)
            current_price = new_price

        return data

    def save_to_database(self, df: pd.DataFrame) -> bool:
        """
        Save generated data to DuckDB database

        Args:
            df: OHLCV DataFrame

        Returns:
            True if successful
        """
        logger.info("Saving synthetic data to DuckDB")
        return store_ohlcv(df)

def main():
    """Main data generation function"""
    print("🚀 NSE NIFTY TRADING AGENT - SYNTHETIC DATA GENERATION")
    print("=" * 60)

    # Setup logging
    setup_logging({
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file_path': 'logs/synthetic_data.log',
        'max_file_size_mb': 50,
        'backup_count': 3
    })

    try:
        # Initialize generator
        generator = SyntheticDataGenerator(seed=42)

        # Generate data for training period (6 months back from now)
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=180)  # 6 months

        symbols = [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
            "HINDUNILVR.NS", "ITC.NS", "KOTAKBANK.NS", "LT.NS", "AXISBANK.NS",
            "BAJFINANCE.NS", "MARUTI.NS", "BAJAJ-AUTO.NS", "WIPRO.NS", "HCLTECH.NS",
            "ADANIPORTS.NS", "DIVISLAB.NS", "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS"
        ]

        logger.info(f"Generating synthetic data from {start_date} to {end_date} for {len(symbols)} symbols")

        # Generate data
        df = generator.generate_ohlcv_data(symbols, start_date.strftime('%Y-%m-%d'),
                                         end_date.strftime('%Y-%m-%d'))

        # Save to database
        success = generator.save_to_database(df)

        if success:
            print(f"✅ Successfully generated and saved {len(df)} synthetic OHLCV records")

            # Show summary
            print("\n📊 Data Summary:")
            print(f"   Date Range: {df['date'].min()} to {df['date'].max()}")
            print(f"   Symbols: {df['symbol'].nunique()}")
            print(f"   Trading Days per Symbol: {len(df) // df['symbol'].nunique()}")
            print(".2f")
            print(".2f")

            # Show sample data
            print("\n📋 Sample Data (first 5 rows):")
            sample = df.head()
            for _, row in sample.iterrows():
                print(f"   {row['symbol']}: {row['date']} | O:{row['open']:.2f} H:{row['high']:.2f} L:{row['low']:.2f} C:{row['close']:.2f} V:{row['volume']:,}")

            return 0
        else:
            logger.error("Failed to save data to database")
            return 1

    except Exception as e:
        logger.error(f"Synthetic data generation failed: {e}", exc_info=True)
        print(f"❌ Synthetic data generation failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
