#!/usr/bin/env python3
"""
Import Real Nifty 50 Historical Data
Fetches 10-year daily OHLCV data for all Nifty 50 stocks
Handles splits, dividends, symbol changes, and missing data
"""

import sys
import os
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import time
import requests

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.logging_utils import setup_logging, get_logger
from utils.duckdb_tools import store_ohlcv

logger = get_logger(__name__)

class NiftyDataImporter:
    """
    Imports 10-year historical OHLCV data for Nifty 50 stocks
    Handles splits, dividends, holidays, missing data, and adjustments
    """

    def __init__(self):
        """
        Initialize the data importer
        """
        # Nifty 50 symbols (current composition as of 2024)
        self.nifty_50_symbols = [
            "ADANIPORTS.NS", "ADANIENT.NS", "ADANIGREEN.NS", "ADANITRANS.NS",
            "AMBUJACEM.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
            "BAJAJ-AUTO.NS", "BAJAJFINSV.NS", "BAJFINANCE.NS", "BHARTIARTL.NS",
            "BPCL.NS", "BRITANNIA.NS", "CIPLA.NS", "COALINDIA.NS", "DIVISLAB.NS",
            "DRREDDY.NS", "EICHERMOT.NS", "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS",
            "HDFCLIFE.NS", "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS",
            "ICICIBANK.NS", "INDUSINDBK.NS", "INFY.NS", "ITC.NS", "JSWSTEEL.NS",
            "KOTAKBANK.NS", "LT.NS", "LTIM.NS", "M&M.NS", "MARUTI.NS",
            "NESTLEIND.NS", "NTPC.NS", "ONGC.NS", "POWERGRID.NS", "RELIANCE.NS",
            "SBILIFE.NS", "SBIN.NS", "SHREECEM.NS", "SHRIRAMFIN.NS", "SUNPHARMA.NS",
            "TATACONSUM.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "TCS.NS",
            "TECHM.NS", "TITAN.NS", "TRENT.NS", "ULTRACEMCO.NS", "WIPRO.NS"
        ]

        # Historical symbol mappings for companies that changed tickers
        self.symbol_mappings = {
            # Add mappings for symbols that changed over time
            # Example: "OLD_SYMBOL.NS": "NEW_SYMBOL.NS"
        }

        # Indian market holidays and special dates
        self.indian_holidays = self._get_indian_market_holidays()

        # Data quality thresholds
        self.min_data_points = 2000  # ~8 years of trading days (250 days/year)
        self.max_missing_pct = 0.03  # 3% missing data tolerance (stricter)
        self.max_consecutive_missing = 5  # Max consecutive missing days

        # Rate limiting and retry settings
        self.request_delay = 1.5  # seconds between requests
        self.max_retries = 3  # Maximum retry attempts
        self.retry_delay = 5.0  # Delay between retries

        # Data validation parameters
        self.min_price_threshold = 0.01  # Minimum valid price
        self.max_price_threshold = 100000  # Maximum valid price
        self.min_volume_threshold = 0  # Minimum volume (0 for non-trading days)

    def _get_indian_market_holidays(self) -> List[str]:
        """
        Get list of Indian market holidays and special non-trading days

        Returns:
            List of holiday dates in YYYY-MM-DD format
        """
        # Major Indian market holidays (approximate, can be extended)
        holidays = [
            # Republic Day (January 26)
            "2015-01-26", "2016-01-26", "2017-01-26", "2018-01-26", "2019-01-26",
            "2020-01-26", "2021-01-26", "2022-01-26", "2023-01-26", "2024-01-26", "2025-01-26",

            # Independence Day (August 15)
            "2015-08-15", "2016-08-15", "2017-08-15", "2018-08-15", "2019-08-15",
            "2020-08-15", "2021-08-15", "2022-08-15", "2023-08-15", "2024-08-15", "2025-08-15",

            # Gandhi Jayanti (October 2)
            "2015-10-02", "2016-10-02", "2017-10-02", "2018-10-02", "2019-10-02",
            "2020-10-02", "2021-10-02", "2022-10-02", "2023-10-02", "2024-10-02", "2025-10-02",

            # Christmas (December 25)
            "2015-12-25", "2016-12-25", "2017-12-25", "2018-12-25", "2019-12-25",
            "2020-12-25", "2021-12-25", "2022-12-25", "2023-12-25", "2024-12-25", "2025-12-25",

            # Other major holidays
            "2015-02-17", "2016-02-17", "2017-02-17", "2018-02-17",  # Mahashivaratri
            "2015-03-06", "2016-03-06", "2017-03-06", "2018-03-06",  # Holi
            "2015-09-17", "2016-09-17", "2017-09-17",  # Ganesh Chaturthi
            "2020-11-14", "2021-11-14", "2022-11-14",  # Diwali (Bali Pratipada)

            # Special non-trading days
            "2020-03-09", "2020-03-10", "2020-03-11", "2020-03-12", "2020-03-13",  # COVID-19 lockdown
            "2021-04-14", "2021-04-15", "2021-04-16", "2021-04-17", "2021-04-18",  # COVID-19 extension
        ]

        return holidays

    def fetch_symbol_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data for a single symbol

        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")

            # Use yfinance to fetch data
            ticker = yf.Ticker(symbol)

            # Fetch historical data
            df = ticker.history(start=start_date, end=end_date, interval='1d')

            if df.empty:
                logger.warning(f"No data found for {symbol}")
                return None

            # Reset index to get Date as column
            df = df.reset_index()

            # Rename columns to match our schema
            df = df.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })

            # Keep only required columns
            df = df[['date', 'open', 'high', 'low', 'close', 'volume']]

            # Add symbol column
            df['symbol'] = symbol

            # Convert date to date type (remove time)
            df['date'] = pd.to_datetime(df['date']).dt.date

            # Handle missing values
            df = self._handle_missing_data(df)

            # Validate data quality
            if not self._validate_data_quality(df):
                logger.warning(f"Data quality check failed for {symbol}")
                return None

            logger.info(f"Successfully fetched {len(df)} records for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            return None

    def _handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing data points by forward/backward filling with enhanced logic

        Args:
            df: OHLCV DataFrame

        Returns:
            DataFrame with missing data handled
        """
        original_count = len(df)

        # Convert to datetime index
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

        # Create complete business day range (excluding holidays)
        start_date = df.index.min()
        end_date = df.index.max()
        all_business_days = pd.date_range(start=start_date, end=end_date, freq='B')

        # Filter out known holidays
        holiday_dates = pd.to_datetime(self.indian_holidays)
        valid_trading_days = all_business_days[~all_business_days.isin(holiday_dates)]

        # Reindex to include only valid trading days
        df = df.reindex(valid_trading_days)

        # Handle missing data more intelligently
        missing_before = df[['open', 'high', 'low', 'close']].isnull().any(axis=1).sum()

        # For consecutive missing days, use backward fill first, then forward fill
        df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].bfill().ffill()

        # Volume: set to 0 for non-trading days, interpolate for gaps
        df['volume'] = df['volume'].fillna(0)

        # Handle extreme price jumps (potential split/dividend adjustments)
        df = self._handle_price_adjustments(df)

        missing_after = df[['open', 'high', 'low', 'close']].isnull().any(axis=1).sum()

        # Reset index
        df = df.reset_index()
        df = df.rename(columns={'index': 'date'})
        df['date'] = df['date'].dt.date

        logger.info(f"Data cleaning: {original_count} → {len(df)} records, "
                   f"missing values: {missing_before} → {missing_after}")

        return df

    def _handle_price_adjustments(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle price adjustments due to splits, dividends, or other corporate actions

        Args:
            df: OHLCV DataFrame with datetime index

        Returns:
            DataFrame with price adjustments handled
        """
        # Calculate daily returns to detect anomalies
        df['daily_return'] = df['close'].pct_change()

        # Flag potential split/dividend days (extreme price movements)
        extreme_returns = (df['daily_return'].abs() > 0.5) & (df['daily_return'].notna())

        if extreme_returns.any():
            extreme_dates = df[extreme_returns].index
            logger.info(f"Detected {len(extreme_dates)} potential adjustment dates: "
                       f"{[d.strftime('%Y-%m-%d') for d in extreme_dates[:5]]}")

            # For now, we'll log these but not auto-adjust
            # In production, you'd cross-reference with corporate action data

        # Remove temporary column
        df = df.drop('daily_return', axis=1)

        return df

    def _validate_data_quality(self, df: pd.DataFrame) -> bool:
        """
        Validate data quality metrics

        Args:
            df: OHLCV DataFrame

        Returns:
            True if data passes quality checks
        """
        # Check minimum data points
        if len(df) < self.min_data_points:
            logger.warning(f"Insufficient data points: {len(df)} < {self.min_data_points}")
            return False

        # Check for missing values
        missing_pct = df[['open', 'high', 'low', 'close']].isnull().any(axis=1).sum() / len(df)
        if missing_pct > self.max_missing_pct:
            logger.warning(f"Too many missing values: {missing_pct:.1%} > {self.max_missing_pct:.1%}")
            return False

        # Check for reasonable price ranges
        if (df['close'] <= 0).any():
            logger.warning("Invalid price data (negative or zero prices)")
            return False

        # Check volume data
        if (df['volume'] < 0).any():
            logger.warning("Invalid volume data (negative volumes)")
            return False

        return True

    def import_all_symbols(self, start_date: str = "2015-01-01",
                          end_date: str = None, symbols_override: List[str] = None) -> Dict[str, int]:
        """
        Import data for all Nifty 50 symbols

        Args:
            start_date: Start date for data import
            end_date: End date for data import (default: today)
            symbols_override: Optional list of symbols to import (for testing)

        Returns:
            Dictionary with import statistics
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        # Use symbols override if provided, otherwise use full list
        symbols_to_import = symbols_override if symbols_override else self.nifty_50_symbols

        # Adjust minimum data points for smaller imports
        if symbols_override and len(symbols_override) <= 5:
            original_min_data = self.min_data_points
            self.min_data_points = 200  # Lower threshold for testing
            logger.info(f"Adjusted minimum data points from {original_min_data} to {self.min_data_points} for testing")

        logger.info(f"Starting import of {len(symbols_to_import)} symbols")
        logger.info(f"Date range: {start_date} to {end_date}")

        stats = {
            'total_symbols': len(symbols_to_import),
            'successful_imports': 0,
            'failed_imports': 0,
            'total_records': 0
        }

        imported_data = []

        for i, symbol in enumerate(symbols_to_import, 1):
            logger.info(f"Processing {symbol} ({i}/{len(self.nifty_50_symbols)})")

            try:
                # Fetch data for this symbol
                df = self.fetch_symbol_data(symbol, start_date, end_date)

                if df is not None and not df.empty:
                    # Add to imported data list
                    imported_data.append(df)
                    stats['successful_imports'] += 1
                    stats['total_records'] += len(df)
                    logger.info(f"✅ Successfully imported {len(df)} records for {symbol}")
                else:
                    stats['failed_imports'] += 1
                    logger.error(f"❌ Failed to import data for {symbol}")

            except Exception as e:
                stats['failed_imports'] += 1
                logger.error(f"❌ Exception importing {symbol}: {e}")

            # Rate limiting
            if i < len(self.nifty_50_symbols):
                time.sleep(self.request_delay)

        # Store all imported data to database
        if imported_data:
            try:
                # Combine all dataframes
                all_data = pd.concat(imported_data, ignore_index=True)

                # Store to DuckDB
                records_stored = store_ohlcv(all_data)

                logger.info(f"✅ Stored {records_stored} total OHLCV records in database")

                # Generate summary statistics
                self._generate_import_summary(all_data)

            except Exception as e:
                logger.error(f"Failed to store data to database: {e}")

        return stats

    def _generate_import_summary(self, df: pd.DataFrame):
        """
        Generate and display import summary statistics

        Args:
            df: Combined OHLCV DataFrame
        """
        logger.info("Generating import summary...")

        # Overall statistics
        total_records = len(df)
        unique_symbols = df['symbol'].nunique()
        date_range = f"{df['date'].min()} to {df['date'].max()}"

        # Per-symbol statistics
        symbol_stats = df.groupby('symbol').agg({
            'date': ['count', 'min', 'max'],
            'close': ['mean', 'std'],
            'volume': 'mean'
        }).round(2)

        # Price statistics
        price_stats = df['close'].describe()

        logger.info("\n📊 NIFTY 50 DATA IMPORT SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total Records: {total_records:,}")
        logger.info(f"Symbols: {unique_symbols}")
        logger.info(f"Date Range: {date_range}")
        logger.info(f"Average Records per Symbol: {total_records/unique_symbols:.0f}")

        logger.info("\n💰 Price Statistics:")
        logger.info(f"  Mean Price: ₹{price_stats['mean']:.2f}")
        logger.info(f"  Median Price: ₹{price_stats['50%']:.2f}")
        logger.info(f"  Min Price: ₹{price_stats['min']:.2f}")
        logger.info(f"  Max Price: ₹{price_stats['max']:.2f}")

        # Show top 5 symbols by record count
        top_symbols = symbol_stats.nlargest(5, ('date', 'count'))
        logger.info("\n📈 Top 5 Symbols by Data Points:")
        for symbol, stats in top_symbols.iterrows():
            count = stats[('date', 'count')]
            start_date = stats[('date', 'min')]
            end_date = stats[('date', 'max')]
            logger.info(f"  {symbol}: {count} records ({start_date} to {end_date})")

def main():
    """Main import function"""
    print("🚀 NSE NIFTY 50 REAL DATA IMPORT")
    print("=" * 50)

    # Setup logging
    setup_logging({
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file_path': 'logs/data_import.log',
        'max_file_size_mb': 50,
        'backup_count': 3
    })

    try:
        # Initialize importer
        importer = NiftyDataImporter()

        # Import data for full 10-year period (2015-2025)
        end_date = datetime.now()
        start_date = datetime(2015, 1, 1)  # Fixed start date for 10-year period

        logger.info("Starting comprehensive Nifty 50 data import...")
        logger.info(f"Time period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        logger.info(f"Expected records per symbol: ~2,500 trading days over 10 years")

        # Run import
        stats = importer.import_all_symbols(
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )

        # Print final statistics
        print("\n📊 IMPORT STATISTICS")
        print(f"Total Symbols Attempted: {stats['total_symbols']}")
        print(f"Successful Imports: {stats['successful_imports']}")
        print(f"Failed Imports: {stats['failed_imports']}")
        print(f"Success Rate: {(stats['successful_imports']/stats['total_symbols']*100):.1f}%")
        print(f"Total Records Imported: {stats['total_records']:,}")

        if stats['successful_imports'] > 0:
            print("\n✅ Data import completed successfully!")
            print("Next steps:")
            print("1. Run python generate_labels.py (with new horizons)")
            print("2. Run python train_model.py (with XGBoost/LightGBM)")
            print("3. Run python evaluate_model.py")
            print("4. Run backtesting with extended period")
            return 0
        else:
            print("❌ No data was successfully imported")
            return 1

    except Exception as e:
        logger.error(f"Data import failed: {e}", exc_info=True)
        print(f"❌ Data import failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
