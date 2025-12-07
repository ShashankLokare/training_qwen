#!/usr/bin/env python3
"""
Label Generation Script v2 for Nifty Trading Agent
Generates realistic, less rare labels for ML training without lookahead bias
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Dict

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.logging_utils import setup_logging, get_logger
from utils.duckdb_tools import load_ohlcv, store_features
from utils.io_utils import load_yaml_config

logger = get_logger(__name__)

class LabelGeneratorV2:
    """
    Generates realistic labels for v2 ML training
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize label generator v2

        Args:
            config_path: Path to configuration file
        """
        self.config = load_yaml_config(config_path)

    def generate_labels_for_symbols(self, symbols: List[str],
                                   start_date: str, end_date: str,
                                   index_symbol: str = "^NSEI") -> pd.DataFrame:
        """
        Generate v2 labels for specified symbols and date range

        Args:
            symbols: List of stock symbols
            start_date: Start date for label generation
            end_date: End date for label generation
            index_symbol: Nifty index symbol for relative labels

        Returns:
            DataFrame with v2 labels
        """
        logger.info(f"Generating v2 labels for {len(symbols)} symbols from {start_date} to {end_date}")

        all_labels = []

        # Load index data for relative labels
        index_data = self._load_index_data(index_symbol, start_date, end_date)

        for symbol in symbols:
            try:
                # Load OHLCV data with buffer for forward-looking calculations
                buffer_start = pd.to_datetime(start_date) - timedelta(days=15)
                buffer_end = pd.to_datetime(end_date) + timedelta(days=15)

                ohlcv_data = load_ohlcv([symbol], buffer_start.strftime('%Y-%m-%d'), buffer_end.strftime('%Y-%m-%d'))

                if ohlcv_data.empty:
                    logger.warning(f"No OHLCV data found for {symbol}")
                    continue

                # Generate labels for this symbol
                symbol_labels = self._generate_labels_for_symbol(
                    ohlcv_data, symbol, start_date, end_date, index_data
                )
                all_labels.append(symbol_labels)

                logger.info(f"Generated {len(symbol_labels)} v2 labels for {symbol}")

            except Exception as e:
                logger.error(f"Error generating v2 labels for {symbol}: {e}")
                continue

        if not all_labels:
            logger.warning("No v2 labels generated for any symbol")
            return pd.DataFrame()

        # Combine all symbol labels
        combined_labels = pd.concat(all_labels, ignore_index=True)
        combined_labels = combined_labels.sort_values(['symbol', 'date']).reset_index(drop=True)

        logger.info(f"Generated total of {len(combined_labels)} v2 labels")
        return combined_labels

    def _load_index_data(self, index_symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Load Nifty index data for relative labels

        Args:
            index_symbol: Index symbol (e.g., ^NSEI)
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with index OHLCV data
        """
        try:
            buffer_start = pd.to_datetime(start_date) - timedelta(days=15)
            buffer_end = pd.to_datetime(end_date) + timedelta(days=15)

            index_data = load_ohlcv([index_symbol], buffer_start.strftime('%Y-%m-%d'), buffer_end.strftime('%Y-%m-%d'))

            if index_data.empty:
                logger.warning(f"No index data found for {index_symbol}, using stock average as proxy")
                # Use average of available stocks as proxy for index
                return self._create_index_proxy(start_date, end_date)
            else:
                logger.info(f"Loaded index data for {index_symbol}")
                return index_data

        except Exception as e:
            logger.warning(f"Error loading index data: {e}, using proxy")
            return self._create_index_proxy(start_date, end_date)

    def _create_index_proxy(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Create a proxy index using average of available stocks

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with proxy index data
        """
        try:
            # Get a few major stocks to create index proxy
            proxy_symbols = ['HINDUNILVR.NS', 'ICICIBANK.NS', 'ITC.NS', 'BAJAJFINSV.NS']
            buffer_start = pd.to_datetime(start_date) - timedelta(days=15)
            buffer_end = pd.to_datetime(end_date) + timedelta(days=15)

            proxy_data = load_ohlcv(proxy_symbols, buffer_start.strftime('%Y-%m-%d'), buffer_end.strftime('%Y-%m-%d'))

            if proxy_data.empty:
                logger.error("Cannot create index proxy - no stock data available")
                return pd.DataFrame()

            # Calculate equal-weighted index
            pivot_data = proxy_data.pivot(index='date', columns='symbol', values='close')
            pivot_data['proxy_index'] = pivot_data.mean(axis=1)

            # Create index-like DataFrame
            index_df = pd.DataFrame({
                'symbol': '^NSEI_PROXY',
                'date': pivot_data.index,
                'close': pivot_data['proxy_index']
            })

            logger.info("Created proxy index using stock average")
            return index_df

        except Exception as e:
            logger.error(f"Error creating index proxy: {e}")
            return pd.DataFrame()

    def _generate_labels_for_symbol(self, ohlcv_data: pd.DataFrame, symbol: str,
                                   start_date: str, end_date: str,
                                   index_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate v2 labels for a single symbol

        Args:
            ohlcv_data: OHLCV data for the symbol
            symbol: Stock symbol
            start_date: Start date for labels
            end_date: End date for labels
            index_data: Index data for relative labels

        Returns:
            DataFrame with v2 labels for this symbol
        """
        # Ensure data is sorted by date
        ohlcv_data = ohlcv_data.sort_values('date').reset_index(drop=True)

        labels_data = []

        # Iterate through each date in the range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        for current_date in date_range:
            try:
                # Find the row for current date
                current_row = ohlcv_data[ohlcv_data['date'] == current_date]

                if current_row.empty:
                    continue

                current_price = current_row['close'].iloc[0]

                # Calculate forward returns
                fwd_returns = self._calculate_forward_returns(
                    ohlcv_data, current_date
                )

                if fwd_returns is None:
                    continue

                # Calculate index-relative return
                idx_fwd_return = self._calculate_index_forward_return(
                    index_data, current_date
                )

                # Generate v2 labels
                labels = self._create_v2_labels(current_price, fwd_returns, idx_fwd_return)

                # Create label record
                label_record = {
                    'symbol': symbol,
                    'date': current_date,
                    'fwd_5d_return': fwd_returns.get('5d_return'),
                    'fwd_10d_return': fwd_returns.get('10d_return'),
                    'idx_fwd_5d_return': idx_fwd_return,
                    'label_3p_5d': labels['label_3p_5d'],
                    'label_5p_10d': labels['label_5p_10d'],
                    'label_outperf_5d': labels['label_outperf_5d']
                }

                labels_data.append(label_record)

            except Exception as e:
                logger.debug(f"Error processing {symbol} on {current_date}: {e}")
                continue

        return pd.DataFrame(labels_data)

    def _calculate_forward_returns(self, ohlcv_data: pd.DataFrame,
                                  start_date: pd.Timestamp) -> Optional[Dict[str, float]]:
        """
        Calculate forward returns for v2 labels

        Args:
            ohlcv_data: OHLCV data
            start_date: Starting date

        Returns:
            Dictionary with forward return metrics
        """
        try:
            # Find starting price
            start_row = ohlcv_data[ohlcv_data['date'] == start_date]
            if start_row.empty:
                return None

            start_price = start_row['close'].iloc[0]

            # Calculate 5-day forward return
            future_5d = start_date + timedelta(days=5)
            future_5d_row = ohlcv_data[ohlcv_data['date'] == future_5d]
            if future_5d_row.empty:
                return None
            price_5d = future_5d_row['close'].iloc[0]
            fwd_5d_return = (price_5d / start_price) - 1

            # Calculate 10-day forward return
            future_10d = start_date + timedelta(days=10)
            future_10d_row = ohlcv_data[ohlcv_data['date'] == future_10d]
            if future_10d_row.empty:
                return None
            price_10d = future_10d_row['close'].iloc[0]
            fwd_10d_return = (price_10d / start_price) - 1

            return {
                '5d_return': fwd_5d_return,
                '10d_return': fwd_10d_return
            }

        except Exception as e:
            logger.debug(f"Error calculating forward returns: {e}")
            return None

    def _calculate_index_forward_return(self, index_data: pd.DataFrame,
                                       start_date: pd.Timestamp) -> Optional[float]:
        """
        Calculate 5-day forward return for the index

        Args:
            index_data: Index OHLCV data
            start_date: Starting date

        Returns:
            5-day forward return for index, or None
        """
        try:
            if index_data.empty:
                return 0.0  # Neutral if no index data

            # Find starting price
            start_row = index_data[index_data['date'] == start_date]
            if start_row.empty:
                return 0.0

            start_price = start_row['close'].iloc[0]

            # Calculate 5-day forward return
            future_5d = start_date + timedelta(days=5)
            future_5d_row = index_data[index_data['date'] == future_5d]
            if future_5d_row.empty:
                return 0.0

            price_5d = future_5d_row['close'].iloc[0]
            idx_fwd_return = (price_5d / start_price) - 1

            return idx_fwd_return

        except Exception as e:
            logger.debug(f"Error calculating index forward return: {e}")
            return 0.0

    def _create_v2_labels(self, start_price: float, fwd_returns: Dict[str, float],
                         idx_fwd_return: float) -> Dict[str, bool]:
        """
        Create v2 classification labels

        Args:
            start_price: Starting price
            fwd_returns: Forward return calculations
            idx_fwd_return: Index 5-day forward return

        Returns:
            Dictionary with v2 binary labels
        """
        fwd_5d_return = fwd_returns.get('5d_return', 0)
        fwd_10d_return = fwd_returns.get('10d_return', 0)

        # Label 1: 3% return in 5 days (more achievable)
        label_3p_5d = fwd_5d_return >= 0.03

        # Label 2: 5% return in 10 days (primary target)
        label_5p_10d = fwd_10d_return >= 0.05

        # Label 3: Outperform index by 1.5% in 5 days
        stock_outperformance = fwd_5d_return - idx_fwd_return
        label_outperf_5d = stock_outperformance >= 0.015

        return {
            'label_3p_5d': label_3p_5d,
            'label_5p_10d': label_5p_10d,
            'label_outperf_5d': label_outperf_5d
        }

def main():
    """Main v2 label generation function"""
    print("🚀 NIFTY TRADING AGENT - V2 LABEL GENERATION")
    print("=" * 50)

    # Setup logging
    setup_logging({
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file_path': 'logs/generate_labels_v2.log',
        'max_file_size_mb': 50,
        'backup_count': 3
    })

    try:
        # Load configuration
        config = load_yaml_config("config/config.yaml")

        # Initialize v2 label generator
        generator = LabelGeneratorV2()

        # Get symbols and date range
        symbols = config.get('universe', {}).get('tickers', [])

        # Use v2 training date ranges (will be updated in config)
        model_params = config.get('model_params', {})
        start_date = model_params.get('training_start_date', '2015-01-01')
        end_date = model_params.get('training_end_date', '2022-12-31')

        logger.info(f"Generating v2 labels for {len(symbols)} symbols from {start_date} to {end_date}")

        # Generate v2 labels
        labels_df = generator.generate_labels_for_symbols(symbols, start_date, end_date)

        if labels_df.empty:
            logger.error("No v2 labels generated")
            return 1

        # Store in DuckDB (extend features_nifty with v2 columns)
        success = store_features(labels_df)
        if success:
            logger.info(f"Successfully stored {len(labels_df)} v2 label records")

            # Print v2 label statistics
            print("\n📊 V2 Label Statistics:")
            print("=" * 30)

            total_labels = len(labels_df)

            # Label 3p_5d stats
            pos_3p_5d = labels_df['label_3p_5d'].sum()
            rate_3p_5d = pos_3p_5d / total_labels * 100

            # Label 5p_10d stats (primary target)
            pos_5p_10d = labels_df['label_5p_10d'].sum()
            rate_5p_10d = pos_5p_10d / total_labels * 100

            # Label outperf_5d stats
            pos_outperf = labels_df['label_outperf_5d'].sum()
            rate_outperf = pos_outperf / total_labels * 100

            print(f"   Total labels: {total_labels}")
            print(f"   Symbols covered: {labels_df['symbol'].nunique()}")
            print()
            print(f"   Label 3p_5d (+3% in 5 days):")
            print(f"     Positive: {pos_3p_5d} ({rate_3p_5d:.1f}%)")
            print()
            print(f"   Label 5p_10d (+5% in 10 days) - PRIMARY TARGET:")
            print(f"     Positive: {pos_5p_10d} ({rate_5p_10d:.1f}%)")
            print()
            print(f"   Label outperf_5d (outperform index by 1.5% in 5 days):")
            print(f"     Positive: {pos_outperf} ({rate_outperf:.1f}%)")

            # Check if rates are in target range (5-20%)
            if 5 <= rate_5p_10d <= 20:
                print(f"\n✅ Primary target label_5p_10d has realistic rate: {rate_5p_10d:.1f}%")
            else:
                print(f"\n⚠️  Primary target label_5p_10d rate {rate_5p_10d:.1f}% outside 5-20% target range")

            # Yearly breakdown
            labels_df['year'] = labels_df['date'].dt.year
            yearly_stats = labels_df.groupby('year').agg({
                'label_5p_10d': ['count', 'sum', lambda x: x.sum()/x.count()*100]
            }).round(1)
            yearly_stats.columns = ['total', 'positive', 'rate_pct']

            print(f"\n   Yearly breakdown for label_5p_10d:")
            print(yearly_stats.to_string())

            return 0
        else:
            logger.error("Failed to store v2 labels")
            return 1

    except Exception as e:
        logger.error(f"V2 label generation failed: {e}", exc_info=True)
        print(f"❌ V2 label generation failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
