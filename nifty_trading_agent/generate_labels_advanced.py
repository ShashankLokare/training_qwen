#!/usr/bin/env python3
"""
Advanced Label Generation Script for Nifty Trading Agent
Generates forward-looking labels with multiple horizons and trading-oriented labels
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.logging_utils import setup_logging, get_logger
from utils.duckdb_tools import load_ohlcv, store_features
from utils.io_utils import load_yaml_config

logger = get_logger(__name__)

class AdvancedLabelGenerator:
    """
    Generates advanced forward-looking labels for ML training
    Supports multiple horizons and trading-oriented labels
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize advanced label generator

        Args:
            config_path: Path to configuration file
        """
        self.config = load_yaml_config(config_path)
        self.prediction_horizons = [10, 20]  # 10-day and 20-day horizons
        self.target_return_pct = 10.0  # +10% target
        self.stop_loss_pct = 5.0  # -5% stop loss

    def generate_labels_for_symbols(self, symbols: List[str],
                                   start_date: str, end_date: str) -> pd.DataFrame:
        """
        Generate advanced labels for specified symbols and date range

        Args:
            symbols: List of stock symbols
            start_date: Start date for label generation
            end_date: End date for label generation

        Returns:
            DataFrame with advanced labels
        """
        logger.info(f"Generating advanced labels for {len(symbols)} symbols from {start_date} to {end_date}")

        all_labels = []

        for symbol in symbols:
            try:
                # Load OHLCV data with buffer for forward-looking calculations
                buffer_days = max(self.prediction_horizons) + 30  # Extra buffer
                buffer_start = pd.to_datetime(start_date) - timedelta(days=buffer_days)
                buffer_end = pd.to_datetime(end_date) + timedelta(days=buffer_days)

                ohlcv_data = load_ohlcv([symbol], buffer_start.strftime('%Y-%m-%d'),
                                       buffer_end.strftime('%Y-%m-%d'))

                if ohlcv_data.empty or len(ohlcv_data) < 100:
                    logger.warning(f"Insufficient data for {symbol}: {len(ohlcv_data)} records")
                    continue

                # Generate labels for this symbol
                symbol_labels = self._generate_labels_for_symbol(ohlcv_data, symbol, start_date, end_date)
                if not symbol_labels.empty:
                    all_labels.append(symbol_labels)
                    logger.info(f"Generated {len(symbol_labels)} labels for {symbol}")

            except Exception as e:
                logger.error(f"Error generating labels for {symbol}: {e}")
                continue

        if not all_labels:
            logger.warning("No labels generated for any symbol")
            return pd.DataFrame()

        # Combine all symbol labels
        combined_labels = pd.concat(all_labels, ignore_index=True)
        combined_labels = combined_labels.sort_values(['symbol', 'date']).reset_index(drop=True)

        logger.info(f"Generated total of {len(combined_labels)} advanced labels")
        return combined_labels

    def _generate_labels_for_symbol(self, ohlcv_data: pd.DataFrame, symbol: str,
                                   start_date: str, end_date: str) -> pd.DataFrame:
        """
        Generate advanced labels for a single symbol

        Args:
            ohlcv_data: OHLCV data for the symbol
            symbol: Stock symbol
            start_date: Start date for labels
            end_date: End date for labels

        Returns:
            DataFrame with advanced labels for this symbol
        """
        # Ensure data is sorted by date
        ohlcv_data = ohlcv_data.sort_values('date').reset_index(drop=True)

        labels_data = []

        # Iterate through each date in the range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        for current_date in date_range:
            try:
                # Skip if not a trading day or no data
                if not ohlcv_data[ohlcv_data['date'] == current_date].any().any():
                    continue

                # Calculate forward returns for all horizons
                forward_returns = {}
                valid_horizons = []

                for horizon in self.prediction_horizons:
                    returns = self._calculate_forward_returns(
                        ohlcv_data, current_date, horizon
                    )
                    if returns is not None:
                        forward_returns[horizon] = returns
                        valid_horizons.append(horizon)

                if not valid_horizons:
                    continue

                # Generate trading-oriented labels
                labels = self._create_advanced_labels(forward_returns)

                # Create label record
                label_record = {
                    'symbol': symbol,
                    'date': current_date,
                }

                # Add forward returns for all horizons
                for horizon in valid_horizons:
                    returns = forward_returns[horizon]
                    label_record[f'forward_{horizon}d_max_return_pct'] = returns['max_return_pct']
                    label_record[f'forward_{horizon}d_min_return_pct'] = returns['min_return_pct']
                    label_record[f'forward_{horizon}d_volatility'] = returns['volatility']

                # Add labels
                label_record.update(labels)

                labels_data.append(label_record)

            except Exception as e:
                logger.debug(f"Error processing {symbol} on {current_date}: {e}")
                continue

        return pd.DataFrame(labels_data)

    def _calculate_forward_returns(self, ohlcv_data: pd.DataFrame,
                                  start_date: pd.Timestamp, horizon_days: int) -> Optional[Dict[str, float]]:
        """
        Calculate comprehensive forward returns over the next N trading days

        Args:
            ohlcv_data: OHLCV data
            start_date: Starting date
            horizon_days: Number of days to look forward

        Returns:
            Dictionary with forward return metrics
        """
        try:
            # Find starting price
            start_row = ohlcv_data[ohlcv_data['date'] == start_date]
            if start_row.empty:
                return None

            start_price = start_row['close'].iloc[0]

            # Get future prices for the next horizon_days
            future_data = ohlcv_data[
                (ohlcv_data['date'] > start_date) &
                (ohlcv_data['date'] <= start_date + timedelta(days=horizon_days))
            ]

            if len(future_data) < max(5, horizon_days // 2):  # Need reasonable amount of data
                return None

            # Calculate returns for each future day
            future_prices = future_data['close'].values
            daily_returns = (future_prices - start_price) / start_price

            # Return metrics
            max_return_pct = daily_returns.max() * 100
            min_return_pct = daily_returns.min() * 100
            volatility = daily_returns.std() * 100  # Daily volatility over period

            # Check target and stop loss achievement
            target_price = start_price * (1 + self.target_return_pct / 100)
            stop_price = start_price * (1 - self.stop_loss_pct / 100)

            hit_target = (future_prices >= target_price).any()
            hit_stop = (future_prices <= stop_price).any()

            # Find first day we hit target or stop
            first_target_day = None
            first_stop_day = None

            for i, price in enumerate(future_prices):
                if first_target_day is None and price >= target_price:
                    first_target_day = i + 1
                if first_stop_day is None and price <= stop_price:
                    first_stop_day = i + 1

            return {
                'max_return_pct': max_return_pct,
                'min_return_pct': min_return_pct,
                'volatility': volatility,
                'hit_target': hit_target,
                'hit_stop': hit_stop,
                'first_target_day': first_target_day,
                'first_stop_day': first_stop_day,
                'avg_daily_return': daily_returns.mean() * 100,
                'total_return': daily_returns[-1] * 100 if len(daily_returns) > 0 else 0
            }

        except Exception as e:
            logger.debug(f"Error calculating forward returns: {e}")
            return None

    def _create_advanced_labels(self, forward_returns: Dict[int, Dict[str, float]]) -> Dict[str, int]:
        """
        Create advanced binary labels based on forward returns across horizons

        Args:
            forward_returns: Forward return calculations for each horizon

        Returns:
            Dictionary with binary labels
        """
        labels = {}

        # Get the best horizon (10-day by default, but check if 20-day is available)
        primary_horizon = 10
        alt_horizon = 20

        primary_returns = forward_returns.get(primary_horizon)
        alt_returns = forward_returns.get(alt_horizon)

        if primary_returns is None:
            return {}  # No valid labels

        # Main labels: hit +10% within horizon
        labels['label_up_10pct_10d'] = 1 if primary_returns['hit_target'] else 0
        if alt_returns:
            labels['label_up_10pct_20d'] = 1 if alt_returns['hit_target'] else 0

        # Hit target before stop loss (win before loss)
        labels['label_win_before_loss_10d'] = 1 if (
            primary_returns['hit_target'] and
            (primary_returns['first_target_day'] or 999) < (primary_returns['first_stop_day'] or 999)
        ) else 0

        if alt_returns:
            labels['label_win_before_loss_20d'] = 1 if (
                alt_returns['hit_target'] and
                (alt_returns['first_target_day'] or 999) < (alt_returns['first_stop_day'] or 999)
            ) else 0

        # Hit stop loss before target (loss before win)
        labels['label_loss_before_win_10d'] = 1 if (
            primary_returns['hit_stop'] and
            (primary_returns['first_stop_day'] or 999) < (primary_returns['first_target_day'] or 999)
        ) else 0

        if alt_returns:
            labels['label_loss_before_win_20d'] = 1 if (
                alt_returns['hit_stop'] and
                (alt_returns['first_stop_day'] or 999) < (alt_returns['first_target_day'] or 999)
            ) else 0

        # Volatility-based labels
        if primary_returns['volatility'] < 2.0:  # Low volatility
            labels['label_low_vol_up_10pct'] = labels['label_up_10pct_10d']
        else:  # High volatility
            labels['label_high_vol_up_10pct'] = labels['label_up_10pct_10d']

        return labels

def _print_label_summary_static(df: pd.DataFrame):
    """Print detailed label summary"""
    print("\n📊 ADVANCED LABEL STATISTICS")
    print("=" * 50)

    total_labels = len(df)
    symbols_count = df['symbol'].nunique()

    print(f"Total labels: {total_labels:,}")
    print(f"Symbols covered: {symbols_count}")
    print(f"Average labels per symbol: {total_labels/symbols_count:.0f}")

    # 10-day horizon stats
    if 'label_up_10pct_10d' in df.columns:
        positive_10d = df['label_up_10pct_10d'].sum()
        rate_10d = positive_10d / total_labels * 100
        print("\n🎯 10-Day Horizon:")
        print(f"   Positive labels (+10%): {positive_10d:,} ({rate_10d:.1f}%)")

        win_before_loss = df['label_win_before_loss_10d'].sum()
        loss_before_win = df['label_loss_before_win_10d'].sum()
        print(f"   Win before loss: {win_before_loss:,}")
        print(f"   Loss before win: {loss_before_win:,}")

    # 20-day horizon stats
    if 'label_up_10pct_20d' in df.columns:
        positive_20d = df['label_up_10pct_20d'].sum()
        rate_20d = positive_20d / total_labels * 100
        print("\n🎯 20-Day Horizon:")
        print(f"   Positive labels (+10%): {positive_20d:,} ({rate_20d:.1f}%)")

    # Forward returns statistics
    if 'forward_10d_max_return_pct' in df.columns:
        avg_max_return = df['forward_10d_max_return_pct'].mean()
        median_max_return = df['forward_10d_max_return_pct'].median()
        print("\n📈 Forward Returns (10-day):")
        print(".1f")
        print(".1f")

def main():
    """Main advanced label generation function"""
    print("🚀 NIFTY TRADING AGENT - ADVANCED LABEL GENERATION")
    print("=" * 60)

    # Setup logging
    setup_logging({
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file_path': 'logs/advanced_label_generation.log',
        'max_file_size_mb': 50,
        'backup_count': 3
    })

    try:
        # Load configuration
        config = load_yaml_config("config/config.yaml")

        # Initialize advanced label generator
        generator = AdvancedLabelGenerator()

        # Get symbols and date range
        symbols = config.get('universe', {}).get('tickers', [])

        # Use extended training date ranges for real data
        model_params = config.get('model_params', {})
        start_date = "2018-01-01"  # 7 years back from 2025
        end_date = "2024-06-30"   # Leave room for forward-looking labels

        logger.info(f"Generating advanced labels for {len(symbols)} symbols from {start_date} to {end_date}")

        # Generate labels
        labels_df = generator.generate_labels_for_symbols(symbols, start_date, end_date)

        if labels_df.empty:
            logger.error("No labels generated")
            return 1

        # Save to DuckDB
        success = store_features(labels_df)
        if success:
            logger.info(f"Successfully stored {len(labels_df)} advanced label records")
            print(f"✅ Generated and stored {len(labels_df)} advanced labels")

            # Print detailed summary statistics
            _print_label_summary_static(labels_df)

            return 0
        else:
            logger.error("Failed to store labels")
            return 1

    except Exception as e:
        logger.error(f"Advanced label generation failed: {e}", exc_info=True)
        print(f"❌ Advanced label generation failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
