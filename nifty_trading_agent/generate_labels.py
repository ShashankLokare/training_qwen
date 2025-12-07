#!/usr/bin/env python3
"""
Label Generation Script for Nifty Trading Agent
Generates forward-looking labels for ML training without lookahead bias
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.logging_utils import setup_logging, get_logger
from utils.duckdb_tools import load_ohlcv, store_features
from utils.io_utils import load_yaml_config

logger = get_logger(__name__)

class LabelGenerator:
    """
    Generates forward-looking labels for ML training
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize label generator

        Args:
            config_path: Path to configuration file
        """
        self.config = load_yaml_config(config_path)
        self.horizon_days = self.config.get('model_params', {}).get('prediction_horizon_days', 10)
        self.stop_loss_pct = self.config.get('risk_settings', {}).get('stop_loss_atr_multiplier', 1.5) * 0.05  # Rough stop loss

    def generate_labels_for_symbols(self, symbols: List[str],
                                   start_date: str, end_date: str) -> pd.DataFrame:
        """
        Generate labels for specified symbols and date range

        Args:
            symbols: List of stock symbols
            start_date: Start date for label generation
            end_date: End date for label generation

        Returns:
            DataFrame with labels
        """
        logger.info(f"Generating labels for {len(symbols)} symbols from {start_date} to {end_date}")

        all_labels = []

        for symbol in symbols:
            try:
                # Load OHLCV data with buffer for forward-looking calculations
                buffer_start = pd.to_datetime(start_date) - timedelta(days=self.horizon_days + 10)
                buffer_end = pd.to_datetime(end_date) + timedelta(days=self.horizon_days + 10)

                ohlcv_data = load_ohlcv([symbol], buffer_start.strftime('%Y-%m-%d'), buffer_end.strftime('%Y-%m-%d'))

                if ohlcv_data.empty:
                    logger.warning(f"No OHLCV data found for {symbol}")
                    continue

                # Generate labels for this symbol
                symbol_labels = self._generate_labels_for_symbol(ohlcv_data, symbol, start_date, end_date)
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

        logger.info(f"Generated total of {len(combined_labels)} labels")
        return combined_labels

    def _generate_labels_for_symbol(self, ohlcv_data: pd.DataFrame, symbol: str,
                                   start_date: str, end_date: str) -> pd.DataFrame:
        """
        Generate labels for a single symbol

        Args:
            ohlcv_data: OHLCV data for the symbol
            symbol: Stock symbol
            start_date: Start date for labels
            end_date: End date for labels

        Returns:
            DataFrame with labels for this symbol
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

                # Calculate forward returns over the next N trading days
                forward_returns = self._calculate_forward_returns(
                    ohlcv_data, current_date, self.horizon_days
                )

                if forward_returns is None:
                    continue

                # Generate labels
                labels = self._create_labels(current_price, forward_returns)

                # Create label record
                label_record = {
                    'symbol': symbol,
                    'date': current_date,
                    'forward_10d_max_return_pct': forward_returns['max_return_pct'],
                    'label_up_10pct': labels['up_10pct'],
                    'label_win_before_loss': labels['win_before_loss'],
                    'label_loss_before_win': labels['loss_before_win']
                }

                labels_data.append(label_record)

            except Exception as e:
                logger.debug(f"Error processing {symbol} on {current_date}: {e}")
                continue

        return pd.DataFrame(labels_data)

    def _calculate_forward_returns(self, ohlcv_data: pd.DataFrame,
                                  start_date: pd.Timestamp, horizon_days: int) -> Optional[Dict[str, float]]:
        """
        Calculate forward returns over the next N trading days

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

            if future_data.empty:
                return None

            # Calculate returns for each future day
            future_prices = future_data['close'].values
            daily_returns = (future_prices - start_price) / start_price

            # Find maximum return over the period
            max_return_pct = daily_returns.max() * 100

            # Check if we hit stop loss before target
            stop_loss_price = start_price * (1 - self.stop_loss_pct)
            target_price = start_price * 1.10  # +10%

            # Find first day we hit stop loss or target
            for i, price in enumerate(future_prices):
                if price <= stop_loss_price:
                    # Hit stop loss first
                    return {
                        'max_return_pct': max_return_pct,
                        'hit_stop_loss_first': True,
                        'hit_target_first': False,
                        'days_to_event': i + 1
                    }
                elif price >= target_price:
                    # Hit target first
                    return {
                        'max_return_pct': max_return_pct,
                        'hit_stop_loss_first': False,
                        'hit_target_first': True,
                        'days_to_event': i + 1
                    }

            # Neither target nor stop loss hit within horizon
            return {
                'max_return_pct': max_return_pct,
                'hit_stop_loss_first': False,
                'hit_target_first': False,
                'days_to_event': None
            }

        except Exception as e:
            logger.debug(f"Error calculating forward returns: {e}")
            return None

    def _create_labels(self, start_price: float, forward_returns: Dict[str, float]) -> Dict[str, bool]:
        """
        Create binary labels based on forward returns

        Args:
            start_price: Starting price
            forward_returns: Forward return calculations

        Returns:
            Dictionary with binary labels
        """
        max_return_pct = forward_returns['max_return_pct']

        # Main label: did we achieve +10% return within horizon?
        label_up_10pct = max_return_pct >= 10.0

        # Trading-oriented labels
        hit_stop_loss_first = forward_returns.get('hit_stop_loss_first', False)
        hit_target_first = forward_returns.get('hit_target_first', False)

        # Win before loss: hit +10% before -5%
        label_win_before_loss = hit_target_first

        # Loss before win: hit -5% before +10%
        label_loss_before_win = hit_stop_loss_first

        return {
            'up_10pct': label_up_10pct,
            'win_before_loss': label_win_before_loss,
            'loss_before_win': label_loss_before_win
        }

def main():
    """Main label generation function"""
    print("🚀 NIFTY TRADING AGENT - LABEL GENERATION")
    print("=" * 50)

    # Setup logging
    setup_logging({
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file_path': 'logs/label_generation.log',
        'max_file_size_mb': 50,
        'backup_count': 3
    })

    try:
        # Load configuration
        config = load_yaml_config("config/config.yaml")

        # Initialize label generator
        generator = LabelGenerator()

        # Get symbols and date range
        symbols = config.get('universe', {}).get('tickers', [])

        # Use model training date ranges
        model_params = config.get('model_params', {})
        start_date = model_params.get('training_start_date', '2020-01-01')
        end_date = model_params.get('training_end_date', '2024-12-31')

        logger.info(f"Generating labels for {len(symbols)} symbols from {start_date} to {end_date}")

        # Generate labels
        labels_df = generator.generate_labels_for_symbols(symbols, start_date, end_date)

        if labels_df.empty:
            logger.error("No labels generated")
            return 1

        # Save to DuckDB
        success = store_features(labels_df)
        if success:
            logger.info(f"Successfully stored {len(labels_df)} label records")
            print(f"✅ Generated and stored {len(labels_df)} labels")

            # Print summary statistics
            print("\n📊 Label Statistics:")
            total_labels = len(labels_df)
            positive_labels = labels_df['label_up_10pct'].sum()
            positive_rate = positive_labels / total_labels * 100

            print(f"   Total labels: {total_labels}")
            print(f"   Positive labels (+10%): {positive_labels}")
            print(".1f")
            print(f"   Symbols covered: {labels_df['symbol'].nunique()}")

            return 0
        else:
            logger.error("Failed to store labels")
            return 1

    except Exception as e:
        logger.error(f"Label generation failed: {e}", exc_info=True)
        print(f"❌ Label generation failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
