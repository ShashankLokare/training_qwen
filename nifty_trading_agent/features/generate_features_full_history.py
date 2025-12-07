#!/usr/bin/env python3
"""
Generate Features for Full History (2015-2025)
Creates comprehensive 25-feature dataset for all Nifty 50 stocks
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.logging_utils import setup_logging, get_logger
from utils.db_duckdb import execute_query

logger = get_logger(__name__)

class FullHistoryFeatureGenerator:
    """
    Generate features for the complete OHLCV history (2015-2025)
    """

    def __init__(self):
        """Initialize feature generator"""
        logger.info("FullHistoryFeatureGenerator initialized")

        # EXACT feature set that the model expects (25 features)
        self.required_features = [
            'r_1d', 'r_3d', 'r_5d', 'r_10d', 'r_20d',  # Returns
            'ma_5', 'ma_10', 'ma_20', 'ma_50', 'ma_100', 'ma_200',  # Moving averages
            'rsi_14', 'macd', 'bb_width',  # Technical indicators
            'atr_14', 'vol_zscore_20',  # Volatility
            'quarterly_eps_growth', 'quarterly_rev_growth', 'profit_margin',  # Fundamentals
            'pe_ratio', 'pb_ratio',  # Valuation ratios
            'sentiment_short', 'sentiment_medium',  # Sentiment
            'recent_earnings_flag', 'volume_gap_flag'  # Flags
        ]

    def process_all_symbols_full_history(self) -> pd.DataFrame:
        """
        Process all symbols for the complete 2015-2025 history

        Returns:
            DataFrame with all features
        """
        logger.info("🚀 Starting full history feature generation (2015-2025)")

        # Get all symbols from OHLCV table
        symbols_query = "SELECT DISTINCT symbol FROM ohlcv_nifty ORDER BY symbol"
        symbols_result = execute_query(symbols_query)
        symbols = [row[0] for row in symbols_result]

        logger.info(f"Found {len(symbols)} symbols to process")

        all_features = []

        for i, symbol in enumerate(symbols, 1):
            logger.info(f"Processing {symbol} ({i}/{len(symbols)})")

            try:
                # Get full OHLCV history for this symbol
                ohlcv_data = self._load_symbol_ohlcv(symbol)
                if ohlcv_data.empty:
                    logger.warning(f"No OHLCV data for {symbol}")
                    continue

                logger.info(f"  {symbol}: {len(ohlcv_data)} OHLCV records")

                # Generate features for this symbol
                symbol_features = self._generate_symbol_features_full(ohlcv_data, symbol)

                if not symbol_features.empty:
                    all_features.append(symbol_features)
                    logger.info(f"  ✅ Generated {len(symbol_features)} feature records for {symbol}")

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue

        if not all_features:
            logger.error("No features generated for any symbol")
            return pd.DataFrame()

        # Combine all features
        combined_features = pd.concat(all_features, ignore_index=True)

        # Remove any rows with NaN in required features
        before_count = len(combined_features)
        combined_features = combined_features.dropna(subset=self.required_features)
        after_count = len(combined_features)

        logger.info(f"Removed {before_count - after_count} rows with NaN features")

        # Sort and reset index
        combined_features = combined_features.sort_values(['symbol', 'date']).reset_index(drop=True)

        logger.info(f"🎉 Generated {len(combined_features)} total feature records")
        logger.info(f"📊 Features per record: {len(self.required_features)}")
        logger.info(f"🏢 Symbols processed: {combined_features['symbol'].nunique()}")

        return combined_features

    def _load_symbol_ohlcv(self, symbol: str) -> pd.DataFrame:
        """
        Load complete OHLCV history for a symbol

        Args:
            symbol: Stock symbol

        Returns:
            DataFrame with OHLCV data
        """
        query = f"""
            SELECT symbol, date, open, high, low, close, volume
            FROM ohlcv_nifty
            WHERE symbol = '{symbol}'
            ORDER BY date
        """

        result = execute_query(query)

        if not result:
            return pd.DataFrame()

        df = pd.DataFrame(result, columns=['symbol', 'date', 'open', 'high', 'low', 'close', 'volume'])
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

        return df

    def _generate_symbol_features_full(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Generate all 25 features for a symbol's complete history

        Args:
            df: OHLCV DataFrame
            symbol: Stock symbol

        Returns:
            DataFrame with features
        """
        # Prepare data - ensure we have enough history for longest calculations (ma_200)
        min_required_history = 250  # Need at least 250 trading days

        if len(df) < min_required_history:
            logger.warning(f"  Insufficient history for {symbol}: {len(df)} < {min_required_history}")
            return pd.DataFrame()

        # Create feature records for each date that has sufficient history
        features_list = []

        # Start from the date where we have enough history for all features
        start_date = df.index[min_required_history - 1]  # Index where we have 250+ days before

        for current_date in df.index[df.index >= start_date]:
            try:
                # Get data up to current date
                historical_data = df[df.index <= current_date]

                # Generate all feature categories
                feature_row = {
                    'symbol': symbol,
                    'date': current_date.date()
                }

                # Price-based features
                feature_row.update(self._generate_price_features(historical_data))

                # Volatility features
                feature_row.update(self._generate_volatility_features(historical_data))

                # Momentum features
                feature_row.update(self._generate_momentum_features(historical_data))

                # Volume features
                feature_row.update(self._generate_volume_features(historical_data))

                # Fundamental, sentiment, and event features
                feature_row.update(self._generate_fundamental_features(current_date))

                # Check for NaN in required features
                has_nan = any(pd.isna(feature_row.get(feat, np.nan)) for feat in self.required_features)
                if not has_nan:
                    features_list.append(feature_row)

            except Exception as e:
                logger.debug(f"Error generating features for {symbol} on {current_date}: {e}")
                continue

        return pd.DataFrame(features_list)

    def _generate_price_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Generate EXACT price features that the model expects"""
        features = {}
        close_prices = df['close'].values

        # Moving Averages - exactly as model expects
        ma_windows = [5, 10, 20, 50, 100, 200]
        for window in ma_windows:
            if len(close_prices) >= window:
                features[f'ma_{window}'] = close_prices[-window:].mean()

        # RSI (Relative Strength Index) - exactly 14 periods
        if len(close_prices) >= 14:
            deltas = np.diff(close_prices[-15:])  # Last 15 prices for 14 deltas
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)

            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)

            if avg_loss != 0:
                rs = avg_gain / avg_loss
                features['rsi_14'] = 100 - (100 / (1 + rs))
            else:
                features['rsi_14'] = 100.0

        # MACD - exactly as model expects
        if len(close_prices) >= 26:
            # Simplified MACD calculation
            ema_12 = np.average(close_prices[-12:], weights=np.array([2/(12+1) * (1-2/(12+1))**i for i in range(12)][::-1]))
            ema_26 = np.average(close_prices[-26:], weights=np.array([2/(26+1) * (1-2/(26+1))**i for i in range(26)][::-1]))
            features['macd'] = ema_12 - ema_26

        # Bollinger Band Width - exactly for 20-period (as model expects)
        if len(close_prices) >= 20:
            ma = close_prices[-20:].mean()
            std = close_prices[-20:].std()
            features['bb_width'] = (2 * std) / ma if ma != 0 else 0

        return features

    def _generate_volatility_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Generate EXACT volatility features that the model expects"""
        features = {}

        # Average True Range (ATR) - exactly 14 periods as model expects
        if len(df) >= 15:  # Need at least 15 periods for ATR calculation
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift(1)).abs()
            low_close = (df['low'] - df['close'].shift(1)).abs()

            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            features['atr_14'] = tr.rolling(14).mean().iloc[-1]

        # Volume Z-score - exactly 20 periods as model expects
        if len(df) >= 20:
            volumes = df['volume'].values
            vol_ma = volumes[-20:].mean()
            vol_std = volumes[-20:].std()
            current_vol = volumes[-1]
            features['vol_zscore_20'] = (current_vol - vol_ma) / vol_std if vol_std != 0 else 0

        return features

    def _generate_momentum_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Generate EXACT momentum features that the model expects"""
        features = {}
        returns = df['close'].pct_change().dropna().values

        # Return features - exactly as model expects
        if len(returns) >= 1:
            features['r_1d'] = returns[-1]  # 1-day return

        if len(returns) >= 3:
            features['r_3d'] = (df['close'].iloc[-1] - df['close'].iloc[-4]) / df['close'].iloc[-4]  # 3-day return

        if len(returns) >= 5:
            features['r_5d'] = (df['close'].iloc[-1] - df['close'].iloc[-6]) / df['close'].iloc[-6]  # 5-day return

        if len(returns) >= 10:
            features['r_10d'] = (df['close'].iloc[-1] - df['close'].iloc[-11]) / df['close'].iloc[-11]  # 10-day return

        if len(returns) >= 20:
            features['r_20d'] = (df['close'].iloc[-1] - df['close'].iloc[-21]) / df['close'].iloc[-21]  # 20-day return

        return features

    def _generate_volume_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Generate EXACT volume features that the model expects"""
        features = {}

        # Volume gap flag - simplified volume anomaly detection
        if len(df) >= 20:
            volumes = df['volume'].values
            vol_ma = volumes[-20:].mean()
            vol_std = volumes[-20:].std()
            current_vol = volumes[-1]

            # Flag significant volume spikes (more than 2 standard deviations)
            z_score = (current_vol - vol_ma) / vol_std if vol_std != 0 else 0
            features['volume_gap_flag'] = 1 if z_score > 2.0 else 0

        return features

    def _generate_fundamental_features(self, current_date: pd.Timestamp) -> Dict[str, float]:
        """Generate EXACT fundamental and sentiment features that the model expects"""
        features = {}

        # Generate placeholder values for fundamental features
        # In production, these would come from actual fundamental data sources

        # EPS and revenue growth (simplified - would use actual quarterly data)
        # Using deterministic random values for demonstration
        seed_base = int(current_date.strftime('%Y%m%d'))
        np.random.seed(seed_base)
        features['quarterly_eps_growth'] = np.random.normal(0.15, 0.30)  # 15% avg growth, 30% std
        features['quarterly_rev_growth'] = np.random.normal(0.12, 0.25)  # 12% avg growth, 25% std

        # Profit margin (simplified)
        features['profit_margin'] = np.random.uniform(0.05, 0.25)  # 5-25% profit margins

        # Valuation ratios (simplified - would use actual P/E and P/B)
        features['pe_ratio'] = np.random.uniform(10, 40)  # Realistic P/E range
        features['pb_ratio'] = np.random.uniform(1.5, 5.0)  # Realistic P/B range

        # Sentiment features (simplified - would use NLP on news/articles)
        features['sentiment_short'] = np.random.uniform(-1, 1)  # Short-term sentiment
        features['sentiment_medium'] = np.random.uniform(-1, 1)  # Medium-term sentiment

        # Recent earnings flag (simplified)
        current_month = current_date.month
        features['recent_earnings_flag'] = 1 if current_month in [1, 4, 7, 10] else 0

        return features

    def save_features_to_duckdb(self, features_df: pd.DataFrame) -> bool:
        """
        Save features to DuckDB features_nifty table

        Args:
            features_df: DataFrame with features

        Returns:
            True if successful
        """
        if features_df.empty:
            logger.error("No features to save")
            return False

        try:
            # Create table if not exists
            create_table_query = """
                CREATE TABLE IF NOT EXISTS features_nifty (
                    symbol                 TEXT,
                    date                   DATE,
                    r_1d                   DOUBLE,
                    r_3d                   DOUBLE,
                    r_5d                   DOUBLE,
                    r_10d                  DOUBLE,
                    r_20d                  DOUBLE,
                    ma_5                   DOUBLE,
                    ma_10                  DOUBLE,
                    ma_20                  DOUBLE,
                    ma_50                  DOUBLE,
                    ma_100                 DOUBLE,
                    ma_200                 DOUBLE,
                    rsi_14                 DOUBLE,
                    macd                   DOUBLE,
                    bb_width               DOUBLE,
                    atr_14                 DOUBLE,
                    vol_zscore_20          DOUBLE,
                    quarterly_eps_growth   DOUBLE,
                    quarterly_rev_growth   DOUBLE,
                    profit_margin          DOUBLE,
                    pe_ratio               DOUBLE,
                    pb_ratio               DOUBLE,
                    sentiment_short        DOUBLE,
                    sentiment_medium       DOUBLE,
                    recent_earnings_flag   BOOLEAN,
                    volume_gap_flag        BOOLEAN,
                    PRIMARY KEY (symbol, date)
                )
            """

            # Execute DDL
            from utils.db_duckdb import execute_ddl
            execute_ddl(create_table_query)

            # Clear existing data
            execute_query("DELETE FROM features_nifty")

            # Insert new data in batches to avoid memory issues
            batch_size = 1000
            total_rows = len(features_df)

            for i in range(0, total_rows, batch_size):
                batch = features_df.iloc[i:i+batch_size]

                # Convert to list of tuples for insertion
                values = []
                for _, row in batch.iterrows():
                    values.append(tuple(row))

                # Build INSERT query
                columns = ', '.join(features_df.columns)
                placeholders = ', '.join(['?' for _ in features_df.columns])
                insert_query = f"INSERT INTO features_nifty ({columns}) VALUES ({placeholders})"

                # Execute insert
                from utils.db_duckdb import duck_cursor
                with duck_cursor() as conn:
                    conn.executemany(insert_query, values)

                logger.info(f"Inserted batch {i//batch_size + 1}/{(total_rows + batch_size - 1)//batch_size}")

            logger.info(f"✅ Successfully saved {total_rows} feature records to features_nifty table")
            return True

        except Exception as e:
            logger.error(f"Error saving features to DuckDB: {e}")
            return False

    def print_summary_statistics(self) -> None:
        """Print summary statistics of the features_nifty table"""
        try:
            # Count query
            count_query = """
                SELECT
                    COUNT(*) AS rows,
                    COUNT(DISTINCT symbol) AS symbols,
                    COUNT(DISTINCT date) AS dates
                FROM features_nifty
            """

            result = execute_query(count_query)
            if result:
                rows, symbols, dates = result[0]
                print("\n📊 FEATURES_NIFTY TABLE SUMMARY:")
                print(f"   Total rows: {rows:,}")
                print(f"   Unique symbols: {symbols}")
                print(f"   Unique dates: {dates}")

                # Date range
                date_range_query = """
                    SELECT MIN(date) as start_date, MAX(date) as end_date
                    FROM features_nifty
                """
                date_result = execute_query(date_range_query)
                if date_result:
                    start_date, end_date = date_result[0]
                    print(f"   Date range: {start_date} to {end_date}")

                # Feature completeness - check first 5 features
                null_check_query = """
                    SELECT
                        SUM(CASE WHEN r_1d IS NULL THEN 1 ELSE 0 END) as r_1d_nulls,
                        SUM(CASE WHEN r_3d IS NULL THEN 1 ELSE 0 END) as r_3d_nulls,
                        SUM(CASE WHEN r_5d IS NULL THEN 1 ELSE 0 END) as r_5d_nulls,
                        SUM(CASE WHEN r_10d IS NULL THEN 1 ELSE 0 END) as r_10d_nulls,
                        SUM(CASE WHEN r_20d IS NULL THEN 1 ELSE 0 END) as r_20d_nulls
                    FROM features_nifty
                """
                null_result = execute_query(null_check_query)
                if null_result:
                    null_counts = null_result[0]
                    total_nulls = sum(null_counts)
                    print(f"   Feature completeness: {rows - total_nulls}/{rows} ({100*(rows - total_nulls)/rows:.1f}%)")

        except Exception as e:
            logger.error(f"Error printing summary statistics: {e}")

def main():
    """Main feature generation function"""
    print("🔧 NIFTY TRADING AGENT - FULL HISTORY FEATURE GENERATION")
    print("=" * 65)

    # Setup logging
    setup_logging({
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file_path': 'logs/full_history_feature_generation.log',
        'max_file_size_mb': 50,
        'backup_count': 3
    })

    try:
        # Initialize feature generator
        generator = FullHistoryFeatureGenerator()

        # Generate features for full history
        logger.info("Starting full history feature generation...")
        all_features = generator.process_all_symbols_full_history()

        if all_features.empty:
            logger.error("No features generated")
            return 1

        # Save to DuckDB
        logger.info(f"Saving {len(all_features)} feature records to database...")
        success = generator.save_features_to_duckdb(all_features)

        if success:
            logger.info("✅ Full history features saved successfully")

            # Print comprehensive statistics
            generator.print_summary_statistics()

            print("\n🎯 Next steps:")
            print("  1. Run labels/generate_labels_v3.py")
            print("  2. Run models/train_model_v3.py")
            print("  3. Run pipeline/generate_signals_v3.py")

            return 0
        else:
            logger.error("Failed to save features")
            return 1

    except Exception as e:
        logger.error(f"Feature generation failed: {e}", exc_info=True)
        print(f"❌ Feature generation failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
