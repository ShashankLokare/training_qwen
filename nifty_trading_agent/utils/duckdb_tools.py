"""
DuckDB Tools for Agent Operations
High-level functions for agents to interact with DuckDB analytical database
"""

import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime, date
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.logging_utils import get_logger
from utils.db_duckdb import duck_cursor, execute_query, execute_ddl

logger = get_logger(__name__)

def store_ohlcv(df_ohlcv: pd.DataFrame) -> bool:
    """
    Store OHLCV data in DuckDB

    Args:
        df_ohlcv: DataFrame with columns: symbol, date, open, high, low, close, volume

    Returns:
        True if successful, False otherwise
    """
    if df_ohlcv.empty:
        logger.warning("Empty OHLCV DataFrame provided")
        return True

    try:
        # Ensure required columns exist
        required_cols = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df_ohlcv.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False

        with duck_cursor() as conn:
            # Create table if it doesn't exist
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv_nifty (
                    symbol TEXT,
                    date DATE,
                    open DOUBLE,
                    high DOUBLE,
                    low DOUBLE,
                    close DOUBLE,
                    volume DOUBLE,
                    PRIMARY KEY (symbol, date)
                )
            """)

            # Insert data (UPSERT - update if exists, insert if not)
            for _, row in df_ohlcv.iterrows():
                conn.execute("""
                    INSERT OR REPLACE INTO ohlcv_nifty
                    (symbol, date, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    row['symbol'], row['date'], row['open'], row['high'],
                    row['low'], row['close'], row['volume']
                ))

            logger.info(f"Stored {len(df_ohlcv)} OHLCV records")
            return True

    except Exception as e:
        logger.error(f"Failed to store OHLCV data: {e}")
        return False

def load_ohlcv(symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Load OHLCV data for specified symbols and date range

    Args:
        symbols: List of stock symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame with OHLCV data
    """
    try:
        symbols_str = "', '".join(symbols)

        query = f"""
            SELECT symbol, date, open, high, low, close, volume
            FROM ohlcv_nifty
            WHERE symbol IN ('{symbols_str}')
              AND date >= '{start_date}'
              AND date <= '{end_date}'
            ORDER BY symbol, date
        """

        results = execute_query(query)

        if not results:
            logger.info(f"No OHLCV data found for {symbols} between {start_date} and {end_date}")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(results, columns=['symbol', 'date', 'open', 'high', 'low', 'close', 'volume'])

        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])

        logger.info(f"Loaded {len(df)} OHLCV records for {len(symbols)} symbols")
        return df

    except Exception as e:
        logger.error(f"Failed to load OHLCV data: {e}")
        return pd.DataFrame()

def store_features(df_features: pd.DataFrame) -> bool:
    """
    Store engineered features in DuckDB

    Args:
        df_features: DataFrame with feature columns

    Returns:
        True if successful, False otherwise
    """
    if df_features.empty:
        logger.warning("Empty features DataFrame provided")
        return True

    try:
        with duck_cursor() as conn:
            # Create features table if it doesn't exist
            conn.execute("""
                CREATE TABLE IF NOT EXISTS features_nifty (
                    symbol TEXT,
                    date DATE,
                    r_1d DOUBLE,
                    r_3d DOUBLE,
                    r_5d DOUBLE,
                    r_10d DOUBLE,
                    r_20d DOUBLE,
                    ma_5 DOUBLE,
                    ma_10 DOUBLE,
                    ma_20 DOUBLE,
                    ma_50 DOUBLE,
                    ma_100 DOUBLE,
                    ma_200 DOUBLE,
                    rsi_14 DOUBLE,
                    macd DOUBLE,
                    bb_width DOUBLE,
                    atr_14 DOUBLE,
                    vol_zscore_20 DOUBLE,
                    quarterly_eps_growth DOUBLE,
                    quarterly_rev_growth DOUBLE,
                    pe_ratio DOUBLE,
                    pb_ratio DOUBLE,
                    sentiment_short DOUBLE,
                    sentiment_medium DOUBLE,
                    forward_5d_ret_pct DOUBLE,
                    forward_10d_ret_pct DOUBLE,
                    label_up_10pct BOOLEAN,
                    PRIMARY KEY (symbol, date)
                )
            """)

            # Get column names from DataFrame
            columns = df_features.columns.tolist()

            # Insert data in batches for better performance
            batch_size = 1000
            for i in range(0, len(df_features), batch_size):
                batch = df_features.iloc[i:i+batch_size]

                # Build dynamic INSERT statement
                placeholders = ', '.join(['?' for _ in columns])
                columns_str = ', '.join(columns)

                insert_query = f"""
                    INSERT OR REPLACE INTO features_nifty
                    ({columns_str})
                    VALUES ({placeholders})
                """

                # Execute for each row in batch
                for _, row in batch.iterrows():
                    values = tuple(row[col] for col in columns)
                    conn.execute(insert_query, values)

            logger.info(f"Stored {len(df_features)} feature records")
            return True

    except Exception as e:
        logger.error(f"Failed to store features: {e}")
        return False

def load_features_for_training(start_date: str, end_date: str,
                              symbols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Load features for model training

    Args:
        start_date: Start date for training data
        end_date: End date for training data
        symbols: Optional list of symbols to filter

    Returns:
        DataFrame with features for training
    """
    try:
        query = """
            SELECT * FROM features_nifty
            WHERE date >= ?
              AND date <= ?
        """

        params = [start_date, end_date]

        if symbols:
            symbols_str = "', '".join(symbols)
            query += f" AND symbol IN ('{symbols_str}')"

        query += " ORDER BY symbol, date"

        results = execute_query(query, tuple(params))

        if not results:
            logger.info(f"No feature data found for training period {start_date} to {end_date}")
            return pd.DataFrame()

        # Convert to DataFrame (column names are known from schema)
        columns = [
            'symbol', 'date', 'r_1d', 'r_3d', 'r_5d', 'r_10d', 'r_20d',
            'ma_5', 'ma_10', 'ma_20', 'ma_50', 'ma_100', 'ma_200',
            'rsi_14', 'macd', 'bb_width', 'atr_14', 'vol_zscore_20',
            'quarterly_eps_growth', 'quarterly_rev_growth', 'profit_margin',
            'pe_ratio', 'pb_ratio', 'sentiment_short', 'sentiment_medium',
            'recent_earnings_flag', 'volume_gap_flag',
            'forward_5d_ret_pct', 'forward_10d_ret_pct', 'forward_10d_max_return_pct',
            'label_up_10pct', 'label_win_before_loss', 'label_loss_before_win'
        ]

        df = pd.DataFrame(results, columns=columns)
        df['date'] = pd.to_datetime(df['date'])

        logger.info(f"Loaded {len(df)} feature records for training")
        return df

    except Exception as e:
        logger.error(f"Failed to load features for training: {e}")
        return pd.DataFrame()

def store_backtest_results(strategy_id: str, run_id: str, df_equity: pd.DataFrame) -> bool:
    """
    Store backtest equity curve and metrics

    Args:
        strategy_id: Strategy identifier
        run_id: Unique run identifier
        df_equity: DataFrame with columns: date, equity, drawdown, exposure

    Returns:
        True if successful, False otherwise
    """
    if df_equity.empty:
        logger.warning("Empty equity DataFrame provided")
        return True

    try:
        with duck_cursor() as conn:
            # Create backtest results table if it doesn't exist
            conn.execute("""
                CREATE TABLE IF NOT EXISTS backtest_results (
                    strategy_id TEXT,
                    run_id TEXT,
                    date DATE,
                    equity DOUBLE,
                    drawdown DOUBLE,
                    exposure DOUBLE
                )
            """)

            # Create index for faster queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_backtest_strategy_date
                ON backtest_results(strategy_id, date)
            """)

            # Insert data
            for _, row in df_equity.iterrows():
                conn.execute("""
                    INSERT INTO backtest_results
                    (strategy_id, run_id, date, equity, drawdown, exposure)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    strategy_id, run_id, row['date'],
                    row.get('equity', 0),
                    row.get('drawdown', 0),
                    row.get('exposure', 0)
                ))

            logger.info(f"Stored backtest results: {strategy_id} - {run_id} ({len(df_equity)} records)")
            return True

    except Exception as e:
        logger.error(f"Failed to store backtest results: {e}")
        return False

def load_backtest_results(strategy_id: str) -> pd.DataFrame:
    """
    Load backtest results for a strategy

    Args:
        strategy_id: Strategy identifier

    Returns:
        DataFrame with backtest results
    """
    try:
        query = """
            SELECT run_id, date, equity, drawdown, exposure
            FROM backtest_results
            WHERE strategy_id = ?
            ORDER BY run_id, date
        """

        results = execute_query(query, (strategy_id,))

        if not results:
            logger.info(f"No backtest results found for strategy {strategy_id}")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(results, columns=['run_id', 'date', 'equity', 'drawdown', 'exposure'])
        df['date'] = pd.to_datetime(df['date'])

        logger.info(f"Loaded backtest results for {strategy_id}: {len(df)} records")
        return df

    except Exception as e:
        logger.error(f"Failed to load backtest results for {strategy_id}: {e}")
        return pd.DataFrame()

def get_latest_ohlcv_date(symbol: str) -> Optional[date]:
    """
    Get the latest OHLCV date for a symbol

    Args:
        symbol: Stock symbol

    Returns:
        Latest date or None if no data
    """
    try:
        query = """
            SELECT MAX(date) as latest_date
            FROM ohlcv_nifty
            WHERE symbol = ?
        """

        results = execute_query(query, (symbol,))

        if results and results[0][0]:
            return results[0][0]
        else:
            return None

    except Exception as e:
        logger.error(f"Failed to get latest OHLCV date for {symbol}: {e}")
        return None

def get_feature_statistics(symbol: Optional[str] = None, days: int = 30) -> Dict[str, Any]:
    """
    Get feature statistics for analysis

    Args:
        symbol: Optional symbol filter
        days: Number of recent days to analyze

    Returns:
        Dictionary with feature statistics
    """
    try:
        query = f"""
            SELECT
                COUNT(*) as total_records,
                COUNT(DISTINCT symbol) as unique_symbols,
                AVG(forward_5d_ret_pct) as avg_5d_return,
                STDDEV(forward_5d_ret_pct) as std_5d_return,
                SUM(CASE WHEN label_up_10pct THEN 1 ELSE 0 END) as positive_labels,
                AVG(rsi_14) as avg_rsi,
                AVG(bb_width) as avg_bb_width
            FROM features_nifty
            WHERE date >= (SELECT MAX(date) FROM features_nifty) - INTERVAL '{days}' DAY
        """

        if symbol:
            query += f" AND symbol = '{symbol}'"

        results = execute_query(query)

        if results:
            row = results[0]
            stats = {
                'total_records': row[0],
                'unique_symbols': row[1],
                'avg_5d_return': row[2],
                'std_5d_return': row[3],
                'positive_labels': row[4],
                'positive_label_ratio': row[4] / row[0] if row[0] > 0 else 0,
                'avg_rsi': row[5],
                'avg_bb_width': row[6]
            }
            return stats
        else:
            return {}

    except Exception as e:
        logger.error(f"Failed to get feature statistics: {e}")
        return {}

def delete_old_data(older_than_days: int = 365) -> int:
    """
    Delete data older than specified days

    Args:
        older_than_days: Delete data older than this many days

    Returns:
        Number of records deleted
    """
    try:
        cutoff_date = date.today() - pd.Timedelta(days=older_than_days)

        with duck_cursor() as conn:
            # Delete from OHLCV table
            result1 = conn.execute(f"""
                DELETE FROM ohlcv_nifty
                WHERE date < '{cutoff_date}'
            """)

            # Delete from features table
            result2 = conn.execute(f"""
                DELETE FROM features_nifty
                WHERE date < '{cutoff_date}'
            """)

            deleted_count = result1.fetchall()[0][0] + result2.fetchall()[0][0]

            logger.info(f"Deleted {deleted_count} old records (older than {older_than_days} days)")
            return deleted_count

    except Exception as e:
        logger.error(f"Failed to delete old data: {e}")
        return 0

def get_data_quality_report() -> Dict[str, Any]:
    """
    Generate a data quality report

    Returns:
        Dictionary with data quality metrics
    """
    try:
        report = {}

        # OHLCV data quality
        ohlcv_query = """
            SELECT
                COUNT(*) as total_records,
                COUNT(DISTINCT symbol) as symbols,
                MIN(date) as earliest_date,
                MAX(date) as latest_date,
                AVG(volume) as avg_volume
            FROM ohlcv_nifty
        """

        ohlcv_results = execute_query(ohlcv_query)
        if ohlcv_results:
            row = ohlcv_results[0]
            report['ohlcv'] = {
                'total_records': row[0],
                'unique_symbols': row[1],
                'date_range': f"{row[2]} to {row[3]}" if row[2] and row[3] else "N/A",
                'avg_volume': row[4]
            }

        # Features data quality
        features_query = """
            SELECT
                COUNT(*) as total_records,
                COUNT(DISTINCT symbol) as symbols,
                AVG(CASE WHEN rsi_14 IS NULL THEN 1 ELSE 0 END) as null_rsi_ratio,
                AVG(CASE WHEN forward_5d_ret_pct IS NULL THEN 1 ELSE 0 END) as null_return_ratio
            FROM features_nifty
        """

        features_results = execute_query(features_query)
        if features_results:
            row = features_results[0]
            report['features'] = {
                'total_records': row[0],
                'unique_symbols': row[1],
                'null_rsi_ratio': row[2],
                'null_return_ratio': row[3]
            }

        logger.info("Generated data quality report")
        return report

    except Exception as e:
        logger.error(f"Failed to generate data quality report: {e}")
        return {}
