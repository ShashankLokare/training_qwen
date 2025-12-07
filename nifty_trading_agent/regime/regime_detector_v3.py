#!/usr/bin/env python3
"""
V3 Regime Detection for Nifty Trading Agent
Enhanced market regime detection with DuckDB storage and advanced classification
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.logging_utils import get_logger
from utils.db_duckdb import execute_query, get_duck_conn

logger = get_logger(__name__)

class RegimeDetectorV3:
    """
    V3 Enhanced regime detector with DuckDB storage and advanced classification
    """

    def __init__(self, index_symbol: str = "^NSEI", window_days: int = 60):
        """
        Initialize V3 regime detector

        Args:
            index_symbol: Symbol for the market index (will derive if not available)
            window_days: Rolling window for regime calculation (default 60 days)
        """
        self.index_symbol = index_symbol
        self.window_days = window_days

        # Regime detection parameters
        self.min_data_points = max(window_days + 20, 100)  # Minimum data needed

        # Regime thresholds (can be tuned)
        self.bull_return_threshold = 0.05      # +5% return for bull
        self.bear_return_threshold = -0.05     # -5% return for bear
        self.high_vol_multiplier = 1.5         # 1.5x average vol for high vol
        self.low_vol_multiplier = 0.7          # 0.7x average vol for low vol

        logger.info(f"RegimeDetectorV3 initialized with {window_days}d window")

    def create_regime_table(self) -> None:
        """
        Create the market_regimes DuckDB table if it doesn't exist
        """
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS market_regimes (
            date DATE PRIMARY KEY,
            regime TEXT NOT NULL,
            rolling_return_60d FLOAT,
            rolling_volatility_60d FLOAT,
            index_close FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """

        try:
            execute_query(create_table_sql)
            logger.info("Created market_regimes table")
        except Exception as e:
            logger.error(f"Failed to create market_regimes table: {e}")
            raise

    def compute_market_index(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Compute equal-weighted market index from available stocks

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with date and close columns for the index
        """
        # Get all available symbols for the period
        symbols_query = f"""
        SELECT DISTINCT symbol
        FROM ohlcv_nifty
        WHERE date BETWEEN '{start_date}' AND '{end_date}'
        AND symbol NOT LIKE '^%'  -- Exclude index symbols
        ORDER BY symbol
        """

        symbols_result = execute_query(symbols_query)
        if not symbols_result:
            logger.warning("No symbols found for market index calculation")
            return pd.DataFrame()

        symbols = [row[0] for row in symbols_result]
        logger.info(f"Computing market index from {len(symbols)} symbols")

        # Get OHLCV data for all symbols
        index_query = f"""
        SELECT
            date,
            symbol,
            close
        FROM ohlcv_nifty
        WHERE date BETWEEN '{start_date}' AND '{end_date}'
        AND symbol IN ({','.join([f"'{s}'" for s in symbols])})
        ORDER BY date, symbol
        """

        data_result = execute_query(index_query)
        if not data_result:
            logger.warning("No OHLCV data found for market index calculation")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(data_result, columns=['date', 'symbol', 'close'])
        df['date'] = pd.to_datetime(df['date'])

        # Compute equal-weighted index
        index_data = df.groupby('date')['close'].mean().reset_index()
        index_data.columns = ['date', 'close']

        logger.info(f"Computed market index with {len(index_data)} data points")
        return index_data

    def calculate_regime_indicators(self, index_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate regime indicators (rolling return and volatility)

        Args:
            index_data: DataFrame with date and close columns

        Returns:
            DataFrame with regime indicators
        """
        if index_data.empty:
            return pd.DataFrame()

        df = index_data.copy()
        df = df.sort_values('date').reset_index(drop=True)

        # Calculate daily returns
        df['daily_return'] = df['close'].pct_change()

        # Calculate rolling metrics
        df['rolling_return_60d'] = df['close'].pct_change(periods=self.window_days)
        df['rolling_volatility_60d'] = df['daily_return'].rolling(window=self.window_days).std() * np.sqrt(252)  # Annualized

        # Remove rows with insufficient data
        df = df.dropna(subset=['rolling_return_60d', 'rolling_volatility_60d']).reset_index(drop=True)

        logger.info(f"Calculated regime indicators for {len(df)} dates")
        return df

    def classify_regime(self, rolling_return: float, rolling_volatility: float,
                       avg_volatility: float) -> str:
        """
        Classify market regime based on return and volatility

        Args:
            rolling_return: 60-day rolling return
            rolling_volatility: 60-day rolling volatility (annualized)
            avg_volatility: Long-term average volatility

        Returns:
            Regime classification string
        """
        # Bull market: positive return above threshold and reasonable volatility
        if (rolling_return > self.bull_return_threshold and
            rolling_volatility < self.high_vol_multiplier * avg_volatility):
            return 'bull'

        # Bear market: negative return below threshold
        elif rolling_return < self.bear_return_threshold:
            return 'bear'

        # High volatility: regardless of return direction
        elif rolling_volatility > self.high_vol_multiplier * avg_volatility:
            return 'high_vol'

        # Sideways: everything else
        else:
            return 'sideways'

    def populate_regime_table(self, start_date: str, end_date: str) -> None:
        """
        Populate the market_regimes table with regime data

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        """
        logger.info(f"Populating market_regimes table from {start_date} to {end_date}")

        # Create table if needed
        self.create_regime_table()

        # Compute market index
        index_data = self.compute_market_index(start_date, end_date)
        if index_data.empty:
            logger.error("Could not compute market index")
            return

        # Calculate regime indicators
        regime_data = self.calculate_regime_indicators(index_data)
        if regime_data.empty:
            logger.error("Could not calculate regime indicators")
            return

        # Calculate long-term average volatility for classification
        avg_volatility = regime_data['rolling_volatility_60d'].mean()

        # Classify regimes
        regimes = []
        for _, row in regime_data.iterrows():
            regime = self.classify_regime(
                row['rolling_return_60d'],
                row['rolling_volatility_60d'],
                avg_volatility
            )

            regimes.append({
                'date': row['date'],
                'regime': regime,
                'rolling_return_60d': row['rolling_return_60d'],
                'rolling_volatility_60d': row['rolling_volatility_60d'],
                'index_close': row['close']
            })

        # Insert into database using execute_query for consistency
        try:
            # Clear existing data for the date range
            execute_query(f"""
                DELETE FROM market_regimes
                WHERE date BETWEEN '{start_date}' AND '{end_date}'
            """)

            # Insert new data
            for regime_data in regimes:
                execute_query("""
                    INSERT INTO market_regimes
                    (date, regime, rolling_return_60d, rolling_volatility_60d, index_close)
                    VALUES (?, ?, ?, ?, ?)
                """, [
                    regime_data['date'],
                    regime_data['regime'],
                    regime_data['rolling_return_60d'],
                    regime_data['rolling_volatility_60d'],
                    regime_data['index_close']
                ])

            logger.info(f"Inserted {len(regimes)} regime records into database")

        except Exception as e:
            logger.error(f"Failed to populate regime table: {e}")
            raise

    def get_regime_for_date(self, target_date: str) -> Optional[str]:
        """
        Get regime for a specific date from the database

        Args:
            target_date: Date string (YYYY-MM-DD)

        Returns:
            Regime string or None if not found
        """
        query = f"""
        SELECT regime
        FROM market_regimes
        WHERE date = '{target_date}'
        """

        result = execute_query(query)
        if result and len(result) > 0:
            return result[0][0]
        else:
            logger.warning(f"No regime found for date {target_date}")
            return None

    def get_regime_history(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get regime history for a date range

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with regime data
        """
        query = f"""
        SELECT
            date,
            regime,
            rolling_return_60d,
            rolling_volatility_60d,
            index_close
        FROM market_regimes
        WHERE date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY date
        """

        result = execute_query(query)
        if not result:
            logger.warning(f"No regime data found for {start_date} to {end_date}")
            return pd.DataFrame()

        df = pd.DataFrame(result, columns=[
            'date', 'regime', 'rolling_return_60d',
            'rolling_volatility_60d', 'index_close'
        ])
        df['date'] = pd.to_datetime(df['date'])

        logger.info(f"Retrieved {len(df)} regime records")
        return df

    def analyze_regime_statistics(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Analyze regime statistics for a date range

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Dictionary with regime statistics
        """
        regime_df = self.get_regime_history(start_date, end_date)
        if regime_df.empty:
            return {}

        stats = {
            'total_days': len(regime_df),
            'regime_distribution': {},
            'average_regime_duration': {},
            'regime_transitions': 0,
            'most_common_regime': None,
            'regime_volatility': {},
            'regime_returns': {}
        }

        # Regime distribution
        regime_counts = regime_df['regime'].value_counts()
        total_days = len(regime_df)

        for regime, count in regime_counts.items():
            percentage = (count / total_days) * 100
            stats['regime_distribution'][regime] = {
                'days': int(count),
                'percentage': round(percentage, 1)
            }

            # Average return and volatility by regime
            regime_data = regime_df[regime_df['regime'] == regime]
            if not regime_data.empty:
                stats['regime_returns'][regime] = regime_data['rolling_return_60d'].mean()
                stats['regime_volatility'][regime] = regime_data['rolling_volatility_60d'].mean()

        # Most common regime
        stats['most_common_regime'] = regime_counts.index[0] if not regime_counts.empty else None

        # Regime transitions
        regime_df['regime_changed'] = regime_df['regime'] != regime_df['regime'].shift(1)
        stats['regime_transitions'] = int(regime_df['regime_changed'].sum())

        return stats

    def print_regime_report(self, start_date: str, end_date: str) -> None:
        """
        Print comprehensive regime analysis report

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        """
        print("\n🌦️  V3 MARKET REGIME ANALYSIS")
        print("=" * 40)

        # Get regime statistics
        stats = self.analyze_regime_statistics(start_date, end_date)

        if not stats:
            print("No regime data available")
            return

        print(f"Analysis period: {start_date} to {end_date}")
        print(f"Total trading days: {stats['total_days']}")
        print()

        print("Regime Distribution:")
        for regime, data in stats['regime_distribution'].items():
            print(f"  {regime.capitalize():9}: {data['days']:4} days ({data['percentage']:4.1f}%)")
        print()

        print("Regime Performance:")
        for regime in stats['regime_distribution'].keys():
            if regime in stats['regime_returns'] and regime in stats['regime_volatility']:
                ret = stats['regime_returns'][regime] * 100
                vol = stats['regime_volatility'][regime] * 100
                print(f"  {regime.capitalize():9}: {ret:6.1f}% return, {vol:5.1f}% vol")
        print()

        print(f"Most common regime: {stats['most_common_regime'].capitalize() if stats['most_common_regime'] else 'N/A'}")
        print(f"Regime transitions: {stats['regime_transitions']}")
        print()

        # Current regime
        current_query = """
        SELECT date, regime, rolling_return_60d, rolling_volatility_60d
        FROM market_regimes
        ORDER BY date DESC
        LIMIT 1
        """

        current_result = execute_query(current_query)
        if current_result:
            date, regime, ret_60d, vol_60d = current_result[0]
            print("Current Regime:")
            print(f"  Date: {date}")
            print(f"  Regime: {regime.capitalize()}")
            print(f"  60d Return: {ret_60d*100:.1f}%")
            print(f"  60d Volatility: {vol_60d*100:.1f}%")
        print()

        # Recommendations
        print("V3 Trading Recommendations:")
        if stats['most_common_regime'] == 'bull':
            print("  • Bull regime dominant: Favor momentum strategies")
            print("  • Increase position sizes in winning trades")
            print("  • Use trailing stops to capture trends")
        elif stats['most_common_regime'] == 'bear':
            print("  • Bear regime dominant: Focus on short/cash positions")
            print("  • Tighten stop losses and reduce position sizes")
            print("  • Consider put options for hedging")
        elif stats['most_common_regime'] == 'high_vol':
            print("  • High volatility: Reduce position sizes")
            print("  • Focus on options strategies or VIX-related trades")
            print("  • Use wider stops to avoid whipsaws")
        else:
            print("  • Sideways regime: Focus on mean-reversion")
            print("  • Use shorter holding periods")
            print("  • Consider pairs trading or arbitrage")

def main():
    """Demo function for V3 regime detection"""
    print("🌦️  V3 REGIME DETECTOR - DEMO")
    print("=" * 32)

    # Initialize V3 detector
    detector = RegimeDetectorV3()

    # Populate regime table for recent period
    print("Populating regime table for 2023-2024...")
    try:
        detector.populate_regime_table('2023-01-01', '2024-12-31')
        print("✅ Regime table populated successfully")
    except Exception as e:
        print(f"❌ Failed to populate regime table: {e}")
        return

    # Test regime retrieval
    test_date = '2024-01-15'
    regime = detector.get_regime_for_date(test_date)
    if regime:
        print(f"✅ Regime for {test_date}: {regime}")
    else:
        print(f"❌ No regime found for {test_date}")

    # Print regime report
    detector.print_regime_report('2023-01-01', '2024-12-31')

if __name__ == "__main__":
    main()
