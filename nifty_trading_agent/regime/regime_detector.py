#!/usr/bin/env python3
"""
Regime Detection for Nifty Trading Agent v2
Detects market regimes (bull, sideways, bear) based on index trends and volatility
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.logging_utils import get_logger
from utils.duckdb_tools import load_ohlcv

logger = get_logger(__name__)

class RegimeDetector:
    """
    Detects market regimes based on Nifty index data
    """

    def __init__(self, index_symbol: str = "^NSEI"):
        """
        Initialize regime detector

        Args:
            index_symbol: Symbol for the market index (e.g., ^NSEI)
        """
        self.index_symbol = index_symbol

        # Regime detection parameters
        self.ma_short_period = 20   # Short-term MA for trend
        self.ma_long_period = 100   # Long-term MA for trend
        self.vol_period = 20        # Volatility lookback period
        self.min_data_points = 200  # Minimum data points needed

        # Regime thresholds
        self.bull_threshold = 0.02      # +2% above long MA for bull
        self.bear_threshold = -0.02     # -2% below long MA for bear
        self.high_vol_threshold = 1.5   # 1.5x average volatility for high vol
        self.sideways_vol_threshold = 0.8  # 0.8x average volatility for low vol

        logger.info("RegimeDetector initialized")

    def get_regime_for_date(self, date: pd.Timestamp,
                           use_cache: bool = True) -> str:
        """
        Get market regime for a specific date

        Args:
            date: Date to analyze
            use_cache: Whether to use cached regime data

        Returns:
            Regime string: 'bull', 'sideways', or 'bear'
        """
        try:
            # Load sufficient historical data for regime calculation
            start_date = date - timedelta(days=self.min_data_points * 2)  # Extra buffer
            end_date = date + timedelta(days=5)  # Small buffer for future data

            index_data = load_ohlcv([self.index_symbol], start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

            if index_data.empty:
                logger.warning(f"No index data available for {self.index_symbol} on {date}")
                return 'sideways'  # Default to sideways when no data

            # Calculate regime
            regime = self._calculate_regime_for_date(index_data, date)

            logger.debug(f"Regime for {date}: {regime}")
            return regime

        except Exception as e:
            logger.error(f"Error detecting regime for {date}: {e}")
            return 'sideways'  # Default to sideways on error

    def _calculate_regime_for_date(self, index_data: pd.DataFrame,
                                  target_date: pd.Timestamp) -> str:
        """
        Calculate regime based on trend and volatility indicators

        Args:
            index_data: OHLCV data for the index
            target_date: Date to analyze

        Returns:
            Regime classification
        """
        # Ensure data is sorted
        index_data = index_data.sort_values('date').reset_index(drop=True)

        # Find the target date row
        target_row = index_data[index_data['date'] == target_date]
        if target_row.empty:
            # Find closest date
            index_data['date_diff'] = (index_data['date'] - target_date).abs()
            closest_row = index_data.loc[index_data['date_diff'].idxmin()]
            target_idx = closest_row.name
            logger.debug(f"Using closest date {closest_row['date']} for target {target_date}")
        else:
            target_idx = target_row.index[0]

        # Need sufficient historical data
        if target_idx < self.ma_long_period:
            logger.warning(f"Insufficient historical data for regime calculation on {target_date}")
            return 'sideways'

        # Calculate moving averages
        prices = index_data['close'][:target_idx + 1]  # Up to target date

        ma_short = prices.rolling(window=self.ma_short_period).mean()
        ma_long = prices.rolling(window=self.ma_long_period).mean()

        current_price = prices.iloc[-1]
        current_ma_short = ma_short.iloc[-1]
        current_ma_long = ma_long.iloc[-1]

        # Calculate trend indicators
        price_vs_long_ma = (current_price / current_ma_long) - 1
        short_vs_long_ma = (current_ma_short / current_ma_long) - 1

        # Calculate volatility
        returns = prices.pct_change()
        current_vol = returns.rolling(window=self.vol_period).std().iloc[-1]
        avg_vol = returns.rolling(window=self.ma_long_period).std().iloc[-1]

        # Normalize volatility
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0

        # Determine regime based on trend and volatility
        regime = self._classify_regime(price_vs_long_ma, short_vs_long_ma, vol_ratio)

        logger.debug(f"Regime calculation for {target_date}:")
        logger.debug(f"  Price: {current_price:.2f}, Short MA: {current_ma_short:.2f}, Long MA: {current_ma_long:.2f}")
        logger.debug(f"  Price vs Long MA: {price_vs_long_ma:.3f}")
        logger.debug(f"  Short vs Long MA: {short_vs_long_ma:.3f}")
        logger.debug(f"  Volatility ratio: {vol_ratio:.3f}")
        logger.debug(f"  Final regime: {regime}")

        return regime

    def _classify_regime(self, price_vs_long_ma: float, short_vs_long_ma: float,
                        vol_ratio: float) -> str:
        """
        Classify regime based on trend and volatility metrics

        Args:
            price_vs_long_ma: Current price vs long-term MA
            short_vs_long_ma: Short-term MA vs long-term MA
            vol_ratio: Current volatility vs average

        Returns:
            Regime classification
        """
        # Bull market conditions
        if (price_vs_long_ma > self.bull_threshold and
            short_vs_long_ma > 0.01 and
            vol_ratio < self.high_vol_threshold):
            return 'bull'

        # Bear market conditions
        elif (price_vs_long_ma < self.bear_threshold and
              short_vs_long_ma < -0.01):
            return 'bear'

        # Sideways/high volatility conditions
        else:
            return 'sideways'

    def get_regime_history(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get regime history for a date range

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with date and regime columns
        """
        try:
            # Load index data
            index_data = load_ohlcv([self.index_symbol], start_date, end_date)

            if index_data.empty:
                logger.warning("No index data available for regime history")
                return pd.DataFrame()

            # Calculate regimes for each date
            regimes = []
            dates = pd.date_range(start=start_date, end=end_date, freq='D')

            for date in dates:
                try:
                    regime = self.get_regime_for_date(date)
                    regimes.append({
                        'date': date,
                        'regime': regime
                    })
                except Exception as e:
                    logger.debug(f"Could not calculate regime for {date}: {e}")
                    continue

            regime_df = pd.DataFrame(regimes)

            # Add regime transition indicators
            if not regime_df.empty:
                regime_df['regime_changed'] = regime_df['regime'] != regime_df['regime'].shift(1)
                regime_df['days_in_regime'] = regime_df.groupby(
                    (regime_df['regime'] != regime_df['regime'].shift(1)).cumsum()
                ).cumcount() + 1

            logger.info(f"Generated regime history for {len(regime_df)} dates")
            return regime_df

        except Exception as e:
            logger.error(f"Error generating regime history: {e}")
            return pd.DataFrame()

    def analyze_regime_statistics(self, regime_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze regime statistics

        Args:
            regime_df: DataFrame with regime data

        Returns:
            Dictionary with regime statistics
        """
        if regime_df.empty:
            return {}

        stats = {
            'total_days': len(regime_df),
            'regime_distribution': {},
            'average_regime_duration': {},
            'regime_transitions': 0,
            'most_common_regime': None
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

        # Most common regime
        stats['most_common_regime'] = regime_counts.index[0]

        # Average regime duration
        for regime in regime_counts.index:
            regime_data = regime_df[regime_df['regime'] == regime]
            avg_duration = regime_data['days_in_regime'].max()  # Last streak
            stats['average_regime_duration'][regime] = int(avg_duration)

        # Regime transitions
        stats['regime_transitions'] = regime_df['regime_changed'].sum()

        return stats

    def print_regime_report(self, start_date: str, end_date: str):
        """
        Print a comprehensive regime analysis report

        Args:
            start_date: Start date for analysis
            end_date: End date for analysis
        """
        print("\n🌦️  MARKET REGIME ANALYSIS")
        print("=" * 35)

        # Get regime history
        regime_df = self.get_regime_history(start_date, end_date)

        if regime_df.empty:
            print("No regime data available")
            return

        # Analyze statistics
        stats = self.analyze_regime_statistics(regime_df)

        print(f"Analysis period: {start_date} to {end_date}")
        print(f"Total trading days: {stats['total_days']}")
        print()

        print("Regime Distribution:")
        for regime, data in stats['regime_distribution'].items():
            print(f"  {regime.capitalize():8}: {data['days']:4} days ({data['percentage']:4.1f}%)")
        print()

        print(f"Most common regime: {stats['most_common_regime'].capitalize()}")
        print(f"Regime transitions: {stats['regime_transitions']}")
        print()

        print("Current Regime:")
        latest_regime = regime_df['regime'].iloc[-1]
        days_in_current = regime_df['days_in_regime'].iloc[-1]
        print(f"  Current: {latest_regime.capitalize()}")
        print(f"  Days in current regime: {days_in_current}")
        print()

        # Recent transitions
        transitions = regime_df[regime_df['regime_changed']].tail(5)
        if not transitions.empty:
            print("Recent Regime Transitions:")
            for _, row in transitions.iterrows():
                print(f"  {row['date'].strftime('%Y-%m-%d')}: {row['regime'].capitalize()}")
        print()

        # Recommendations based on current regime
        print("Trading Recommendations:")
        if latest_regime == 'bull':
            print("  • Bull market: Favor long positions")
            print("  • Use trailing stops to protect profits")
            print("  • Consider increasing position sizes")
        elif latest_regime == 'bear':
            print("  • Bear market: Consider short positions or cash")
            print("  • Use tight stops to limit losses")
            print("  • Focus on defensive stocks")
        else:  # sideways
            print("  • Sideways market: Focus on range trading")
            print("  • Use shorter timeframes for entries")
            print("  • Consider options strategies")

def get_regime_for_date(date: pd.Timestamp,
                       index_symbol: str = "^NSEI") -> str:
    """
    Convenience function to get regime for a single date

    Args:
        date: Date to analyze
        index_symbol: Index symbol to use

    Returns:
        Regime string
    """
    detector = RegimeDetector(index_symbol)
    return detector.get_regime_for_date(date)

def main():
    """Demo function for regime detection"""
    print("🌦️  REGIME DETECTOR - DEMO")
    print("=" * 28)

    # Initialize detector
    detector = RegimeDetector()

    # Test current date (will use closest available data)
    test_date = pd.Timestamp('2024-01-15')
    regime = detector.get_regime_for_date(test_date)
    print(f"Regime for {test_date.strftime('%Y-%m-%d')}: {regime}")

    # Test regime history
    start_date = "2023-01-01"
    end_date = "2023-12-31"

    print(f"\nAnalyzing regime history from {start_date} to {end_date}...")
    detector.print_regime_report(start_date, end_date)

if __name__ == "__main__":
    main()
