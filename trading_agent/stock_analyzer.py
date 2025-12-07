"""
Stock Analyzer for Indian Market Trading Agent
Analyzes stocks for consistent upward momentum over specified periods
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockAnalyzer:
    """Analyzes stocks for consistent upward momentum"""

    def __init__(self):
        """Initialize the stock analyzer"""
        pass

    def calculate_daily_returns(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate daily percentage returns

        Args:
            data: DataFrame with stock price data

        Returns:
            Series of daily percentage returns
        """
        if 'Close' not in data.columns:
            logger.warning("No 'Close' column in data")
            return pd.Series()

        # Calculate percentage change from previous close
        returns = data['Close'].pct_change() * 100
        return returns

    def calculate_cumulative_return(self, data: pd.DataFrame, days: int = 5) -> float:
        """
        Calculate cumulative return over the last N days

        Args:
            data: DataFrame with stock price data
            days: Number of days to calculate cumulative return for

        Returns:
            Cumulative percentage return over the period
        """
        if len(data) < days:
            logger.warning(f"Insufficient data for {days} days analysis")
            return 0.0

        # Take the last N days of data
        recent_data = data.tail(days)

        if len(recent_data) < 2:
            return 0.0

        # Calculate cumulative return
        start_price = recent_data['Close'].iloc[0]
        end_price = recent_data['Close'].iloc[-1]

        cumulative_return = ((end_price - start_price) / start_price) * 100
        return cumulative_return

    def analyze_consistency(self, data: pd.DataFrame, min_gain: float = 2.0,
                          max_gain: float = 5.0, days: int = 5) -> Dict[str, float]:
        """
        Analyze if a stock has shown consistent gains within the specified range

        Args:
            data: DataFrame with stock price data
            min_gain: Minimum daily gain percentage
            max_gain: Maximum daily gain percentage
            days: Number of days to analyze

        Returns:
            Dictionary with analysis results
        """
        if len(data) < days:
            return {
                'is_consistent': False,
                'cumulative_return': 0.0,
                'days_analyzed': len(data),
                'consistent_days': 0,
                'avg_daily_return': 0.0
            }

        # Get recent data
        recent_data = data.tail(days)
        daily_returns = self.calculate_daily_returns(recent_data)

        # Remove the first NaN value (first day has no previous day to compare)
        daily_returns = daily_returns.dropna()

        if len(daily_returns) == 0:
            return {
                'is_consistent': False,
                'cumulative_return': 0.0,
                'days_analyzed': 0,
                'consistent_days': 0,
                'avg_daily_return': 0.0
            }

        # Count days with gains in the specified range
        consistent_days = ((daily_returns >= min_gain) & (daily_returns <= max_gain)).sum()

        # Calculate metrics
        cumulative_return = self.calculate_cumulative_return(recent_data, days)
        avg_daily_return = daily_returns.mean()
        consistency_ratio = consistent_days / len(daily_returns)

        # Determine if the stock is consistently gaining
        # Require at least 60% of days to be in the gain range
        is_consistent = consistency_ratio >= 0.6 and consistent_days >= 3

        return {
            'is_consistent': is_consistent,
            'cumulative_return': cumulative_return,
            'days_analyzed': len(daily_returns),
            'consistent_days': int(consistent_days),
            'avg_daily_return': avg_daily_return,
            'consistency_ratio': consistency_ratio
        }

    def find_consistent_gainers(self, stocks_data: Dict[str, pd.DataFrame],
                               min_gain: float = 2.0, max_gain: float = 5.0,
                               days: int = 5) -> List[Dict]:
        """
        Find stocks that have shown consistent gains in the specified range

        Args:
            stocks_data: Dictionary mapping stock symbols to their DataFrames
            min_gain: Minimum daily gain percentage
            max_gain: Maximum daily gain percentage
            days: Number of days to analyze

        Returns:
            List of dictionaries with stock analysis results
        """
        consistent_gainers = []

        for symbol, data in stocks_data.items():
            if data.empty:
                continue

            analysis = self.analyze_consistency(data, min_gain, max_gain, days)

            if analysis['is_consistent']:
                result = {
                    'symbol': symbol,
                    'cumulative_return': analysis['cumulative_return'],
                    'consistent_days': analysis['consistent_days'],
                    'avg_daily_return': analysis['avg_daily_return'],
                    'consistency_ratio': analysis['consistency_ratio'],
                    'analysis_period_days': days
                }
                consistent_gainers.append(result)

        # Sort by cumulative return (highest first)
        consistent_gainers.sort(key=lambda x: x['cumulative_return'], reverse=True)

        logger.info(f"Found {len(consistent_gainers)} stocks with consistent {min_gain}-{max_gain}% gains over {days} days")
        return consistent_gainers

    def get_stock_summary(self, symbol: str, data: pd.DataFrame) -> Dict:
        """
        Get comprehensive summary for a single stock

        Args:
            symbol: Stock symbol
            data: Stock price data

        Returns:
            Dictionary with stock summary
        """
        if data.empty:
            return {'symbol': symbol, 'error': 'No data available'}

        latest_data = data.iloc[-1]
        previous_data = data.iloc[-2] if len(data) > 1 else latest_data

        # Calculate recent performance
        week_5_analysis = self.analyze_consistency(data, days=5)
        week_10_analysis = self.analyze_consistency(data, days=10)

        summary = {
            'symbol': symbol,
            'current_price': latest_data['Close'],
            'previous_close': previous_data['Close'],
            'daily_change_pct': ((latest_data['Close'] - previous_data['Close']) / previous_data['Close']) * 100,
            'volume': latest_data['Volume'],
            '5_day_analysis': week_5_analysis,
            '10_day_analysis': week_10_analysis,
            'data_points': len(data)
        }

        return summary

    def rank_stocks_by_momentum(self, stocks_data: Dict[str, pd.DataFrame],
                               days: int = 5) -> List[Dict]:
        """
        Rank all stocks by their momentum over the specified period

        Args:
            stocks_data: Dictionary mapping stock symbols to their DataFrames
            days: Number of days to analyze momentum

        Returns:
            List of stocks ranked by momentum score
        """
        momentum_scores = []

        for symbol, data in stocks_data.items():
            if data.empty or len(data) < days:
                continue

            cumulative_return = self.calculate_cumulative_return(data, days)
            volatility = data['Close'].pct_change().std() * 100  # Daily volatility

            # Calculate momentum score (return adjusted for volatility)
            if volatility > 0:
                momentum_score = cumulative_return / volatility
            else:
                momentum_score = cumulative_return

            momentum_scores.append({
                'symbol': symbol,
                'cumulative_return': cumulative_return,
                'volatility': volatility,
                'momentum_score': momentum_score,
                'current_price': data['Close'].iloc[-1]
            })

        # Sort by momentum score
        momentum_scores.sort(key=lambda x: x['momentum_score'], reverse=True)

        logger.info(f"Ranked {len(momentum_scores)} stocks by momentum over {days} days")
        return momentum_scores
