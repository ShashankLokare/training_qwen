"""
Fundamentals Data Provider for Nifty Trading Agent
Fetches quarterly results, financial statements, and valuation metrics
"""

import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import json

from ..utils.logging_utils import get_logger
from ..utils.io_utils import save_json_file, load_json_file, file_exists

logger = get_logger(__name__)

class FundamentalDataProvider:
    """
    Provides fundamental data for Indian stocks including quarterly results,
    valuation metrics, and financial ratios.
    """

    def __init__(self, cache_dir: str = "data/fundamentals_data", cache_expiry_hours: int = 168):
        """
        Initialize the fundamentals data provider

        Args:
            cache_dir: Directory to store cached data
            cache_expiry_hours: Cache expiry time in hours (default: 1 week)
        """
        from pathlib import Path
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_expiry_hours = cache_expiry_hours
        self.cache_expiry_seconds = cache_expiry_hours * 3600

        logger.info(f"FundamentalDataProvider initialized with cache_dir: {cache_dir}")

    def get_quarterly_results(self, symbol: str, num_quarters: int = 8) -> pd.DataFrame:
        """
        Get quarterly financial results for a company

        Args:
            symbol: Stock symbol (e.g., 'RELIANCE.NS')
            num_quarters: Number of quarters to fetch

        Returns:
            DataFrame with quarterly results
        """
        logger.info(f"Fetching quarterly results for {symbol}")

        # Check cache first
        cache_key = f"{symbol}_quarterly_{num_quarters}"
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            logger.info(f"Loaded quarterly data from cache for {symbol}")
            return pd.DataFrame(cached_data)

        try:
            # For demo purposes, generate synthetic quarterly data
            # In production, this would integrate with NSE APIs or financial data providers
            quarterly_data = self._generate_synthetic_quarterly_data(symbol, num_quarters)

            # Cache the data
            self._save_to_cache(cache_key, quarterly_data.to_dict('records'))

            logger.info(f"Successfully fetched {len(quarterly_data)} quarters of data for {symbol}")
            return quarterly_data

        except Exception as e:
            logger.error(f"Error fetching quarterly results for {symbol}: {e}")
            return pd.DataFrame()

    def get_latest_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """
        Get latest fundamental metrics for a company

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with fundamental metrics
        """
        logger.info(f"Fetching latest fundamentals for {symbol}")

        # Check cache first
        cache_key = f"{symbol}_latest_fundamentals"
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            logger.info(f"Loaded latest fundamentals from cache for {symbol}")
            return cached_data

        try:
            # Generate synthetic fundamental data
            fundamentals = self._generate_synthetic_fundamentals(symbol)

            # Cache the data
            self._save_to_cache(cache_key, fundamentals)

            logger.info(f"Successfully fetched fundamentals for {symbol}")
            return fundamentals

        except Exception as e:
            logger.error(f"Error fetching fundamentals for {symbol}: {e}")
            return {}

    def get_valuation_metrics(self, symbol: str) -> Dict[str, float]:
        """
        Get valuation metrics like P/E, P/B, EV/EBITDA, etc.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with valuation metrics
        """
        logger.info(f"Fetching valuation metrics for {symbol}")

        fundamentals = self.get_latest_fundamentals(symbol)

        if not fundamentals:
            return {}

        # Calculate valuation metrics
        try:
            current_price = fundamentals.get('current_price', 0)
            if current_price <= 0:
                return {}

            valuation = {
                'pe_ratio': fundamentals.get('pe_ratio', 0),
                'pb_ratio': fundamentals.get('pb_ratio', 0),
                'dividend_yield': fundamentals.get('dividend_yield', 0),
                'roe': fundamentals.get('roe', 0),
                'roce': fundamentals.get('roce', 0),
                'debt_to_equity': fundamentals.get('debt_to_equity', 0),
                'current_ratio': fundamentals.get('current_ratio', 0),
                'gross_margin': fundamentals.get('gross_margin', 0),
                'net_margin': fundamentals.get('net_margin', 0),
                'market_cap': fundamentals.get('market_cap', 0)
            }

            return valuation

        except Exception as e:
            logger.error(f"Error calculating valuation metrics for {symbol}: {e}")
            return {}

    def get_growth_metrics(self, symbol: str, periods: List[str] = ['1Y', '3Y', '5Y']) -> Dict[str, Dict[str, float]]:
        """
        Get growth metrics over different time periods

        Args:
            symbol: Stock symbol
            periods: List of periods to analyze

        Returns:
            Dictionary with growth metrics by period
        """
        logger.info(f"Fetching growth metrics for {symbol}")

        quarterly_data = self.get_quarterly_results(symbol, num_quarters=20)

        if quarterly_data.empty:
            return {}

        growth_metrics = {}

        try:
            # Sort by date
            quarterly_data = quarterly_data.sort_values('date')

            for period in periods:
                if period == '1Y':
                    quarters = 4
                elif period == '3Y':
                    quarters = 12
                elif period == '5Y':
                    quarters = 20
                else:
                    continue

                if len(quarterly_data) >= quarters:
                    recent_data = quarterly_data.tail(quarters)

                    # Calculate growth rates
                    start_revenue = recent_data['revenue'].iloc[0]
                    end_revenue = recent_data['revenue'].iloc[-1]

                    start_eps = recent_data['eps'].iloc[0]
                    end_eps = recent_data['eps'].iloc[-1]

                    revenue_growth = ((end_revenue - start_revenue) / start_revenue) * 100 if start_revenue > 0 else 0
                    eps_growth = ((end_eps - start_eps) / start_eps) * 100 if start_eps > 0 else 0

                    growth_metrics[period] = {
                        'revenue_growth_pct': revenue_growth,
                        'eps_growth_pct': eps_growth,
                        'period_quarters': quarters
                    }

            return growth_metrics

        except Exception as e:
            logger.error(f"Error calculating growth metrics for {symbol}: {e}")
            return {}

    def _generate_synthetic_quarterly_data(self, symbol: str, num_quarters: int) -> pd.DataFrame:
        """
        Generate synthetic quarterly financial data for demonstration

        Args:
            symbol: Stock symbol
            num_quarters: Number of quarters to generate

        Returns:
            DataFrame with quarterly data
        """
        import numpy as np

        # Base values for different companies
        company_bases = {
            'RELIANCE.NS': {'revenue': 250000, 'profit': 18000, 'eps': 45},
            'TCS.NS': {'revenue': 65000, 'profit': 12000, 'eps': 52},
            'HDFCBANK.NS': {'revenue': 45000, 'profit': 9500, 'eps': 28},
            'ICICIBANK.NS': {'revenue': 38000, 'profit': 7200, 'eps': 22},
            'INFY.NS': {'revenue': 42000, 'profit': 8500, 'eps': 38},
            'HINDUNILVR.NS': {'revenue': 15000, 'profit': 2800, 'eps': 25},
            'ITC.NS': {'revenue': 18000, 'profit': 3200, 'eps': 18},
            'KOTAKBANK.NS': {'revenue': 12000, 'profit': 2500, 'eps': 35},
            'LT.NS': {'revenue': 22000, 'profit': 1800, 'eps': 15},
            'AXISBANK.NS': {'revenue': 25000, 'profit': 4800, 'eps': 12}
        }

        base_values = company_bases.get(symbol, {'revenue': 10000, 'profit': 1500, 'eps': 10})

        quarterly_data = []

        # Generate quarterly data going backwards
        current_date = datetime.now()

        for i in range(num_quarters):
            # Calculate quarter date
            quarter_date = current_date - timedelta(days=i * 90)

            # Add some random variation and slight growth trend
            growth_factor = 1 + (i * 0.02)  # Slight growth over time
            noise_factor = np.random.normal(1, 0.1)  # Random noise

            revenue = base_values['revenue'] * growth_factor * noise_factor
            profit = base_values['profit'] * growth_factor * noise_factor
            eps = base_values['eps'] * growth_factor * noise_factor

            quarterly_data.append({
                'date': quarter_date.strftime('%Y-%m-%d'),
                'quarter': f"Q{((quarter_date.month - 1) // 3) + 1} {quarter_date.year}",
                'revenue': round(revenue, 2),
                'profit': round(profit, 2),
                'eps': round(eps, 2),
                'margin': round((profit / revenue) * 100, 2) if revenue > 0 else 0
            })

        return pd.DataFrame(quarterly_data)

    def _generate_synthetic_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """
        Generate synthetic fundamental data for demonstration

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with fundamental data
        """
        import numpy as np

        # Base fundamentals for different companies
        company_fundamentals = {
            'RELIANCE.NS': {
                'current_price': 1540, 'pe_ratio': 22.5, 'pb_ratio': 2.8,
                'roe': 12.5, 'roce': 14.2, 'debt_to_equity': 0.45,
                'market_cap': 1800000, 'dividend_yield': 0.35
            },
            'TCS.NS': {
                'current_price': 3240, 'pe_ratio': 28.5, 'pb_ratio': 12.8,
                'roe': 45.2, 'roce': 52.1, 'debt_to_equity': 0.12,
                'market_cap': 1400000, 'dividend_yield': 1.8
            },
            'HDFCBANK.NS': {
                'current_price': 1000, 'pe_ratio': 18.2, 'pb_ratio': 2.9,
                'roe': 16.8, 'roce': 6.2, 'debt_to_equity': 6.8,
                'market_cap': 850000, 'dividend_yield': 1.2
            },
            'ICICIBANK.NS': {
                'current_price': 1390, 'pe_ratio': 16.8, 'pb_ratio': 3.1,
                'roe': 18.5, 'roce': 5.8, 'debt_to_equity': 7.2,
                'market_cap': 720000, 'dividend_yield': 0.8
            },
            'INFY.NS': {
                'current_price': 1616, 'pe_ratio': 24.8, 'pb_ratio': 7.2,
                'roe': 28.5, 'roce': 35.2, 'debt_to_equity': 0.08,
                'market_cap': 650000, 'dividend_yield': 2.1
            }
        }

        base_fundamentals = company_fundamentals.get(symbol, {
            'current_price': 500, 'pe_ratio': 20.0, 'pb_ratio': 3.0,
            'roe': 15.0, 'roce': 18.0, 'debt_to_equity': 0.5,
            'market_cap': 100000, 'dividend_yield': 1.0
        })

        # Add some variation
        fundamentals = {}
        for key, value in base_fundamentals.items():
            if isinstance(value, (int, float)):
                # Add 10% random variation
                variation = np.random.normal(1, 0.1)
                fundamentals[key] = round(value * variation, 2)
            else:
                fundamentals[key] = value

        # Add additional metrics
        fundamentals.update({
            'current_ratio': round(np.random.normal(1.5, 0.3), 2),
            'gross_margin': round(np.random.normal(35, 5), 2),
            'net_margin': round(np.random.normal(12, 3), 2),
            'beta': round(np.random.normal(1.0, 0.2), 2),
            'sector': self._get_sector_for_symbol(symbol),
            'last_updated': datetime.now().isoformat()
        })

        return fundamentals

    def _get_sector_for_symbol(self, symbol: str) -> str:
        """Get sector classification for a symbol"""
        sector_map = {
            'RELIANCE.NS': 'Energy',
            'TCS.NS': 'IT Services',
            'HDFCBANK.NS': 'Banking',
            'ICICIBANK.NS': 'Banking',
            'INFY.NS': 'IT Services',
            'HINDUNILVR.NS': 'FMCG',
            'ITC.NS': 'FMCG',
            'KOTAKBANK.NS': 'Banking',
            'LT.NS': 'Construction',
            'AXISBANK.NS': 'Banking'
        }
        return sector_map.get(symbol, 'Unknown')

    def _load_from_cache(self, cache_key: str) -> Optional[Any]:
        """Load data from cache"""
        cache_file = self.cache_dir / f"{cache_key}.json"

        if not cache_file.exists():
            return None

        # Check if cache is expired
        if not self._is_cache_valid(cache_file):
            cache_file.unlink()
            return None

        try:
            return load_json_file(str(cache_file))
        except Exception as e:
            logger.warning(f"Error loading cache for {cache_key}: {e}")
            return None

    def _save_to_cache(self, cache_key: str, data: Any) -> None:
        """Save data to cache"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            save_json_file(data, str(cache_file))
        except Exception as e:
            logger.warning(f"Error saving cache for {cache_key}: {e}")

    def _is_cache_valid(self, cache_file: str) -> bool:
        """Check if cache file is still valid"""
        from ..utils.io_utils import get_file_modification_time
        import time

        mtime = get_file_modification_time(cache_file)
        if mtime is None:
            return False

        current_time = time.time()
        return (current_time - mtime) <= self.cache_expiry_seconds
