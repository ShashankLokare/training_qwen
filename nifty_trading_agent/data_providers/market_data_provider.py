"""
Market Data Provider for Nifty Trading Agent
Fetches historical OHLCV data for Indian stocks using Yahoo Finance
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.logging_utils import get_logger
from utils.io_utils import save_csv_file, load_csv_file, file_exists, get_file_modification_time
from utils.date_utils import date_to_pandas_timestamp

logger = get_logger(__name__)

class MarketDataProvider:
    """
    Provides market data for Indian stocks and indices

    Uses Yahoo Finance as the data source with caching capabilities.
    """

    def __init__(self, cache_dir: str = "data/price_data", cache_expiry_hours: int = 24):
        """
        Initialize the market data provider

        Args:
            cache_dir: Directory to store cached data
            cache_expiry_hours: Cache expiry time in hours
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_expiry_hours = cache_expiry_hours
        self.cache_expiry_seconds = cache_expiry_hours * 3600

        logger.info(f"MarketDataProvider initialized with cache_dir: {cache_dir}")

    def get_ohlcv(self, symbol: str, start_date: str, end_date: str,
                  interval: str = "1d", use_cache: bool = True) -> pd.DataFrame:
        """
        Get OHLCV (Open, High, Low, Close, Volume) data for a symbol

        Args:
            symbol: Stock symbol (e.g., 'RELIANCE.NS')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval ('1d', '1h', '30m', '15m', '5m', '1m')
            use_cache: Whether to use cached data if available

        Returns:
            DataFrame with OHLCV data and Date as index
        """
        logger.info(f"Fetching OHLCV data for {symbol} from {start_date} to {end_date}")

        # Check cache first if enabled
        if use_cache:
            cached_data = self._load_from_cache(symbol, start_date, end_date, interval)
            if cached_data is not None:
                logger.info(f"Loaded {len(cached_data)} rows from cache for {symbol}")
                return cached_data

        try:
            # Fetch data from Yahoo Finance
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True,
                prepost=False
            )

            if data.empty:
                logger.warning(f"No data retrieved for {symbol}")
                return pd.DataFrame()

            # Clean and format data
            data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

            # Ensure we have the expected columns
            if len(data) == 0:
                logger.warning(f"No valid OHLCV data for {symbol}")
                return pd.DataFrame()

            # Reset index to have Date as a column
            data = data.reset_index()
            data['Date'] = pd.to_datetime(data['Date']).dt.date

            logger.info(f"Successfully fetched {len(data)} rows of OHLCV data for {symbol}")

            # Cache the data
            if use_cache:
                self._save_to_cache(data, symbol, start_date, end_date, interval)

            return data

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def get_index_data(self, index_symbol: str, start_date: str, end_date: str,
                      interval: str = "1d", use_cache: bool = True) -> pd.DataFrame:
        """
        Get data for an index (e.g., Nifty 50, Bank Nifty)

        Args:
            index_symbol: Index symbol (e.g., '^NSEI' for Nifty 50)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval
            use_cache: Whether to use cached data

        Returns:
            DataFrame with index data
        """
        logger.info(f"Fetching index data for {index_symbol}")
        return self.get_ohlcv(index_symbol, start_date, end_date, interval, use_cache)

    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current/latest price for a symbol

        Args:
            symbol: Stock symbol

        Returns:
            Current price or None if unavailable
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Try different price fields
            price_fields = ['regularMarketPrice', 'currentPrice', 'previousClose']
            for field in price_fields:
                price = info.get(field)
                if price is not None:
                    return float(price)

            # Fallback: get latest historical data
            data = ticker.history(period="1d")
            if not data.empty:
                return float(data['Close'].iloc[-1])

            logger.warning(f"Could not get current price for {symbol}")
            return None

        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None

    def get_multiple_symbols(self, symbols: list, start_date: str, end_date: str,
                           interval: str = "1d", use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Get OHLCV data for multiple symbols

        Args:
            symbols: List of stock symbols
            start_date: Start date
            end_date: End date
            interval: Data interval
            use_cache: Whether to use cache

        Returns:
            Dictionary mapping symbols to their DataFrames
        """
        results = {}
        total_symbols = len(symbols)

        logger.info(f"Fetching data for {total_symbols} symbols")

        for i, symbol in enumerate(symbols, 1):
            logger.info(f"Processing {symbol} ({i}/{total_symbols})")
            data = self.get_ohlcv(symbol, start_date, end_date, interval, use_cache)

            if not data.empty:
                results[symbol] = data

        successful_fetches = len(results)
        logger.info(f"Successfully fetched data for {successful_fetches}/{total_symbols} symbols")

        return results

    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get basic information about a symbol

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with symbol information or None if unavailable
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            if not info:
                return None

            # Extract relevant information
            symbol_info = {
                'symbol': symbol,
                'name': info.get('longName', info.get('shortName', symbol)),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'pb_ratio': info.get('priceToBook'),
                'dividend_yield': info.get('dividendYield'),
                'beta': info.get('beta'),
                'currency': info.get('currency', 'INR')
            }

            return symbol_info

        except Exception as e:
            logger.error(f"Error getting symbol info for {symbol}: {e}")
            return None

    def _get_cache_filename(self, symbol: str, start_date: str, end_date: str, interval: str) -> str:
        """
        Generate cache filename for the given parameters

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            interval: Data interval

        Returns:
            Cache filename
        """
        # Sanitize symbol for filename
        safe_symbol = symbol.replace('^', '').replace('.', '_')
        filename = f"{safe_symbol}_{start_date}_{end_date}_{interval}.csv"
        return self.cache_dir / filename

    def _load_from_cache(self, symbol: str, start_date: str, end_date: str, interval: str) -> Optional[pd.DataFrame]:
        """
        Load data from cache if available and not expired

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            interval: Data interval

        Returns:
            Cached DataFrame or None if not available/expired
        """
        cache_file = self._get_cache_filename(symbol, start_date, end_date, interval)

        if not cache_file.exists():
            return None

        # Check if cache is expired
        if not self._is_cache_valid(str(cache_file)):
            logger.debug(f"Cache expired for {symbol}, removing old cache")
            cache_file.unlink()  # Remove expired cache
            return None

        try:
            # Load cached data
            data = load_csv_file(str(cache_file))

            # Convert Date column back to datetime if present
            if 'Date' in data.columns:
                data['Date'] = pd.to_datetime(data['Date']).dt.date

            logger.debug(f"Loaded {len(data)} rows from cache for {symbol}")
            return data

        except Exception as e:
            logger.warning(f"Error loading cache for {symbol}: {e}")
            # Remove corrupted cache file
            if cache_file.exists():
                cache_file.unlink()
            return None

    def _save_to_cache(self, data: pd.DataFrame, symbol: str, start_date: str, end_date: str, interval: str) -> None:
        """
        Save data to cache

        Args:
            data: DataFrame to cache
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            interval: Data interval
        """
        if data.empty:
            return

        try:
            cache_file = self._get_cache_filename(symbol, start_date, end_date, interval)
            save_csv_file(data, str(cache_file), index=False)
            logger.debug(f"Cached {len(data)} rows for {symbol}")

        except Exception as e:
            logger.warning(f"Error saving cache for {symbol}: {e}")

    def _is_cache_valid(self, cache_file_path: str) -> bool:
        """
        Check if cache file is still valid (not expired)

        Args:
            cache_file_path: Path to cache file

        Returns:
            True if cache is valid, False otherwise
        """
        mtime = get_file_modification_time(cache_file_path)
        if mtime is None:
            return False

        import time
        current_time = time.time()
        return (current_time - mtime) <= self.cache_expiry_seconds

    def clear_cache(self, pattern: str = "*") -> int:
        """
        Clear cache files matching a pattern

        Args:
            pattern: File pattern to match (default: all files)

        Returns:
            Number of files removed
        """
        import glob

        cache_pattern = str(self.cache_dir / pattern)
        cache_files = glob.glob(cache_pattern)

        removed_count = 0
        for cache_file in cache_files:
            try:
                Path(cache_file).unlink()
                removed_count += 1
            except Exception as e:
                logger.warning(f"Error removing cache file {cache_file}: {e}")

        logger.info(f"Cleared {removed_count} cache files")
        return removed_count

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics

        Returns:
            Dictionary with cache statistics
        """
        total_files = 0
        total_size = 0

        if self.cache_dir.exists():
            for cache_file in self.cache_dir.iterdir():
                if cache_file.is_file():
                    total_files += 1
                    total_size += cache_file.stat().st_size

        return {
            'cache_directory': str(self.cache_dir),
            'total_files': total_files,
            'total_size_mb': total_size / (1024 * 1024),
            'cache_expiry_hours': self.cache_expiry_hours
        }
