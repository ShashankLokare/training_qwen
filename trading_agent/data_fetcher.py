"""
Data Fetcher for Indian Market Trading Agent
Fetches Nifty50 and individual stock data from Yahoo Finance
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IndianMarketDataFetcher:
    """Fetches data from Indian stock market indices and stocks"""

    # Nifty50 symbols on Yahoo Finance
    NIFTY50_SYMBOL = "^NSEI"

    # Some major Nifty50 stocks (can be expanded)
    NIFTY50_STOCKS = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
        "HINDUNILVR.NS", "ITC.NS", "KOTAKBANK.NS", "LT.NS", "AXISBANK.NS",
        "BAJFINANCE.NS", "MARUTI.NS", "BAJAJ-AUTO.NS", "WIPRO.NS", "HCLTECH.NS"
    ]

    def __init__(self):
        """Initialize the data fetcher"""
        self.session = None

    def fetch_nifty50_data(self, days: int = 30) -> pd.DataFrame:
        """
        Fetch Nifty50 opening and closing values for the last N days

        Args:
            days: Number of days of historical data to fetch

        Returns:
            DataFrame with Open, Close, High, Low, Volume columns
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            logger.info(f"Fetching Nifty50 data from {start_date.date()} to {end_date.date()}")

            nifty = yf.Ticker(self.NIFTY50_SYMBOL)
            data = nifty.history(start=start_date, end=end_date, interval="1d")

            if data.empty:
                logger.warning("No Nifty50 data retrieved")
                return pd.DataFrame()

            # Keep only relevant columns
            data = data[['Open', 'Close', 'High', 'Low', 'Volume']]

            logger.info(f"Successfully fetched {len(data)} days of Nifty50 data")
            return data

        except Exception as e:
            logger.error(f"Error fetching Nifty50 data: {e}")
            return pd.DataFrame()

    def fetch_stock_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """
        Fetch individual stock data

        Args:
            symbol: Stock symbol (e.g., 'RELIANCE.NS')
            days: Number of days of historical data to fetch

        Returns:
            DataFrame with stock price data
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            logger.info(f"Fetching {symbol} data from {start_date.date()} to {end_date.date()}")

            stock = yf.Ticker(symbol)
            data = stock.history(start=start_date, end=end_date, interval="1d")

            if data.empty:
                logger.warning(f"No data retrieved for {symbol}")
                return pd.DataFrame()

            # Keep only relevant columns
            data = data[['Open', 'Close', 'High', 'Low', 'Volume']]

            logger.info(f"Successfully fetched {len(data)} days of {symbol} data")
            return data

        except Exception as e:
            logger.error(f"Error fetching {symbol} data: {e}")
            return pd.DataFrame()

    def fetch_multiple_stocks(self, symbols: List[str], days: int = 30) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks

        Args:
            symbols: List of stock symbols
            days: Number of days of historical data to fetch

        Returns:
            Dictionary mapping symbols to their DataFrames
        """
        results = {}
        for symbol in symbols:
            data = self.fetch_stock_data(symbol, days)
            if not data.empty:
                results[symbol] = data

        logger.info(f"Successfully fetched data for {len(results)} out of {len(symbols)} stocks")
        return results

    def get_nifty50_stocks_list(self) -> List[str]:
        """
        Get list of Nifty50 stock symbols

        Returns:
            List of stock symbols
        """
        return self.NIFTY50_STOCKS.copy()

    def get_current_nifty_level(self) -> Optional[float]:
        """
        Get current Nifty50 level

        Returns:
            Current Nifty50 value or None if unavailable
        """
        try:
            nifty = yf.Ticker(self.NIFTY50_SYMBOL)
            info = nifty.info
            return info.get('regularMarketPrice')

        except Exception as e:
            logger.error(f"Error fetching current Nifty level: {e}")
            return None
