"""
Date and time utilities for the Nifty Trading Agent
"""

from datetime import datetime, timedelta, date
from typing import List, Tuple, Optional
import pandas as pd

def get_trading_days(start_date: date, end_date: date) -> List[date]:
    """
    Get list of trading days between start and end dates
    Excludes weekends (Saturday, Sunday)

    Args:
        start_date: Start date
        end_date: End date

    Returns:
        List of trading days
    """
    trading_days = []
    current_date = start_date

    while current_date <= end_date:
        # Monday = 0, Sunday = 6
        if current_date.weekday() < 5:  # Monday to Friday
            trading_days.append(current_date)
        current_date += timedelta(days=1)

    return trading_days

def get_business_days(start_date: date, end_date: date) -> List[date]:
    """
    Get list of business days (same as trading days for NSE)

    Args:
        start_date: Start date
        end_date: End date

    Returns:
        List of business days
    """
    return get_trading_days(start_date, end_date)

def is_trading_day(input_date: date) -> bool:
    """
    Check if a given date is a trading day

    Args:
        input_date: Date to check

    Returns:
        True if trading day, False otherwise
    """
    # Monday = 0, Sunday = 6
    return input_date.weekday() < 5

def get_next_trading_day(from_date: date) -> date:
    """
    Get the next trading day from a given date

    Args:
        from_date: Starting date

    Returns:
        Next trading day
    """
    next_date = from_date + timedelta(days=1)
    while not is_trading_day(next_date):
        next_date += timedelta(days=1)
    return next_date

def get_previous_trading_day(from_date: date) -> date:
    """
    Get the previous trading day from a given date

    Args:
        from_date: Starting date

    Returns:
        Previous trading day
    """
    prev_date = from_date - timedelta(days=1)
    while not is_trading_day(prev_date):
        prev_date -= timedelta(days=1)
    return prev_date

def parse_date(date_str: str, format_str: str = "%Y-%m-%d") -> Optional[date]:
    """
    Parse date string to date object

    Args:
        date_str: Date string
        format_str: Date format string

    Returns:
        Date object or None if parsing fails
    """
    try:
        dt = datetime.strptime(date_str, format_str)
        return dt.date()
    except ValueError:
        return None

def format_date(input_date: date, format_str: str = "%Y-%m-%d") -> str:
    """
    Format date object to string

    Args:
        input_date: Date object
        format_str: Date format string

    Returns:
        Formatted date string
    """
    return input_date.strftime(format_str)

def get_date_range(start_date: str, end_date: str) -> Tuple[date, date]:
    """
    Parse date range strings and return date objects

    Args:
        start_date: Start date string
        end_date: End date string

    Returns:
        Tuple of (start_date, end_date) as date objects
    """
    start = parse_date(start_date)
    end = parse_date(end_date)

    if start is None or end is None:
        raise ValueError("Invalid date format. Use YYYY-MM-DD format.")

    if start > end:
        raise ValueError("Start date cannot be after end date.")

    return start, end

def get_trading_days_count(start_date: date, end_date: date) -> int:
    """
    Get the number of trading days between two dates (inclusive)

    Args:
        start_date: Start date
        end_date: End date

    Returns:
        Number of trading days
    """
    trading_days = get_trading_days(start_date, end_date)
    return len(trading_days)

def add_trading_days(from_date: date, days: int) -> date:
    """
    Add a specified number of trading days to a date

    Args:
        from_date: Starting date
        days: Number of trading days to add (can be negative)

    Returns:
        Resulting date
    """
    if days == 0:
        return from_date

    current_date = from_date
    remaining_days = abs(days)

    while remaining_days > 0:
        if days > 0:
            current_date = get_next_trading_day(current_date)
        else:
            current_date = get_previous_trading_day(current_date)
        remaining_days -= 1

    return current_date

def get_market_hours_check(input_datetime: Optional[datetime] = None) -> bool:
    """
    Check if current time is within NSE market hours
    NSE trading hours: 9:15 AM to 3:30 PM IST, Monday to Friday

    Args:
        input_datetime: Datetime to check (default: current time)

    Returns:
        True if within market hours, False otherwise
    """
    if input_datetime is None:
        input_datetime = datetime.now()

    # Check if it's a trading day
    if not is_trading_day(input_datetime.date()):
        return False

    # NSE market hours: 9:15 AM to 3:30 PM IST
    market_start = input_datetime.replace(hour=9, minute=15, second=0, microsecond=0)
    market_end = input_datetime.replace(hour=15, minute=30, second=0, microsecond=0)

    return market_start <= input_datetime <= market_end

def get_quarter_dates(year: int, quarter: int) -> Tuple[date, date]:
    """
    Get start and end dates for a specific quarter

    Args:
        year: Year
        quarter: Quarter (1-4)

    Returns:
        Tuple of (start_date, end_date)
    """
    if quarter == 1:
        start_date = date(year, 1, 1)
        end_date = date(year, 3, 31)
    elif quarter == 2:
        start_date = date(year, 4, 1)
        end_date = date(year, 6, 30)
    elif quarter == 3:
        start_date = date(year, 7, 1)
        end_date = date(year, 9, 30)
    elif quarter == 4:
        start_date = date(year, 10, 1)
        end_date = date(year, 12, 31)
    else:
        raise ValueError("Quarter must be between 1 and 4")

    return start_date, end_date

def get_current_quarter() -> Tuple[int, int]:
    """
    Get current quarter and year

    Returns:
        Tuple of (year, quarter)
    """
    today = date.today()
    year = today.year
    month = today.month

    if month <= 3:
        quarter = 1
    elif month <= 6:
        quarter = 2
    elif month <= 9:
        quarter = 3
    else:
        quarter = 4

    return year, quarter

def date_to_pandas_timestamp(input_date: date) -> pd.Timestamp:
    """
    Convert date object to pandas Timestamp

    Args:
        input_date: Date object

    Returns:
        Pandas Timestamp
    """
    return pd.Timestamp(input_date)

def pandas_timestamp_to_date(timestamp: pd.Timestamp) -> date:
    """
    Convert pandas Timestamp to date object

    Args:
        timestamp: Pandas Timestamp

    Returns:
        Date object
    """
    return timestamp.date()
