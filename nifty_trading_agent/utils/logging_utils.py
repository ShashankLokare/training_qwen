"""
Logging utilities for the Nifty Trading Agent
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Dict, Optional

def setup_logging(config: Dict) -> None:
    """
    Setup logging configuration for the application

    Args:
        config: Dictionary containing logging configuration
               - level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
               - format: Log format string
               - file_path: Path to log file (optional)
               - max_file_size_mb: Maximum file size in MB (optional)
               - backup_count: Number of backup files to keep (optional)
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, config.get('level', 'INFO')))

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    file_path = config.get('file_path')
    if file_path:
        # Ensure directory exists
        log_file = Path(file_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # Rotating file handler
        max_bytes = config.get('max_file_size_mb', 10) * 1024 * 1024
        backup_count = config.get('backup_count', 5)

        file_handler = logging.handlers.RotatingFileHandler(
            file_path,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)

def log_function_call(func_name: str, args: Optional[Dict] = None) -> None:
    """
    Log function call for debugging purposes

    Args:
        func_name: Function name
        args: Function arguments (optional)
    """
    logger = get_logger(__name__)
    if args:
        logger.debug(f"Calling {func_name} with args: {args}")
    else:
        logger.debug(f"Calling {func_name}")

def log_performance(func_name: str, start_time: float, end_time: float) -> None:
    """
    Log function performance

    Args:
        func_name: Function name
        start_time: Start time (from time.time())
        end_time: End time (from time.time())
    """
    duration = end_time - start_time
    logger = get_logger(__name__)
    logger.info(".4f")

def setup_daily_log_file(base_log_dir: str = "logs") -> str:
    """
    Setup a daily log file

    Args:
        base_log_dir: Base directory for logs

    Returns:
        Path to the daily log file
    """
    from datetime import datetime

    # Create logs directory if it doesn't exist
    log_dir = Path(base_log_dir)
    log_dir.mkdir(exist_ok=True)

    # Create daily log file name
    today = datetime.now().strftime("%Y%m%d")
    log_file = log_dir / f"trading_agent_{today}.log"

    return str(log_file)
