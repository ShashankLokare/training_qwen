"""
PostgreSQL Database Connection Helper
Provides connection management and context managers for PostgreSQL operations
"""

import psycopg2
from psycopg2 import pool
from contextlib import contextmanager
from typing import Optional, Any, Dict
import logging

try:
    from ..utils.logging_utils import get_logger
    from ..utils.io_utils import load_yaml_config
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.logging_utils import get_logger
    from utils.io_utils import load_yaml_config

logger = get_logger(__name__)

class PostgresConnectionManager:
    """
    Manages PostgreSQL connections and provides context managers for safe operations
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the connection manager

        Args:
            config: Database configuration. If None, loads from config.yaml
        """
        if config is None:
            full_config = load_yaml_config("config/config.yaml")
            config = full_config.get('databases', {}).get('postgres', {})

        self.config = config
        self._connection_pool = None
        self._initialize_pool()

    def _initialize_pool(self) -> None:
        """Initialize the connection pool"""
        try:
            self._connection_pool = pool.SimpleConnectionPool(
                minconn=1,
                maxconn=self.config.get('connection_pool_size', 10),
                host=self.config.get('host', 'localhost'),
                port=self.config.get('port', 5432),
                dbname=self.config.get('dbname', 'nifty_trading'),
                user=self.config.get('user', 'trader'),
                password=self.config.get('password', 'secret123'),
                sslmode=self.config.get('sslmode', 'prefer'),
                connect_timeout=self.config.get('connection_timeout', 30)
            )
            logger.info(f"PostgreSQL connection pool initialized for {self.config.get('dbname')}")
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL connection pool: {e}")
            raise

    def get_connection(self):
        """
        Get a connection from the pool

        Returns:
            PostgreSQL connection object
        """
        if self._connection_pool is None:
            raise RuntimeError("Connection pool not initialized")

        try:
            conn = self._connection_pool.getconn()
            return conn
        except Exception as e:
            logger.error(f"Failed to get connection from pool: {e}")
            raise

    def return_connection(self, conn) -> None:
        """
        Return a connection to the pool

        Args:
            conn: Connection to return
        """
        if self._connection_pool is not None:
            self._connection_pool.putconn(conn)

    def close_all(self) -> None:
        """Close all connections in the pool"""
        if self._connection_pool is not None:
            self._connection_pool.closeall()
            logger.info("All PostgreSQL connections closed")

    @contextmanager
    def get_cursor(self, commit: bool = True):
        """
        Context manager for database operations with automatic cleanup

        Args:
            commit: Whether to commit changes automatically

        Yields:
            Database cursor
        """
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            yield cursor

            if commit:
                conn.commit()
                logger.debug("Transaction committed")

        except Exception as e:
            if conn and not conn.closed:
                conn.rollback()
                logger.warning("Transaction rolled back due to error")
            logger.error(f"Database operation failed: {e}")
            raise
        finally:
            if cursor:
                cursor.close()
            if conn:
                self.return_connection(conn)

    @contextmanager
    def transaction(self):
        """
        Context manager for explicit transaction management

        Yields:
            Database cursor within a transaction
        """
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            yield cursor
            # Transaction will be committed by caller or rolled back on error
        except Exception as e:
            if conn and not conn.closed:
                conn.rollback()
                logger.warning("Transaction rolled back due to error")
            logger.error(f"Transaction failed: {e}")
            raise
        finally:
            if cursor:
                cursor.close()
            if conn:
                self.return_connection(conn)

# Global connection manager instance
_connection_manager: Optional[PostgresConnectionManager] = None

def get_pg_conn():
    """
    Get a PostgreSQL connection (convenience function)

    Returns:
        PostgreSQL connection object
    """
    global _connection_manager
    if _connection_manager is None:
        _connection_manager = PostgresConnectionManager()
    return _connection_manager.get_connection()

def initialize_postgres_connection(config: Optional[Dict[str, Any]] = None) -> PostgresConnectionManager:
    """
    Initialize the global PostgreSQL connection manager

    Args:
        config: Database configuration

    Returns:
        Connection manager instance
    """
    global _connection_manager
    _connection_manager = PostgresConnectionManager(config)
    return _connection_manager

def close_postgres_connection() -> None:
    """Close the global PostgreSQL connection manager"""
    global _connection_manager
    if _connection_manager:
        _connection_manager.close_all()
        _connection_manager = None

# Context manager for safe database operations
@contextmanager
def pg_cursor(commit: bool = True):
    """
    Context manager for PostgreSQL cursor operations

    Args:
        commit: Whether to commit changes automatically

    Yields:
        Database cursor
    """
    global _connection_manager
    if _connection_manager is None:
        _connection_manager = PostgresConnectionManager()

    with _connection_manager.get_cursor(commit=commit) as cursor:
        yield cursor

@contextmanager
def pg_transaction():
    """
    Context manager for PostgreSQL transactions

    Yields:
        Database cursor within a transaction
    """
    global _connection_manager
    if _connection_manager is None:
        _connection_manager = PostgresConnectionManager()

    with _connection_manager.transaction() as cursor:
        yield cursor

def test_connection() -> bool:
    """
    Test the PostgreSQL connection

    Returns:
        True if connection successful, False otherwise
    """
    try:
        with pg_cursor(commit=False) as cursor:
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            if result and result[0] == 1:
                logger.info("PostgreSQL connection test successful")
                return True
            else:
                logger.error("PostgreSQL connection test failed - unexpected result")
                return False
    except Exception as e:
        logger.error(f"PostgreSQL connection test failed: {e}")
        return False
