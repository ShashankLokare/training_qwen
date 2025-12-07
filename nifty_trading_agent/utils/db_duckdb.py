"""
DuckDB Connection Helper
Provides connection management for DuckDB analytical database
"""

import duckdb
from contextlib import contextmanager
from typing import Optional, Any, Dict
import os
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.logging_utils import get_logger
from utils.io_utils import load_yaml_config

logger = get_logger(__name__)

class DuckDBConnectionManager:
    """
    Manages DuckDB connections for analytical operations
    """

    def __init__(self, db_path: str = "data/nifty_analytics.duckdb",
                 read_only: bool = False, memory_limit: str = "2GB"):
        """
        Initialize the DuckDB connection manager

        Args:
            db_path: Path to DuckDB file
            read_only: Whether to open in read-only mode
            memory_limit: Memory limit for DuckDB operations
        """
        self.db_path = db_path
        self.read_only = read_only
        self.memory_limit = memory_limit
        self._connection = None

        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        logger.info(f"DuckDB manager initialized for {db_path}")

    def get_connection(self) -> duckdb.DuckDBPyConnection:
        """
        Get a DuckDB connection

        Returns:
            DuckDB connection object
        """
        if self._connection is None:
            try:
                self._connection = duckdb.connect(
                    database=self.db_path,
                    read_only=self.read_only
                )

                # Set memory limit and other configurations
                self._connection.execute(f"SET memory_limit = '{self.memory_limit}';")
                self._connection.execute("SET threads = 4;")  # Use 4 threads for better performance

                logger.info(f"DuckDB connection established to {self.db_path}")
            except Exception as e:
                logger.error(f"Failed to connect to DuckDB: {e}")
                raise

        return self._connection

    def close_connection(self) -> None:
        """Close the DuckDB connection"""
        if self._connection is not None:
            try:
                self._connection.close()
                self._connection = None
                logger.info("DuckDB connection closed")
            except Exception as e:
                logger.warning(f"Error closing DuckDB connection: {e}")

    @contextmanager
    def get_cursor(self):
        """
        Context manager for DuckDB operations

        Yields:
            DuckDB connection (cursor-like interface)
        """
        conn = None
        try:
            conn = self.get_connection()
            yield conn
        except Exception as e:
            logger.error(f"DuckDB operation failed: {e}")
            raise
        finally:
            # DuckDB connections are lightweight, we don't close them here
            # They remain open for the session
            pass

# Global connection manager instance
_connection_manager: Optional[DuckDBConnectionManager] = None

def get_duck_conn() -> duckdb.DuckDBPyConnection:
    """
    Get a DuckDB connection (convenience function)

    Returns:
        DuckDB connection object
    """
    global _connection_manager
    if _connection_manager is None:
        _connection_manager = DuckDBConnectionManager()
    return _connection_manager.get_connection()

def initialize_duckdb_connection(config: Optional[Dict[str, Any]] = None) -> DuckDBConnectionManager:
    """
    Initialize the global DuckDB connection manager

    Args:
        config: Database configuration

    Returns:
        Connection manager instance
    """
    global _connection_manager
    if config is None:
        full_config = load_yaml_config("config/config.yaml")
        config = full_config.get('databases', {}).get('duckdb', {})

    db_path = config.get('path', 'data/nifty_analytics.duckdb')
    read_only = config.get('read_only', False)
    memory_limit = config.get('memory_limit', '2GB')

    _connection_manager = DuckDBConnectionManager(
        db_path=db_path,
        read_only=read_only,
        memory_limit=memory_limit
    )
    return _connection_manager

def close_duckdb_connection() -> None:
    """Close the global DuckDB connection manager"""
    global _connection_manager
    if _connection_manager:
        _connection_manager.close_connection()
        _connection_manager = None

@contextmanager
def duck_cursor():
    """
    Context manager for DuckDB operations

    Yields:
        DuckDB connection
    """
    global _connection_manager
    if _connection_manager is None:
        _connection_manager = DuckDBConnectionManager()

    with _connection_manager.get_cursor() as conn:
        yield conn

def test_connection() -> bool:
    """
    Test the DuckDB connection

    Returns:
        True if connection successful, False otherwise
    """
    try:
        with duck_cursor() as conn:
            result = conn.execute("SELECT 42 as test_value").fetchone()
            if result and result[0] == 42:
                logger.info("DuckDB connection test successful")
                return True
            else:
                logger.error("DuckDB connection test failed - unexpected result")
                return False
    except Exception as e:
        logger.error(f"DuckDB connection test failed: {e}")
        return False

def execute_query(query: str, params: Optional[tuple] = None) -> list:
    """
    Execute a query and return results

    Args:
        query: SQL query string
        params: Query parameters

    Returns:
        Query results as list of tuples
    """
    try:
        with duck_cursor() as conn:
            if params:
                result = conn.execute(query, params).fetchall()
            else:
                result = conn.execute(query).fetchall()
            return result
    except Exception as e:
        logger.error(f"Query execution failed: {e}")
        return []

def execute_ddl(ddl_statement: str) -> bool:
    """
    Execute a DDL statement (CREATE, DROP, ALTER, etc.)

    Args:
        ddl_statement: DDL SQL statement

    Returns:
        True if successful, False otherwise
    """
    try:
        with duck_cursor() as conn:
            conn.execute(ddl_statement)
            logger.debug(f"DDL executed: {ddl_statement[:50]}...")
            return True
    except Exception as e:
        logger.error(f"DDL execution failed: {e}")
        return False
