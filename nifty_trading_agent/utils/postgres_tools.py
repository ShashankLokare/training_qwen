"""
PostgreSQL Tools for Agent Operations
High-level functions for agents to interact with PostgreSQL database
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, date
import json

try:
    from ..utils.logging_utils import get_logger
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.logging_utils import get_logger
from .db_postgres import pg_cursor, pg_transaction

logger = get_logger(__name__)

def record_signals(signals: List[Dict[str, Any]]) -> List[int]:
    """
    Record trading signals in the database

    Args:
        signals: List of signal dictionaries with keys:
                symbol, signal_date, entry_low, entry_high, target_price,
                stop_loss, position_size_pct, conviction, notes, model_version

    Returns:
        List of signal IDs created
    """
    signal_ids = []

    for signal in signals:
        try:
            with pg_transaction() as cursor:
                cursor.execute("""
                    INSERT INTO daily_signals
                    (symbol, signal_date, entry_low, entry_high, target_price,
                     stop_loss, position_size_pct, conviction, notes, model_version)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    signal['symbol'],
                    signal.get('signal_date', datetime.now().date()),
                    signal.get('entry_low'),
                    signal.get('entry_high'),
                    signal.get('target_price'),
                    signal.get('stop_loss'),
                    signal.get('position_size_pct'),
                    signal.get('conviction'),
                    signal.get('notes', ''),
                    signal.get('model_version', 'default')
                ))

                signal_id = cursor.fetchone()[0]
                signal_ids.append(signal_id)
                logger.info(f"Recorded signal for {signal['symbol']} with ID {signal_id}")

        except Exception as e:
            logger.error(f"Failed to record signal for {signal['symbol']}: {e}")
            continue

    return signal_ids

def fetch_signals_for_date(signal_date: date) -> List[Dict[str, Any]]:
    """
    Fetch all signals for a specific date

    Args:
        signal_date: Date to fetch signals for

    Returns:
        List of signal dictionaries
    """
    try:
        with pg_cursor(commit=False) as cursor:
            cursor.execute("""
                SELECT id, symbol, signal_date, entry_low, entry_high, target_price,
                       stop_loss, position_size_pct, conviction, notes, model_version,
                       created_at
                FROM daily_signals
                WHERE signal_date = %s
                ORDER BY conviction DESC, created_at DESC
            """, (signal_date,))

            columns = [desc[0] for desc in cursor.description]
            signals = []

            for row in cursor.fetchall():
                signal_dict = dict(zip(columns, row))
                # Convert decimal types to float for JSON serialization
                for key in ['entry_low', 'entry_high', 'target_price', 'stop_loss', 'position_size_pct', 'conviction']:
                    if signal_dict[key] is not None:
                        signal_dict[key] = float(signal_dict[key])
                signals.append(signal_dict)

            logger.info(f"Fetched {len(signals)} signals for {signal_date}")
            return signals

    except Exception as e:
        logger.error(f"Failed to fetch signals for {signal_date}: {e}")
        return []

def create_order_from_signal(signal_id: int, side: str, quantity: float,
                           price: float, agent_name: str = "SignalAgent") -> Optional[int]:
    """
    Create an order from a signal

    Args:
        signal_id: ID of the signal to create order from
        side: 'BUY' or 'SELL'
        quantity: Order quantity
        price: Order price
        agent_name: Name of the agent creating the order

    Returns:
        Order ID if successful, None otherwise
    """
    try:
        with pg_transaction() as cursor:
            cursor.execute("""
                INSERT INTO orders (symbol, side, quantity, price, status, signal_id, agent_name)
                SELECT ds.symbol, %s, %s, %s, 'NEW', %s, %s
                FROM daily_signals ds
                WHERE ds.id = %s
                RETURNING id
            """, (side, quantity, price, signal_id, agent_name, signal_id))

            order_id = cursor.fetchone()
            if order_id:
                order_id = order_id[0]
                logger.info(f"Created order {order_id} for signal {signal_id}")
                return order_id
            else:
                logger.error(f"Failed to create order for signal {signal_id}")
                return None

    except Exception as e:
        logger.error(f"Failed to create order from signal {signal_id}: {e}")
        return None

def get_open_positions() -> List[Dict[str, Any]]:
    """
    Get all open positions

    Returns:
        List of open position dictionaries
    """
    try:
        with pg_cursor(commit=False) as cursor:
            cursor.execute("""
                SELECT id, symbol, quantity, avg_price, current_price,
                       unrealized_pnl, realized_pnl, opened_at, updated_at
                FROM positions
                WHERE is_open = TRUE
                ORDER BY opened_at DESC
            """)

            columns = [desc[0] for desc in cursor.description]
            positions = []

            for row in cursor.fetchall():
                position_dict = dict(zip(columns, row))
                # Convert decimal types to float
                for key in ['quantity', 'avg_price', 'current_price', 'unrealized_pnl', 'realized_pnl']:
                    if position_dict[key] is not None:
                        position_dict[key] = float(position_dict[key])
                positions.append(position_dict)

            logger.info(f"Fetched {len(positions)} open positions")
            return positions

    except Exception as e:
        logger.error(f"Failed to fetch open positions: {e}")
        return []

def log_agent_run(agent_name: str, run_type: str, status: str = "SUCCESS",
                 meta: Optional[Dict[str, Any]] = None) -> Optional[int]:
    """
    Log an agent run

    Args:
        agent_name: Name of the agent
        run_type: Type of run (e.g., 'daily_pipeline', 'backtest')
        status: Run status ('SUCCESS', 'FAILURE', 'PARTIAL')
        meta: Additional metadata as JSON

    Returns:
        Run ID if successful, None otherwise
    """
    try:
        meta_json = json.dumps(meta) if meta else None

        with pg_transaction() as cursor:
            cursor.execute("""
                INSERT INTO agent_runs (agent_name, run_type, status, meta_json)
                VALUES (%s, %s, %s, %s)
                RETURNING id
            """, (agent_name, run_type, status, meta_json))

            run_id = cursor.fetchone()[0]
            logger.info(f"Logged agent run: {agent_name} - {run_type} - {status} (ID: {run_id})")
            return run_id

    except Exception as e:
        logger.error(f"Failed to log agent run for {agent_name}: {e}")
        return None

def update_agent_run(run_id: int, status: str, meta: Optional[Dict[str, Any]] = None) -> bool:
    """
    Update an existing agent run

    Args:
        run_id: ID of the run to update
        status: New status
        meta: Updated metadata

    Returns:
        True if successful, False otherwise
    """
    try:
        meta_json = json.dumps(meta) if meta else None

        with pg_transaction() as cursor:
            cursor.execute("""
                UPDATE agent_runs
                SET finished_at = NOW(), status = %s, meta_json = %s
                WHERE id = %s
            """, (status, meta_json, run_id))

            logger.info(f"Updated agent run {run_id} with status {status}")
            return True

    except Exception as e:
        logger.error(f"Failed to update agent run {run_id}: {e}")
        return False

def get_recent_signals(days: int = 7) -> List[Dict[str, Any]]:
    """
    Get recent signals from the last N days

    Args:
        days: Number of days to look back

    Returns:
        List of recent signal dictionaries
    """
    try:
        with pg_cursor(commit=False) as cursor:
            cursor.execute("""
                SELECT id, symbol, signal_date, entry_low, entry_high, target_price,
                       stop_loss, position_size_pct, conviction, notes, model_version,
                       created_at
                FROM daily_signals
                WHERE signal_date >= CURRENT_DATE - INTERVAL '%s days'
                ORDER BY signal_date DESC, conviction DESC
            """, (days,))

            columns = [desc[0] for desc in cursor.description]
            signals = []

            for row in cursor.fetchall():
                signal_dict = dict(zip(columns, row))
                # Convert decimal types to float
                for key in ['entry_low', 'entry_high', 'target_price', 'stop_loss', 'position_size_pct', 'conviction']:
                    if signal_dict[key] is not None:
                        signal_dict[key] = float(signal_dict[key])
                signals.append(signal_dict)

            logger.info(f"Fetched {len(signals)} signals from last {days} days")
            return signals

    except Exception as e:
        logger.error(f"Failed to fetch recent signals: {e}")
        return []

def get_agent_run_history(agent_name: Optional[str] = None, days: int = 30) -> List[Dict[str, Any]]:
    """
    Get agent run history

    Args:
        agent_name: Specific agent name, or None for all agents
        days: Number of days to look back

    Returns:
        List of agent run dictionaries
    """
    try:
        with pg_cursor(commit=False) as cursor:
            if agent_name:
                cursor.execute("""
                    SELECT id, agent_name, run_type, started_at, finished_at, status, meta_json
                    FROM agent_runs
                    WHERE agent_name = %s AND started_at >= CURRENT_TIMESTAMP - INTERVAL '%s days'
                    ORDER BY started_at DESC
                """, (agent_name, days))
            else:
                cursor.execute("""
                    SELECT id, agent_name, run_type, started_at, finished_at, status, meta_json
                    FROM agent_runs
                    WHERE started_at >= CURRENT_TIMESTAMP - INTERVAL '%s days'
                    ORDER BY started_at DESC
                """, (days,))

            columns = [desc[0] for desc in cursor.description]
            runs = []

            for row in cursor.fetchall():
                run_dict = dict(zip(columns, row))
                # Parse meta_json if present
                if run_dict['meta_json']:
                    try:
                        run_dict['meta'] = json.loads(run_dict['meta_json'])
                    except:
                        run_dict['meta'] = {}
                    del run_dict['meta_json']
                else:
                    run_dict['meta'] = {}
                runs.append(run_dict)

            logger.info(f"Fetched {len(runs)} agent runs from last {days} days")
            return runs

    except Exception as e:
        logger.error(f"Failed to fetch agent run history: {e}")
        return []

def create_symbol(symbol: str, name: str = "", sector: str = "", is_active: bool = True) -> bool:
    """
    Create or update a symbol in the database

    Args:
        symbol: Stock symbol
        name: Company name
        sector: Sector classification
        is_active: Whether the symbol is actively traded

    Returns:
        True if successful, False otherwise
    """
    try:
        with pg_transaction() as cursor:
            cursor.execute("""
                INSERT INTO symbols (symbol, name, sector, is_active)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (symbol) DO UPDATE SET
                    name = EXCLUDED.name,
                    sector = EXCLUDED.sector,
                    is_active = EXCLUDED.is_active
            """, (symbol, name, sector, is_active))

            logger.info(f"Created/updated symbol: {symbol}")
            return True

    except Exception as e:
        logger.error(f"Failed to create symbol {symbol}: {e}")
        return False

def get_symbols(active_only: bool = True) -> List[Dict[str, Any]]:
    """
    Get all symbols from the database

    Args:
        active_only: Whether to return only active symbols

    Returns:
        List of symbol dictionaries
    """
    try:
        with pg_cursor(commit=False) as cursor:
            if active_only:
                cursor.execute("""
                    SELECT symbol, name, sector, is_active, created_at
                    FROM symbols
                    WHERE is_active = TRUE
                    ORDER BY symbol
                """)
            else:
                cursor.execute("""
                    SELECT symbol, name, sector, is_active, created_at
                    FROM symbols
                    ORDER BY symbol
                """)

            columns = [desc[0] for desc in cursor.description]
            symbols = [dict(zip(columns, row)) for row in cursor.fetchall()]

            logger.info(f"Fetched {len(symbols)} symbols")
            return symbols

    except Exception as e:
        logger.error(f"Failed to fetch symbols: {e}")
        return []

def update_positions_from_trades(trades: List[Dict[str, Any]]) -> bool:
    """
    Update positions based on executed trades

    Args:
        trades: List of trade dictionaries

    Returns:
        True if successful, False otherwise
    """
    try:
        with pg_transaction() as cursor:
            for trade in trades:
                cursor.execute("""
                    -- Update existing position or create new one
                    INSERT INTO positions (symbol, quantity, avg_price, current_price,
                                         unrealized_pnl, realized_pnl, opened_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (symbol) DO UPDATE SET
                        quantity = positions.quantity + EXCLUDED.quantity,
                        avg_price = CASE
                            WHEN positions.quantity + EXCLUDED.quantity != 0
                            THEN ((positions.quantity * positions.avg_price) +
                                  (EXCLUDED.quantity * EXCLUDED.avg_price)) /
                                 (positions.quantity + EXCLUDED.quantity)
                            ELSE EXCLUDED.avg_price
                        END,
                        current_price = EXCLUDED.current_price,
                        unrealized_pnl = EXCLUDED.unrealized_pnl,
                        realized_pnl = positions.realized_pnl + COALESCE(EXCLUDED.realized_pnl, 0),
                        updated_at = NOW(),
                        is_open = CASE
                            WHEN positions.quantity + EXCLUDED.quantity = 0 THEN FALSE
                            ELSE TRUE
                        END
                """, (
                    trade['symbol'],
                    trade.get('quantity', 0),
                    trade.get('avg_price', 0),
                    trade.get('current_price', trade.get('avg_price', 0)),
                    trade.get('unrealized_pnl', 0),
                    trade.get('realized_pnl', 0),
                    trade.get('opened_at', datetime.now())
                ))

            logger.info(f"Updated positions from {len(trades)} trades")
            return True

    except Exception as e:
        logger.error(f"Failed to update positions from trades: {e}")
        return False
