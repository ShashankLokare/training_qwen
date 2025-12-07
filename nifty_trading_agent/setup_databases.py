#!/usr/bin/env python3
"""
Database Setup Script for Nifty Trading Agent
Creates PostgreSQL and DuckDB schemas and initial data
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.logging_utils import setup_logging
try:
    from utils.db_postgres import pg_cursor, test_connection as test_pg_connection
    from utils.postgres_tools import create_symbol
    POSTGRES_AVAILABLE = True
except ImportError as e:
    print("PostgreSQL not available, running in DuckDB-only mode")
    POSTGRES_AVAILABLE = False

from utils.db_duckdb import execute_ddl, test_connection as test_duck_connection
from utils.duckdb_tools import store_ohlcv
import pandas as pd

def setup_postgres_schema():
    """Create PostgreSQL tables"""
    print("Setting up PostgreSQL schema...")

    # Create symbols table
    with pg_cursor(commit=True) as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS symbols (
                symbol          TEXT PRIMARY KEY,
                name            TEXT,
                sector          TEXT,
                is_active       BOOLEAN DEFAULT TRUE,
                created_at      TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        print("✓ Created symbols table")

    # Create daily_signals table
    with pg_cursor(commit=True) as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_signals (
                id                  BIGSERIAL PRIMARY KEY,
                symbol              TEXT REFERENCES symbols(symbol),
                signal_date         DATE NOT NULL,
                entry_low           NUMERIC(18,4),
                entry_high          NUMERIC(18,4),
                target_price        NUMERIC(18,4),
                stop_loss           NUMERIC(18,4),
                position_size_pct   NUMERIC(8,4),
                conviction          NUMERIC(4,3),
                notes               TEXT,
                model_version       TEXT,
                created_at          TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_daily_signals_date ON daily_signals(signal_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_daily_signals_symbol_date ON daily_signals(symbol, signal_date)")
        print("✓ Created daily_signals table")

    # Create orders table
    with pg_cursor(commit=True) as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                id              BIGSERIAL PRIMARY KEY,
                symbol          TEXT REFERENCES symbols(symbol),
                side            TEXT CHECK (side IN ('BUY', 'SELL')),
                quantity        NUMERIC(18,4),
                price           NUMERIC(18,4),
                status          TEXT DEFAULT 'NEW',
                signal_id       BIGINT REFERENCES daily_signals(id),
                created_at      TIMESTAMPTZ DEFAULT NOW(),
                updated_at      TIMESTAMPTZ DEFAULT NOW(),
                agent_name      TEXT
            )
        """)
        print("✓ Created orders table")

    # Create trades table
    with pg_cursor(commit=True) as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id              BIGSERIAL PRIMARY KEY,
                order_id        BIGINT REFERENCES orders(id),
                symbol          TEXT REFERENCES symbols(symbol),
                quantity        NUMERIC(18,4),
                avg_price       NUMERIC(18,4),
                pnl             NUMERIC(18,4),
                opened_at       TIMESTAMPTZ,
                closed_at       TIMESTAMPTZ,
                holding_days    INTEGER
            )
        """)
        print("✓ Created trades table")

    # Create positions table
    with pg_cursor(commit=True) as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id              BIGSERIAL PRIMARY KEY,
                symbol          TEXT REFERENCES symbols(symbol),
                quantity        NUMERIC(18,4),
                avg_price       NUMERIC(18,4),
                current_price   NUMERIC(18,4),
                unrealized_pnl  NUMERIC(18,4),
                realized_pnl    NUMERIC(18,4),
                opened_at       TIMESTAMPTZ,
                updated_at      TIMESTAMPTZ DEFAULT NOW(),
                is_open         BOOLEAN DEFAULT TRUE
            )
        """)
        print("✓ Created positions table")

    # Create agent_runs table
    with pg_cursor(commit=True) as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_runs (
                id              BIGSERIAL PRIMARY KEY,
                agent_name      TEXT,
                run_type        TEXT,
                started_at      TIMESTAMPTZ DEFAULT NOW(),
                finished_at     TIMESTAMPTZ,
                status          TEXT,
                meta_json       JSONB
            )
        """)
        print("✓ Created agent_runs table")

    # Create config_overrides table
    with pg_cursor(commit=True) as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS config_overrides (
                id              BIGSERIAL PRIMARY KEY,
                key             TEXT,
                value_json      JSONB,
                effective_from  TIMESTAMPTZ,
                created_at      TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        print("✓ Created config_overrides table")

    print("PostgreSQL schema setup complete!")

def setup_duckdb_schema():
    """Create DuckDB tables"""
    print("Setting up DuckDB schema...")

    # Create OHLCV table
    ddl = """
        CREATE TABLE IF NOT EXISTS ohlcv_nifty (
            symbol    TEXT,
            date      DATE,
            open      DOUBLE,
            high      DOUBLE,
            low       DOUBLE,
            close     DOUBLE,
            volume    DOUBLE,
            PRIMARY KEY (symbol, date)
        )
    """
    if execute_ddl(ddl):
        print("✓ Created ohlcv_nifty table")

    # Create features table
    ddl = """
        CREATE TABLE IF NOT EXISTS features_nifty (
            symbol                 TEXT,
            date                   DATE,
            r_1d                   DOUBLE,
            r_3d                   DOUBLE,
            r_5d                   DOUBLE,
            r_10d                  DOUBLE,
            r_20d                  DOUBLE,
            ma_5                   DOUBLE,
            ma_10                  DOUBLE,
            ma_20                  DOUBLE,
            ma_50                  DOUBLE,
            ma_100                 DOUBLE,
            ma_200                 DOUBLE,
            rsi_14                 DOUBLE,
            macd                   DOUBLE,
            bb_width               DOUBLE,
            atr_14                 DOUBLE,
            vol_zscore_20          DOUBLE,
            quarterly_eps_growth   DOUBLE,
            quarterly_rev_growth   DOUBLE,
            profit_margin          DOUBLE,
            pe_ratio               DOUBLE,
            pb_ratio               DOUBLE,
            sentiment_short        DOUBLE,
            sentiment_medium       DOUBLE,
            recent_earnings_flag   BOOLEAN,
            volume_gap_flag        BOOLEAN,
            forward_5d_ret_pct     DOUBLE,
            forward_10d_ret_pct    DOUBLE,
            forward_10d_max_return_pct DOUBLE,
            label_up_10pct         BOOLEAN,
            label_win_before_loss  BOOLEAN,
            label_loss_before_win  BOOLEAN,
            PRIMARY KEY (symbol, date)
        )
    """
    if execute_ddl(ddl):
        print("✓ Created features_nifty table")

    # Create backtest_results table
    ddl = """
        CREATE TABLE IF NOT EXISTS backtest_results (
            strategy_id      TEXT,
            run_id           TEXT,
            date             DATE,
            equity           DOUBLE,
            drawdown         DOUBLE,
            exposure         DOUBLE
        )
    """
    if execute_ddl(ddl):
        print("✓ Created backtest_results table")

        # Create index
        index_ddl = """
            CREATE INDEX IF NOT EXISTS idx_backtest_strategy_date
            ON backtest_results(strategy_id, date)
        """
        execute_ddl(index_ddl)
        print("✓ Created backtest_results index")

    print("DuckDB schema setup complete!")

def populate_initial_data():
    """Populate initial symbol data"""
    print("Populating initial data...")

    # Define initial symbols
    initial_symbols = [
        {"symbol": "RELIANCE.NS", "name": "Reliance Industries Limited", "sector": "Energy"},
        {"symbol": "TCS.NS", "name": "Tata Consultancy Services Limited", "sector": "IT Services"},
        {"symbol": "HDFCBANK.NS", "name": "HDFC Bank Limited", "sector": "Banking"},
        {"symbol": "ICICIBANK.NS", "name": "ICICI Bank Limited", "sector": "Banking"},
        {"symbol": "INFY.NS", "name": "Infosys Limited", "sector": "IT Services"},
        {"symbol": "HINDUNILVR.NS", "name": "Hindustan Unilever Limited", "sector": "FMCG"},
        {"symbol": "ITC.NS", "name": "ITC Limited", "sector": "FMCG"},
        {"symbol": "KOTAKBANK.NS", "name": "Kotak Mahindra Bank Limited", "sector": "Banking"},
        {"symbol": "LT.NS", "name": "Larsen & Toubro Limited", "sector": "Construction"},
        {"symbol": "AXISBANK.NS", "name": "Axis Bank Limited", "sector": "Banking"},
        {"symbol": "BAJFINANCE.NS", "name": "Bajaj Finance Limited", "sector": "Financial Services"},
        {"symbol": "MARUTI.NS", "name": "Maruti Suzuki India Limited", "sector": "Automotive"},
        {"symbol": "BAJAJ-AUTO.NS", "name": "Bajaj Auto Limited", "sector": "Automotive"},
        {"symbol": "WIPRO.NS", "name": "Wipro Limited", "sector": "IT Services"},
        {"symbol": "HCLTECH.NS", "name": "HCL Technologies Limited", "sector": "IT Services"},
        {"symbol": "ADANIPORTS.NS", "name": "Adani Ports and Special Economic Zone Limited", "sector": "Infrastructure"},
        {"symbol": "DIVISLAB.NS", "name": "Divi's Laboratories Limited", "sector": "Pharmaceuticals"},
        {"symbol": "SUNPHARMA.NS", "name": "Sun Pharmaceutical Industries Limited", "sector": "Pharmaceuticals"},
        {"symbol": "DRREDDY.NS", "name": "Dr. Reddy's Laboratories Limited", "sector": "Pharmaceuticals"},
        {"symbol": "CIPLA.NS", "name": "Cipla Limited", "sector": "Pharmaceuticals"}
    ]

    # Insert symbols into PostgreSQL
    for symbol_data in initial_symbols:
        success = create_symbol(
            symbol=symbol_data["symbol"],
            name=symbol_data["name"],
            sector=symbol_data["sector"]
        )
        if success:
            print(f"✓ Added symbol: {symbol_data['symbol']}")
        else:
            print(f"✗ Failed to add symbol: {symbol_data['symbol']}")

    print("Initial data population complete!")

def main():
    """Main setup function"""
    print("🚀 NIFTY TRADING AGENT - DATABASE SETUP")
    print("=" * 50)

    # Setup logging
    setup_logging({
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file_path': 'logs/db_setup.log',
        'max_file_size_mb': 10,
        'backup_count': 2
    })

    try:
        # Test connections
        print("Testing database connections...")

        if POSTGRES_AVAILABLE and not test_pg_connection():
            print("❌ PostgreSQL connection failed!")
            print("Please ensure PostgreSQL is running and credentials are correct in config.yaml")
            return 1

        if POSTGRES_AVAILABLE:
            print("✅ PostgreSQL connection successful")

        if not test_duck_connection():
            print("❌ DuckDB connection failed!")
            return 1

        print("✅ DuckDB connection successful")

        # Setup schemas
        if POSTGRES_AVAILABLE:
            setup_postgres_schema()
        else:
            print("⚠️ PostgreSQL not available, skipping PostgreSQL schema setup")

        setup_duckdb_schema()

        # Populate initial data (skip if PostgreSQL not available)
        if POSTGRES_AVAILABLE:
            populate_initial_data()
        else:
            print("⚠️ PostgreSQL not available, skipping initial data population")

        print("\n🎉 Database setup completed successfully!")
        print("\nNext steps:")
        print("1. Run the interactive setup: python interactive_main.py")
        print("2. Or run the daily pipeline: python main_daily_run.py")
        print("3. Check logs in logs/ directory for any issues")

        return 0

    except Exception as e:
        print(f"\n❌ Database setup failed: {e}")
        print("Check the logs in logs/db_setup.log for details")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
