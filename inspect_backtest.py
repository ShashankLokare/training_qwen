import duckdb
import pandas as pd

# Path to the DuckDB file
DB_PATH = "nifty_trading_agent/data/nifty_analytics.duckdb"

def inspect_backtest_results():
    try:
        conn = duckdb.connect(DB_PATH)

        # First query
        print("=== BACKTEST_RESULTS SUMMARY ===")
        summary = conn.execute("""
            SELECT
              MIN(date) AS min_date,
              MAX(date) AS max_date,
              COUNT(*) AS rows,
              COUNT(DISTINCT date) AS unique_dates,
              COUNT(DISTINCT strategy_id) AS strategies,
              MIN(equity) AS min_equity,
              MAX(equity) AS max_equity
            FROM backtest_results
        """).fetchdf()
        print(summary)

        # Second query
        print("\n=== STRATEGY_ID AND RUN_ID BREAKDOWN ===")
        breakdown = conn.execute("""
            SELECT strategy_id, run_id, COUNT(*) AS rows
            FROM backtest_results
            GROUP BY strategy_id, run_id
            ORDER BY strategy_id, run_id
        """).fetchdf()
        print(breakdown)

        # Additional check for trades if table exists
        try:
            print("\n=== TRADES TABLE CHECK ===")
            trades_count = conn.execute("SELECT COUNT(*) FROM trades").fetchdf()
            print(f"Total trades: {trades_count.iloc[0,0]}")

            if trades_count.iloc[0,0] > 0:
                trades_by_strategy = conn.execute("""
                    SELECT strategy_id, COUNT(*) FROM trades GROUP BY strategy_id
                """).fetchdf()
                print(trades_by_strategy)
        except Exception as e:
            print(f"Trades table not found or error: {e}")

        # Check ohlcv_nifty data
        try:
            print("\n=== OHLCV_NIFTY DATA CHECK ===")
            ohlcv_count = conn.execute("SELECT COUNT(*) FROM ohlcv_nifty").fetchdf()
            print(f"Total OHLCV records: {ohlcv_count.iloc[0,0]}")

            if ohlcv_count.iloc[0,0] > 0:
                symbols = conn.execute("SELECT DISTINCT symbol FROM ohlcv_nifty").fetchdf()
                print(f"Symbols in ohlcv_nifty: {symbols['symbol'].tolist()}")

                date_range = conn.execute("SELECT MIN(date) as min_date, MAX(date) as max_date FROM ohlcv_nifty").fetchdf()
                print(f"Date range: {date_range.iloc[0,0]} to {date_range.iloc[0,1]}")
        except Exception as e:
            print(f"Error checking ohlcv_nifty: {e}")

        conn.close()

    except Exception as e:
        print(f"Error inspecting database: {e}")

if __name__ == "__main__":
    inspect_backtest_results()
