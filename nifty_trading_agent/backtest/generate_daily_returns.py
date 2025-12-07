"""
Generate daily returns directly from DuckDB backtest tables.

Creates a clean daily returns file for Monte Carlo testing.
"""

import duckdb
import pandas as pd
import os

DB_PATH = "data/nifty_duckdb.db"     # update if your path is different
OUTPUT_PATH = "reports/daily_returns.csv"

def main():
    print("Connecting to DuckDB...")
    conn = duckdb.connect(DB_PATH)

    # We expect your backtest to have stored either:
    # - daily equity curve
    # - or daily PnL
    # If your table name differs, replace below.
    QUERY = """
        SELECT
            date,
            daily_return
        FROM backtest_results
        ORDER BY date;
    """

    try:
        df = conn.execute(QUERY).fetchdf()
    except Exception as e:
        raise RuntimeError(
            "Could not find 'daily_return' column in DuckDB. "
            "Please share your backtest table schema so I can adjust the query."
        ) from e

    if df.empty:
        raise ValueError("No daily returns found in DuckDB.")

    # Convert to percent if values are decimals
    if df["daily_return"].abs().max() < 1:
        df["strategy_return"] = df["daily_return"] * 100.0
    else:
        df.rename(columns={"daily_return": "strategy_return"}, inplace=True)

    print(f"Saving daily returns to {OUTPUT_PATH}")
    os.makedirs("reports", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("Done ✔")

if __name__ == "__main__":
    main()
