"""
Pipeline Diagnostics for Nifty Trading Agent

This script inspects:
- OHLCV data coverage
- Features coverage
- Backtest coverage (by strategy_id & run_id)
- Per-stage row counts & date ranges

Usage:
    cd nifty_trading_agent
    python -m diagnostics.pipeline_diagnostics
"""

import os
import duckdb
import pandas as pd

DB_PATH = "data/nifty_duckdb.db"  # adjust if needed


def connect_db(path: str) -> duckdb.DuckDBPyConnection:
    if not os.path.exists(path):
        raise FileNotFoundError(f"DuckDB file not found at: {path}")
    print(f"✅ Connected to DuckDB: {path}")
    return duckdb.connect(path)


def summarize_table(conn, table_name: str, date_col_candidates=("date", "trade_date", "dt")):
    print(f"\n🔍 TABLE: {table_name}")
    try:
        schema = conn.execute(f"DESCRIBE {table_name}").fetchdf()
    except Exception as e:
        print(f"  ⚠️ Could not DESCRIBE table: {e}")
        return

    print(schema)

    cols = {c.lower(): c for c in schema["column_name"].tolist()}
    date_col = None
    for cand in date_col_candidates:
        if cand.lower() in cols:
            date_col = cols[cand.lower()]
            break

    row_count = conn.execute(f"SELECT COUNT(*) AS n FROM {table_name}").fetchdf()["n"][0]
    print(f"  ➜ Total rows: {row_count}")

    if date_col:
        stats = conn.execute(
            f"""
            SELECT
                MIN({date_col}) AS min_date,
                MAX({date_col}) AS max_date,
                COUNT(DISTINCT {date_col}) AS unique_dates
            FROM {table_name}
            """
        ).fetchdf()
        print("  ➜ Date stats:")
        print(stats.to_string(index=False))
    else:
        print("  ⚠️ No obvious date column found; skipping date stats.")


def summarize_backtest_results(conn):
    table = "backtest_results"
    print("\n📊 BACKTEST RESULTS SUMMARY")

    # Basic stats
    base_stats = conn.execute(
        """
        SELECT
            MIN(date) AS min_date,
            MAX(date) AS max_date,
            COUNT(*) AS rows,
            COUNT(DISTINCT date) AS unique_dates,
            COUNT(DISTINCT strategy_id) AS strategies,
            COUNT(DISTINCT run_id) AS runs
        FROM backtest_results
        """
    ).fetchdf()
    print("\n🧱 Global backtest stats:")
    print(base_stats.to_string(index=False))

    # Per strategy
    strat_stats = conn.execute(
        """
        SELECT
            strategy_id,
            MIN(date) AS min_date,
            MAX(date) AS max_date,
            COUNT(*) AS rows,
            COUNT(DISTINCT date) AS unique_dates
        FROM backtest_results
        GROUP BY strategy_id
        ORDER BY strategy_id
        """
    ).fetchdf()
    print("\n📌 Per-strategy coverage:")
    print(strat_stats.to_string(index=False))

    # Per run for ml_model_signals
    ml_runs = conn.execute(
        """
        SELECT
            run_id,
            MIN(date) AS min_date,
            MAX(date) AS max_date,
            COUNT(*) AS rows,
            COUNT(DISTINCT date) AS unique_dates
        FROM backtest_results
        WHERE strategy_id = 'ml_model_signals'
        GROUP BY run_id
        ORDER BY min_date
        """
    ).fetchdf()
    print("\n🎯 ml_model_signals runs:")
    print(ml_runs.to_string(index=False))

    # Show one run with most dates
    if not ml_runs.empty:
        best_run = ml_runs.sort_values("unique_dates", ascending=False).iloc[0]
        print("\n🏆 Best (longest) ml_model_signals run_id candidate:")
        print(best_run.to_string())

        run_id = best_run["run_id"]
        sample = conn.execute(
            f"""
            SELECT date, equity, drawdown, exposure
            FROM backtest_results
            WHERE strategy_id = 'ml_model_signals'
              AND run_id = '{run_id}'
            ORDER BY date
            LIMIT 10
            """
        ).fetchdf()
        print(f"\n📈 Sample rows for run_id = {run_id}:")
        print(sample)


def main():
    print("=" * 80)
    print("NIFTY TRADING AGENT - PIPELINE DIAGNOSTICS")
    print("=" * 80)

    conn = connect_db(DB_PATH)

    # 1) Show available tables
    tables = conn.execute("SHOW TABLES").fetchdf()
    print("\n📚 Available Tables:")
    print(tables.to_string(index=False))

    # 2) Summaries for key tables
    for tbl in ["ohlcv_nifty", "features_nifty", "backtest_results"]:
        if tbl in tables["name"].values:
            summarize_table(conn, tbl)
        else:
            print(f"\n⚠️ Table '{tbl}' not found in DuckDB.")

    # 3) Detailed backtest results analysis
    if "backtest_results" in tables["name"].values:
        summarize_backtest_results(conn)

    print("\n✅ Diagnostics complete.\n")


if __name__ == "__main__":
    main()
