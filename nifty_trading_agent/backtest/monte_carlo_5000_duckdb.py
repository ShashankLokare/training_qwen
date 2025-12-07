"""
Monte Carlo Stress Test - 5000 Scenarios (DuckDB version)

- Connects to DuckDB
- Tries to infer a backtest/equity table
- Derives daily returns
- Runs 5000 bootstrapped scenarios
- Prints risk & performance metrics

Usage:
    cd nifty_trading_agent
    python -m backtest.monte_carlo_5000_duckdb
"""

import os
import duckdb
import numpy as np
import pandas as pd

# =============================================================================
# CONFIG
# =============================================================================

# 🔧 Update this to your actual DuckDB file path if needed
DB_PATH = "data/nifty_analytics.duckdb"

# If you know the exact table name that contains equity / daily PnL, set it here.
# Otherwise keep it as None and let the script try to guess from available tables.
TABLE_NAME_OVERRIDE = None  # e.g. "backtest_equity_v2" or "equity_curve"

# If your equity column has a particular name, set one of these as needed:
EQUITY_CANDIDATE_COLS = ["equity", "equity_curve", "portfolio_value", "nav"]

# If your table has a direct daily return column, we'll prefer that:
RETURN_CANDIDATE_COLS = ["daily_return", "strategy_return", "pnl_pct"]

# Monte Carlo configuration
N_SIMULATIONS = 5000           # number of Monte Carlo scenarios
CONFIDENCE_LEVEL = 0.95        # for VaR and ES
HORIZON_DAYS = None            # if None, use length of historical series
RISK_FREE_RATE = 0.0           # assume 0 for Sharpe

# =============================================================================
# HELPERS
# =============================================================================

def connect_db(path: str) -> duckdb.DuckDBPyConnection:
    if not os.path.exists(path):
        raise FileNotFoundError(f"DuckDB file not found at: {path}")
    return duckdb.connect(path)


def list_tables(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = conn.execute("SHOW TABLES").fetchdf()
    return df


def pick_backtest_table(conn: duckdb.DuckDBPyConnection) -> str:
    if TABLE_NAME_OVERRIDE:
        print(f"Using TABLE_NAME_OVERRIDE = {TABLE_NAME_OVERRIDE}")
        return TABLE_NAME_OVERRIDE

    tables = list_tables(conn)
    table_names = [t[0] for t in tables.values]

    print("\nAvailable tables in DuckDB:")
    for name in table_names:
        print(f"  - {name}")

    # Heuristic: pick table that looks like a backtest/equity table
    candidates = [
        name for name in table_names
        if any(kw in name.lower() for kw in ["backtest", "equity", "portfolio"])
    ]

    if not candidates:
        raise RuntimeError(
            "Could not auto-detect a backtest/equity table.\n"
            "Please set TABLE_NAME_OVERRIDE in this script to the correct table name."
        )

    # Pick the first one for now, but print list so user can refine if needed
    print("\nHeuristic candidate tables for backtest/equity:")
    for c in candidates:
        print(f"  - {c}")

    chosen = candidates[0]
    print(f"\nUsing inferred backtest table: {chosen}")
    return chosen


def get_table_schema(conn: duckdb.DuckDBPyConnection, table_name: str) -> pd.DataFrame:
    df = conn.execute(f"DESCRIBE {table_name}").fetchdf()
    return df


def find_column(columns: pd.DataFrame, candidates: list[str]) -> str | None:
    existing = {col.lower(): col for col in columns["column_name"].tolist()}
    for cand in candidates:
        if cand.lower() in existing:
            return existing[cand.lower()]
    return None


def load_daily_returns_from_duckdb(conn: duckdb.DuckDBPyConnection) -> pd.Series:
    table = pick_backtest_table(conn)
    schema = get_table_schema(conn, table)

    print(f"\nSchema for table '{table}':")
    print(schema)

    # Try to find a daily return column first
    ret_col = find_column(schema, RETURN_CANDIDATE_COLS)
    date_col = find_column(schema, ["date", "trade_date", "dt"])

    if date_col is None:
        raise RuntimeError(
            f"Could not find a date column in table '{table}'. "
            f"Expected one of ['date', 'trade_date', 'dt']."
        )

    if ret_col is not None:
        print(f"\nUsing '{ret_col}' as daily return column.")
        df = conn.execute(
            f"SELECT {date_col} AS date, {ret_col} AS ret FROM {table} ORDER BY {date_col}"
        ).fetchdf()
        # Decide if returns are in percent or decimal
        if df["ret"].abs().max() > 1.0:
            # assume percentage
            daily_returns = df["ret"].astype(float) / 100.0
        else:
            daily_returns = df["ret"].astype(float)
        return daily_returns

    # Otherwise, try to derive returns from equity
    equity_col = find_column(schema, EQUITY_CANDIDATE_COLS)
    if equity_col is None:
        raise RuntimeError(
            f"Could not find a daily return column or an equity column in table '{table}'.\n"
            f"Tried return candidates: {RETURN_CANDIDATE_COLS}\n"
            f"Tried equity candidates: {EQUITY_CANDIDATE_COLS}\n"
            f"Please update this script with the correct column names for your schema."
        )

    print(f"\nUsing '{equity_col}' as equity column to derive daily returns.")
    df = conn.execute(
        f"SELECT {date_col} AS date, {equity_col} AS equity FROM {table} ORDER BY {date_col}"
    ).fetchdf()

    df["equity"] = df["equity"].astype(float)
    df["ret"] = df["equity"].pct_change()
    df = df.dropna(subset=["ret"])

    return df["ret"]


def max_drawdown(equity_curve: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - peak) / peak
    return float(drawdowns.min())


def simulate_paths(returns: np.ndarray, n_sims: int, horizon_days: int) -> np.ndarray:
    n_hist = len(returns)
    if n_hist == 0:
        raise ValueError("Historical returns array is empty.")
    indices = np.random.randint(0, n_hist, size=(n_sims, horizon_days))
    sim_returns = returns[indices]
    return sim_returns


def compute_pnl_stats(sim_returns: np.ndarray) -> dict:
    equity_curves = (1.0 + sim_returns).cumprod(axis=1)
    final_equity = equity_curves[:, -1]
    final_pnl = final_equity - 1.0

    mean_pnl = np.mean(final_pnl)
    std_pnl = np.std(final_pnl)
    median_pnl = np.median(final_pnl)
    best_pnl = np.max(final_pnl)
    worst_pnl = np.min(final_pnl)

    daily_mean = np.mean(sim_returns, axis=1)
    daily_std = np.std(sim_returns, axis=1)
    downside_returns = np.clip(sim_returns, None, 0.0)
    daily_downside_std = np.sqrt(np.mean(downside_returns**2, axis=1))

    ann_factor = np.sqrt(252.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe_paths = (daily_mean - RISK_FREE_RATE / 252.0) / daily_std * ann_factor
        sharpe_paths = np.nan_to_num(sharpe_paths, nan=0.0, posinf=0.0, neginf=0.0)

        sortino_paths = (daily_mean - RISK_FREE_RATE / 252.0) / daily_downside_std * ann_factor
        sortino_paths = np.nan_to_num(sortino_paths, nan=0.0, posinf=0.0, neginf=0.0)

    sharpe = float(np.mean(sharpe_paths))
    sortino = float(np.mean(sortino_paths))

    mdds = np.array([max_drawdown(ec) for ec in equity_curves])
    avg_mdd = float(np.mean(mdds))

    cl_95 = CONFIDENCE_LEVEL
    cl_99 = 0.99

    var_95 = np.percentile(final_pnl, (1 - cl_95) * 100.0)
    var_99 = np.percentile(final_pnl, (1 - cl_99) * 100.0)

    es_95 = final_pnl[final_pnl <= var_95].mean() if np.any(final_pnl <= var_95) else 0.0
    es_99 = final_pnl[final_pnl <= var_99].mean() if np.any(final_pnl <= var_99) else 0.0

    lower_ci = np.percentile(final_pnl, (1 - cl_95) * 100.0)
    upper_ci = np.percentile(final_pnl, cl_95 * 100.0)

    prob_profit = float(np.mean(final_pnl > 0.0))
    prob_loss_10 = float(np.mean(final_pnl <= -0.10))
    prob_loss_20 = float(np.mean(final_pnl <= -0.20))

    return {
        "mean_pnl": mean_pnl,
        "std_pnl": std_pnl,
        "median_pnl": median_pnl,
        "best_pnl": best_pnl,
        "worst_pnl": worst_pnl,
        "sharpe": sharpe,
        "sortino": sortino,
        "avg_max_drawdown": avg_mdd,
        "var_95": var_95,
        "var_99": var_99,
        "es_95": es_95,
        "es_99": es_99,
        "pnL_ci_lower": lower_ci,
        "pnL_ci_upper": upper_ci,
        "prob_profit": prob_profit,
        "prob_loss_10": prob_loss_10,
        "prob_loss_20": prob_loss_20,
    }


def format_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def main():
    print("=" * 80)
    print("MONTE CARLO STRESS TEST REPORT (5000 SCENARIOS, DuckDB)")
    print("=" * 80)

    conn = connect_db(DB_PATH)
    daily_returns = load_daily_returns_from_duckdb(conn)
    returns_array = daily_returns.values

    horizon = HORIZON_DAYS or len(returns_array)

    print("\nINPUT DATA")
    print("-----------")
    print(f"Historical daily points: {len(returns_array)}")
    print(f"Simulation horizon     : {horizon} days")
    print(f"Number of simulations  : {N_SIMULATIONS}")
    print(f"Confidence level       : {CONFIDENCE_LEVEL * 100:.1f}%")

    sim_returns = simulate_paths(returns_array, N_SIMULATIONS, horizon)
    stats = compute_pnl_stats(sim_returns)

    print("\nPERFORMANCE STATISTICS")
    print("-----------------------")
    print(f"Mean PnL        : {format_pct(stats['mean_pnl'])}")
    print(f"Std Dev (PnL)   : {format_pct(stats['std_pnl'])}")
    print(f"Median PnL      : {format_pct(stats['median_pnl'])}")
    print(f"Best Case       : {format_pct(stats['best_pnl'])}")
    print(f"Worst Case      : {format_pct(stats['worst_pnl'])}")

    print("\nRISK METRICS")
    print("-------------")
    print(f"Sharpe Ratio    : {stats['sharpe']:.2f}")
    print(f"Sortino Ratio   : {stats['sortino']:.2f}")
    print(f"Avg Max Drawdown: {format_pct(stats['avg_max_drawdown'])}")

    print("\nVALUE AT RISK (VaR) & EXPECTED SHORTFALL")
    print("-----------------------------------------")
    print(f"VaR  {CONFIDENCE_LEVEL*100:.0f}% : {format_pct(stats['var_95'])}")
    print(f"VaR  99%          : {format_pct(stats['var_99'])}")
    print(f"ES   {CONFIDENCE_LEVEL*100:.0f}% : {format_pct(stats['es_95'])}")
    print(f"ES   99%          : {format_pct(stats['es_99'])}")

    print("\nCONFIDENCE INTERVALS (FINAL PnL)")
    print("--------------------------------")
    print(f"PnL {CONFIDENCE_LEVEL*100:.0f}% CI: [{format_pct(stats['pnL_ci_lower'])}, "
          f"{format_pct(stats['pnL_ci_upper'])}]")

    print("\nPROBABILITY ANALYSIS")
    print("---------------------")
    print(f"Probability of Profit        : {stats['prob_profit']*100:.1f}%")
    print(f"Probability of >10% Loss     : {stats['prob_loss_10']*100:.1f}%")
    print(f"Probability of >20% Loss     : {stats['prob_loss_20']*100:.1f}%")

    print("\nRISK ASSESSMENT")
    print("----------------")
    if stats["sharpe"] < 0:
        assessment = "POOR: Negative risk-adjusted return (Sharpe < 0)."
    elif stats["sharpe"] < 1.0:
        assessment = "WEAK: Low risk-adjusted returns (Sharpe < 1.0)."
    elif stats["sharpe"] < 1.5:
        assessment = "MODERATE: Acceptable but not outstanding."
    else:
        assessment = "STRONG: Attractive risk-adjusted profile."

    print(f"Overall Assessment: {assessment}")

    print("\nRECOMMENDATIONS")
    print("---------------")
    if stats["sharpe"] < 1.0:
        print("• Improve signal quality (features, models, regime handling).")
        print("• Reduce position sizes and enforce stricter risk rules.")
    else:
        print("• Monitor model drift and recalibrate periodically.")
        print("• Consider gradual scaling with strict drawdown controls.")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
