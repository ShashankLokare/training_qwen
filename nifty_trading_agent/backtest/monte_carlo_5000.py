"""
Monte Carlo Stress Test - 5000 Scenarios

This script:
- Loads daily equity curve from backtest output
- Converts it to daily returns
- Runs 5000 bootstrapped Monte Carlo scenarios
- Computes risk/performance metrics:
    - Mean/median PnL
    - Best/worst case
    - Sharpe, Sortino
    - Max Drawdown
    - VaR / Expected Shortfall
    - Probability of profit / >10% / >20% loss
"""

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

EQUITY_CSV = "reports/model_v2_backtest/daily_equity.csv"  # adjust path if needed

N_SIMULATIONS = 5000
CONFIDENCE_LEVEL = 0.95
HORIZON_DAYS = None   # None → use full history length
RISK_FREE_RATE = 0.0  # assume 0 for simplicity

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def load_daily_equity(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Equity CSV not found at: {path}")
    df = pd.read_csv(path)
    if "equity" not in df.columns:
        raise KeyError(f"'equity' column not found in {path}. Columns: {list(df.columns)}")
    if "date" not in df.columns:
        raise KeyError(f"'date' column not found in {path}. Columns: {list(df.columns)}")
    return df


def compute_daily_returns(equity_df: pd.DataFrame) -> pd.Series:
    df = equity_df.copy()
    df = df.sort_values("date")
    df["equity"] = df["equity"].astype(float)
    df["daily_return"] = df["equity"].pct_change()
    df = df.dropna(subset=["daily_return"])
    if df.empty:
        raise ValueError("No valid daily returns after pct_change (equity may be constant).")
    return df["daily_return"]


def max_drawdown(equity_curve: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - peak) / peak
    return float(drawdowns.min())


def simulate_paths(returns: np.ndarray, n_sims: int, horizon_days: int) -> np.ndarray:
    n_hist = len(returns)
    if n_hist == 0:
        raise ValueError("Historical returns array is empty.")
    idx = np.random.randint(0, n_hist, size=(n_sims, horizon_days))
    return returns[idx]


def compute_pnl_stats(sim_returns: np.ndarray) -> dict:
    # cumprod equity per path
    equity_curves = (1.0 + sim_returns).cumprod(axis=1)
    final_equity = equity_curves[:, -1]
    final_pnl = final_equity - 1.0

    mean_pnl = float(np.mean(final_pnl))
    std_pnl = float(np.std(final_pnl))
    median_pnl = float(np.median(final_pnl))
    best_pnl = float(np.max(final_pnl))
    worst_pnl = float(np.min(final_pnl))

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

    var_95 = float(np.percentile(final_pnl, (1 - cl_95) * 100.0))
    var_99 = float(np.percentile(final_pnl, (1 - cl_99) * 100.0))

    es_95 = float(final_pnl[final_pnl <= var_95].mean()) if np.any(final_pnl <= var_95) else 0.0
    es_99 = float(final_pnl[final_pnl <= var_99].mean()) if np.any(final_pnl <= var_99) else 0.0

    lower_ci = float(np.percentile(final_pnl, (1 - cl_95) * 100.0))
    upper_ci = float(np.percentile(final_pnl, cl_95 * 100.0))

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
        "pnl_ci_lower": lower_ci,
        "pnl_ci_upper": upper_ci,
        "prob_profit": prob_profit,
        "prob_loss_10": prob_loss_10,
        "prob_loss_20": prob_loss_20,
    }


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def main():
    print("=" * 80)
    print("MONTE CARLO STRESS TEST REPORT (5000 SCENARIOS)")
    print("=" * 80)

    equity_df = load_daily_equity(EQUITY_CSV)
    daily_returns = compute_daily_returns(equity_df)
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
    print(f"Mean PnL        : {pct(stats['mean_pnl'])}")
    print(f"Std Dev (PnL)   : {pct(stats['std_pnl'])}")
    print(f"Median PnL      : {pct(stats['median_pnl'])}")
    print(f"Best Case       : {pct(stats['best_pnl'])}")
    print(f"Worst Case      : {pct(stats['worst_pnl'])}")

    print("\nRISK METRICS")
    print("-------------")
    print(f"Sharpe Ratio    : {stats['sharpe']:.2f}")
    print(f"Sortino Ratio   : {stats['sortino']:.2f}")
    print(f"Avg Max Drawdown: {pct(stats['avg_max_drawdown'])}")

    print("\nVALUE AT RISK (VaR) & EXPECTED SHORTFALL")
    print("-----------------------------------------")
    print(f"VaR  {CONFIDENCE_LEVEL*100:.0f}% : {pct(stats['var_95'])}")
    print(f"VaR  99%          : {pct(stats['var_99'])}")
    print(f"ES   {CONFIDENCE_LEVEL*100:.0f}% : {pct(stats['es_95'])}")
    print(f"ES   99%          : {pct(stats['es_99'])}")

    print("\nCONFIDENCE INTERVALS (FINAL PnL)")
    print("--------------------------------")
    print(f"PnL {CONFIDENCE_LEVEL*100:.0f}% CI: [{pct(stats['pnl_ci_lower'])}, {pct(stats['pnl_ci_upper'])}]")

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
        print("• Improve signal quality and calibration.")
        print("• Tighten entries and risk management; consider smaller position sizes.")
    else:
        print("• Monitor drift and recalibrate the model periodically.")
        print("• Scale cautiously with strict drawdown and VaR limits.")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
