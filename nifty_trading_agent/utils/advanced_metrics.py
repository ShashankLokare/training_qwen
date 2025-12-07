#!/usr/bin/env python3
"""
Advanced Performance Metrics for Nifty Trading Agent v2
Professional-grade risk-adjusted performance calculations
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from scipy import stats

def calculate_information_ratio(strategy_returns: np.ndarray,
                              benchmark_returns: np.ndarray) -> float:
    """
    Calculate Information Ratio (active return / tracking error)

    Args:
        strategy_returns: Strategy daily returns
        benchmark_returns: Benchmark daily returns

    Returns:
        Information ratio (annualized)
    """
    if len(strategy_returns) != len(benchmark_returns):
        raise ValueError("Strategy and benchmark returns must have same length")

    # Calculate active returns
    active_returns = strategy_returns - benchmark_returns

    # Annualized information ratio
    annualized_active_return = np.mean(active_returns) * 252
    annualized_tracking_error = np.std(active_returns) * np.sqrt(252)

    if annualized_tracking_error == 0:
        return 0.0

    return annualized_active_return / annualized_tracking_error

def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sortino Ratio (return / downside deviation)

    Args:
        returns: Daily returns
        risk_free_rate: Annual risk-free rate

    Returns:
        Sortino ratio (annualized)
    """
    # Calculate downside deviation (only negative returns)
    downside_returns = returns[returns < 0]

    if len(downside_returns) == 0:
        return float('inf')  # No downside risk

    # Annualized metrics
    annualized_return = np.mean(returns) * 252
    annualized_downside_deviation = np.std(downside_returns) * np.sqrt(252)

    if annualized_downside_deviation == 0:
        return float('inf')

    return (annualized_return - risk_free_rate) / annualized_downside_deviation

def calculate_calmar_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
    """
    Calculate Calmar Ratio (annual return / maximum drawdown)

    Args:
        returns: Daily returns
        risk_free_rate: Annual risk-free rate

    Returns:
        Calmar ratio
    """
    if len(returns) == 0:
        return 0.0

    # Calculate cumulative returns
    cumulative = np.cumprod(1 + returns)

    # Find maximum drawdown
    peak = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - peak) / peak
    max_drawdown = np.min(drawdowns)

    if max_drawdown >= 0:
        return 0.0  # No drawdown

    # Annualized return
    annualized_return = np.mean(returns) * 252

    return (annualized_return - risk_free_rate) / abs(max_drawdown)

def calculate_omega_ratio(returns: np.ndarray, threshold: float = 0.0) -> float:
    """
    Calculate Omega Ratio (probability-weighted returns above threshold)

    Args:
        returns: Daily returns
        threshold: Minimum acceptable return (default: 0)

    Returns:
        Omega ratio
    """
    if len(returns) == 0:
        return 1.0

    # Returns above threshold
    upside = returns[returns > threshold] - threshold
    downside = threshold - returns[returns <= threshold]

    if len(downside) == 0:
        return float('inf')  # No downside

    return np.sum(upside) / np.sum(downside)

def calculate_alpha_beta(strategy_returns: np.ndarray,
                        market_returns: np.ndarray,
                        risk_free_rate: float = 0.02) -> Dict[str, float]:
    """
    Calculate Jensen's Alpha and Beta using CAPM

    Args:
        strategy_returns: Strategy daily returns
        market_returns: Market daily returns
        risk_free_rate: Annual risk-free rate

    Returns:
        Dictionary with alpha, beta, and statistics
    """
    if len(strategy_returns) != len(market_returns):
        raise ValueError("Strategy and market returns must have same length")

    # Convert to excess returns
    excess_strategy = strategy_returns - risk_free_rate/252
    excess_market = market_returns - risk_free_rate/252

    # Perform linear regression: excess_strategy = alpha + beta * excess_market
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(excess_market, excess_strategy)

        # Annualize alpha
        annualized_alpha = intercept * 252

        return {
            'alpha': annualized_alpha,
            'beta': slope,
            'r_squared': r_value ** 2,
            'alpha_t_stat': intercept / std_err if std_err != 0 else 0,
            'alpha_p_value': p_value,
            'alpha_significant': p_value < 0.05
        }

    except Exception as e:
        return {
            'error': str(e),
            'alpha': 0.0,
            'beta': 1.0,
            'r_squared': 0.0
        }

def calculate_maximum_drawdown(returns: np.ndarray) -> float:
    """
    Calculate maximum drawdown from peak to trough

    Args:
        returns: Daily returns

    Returns:
        Maximum drawdown as negative decimal
    """
    if len(returns) == 0:
        return 0.0

    # Calculate cumulative returns
    cumulative = np.cumprod(1 + returns)

    # Calculate drawdowns
    peak = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - peak) / peak

    return np.min(drawdowns)  # Most negative value

def calculate_value_at_risk(returns: np.ndarray, confidence_level: float = 0.95) -> float:
    """
    Calculate Value at Risk (VaR)

    Args:
        returns: Daily returns
        confidence_level: Confidence level (0.95 for 95% VaR)

    Returns:
        VaR as negative decimal
    """
    if len(returns) == 0:
        return 0.0

    return np.percentile(returns, (1 - confidence_level) * 100)

def calculate_expected_shortfall(returns: np.ndarray, confidence_level: float = 0.95) -> float:
    """
    Calculate Expected Shortfall (Conditional VaR)

    Args:
        returns: Daily returns
        confidence_level: Confidence level

    Returns:
        Expected shortfall as negative decimal
    """
    if len(returns) == 0:
        return 0.0

    # Find VaR threshold
    var_threshold = np.percentile(returns, (1 - confidence_level) * 100)

    # Average of returns beyond VaR
    tail_losses = returns[returns <= var_threshold]

    if len(tail_losses) == 0:
        return var_threshold

    return np.mean(tail_losses)

def comprehensive_performance_analysis(strategy_returns: np.ndarray,
                                    benchmark_returns: np.ndarray = None,
                                    risk_free_rate: float = 0.02) -> Dict[str, Any]:
    """
    Comprehensive performance analysis with all key metrics

    Args:
        strategy_returns: Strategy daily returns
        benchmark_returns: Benchmark daily returns (optional)
        risk_free_rate: Annual risk-free rate

    Returns:
        Dictionary with all performance metrics
    """
    if benchmark_returns is None:
        # Use zero return as benchmark if not provided
        benchmark_returns = np.zeros_like(strategy_returns)

    analysis = {
        'basic_metrics': {
            'total_return': np.prod(1 + strategy_returns) - 1,
            'annualized_return': np.mean(strategy_returns) * 252,
            'volatility': np.std(strategy_returns) * np.sqrt(252),
            'sharpe_ratio': (np.mean(strategy_returns) * 252 - risk_free_rate) / (np.std(strategy_returns) * np.sqrt(252)),
            'max_drawdown': calculate_maximum_drawdown(strategy_returns)
        },

        'advanced_metrics': {
            'sortino_ratio': calculate_sortino_ratio(strategy_returns, risk_free_rate),
            'calmar_ratio': calculate_calmar_ratio(strategy_returns, risk_free_rate),
            'omega_ratio': calculate_omega_ratio(strategy_returns),
            'information_ratio': calculate_information_ratio(strategy_returns, benchmark_returns)
        },

        'risk_metrics': {
            'var_95': calculate_value_at_risk(strategy_returns, 0.95),
            'var_99': calculate_value_at_risk(strategy_returns, 0.99),
            'expected_shortfall_95': calculate_expected_shortfall(strategy_returns, 0.95),
            'expected_shortfall_99': calculate_expected_shortfall(strategy_returns, 0.99)
        }
    }

    # Add alpha/beta analysis if benchmark provided
    if benchmark_returns is not None and not np.allclose(benchmark_returns, 0):
        analysis['alpha_beta'] = calculate_alpha_beta(strategy_returns, benchmark_returns, risk_free_rate)

    # Add probability metrics
    analysis['probability_metrics'] = {
        'prob_positive_return': np.mean(strategy_returns > 0),
        'prob_negative_return': np.mean(strategy_returns < 0),
        'prob_return_gt_1pct': np.mean(strategy_returns > 0.01),
        'prob_return_lt_minus_1pct': np.mean(strategy_returns < -0.01)
    }

    return analysis

def calculate_kelly_criterion(win_rate: float, win_loss_ratio: float) -> float:
    """
    Calculate optimal position size using Kelly Criterion

    Args:
        win_rate: Probability of winning (0-1)
        win_loss_ratio: Average win / average loss ratio

    Returns:
        Optimal fraction of capital to risk
    """
    if win_rate <= 0 or win_rate >= 1 or win_loss_ratio <= 0:
        return 0.0

    # Kelly formula: K = (bp - q) / b
    # Where: b = odds (win_loss_ratio), p = win_rate, q = loss_rate
    b = win_loss_ratio
    p = win_rate
    q = 1 - p

    kelly_fraction = (b * p - q) / b

    # Half-Kelly for safety (more conservative)
    return max(0, kelly_fraction * 0.5)

def calculate_optimal_portfolio_weights(returns_matrix: np.ndarray,
                                       target_return: float = None,
                                       risk_free_rate: float = 0.02) -> Dict[str, Any]:
    """
    Calculate optimal portfolio weights using Modern Portfolio Theory

    Args:
        returns_matrix: Matrix of asset returns (n_assets x n_periods)
        target_return: Target portfolio return (optional)
        risk_free_rate: Risk-free rate

    Returns:
        Dictionary with optimal weights and portfolio statistics
    """
    if returns_matrix.shape[0] == 0:
        return {'error': 'No assets provided'}

    n_assets = returns_matrix.shape[0]

    # Calculate mean returns and covariance matrix
    mean_returns = np.mean(returns_matrix, axis=1)
    cov_matrix = np.cov(returns_matrix)

    # If only one asset, return simple statistics
    if n_assets == 1:
        return {
            'weights': [1.0],
            'expected_return': mean_returns[0] * 252,
            'volatility': np.std(returns_matrix[0]) * np.sqrt(252),
            'sharpe_ratio': (mean_returns[0] * 252 - risk_free_rate) / (np.std(returns_matrix[0]) * np.sqrt(252))
        }

    try:
        from scipy.optimize import minimize

        # Objective: minimize portfolio variance for given return, or maximize Sharpe
        if target_return is not None:
            # Minimize variance for target return
            def objective(weights):
                portfolio_return = np.sum(mean_returns * weights) * 252
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
                return portfolio_vol

            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
                {'type': 'eq', 'fun': lambda w: np.sum(mean_returns * w) * 252 - target_return},  # Target return
            ]
        else:
            # Maximize Sharpe ratio
            def objective(weights):
                portfolio_return = np.sum(mean_returns * weights) * 252
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
                sharpe = (portfolio_return - risk_free_rate) / portfolio_vol
                return -sharpe  # Minimize negative Sharpe

            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
            ]

        bounds = [(0, 1) for _ in range(n_assets)]  # Long-only constraint
        initial_weights = np.ones(n_assets) / n_assets

        result = minimize(objective, initial_weights, method='SLSQP',
                         bounds=bounds, constraints=constraints)

        if result.success:
            optimal_weights = result.x
            portfolio_return = np.sum(mean_returns * optimal_weights) * 252
            portfolio_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix * 252, optimal_weights)))
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_vol

            return {
                'weights': optimal_weights.tolist(),
                'expected_return': portfolio_return,
                'volatility': portfolio_vol,
                'sharpe_ratio': sharpe_ratio,
                'optimization_success': True
            }
        else:
            # Fallback to equal weighting
            equal_weights = np.ones(n_assets) / n_assets
            return {
                'weights': equal_weights.tolist(),
                'expected_return': np.sum(mean_returns * equal_weights) * 252,
                'volatility': np.sqrt(np.dot(equal_weights.T, np.dot(cov_matrix * 252, equal_weights))),
                'sharpe_ratio': 0.0,
                'optimization_success': False,
                'fallback_reason': result.message
            }

    except ImportError:
        # Fallback if scipy not available
        equal_weights = np.ones(n_assets) / n_assets
        return {
            'weights': equal_weights.tolist(),
            'expected_return': np.sum(mean_returns * equal_weights) * 252,
            'volatility': np.sqrt(np.dot(equal_weights.T, np.dot(cov_matrix * 252, equal_weights))),
            'sharpe_ratio': 0.0,
            'optimization_success': False,
            'fallback_reason': 'scipy.optimize not available'
        }

def calculate_rolling_sharpe_ratio(returns: np.ndarray, window: int = 252) -> np.ndarray:
    """
    Calculate rolling Sharpe ratio

    Args:
        returns: Daily returns
        window: Rolling window size (default: 252 trading days)

    Returns:
        Array of rolling Sharpe ratios
    """
    if len(returns) < window:
        return np.array([])

    rolling_sharpe = []
    risk_free_rate = 0.02 / 252  # Daily risk-free rate

    for i in range(window, len(returns) + 1):
        window_returns = returns[i-window:i]
        mean_return = np.mean(window_returns)
        std_return = np.std(window_returns)

        if std_return > 0:
            sharpe = (mean_return - risk_free_rate) / std_return
        else:
            sharpe = 0.0

        rolling_sharpe.append(sharpe)

    return np.array(rolling_sharpe)

def calculate_performance_attribution(strategy_returns: np.ndarray,
                                    benchmark_returns: np.ndarray,
                                    factor_returns: np.ndarray = None) -> Dict[str, float]:
    """
    Multi-factor performance attribution analysis

    Args:
        strategy_returns: Strategy daily returns
        benchmark_returns: Benchmark daily returns
        factor_returns: Factor returns matrix (optional)

    Returns:
        Performance attribution breakdown
    """
    if len(strategy_returns) != len(benchmark_returns):
        raise ValueError("Strategy and benchmark returns must have same length")

    # Basic attribution
    active_returns = strategy_returns - benchmark_returns
    total_active_return = np.sum(active_returns)

    attribution = {
        'total_active_return': total_active_return,
        'annualized_active_return': total_active_return / len(strategy_returns) * 252,
        'tracking_error': np.std(active_returns) * np.sqrt(252),
        'information_ratio': (np.mean(active_returns) * 252) / (np.std(active_returns) * np.sqrt(252))
    }

    # Multi-factor attribution if factors provided
    if factor_returns is not None and factor_returns.shape[1] > 0:
        try:
            # Run regression: active_returns = beta1*factor1 + beta2*factor2 + ... + alpha
            X = factor_returns.T  # Transpose to get factors as columns
            y = active_returns

            # Add constant for alpha
            X = np.column_stack([np.ones(len(y)), X])

            # OLS regression
            beta = np.linalg.lstsq(X, y, rcond=None)[0]

            attribution['alpha'] = beta[0] * 252  # Annualized alpha
            attribution['factor_betas'] = beta[1:].tolist()
            attribution['r_squared'] = 1 - np.var(y - X @ beta) / np.var(y)

        except Exception as e:
            attribution['factor_attribution_error'] = str(e)

    return attribution

# Example usage and testing functions
def demo_advanced_metrics():
    """Demonstrate advanced metrics calculation"""
    print("📊 ADVANCED METRICS DEMONSTRATION")
    print("=" * 40)

    # Generate sample data
    np.random.seed(42)
    n_days = 252

    # Sample strategy returns (slightly positive with realistic volatility)
    strategy_returns = np.random.normal(0.0008, 0.015, n_days)

    # Sample benchmark returns (market-like)
    benchmark_returns = np.random.normal(0.0005, 0.02, n_days)

    # Comprehensive analysis
    analysis = comprehensive_performance_analysis(
        strategy_returns=strategy_returns,
        benchmark_returns=benchmark_returns,
        risk_free_rate=0.02
    )

    # Print results
    print("\n📈 BASIC METRICS:")
    basic = analysis['basic_metrics']
    print(f"  Total Return: {basic['total_return']:.2%}")
    print(f"  Annualized Return: {basic['annualized_return']:.2%}")
    print(f"  Volatility: {basic['volatility']:.2%}")
    print(f"  Sharpe Ratio: {basic['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {basic['max_drawdown']:.2%}")

    print("\n🎯 ADVANCED METRICS:")
    advanced = analysis['advanced_metrics']
    print(f"  Sortino Ratio: {advanced['sortino_ratio']:.2f}")
    print(f"  Calmar Ratio: {advanced['calmar_ratio']:.2f}")
    print(f"  Omega Ratio: {advanced['omega_ratio']:.2f}")
    print(f"  Information Ratio: {advanced['information_ratio']:.2f}")

    print("\n⚠️  RISK METRICS:")
    risk = analysis['risk_metrics']
    print(f"  VaR 95%: {risk['var_95']:.2%}")
    print(f"  VaR 99%: {risk['var_99']:.2%}")
    print(f"  Expected Shortfall 95%: {risk['expected_shortfall_95']:.2%}")

    if 'alpha_beta' in analysis:
        ab = analysis['alpha_beta']
        print("\n📊 ALPHA/BETA ANALYSIS:")
        print(f"  Alpha: {ab['alpha']:.2%}")
        print(f"  Beta: {ab['beta']:.3f}")
        print(f"  R²: {ab['r_squared']:.3f}")
        print(f"  Alpha Significant: {ab['alpha_significant']}")

    print("\n✅ Advanced metrics demonstration completed!")

    return analysis

if __name__ == "__main__":
    # Run demonstration
    results = demo_advanced_metrics()
