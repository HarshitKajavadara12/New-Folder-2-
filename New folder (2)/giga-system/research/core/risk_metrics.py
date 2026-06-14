"""
GIGA SYSTEM - Risk Metrics
===========================

Risk metrics quantify potential losses in a portfolio. Essential for:
1. Risk Management: Set position limits, allocate capital
2. Regulation: Basel III requires banks to report VaR
3. Performance: Risk-adjusted returns (Sharpe ratio)
4. Decision Making: Compare strategies on risk-reward basis

Key Concepts:
-------------
- VaR (Value at Risk): "What's the most I can lose with X% confidence?"
- CVaR (Expected Shortfall): "If I do lose more than VaR, what's the average loss?"
- Greeks-based Risk: How does portfolio react to market changes?

Historical Context:
-------------------
- VaR developed at J.P. Morgan (RiskMetrics, 1994)
- Widely adopted after 1996 Basel Accord
- Criticized after 2008 crisis (underestimated tail risk)
- CVaR addresses some VaR limitations
"""

import numpy as np
from numba import jit
from scipy import stats
from typing import Tuple, Dict, Optional, List
import math
from datetime import datetime, timedelta

try:
    from data.realtime_manager import get_data_manager
    REAL_DATA_AVAILABLE = True
except ImportError:
    REAL_DATA_AVAILABLE = False


# =============================================================================
# VALUE AT RISK (VaR)
# =============================================================================

def value_at_risk(
    returns: np.ndarray,
    confidence: float = 0.95,
    method: str = "historical"
) -> float:
    """
    Calculate Value at Risk (VaR).
    
    Definition:
    -----------
    VaR_α is the loss threshold such that:
    P(Loss > VaR_α) = 1 - α
    
    Interpretation:
    ---------------
    "With 95% confidence, daily loss will not exceed VaR_0.95"
    
    Example:
    --------
    If VaR_0.95 = $100,000, then:
    - 95% of days, you lose less than $100,000
    - 5% of days, you lose MORE than $100,000
    
    Methods:
    --------
    1. Historical: Use actual past returns (non-parametric)
    2. Parametric: Assume normal distribution
    3. Monte Carlo: Simulate future scenarios
    
    Parameters:
    -----------
    returns : np.ndarray
        Array of historical returns (daily, as decimals)
    confidence : float
        Confidence level (0.95 = 95%)
    method : str
        "historical", "parametric", or "monte_carlo"
    
    Returns:
    --------
    float : VaR as positive number (loss)
    
    Example:
    --------
    >>> returns = np.random.normal(0.001, 0.02, 252)  # 1 year of daily returns
    >>> var = value_at_risk(returns, 0.95, "historical")
    >>> print(f"95% VaR: {var:.4f}")
    95% VaR: 0.0312  # 3.12% daily loss threshold
    """
    returns = np.asarray(returns)
    
    if method == "historical":
        # Historical simulation: Use empirical distribution
        # VaR is the (1-α) percentile of returns
        var = -np.percentile(returns, (1 - confidence) * 100)
        
    elif method == "parametric":
        # Parametric (Variance-Covariance): Assume normal distribution
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)
        
        # VaR = -μ + σ × Z_α
        z_score = stats.norm.ppf(1 - confidence)
        var = -(mu + sigma * z_score)
        
    elif method == "monte_carlo":
        # Monte Carlo: Bootstrap from real historical data if available
        if REAL_DATA_AVAILABLE and len(returns) < 100:
            try:
                # Use broader market data for better sampling
                dm = get_data_manager()
                hist_df = dm.get_historical_data_sync('SPY',
                    (datetime.now() - timedelta(days=252*2)).strftime('%Y-%m-%d'),
                    datetime.now().strftime('%Y-%m-%d')
                )
                if not hist_df.empty:
                    real_returns = hist_df['close'].pct_change().dropna().values
                    # Bootstrap sample
                    simulated = np.random.choice(real_returns, size=10000, replace=True)
                else:
                    mu = np.mean(returns)
                    sigma = np.std(returns, ddof=1)
                    simulated = np.random.normal(mu, sigma, 10000)
            except:
                mu = np.mean(returns)
                sigma = np.std(returns, ddof=1)
                simulated = np.random.normal(mu, sigma, 10000)
        else:
            # Use provided returns for bootstrap
            simulated = np.random.choice(returns, size=10000, replace=True)
        
        var = -np.percentile(simulated, (1 - confidence) * 100)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return max(var, 0)  # VaR is always positive (represents loss)


def value_at_risk_portfolio(
    weights: np.ndarray,
    returns_matrix: np.ndarray,
    confidence: float = 0.95
) -> float:
    """
    Calculate VaR for a multi-asset portfolio.
    
    Parameters:
    -----------
    weights : np.ndarray
        Portfolio weights (should sum to 1)
    returns_matrix : np.ndarray
        Matrix of returns, shape (n_days, n_assets)
    confidence : float
        Confidence level
    
    Returns:
    --------
    float : Portfolio VaR
    """
    # Calculate portfolio returns
    portfolio_returns = returns_matrix @ weights
    
    return value_at_risk(portfolio_returns, confidence, "historical")


# =============================================================================
# CONDITIONAL VALUE AT RISK (CVaR / Expected Shortfall)
# =============================================================================

def conditional_var(
    returns: np.ndarray,
    confidence: float = 0.95,
    method: str = "historical"
) -> float:
    """
    Calculate Conditional Value at Risk (CVaR / Expected Shortfall).
    
    Definition:
    -----------
    CVaR_α = E[Loss | Loss > VaR_α]
    
    "Average loss when losses exceed VaR"
    
    Why CVaR is Better than VaR:
    ----------------------------
    1. Coherent risk measure (VaR is not)
       - Subadditivity: CVaR(A+B) ≤ CVaR(A) + CVaR(B)
       - VaR can violate this (diversification may increase VaR!)
    
    2. Considers tail severity
       - VaR: "You have 5% chance of losing more than $100K"
       - CVaR: "When you lose more than $100K, average loss is $150K"
    
    3. Required by Basel III for market risk capital
    
    Parameters:
    -----------
    Same as value_at_risk
    
    Returns:
    --------
    float : CVaR (Expected Shortfall) as positive number
    """
    returns = np.asarray(returns)
    
    if method == "historical":
        # Find VaR threshold
        var_threshold = np.percentile(returns, (1 - confidence) * 100)
        
        # Average of returns below VaR
        tail_losses = returns[returns <= var_threshold]
        
        if len(tail_losses) == 0:
            return value_at_risk(returns, confidence, method)
        
        cvar = -np.mean(tail_losses)
        
    elif method == "parametric":
        # For normal distribution:
        # CVaR = μ + σ × φ(z_α) / (1-α)
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)
        
        alpha = 1 - confidence
        z_score = stats.norm.ppf(alpha)
        pdf_at_z = stats.norm.pdf(z_score)
        
        cvar = -(mu - sigma * pdf_at_z / alpha)
        
    elif method == "monte_carlo":
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)
        
        simulated = np.random.normal(mu, sigma, 10000)
        var_threshold = np.percentile(simulated, (1 - confidence) * 100)
        tail_losses = simulated[simulated <= var_threshold]
        
        cvar = -np.mean(tail_losses)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return max(cvar, 0)


def expected_shortfall(
    returns: np.ndarray,
    confidence: float = 0.95
) -> float:
    """Alias for conditional_var (same concept, different name)."""
    return conditional_var(returns, confidence, "historical")


# =============================================================================
# PARAMETRIC VAR WITH FAT TAILS
# =============================================================================

def var_student_t(
    returns: np.ndarray,
    confidence: float = 0.95,
    df: Optional[float] = None
) -> float:
    """
    VaR using Student's t-distribution (captures fat tails).
    
    Why Student's t?
    ----------------
    - Financial returns have "fat tails" (more extreme events than normal)
    - Normal distribution: 5-sigma event is 1 in 3.5 million
    - Reality: 5-sigma events happen every few years
    - Student's t with low df captures this behavior
    
    Parameters:
    -----------
    df : float, optional
        Degrees of freedom. If None, estimated from data.
        Lower df = fatter tails
        df=3: Very fat tails
        df=30: Almost normal
    """
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    
    if df is None:
        # Estimate df by maximum likelihood
        # Simple approximation: df ≈ 4 + 6/kurtosis_excess
        kurt = stats.kurtosis(returns)
        if kurt > 0:
            df = 4 + 6 / kurt
        else:
            df = 30  # Near-normal
        df = max(min(df, 100), 2.1)  # Bound df
    
    # VaR using t-distribution
    t_quantile = stats.t.ppf(1 - confidence, df)
    var = -(mu + sigma * t_quantile * np.sqrt((df - 2) / df))
    
    return max(var, 0)


# =============================================================================
# STRESS TESTING
# =============================================================================

def stress_test_scenarios() -> Dict[str, Dict[str, float]]:
    """
    Pre-defined historical stress scenarios.
    
    Each scenario specifies:
    - Equity return (e.g., S&P 500)
    - Volatility multiplier (VIX spike)
    - Interest rate change
    - Credit spread widening
    
    Usage:
    ------
    Apply these scenarios to portfolio to see potential losses.
    """
    return {
        "black_monday_1987": {
            "name": "Black Monday (Oct 19, 1987)",
            "equity_return": -0.226,  # S&P 500 fell 22.6% in one day
            "volatility_mult": 3.0,   # VIX equivalent tripled
            "rate_change": -0.005,    # Flight to safety
            "description": "Largest single-day percentage decline in history"
        },
        "dot_com_crash_2000": {
            "name": "Dot-Com Crash (2000-2002)",
            "equity_return": -0.49,   # NASDAQ fell 49% over period
            "volatility_mult": 2.0,
            "rate_change": -0.03,     # Fed cut rates
            "description": "Tech bubble burst, S&P 500 down 50%"
        },
        "financial_crisis_2008": {
            "name": "Financial Crisis (Sep-Oct 2008)",
            "equity_return": -0.40,   # S&P 500 peak-to-trough
            "volatility_mult": 4.0,   # VIX hit 80
            "rate_change": -0.04,     # Rates to zero
            "credit_spread": 0.05,    # 500 bps spread widening
            "description": "Lehman collapse, global financial panic"
        },
        "covid_crash_2020": {
            "name": "COVID-19 Crash (Feb-Mar 2020)",
            "equity_return": -0.34,   # S&P 500 fell 34% in 23 days
            "volatility_mult": 5.0,   # VIX hit 82
            "rate_change": -0.015,    # Fed emergency cut
            "description": "Fastest 30%+ drop in history"
        },
        "rate_shock_2022": {
            "name": "Rate Shock (2022)",
            "equity_return": -0.20,   # S&P 500 down 20%
            "volatility_mult": 1.5,
            "rate_change": 0.0425,    # Fed raised 425 bps
            "description": "Fastest rate hike cycle in 40 years"
        },
        "hypothetical_tail": {
            "name": "Hypothetical Tail Event",
            "equity_return": -0.50,   # 50% crash
            "volatility_mult": 6.0,   # VIX to 150+
            "rate_change": -0.05,     # Rates to negative
            "description": "Worse than any historical event"
        }
    }


def apply_stress_scenario(
    portfolio_value: float,
    delta: float,
    gamma: float,
    vega: float,
    theta: float,
    rho: float,
    scenario: Dict[str, float],
    time_horizon_days: int = 1
) -> Dict[str, float]:
    """
    Apply stress scenario to a portfolio with Greeks.
    
    Uses Taylor expansion:
    ΔV ≈ Δ×ΔS + ½Γ×(ΔS)² + ν×Δσ + Θ×Δt + ρ×Δr
    
    Parameters:
    -----------
    portfolio_value : float - Current portfolio value
    delta, gamma, vega, theta, rho : float - Portfolio Greeks
    scenario : dict - Stress scenario parameters
    time_horizon_days : int - Time horizon for stress test
    
    Returns:
    --------
    Dict with P&L breakdown by Greek
    """
    # Extract scenario parameters
    equity_return = scenario.get("equity_return", 0)
    vol_mult = scenario.get("volatility_mult", 1.0)
    rate_change = scenario.get("rate_change", 0)
    
    # Assume current stock price = 100 for simplicity (or normalize)
    S = 100
    dS = S * equity_return
    
    # Volatility change (assuming base vol = 20%)
    base_vol = 0.20
    d_sigma = base_vol * (vol_mult - 1)  # Change in volatility
    
    # Time decay
    dt = time_horizon_days / 365.0
    
    # P&L components
    delta_pnl = delta * dS
    gamma_pnl = 0.5 * gamma * dS ** 2
    vega_pnl = vega * d_sigma * 100  # Vega is per 1% vol
    theta_pnl = theta * time_horizon_days  # Theta is daily
    rho_pnl = rho * rate_change * 100  # Rho is per 1% rate
    
    total_pnl = delta_pnl + gamma_pnl + vega_pnl + theta_pnl + rho_pnl
    
    return {
        "scenario_name": scenario.get("name", "Custom"),
        "description": scenario.get("description", ""),
        "total_pnl": total_pnl,
        "delta_pnl": delta_pnl,
        "gamma_pnl": gamma_pnl,
        "vega_pnl": vega_pnl,
        "theta_pnl": theta_pnl,
        "rho_pnl": rho_pnl,
        "pnl_percent": total_pnl / portfolio_value * 100 if portfolio_value > 0 else 0,
        "new_portfolio_value": portfolio_value + total_pnl
    }


def run_all_stress_tests(
    portfolio_value: float,
    delta: float,
    gamma: float,
    vega: float,
    theta: float,
    rho: float,
    spot_price: float = 100.0,
    base_vol: float = 0.20
) -> List[Dict[str, float]]:
    """Run all pre-defined stress scenarios on portfolio."""
    scenarios = stress_test_scenarios()
    results = []
    
    for scenario_id, scenario in scenarios.items():
        result = apply_stress_scenario(
            portfolio_value, delta, gamma, vega, theta, rho, scenario,
            spot_price=spot_price, base_vol=base_vol
        )
        result["scenario_id"] = scenario_id
        results.append(result)
    
    return results


# =============================================================================
# DRAWDOWN ANALYSIS
# =============================================================================

def calculate_drawdown(returns: np.ndarray) -> Tuple[np.ndarray, float, int]:
    """
    Calculate drawdown series and maximum drawdown.
    
    Drawdown Definition:
    --------------------
    Drawdown(t) = (Peak(t) - Value(t)) / Peak(t)
    
    Where Peak(t) = max(Value(s) for s ≤ t)
    
    "How far below your all-time high are you?"
    
    Parameters:
    -----------
    returns : np.ndarray
        Array of period returns (not cumulative)
    
    Returns:
    --------
    Tuple of:
        - drawdown_series : np.ndarray of drawdown values
        - max_drawdown : float, worst drawdown
        - max_dd_duration : int, longest drawdown duration
    """
    # Calculate cumulative returns (wealth curve)
    cumulative = np.cumprod(1 + returns)
    
    # Running maximum (peak)
    running_max = np.maximum.accumulate(cumulative)
    
    # Drawdown series
    drawdown = (running_max - cumulative) / running_max
    
    # Maximum drawdown
    max_drawdown = np.max(drawdown)
    
    # Drawdown duration (periods below peak)
    in_drawdown = cumulative < running_max
    
    # Find longest consecutive drawdown period
    max_duration = 0
    current_duration = 0
    
    for is_dd in in_drawdown:
        if is_dd:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0
    
    return drawdown, max_drawdown, max_duration


def calmar_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """
    Calculate Calmar Ratio: Annual Return / Max Drawdown.
    
    Interpretation:
    ---------------
    - Calmar = 1.0: Annual return equals max drawdown
    - Calmar > 3.0: Excellent (gain 3x what you risk losing)
    - Calmar < 0.5: Poor risk-adjusted returns
    
    Note: Uses absolute value of max drawdown.
    """
    if len(returns) < 2:
        return 0.0
    
    # Annualized return
    cumulative_return = np.prod(1 + returns) - 1
    years = len(returns) / periods_per_year
    annual_return = (1 + cumulative_return) ** (1 / years) - 1 if years > 0 else 0
    
    # Max drawdown
    _, max_dd, _ = calculate_drawdown(returns)
    
    if max_dd == 0:
        return np.inf if annual_return > 0 else 0.0
    
    return annual_return / max_dd


# =============================================================================
# RISK-ADJUSTED PERFORMANCE METRICS
# =============================================================================

def sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe Ratio: (Return - Rf) / Volatility.
    
    Developed by William Sharpe (Nobel Prize 1990).
    
    Interpretation:
    ---------------
    - Sharpe < 0: Losing money
    - Sharpe 0-1: Below average
    - Sharpe 1-2: Good
    - Sharpe 2-3: Very good
    - Sharpe > 3: Excellent (rare, may indicate overfitting)
    
    Annualization:
    --------------
    Sharpe_annual = Sharpe_daily × √252
    """
    if len(returns) < 2:
        return 0.0
    
    # Daily risk-free rate
    rf_daily = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    
    excess_returns = returns - rf_daily
    
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns, ddof=1)
    
    if std_excess == 0:
        return np.inf if mean_excess > 0 else 0.0
    
    # Annualize
    sharpe = (mean_excess / std_excess) * np.sqrt(periods_per_year)
    
    return sharpe


def sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    target_return: float = 0.0
) -> float:
    """
    Calculate Sortino Ratio: Uses downside deviation instead of total volatility.
    
    Why Sortino > Sharpe:
    ---------------------
    - Sharpe penalizes upside volatility (why is that bad?)
    - Sortino only penalizes downside moves (actual risk)
    - Better for asymmetric return distributions
    
    Downside Deviation:
    -------------------
    Only considers returns below target (usually 0 or Rf)
    """
    if len(returns) < 2:
        return 0.0
    
    rf_daily = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    target_daily = target_return / periods_per_year
    
    excess_returns = returns - rf_daily
    
    # Downside deviation: std of returns below target
    downside_returns = returns[returns < target_daily] - target_daily
    
    if len(downside_returns) == 0:
        return np.inf if np.mean(excess_returns) > 0 else 0.0
    
    downside_std = np.sqrt(np.mean(downside_returns ** 2))
    
    if downside_std == 0:
        return np.inf if np.mean(excess_returns) > 0 else 0.0
    
    # Annualize
    sortino = (np.mean(excess_returns) / downside_std) * np.sqrt(periods_per_year)
    
    return sortino


def information_ratio(
    returns: np.ndarray,
    benchmark_returns: np.ndarray,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Information Ratio: Measures consistency of outperformance.
    
    IR = (Return - Benchmark) / Tracking Error
    
    Where Tracking Error = StdDev(Return - Benchmark)
    
    Interpretation:
    ---------------
    - IR > 0.5: Good (consistent outperformance)
    - IR > 1.0: Excellent
    - IR < 0: Underperforming benchmark
    
    Used by: Active fund managers to justify fees
    """
    if len(returns) < 2 or len(returns) != len(benchmark_returns):
        return 0.0
    
    active_returns = returns - benchmark_returns
    
    tracking_error = np.std(active_returns, ddof=1)
    
    if tracking_error == 0:
        return np.inf if np.mean(active_returns) > 0 else 0.0
    
    ir = (np.mean(active_returns) / tracking_error) * np.sqrt(periods_per_year)
    
    return ir


# =============================================================================
# BENCHMARK
# =============================================================================

if __name__ == "__main__":
    # Generate sample returns (simulate 2 years of daily returns)
    np.random.seed(42)
    n_days = 504
    
    # Mean 10% annual return, 20% annual vol
    daily_mean = 0.10 / 252
    daily_vol = 0.20 / np.sqrt(252)
    
    returns = np.random.normal(daily_mean, daily_vol, n_days)
    
    # Add some fat tails (occasional large moves)
    returns[100] = -0.08  # Simulate a bad day
    returns[250] = -0.06  # Another bad day
    
    print("=" * 60)
    print("RISK METRICS DEMONSTRATION")
    print("=" * 60)
    
    print(f"\nSample Data: {n_days} daily returns")
    print(f"Mean daily return: {np.mean(returns)*100:.3f}%")
    print(f"Daily volatility: {np.std(returns)*100:.3f}%")
    
    print("\n--- Value at Risk ---")
    var_hist = value_at_risk(returns, 0.95, "historical")
    var_param = value_at_risk(returns, 0.95, "parametric")
    var_t = var_student_t(returns, 0.95)
    
    print(f"95% VaR (Historical): {var_hist*100:.2f}%")
    print(f"95% VaR (Parametric): {var_param*100:.2f}%")
    print(f"95% VaR (Student-t):  {var_t*100:.2f}%")
    
    print("\n--- Expected Shortfall ---")
    cvar_hist = conditional_var(returns, 0.95, "historical")
    cvar_param = conditional_var(returns, 0.95, "parametric")
    
    print(f"95% CVaR (Historical): {cvar_hist*100:.2f}%")
    print(f"95% CVaR (Parametric): {cvar_param*100:.2f}%")
    
    print("\n--- Drawdown Analysis ---")
    dd_series, max_dd, max_duration = calculate_drawdown(returns)
    print(f"Maximum Drawdown: {max_dd*100:.2f}%")
    print(f"Max DD Duration: {max_duration} days")
    
    print("\n--- Performance Ratios ---")
    print(f"Sharpe Ratio:  {sharpe_ratio(returns, 0.02):.2f}")
    print(f"Sortino Ratio: {sortino_ratio(returns, 0.02):.2f}")
    print(f"Calmar Ratio:  {calmar_ratio(returns):.2f}")
    
    print("\n--- Stress Test Example ---")
    results = run_all_stress_tests(
        portfolio_value=1000000,
        delta=500,  # Equivalent to 500 shares
        gamma=10,
        vega=1000,
        theta=-50,
        rho=200
    )
    
    print("\nPortfolio: $1,000,000 with delta=500, gamma=10, vega=1000")
    print("-" * 50)
    for r in results[:3]:  # Show top 3
        print(f"{r['scenario_name']}: ${r['total_pnl']:,.0f} ({r['pnl_percent']:.1f}%)")
