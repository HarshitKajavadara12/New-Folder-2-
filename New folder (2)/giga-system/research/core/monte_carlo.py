"""
GIGA SYSTEM - Monte Carlo Simulation
=====================================

Monte Carlo methods use random sampling to solve problems that might be
deterministic in principle. For option pricing, we simulate thousands of
possible future price paths and average the payoffs.

Historical Context:
-------------------
- Named after the Monte Carlo Casino in Monaco
- Developed during WWII for nuclear physics (Manhattan Project)
- Applied to finance by Phelim Boyle (1977)
- Essential for pricing path-dependent options (Asian, barriers, lookbacks)

Why Monte Carlo?
----------------
1. Flexibility: Can price ANY derivative (exotic options)
2. Accuracy: Error decreases as O(1/√N) - more simulations = better
3. Parallelizable: Each path is independent → GPU acceleration
4. Path-dependent: Captures complex payoff structures

Limitations:
------------
1. Slow for simple options (Black-Scholes is analytical)
2. Variance can be high (need variance reduction techniques)
3. Greek calculation requires bump-and-reprice

Performance Target: 10,000 paths in <10ms (vectorized NumPy)
"""

import numpy as np
from numba import jit, prange
from typing import Callable, Tuple, Optional, Dict
from datetime import datetime, timedelta
import math

try:
    from data.realtime_manager import get_data_manager
    REAL_DATA_AVAILABLE = True
except ImportError:
    REAL_DATA_AVAILABLE = False


# =============================================================================
# GEOMETRIC BROWNIAN MOTION (GBM)
# =============================================================================

def simulate_gbm_paths(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int = 10000,
    n_steps: int = 252,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Simulate stock price paths using Geometric Brownian Motion.
    
    Mathematical Model:
    -------------------
    GBM assumes stock price follows:
        dS = μS dt + σS dW
    
    Where:
        - μ = drift (expected return)
        - σ = volatility
        - dW = Wiener process increment (Brownian motion)
    
    Solution (Itô's Lemma):
    -----------------------
    S(t) = S(0) × exp[(μ - σ²/2)t + σW(t)]
    
    Discrete simulation:
    S(t+Δt) = S(t) × exp[(r - σ²/2)Δt + σ√Δt × Z]
    
    Where Z ~ N(0,1)
    
    Note: We use risk-neutral measure (μ = r) for option pricing.
    
    Parameters:
    -----------
    S0 : float
        Initial stock price
    r : float
        Risk-free rate (annualized)
    sigma : float
        Volatility (annualized)
    T : float
        Time horizon in years
    n_paths : int
        Number of simulation paths
    n_steps : int
        Number of time steps (252 = daily for 1 year)
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    np.ndarray
        Shape (n_paths, n_steps + 1) array of price paths
        First column is S0, last column is S(T)
    
    Example:
    --------
    >>> paths = simulate_gbm_paths(100, 0.05, 0.20, 1.0, n_paths=10000)
    >>> paths.shape
    (10000, 253)
    >>> paths[:, -1].mean()  # Average final price ≈ S0 * exp(rT)
    ~105.1
    """
    if seed is not None:
        np.random.seed(seed)
    
    dt = T / n_steps
    
    # Pre-compute constants
    drift = (r - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt)
    
    # Generate random increments from historical data if available
    # Otherwise fall back to standard normal
    if REAL_DATA_AVAILABLE and seed is None:
        try:
            # Bootstrap from real market data for more realistic paths
            dm = get_data_manager()
            hist_df = dm.get_historical_data_sync('SPY', 
                (datetime.now() - timedelta(days=252*2)).strftime('%Y-%m-%d'),
                datetime.now().strftime('%Y-%m-%d')
            )
            if not hist_df.empty:
                real_returns = hist_df['close'].pct_change().dropna().values
                # Standardize returns for sampling
                Z_samples = (real_returns - real_returns.mean()) / real_returns.std()
                # Bootstrap sample with replacement
                indices = np.random.choice(len(Z_samples), size=(n_paths, n_steps), replace=True)
                Z = Z_samples[indices]
            else:
                Z = np.random.standard_normal((n_paths, n_steps))
        except:
            Z = np.random.standard_normal((n_paths, n_steps))
    else:
        Z = np.random.standard_normal((n_paths, n_steps))
    
    # Calculate log returns: log(S(t+dt)/S(t)) = drift + diffusion * Z
    log_returns = drift + diffusion * Z
    
    # Cumulative sum to get log(S(t)/S(0))
    log_paths = np.cumsum(log_returns, axis=1)
    
    # Prepend zero column (log(S0/S0) = 0)
    log_paths = np.column_stack([np.zeros(n_paths), log_paths])
    
    # Convert to prices: S(t) = S0 * exp(log_path)
    paths = S0 * np.exp(log_paths)
    
    return paths


def simulate_gbm_terminal(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int = 10000,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Simulate only terminal stock prices (more efficient for European options).
    
    Returns:
    --------
    np.ndarray
        Shape (n_paths,) array of terminal prices S(T)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Use historical bootstrap for more realistic terminal prices
    if REAL_DATA_AVAILABLE and seed is None:
        try:
            dm = get_data_manager()
            hist_df = dm.get_historical_data_sync('SPY',
                (datetime.now() - timedelta(days=int(T*252*2))).strftime('%Y-%m-%d'),
                datetime.now().strftime('%Y-%m-%d')
            )
            if not hist_df.empty:
                real_returns = hist_df['close'].pct_change().dropna().values
                Z_samples = (real_returns - real_returns.mean()) / real_returns.std()
                Z = np.random.choice(Z_samples, size=n_paths, replace=True)
            else:
                Z = np.random.standard_normal(n_paths)
        except:
            Z = np.random.standard_normal(n_paths)
    else:
        Z = np.random.standard_normal(n_paths)
    
    # Direct simulation of S(T)
    ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
    
    return ST


# =============================================================================
# EUROPEAN OPTION PRICING
# =============================================================================

def monte_carlo_european(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = "call",
    n_paths: int = 10000,
    seed: Optional[int] = None
) -> Tuple[float, float]:
    """
    Price a European option using Monte Carlo simulation.
    
    Algorithm:
    ----------
    1. Simulate N terminal prices S(T)
    2. Calculate payoff for each path:
       - Call: max(S(T) - K, 0)
       - Put: max(K - S(T), 0)
    3. Average payoffs and discount to present value
    
    Statistical Properties:
    -----------------------
    - Estimator: V̂ = e^(-rT) × (1/N) × Σ payoff_i
    - This is unbiased: E[V̂] = V (true option value)
    - Standard error: SE = σ_payoff / √N
    - 95% CI: V̂ ± 1.96 × SE
    
    Parameters:
    -----------
    (same as simulate_gbm_paths, plus K for strike)
    
    Returns:
    --------
    Tuple[float, float]
        (option_price, standard_error)
    
    Example:
    --------
    >>> price, se = monte_carlo_european(100, 100, 0.05, 0.20, 1.0, "call")
    >>> print(f"Price: {price:.4f} ± {1.96*se:.4f}")
    Price: 10.4506 ± 0.2134
    """
    # Simulate terminal prices
    ST = simulate_gbm_terminal(S0, r, sigma, T, n_paths, seed)
    
    # Calculate payoffs
    if option_type.lower() == "call":
        payoffs = np.maximum(ST - K, 0)
    else:
        payoffs = np.maximum(K - ST, 0)
    
    # Discount to present value
    discount = np.exp(-r * T)
    discounted_payoffs = discount * payoffs
    
    # Estimate price and standard error
    price = np.mean(discounted_payoffs)
    std_error = np.std(discounted_payoffs, ddof=1) / np.sqrt(n_paths)
    
    return price, std_error


def monte_carlo_price(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = "call",
    n_paths: int = 10000
) -> float:
    """Simplified interface returning only the price."""
    price, _ = monte_carlo_european(S0, K, r, sigma, T, option_type, n_paths)
    return price


# =============================================================================
# VARIANCE REDUCTION TECHNIQUES
# =============================================================================

def monte_carlo_antithetic(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = "call",
    n_paths: int = 10000,
    seed: Optional[int] = None
) -> Tuple[float, float]:
    """
    Monte Carlo with Antithetic Variates (variance reduction).
    
    Technique:
    ----------
    For each random number Z, also use -Z (its "mirror").
    This creates negatively correlated paths that cancel out variance.
    
    Why It Works:
    -------------
    - If Z gives high path, -Z gives low path
    - Averaging reduces variance because errors cancel
    - Effectively doubles sample size without extra random numbers
    
    Variance Reduction:
    -------------------
    - Standard MC variance: σ²/N
    - Antithetic variance: σ²(1+ρ)/(2N), where ρ < 0 for options
    - Typical reduction: 30-50% lower variance
    
    Returns:
    --------
    Same as monte_carlo_european
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Only need half the paths (we'll generate mirror paths)
    half_paths = n_paths // 2
    
    Z = np.random.standard_normal(half_paths)
    
    # Original paths
    ST_original = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
    
    # Antithetic paths (using -Z)
    ST_antithetic = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * (-Z))
    
    # Combine
    ST = np.concatenate([ST_original, ST_antithetic])
    
    # Calculate payoffs
    if option_type.lower() == "call":
        payoffs = np.maximum(ST - K, 0)
    else:
        payoffs = np.maximum(K - ST, 0)
    
    # Discount and estimate
    discount = np.exp(-r * T)
    discounted_payoffs = discount * payoffs
    
    price = np.mean(discounted_payoffs)
    std_error = np.std(discounted_payoffs, ddof=1) / np.sqrt(len(ST))
    
    return price, std_error


def monte_carlo_control_variate(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = "call",
    n_paths: int = 10000,
    seed: Optional[int] = None
) -> Tuple[float, float]:
    """
    Monte Carlo with Control Variate (variance reduction).
    
    Technique:
    ----------
    Use the stock price (known expected value) as a control.
    
    Adjusted estimator:
    V̂_cv = V̂ - β(S̄_T - E[S_T])
    
    Where:
    - S̄_T = sample average of terminal prices
    - E[S_T] = S0 × e^(rT) (known analytically)
    - β = Cov(payoff, S_T) / Var(S_T) (optimal coefficient)
    
    Variance Reduction:
    -------------------
    Reduction factor: 1 - ρ², where ρ = correlation(payoff, S_T)
    For ATM calls: ρ ≈ 0.9 → 81% variance reduction!
    """
    if seed is not None:
        np.random.seed(seed)
    
    ST = simulate_gbm_terminal(S0, r, sigma, T, n_paths, seed)
    
    # Calculate payoffs
    if option_type.lower() == "call":
        payoffs = np.maximum(ST - K, 0)
    else:
        payoffs = np.maximum(K - ST, 0)
    
    # Control variate: stock price
    # E[S_T] under risk-neutral measure
    expected_ST = S0 * np.exp(r * T)
    
    # Optimal beta coefficient
    cov_matrix = np.cov(payoffs, ST)
    beta = cov_matrix[0, 1] / cov_matrix[1, 1]
    
    # Adjusted payoffs
    adjusted_payoffs = payoffs - beta * (ST - expected_ST)
    
    # Discount and estimate
    discount = np.exp(-r * T)
    discounted_payoffs = discount * adjusted_payoffs
    
    price = np.mean(discounted_payoffs)
    std_error = np.std(discounted_payoffs, ddof=1) / np.sqrt(n_paths)
    
    return price, std_error


# =============================================================================
# PATH-DEPENDENT OPTIONS
# =============================================================================

def monte_carlo_asian(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = "call",
    averaging: str = "arithmetic",
    n_paths: int = 10000,
    n_steps: int = 252,
    seed: Optional[int] = None
) -> Tuple[float, float]:
    """
    Price an Asian option (average price option).
    
    Payoff:
    -------
    - Arithmetic Asian call: max(Ā - K, 0), where Ā = (1/n)Σ S(t_i)
    - Geometric Asian call: max(Ḡ - K, 0), where Ḡ = (Π S(t_i))^(1/n)
    
    Why Asian Options?
    ------------------
    - Reduces manipulation risk (single price vs average)
    - Lower premium than vanilla options (averaging reduces volatility)
    - Common for commodities, currencies
    
    Note: Geometric Asian has closed-form solution (for validation)
    """
    paths = simulate_gbm_paths(S0, r, sigma, T, n_paths, n_steps, seed)
    
    if averaging.lower() == "arithmetic":
        average_prices = np.mean(paths, axis=1)
    else:  # geometric
        average_prices = np.exp(np.mean(np.log(paths), axis=1))
    
    if option_type.lower() == "call":
        payoffs = np.maximum(average_prices - K, 0)
    else:
        payoffs = np.maximum(K - average_prices, 0)
    
    discount = np.exp(-r * T)
    discounted_payoffs = discount * payoffs
    
    price = np.mean(discounted_payoffs)
    std_error = np.std(discounted_payoffs, ddof=1) / np.sqrt(n_paths)
    
    return price, std_error


def monte_carlo_barrier(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    barrier: float,
    barrier_type: str = "down-and-out",
    option_type: str = "call",
    n_paths: int = 10000,
    n_steps: int = 252,
    seed: Optional[int] = None
) -> Tuple[float, float]:
    """
    Price a barrier option.
    
    Types:
    ------
    - Down-and-out: Expires worthless if price touches barrier from above
    - Down-and-in: Activates only if price touches barrier from above
    - Up-and-out: Expires worthless if price touches barrier from below
    - Up-and-in: Activates only if price touches barrier from below
    
    Relationships:
    - Down-and-out + Down-and-in = Vanilla option (no barrier)
    - Up-and-out + Up-and-in = Vanilla option
    """
    paths = simulate_gbm_paths(S0, r, sigma, T, n_paths, n_steps, seed)
    
    # Check barrier hits
    if barrier_type.startswith("down"):
        hit_barrier = np.any(paths <= barrier, axis=1)
    else:  # up
        hit_barrier = np.any(paths >= barrier, axis=1)
    
    # Determine which paths pay off
    if barrier_type.endswith("out"):
        active = ~hit_barrier  # Out options die when barrier is hit
    else:  # in
        active = hit_barrier   # In options activate when barrier is hit
    
    # Terminal prices
    ST = paths[:, -1]
    
    # Calculate payoffs (only for active paths)
    if option_type.lower() == "call":
        payoffs = np.where(active, np.maximum(ST - K, 0), 0)
    else:
        payoffs = np.where(active, np.maximum(K - ST, 0), 0)
    
    discount = np.exp(-r * T)
    discounted_payoffs = discount * payoffs
    
    price = np.mean(discounted_payoffs)
    std_error = np.std(discounted_payoffs, ddof=1) / np.sqrt(n_paths)
    
    return price, std_error


def monte_carlo_lookback(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = "call",
    strike_type: str = "floating",
    K: Optional[float] = None,
    n_paths: int = 10000,
    n_steps: int = 252,
    seed: Optional[int] = None
) -> Tuple[float, float]:
    """
    Price a lookback option.
    
    Floating Strike Lookback:
    -------------------------
    - Call: S(T) - min(S(t)) for t ∈ [0,T]
    - Put: max(S(t)) - S(T) for t ∈ [0,T]
    
    Fixed Strike Lookback:
    ----------------------
    - Call: max(max(S(t)) - K, 0)
    - Put: max(K - min(S(t)), 0)
    
    Characteristics:
    ----------------
    - Most expensive options (guaranteed "perfect timing")
    - Useful for: Bonus structures, executive compensation
    - Closed-form exists for continuous monitoring
    """
    paths = simulate_gbm_paths(S0, r, sigma, T, n_paths, n_steps, seed)
    
    ST = paths[:, -1]
    S_max = np.max(paths, axis=1)
    S_min = np.min(paths, axis=1)
    
    if strike_type == "floating":
        if option_type.lower() == "call":
            payoffs = ST - S_min  # Bought at the low!
        else:
            payoffs = S_max - ST  # Sold at the high!
    else:  # fixed strike
        if K is None:
            K = S0
        if option_type.lower() == "call":
            payoffs = np.maximum(S_max - K, 0)
        else:
            payoffs = np.maximum(K - S_min, 0)
    
    discount = np.exp(-r * T)
    discounted_payoffs = discount * payoffs
    
    price = np.mean(discounted_payoffs)
    std_error = np.std(discounted_payoffs, ddof=1) / np.sqrt(n_paths)
    
    return price, std_error


# =============================================================================
# GREEK CALCULATION VIA MONTE CARLO
# =============================================================================

def monte_carlo_delta(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = "call",
    n_paths: int = 10000,
    bump: float = 0.01
) -> float:
    """
    Calculate Delta using bump-and-reprice method.
    
    Method:
    -------
    Δ ≈ [V(S+ε) - V(S-ε)] / (2ε)
    
    This is central difference approximation (O(ε²) error).
    """
    # Use same random seed for both prices
    seed = np.random.randint(0, 2**31)
    
    epsilon = S0 * bump  # Bump as percentage of S0
    
    price_up, _ = monte_carlo_european(S0 + epsilon, K, r, sigma, T, option_type, n_paths, seed)
    price_down, _ = monte_carlo_european(S0 - epsilon, K, r, sigma, T, option_type, n_paths, seed)
    
    delta = (price_up - price_down) / (2 * epsilon)
    
    return delta


def monte_carlo_gamma(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = "call",
    n_paths: int = 10000,
    bump: float = 0.01
) -> float:
    """
    Calculate Gamma using bump-and-reprice method.
    
    Method:
    -------
    Γ ≈ [V(S+ε) - 2V(S) + V(S-ε)] / ε²
    """
    seed = np.random.randint(0, 2**31)
    
    epsilon = S0 * bump
    
    price_up, _ = monte_carlo_european(S0 + epsilon, K, r, sigma, T, option_type, n_paths, seed)
    price_mid, _ = monte_carlo_european(S0, K, r, sigma, T, option_type, n_paths, seed)
    price_down, _ = monte_carlo_european(S0 - epsilon, K, r, sigma, T, option_type, n_paths, seed)
    
    gamma = (price_up - 2 * price_mid + price_down) / (epsilon ** 2)
    
    return gamma


def monte_carlo_vega(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = "call",
    n_paths: int = 10000,
    bump: float = 0.01
) -> float:
    """
    Calculate Vega using bump-and-reprice method.
    
    Method:
    -------
    ν ≈ [V(σ+ε) - V(σ-ε)] / (2ε)
    
    Returns Vega per 1% volatility move.
    """
    seed = np.random.randint(0, 2**31)
    
    epsilon = bump  # Absolute bump for volatility
    
    price_up, _ = monte_carlo_european(S0, K, r, sigma + epsilon, T, option_type, n_paths, seed)
    price_down, _ = monte_carlo_european(S0, K, r, sigma - epsilon, T, option_type, n_paths, seed)
    
    # Vega per 1% vol move
    vega = (price_up - price_down) / (2 * epsilon) / 100
    
    return vega


# =============================================================================
# PERFORMANCE BENCHMARK
# =============================================================================

if __name__ == "__main__":
    import time
    from research.core.black_scholes import black_scholes_call
    
    # Parameters
    S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
    
    # Benchmark: 10,000 paths
    start = time.perf_counter()
    price_mc, se = monte_carlo_european(S0, K, r, sigma, T, "call", n_paths=10000)
    elapsed = (time.perf_counter() - start) * 1000
    
    # Compare with analytical Black-Scholes
    price_bs = black_scholes_call(S0, K, r, sigma, T)
    
    print("=" * 60)
    print("MONTE CARLO PERFORMANCE BENCHMARK")
    print("=" * 60)
    print(f"\nParameters: S={S0}, K={K}, r={r}, σ={sigma}, T={T}")
    print(f"\nMonte Carlo (10,000 paths):")
    print(f"  Price: {price_mc:.4f} ± {1.96*se:.4f} (95% CI)")
    print(f"  Time: {elapsed:.2f} ms")
    print(f"  Target: <10 ms ({'  PASS' if elapsed < 10 else '  FAIL'})")
    print(f"\nBlack-Scholes (analytical):")
    print(f"  Price: {price_bs:.4f}")
    print(f"\nError: {abs(price_mc - price_bs):.4f} ({abs(price_mc - price_bs)/price_bs*100:.2f}%)")
    
    # Variance reduction comparison
    print("\n" + "=" * 60)
    print("VARIANCE REDUCTION COMPARISON")
    print("=" * 60)
    
    _, se_standard = monte_carlo_european(S0, K, r, sigma, T, "call", 10000)
    _, se_antithetic = monte_carlo_antithetic(S0, K, r, sigma, T, "call", 10000)
    _, se_control = monte_carlo_control_variate(S0, K, r, sigma, T, "call", 10000)
    
    print(f"\nStandard MC SE:     {se_standard:.4f}")
    print(f"Antithetic MC SE:   {se_antithetic:.4f} ({(1-se_antithetic/se_standard)*100:.1f}% reduction)")
    print(f"Control Var MC SE:  {se_control:.4f} ({(1-se_control/se_standard)*100:.1f}% reduction)")
