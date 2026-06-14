"""
GIGA SYSTEM - Implied Volatility Solver
=======================================

Implied Volatility (IV) is the market's expectation of future volatility,
backed out from observed option prices using the Black-Scholes formula.

What is Implied Volatility?
---------------------------
- Given: Option price, S, K, r, T
- Find: σ such that BS(S, K, r, σ, T) = Market Price

This is an inverse problem - no closed-form solution exists.
We use numerical root-finding algorithms.

Why IV Matters:
---------------
1. Trading Signal: IV vs Realized Vol → Volatility arbitrage
2. Risk Assessment: High IV = Market uncertainty
3. Relative Value: Compare IV across strikes (volatility smile)
4. VIX Index: IV of S&P 500 options ("Fear Gauge")

Performance Target: <0.1ms per IV calculation
"""

import numpy as np
from numba import jit, float64
from scipy.optimize import brentq, newton
from typing import Optional, Tuple
import math

from research.core.black_scholes import black_scholes_call, black_scholes_put
from research.core.greeks import vega


# =============================================================================
# NEWTON-RAPHSON METHOD (Fastest)
# =============================================================================

@jit(nopython=True, cache=True)
def _fast_norm_cdf(x: float) -> float:
    """Fast normal CDF approximation."""
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = 1.0 if x >= 0 else -1.0
    x = abs(x)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x / 2.0)
    return 0.5 * (1.0 + sign * y)


@jit(nopython=True, cache=True)
def _fast_norm_pdf(x: float) -> float:
    """Fast normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


@jit(nopython=True, cache=True)
def _bs_price_and_vega(
    S: float, K: float, r: float, sigma: float, T: float, is_call: bool
) -> Tuple[float, float]:
    """
    Calculate BS price and Vega in one pass (efficient for Newton-Raphson).
    
    Returns:
    --------
    Tuple of (price, vega)
    """
    if sigma <= 0 or T <= 0:
        # Edge case: no time value
        if is_call:
            return max(S - K, 0.0), 0.0
        return max(K - S, 0.0), 0.0
    
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    
    n_d1 = _fast_norm_pdf(d1)
    N_d1 = _fast_norm_cdf(d1)
    N_d2 = _fast_norm_cdf(d2)
    
    if is_call:
        price = S * N_d1 - K * math.exp(-r * T) * N_d2
    else:
        price = K * math.exp(-r * T) * _fast_norm_cdf(-d2) - S * _fast_norm_cdf(-d1)
    
    # Vega (same for calls and puts)
    vega_val = S * sqrt_T * n_d1
    
    return price, vega_val


@jit(nopython=True, cache=True)
def implied_volatility_newton_jit(
    market_price: float,
    S: float,
    K: float,
    r: float,
    T: float,
    is_call: bool,
    sigma_init: float = 0.3,
    tol: float = 1e-8,
    max_iter: int = 100
) -> float:
    """
    Calculate implied volatility using Newton-Raphson method (JIT-compiled).
    
    Algorithm:
    ----------
    Newton-Raphson finds root of f(σ) = BS(σ) - Market_Price
    
    σ_{n+1} = σ_n - f(σ_n) / f'(σ_n)
            = σ_n - (BS(σ_n) - Market_Price) / Vega(σ_n)
    
    Why Newton-Raphson?
    -------------------
    - Quadratic convergence: Each iteration doubles correct digits
    - Typically converges in 3-5 iterations
    - Requires derivative (Vega), which we have analytically
    
    Convergence Conditions:
    -----------------------
    - Good initial guess (σ_init ≈ 0.2-0.4 usually works)
    - Vega ≠ 0 (fails for very deep ITM/OTM or near expiration)
    
    Parameters:
    -----------
    market_price : float - Observed option price
    S, K, r, T : float - Standard BS parameters
    is_call : bool - True for call, False for put
    sigma_init : float - Initial volatility guess
    tol : float - Convergence tolerance
    max_iter : int - Maximum iterations
    
    Returns:
    --------
    float : Implied volatility (annualized)
    
    Returns NaN if:
    - No solution (price outside arbitrage bounds)
    - Convergence fails
    """
    sigma = sigma_init
    
    for i in range(max_iter):
        price, vega_val = _bs_price_and_vega(S, K, r, sigma, T, is_call)
        
        # Error
        diff = price - market_price
        
        # Check convergence
        if abs(diff) < tol:
            return sigma
        
        # Check if Vega is too small (near-zero sensitivity)
        if abs(vega_val) < 1e-10:
            # Vega too small, try adjusting sigma
            if diff > 0:
                sigma *= 0.5  # Price too high, lower vol
            else:
                sigma *= 2.0  # Price too low, raise vol
            continue
        
        # Newton-Raphson update
        sigma_new = sigma - diff / vega_val
        
        # Ensure sigma stays positive
        if sigma_new <= 0:
            sigma_new = sigma / 2.0
        
        # Prevent extreme jumps
        if sigma_new > 5.0:
            sigma_new = 5.0
        
        sigma = sigma_new
    
    # Did not converge
    return math.nan


def implied_volatility_newton(
    market_price: float,
    S: float,
    K: float,
    r: float,
    T: float,
    option_type: str = "call",
    sigma_init: float = 0.3,
    tol: float = 1e-8,
    max_iter: int = 100
) -> float:
    """
    Python wrapper for Newton-Raphson IV solver.
    
    Example:
    --------
    >>> # ATM call priced at $10.45 (Black-Scholes with σ=0.20)
    >>> iv = implied_volatility_newton(10.4506, 100, 100, 0.05, 1.0, "call")
    >>> print(f"IV: {iv:.4f}")
    IV: 0.2000
    """
    is_call = option_type.lower() == "call"
    return implied_volatility_newton_jit(market_price, S, K, r, T, is_call, sigma_init, tol, max_iter)


# =============================================================================
# BISECTION METHOD (More Robust)
# =============================================================================

def implied_volatility_bisection(
    market_price: float,
    S: float,
    K: float,
    r: float,
    T: float,
    option_type: str = "call",
    sigma_low: float = 0.001,
    sigma_high: float = 5.0,
    tol: float = 1e-8,
    max_iter: int = 100
) -> float:
    """
    Calculate implied volatility using Bisection method.
    
    Algorithm:
    ----------
    Binary search on volatility:
    1. Start with interval [σ_low, σ_high]
    2. Calculate midpoint σ_mid = (σ_low + σ_high) / 2
    3. If BS(σ_mid) > target, σ_high = σ_mid
       Else σ_low = σ_mid
    4. Repeat until convergence
    
    Advantages over Newton-Raphson:
    -------------------------------
    - Guaranteed convergence if solution exists in interval
    - Doesn't require derivatives
    - More robust for edge cases
    
    Disadvantages:
    --------------
    - Slower (linear convergence vs quadratic)
    - Needs initial bracket [σ_low, σ_high]
    
    When to Use Bisection:
    ----------------------
    - Newton fails (Vega ≈ 0)
    - Need robustness over speed
    - Unsure of good initial guess
    """
    is_call = option_type.lower() == "call"
    
    # Check if solution exists (price within arbitrage bounds)
    if is_call:
        # Call: 0 < C < S, and C > max(S - K*e^(-rT), 0)
        lower_bound = max(S - K * np.exp(-r * T), 0)
        upper_bound = S
    else:
        # Put: 0 < P < K*e^(-rT), and P > max(K*e^(-rT) - S, 0)
        lower_bound = max(K * np.exp(-r * T) - S, 0)
        upper_bound = K * np.exp(-r * T)
    
    if market_price < lower_bound or market_price > upper_bound:
        return np.nan  # Price violates arbitrage bounds
    
    # Verify bracket contains root
    if is_call:
        price_low = black_scholes_call(S, K, r, sigma_low, T)
        price_high = black_scholes_call(S, K, r, sigma_high, T)
    else:
        price_low = black_scholes_put(S, K, r, sigma_low, T)
        price_high = black_scholes_put(S, K, r, sigma_high, T)
    
    if (price_low - market_price) * (price_high - market_price) > 0:
        # Root not bracketed, expand search
        sigma_low = 0.0001
        sigma_high = 10.0
    
    # Bisection loop
    for _ in range(max_iter):
        sigma_mid = (sigma_low + sigma_high) / 2.0
        
        if is_call:
            price_mid = black_scholes_call(S, K, r, sigma_mid, T)
        else:
            price_mid = black_scholes_put(S, K, r, sigma_mid, T)
        
        if abs(price_mid - market_price) < tol:
            return sigma_mid
        
        if price_mid > market_price:
            sigma_high = sigma_mid
        else:
            sigma_low = sigma_mid
        
        if sigma_high - sigma_low < tol:
            return sigma_mid
    
    # Return best estimate
    return (sigma_low + sigma_high) / 2.0


# =============================================================================
# BRENT'S METHOD (SciPy - Most Robust)
# =============================================================================

def implied_volatility_brent(
    market_price: float,
    S: float,
    K: float,
    r: float,
    T: float,
    option_type: str = "call",
    sigma_low: float = 0.001,
    sigma_high: float = 5.0
) -> float:
    """
    Calculate IV using Brent's method (scipy.optimize.brentq).
    
    Brent's Method:
    ---------------
    Combines bisection, secant, and inverse quadratic interpolation.
    - Superlinear convergence when function is smooth
    - Falls back to bisection when needed
    - Guaranteed to converge if root is bracketed
    
    Best for:
    ---------
    - Production code where robustness is critical
    - When you need the answer or an error (no silent failures)
    """
    is_call = option_type.lower() == "call"
    
    def objective(sigma):
        if is_call:
            return black_scholes_call(S, K, r, sigma, T) - market_price
        return black_scholes_put(S, K, r, sigma, T) - market_price
    
    try:
        iv = brentq(objective, sigma_low, sigma_high)
        return iv
    except ValueError:
        # Root not in interval
        return np.nan


# =============================================================================
# RATIONAL APPROXIMATION (Fastest, Approximate)
# =============================================================================

def implied_volatility_rational(
    market_price: float,
    S: float,
    K: float,
    r: float,
    T: float,
    option_type: str = "call"
) -> float:
    """
    Fast IV approximation using Brenner-Subrahmanyam formula.
    
    Formula (for ATM options):
    --------------------------
    σ ≈ √(2π/T) × (C/S)
    
    This is accurate for ATM options, less accurate for ITM/OTM.
    
    Use Case:
    ---------
    - Initial guess for Newton-Raphson
    - Quick estimate when precision isn't critical
    - Processing millions of options (speed > accuracy)
    
    Accuracy:
    ---------
    - ATM: ~1% error
    - 10% OTM: ~5% error
    - 20% OTM: ~10% error
    """
    is_call = option_type.lower() == "call"
    
    # Forward price
    F = S * np.exp(r * T)
    
    # Moneyness
    m = np.log(F / K)
    
    # Brenner-Subrahmanyam for ATM
    if is_call:
        # Adjust for ITM/OTM
        intrinsic = max(S - K * np.exp(-r * T), 0)
        time_value = market_price - intrinsic
    else:
        intrinsic = max(K * np.exp(-r * T) - S, 0)
        time_value = market_price - intrinsic
    
    if time_value <= 0:
        return 0.0
    
    # Approximation
    sigma_approx = np.sqrt(2 * np.pi / T) * (time_value / S)
    
    # Adjust for moneyness
    sigma_approx *= np.exp(0.5 * m ** 2)
    
    return sigma_approx


# =============================================================================
# VECTORIZED IV CALCULATION (For Options Chains)
# =============================================================================

def implied_volatility_vectorized(
    market_prices: np.ndarray,
    S: float,
    K: np.ndarray,
    r: float,
    T: np.ndarray,
    option_type: str = "call"
) -> np.ndarray:
    """
    Calculate IV for an entire options chain.
    
    Strategy:
    ---------
    1. Use rational approximation for initial guesses
    2. Refine with Newton-Raphson (vectorized where possible)
    3. Fall back to Brent's for failures
    
    Performance:
    ------------
    1000 options in <100ms
    """
    n = len(market_prices)
    ivs = np.zeros(n)
    
    for i in range(n):
        # Try Newton-Raphson first (fast)
        iv = implied_volatility_newton(
            market_prices[i], S, K[i], r, T[i], option_type
        )
        
        if np.isnan(iv):
            # Fall back to Brent's (robust)
            iv = implied_volatility_brent(
                market_prices[i], S, K[i], r, T[i], option_type
            )
        
        ivs[i] = iv
    
    return ivs


# =============================================================================
# IV SURFACE CONSTRUCTION
# =============================================================================

def construct_iv_surface(
    option_prices: np.ndarray,
    strikes: np.ndarray,
    expiries: np.ndarray,
    S: float,
    r: float,
    option_type: str = "call"
) -> np.ndarray:
    """
    Construct implied volatility surface from option prices.
    
    The IV Surface:
    ---------------
    A 2D function σ(K, T) showing how IV varies with:
    - Strike (volatility smile/skew)
    - Expiry (term structure)
    
    Market Features:
    ----------------
    1. Volatility Smile: IV higher for OTM options (both sides)
       - Caused by fat tails in return distribution
       - Jump risk (crashes)
       
    2. Volatility Skew: IV higher for OTM puts than OTM calls
       - Fear of downside (crash protection demand)
       - Typical for equity indices
       
    3. Term Structure: Short-term IV often differs from long-term
       - Mean reversion of volatility
       - Event risk (earnings, FOMC)
    
    Parameters:
    -----------
    option_prices : np.ndarray - Shape (n_strikes, n_expiries)
    strikes : np.ndarray - Array of strike prices
    expiries : np.ndarray - Array of expiration times (years)
    S : float - Current stock price
    r : float - Risk-free rate
    
    Returns:
    --------
    np.ndarray - IV surface, shape (n_strikes, n_expiries)
    """
    n_strikes = len(strikes)
    n_expiries = len(expiries)
    
    iv_surface = np.zeros((n_strikes, n_expiries))
    
    for i, K in enumerate(strikes):
        for j, T in enumerate(expiries):
            iv = implied_volatility_newton(
                option_prices[i, j], S, K, r, T, option_type
            )
            iv_surface[i, j] = iv
    
    return iv_surface


# =============================================================================
# PERFORMANCE BENCHMARK
# =============================================================================

if __name__ == "__main__":
    import time
    
    # Test parameters
    S, K, r, T = 100.0, 100.0, 0.05, 1.0
    true_sigma = 0.20
    
    # Generate "market" price from known volatility
    market_price = black_scholes_call(S, K, r, true_sigma, T)
    
    print("=" * 60)
    print("IMPLIED VOLATILITY SOLVER BENCHMARK")
    print("=" * 60)
    print(f"\nTrue volatility: {true_sigma:.4f}")
    print(f"Market price: {market_price:.4f}")
    
    # Benchmark different methods
    methods = [
        ("Newton-Raphson", lambda: implied_volatility_newton(market_price, S, K, r, T, "call")),
        ("Bisection", lambda: implied_volatility_bisection(market_price, S, K, r, T, "call")),
        ("Brent", lambda: implied_volatility_brent(market_price, S, K, r, T, "call")),
        ("Rational (approx)", lambda: implied_volatility_rational(market_price, S, K, r, T, "call")),
    ]
    
    print("\nMethod Comparison:")
    print("-" * 50)
    
    for name, func in methods:
        # Warm up (JIT compilation)
        _ = func()
        
        # Benchmark
        n_iterations = 10000
        start = time.perf_counter()
        for _ in range(n_iterations):
            iv = func()
        elapsed = (time.perf_counter() - start) / n_iterations * 1e6  # microseconds
        
        error = abs(iv - true_sigma) * 100 if not np.isnan(iv) else float('inf')
        print(f"{name:20s}: IV={iv:.6f}, Error={error:.4f}%, Time={elapsed:.2f}μs")
    
    print("\nTarget: <100 μs per calculation")
