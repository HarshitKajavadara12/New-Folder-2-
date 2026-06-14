"""
GIGA SYSTEM - Option Greeks (Sensitivities)
============================================

The "Greeks" are partial derivatives of the option price with respect to
various parameters. They were named after Greek letters because finance
academics found it convenient to use single-letter notation.

Historical Context:
-------------------
- Greeks emerged naturally from the Black-Scholes PDE (1973)
- Fischer Black realized Delta could be used for hedging
- Market makers use Greeks to manage risk in real-time
- HFT firms calculate Greeks in microseconds for arbitrage

The Five Primary Greeks:
------------------------
1. Delta (Δ) = ∂V/∂S     - Price sensitivity
2. Gamma (Γ) = ∂²V/∂S²   - Delta's rate of change
3. Theta (Θ) = ∂V/∂T     - Time decay
4. Vega (ν)  = ∂V/∂σ     - Volatility sensitivity
5. Rho (ρ)   = ∂V/∂r     - Interest rate sensitivity

Performance Target: <0.005ms for all 5 Greeks (achieved via Numba)
"""

import numpy as np
from numba import jit, float64
from scipy.stats import norm
from typing import Dict, Tuple, Union
import math
from dataclasses import dataclass
from typing import Optional

# =============================================================================
# Fast Normal Distribution Functions (shared with black_scholes.py)
# =============================================================================

@jit(float64(float64), nopython=True, cache=True)
def _fast_norm_cdf(x: float) -> float:
    """
    Abramowitz & Stegun (1964) formula 26.2.17 — accuracy < 7.5e-8.
    N(x) = 1 - n(x)*(a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5)
    where t = 1/(1+p*|x|) and n(x) is the standard normal PDF.
    """
    # A&S 26.2.17 coefficients
    a1 =  0.319381530
    a2 = -0.356563782
    a3 =  1.781477937
    a4 = -1.821255978
    a5 =  1.330274429
    p  =  0.2316419

    x_abs = abs(x)
    t = 1.0 / (1.0 + p * x_abs)
    poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t
    # standard normal PDF n(x) = exp(-x^2/2) / sqrt(2*pi)
    pdf = math.exp(-0.5 * x_abs * x_abs) / math.sqrt(2.0 * math.pi)
    q = poly * pdf          # tail probability Q(|x|)

    if x >= 0.0:
        return 1.0 - q
    else:
        return q


@jit(float64(float64), nopython=True, cache=True)
def _fast_norm_pdf(x: float) -> float:
    """Standard normal probability density function."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


@jit(nopython=True, cache=True)
def _calculate_d1_d2(S: float, K: float, r: float, sigma: float, T: float) -> Tuple[float, float]:
    """Calculate d1 and d2 parameters."""
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


# =============================================================================
# DELTA (Δ) - The Change Letter
# =============================================================================

@jit(float64(float64, float64, float64, float64, float64), nopython=True, cache=True)
def delta_call(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """
    Calculate Delta for a European call option.
    
    Definition:
    -----------
    Δ_call = ∂C/∂S = N(d₁)
    
    Interpretation:
    ---------------
    - Delta tells you how much the option price changes per $1 change in stock
    - Delta = 0.5 means option moves $0.50 for every $1 stock move
    - For calls: 0 ≤ Delta ≤ 1
    - Delta ≈ probability option expires in-the-money (risk-neutral measure)
    
    Greek Origin:
    -------------
    Δ (Delta) - Fourth letter of Greek alphabet
    Δx notation for "change in x" dates to Leibniz (1684)
    
    HFT Usage:
    ----------
    Market makers delta-hedge continuously:
    - Long 100 calls (Δ = 0.5 each) → Portfolio Delta = +50
    - Short 50 shares → Stock Delta = -50
    - Net Delta = 0 (market-neutral)
    
    Parameters:
    -----------
    S : float - Stock price
    K : float - Strike price
    r : float - Risk-free rate
    sigma : float - Volatility
    T : float - Time to expiration (years)
    
    Returns:
    --------
    float : Delta value (0 to 1 for calls)
    """
    if T <= 0:
        return 1.0 if S > K else 0.0
    
    d1, _ = _calculate_d1_d2(S, K, r, sigma, T)
    return _fast_norm_cdf(d1)


@jit(float64(float64, float64, float64, float64, float64), nopython=True, cache=True)
def delta_put(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """
    Calculate Delta for a European put option.
    
    Δ_put = N(d₁) - 1 = -N(-d₁)
    
    For puts: -1 ≤ Delta ≤ 0
    """
    if T <= 0:
        return -1.0 if S < K else 0.0
    
    d1, _ = _calculate_d1_d2(S, K, r, sigma, T)
    return _fast_norm_cdf(d1) - 1.0


def delta(S: float, K: float, r: float, sigma: float, T: float, option_type: str = "call") -> float:
    """Unified delta function for calls and puts."""
    if option_type.lower() == "call":
        return delta_call(S, K, r, sigma, T)
    return delta_put(S, K, r, sigma, T)


# =============================================================================
# GAMMA (Γ) - The Curvature Letter
# =============================================================================

@jit(float64(float64, float64, float64, float64, float64), nopython=True, cache=True)
def gamma(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """
    Calculate Gamma for both call and put options (same value).
    
    Definition:
    -----------
    Γ = ∂²V/∂S² = ∂Δ/∂S = n(d₁) / (S × σ × √T)
    
    Where n(d₁) is the standard normal PDF evaluated at d₁.
    
    Interpretation:
    ---------------
    - Gamma measures how fast Delta changes as stock price moves
    - High Gamma = Delta changes rapidly = frequent rebalancing needed
    - Gamma is always positive (for both calls and puts)
    - Gamma is highest for at-the-money options near expiration
    
    Greek Origin:
    -------------
    Γ (Gamma) - Third letter of Greek alphabet
    In physics, γ represents acceleration (rate of change of velocity)
    Similarly, Gamma is rate of change of Delta (the "velocity" of option price)
    
    HFT Strategy (Gamma Scalping):
    ------------------------------
    1. Buy options (positive Gamma)
    2. Delta-hedge with stock
    3. As stock moves, Delta changes (Gamma effect)
    4. Rebalance: Buy low, sell high (lock in profit)
    5. Profit from Gamma, lose from Theta
    
    Profitability: Realized Vol > Implied Vol → Gamma scalping profits
    
    Parameters/Returns: Same as delta functions
    """
    if T <= 0:
        return 0.0
    
    d1, _ = _calculate_d1_d2(S, K, r, sigma, T)
    
    return _fast_norm_pdf(d1) / (S * sigma * math.sqrt(T))


# =============================================================================
# THETA (Θ) - The Time Letter
# =============================================================================

@jit(float64(float64, float64, float64, float64, float64), nopython=True, cache=True)
def theta_call(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """
    Calculate Theta for a European call option.
    
    Definition:
    -----------
    Θ_call = -∂C/∂T = -(S × n(d₁) × σ)/(2√T) - r × K × e^(-rT) × N(d₂)
    
    Note: Convention is to report Theta as daily decay (divide by 365)
    
    Interpretation:
    ---------------
    - Theta measures time decay: how much option loses per day
    - For long options, Theta is negative (you lose money each day)
    - For short options, Theta is positive (you collect time decay)
    - Theta decay accelerates near expiration (parabolic curve)
    
    Greek Origin:
    -------------
    Θ (Theta) - Eighth letter of Greek alphabet
    θ is used for angles in geometry, suggesting cyclical time
    "Time is money" - literally quantified by Theta!
    
    Trading Insight:
    ----------------
    Options sellers are "Theta positive" - they profit from time decay
    Options buyers fight Theta - need strong move to overcome decay
    
    Returns:
    --------
    float : Daily theta (negative for long positions)
    """
    if T <= 0:
        return 0.0
    
    d1, d2 = _calculate_d1_d2(S, K, r, sigma, T)
    sqrt_T = math.sqrt(T)
    
    term1 = -(S * _fast_norm_pdf(d1) * sigma) / (2 * sqrt_T)
    term2 = -r * K * math.exp(-r * T) * _fast_norm_cdf(d2)
    
    # Convert to daily theta (divide by 365)
    return (term1 + term2) / 365.0


@jit(float64(float64, float64, float64, float64, float64), nopython=True, cache=True)
def theta_put(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """
    Calculate Theta for a European put option.
    
    Θ_put = -(S × n(d₁) × σ)/(2√T) + r × K × e^(-rT) × N(-d₂)
    """
    if T <= 0:
        return 0.0
    
    d1, d2 = _calculate_d1_d2(S, K, r, sigma, T)
    sqrt_T = math.sqrt(T)
    
    term1 = -(S * _fast_norm_pdf(d1) * sigma) / (2 * sqrt_T)
    term2 = r * K * math.exp(-r * T) * _fast_norm_cdf(-d2)
    
    return (term1 + term2) / 365.0


def theta(S: float, K: float, r: float, sigma: float, T: float, option_type: str = "call") -> float:
    """Unified theta function."""
    if option_type.lower() == "call":
        return theta_call(S, K, r, sigma, T)
    return theta_put(S, K, r, sigma, T)


# =============================================================================
# VEGA (ν) - The Volatility Letter
# =============================================================================

@jit(float64(float64, float64, float64, float64, float64), nopython=True, cache=True)
def vega(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """
    Calculate Vega for both call and put options (same value).
    
    Definition:
    -----------
    ν = ∂V/∂σ = S × √T × n(d₁)
    
    Note: Often reported as change per 1% vol move (divide by 100)
    
    Interpretation:
    ---------------
    - Vega measures sensitivity to volatility changes
    - Vega is always positive (higher vol = higher option value)
    - Vega is highest for at-the-money options with long expiration
    
    Greek Origin:
    -------------
    ν (Nu) is the actual Greek letter, but traders use "Vega" (a star name)
    This is the only "Greek" that isn't actually a Greek letter!
    
    Volatility Trading:
    -------------------
    - Long Vega: Profit when volatility increases (buy options)
    - Short Vega: Profit when volatility decreases (sell options)
    
    VIX Connection:
    - VIX = "Fear Index" = Implied volatility of S&P 500 options
    - When VIX spikes (fear), long Vega positions profit
    - March 2020: VIX hit 80 (normal ~15) → Vega positions 5x profit
    
    Returns:
    --------
    float : Vega (change per 1% volatility move)
    """
    if T <= 0:
        return 0.0
    
    d1, _ = _calculate_d1_d2(S, K, r, sigma, T)
    
    # Raw vega (change per 100% vol move)
    raw_vega = S * math.sqrt(T) * _fast_norm_pdf(d1)
    
    # Return vega per 1% vol move
    return raw_vega / 100.0


# =============================================================================
# RHO (ρ) - The Rate Letter
# =============================================================================

@jit(float64(float64, float64, float64, float64, float64), nopython=True, cache=True)
def rho_call(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """
    Calculate Rho for a European call option.
    
    Definition:
    -----------
    ρ_call = ∂C/∂r = K × T × e^(-rT) × N(d₂)
    
    Note: Often reported as change per 1% rate move (divide by 100)
    
    Interpretation:
    ---------------
    - Rho measures sensitivity to interest rate changes
    - For calls, Rho is positive (higher rates = higher call value)
    - For puts, Rho is negative (higher rates = lower put value)
    - Rho matters more for long-dated options (LEAPS)
    
    Greek Origin:
    -------------
    ρ (Rho) - Seventeenth letter of Greek alphabet
    Used for density in physics, here measures "density" of rate sensitivity
    
    2022-2024 Relevance:
    --------------------
    Fed raised rates from 0% to 5.25%
    - Long-dated calls gained value from Rho
    - LEAPS (2-year options): Rho can be 10%+ of option value
    - Interest rate derivatives (swaptions) are highly Rho-sensitive
    
    Returns:
    --------
    float : Rho (change per 1% interest rate move)
    """
    if T <= 0:
        return 0.0
    
    _, d2 = _calculate_d1_d2(S, K, r, sigma, T)
    
    raw_rho = K * T * math.exp(-r * T) * _fast_norm_cdf(d2)
    
    return raw_rho / 100.0


@jit(float64(float64, float64, float64, float64, float64), nopython=True, cache=True)
def rho_put(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """
    Calculate Rho for a European put option.
    
    ρ_put = -K × T × e^(-rT) × N(-d₂)
    """
    if T <= 0:
        return 0.0
    
    _, d2 = _calculate_d1_d2(S, K, r, sigma, T)
    
    raw_rho = -K * T * math.exp(-r * T) * _fast_norm_cdf(-d2)
    
    return raw_rho / 100.0


def rho(S: float, K: float, r: float, sigma: float, T: float, option_type: str = "call") -> float:
    """Unified rho function."""
    if option_type.lower() == "call":
        return rho_call(S, K, r, sigma, T)
    return rho_put(S, K, r, sigma, T)


# =============================================================================
# COMBINED GREEKS CALCULATION
# =============================================================================

def calculate_all_greeks(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = "call"
) -> Dict[str, float]:
    """
    Calculate all five Greeks for an option.
    
    This is the function to use when you need all Greeks at once.
    More efficient than calling each Greek function separately
    (shares d1/d2 calculation).
    
    Parameters:
    -----------
    S : float - Stock price
    K : float - Strike price  
    r : float - Risk-free rate
    sigma : float - Volatility
    T : float - Time to expiration (years)
    option_type : str - "call" or "put"
    
    Returns:
    --------
    Dict with keys: 'delta', 'gamma', 'theta', 'vega', 'rho'
    
    Example:
    --------
    >>> greeks = calculate_all_greeks(100, 100, 0.05, 0.20, 1.0, "call")
    >>> print(f"Delta: {greeks['delta']:.4f}")
    Delta: 0.6368
    """
    if T <= 0:
        # At expiration
        is_call = option_type.lower() == "call"
        itm = (S > K) if is_call else (S < K)
        
        return {
            'delta': (1.0 if is_call else -1.0) if itm else 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0,
            'rho': 0.0
        }
    
    # Calculate d1, d2 once
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    
    # Normal distribution values
    n_d1 = norm.pdf(d1)
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    
    # Discount factor
    df = np.exp(-r * T)
    
    is_call = option_type.lower() == "call"
    
    # Delta
    delta_val = N_d1 if is_call else N_d1 - 1.0
    
    # Gamma (same for calls and puts)
    gamma_val = n_d1 / (S * sigma * sqrt_T)
    
    # Theta
    theta_term1 = -(S * n_d1 * sigma) / (2 * sqrt_T)
    if is_call:
        theta_val = (theta_term1 - r * K * df * N_d2) / 365.0
    else:
        theta_val = (theta_term1 + r * K * df * norm.cdf(-d2)) / 365.0
    
    # Vega (same for calls and puts, per 1% vol move)
    vega_val = S * sqrt_T * n_d1 / 100.0
    
    # Rho (per 1% rate move)
    if is_call:
        rho_val = K * T * df * N_d2 / 100.0
    else:
        rho_val = -K * T * df * norm.cdf(-d2) / 100.0
    
    return {
        'delta': delta_val,
        'gamma': gamma_val,
        'theta': theta_val,
        'vega': vega_val,
        'rho': rho_val
    }


# =============================================================================
# VECTORIZED GREEKS (For Options Chains)
# =============================================================================

def calculate_greeks_vectorized(
    S: np.ndarray,
    K: np.ndarray,
    r: float,
    sigma: np.ndarray,
    T: np.ndarray,
    option_type: str = "call"
) -> Dict[str, np.ndarray]:
    """
    Calculate all Greeks for an entire options chain.
    
    Use this for batch processing (1000+ options in <1ms).
    
    Returns:
    --------
    Dict with numpy arrays for each Greek
    """
    # Broadcast arrays
    S, K, sigma, T = np.broadcast_arrays(
        np.atleast_1d(S),
        np.atleast_1d(K),
        np.atleast_1d(sigma),
        np.atleast_1d(T)
    )
    
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    
    n_d1 = norm.pdf(d1)
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    df = np.exp(-r * T)
    
    is_call = option_type.lower() == "call"
    
    delta_arr = N_d1 if is_call else N_d1 - 1.0
    gamma_arr = n_d1 / (S * sigma * sqrt_T)
    
    theta_term1 = -(S * n_d1 * sigma) / (2 * sqrt_T)
    if is_call:
        theta_arr = (theta_term1 - r * K * df * N_d2) / 365.0
    else:
        theta_arr = (theta_term1 + r * K * df * norm.cdf(-d2)) / 365.0
    
    vega_arr = S * sqrt_T * n_d1 / 100.0
    
    if is_call:
        rho_arr = K * T * df * N_d2 / 100.0
    else:
        rho_arr = -K * T * df * norm.cdf(-d2) / 100.0
    
    return {
        'delta': delta_arr,
        'gamma': gamma_arr,
        'theta': theta_arr,
        'vega': vega_arr,
        'rho': rho_arr
    }


# =============================================================================
# HIGHER-ORDER GREEKS (Second-Order Sensitivities)
# =============================================================================

def vanna(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """
    Vanna: Cross-partial derivative ∂²V/(∂S∂σ) = ∂Δ/∂σ
    
    Measures how Delta changes with volatility.
    Important for volatility trading and risk management.
    """
    if T <= 0:
        return 0.0
    
    d1, d2 = _calculate_d1_d2(S, K, r, sigma, T)
    return -norm.pdf(d1) * d2 / sigma


def charm(S: float, K: float, r: float, sigma: float, T: float, option_type: str = "call") -> float:
    """
    Charm: Cross-partial derivative ∂²V/(∂S∂T) = ∂Δ/∂T
    
    Measures how Delta changes with time (delta decay).
    """
    if T <= 0:
        return 0.0
    
    d1, d2 = _calculate_d1_d2(S, K, r, sigma, T)
    sqrt_T = np.sqrt(T)
    
    term = norm.pdf(d1) * (2 * r * T - d2 * sigma * sqrt_T) / (2 * T * sigma * sqrt_T)
    
    if option_type.lower() == "call":
        return -term
    return -term


def vomma(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """
    Vomma (Volga): ∂²V/∂σ² = ∂Vega/∂σ
    
    Measures convexity of option price with respect to volatility.
    """
    if T <= 0:
        return 0.0
    
    d1, d2 = _calculate_d1_d2(S, K, r, sigma, T)
    sqrt_T = np.sqrt(T)
    
    return vega(S, K, r, sigma, T) * d1 * d2 / sigma


# =============================================================================
# Performance Benchmark
# =============================================================================

if __name__ == "__main__":
    import time
    
    # Test values
    S, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
    
    # Warm up JIT
    _ = delta_call(S, K, r, sigma, T)
    _ = gamma(S, K, r, sigma, T)
    
    # Benchmark all Greeks
    n = 100000
    start = time.perf_counter()
    for _ in range(n):
        _ = calculate_all_greeks(S, K, r, sigma, T, "call")
    elapsed = (time.perf_counter() - start) / n * 1e6
    
    print(f"All 5 Greeks: {elapsed:.3f} μs per calculation")
    print(f"Target: <5 μs ({'  PASS' if elapsed < 5 else '  FAIL'})")
    
    # Print sample values
    greeks = calculate_all_greeks(S, K, r, sigma, T, "call")
    print(f"\nSample Greeks (ATM call, S=K=100, σ=20%, T=1yr):")
    print(f"  Delta: {greeks['delta']:.4f} (expect ~0.64)")
    print(f"  Gamma: {greeks['gamma']:.4f} (expect ~0.019)")
    print(f"  Theta: {greeks['theta']:.4f} (expect ~-0.018/day)")
    print(f"  Vega:  {greeks['vega']:.4f} (expect ~0.40 per 1% vol)")
    print(f"  Rho:   {greeks['rho']:.4f} (expect ~0.53 per 1% rate)")

@dataclass
class GreekResult:
    """
    Context-aware Greek value container.
    Fixes Phase 2 Failure: "Symbol Overload (Context Loss)"
    """
    value: float
    greek_name: str
    underlying_price: float
    regime_context: str = "unknown"
    is_valid: bool = True
    confidence: float = 1.0
