"""
GIGA SYSTEM - Binomial Tree Option Pricing
==========================================

The Binomial model (Cox-Ross-Rubinstein, 1979) discretizes the continuous
price process into a tree of possible price movements.

Advantages over Black-Scholes:
------------------------------
1. Can price American options (early exercise)
2. Intuitive discrete-time framework
3. Converges to Black-Scholes as steps → ∞
4. Easy to incorporate dividends

Historical Context:
-------------------
- Developed by Cox, Ross, and Rubinstein (1979)
- Originally for pedagogical purposes (simpler than Black-Scholes PDE)
- Now widely used for American options, exotic options
- Foundation for many numerical methods in finance

Algorithm:
----------
1. Build tree forward (calculate possible prices at each time step)
2. Calculate payoffs at expiration (terminal nodes)
3. Work backward, taking max(exercise, hold) at each node
"""

import numpy as np
from numba import jit, prange
from typing import Tuple, Optional
import math


@jit(nopython=True, cache=True)
def _build_price_tree(S0: float, u: float, d: float, n_steps: int) -> np.ndarray:
    """
    Build the stock price tree.
    
    At each node (i, j):
    - i = time step (0 to n_steps)
    - j = number of up moves (0 to i)
    - S(i,j) = S0 * u^j * d^(i-j)
    
    Tree structure (3 steps):
                        S*u³
                  S*u²
            S*u        S*u²*d
       S         S*u*d
            S*d        S*u*d²
                  S*d²
                        S*d³
    """
    tree = np.zeros((n_steps + 1, n_steps + 1))
    
    for i in range(n_steps + 1):
        for j in range(i + 1):
            tree[j, i] = S0 * (u ** j) * (d ** (i - j))
    
    return tree


@jit(nopython=True, cache=True)
def _european_option_tree(
    price_tree: np.ndarray,
    K: float,
    r: float,
    dt: float,
    p: float,
    n_steps: int,
    is_call: bool
) -> float:
    """
    Price European option using backward induction.
    
    At terminal nodes: payoff = max(S-K, 0) for call or max(K-S, 0) for put
    At interior nodes: discounted expected value
        V(i,j) = e^(-r*dt) * [p*V(i+1,j+1) + (1-p)*V(i+1,j)]
    """
    # Option value tree
    option_tree = np.zeros((n_steps + 1, n_steps + 1))
    
    # Terminal payoffs
    for j in range(n_steps + 1):
        if is_call:
            option_tree[j, n_steps] = max(price_tree[j, n_steps] - K, 0.0)
        else:
            option_tree[j, n_steps] = max(K - price_tree[j, n_steps], 0.0)
    
    # Backward induction
    discount = math.exp(-r * dt)
    
    for i in range(n_steps - 1, -1, -1):
        for j in range(i + 1):
            option_tree[j, i] = discount * (
                p * option_tree[j + 1, i + 1] + 
                (1 - p) * option_tree[j, i + 1]
            )
    
    return option_tree[0, 0]


@jit(nopython=True, cache=True)
def _american_option_tree(
    price_tree: np.ndarray,
    K: float,
    r: float,
    dt: float,
    p: float,
    n_steps: int,
    is_call: bool
) -> float:
    """
    Price American option with early exercise.
    
    At each node, compare:
    - Continue value: discounted expected value
    - Exercise value: intrinsic value (S-K or K-S)
    
    V(i,j) = max(intrinsic, e^(-r*dt) * [p*V(i+1,j+1) + (1-p)*V(i+1,j)])
    
    Why American Options Matter:
    ----------------------------
    - American calls on non-dividend stocks = European calls (never early exercise)
    - American puts can be worth more (early exercise when deep ITM)
    - American calls on dividend stocks may early exercise before ex-div
    """
    option_tree = np.zeros((n_steps + 1, n_steps + 1))
    
    # Terminal payoffs
    for j in range(n_steps + 1):
        if is_call:
            option_tree[j, n_steps] = max(price_tree[j, n_steps] - K, 0.0)
        else:
            option_tree[j, n_steps] = max(K - price_tree[j, n_steps], 0.0)
    
    discount = math.exp(-r * dt)
    
    # Backward induction with early exercise check
    for i in range(n_steps - 1, -1, -1):
        for j in range(i + 1):
            # Continuation value
            hold = discount * (
                p * option_tree[j + 1, i + 1] + 
                (1 - p) * option_tree[j, i + 1]
            )
            
            # Exercise value
            if is_call:
                exercise = max(price_tree[j, i] - K, 0.0)
            else:
                exercise = max(K - price_tree[j, i], 0.0)
            
            # American option: take max
            option_tree[j, i] = max(hold, exercise)
    
    return option_tree[0, 0]


def binomial_european(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = "call",
    n_steps: int = 100
) -> float:
    """
    Price a European option using binomial tree.
    
    Cox-Ross-Rubinstein Parameters:
    -------------------------------
    dt = T / n_steps
    u = exp(σ√dt)           # Up factor
    d = 1/u = exp(-σ√dt)    # Down factor
    p = (exp(r*dt) - d) / (u - d)  # Risk-neutral probability
    
    As n_steps → ∞:
    - Tree converges to GBM
    - Price converges to Black-Scholes
    
    Parameters:
    -----------
    S0 : float - Stock price
    K : float - Strike price
    r : float - Risk-free rate
    sigma : float - Volatility
    T : float - Time to expiration
    option_type : str - "call" or "put"
    n_steps : int - Number of tree steps (more = accurate, but slower)
    
    Returns:
    --------
    float : Option price
    
    Example:
    --------
    >>> binomial_european(100, 100, 0.05, 0.20, 1.0, "call", n_steps=500)
    10.4506  # Matches Black-Scholes to 4 decimal places
    """
    dt = T / n_steps
    
    # CRR parameters
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    
    # Risk-neutral probability
    p = (np.exp(r * dt) - d) / (u - d)
    
    # Build price tree
    price_tree = _build_price_tree(S0, u, d, n_steps)
    
    # Calculate option price
    is_call = option_type.lower() == "call"
    price = _european_option_tree(price_tree, K, r, dt, p, n_steps, is_call)
    
    return price


def binomial_american(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = "call",
    n_steps: int = 100
) -> float:
    """
    Price an American option using binomial tree.
    
    American Option Properties:
    ---------------------------
    - Can be exercised any time before expiration
    - American option >= European option (always)
    - American call on non-dividend stock = European call
    - American put > European put (early exercise premium)
    
    Early Exercise Decision:
    ------------------------
    Exercise when: Intrinsic value > Continuation value
    
    For puts: Exercise when stock is very low
        - You get K now
        - Time value of money: K today > K × e^(-rT) at expiration
        - Plus: No risk of stock rebounding
    
    For calls with dividends: Exercise just before ex-dividend
        - Capture dividend by owning stock
        - Especially when ITM and near ex-div date
    """
    dt = T / n_steps
    
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    
    price_tree = _build_price_tree(S0, u, d, n_steps)
    
    is_call = option_type.lower() == "call"
    price = _american_option_tree(price_tree, K, r, dt, p, n_steps, is_call)
    
    return price


def binomial_american_dividend(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    dividend_yield: float,
    option_type: str = "call",
    n_steps: int = 100
) -> float:
    """
    Price American option on dividend-paying stock.
    
    Continuous Dividend Yield Model:
    --------------------------------
    Stock grows at rate (r - q) in risk-neutral world, where q = dividend yield.
    
    Modified parameters:
    u = exp(σ√dt)
    d = 1/u
    p = (exp((r-q)*dt) - d) / (u - d)
    
    Why Dividends Matter:
    ---------------------
    - Dividends reduce stock price on ex-date
    - Call holders don't receive dividends
    - May be optimal to exercise call just before ex-dividend
    """
    dt = T / n_steps
    q = dividend_yield
    
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    
    # Modified risk-neutral probability
    p = (np.exp((r - q) * dt) - d) / (u - d)
    
    # Stock prices adjusted for dividends
    price_tree = np.zeros((n_steps + 1, n_steps + 1))
    for i in range(n_steps + 1):
        for j in range(i + 1):
            # Price adjusted for dividend yield
            price_tree[j, i] = S0 * np.exp(-q * i * dt) * (u ** j) * (d ** (i - j))
    
    is_call = option_type.lower() == "call"
    price = _american_option_tree(price_tree, K, r, dt, p, n_steps, is_call)
    
    return price


def early_exercise_boundary(
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int = 100,
    option_type: str = "put",
    S0: float = None
) -> np.ndarray:
    """
    Calculate the early exercise boundary for American options.
    
    Uses a single-tree O(n²) algorithm instead of rebuilding subtrees at every
    node. Builds one full binomial tree, performs backward induction, and
    extracts the exercise boundary at each time step.
    
    The boundary S*(t) is the critical stock price where:
    - For puts: Exercise if S < S*(t)  (highest S where exercise is optimal)
    - For calls (with dividends): Exercise if S > S*(t)  (lowest S where exercise is optimal)
    
    Parameters
    ----------
    K : float       Strike price
    r : float       Risk-free rate
    sigma : float   Volatility
    T : float       Time to expiry (years)
    n_steps : int   Number of tree steps
    option_type : str  "put" or "call"
    S0 : float      Initial stock price (defaults to K if not provided)
    
    Returns
    -------
    np.ndarray
        Array of shape (M, 2) with (time, critical_price) pairs
    """
    if S0 is None:
        S0 = K
    
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    discount = np.exp(-r * dt)
    p = (np.exp(r * dt) - d) / (u - d)
    is_call = option_type.lower() == "call"
    
    # --- Build full price tree (column-major: tree[j, i]) ---
    price_tree = np.zeros((n_steps + 1, n_steps + 1))
    for i in range(n_steps + 1):
        for j in range(i + 1):
            price_tree[j, i] = S0 * (u ** j) * (d ** (i - j))
    
    # --- Terminal payoffs ---
    option_tree = np.zeros((n_steps + 1, n_steps + 1))
    for j in range(n_steps + 1):
        S = price_tree[j, n_steps]
        if is_call:
            option_tree[j, n_steps] = max(S - K, 0.0)
        else:
            option_tree[j, n_steps] = max(K - S, 0.0)
    
    # --- Backward induction + boundary extraction ---
    boundary = []
    
    for i in range(n_steps - 1, -1, -1):
        critical_price = None
        
        for j in range(i + 1):
            S = price_tree[j, i]
            hold = discount * (p * option_tree[j + 1, i + 1] + (1 - p) * option_tree[j, i + 1])
            
            if is_call:
                exercise = max(S - K, 0.0)
            else:
                exercise = max(K - S, 0.0)
            
            option_tree[j, i] = max(hold, exercise)
            
            # Detect boundary: node where early exercise is optimal
            if exercise > 0 and exercise >= hold:
                # For puts: boundary is the highest S where exercise wins.
                # Nodes are ordered j=0 (lowest S) to j=i (highest S), so
                # keep updating — the last exercised node is the boundary.
                # For calls: boundary is the lowest S where exercise wins.
                if is_call:
                    if critical_price is None:
                        critical_price = S
                else:
                    critical_price = S  # keep overwriting → highest S
        
        if critical_price is not None:
            t = i * dt
            boundary.append((t, critical_price))
    
    # Reverse so time goes forward (we iterated backward)
    boundary.reverse()
    
    if len(boundary) == 0:
        return np.empty((0, 2))
    
    return np.array(boundary)


# =============================================================================
# TRINOMIAL TREE (More Accurate)
# =============================================================================

def trinomial_european(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = "call",
    n_steps: int = 100
) -> float:
    """
    Price European option using trinomial tree.
    
    Trinomial vs Binomial:
    ----------------------
    - Three possible moves: up, middle, down
    - More accurate for same number of steps
    - Better for barrier options (price can stay at barrier)
    
    Parameters:
    -----------
    u = exp(σ√(2dt))
    d = 1/u
    m = 1 (middle = no change)
    
    Probabilities:
    p_u = ((exp(r*dt/2) - exp(-σ√(dt/2))) / (exp(σ√(dt/2)) - exp(-σ√(dt/2))))²
    p_d = ((exp(σ√(dt/2)) - exp(r*dt/2)) / (exp(σ√(dt/2)) - exp(-σ√(dt/2))))²
    p_m = 1 - p_u - p_d
    """
    dt = T / n_steps
    
    # Trinomial parameters
    u = np.exp(sigma * np.sqrt(2 * dt))
    d = 1.0 / u
    m = 1.0  # middle
    
    # Calculate probabilities
    nu = r - 0.5 * sigma ** 2
    dx = sigma * np.sqrt(2 * dt)
    
    p_u = ((np.exp(nu * dt / 2) - np.exp(-dx / 2)) / 
           (np.exp(dx / 2) - np.exp(-dx / 2))) ** 2
    p_d = ((np.exp(dx / 2) - np.exp(nu * dt / 2)) / 
           (np.exp(dx / 2) - np.exp(-dx / 2))) ** 2
    p_m = 1.0 - p_u - p_d
    
    # Tree has 2*n_steps + 1 nodes at each level
    n_nodes = 2 * n_steps + 1
    
    # Build price tree (only need terminal prices)
    prices = np.zeros(n_nodes)
    for j in range(n_nodes):
        n_up = j - n_steps  # Can be negative (net down moves)
        prices[j] = S0 * (u ** max(n_up, 0)) * (d ** max(-n_up, 0))
    
    # Terminal payoffs
    is_call = option_type.lower() == "call"
    if is_call:
        values = np.maximum(prices - K, 0)
    else:
        values = np.maximum(K - prices, 0)
    
    # Backward induction
    discount = np.exp(-r * dt)
    
    for i in range(n_steps - 1, -1, -1):
        n_current = 2 * i + 1
        new_values = np.zeros(n_current)
        
        for j in range(n_current):
            # Value from up, middle, down moves
            new_values[j] = discount * (
                p_u * values[j + 2] + 
                p_m * values[j + 1] + 
                p_d * values[j]
            )
        
        values = new_values
    
    return values[0]


# =============================================================================
# GREEKS FROM TREE
# =============================================================================

def binomial_delta(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = "call",
    n_steps: int = 100
) -> float:
    """
    Calculate Delta from binomial tree.
    
    Method:
    -------
    Use first two nodes of the tree:
    Δ = (V_u - V_d) / (S_u - S_d)
    
    Where V_u, V_d are option values after one up/down move.
    """
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    
    S_u = S0 * u
    S_d = S0 * d
    
    # Price options starting from S_u and S_d
    V_u = binomial_european(S_u, K, r, sigma, T - dt, option_type, n_steps - 1)
    V_d = binomial_european(S_d, K, r, sigma, T - dt, option_type, n_steps - 1)
    
    delta = (V_u - V_d) / (S_u - S_d)
    
    return delta


def binomial_gamma(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = "call",
    n_steps: int = 100
) -> float:
    """
    Calculate Gamma from binomial tree.
    
    Method:
    -------
    Use second level of tree:
    Γ = [(V_uu - V_ud)/(S_uu - S_ud) - (V_ud - V_dd)/(S_ud - S_dd)] / [(S_uu - S_dd)/2]
    """
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    
    S_uu = S0 * u * u
    S_ud = S0 * u * d  # = S0
    S_dd = S0 * d * d
    
    T_remaining = T - 2 * dt
    steps_remaining = max(n_steps - 2, 1)
    
    V_uu = binomial_european(S_uu, K, r, sigma, T_remaining, option_type, steps_remaining)
    V_ud = binomial_european(S_ud, K, r, sigma, T_remaining, option_type, steps_remaining)
    V_dd = binomial_european(S_dd, K, r, sigma, T_remaining, option_type, steps_remaining)
    
    delta_up = (V_uu - V_ud) / (S_uu - S_ud)
    delta_down = (V_ud - V_dd) / (S_ud - S_dd)
    
    gamma = (delta_up - delta_down) / ((S_uu - S_dd) / 2)
    
    return gamma


# =============================================================================
# BENCHMARK
# =============================================================================

if __name__ == "__main__":
    import time
    from research.core.black_scholes import black_scholes_call, black_scholes_put
    
    # Parameters
    S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
    
    print("=" * 60)
    print("BINOMIAL TREE PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    # European options (compare with Black-Scholes)
    bs_call = black_scholes_call(S0, K, r, sigma, T)
    bs_put = black_scholes_put(S0, K, r, sigma, T)
    
    for n_steps in [50, 100, 200, 500]:
        start = time.perf_counter()
        tree_call = binomial_european(S0, K, r, sigma, T, "call", n_steps)
        tree_put = binomial_european(S0, K, r, sigma, T, "put", n_steps)
        elapsed = (time.perf_counter() - start) * 1000
        
        print(f"\n{n_steps} steps (Time: {elapsed:.2f} ms):")
        print(f"  Call: Tree={tree_call:.4f}, BS={bs_call:.4f}, Error={abs(tree_call-bs_call):.4f}")
        print(f"  Put:  Tree={tree_put:.4f}, BS={bs_put:.4f}, Error={abs(tree_put-bs_put):.4f}")
    
    # American vs European put
    print("\n" + "=" * 60)
    print("AMERICAN VS EUROPEAN PUT")
    print("=" * 60)
    
    euro_put = binomial_european(S0, K, r, sigma, T, "put", 200)
    amer_put = binomial_american(S0, K, r, sigma, T, "put", 200)
    
    print(f"\nEuropean Put: {euro_put:.4f}")
    print(f"American Put: {amer_put:.4f}")
    print(f"Early Exercise Premium: {amer_put - euro_put:.4f}")
