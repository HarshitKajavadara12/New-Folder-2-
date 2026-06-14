"""
GIGA SYSTEM - Core Package Initialization
Greek Intelligence for Global Analysis
"""

from .black_scholes import (
    black_scholes_call,
    black_scholes_put,
    black_scholes_price,
)
from .greeks import (
    delta,
    gamma,
    theta,
    vega,
    rho,
    calculate_all_greeks,
)
from .monte_carlo import (
    monte_carlo_price,
    simulate_gbm_paths,
)
from .implied_volatility import (
    implied_volatility_newton,
    implied_volatility_bisection,
)
from .risk_metrics import (
    value_at_risk,
    conditional_var,
    expected_shortfall,
)

__version__ = "1.0.0"
__author__ = "GIGA Contributors"

__all__ = [
    # Black-Scholes
    "black_scholes_call",
    "black_scholes_put",
    "black_scholes_price",
    # Greeks
    "delta",
    "gamma", 
    "theta",
    "vega",
    "rho",
    "calculate_all_greeks",
    # Monte Carlo
    "monte_carlo_price",
    "simulate_gbm_paths",
    # Implied Volatility
    "implied_volatility_newton",
    "implied_volatility_bisection",
    # Risk Metrics
    "value_at_risk",
    "conditional_var",
    "expected_shortfall",
]
