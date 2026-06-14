"""
DOMAIN 3: STOCHASTIC MOTION (μ, σ, κ)
Greek Concepts:
- μ (mu) → Drift (Trend Component)
- σ (sigma) → Diffusion (Noise Component)
- κ (kappa) → Mean Reversion Force (Elasticity)

Implementation:
Fits stochastic differential equations (SDEs) to market data.
Alpha = Exploiting parameter instability (e.g., κ rapidly increasing).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple

@dataclass
class StochasticParams:
    mu: float     # Drift
    sigma: float  # Volatility
    kappa: float  # Mean Reversion Speed
    theta: float  # Long-term Mean (for OU process)

class StochasticModeler:
    """
    Fits SDE parameters to time series data.
    """
    
    def fit_ornstein_uhlenbeck(self, prices: pd.Series, dt: float = 1/252) -> StochasticParams:
        """
        Fit Ornstein-Uhlenbeck Process: dX = κ(θ - X)dt + σdW
        Returns: κ, θ, σ
        """
        if len(prices) < 30:
            return StochasticParams(0, 0, 0, 0)
        
        # Regression: X_{t+1} = a + b*X_t + ε
        # Then: κ = -ln(b)/dt, θ = a/(1-b), σ = std(ε) * sqrt(-2ln(b) / (dt(1-b^2)))
        
        y = prices.values[1:]
        x = prices.values[:-1]
        
        # Linear Regression
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xx = np.sum(x*x)
        sum_xy = np.sum(x*y)
        
        b = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x ** 2)
        a = (sum_y - b * sum_x) / n
        
        epsilon = y - (a + b * x)
        sd_epsilon = np.std(epsilon)
        
        # Recover SDE parameters
        try:
            kappa = -np.log(b) / dt
            theta_mean = a / (1 - b)
            sigma = sd_epsilon * np.sqrt(-2 * np.log(b) / (dt * (1 - b**2)))
        except (ValueError, ZeroDivisionError, RuntimeWarning):
             # Fallback for non-mean-reverting series (b >= 1)
             kappa = 0
             theta_mean = prices.mean()
             sigma = prices.std()

        # Pure Drift (Geometric Brownian Motion approximation)
        returns = prices.pct_change().dropna()
        mu = returns.mean() / dt
        
        return StochasticParams(
            mu=mu,
            sigma=sigma,
            kappa=kappa,
            theta=theta_mean
        )
