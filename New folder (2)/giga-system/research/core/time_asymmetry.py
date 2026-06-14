"""
DOMAIN 4: ERGODICITY & TIME (τ, ε)
Greek Concepts:
- τ (tau) → Time to Equilibrium (Relaxation Time)
- ε (epsilon) → Perturbation / Fluctuation Scale

Key Insight:
Markets are Non-Ergodic. The Ensemble Average (Expectation) != Time Average (Growth).
Alpha = Optimizing for Time Average (Geometric Growth) rather than Expectation.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class ErgodicityResult:
    ensemble_average: float
    time_average: float
    is_ergodic: bool
    kelly_fraction: float

class TimeAsymmetryAnalyzer:
    """
    Analyzes the disconnect between probabilistic expectation and realized growth.
    """
    
    def check_ergodicity(self, returns: pd.Series) -> ErgodicityResult:
        """
        Compare Ensemble Average (Arithmetic Mean) vs Time Average (Geometric Mean).
        """
        if len(returns) == 0:
            return ErgodicityResult(0, 0, True, 0)
            
        # 1. Ensemble Average (Arithmetic Mean) - What you expect in one bet
        ensemble_avg = returns.mean()
        
        # 2. Time Average (Geometric Mean) - What you actually get over time
        # Log-growth rate: E[ln(1+r)]
        time_avg = np.log1p(returns).mean()
        
        # 3. Non-Ergodicity Gap
        gap = abs(ensemble_avg - time_avg)
        is_ergodic = gap < 1e-6
        
        # 4. Optimal Leverage (Kelly Criterion approx)
        # f* = mu / sigma^2
        mu = returns.mean()
        var = returns.var()
        kelly = mu / var if var > 0 else 0.0
        
        return ErgodicityResult(
            ensemble_average=ensemble_avg,
            time_average=time_avg,
            is_ergodic=is_ergodic,
            kelly_fraction=kelly
        )
    
    def calculate_relaxation_time(self, price_series: pd.Series) -> float:
        """
        Calculate τ (Tau): Time to decay to 1/e of perturbation.
        """
        # Autocorrelation decay method
        prev_corr = 1.0
        for lag in range(1, 100):
            corr = price_series.autocorr(lag=lag)
            if corr <= 0.368: # 1/e
                return float(lag)
        return 100.0
