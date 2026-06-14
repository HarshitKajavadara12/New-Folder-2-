"""
DOMAIN 2: VARIATIONAL SENSITIVITY (Δ, Γ, Θ)
Greek Concepts:
- Δ (Delta) → Sensitivity of PnL to Price Change (First Variation)
- Γ (Gamma) → Sensitivity of Delta to Price Change (Second Variation / Convexity)
- Θ (Theta) → Sensitivity of Signal Strength to Time (Decay Rate)

Implementation:
Calculates local derivatives of the PnL surface using genuine
mathematical methods inspired by Greek mathematics:

- Delta: Archimedean tangent line method (secant line → limit)
  NOT just correlation — computes actual ∂PnL/∂Price via finite differences
  
- Gamma: Apollonius curvature — second derivative via three-point formula
  NOT just correlation — computes actual ∂²PnL/∂Price² (convexity)
  
- Theta: Eudoxian exhaustion — measures rate of approach to zero
  NOT just autocorrelation — computes exponential decay rate via log-regression

- Vega: Pythagorean sensitivity — change in PnL per unit volatility change

Alpha Condition: High Γ (Convexity) + Slow Θ (Persistence).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class SensitivityProfile:
    delta: float  # ∂PnL / ∂Price
    gamma: float  # ∂²PnL / ∂Price²
    theta: float  # ∂Signal / ∂Time
    vega: float   # ∂PnL / ∂Vol

class VariationalAnalyzer:
    """
    Calculates the Greek response of a strategy or signal.
    
    Uses genuine mathematical methods:
    - Archimedean tangent lines for Delta (first derivative)
    - Apollonius curvature for Gamma (second derivative)
    - Eudoxian exhaustion for Theta (decay rate)
    - Pythagorean decomposition for Vega (volatility sensitivity)
    """
    
    @staticmethod
    def calculate_delta(pnl_series: pd.Series, price_series: pd.Series) -> float:
        """
        Calculate Delta (Δ) — Archimedean Tangent Line Method.
        
        Archimedes computed tangent lines by taking the limit of secant lines.
        Here we compute ∂PnL/∂Price as the OLS regression slope:
        
          Δ = Cov(ΔPnL, ΔPrice) / Var(ΔPrice)
        
        This IS the actual first derivative (not correlation),
        giving the dollar change in PnL per unit price change.
        """
        if len(pnl_series) < 10:
            return 0.0
        
        # Finite differences (secant lines)
        dpnl = pnl_series.diff().dropna()
        dprice = price_series.diff().dropna()
        
        # Align indices
        common_idx = dpnl.index.intersection(dprice.index)
        if len(common_idx) == 0:
            return 0.0
        
        dpnl = dpnl.loc[common_idx]
        dprice = dprice.loc[common_idx]
        
        # True derivative via OLS: Δ = Cov(ΔPnL, ΔPrice) / Var(ΔPrice)
        # (Not correlation — this gives actual slope in dollar terms)
        var_price = dprice.var()
        if var_price < 1e-15:
            return 0.0
        
        delta = dpnl.cov(dprice) / var_price
        return float(delta)

    @staticmethod
    def calculate_gamma(pnl_series: pd.Series, price_series: pd.Series) -> float:
        """
        Calculate Gamma (Γ) — Apollonius Curvature Method.
        
        Apollonius studied curvature of conic sections.
        Gamma is the second derivative ∂²PnL/∂Price², measuring convexity.
        
        Uses the three-point central difference formula:
          Γ ≈ (f(x+h) - 2f(x) + f(x-h)) / h²
        
        Computed via quadratic regression:
          PnL = a·Price² + b·Price + c → Γ = 2a
        """
        if len(pnl_series) < 10:
            return 0.0
        
        # Quadratic fit: PnL = a·Price² + b·Price + c
        # Gamma = 2a (coefficient of quadratic term = half the second derivative)
        try:
            # Normalize to avoid numerical issues  
            p_mean = price_series.mean()
            p_std = price_series.std()
            if p_std < 1e-15:
                return 0.0
            
            p_norm = (price_series.values - p_mean) / p_std
            coeffs = np.polyfit(p_norm, pnl_series.values, 2)
            
            # Second derivative in original units: d²PnL/dPrice² = 2a / std²
            gamma = 2 * coeffs[0] / (p_std ** 2)
            return float(gamma)
        except (np.linalg.LinAlgError, ValueError, TypeError):
            return 0.0

    @staticmethod
    def calculate_theta(signal_series: pd.Series) -> float:
        """
        Calculate Theta (Θ) — Eudoxian Exhaustion Decay Rate.
        
        Eudoxus's method of exhaustion measures how a quantity
        progressively approaches a limit. Here we measure the
        exponential decay rate of signal autocorrelation:
        
          ACF(lag) ≈ exp(-θ · lag)
          → θ = -slope of log(|ACF|) vs lag
        
        A slower decay (smaller |θ|) means the signal persists longer.
        """
        if len(signal_series) < 20:
            return 0.0
        
        # Compute autocorrelation at multiple lags
        max_lag = min(10, len(signal_series) // 3)
        lags = list(range(1, max_lag + 1))
        acfs = []
        
        for lag in lags:
            acf = signal_series.autocorr(lag=lag)
            if np.isnan(acf) or acf <= 0:
                break
            acfs.append(acf)
        
        if len(acfs) < 3:
            # Fallback: simple autocorrelation difference
            acf1 = signal_series.autocorr(lag=1)
            return float(acf1 - 1.0) if not np.isnan(acf1) else 0.0
        
        # Fit exponential decay: log(ACF) = -θ · lag + const
        log_acf = np.log(np.array(acfs))
        lag_arr = np.array(lags[:len(acfs)])
        
        try:
            slope, intercept = np.polyfit(lag_arr, log_acf, 1)
            # slope = -θ, so θ = -slope
            # Negative theta = signal decays, magnitude = speed
            return float(slope)  # Negative value = decay
        except (np.linalg.LinAlgError, ValueError):
            return 0.0

    @staticmethod
    def calculate_vega(pnl_series: pd.Series, price_series: pd.Series) -> float:
        """
        Calculate Vega (ν) — Pythagorean Volatility Sensitivity.
        
        Inspired by Pythagoras's decomposition of quantities into
        orthogonal components. Measures ∂PnL/∂σ by regressing
        PnL changes against changes in realized volatility.
        """
        if len(pnl_series) < 30:
            return 0.0
        
        # Compute rolling realized volatility
        returns = price_series.pct_change().dropna()
        if len(returns) < 20:
            return 0.0
        
        realized_vol = returns.rolling(10).std() * np.sqrt(252)
        
        # Changes in PnL and vol  
        dpnl = pnl_series.diff().dropna()
        dvol = realized_vol.diff().dropna()
        
        # Align
        common = dpnl.index.intersection(dvol.index)
        if len(common) < 10:
            return 0.0
        
        dpnl = dpnl.loc[common]
        dvol = dvol.loc[common]
        
        # Regression: ΔPnL = α + ν·Δσ + ε
        var_vol = dvol.var()
        if var_vol < 1e-15:
            return 0.0
        
        vega = dpnl.cov(dvol) / var_vol
        return float(vega)

    def analyze_convexity(self, pnl: pd.Series, price: pd.Series) -> SensitivityProfile:
        """
        Generate full Greek sensitivity profile.
        
        Uses genuine mathematical methods:
        - Archimedean secant → tangent for Delta
        - Apollonius curvature for Gamma
        - Eudoxian exhaustion for Theta
        - Pythagorean decomposition for Vega
        """
        delta = self.calculate_delta(pnl, price)
        gamma = self.calculate_gamma(pnl, price)
        theta = self.calculate_theta(pnl.pct_change().dropna() if len(pnl) > 1 else pnl)
        vega = self.calculate_vega(pnl, price)
        
        return SensitivityProfile(delta, gamma, theta, vega)
