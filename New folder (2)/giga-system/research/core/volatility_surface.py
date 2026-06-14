"""
VOLATILITY SURFACE — Full IV Surface Construction
===================================================

Addresses Missing Concepts 3.2, 3.3, 3.4:
  3.2 — Volatility Surface Construction (SVI parameterization)
  3.3 — Term Structure Analysis (contango / backwardation)
  3.4 — Skew Analysis (put-call skew, risk reversal)
"""

import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from scipy.optimize import minimize
from scipy.interpolate import RectBivariateSpline, griddata
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class VolSurfacePoint:
    """Single point on the IV surface."""
    strike: float
    expiry_years: float
    iv: float
    moneyness: float  # log(K/S)
    delta: float = 0.0
    option_type: str = "call"


@dataclass
class SVIParams:
    """
    SVI (Stochastic Volatility Inspired) parameterization.
    Gatheral (2004): w(k) = a + b * (ρ*(k-m) + sqrt((k-m)² + σ²))
    where k = log(K/F) is log-moneyness.
    """
    a: float       # Overall variance level
    b: float       # Curvature (slope of wings)
    rho: float     # Skew (-1 to 1)
    m: float       # Translation (ATM shift)
    sigma: float   # ATM curvature smoothness


@dataclass
class TermStructure:
    """Term structure of implied volatility."""
    expiries: List[float]
    atm_ivs: List[float]
    slope: float  # Term structure slope
    shape: str    # "contango", "backwardation", "flat", "humped"
    curvature: float


@dataclass
class SkewMetrics:
    """Volatility skew metrics for a single expiry."""
    expiry: float
    atm_iv: float
    put_25d_iv: float
    call_25d_iv: float
    skew_25d: float         # put_25d - call_25d
    risk_reversal: float    # call_25d - put_25d
    butterfly: float        # (put_25d + call_25d) / 2 - atm
    skew_ratio: float       # put_25d / call_25d


class VolatilitySurface:
    """
    Full implied volatility surface with SVI parameterization.
    Supports construction, interpolation, term structure, and skew analysis.
    """

    def __init__(self, underlying_price: float = 100.0, risk_free_rate: float = 0.05):
        self.S = underlying_price
        self.r = risk_free_rate
        self.surface_points: List[VolSurfacePoint] = []
        self.svi_params_by_expiry: Dict[float, SVIParams] = {}
        self._interpolator = None

    def add_point(self, strike: float, expiry: float, iv: float, option_type: str = "call"):
        """Add a single IV observation to the surface."""
        moneyness = np.log(strike / self.S)
        d1 = (np.log(self.S / strike) + (self.r + 0.5 * iv**2) * expiry) / (iv * np.sqrt(expiry) + 1e-15)
        delta = stats.norm.cdf(d1) if option_type == "call" else stats.norm.cdf(d1) - 1

        self.surface_points.append(VolSurfacePoint(
            strike=strike, expiry_years=expiry, iv=iv,
            moneyness=moneyness, delta=delta, option_type=option_type,
        ))

    def add_chain(self, strikes: np.ndarray, expiry: float, ivs: np.ndarray, option_type: str = "call"):
        """Add an entire options chain."""
        for k, iv in zip(strikes, ivs):
            if iv > 0.01:
                self.add_point(k, expiry, iv, option_type)

    def fit_svi(self, expiry: float) -> SVIParams:
        """
        Fit SVI parameterization to a single expiry slice.
        Gatheral (2004): w(k) = a + b*(ρ*(k-m) + √((k-m)² + σ²))
        """
        points = [p for p in self.surface_points if abs(p.expiry_years - expiry) < 0.01]
        if len(points) < 3:
            return SVIParams(a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.1)

        k_data = np.array([p.moneyness for p in points])
        w_data = np.array([p.iv**2 * p.expiry_years for p in points])  # Total variance

        def svi_total_var(params, k):
            a, b, rho, m, sig = params
            return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sig**2))

        def objective(params):
            pred = svi_total_var(params, k_data)
            return np.sum((pred - w_data)**2)

        # Initial guess and bounds
        x0 = [np.mean(w_data), 0.1, -0.3, 0.0, 0.1]
        bounds = [(0, None), (0, None), (-0.999, 0.999), (-1, 1), (0.001, 2)]

        try:
            res = minimize(objective, x0, bounds=bounds, method="L-BFGS-B")
            params = SVIParams(a=res.x[0], b=res.x[1], rho=res.x[2], m=res.x[3], sigma=res.x[4])
        except Exception:
            params = SVIParams(a=np.mean(w_data), b=0.1, rho=-0.3, m=0.0, sigma=0.1)

        self.svi_params_by_expiry[expiry] = params
        return params

    def get_iv(self, strike: float, expiry: float) -> float:
        """Interpolate IV at any (strike, expiry) point."""
        # Try SVI first
        if expiry in self.svi_params_by_expiry:
            p = self.svi_params_by_expiry[expiry]
            k = np.log(strike / self.S)
            w = p.a + p.b * (p.rho * (k - p.m) + np.sqrt((k - p.m)**2 + p.sigma**2))
            return np.sqrt(max(w / expiry, 0.0001))

        # Fallback: nearest-neighbor interpolation from raw points
        if not self.surface_points:
            return 0.20  # Default 20% IV

        distances = [(abs(p.strike - strike) + abs(p.expiry_years - expiry) * 100, p.iv)
                     for p in self.surface_points]
        distances.sort()
        return distances[0][1]

    def build_surface_grid(self, n_strikes: int = 20, n_expiries: int = 10) -> Dict:
        """Build a full NxM IV surface grid for visualization."""
        if not self.surface_points:
            return self._generate_synthetic_surface(n_strikes, n_expiries)

        expiries = sorted(set(p.expiry_years for p in self.surface_points))
        strikes = sorted(set(p.strike for p in self.surface_points))

        # Fit SVI for each expiry
        for exp in expiries:
            if exp not in self.svi_params_by_expiry:
                self.fit_svi(exp)

        # Build grid
        strike_grid = np.linspace(min(strikes) * 0.9, max(strikes) * 1.1, n_strikes)
        expiry_grid = np.linspace(min(expiries), max(expiries), n_expiries)

        iv_grid = np.zeros((n_expiries, n_strikes))
        for i, exp in enumerate(expiry_grid):
            # Find nearest fitted expiry
            nearest_exp = min(expiries, key=lambda e: abs(e - exp))
            for j, k in enumerate(strike_grid):
                iv_grid[i, j] = self.get_iv(k, nearest_exp)

        return {
            "strikes": strike_grid.tolist(),
            "expiries": expiry_grid.tolist(),
            "ivs": iv_grid.tolist(),
            "n_raw_points": len(self.surface_points),
        }

    def _generate_synthetic_surface(self, n_strikes: int, n_expiries: int) -> Dict:
        """Generate realistic synthetic vol surface for demo."""
        strikes = np.linspace(self.S * 0.7, self.S * 1.3, n_strikes)
        expiries = np.linspace(0.05, 2.0, n_expiries)

        iv_grid = np.zeros((n_expiries, n_strikes))
        for i, T in enumerate(expiries):
            for j, K in enumerate(strikes):
                moneyness = np.log(K / self.S)
                # ATM vol + skew + term structure + smile
                atm = 0.20 + 0.02 * np.sqrt(T)
                skew = -0.15 * moneyness
                smile = 0.10 * moneyness**2
                iv_grid[i, j] = max(atm + skew + smile, 0.05)

        return {
            "strikes": strikes.tolist(),
            "expiries": expiries.tolist(),
            "ivs": iv_grid.tolist(),
            "n_raw_points": 0,
        }

    # =========================================================================
    # 3.3 — Term Structure Analysis
    # =========================================================================

    def analyze_term_structure(self) -> TermStructure:
        """
        Analyze IV term structure: how ATM IV changes across expiries.
        Contango = rising = normal. Backwardation = falling = fear.
        """
        expiries = sorted(set(p.expiry_years for p in self.surface_points))
        if len(expiries) < 2:
            # Use synthetic
            expiries = [0.08, 0.17, 0.25, 0.5, 1.0]
            atm_ivs = [0.25, 0.23, 0.22, 0.21, 0.20]
        else:
            atm_ivs = []
            for exp in expiries:
                iv = self.get_iv(self.S, exp)  # ATM strike
                atm_ivs.append(iv)

        # Fit slope
        if len(expiries) >= 2:
            slope = np.polyfit(expiries, atm_ivs, 1)[0]
        else:
            slope = 0.0

        # Curvature (second derivative)
        if len(expiries) >= 3:
            curvature = np.polyfit(expiries, atm_ivs, 2)[0]
        else:
            curvature = 0.0

        # Classify shape
        if slope > 0.01:
            shape = "contango"
        elif slope < -0.01:
            shape = "backwardation"
        elif abs(curvature) > 0.05:
            shape = "humped"
        else:
            shape = "flat"

        return TermStructure(
            expiries=expiries, atm_ivs=atm_ivs,
            slope=float(slope), shape=shape, curvature=float(curvature)
        )

    # =========================================================================
    # 3.4 — Skew Analysis
    # =========================================================================

    def analyze_skew(self, expiry: float = 0.25) -> SkewMetrics:
        """
        Analyze volatility skew at a specific expiry.
        25-delta put vs 25-delta call spread.
        """
        atm_iv = self.get_iv(self.S, expiry)

        # 25-delta strikes (approximate)
        # For 25d put: K ≈ S * exp(-0.675 * σ * √T)
        # For 25d call: K ≈ S * exp(+0.675 * σ * √T)
        sqrt_t = np.sqrt(expiry)
        put_25d_strike = self.S * np.exp(-0.675 * atm_iv * sqrt_t)
        call_25d_strike = self.S * np.exp(0.675 * atm_iv * sqrt_t)

        put_25d_iv = self.get_iv(put_25d_strike, expiry)
        call_25d_iv = self.get_iv(call_25d_strike, expiry)

        skew = put_25d_iv - call_25d_iv
        risk_reversal = call_25d_iv - put_25d_iv
        butterfly = (put_25d_iv + call_25d_iv) / 2.0 - atm_iv
        skew_ratio = put_25d_iv / (call_25d_iv + 1e-15)

        return SkewMetrics(
            expiry=expiry, atm_iv=atm_iv,
            put_25d_iv=put_25d_iv, call_25d_iv=call_25d_iv,
            skew_25d=float(skew), risk_reversal=float(risk_reversal),
            butterfly=float(butterfly), skew_ratio=float(skew_ratio),
        )

    def analyze_skew_term_structure(self) -> List[SkewMetrics]:
        """Analyze how skew changes across expiries."""
        expiries = sorted(set(p.expiry_years for p in self.surface_points))
        if not expiries:
            expiries = [0.08, 0.17, 0.25, 0.5, 1.0]
        return [self.analyze_skew(exp) for exp in expiries]
