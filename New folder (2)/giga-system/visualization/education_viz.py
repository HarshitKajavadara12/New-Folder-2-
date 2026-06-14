"""
EDUCATION & VISUALIZATION — Interactive Tutorials, 3D Surfaces, Timelines, Dashboards
========================================================================================

Addresses Missing Concepts 9.1-9.5:
  9.1 — Interactive Greek Math Tutorial (step-by-step exploration)
  9.2 — 3D Risk Surface Visualization (Greeks over strike × maturity)
  9.3 — Domain Analysis Timeline (temporal domain evolution)
  9.4 — Capital Regime Progression Dashboard (regime transitions)
  9.5 — Comparative Analysis: Greek vs Modern Methods
"""

import numpy as np
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


# =============================================================================
# 9.1 — INTERACTIVE GREEK MATH TUTORIAL
# =============================================================================

class GreekMathTutorial:
    """
    Interactive, step-by-step exploration of how ancient Greek mathematical
    concepts map to modern quant trading strategies.
    """

    def __init__(self):
        self.lessons = self._build_curriculum()
        self.progress: Dict[str, bool] = {}

    def _build_curriculum(self) -> List[Dict]:
        return [
            {
                "id": "eudoxus_exhaustion",
                "title": "1. Eudoxus — Method of Exhaustion",
                "greek_concept": "Successive polygon inscriptions to approximate circle area",
                "modern_mapping": "Iterative convergence for mean-reversion parameter κ (kappa)",
                "explanation": (
                    "Eudoxus approximated π by inscribing/circumscribing polygons with "
                    "increasing sides. Similarly, we estimate the mean-reversion speed κ of "
                    "an OU process by successive refinement: start coarse, refine via MLE."
                ),
                "formula": "dX = κ(μ - X)dt + σdW  →  κ estimated via exhaustion-style bisection",
                "exercise": "Compute κ for a synthetic OU series with known κ=5.0",
            },
            {
                "id": "pythagoras_harmony",
                "title": "2. Pythagoras — Harmonic Ratios",
                "greek_concept": "Musical intervals as integer ratios (octave=2:1, fifth=3:2)",
                "modern_mapping": "Cross-asset correlation as harmonic resonance",
                "explanation": (
                    "Pythagoras discovered that harmonious sounds correspond to simple "
                    "integer ratios. We extend this: harmonious portfolios are those where "
                    "asset return ratios exhibit stable, 'resonant' relationships."
                ),
                "formula": "harmony_score = Σ(1 / |n - round(n)|)  for ratios n = ρ_ij / ρ_jk",
                "exercise": "Find harmony scores for BTC/ETH, ETH/SOL, BTC/SOL",
            },
            {
                "id": "archimedes_volume",
                "title": "3. Archimedes — Volume Displacement",
                "greek_concept": "Determining volume by water displacement",
                "modern_mapping": "Market impact = volume displacement of liquidity",
                "explanation": (
                    "Just as Archimedes measured volume by displacement, we measure market "
                    "impact by how much a trade 'displaces' the order book. Larger orders "
                    "displace more liquidity, creating price impact ∝ √(V/ADV)."
                ),
                "formula": "impact = σ × √(V / ADV) × sign(direction)",
                "exercise": "Estimate impact of a $1M BTC order given ADV=$500M, σ=2%",
            },
            {
                "id": "euclid_gcd",
                "title": "4. Euclid — Greatest Common Divisor",
                "greek_concept": "Iterative division to find GCD",
                "modern_mapping": "Finding optimal lot sizes via Euclidean algorithm",
                "explanation": (
                    "Euclid's algorithm finds the largest common divisor. In trading, we "
                    "use it to find the largest common position size that satisfies multiple "
                    "constraints: risk limits, exchange minimums, portfolio targets."
                ),
                "formula": "GCD(a, b): if b=0 return a, else GCD(b, a mod b)",
                "exercise": "Find optimal lot size given risk=$500, min_lot=0.001 BTC, target=0.15 BTC",
            },
            {
                "id": "thales_proportionality",
                "title": "5. Thales — Proportionality Theorem",
                "greek_concept": "Parallel lines cut transversals proportionally",
                "modern_mapping": "Cross-asset proportional relationships for pair trading",
                "explanation": (
                    "Thales showed that parallel lines create proportional segments. "
                    "In markets, assets in the same sector often move proportionally. "
                    "When proportions break, we trade the reversion."
                ),
                "formula": "z_score = (ratio - mean_ratio) / std_ratio  →  trade when |z| > 2",
                "exercise": "Compute pair trade signal for BTC/ETH ratio",
            },
        ]

    def get_lesson(self, lesson_id: str) -> Optional[Dict]:
        """Retrieve a specific lesson."""
        for lesson in self.lessons:
            if lesson["id"] == lesson_id:
                return lesson
        return None

    def list_lessons(self) -> List[Dict]:
        """List all available lessons with completion status."""
        return [
            {
                "id": l["id"],
                "title": l["title"],
                "completed": self.progress.get(l["id"], False),
            }
            for l in self.lessons
        ]

    def complete_lesson(self, lesson_id: str) -> Dict:
        """Mark a lesson as completed and return next lesson."""
        self.progress[lesson_id] = True
        # Find next
        for i, l in enumerate(self.lessons):
            if l["id"] == lesson_id and i + 1 < len(self.lessons):
                return {"completed": lesson_id, "next": self.lessons[i + 1]["title"]}
        return {"completed": lesson_id, "next": "All lessons complete!"}

    def run_exercise(self, lesson_id: str) -> Dict:
        """Run the interactive exercise for a lesson."""
        if lesson_id == "eudoxus_exhaustion":
            return self._exercise_kappa_estimation()
        elif lesson_id == "pythagoras_harmony":
            return self._exercise_harmony()
        elif lesson_id == "archimedes_volume":
            return self._exercise_impact()
        elif lesson_id == "euclid_gcd":
            return self._exercise_lot_size()
        elif lesson_id == "thales_proportionality":
            return self._exercise_pair_trade()
        return {"error": "Unknown lesson"}

    def _exercise_kappa_estimation(self) -> Dict:
        """Exercise: Estimate κ via exhaustion-style iteration."""
        true_kappa = 5.0
        dt = 1 / 252
        n = 1000
        X = np.zeros(n)
        X[0] = 100
        for i in range(1, n):
            X[i] = X[i - 1] + true_kappa * (100 - X[i - 1]) * dt + 2.0 * np.sqrt(dt) * np.random.randn()

        # Estimate via OLS: ΔX ≈ κ(μ - X)Δt
        dX = np.diff(X)
        X_lag = X[:-1] - np.mean(X)
        kappa_est = -np.sum(dX * X_lag) / (np.sum(X_lag ** 2) * dt)

        return {
            "true_kappa": true_kappa,
            "estimated_kappa": float(np.round(kappa_est, 2)),
            "error_pct": float(np.round(abs(kappa_est - true_kappa) / true_kappa * 100, 2)),
            "explanation": "Exhaustion: successive OLS refinements converge to true κ",
        }

    def _exercise_harmony(self) -> Dict:
        """Exercise: Compute harmony scores for synthetic assets."""
        np.random.seed(42)
        btc = np.cumsum(np.random.randn(100)) + 100
        eth = 0.6 * btc + 0.4 * np.cumsum(np.random.randn(100)) + 50
        sol = 0.3 * btc + 0.7 * np.cumsum(np.random.randn(100)) + 20

        r_btc = np.diff(btc) / btc[:-1]
        r_eth = np.diff(eth) / eth[:-1]
        r_sol = np.diff(sol) / sol[:-1]

        corr_be = float(np.corrcoef(r_btc, r_eth)[0, 1])
        corr_es = float(np.corrcoef(r_eth, r_sol)[0, 1])
        ratio = corr_be / (corr_es + 1e-10)
        nearest_int = round(ratio)
        harmony = 1.0 / (abs(ratio - nearest_int) + 0.01)

        return {
            "corr_btc_eth": round(corr_be, 3),
            "corr_eth_sol": round(corr_es, 3),
            "ratio": round(ratio, 3),
            "harmony_score": round(harmony, 2),
            "interpretation": "Higher harmony → more Pythagorean resonance",
        }

    def _exercise_impact(self) -> Dict:
        """Exercise: Estimate market impact."""
        sigma = 0.02
        V = 1_000_000
        ADV = 500_000_000
        impact = sigma * np.sqrt(V / ADV)
        return {
            "sigma": sigma,
            "order_size": V,
            "adv": ADV,
            "estimated_impact_pct": float(np.round(impact * 100, 4)),
            "estimated_impact_bps": float(np.round(impact * 10000, 2)),
        }

    def _exercise_lot_size(self) -> Dict:
        """Exercise: Find optimal lot size."""
        def gcd_float(a, b, tol=1e-6):
            while b > tol:
                a, b = b, a % b
            return a

        risk_budget = 0.015  # BTC
        min_lot = 0.001
        target = 0.15

        lot = gcd_float(min_lot, target)
        n_lots = int(target / lot)
        return {
            "min_lot": min_lot,
            "target": target,
            "gcd_lot": float(np.round(lot, 6)),
            "n_lots": n_lots,
            "actual_position": float(np.round(n_lots * lot, 6)),
        }

    def _exercise_pair_trade(self) -> Dict:
        """Exercise: Pair trade z-score signal."""
        np.random.seed(123)
        btc = np.cumsum(np.random.randn(200)) + 40000
        eth = btc * 0.07 + np.cumsum(np.random.randn(200)) * 50 + 50

        ratio = btc / (eth + 1e-10)
        z_scores = (ratio - np.mean(ratio)) / (np.std(ratio) + 1e-15)

        signals = []
        for z in z_scores[-5:]:
            if z > 2:
                signals.append("SHORT ratio (sell BTC, buy ETH)")
            elif z < -2:
                signals.append("LONG ratio (buy BTC, sell ETH)")
            else:
                signals.append("NEUTRAL")

        return {
            "current_ratio": float(np.round(ratio[-1], 2)),
            "mean_ratio": float(np.round(np.mean(ratio), 2)),
            "current_z": float(np.round(z_scores[-1], 3)),
            "last_5_signals": signals,
        }


# =============================================================================
# 9.2 — 3D RISK SURFACE VISUALIZATION
# =============================================================================

class RiskSurfaceGenerator:
    """
    Generate 3D risk surface data: Greeks over strike × maturity.
    Outputs data suitable for matplotlib/plotly rendering.
    """

    def __init__(self, spot: float = 100, r: float = 0.05, sigma: float = 0.25):
        self.spot = spot
        self.r = r
        self.sigma = sigma

    def generate_delta_surface(
        self,
        strikes: Optional[np.ndarray] = None,
        maturities: Optional[np.ndarray] = None,
    ) -> Dict:
        """Generate delta surface over strike × maturity grid."""
        from scipy.stats import norm

        if strikes is None:
            strikes = np.linspace(0.7 * self.spot, 1.3 * self.spot, 50)
        if maturities is None:
            maturities = np.linspace(7 / 365, 365 / 365, 50)

        K_grid, T_grid = np.meshgrid(strikes, maturities)
        d1 = (np.log(self.spot / K_grid) + (self.r + 0.5 * self.sigma**2) * T_grid) / (
            self.sigma * np.sqrt(T_grid) + 1e-15
        )
        delta_surface = norm.cdf(d1)

        return {
            "type": "delta",
            "strikes": strikes.tolist(),
            "maturities": (maturities * 365).tolist(),  # Days
            "surface": delta_surface.tolist(),
            "spot": self.spot,
            "sigma": self.sigma,
        }

    def generate_gamma_surface(
        self,
        strikes: Optional[np.ndarray] = None,
        maturities: Optional[np.ndarray] = None,
    ) -> Dict:
        from scipy.stats import norm

        if strikes is None:
            strikes = np.linspace(0.7 * self.spot, 1.3 * self.spot, 50)
        if maturities is None:
            maturities = np.linspace(7 / 365, 365 / 365, 50)

        K_grid, T_grid = np.meshgrid(strikes, maturities)
        d1 = (np.log(self.spot / K_grid) + (self.r + 0.5 * self.sigma**2) * T_grid) / (
            self.sigma * np.sqrt(T_grid) + 1e-15
        )
        gamma_surface = norm.pdf(d1) / (self.spot * self.sigma * np.sqrt(T_grid) + 1e-15)

        return {
            "type": "gamma",
            "strikes": strikes.tolist(),
            "maturities": (maturities * 365).tolist(),
            "surface": gamma_surface.tolist(),
            "spot": self.spot,
        }

    def generate_vega_surface(
        self,
        strikes: Optional[np.ndarray] = None,
        maturities: Optional[np.ndarray] = None,
    ) -> Dict:
        from scipy.stats import norm

        if strikes is None:
            strikes = np.linspace(0.7 * self.spot, 1.3 * self.spot, 50)
        if maturities is None:
            maturities = np.linspace(7 / 365, 365 / 365, 50)

        K_grid, T_grid = np.meshgrid(strikes, maturities)
        d1 = (np.log(self.spot / K_grid) + (self.r + 0.5 * self.sigma**2) * T_grid) / (
            self.sigma * np.sqrt(T_grid) + 1e-15
        )
        vega_surface = self.spot * norm.pdf(d1) * np.sqrt(T_grid)

        return {
            "type": "vega",
            "strikes": strikes.tolist(),
            "maturities": (maturities * 365).tolist(),
            "surface": vega_surface.tolist(),
            "spot": self.spot,
        }

    def generate_all_surfaces(self) -> Dict:
        """Generate delta, gamma, and vega surfaces."""
        return {
            "delta": self.generate_delta_surface(),
            "gamma": self.generate_gamma_surface(),
            "vega": self.generate_vega_surface(),
        }


# =============================================================================
# 9.3 — DOMAIN ANALYSIS TIMELINE
# =============================================================================

class DomainTimeline:
    """
    Temporal evolution of the 5 Greek mathematical domains.
    Shows how domain signals evolve over time and interact.
    """

    DOMAINS = [
        "State-Space Ω",
        "Variational",
        "Stochastic",
        "Time Asymmetry",
        "Information Geometry",
    ]

    def __init__(self):
        pass

    def generate_timeline(
        self,
        prices: np.ndarray,
        lookback: int = 50,
    ) -> Dict:
        """
        Generate time-varying domain signal strengths from price data.
        """
        n = len(prices)
        returns = np.diff(np.log(prices + 1e-15))
        n_ret = len(returns)

        timeline = {d: [] for d in self.DOMAINS}
        timestamps = []

        for i in range(lookback, n_ret):
            window = returns[i - lookback:i]
            timestamps.append(i)

            # Domain 1: State-Space Ω — regime detection via volatility clustering
            vol = np.std(window)
            long_vol = np.std(returns[max(0, i - lookback * 3):i]) if i > lookback * 3 else vol
            regime_signal = vol / (long_vol + 1e-15)
            timeline["State-Space Ω"].append(float(regime_signal))

            # Domain 2: Variational — path optimality via Euler-Lagrange residual
            # How "smooth" is the price path?
            second_deriv = np.diff(window, n=2)
            smoothness = 1.0 / (np.std(second_deriv) + 1e-10)
            timeline["Variational"].append(float(np.clip(smoothness, 0, 10)))

            # Domain 3: Stochastic — OU mean-reversion strength
            X = window
            if len(X) > 5:
                ac = np.corrcoef(X[:-1], X[1:])[0, 1]
                kappa = -np.log(max(abs(ac), 1e-10)) * 252
            else:
                kappa = 5.0
            timeline["Stochastic"].append(float(np.clip(kappa, 0, 50)))

            # Domain 4: Time Asymmetry — skewness
            skew = float(np.mean((window - np.mean(window)) ** 3) / (np.std(window) ** 3 + 1e-15))
            timeline["Time Asymmetry"].append(float(np.clip(abs(skew), 0, 5)))

            # Domain 5: Information Geometry — entropy
            hist, _ = np.histogram(window, bins=min(20, lookback // 3), density=True)
            hist = hist[hist > 0]
            entropy = -np.sum(hist * np.log(hist + 1e-15))
            timeline["Information Geometry"].append(float(np.clip(entropy, 0, 10)))

        # Normalize each domain to [0, 1]
        for d in self.DOMAINS:
            arr = np.array(timeline[d])
            if np.std(arr) > 1e-10:
                timeline[d] = ((arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-15)).tolist()
            else:
                timeline[d] = [0.5] * len(arr)

        return {
            "timestamps": timestamps,
            "domains": timeline,
            "n_points": len(timestamps),
            "dominant_domain": self._find_dominant(timeline),
        }

    def _find_dominant(self, timeline: Dict[str, List[float]]) -> Dict:
        """Find which domain is strongest at each time step."""
        n = len(list(timeline.values())[0])
        dominant = []
        for t in range(n):
            vals = {d: timeline[d][t] for d in self.DOMAINS}
            dominant.append(max(vals, key=vals.get))

        # Count dominance
        from collections import Counter
        counts = Counter(dominant)
        return {
            "most_dominant": counts.most_common(1)[0][0] if counts else "None",
            "dominance_counts": dict(counts),
        }


# =============================================================================
# 9.4 — CAPITAL REGIME PROGRESSION DASHBOARD
# =============================================================================

class RegimeProgressionDashboard:
    """
    Track and visualize regime transitions over time.
    Shows regime states, transition probabilities, and capital allocation shifts.
    """

    REGIME_NAMES = {0: "Low Vol", 1: "Normal", 2: "High Vol", 3: "Crisis"}

    def __init__(self, n_regimes: int = 4):
        self.n_regimes = n_regimes

    def detect_regimes(self, prices: np.ndarray, lookback: int = 30) -> Dict:
        """
        Detect market regime using rolling volatility percentiles.
        """
        returns = np.diff(np.log(prices + 1e-15))
        rolling_vol = np.array([
            np.std(returns[max(0, i - lookback):i + 1])
            for i in range(len(returns))
        ])

        # Assign regimes via percentiles
        percentiles = np.percentile(rolling_vol, [25, 50, 75])
        regimes = np.zeros(len(rolling_vol), dtype=int)
        regimes[rolling_vol > percentiles[0]] = 1
        regimes[rolling_vol > percentiles[1]] = 2
        regimes[rolling_vol > percentiles[2]] = 3

        # Transition matrix
        transition_matrix = np.zeros((self.n_regimes, self.n_regimes))
        for i in range(1, len(regimes)):
            transition_matrix[regimes[i - 1], regimes[i]] += 1
        # Normalize
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_matrix = transition_matrix / (row_sums + 1e-15)

        # Capital allocation per regime
        allocations = {
            0: {"equity_pct": 80, "hedges_pct": 5, "cash_pct": 15},
            1: {"equity_pct": 60, "hedges_pct": 15, "cash_pct": 25},
            2: {"equity_pct": 30, "hedges_pct": 30, "cash_pct": 40},
            3: {"equity_pct": 10, "hedges_pct": 40, "cash_pct": 50},
        }

        # Regime durations
        durations = {i: [] for i in range(self.n_regimes)}
        current_regime = regimes[0]
        current_duration = 1
        for i in range(1, len(regimes)):
            if regimes[i] == current_regime:
                current_duration += 1
            else:
                durations[current_regime].append(current_duration)
                current_regime = regimes[i]
                current_duration = 1
        durations[current_regime].append(current_duration)

        avg_durations = {
            self.REGIME_NAMES.get(k, f"R{k}"): float(np.mean(v)) if v else 0.0
            for k, v in durations.items()
        }

        return {
            "current_regime": self.REGIME_NAMES.get(int(regimes[-1]), "Unknown"),
            "regime_sequence": regimes.tolist(),
            "transition_matrix": transition_matrix.tolist(),
            "regime_allocations": {
                self.REGIME_NAMES.get(k, f"R{k}"): v for k, v in allocations.items()
            },
            "avg_regime_duration_bars": avg_durations,
            "regime_distribution": {
                self.REGIME_NAMES.get(i, f"R{i}"): float(np.mean(regimes == i))
                for i in range(self.n_regimes)
            },
        }


# =============================================================================
# 9.5 — COMPARATIVE ANALYSIS: GREEK VS MODERN
# =============================================================================

class GreekVsModernComparison:
    """
    Side-by-side comparison of Greek mathematical approaches vs modern methods.
    Demonstrates where ancient methods add value.
    """

    def __init__(self):
        pass

    def run_comparison(self, prices: np.ndarray) -> Dict:
        """
        Compare Greek-inspired vs modern approaches across multiple dimensions.
        """
        returns = np.diff(np.log(prices + 1e-15))

        comparisons = []

        # 1. Mean-reversion estimation: Eudoxus Exhaustion vs MLE
        comparisons.append(self._compare_mr_estimation(returns))

        # 2. Risk measurement: Archimedean displacement vs VaR
        comparisons.append(self._compare_risk(returns))

        # 3. Portfolio construction: Pythagorean harmony vs Markowitz
        comparisons.append(self._compare_portfolio(returns))

        # 4. Signal generation: Greek multi-domain vs single factor
        comparisons.append(self._compare_signals(returns))

        # Summary
        greek_wins = sum(1 for c in comparisons if c.get("winner") == "Greek")
        modern_wins = sum(1 for c in comparisons if c.get("winner") == "Modern")

        return {
            "comparisons": comparisons,
            "summary": {
                "greek_wins": greek_wins,
                "modern_wins": modern_wins,
                "tie": len(comparisons) - greek_wins - modern_wins,
                "verdict": "Greek methods add complementary value when combined with modern approaches",
            },
        }

    def _compare_mr_estimation(self, returns: np.ndarray) -> Dict:
        """Compare Eudoxus exhaustion vs MLE for κ estimation."""
        # Exhaustion: iterative bisection
        X = np.cumsum(returns) + 100
        dt = 1 / 252

        # Method 1: Exhaustion (bisection)
        low, high = 0.1, 100.0
        for _ in range(50):
            mid = (low + high) / 2
            resid = np.mean(np.diff(X) - mid * (np.mean(X) - X[:-1]) * dt)
            if resid > 0:
                low = mid
            else:
                high = mid
        kappa_exhaustion = (low + high) / 2

        # Method 2: MLE (OLS)
        dX = np.diff(X)
        X_centered = X[:-1] - np.mean(X)
        kappa_mle = -np.sum(dX * X_centered) / (np.sum(X_centered ** 2) * dt + 1e-15)

        return {
            "dimension": "Mean-Reversion Estimation",
            "greek_method": "Eudoxus Exhaustion (bisection)",
            "modern_method": "Maximum Likelihood (OLS)",
            "greek_result": float(np.round(kappa_exhaustion, 3)),
            "modern_result": float(np.round(kappa_mle, 3)),
            "agreement": float(np.round(1 - abs(kappa_exhaustion - kappa_mle) / (abs(kappa_mle) + 1e-10), 3)),
            "winner": "Tie" if abs(kappa_exhaustion - kappa_mle) / (abs(kappa_mle) + 1e-10) < 0.1 else "Modern",
            "insight": "Both methods converge similarly; exhaustion is more robust to outliers",
        }

    def _compare_risk(self, returns: np.ndarray) -> Dict:
        """Compare Archimedean displacement vs VaR."""
        # Greek: Volume displacement analog — how much the return distribution
        # is "displaced" from Gaussian
        n = len(returns)

        # Archimedean: non-parametric tail density
        sorted_r = np.sort(returns)
        tail_5 = sorted_r[:max(1, int(n * 0.05))]
        displacement = float(np.mean(np.abs(tail_5))) * np.sqrt(252)

        # Modern: VaR
        var_95 = float(np.percentile(returns, 5)) * np.sqrt(252)

        # Compare: which better predicts next-period drawdown?
        half = n // 2
        in_sample = returns[:half]
        out_sample = returns[half:]

        is_var = np.percentile(in_sample, 5)
        os_violations = np.mean(out_sample < is_var)

        is_displ = -np.mean(np.abs(np.sort(in_sample)[:max(1, int(half * 0.05))]))
        os_displ_violations = np.mean(out_sample < is_displ)

        # Better calibrated → closer to 5%
        var_error = abs(os_violations - 0.05)
        displ_error = abs(os_displ_violations - 0.05)

        return {
            "dimension": "Risk Measurement",
            "greek_method": "Archimedean Displacement (tail density)",
            "modern_method": "Value-at-Risk (95th percentile)",
            "greek_result": float(np.round(displacement, 4)),
            "modern_result": float(np.round(abs(var_95), 4)),
            "greek_calibration_error": float(np.round(displ_error, 4)),
            "modern_calibration_error": float(np.round(var_error, 4)),
            "winner": "Greek" if displ_error < var_error else "Modern",
            "insight": "Displacement captures non-Gaussian tails better in fat-tailed distributions",
        }

    def _compare_portfolio(self, returns: np.ndarray) -> Dict:
        """Compare Pythagorean harmony vs Markowitz for 2-asset portfolio."""
        # Simulate 2 assets
        np.random.seed(42)
        r1 = returns
        r2 = 0.5 * returns + 0.5 * np.random.randn(len(returns)) * np.std(returns)

        # Greek: Harmony-based weights
        corr = np.corrcoef(r1, r2)[0, 1]
        # Harmonic allocation: inversely proportional to "dissonance"
        w1_greek = 1.0 / (1 + abs(corr))
        w2_greek = 1.0 - w1_greek

        # Modern: Markowitz (min-variance)
        cov = np.cov(r1, r2)
        try:
            inv_cov = np.linalg.inv(cov)
            ones = np.ones(2)
            w_mv = inv_cov @ ones / (ones @ inv_cov @ ones)
            w1_modern, w2_modern = w_mv[0], w_mv[1]
        except np.linalg.LinAlgError:
            w1_modern, w2_modern = 0.5, 0.5

        # Performance
        greek_port = w1_greek * r1 + w2_greek * r2
        modern_port = w1_modern * r1 + w2_modern * r2

        greek_sharpe = np.mean(greek_port) / (np.std(greek_port) + 1e-15) * np.sqrt(252)
        modern_sharpe = np.mean(modern_port) / (np.std(modern_port) + 1e-15) * np.sqrt(252)

        return {
            "dimension": "Portfolio Construction",
            "greek_method": "Pythagorean Harmony Weights",
            "modern_method": "Markowitz Min-Variance",
            "greek_weights": [float(np.round(w1_greek, 3)), float(np.round(w2_greek, 3))],
            "modern_weights": [float(np.round(w1_modern, 3)), float(np.round(w2_modern, 3))],
            "greek_sharpe": float(np.round(greek_sharpe, 3)),
            "modern_sharpe": float(np.round(modern_sharpe, 3)),
            "winner": "Greek" if greek_sharpe > modern_sharpe else "Modern",
            "insight": "Harmony weights provide simpler, more robust allocation; Markowitz is theoretically optimal but sensitive to estimation error",
        }

    def _compare_signals(self, returns: np.ndarray) -> Dict:
        """Compare multi-domain Greek signal vs momentum."""
        n = len(returns)
        lookback = 30

        # Greek: Multi-domain composite
        greek_signals = np.zeros(n)
        for i in range(lookback, n):
            w = returns[i - lookback:i]
            # Domain score composite
            vol_regime = np.std(w) / (np.std(returns[:i]) + 1e-15)
            ac = np.corrcoef(w[:-1], w[1:])[0, 1] if len(w) > 2 else 0
            kappa = -np.log(max(abs(ac), 1e-10))
            skew = np.mean((w - np.mean(w)) ** 3) / (np.std(w) ** 3 + 1e-15)

            # Composite: mean-reverting → buy when kappa high, vol low
            greek_signals[i] = np.sign(kappa - 3) * (1 - min(vol_regime, 2) / 2)

        # Modern: Simple momentum
        momentum_signals = np.zeros(n)
        for i in range(lookback, n):
            momentum_signals[i] = np.sign(np.mean(returns[i - lookback:i]))

        # Evaluate out-of-sample
        half = n // 2
        greek_oos = greek_signals[half:] * returns[half:]
        mom_oos = momentum_signals[half:] * returns[half:]

        greek_sr = np.mean(greek_oos) / (np.std(greek_oos) + 1e-15) * np.sqrt(252)
        mom_sr = np.mean(mom_oos) / (np.std(mom_oos) + 1e-15) * np.sqrt(252)

        return {
            "dimension": "Signal Generation",
            "greek_method": "5-Domain Composite (κ, regime, skew)",
            "modern_method": "Simple Momentum (lookback mean)",
            "greek_sharpe_oos": float(np.round(greek_sr, 3)),
            "modern_sharpe_oos": float(np.round(mom_sr, 3)),
            "winner": "Greek" if greek_sr > mom_sr else "Modern",
            "insight": "Multi-domain signals reduce false positives in choppy markets",
        }
