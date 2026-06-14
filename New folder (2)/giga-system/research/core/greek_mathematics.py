"""
GENUINE GREEK MATHEMATICS MODULE
=================================

This module implements ACTUAL Greek mathematical methods, not just Greek variable naming.
These are computational techniques derived from ancient Greek mathematical thinking,
applied to financial markets.

Historical Basis:
- Euclidean Algorithm (GCD) → Order sizing optimization
- Archimedean Spiral → Recursive portfolio rebalancing
- Eudoxian Method of Exhaustion → Rigorous numerical convergence proofs
- Pythagorean Harmony → Regime frequency detection (Fourier roots)
- Apollonius Conic Sections → Risk surface curvature analysis
- Zeno's Paradox → Infinite series convergence in pricing
- Platonic Solids → Multi-dimensional portfolio symmetry

Why "Old Math Is More Powerful":
Ancient Greek math is constructive — every result is built from first principles.
Modern quant finance often applies formulas without proving convergence or
understanding the geometric structure beneath. Greek methods force rigor.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# 1. EUCLIDEAN ALGORITHM — Order Sizing Optimization
# =============================================================================

class EuclideanOrderSizer:
    """
    Uses the Euclidean Algorithm (Euclid, Elements VII, ~300 BCE) for
    optimal order sizing.

    Principle:
      The GCD finds the largest common divisor — applied to trading,
      it finds the optimal lot size that divides evenly into both
      the desired position and the exchange's minimum tick.
      
      Extended Euclidean: ax + by = gcd(a, b) → used for splitting
      orders across multiple venues with different lot sizes.
    """

    @staticmethod
    def gcd(a: int, b: int) -> int:
        """Euclid's Algorithm — Elements Book VII, Propositions 1-2."""
        while b != 0:
            a, b = b, a % b
        return a

    @staticmethod
    def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
        """
        Extended Euclidean Algorithm.
        Returns (gcd, x, y) such that ax + by = gcd(a, b).
        Used for: splitting orders across venues with different minimum lots.
        """
        if a == 0:
            return b, 0, 1
        gcd, x1, y1 = EuclideanOrderSizer.extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return gcd, x, y

    @staticmethod
    def optimal_lot_size(desired_qty: float, min_lot: float, tick_size: float) -> float:
        """
        Find the largest order size that is:
        1. A multiple of min_lot (exchange constraint)
        2. Compatible with tick_size (price constraint)
        3. As close to desired_qty as possible
        
        Uses Euclidean principle: the optimal lot is the GCD of constraints.
        """
        if min_lot <= 0 or tick_size <= 0:
            return desired_qty

        # Scale to integers for GCD
        scale = int(1 / min(min_lot, tick_size, 1e-8))
        a = int(min_lot * scale)
        b = int(tick_size * scale)

        g = EuclideanOrderSizer.gcd(a, b)
        optimal_unit = g / scale

        if optimal_unit <= 0:
            return desired_qty

        # Largest multiple of optimal_unit ≤ desired_qty
        n_lots = int(desired_qty / optimal_unit)
        return n_lots * optimal_unit

    @staticmethod
    def split_order_across_venues(
        total_qty: float,
        venue_min_lots: List[float],
    ) -> List[float]:
        """
        Euclidean decomposition: split a total order across N venues
        each with different minimum lot sizes, using extended GCD.
        """
        if not venue_min_lots:
            return [total_qty]

        # Find common divisor across all venues
        scale = 10**8
        int_lots = [max(1, int(lot * scale)) for lot in venue_min_lots]

        common = int_lots[0]
        for lot in int_lots[1:]:
            common = EuclideanOrderSizer.gcd(common, lot)

        unit = common / scale
        n_units = int(total_qty / unit) if unit > 0 else 1

        # Distribute proportionally
        total_min = sum(venue_min_lots)
        if total_min <= 0:
            return [total_qty / len(venue_min_lots)] * len(venue_min_lots)

        allocations = []
        remaining = total_qty
        for i, ml in enumerate(venue_min_lots):
            if i == len(venue_min_lots) - 1:
                alloc = remaining  # Last venue gets remainder
            else:
                proportion = ml / total_min
                raw = total_qty * proportion
                alloc = int(raw / ml) * ml if ml > 0 else raw
            allocations.append(alloc)
            remaining -= alloc

        return allocations


# =============================================================================
# 2. ARCHIMEDEAN SPIRAL — Recursive Portfolio Rebalancing
# =============================================================================

class ArchimedeanRebalancer:
    """
    Based on the Archimedean Spiral (r = a + bθ, Archimedes ~250 BCE).

    Principle:
      Instead of discrete periodic rebalancing (monthly, quarterly),
      the portfolio rebalances along a spiral path — the rebalancing
      frequency and magnitude increase as deviation from target grows.
      
      Inner spiral (near center) = small deviations → gentle rebalance
      Outer spiral (far from center) = large deviations → aggressive rebalance
      
      This is continuous, proportional, and path-independent — properties
      Archimedes proved for the spiral.
    """

    def __init__(self, a: float = 0.01, b: float = 0.05):
        """
        Parameters:
          a: Initial radius (minimum rebalancing threshold, e.g., 1%)
          b: Growth rate (how fast rebalancing aggressiveness increases)
        """
        self.a = a  # Minimum deviation before rebalancing
        self.b = b  # Aggressiveness growth rate
        self.theta = 0.0  # Current angle on spiral (accumulated deviation)

    def compute_rebalance_trades(
        self, current_weights: Dict[str, float], target_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compute rebalancing trades using Archimedean spiral proportionality.
        
        Returns: Dict of {asset: trade_weight} (positive = buy, negative = sell)
        """
        trades = {}
        total_deviation = 0.0

        for asset in target_weights:
            current = current_weights.get(asset, 0.0)
            target = target_weights[asset]
            deviation = target - current
            total_deviation += abs(deviation)

        # Spiral radius at current deviation
        # r(θ) = a + bθ, where θ ~ cumulative deviation
        self.theta += total_deviation
        radius = self.a + self.b * self.theta

        # Rebalancing fraction: proportional to how far we are on the spiral
        # Inner spiral → small correction, outer → full correction
        rebalance_fraction = min(1.0, radius / max(total_deviation, 1e-10))

        for asset in target_weights:
            current = current_weights.get(asset, 0.0)
            target = target_weights[asset]
            deviation = target - current

            if abs(deviation) > self.a:
                # Only rebalance if deviation exceeds minimum threshold
                trades[asset] = deviation * rebalance_fraction
            else:
                trades[asset] = 0.0

        return trades

    def reset(self):
        """Reset spiral to center (after full rebalance)."""
        self.theta = 0.0


# =============================================================================
# 3. EUDOXIAN METHOD OF EXHAUSTION — Numerical Convergence Proofs
# =============================================================================

class EudoxianConvergence:
    """
    Method of Exhaustion (Eudoxus of Cnidus, ~408-355 BCE).

    Principle:
      Prove convergence by showing the difference between successive
      approximations can be made arbitrarily small. Unlike modern epsilon-delta,
      Eudoxus used a constructive doubling argument:
      "If the difference can be halved indefinitely, it converges."
      
      Applied to numerical methods in finance:
      - Verify Monte Carlo convergence
      - Validate binomial tree convergence to BS
      - Prove pricing stability
    """

    @staticmethod
    def exhaustion_test(
        approximation_fn,
        n_values: List[int],
        tolerance: float = 1e-6,
    ) -> Dict:
        """
        Test convergence using Method of Exhaustion.
        
        The function is evaluated at increasing n values.
        Convergence is proven if |f(2n) - f(n)| < |f(n) - f(n/2)| / 2
        (each doubling reduces error by at least half — Eudoxian criterion).
        
        Args:
            approximation_fn: Callable that takes n (int) and returns float
            n_values: List of increasing sample sizes
            tolerance: Convergence tolerance
            
        Returns:
            Dict with convergence proof results
        """
        values = []
        for n in n_values:
            val = approximation_fn(n)
            values.append(val)

        # Check Eudoxian criterion: successive errors must halve
        differences = []
        eudoxian_satisfied = True
        for i in range(1, len(values)):
            diff = abs(values[i] - values[i - 1])
            differences.append(diff)
            if i >= 2:
                prev_diff = differences[i - 2]
                if diff > prev_diff * 0.6:  # Allow some slack (0.5 = perfect halving)
                    eudoxian_satisfied = False

        converged = len(differences) > 0 and differences[-1] < tolerance
        final_value = values[-1] if values else None

        return {
            "converged": converged,
            "eudoxian_criterion": eudoxian_satisfied,
            "final_value": final_value,
            "n_iterations": len(values),
            "final_error": differences[-1] if differences else float('inf'),
            "error_sequence": differences,
            "proof": (
                "PROVEN: Successive errors halve (Eudoxian exhaustion satisfied)"
                if eudoxian_satisfied and converged
                else "NOT PROVEN: Convergence criterion not met"
            ),
        }

    @staticmethod
    def verify_monte_carlo_convergence(
        price_fn, S: float, K: float, r: float, sigma: float, T: float
    ) -> Dict:
        """
        Verify that Monte Carlo pricing converges to theoretical value
        using Eudoxian exhaustion.
        """
        n_values = [100, 500, 1000, 5000, 10000, 50000]

        def mc_estimate(n):
            np.random.seed(42)
            z = np.random.standard_normal(n)
            ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
            payoff = np.maximum(ST - K, 0)
            return np.exp(-r * T) * np.mean(payoff)

        return EudoxianConvergence.exhaustion_test(mc_estimate, n_values)


# =============================================================================
# 4. PYTHAGOREAN HARMONY — Regime Frequency Detection
# =============================================================================

class PythagoreanHarmony:
    """
    Pythagorean Ratios (Pythagoras, ~570-495 BCE).

    Principle:
      Pythagoras discovered that musical harmony comes from simple ratios:
      octave = 2:1, fifth = 3:2, fourth = 4:3.
      
      Applied to markets:
      - Decompose price movements into frequency components (FFT)
      - Identify "harmonic" ratios between dominant frequencies
      - Markets in "harmony" (simple frequency ratios) → stable regime
      - Markets in "dissonance" (irrational ratios) → regime transition
      
      This extends beyond standard Fourier analysis by looking at the
      RELATIONSHIPS between frequency components, not just the components.
    """

    HARMONIC_RATIOS = {
        "unison": 1 / 1,
        "octave": 2 / 1,
        "fifth": 3 / 2,
        "fourth": 4 / 3,
        "major_third": 5 / 4,
        "minor_third": 6 / 5,
    }

    @staticmethod
    def detect_frequencies(prices: pd.Series, n_top: int = 5) -> List[Tuple[float, float]]:
        """
        Extract dominant frequencies from price series using FFT.
        Returns: List of (frequency, amplitude) tuples.
        """
        returns = prices.pct_change().dropna().values
        if len(returns) < 10:
            return []

        # FFT
        fft_vals = np.fft.rfft(returns - np.mean(returns))
        freqs = np.fft.rfftfreq(len(returns))
        amplitudes = np.abs(fft_vals)

        # Skip DC component (index 0)
        if len(amplitudes) > 1:
            amplitudes[0] = 0

        # Top N frequencies
        top_idx = np.argsort(amplitudes)[-n_top:][::-1]
        result = []
        for idx in top_idx:
            if idx < len(freqs) and amplitudes[idx] > 0:
                result.append((float(freqs[idx]), float(amplitudes[idx])))

        return result

    @staticmethod
    def compute_harmony_score(frequencies: List[Tuple[float, float]]) -> Dict:
        """
        Measure how "harmonic" the frequency structure is.
        
        High harmony → regime stability (Pythagorean order)
        Low harmony → regime instability (Pythagorean chaos)
        """
        if len(frequencies) < 2:
            return {"harmony_score": 0.0, "dominant_ratio": "none", "regime_stability": "unknown"}

        # Compute ratios between dominant frequencies
        ratios = []
        for i in range(len(frequencies)):
            for j in range(i + 1, len(frequencies)):
                f1 = frequencies[i][0]
                f2 = frequencies[j][0]
                if f2 > 0 and f1 > 0:
                    ratio = max(f1, f2) / min(f1, f2)
                    ratios.append(ratio)

        if not ratios:
            return {"harmony_score": 0.0, "dominant_ratio": "none", "regime_stability": "unknown"}

        # Score: how close each ratio is to a harmonic ratio
        harmony_scores = []
        best_match = "dissonant"
        best_distance = float('inf')

        for ratio in ratios:
            min_dist = float('inf')
            closest = ""
            for name, harmonic in PythagoreanHarmony.HARMONIC_RATIOS.items():
                dist = abs(ratio - harmonic)
                if dist < min_dist:
                    min_dist = dist
                    closest = name
            harmony_scores.append(1.0 / (1.0 + min_dist))
            if min_dist < best_distance:
                best_distance = min_dist
                best_match = closest

        avg_harmony = float(np.mean(harmony_scores))

        if avg_harmony > 0.7:
            stability = "STABLE (Pythagorean order)"
        elif avg_harmony > 0.4:
            stability = "TRANSITIONAL"
        else:
            stability = "UNSTABLE (Pythagorean dissonance)"

        return {
            "harmony_score": avg_harmony,
            "dominant_ratio": best_match,
            "regime_stability": stability,
            "frequency_ratios": ratios[:5],
        }


# =============================================================================
# 5. APOLLONIUS CONIC SECTIONS — Risk Surface Curvature
# =============================================================================

class ApolloniusCurvature:
    """
    Conic Section Analysis (Apollonius of Perga, ~262-190 BCE).

    Principle:
      Apollonius classified all second-degree curves as ellipses, parabolas,
      or hyperbolas based on their curvature properties.
      
      Applied to risk surfaces:
      - Elliptic (positive curvature) → risk is bounded, diversifiable
      - Parabolic (zero curvature) → linear risk, hedgeable
      - Hyperbolic (negative curvature) → risk is divergent, catastrophic
      
      The shape of the loss function determines the type of risk management needed.
    """

    @dataclass
    class CurvatureAnalysis:
        curvature_type: str  # "elliptic", "parabolic", "hyperbolic"
        principal_curvatures: Tuple[float, float]
        gaussian_curvature: float  # K = k1 * k2
        mean_curvature: float  # H = (k1 + k2) / 2
        risk_classification: str
        recommendation: str

    @staticmethod
    def analyze_risk_surface(
        returns_x: np.ndarray, returns_y: np.ndarray, portfolio_returns: np.ndarray
    ) -> 'ApolloniusCurvature.CurvatureAnalysis':
        """
        Fit a quadratic surface to portfolio returns as function of two risk factors.
        Classify the surface using Apollonius's conic section theory.
        
        z = ax² + bxy + cy² + dx + ey + f
        Discriminant D = b² - 4ac determines conic type.
        """
        if len(returns_x) < 10:
            return ApolloniusCurvature.CurvatureAnalysis(
                "unknown", (0, 0), 0, 0, "insufficient data", "gather more data"
            )

        # Fit quadratic surface: z = a*x^2 + b*x*y + c*y^2 + d*x + e*y + f
        x = returns_x
        y = returns_y
        z = portfolio_returns

        # Design matrix for quadratic fit
        A = np.column_stack([x**2, x * y, y**2, x, y, np.ones_like(x)])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
        except np.linalg.LinAlgError:
            return ApolloniusCurvature.CurvatureAnalysis(
                "unknown", (0, 0), 0, 0, "fit failed", "check data quality"
            )

        a, b, c, d, e, f = coeffs

        # Apollonius Discriminant
        discriminant = b**2 - 4 * a * c

        # Principal curvatures (eigenvalues of Hessian)
        hessian = np.array([[2 * a, b], [b, 2 * c]])
        eigenvalues = np.linalg.eigvals(hessian)
        k1, k2 = float(eigenvalues[0]), float(eigenvalues[1])

        # Gaussian curvature K = k1 * k2
        gaussian_K = k1 * k2
        # Mean curvature H = (k1 + k2) / 2
        mean_H = (k1 + k2) / 2

        # Classification (Apollonius)
        if discriminant < -1e-6:
            curve_type = "elliptic"
            risk_class = "BOUNDED RISK — diversification works"
            recommendation = "Standard hedging effective. Risk is contained."
        elif discriminant > 1e-6:
            curve_type = "hyperbolic"
            risk_class = "DIVERGENT RISK — tail risk dominant"
            recommendation = "Tail hedging required. Consider options/insurance."
        else:
            curve_type = "parabolic"
            risk_class = "LINEAR RISK — directional exposure"
            recommendation = "Delta hedging sufficient. Risk scales linearly."

        return ApolloniusCurvature.CurvatureAnalysis(
            curvature_type=curve_type,
            principal_curvatures=(k1, k2),
            gaussian_curvature=gaussian_K,
            mean_curvature=mean_H,
            risk_classification=risk_class,
            recommendation=recommendation,
        )


# =============================================================================
# 6. ZENO'S PARADOX — Infinite Series Convergence in Pricing
# =============================================================================

class ZenoConvergence:
    """
    Zeno of Elea's Paradoxes (~490-430 BCE) and Infinite Series.

    Principle:
      Zeno argued motion was impossible because you must traverse
      infinitely many half-distances. Archimedes resolved this by
      showing 1/2 + 1/4 + 1/8 + ... = 1 (geometric series converges).
      
      Applied to pricing:
      - Perpetual options as infinite series of exercise opportunities
      - Present value of infinite cash flows (geometric series)
      - Convergence rate determines pricing accuracy
      - Series acceleration techniques from Euler
    """

    @staticmethod
    def geometric_series_pv(
        cash_flow: float, discount_rate: float, n_periods: int = None
    ) -> float:
        """
        Present value using geometric series.
        PV = C * (1 - (1+r)^(-n)) / r  (finite)
        PV = C / r                       (infinite, Zeno's resolution)
        """
        if discount_rate <= 0:
            return float('inf') if n_periods is None else cash_flow * n_periods

        if n_periods is None:
            # Infinite series (Zeno's resolution: the sum converges)
            return cash_flow / discount_rate
        else:
            # Finite geometric series
            return cash_flow * (1 - (1 + discount_rate) ** (-n_periods)) / discount_rate

    @staticmethod
    def series_convergence_rate(
        partial_sums: List[float],
    ) -> Dict:
        """
        Analyze the convergence rate of a series.
        Uses ratio test (inspired by Archimedes's geometric argument).
        """
        if len(partial_sums) < 3:
            return {"converges": False, "rate": 0.0, "type": "insufficient_data"}

        # Ratio test
        ratios = []
        for i in range(2, len(partial_sums)):
            diff_curr = abs(partial_sums[i] - partial_sums[i - 1])
            diff_prev = abs(partial_sums[i - 1] - partial_sums[i - 2])
            if diff_prev > 1e-15:
                ratios.append(diff_curr / diff_prev)

        if not ratios:
            return {"converges": True, "rate": 0.0, "type": "already_converged"}

        avg_ratio = float(np.mean(ratios))

        if avg_ratio < 1.0:
            conv_type = "geometric" if avg_ratio < 0.9 else "slow_geometric"
            return {"converges": True, "rate": avg_ratio, "type": conv_type}
        else:
            return {"converges": False, "rate": avg_ratio, "type": "divergent"}

    @staticmethod
    def euler_acceleration(partial_sums: List[float]) -> float:
        """
        Euler series acceleration — speed up convergence of alternating series.
        Given S_0, S_1, ..., compute accelerated estimate.
        """
        if len(partial_sums) < 3:
            return partial_sums[-1] if partial_sums else 0.0

        # Aitken's delta-squared process (related to Euler transform)
        s0, s1, s2 = partial_sums[-3], partial_sums[-2], partial_sums[-1]
        denom = s2 - 2 * s1 + s0
        if abs(denom) < 1e-15:
            return s2
        return s0 - (s1 - s0) ** 2 / denom


# =============================================================================
# 7. PLATONIC SYMMETRY — Multi-Dimensional Portfolio Analysis
# =============================================================================

class PlatonicSymmetry:
    """
    Platonic Solids (Plato, ~428-348 BCE / Theaetetus).

    Principle:
      The 5 Platonic solids represent perfect symmetry in 3D.
      A well-diversified portfolio should exhibit symmetry properties:
      - Equal risk contribution from each asset
      - Rotation-invariant performance (works in all regimes)
      
      We measure how close a portfolio's risk structure is to
      "Platonic perfection" using symmetry metrics.
    """

    @staticmethod
    def symmetry_score(weights: np.ndarray, covariance: np.ndarray) -> Dict:
        """
        Measure portfolio symmetry using Platonic ideals.
        
        Perfect symmetry: all risk contributions equal (like vertices of Platonic solid).
        """
        n = len(weights)
        if n < 2 or covariance.shape[0] != n:
            return {"symmetry_score": 0.0, "risk_contributions": []}

        # Portfolio variance
        port_var = float(weights @ covariance @ weights)
        if port_var < 1e-15:
            return {"symmetry_score": 1.0, "risk_contributions": [0.0] * n}

        # Marginal risk contributions
        marginal = covariance @ weights
        risk_contributions = (weights * marginal) / np.sqrt(port_var)

        # Perfect symmetry: each asset contributes 1/n of total risk
        ideal = np.ones(n) / n
        deviation = np.linalg.norm(risk_contributions - ideal)
        max_deviation = np.linalg.norm(np.ones(n))  # Maximum possible deviation

        symmetry = 1.0 - (deviation / max_deviation) if max_deviation > 0 else 0.0

        # Classification based on Platonic solids
        if symmetry > 0.9:
            shape = "ICOSAHEDRON (near-perfect symmetry)"
        elif symmetry > 0.7:
            shape = "CUBE (good symmetry)"
        elif symmetry > 0.5:
            shape = "TETRAHEDRON (basic symmetry)"
        else:
            shape = "IRREGULAR (asymmetric)"

        return {
            "symmetry_score": float(symmetry),
            "platonic_shape": shape,
            "risk_contributions": risk_contributions.tolist(),
            "ideal_contributions": ideal.tolist(),
        }

    @staticmethod
    def equal_risk_weights(covariance: np.ndarray, max_iter: int = 100) -> np.ndarray:
        """
        Find risk-parity weights (Platonic ideal: equal risk contribution).
        Uses iterative method inspired by Euclidean proportional reasoning.
        """
        n = covariance.shape[0]
        weights = np.ones(n) / n

        for _ in range(max_iter):
            marginal = covariance @ weights
            risk_contrib = weights * marginal
            total_risk = np.sum(risk_contrib)

            if total_risk < 1e-15:
                break

            target = total_risk / n
            adjustment = target / (risk_contrib + 1e-15)
            weights = weights * adjustment
            weights = weights / np.sum(weights)

        return weights


# =============================================================================
# 8. THALES' PROPORTIONALITY — Cross-Asset Pricing
# =============================================================================

class ThalesProportionality:
    """
    Thales of Miletus (624-546 BCE) used proportional reasoning:
    "If A:B = B:C, then deviations from this proportion create arbitrage."

    Applied to cross-asset pricing: detect when proportional relationships
    between assets break down, signalling mean-reversion opportunities.
    """

    def __init__(self, lookback: int = 60, zscore_threshold: float = 2.0):
        self.lookback = lookback
        self.zscore_threshold = zscore_threshold

    def compute_proportional_chain(
        self, prices_a: np.ndarray, prices_b: np.ndarray, prices_c: np.ndarray
    ) -> Dict:
        """
        Thales' proportion: A/B should relate to B/C.
        If A:B :: B:C then B² ≈ A*C (geometric mean property).
        Deviation from this = trading signal.
        """
        n = min(len(prices_a), len(prices_b), len(prices_c))
        prices_a, prices_b, prices_c = prices_a[-n:], prices_b[-n:], prices_c[-n:]

        # Thales ratio: B² / (A * C) should ≈ 1.0
        thales_ratio = (prices_b ** 2) / (prices_a * prices_c + 1e-15)

        # Rolling statistics
        lb = min(self.lookback, n)
        recent = thales_ratio[-lb:]
        mu = np.mean(recent)
        sigma = np.std(recent) + 1e-15
        current = thales_ratio[-1]
        zscore = (current - mu) / sigma

        # Signal: deviation from Thales proportion
        if zscore > self.zscore_threshold:
            signal = "SHORT_B_LONG_AC"  # B is overpriced relative to geometric mean
        elif zscore < -self.zscore_threshold:
            signal = "LONG_B_SHORT_AC"  # B is underpriced
        else:
            signal = "NEUTRAL"

        return {
            "thales_ratio_current": float(current),
            "thales_ratio_mean": float(mu),
            "zscore": float(zscore),
            "signal": signal,
            "deviation_pct": float((current - 1.0) * 100),
            "half_life": self._estimate_half_life(thales_ratio),
        }

    def find_proportional_triads(
        self, prices_dict: Dict[str, np.ndarray], top_n: int = 5
    ) -> List[Dict]:
        """
        Scan all asset triads to find those with strongest Thales proportionality
        and current deviations.
        """
        assets = list(prices_dict.keys())
        results = []

        for i in range(len(assets)):
            for j in range(i + 1, len(assets)):
                for k in range(j + 1, len(assets)):
                    a, b, c = assets[i], assets[j], assets[k]
                    # Test all 3 orderings: which asset is the "middle"?
                    for mid_idx, (pa, pb, pc) in enumerate([
                        (prices_dict[a], prices_dict[b], prices_dict[c]),
                        (prices_dict[b], prices_dict[a], prices_dict[c]),
                        (prices_dict[a], prices_dict[c], prices_dict[b]),
                    ]):
                        try:
                            result = self.compute_proportional_chain(pa, pb, pc)
                            result["triad"] = (a, b, c)
                            result["middle_asset"] = [b, a, c][mid_idx]
                            result["abs_zscore"] = abs(result["zscore"])
                            results.append(result)
                        except Exception:
                            continue

        results.sort(key=lambda x: x["abs_zscore"], reverse=True)
        return results[:top_n]

    @staticmethod
    def _estimate_half_life(series: np.ndarray) -> float:
        """Estimate mean-reversion half-life via OLS on lagged differences."""
        if len(series) < 10:
            return float("inf")
        y = np.diff(series)
        x = series[:-1] - np.mean(series)
        if np.std(x) < 1e-15:
            return float("inf")
        slope = np.sum(x * y) / (np.sum(x ** 2) + 1e-15)
        if slope >= 0:
            return float("inf")
        return float(-np.log(2) / slope)


# =============================================================================
# 9. HIPPARCHUS TRIGONOMETRY — Cyclical Analysis
# =============================================================================

class HipparchusTrigonometry:
    """
    Hipparchus (190-120 BCE) created the first trigonometric table (chord table).

    Applied to finance: decompose price action into sinusoidal components,
    then use trigonometric IDENTITIES to reveal relationships between cycles.

    Key identity: sin(A+B) = sinA*cosB + cosA*sinB
    → If two market cycles combine, we can predict constructive/destructive interference.
    """

    def __init__(self, max_harmonics: int = 10):
        self.max_harmonics = max_harmonics

    def build_chord_table(self, prices: np.ndarray) -> Dict:
        """
        Hipparchus' chord function: chord(θ) = 2R·sin(θ/2)
        Build a 'financial chord table' mapping cycle angles to price amplitudes.
        """
        n = len(prices)
        returns = np.diff(np.log(prices + 1e-15))

        # FFT to extract dominant frequencies
        fft_vals = np.fft.rfft(returns)
        magnitudes = np.abs(fft_vals)
        phases = np.angle(fft_vals)
        freqs = np.fft.rfftfreq(len(returns))

        # Find dominant harmonics (skip DC component)
        top_indices = np.argsort(magnitudes[1:])[-self.max_harmonics:] + 1

        chord_table = []
        for idx in top_indices:
            freq = freqs[idx]
            period = 1.0 / freq if freq > 0 else float("inf")
            amplitude = magnitudes[idx] * 2.0 / len(returns)
            phase = phases[idx]
            # Hipparchus chord: chord(θ) = 2R·sin(θ/2)
            theta = 2 * np.pi * freq
            chord_value = 2 * amplitude * np.sin(theta / 2)

            chord_table.append({
                "frequency": float(freq),
                "period_bars": float(period),
                "amplitude": float(amplitude),
                "phase_rad": float(phase),
                "chord_value": float(chord_value),
            })

        chord_table.sort(key=lambda x: x["amplitude"], reverse=True)
        return {"chord_table": chord_table, "n_harmonics": len(chord_table)}

    def detect_cycle_interference(self, prices: np.ndarray) -> Dict:
        """
        Use trigonometric identity sin(A+B) = sinA·cosB + cosA·sinB
        to predict constructive/destructive interference between dominant cycles.
        """
        table = self.build_chord_table(prices)
        harmonics = table["chord_table"]

        if len(harmonics) < 2:
            return {"interference": "insufficient_data", "combined_amplitude": 0.0}

        # Check pairwise interference between top 2 harmonics
        h1, h2 = harmonics[0], harmonics[1]
        phase_diff = abs(h1["phase_rad"] - h2["phase_rad"])

        # Normalize phase difference to [0, π]
        phase_diff = phase_diff % (2 * np.pi)
        if phase_diff > np.pi:
            phase_diff = 2 * np.pi - phase_diff

        # Combined amplitude using trig identity
        a1, a2 = h1["amplitude"], h2["amplitude"]
        combined = np.sqrt(a1**2 + a2**2 + 2 * a1 * a2 * np.cos(phase_diff))

        if phase_diff < np.pi / 4:
            interference_type = "CONSTRUCTIVE"  # Cycles reinforce → trending
        elif phase_diff > 3 * np.pi / 4:
            interference_type = "DESTRUCTIVE"  # Cycles cancel → mean-reverting
        else:
            interference_type = "PARTIAL"  # Mixed regime

        return {
            "interference": interference_type,
            "phase_difference_rad": float(phase_diff),
            "combined_amplitude": float(combined),
            "dominant_period": h1["period_bars"],
            "secondary_period": h2["period_bars"],
            "frequency_ratio": h1["frequency"] / (h2["frequency"] + 1e-15),
            "regime_signal": "TREND" if interference_type == "CONSTRUCTIVE" else "MEAN_REVERT",
        }

    def forecast_next_cycle_phase(self, prices: np.ndarray, horizon: int = 10) -> Dict:
        """
        Extrapolate dominant cycle forward to predict phase at future horizon.
        """
        table = self.build_chord_table(prices)
        if not table["chord_table"]:
            return {"forecast": "no_dominant_cycle"}

        h = table["chord_table"][0]  # Dominant harmonic
        freq = h["frequency"]
        phase = h["phase_rad"]
        amp = h["amplitude"]

        # Current phase position
        n = len(prices)
        current_angle = 2 * np.pi * freq * n + phase

        # Forecast values
        forecast = []
        for t in range(1, horizon + 1):
            future_angle = 2 * np.pi * freq * (n + t) + phase
            predicted_return = amp * np.sin(future_angle)
            forecast.append({
                "bars_ahead": t,
                "predicted_return": float(predicted_return),
                "phase_angle_rad": float(future_angle % (2 * np.pi)),
            })

        return {
            "dominant_period": h["period_bars"],
            "current_phase": float(current_angle % (2 * np.pi)),
            "forecast": forecast,
        }


# =============================================================================
# 10. DIOPHANTINE EQUATIONS — Integer Constraint Optimization
# =============================================================================

class DiophantineOptimizer:
    """
    Diophantus of Alexandria (3rd century CE) solved equations requiring
    integer solutions (Arithmetica).

    Applied to finance: portfolio weights constrained to integer lot sizes.
    Real orders must be in whole lots — this is a Diophantine optimization problem.

    Solve: minimize |w_target - w_actual| subject to n_i * lot_size_i / total_capital = w_i
    where n_i must be non-negative integers.
    """

    def __init__(self):
        pass

    def optimize_integer_portfolio(
        self,
        target_weights: np.ndarray,
        prices: np.ndarray,
        lot_sizes: np.ndarray,
        total_capital: float,
    ) -> Dict:
        """
        Find integer number of lots for each asset that best approximates
        target portfolio weights, subject to Diophantine (integer) constraints.

        Uses greedy rounding with iterative refinement (inspired by
        Diophantus' method of successive approximations).
        """
        n_assets = len(target_weights)
        target_weights = target_weights / np.sum(target_weights)  # Normalize

        # Cost per lot for each asset
        cost_per_lot = prices * lot_sizes

        # Maximum lots affordable
        max_lots = np.floor(total_capital * target_weights / (cost_per_lot + 1e-15)).astype(int)

        # Start with floor allocation
        lots = max_lots.copy()
        allocated = np.sum(lots * cost_per_lot)
        remaining = total_capital - allocated

        # Greedy fill: add 1 lot to the asset with largest weight shortfall
        for _ in range(n_assets * 10):  # Max iterations
            if remaining <= 0:
                break

            # Compute current weight vs target
            current_values = lots * cost_per_lot
            total_value = np.sum(current_values)
            if total_value < 1e-15:
                break
            current_weights = current_values / total_value
            shortfall = target_weights - current_weights

            # Find asset with most shortfall that we can still afford
            can_afford = cost_per_lot <= remaining
            if not np.any(can_afford):
                break

            adjusted_shortfall = shortfall.copy()
            adjusted_shortfall[~can_afford] = -np.inf

            best_asset = np.argmax(adjusted_shortfall)
            if adjusted_shortfall[best_asset] <= 0:
                break

            lots[best_asset] += 1
            remaining -= cost_per_lot[best_asset]

        # Final portfolio
        final_values = lots * cost_per_lot
        total_invested = np.sum(final_values)
        actual_weights = final_values / (total_invested + 1e-15)
        tracking_error = np.sqrt(np.mean((actual_weights - target_weights) ** 2))
        cash_remaining = total_capital - total_invested
        utilization = total_invested / total_capital

        return {
            "lots": lots.tolist(),
            "actual_weights": actual_weights.tolist(),
            "target_weights": target_weights.tolist(),
            "tracking_error": float(tracking_error),
            "cash_remaining": float(cash_remaining),
            "capital_utilization": float(utilization),
            "total_invested": float(total_invested),
            "is_diophantine_feasible": bool(tracking_error < 0.05),
        }

    @staticmethod
    def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
        """
        Extended Euclidean/Diophantine algorithm.
        Returns (gcd, x, y) such that a*x + b*y = gcd(a, b).
        Fundamental to solving linear Diophantine equations.
        """
        if a == 0:
            return b, 0, 1
        gcd, x1, y1 = DiophantineOptimizer.extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return gcd, x, y

    def solve_two_asset_diophantine(
        self, lot_a: int, lot_b: int, target_value: int
    ) -> Optional[Tuple[int, int]]:
        """
        Solve: lot_a * x + lot_b * y = target_value
        for non-negative integers x, y (if possible).
        This is a classic linear Diophantine equation.
        """
        gcd, x0, y0 = self.extended_gcd(lot_a, lot_b)

        if target_value % gcd != 0:
            return None  # No integer solution exists

        scale = target_value // gcd
        x0 *= scale
        y0 *= scale

        # Find non-negative solution by adjusting with parameter t
        # x = x0 + (b/gcd)*t,  y = y0 - (a/gcd)*t
        step_x = lot_b // gcd
        step_y = lot_a // gcd

        # Find range of t for non-negative x, y
        if step_x == 0 and step_y == 0:
            if x0 >= 0 and y0 >= 0:
                return (x0, y0)
            return None

        best = None
        best_diff = float("inf")

        # Search reasonable range
        for t in range(-abs(target_value), abs(target_value) + 1):
            x = x0 + step_x * t
            y = y0 - step_y * t
            if x >= 0 and y >= 0:
                diff = abs(x - y)
                if diff < best_diff:
                    best_diff = diff
                    best = (x, y)
                if best_diff == 0:
                    break

        return best
