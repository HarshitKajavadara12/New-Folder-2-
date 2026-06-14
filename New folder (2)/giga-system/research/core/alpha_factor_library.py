"""
ALPHA FACTOR LIBRARY — Modular Alpha Factors from Greek Domain Outputs
=======================================================================

Addresses Missing Concepts 4.1-4.6:
  4.1 — Alpha Factor Library (modular, pluggable factors)
  4.2 — Alpha Research Pipeline (systematic hypothesis testing)
  4.3 — Alpha Decay (how long signals persist)
  4.4 — Alpha Combination (optimal multi-domain weighting)
  4.5 — Cross-Asset Greek Alpha (multi-asset domain analysis)
  4.6 — Alpha Attribution (decompose P&L by factor)
"""

import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from datetime import datetime
from scipy import stats

logger = logging.getLogger(__name__)


# =============================================================================
# 4.1 — ALPHA FACTOR LIBRARY
# =============================================================================

@dataclass
class AlphaFactorResult:
    """Result from a single alpha factor."""
    name: str
    value: float  # Raw alpha score
    zscore: float  # Standardized
    ic: float  # Information Coefficient
    signal: str  # "LONG", "SHORT", "NEUTRAL"
    confidence: float  # 0-1
    half_life_bars: float = float("inf")  # Bars until decay to 50%


class AlphaFactor:
    """Base class for Greek-domain alpha factors."""

    def __init__(self, name: str, threshold: float = 0.0, scale: float = 1.0):
        self.name = name
        self.threshold = threshold
        self.scale = scale
        self.history: List[float] = []

    def compute(self, features: Dict[str, float]) -> float:
        """Compute raw alpha score. Override in subclass."""
        raise NotImplementedError

    def score(self, features: Dict[str, float]) -> AlphaFactorResult:
        """Full scoring with z-score and signal."""
        raw = self.compute(features)
        self.history.append(raw)

        # Z-score
        if len(self.history) > 10:
            mu = np.mean(self.history[-100:])
            sigma = np.std(self.history[-100:]) + 1e-15
            zscore = (raw - mu) / sigma
        else:
            zscore = raw

        # IC (auto-correlation as proxy if we have history)
        if len(self.history) > 20:
            ic = np.corrcoef(self.history[-21:-1], self.history[-20:])[0, 1]
        else:
            ic = 0.0

        # Signal
        if zscore > 1.0:
            signal = "LONG"
            confidence = min(1.0, abs(zscore) / 3.0)
        elif zscore < -1.0:
            signal = "SHORT"
            confidence = min(1.0, abs(zscore) / 3.0)
        else:
            signal = "NEUTRAL"
            confidence = 0.0

        # Decay estimation
        hl = self._estimate_half_life() if len(self.history) > 30 else float("inf")

        return AlphaFactorResult(
            name=self.name, value=float(raw), zscore=float(zscore),
            ic=float(ic), signal=signal, confidence=float(confidence),
            half_life_bars=float(hl),
        )

    def _estimate_half_life(self) -> float:
        """4.3 — Estimate alpha decay half-life."""
        series = np.array(self.history[-100:])
        if len(series) < 10:
            return float("inf")
        y = np.diff(series)
        x = series[:-1] - np.mean(series)
        if np.std(x) < 1e-15:
            return float("inf")
        slope = np.sum(x * y) / (np.sum(x**2) + 1e-15)
        if slope >= 0:
            return float("inf")
        return float(-np.log(2) / slope)


class KappaAlpha(AlphaFactor):
    """Buy when mean-reversion speed is high (κ > threshold)."""

    def __init__(self, threshold: float = 5.0, scale: float = 2.0):
        super().__init__("kappa_alpha", threshold, scale)

    def compute(self, features: Dict[str, float]) -> float:
        kappa = features.get("ou_kappa", features.get("kappa", 0.0))
        return float(np.tanh((kappa - self.threshold) / self.scale))


class EntropyAlpha(AlphaFactor):
    """Go short when entropy is very high (market chaos), long when low."""

    def __init__(self, threshold: float = 3.5, scale: float = 1.5):
        super().__init__("entropy_alpha", threshold, scale)

    def compute(self, features: Dict[str, float]) -> float:
        entropy = features.get("entropy", features.get("shannon_entropy", 3.0))
        return float(-np.tanh((entropy - self.threshold) / self.scale))


class GammaAlpha(AlphaFactor):
    """Exploit convexity: positive gamma = accelerating returns."""

    def __init__(self, threshold: float = 0.0, scale: float = 0.01):
        super().__init__("gamma_alpha", threshold, scale)

    def compute(self, features: Dict[str, float]) -> float:
        gamma = features.get("gamma", 0.0)
        return float(np.tanh((gamma - self.threshold) / self.scale))


class ErgodicityAlpha(AlphaFactor):
    """Trade when ergodicity gap is large (time avg ≠ ensemble avg)."""

    def __init__(self, threshold: float = 0.01, scale: float = 0.05):
        super().__init__("ergodicity_alpha", threshold, scale)

    def compute(self, features: Dict[str, float]) -> float:
        gap = features.get("ergodicity_gap", 0.0)
        kelly = features.get("kelly_fraction", 0.0)
        return float(np.sign(kelly) * np.tanh(abs(gap) / self.scale))


class MomentumAlpha(AlphaFactor):
    """Trend-following based on realized momentum."""

    def __init__(self, lookback: int = 20):
        super().__init__("momentum_alpha", 0.0, 1.0)
        self.lookback = lookback

    def compute(self, features: Dict[str, float]) -> float:
        momentum = features.get("momentum", features.get("trend_ann", 0.0))
        return float(np.tanh(momentum / 0.1))


class VolRegimeAlpha(AlphaFactor):
    """Trade vol regime transitions: low→high = risk off, high→low = risk on."""

    def __init__(self):
        super().__init__("vol_regime_alpha", 0.0, 1.0)
        self.prev_vol = None

    def compute(self, features: Dict[str, float]) -> float:
        vol = features.get("volatility_ann", features.get("sigma", 0.2))
        if self.prev_vol is None:
            self.prev_vol = vol
            return 0.0
        vol_change = vol - self.prev_vol
        self.prev_vol = vol
        return float(-np.tanh(vol_change / 0.05))  # Negative: rising vol = bearish


# =============================================================================
# 4.2 — ALPHA RESEARCH PIPELINE
# =============================================================================

@dataclass
class HypothesisTest:
    """Result of testing a Greek-derived hypothesis."""
    hypothesis: str
    factor_name: str
    n_observations: int
    mean_return_when_active: float
    mean_return_when_inactive: float
    t_statistic: float
    p_value: float
    significant: bool
    sharpe_ratio: float
    information_ratio: float


class AlphaResearchPipeline:
    """
    Systematic testing of Greek-derived hypotheses against historical data.
    """

    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        self.test_results: List[HypothesisTest] = []
        self.factors = [
            KappaAlpha(), EntropyAlpha(), GammaAlpha(),
            ErgodicityAlpha(), MomentumAlpha(), VolRegimeAlpha(),
        ]

    def test_hypothesis(
        self,
        factor: AlphaFactor,
        features_series: List[Dict[str, float]],
        returns: np.ndarray,
        hypothesis: str = "",
    ) -> HypothesisTest:
        """
        Test if a factor generates statistically significant alpha.
        """
        n = min(len(features_series), len(returns))
        signals = []
        for feat in features_series[:n]:
            result = factor.score(feat)
            signals.append(result.value)

        signals = np.array(signals)
        rets = returns[:n]

        # Split into active (signal > 0) and inactive
        active_mask = signals > 0
        if np.sum(active_mask) < 5 or np.sum(~active_mask) < 5:
            return HypothesisTest(
                hypothesis=hypothesis or f"{factor.name}_generates_alpha",
                factor_name=factor.name, n_observations=n,
                mean_return_when_active=0.0, mean_return_when_inactive=0.0,
                t_statistic=0.0, p_value=1.0, significant=False,
                sharpe_ratio=0.0, information_ratio=0.0,
            )

        active_returns = rets[active_mask]
        inactive_returns = rets[~active_mask]

        # Two-sample t-test
        t_stat, p_val = stats.ttest_ind(active_returns, inactive_returns)

        # Sharpe & IR
        strategy_returns = signals * rets
        sr = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-15) * np.sqrt(252)
        # IR: mean active/inactive spread normalised by strategy return std
        spread = float(np.mean(active_returns)) - float(np.mean(inactive_returns))
        ir = spread / (np.std(strategy_returns) + 1e-15) * np.sqrt(252)

        result = HypothesisTest(
            hypothesis=hypothesis or f"{factor.name}_generates_alpha",
            factor_name=factor.name, n_observations=n,
            mean_return_when_active=float(np.mean(active_returns)),
            mean_return_when_inactive=float(np.mean(inactive_returns)),
            t_statistic=float(t_stat), p_value=float(p_val),
            significant=bool(p_val < self.significance_level),
            sharpe_ratio=float(sr), information_ratio=float(ir),
        )
        self.test_results.append(result)
        return result

    def run_all_tests(
        self, features_series: List[Dict[str, float]], returns: np.ndarray
    ) -> List[HypothesisTest]:
        """Test all registered factors."""
        results = []
        for factor in self.factors:
            hypothesis = f"High {factor.name} predicts positive returns"
            result = self.test_hypothesis(factor, features_series, returns, hypothesis)
            results.append(result)
        return results

    def fdr_correction(self, results: List[HypothesisTest]) -> List[HypothesisTest]:
        """
        Apply Benjamini-Hochberg FDR correction for multiple testing.
        """
        p_values = [r.p_value for r in results]
        n = len(p_values)
        if n == 0:
            return results

        sorted_indices = np.argsort(p_values)
        adjusted = np.zeros(n)
        for rank, idx in enumerate(sorted_indices, 1):
            adjusted[idx] = p_values[idx] * n / rank

        # Enforce monotonicity
        for i in range(n - 2, -1, -1):
            adjusted[sorted_indices[i]] = min(
                adjusted[sorted_indices[i]],
                adjusted[sorted_indices[i + 1]] if i + 1 < n else 1.0
            )

        for i, result in enumerate(results):
            result.p_value = min(float(adjusted[i]), 1.0)
            result.significant = result.p_value < self.significance_level

        return results


# =============================================================================
# 4.4 — ALPHA COMBINATION
# =============================================================================

class AlphaCombiner:
    """
    Optimally combine signals from all factors/domains.
    Uses ridge regression on Information Coefficient.
    """

    def __init__(self, factors: Optional[List[AlphaFactor]] = None):
        self.factors = factors or [
            KappaAlpha(), EntropyAlpha(), GammaAlpha(),
            ErgodicityAlpha(), MomentumAlpha(), VolRegimeAlpha(),
        ]
        self.weights = np.ones(len(self.factors)) / len(self.factors)
        self.fitted = False

    def fit(self, features_series: List[Dict[str, float]], returns: np.ndarray,
            ridge_alpha: float = 1.0):
        """
        Fit optimal combination weights using ridge regression on IC.
        """
        n = min(len(features_series), len(returns))
        n_factors = len(self.factors)

        # Compute factor scores
        X = np.zeros((n, n_factors))
        for t in range(n):
            for j, factor in enumerate(self.factors):
                result = factor.score(features_series[t])
                X[t, j] = result.value

        y = returns[:n]

        # Ridge regression
        XtX = X.T @ X + ridge_alpha * np.eye(n_factors)
        Xty = X.T @ y
        try:
            self.weights = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            self.weights = np.ones(n_factors) / n_factors

        # Normalize weights
        total = np.sum(np.abs(self.weights)) + 1e-15
        self.weights = self.weights / total
        self.fitted = True

        return {
            "weights": {f.name: float(w) for f, w in zip(self.factors, self.weights)},
            "r_squared": float(1 - np.var(y - X @ self.weights) / (np.var(y) + 1e-15)),
        }

    def combine(self, features: Dict[str, float]) -> Dict:
        """Produce combined alpha signal from all factors."""
        scores = []
        details = {}
        for i, factor in enumerate(self.factors):
            result = factor.score(features)
            scores.append(result.value * self.weights[i])
            details[factor.name] = {
                "raw": float(result.value),
                "weight": float(self.weights[i]),
                "contribution": float(result.value * self.weights[i]),
            }

        combined = float(np.sum(scores))
        if combined > 0.1:
            signal = "LONG"
        elif combined < -0.1:
            signal = "SHORT"
        else:
            signal = "NEUTRAL"

        return {
            "combined_alpha": combined,
            "signal": signal,
            "confidence": float(min(1.0, abs(combined))),
            "factor_details": details,
        }


# =============================================================================
# 4.5 — CROSS-ASSET GREEK ALPHA
# =============================================================================

class CrossAssetGreekAlpha:
    """
    Apply 5-domain analysis to multiple assets simultaneously.
    Find cross-asset Greek harmonies.
    """

    def __init__(self):
        self.combiner = AlphaCombiner()

    def analyze_multi_asset(
        self, asset_features: Dict[str, Dict[str, float]]
    ) -> Dict:
        """
        Run alpha combination for each asset, then rank.
        """
        asset_scores = {}
        for asset, features in asset_features.items():
            result = self.combiner.combine(features)
            asset_scores[asset] = result

        # Rank by combined alpha
        ranked = sorted(
            asset_scores.items(),
            key=lambda x: x[1]["combined_alpha"],
            reverse=True,
        )

        n = len(ranked)
        long_assets = [a for a, s in ranked[:max(1, n // 4)] if s["signal"] == "LONG"]
        short_assets = [a for a, s in ranked[-(max(1, n // 4)):] if s["signal"] == "SHORT"]

        # Cross-asset harmony: correlation of domain scores across assets
        kappas = [f.get("kappa", 5.0) for f in asset_features.values()]
        entropies = [f.get("entropy", 3.0) for f in asset_features.values()]
        harmony = abs(np.corrcoef(kappas, entropies)[0, 1]) if len(kappas) > 2 else 0.0

        return {
            "asset_scores": asset_scores,
            "rankings": {a: i + 1 for i, (a, _) in enumerate(ranked)},
            "long_basket": long_assets,
            "short_basket": short_assets,
            "cross_asset_harmony": float(harmony),
            "dispersion": float(np.std([s["combined_alpha"] for s in asset_scores.values()])),
        }


# =============================================================================
# 4.6 — ALPHA ATTRIBUTION
# =============================================================================

class AlphaAttribution:
    """
    Decompose P&L into contributions from each factor/domain.
    """

    def __init__(self, factors: Optional[List[AlphaFactor]] = None):
        self.factors = factors or [
            KappaAlpha(), EntropyAlpha(), GammaAlpha(),
            ErgodicityAlpha(), MomentumAlpha(), VolRegimeAlpha(),
        ]
        self.attribution_history: List[Dict] = []

    def attribute(
        self,
        features_series: List[Dict[str, float]],
        returns: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Decompose total P&L into factor contributions.
        Attribution = weight_i * score_i * return_t, summed over time.
        """
        n = min(len(features_series), len(returns))
        n_factors = len(self.factors)

        if weights is None:
            weights = np.ones(n_factors) / n_factors

        factor_pnl = np.zeros(n_factors)
        factor_scores = np.zeros((n, n_factors))

        for t in range(n):
            for j, factor in enumerate(self.factors):
                result = factor.score(features_series[t])
                factor_scores[t, j] = result.value

        for j in range(n_factors):
            factor_pnl[j] = np.sum(factor_scores[:n, j] * weights[j] * returns[:n])

        total_pnl = np.sum(factor_pnl)
        residual = np.sum(returns[:n]) - total_pnl

        attribution = {
            "total_pnl": float(total_pnl),
            "residual_pnl": float(residual),
            "factor_contributions": {},
        }

        for j, factor in enumerate(self.factors):
            pct = factor_pnl[j] / (abs(total_pnl) + 1e-15) * 100
            attribution["factor_contributions"][factor.name] = {
                "pnl": float(factor_pnl[j]),
                "pct_of_total": float(pct),
                "avg_score": float(np.mean(factor_scores[:n, j])),
                "hit_rate": float(np.mean((factor_scores[:n, j] * returns[:n]) > 0)),
            }

        self.attribution_history.append(attribution)
        return attribution
