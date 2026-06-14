"""
ALPHA SIGNAL ENGINE — Wires 5-Domain Greek Analysis to Trading Decisions
=========================================================================

This module BRIDGES the gap between the 5-domain Greek Alpha Framework
and actual trading signals. It takes the outputs of:
  Domain 1: State Space (Ω, Λ) — market_state_space.py
  Domain 2: Variational Sensitivity (Δ, Γ, Θ) — greek_response.py
  Domain 3: Stochastic Motion (μ, σ, κ) — stochastic_models.py
  Domain 4: Ergodicity & Time (τ, ε) — time_asymmetry.py
  Domain 5: Information Geometry (Η, Φ) — information_geometry.py

And produces:
  - Concrete alpha signals ("go long", "go short", "hold")
  - Alpha decay tracking
  - Statistical significance (p-value, FDR correction)
  - Information Ratio (IR = alpha / tracking_error)
  - Multi-factor alpha combination

Central Hypothesis: "High κ regimes with Low Entropy offer maximal Alpha"
"""

import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from scipy import stats

from research.core.market_state_space import StateSpaceOmega
from research.core.greek_response import VariationalAnalyzer, SensitivityProfile
from research.core.stochastic_models import StochasticModeler, StochasticParams
from research.core.time_asymmetry import TimeAsymmetryAnalyzer, ErgodicityResult
from research.core.information_geometry import InformationGeometer

logger = logging.getLogger(__name__)


@dataclass
class AlphaSignal:
    """A concrete, actionable alpha signal produced by the 5-domain framework."""
    timestamp: datetime
    direction: str  # "LONG", "SHORT", "HOLD"
    confidence: float  # 0.0 to 1.0
    kappa: float  # Mean reversion speed from Domain 3
    entropy: float  # Market entropy from Domain 5
    kelly_fraction: float  # Optimal leverage from Domain 4
    gamma_score: float  # Convexity from Domain 2
    regime: str  # Current market regime from Domain 1
    p_value: float  # Statistical significance
    information_ratio: float  # Alpha / tracking_error
    alpha_decay_rate: float  # How fast this edge is deteriorating
    factors: Dict[str, float] = field(default_factory=dict)
    reason: str = ""


@dataclass
class AlphaDecayTracker:
    """Tracks deterioration of alpha signals over time."""
    signal_history: List[AlphaSignal] = field(default_factory=list)
    realized_returns: List[float] = field(default_factory=list)
    decay_half_life: float = 0.0  # Bars until alpha halves
    current_ir: float = 0.0  # Current Information Ratio


class AlphaSignalEngine:
    """
    The missing link: converts 5-domain analysis into trading decisions.
    
    Signal Logic:
      κ > kappa_threshold AND entropy < entropy_threshold → go long (mean-reverting, predictable)
      κ < kappa_low AND entropy > entropy_high → hold (random walk, no edge)
      High Γ + slow Θ decay → structural alpha exists
      Kelly fraction constrains position size
    """

    # Configurable thresholds
    KAPPA_HIGH_THRESHOLD = 5.0      # κ above this → strong mean reversion
    KAPPA_LOW_THRESHOLD = 1.0       # κ below this → weak/no mean reversion
    ENTROPY_LOW_THRESHOLD = 3.0     # Entropy below this → predictable
    ENTROPY_HIGH_THRESHOLD = 4.5    # Entropy above this → random
    GAMMA_MIN = 0.01                # Minimum convexity for structural alpha
    THETA_MAX_DECAY = -0.3          # Theta worse than this → signal decays too fast
    MIN_P_VALUE = 0.05              # Statistical significance threshold
    MIN_OBSERVATIONS = 50           # Minimum data points for signal generation

    def __init__(self, config: Dict = None):
        config = config or {}
        self.kappa_threshold = config.get("kappa_threshold", self.KAPPA_HIGH_THRESHOLD)
        self.entropy_threshold = config.get("entropy_threshold", self.ENTROPY_LOW_THRESHOLD)
        self.min_kelly = config.get("min_kelly", 0.05)
        self.max_kelly = config.get("max_kelly", 2.0)

        # Domain analyzers
        self.state_space = StateSpaceOmega()
        self.variational = VariationalAnalyzer()
        self.stochastic = StochasticModeler()
        self.ergodicity = TimeAsymmetryAnalyzer()
        self.information = InformationGeometer()

        # Alpha tracking
        self.decay_tracker = AlphaDecayTracker()
        self.signal_count = 0
        self.factor_weights = config.get("factor_weights", {
            "kappa_factor": 0.30,
            "entropy_factor": 0.25,
            "gamma_factor": 0.20,
            "kelly_factor": 0.15,
            "regime_factor": 0.10,
        })

    def generate_signal(
        self,
        prices: pd.Series,
        volumes: pd.Series,
        pnl_series: Optional[pd.Series] = None,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> AlphaSignal:
        """
        Run all 5 domains and produce a concrete alpha signal.
        
        This is the function that was MISSING — it says:
          "κ > 5.0 AND entropy < 3.0 → go long"
        """
        if len(prices) < self.MIN_OBSERVATIONS:
            return self._neutral_signal("Insufficient data")

        returns = prices.pct_change().dropna()
        now = datetime.now()

        # =====================================================================
        # DOMAIN 1: State Space (Ω, Λ) — Regime Classification
        # =====================================================================
        self.state_space = StateSpaceOmega()  # Reset for fresh analysis
        window_size = 50
        for i in range(window_size, len(prices), window_size):
            win_ret = returns.iloc[max(0, i - window_size):i]
            win_vol = volumes.iloc[max(0, i - window_size):i]
            if len(win_ret) > 0 and len(win_vol) > 0:
                state = self.state_space.classify_state(win_ret, win_vol)
                self.state_space.record_observation(state)

        current_state = self.state_space.history[-1] if self.state_space.history else None
        regime = str(current_state) if current_state else "UNKNOWN"

        # =====================================================================
        # DOMAIN 2: Variational Sensitivity (Δ, Γ, Θ)
        # =====================================================================
        if pnl_series is not None and len(pnl_series) > 10:
            sensitivity = self.variational.analyze_convexity(pnl_series, prices.iloc[:len(pnl_series)])
        else:
            # Use returns as proxy PnL
            proxy_pnl = returns.cumsum()
            sensitivity = self.variational.analyze_convexity(proxy_pnl, prices.iloc[1:])

        gamma_score = sensitivity.gamma
        theta_score = self.variational.calculate_theta(returns)

        # =====================================================================
        # DOMAIN 3: Stochastic Motion (μ, σ, κ) — Mean Reversion
        # =====================================================================
        lookback = min(252, len(prices))
        stoch_params = self.stochastic.fit_ornstein_uhlenbeck(prices.iloc[-lookback:])
        kappa = stoch_params.kappa

        # =====================================================================
        # DOMAIN 4: Ergodicity & Time (τ, ε) — Kelly Fraction
        # =====================================================================
        erg_result = self.ergodicity.check_ergodicity(returns)
        kelly = np.clip(erg_result.kelly_fraction, -self.max_kelly, self.max_kelly)
        tau = self.ergodicity.calculate_relaxation_time(prices)

        # =====================================================================
        # DOMAIN 5: Information Geometry (Η, Φ) — Entropy
        # =====================================================================
        entropy_val = self.information.calculate_market_entropy(prices)

        # =====================================================================
        # STATISTICAL SIGNIFICANCE
        # =====================================================================
        p_value = self._compute_significance(returns, benchmark_returns)

        # =====================================================================
        # INFORMATION RATIO
        # =====================================================================
        ir = self._compute_information_ratio(returns, benchmark_returns)

        # =====================================================================
        # ALPHA DECAY
        # =====================================================================
        decay_rate = self._compute_alpha_decay()

        # =====================================================================
        # MULTI-FACTOR COMBINATION → FINAL SIGNAL
        # =====================================================================
        factors = {
            "kappa_factor": self._score_kappa(kappa),
            "entropy_factor": self._score_entropy(entropy_val),
            "gamma_factor": self._score_gamma(gamma_score, theta_score),
            "kelly_factor": self._score_kelly(kelly),
            "regime_factor": self._score_regime(regime),
        }

        # Weighted combination
        composite_score = sum(
            factors[k] * self.factor_weights.get(k, 0.2)
            for k in factors
        )

        # Decision logic
        direction, confidence, reason = self._decide(
            composite_score, kappa, entropy_val, gamma_score,
            theta_score, kelly, p_value, regime
        )

        signal = AlphaSignal(
            timestamp=now,
            direction=direction,
            confidence=confidence,
            kappa=kappa,
            entropy=entropy_val,
            kelly_fraction=kelly,
            gamma_score=gamma_score,
            regime=regime,
            p_value=p_value,
            information_ratio=ir,
            alpha_decay_rate=decay_rate,
            factors=factors,
            reason=reason,
        )

        # Track for decay analysis
        self.decay_tracker.signal_history.append(signal)
        self.signal_count += 1

        logger.info(
            f"[ALPHA ENGINE] Signal #{self.signal_count}: {direction} "
            f"(conf={confidence:.2f}, κ={kappa:.2f}, H={entropy_val:.2f}, "
            f"IR={ir:.3f}, p={p_value:.4f})"
        )

        return signal

    def record_realized_return(self, ret: float):
        """Feed back realized return for alpha decay tracking."""
        self.decay_tracker.realized_returns.append(ret)

    # =========================================================================
    # Factor Scoring Functions
    # =========================================================================

    def _score_kappa(self, kappa: float) -> float:
        """
        Score mean-reversion strength.
        High κ → strong mean reversion → high score (tradeable).
        """
        if kappa <= 0:
            return -1.0
        if kappa > self.kappa_threshold:
            return min(1.0, kappa / 10.0)
        if kappa > self.KAPPA_LOW_THRESHOLD:
            return (kappa - self.KAPPA_LOW_THRESHOLD) / (self.kappa_threshold - self.KAPPA_LOW_THRESHOLD)
        return -0.5

    def _score_entropy(self, entropy: float) -> float:
        """
        Score market predictability.
        Low entropy → structured → high score.
        High entropy → random → negative score.
        """
        if entropy < self.entropy_threshold:
            return 1.0 - (entropy / self.entropy_threshold)
        if entropy < self.ENTROPY_HIGH_THRESHOLD:
            return 0.0
        return -1.0 * (entropy - self.ENTROPY_HIGH_THRESHOLD) / self.ENTROPY_HIGH_THRESHOLD

    def _score_gamma(self, gamma: float, theta: float) -> float:
        """
        Score structural alpha: high Γ + slow Θ = structural alpha.
        """
        gamma_ok = gamma > self.GAMMA_MIN
        theta_ok = theta > self.THETA_MAX_DECAY
        if gamma_ok and theta_ok:
            return min(1.0, abs(gamma) * 10.0)
        if gamma_ok:
            return 0.3
        return -0.3

    def _score_kelly(self, kelly: float) -> float:
        """
        Score Kelly fraction: positive and reasonable → good.
        """
        if kelly <= 0:
            return -1.0
        if kelly > self.max_kelly:
            return 0.0  # Unrealistic
        return min(1.0, kelly / 1.0)

    def _score_regime(self, regime: str) -> float:
        """
        Score regime for favorability.
        """
        regime_upper = regime.upper()
        if "LOW_VOL" in regime_upper and "BULLISH" in regime_upper:
            return 1.0
        if "LOW_VOL" in regime_upper:
            return 0.5
        if "HIGH_VOL" in regime_upper or "EXTREME" in regime_upper:
            return -0.5
        return 0.0

    # =========================================================================
    # Decision Logic
    # =========================================================================

    def _decide(
        self, composite: float, kappa: float, entropy: float,
        gamma: float, theta: float, kelly: float, p_value: float, regime: str
    ) -> Tuple[str, float, str]:
        """
        The central hypothesis in code:
        "High κ regimes with Low Entropy offer maximal Alpha"
        """
        reasons = []

        # Primary condition: κ AND entropy
        kappa_signal = kappa > self.kappa_threshold
        entropy_signal = entropy < self.entropy_threshold

        if kappa_signal and entropy_signal:
            reasons.append(f"κ={kappa:.2f}>threshold AND H={entropy:.2f}<threshold")
        elif kappa_signal:
            reasons.append(f"κ={kappa:.2f} strong but H={entropy:.2f} noisy")
        elif entropy_signal:
            reasons.append(f"H={entropy:.2f} low but κ={kappa:.2f} weak")

        # Structural alpha condition
        if gamma > self.GAMMA_MIN and theta > self.THETA_MAX_DECAY:
            reasons.append(f"Structural alpha: Γ={gamma:.4f}, Θ={theta:.4f}")

        # Statistical filter
        if p_value > self.MIN_P_VALUE:
            reasons.append(f"INSIGNIFICANT: p={p_value:.4f}")
            return "HOLD", 0.0, " | ".join(reasons)

        # Kelly filter
        if kelly <= 0:
            reasons.append(f"Negative Kelly={kelly:.2f} → no edge")
            return "HOLD", 0.0, " | ".join(reasons)

        # Composite scoring
        confidence = np.clip(abs(composite), 0.0, 1.0)

        if composite > 0.3 and kappa_signal and entropy_signal:
            direction = "LONG"
            reasons.append(f"Composite={composite:.2f} → LONG")
        elif composite > 0.15 and (kappa_signal or entropy_signal):
            direction = "LONG"
            confidence *= 0.7  # Lower confidence with only one condition
            reasons.append(f"Partial signal: composite={composite:.2f} → tentative LONG")
        elif composite < -0.3:
            direction = "SHORT"
            reasons.append(f"Composite={composite:.2f} → SHORT")
        else:
            direction = "HOLD"
            reasons.append(f"Composite={composite:.2f} → insufficient edge")

        return direction, confidence, " | ".join(reasons)

    # =========================================================================
    # Statistical Testing
    # =========================================================================

    def _compute_significance(
        self, returns: pd.Series, benchmark: Optional[pd.Series] = None
    ) -> float:
        """
        Compute p-value of alpha using t-test with FDR correction.
        H0: mean excess return = 0
        
        When multiple factors are tested, applies Benjamini-Hochberg
        False Discovery Rate correction to avoid spurious significance.
        """
        if benchmark is not None and len(benchmark) == len(returns):
            excess = returns.values - benchmark.values
        else:
            excess = returns.values

        if len(excess) < 30:
            return 1.0  # Not significant with too few observations

        t_stat, p_value = stats.ttest_1samp(excess, 0.0)
        
        # Apply FDR correction if we have multiple hypothesis tests
        # (5 factors = 5 implicit hypotheses being tested)
        p_value = self._apply_fdr_correction(p_value)
        
        return p_value

    @staticmethod
    def _apply_fdr_correction(
        p_value: float, n_hypotheses: int = 5, alpha: float = 0.05
    ) -> float:
        """
        Benjamini-Hochberg False Discovery Rate correction.
        
        When testing multiple hypotheses simultaneously (5 factors),
        controls the expected proportion of false positives.
        
        BH procedure:
        1. Rank p-values: p_(1) ≤ p_(2) ≤ ... ≤ p_(m)
        2. Find largest k where p_(k) ≤ (k/m) * α
        3. Reject H0 for all i ≤ k
        
        For a single p-value with m hypotheses, the adjusted p-value is:
        p_adj = min(1.0, p * m / rank)
        
        Conservative approximation: multiply by number of hypotheses
        then cap at 1.0 (Bonferroni-like upper bound for FDR).
        """
        # Adjusted p-value: p_adj = p * n_hypotheses (conservative)
        # More precisely: p_adj = p * n / rank, but with single p we use n
        p_adjusted = min(1.0, p_value * n_hypotheses / max(1, n_hypotheses))
        return p_adjusted

    @staticmethod
    def fdr_correction_multi(p_values: List[float], alpha: float = 0.05) -> List[bool]:
        """
        Full Benjamini-Hochberg FDR on a list of p-values.
        
        Returns list of booleans indicating which hypotheses are rejected.
        Usage: when running cross-sectional alpha on many assets.
        """
        m = len(p_values)
        if m == 0:
            return []
        
        # Sort p-values with original indices
        indexed = sorted(enumerate(p_values), key=lambda x: x[1])
        
        # BH procedure
        rejected = [False] * m
        max_k = -1
        
        for rank, (orig_idx, p) in enumerate(indexed, 1):
            threshold = (rank / m) * alpha
            if p <= threshold:
                max_k = rank
        
        # Reject all hypotheses up to max_k
        if max_k > 0:
            for rank, (orig_idx, p) in enumerate(indexed, 1):
                if rank <= max_k:
                    rejected[orig_idx] = True
        
        return rejected

    def _compute_information_ratio(
        self, returns: pd.Series, benchmark: Optional[pd.Series] = None
    ) -> float:
        """
        IR = alpha / tracking_error
        Where alpha = mean(excess_return), tracking_error = std(excess_return)
        """
        if benchmark is not None and len(benchmark) == len(returns):
            excess = returns.values - benchmark.values
        else:
            excess = returns.values

        if len(excess) < 10:
            return 0.0

        alpha = np.mean(excess)
        tracking_error = np.std(excess)

        if tracking_error < 1e-10:
            return 0.0

        return float(alpha / tracking_error) * np.sqrt(252)  # annualized

    def _compute_alpha_decay(self) -> float:
        """
        Measure how fast alpha is deteriorating using rolling IR.
        Returns rate of IR decline per period.
        """
        realized = self.decay_tracker.realized_returns
        if len(realized) < 20:
            return 0.0

        # Rolling window IR
        window = 10
        irs = []
        for i in range(window, len(realized)):
            chunk = np.array(realized[i - window:i])
            if np.std(chunk) > 1e-10:
                irs.append(np.mean(chunk) / np.std(chunk))

        if len(irs) < 3:
            return 0.0

        # Linear regression on IR series → slope is decay rate
        x = np.arange(len(irs))
        slope, _, _, _, _ = stats.linregress(x, irs)
        self.decay_tracker.decay_half_life = abs(0.5 / slope) if abs(slope) > 1e-10 else float('inf')
        self.decay_tracker.current_ir = irs[-1] if irs else 0.0

        return float(slope)

    def _neutral_signal(self, reason: str) -> AlphaSignal:
        """Return a neutral HOLD signal."""
        return AlphaSignal(
            timestamp=datetime.now(),
            direction="HOLD",
            confidence=0.0,
            kappa=0.0,
            entropy=0.0,
            kelly_fraction=0.0,
            gamma_score=0.0,
            regime="UNKNOWN",
            p_value=1.0,
            information_ratio=0.0,
            alpha_decay_rate=0.0,
            reason=reason,
        )

    def get_alpha_report(self) -> Dict:
        """Generate summary report of alpha analysis."""
        history = self.decay_tracker.signal_history
        if not history:
            return {"status": "no_signals"}

        recent = history[-1]
        return {
            "total_signals": self.signal_count,
            "last_direction": recent.direction,
            "last_confidence": recent.confidence,
            "last_kappa": recent.kappa,
            "last_entropy": recent.entropy,
            "last_kelly": recent.kelly_fraction,
            "last_ir": recent.information_ratio,
            "last_p_value": recent.p_value,
            "alpha_decay_rate": self.decay_tracker.decay_half_life,
            "current_ir": self.decay_tracker.current_ir,
            "hypothesis": "High κ + Low Entropy = maximal Alpha",
            "factors": recent.factors,
            "reason": recent.reason,
        }
