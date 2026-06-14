"""
GIGA SYSTEM - Feedback & Adaptive Learning
==========================================

The Learning Engine.
Closes the loop between Live Execution outcomes and Research/Reducer parameters.
"""

from datetime import datetime
from typing import Dict, Any, List, Tuple
from enum import Enum, auto
import math
import logging

logger = logging.getLogger(__name__)

class AdaptiveEngine:
    """
    Evaluates execution quality and adjusts strategy parameters.
    Includes guardrails to prevent runaway adaptation.
    """
    
    # Guardrail constants
    LOSS_CUT_FACTOR = 0.95      # Cut risk by 5% on loss (asymmetric — slower than gain)
    GAIN_GROW_FACTOR = 1.03     # Grow risk by 3% on gain (conservative)
    MIN_LIMIT_RATIO = 0.30      # Floor: never shrink below 30% of original
    MAX_LIMIT_RATIO = 3.0       # Ceiling: never grow beyond 3× original
    LOOKBACK_WINDOW = 20        # Minimum trades before adapting
    COOLDOWN_TRADES = 5         # Min trades between adaptations
    
    def __init__(self, config: Dict[str, Any]):
        self.history: List[Dict[str, Any]] = []
        self.config = config
        self.loss_threshold = config.get("LOSS_THRESHOLD", -500.0)
        
        # Track original limit for floor/ceiling calculations
        self._original_limit: float = config.get("RISK_LIMIT", 10000.0)
        self._trades_since_adaptation: int = 0
        self._adaptation_count: int = 0

    def evaluate_signal(self, signal: str, pnl: float, confidence: float) -> float:
        """
        Record the outcome of a signal.
        Returns a 'quality_score' (positive = good, negative = bad).
        """
        quality_score = pnl
        
        self.history.append({
            "timestamp": datetime.now(),
            "signal": signal,
            "pnl": pnl,
            "confidence": confidence,
            "quality_score": quality_score
        })
        self._trades_since_adaptation += 1
        return quality_score

    def adjust_parameters(self, current_params: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        """
        Proposes adjustments to Reducer parameters based on recent history.
        
        Guardrails:
        - Minimum lookback window (20 trades) before any adaptation
        - Cooldown between adaptations (5 trades minimum)
        - Asymmetric scaling (cut 5% on loss, grow 3% on gain)
        - Floor at 30% of original limit
        - Ceiling at 3× original limit
        """
        adjusted_params = current_params.copy()
        
        # Don't adapt with too little data
        if len(self.history) < self.LOOKBACK_WINDOW:
            return adjusted_params, False
        
        # Cooldown: don't adapt too frequently
        if self._trades_since_adaptation < self.COOLDOWN_TRADES:
            return adjusted_params, False
        
        # Use proper lookback window
        recent_history = self.history[-self.LOOKBACK_WINDOW:]
        cumulative_pnl = sum(h["pnl"] for h in recent_history)
        
        current_limit = adjusted_params.get("RISK_LIMIT", self._original_limit)
        
        # Calculate floor and ceiling from original
        floor = self._original_limit * self.MIN_LIMIT_RATIO
        ceiling = self._original_limit * self.MAX_LIMIT_RATIO
        
        adapted = False
        
        if cumulative_pnl < self.loss_threshold:
            # Defensive: cut risk by 5% (asymmetric — favor safety)
            new_limit = max(current_limit * self.LOSS_CUT_FACTOR, floor)
            adjusted_params["RISK_LIMIT"] = new_limit
            adapted = True
            logger.info(
                f"ADAPTIVE: Defensive cut. PnL={cumulative_pnl:.2f}, "
                f"Limit: {current_limit:.0f} → {new_limit:.0f} "
                f"(floor={floor:.0f})"
            )
            
        elif cumulative_pnl > abs(self.loss_threshold):
            # Offensive: grow risk by 3% (conservative growth)
            new_limit = min(current_limit * self.GAIN_GROW_FACTOR, ceiling)
            adjusted_params["RISK_LIMIT"] = new_limit
            adapted = True
            logger.info(
                f"ADAPTIVE: Offensive grow. PnL={cumulative_pnl:.2f}, "
                f"Limit: {current_limit:.0f} → {new_limit:.0f} "
                f"(ceiling={ceiling:.0f})"
            )
        
        if adapted:
            self._trades_since_adaptation = 0
            self._adaptation_count += 1
            
        return adjusted_params, adapted

    def feedback_summary(self) -> Dict[str, Any]:
        """
        Returns rolling metrics for dashboard.
        """
        total_signals = len(self.history)
        if total_signals == 0:
            return {"total_signals": 0, "cum_pnl": 0.0}
            
        cum_pnl = sum(h["pnl"] for h in self.history)
        return {"total_signals": total_signals, "cum_pnl": cum_pnl}

class CapitalRegime(Enum):
    SEED = auto()        # < 50k
    GROWTH = auto()      # 50k - 500k
    SCALE = auto()       # 500k - 5M
    INSTITUTION = auto() # > 5M

class CapitalRegimeEngine:
    """
    PHASE 10: Capital Regime Engine
    Classifies capital state to govern behavior.
    """
    def __init__(self):
        self.current_regime = CapitalRegime.SEED

    def update(self, equity: float) -> CapitalRegime:
        if equity < 50000:
            self.current_regime = CapitalRegime.SEED
        elif equity < 500000:
            self.current_regime = CapitalRegime.GROWTH
        elif equity < 5000000:
            self.current_regime = CapitalRegime.SCALE
        else:
            self.current_regime = CapitalRegime.INSTITUTION
        return self.current_regime

class PositionSizer:
    """
    PHASE 10: Sub-linear Position Sizing
    Rule: Size grows logarithmically with capital, never linearly.
    """
    def __init__(self, base_unit: float = 0.1): # 0.1 BTC base
        self.base_unit = base_unit
        self.k_factor = 10000.0 # Scaling constant

    def calculate_size(self, equity: float, regime: CapitalRegime) -> float:
        # P10 Formula: size = base * log10(1 + equity/K)
        # Prevents explosion at high equity
        
        # Regime Multipliers (Conservative)
        regime_mult = 1.0
        if regime == CapitalRegime.SEED: regime_mult = 1.0
        elif regime == CapitalRegime.GROWTH: regime_mult = 0.8 # Slow down
        elif regime == CapitalRegime.SCALE: regime_mult = 0.5 # Major brake
        elif regime == CapitalRegime.INSTITUTION: regime_mult = 0.2 # Preservation
        
        # Logarithmic Scaling
        raw_size = self.base_unit * math.log10(1 + (equity / self.k_factor))
        
        # Apply brakes
        final_size = raw_size * regime_mult
        
        # Hard limits
        if final_size < 0.001: final_size = 0.001 # Min size
        
        return round(final_size, 4)
