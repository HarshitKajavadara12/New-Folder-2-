"""
DOMAIN 1: STATE SPACE (Λ, Ω)
Greek Concepts:
- Ω (Omega) → All possible market states (Global Topology)
- Λ (Lambda) → Allowed transitions (Transition Matrix)

Implementation:
Market is modeled as a directed graph where nodes are (regime) tuples.
Alpha = Identification of rare but high-probability transitions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from enum import Enum

class VolatilityRegime(Enum):
    LOW = "LOW_VOL"
    NORMAL = "NORMAL_VOL"
    HIGH = "HIGH_VOL"
    EXTREME = "EXTREME_VOL"

class TrendRegime(Enum):
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    BULLISH = "BULLISH"

class LiquidityRegime(Enum):
    LIQUID = "LIQUID"
    ILLIQUID = "ILLIQUID"
    FRAGMENTED = "FRAGMENTED"

@dataclass(frozen=True)
class MarketState:
    """A single node in the State Space Graph Ω."""
    volatility: VolatilityRegime
    trend: TrendRegime
    liquidity: LiquidityRegime
    
    def __str__(self):
        return f"[{self.volatility.value}|{self.trend.value}|{self.liquidity.value}]"

class StateSpaceOmega:
    """
    Represents the topological space of the market.
    """
    def __init__(self):
        self.states: Set[MarketState] = set()
        self.transitions: Dict[MarketState, Dict[MarketState, float]] = {} # Adjacency Matrix
        self.history: List[MarketState] = []
        
    def classify_state(self, returns: pd.Series, volume: pd.Series) -> MarketState:
        """Map raw data to discrete State Node."""
        # 1. Volatility (Annualized std dev)
        vol = returns.std() * np.sqrt(252)
        if vol < 0.2: vol_regime = VolatilityRegime.LOW
        elif vol < 0.5: vol_regime = VolatilityRegime.NORMAL
        elif vol < 1.0: vol_regime = VolatilityRegime.HIGH
        else: vol_regime = VolatilityRegime.EXTREME
        
        # 2. Trend (Simple moving average slope or return sign)
        total_ret = returns.sum()
        if total_ret > 0.05: trend_regime = TrendRegime.BULLISH
        elif total_ret < -0.05: trend_regime = TrendRegime.BEARISH
        else: trend_regime = TrendRegime.NEUTRAL
        
        # 3. Liquidity (Volume consistency)
        vol_consistency = volume.std() / volume.mean()
        if vol_consistency < 0.5: liq_regime = LiquidityRegime.LIQUID
        elif vol_consistency < 1.5: liq_regime = LiquidityRegime.ILLIQUID
        else: liq_regime = LiquidityRegime.FRAGMENTED
        
        return MarketState(vol_regime, trend_regime, liq_regime)

    def record_observation(self, state: MarketState):
        """Update Ω with new observation."""
        self.states.add(state)
        self.history.append(state)
        
        # Update Lambda (Transition Probabilities)
        if len(self.history) > 1:
            prev = self.history[-2]
            curr = self.history[-1]
            
            if prev not in self.transitions:
                self.transitions[prev] = {}
            
            self.transitions[prev][curr] = self.transitions[prev].get(curr, 0) + 1

    def get_lambda_matrix(self) -> Dict[str, Dict[str, float]]:
        """Compute transition probabilities Λ."""
        matrix = {}
        for start_node, neighbors in self.transitions.items():
            total = sum(neighbors.values())
            matrix[str(start_node)] = {
                str(end): count / total 
                for end, count in neighbors.items()
            }
        return matrix
