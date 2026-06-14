"""
MARKET MICROSTRUCTURE ALPHA
============================

Implements order-flow and microstructure-based alpha signals:

1. VPIN (Volume-Synchronized Probability of Informed Trading)
   - Easley, Lopez de Prado, O'Hara (2012)
   - Measures toxic flow / information asymmetry
   
2. Order Flow Imbalance (OFI)
   - Measures pressure from buyer vs seller aggression
   
3. Kyle's Lambda (λ)
   - Kyle (1985) — permanent price impact per unit of order flow
   
4. Amihud Illiquidity Ratio
   - |return| / volume — higher = more illiquid
   
5. Roll Spread Estimator
   - Roll (1984) — estimates effective bid-ask spread from serial covariance

This was the remaining gap: "No order flow, no VPIN"
"""

import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class MicrostructureSignal:
    """Microstructure-based alpha signal."""
    timestamp: datetime
    vpin: float  # Volume-Sync Probability of Informed Trading
    ofi: float  # Order Flow Imbalance
    kyle_lambda: float  # Price impact coefficient
    amihud_illiquidity: float  # Amihud ratio
    roll_spread: float  # Estimated bid-ask spread
    toxicity_regime: str  # "LOW_TOXICITY", "NORMAL", "HIGH_TOXICITY", "EXTREME_TOXICITY"
    direction: str  # "LONG", "SHORT", "HOLD"
    confidence: float
    reason: str = ""


class VPINCalculator:
    """
    Volume-Synchronized Probability of Informed Trading.
    
    VPIN approximates PIN (Probability of Informed Trading) without
    requiring trade-by-trade buyer/seller classification.
    
    Method:
    1. Classify volume into buy/sell using bulk volume classification (BVC)
    2. Group trades into volume buckets of size V
    3. VPIN = Σ|V_buy - V_sell| / (n * V)
    
    High VPIN → toxic flow → likely adverse selection → avoid providing liquidity.
    """

    def __init__(self, bucket_size: int = 50, n_buckets: int = 50):
        """
        Args:
            bucket_size: Volume per bucket (in units)
            n_buckets: Number of buckets for VPIN estimation window
        """
        self.bucket_size = bucket_size
        self.n_buckets = n_buckets

    def calculate(
        self, prices: pd.Series, volumes: pd.Series
    ) -> pd.Series:
        """
        Calculate VPIN time series.
        
        Uses Bulk Volume Classification (BVC):
        - Returns standardized by σ → CDF gives buy probability
        - V_buy = V * Φ(ΔP / σ), V_sell = V - V_buy
        """
        if len(prices) < self.n_buckets + 10:
            return pd.Series(dtype=float)

        returns = prices.pct_change().dropna()
        vol = returns.rolling(20).std()

        # BVC: classify each bar's volume
        standardized = returns / (vol + 1e-10)
        # Using normal CDF approximation
        buy_pct = self._normal_cdf(standardized)
        
        buy_volume = volumes * buy_pct
        sell_volume = volumes * (1 - buy_pct)

        # Volume bucket aggregation
        cum_vol = volumes.cumsum()
        total_vol = volumes.sum()
        
        if total_vol == 0:
            return pd.Series(0.0, index=prices.index)

        # Adaptive bucketing: use rolling window instead of exact volume buckets
        window = self.n_buckets
        imbalance = np.abs(buy_volume - sell_volume)
        
        vpin_series = imbalance.rolling(window).sum() / (volumes.rolling(window).sum() + 1e-10)

        return vpin_series.reindex(prices.index)

    @staticmethod
    def _normal_cdf(x: pd.Series) -> pd.Series:
        """Approximate standard normal CDF: Φ(x) ≈ 1/(1 + exp(-1.7*x))."""
        return 1.0 / (1.0 + np.exp(-1.7 * x.clip(-10, 10)))


class OrderFlowImbalance:
    """
    Order Flow Imbalance (OFI).
    
    Measures the net pressure from buyer vs seller aggression using
    changes in best bid/ask queue sizes.
    
    OFI = Σ (ΔBid_size * I(bid↑) - ΔAsk_size * I(ask↓))
    
    Simplified version using OHLCV:
    OFI ≈ sign(close - open) * volume / avg_volume
    """

    def __init__(self, lookback: int = 20):
        self.lookback = lookback

    def calculate(
        self,
        prices: pd.Series,
        volumes: pd.Series,
        highs: Optional[pd.Series] = None,
        lows: Optional[pd.Series] = None,
    ) -> pd.Series:
        """Calculate OFI proxy from OHLCV data."""
        if len(prices) < self.lookback:
            return pd.Series(0.0, index=prices.index)

        # Direction: sign of close-to-close return
        returns = prices.pct_change()
        direction = np.sign(returns)

        # Normalize volume
        avg_vol = volumes.rolling(self.lookback).mean()
        rel_vol = volumes / (avg_vol + 1e-10)

        # OFI = direction * relative_volume
        ofi = direction * rel_vol

        # Smooth
        ofi_smooth = ofi.rolling(5).mean()

        # If we have high/low, use them for additional info
        if highs is not None and lows is not None:
            # Bar range as fraction of price
            bar_range = (highs - lows) / (prices + 1e-10)
            # Close location within bar: (close - low) / (high - low)
            close_loc = (prices - lows) / (highs - lows + 1e-10)
            # If close is near high with high volume → strong buying
            buying_pressure = close_loc * rel_vol
            selling_pressure = (1 - close_loc) * rel_vol
            ofi_smooth = (buying_pressure - selling_pressure).rolling(5).mean()

        return ofi_smooth

    def get_imbalance_signal(self, ofi: pd.Series, threshold: float = 1.5) -> str:
        """Convert OFI to trading signal."""
        if len(ofi.dropna()) == 0:
            return "HOLD"
        
        current = ofi.iloc[-1]
        if np.isnan(current):
            return "HOLD"
            
        if current > threshold:
            return "LONG"  # Strong buying pressure
        elif current < -threshold:
            return "SHORT"  # Strong selling pressure
        return "HOLD"


class KyleLambda:
    """
    Kyle's Lambda (λ) — Permanent Price Impact.
    
    From Kyle (1985): ΔP = λ * OFI + ε
    
    λ measures how much price moves per unit of order flow.
    High λ → illiquid, information-rich → adverse selection risk.
    Low λ → liquid, safe to trade.
    
    Estimated via regression: returns = α + λ * signed_volume + ε
    """

    def __init__(self, window: int = 50):
        self.window = window

    def calculate(
        self, prices: pd.Series, volumes: pd.Series
    ) -> pd.Series:
        """Calculate rolling Kyle's Lambda."""
        returns = prices.pct_change()
        signed_vol = np.sign(returns) * volumes

        lambdas = pd.Series(np.nan, index=prices.index)

        for i in range(self.window, len(prices)):
            ret_window = returns.iloc[i - self.window:i].values
            sv_window = signed_vol.iloc[i - self.window:i].values

            # Remove NaN
            mask = ~(np.isnan(ret_window) | np.isnan(sv_window))
            if mask.sum() < 20:
                continue

            ret_clean = ret_window[mask]
            sv_clean = sv_window[mask]

            # OLS: return = α + λ * signed_volume
            if np.std(sv_clean) < 1e-10:
                continue

            x = np.column_stack([sv_clean, np.ones(len(sv_clean))])
            try:
                beta = np.linalg.lstsq(x, ret_clean, rcond=None)[0]
                lambdas.iloc[i] = abs(beta[0])
            except np.linalg.LinAlgError:
                continue

        return lambdas


class AmihudIlliquidity:
    """
    Amihud (2002) Illiquidity Ratio.
    
    ILLIQ = (1/N) * Σ |return_t| / volume_t
    
    Higher ratio → more illiquid → larger price impact per dollar traded.
    """

    def __init__(self, window: int = 20):
        self.window = window

    def calculate(
        self, prices: pd.Series, volumes: pd.Series
    ) -> pd.Series:
        """Calculate rolling Amihud illiquidity ratio."""
        abs_returns = prices.pct_change().abs()
        # Avoid division by zero
        dollar_volume = volumes * prices
        ratio = abs_returns / (dollar_volume + 1e-10)
        return ratio.rolling(self.window).mean() * 1e6  # Scale for readability


class RollSpread:
    """
    Roll (1984) Effective Spread Estimator.
    
    Spread = 2 * sqrt(-Cov(ΔP_t, ΔP_{t-1}))
    
    Uses serial covariance of price changes to estimate
    effective bid-ask spread without needing quote data.
    """

    def __init__(self, window: int = 20):
        self.window = window

    def calculate(self, prices: pd.Series) -> pd.Series:
        """Calculate rolling Roll spread estimate."""
        dp = prices.diff()
        dp_lag = dp.shift(1)

        spreads = pd.Series(np.nan, index=prices.index)

        for i in range(self.window + 1, len(prices)):
            window_dp = dp.iloc[i - self.window:i]
            window_lag = dp_lag.iloc[i - self.window:i]
            
            mask = ~(window_dp.isna() | window_lag.isna())
            if mask.sum() < 10:
                continue

            cov = np.cov(window_dp[mask], window_lag[mask])[0, 1]

            if cov < 0:
                spreads.iloc[i] = 2.0 * np.sqrt(-cov) / (prices.iloc[i] + 1e-10)
            else:
                spreads.iloc[i] = 0.0  # No negative serial covariance

        return spreads


class MicrostructureAlphaEngine:
    """
    Complete microstructure alpha engine combining all components.
    
    Produces a MicrostructureSignal that classifies the current
    regime and generates trading signals based on order flow.
    """

    def __init__(self, config: Dict = None):
        config = config or {}
        self.vpin_calc = VPINCalculator(
            bucket_size=config.get("bucket_size", 50),
            n_buckets=config.get("n_buckets", 50),
        )
        self.ofi_calc = OrderFlowImbalance(
            lookback=config.get("ofi_lookback", 20),
        )
        self.kyle_calc = KyleLambda(
            window=config.get("kyle_window", 50),
        )
        self.amihud_calc = AmihudIlliquidity(
            window=config.get("amihud_window", 20),
        )
        self.roll_calc = RollSpread(
            window=config.get("roll_window", 20),
        )
        
        self.vpin_toxic_threshold = config.get("vpin_toxic_threshold", 0.7)
        self.vpin_extreme_threshold = config.get("vpin_extreme_threshold", 0.85)
        self.ofi_threshold = config.get("ofi_threshold", 1.5)
        
        self.signal_history: List[MicrostructureSignal] = []

    def generate_signal(
        self,
        prices: pd.Series,
        volumes: pd.Series,
        highs: Optional[pd.Series] = None,
        lows: Optional[pd.Series] = None,
    ) -> MicrostructureSignal:
        """
        Generate microstructure-based alpha signal.
        
        Args:
            prices: Close prices
            volumes: Trading volumes
            highs: Optional high prices
            lows: Optional low prices
            
        Returns:
            MicrostructureSignal with VPIN, OFI, Kyle's λ, etc.
        """
        # Calculate all microstructure metrics
        vpin_series = self.vpin_calc.calculate(prices, volumes)
        ofi_series = self.ofi_calc.calculate(prices, volumes, highs, lows)
        kyle_series = self.kyle_calc.calculate(prices, volumes)
        amihud_series = self.amihud_calc.calculate(prices, volumes)
        roll_series = self.roll_calc.calculate(prices)

        # Current values
        vpin = float(vpin_series.iloc[-1]) if len(vpin_series.dropna()) > 0 else 0.0
        ofi = float(ofi_series.iloc[-1]) if len(ofi_series.dropna()) > 0 else 0.0
        kyle_lambda = float(kyle_series.iloc[-1]) if len(kyle_series.dropna()) > 0 else 0.0
        amihud = float(amihud_series.iloc[-1]) if len(amihud_series.dropna()) > 0 else 0.0
        roll_spread = float(roll_series.iloc[-1]) if len(roll_series.dropna()) > 0 else 0.0

        # Handle NaN
        vpin = 0.0 if np.isnan(vpin) else vpin
        ofi = 0.0 if np.isnan(ofi) else ofi
        kyle_lambda = 0.0 if np.isnan(kyle_lambda) else kyle_lambda
        amihud = 0.0 if np.isnan(amihud) else amihud
        roll_spread = 0.0 if np.isnan(roll_spread) else roll_spread

        # Classify toxicity regime
        toxicity_regime = self._classify_toxicity(vpin, kyle_lambda, amihud)

        # Generate direction signal
        direction, confidence, reason = self._decide(
            vpin, ofi, kyle_lambda, amihud, roll_spread, toxicity_regime
        )

        signal = MicrostructureSignal(
            timestamp=datetime.now(),
            vpin=vpin,
            ofi=ofi,
            kyle_lambda=kyle_lambda,
            amihud_illiquidity=amihud,
            roll_spread=roll_spread,
            toxicity_regime=toxicity_regime,
            direction=direction,
            confidence=confidence,
            reason=reason,
        )

        self.signal_history.append(signal)
        logger.info(
            f"[MICROSTRUCTURE] {direction}(conf={confidence:.2f}) | "
            f"VPIN={vpin:.3f}, OFI={ofi:.3f}, λ={kyle_lambda:.6f}, "
            f"Amihud={amihud:.4f}, Spread={roll_spread:.6f} | "
            f"Regime={toxicity_regime}"
        )

        return signal

    def _classify_toxicity(
        self, vpin: float, kyle_lambda: float, amihud: float
    ) -> str:
        """Classify current toxicity/liquidity regime."""
        if vpin > self.vpin_extreme_threshold:
            return "EXTREME_TOXICITY"
        elif vpin > self.vpin_toxic_threshold:
            return "HIGH_TOXICITY"
        elif vpin > 0.4:
            return "NORMAL"
        else:
            return "LOW_TOXICITY"

    def _decide(
        self,
        vpin: float,
        ofi: float,
        kyle_lambda: float,
        amihud: float,
        roll_spread: float,
        regime: str,
    ) -> Tuple[str, float, str]:
        """Generate trading signal from microstructure metrics."""
        reasons = []

        # Rule 1: Extreme toxicity → HOLD (don't trade into toxic flow)
        if regime == "EXTREME_TOXICITY":
            reasons.append(f"VPIN={vpin:.3f} → extreme toxicity, avoid")
            return "HOLD", 0.0, " | ".join(reasons)

        # Rule 2: High toxicity → reduce confidence
        toxicity_penalty = 1.0
        if regime == "HIGH_TOXICITY":
            toxicity_penalty = 0.5
            reasons.append(f"VPIN={vpin:.3f} → high toxicity, reduced confidence")

        # Rule 3: OFI direction
        if abs(ofi) > self.ofi_threshold:
            if ofi > 0:
                direction = "LONG"
                reasons.append(f"OFI={ofi:.3f} → strong buying pressure")
            else:
                direction = "SHORT"
                reasons.append(f"OFI={ofi:.3f} → strong selling pressure")

            confidence = min(1.0, abs(ofi) / 3.0) * toxicity_penalty
            return direction, confidence, " | ".join(reasons)

        # Rule 4: Low toxicity + no directional signal → safe to provide liquidity
        if regime == "LOW_TOXICITY":
            reasons.append(f"VPIN={vpin:.3f} → low toxicity, safe to provide liquidity")
            # In low-toxicity regime, slight mean-reversion tendency
            return "HOLD", 0.0, " | ".join(reasons)

        reasons.append("No strong microstructure signal")
        return "HOLD", 0.0, " | ".join(reasons)

    def get_report(self) -> Dict:
        """Summary report."""
        if not self.signal_history:
            return {"status": "no_signals"}
        
        last = self.signal_history[-1]
        return {
            "total_signals": len(self.signal_history),
            "last_vpin": last.vpin,
            "last_ofi": last.ofi,
            "last_kyle_lambda": last.kyle_lambda,
            "last_amihud": last.amihud_illiquidity,
            "last_roll_spread": last.roll_spread,
            "last_regime": last.toxicity_regime,
            "last_direction": last.direction,
            "last_confidence": last.confidence,
        }
