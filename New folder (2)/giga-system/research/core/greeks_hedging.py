"""
GREEKS HEDGING ENGINE — Delta-Neutral, Gamma Scalping, Vega Trading
====================================================================

Addresses Missing Concept 3.5: Greeks-based hedging strategies.
  - Delta-hedge portfolio using computed Greeks
  - Gamma scalp strategy
  - Vega trade around volatility events

Also addresses 3.6: Options Strategy Builder
  - Multi-leg strategy construction (straddles, strangles, butterflies, etc.)
  - Greek aggregation across legs
  - P&L profile computation
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from scipy.stats import norm

logger = logging.getLogger(__name__)


@dataclass
class OptionLeg:
    """Single option leg in a strategy."""
    strike: float
    expiry: float  # Years
    option_type: str  # "call" or "put"
    quantity: int  # Positive = long, negative = short
    premium: float = 0.0
    iv: float = 0.25


@dataclass
class OptionGreeks:
    """Greeks for a single option or aggregated position."""
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0


@dataclass
class HedgeAction:
    """A specific hedging action to take."""
    action: str  # "BUY_UNDERLYING", "SELL_UNDERLYING", "BUY_OPTION", "SELL_OPTION"
    quantity: float
    instrument: str
    reason: str
    urgency: str = "normal"  # "low", "normal", "high", "critical"


@dataclass
class StrategyProfile:
    """Complete profile of a multi-leg options strategy."""
    name: str
    legs: List[OptionLeg]
    net_greeks: OptionGreeks
    max_profit: float
    max_loss: float
    breakeven_points: List[float]
    net_premium: float
    pnl_at_expiry: List[Tuple[float, float]]  # (price, pnl) pairs


class BSGreeksCalculator:
    """Black-Scholes Greeks calculator for hedging engine."""

    def __init__(self, r: float = 0.05):
        self.r = r

    def _d1d2(self, S: float, K: float, T: float, sigma: float) -> Tuple[float, float]:
        T = max(T, 1e-10)
        sigma = max(sigma, 1e-10)
        d1 = (np.log(S / K) + (self.r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2

    def price(self, S: float, K: float, T: float, sigma: float, option_type: str = "call") -> float:
        d1, d2 = self._d1d2(S, K, T, sigma)
        if option_type == "call":
            return S * norm.cdf(d1) - K * np.exp(-self.r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-self.r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    def greeks(self, S: float, K: float, T: float, sigma: float, option_type: str = "call") -> OptionGreeks:
        d1, d2 = self._d1d2(S, K, T, sigma)
        sqrt_t = np.sqrt(max(T, 1e-10))

        # Delta
        if option_type == "call":
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1

        # Gamma (same for call/put)
        gamma = norm.pdf(d1) / (S * sigma * sqrt_t + 1e-15)

        # Theta
        common = -(S * norm.pdf(d1) * sigma) / (2 * sqrt_t + 1e-15)
        if option_type == "call":
            theta = common - self.r * K * np.exp(-self.r * T) * norm.cdf(d2)
        else:
            theta = common + self.r * K * np.exp(-self.r * T) * norm.cdf(-d2)
        theta /= 252  # Daily theta

        # Vega
        vega = S * norm.pdf(d1) * sqrt_t / 100  # Per 1% IV move

        # Rho
        if option_type == "call":
            rho = K * T * np.exp(-self.r * T) * norm.cdf(d2) / 100
        else:
            rho = -K * T * np.exp(-self.r * T) * norm.cdf(-d2) / 100

        return OptionGreeks(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)


class GreeksHedgingEngine:
    """
    3.5 — Greeks-based hedging engine.
    Delta-hedge, gamma scalp, vega trade.
    """

    def __init__(self, risk_free_rate: float = 0.05):
        self.calc = BSGreeksCalculator(r=risk_free_rate)
        self.position_greeks = OptionGreeks()
        self.hedge_history: List[HedgeAction] = []

    def compute_position_greeks(
        self, legs: List[OptionLeg], S: float
    ) -> OptionGreeks:
        """Aggregate Greeks across all position legs."""
        total = OptionGreeks()
        for leg in legs:
            g = self.calc.greeks(S, leg.strike, leg.expiry, leg.iv, leg.option_type)
            total.delta += g.delta * leg.quantity
            total.gamma += g.gamma * leg.quantity
            total.theta += g.theta * leg.quantity
            total.vega += g.vega * leg.quantity
            total.rho += g.rho * leg.quantity
        self.position_greeks = total
        return total

    def delta_hedge(self, legs: List[OptionLeg], S: float, current_hedge_shares: float = 0.0) -> HedgeAction:
        """
        Delta-hedge: make portfolio delta-neutral by trading underlying.
        """
        greeks = self.compute_position_greeks(legs, S)
        total_delta = greeks.delta + current_hedge_shares
        hedge_needed = -total_delta

        if abs(hedge_needed) < 0.01:
            action = HedgeAction(
                action="NO_ACTION", quantity=0, instrument="underlying",
                reason=f"Delta already neutral ({total_delta:.4f})"
            )
        elif hedge_needed > 0:
            action = HedgeAction(
                action="BUY_UNDERLYING", quantity=abs(hedge_needed), instrument="underlying",
                reason=f"Portfolio delta={total_delta:.4f}, buying {abs(hedge_needed):.2f} shares",
                urgency="high" if abs(total_delta) > 0.5 else "normal"
            )
        else:
            action = HedgeAction(
                action="SELL_UNDERLYING", quantity=abs(hedge_needed), instrument="underlying",
                reason=f"Portfolio delta={total_delta:.4f}, selling {abs(hedge_needed):.2f} shares",
                urgency="high" if abs(total_delta) > 0.5 else "normal"
            )

        self.hedge_history.append(action)
        return action

    def gamma_scalp_signal(self, legs: List[OptionLeg], S: float,
                            realized_vol: float, implied_vol: float) -> Dict:
        """
        Gamma scalping: profit from gamma when realized vol > implied vol.
        Delta-hedge frequently to capture gamma P&L.
        """
        greeks = self.compute_position_greeks(legs, S)
        vol_edge = realized_vol - implied_vol

        # Daily gamma P&L estimate
        daily_gamma_pnl = 0.5 * greeks.gamma * (S * realized_vol / np.sqrt(252))**2
        daily_theta_cost = greeks.theta

        net_daily = daily_gamma_pnl + daily_theta_cost  # theta is negative for long gamma

        return {
            "gamma": float(greeks.gamma),
            "theta_daily": float(daily_theta_cost),
            "vol_edge": float(vol_edge),
            "daily_gamma_pnl": float(daily_gamma_pnl),
            "net_daily_pnl": float(net_daily),
            "signal": "SCALP" if vol_edge > 0.02 and greeks.gamma > 0 else "HOLD",
            "rehedge_frequency": "hourly" if abs(greeks.gamma) > 0.01 else "daily",
        }

    def vega_trade_signal(self, legs: List[OptionLeg], S: float,
                           iv_forecast: float, current_iv: float) -> Dict:
        """
        Vega trade: profit from IV changes around events.
        """
        greeks = self.compute_position_greeks(legs, S)
        iv_change_expected = iv_forecast - current_iv

        vega_pnl = greeks.vega * iv_change_expected * 100  # Vega per 1% move

        return {
            "vega": float(greeks.vega),
            "iv_change_expected": float(iv_change_expected),
            "vega_pnl_estimate": float(vega_pnl),
            "signal": "LONG_VOL" if iv_forecast > current_iv * 1.05 else
                      "SHORT_VOL" if iv_forecast < current_iv * 0.95 else "NEUTRAL",
        }


class OptionsStrategyBuilder:
    """
    3.6 — Multi-leg options strategy construction.
    Build straddles, strangles, butterflies, iron condors, calendars, diagonals.
    """

    def __init__(self, risk_free_rate: float = 0.05):
        self.calc = BSGreeksCalculator(r=risk_free_rate)

    def straddle(self, S: float, K: float, T: float, iv: float) -> StrategyProfile:
        """Long straddle: buy ATM call + buy ATM put."""
        call_leg = OptionLeg(strike=K, expiry=T, option_type="call", quantity=1, iv=iv)
        put_leg = OptionLeg(strike=K, expiry=T, option_type="put", quantity=1, iv=iv)
        call_leg.premium = self.calc.price(S, K, T, iv, "call")
        put_leg.premium = self.calc.price(S, K, T, iv, "put")
        return self._build_profile("Long Straddle", [call_leg, put_leg], S)

    def strangle(self, S: float, K_put: float, K_call: float, T: float, iv: float) -> StrategyProfile:
        """Long strangle: buy OTM put + buy OTM call."""
        call_leg = OptionLeg(strike=K_call, expiry=T, option_type="call", quantity=1, iv=iv)
        put_leg = OptionLeg(strike=K_put, expiry=T, option_type="put", quantity=1, iv=iv)
        call_leg.premium = self.calc.price(S, K_call, T, iv, "call")
        put_leg.premium = self.calc.price(S, K_put, T, iv, "put")
        return self._build_profile("Long Strangle", [call_leg, put_leg], S)

    def butterfly(self, S: float, K_low: float, K_mid: float, K_high: float,
                  T: float, iv: float) -> StrategyProfile:
        """Long butterfly: buy 1 low call, sell 2 mid calls, buy 1 high call."""
        legs = [
            OptionLeg(strike=K_low, expiry=T, option_type="call", quantity=1, iv=iv),
            OptionLeg(strike=K_mid, expiry=T, option_type="call", quantity=-2, iv=iv),
            OptionLeg(strike=K_high, expiry=T, option_type="call", quantity=1, iv=iv),
        ]
        for leg in legs:
            leg.premium = self.calc.price(S, leg.strike, T, iv, leg.option_type)
        return self._build_profile("Long Butterfly", legs, S)

    def iron_condor(self, S: float, K1: float, K2: float, K3: float, K4: float,
                    T: float, iv: float) -> StrategyProfile:
        """Iron condor: sell put spread + sell call spread."""
        legs = [
            OptionLeg(strike=K1, expiry=T, option_type="put", quantity=1, iv=iv),   # Buy OTM put
            OptionLeg(strike=K2, expiry=T, option_type="put", quantity=-1, iv=iv),  # Sell put
            OptionLeg(strike=K3, expiry=T, option_type="call", quantity=-1, iv=iv), # Sell call
            OptionLeg(strike=K4, expiry=T, option_type="call", quantity=1, iv=iv),  # Buy OTM call
        ]
        for leg in legs:
            leg.premium = self.calc.price(S, leg.strike, T, iv, leg.option_type)
        return self._build_profile("Iron Condor", legs, S)

    def calendar_spread(self, S: float, K: float, T_near: float, T_far: float,
                        iv: float) -> StrategyProfile:
        """Calendar spread: sell near-term, buy far-term at same strike."""
        legs = [
            OptionLeg(strike=K, expiry=T_near, option_type="call", quantity=-1, iv=iv),
            OptionLeg(strike=K, expiry=T_far, option_type="call", quantity=1, iv=iv),
        ]
        for leg in legs:
            leg.premium = self.calc.price(S, leg.strike, leg.expiry, iv, leg.option_type)
        return self._build_profile("Calendar Spread", legs, S)

    def vertical_spread(self, S: float, K_buy: float, K_sell: float,
                        T: float, iv: float, spread_type: str = "bull_call") -> StrategyProfile:
        """Vertical spread (bull call, bear put, etc.)."""
        if spread_type == "bull_call":
            legs = [
                OptionLeg(strike=K_buy, expiry=T, option_type="call", quantity=1, iv=iv),
                OptionLeg(strike=K_sell, expiry=T, option_type="call", quantity=-1, iv=iv),
            ]
            name = "Bull Call Spread"
        else:
            legs = [
                OptionLeg(strike=K_sell, expiry=T, option_type="put", quantity=-1, iv=iv),
                OptionLeg(strike=K_buy, expiry=T, option_type="put", quantity=1, iv=iv),
            ]
            name = "Bear Put Spread"
        for leg in legs:
            leg.premium = self.calc.price(S, leg.strike, T, iv, leg.option_type)
        return self._build_profile(name, legs, S)

    def _build_profile(self, name: str, legs: List[OptionLeg], S: float) -> StrategyProfile:
        """Build complete strategy profile with Greeks, P&L, breakevens."""
        # Net premium
        net_premium = sum(leg.premium * leg.quantity for leg in legs)

        # Aggregate Greeks
        net_greeks = OptionGreeks()
        for leg in legs:
            g = self.calc.greeks(S, leg.strike, leg.expiry, leg.iv, leg.option_type)
            net_greeks.delta += g.delta * leg.quantity
            net_greeks.gamma += g.gamma * leg.quantity
            net_greeks.theta += g.theta * leg.quantity
            net_greeks.vega += g.vega * leg.quantity

        # P&L at expiry
        price_range = np.linspace(S * 0.5, S * 1.5, 200)
        pnl_curve = []
        for price in price_range:
            pnl = -net_premium  # Start with premium paid/received
            for leg in legs:
                if leg.option_type == "call":
                    intrinsic = max(price - leg.strike, 0)
                else:
                    intrinsic = max(leg.strike - price, 0)
                pnl += intrinsic * leg.quantity
            pnl_curve.append((float(price), float(pnl)))

        pnl_values = [p[1] for p in pnl_curve]
        max_profit = max(pnl_values)
        max_loss = min(pnl_values)

        # Breakeven points
        breakevens = []
        for i in range(1, len(pnl_curve)):
            if pnl_curve[i - 1][1] * pnl_curve[i][1] < 0:
                # Linear interpolation
                x1, y1 = pnl_curve[i - 1]
                x2, y2 = pnl_curve[i]
                be = x1 + (x2 - x1) * (-y1) / (y2 - y1 + 1e-15)
                breakevens.append(float(be))

        return StrategyProfile(
            name=name, legs=legs, net_greeks=net_greeks,
            max_profit=float(max_profit), max_loss=float(max_loss),
            breakeven_points=breakevens, net_premium=float(net_premium),
            pnl_at_expiry=pnl_curve,
        )

    def recommend_strategy(self, S: float, iv_current: float, iv_forecast: float,
                            direction_view: str = "neutral", T: float = 0.25) -> StrategyProfile:
        """
        Recommend optimal strategy based on market view.
        direction_view: "bullish", "bearish", "neutral"
        """
        K = S  # ATM

        if direction_view == "neutral":
            if iv_forecast > iv_current * 1.1:
                return self.straddle(S, K, T, iv_current)  # Long vol
            elif iv_forecast < iv_current * 0.9:
                return self.iron_condor(S, S * 0.95, S * 0.97, S * 1.03, S * 1.05, T, iv_current)
            else:
                return self.butterfly(S, S * 0.95, K, S * 1.05, T, iv_current)
        elif direction_view == "bullish":
            return self.vertical_spread(S, S * 0.98, S * 1.05, T, iv_current, "bull_call")
        else:  # bearish
            return self.vertical_spread(S, S * 1.02, S * 0.95, T, iv_current, "bear_put")
