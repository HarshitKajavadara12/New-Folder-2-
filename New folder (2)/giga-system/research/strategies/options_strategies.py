"""
GIGA SYSTEM - Options Trading Strategies
Greeks-based strategies with volatility surface analysis
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from .base import Strategy, Signal, Side

# Import core mathematics
import sys
sys.path.append('..')
try:
    from core.black_scholes import black_scholes_price, black_scholes_greeks
    from core.greeks import Greeks
    from core.implied_volatility import implied_volatility_newton
except ImportError:
    # Minimal fallbacks so the module still loads without core package
    import math

    def _bs_d1(S, K, r, sigma, T):
        return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))

    def _norm_cdf(x):
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def black_scholes_price(S, K, r, sigma, T, option_type='call'):
        if T <= 0 or sigma <= 0:
            return max(S - K, 0) if option_type == 'call' else max(K - S, 0)
        d1 = _bs_d1(S, K, r, sigma, T)
        d2 = d1 - sigma * math.sqrt(T)
        if option_type == 'call':
            return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
        return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)

    def black_scholes_greeks(S, K, r, sigma, T, option_type='call'):
        d1 = _bs_d1(S, K, r, sigma, T) if T > 0 and sigma > 0 else 0.0
        delta = _norm_cdf(d1) if option_type == 'call' else _norm_cdf(d1) - 1
        return {'delta': delta, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0, 'rho': 0.0}

    class Greeks:
        pass

    def implied_volatility_newton(price, S, K, r, T, option_type='call'):
        return 0.20  # Default IV when solver unavailable


class OptionType(Enum):
    """Option type enumeration."""
    CALL = "call"
    PUT = "put"


@dataclass
class OptionContract:
    """Option contract specification."""
    symbol: str
    underlying: str
    strike: float
    expiry: datetime
    option_type: OptionType
    multiplier: float = 100.0
    
    @property
    def is_call(self) -> bool:
        return self.option_type == OptionType.CALL
    
    @property
    def is_put(self) -> bool:
        return self.option_type == OptionType.PUT


@dataclass
class OptionQuote:
    """Option market quote with Greeks."""
    contract: OptionContract
    bid: float
    ask: float
    mid: float
    last: float
    volume: int
    open_interest: int
    implied_vol: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    underlying_price: float
    timestamp: datetime


class DeltaHedgingStrategy(Strategy):
    """
    Delta-Neutral Options Strategy.
    
    Concept:
    - Sell options to collect premium (theta decay)
    - Hedge delta exposure with underlying
    - Profit from time decay while managing directional risk
    
    Position:
    - Short options (short gamma, positive theta)
    - Delta-hedged with underlying
    
    Rebalancing:
    - When delta exceeds threshold, rebalance
    - Frequency based on gamma and time to expiry
    
    Risk Management:
    - Maximum gamma exposure
    - Vega limits (volatility risk)
    - Early exit if IV spikes
    """
    
    def __init__(self,
                 underlying: str,
                 delta_threshold: float = 0.1,
                 max_gamma: float = 0.05,
                 max_vega: float = 0.1,
                 vol_spike_threshold: float = 0.5,
                 risk_free_rate: float = 0.05):
        """
        Initialize delta hedging strategy.
        
        Parameters
        ----------
        underlying : str
            Underlying symbol.
        delta_threshold : float
            Delta threshold for rebalancing (default 0.1).
        max_gamma : float
            Maximum gamma exposure (default 0.05).
        max_vega : float
            Maximum vega exposure (default 0.1).
        vol_spike_threshold : float
            IV increase threshold for exit (default 0.5 = 50%).
        risk_free_rate : float
            Risk-free interest rate.
        """
        super().__init__(
            name="Delta_Hedging",
            symbols=[underlying],
            delta_threshold=delta_threshold
        )
        
        self.underlying = underlying
        self.delta_threshold = delta_threshold
        self.max_gamma = max_gamma
        self.max_vega = max_vega
        self.vol_spike_threshold = vol_spike_threshold
        self.risk_free_rate = risk_free_rate
        
        # Position state
        self._option_positions: List[Dict] = []  # Short option contracts
        self._hedge_position: float = 0  # Underlying shares
        self._portfolio_delta: float = 0
        self._portfolio_gamma: float = 0
        self._portfolio_vega: float = 0
        self._portfolio_theta: float = 0
        
        # IV tracking
        self._entry_iv: float = 0
        self._current_iv: float = 0
    
    def initialize(self, **kwargs):
        """Initialize strategy."""
        super().initialize(**kwargs)
        self._option_positions = []
        self._hedge_position = 0
        self._portfolio_delta = 0
    
    def calculate_portfolio_greeks(self, underlying_price: float,
                                   options: List[OptionQuote]) -> Dict[str, float]:
        """
        Calculate aggregate portfolio Greeks.
        
        Parameters
        ----------
        underlying_price : float
            Current underlying price.
        options : list
            List of OptionQuote objects in portfolio.
        
        Returns
        -------
        dict
            Portfolio Greeks.
        """
        total_delta = self._hedge_position  # Start with underlying position
        total_gamma = 0.0
        total_vega = 0.0
        total_theta = 0.0
        
        for opt in options:
            pos_size = -1  # We're short options
            multiplier = opt.contract.multiplier
            
            total_delta += pos_size * opt.delta * multiplier
            total_gamma += pos_size * opt.gamma * multiplier
            total_vega += pos_size * opt.vega * multiplier
            total_theta += pos_size * opt.theta * multiplier
        
        return {
            'delta': total_delta,
            'gamma': total_gamma,
            'vega': total_vega,
            'theta': total_theta
        }
    
    def calculate_hedge_shares(self, target_delta: float = 0) -> float:
        """
        Calculate shares needed to achieve target delta.
        
        Shares = Target Delta - Current Option Delta
        
        Parameters
        ----------
        target_delta : float
            Target portfolio delta (default 0 = neutral).
        
        Returns
        -------
        float
            Number of shares to buy (positive) or sell (negative).
        """
        option_delta = self._portfolio_delta - self._hedge_position
        shares_needed = target_delta - option_delta
        return shares_needed
    
    def should_rebalance(self) -> Tuple[bool, str]:
        """
        Determine if delta hedge needs rebalancing.
        
        Returns
        -------
        tuple
            (should_rebalance, reason)
        """
        if abs(self._portfolio_delta) > self.delta_threshold:
            return True, "delta_threshold"
        
        if abs(self._portfolio_gamma) > self.max_gamma:
            return True, "gamma_limit"
        
        if abs(self._portfolio_vega) > self.max_vega:
            return True, "vega_limit"
        
        # IV spike check
        if self._entry_iv > 0:
            iv_change = (self._current_iv - self._entry_iv) / self._entry_iv
            if iv_change > self.vol_spike_threshold:
                return True, "iv_spike"
        
        return False, ""
    
    def on_bar(self, timestamp: datetime, bars: Dict[str, Dict[str, float]]) -> List[Signal]:
        """Process bar and generate hedging signals based on underlying price movement."""
        if self.underlying not in bars:
            return []
        
        signals = []
        price = bars[self.underlying]['close']
        
        # If we have option positions, check if delta needs rebalancing
        if self._option_positions:
            # Re-estimate portfolio delta from price change
            # Each option position contributes delta * multiplier
            option_delta_total = 0.0
            for pos in self._option_positions:
                option_delta_total += pos.get('delta', 0.0) * pos.get('quantity', 0) * pos.get('multiplier', 100)
            
            self._portfolio_delta = self._hedge_position + option_delta_total
            
            should_rebal, reason = self.should_rebalance()
            if should_rebal:
                shares_needed = self.calculate_hedge_shares(target_delta=0)
                if abs(shares_needed) > 0:
                    signals.append(Signal(
                        timestamp=timestamp,
                        symbol=self.underlying,
                        side=Side.BUY if shares_needed > 0 else Side.SELL,
                        strength=min(abs(self._portfolio_delta) / self.delta_threshold, 1.0),
                        confidence=0.8,
                        metadata={
                            'strategy': 'delta_hedging',
                            'action': 'rebalance',
                            'reason': reason,
                            'shares': abs(shares_needed),
                            'current_delta': self._portfolio_delta,
                            'price': price
                        }
                    ))
                    self._hedge_position += shares_needed
        
        self._signals.extend(signals)
        return signals
    
    def on_option_quote(self, timestamp: datetime, option: OptionQuote) -> List[Signal]:
        """
        Process option quote update.
        
        Parameters
        ----------
        timestamp : datetime
            Quote timestamp.
        option : OptionQuote
            Updated option quote with Greeks.
        
        Returns
        -------
        list
            Hedging signals.
        """
        signals = []
        
        # Update IV tracking
        self._current_iv = option.implied_vol
        if self._entry_iv == 0:
            self._entry_iv = option.implied_vol
        
        # Recalculate portfolio Greeks
        greeks = self.calculate_portfolio_greeks(
            option.underlying_price,
            [option]  # Simplified: single option
        )
        
        self._portfolio_delta = greeks['delta']
        self._portfolio_gamma = greeks['gamma']
        self._portfolio_vega = greeks['vega']
        self._portfolio_theta = greeks['theta']
        
        # Check if rebalancing needed
        should_rebal, reason = self.should_rebalance()
        
        if should_rebal:
            if reason == "iv_spike":
                # Close positions on IV spike
                signals.append(Signal(
                    timestamp=timestamp,
                    symbol=self.underlying,
                    side=Side.SELL if self._hedge_position > 0 else Side.BUY,
                    strength=1.0,
                    confidence=0.9,
                    metadata={
                        'strategy': 'delta_hedging',
                        'action': 'close_hedge',
                        'reason': reason,
                        'iv_change': (self._current_iv - self._entry_iv) / self._entry_iv
                    }
                ))
            else:
                # Rebalance delta
                shares_needed = self.calculate_hedge_shares(target_delta=0)
                
                if abs(shares_needed) > 0:
                    signals.append(Signal(
                        timestamp=timestamp,
                        symbol=self.underlying,
                        side=Side.BUY if shares_needed > 0 else Side.SELL,
                        strength=abs(self._portfolio_delta) / self.delta_threshold,
                        confidence=0.8,
                        metadata={
                            'strategy': 'delta_hedging',
                            'action': 'rebalance',
                            'reason': reason,
                            'shares': abs(shares_needed),
                            'current_delta': self._portfolio_delta,
                            'gamma': self._portfolio_gamma,
                            'theta': self._portfolio_theta
                        }
                    ))
                    
                    self._hedge_position += shares_needed
        
        self._signals.extend(signals)
        return signals


class VolatilityArbitrageStrategy(Strategy):
    """
    Volatility Arbitrage Strategy.
    
    Concept:
    - Compare implied volatility to realized/forecast volatility
    - Sell options when IV > forecast vol (overpriced)
    - Buy options when IV < forecast vol (underpriced)
    
    Position Types:
    - Straddle: Long/short ATM call + put
    - Strangle: Long/short OTM call + put
    - Calendar spread: Different expirations
    
    Signal Generation:
    - Calculate vol premium: IV - Realized Vol
    - Entry when premium exceeds threshold
    - Exit when premium normalizes
    """
    
    def __init__(self,
                 underlying: str,
                 vol_window: int = 20,
                 entry_threshold: float = 0.05,
                 exit_threshold: float = 0.02,
                 use_forecast: bool = True):
        """
        Initialize volatility arbitrage strategy.
        
        Parameters
        ----------
        underlying : str
            Underlying symbol.
        vol_window : int
            Window for realized vol calculation.
        entry_threshold : float
            IV premium threshold for entry (default 5%).
        exit_threshold : float
            IV premium threshold for exit (default 2%).
        use_forecast : bool
            Use GARCH forecast vs historical vol.
        """
        super().__init__(
            name="Vol_Arbitrage",
            symbols=[underlying],
            vol_window=vol_window
        )
        
        self.underlying = underlying
        self.vol_window = vol_window
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.use_forecast = use_forecast
        
        # Price history for realized vol
        self._prices: List[float] = []
        self._returns: List[float] = []
        
        # Position
        self._position_type: str = ""  # "short_vol", "long_vol"
        self._entry_premium: float = 0
    
    def initialize(self, **kwargs):
        """Initialize strategy."""
        super().initialize(**kwargs)
        self._prices = []
        self._returns = []
        self._position_type = ""
    
    def calculate_realized_vol(self) -> float:
        """
        Calculate realized volatility from price history.
        
        σ_realized = σ(returns) * √252
        
        Returns
        -------
        float
            Annualized realized volatility.
        """
        if len(self._returns) < self.vol_window:
            return 0.0
        
        recent_returns = np.array(self._returns[-self.vol_window:])
        daily_vol = np.std(recent_returns)
        annualized_vol = daily_vol * np.sqrt(252)
        
        return annualized_vol
    
    def forecast_volatility(self) -> float:
        """
        Forecast volatility using GARCH-like approach.
        
        Simple exponential weighted moving average of squared returns:
        σ²_t = λ * σ²_{t-1} + (1-λ) * r²_{t-1}
        
        Returns
        -------
        float
            Forecast annualized volatility.
        """
        if len(self._returns) < 2:
            return 0.0
        
        decay = 0.94  # RiskMetrics decay factor
        returns = np.array(self._returns)
        
        # EWMA variance
        variance = returns[0] ** 2
        for ret in returns[1:]:
            variance = decay * variance + (1 - decay) * ret ** 2
        
        daily_vol = np.sqrt(variance)
        annualized_vol = daily_vol * np.sqrt(252)
        
        return annualized_vol
    
    def calculate_vol_premium(self, implied_vol: float) -> float:
        """
        Calculate volatility premium.
        
        Premium = IV - Fair Vol
        
        Parameters
        ----------
        implied_vol : float
            Current implied volatility.
        
        Returns
        -------
        float
            Volatility premium (positive = IV expensive).
        """
        if self.use_forecast:
            fair_vol = self.forecast_volatility()
        else:
            fair_vol = self.calculate_realized_vol()
        
        if fair_vol == 0:
            return 0.0
        
        premium = implied_vol - fair_vol
        return premium
    
    def on_bar(self, timestamp: datetime, bars: Dict[str, Dict[str, float]]) -> List[Signal]:
        """Update price history."""
        if self.underlying not in bars:
            return []
        
        price = bars[self.underlying]['close']
        self._prices.append(price)
        
        if len(self._prices) > 1:
            ret = np.log(price / self._prices[-2])
            self._returns.append(ret)
        
        # Trim
        max_hist = self.vol_window * 2
        if len(self._prices) > max_hist:
            self._prices = self._prices[-max_hist:]
            self._returns = self._returns[-max_hist:]
        
        return []
    
    def on_volatility_update(self, timestamp: datetime, 
                             implied_vol: float,
                             underlying_price: float) -> List[Signal]:
        """
        Process implied volatility update.
        
        Parameters
        ----------
        timestamp : datetime
            Update timestamp.
        implied_vol : float
            Current ATM implied volatility.
        underlying_price : float
            Current underlying price.
        
        Returns
        -------
        list
            Trading signals.
        """
        signals = []
        
        if len(self._returns) < self.vol_window:
            return []
        
        # Calculate vol premium
        premium = self.calculate_vol_premium(implied_vol)
        realized_vol = self.calculate_realized_vol()
        forecast_vol = self.forecast_volatility() if self.use_forecast else realized_vol
        
        # Entry signals
        if self._position_type == "":
            if premium > self.entry_threshold:
                # IV expensive -> Sell volatility (short straddle)
                self._position_type = "short_vol"
                self._entry_premium = premium
                
                signals.append(Signal(
                    timestamp=timestamp,
                    symbol=f"{self.underlying}_STRADDLE",
                    side=Side.SELL,
                    strength=premium / self.entry_threshold,
                    confidence=0.7,
                    metadata={
                        'strategy': 'vol_arbitrage',
                        'action': 'sell_straddle',
                        'implied_vol': implied_vol,
                        'realized_vol': realized_vol,
                        'forecast_vol': forecast_vol,
                        'premium': premium
                    }
                ))
            
            elif premium < -self.entry_threshold:
                # IV cheap -> Buy volatility (long straddle)
                self._position_type = "long_vol"
                self._entry_premium = premium
                
                signals.append(Signal(
                    timestamp=timestamp,
                    symbol=f"{self.underlying}_STRADDLE",
                    side=Side.BUY,
                    strength=abs(premium) / self.entry_threshold,
                    confidence=0.7,
                    metadata={
                        'strategy': 'vol_arbitrage',
                        'action': 'buy_straddle',
                        'implied_vol': implied_vol,
                        'realized_vol': realized_vol,
                        'forecast_vol': forecast_vol,
                        'premium': premium
                    }
                ))
        
        # Exit signals
        elif self._position_type == "short_vol":
            if premium < self.exit_threshold:
                pnl_vol = self._entry_premium - premium
                self._position_type = ""
                
                signals.append(Signal(
                    timestamp=timestamp,
                    symbol=f"{self.underlying}_STRADDLE",
                    side=Side.BUY,
                    strength=0.0,
                    confidence=0.9,
                    metadata={
                        'strategy': 'vol_arbitrage',
                        'action': 'close_short_straddle',
                        'premium_captured': pnl_vol,
                        'implied_vol': implied_vol
                    }
                ))
        
        elif self._position_type == "long_vol":
            if premium > -self.exit_threshold:
                pnl_vol = premium - self._entry_premium
                self._position_type = ""
                
                signals.append(Signal(
                    timestamp=timestamp,
                    symbol=f"{self.underlying}_STRADDLE",
                    side=Side.SELL,
                    strength=0.0,
                    confidence=0.9,
                    metadata={
                        'strategy': 'vol_arbitrage',
                        'action': 'close_long_straddle',
                        'premium_captured': pnl_vol,
                        'implied_vol': implied_vol
                    }
                ))
        
        self._signals.extend(signals)
        return signals


class IronCondorStrategy(Strategy):
    """
    Iron Condor Strategy - Range-bound volatility selling.
    
    Structure:
    - Sell OTM put (support level)
    - Buy further OTM put (protection)
    - Sell OTM call (resistance level)
    - Buy further OTM call (protection)
    
    Profit: When underlying stays within range
    Max Loss: Width of spread - premium received
    
    Greeks Profile:
    - Delta: ~0 (neutral)
    - Gamma: Negative
    - Theta: Positive (time decay profit)
    - Vega: Negative (benefits from vol decrease)
    """
    
    def __init__(self,
                 underlying: str,
                 wing_width: float = 0.05,  # 5% OTM for short strikes
                 spread_width: float = 0.02,  # 2% spread between long/short
                 days_to_expiry: int = 30,
                 delta_target: float = 0.16,  # ~1 std dev
                 profit_target: float = 0.5,  # Close at 50% profit
                 loss_limit: float = 2.0):     # Close at 2x credit
        """
        Initialize iron condor strategy.
        
        Parameters
        ----------
        underlying : str
            Underlying symbol.
        wing_width : float
            Distance for short strikes (% from spot).
        spread_width : float
            Width between long and short strikes.
        days_to_expiry : int
            Target DTE for entry.
        delta_target : float
            Target delta for short strikes.
        profit_target : float
            Close at this percentage of max profit.
        loss_limit : float
            Close at this multiple of initial credit.
        """
        super().__init__(
            name="Iron_Condor",
            symbols=[underlying],
            wing_width=wing_width
        )
        
        self.underlying = underlying
        self.wing_width = wing_width
        self.spread_width = spread_width
        self.days_to_expiry = days_to_expiry
        self.delta_target = delta_target
        self.profit_target = profit_target
        self.loss_limit = loss_limit
        
        # Position tracking
        self._has_position = False
        self._entry_credit: float = 0
        self._upper_strike: float = 0
        self._lower_strike: float = 0
        self._entry_price: float = 0
    
    def initialize(self, **kwargs):
        """Initialize strategy."""
        super().initialize(**kwargs)
        self._has_position = False
        self._entry_credit = 0
    
    def calculate_strikes(self, spot: float, vol: float, 
                         days_to_expiry: int) -> Dict[str, float]:
        """
        Calculate optimal strikes for iron condor.
        
        Uses delta targeting or percentage distance.
        
        Parameters
        ----------
        spot : float
            Current underlying price.
        vol : float
            Implied volatility.
        days_to_expiry : int
            Days until expiration.
        
        Returns
        -------
        dict
            Strike levels for each leg.
        """
        # Standard deviation for the period
        std_move = spot * vol * np.sqrt(days_to_expiry / 365)
        
        # Strikes at approximately 1 std dev (16 delta)
        short_put = spot - std_move
        short_call = spot + std_move
        
        # Long strikes for protection
        long_put = short_put - spot * self.spread_width
        long_call = short_call + spot * self.spread_width
        
        return {
            'short_put': round(short_put, 2),
            'long_put': round(long_put, 2),
            'short_call': round(short_call, 2),
            'long_call': round(long_call, 2)
        }
    
    def on_bar(self, timestamp: datetime, bars: Dict[str, Dict[str, float]]) -> List[Signal]:
        """Process bar for iron condor management."""
        if self.underlying not in bars:
            return []
        
        signals = []
        price = bars[self.underlying]['close']
        
        # Position management
        if self._has_position:
            # Check if price is testing strikes
            buffer = (self._upper_strike - self._lower_strike) * 0.1
            
            if price > self._upper_strike - buffer:
                # Price approaching upper strike - consider adjustment
                signals.append(Signal(
                    timestamp=timestamp,
                    symbol=f"{self.underlying}_IC",
                    side=Side.SELL,
                    strength=0.5,
                    confidence=0.6,
                    metadata={
                        'strategy': 'iron_condor',
                        'action': 'warning_upper',
                        'price': price,
                        'upper_strike': self._upper_strike
                    }
                ))
            
            elif price < self._lower_strike + buffer:
                # Price approaching lower strike
                signals.append(Signal(
                    timestamp=timestamp,
                    symbol=f"{self.underlying}_IC",
                    side=Side.SELL,
                    strength=0.5,
                    confidence=0.6,
                    metadata={
                        'strategy': 'iron_condor',
                        'action': 'warning_lower',
                        'price': price,
                        'lower_strike': self._lower_strike
                    }
                ))
        
        self._signals.extend(signals)
        return signals
    
    def on_option_chain(self, timestamp: datetime, 
                        spot: float, 
                        implied_vol: float,
                        days_to_expiry: int) -> List[Signal]:
        """
        Process option chain for entry/exit.
        
        Parameters
        ----------
        timestamp : datetime
            Current timestamp.
        spot : float
            Underlying spot price.
        implied_vol : float
            ATM implied volatility.
        days_to_expiry : int
            Days until target expiration.
        
        Returns
        -------
        list
            Entry/exit signals.
        """
        signals = []
        
        # Entry logic
        if not self._has_position:
            if days_to_expiry >= self.days_to_expiry - 5:
                # Calculate strikes
                strikes = self.calculate_strikes(spot, implied_vol, days_to_expiry)
                
                self._has_position = True
                self._upper_strike = strikes['short_call']
                self._lower_strike = strikes['short_put']
                self._entry_price = spot
                
                # Estimate entry credit from spread pricing
                put_spread_width = strikes['short_put'] - strikes['long_put']
                call_spread_width = strikes['long_call'] - strikes['short_call']
                # Approximate credit as fraction of wing widths (conservative)
                self._entry_credit = (put_spread_width + call_spread_width) * 0.30
                
                signals.append(Signal(
                    timestamp=timestamp,
                    symbol=f"{self.underlying}_IC",
                    side=Side.SELL,
                    strength=1.0,
                    confidence=0.7,
                    metadata={
                        'strategy': 'iron_condor',
                        'action': 'open',
                        'short_put': strikes['short_put'],
                        'long_put': strikes['long_put'],
                        'short_call': strikes['short_call'],
                        'long_call': strikes['long_call'],
                        'dte': days_to_expiry,
                        'implied_vol': implied_vol
                    }
                ))
        
        # Exit logic: profit target / loss limit / strike breach
        else:
            # Estimate current spread value based on distance from strikes
            dist_to_upper = max(0, self._upper_strike - spot) / (self._upper_strike - self._lower_strike)
            dist_to_lower = max(0, spot - self._lower_strike) / (self._upper_strike - self._lower_strike)
            safety = min(dist_to_upper, dist_to_lower)  # 0 = at strike, 0.5 = dead center
            
            # Approximate current value (decays with time and safety distance)
            dte_pct = max(days_to_expiry / self.days_to_expiry, 0.01)
            current_value = self._entry_credit * dte_pct * (1 - safety)
            profit_pct = (self._entry_credit - current_value) / self._entry_credit if self._entry_credit > 0 else 0
            
            should_close = False
            close_reason = ""
            
            # Profit target hit (e.g., captured 50% of credit)
            if profit_pct >= self.profit_target:
                should_close = True
                close_reason = "profit_target"
            
            # Loss limit hit (current loss exceeds N× credit)
            elif current_value > self._entry_credit * self.loss_limit:
                should_close = True
                close_reason = "loss_limit"
            
            # Price breached strikes
            elif spot >= self._upper_strike or spot <= self._lower_strike:
                should_close = True
                close_reason = "strike_breach"
            
            if should_close:
                self._has_position = False
                signals.append(Signal(
                    timestamp=timestamp,
                    symbol=f"{self.underlying}_IC",
                    side=Side.BUY,  # Buy back the short spreads
                    strength=1.0,
                    confidence=0.9,
                    metadata={
                        'strategy': 'iron_condor',
                        'action': 'close',
                        'reason': close_reason,
                        'entry_credit': self._entry_credit,
                        'current_value': current_value,
                        'profit_pct': profit_pct,
                        'spot': spot,
                        'dte': days_to_expiry
                    }
                ))
                self._entry_credit = 0
        
        self._signals.extend(signals)
        return signals

if __name__ == "__main__":
    from datetime import datetime, timedelta
    
    print("=" * 60)
    print("OPTIONS STRATEGIES TEST")
    print("=" * 60)
    
    # Test Volatility Arbitrage
    vol_strategy = VolatilityArbitrageStrategy(
        underlying="SPY",
        vol_window=20,
        entry_threshold=0.05
    )
    vol_strategy.initialize()
    
    # Use REAL SPY price history
    try:
        from data.realtime_manager import get_data_manager
        dm = get_data_manager()
        
        spy_df = dm.get_historical_data_sync('SPY', '2023-01-01', '2023-03-31')
        
        if not spy_df.empty:
            print("  Using REAL SPY data")
            for idx, row in spy_df.iterrows():
                bars = {
                    "SPY": {
                        'open': row['open'],
                        'high': row['high'],
                        'low': row['low'],
                        'close': row['close'],
                        'volume': row['volume']
                    }
                }
                vol_strategy.on_bar(row['timestamp'], bars)
        else:
            raise Exception("No data")
    except Exception as e:
        print(f"  Real SPY data unavailable: {e}")
        print("  Volatility strategy demonstration requires SPY historical data")
        import sys
        sys.exit(0)
    
    # Simulate vol update
    realized = vol_strategy.calculate_realized_vol()
    forecast = vol_strategy.forecast_volatility()
    
    # Get current price
    current_price = spy_df.iloc[-1]['close']
    timestamp = spy_df.iloc[-1]['timestamp']
    
    # Test with IV higher than forecast
    signals = vol_strategy.on_volatility_update(
        timestamp=timestamp,
        implied_vol=0.25,  # 25% IV
        underlying_price=current_price
    )
    
    print(f"Realized Vol: {realized:.2%}")
    print(f"Forecast Vol: {forecast:.2%}")
    print(f"Signals generated: {len(signals)}")
    
    if signals:
        for s in signals:
            print(f"  {s.side.name}: {s.symbol}")
            print(f"    Premium: {s.metadata.get('premium', 0):.4f}")
    
    # Test Iron Condor
    print("\n" + "-" * 40)
    print("Iron Condor Strategy")
    
    ic_strategy = IronCondorStrategy(
        underlying="SPY",
        wing_width=0.05,
        days_to_expiry=30
    )
    ic_strategy.initialize()
    
    signals = ic_strategy.on_option_chain(
        timestamp=datetime.now(),
        spot=450.0,
        implied_vol=0.18,
        days_to_expiry=30
    )
    
    if signals:
        s = signals[0]
        print(f"Open Iron Condor:")
        print(f"  Short Put: {s.metadata['short_put']}")
        print(f"  Long Put: {s.metadata['long_put']}")
        print(f"  Short Call: {s.metadata['short_call']}")
        print(f"  Long Call: {s.metadata['long_call']}")
