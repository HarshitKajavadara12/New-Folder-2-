"""
GIGA SYSTEM - Momentum Strategies
Trend following and breakout detection with multiple timeframe analysis
"""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .base import Strategy, Signal, Side


class TrendState(Enum):
    """Trend classification."""
    STRONG_UP = 2
    WEAK_UP = 1
    NEUTRAL = 0
    WEAK_DOWN = -1
    STRONG_DOWN = -2


@dataclass
class MomentumState:
    """Current momentum state for a symbol."""
    symbol: str
    trend: TrendState
    strength: float
    rsi: float
    macd_histogram: float
    price_vs_sma: float
    atr: float
    breakout_level: Optional[float] = None
    breakdown_level: Optional[float] = None


class TrendFollowingStrategy(Strategy):
    """
    Multi-indicator Trend Following Strategy.
    
    Philosophy:
    - "The trend is your friend"
    - Enter on confirmed trend, ride momentum
    - Exit when momentum exhausts
    
    Indicators Used:
    - Moving Average (fast/slow crossover)
    - RSI (momentum confirmation)
    - MACD (trend strength)
    - ATR (volatility-adjusted stops)
    
    Entry Rules:
    - Fast MA > Slow MA (uptrend) or Fast MA < Slow MA (downtrend)
    - RSI confirms (not overbought for longs, not oversold for shorts)
    - MACD histogram in same direction
    - Price above/below key moving average
    
    Exit Rules:
    - MA crossover reversal
    - RSI extreme (overbought/oversold)
    - Trailing stop based on ATR
    """
    
    def __init__(self,
                 symbols: List[str],
                 fast_period: int = 10,
                 slow_period: int = 30,
                 signal_period: int = 9,
                 rsi_period: int = 14,
                 atr_period: int = 14,
                 rsi_overbought: float = 70.0,
                 rsi_oversold: float = 30.0,
                 atr_multiplier: float = 2.0):
        """
        Initialize trend following strategy.
        
        Parameters
        ----------
        symbols : list
            List of symbols to trade.
        fast_period : int
            Fast EMA period (default 10).
        slow_period : int
            Slow EMA period (default 30).
        signal_period : int
            MACD signal line period (default 9).
        rsi_period : int
            RSI calculation period (default 14).
        atr_period : int
            ATR period (default 14).
        rsi_overbought : float
            RSI overbought threshold (default 70).
        rsi_oversold : float
            RSI oversold threshold (default 30).
        atr_multiplier : float
            ATR multiplier for trailing stop (default 2.0).
        """
        super().__init__(
            name="Trend_Following",
            symbols=symbols,
            fast_period=fast_period,
            slow_period=slow_period,
            rsi_period=rsi_period
        )
        
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.rsi_period = rsi_period
        self.atr_period = atr_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.atr_multiplier = atr_multiplier
        
        # Price history per symbol
        self._prices: Dict[str, List[float]] = {}
        self._highs: Dict[str, List[float]] = {}
        self._lows: Dict[str, List[float]] = {}
        
        # Indicator cache
        self._momentum_states: Dict[str, MomentumState] = {}
        
        # Position tracking
        self._positions: Dict[str, int] = {}  # 1=long, -1=short, 0=flat
        self._trailing_stops: Dict[str, float] = {}
    
    def initialize(self, **kwargs):
        """Initialize strategy."""
        super().initialize(**kwargs)
        for symbol in self.symbols:
            self._prices[symbol] = []
            self._highs[symbol] = []
            self._lows[symbol] = []
            self._positions[symbol] = 0
            self._trailing_stops[symbol] = 0.0
    
    # =========================================================================
    # TECHNICAL INDICATORS
    # =========================================================================
    
    def _ema(self, data: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average."""
        if len(data) < period:
            return np.mean(data) if len(data) > 0 else 0.0
        
        multiplier = 2.0 / (period + 1)
        ema = data[0]
        
        for price in data[1:]:
            ema = (price - ema) * multiplier + ema
        
        return ema
    
    def _sma(self, data: np.ndarray, period: int) -> float:
        """Calculate Simple Moving Average."""
        if len(data) < period:
            return np.mean(data) if len(data) > 0 else 0.0
        return np.mean(data[-period:])
    
    def _rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """
        Calculate Relative Strength Index.
        
        RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss
        """
        if len(prices) < period + 1:
            return 50.0
        
        # Calculate returns
        deltas = np.diff(prices[-period-1:])
        
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        
        return rsi
    
    def _macd(self, prices: np.ndarray) -> tuple:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        MACD Line = 12-EMA - 26-EMA
        Signal Line = 9-EMA of MACD Line
        Histogram = MACD Line - Signal Line
        
        Returns
        -------
        tuple
            (macd_line, signal_line, histogram)
        """
        if len(prices) < 26:
            return 0.0, 0.0, 0.0
        
        # Build MACD series over recent history for proper signal line
        macd_series = []
        for i in range(26, len(prices) + 1):
            e12 = self._ema(prices[:i], 12)
            e26 = self._ema(prices[:i], 26)
            macd_series.append(e12 - e26)
        
        macd_line = macd_series[-1]
        
        # Signal line = 9-period EMA of the MACD series (proper calculation)
        if len(macd_series) >= self.signal_period:
            macd_arr = np.array(macd_series)
            signal_line = self._ema(macd_arr, self.signal_period)
        else:
            signal_line = self._ema(np.array(macd_series), len(macd_series))
        
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def _atr(self, highs: np.ndarray, lows: np.ndarray, 
             closes: np.ndarray, period: int = 14) -> float:
        """
        Calculate Average True Range.
        
        True Range = max(H-L, |H-Pc|, |L-Pc|)
        ATR = EMA of True Range
        """
        if len(closes) < 2:
            return highs[-1] - lows[-1] if len(highs) > 0 else 0.0
        
        n = min(len(closes), period + 1)
        
        # Calculate True Range
        tr = np.zeros(n - 1)
        for i in range(1, n):
            high_low = highs[-n+i] - lows[-n+i]
            high_close = abs(highs[-n+i] - closes[-n+i-1])
            low_close = abs(lows[-n+i] - closes[-n+i-1])
            tr[i-1] = max(high_low, high_close, low_close)
        
        return np.mean(tr)
    
    # =========================================================================
    # TREND ANALYSIS
    # =========================================================================
    
    def _classify_trend(self, symbol: str) -> TrendState:
        """
        Classify current trend using multiple indicators.
        
        Scoring System:
        - MA crossover: +/-1
        - Price vs SMA: +/-1  
        - RSI: +/-0.5
        - MACD histogram: +/-0.5
        
        Score > 1.5: Strong Up
        Score > 0.5: Weak Up
        Score < -0.5: Weak Down
        Score < -1.5: Strong Down
        """
        prices = np.array(self._prices[symbol])
        
        if len(prices) < self.slow_period:
            return TrendState.NEUTRAL
        
        score = 0.0
        
        # MA crossover
        fast_ma = self._ema(prices, self.fast_period)
        slow_ma = self._ema(prices, self.slow_period)
        
        if fast_ma > slow_ma:
            score += 1.0
        else:
            score -= 1.0
        
        # Price vs slow MA
        current_price = prices[-1]
        if current_price > slow_ma:
            score += 1.0
        else:
            score -= 1.0
        
        # RSI
        rsi = self._rsi(prices, self.rsi_period)
        if rsi > 50:
            score += 0.5
        else:
            score -= 0.5
        
        # MACD histogram
        _, _, histogram = self._macd(prices)
        if histogram > 0:
            score += 0.5
        else:
            score -= 0.5
        
        # Classify
        if score >= 1.5:
            return TrendState.STRONG_UP
        elif score >= 0.5:
            return TrendState.WEAK_UP
        elif score <= -1.5:
            return TrendState.STRONG_DOWN
        elif score <= -0.5:
            return TrendState.WEAK_DOWN
        else:
            return TrendState.NEUTRAL
    
    def _calculate_momentum_state(self, symbol: str) -> MomentumState:
        """Calculate full momentum state for a symbol."""
        prices = np.array(self._prices[symbol])
        highs = np.array(self._highs[symbol])
        lows = np.array(self._lows[symbol])
        
        if len(prices) < self.slow_period:
            return MomentumState(
                symbol=symbol,
                trend=TrendState.NEUTRAL,
                strength=0.0,
                rsi=50.0,
                macd_histogram=0.0,
                price_vs_sma=0.0,
                atr=0.0
            )
        
        # Calculate indicators
        fast_ma = self._ema(prices, self.fast_period)
        slow_ma = self._ema(prices, self.slow_period)
        rsi = self._rsi(prices, self.rsi_period)
        _, _, macd_hist = self._macd(prices)
        atr = self._atr(highs, lows, prices, self.atr_period)
        
        # Trend classification
        trend = self._classify_trend(symbol)
        
        # Trend strength (0 to 1)
        ma_diff = (fast_ma - slow_ma) / slow_ma if slow_ma != 0 else 0
        strength = min(1.0, abs(ma_diff) * 100)  # Normalize
        
        # Price vs SMA percentage
        price_vs_sma = (prices[-1] - slow_ma) / slow_ma * 100 if slow_ma != 0 else 0
        
        # Breakout levels (recent high/low)
        lookback = min(20, len(prices))
        breakout_level = max(highs[-lookback:]) if len(highs) >= lookback else None
        breakdown_level = min(lows[-lookback:]) if len(lows) >= lookback else None
        
        return MomentumState(
            symbol=symbol,
            trend=trend,
            strength=strength,
            rsi=rsi,
            macd_histogram=macd_hist,
            price_vs_sma=price_vs_sma,
            atr=atr,
            breakout_level=breakout_level,
            breakdown_level=breakdown_level
        )
    
    # =========================================================================
    # SIGNAL GENERATION
    # =========================================================================
    
    def on_bar(self, timestamp: datetime, bars: Dict[str, Dict[str, float]]) -> List[Signal]:
        """
        Process new bar and generate signals.
        
        Parameters
        ----------
        timestamp : datetime
            Bar timestamp.
        bars : dict
            Symbol -> OHLCV data.
        
        Returns
        -------
        list
            List of Signal objects.
        """
        signals = []
        
        for symbol in self.symbols:
            if symbol not in bars:
                continue
            
            bar = bars[symbol]
            
            # Update price history
            self._prices[symbol].append(bar['close'])
            self._highs[symbol].append(bar['high'])
            self._lows[symbol].append(bar['low'])
            
            # Trim history
            max_history = self.slow_period * 3
            if len(self._prices[symbol]) > max_history:
                self._prices[symbol] = self._prices[symbol][-max_history:]
                self._highs[symbol] = self._highs[symbol][-max_history:]
                self._lows[symbol] = self._lows[symbol][-max_history:]
            
            # Not enough data
            if len(self._prices[symbol]) < self.slow_period:
                continue
            
            # Calculate momentum state
            state = self._calculate_momentum_state(symbol)
            self._momentum_states[symbol] = state
            
            # Generate signals based on position
            current_pos = self._positions[symbol]
            signal = self._generate_signal(timestamp, symbol, state, bar, current_pos)
            
            if signal:
                signals.append(signal)
                self._signals.append(signal)
        
        return signals
    
    def _generate_signal(self, timestamp: datetime, symbol: str, 
                        state: MomentumState, bar: Dict[str, float],
                        current_pos: int) -> Optional[Signal]:
        """Generate trading signal based on momentum state."""
        
        # Check trailing stop first
        if current_pos != 0:
            stop_triggered = self._check_trailing_stop(symbol, bar['close'], state.atr)
            if stop_triggered:
                self._positions[symbol] = 0
                return Signal(
                    timestamp=timestamp,
                    symbol=symbol,
                    side=Side.SELL if current_pos > 0 else Side.BUY,
                    strength=0.0,
                    confidence=1.0,
                    metadata={
                        'strategy': 'trend_following',
                        'action': 'stop_loss',
                        'stop_price': self._trailing_stops[symbol]
                    }
                )
        
        # Entry signals
        if current_pos == 0:
            # Long entry
            if state.trend in [TrendState.STRONG_UP, TrendState.WEAK_UP]:
                if state.rsi < self.rsi_overbought and state.macd_histogram > 0:
                    self._positions[symbol] = 1
                    self._trailing_stops[symbol] = bar['close'] - state.atr * self.atr_multiplier
                    
                    return Signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        side=Side.BUY,
                        strength=state.strength,
                        confidence=0.5 + state.strength * 0.5,
                        stop_loss=self._trailing_stops[symbol],
                        metadata={
                            'strategy': 'trend_following',
                            'action': 'long_entry',
                            'trend': state.trend.name,
                            'rsi': state.rsi,
                            'macd_hist': state.macd_histogram
                        }
                    )
            
            # Short entry
            elif state.trend in [TrendState.STRONG_DOWN, TrendState.WEAK_DOWN]:
                if state.rsi > self.rsi_oversold and state.macd_histogram < 0:
                    self._positions[symbol] = -1
                    self._trailing_stops[symbol] = bar['close'] + state.atr * self.atr_multiplier
                    
                    return Signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        side=Side.SELL,
                        strength=state.strength,
                        confidence=0.5 + state.strength * 0.5,
                        stop_loss=self._trailing_stops[symbol],
                        metadata={
                            'strategy': 'trend_following',
                            'action': 'short_entry',
                            'trend': state.trend.name,
                            'rsi': state.rsi,
                            'macd_hist': state.macd_histogram
                        }
                    )
        
        # Exit signals (trend reversal)
        elif current_pos > 0:  # Long position
            if state.trend in [TrendState.STRONG_DOWN, TrendState.WEAK_DOWN]:
                self._positions[symbol] = 0
                return Signal(
                    timestamp=timestamp,
                    symbol=symbol,
                    side=Side.SELL,
                    strength=0.0,
                    confidence=state.strength,
                    metadata={
                        'strategy': 'trend_following',
                        'action': 'long_exit',
                        'reason': 'trend_reversal',
                        'new_trend': state.trend.name
                    }
                )
            elif state.rsi > self.rsi_overbought:
                self._positions[symbol] = 0
                return Signal(
                    timestamp=timestamp,
                    symbol=symbol,
                    side=Side.SELL,
                    strength=0.0,
                    confidence=0.8,
                    metadata={
                        'strategy': 'trend_following',
                        'action': 'long_exit',
                        'reason': 'rsi_overbought',
                        'rsi': state.rsi
                    }
                )
        
        elif current_pos < 0:  # Short position
            if state.trend in [TrendState.STRONG_UP, TrendState.WEAK_UP]:
                self._positions[symbol] = 0
                return Signal(
                    timestamp=timestamp,
                    symbol=symbol,
                    side=Side.BUY,
                    strength=0.0,
                    confidence=state.strength,
                    metadata={
                        'strategy': 'trend_following',
                        'action': 'short_exit',
                        'reason': 'trend_reversal',
                        'new_trend': state.trend.name
                    }
                )
            elif state.rsi < self.rsi_oversold:
                self._positions[symbol] = 0
                return Signal(
                    timestamp=timestamp,
                    symbol=symbol,
                    side=Side.BUY,
                    strength=0.0,
                    confidence=0.8,
                    metadata={
                        'strategy': 'trend_following',
                        'action': 'short_exit',
                        'reason': 'rsi_oversold',
                        'rsi': state.rsi
                    }
                )
        
        # Update trailing stop
        if current_pos != 0:
            self._update_trailing_stop(symbol, bar['close'], state.atr, current_pos)
        
        return None
    
    def _check_trailing_stop(self, symbol: str, price: float, atr: float) -> bool:
        """Check if trailing stop is triggered."""
        stop = self._trailing_stops.get(symbol, 0)
        pos = self._positions.get(symbol, 0)
        
        if pos > 0 and price <= stop:
            return True
        elif pos < 0 and price >= stop:
            return True
        
        return False
    
    def _update_trailing_stop(self, symbol: str, price: float, 
                              atr: float, position: int):
        """Update trailing stop based on price movement."""
        current_stop = self._trailing_stops.get(symbol, 0)
        
        if position > 0:  # Long
            new_stop = price - atr * self.atr_multiplier
            self._trailing_stops[symbol] = max(current_stop, new_stop)
        elif position < 0:  # Short
            new_stop = price + atr * self.atr_multiplier
            self._trailing_stops[symbol] = min(current_stop, new_stop) if current_stop > 0 else new_stop


class BreakoutStrategy(Strategy):
    """
    Channel Breakout Strategy.
    
    Concept:
    - Price consolidates in a range (Donchian channel)
    - Breakout above range = bullish signal
    - Breakdown below range = bearish signal
    
    Entry:
    - Price breaks above N-period high -> Long
    - Price breaks below N-period low -> Short
    
    Exit:
    - Price crosses opposite channel boundary
    - Or trailing stop
    """
    
    def __init__(self,
                 symbols: List[str],
                 entry_period: int = 20,
                 exit_period: int = 10,
                 atr_period: int = 14,
                 atr_multiplier: float = 2.0):
        """
        Initialize breakout strategy.
        
        Parameters
        ----------
        symbols : list
            Symbols to trade.
        entry_period : int
            Period for entry channel (default 20).
        exit_period : int
            Period for exit channel (default 10).
        atr_period : int
            ATR period for stops.
        atr_multiplier : float
            ATR multiplier for position sizing.
        """
        super().__init__(
            name="Breakout",
            symbols=symbols,
            entry_period=entry_period,
            exit_period=exit_period
        )
        
        self.entry_period = entry_period
        self.exit_period = exit_period
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        
        # Price history
        self._highs: Dict[str, List[float]] = {}
        self._lows: Dict[str, List[float]] = {}
        self._closes: Dict[str, List[float]] = {}
        
        # Position tracking
        self._positions: Dict[str, int] = {}
        self._entry_prices: Dict[str, float] = {}
    
    def initialize(self, **kwargs):
        """Initialize strategy."""
        super().initialize(**kwargs)
        for symbol in self.symbols:
            self._highs[symbol] = []
            self._lows[symbol] = []
            self._closes[symbol] = []
            self._positions[symbol] = 0
            self._entry_prices[symbol] = 0.0
    
    def _donchian_channel(self, highs: np.ndarray, lows: np.ndarray,
                          period: int) -> tuple:
        """
        Calculate Donchian Channel.
        
        Upper Band = N-period high
        Lower Band = N-period low
        Middle = (Upper + Lower) / 2
        
        Returns
        -------
        tuple
            (upper, middle, lower)
        """
        if len(highs) < period:
            return highs[-1], (highs[-1] + lows[-1])/2, lows[-1]
        
        upper = max(highs[-period:])
        lower = min(lows[-period:])
        middle = (upper + lower) / 2
        
        return upper, middle, lower
    
    def on_bar(self, timestamp: datetime, bars: Dict[str, Dict[str, float]]) -> List[Signal]:
        """Process bar and generate breakout signals."""
        signals = []
        
        for symbol in self.symbols:
            if symbol not in bars:
                continue
            
            bar = bars[symbol]
            
            # Update history
            self._highs[symbol].append(bar['high'])
            self._lows[symbol].append(bar['low'])
            self._closes[symbol].append(bar['close'])
            
            # Trim
            max_hist = self.entry_period * 2
            if len(self._highs[symbol]) > max_hist:
                self._highs[symbol] = self._highs[symbol][-max_hist:]
                self._lows[symbol] = self._lows[symbol][-max_hist:]
                self._closes[symbol] = self._closes[symbol][-max_hist:]
            
            if len(self._highs[symbol]) < self.entry_period:
                continue
            
            highs = np.array(self._highs[symbol])
            lows = np.array(self._lows[symbol])
            
            # Calculate channels
            entry_upper, _, entry_lower = self._donchian_channel(
                highs[:-1], lows[:-1], self.entry_period  # Exclude current bar
            )
            exit_upper, _, exit_lower = self._donchian_channel(
                highs[:-1], lows[:-1], self.exit_period
            )
            
            current_pos = self._positions[symbol]
            price = bar['close']
            
            # Entry signals
            if current_pos == 0:
                # Breakout above entry channel -> Long
                if price > entry_upper:
                    self._positions[symbol] = 1
                    self._entry_prices[symbol] = price
                    
                    signals.append(Signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        side=Side.BUY,
                        strength=1.0,
                        confidence=0.7,
                        metadata={
                            'strategy': 'breakout',
                            'action': 'long_entry',
                            'breakout_level': entry_upper,
                            'channel_width': entry_upper - entry_lower
                        }
                    ))
                
                # Breakdown below entry channel -> Short
                elif price < entry_lower:
                    self._positions[symbol] = -1
                    self._entry_prices[symbol] = price
                    
                    signals.append(Signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        side=Side.SELL,
                        strength=1.0,
                        confidence=0.7,
                        metadata={
                            'strategy': 'breakout',
                            'action': 'short_entry',
                            'breakdown_level': entry_lower,
                            'channel_width': entry_upper - entry_lower
                        }
                    ))
            
            # Exit signals
            elif current_pos > 0:  # Long
                if price < exit_lower:
                    pnl_pct = (price - self._entry_prices[symbol]) / self._entry_prices[symbol]
                    self._positions[symbol] = 0
                    
                    signals.append(Signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        side=Side.SELL,
                        strength=0.0,
                        confidence=1.0,
                        metadata={
                            'strategy': 'breakout',
                            'action': 'long_exit',
                            'exit_level': exit_lower,
                            'pnl_pct': pnl_pct
                        }
                    ))
            
            elif current_pos < 0:  # Short
                if price > exit_upper:
                    pnl_pct = (self._entry_prices[symbol] - price) / self._entry_prices[symbol]
                    self._positions[symbol] = 0
                    
                    signals.append(Signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        side=Side.BUY,
                        strength=0.0,
                        confidence=1.0,
                        metadata={
                            'strategy': 'breakout',
                            'action': 'short_exit',
                            'exit_level': exit_upper,
                            'pnl_pct': pnl_pct
                        }
                    ))
        
        self._signals.extend(signals)
        return signals


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    import numpy as np
    from datetime import datetime, timedelta
    
    # Use REAL trending market data
    try:
        from data.realtime_manager import get_data_manager
        dm = get_data_manager()
        
        # QQQ often shows trends
        qqq_df = dm.get_historical_data_sync('QQQ', '2023-01-01', '2023-12-31')
        
        if not qqq_df.empty:
            print("  Using REAL QQQ data")
            n = len(qqq_df)
            prices = qqq_df['close'].values.tolist()
            highs = qqq_df['high'].values.tolist()
            lows = qqq_df['low'].values.tolist()
            timestamps = qqq_df['timestamp'].tolist()
        else:
            raise Exception("No data")
    except Exception as e:
        print(f"  Real SPY data unavailable: {e}")
        print("  Momentum strategy demonstration requires SPY historical data")
        import sys
        sys.exit(0)
    
    # Test Trend Following Strategy
    print("=" * 60)
    print("TREND FOLLOWING STRATEGY TEST")
    print("=" * 60)
    
    # Extract data from spy_df
    n = len(spy_df)
    timestamps = spy_df['timestamp'].tolist()
    prices = spy_df['close'].tolist()
    highs = spy_df['high'].tolist()
    lows = spy_df['low'].tolist()
    
    trend_strategy = TrendFollowingStrategy(
        symbols=["SPY"],
        fast_period=10,
        slow_period=30
    )
    trend_strategy.initialize()
    
    for i in range(n):
        timestamp = timestamps[i]
        bars = {
            "SPY": {
                'open': prices[i],
                'high': highs[i],
                'low': lows[i],
                'close': prices[i],
                'volume': 1000000
            }
        }
        trend_strategy.on_bar(timestamp, bars)
    
    summary = trend_strategy.finalize()
    print(f"Total signals: {summary['total_signals']}")
    
    # Test Breakout Strategy
    print("\n" + "=" * 60)
    print("BREAKOUT STRATEGY TEST")
    print("=" * 60)
    
    breakout_strategy = BreakoutStrategy(
        symbols=["TEST"],
        entry_period=20,
        exit_period=10
    )
    breakout_strategy.initialize()
    
    for i in range(n):
        timestamp = base_date + timedelta(days=i)
        bars = {
            "TEST": {
                'open': prices[i],
                'high': highs[i],
                'low': lows[i],
                'close': prices[i],
                'volume': 1000000
            }
        }
        breakout_strategy.on_bar(timestamp, bars)
    
    summary = breakout_strategy.finalize()
    print(f"Total signals: {summary['total_signals']}")
# ------------------------------------------------------------------------------
# PHASE 8 LIVE IMPLEMENTATION
# ------------------------------------------------------------------------------

class LiveMomentumStrategy:
    """
    LAYER 2: Signal Generation
    Simple, robust SMA Crossover for Live Tick Loop.
    """
    def __init__(self, fast_period=5, slow_period=15):
        self.prices = []
        self.fast = fast_period
        self.slow = slow_period
        self.name = "Momentum_V1"

    def update(self, price: float, timestamp: float) -> Optional[dict]:
        self.prices.append(price)
        if len(self.prices) > self.slow + 1:
            self.prices.pop(0)

        if len(self.prices) < self.slow:
            return None # Not enough data

        # Calculate SMAs
        fast_sma = sum(self.prices[-self.fast:]) / self.fast
        slow_sma = sum(self.prices[-self.slow:]) / self.slow
        
        # Calculate Slope (very short term)
        slope = 0
        if len(self.prices) >= 3:
            slope = self.prices[-1] - self.prices[-3]

        # Signal Logic
        # BUY if Fast > Slow AND Slope > 0
        if fast_sma > slow_sma and slope > 0:
            return {
                "source": self.name,
                "action": "ENTER_LONG",
                "confidence": 0.75,
                "reason": f"Golden Cross + Momentum ({price:.2f})"
            }
        
        # SELL if Fast < Slow AND Slope < 0
        elif fast_sma < slow_sma and slope < 0:
            return {
                "source": self.name,
                "action": "ENTER_SHORT",
                "confidence": 0.75,
                "reason": f"Death Cross + Momentum ({price:.2f})"
            }
            
        return {
            "source": self.name,
            "action": "HOLD",
            "confidence": 0.1,
            "reason": "Choppy or Neutral"
        }
