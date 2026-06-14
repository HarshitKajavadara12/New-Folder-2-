"""
GIGA SYSTEM - Strategy Base Classes
Abstract interfaces for all trading strategies
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class Side(Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class OrderType(Enum):
    """Order type."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


@dataclass
class Signal:
    """Trading signal from strategy."""
    timestamp: datetime
    symbol: str
    side: Side
    strength: float  # -1 to 1 (negative = sell, positive = buy)
    confidence: float  # 0 to 1
    target_position: Optional[float] = None  # Target position size
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_entry(self) -> bool:
        return self.side != Side.HOLD
    
    @property
    def direction(self) -> int:
        """1 for buy, -1 for sell, 0 for hold."""
        if self.side == Side.BUY:
            return 1
        elif self.side == Side.SELL:
            return -1
        return 0


@dataclass
class Order:
    """Order to be executed."""
    timestamp: datetime
    symbol: str
    side: Side
    order_type: OrderType
    quantity: float
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"  # Good Till Cancelled
    
    def __post_init__(self):
        if self.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            if self.limit_price is None:
                raise ValueError("Limit price required for limit orders")
        if self.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
            if self.stop_price is None:
                raise ValueError("Stop price required for stop orders")


@dataclass
class Position:
    """Current position in an asset."""
    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    @property
    def is_long(self) -> bool:
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        return self.quantity < 0
    
    @property
    def is_flat(self) -> bool:
        return self.quantity == 0
    
    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price
    
    def update_price(self, price: float):
        """Update current price and unrealized P&L."""
        self.current_price = price
        self.unrealized_pnl = (price - self.entry_price) * self.quantity


@dataclass
class Trade:
    """Executed trade details."""
    id: str
    timestamp: datetime
    symbol: str
    side: Side
    price: float
    quantity: float
    fees: float = 0.0
    pnl: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class Strategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    Strategies implement:
    1. generate_signals(): Produce trading signals from market data
    2. on_bar(): Called on each new bar (OHLCV)
    3. on_tick(): Called on each new tick (optional)
    
    Lifecycle:
    1. __init__(): Initialize parameters
    2. initialize(): Called before backtest starts
    3. on_bar() / on_tick(): Process market data
    4. generate_signals(): Produce signals
    5. finalize(): Called after backtest ends
    """
    
    def __init__(self, name: str, symbols: List[str], **params):
        """
        Initialize strategy.
        
        Parameters
        ----------
        name : str
            Strategy name.
        symbols : list
            List of symbols to trade.
        **params
            Strategy-specific parameters.
        """
        self.name = name
        self.symbols = symbols
        self.params = params
        self._positions: Dict[str, Position] = {}
        self._signals: List[Signal] = []
        self._initialized = False
    
    def initialize(self, **kwargs):
        """
        Called before backtest starts.
        Override to initialize indicators, load models, etc.
        """
        self._initialized = True
    
    def finalize(self) -> Dict[str, Any]:
        """
        Called after backtest ends.
        Override to compute final metrics, save state, etc.
        
        Returns
        -------
        dict
            Strategy summary and custom metrics.
        """
        return {
            'name': self.name,
            'symbols': self.symbols,
            'params': self.params,
            'n_signals': len(self._signals)
        }
    
    @abstractmethod
    def on_bar(self, timestamp: datetime, bars: Dict[str, Dict[str, float]]) -> List[Signal]:
        """
        Process new bar data.
        
        Parameters
        ----------
        timestamp : datetime
            Bar timestamp.
        bars : dict
            Dictionary of symbol -> OHLCV dict.
            Example: {'AAPL': {'open': 150, 'high': 152, 'low': 149, 'close': 151, 'volume': 1000000}}
        
        Returns
        -------
        list
            List of Signal objects (can be empty).
        """
        pass
    
    def on_tick(self, timestamp: datetime, ticks: Dict[str, Dict[str, float]]) -> List[Signal]:
        """
        Process new tick data (optional).
        
        Parameters
        ----------
        timestamp : datetime
            Tick timestamp.
        ticks : dict
            Dictionary of symbol -> tick dict.
            Example: {'AAPL': {'price': 150.5, 'volume': 100, 'side': 'buy'}}
        
        Returns
        -------
        list
            List of Signal objects.
        """
        return []
    
    def on_order_fill(self, order: Order, fill_price: float, fill_qty: float):
        """
        Called when an order is filled.
        
        Parameters
        ----------
        order : Order
            The filled order.
        fill_price : float
            Execution price.
        fill_qty : float
            Filled quantity.
        """
        pass
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get current position for symbol."""
        return self._positions.get(symbol)
    
    def set_position(self, position: Position):
        """Update position for symbol."""
        self._positions[position.symbol] = position
    
    def clear_position(self, symbol: str):
        """Clear position for symbol."""
        if symbol in self._positions:
            del self._positions[symbol]
    
    @property
    def positions(self) -> Dict[str, Position]:
        """All current positions."""
        return self._positions.copy()
    
    @property
    def signals(self) -> List[Signal]:
        """All generated signals."""
        return self._signals.copy()


class IndicatorStrategy(Strategy):
    """
    Strategy base class with built-in indicator management.
    
    Automatically computes and caches technical indicators.
    """
    
    def __init__(self, name: str, symbols: List[str], 
                 lookback: int = 100, **params):
        """
        Initialize indicator-based strategy.
        
        Parameters
        ----------
        name : str
            Strategy name.
        symbols : list
            Symbols to trade.
        lookback : int
            Number of bars to keep for indicator calculation.
        **params
            Strategy parameters.
        """
        super().__init__(name, symbols, **params)
        self.lookback = lookback
        self._price_history: Dict[str, Dict[str, List[float]]] = {}
        self._indicators: Dict[str, Dict[str, np.ndarray]] = {}
    
    def initialize(self, **kwargs):
        """Initialize price history buffers."""
        super().initialize(**kwargs)
        
        for symbol in self.symbols:
            self._price_history[symbol] = {
                'open': [],
                'high': [],
                'low': [],
                'close': [],
                'volume': []
            }
            self._indicators[symbol] = {}
    
    def _update_history(self, symbol: str, bar: Dict[str, float]):
        """Update price history with new bar."""
        history = self._price_history[symbol]
        
        for field in ['open', 'high', 'low', 'close', 'volume']:
            history[field].append(bar.get(field, 0))
            
            # Trim to lookback
            if len(history[field]) > self.lookback:
                history[field] = history[field][-self.lookback:]
    
    def get_prices(self, symbol: str, field: str = 'close') -> np.ndarray:
        """Get price history as numpy array."""
        return np.array(self._price_history[symbol][field])
    
    def set_indicator(self, symbol: str, name: str, values: np.ndarray):
        """Cache indicator values."""
        self._indicators[symbol][name] = values
    
    def get_indicator(self, symbol: str, name: str) -> Optional[np.ndarray]:
        """Get cached indicator values."""
        return self._indicators[symbol].get(name)
    
    @abstractmethod
    def compute_indicators(self, symbol: str):
        """
        Compute indicators for symbol.
        
        Override this method to compute strategy-specific indicators.
        Call set_indicator() to cache values.
        """
        pass
    
    def on_bar(self, timestamp: datetime, bars: Dict[str, Dict[str, float]]) -> List[Signal]:
        """Process bar and generate signals."""
        signals = []
        
        for symbol, bar in bars.items():
            if symbol in self.symbols:
                # Update price history
                self._update_history(symbol, bar)
                
                # Compute indicators (if enough data)
                if len(self._price_history[symbol]['close']) >= self.lookback // 2:
                    self.compute_indicators(symbol)
                    
                    # Generate signal
                    signal = self.generate_signal(timestamp, symbol, bar)
                    if signal is not None:
                        signals.append(signal)
                        self._signals.append(signal)
        
        return signals
    
    @abstractmethod
    def generate_signal(self, timestamp: datetime, symbol: str, 
                        bar: Dict[str, float]) -> Optional[Signal]:
        """
        Generate trading signal for symbol.
        
        Parameters
        ----------
        timestamp : datetime
            Current timestamp.
        symbol : str
            Symbol to generate signal for.
        bar : dict
            Current OHLCV bar.
        
        Returns
        -------
        Signal or None
            Trading signal (None if no signal).
        """
        pass


class MultiAssetStrategy(Strategy):
    """
    Strategy base class for multi-asset/portfolio strategies.
    
    Handles cross-asset signals and portfolio-level constraints.
    """
    
    def __init__(self, name: str, symbols: List[str],
                 max_positions: int = 10,
                 max_position_size: float = 0.1,  # 10% of portfolio
                 **params):
        """
        Initialize multi-asset strategy.
        
        Parameters
        ----------
        name : str
            Strategy name.
        symbols : list
            Universe of symbols.
        max_positions : int
            Maximum concurrent positions.
        max_position_size : float
            Maximum position size as fraction of portfolio.
        **params
            Strategy parameters.
        """
        super().__init__(name, symbols, **params)
        self.max_positions = max_positions
        self.max_position_size = max_position_size
        self._portfolio_value = 0.0
    
    def set_portfolio_value(self, value: float):
        """Update portfolio value."""
        self._portfolio_value = value
    
    def get_target_weight(self, symbol: str) -> float:
        """
        Get target portfolio weight for symbol.
        
        Override this method to implement allocation logic.
        
        Returns
        -------
        float
            Target weight (0 to 1).
        """
        return self.max_position_size
    
    def rank_signals(self, signals: List[Signal]) -> List[Signal]:
        """
        Rank and filter signals.
        
        Override to implement signal ranking/filtering.
        
        Parameters
        ----------
        signals : list
            Candidate signals.
        
        Returns
        -------
        list
            Ranked and filtered signals.
        """
        # Default: sort by strength
        return sorted(signals, key=lambda s: abs(s.strength), reverse=True)[:self.max_positions]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_position_size(signal: Signal, portfolio_value: float,
                           current_price: float, 
                           max_risk_pct: float = 0.02) -> float:
    """
    Calculate position size based on signal and risk parameters.
    
    Uses fixed fractional position sizing:
    Position Size = (Portfolio Value * Max Risk %) / (Entry Price - Stop Loss)
    
    Parameters
    ----------
    signal : Signal
        Trading signal.
    portfolio_value : float
        Current portfolio value.
    current_price : float
        Current asset price.
    max_risk_pct : float
        Maximum risk per trade as percentage.
    
    Returns
    -------
    float
        Position size in units.
    """
    if signal.stop_loss is None:
        # Default to 2% stop
        risk_per_unit = current_price * 0.02
    else:
        risk_per_unit = abs(current_price - signal.stop_loss)
    
    if risk_per_unit == 0:
        return 0
    
    max_risk = portfolio_value * max_risk_pct
    size = max_risk / risk_per_unit
    
    # Scale by confidence
    size *= signal.confidence
    
    return size


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example: Simple Moving Average Crossover Strategy
    from data.indicators import sma, ema
    
    class SMACrossover(IndicatorStrategy):
        """Simple Moving Average Crossover Strategy."""
        
        def __init__(self, symbols: List[str], fast_period: int = 10, 
                     slow_period: int = 30):
            super().__init__(
                name="SMA_Crossover",
                symbols=symbols,
                lookback=max(fast_period, slow_period) + 10,
                fast_period=fast_period,
                slow_period=slow_period
            )
            self.fast_period = fast_period
            self.slow_period = slow_period
        
        def compute_indicators(self, symbol: str):
            close = self.get_prices(symbol, 'close')
            
            if len(close) >= self.slow_period:
                fast_sma = sma(close, self.fast_period)
                slow_sma = sma(close, self.slow_period)
                
                self.set_indicator(symbol, 'fast_sma', fast_sma)
                self.set_indicator(symbol, 'slow_sma', slow_sma)
        
        def generate_signal(self, timestamp, symbol, bar) -> Optional[Signal]:
            fast = self.get_indicator(symbol, 'fast_sma')
            slow = self.get_indicator(symbol, 'slow_sma')
            
            if fast is None or slow is None:
                return None
            
            # Need at least 2 values for crossover detection
            if len(fast) < 2:
                return None
            
            # Crossover detection
            prev_cross = fast[-2] - slow[-2]
            curr_cross = fast[-1] - slow[-1]
            
            if prev_cross < 0 and curr_cross > 0:
                # Bullish crossover
                return Signal(
                    timestamp=timestamp,
                    symbol=symbol,
                    side=Side.BUY,
                    strength=1.0,
                    confidence=0.7,
                    metadata={'crossover': 'bullish'}
                )
            elif prev_cross > 0 and curr_cross < 0:
                # Bearish crossover
                return Signal(
                    timestamp=timestamp,
                    symbol=symbol,
                    side=Side.SELL,
                    strength=-1.0,
                    confidence=0.7,
                    metadata={'crossover': 'bearish'}
                )
            
            return None
    
    # Test
    strategy = SMACrossover(['AAPL'], fast_period=5, slow_period=20)
    strategy.initialize()
    
    print(f"Strategy: {strategy.name}")
    print(f"Symbols: {strategy.symbols}")
    print(f"Parameters: fast={strategy.fast_period}, slow={strategy.slow_period}")
    print("\nStrategy base classes test complete!")

class PositionSizer(ABC):
    """Abstract base class for position sizing strategies."""
    
    @abstractmethod
    def calculate_size(self, signal: Signal, price: float, portfolio_value: float) -> float:
        """Calculate position size for a signal."""
        pass

class FixedFractionSizer(PositionSizer):
    def __init__(self, fraction: float = 0.02):
        self.fraction = fraction

    def calculate_size(self, signal: Signal, price: float, portfolio_value: float) -> float:
        risk_amt = portfolio_value * self.fraction
        return risk_amt / price

class KellyCriterionSizer(PositionSizer):
    def calculate_size(self, signal: Signal, price: float, portfolio_value: float) -> float:
        # Simplified Kelly: f = p - q (assuming b=1)
        # Real impl needs win rates. Returning conservative 1%
        return (portfolio_value * 0.01) / price
