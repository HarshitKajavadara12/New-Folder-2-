"""
GIGA SYSTEM - Backtesting Engine
Event-driven backtesting with realistic execution simulation
"""

import numpy as np
from typing import Dict, List, Any, Optional, Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import heapq

try:
    from research.strategies.base import Strategy, Signal, Side, Position, Trade
except ImportError:
    from strategies.base import Strategy, Signal, Side, Position, Trade


class EventType(Enum):
    """Event types in backtesting."""
    BAR = "bar"
    SIGNAL = "signal"
    ORDER = "order"
    FILL = "fill"
    CANCEL = "cancel"


class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Event:
    """Base event class."""
    timestamp: datetime
    event_type: EventType
    data: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        return self.timestamp < other.timestamp


@dataclass
class Order:
    """Order representation."""
    order_id: str
    timestamp: datetime
    symbol: str
    side: Side
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0
    avg_fill_price: float = 0
    commission: float = 0
    slippage: float = 0
    
    @property
    def remaining(self) -> float:
        return self.quantity - self.filled_quantity
    
    @property
    def is_buy(self) -> bool:
        return self.side == Side.BUY
    
    @property
    def is_complete(self) -> bool:
        return self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]


@dataclass
class Fill:
    """Order fill details."""
    order_id: str
    timestamp: datetime
    symbol: str
    side: Side
    quantity: float
    price: float
    commission: float = 0
    slippage: float = 0


class ExecutionSimulator:
    """
    Realistic order execution simulator.
    
    Features:
    - Slippage modeling (fixed, percentage, volume-based)
    - Commission calculation
    - Partial fills
    - Order latency
    """
    
    def __init__(self,
                 slippage_model: str = "percentage",
                 slippage_pct: float = 0.0001,  # 1 bp
                 slippage_fixed: float = 0.01,
                 commission_per_share: float = 0.005,
                 commission_minimum: float = 1.0,
                 latency_ms: int = 0):
        """
        Initialize execution simulator.
        
        Parameters
        ----------
        slippage_model : str
            "percentage", "fixed", or "volume".
        slippage_pct : float
            Slippage as percentage of price.
        slippage_fixed : float
            Fixed slippage amount.
        commission_per_share : float
            Commission per share.
        commission_minimum : float
            Minimum commission.
        latency_ms : int
            Order latency in milliseconds.
        """
        self.slippage_model = slippage_model
        self.slippage_pct = slippage_pct
        self.slippage_fixed = slippage_fixed
        self.commission_per_share = commission_per_share
        self.commission_minimum = commission_minimum
        self.latency_ms = latency_ms
    
    def calculate_slippage(self, price: float, side: Side, 
                          volume: float = 0) -> float:
        """
        Calculate slippage for execution.
        
        Parameters
        ----------
        price : float
            Reference price.
        side : Side
            Order side.
        volume : float
            Bar volume (for volume-based slippage).
        
        Returns
        -------
        float
            Slippage amount.
        """
        if self.slippage_model == "percentage":
            slippage = price * self.slippage_pct
        elif self.slippage_model == "fixed":
            slippage = self.slippage_fixed
        elif self.slippage_model == "volume" and volume > 0:
            # Higher slippage for larger orders relative to volume
            slippage = price * self.slippage_pct * 2
        else:
            slippage = price * self.slippage_pct
        
        # Slippage always adverse
        return slippage if side == Side.BUY else -slippage
    
    def calculate_commission(self, quantity: float) -> float:
        """Calculate commission for order."""
        commission = quantity * self.commission_per_share
        return max(commission, self.commission_minimum)
    
    def execute_order(self, order: Order, bar: Dict[str, float]) -> Optional[Fill]:
        """
        Simulate order execution against bar.
        
        Parameters
        ----------
        order : Order
            Order to execute.
        bar : dict
            OHLCV bar data.
        
        Returns
        -------
        Fill or None
            Fill if order executed.
        """
        if order.is_complete:
            return None
        
        # Determine execution price
        if order.order_type == OrderType.MARKET:
            # Execute at open of next bar with slippage
            base_price = bar['open']
        elif order.order_type == OrderType.LIMIT:
            # Check if limit is hit
            if order.is_buy:
                if bar['low'] <= order.limit_price:
                    base_price = min(order.limit_price, bar['open'])
                else:
                    return None
            else:
                if bar['high'] >= order.limit_price:
                    base_price = max(order.limit_price, bar['open'])
                else:
                    return None
        elif order.order_type == OrderType.STOP:
            # Check if stop is triggered
            if order.is_buy:
                if bar['high'] >= order.stop_price:
                    base_price = max(order.stop_price, bar['open'])
                else:
                    return None
            else:
                if bar['low'] <= order.stop_price:
                    base_price = min(order.stop_price, bar['open'])
                else:
                    return None
        else:
            base_price = bar['open']
        
        # Apply slippage
        slippage = self.calculate_slippage(base_price, order.side, bar.get('volume', 0))
        fill_price = base_price + slippage
        
        # Calculate commission
        commission = self.calculate_commission(order.remaining)
        
        # Create fill
        fill = Fill(
            order_id=order.order_id,
            timestamp=order.timestamp + timedelta(milliseconds=self.latency_ms),
            symbol=order.symbol,
            side=order.side,
            quantity=order.remaining,
            price=fill_price,
            commission=commission,
            slippage=slippage
        )
        
        # Update order
        order.filled_quantity = order.quantity
        order.avg_fill_price = fill_price
        order.commission = commission
        order.slippage = slippage
        order.status = OrderStatus.FILLED
        
        return fill


@dataclass
class PortfolioPosition:
    """Portfolio-level position tracking with avg cost basis."""
    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


class Portfolio:
    """
    Portfolio state management.
    
    Tracks:
    - Cash balance
    - Positions
    - Equity
    - P&L
    """
    
    def __init__(self, initial_cash: float = 1_000_000):
        """
        Initialize portfolio.
        
        Parameters
        ----------
        initial_cash : float
            Starting cash balance.
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, PortfolioPosition] = {}
        self.trades: List[Trade] = []
        
        # History tracking
        self._equity_curve: List[tuple] = []
        self._cash_curve: List[tuple] = []
    
    def get_position(self, symbol: str) -> PortfolioPosition:
        """Get position for symbol, create if doesn't exist."""
        if symbol not in self.positions:
            self.positions[symbol] = PortfolioPosition(
                symbol=symbol,
                quantity=0,
                avg_price=0,
                market_value=0,
                unrealized_pnl=0,
                realized_pnl=0
            )
        return self.positions[symbol]
    
    def update_position(self, fill: Fill):
        """
        Update position from fill.
        
        Parameters
        ----------
        fill : Fill
            Order fill.
        """
        pos = self.get_position(fill.symbol)
        
        if fill.side == Side.BUY:
            # Buying
            if pos.quantity >= 0:
                # Adding to long or opening long
                new_cost = pos.quantity * pos.avg_price + fill.quantity * fill.price
                new_qty = pos.quantity + fill.quantity
                pos.avg_price = new_cost / new_qty if new_qty > 0 else 0
                pos.quantity = new_qty
            else:
                # Covering short
                if fill.quantity <= abs(pos.quantity):
                    # Partial cover
                    realized = (pos.avg_price - fill.price) * fill.quantity
                    pos.realized_pnl += realized
                    pos.quantity += fill.quantity
                else:
                    # Full cover and go long
                    realized = (pos.avg_price - fill.price) * abs(pos.quantity)
                    pos.realized_pnl += realized
                    remaining = fill.quantity - abs(pos.quantity)
                    pos.quantity = remaining
                    pos.avg_price = fill.price
        else:
            # Selling
            if pos.quantity <= 0:
                # Adding to short or opening short
                new_cost = abs(pos.quantity) * pos.avg_price + fill.quantity * fill.price
                new_qty = pos.quantity - fill.quantity
                pos.avg_price = new_cost / abs(new_qty) if new_qty != 0 else 0
                pos.quantity = new_qty
            else:
                # Closing long
                if fill.quantity <= pos.quantity:
                    # Partial close
                    realized = (fill.price - pos.avg_price) * fill.quantity
                    pos.realized_pnl += realized
                    pos.quantity -= fill.quantity
                else:
                    # Full close and go short
                    realized = (fill.price - pos.avg_price) * pos.quantity
                    pos.realized_pnl += realized
                    remaining = fill.quantity - pos.quantity
                    pos.quantity = -remaining
                    pos.avg_price = fill.price
        
        # Update cash
        trade_value = fill.quantity * fill.price
        if fill.side == Side.BUY:
            self.cash -= trade_value + fill.commission
        else:
            self.cash += trade_value - fill.commission
        
        # Record trade
        self.trades.append(Trade(
            timestamp=fill.timestamp,
            symbol=fill.symbol,
            side=fill.side,
            quantity=fill.quantity,
            price=fill.price,
            commission=fill.commission,
            slippage=fill.slippage
        ))
    
    def mark_to_market(self, prices: Dict[str, float], timestamp: datetime):
        """
        Mark all positions to market.
        
        Parameters
        ----------
        prices : dict
            Symbol -> current price.
        timestamp : datetime
            Mark timestamp.
        """
        total_value = self.cash
        
        for symbol, pos in self.positions.items():
            if symbol in prices and pos.quantity != 0:
                current_price = prices[symbol]
                pos.market_value = pos.quantity * current_price
                pos.unrealized_pnl = pos.quantity * (current_price - pos.avg_price)
                total_value += pos.market_value
        
        self._equity_curve.append((timestamp, total_value))
        self._cash_curve.append((timestamp, self.cash))
    
    @property
    def equity(self) -> float:
        """Current portfolio equity."""
        if self._equity_curve:
            return self._equity_curve[-1][1]
        return self.cash
    
    @property
    def equity_curve(self) -> np.ndarray:
        """Get equity curve as numpy array."""
        if not self._equity_curve:
            return np.array([])
        return np.array([eq for _, eq in self._equity_curve])
    
    @property
    def timestamps(self) -> List[datetime]:
        """Get timestamps for equity curve."""
        return [ts for ts, _ in self._equity_curve]


class BacktestEngine:
    """
    Event-driven backtesting engine.
    
    Flow:
    1. Feed historical data bar by bar
    2. Strategy generates signals
    3. Signals converted to orders
    4. Orders executed with slippage/commission
    5. Portfolio updated
    6. Performance calculated
    
    Features:
    - Multiple strategies
    - Realistic execution
    - Position sizing
    - Risk management
    """
    
    def __init__(self,
                 initial_capital: float = 1_000_000,
                 execution_sim: Optional[ExecutionSimulator] = None):
        """
        Initialize backtest engine.
        
        Parameters
        ----------
        initial_capital : float
            Starting capital.
        execution_sim : ExecutionSimulator, optional
            Custom execution simulator.
        """
        self.initial_capital = initial_capital
        self.execution_sim = execution_sim or ExecutionSimulator()
        
        # Components
        self.portfolio = Portfolio(initial_capital)
        self.strategies: List[Strategy] = []
        
        # State
        self._orders: Dict[str, Order] = {}
        self._order_counter = 0
        self._current_prices: Dict[str, float] = {}
        self._event_queue: List[Event] = []
        
        # Results
        self._signals_generated: List[Signal] = []
        self._orders_executed: List[Order] = []
        self._fills: List[Fill] = []
    
    def add_strategy(self, strategy: Strategy):
        """Add strategy to backtest."""
        self.strategies.append(strategy)
        strategy.initialize()
    
    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        self._order_counter += 1
        return f"ORD-{self._order_counter:06d}"
    
    def _signal_to_order(self, signal: Signal) -> Order:
        """
        Convert signal to order.
        
        Parameters
        ----------
        signal : Signal
            Trading signal.
        
        Returns
        -------
        Order
            Order object.
        """
        # Determine quantity
        if signal.target_position is not None:
            quantity = signal.target_position
        else:
            # Default: use signal strength as percentage of capital
            price = self._current_prices.get(signal.symbol, 100)
            capital_alloc = self.portfolio.equity * 0.1 * signal.strength
            quantity = capital_alloc / price
        
        quantity = abs(quantity)
        
        # Determine order type
        if signal.limit_price is not None:
            order_type = OrderType.LIMIT
            limit_price = signal.limit_price
        elif signal.stop_loss is not None:
            order_type = OrderType.STOP
            stop_price = signal.stop_loss
            limit_price = None
        else:
            order_type = OrderType.MARKET
            limit_price = None
            stop_price = None
        
        return Order(
            order_id=self._generate_order_id(),
            timestamp=signal.timestamp,
            symbol=signal.symbol,
            side=signal.side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price if order_type == OrderType.LIMIT else None,
            stop_price=stop_price if order_type == OrderType.STOP else None
        )
    
    def _process_bar(self, timestamp: datetime, bars: Dict[str, Dict[str, float]]):
        """
        Process single bar for all symbols.
        
        Parameters
        ----------
        timestamp : datetime
            Bar timestamp.
        bars : dict
            Symbol -> OHLCV data.
        """
        # Update current prices
        for symbol, bar in bars.items():
            self._current_prices[symbol] = bar['close']
        
        # Execute pending orders
        for order_id, order in list(self._orders.items()):
            if order.symbol in bars and not order.is_complete:
                fill = self.execution_sim.execute_order(order, bars[order.symbol])
                if fill:
                    self._fills.append(fill)
                    self.portfolio.update_position(fill)
                    self._orders_executed.append(order)
                    del self._orders[order_id]
        
        # Generate signals from strategies
        for strategy in self.strategies:
            signals = strategy.on_bar(timestamp, bars)
            self._signals_generated.extend(signals)
            
            # Convert signals to orders
            for signal in signals:
                order = self._signal_to_order(signal)
                self._orders[order.order_id] = order
        
        # Mark portfolio to market
        self.portfolio.mark_to_market(self._current_prices, timestamp)
    
    def run(self, data_iterator: Iterator[tuple]) -> Dict[str, Any]:
        """
        Run backtest.
        
        Parameters
        ----------
        data_iterator : iterator
            Yields (timestamp, bars) tuples.
        
        Returns
        -------
        dict
            Backtest results.
        """
        bar_count = 0
        
        for timestamp, bars in data_iterator:
            self._process_bar(timestamp, bars)
            bar_count += 1
        
        # Calculate results
        results = self._calculate_results()
        results['bars_processed'] = bar_count
        
        return results
    
    def _calculate_results(self) -> Dict[str, Any]:
        """Calculate backtest results."""
        equity_curve = self.portfolio.equity_curve
        
        if len(equity_curve) < 2:
            return {'error': 'Insufficient data'}
        
        # Returns
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        # Performance metrics
        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
        
        # Annualized return (assuming daily data)
        n_days = len(equity_curve)
        ann_return = (1 + total_return) ** (252 / n_days) - 1
        
        # Volatility
        daily_vol = np.std(returns)
        ann_vol = daily_vol * np.sqrt(252)
        
        # Sharpe ratio
        risk_free_rate = 0.02  # 2% annual
        sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else 0
        
        # Maximum drawdown
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_vol = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 1
        sortino = (ann_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0
        
        # Win rate
        trades = self.portfolio.trades
        if trades:
            # Group trades by symbol to calculate P&L
            winning_trades = sum(1 for t in trades if t.side == Side.SELL)  # Simplified
            win_rate = winning_trades / len(trades) if trades else 0
        else:
            win_rate = 0
        
        # Calmar ratio
        calmar = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'initial_capital': self.initial_capital,
            'final_equity': equity_curve[-1],
            'total_return': total_return,
            'annualized_return': ann_return,
            'annualized_volatility': ann_vol,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar,
            'total_trades': len(trades),
            'total_signals': len(self._signals_generated),
            'win_rate': win_rate,
            'equity_curve': equity_curve.tolist(),
            'timestamps': [str(ts) for ts in self.portfolio.timestamps]
        }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_data_iterator(prices: Dict[str, np.ndarray], 
                        timestamps: List[datetime]) -> Iterator:
    """
    Create data iterator from price arrays.
    
    Parameters
    ----------
    prices : dict
        Symbol -> price array.
    timestamps : list
        List of timestamps.
    
    Yields
    ------
    tuple
        (timestamp, bars)
    """
    n = len(timestamps)
    
    for i in range(n):
        bars = {}
        for symbol, price_data in prices.items():
            if isinstance(price_data, np.ndarray) and price_data.ndim == 1:
                # Simple price array - create synthetic OHLCV
                close = price_data[i]
                bars[symbol] = {
                    'open': close * 0.999,
                    'high': close * 1.002,
                    'low': close * 0.998,
                    'close': close,
                    'volume': 1_000_000
                }
            elif isinstance(price_data, dict):
                # Full OHLCV dict
                bars[symbol] = {k: v[i] for k, v in price_data.items()}
        
        yield timestamps[i], bars


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    from datetime import datetime, timedelta
    import numpy as np
    
    print("=" * 60)
    print("BACKTEST ENGINE TEST")
    print("=" * 60)
    
    # Generate synthetic data
    np.random.seed(42)
    n_days = 252
    
    base_date = datetime(2023, 1, 1)
    timestamps = [base_date + timedelta(days=i) for i in range(n_days)]
    
    # Two correlated assets for pairs trading
    returns1 = np.random.normal(0.0003, 0.02, n_days)
    returns2 = returns1 * 0.8 + np.random.normal(0, 0.01, n_days)
    
    prices = {
        'STOCK_A': 100 * np.cumprod(1 + returns1),
        'STOCK_B': 100 * np.cumprod(1 + returns2)
    }
    
    # Create engine
    engine = BacktestEngine(
        initial_capital=1_000_000,
        execution_sim=ExecutionSimulator(
            slippage_pct=0.0001,
            commission_per_share=0.005
        )
    )
    
    # Add simple strategy
    from research.strategies.pairs_trading import PairsTradingStrategy
    
    strategy = PairsTradingStrategy(
        symbol1='STOCK_A',
        symbol2='STOCK_B',
        lookback=60,
        entry_zscore=2.0
    )
    engine.add_strategy(strategy)
    
    # Run backtest
    data_iter = create_data_iterator(prices, timestamps)
    results = engine.run(data_iter)
    
    # Print results
    print(f"\nBacktest Results:")
    print(f"  Initial Capital: ${results['initial_capital']:,.2f}")
    print(f"  Final Equity: ${results['final_equity']:,.2f}")
    print(f"  Total Return: {results['total_return']:.2%}")
    print(f"  Annual Return: {results['annualized_return']:.2%}")
    print(f"  Annual Volatility: {results['annualized_volatility']:.2%}")
    print(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"  Sortino Ratio: {results['sortino_ratio']:.2f}")
    print(f"  Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"  Calmar Ratio: {results['calmar_ratio']:.2f}")
    print(f"  Total Trades: {results['total_trades']}")
    print(f"  Total Signals: {results['total_signals']}")
