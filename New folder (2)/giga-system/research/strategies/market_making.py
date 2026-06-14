"""
GIGA SYSTEM - Market Making Strategy
High-frequency bid-ask spread capture with inventory management
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import deque

from .base import Strategy, Signal, Side


@dataclass
class OrderBook:
    """Order book snapshot."""
    timestamp: datetime
    symbol: str
    bids: List[Tuple[float, float]]  # (price, size) sorted desc by price
    asks: List[Tuple[float, float]]  # (price, size) sorted asc by price
    
    @property
    def best_bid(self) -> float:
        return self.bids[0][0] if self.bids else 0.0
    
    @property
    def best_ask(self) -> float:
        return self.asks[0][0] if self.asks else float('inf')
    
    @property
    def mid_price(self) -> float:
        return (self.best_bid + self.best_ask) / 2
    
    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid
    
    @property
    def spread_bps(self) -> float:
        """Spread in basis points."""
        return self.spread / self.mid_price * 10000
    
    @property
    def imbalance(self) -> float:
        """Order book imbalance [-1, 1]. Positive = more bids."""
        bid_vol = sum(size for _, size in self.bids[:5])
        ask_vol = sum(size for _, size in self.asks[:5])
        total = bid_vol + ask_vol
        if total == 0:
            return 0.0
        return (bid_vol - ask_vol) / total


@dataclass
class Quote:
    """Market maker quote."""
    symbol: str
    bid_price: float
    bid_size: float
    ask_price: float
    ask_size: float
    timestamp: datetime
    
    @property
    def mid(self) -> float:
        return (self.bid_price + self.ask_price) / 2
    
    @property
    def spread(self) -> float:
        return self.ask_price - self.bid_price


class AvellanedaStoikovMM(Strategy):
    """
    Avellaneda-Stoikov Market Making Strategy.
    
    Mathematical Foundation:
    The optimal bid and ask prices are derived from the solution to:
    
    Reservation Price:
    r(s,q,t) = s - q * γ * σ² * (T-t)
    
    Optimal Spread:
    δ(s,q,t) = γ * σ² * (T-t) + (2/γ) * ln(1 + γ/k)
    
    Where:
    - s = mid price
    - q = inventory position
    - γ = risk aversion parameter
    - σ = volatility
    - T-t = time to end of trading
    - k = market order arrival rate parameter
    
    Quote Placement:
    bid = r - δ/2
    ask = r + δ/2
    
    Key Properties:
    - Quotes adjust based on inventory (skew)
    - Higher volatility → wider spreads
    - Risk aversion controls inventory penalty
    """
    
    def __init__(self,
                 symbol: str,
                 gamma: float = 0.1,          # Risk aversion
                 k: float = 1.5,              # Order arrival parameter
                 base_spread_bps: float = 2.0, # Minimum spread in bps
                 max_position: float = 1000,   # Max inventory
                 vol_window: int = 20,         # Volatility calculation window
                 quote_size: float = 100):     # Default quote size
        """
        Initialize Avellaneda-Stoikov market maker.
        
        Parameters
        ----------
        symbol : str
            Symbol to market make.
        gamma : float
            Risk aversion (higher = more inventory penalty).
        k : float
            Order arrival rate parameter.
        base_spread_bps : float
            Minimum spread in basis points.
        max_position : float
            Maximum inventory position.
        vol_window : int
            Window for volatility estimation.
        quote_size : float
            Default quote size.
        """
        super().__init__(
            name="Avellaneda_Stoikov_MM",
            symbols=[symbol],
            gamma=gamma,
            k=k
        )
        
        self.symbol = symbol
        self.gamma = gamma
        self.k = k
        self.base_spread_bps = base_spread_bps
        self.max_position = max_position
        self.vol_window = vol_window
        self.quote_size = quote_size
        
        # State
        self._inventory: float = 0  # Current position
        self._mid_prices: deque = deque(maxlen=vol_window)
        self._current_quote: Optional[Quote] = None
        self._pnl: float = 0
        self._trades_made: int = 0
        
        # Trading session
        self._session_start: Optional[datetime] = None
        self._session_end: Optional[datetime] = None
    
    def initialize(self, session_duration_hours: float = 6.5, **kwargs):
        """
        Initialize for trading session.
        
        Parameters
        ----------
        session_duration_hours : float
            Trading session length (default 6.5 = regular market hours).
        """
        super().initialize(**kwargs)
        self._inventory = 0
        self._mid_prices.clear()
        self._current_quote = None
        self._pnl = 0
        self._trades_made = 0
        
        # Will be set on first update
        self._session_start = None
        self._session_duration = session_duration_hours * 3600  # seconds
    
    # =========================================================================
    # VOLATILITY ESTIMATION
    # =========================================================================
    
    def estimate_volatility(self) -> float:
        """
        Estimate instantaneous volatility from mid prices.
        
        σ = std(returns) * √(freq)
        
        Returns
        -------
        float
            Annualized volatility.
        """
        if len(self._mid_prices) < 2:
            return 0.20  # Default 20% vol
        
        prices = np.array(self._mid_prices)
        returns = np.diff(np.log(prices))
        
        if len(returns) == 0:
            return 0.20
        
        # Assuming 1-second updates, annualize
        # ~252 days * 6.5 hours * 3600 seconds
        freq = 252 * 6.5 * 3600
        volatility = np.std(returns) * np.sqrt(freq)
        
        # Clip to reasonable range
        return np.clip(volatility, 0.05, 2.0)
    
    # =========================================================================
    # AVELLANEDA-STOIKOV MODEL
    # =========================================================================
    
    def calculate_reservation_price(self, mid: float, time_remaining: float,
                                    sigma: float) -> float:
        """
        Calculate reservation price.
        
        r = s - q * γ * σ² * (T-t)
        
        Parameters
        ----------
        mid : float
            Current mid price.
        time_remaining : float
            Time remaining in session (normalized 0-1).
        sigma : float
            Volatility.
        
        Returns
        -------
        float
            Reservation price.
        """
        inventory_adjustment = (self._inventory * self.gamma * 
                                sigma ** 2 * time_remaining)
        
        return mid - inventory_adjustment
    
    def calculate_optimal_spread(self, time_remaining: float, 
                                 sigma: float) -> float:
        """
        Calculate optimal spread.
        
        δ = γ * σ² * (T-t) + (2/γ) * ln(1 + γ/k)
        
        Parameters
        ----------
        time_remaining : float
            Time remaining (normalized).
        sigma : float
            Volatility.
        
        Returns
        -------
        float
            Optimal spread.
        """
        # Time component
        time_component = self.gamma * sigma ** 2 * time_remaining
        
        # Inventory risk component
        inventory_component = (2 / self.gamma) * np.log(1 + self.gamma / self.k)
        
        optimal_spread = time_component + inventory_component
        
        return optimal_spread
    
    def calculate_quotes(self, book: OrderBook, 
                        time_remaining: float) -> Tuple[float, float]:
        """
        Calculate optimal bid and ask prices.
        
        Parameters
        ----------
        book : OrderBook
            Current order book.
        time_remaining : float
            Time remaining in session.
        
        Returns
        -------
        tuple
            (bid_price, ask_price)
        """
        mid = book.mid_price
        sigma = self.estimate_volatility()
        
        # Reservation price (inventory-adjusted mid)
        reservation = self.calculate_reservation_price(mid, time_remaining, sigma)
        
        # Optimal spread
        spread = self.calculate_optimal_spread(time_remaining, sigma)
        
        # Ensure minimum spread
        min_spread = mid * self.base_spread_bps / 10000
        spread = max(spread, min_spread)
        
        # Calculate quotes
        bid_price = reservation - spread / 2
        ask_price = reservation + spread / 2
        
        # Adjust for order book imbalance
        imbalance = book.imbalance
        skew = imbalance * spread * 0.25  # Skew up to 25% of spread
        
        bid_price -= skew
        ask_price -= skew
        
        return bid_price, ask_price
    
    # =========================================================================
    # INVENTORY MANAGEMENT
    # =========================================================================
    
    def calculate_quote_sizes(self) -> Tuple[float, float]:
        """
        Calculate bid and ask sizes based on inventory.
        
        Reduce size on side that would increase inventory risk.
        
        Returns
        -------
        tuple
            (bid_size, ask_size)
        """
        base_size = self.quote_size
        
        # Inventory ratio [-1, 1]
        inv_ratio = self._inventory / self.max_position
        
        # Reduce bid size when long, ask size when short
        if inv_ratio > 0:
            bid_size = base_size * (1 - inv_ratio * 0.5)
            ask_size = base_size * (1 + inv_ratio * 0.5)
        else:
            bid_size = base_size * (1 + abs(inv_ratio) * 0.5)
            ask_size = base_size * (1 - abs(inv_ratio) * 0.5)
        
        # Ensure minimum size
        bid_size = max(1, bid_size)
        ask_size = max(1, ask_size)
        
        return bid_size, ask_size
    
    def should_quote(self) -> Tuple[bool, bool]:
        """
        Determine if should quote bid/ask.
        
        Returns
        -------
        tuple
            (quote_bid, quote_ask)
        """
        quote_bid = True
        quote_ask = True
        
        # Don't buy if at max long
        if self._inventory >= self.max_position:
            quote_bid = False
        
        # Don't sell if at max short
        if self._inventory <= -self.max_position:
            quote_ask = False
        
        return quote_bid, quote_ask
    
    # =========================================================================
    # SIGNAL GENERATION
    # =========================================================================
    
    def on_book_update(self, book: OrderBook) -> List[Signal]:
        """
        Process order book update and generate quotes.
        
        Parameters
        ----------
        book : OrderBook
            Current order book snapshot.
        
        Returns
        -------
        list
            Quote signals.
        """
        signals = []
        
        # Set session start if first update
        if self._session_start is None:
            self._session_start = book.timestamp
        
        # Update mid price history
        self._mid_prices.append(book.mid_price)
        
        # Calculate time remaining
        elapsed = (book.timestamp - self._session_start).total_seconds()
        time_remaining = max(0, 1 - elapsed / self._session_duration)
        
        # End of session - flatten inventory
        if time_remaining < 0.01:
            return self._generate_flatten_signals(book.timestamp, book.mid_price)
        
        # Calculate optimal quotes
        bid_price, ask_price = self.calculate_quotes(book, time_remaining)
        bid_size, ask_size = self.calculate_quote_sizes()
        quote_bid, quote_ask = self.should_quote()
        
        # Generate quote signals
        if quote_bid:
            signals.append(Signal(
                timestamp=book.timestamp,
                symbol=self.symbol,
                side=Side.BUY,
                strength=1.0,
                confidence=0.8,
                limit_price=round(bid_price, 2),
                target_position=bid_size,
                metadata={
                    'strategy': 'avellaneda_stoikov',
                    'action': 'quote_bid',
                    'mid': book.mid_price,
                    'spread': ask_price - bid_price,
                    'inventory': self._inventory,
                    'time_remaining': time_remaining,
                    'volatility': self.estimate_volatility()
                }
            ))
        
        if quote_ask:
            signals.append(Signal(
                timestamp=book.timestamp,
                symbol=self.symbol,
                side=Side.SELL,
                strength=1.0,
                confidence=0.8,
                limit_price=round(ask_price, 2),
                target_position=ask_size,
                metadata={
                    'strategy': 'avellaneda_stoikov',
                    'action': 'quote_ask',
                    'mid': book.mid_price,
                    'spread': ask_price - bid_price,
                    'inventory': self._inventory,
                    'time_remaining': time_remaining
                }
            ))
        
        # Store current quote
        self._current_quote = Quote(
            symbol=self.symbol,
            bid_price=bid_price if quote_bid else 0,
            bid_size=bid_size if quote_bid else 0,
            ask_price=ask_price if quote_ask else 0,
            ask_size=ask_size if quote_ask else 0,
            timestamp=book.timestamp
        )
        
        self._signals.extend(signals)
        return signals
    
    def _generate_flatten_signals(self, timestamp: datetime, 
                                  mid_price: float) -> List[Signal]:
        """Generate signals to flatten inventory at end of session."""
        if self._inventory == 0:
            return []
        
        signals = [Signal(
            timestamp=timestamp,
            symbol=self.symbol,
            side=Side.SELL if self._inventory > 0 else Side.BUY,
            strength=1.0,
            confidence=1.0,
            metadata={
                'strategy': 'avellaneda_stoikov',
                'action': 'flatten',
                'inventory': self._inventory,
                'mid': mid_price
            }
        )]
        
        return signals
    
    def on_fill(self, timestamp: datetime, side: Side, 
                price: float, size: float):
        """
        Process fill notification.
        
        Parameters
        ----------
        timestamp : datetime
            Fill timestamp.
        side : Side
            BUY or SELL.
        price : float
            Fill price.
        size : float
            Fill size.
        """
        if side == Side.BUY:
            self._inventory += size
        else:
            self._inventory -= size
        
        self._trades_made += 1
        
        # Track P&L (simplified)
        if self._current_quote:
            if side == Side.BUY:
                edge = self._current_quote.mid - price
            else:
                edge = price - self._current_quote.mid
            self._pnl += edge * size
    
    def finalize(self) -> Dict[str, Any]:
        """Return market making summary."""
        base_summary = super().finalize()
        
        base_summary.update({
            'final_inventory': self._inventory,
            'total_trades': self._trades_made,
            'estimated_pnl': self._pnl,
            'avg_volatility': self.estimate_volatility()
        })
        
        return base_summary


class SimpleSpreadMM(Strategy):
    """
    Simple Spread-Based Market Maker.
    
    Simpler alternative to Avellaneda-Stoikov:
    - Fixed spread around mid
    - Linear inventory skew
    - No stochastic calculus required
    
    Good for:
    - Less liquid markets
    - Simpler implementation
    - Lower frequency
    """
    
    def __init__(self,
                 symbol: str,
                 spread_bps: float = 5.0,
                 skew_factor: float = 0.5,
                 max_position: float = 1000,
                 quote_size: float = 100):
        """
        Initialize simple market maker.
        
        Parameters
        ----------
        symbol : str
            Symbol to market make.
        spread_bps : float
            Spread in basis points (default 5 bps).
        skew_factor : float
            How much to skew for inventory (0-1).
        max_position : float
            Maximum inventory.
        quote_size : float
            Default quote size.
        """
        super().__init__(
            name="Simple_Spread_MM",
            symbols=[symbol],
            spread_bps=spread_bps
        )
        
        self.symbol = symbol
        self.spread_bps = spread_bps
        self.skew_factor = skew_factor
        self.max_position = max_position
        self.quote_size = quote_size
        
        self._inventory: float = 0
    
    def initialize(self, **kwargs):
        """Initialize strategy."""
        super().initialize(**kwargs)
        self._inventory = 0
    
    def calculate_quotes(self, mid: float) -> Tuple[float, float]:
        """
        Calculate bid and ask prices.
        
        Parameters
        ----------
        mid : float
            Current mid price.
        
        Returns
        -------
        tuple
            (bid, ask)
        """
        half_spread = mid * self.spread_bps / 10000 / 2
        
        # Inventory skew
        inv_ratio = self._inventory / self.max_position if self.max_position > 0 else 0
        skew = inv_ratio * half_spread * self.skew_factor
        
        bid = mid - half_spread - skew
        ask = mid + half_spread - skew
        
        return bid, ask
    
    def on_bar(self, timestamp: datetime, bars: Dict[str, Dict[str, float]]) -> List[Signal]:
        """Process bar and generate quotes."""
        if self.symbol not in bars:
            return []
        
        bar = bars[self.symbol]
        mid = (bar['high'] + bar['low']) / 2
        
        bid, ask = self.calculate_quotes(mid)
        
        signals = []
        
        # Quote bid if not at max long
        if self._inventory < self.max_position:
            signals.append(Signal(
                timestamp=timestamp,
                symbol=self.symbol,
                side=Side.BUY,
                strength=1.0,
                confidence=0.7,
                limit_price=round(bid, 2),
                target_position=self.quote_size,
                metadata={
                    'strategy': 'simple_mm',
                    'action': 'quote_bid',
                    'mid': mid,
                    'inventory': self._inventory
                }
            ))
        
        # Quote ask if not at max short
        if self._inventory > -self.max_position:
            signals.append(Signal(
                timestamp=timestamp,
                symbol=self.symbol,
                side=Side.SELL,
                strength=1.0,
                confidence=0.7,
                limit_price=round(ask, 2),
                target_position=self.quote_size,
                metadata={
                    'strategy': 'simple_mm',
                    'action': 'quote_ask',
                    'mid': mid,
                    'inventory': self._inventory
                }
            ))
        
        self._signals.extend(signals)
        return signals


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    from datetime import datetime, timedelta
    
    print("=" * 60)
    print("MARKET MAKING STRATEGY TEST")
    print("=" * 60)
    
    # Test Avellaneda-Stoikov
    mm = AvellanedaStoikovMM(
        symbol="AAPL",
        gamma=0.1,
        k=1.5,
        base_spread_bps=2.0,
        max_position=1000
    )
    mm.initialize(session_duration_hours=6.5)
    
    # Simulate order book updates using real SPY price movements
    try:
        from data.realtime_manager import get_data_manager
        import datetime as dt
        
        manager = get_data_manager()
        end_date = dt.datetime.now()
        start_date = end_date - dt.timedelta(days=5)
        
        spy_data = manager.get_historical_data_sync('SPY', start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), '1m')
        prices = spy_data['close'].values[-100:]
        mid = prices[0]
        
        print(f"Using real SPY minute data for order book simulation")
    except Exception as e:
        print(f"  Real SPY data unavailable: {e}")
        print("  Market making demonstration requires SPY 1-minute historical data")
        import sys
        sys.exit(0)
    
    base_time = datetime(2023, 6, 15, 9, 30)
    
    for i in range(100):
        mid = prices[i]
        
        # Synthetic order book
        book = OrderBook(
            timestamp=base_time + timedelta(seconds=i),
            symbol="AAPL",
            bids=[
                (mid - 0.01, 100),
                (mid - 0.02, 200),
                (mid - 0.03, 300),
            ],
            asks=[
                (mid + 0.01, 100),
                (mid + 0.02, 200),
                (mid + 0.03, 300),
            ]
        )
        
        signals = mm.on_book_update(book)
        
        # Simulate random fills
        if np.random.random() < 0.1:
            side = Side.BUY if np.random.random() < 0.5 else Side.SELL
            fill_price = mid - 0.01 if side == Side.BUY else mid + 0.01
            mm.on_fill(book.timestamp, side, fill_price, 50)
    
    summary = mm.finalize()
    
    print(f"\nAvellaneda-Stoikov Summary:")
    print(f"  Total signals: {summary['total_signals']}")
    print(f"  Final inventory: {summary['final_inventory']}")
    print(f"  Trades made: {summary['total_trades']}")
    print(f"  Estimated P&L: ${summary['estimated_pnl']:.2f}")
    print(f"  Volatility: {summary['avg_volatility']:.2%}")
    
    # Test Simple MM
    print("\n" + "-" * 40)
    print("Simple Spread Market Maker")
    
    simple_mm = SimpleSpreadMM(
        symbol="AAPL",
        spread_bps=5.0,
        max_position=1000
    )
    simple_mm.initialize()
    
    signals = simple_mm.on_bar(
        timestamp=datetime.now(),
        bars={
            "AAPL": {
                'open': 150.0,
                'high': 150.5,
                'low': 149.5,
                'close': 150.2,
                'volume': 1000000
            }
        }
    )
    
    print(f"  Signals generated: {len(signals)}")
    for s in signals:
        print(f"    {s.side.name}: ${s.limit_price}")
