"""
High-Frequency Order Manager
Ultra-low latency order management system

Features:
- Microsecond-level order tracking
- Pre-trade risk checks (<100μs)
- Order book integration
- Fill simulation with realistic slippage
"""

import time
import logging
import numpy as np
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)
from dataclasses import dataclass, field
from datetime import datetime
import threading


class OrderType(Enum):
    """Order types for execution."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill
    ICEBERG = "ICEBERG"


class OrderSide(Enum):
    """Order side."""
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    """Order lifecycle status."""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"     # Sent to exchange
    ACKNOWLEDGED = "ACKNOWLEDGED" # Exchange confirmed receipt (Phase 11)
    PARTIAL_FILL = "PARTIAL_FILL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class Order:
    """HFT-level order representation."""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: Optional[float] = None
    stop_price: Optional[float] = None
    
    # HFT-specific fields
    order_id: str = field(default_factory=lambda: f"ORD_{int(time.time()*1e6)}")
    timestamp_created: float = field(default_factory=lambda: time.perf_counter())
    timestamp_submitted: Optional[float] = None
    timestamp_acked: Optional[float] = None   # Phase 11: Ack timestamp
    timestamp_filled: Optional[float] = None
    
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    
    # Execution metadata
    latency_submit_us: Optional[float] = None
    latency_ack_us: Optional[float] = None    # Phase 11: Time to ACK
    latency_fill_us: Optional[float] = None
    exchange_order_id: Optional[str] = None
    
    # Risk limits
    max_slippage_bps: int = 10  # 10 basis points max slippage
    time_in_force: str = "DAY"
    
    def is_active(self) -> bool:
        """Check if order is still active."""
        return self.status in [
            OrderStatus.PENDING, 
            OrderStatus.SUBMITTED, 
            OrderStatus.ACKNOWLEDGED,
            OrderStatus.PARTIAL_FILL
        ]
    
    def remaining_quantity(self) -> int:
        """Get unfilled quantity."""
        return self.quantity - self.filled_quantity
    
    def calculate_latency(self, phase: str) -> float:
        """Calculate latency in microseconds."""
        if phase == "submit" and self.timestamp_submitted:
            latency = (self.timestamp_submitted - self.timestamp_created) * 1e6
            self.latency_submit_us = latency
            return latency
        elif phase == "ack" and self.timestamp_acked:
            latency = (self.timestamp_acked - self.timestamp_submitted) * 1e6
            self.latency_ack_us = latency
            return latency
        elif phase == "fill" and self.timestamp_filled:
            # Fill latency is from ACK to FILL (Phase 11 definition)
            start = self.timestamp_acked if self.timestamp_acked else self.timestamp_submitted
            latency = (self.timestamp_filled - start) * 1e6
            self.latency_fill_us = latency
            return latency
        return 0.0


class OrderManager:
    """
    High-frequency order management system.
    
    Manages order lifecycle with microsecond precision:
    - Pre-trade risk checks (<100μs)
    - Order routing and tracking
    - Fill management
    - Position tracking
    """
    
    def __init__(self, initial_capital: float = 1_000_000):
        """Initialize order manager."""
        self.capital = initial_capital
        self.initial_capital = initial_capital
        
        # Order tracking
        self.active_orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        
        # Position tracking
        self.positions: Dict[str, int] = {}  # symbol -> quantity
        self.avg_prices: Dict[str, float] = {}  # symbol -> avg price
        
        # Risk limits
        self.max_position_size: Dict[str, int] = {}
        self.max_order_value: float = 100_000
        self.max_daily_loss: float = 50_000
        self.daily_pnl: float = 0.0
        
        # Performance metrics
        self.total_orders: int = 0
        self.filled_orders: int = 0
        self.rejected_orders: int = 0
        self.avg_latency_us: float = 0.0
        
        # Thread safety
        self.lock = threading.Lock()
    
    def pre_trade_risk_check(self, order: Order, current_price: float) -> Tuple[bool, str]:
        """
        Ultra-fast pre-trade risk check.
        Target: <100 microseconds
        
        Returns:
            (is_valid, reason)
        """
        start = time.perf_counter()
        
        # Check 1: Order value limit
        order_value = order.quantity * current_price
        if order_value > self.max_order_value:
            return False, f"Order value ${order_value:,.0f} exceeds limit ${self.max_order_value:,.0f}"
        
        # Check 2: Daily loss limit
        if self.daily_pnl < -self.max_daily_loss:
            return False, f"Daily loss limit breached: ${self.daily_pnl:,.0f}"
        
        # Check 3: Position limits
        current_position = self.positions.get(order.symbol, 0)
        new_position = current_position + (order.quantity if order.side == OrderSide.BUY else -order.quantity)
        
        if order.symbol in self.max_position_size:
            if abs(new_position) > self.max_position_size[order.symbol]:
                return False, f"Position limit exceeded for {order.symbol}"
        
        # Check 4: Sufficient capital
        required_capital = order_value if order.side == OrderSide.BUY else 0
        if required_capital > self.capital:
            return False, f"Insufficient capital: ${self.capital:,.0f} < ${required_capital:,.0f}"
        
        # Check 5: Slippage tolerance
        if order.price and order.order_type == OrderType.LIMIT:
            slippage = abs(current_price - order.price) / current_price * 10000  # bps
            if slippage > order.max_slippage_bps:
                return False, f"Slippage {slippage:.1f}bps exceeds limit {order.max_slippage_bps}bps"
        
        elapsed_us = (time.perf_counter() - start) * 1e6
        return True, f"Risk check passed in {elapsed_us:.1f}μs"
    
    def submit_order(self, order: Order, current_price: float) -> Tuple[bool, str]:
        """
        Submit order with pre-trade checks.
        
        Returns:
            (success, message)
        """
        with self.lock:
            # Pre-trade risk check
            is_valid, reason = self.pre_trade_risk_check(order, current_price)
            if not is_valid:
                order.status = OrderStatus.REJECTED
                self.rejected_orders += 1
                self.order_history.append(order)
                return False, reason
            
            # Submit order
            order.timestamp_submitted = time.perf_counter()
            order.status = OrderStatus.SUBMITTED
            order.calculate_latency("submit")
            
            self.active_orders[order.order_id] = order
            self.total_orders += 1
            
            return True, f"Order {order.order_id} submitted (latency: {order.latency_submit_us:.1f}μs)"
    
    def fill_order(self, order_id: str, fill_price: float, fill_quantity: int) -> bool:
        """
        Process order fill.
        
        Returns:
            success
        """
        with self.lock:
            if order_id not in self.active_orders:
                return False
            
            order = self.active_orders[order_id]
            order.timestamp_filled = time.perf_counter()
            
            # Update fill info
            total_filled = order.filled_quantity + fill_quantity
            order.avg_fill_price = (
                (order.avg_fill_price * order.filled_quantity + fill_price * fill_quantity) / total_filled
            )
            order.filled_quantity = total_filled
            
            # Update position
            current_pos = self.positions.get(order.symbol, 0)
            current_avg = self.avg_prices.get(order.symbol, 0.0)
            
            if order.side == OrderSide.BUY:
                new_pos = current_pos + fill_quantity
                self.capital -= fill_price * fill_quantity
                
                # Update weighted average price (BUG#5 FIX)
                if current_pos >= 0:
                    # Adding to long or opening long
                    total_cost = current_avg * current_pos + fill_price * fill_quantity
                    self.avg_prices[order.symbol] = total_cost / new_pos if new_pos != 0 else 0.0
                else:
                    # Covering short — realize P&L (BUG#3 FIX)
                    closed_qty = min(fill_quantity, abs(current_pos))
                    realized_pnl = (current_avg - fill_price) * closed_qty  # short: sold high, bought low
                    self.daily_pnl += realized_pnl
                    remaining_buy = fill_quantity - closed_qty
                    if remaining_buy > 0:
                        # Flipped to long
                        self.avg_prices[order.symbol] = fill_price
                    elif new_pos < 0:
                        pass  # Still short, avg stays same
                    else:
                        self.avg_prices[order.symbol] = 0.0  # Flat
            else:
                new_pos = current_pos - fill_quantity
                self.capital += fill_price * fill_quantity
                
                # Update weighted average price (BUG#5 FIX)
                if current_pos <= 0:
                    # Adding to short or opening short
                    total_cost = current_avg * abs(current_pos) + fill_price * fill_quantity
                    self.avg_prices[order.symbol] = total_cost / abs(new_pos) if new_pos != 0 else 0.0
                else:
                    # Closing long — realize P&L (BUG#3 FIX)
                    closed_qty = min(fill_quantity, current_pos)
                    realized_pnl = (fill_price - current_avg) * closed_qty  # long: bought low, sold high
                    self.daily_pnl += realized_pnl
                    remaining_sell = fill_quantity - closed_qty
                    if remaining_sell > 0:
                        # Flipped to short
                        self.avg_prices[order.symbol] = fill_price
                    elif new_pos > 0:
                        pass  # Still long, avg stays same
                    else:
                        self.avg_prices[order.symbol] = 0.0  # Flat
            
            self.positions[order.symbol] = new_pos
            
            # Clean up flat positions
            if new_pos == 0 and order.symbol in self.avg_prices:
                del self.avg_prices[order.symbol]
            
            # Update order status
            if order.filled_quantity >= order.quantity:
                order.status = OrderStatus.FILLED
                order.calculate_latency("fill")
                self.filled_orders += 1
                del self.active_orders[order_id]
                self.order_history.append(order)
            else:
                order.status = OrderStatus.PARTIAL_FILL
            
            # Update latency stats
            if order.latency_fill_us:
                self.avg_latency_us = (
                    (self.avg_latency_us * (self.filled_orders - 1) + order.latency_fill_us) / 
                    self.filled_orders
                )
            
            return True
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel active order."""
        with self.lock:
            if order_id not in self.active_orders:
                return False
            
            order = self.active_orders[order_id]
            order.status = OrderStatus.CANCELLED
            
            del self.active_orders[order_id]
            self.order_history.append(order)
            return True
    
    def cancel_all_orders(self) -> int:
        """Cancel all active orders. Returns count of cancelled orders."""
        with self.lock:
            cancelled = 0
            for order_id in list(self.active_orders.keys()):
                order = self.active_orders[order_id]
                order.status = OrderStatus.CANCELLED
                self.order_history.append(order)
                cancelled += 1
            self.active_orders.clear()
            return cancelled
    
    def acknowledge_order(self, order_id: str) -> bool:
        """Phase 11: Acknowledge order receipt by exchange."""
        with self.lock:
            if order_id not in self.active_orders:
                return False
            order = self.active_orders[order_id]
            if order.status == OrderStatus.SUBMITTED:
                order.status = OrderStatus.ACKNOWLEDGED
                order.timestamp_acked = time.perf_counter()
                order.calculate_latency("ack")
                return True
            return False

    def get_position(self, symbol: str) -> int:
        """Get current position for symbol."""
        return self.positions.get(symbol, 0)
    
    def get_pnl(self, current_prices: Dict[str, float]) -> float:
        """Calculate current P&L."""
        pnl = 0.0
        for symbol, quantity in self.positions.items():
            if symbol in current_prices:
                avg_price = self.avg_prices.get(symbol, current_prices[symbol])
                pnl += quantity * (current_prices[symbol] - avg_price)
        
        pnl += (self.capital - self.initial_capital)
        return pnl
    
    def reset_daily_pnl(self):
        """Reset daily P&L counter. Call at start of each trading day."""
        with self.lock:
            logger.info(f"Daily P&L reset. Previous: ${self.daily_pnl:,.2f}")
            self.daily_pnl = 0.0
    
    def get_position_summary(self) -> Dict:
        """Get summary of all positions and order state."""
        return {
            'positions': dict(self.positions),
            'avg_prices': dict(self.avg_prices),
            'active_orders': len(self.active_orders),
            'daily_pnl': self.daily_pnl,
            'capital': self.capital,
            'total_orders': self.total_orders,
            'filled_orders': self.filled_orders,
            'rejected_orders': self.rejected_orders,
        }
    
    def get_metrics(self) -> Dict:
        """Get execution metrics."""
        fill_rate = self.filled_orders / self.total_orders if self.total_orders > 0 else 0
        
        return {
            'total_orders': self.total_orders,
            'filled_orders': self.filled_orders,
            'rejected_orders': self.rejected_orders,
            'active_orders': len(self.active_orders),
            'fill_rate': fill_rate,
            'avg_latency_us': self.avg_latency_us,
            'capital': self.capital,
            'daily_pnl': self.daily_pnl,
            'positions': len(self.positions)
        }


class ExposureGovernor:
    """
    Phase 10: Logic for Safety Checks.
    """
    def __init__(self, max_exposure_per_symbol: float = 0.30):
        self.MAX_SYMBOL_EXPOSURE = max_exposure_per_symbol

    def check_exposure(self, symbol: str, proposed_value: float, current_equity: float, current_positions: dict) -> bool:
        """
        Returns TRUE if exposure is SAFE.
        """
        # Calculate current exposure
        current_symbol_value = current_positions.get(symbol, 0)
        total_market_value = sum(current_positions.values()) # Simplified
        
        projected_exposure = (current_symbol_value + proposed_value) / current_equity
        
        if projected_exposure > self.MAX_SYMBOL_EXPOSURE:
            logger.warning(f"[GOVERNOR] REJECT: Exposure {projected_exposure:.2%} > Limit {self.MAX_SYMBOL_EXPOSURE:.0%}")
            return False
            
        return True
    

# Demo
if __name__ == "__main__":
    print("=" * 60)
    print("HFT ORDER MANAGER DEMO")
    print("=" * 60)
    
    # Initialize manager
    manager = OrderManager(initial_capital=1_000_000)
    manager.max_position_size["SPY"] = 10000
    
    # Create orders
    orders = [
        Order("SPY", OrderSide.BUY, OrderType.LIMIT, 100, price=450.50),
        Order("SPY", OrderSide.BUY, OrderType.MARKET, 50),
        Order("QQQ", OrderSide.SELL, OrderType.LIMIT, 200, price=380.25),
    ]
    
    # Submit orders
    current_prices = {"SPY": 450.75, "QQQ": 380.50}
    
    for order in orders:
        price = current_prices.get(order.symbol, 100.0)
        success, msg = manager.submit_order(order, price)
        print(f"\n{order.symbol} {order.side.value} {order.quantity}")
        print(f"  {msg}")
        
        if success:
            # Simulate fill
            fill_price = price * (1 + np.random.uniform(-0.0001, 0.0001))
            manager.fill_order(order.order_id, fill_price, order.quantity)
            print(f"  Filled @ ${fill_price:.2f}")
    
    # Display metrics
    print("\n" + "=" * 60)
    print("EXECUTION METRICS")
    print("=" * 60)
    
    metrics = manager.get_metrics()
    for key, value in metrics.items():
        print(f"{key:20s}: {value}")
    
    print("\nPositions:")
    for symbol, qty in manager.positions.items():
        print(f"  {symbol}: {qty:+d} shares")
    
    pnl = manager.get_pnl(current_prices)
    print(f"\nP&L: ${pnl:,.2f}")
