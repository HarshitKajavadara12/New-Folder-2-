
import time
import logging
from typing import Dict, List, Optional, Callable, Tuple

logger = logging.getLogger(__name__)

class SessionGuard:
    """
    PHASE 6: Survival Mechanism
    Responsibility: Kill switch for the live session.
    
    Features:
    - Drawdown-based kill switch
    - Order rate limiting
    - Emergency shutdown: cancel all orders + flatten positions
    - Hooks into OrderManager and Exchange for real action
    """
    def __init__(self, max_loss: float = 500.0, max_orders_per_min: int = 5,
                 max_drawdown_pct: float = 0.05, max_session_hours: float = 24.0):
        self.max_loss = max_loss
        self.max_orders_per_min = max_orders_per_min
        self.max_drawdown_pct = max_drawdown_pct
        self.max_session_hours = max_session_hours
        self.start_equity = 0.0
        self.peak_equity = 0.0
        self.order_timestamps: List[float] = []
        self.triggered = False
        self.trigger_reason = ""
        self.shutdown_callbacks: List[Callable] = []
        self.equity_history: List[Tuple[float, float]] = []
        self.max_equity_history: int = 10000
        self.start_time: Optional[float] = None
        
        # References to live components (set via wire())
        self._order_manager = None
        self._exchange = None
    
    def wire(self, order_manager=None, exchange=None):
        """Connect to live order manager and exchange for real kill switch."""
        self._order_manager = order_manager
        self._exchange = exchange
    
    def register_shutdown_callback(self, callback: Callable):
        """Register a function to call on emergency shutdown."""
        self.shutdown_callbacks.append(callback)

    def initialize_account(self, initial_equity: float):
        self.start_equity = initial_equity
        self.peak_equity = initial_equity
        self.start_time = time.time()

    def check_health(self, current_equity: float) -> bool:
        """
        Returns False if system must DIE.
        Checks absolute loss AND percentage drawdown from peak.
        """
        # Track equity history
        self.equity_history.append((time.time(), current_equity))
        if len(self.equity_history) > self.max_equity_history:
            self.equity_history = self.equity_history[-self.max_equity_history:]
        
        # Track peak equity
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        absolute_dd = self.start_equity - current_equity
        pct_dd = (self.peak_equity - current_equity) / self.peak_equity if self.peak_equity > 0 else 0
        
        # 1. Absolute Drawdown Check
        if absolute_dd > self.max_loss:
            self.triggered = True
            self.trigger_reason = f"MAX LOSS EXCEEDED: -${absolute_dd:.2f} (limit: ${self.max_loss:.2f})"
            logger.critical(self.trigger_reason)
            self.emergency_shutdown()
            return False
        
        # 2. Percentage Drawdown Check
        if pct_dd > self.max_drawdown_pct:
            self.triggered = True
            self.trigger_reason = f"DRAWDOWN LIMIT: -{pct_dd:.2%} (limit: {self.max_drawdown_pct:.2%})"
            logger.critical(self.trigger_reason)
            self.emergency_shutdown()
            return False
        
        # 3. Session time limit
        if self.start_time is not None:
            elapsed_hours = (time.time() - self.start_time) / 3600
            if elapsed_hours > self.max_session_hours:
                self.triggered = True
                self.trigger_reason = f"SESSION TIME LIMIT: {elapsed_hours:.1f}h > {self.max_session_hours:.0f}h"
                logger.critical(self.trigger_reason)
                self.emergency_shutdown()
                return False
            
        return True

    def validate_order_rate(self) -> bool:
        """
        Returns False if trading too fast.
        """
        now = time.time()
        # Clean old timestamps (older than 60s)
        self.order_timestamps = [t for t in self.order_timestamps if now - t < 60]
        
        if len(self.order_timestamps) >= self.max_orders_per_min:
            self.triggered = True
            self.trigger_reason = f"RATE LIMIT EXCEEDED: {len(self.order_timestamps)}/{self.max_orders_per_min} orders/min"
            logger.critical(self.trigger_reason)
            self.emergency_shutdown()
            return False
            
        self.order_timestamps.append(now)
        return True
    
    def emergency_shutdown(self):
        """
        REAL KILL SWITCH: Cancel all orders and flatten all positions.
        This is not a flag-setter — it takes ACTION.
        """
        logger.critical(f"EMERGENCY SHUTDOWN ENGAGED: {self.trigger_reason}")
        
        cancelled_orders = 0
        closed_positions = 0
        
        # Step 1: Cancel all active orders
        if self._order_manager is not None:
            active_ids = list(self._order_manager.active_orders.keys())
            for order_id in active_ids:
                try:
                    self._order_manager.cancel_order(order_id)
                    cancelled_orders += 1
                except Exception as e:
                    logger.error(f"Failed to cancel order {order_id}: {e}")
            logger.info(f"Cancelled {cancelled_orders} active orders")
        
        # Step 2: Flatten all positions
        if self._order_manager is not None and self._exchange is not None:
            for symbol, qty in list(self._order_manager.positions.items()):
                if qty == 0:
                    continue
                try:
                    side = "SELL" if qty > 0 else "BUY"
                    abs_qty = abs(qty)
                    # Use exchange to close position at market
                    result = self._exchange.post_order(
                        symbol=symbol,
                        side=side,
                        quantity=abs_qty,
                        price=0  # Market order
                    )
                    closed_positions += 1
                    logger.info(f"Flattened {symbol}: {side} {abs_qty} -> {result}")
                except Exception as e:
                    logger.error(f"CRITICAL: Failed to flatten {symbol} ({qty}): {e}")
        
        # Step 3: Fire registered callbacks (e.g., alert system, logging)
        for callback in self.shutdown_callbacks:
            try:
                callback(self.trigger_reason)
            except Exception as e:
                logger.error(f"Shutdown callback error: {e}")
        
        logger.critical(
            f"SHUTDOWN COMPLETE: {cancelled_orders} orders cancelled, "
            f"{closed_positions} positions flattened"
        )

    def get_drawdown_details(self) -> Dict:
        """Get current drawdown details."""
        current_equity = self.equity_history[-1][1] if self.equity_history else self.start_equity
        absolute_dd = self.start_equity - current_equity
        pct_dd = (self.peak_equity - current_equity) / self.peak_equity if self.peak_equity > 0 else 0
        return {
            'current_equity': current_equity,
            'start_equity': self.start_equity,
            'peak_equity': self.peak_equity,
            'absolute_drawdown': absolute_dd,
            'pct_drawdown_from_peak': pct_dd,
            'pct_drawdown_from_start': absolute_dd / self.start_equity if self.start_equity > 0 else 0,
            'equity_history_length': len(self.equity_history),
            'triggered': self.triggered,
        }

    def get_status(self) -> Dict:
        if self.triggered:
            return {
                "status": "KILLED",
                "reason": self.trigger_reason,
                "message": f"  KILL SWITCH ENGAGED: {self.trigger_reason}"
            }
        return {
            "status": "ACTIVE",
            "reason": "",
            "message": "  GUARD ACTIVE"
        }
    
    def reset(self, new_equity: float = None):
        """Reset the guard after manual review. Requires explicit call."""
        self.triggered = False
        self.trigger_reason = ""
        self.order_timestamps.clear()
        if new_equity is not None:
            self.start_equity = new_equity
            self.peak_equity = new_equity
        logger.info("SessionGuard RESET — manual review acknowledged")
