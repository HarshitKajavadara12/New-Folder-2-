"""
GIGA SYSTEM - Live Account Manager
Implementation: Real PnL, Margin, Leverage, and Fees

Features:
- Tracks Cash Balance + Unrealized/Realized P&L
- Proper Weighted Average Price on position increase
- Correct P&L for both long and short closes
- Fee deduction on every trade
- Equity = Cash + Unrealized P&L
- Margin tracking with leverage awareness
- Liquidation price estimation
"""

from typing import Dict, List, Optional, Tuple
import time
import logging

logger = logging.getLogger(__name__)


class LiveAccount:
    def __init__(self, start_balance: float = 10000.0, fee_rate: float = 0.0004,
                 max_leverage: float = 1.0, maintenance_margin_rate: float = 0.005):
        """
        Args:
            start_balance: Starting cash balance.
            fee_rate: Taker fee rate (e.g. 0.0004 = 0.04%).
            max_leverage: Maximum allowed leverage (1.0 = spot / no leverage).
            maintenance_margin_rate: Exchange maintenance margin rate for liquidation calc.
        """
        self.cash = start_balance
        self.initial_balance = start_balance
        self.equity = start_balance
        self.fee_rate = fee_rate
        self.max_leverage = max_leverage
        self.maintenance_margin_rate = maintenance_margin_rate
        self.positions: Dict[str, Dict] = {}
        self.history: List[Dict] = []
        self.total_fees = 0.0
        self.total_realized_pnl = 0.0
        
    def update_mark_price(self, symbol: str, price: float) -> float:
        """Update Unrealized PnL based on Mark Price."""
        unrealized_pnl = 0.0
        
        for sym, pos in self.positions.items():
            size = pos['size']
            entry = pos['entry_price']
            mark = price if sym == symbol else pos.get('mark_price', entry)
            
            if sym == symbol:
                pos['mark_price'] = price
            
            # Linear PnL: Long = (Mark - Entry) * Size, Short (size<0) works automatically
            pnl = (mark - entry) * size
            unrealized_pnl += pnl
            
        self.equity = self.cash + unrealized_pnl
        return self.equity

    def execute_trade(self, symbol: str, side: str, price: float, size: float) -> Tuple[float, float]:
        """
        Process a fill with proper weighted average price and P&L.
        
        Args:
            symbol: Trading pair
            side: "BUY" or "SELL"
            price: Fill price
            size: Fill size (always positive)
            
        Returns:
            (realized_pnl, fee)
        """
        # Fee deduction
        notional = price * size
        fee = notional * self.fee_rate
        self.cash -= fee
        self.total_fees += fee
        
        # Current position state
        current_pos = self.positions.get(symbol, {'size': 0.0, 'entry_price': 0.0, 'mark_price': price})
        curr_size = current_pos['size']
        curr_entry = current_pos['entry_price']
        
        # Determine new size
        if side == "BUY":
            new_size = curr_size + size
        elif side == "SELL":
            new_size = curr_size - size
        else:
            logger.error(f"Invalid side: {side}")
            return 0.0, fee
            
        # P&L Realization
        realized_pnl = 0.0
        new_entry = curr_entry
        
        if curr_size > 0 and side == "SELL":
            # Closing long (partially or fully)
            closed_qty = min(size, curr_size)
            realized_pnl = (price - curr_entry) * closed_qty
            self.cash += realized_pnl
            
            if size > curr_size:
                # Flipped to short
                new_entry = price
            # If partially closed, entry stays the same
            
        elif curr_size < 0 and side == "BUY":
            # Closing short (partially or fully)
            closed_qty = min(size, abs(curr_size))
            realized_pnl = (curr_entry - price) * closed_qty  # Short: profit when price drops
            self.cash += realized_pnl
            
            if size > abs(curr_size):
                # Flipped to long
                new_entry = price
            # If partially closed, entry stays the same
            
        elif (curr_size >= 0 and side == "BUY") or (curr_size <= 0 and side == "SELL"):
            # Adding to position — weighted average price
            if curr_size == 0:
                new_entry = price
            else:
                total_cost = abs(curr_size) * curr_entry + size * price
                new_entry = total_cost / (abs(curr_size) + size)
        
        self.total_realized_pnl += realized_pnl
        
        # Update position state
        if abs(new_size) < 1e-10:
            # Position closed
            if symbol in self.positions:
                del self.positions[symbol]
        else:
            self.positions[symbol] = {
                'size': new_size,
                'entry_price': new_entry,
                'mark_price': price
            }
            
        # Record trade
        self.history.append({
            "ts": time.time(),
            "symbol": symbol,
            "side": side,
            "price": price,
            "size": size,
            "fee": fee,
            "realized_pnl": realized_pnl,
            "cash_after": self.cash,
            "position_after": new_size
        })
        
        logger.info(
            f"[ACCOUNT] {side} {size} {symbol} @ {price:.2f} | "
            f"PnL: {realized_pnl:+.2f} | Fee: {fee:.2f} | Cash: {self.cash:.2f}"
        )
        
        return realized_pnl, fee
    
    def get_equity(self) -> float:
        """Get current equity (cash + unrealized P&L)."""
        return self.equity
    
    def get_total_pnl(self) -> float:
        """Total realized P&L minus fees."""
        return self.total_realized_pnl - self.total_fees
    
    def get_position_summary(self) -> Dict:
        """Summary of all open positions."""
        return {
            sym: {
                'size': pos['size'],
                'entry': pos['entry_price'],
                'side': 'LONG' if pos['size'] > 0 else 'SHORT',
                'notional': abs(pos['size'] * pos['entry_price'])
            }
            for sym, pos in self.positions.items()
        }
    
    def get_account_summary(self) -> Dict:
        """Full account summary including leverage metrics."""
        margin_info = self.get_margin_info()
        return {
            'cash': self.cash,
            'equity': self.equity,
            'initial_balance': self.initial_balance,
            'total_realized_pnl': self.total_realized_pnl,
            'total_fees': self.total_fees,
            'net_pnl': self.total_realized_pnl - self.total_fees,
            'return_pct': (self.equity - self.initial_balance) / self.initial_balance * 100,
            'open_positions': len(self.positions),
            'total_trades': len(self.history),
            'leverage': margin_info['current_leverage'],
            'max_leverage': self.max_leverage,
            'margin_used': margin_info['margin_used'],
            'margin_available': margin_info['margin_available'],
            'margin_ratio': margin_info['margin_ratio'],
        }

    # ------------------------------------------------------------------
    # Margin & Leverage
    # ------------------------------------------------------------------

    def get_gross_exposure(self) -> float:
        """Total notional exposure across all positions."""
        exposure = 0.0
        for pos in self.positions.values():
            mark = pos.get('mark_price', pos['entry_price'])
            exposure += abs(pos['size'] * mark)
        return exposure

    def get_margin_info(self) -> Dict:
        """
        Return margin & leverage metrics.

        Returns dict with:
            current_leverage  – gross exposure / equity
            margin_used       – notional / max_leverage (how much equity is "locked")
            margin_available  – equity - margin_used
            margin_ratio      – margin_used / equity (0-1, >1 means overleveraged)
            can_open_more     – whether margin allows new positions
        """
        exposure = self.get_gross_exposure()
        equity = max(self.equity, 1e-9)  # avoid div-by-zero
        current_leverage = exposure / equity

        if self.max_leverage > 0:
            margin_used = exposure / self.max_leverage
        else:
            margin_used = exposure

        margin_available = max(0.0, equity - margin_used)
        margin_ratio = margin_used / equity if equity > 0 else 0.0

        return {
            'current_leverage': round(current_leverage, 4),
            'margin_used': round(margin_used, 2),
            'margin_available': round(margin_available, 2),
            'margin_ratio': round(margin_ratio, 4),
            'can_open_more': margin_ratio < 0.95,
        }

    def check_leverage_limit(self, symbol: str, additional_notional: float) -> bool:
        """
        Check whether adding *additional_notional* would breach max leverage.

        Returns True if the order is safe, False if it would exceed limits.
        """
        new_exposure = self.get_gross_exposure() + additional_notional
        equity = max(self.equity, 1e-9)
        projected_leverage = new_exposure / equity
        if projected_leverage > self.max_leverage:
            logger.warning(
                f"[MARGIN] Leverage check FAILED for {symbol}: "
                f"projected {projected_leverage:.2f}x > max {self.max_leverage:.1f}x"
            )
            return False
        return True

    def estimate_liquidation_price(self, symbol: str) -> Optional[float]:
        """
        Estimate liquidation price for a position based on maintenance margin.

        Uses simplified formula:
            Long : liq_price = entry * (1 - 1/leverage + mmr)
            Short: liq_price = entry * (1 + 1/leverage - mmr)
        """
        pos = self.positions.get(symbol)
        if not pos:
            return None

        entry = pos['entry_price']
        size = pos['size']
        # Position-level leverage (notional / margin allocated)
        notional = abs(size * entry)
        margin_allocated = notional / self.max_leverage if self.max_leverage > 0 else notional
        pos_leverage = notional / margin_allocated if margin_allocated > 0 else 1.0
        mmr = self.maintenance_margin_rate

        if size > 0:
            liq = entry * (1 - 1.0 / pos_leverage + mmr)
        else:
            liq = entry * (1 + 1.0 / pos_leverage - mmr)

        return max(0.0, liq)

