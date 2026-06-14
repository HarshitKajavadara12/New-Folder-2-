
import time
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class OrderRouter:
    """
    PHASE 6: Execution Gateway
    Routes signals to the correct executor/exchange.
    Supports multiple symbols and exchanges.
    """
    def __init__(self, executor, default_symbol: str = "BTCUSDT"):
        self.executor = executor
        self.default_symbol = default_symbol
        self.active_routes = ["BINANCE"]
        self._supported_symbols: set = set()  # populated lazily
        self._order_count = 0

    def route_order(self, signal: Dict, current_price: float = 0.0,
                    quantity_override: float = 0.0) -> Dict:
        """
        Convert a BRAIN signal into an EXCHANGE order.
        
        Accepts ``symbol`` from the signal dict so the router is
        not locked to a single pair.

        Args:
            signal: Signal dict from brain/reducer. Expected keys:
                    - action: "EXECUTE_ENTRY" | "EXECUTE_EXIT" | …
                    - symbol (optional): Trading pair, e.g. "ETHUSDT"
                    - side   (optional): Explicit "BUY" / "SELL"
            current_price: Latest market price for the symbol.
            quantity_override: If >0, use this qty instead of default.

        Returns:
            Execution result dict from the underlying executor.
        """
        if "action" not in signal:
            return {"status": "SKIPPED", "reason": "No Action"}

        # --- Determine symbol (multi-symbol support) ---
        symbol = signal.get("symbol", self.default_symbol).upper()

        # --- Determine side ---
        if "side" in signal:
            side = signal["side"].upper()
        else:
            side = "BUY" if "ENTRY" in signal["action"] else "SELL"

        # --- Determine quantity ---
        qty = quantity_override if quantity_override > 0 else signal.get("quantity", 0.001)

        # --- Route to executor ---
        logger.info(f"[ROUTER] Routing {side} {qty} {symbol} to Binance")
        self._order_count += 1

        result = self.executor.post_order(
            symbol=symbol,
            side=side,
            quantity=qty,
            price=current_price,
        )

        return result

    def route_batch(self, signals: List[Dict], current_prices: Optional[Dict[str, float]] = None) -> List[Dict]:
        """
        Route a batch of signals (one per symbol).

        Args:
            signals: List of signal dicts, each with ``symbol``.
            current_prices: Mapping symbol → latest price.

        Returns:
            List of execution result dicts.
        """
        results = []
        current_prices = current_prices or {}
        for sig in signals:
            sym = sig.get("symbol", self.default_symbol).upper()
            price = current_prices.get(sym, 0.0)
            results.append(self.route_order(sig, current_price=price))
        return results

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "orders_routed": self._order_count,
            "active_routes": self.active_routes,
            "default_symbol": self.default_symbol,
        }
