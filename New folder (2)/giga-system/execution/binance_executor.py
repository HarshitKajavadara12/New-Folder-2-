"""
GIGA SYSTEM - Real Execution Logic
Implementation: Binance Executor (Paper/Live Switchable)

Features:
  - Paper mode: realistic fill simulation via ExecutionEngine
  - Live mode: real order placement via ccxt (Binance / Binance Testnet)
  - Order cancellation
  - Position reconciliation
  - Secure API key management (env vars)
  - Proper logging (no print statements)
"""

import os
import time
import hmac
import hashlib
import uuid
import logging
from typing import Dict, Any, Optional, List
import random
from .execution_engine import ExecutionEngine, FillModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# API Key Management — load keys from environment variables
# ---------------------------------------------------------------------------

def _load_api_keys(api_key: Optional[str], api_secret: Optional[str]):
    """
    Resolve API credentials.  Priority:
      1. Explicit arguments
      2. BINANCE_API_KEY / BINANCE_API_SECRET env vars
      3. None (paper mode only)
    """
    key = api_key or os.environ.get("BINANCE_API_KEY")
    secret = api_secret or os.environ.get("BINANCE_API_SECRET")
    return key, secret


class BinanceExecutor:
    def __init__(self, api_key: str = None, api_secret: str = None,
                 paper_mode: bool = True, testnet: bool = True):
        """
        Parameters
        ----------
        api_key, api_secret : str, optional
            Credentials.  Falls back to env vars BINANCE_API_KEY / BINANCE_API_SECRET.
        paper_mode : bool
            True → simulated fills.  False → real exchange via ccxt.
        testnet : bool
            If live mode, use Binance testnet (testnet.binance.vision).
            Always start on testnet before going to production.
        """
        self.api_key, self.api_secret = _load_api_keys(api_key, api_secret)
        self.paper_mode = paper_mode
        self.testnet = testnet
        self.base_url = "https://api.binance.com"
        
        # Track open orders and positions for reconciliation
        self._open_orders: Dict[str, Dict] = {}   # orderId → order info
        self._positions: Dict[str, float] = {}     # symbol → net qty

        if self.paper_mode:
            self.sim_engine = ExecutionEngine(fill_model=FillModel.CHAOTIC)
            self._exchange = None
            logger.info("[EXEC] Initialized — Mode: PAPER (Phase 11 Chaos Active)")
        else:
            self._exchange = self._init_ccxt_exchange()
            self.sim_engine = None
            mode_label = "LIVE-TESTNET" if self.testnet else "LIVE-PRODUCTION"
            logger.info(f"[EXEC] Initialized — Mode: {mode_label}")

    # ------------------------------------------------------------------
    # ccxt exchange initialisation
    # ------------------------------------------------------------------

    def _init_ccxt_exchange(self):
        """Create and configure a ccxt Binance exchange instance."""
        try:
            import ccxt
        except ImportError:
            raise ImportError(
                "ccxt is required for live trading.  Install via: pip install ccxt"
            )

        if not self.api_key or not self.api_secret:
            raise ValueError(
                "Live mode requires API credentials.  Set BINANCE_API_KEY and "
                "BINANCE_API_SECRET environment variables, or pass them explicitly."
            )

        exchange = ccxt.binance({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'},
        })

        if self.testnet:
            exchange.set_sandbox_mode(True)
            logger.info("[EXEC] Using Binance TESTNET (testnet.binance.vision)")

        # Warmup: load markets so symbol validation works
        exchange.load_markets()
        return exchange

    # ------------------------------------------------------------------
    # Signature (kept for any direct REST usage)
    # ------------------------------------------------------------------

    def sign_request(self, query_string: str) -> str:
        """HMAC SHA256 signature for signed endpoints."""
        if not self.api_secret:
            return "MOCK_SIG"
        return hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    # ------------------------------------------------------------------
    # Place order
    # ------------------------------------------------------------------

    def post_order(self, symbol: str, side: str, quantity: float, price: float = 0.0,
                   last_known_price: float = 0.0) -> Dict:
        """
        Place a new order (paper or live).

        Parameters
        ----------
        symbol : str   Trading pair, e.g. 'BTCUSDT' or 'BTC/USDT'
        side : str     'BUY' or 'SELL'
        quantity : float   Order size
        price : float  Limit price (0 = market order)
        last_known_price : float  Reference price for market orders
        """
        if quantity <= 0:
            return {"status": "REJECTED", "reason": "MinQty"}

        if self.paper_mode:
            return self._simulate_network_call({
                "symbol": symbol,
                "side": side,
                "type": "MARKET" if price == 0 else "LIMIT",
                "quantity": quantity,
                "price": price,
                "last_known_price": last_known_price
            })

        # ---- LIVE MODE via ccxt ----
        return self._live_post_order(symbol, side, quantity, price)

    def _live_post_order(self, symbol: str, side: str, quantity: float,
                         price: float) -> Dict:
        """Place a real order on Binance via ccxt."""
        try:
            # Normalise symbol for ccxt (BTCUSDT → BTC/USDT)
            ccxt_symbol = self._normalise_symbol(symbol)
            order_type = 'limit' if price > 0 else 'market'
            ccxt_side = side.lower()

            if order_type == 'limit':
                order = self._exchange.create_order(
                    symbol=ccxt_symbol,
                    type='limit',
                    side=ccxt_side,
                    amount=quantity,
                    price=price,
                )
            else:
                order = self._exchange.create_order(
                    symbol=ccxt_symbol,
                    type='market',
                    side=ccxt_side,
                    amount=quantity,
                )

            order_id = order.get('id', str(uuid.uuid4()))
            status = order.get('status', 'open').upper()
            if status == 'CLOSED':
                status = 'FILLED'

            result = {
                "status": status,
                "orderId": order_id,
                "transactTime": int(time.time() * 1000),
                "executedQty": order.get('filled', 0),
                "origQty": quantity,
                "avgPrice": order.get('average', order.get('price', 0)),
                "ccxt_raw": order,
            }

            # Track open order
            if status not in ('FILLED', 'CANCELED', 'REJECTED', 'EXPIRED'):
                self._open_orders[order_id] = result

            # Update local position
            filled = order.get('filled', 0)
            if filled > 0:
                sign = 1 if ccxt_side == 'buy' else -1
                self._positions[ccxt_symbol] = self._positions.get(ccxt_symbol, 0) + sign * filled

            logger.info(f"[EXEC] {status} {ccxt_side} {filled}/{quantity} {ccxt_symbol} @ {result['avgPrice']}")
            return result

        except Exception as e:
            logger.error(f"[EXEC] Live order failed: {e}")
            return {"status": "ERROR", "reason": str(e)}

    # ------------------------------------------------------------------
    # Cancel order
    # ------------------------------------------------------------------

    def cancel_order(self, order_id: str, symbol: str) -> Dict:
        """
        Cancel an open order.

        Parameters
        ----------
        order_id : str   Exchange order ID
        symbol : str     Trading pair
        """
        if self.paper_mode:
            removed = self._open_orders.pop(order_id, None)
            return {"status": "CANCELED", "orderId": order_id,
                    "was_tracked": removed is not None}

        try:
            ccxt_symbol = self._normalise_symbol(symbol)
            result = self._exchange.cancel_order(order_id, ccxt_symbol)
            self._open_orders.pop(order_id, None)
            logger.info(f"[EXEC] Canceled order {order_id} on {ccxt_symbol}")
            return {"status": "CANCELED", "orderId": order_id, "ccxt_raw": result}
        except Exception as e:
            logger.error(f"[EXEC] Cancel failed: {e}")
            return {"status": "ERROR", "reason": str(e)}

    def cancel_all_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Cancel all open orders, optionally filtered by symbol."""
        results = []
        if self.paper_mode:
            ids = list(self._open_orders.keys())
            for oid in ids:
                results.append(self.cancel_order(oid, symbol or ""))
            return results

        try:
            ccxt_symbol = self._normalise_symbol(symbol) if symbol else None
            open_orders = self._exchange.fetch_open_orders(ccxt_symbol)
            for order in open_orders:
                r = self.cancel_order(order['id'], order.get('symbol', ''))
                results.append(r)
            logger.info(f"[EXEC] Canceled {len(results)} open orders")
        except Exception as e:
            logger.error(f"[EXEC] Cancel-all failed: {e}")
            results.append({"status": "ERROR", "reason": str(e)})
        return results

    # ------------------------------------------------------------------
    # Position reconciliation
    # ------------------------------------------------------------------

    def reconcile_positions(self) -> Dict[str, Any]:
        """
        Sync local position tracking with exchange reality.

        Returns a dict of mismatches found and whether corrections were applied.
        """
        if self.paper_mode:
            return {"mode": "paper", "positions": dict(self._positions)}

        mismatches = []
        try:
            balance = self._exchange.fetch_balance()
            exchange_positions: Dict[str, float] = {}

            # Spot balances: sum free + used for non-zero assets
            for asset, details in balance.get('total', {}).items():
                if details and float(details) > 0:
                    exchange_positions[asset] = float(details)

            all_symbols = set(list(self._positions.keys()) + list(exchange_positions.keys()))

            for sym in all_symbols:
                local_qty = self._positions.get(sym, 0)
                exchange_qty = exchange_positions.get(sym, 0)
                if abs(exchange_qty - local_qty) > 1e-8:
                    mismatches.append({
                        "symbol": sym,
                        "local": local_qty,
                        "exchange": exchange_qty,
                        "diff": exchange_qty - local_qty,
                    })
                    # Force local to match exchange
                    self._positions[sym] = exchange_qty
                    logger.warning(
                        f"[RECON] Position mismatch {sym}: "
                        f"local={local_qty} exchange={exchange_qty} → corrected"
                    )

            logger.info(f"[RECON] Reconciliation complete — {len(mismatches)} mismatch(es)")
            return {"mismatches": mismatches, "positions": dict(self._positions)}

        except Exception as e:
            logger.error(f"[RECON] Reconciliation failed: {e}")
            return {"status": "ERROR", "reason": str(e)}

    def fetch_balance(self) -> Dict:
        """Fetch account balance from exchange."""
        if self.paper_mode:
            return {"mode": "paper", "positions": dict(self._positions)}

        try:
            balance = self._exchange.fetch_balance()
            return {
                "free": balance.get("free", {}),
                "used": balance.get("used", {}),
                "total": balance.get("total", {}),
            }
        except Exception as e:
            logger.error(f"[EXEC] Fetch balance failed: {e}")
            return {"status": "ERROR", "reason": str(e)}

    # ------------------------------------------------------------------
    # Paper-mode simulation (unchanged logic)
    # ------------------------------------------------------------------

    def _simulate_network_call(self, params: Dict) -> Dict:
        """Phase 11: Velocity & Chaos Simulation via ExecutionEngine."""
        target_price = params.get('price', 0)
        if target_price == 0:
            target_price = params.get('last_known_price', 0)
            if target_price == 0:
                return {
                    "status": "REJECTED",
                    "reason": "NO_PRICE: Market order requires a reference price. "
                              "Pass 'last_known_price' or use LIMIT order."
                }

        res = self.sim_engine.execute_order(
            quantity=params['quantity'],
            target_price=target_price,
            urgency=0.9 if params['type'] == 'MARKET' else 0.5
        )

        time.sleep(res.latency_us / 1_000_000)

        if hasattr(res, 'status') and res.status == "REJECTED":
            return {
                "status": "REJECTED",
                "reason": res.error_message or "Exchange System Error"
            }

        status = "FILLED"
        if res.partial_fill:
            status = "PARTIAL_FILL"

        order_id = str(uuid.uuid4())

        result = {
            "status": status,
            "orderId": order_id,
            "transactTime": int(time.time() * 1000),
            "executedQty": res.filled_quantity,
            "origQty": params['quantity'],
            "avgPrice": res.avg_fill_price,
            "latency_ms": res.latency_us / 1000
        }

        # Track in local positions
        sign = 1 if params['side'] == 'BUY' else -1
        sym = params['symbol']
        self._positions[sym] = self._positions.get(sym, 0) + sign * res.filled_quantity

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_symbol(symbol: str) -> str:
        """Convert BTCUSDT → BTC/USDT if needed."""
        if '/' in symbol:
            return symbol
        # Common quote currencies
        for quote in ('USDT', 'BUSD', 'USDC', 'BTC', 'ETH', 'BNB', 'USD'):
            if symbol.endswith(quote) and len(symbol) > len(quote):
                base = symbol[:-len(quote)]
                return f"{base}/{quote}"
        return symbol
