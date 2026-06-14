"""
GIGA SYSTEM - Real Market Data Feed
Implementation: Binance WebSocket (Aggregated Trade Stream)

REALITY CHECK:
- Connects to wss://stream.binance.com:9443
- Buffers real-time trades
- Tracks latency (Exchange Time vs Local Time)
- Handles disconnected states
"""

import asyncio
import json
import time
import ssl
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)

# Try importing websockets, else warn (but code is real)
try:
    import websockets
except ImportError:
    logger.critical("'websockets' library missing. Install via pip.")
    websockets = None

class BinanceWebSocketFeed:
    """
    Real-time Binance WebSocket Client.
    Stream: <symbol>@aggTrade
    Features: SSL verified, auto-reconnect with exponential backoff.
    """
    
    MAX_RECONNECT_ATTEMPTS = 20
    INITIAL_BACKOFF_SEC = 1.0
    MAX_BACKOFF_SEC = 60.0
    
    def __init__(self, symbol: str = "btcusdt"):
        self.symbol = symbol.lower()
        self.url = f"wss://stream.binance.com:9443/ws/{self.symbol}@aggTrade"
        self.last_price = 0.0
        self.last_update = 0
        self.latency_ms = 0.0
        self.connected = False
        self.queue = asyncio.Queue()
        self.running = False
        self._reconnect_count = 0
        
    async def connect(self):
        if not websockets:
            raise ImportError("Websockets library required for Real Feed")
        
        # Proper SSL: verify certificates (safe for Binance public API)
        ssl_context = ssl.create_default_context()
        # check_hostname and verify_mode are True by default — no need to disable
        
        backoff = self.INITIAL_BACKOFF_SEC
        
        while self._reconnect_count < self.MAX_RECONNECT_ATTEMPTS:
            logger.info(f"[FEED] Connecting to {self.url} (attempt {self._reconnect_count + 1})...")
            
            try:
                async with websockets.connect(
                    self.url, 
                    ssl=ssl_context,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5
                ) as ws:
                    self.connected = True
                    self.running = True
                    self._reconnect_count = 0  # Reset on successful connect
                    backoff = self.INITIAL_BACKOFF_SEC
                    logger.info("[FEED] CONNECTED to Binance Public Stream (SSL verified)")
                    
                    while self.running:
                        try:
                            message = await asyncio.wait_for(ws.recv(), timeout=30.0)
                            data = json.loads(message)
                            self._process_msg(data)
                        except asyncio.TimeoutError:
                            logger.warning("[FEED] No data for 30s — sending ping...")
                            try:
                                pong = await ws.ping()
                                await asyncio.wait_for(pong, timeout=10)
                            except Exception:
                                logger.warning("[FEED] Ping failed — reconnecting...")
                                break
                        except websockets.exceptions.ConnectionClosed as e:
                            logger.warning(f"[FEED] Connection closed: {e} — reconnecting...")
                            break
                        except Exception as e:
                            logger.error(f"[FEED] Stream error: {e}")
                            break
                            
            except Exception as e:
                logger.error(f"[FEED] Connection failed: {e}")
            
            # Reconnection with exponential backoff
            self.connected = False
            self._reconnect_count += 1
            
            if self._reconnect_count >= self.MAX_RECONNECT_ATTEMPTS:
                logger.error(f"[FEED] Max reconnect attempts ({self.MAX_RECONNECT_ATTEMPTS}) reached. Giving up.")
                break
            
            logger.info(f"[FEED] Reconnecting in {backoff:.1f}s (attempt {self._reconnect_count}/{self.MAX_RECONNECT_ATTEMPTS})...")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, self.MAX_BACKOFF_SEC)  # Exponential backoff with cap
        
        self.connected = False
        self.running = False

    def _process_msg(self, data: Dict[str, Any]):
        """
        Process aggregate trade message.
        {
          "e": "aggTrade",  // Event type
          "E": 123456789,   // Event time
          "s": "BNBBTC",    // Symbol
          "p": "0.001",     // Price
          "q": "100",       // Quantity
          ...
        }
        """
        try:
            self.last_price = float(data['p'])
            event_time = int(data['E']) # ms
            local_time = int(time.time() * 1000)
            self.latency_ms = local_time - event_time
            self.last_update = time.time()
            
            # Put in queue for consumer
            if not self.queue.full():
                self.queue.put_nowait({
                    "price": self.last_price,
                    "qty": float(data['q']),
                    "timestamp": event_time,
                    "latency": self.latency_ms,
                    "is_buyer_maker": data['m']
                })
        except Exception as e:
            logger.debug(f"[FEED] Message parse error: {e}")  # Log instead of silent drop

    async def get_latest(self) -> Dict[str, Any]:
        """Fetch latest packet from queue"""
        return await self.queue.get()

    def get_snapshot(self) -> Dict[str, Any]:
        """Synchronous snapshot of state"""
        return {
            "price": self.last_price,
            "latency": self.latency_ms,
            "connected": self.connected,
            "ts": self.last_update
        }
