
import time
import requests
import threading
from typing import Dict, Any, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class TokenBucketRateLimiter:
    """Thread-safe token bucket rate limiter for API calls."""
    
    def __init__(self, rate: float = 5.0, burst: int = 10):
        """
        Args:
            rate: Tokens per second (sustained request rate).
            burst: Maximum burst size (bucket capacity).
        """
        self.rate = rate
        self.burst = burst
        self._tokens = float(burst)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()
    
    def acquire(self, timeout: float = 10.0) -> bool:
        """Block until a token is available or timeout expires."""
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_refill
                self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
                self._last_refill = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True
            # Wait a fraction of the refill interval
            if time.monotonic() >= deadline:
                return False
            time.sleep(1.0 / self.rate)


class MarketStream:
    """
    PHASE 9: High-Speed Event Stream
    Responsibility: Async Tick Generation with proper rate limiting.
    """
    def __init__(self, symbol: str = "BTCUSDT", callback: Callable[[Dict], None] = None,
                 rate_limit: float = 5.0):
        """
        Args:
            symbol: Trading pair symbol.
            callback: Function called with each tick dict.
            rate_limit: Max requests per second (default 5, well under Binance's 10/s cap).
        """
        self.symbol = symbol.upper()
        self.running = False
        self.last_tick: Dict[str, Any] = {}
        self.session = requests.Session()
        self.callback = callback
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self._rate_limiter = TokenBucketRateLimiter(rate=rate_limit, burst=int(rate_limit * 2))
        self._consecutive_errors = 0
        self._MAX_BACKOFF = 30.0  # seconds

    def start(self):
        self.running = True
        self.thread.start()
        logger.info(f"[STREAM] Connected to {self.symbol} @ Binance (Async Mode)")

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=5.0)
        logger.info(f"[STREAM] Stopped {self.symbol} stream")

    def _loop(self):
        while self.running:
            try:
                # Respect rate limits before each request
                if not self._rate_limiter.acquire(timeout=5.0):
                    logger.warning("[STREAM] Rate limiter timeout — backing off")
                    time.sleep(1.0)
                    continue
                
                t0 = time.perf_counter()
                
                url = f"https://api.binance.com/api/v3/ticker/price?symbol={self.symbol}"
                resp = self.session.get(url, timeout=5)
                resp.raise_for_status()
                data = resp.json()
                
                tick = {
                    "symbol": self.symbol,
                    "price": float(data['price']),
                    "volume": 0.0,
                    "timestamp": time.time(),
                    "source": "BINANCE_REST",
                    "latency_poll": (time.perf_counter() - t0) * 1000
                }
                
                self.last_tick = tick
                self._consecutive_errors = 0  # reset on success
                
                if self.callback:
                    self.callback(tick)
                
            except requests.exceptions.HTTPError as e:
                self._consecutive_errors += 1
                status = getattr(e.response, 'status_code', None)
                if status == 429:
                    # Rate-limited by Binance — long backoff
                    backoff = min(self._MAX_BACKOFF, 5.0 * self._consecutive_errors)
                    logger.error(f"[STREAM] Rate limited (HTTP 429). Backing off {backoff:.1f}s")
                    time.sleep(backoff)
                elif status and 500 <= status < 600:
                    backoff = min(self._MAX_BACKOFF, 2.0 ** self._consecutive_errors)
                    logger.warning(f"[STREAM] Server error {status}. Retry in {backoff:.1f}s")
                    time.sleep(backoff)
                else:
                    logger.error(f"[STREAM] HTTP error: {e}")
                    time.sleep(1.0)
            except requests.exceptions.ConnectionError as e:
                self._consecutive_errors += 1
                backoff = min(self._MAX_BACKOFF, 2.0 ** self._consecutive_errors)
                logger.error(f"[STREAM] Connection error. Retry in {backoff:.1f}s: {e}")
                time.sleep(backoff)
            except Exception as e:
                self._consecutive_errors += 1
                backoff = min(self._MAX_BACKOFF, 2.0 ** self._consecutive_errors)
                logger.error(f"[STREAM] Unexpected error ({type(e).__name__}): {e}. Retry in {backoff:.1f}s")
                time.sleep(backoff)

    def get_latest_tick(self) -> Dict[str, Any]:
        """Return most recent tick data."""
        return self.last_tick
