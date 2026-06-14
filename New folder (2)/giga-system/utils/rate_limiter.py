"""
GIGA SYSTEM - Rate Limiter
===========================

Thread-safe rate limiters for API calls.

- TokenBucketLimiter : smooth, allows bursts up to bucket capacity
- SlidingWindowLimiter : strict per-window cap (e.g. 1200 req / 60s)

Usage
-----
    from utils.rate_limiter import TokenBucketLimiter

    limiter = TokenBucketLimiter(rate=5.0, burst=10)
    limiter.acquire()          # blocks until a token is available
    limiter.try_acquire()      # returns True/False immediately
"""

import logging
import threading
import time
from collections import deque

logger = logging.getLogger(__name__)


class TokenBucketLimiter:
    """
    Token-bucket rate limiter (thread-safe).

    Parameters
    ----------
    rate : float
        Sustained token refill rate (tokens per second).
    burst : int
        Maximum number of tokens the bucket can hold (burst capacity).
    """

    def __init__(self, rate: float = 5.0, burst: int = 10):
        if rate <= 0:
            raise ValueError("rate must be > 0")
        self.rate = rate
        self.burst = burst
        self._tokens = float(burst)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
        self._last_refill = now

    def try_acquire(self) -> bool:
        """Return True if a token was consumed, False otherwise (non-blocking)."""
        with self._lock:
            self._refill()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
        return False

    def acquire(self, timeout: float = 30.0) -> bool:
        """
        Block until a token is available or *timeout* seconds elapse.

        Returns True on success, False on timeout.
        """
        deadline = time.monotonic() + timeout
        while True:
            if self.try_acquire():
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(1.0 / self.rate, remaining))


class SlidingWindowLimiter:
    """
    Sliding-window rate limiter (thread-safe).

    Enforces at most *max_requests* within any *window_seconds* window.
    Useful for exchange-imposed hard caps (e.g. 1200 weight / 60s).
    """

    def __init__(self, max_requests: int = 1200, window_seconds: float = 60.0):
        self.max_requests = max_requests
        self.window = window_seconds
        self._timestamps: deque = deque()
        self._lock = threading.Lock()

    def _prune(self) -> None:
        cutoff = time.monotonic() - self.window
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

    def try_acquire(self, weight: int = 1) -> bool:
        """Try to consume *weight* request slots (non-blocking)."""
        with self._lock:
            self._prune()
            if len(self._timestamps) + weight <= self.max_requests:
                now = time.monotonic()
                for _ in range(weight):
                    self._timestamps.append(now)
                return True
        return False

    def acquire(self, weight: int = 1, timeout: float = 30.0) -> bool:
        """Block until *weight* slots are available or timeout."""
        deadline = time.monotonic() + timeout
        while True:
            if self.try_acquire(weight):
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(0.1, remaining))

    @property
    def usage(self) -> int:
        """Current number of requests in the sliding window."""
        with self._lock:
            self._prune()
            return len(self._timestamps)
