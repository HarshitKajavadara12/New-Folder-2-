"""
GIGA SYSTEM - Retry & Error Recovery Utilities
================================================

Provides decorators and helpers for resilient operation:
- Exponential-backoff retry decorator (sync & async)
- Circuit breaker for repeated failures
- Crash-recovery state persistence

Usage
-----
    from utils.retry import retry, async_retry, CircuitBreaker

    @retry(max_attempts=3, backoff=2.0, exceptions=(ConnectionError,))
    def fetch_data():
        ...

    @async_retry(max_attempts=5)
    async def stream_data():
        ...

    cb = CircuitBreaker(failure_threshold=5, reset_timeout=60)
    with cb:
        do_risky_operation()
"""

import asyncio
import functools
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Type

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Synchronous retry decorator
# ---------------------------------------------------------------------------

def retry(
    max_attempts: int = 3,
    backoff: float = 2.0,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: Sequence[Type[Exception]] = (Exception,),
    on_retry: Optional[Callable] = None,
):
    """
    Retry a function on failure with exponential backoff.

    Parameters
    ----------
    max_attempts : int
        Maximum number of attempts (including the first call).
    backoff : float
        Multiplier for delay after each failure.
    initial_delay : float
        Delay in seconds before the first retry.
    max_delay : float
        Cap on delay between retries.
    exceptions : tuple
        Exception types that trigger a retry.
    on_retry : callable, optional
        Callback ``(attempt, exception)`` invoked before each retry.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exc = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt == max_attempts:
                        logger.error(
                            f"[RETRY] {func.__name__} failed after {max_attempts} attempts: {exc}"
                        )
                        raise
                    if on_retry:
                        on_retry(attempt, exc)
                    logger.warning(
                        f"[RETRY] {func.__name__} attempt {attempt}/{max_attempts} "
                        f"failed ({exc}), retrying in {delay:.1f}s…"
                    )
                    time.sleep(delay)
                    delay = min(delay * backoff, max_delay)
            raise last_exc  # should not reach here
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Async retry decorator
# ---------------------------------------------------------------------------

def async_retry(
    max_attempts: int = 3,
    backoff: float = 2.0,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: Sequence[Type[Exception]] = (Exception,),
    on_retry: Optional[Callable] = None,
):
    """Async version of :func:`retry`."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exc = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt == max_attempts:
                        logger.error(
                            f"[RETRY] {func.__name__} failed after {max_attempts} attempts: {exc}"
                        )
                        raise
                    if on_retry:
                        on_retry(attempt, exc)
                    logger.warning(
                        f"[RETRY] {func.__name__} attempt {attempt}/{max_attempts} "
                        f"failed ({exc}), retrying in {delay:.1f}s…"
                    )
                    await asyncio.sleep(delay)
                    delay = min(delay * backoff, max_delay)
            raise last_exc
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------

class CircuitBreaker:
    """
    Prevents repeated calls to a failing service.

    States:
        CLOSED  — Normal operation.  Failures are counted.
        OPEN    — All calls are immediately rejected for ``reset_timeout`` seconds.
        HALF-OPEN — After timeout, one trial call is allowed.

    Usage::

        cb = CircuitBreaker(failure_threshold=5, reset_timeout=60)

        # As context manager
        with cb:
            risky_call()

        # Or manually
        if cb.allow_request():
            try:
                risky_call()
                cb.record_success()
            except Exception:
                cb.record_failure()
    """

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"

    def __init__(self, failure_threshold: int = 5, reset_timeout: float = 60.0,
                 name: str = "default"):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.name = name

        self._state = self.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0
        self._success_count = 0

    @property
    def state(self) -> str:
        if self._state == self.OPEN:
            if time.time() - self._last_failure_time >= self.reset_timeout:
                self._state = self.HALF_OPEN
        return self._state

    def allow_request(self) -> bool:
        s = self.state
        if s == self.CLOSED:
            return True
        if s == self.HALF_OPEN:
            return True
        return False

    def record_success(self):
        self._failure_count = 0
        self._state = self.CLOSED
        self._success_count += 1

    def record_failure(self):
        self._failure_count += 1
        self._last_failure_time = time.time()
        if self._failure_count >= self.failure_threshold:
            self._state = self.OPEN
            logger.warning(
                f"[CIRCUIT] {self.name} OPEN after {self._failure_count} failures. "
                f"Blocking for {self.reset_timeout}s."
            )

    def __enter__(self):
        if not self.allow_request():
            raise RuntimeError(
                f"CircuitBreaker '{self.name}' is {self.state} — call blocked"
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.record_success()
        else:
            self.record_failure()
        return False  # don't suppress exception


# ---------------------------------------------------------------------------
# State Persistence (crash recovery)
# ---------------------------------------------------------------------------

class StatePersistence:
    """
    Simple JSON-based state persistence for crash recovery.

    Saves key system state (positions, orders, daily P&L) to disk so that
    a restart can resume without data loss.

    Usage::

        sp = StatePersistence("state/system_state.json")
        sp.save({"positions": {...}, "daily_pnl": -120.50})
        ...
        state = sp.load()  # after restart
    """

    def __init__(self, filepath: str = "state/system_state.json"):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def save(self, state: Dict[str, Any]) -> None:
        """Atomically save state to disk."""
        tmp = self.filepath.with_suffix(".tmp")
        try:
            with open(tmp, "w") as f:
                json.dump(state, f, indent=2, default=str)
            tmp.replace(self.filepath)  # atomic on most OSes
            logger.debug(f"[STATE] Persisted state to {self.filepath}")
        except Exception as e:
            logger.error(f"[STATE] Failed to persist state: {e}")
            if tmp.exists():
                tmp.unlink()

    def load(self) -> Dict[str, Any]:
        """Load state from disk.  Returns empty dict if no state exists."""
        if not self.filepath.exists():
            return {}
        try:
            with open(self.filepath, "r") as f:
                state = json.load(f)
            logger.info(f"[STATE] Recovered state from {self.filepath}")
            return state
        except Exception as e:
            logger.error(f"[STATE] Failed to load state: {e}")
            return {}

    def clear(self) -> None:
        """Remove persisted state."""
        if self.filepath.exists():
            self.filepath.unlink()
