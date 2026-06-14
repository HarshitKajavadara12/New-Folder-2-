"""
GIGA SYSTEM - Per-Strategy Circuit Breaker
============================================

Isolates strategy-level failures so one broken strategy doesn't
take down the entire portfolio.

Each strategy gets:
- Its own CircuitBreaker (failure threshold → OPEN state)
- A max-consecutive-loss counter
- A max-drawdown-from-peak check
- A daily loss cap

Usage
-----
    from risk.strategy_breaker import StrategyBreakerManager

    mgr = StrategyBreakerManager(
        max_consecutive_losses=5,
        max_strategy_drawdown_pct=0.10,
        daily_loss_limit=200.0,
    )

    if mgr.allow_trade("momentum"):
        execute_signal(...)
        # After fill:
        mgr.record_result("momentum", pnl=+45.0)
    else:
        logger.warning("momentum circuit breaker is OPEN")
"""

import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class StrategyBreaker:
    """Per-strategy risk circuit breaker."""

    def __init__(
        self,
        name: str,
        max_consecutive_losses: int = 5,
        max_drawdown_pct: float = 0.10,
        daily_loss_limit: float = 200.0,
        cooldown_seconds: float = 300.0,
    ):
        self.name = name
        self.max_consecutive_losses = max_consecutive_losses
        self.max_drawdown_pct = max_drawdown_pct
        self.daily_loss_limit = daily_loss_limit
        self.cooldown_seconds = cooldown_seconds

        # Internal state
        self._consecutive_losses = 0
        self._peak_pnl = 0.0
        self._cumulative_pnl = 0.0
        self._daily_pnl = 0.0
        self._daily_reset_ts = time.time()
        self._tripped = False
        self._trip_time: float = 0
        self._trip_reason = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def allow_trade(self) -> bool:
        """Return True if the strategy is allowed to trade."""
        if self._tripped:
            # Auto-reset after cooldown
            if time.time() - self._trip_time >= self.cooldown_seconds:
                logger.info(
                    f"[BREAKER] {self.name} — cooldown expired, resetting to CLOSED"
                )
                self.reset()
                return True
            return False
        return True

    def record_result(self, pnl: float) -> None:
        """Record a trade result and check trip conditions."""
        self._maybe_reset_daily()
        self._cumulative_pnl += pnl
        self._daily_pnl += pnl

        if pnl >= 0:
            self._consecutive_losses = 0
            self._peak_pnl = max(self._peak_pnl, self._cumulative_pnl)
        else:
            self._consecutive_losses += 1

        # --- Check trip conditions ---
        # 1. Consecutive losses
        if self._consecutive_losses >= self.max_consecutive_losses:
            self._trip(f"{self._consecutive_losses} consecutive losses")

        # 2. Drawdown from peak
        if self._peak_pnl > 0:
            dd = (self._peak_pnl - self._cumulative_pnl) / self._peak_pnl
            if dd >= self.max_drawdown_pct:
                self._trip(f"drawdown {dd:.1%} >= {self.max_drawdown_pct:.1%}")

        # 3. Daily loss cap
        if self._daily_pnl <= -self.daily_loss_limit:
            self._trip(f"daily loss ${abs(self._daily_pnl):.2f} >= limit ${self.daily_loss_limit:.2f}")

    def reset(self) -> None:
        """Manually reset the breaker to CLOSED."""
        self._consecutive_losses = 0
        self._tripped = False
        self._trip_reason = ""

    @property
    def state(self) -> str:
        return "OPEN" if self._tripped else "CLOSED"

    @property
    def info(self) -> Dict[str, Any]:
        return {
            "strategy": self.name,
            "state": self.state,
            "consecutive_losses": self._consecutive_losses,
            "cumulative_pnl": round(self._cumulative_pnl, 2),
            "daily_pnl": round(self._daily_pnl, 2),
            "trip_reason": self._trip_reason,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _trip(self, reason: str) -> None:
        if self._tripped:
            return  # already tripped
        self._tripped = True
        self._trip_time = time.time()
        self._trip_reason = reason
        logger.warning(f"[BREAKER] {self.name} TRIPPED: {reason}")

    def _maybe_reset_daily(self) -> None:
        """Reset daily P&L at midnight (approximate — 24h rolling)."""
        if time.time() - self._daily_reset_ts >= 86400:
            self._daily_pnl = 0.0
            self._daily_reset_ts = time.time()


class StrategyBreakerManager:
    """
    Manages circuit breakers for multiple strategies.

    Strategies are auto-registered on first access.
    """

    def __init__(
        self,
        max_consecutive_losses: int = 5,
        max_strategy_drawdown_pct: float = 0.10,
        daily_loss_limit: float = 200.0,
        cooldown_seconds: float = 300.0,
    ):
        self._defaults = dict(
            max_consecutive_losses=max_consecutive_losses,
            max_drawdown_pct=max_strategy_drawdown_pct,
            daily_loss_limit=daily_loss_limit,
            cooldown_seconds=cooldown_seconds,
        )
        self._breakers: Dict[str, StrategyBreaker] = {}

    def _get(self, name: str) -> StrategyBreaker:
        if name not in self._breakers:
            self._breakers[name] = StrategyBreaker(name=name, **self._defaults)
        return self._breakers[name]

    def allow_trade(self, strategy_name: str) -> bool:
        return self._get(strategy_name).allow_trade()

    def record_result(self, strategy_name: str, pnl: float) -> None:
        self._get(strategy_name).record_result(pnl)

    def get_all_states(self) -> Dict[str, Dict]:
        return {name: b.info for name, b in self._breakers.items()}

    def reset(self, strategy_name: str) -> None:
        if strategy_name in self._breakers:
            self._breakers[strategy_name].reset()

    def reset_all(self) -> None:
        for b in self._breakers.values():
            b.reset()
