"""
Unit tests for risk management: SessionGuard, StrategyBreaker, CircuitBreaker.
"""
import sys
import os
import time
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from risk.strategy_breaker import StrategyBreaker, StrategyBreakerManager
from utils.retry import CircuitBreaker


class TestCircuitBreaker(unittest.TestCase):

    def test_closed_by_default(self):
        cb = CircuitBreaker(failure_threshold=3, reset_timeout=1.0)
        self.assertEqual(cb.state, CircuitBreaker.CLOSED)
        self.assertTrue(cb.allow_request())

    def test_opens_after_threshold(self):
        cb = CircuitBreaker(failure_threshold=3, reset_timeout=60)
        for _ in range(3):
            cb.record_failure()
        self.assertEqual(cb.state, CircuitBreaker.OPEN)
        self.assertFalse(cb.allow_request())

    def test_resets_on_success(self):
        cb = CircuitBreaker(failure_threshold=3, reset_timeout=60)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        self.assertEqual(cb.state, CircuitBreaker.CLOSED)

    def test_half_open_after_timeout(self):
        cb = CircuitBreaker(failure_threshold=2, reset_timeout=0.1)
        cb.record_failure()
        cb.record_failure()
        self.assertEqual(cb.state, CircuitBreaker.OPEN)
        time.sleep(0.15)
        self.assertEqual(cb.state, CircuitBreaker.HALF_OPEN)
        self.assertTrue(cb.allow_request())


class TestStrategyBreaker(unittest.TestCase):

    def test_allows_by_default(self):
        sb = StrategyBreaker("momentum")
        self.assertTrue(sb.allow_trade())

    def test_trips_on_consecutive_losses(self):
        sb = StrategyBreaker("momentum", max_consecutive_losses=3, cooldown_seconds=0.1)
        for _ in range(3):
            sb.record_result(-50.0)
        self.assertFalse(sb.allow_trade())
        self.assertEqual(sb.state, "OPEN")

    def test_resets_loss_streak_on_win(self):
        sb = StrategyBreaker("momentum", max_consecutive_losses=3)
        sb.record_result(-50.0)
        sb.record_result(-50.0)
        sb.record_result(100.0)  # win resets streak
        sb.record_result(-50.0)
        self.assertTrue(sb.allow_trade())

    def test_trips_on_daily_loss(self):
        sb = StrategyBreaker("pairs", daily_loss_limit=100, cooldown_seconds=0.1)
        sb.record_result(-60)
        sb.record_result(-50)
        self.assertFalse(sb.allow_trade())

    def test_cooldown_resets(self):
        sb = StrategyBreaker("vol", max_consecutive_losses=2, cooldown_seconds=0.1)
        sb.record_result(-10)
        sb.record_result(-10)
        self.assertFalse(sb.allow_trade())
        time.sleep(0.15)
        self.assertTrue(sb.allow_trade())


class TestStrategyBreakerManager(unittest.TestCase):

    def test_auto_registers_strategy(self):
        mgr = StrategyBreakerManager()
        self.assertTrue(mgr.allow_trade("new_strategy"))

    def test_independent_strategies(self):
        mgr = StrategyBreakerManager(max_consecutive_losses=2, cooldown_seconds=0.1)
        mgr.record_result("bad_strat", -100)
        mgr.record_result("bad_strat", -100)
        self.assertFalse(mgr.allow_trade("bad_strat"))
        self.assertTrue(mgr.allow_trade("good_strat"))

    def test_get_all_states(self):
        mgr = StrategyBreakerManager()
        mgr.record_result("a", 10)
        mgr.record_result("b", -5)
        states = mgr.get_all_states()
        self.assertIn("a", states)
        self.assertIn("b", states)


if __name__ == "__main__":
    unittest.main()
