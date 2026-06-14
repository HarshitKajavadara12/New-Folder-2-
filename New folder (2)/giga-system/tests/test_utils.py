"""
Unit tests for rate limiter and storage manager SQL safety.
"""
import sys
import os
import re
import time
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.rate_limiter import TokenBucketLimiter, SlidingWindowLimiter


class TestTokenBucketLimiter(unittest.TestCase):

    def test_burst_allows_initial(self):
        lim = TokenBucketLimiter(rate=1.0, burst=5)
        acquired = sum(1 for _ in range(5) if lim.try_acquire())
        self.assertEqual(acquired, 5)

    def test_exceeds_burst_blocks(self):
        lim = TokenBucketLimiter(rate=1.0, burst=3)
        for _ in range(3):
            lim.try_acquire()
        self.assertFalse(lim.try_acquire())

    def test_refills_over_time(self):
        lim = TokenBucketLimiter(rate=10.0, burst=2)
        lim.try_acquire()
        lim.try_acquire()
        self.assertFalse(lim.try_acquire())
        time.sleep(0.15)
        self.assertTrue(lim.try_acquire())

    def test_acquire_blocking(self):
        lim = TokenBucketLimiter(rate=10.0, burst=1)
        lim.try_acquire()
        t0 = time.monotonic()
        ok = lim.acquire(timeout=1.0)
        elapsed = time.monotonic() - t0
        self.assertTrue(ok)
        self.assertGreater(elapsed, 0.05)


class TestSlidingWindowLimiter(unittest.TestCase):

    def test_allows_within_limit(self):
        lim = SlidingWindowLimiter(max_requests=5, window_seconds=1.0)
        for _ in range(5):
            self.assertTrue(lim.try_acquire())

    def test_blocks_over_limit(self):
        lim = SlidingWindowLimiter(max_requests=3, window_seconds=1.0)
        for _ in range(3):
            lim.try_acquire()
        self.assertFalse(lim.try_acquire())

    def test_window_expires(self):
        lim = SlidingWindowLimiter(max_requests=2, window_seconds=0.1)
        lim.try_acquire()
        lim.try_acquire()
        self.assertFalse(lim.try_acquire())
        time.sleep(0.15)
        self.assertTrue(lim.try_acquire())

    def test_usage_property(self):
        lim = SlidingWindowLimiter(max_requests=10, window_seconds=1.0)
        lim.try_acquire()
        lim.try_acquire()
        self.assertEqual(lim.usage, 2)


class TestSQLIdentifierValidation(unittest.TestCase):
    """Test the SQL identifier validation in storage_manager."""

    def setUp(self):
        # Import the validation function
        from data.storage_manager import _validate_identifier, _ALLOWED_TABLES
        self.validate = _validate_identifier
        self.allowed = _ALLOWED_TABLES

    def test_allowed_tables_pass(self):
        for t in ("market_data", "trades", "positions"):
            self.assertEqual(self.validate(t), t)

    def test_valid_identifier(self):
        self.assertEqual(self.validate("my_table_123"), "my_table_123")

    def test_rejects_injection(self):
        with self.assertRaises(ValueError):
            self.validate("users; DROP TABLE--")

    def test_rejects_empty(self):
        with self.assertRaises(ValueError):
            self.validate("")

    def test_rejects_special_chars(self):
        with self.assertRaises(ValueError):
            self.validate("table'name")


if __name__ == "__main__":
    unittest.main()
