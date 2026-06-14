"""
Unit tests for LiveAccount — WAP, P&L, margin, and leverage.
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from account.live_account import LiveAccount


class TestLiveAccount(unittest.TestCase):

    def setUp(self):
        self.acc = LiveAccount(start_balance=10_000, fee_rate=0.0004, max_leverage=5.0)

    # ------------------------------------------------------------------
    # Basic P&L
    # ------------------------------------------------------------------
    def test_buy_and_sell_profit(self):
        pnl, fee1 = self.acc.execute_trade("BTCUSDT", "BUY", 50000, 0.1)
        self.assertEqual(pnl, 0.0)  # opening trade, no realized pnl
        pnl2, fee2 = self.acc.execute_trade("BTCUSDT", "SELL", 51000, 0.1)
        # profit = (51000 - 50000) * 0.1 = 100
        self.assertAlmostEqual(pnl2, 100.0, places=2)

    def test_short_profit(self):
        self.acc.execute_trade("ETHUSDT", "SELL", 3000, 1.0)
        pnl, _ = self.acc.execute_trade("ETHUSDT", "BUY", 2800, 1.0)
        # Short profit = (3000 - 2800) * 1.0 = 200
        self.assertAlmostEqual(pnl, 200.0, places=2)

    def test_partial_close(self):
        self.acc.execute_trade("BTCUSDT", "BUY", 50000, 1.0)
        pnl, _ = self.acc.execute_trade("BTCUSDT", "SELL", 52000, 0.5)
        self.assertAlmostEqual(pnl, 1000.0, places=2)
        # Still have 0.5 BTC open
        self.assertAlmostEqual(self.acc.positions["BTCUSDT"]["size"], 0.5, places=4)

    def test_wap_on_add(self):
        self.acc.execute_trade("BTCUSDT", "BUY", 40000, 1.0)
        self.acc.execute_trade("BTCUSDT", "BUY", 50000, 1.0)
        # WAP = (40000 + 50000) / 2 = 45000
        self.assertAlmostEqual(self.acc.positions["BTCUSDT"]["entry_price"], 45000, places=0)

    # ------------------------------------------------------------------
    # Fees
    # ------------------------------------------------------------------
    def test_fee_deduction(self):
        self.acc.execute_trade("BTCUSDT", "BUY", 50000, 0.1)
        # fee = 50000 * 0.1 * 0.0004 = 2.0
        self.assertAlmostEqual(self.acc.total_fees, 2.0, places=4)

    # ------------------------------------------------------------------
    # Equity & Mark Price
    # ------------------------------------------------------------------
    def test_equity_updates_with_mark(self):
        self.acc.execute_trade("BTCUSDT", "BUY", 50000, 1.0)
        eq = self.acc.update_mark_price("BTCUSDT", 55000)
        # Unrealized = (55000-50000)*1 = 5000
        # cash = 10000 - fee
        fee = 50000 * 1.0 * 0.0004
        expected = (10000 - fee) + 5000
        self.assertAlmostEqual(eq, expected, places=2)

    # ------------------------------------------------------------------
    # Margin & Leverage
    # ------------------------------------------------------------------
    def test_leverage_calculation(self):
        self.acc.execute_trade("BTCUSDT", "BUY", 50000, 1.0)
        self.acc.update_mark_price("BTCUSDT", 50000)
        info = self.acc.get_margin_info()
        # exposure = 50000, equity ~ 10000 → leverage ~ 5x
        self.assertAlmostEqual(info["current_leverage"], 5.0, delta=0.1)

    def test_leverage_check_blocks(self):
        self.acc.execute_trade("BTCUSDT", "BUY", 50000, 1.0)
        self.acc.update_mark_price("BTCUSDT", 50000)
        # Already at ~5x, adding more should fail
        ok = self.acc.check_leverage_limit("ETHUSDT", 10000)
        self.assertFalse(ok)

    def test_leverage_check_allows(self):
        ok = self.acc.check_leverage_limit("BTCUSDT", 1000)
        self.assertTrue(ok)

    def test_liquidation_price_long(self):
        self.acc.execute_trade("BTCUSDT", "BUY", 50000, 1.0)
        liq = self.acc.estimate_liquidation_price("BTCUSDT")
        self.assertIsNotNone(liq)
        self.assertLess(liq, 50000)  # liq price should be below entry for longs

    def test_liquidation_price_short(self):
        self.acc.execute_trade("BTCUSDT", "SELL", 50000, 1.0)
        liq = self.acc.estimate_liquidation_price("BTCUSDT")
        self.assertIsNotNone(liq)
        self.assertGreater(liq, 50000)  # above entry for shorts

    # ------------------------------------------------------------------
    # Account Summary
    # ------------------------------------------------------------------
    def test_account_summary_has_leverage(self):
        summary = self.acc.get_account_summary()
        self.assertIn("leverage", summary)
        self.assertIn("margin_used", summary)
        self.assertIn("margin_available", summary)


if __name__ == "__main__":
    unittest.main()
