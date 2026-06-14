"""
Unit tests for OrderRouter and SmartOrderRouter.
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from execution.order_router import OrderRouter
from execution.smart_router import SmartOrderRouter, Venue, VenueMetrics


# Fake executor for testing
class MockExecutor:
    def __init__(self):
        self.orders = []

    def post_order(self, symbol, side, quantity, price=0.0, **kw):
        order = {"symbol": symbol, "side": side, "quantity": quantity, "price": price, "status": "FILLED"}
        self.orders.append(order)
        return order


class TestOrderRouter(unittest.TestCase):

    def setUp(self):
        self.executor = MockExecutor()
        self.router = OrderRouter(self.executor)

    def test_skip_no_action(self):
        result = self.router.route_order({})
        self.assertEqual(result["status"], "SKIPPED")

    def test_routes_symbol_from_signal(self):
        sig = {"action": "EXECUTE_ENTRY", "symbol": "ETHUSDT"}
        result = self.router.route_order(sig, current_price=3500, quantity_override=2.0)
        self.assertEqual(result["symbol"], "ETHUSDT")
        self.assertEqual(result["side"], "BUY")
        self.assertAlmostEqual(result["quantity"], 2.0)

    def test_default_symbol_used(self):
        sig = {"action": "EXECUTE_ENTRY"}
        result = self.router.route_order(sig, current_price=65000)
        self.assertEqual(result["symbol"], "BTCUSDT")

    def test_exit_maps_to_sell(self):
        sig = {"action": "EXECUTE_EXIT", "symbol": "SOLUSDT"}
        result = self.router.route_order(sig)
        self.assertEqual(result["side"], "SELL")

    def test_batch_routing(self):
        signals = [
            {"action": "EXECUTE_ENTRY", "symbol": "BTCUSDT"},
            {"action": "EXECUTE_ENTRY", "symbol": "ETHUSDT"},
        ]
        prices = {"BTCUSDT": 65000, "ETHUSDT": 3500}
        results = self.router.route_batch(signals, prices)
        self.assertEqual(len(results), 2)
        symbols = [r["symbol"] for r in results]
        self.assertIn("BTCUSDT", symbols)
        self.assertIn("ETHUSDT", symbols)


class TestSmartOrderRouter(unittest.TestCase):

    def setUp(self):
        self.sor = SmartOrderRouter()

    def test_venues_are_crypto(self):
        venues = [v.value for v in Venue]
        self.assertIn("BINANCE", venues)
        self.assertIn("COINBASE", venues)
        self.assertNotIn("NYSE", venues)

    def test_route_returns_allocations(self):
        allocs = self.sor.route_order("BTCUSDT", 1.0, urgency=0.5, max_venues=3, price=65000)
        self.assertTrue(len(allocs) > 0)
        total_qty = sum(a[1] for a in allocs)
        self.assertAlmostEqual(total_qty, 1.0, places=4)

    def test_best_venue_returns_venue(self):
        venue = self.sor.get_best_venue("BTCUSDT", 1.0)
        self.assertIsInstance(venue, Venue)

    def test_routing_stats(self):
        self.sor.route_order("BTCUSDT", 0.5, price=65000)
        stats = self.sor.get_routing_stats()
        self.assertGreater(stats["total_orders_routed"], 0)

    def test_update_venue_metrics(self):
        original = self.sor.venue_metrics[Venue.BINANCE].avg_latency_ms
        self.sor.update_venue_metrics(Venue.BINANCE, avg_latency_ms=200)
        updated = self.sor.venue_metrics[Venue.BINANCE].avg_latency_ms
        # EMA: 0.9 * original + 0.1 * 200
        expected = 0.9 * original + 0.1 * 200
        self.assertAlmostEqual(updated, expected, places=1)


if __name__ == "__main__":
    unittest.main()
