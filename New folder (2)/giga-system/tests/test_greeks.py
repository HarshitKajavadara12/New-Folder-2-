"""
Unit tests for Black-Scholes pricing and Greeks.
"""
import math
import sys
import os
import unittest

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestBlackScholes(unittest.TestCase):
    """Smoke tests for BS pricing using closed-form benchmarks."""

    def setUp(self):
        # Standard test params: S=100, K=100, T=1, r=0.05, sigma=0.2
        self.S = 100.0
        self.K = 100.0
        self.T = 1.0
        self.r = 0.05
        self.sigma = 0.2

    # ------------------------------------------------------------------
    # Helpers (pure-Python BS for reference, no dependency on project code)
    # ------------------------------------------------------------------
    @staticmethod
    def _norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def _bs_call(self, S, K, T, r, sigma):
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return S * self._norm_cdf(d1) - K * math.exp(-r * T) * self._norm_cdf(d2)

    def _bs_put(self, S, K, T, r, sigma):
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return K * math.exp(-r * T) * self._norm_cdf(-d2) - S * self._norm_cdf(-d1)

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------
    def test_call_price_at_the_money(self):
        price = self._bs_call(self.S, self.K, self.T, self.r, self.sigma)
        # ATM call with these params should be ~$10.45
        self.assertAlmostEqual(price, 10.4506, places=2)

    def test_put_call_parity(self):
        call = self._bs_call(self.S, self.K, self.T, self.r, self.sigma)
        put = self._bs_put(self.S, self.K, self.T, self.r, self.sigma)
        # C - P = S - K * exp(-rT)
        lhs = call - put
        rhs = self.S - self.K * math.exp(-self.r * self.T)
        self.assertAlmostEqual(lhs, rhs, places=6)

    def test_deep_itm_call(self):
        # Deep ITM call should be close to S - K*exp(-rT)
        call = self._bs_call(200, 100, 1, 0.05, 0.2)
        intrinsic = 200 - 100 * math.exp(-0.05)
        self.assertGreater(call, intrinsic)
        self.assertAlmostEqual(call, intrinsic, delta=2.0)

    def test_deep_otm_put(self):
        # Deep OTM put should be near zero
        put = self._bs_put(200, 100, 1, 0.05, 0.2)
        self.assertAlmostEqual(put, 0.0, places=2)

    def test_zero_vol_call(self):
        # With sigma → 0, call = max(0, S - K*exp(-rT))
        # Use tiny sigma to avoid div-by-zero
        call = self._bs_call(100, 95, 1, 0.05, 0.0001)
        expected = max(0, 100 - 95 * math.exp(-0.05))
        self.assertAlmostEqual(call, expected, places=1)

    def test_delta_approximation(self):
        """Finite-difference delta should be ~0.64 for ATM call."""
        eps = 0.01
        c_up = self._bs_call(self.S + eps, self.K, self.T, self.r, self.sigma)
        c_dn = self._bs_call(self.S - eps, self.K, self.T, self.r, self.sigma)
        delta = (c_up - c_dn) / (2 * eps)
        self.assertAlmostEqual(delta, 0.6368, places=2)


class TestGreeks(unittest.TestCase):
    """Tests for Greek sensitivities via finite differences."""

    @staticmethod
    def _norm_cdf(x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def _bs_call(self, S, K, T, r, sigma):
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return S * self._norm_cdf(d1) - K * math.exp(-r * T) * self._norm_cdf(d2)

    def test_gamma_positive(self):
        """Gamma (∂²C/∂S²) should always be positive for a call."""
        S, K, T, r, sig = 100, 100, 1, 0.05, 0.2
        eps = 0.5
        c_up = self._bs_call(S + eps, K, T, r, sig)
        c_mid = self._bs_call(S, K, T, r, sig)
        c_dn = self._bs_call(S - eps, K, T, r, sig)
        gamma = (c_up - 2 * c_mid + c_dn) / (eps ** 2)
        self.assertGreater(gamma, 0)

    def test_theta_negative_for_call(self):
        """Theta (∂C/∂T) should be negative for a plain-vanilla call."""
        S, K, T, r, sig = 100, 100, 1, 0.05, 0.2
        eps = 1 / 365
        c_now = self._bs_call(S, K, T, r, sig)
        c_later = self._bs_call(S, K, T - eps, r, sig)
        theta = (c_later - c_now)  # per day
        self.assertLess(theta, 0)

    def test_vega_positive(self):
        """Vega (∂C/∂σ) should be positive."""
        S, K, T, r, sig = 100, 100, 1, 0.05, 0.2
        eps = 0.001
        c_up = self._bs_call(S, K, T, r, sig + eps)
        c_dn = self._bs_call(S, K, T, r, sig - eps)
        vega = (c_up - c_dn) / (2 * eps)
        self.assertGreater(vega, 0)


if __name__ == "__main__":
    unittest.main()
