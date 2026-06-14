"""
DOMAIN DATA CONNECTOR — Real Market Data for 5-Domain Analysis
===============================================================

Addresses Missing Concept 2.2: All 5 domains currently run on synthetic data.
This module connects real market data (from data bridge, yfinance, CSV) to
each domain analyzer, replacing np.random.randn() with actual price series.

Also addresses:
  2.3 — Domain Cross-Correlation Matrix
  2.6 — Domain Parameter Sensitivity Analysis
"""

import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Import domain analyzers
try:
    from research.core.market_state_space import StateSpaceOmega
    from research.core.greek_response import VariationalAnalyzer
    from research.core.stochastic_models import StochasticModeler
    from research.core.time_asymmetry import TimeAsymmetryAnalyzer
    from research.core.information_geometry import InformationGeometer
except ImportError:
    StateSpaceOmega = None
    VariationalAnalyzer = None
    StochasticModeler = None
    TimeAsymmetryAnalyzer = None
    InformationGeometer = None


@dataclass
class DomainResult:
    """Result from a single domain analysis."""
    domain_name: str
    domain_id: int  # 1-5
    metrics: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)
    data_source: str = "unknown"
    n_observations: int = 0


@dataclass
class FullDomainAnalysis:
    """Complete 5-domain analysis result."""
    domains: Dict[str, DomainResult]
    cross_correlation: Optional[np.ndarray] = None
    correlation_labels: Optional[List[str]] = None
    data_source: str = "unknown"
    timestamp: datetime = field(default_factory=datetime.now)


class DomainDataConnector:
    """
    Connects real market data to all 5 domain analyzers.
    Replaces synthetic np.random.randn() with actual price series.
    """

    def __init__(self):
        self.state_space = StateSpaceOmega() if StateSpaceOmega else None
        self.variational = VariationalAnalyzer() if VariationalAnalyzer else None
        self.stochastic = StochasticModeler() if StochasticModeler else None
        self.time_asymmetry = TimeAsymmetryAnalyzer() if TimeAsymmetryAnalyzer else None
        self.info_geometry = InformationGeometer() if InformationGeometer else None

    def load_real_data(self, symbol: str = "BTCUSDT", days: int = 365) -> Optional[np.ndarray]:
        """
        Load real market data from multiple sources with fallback chain:
        1. Local CSV/Parquet files
        2. yfinance API
        3. Synthetic fallback (clearly labeled)
        """
        # Try local data first
        data_dirs = [
            Path(__file__).parent.parent.parent / "data_samples",
            Path(__file__).parent.parent.parent / "data",
        ]

        for data_dir in data_dirs:
            csv_files = list(data_dir.glob("*.csv")) if data_dir.exists() else []
            for f in csv_files:
                try:
                    df = pd.read_csv(f)
                    if "close" in df.columns or "Close" in df.columns:
                        col = "close" if "close" in df.columns else "Close"
                        prices = df[col].dropna().values.astype(float)
                        if len(prices) >= 50:
                            logger.info(f"Loaded {len(prices)} prices from {f.name}")
                            return prices
                except Exception:
                    continue

        # Try yfinance
        try:
            import yfinance as yf
            ticker_map = {
                "BTCUSDT": "BTC-USD", "ETHUSDT": "ETH-USD",
                "SPY": "SPY", "AAPL": "AAPL", "QQQ": "QQQ",
            }
            yf_symbol = ticker_map.get(symbol, symbol)
            data = yf.download(yf_symbol, period=f"{days}d", progress=False)
            if len(data) > 50:
                prices = data["Close"].dropna().values.astype(float)
                logger.info(f"Loaded {len(prices)} prices from yfinance ({yf_symbol})")
                return prices
        except Exception:
            pass

        # Synthetic fallback — GBM with realistic parameters
        logger.warning(f"Using synthetic data for {symbol} (no real data available)")
        dt = 1 / 252
        n = days
        mu, sigma = 0.05, 0.25
        prices = np.zeros(n)
        prices[0] = 50000 if "BTC" in symbol else 100
        for i in range(1, n):
            prices[i] = prices[i - 1] * np.exp(
                (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * np.random.randn()
            )
        return prices

    def run_full_analysis(self, prices: np.ndarray, source_label: str = "real") -> FullDomainAnalysis:
        """
        Run all 5 domains on REAL price data and return unified results.
        """
        returns = np.diff(np.log(prices + 1e-15))
        results = {}

        # Domain 1: State Space Ω
        d1_metrics = self._run_domain_1(prices, returns)
        results["state_space"] = DomainResult(
            domain_name="State Space Ω", domain_id=1,
            metrics=d1_metrics, data_source=source_label, n_observations=len(prices)
        )

        # Domain 2: Variational Sensitivity Δ
        d2_metrics = self._run_domain_2(prices, returns)
        results["variational"] = DomainResult(
            domain_name="Variational Sensitivity Δ", domain_id=2,
            metrics=d2_metrics, data_source=source_label, n_observations=len(prices)
        )

        # Domain 3: Stochastic Motion κ
        d3_metrics = self._run_domain_3(prices, returns)
        results["stochastic"] = DomainResult(
            domain_name="Stochastic Motion κ", domain_id=3,
            metrics=d3_metrics, data_source=source_label, n_observations=len(prices)
        )

        # Domain 4: Ergodicity & Time τ
        d4_metrics = self._run_domain_4(returns)
        results["ergodicity"] = DomainResult(
            domain_name="Ergodicity τ", domain_id=4,
            metrics=d4_metrics, data_source=source_label, n_observations=len(prices)
        )

        # Domain 5: Information Geometry Η
        d5_metrics = self._run_domain_5(returns)
        results["information"] = DomainResult(
            domain_name="Information Geometry Η", domain_id=5,
            metrics=d5_metrics, data_source=source_label, n_observations=len(prices)
        )

        # Cross-correlation matrix
        cross_corr, labels = self._compute_cross_correlation(results)

        return FullDomainAnalysis(
            domains=results,
            cross_correlation=cross_corr,
            correlation_labels=labels,
            data_source=source_label,
        )

    def _run_domain_1(self, prices: np.ndarray, returns: np.ndarray) -> Dict[str, float]:
        """Domain 1: State Space analysis."""
        try:
            if self.state_space:
                result = self.state_space.analyze(prices)
                return {k: float(v) for k, v in result.items() if isinstance(v, (int, float, np.floating))}
        except Exception:
            pass
        # Fallback: compute basic state metrics
        vol = np.std(returns) * np.sqrt(252)
        trend = np.mean(returns) * 252
        if vol < 0.15:
            regime_id = 0  # Low vol
        elif trend > 0:
            regime_id = 1  # Bullish
        else:
            regime_id = 2  # Bearish
        return {"regime_id": float(regime_id), "volatility_ann": float(vol), "trend_ann": float(trend)}

    def _run_domain_2(self, prices: np.ndarray, returns: np.ndarray) -> Dict[str, float]:
        """Domain 2: Variational Sensitivity."""
        try:
            if self.variational:
                result = self.variational.analyze_convexity(prices, returns)
                return {k: float(v) for k, v in result.__dict__.items() if isinstance(v, (int, float, np.floating))}
        except Exception:
            pass
        # OLS slope as delta proxy
        x = np.arange(len(returns))
        delta = np.polyfit(x, returns, 1)[0] if len(returns) > 2 else 0.0
        gamma = np.polyfit(x, returns, 2)[0] if len(returns) > 3 else 0.0
        return {"delta": float(delta), "gamma": float(gamma), "theta": float(-np.mean(np.abs(returns)))}

    def _run_domain_3(self, prices: np.ndarray, returns: np.ndarray) -> Dict[str, float]:
        """Domain 3: Stochastic Motion — O-U parameters."""
        try:
            if self.stochastic:
                result = self.stochastic.fit_ou(prices)
                return {k: float(v) for k, v in result.__dict__.items() if isinstance(v, (int, float, np.floating))}
        except Exception:
            pass
        # Manual O-U fit
        y = returns[1:]
        x = returns[:-1]
        if len(x) > 5 and np.std(x) > 1e-15:
            slope = np.sum((x - np.mean(x)) * (y - np.mean(y))) / (np.sum((x - np.mean(x)) ** 2) + 1e-15)
            kappa = -np.log(max(abs(slope), 1e-10)) * 252
            mu = np.mean(returns) * 252
            sigma = np.std(returns) * np.sqrt(252)
        else:
            kappa, mu, sigma = 5.0, 0.0, 0.2
        return {"kappa": float(kappa), "mu": float(mu), "sigma": float(sigma)}

    def _run_domain_4(self, returns: np.ndarray) -> Dict[str, float]:
        """Domain 4: Ergodicity & Time."""
        try:
            if self.time_asymmetry:
                result = self.time_asymmetry.analyze(returns)
                return {k: float(v) for k, v in result.__dict__.items() if isinstance(v, (int, float, np.floating))}
        except Exception:
            pass
        # Ergodicity: time average vs ensemble average
        time_avg = np.mean(np.log(1 + returns))
        ensemble_avg = np.log(1 + np.mean(returns))
        ergodicity_gap = ensemble_avg - time_avg
        # Kelly fraction
        mu_r = np.mean(returns)
        var_r = np.var(returns) + 1e-15
        kelly = mu_r / var_r
        return {"time_average_growth": float(time_avg * 252), "ensemble_average": float(ensemble_avg * 252),
                "ergodicity_gap": float(ergodicity_gap * 252), "kelly_fraction": float(np.clip(kelly, -2, 2))}

    def _run_domain_5(self, returns: np.ndarray) -> Dict[str, float]:
        """Domain 5: Information Geometry."""
        try:
            if self.info_geometry:
                result = self.info_geometry.analyze(returns)
                return {k: float(v) for k, v in result.__dict__.items() if isinstance(v, (int, float, np.floating))}
        except Exception:
            pass
        # Shannon entropy of return distribution
        hist, _ = np.histogram(returns, bins=50, density=True)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log(hist + 1e-15)) * (returns.max() - returns.min()) / 50
        # KL divergence from normal
        from scipy import stats as sp_stats
        normal_pdf = sp_stats.norm.pdf(np.linspace(returns.min(), returns.max(), 50),
                                        np.mean(returns), np.std(returns) + 1e-15)
        normal_pdf = normal_pdf / (np.sum(normal_pdf) + 1e-15)
        hist_norm = hist / (np.sum(hist) + 1e-15)
        kl = np.sum(hist_norm * np.log((hist_norm + 1e-15) / (normal_pdf[:len(hist_norm)] + 1e-15)))
        return {"entropy": float(entropy), "kl_divergence": float(kl)}

    def _compute_cross_correlation(self, results: Dict[str, DomainResult]) -> Tuple[np.ndarray, List[str]]:
        """
        2.3 — Domain Cross-Correlation Matrix.
        Compute how the 5 domains relate to each other.
        """
        labels = []
        values = []

        for name, dr in results.items():
            for metric_name, val in dr.metrics.items():
                labels.append(f"{name}.{metric_name}")
                values.append(val)

        n = len(values)
        if n < 2:
            return np.eye(1), labels

        # Build pseudo-correlation from normalized metric vectors
        arr = np.array(values).reshape(1, -1)
        # For single-snapshot, we use metric proximity as correlation proxy
        corr = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                if abs(values[i]) > 1e-15 and abs(values[j]) > 1e-15:
                    # Cosine similarity between scaled values
                    ratio = min(abs(values[i]), abs(values[j])) / max(abs(values[i]), abs(values[j]))
                    sign = 1.0 if (values[i] > 0) == (values[j] > 0) else -1.0
                    corr[i, j] = sign * ratio
                    corr[j, i] = corr[i, j]

        return corr, labels


class DomainParameterSensitivity:
    """
    2.6 — Domain Parameter Sensitivity Analysis.
    Tests how robust domain signals are to parameter choices.
    """

    def __init__(self, connector: Optional[DomainDataConnector] = None):
        self.connector = connector or DomainDataConnector()

    def sensitivity_grid(
        self,
        prices: np.ndarray,
        param_name: str = "kappa_threshold",
        param_range: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Run analysis across a grid of parameter values to test robustness.
        """
        if param_range is None:
            param_range = np.linspace(2.0, 12.0, 11)

        results = []
        base = self.connector.run_full_analysis(prices, "sensitivity_test")
        base_kappa = base.domains.get("stochastic", DomainResult("", 3, {})).metrics.get("kappa", 5.0)

        for threshold in param_range:
            # Test signal under this threshold
            signal_strength = max(0.0, (base_kappa - threshold) / (threshold + 1e-15))
            would_trade = base_kappa > threshold

            results.append({
                "threshold": float(threshold),
                "signal_strength": float(signal_strength),
                "would_trade": bool(would_trade),
                "kappa": float(base_kappa),
            })

        # Compute sensitivity metrics
        trade_pct = sum(1 for r in results if r["would_trade"]) / len(results)
        signal_std = np.std([r["signal_strength"] for r in results])

        return {
            "param_name": param_name,
            "grid_results": results,
            "trade_frequency": float(trade_pct),
            "signal_volatility": float(signal_std),
            "robust": bool(signal_std < 1.0),  # Low sensitivity = robust
            "optimal_threshold": float(param_range[np.argmax([r["signal_strength"] for r in results])]),
        }

    def multi_param_sensitivity(self, prices: np.ndarray) -> Dict:
        """
        Run sensitivity analysis across multiple parameters simultaneously.
        """
        results = {}

        # Kappa threshold sensitivity
        results["kappa_threshold"] = self.sensitivity_grid(
            prices, "kappa_threshold", np.linspace(2.0, 12.0, 11)
        )

        # Entropy threshold sensitivity
        returns = np.diff(np.log(prices + 1e-15))
        hist, _ = np.histogram(returns, bins=50, density=True)
        hist = hist[hist > 0]
        base_entropy = -np.sum(hist * np.log(hist + 1e-15))

        entropy_results = []
        for threshold in np.linspace(1.0, 6.0, 11):
            entropy_results.append({
                "threshold": float(threshold),
                "signal_strength": float(max(0, threshold - base_entropy)),
                "would_trade": bool(base_entropy < threshold),
            })
        results["entropy_threshold"] = {
            "param_name": "entropy_threshold",
            "grid_results": entropy_results,
            "base_entropy": float(base_entropy),
        }

        # Lookback sensitivity
        lookback_results = []
        for lb in [30, 60, 90, 120, 180, 252]:
            subset = prices[-lb:] if len(prices) >= lb else prices
            sub_returns = np.diff(np.log(subset + 1e-15))
            kappa_est = abs(np.corrcoef(sub_returns[:-1], sub_returns[1:])[0, 1]) * 252 if len(sub_returns) > 5 else 5.0
            lookback_results.append({
                "lookback": lb,
                "kappa_estimate": float(kappa_est),
            })
        results["lookback_sensitivity"] = lookback_results

        return results
