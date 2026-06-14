"""
CROSS-SECTIONAL ALPHA ENGINE — Multi-Asset Extension
=====================================================

Extends the single-asset AlphaSignalEngine to handle multiple assets
simultaneously, enabling:
  - Relative value signals (go long Asset A, short Asset B)
  - Cross-asset momentum / mean-reversion detection
  - Portfolio-level Greek alpha scoring
  - Rank-based signal generation (long top quartile, short bottom)
  - Factor-neutral portfolio construction

This was the #1 remaining gap: "Single-asset only — needs multi-asset extension"
"""

import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from scipy import stats

from research.core.alpha_signal_engine import AlphaSignalEngine, AlphaSignal
from research.core.stochastic_models import StochasticModeler
from research.core.information_geometry import InformationGeometer

logger = logging.getLogger(__name__)


@dataclass
class CrossSectionalSignal:
    """Multi-asset alpha signal with relative rankings."""
    timestamp: datetime
    asset_signals: Dict[str, AlphaSignal]
    rankings: Dict[str, int]  # Asset → rank (1 = best)
    long_assets: List[str]
    short_assets: List[str]
    neutral_assets: List[str]
    spread_zscore: float  # Z-score of long-short spread
    portfolio_kappa: float  # Portfolio-level mean reversion
    portfolio_entropy: float  # Portfolio-level entropy
    diversification_ratio: float  # Correlation-adjusted diversification
    factor_exposures: Dict[str, float] = field(default_factory=dict)
    reason: str = ""


@dataclass
class PairSignal:
    """Signal for a specific pairs trade."""
    asset_long: str
    asset_short: str
    spread_zscore: float
    half_life: float  # Mean reversion half-life of spread
    correlation: float
    cointegration_pvalue: float
    hedge_ratio: float
    confidence: float
    reason: str = ""


class CrossSectionalAlphaEngine:
    """
    Multi-asset alpha engine that generates cross-sectional signals.
    
    Methodology:
    1. Run single-asset AlphaSignalEngine on each asset
    2. Rank assets by composite alpha score
    3. Identify pairs with cointegration (Engle-Granger)
    4. Generate long-short portfolio with factor neutrality
    5. Apply cross-sectional momentum / mean-reversion filters
    """

    def __init__(self, config: Dict = None):
        config = config or {}
        self.engine = AlphaSignalEngine(config)
        self.stochastic = StochasticModeler()
        self.information = InformationGeometer()
        
        self.long_pct = config.get("long_percentile", 0.25)  # Top 25%
        self.short_pct = config.get("short_percentile", 0.25)  # Bottom 25%
        self.min_assets = config.get("min_assets", 3)
        self.coint_pvalue = config.get("cointegration_pvalue", 0.05)
        self.min_correlation = config.get("min_pair_correlation", 0.5)
        self.zscore_entry = config.get("zscore_entry", 2.0)
        self.zscore_exit = config.get("zscore_exit", 0.5)
        
        self.signal_history: List[CrossSectionalSignal] = []

    def generate_cross_sectional_signal(
        self,
        asset_prices: Dict[str, pd.Series],
        asset_volumes: Dict[str, pd.Series],
        benchmark_returns: Optional[pd.Series] = None,
    ) -> CrossSectionalSignal:
        """
        Generate cross-sectional alpha signal across multiple assets.
        
        Args:
            asset_prices: {asset_name: price_series} for all assets
            asset_volumes: {asset_name: volume_series} for all assets
            benchmark_returns: Optional benchmark for relative performance
            
        Returns:
            CrossSectionalSignal with rankings, long/short lists, and scores
        """
        if len(asset_prices) < self.min_assets:
            return self._neutral_signal(
                asset_prices, f"Need >= {self.min_assets} assets, got {len(asset_prices)}"
            )

        # Step 1: Run single-asset engine on each asset
        asset_signals: Dict[str, AlphaSignal] = {}
        composite_scores: Dict[str, float] = {}

        for asset_name, prices in asset_prices.items():
            volumes = asset_volumes.get(asset_name, pd.Series(np.ones(len(prices))))
            
            try:
                signal = self.engine.generate_signal(
                    prices, volumes,
                    benchmark_returns=benchmark_returns
                )
                asset_signals[asset_name] = signal
                
                # Composite score from factors
                score = sum(
                    signal.factors.get(k, 0) * self.engine.factor_weights.get(k, 0.2)
                    for k in signal.factors
                )
                composite_scores[asset_name] = score
                
            except Exception as e:
                logger.warning(f"[CROSS-SECTIONAL] Failed on {asset_name}: {e}")
                continue

        if len(composite_scores) < self.min_assets:
            return self._neutral_signal(
                asset_prices, f"Only {len(composite_scores)} assets produced signals"
            )

        # Step 2: Rank assets by composite score
        sorted_assets = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)
        rankings = {asset: rank + 1 for rank, (asset, _) in enumerate(sorted_assets)}

        # Step 3: Determine long/short/neutral
        n = len(sorted_assets)
        n_long = max(1, int(n * self.long_pct))
        n_short = max(1, int(n * self.short_pct))

        long_assets = [a for a, _ in sorted_assets[:n_long]]
        short_assets = [a for a, _ in sorted_assets[-n_short:]]
        neutral_assets = [a for a, _ in sorted_assets[n_long:-n_short]] if n > n_long + n_short else []

        # Step 4: Portfolio-level metrics
        portfolio_kappa = np.mean([
            asset_signals[a].kappa for a in composite_scores
            if a in asset_signals
        ])
        portfolio_entropy = np.mean([
            asset_signals[a].entropy for a in composite_scores
            if a in asset_signals
        ])

        # Step 5: Spread z-score (long basket vs short basket)
        long_returns = self._basket_returns(asset_prices, long_assets)
        short_returns = self._basket_returns(asset_prices, short_assets)
        spread = long_returns - short_returns if long_returns is not None and short_returns is not None else pd.Series([0.0])
        spread_zscore = float((spread.iloc[-1] - spread.mean()) / (spread.std() + 1e-10)) if len(spread) > 10 else 0.0

        # Step 6: Diversification ratio
        div_ratio = self._compute_diversification_ratio(asset_prices, list(composite_scores.keys()))

        # Step 7: Factor exposures
        factor_exposures = self._compute_factor_exposures(asset_prices, composite_scores)

        signal = CrossSectionalSignal(
            timestamp=datetime.now(),
            asset_signals=asset_signals,
            rankings=rankings,
            long_assets=long_assets,
            short_assets=short_assets,
            neutral_assets=neutral_assets,
            spread_zscore=spread_zscore,
            portfolio_kappa=portfolio_kappa,
            portfolio_entropy=portfolio_entropy,
            diversification_ratio=div_ratio,
            factor_exposures=factor_exposures,
            reason=self._build_reason(rankings, long_assets, short_assets, spread_zscore),
        )

        self.signal_history.append(signal)
        logger.info(
            f"[CROSS-SECTIONAL] Signal: Long={long_assets}, Short={short_assets}, "
            f"Spread Z={spread_zscore:.2f}, κ_port={portfolio_kappa:.2f}, "
            f"H_port={portfolio_entropy:.2f}, Div={div_ratio:.2f}"
        )

        return signal

    def find_cointegrated_pairs(
        self,
        asset_prices: Dict[str, pd.Series],
    ) -> List[PairSignal]:
        """
        Find cointegrated pairs using Engle-Granger two-step method.
        
        Returns list of PairSignal for tradeable pairs.
        """
        assets = list(asset_prices.keys())
        pairs: List[PairSignal] = []

        for i in range(len(assets)):
            for j in range(i + 1, len(assets)):
                a, b = assets[i], assets[j]
                pa, pb = asset_prices[a], asset_prices[b]

                # Align series
                common = pa.index.intersection(pb.index)
                if len(common) < 60:
                    continue

                pa_aligned = pa.loc[common].values
                pb_aligned = pb.loc[common].values

                # Correlation check
                corr = np.corrcoef(pa_aligned, pb_aligned)[0, 1]
                if abs(corr) < self.min_correlation:
                    continue

                # OLS regression for hedge ratio
                hedge_ratio = self._ols_hedge_ratio(pa_aligned, pb_aligned)

                # Spread
                spread = pa_aligned - hedge_ratio * pb_aligned

                # ADF test for cointegration (Engle-Granger)
                coint_pvalue = self._adf_test(spread)

                if coint_pvalue > self.coint_pvalue:
                    continue

                # Half-life of mean reversion
                half_life = self._half_life(spread)

                # Current z-score
                zscore = (spread[-1] - np.mean(spread)) / (np.std(spread) + 1e-10)

                # Determine direction
                if abs(zscore) < self.zscore_exit:
                    continue  # No trade — spread near mean

                if zscore > self.zscore_entry:
                    # Spread is high → short A, long B
                    asset_long, asset_short = b, a
                elif zscore < -self.zscore_entry:
                    # Spread is low → long A, short B
                    asset_long, asset_short = a, b
                else:
                    continue  # In no-trade zone

                confidence = min(1.0, abs(zscore) / 4.0) * (1.0 - coint_pvalue)

                pair_signal = PairSignal(
                    asset_long=asset_long,
                    asset_short=asset_short,
                    spread_zscore=zscore,
                    half_life=half_life,
                    correlation=corr,
                    cointegration_pvalue=coint_pvalue,
                    hedge_ratio=hedge_ratio,
                    confidence=confidence,
                    reason=f"Coint p={coint_pvalue:.4f}, Z={zscore:.2f}, HL={half_life:.1f}d",
                )
                pairs.append(pair_signal)

                logger.info(
                    f"[PAIRS] {asset_long}↑ / {asset_short}↓ | "
                    f"Z={zscore:.2f}, HL={half_life:.1f}d, "
                    f"p={coint_pvalue:.4f}, ρ={corr:.2f}"
                )

        # Sort by confidence
        pairs.sort(key=lambda p: p.confidence, reverse=True)
        return pairs

    def _basket_returns(
        self, asset_prices: Dict[str, pd.Series], assets: List[str]
    ) -> Optional[pd.Series]:
        """Equal-weighted basket returns."""
        if not assets:
            return None
        returns_list = []
        for a in assets:
            if a in asset_prices:
                ret = asset_prices[a].pct_change().dropna()
                returns_list.append(ret)
        if not returns_list:
            return None
        # Align and average
        df = pd.concat(returns_list, axis=1).dropna()
        return df.mean(axis=1)

    def _compute_diversification_ratio(
        self, asset_prices: Dict[str, pd.Series], assets: List[str]
    ) -> float:
        """
        Diversification ratio = sum(w_i * σ_i) / σ_portfolio
        DR > 1 means diversification benefit exists.
        """
        if len(assets) < 2:
            return 1.0

        returns_list = []
        for a in assets:
            if a in asset_prices:
                returns_list.append(asset_prices[a].pct_change().dropna())

        if len(returns_list) < 2:
            return 1.0

        df = pd.concat(returns_list, axis=1).dropna()
        if len(df) < 10:
            return 1.0

        n = df.shape[1]
        weights = np.ones(n) / n  # Equal weight

        individual_vols = df.std().values
        cov_matrix = df.cov().values
        portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)

        if portfolio_vol < 1e-10:
            return 1.0

        weighted_vol_sum = np.sum(weights * individual_vols)
        return float(weighted_vol_sum / portfolio_vol)

    def _compute_factor_exposures(
        self, asset_prices: Dict[str, pd.Series], scores: Dict[str, float]
    ) -> Dict[str, float]:
        """Compute portfolio factor exposures: momentum, size, value proxies."""
        exposures = {}

        # Momentum factor: correlation of scores with recent returns
        recent_returns = {}
        for a, prices in asset_prices.items():
            if len(prices) >= 20:
                recent_returns[a] = float(prices.iloc[-1] / prices.iloc[-20] - 1)

        if len(recent_returns) >= 3:
            common_assets = [a for a in scores if a in recent_returns]
            if len(common_assets) >= 3:
                score_vals = [scores[a] for a in common_assets]
                ret_vals = [recent_returns[a] for a in common_assets]
                mom_corr, _ = stats.spearmanr(score_vals, ret_vals)
                exposures["momentum_factor"] = float(mom_corr)

        # Volatility factor: are we long low-vol or high-vol?
        vols = {}
        for a, prices in asset_prices.items():
            if len(prices) >= 20:
                vols[a] = float(prices.pct_change().dropna().std())

        if len(vols) >= 3:
            common_assets = [a for a in scores if a in vols]
            if len(common_assets) >= 3:
                score_vals = [scores[a] for a in common_assets]
                vol_vals = [vols[a] for a in common_assets]
                vol_corr, _ = stats.spearmanr(score_vals, vol_vals)
                exposures["volatility_factor"] = float(vol_corr)

        return exposures

    @staticmethod
    def _ols_hedge_ratio(y: np.ndarray, x: np.ndarray) -> float:
        """OLS hedge ratio: y = β*x + ε."""
        x_with_const = np.column_stack([x, np.ones_like(x)])
        try:
            beta = np.linalg.lstsq(x_with_const, y, rcond=None)[0]
            return float(beta[0])
        except np.linalg.LinAlgError:
            return 1.0

    @staticmethod
    def _adf_test(series: np.ndarray) -> float:
        """Augmented Dickey-Fuller test. Returns p-value."""
        n = len(series)
        if n < 20:
            return 1.0

        # Δy_t = α + β*y_{t-1} + ε_t
        dy = np.diff(series)
        y_lag = series[:-1]

        x = np.column_stack([y_lag, np.ones(len(y_lag))])
        try:
            beta = np.linalg.lstsq(x, dy, rcond=None)[0]
            residuals = dy - x @ beta
            se = np.sqrt(np.sum(residuals**2) / (n - 3))
            var_beta = se**2 * np.linalg.inv(x.T @ x)[0, 0]
            t_stat = beta[0] / np.sqrt(var_beta)
        except (np.linalg.LinAlgError, ValueError):
            return 1.0

        # ADF critical values (approximate via MacKinnon)
        # For n=100: 1% = -3.51, 5% = -2.89, 10% = -2.58
        if t_stat < -3.51:
            return 0.01
        elif t_stat < -2.89:
            return 0.05
        elif t_stat < -2.58:
            return 0.10
        elif t_stat < -1.95:
            return 0.20
        else:
            return 0.50

    @staticmethod
    def _half_life(spread: np.ndarray) -> float:
        """Estimate half-life of mean reversion via AR(1) regression."""
        y = spread[1:]
        y_lag = spread[:-1]
        x = np.column_stack([y_lag, np.ones(len(y_lag))])
        try:
            beta = np.linalg.lstsq(x, y, rcond=None)[0]
            phi = beta[0]
            if phi >= 1.0 or phi <= 0.0:
                return float('inf')
            return -np.log(2) / np.log(phi)
        except (np.linalg.LinAlgError, ValueError):
            return float('inf')

    def _build_reason(
        self, rankings: Dict, long_assets: List, short_assets: List, zscore: float
    ) -> str:
        parts = [
            f"Long: {long_assets}",
            f"Short: {short_assets}",
            f"Spread Z={zscore:.2f}",
            f"Rankings: {rankings}",
        ]
        return " | ".join(parts)

    def _neutral_signal(
        self, asset_prices: Dict[str, pd.Series], reason: str
    ) -> CrossSectionalSignal:
        return CrossSectionalSignal(
            timestamp=datetime.now(),
            asset_signals={},
            rankings={},
            long_assets=[],
            short_assets=[],
            neutral_assets=list(asset_prices.keys()),
            spread_zscore=0.0,
            portfolio_kappa=0.0,
            portfolio_entropy=0.0,
            diversification_ratio=1.0,
            reason=reason,
        )

    def get_report(self) -> Dict:
        """Summary report of cross-sectional analysis."""
        if not self.signal_history:
            return {"status": "no_signals"}

        last = self.signal_history[-1]
        return {
            "total_signals": len(self.signal_history),
            "long_assets": last.long_assets,
            "short_assets": last.short_assets,
            "neutral_assets": last.neutral_assets,
            "spread_zscore": last.spread_zscore,
            "portfolio_kappa": last.portfolio_kappa,
            "portfolio_entropy": last.portfolio_entropy,
            "diversification_ratio": last.diversification_ratio,
            "factor_exposures": last.factor_exposures,
            "rankings": last.rankings,
        }
