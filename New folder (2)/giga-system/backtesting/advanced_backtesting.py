"""
BACKTESTING ENHANCEMENTS — Comparison Dashboard, Greek-Aware, Options, Multi-Asset
====================================================================================

Addresses Missing Concepts 8.2-8.5:
  8.2 — Backtest Result Comparison Dashboard
  8.3 — Greek-Aware Backtesting (domain signals as features)
  8.4 — Options Strategy Backtesting (with historical IV)
  8.5 — Multi-Asset Backtest (portfolio with rebalancing)
"""

import numpy as np
import pandas as pd
import logging
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


# =============================================================================
# 8.2 — BACKTEST RESULT COMPARISON DASHBOARD
# =============================================================================

@dataclass
class BacktestSummary:
    """Summary metrics for a single backtest run."""
    run_id: str
    strategy: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    n_trades: int
    start_date: str
    end_date: str
    parameters: Dict = field(default_factory=dict)


class BacktestComparison:
    """
    Compare multiple backtest runs side-by-side.
    """

    def __init__(self, results_dir: Optional[Path] = None):
        self.results_dir = results_dir or Path(__file__).parent.parent / "artifacts" / "backtests"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.runs: List[BacktestSummary] = []

    def add_run(self, summary: BacktestSummary):
        """Add a backtest run to comparison."""
        self.runs.append(summary)
        # Persist
        path = self.results_dir / f"{summary.run_id}.json"
        path.write_text(json.dumps({
            "run_id": summary.run_id, "strategy": summary.strategy,
            "total_return": summary.total_return, "sharpe_ratio": summary.sharpe_ratio,
            "max_drawdown": summary.max_drawdown, "win_rate": summary.win_rate,
            "n_trades": summary.n_trades, "start_date": summary.start_date,
            "end_date": summary.end_date, "parameters": summary.parameters,
        }, indent=2, default=str), encoding="utf-8")

    def load_all(self) -> List[BacktestSummary]:
        """Load all saved backtest results."""
        self.runs = []
        for f in sorted(self.results_dir.glob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                self.runs.append(BacktestSummary(**data))
            except Exception:
                continue
        return self.runs

    def compare(self, run_ids: Optional[List[str]] = None) -> Dict:
        """
        Compare runs side-by-side.
        """
        runs = self.runs
        if run_ids:
            runs = [r for r in runs if r.run_id in run_ids]

        if not runs:
            return {"error": "No runs to compare"}

        comparison = {
            "n_runs": len(runs),
            "best_sharpe": max(runs, key=lambda r: r.sharpe_ratio).__dict__,
            "best_return": max(runs, key=lambda r: r.total_return).__dict__,
            "lowest_drawdown": min(runs, key=lambda r: r.max_drawdown).__dict__,
            "runs": [r.__dict__ for r in runs],
            "summary_table": {
                "strategies": [r.strategy for r in runs],
                "returns": [r.total_return for r in runs],
                "sharpes": [r.sharpe_ratio for r in runs],
                "drawdowns": [r.max_drawdown for r in runs],
                "win_rates": [r.win_rate for r in runs],
            },
        }
        return comparison

    def rank_runs(self, metric: str = "sharpe_ratio") -> List[BacktestSummary]:
        """Rank runs by a given metric."""
        return sorted(self.runs, key=lambda r: getattr(r, metric, 0), reverse=True)


# =============================================================================
# 8.3 — GREEK-AWARE BACKTESTING
# =============================================================================

class GreekAwareBacktester:
    """
    Run backtest with Greek domain signals as additional features.
    Tests whether the 5-domain Greek framework adds alpha.
    """

    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital

    def run(
        self,
        prices: np.ndarray,
        domain_signals: Optional[List[Dict[str, float]]] = None,
        kappa_threshold: float = 5.0,
        entropy_threshold: float = 3.5,
        position_size_pct: float = 2.0,
    ) -> Dict:
        """
        Backtest using Greek domain signals.
        Compare with and without Greek framework.
        """
        n = len(prices)
        returns = np.diff(prices) / prices[:-1]

        # Generate domain signals if not provided
        if domain_signals is None:
            domain_signals = self._generate_synthetic_signals(prices)

        # Strategy 1: Greek framework signals
        greek_positions = np.zeros(n - 1)
        for i in range(min(len(domain_signals), n - 1)):
            sig = domain_signals[i]
            kappa = sig.get("kappa", 5.0)
            entropy = sig.get("entropy", 3.0)

            if kappa > kappa_threshold and entropy < entropy_threshold:
                greek_positions[i] = position_size_pct / 100.0
            elif kappa < kappa_threshold * 0.5 and entropy > entropy_threshold * 1.5:
                greek_positions[i] = -position_size_pct / 100.0
            else:
                greek_positions[i] = 0.0

        greek_returns = greek_positions * returns

        # Strategy 2: Buy-and-hold baseline
        bnh_returns = returns * (position_size_pct / 100.0)

        # Strategy 3: Momentum baseline
        momentum_positions = np.zeros(n - 1)
        lookback = 20
        for i in range(lookback, n - 1):
            mom = np.mean(returns[i - lookback:i])
            momentum_positions[i] = np.sign(mom) * position_size_pct / 100.0
        momentum_returns = momentum_positions * returns

        # Compute metrics
        def compute_metrics(rets: np.ndarray) -> Dict:
            cum = np.cumsum(rets)
            total_ret = float(cum[-1]) if len(cum) > 0 else 0.0
            sr = float(np.mean(rets) / (np.std(rets) + 1e-15) * np.sqrt(252))
            # Max drawdown
            running_max = np.maximum.accumulate(cum)
            dd = running_max - cum
            max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0
            # Win rate
            trades = rets[rets != 0]
            wr = float(np.mean(trades > 0)) if len(trades) > 0 else 0.0
            return {
                "total_return": total_ret, "sharpe_ratio": sr,
                "max_drawdown": max_dd, "win_rate": wr,
                "n_trades": int(np.sum(np.diff(np.sign(rets[rets != 0])) != 0)) if len(trades) > 2 else 0,
            }

        greek_metrics = compute_metrics(greek_returns)
        bnh_metrics = compute_metrics(bnh_returns)
        momentum_metrics = compute_metrics(momentum_returns)

        greek_alpha = greek_metrics["sharpe_ratio"] - bnh_metrics["sharpe_ratio"]

        return {
            "greek_framework": greek_metrics,
            "buy_and_hold": bnh_metrics,
            "momentum_baseline": momentum_metrics,
            "greek_alpha_sharpe": float(greek_alpha),
            "greek_adds_value": bool(greek_alpha > 0),
            "n_bars": n,
        }

    def _generate_synthetic_signals(self, prices: np.ndarray) -> List[Dict[str, float]]:
        """Generate synthetic domain signals for backtesting."""
        returns = np.diff(np.log(prices + 1e-15))
        signals = []
        for i in range(len(returns)):
            # Estimate kappa from recent auto-correlation
            lb = min(i + 1, 30)
            if lb > 5:
                recent = returns[max(0, i - lb):i + 1]
                ac = np.corrcoef(recent[:-1], recent[1:])[0, 1] if len(recent) > 2 else 0.0
                kappa = -np.log(max(abs(ac), 1e-10)) * 252
            else:
                kappa = 5.0

            # Estimate entropy
            if lb > 10:
                hist, _ = np.histogram(recent, bins=10, density=True)
                hist = hist[hist > 0]
                entropy = -np.sum(hist * np.log(hist + 1e-15))
            else:
                entropy = 3.0

            signals.append({"kappa": float(kappa), "entropy": float(entropy)})

        return signals


# =============================================================================
# 8.4 — OPTIONS STRATEGY BACKTESTING
# =============================================================================

class OptionsBacktester:
    """
    Backtest options strategies with historical IV data.
    """

    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital

    def backtest_straddle(
        self,
        prices: np.ndarray,
        ivs: np.ndarray,
        hold_period: int = 20,
        entry_iv_percentile: float = 25,
    ) -> Dict:
        """
        Backtest long straddle strategy:
          - Enter when IV is low (below percentile)
          - Hold for fixed period
          - Profit from realized vol > implied vol
        """
        n = len(prices)
        lookback = 60
        trades = []
        equity = self.initial_capital

        i = lookback
        while i < n - hold_period:
            # IV percentile
            recent_ivs = ivs[max(0, i - lookback):i]
            iv_pctl = np.percentile(recent_ivs, entry_iv_percentile)

            if ivs[i] <= iv_pctl:
                # Enter straddle
                entry_price = prices[i]
                entry_iv = ivs[i]

                # Approximate straddle cost: ~2 * S * σ * √T / √(2π)
                T = hold_period / 252
                straddle_cost = 2 * entry_price * entry_iv * np.sqrt(T) / np.sqrt(2 * np.pi)

                # Exit: realized move vs cost
                exit_price = prices[min(i + hold_period, n - 1)]
                realized_move = abs(exit_price - entry_price)
                pnl = realized_move - straddle_cost

                trades.append({
                    "entry_bar": i,
                    "entry_price": float(entry_price),
                    "entry_iv": float(entry_iv),
                    "exit_price": float(exit_price),
                    "straddle_cost": float(straddle_cost),
                    "realized_move": float(realized_move),
                    "pnl": float(pnl),
                })
                equity += pnl
                i += hold_period  # Skip ahead
            else:
                i += 1

        total_pnl = sum(t["pnl"] for t in trades)
        win_rate = np.mean([t["pnl"] > 0 for t in trades]) if trades else 0.0

        return {
            "strategy": "long_straddle",
            "n_trades": len(trades),
            "total_pnl": float(total_pnl),
            "win_rate": float(win_rate),
            "avg_pnl": float(total_pnl / max(len(trades), 1)),
            "final_equity": float(equity),
            "return_pct": float((equity - self.initial_capital) / self.initial_capital * 100),
            "trades": trades[-10:],  # Last 10 trades
        }

    def backtest_delta_hedge(
        self,
        prices: np.ndarray,
        ivs: np.ndarray,
        rehedge_frequency: int = 1,
    ) -> Dict:
        """
        Backtest delta-hedged option position.
        Profit/loss comes from gamma × (realized vol - implied vol).
        """
        from scipy.stats import norm

        n = len(prices)
        K = prices[0]  # ATM strike
        T_initial = 30 / 252  # 30-day option
        r = 0.05

        total_pnl = 0.0
        rehedge_pnls = []

        for i in range(0, n - 1, rehedge_frequency):
            S = prices[i]
            T = max(T_initial - i / 252, 1 / 252)
            sigma = ivs[i]

            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T) + 1e-15)
            delta = norm.cdf(d1)
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T) + 1e-15)

            # Realized move
            next_i = min(i + rehedge_frequency, n - 1)
            dS = prices[next_i] - S
            realized_var = dS**2

            # Gamma P&L = 0.5 * Γ * (ΔS² - σ²S²Δt)
            implied_var = sigma**2 * S**2 * (rehedge_frequency / 252)
            gamma_pnl = 0.5 * gamma * (realized_var - implied_var)

            total_pnl += gamma_pnl
            rehedge_pnls.append(gamma_pnl)

        return {
            "strategy": "delta_hedge_gamma_scalp",
            "total_pnl": float(total_pnl),
            "n_rehedges": len(rehedge_pnls),
            "avg_gamma_pnl": float(np.mean(rehedge_pnls)) if rehedge_pnls else 0.0,
            "gamma_sharpe": float(np.mean(rehedge_pnls) / (np.std(rehedge_pnls) + 1e-15) * np.sqrt(252 / rehedge_frequency)) if rehedge_pnls else 0.0,
        }


# =============================================================================
# 8.5 — MULTI-ASSET BACKTEST
# =============================================================================

class MultiAssetBacktester:
    """
    Backtest portfolio of instruments simultaneously with rebalancing.
    """

    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital

    def run(
        self,
        asset_prices: Dict[str, np.ndarray],
        target_weights: Dict[str, float],
        rebalance_frequency: int = 20,  # Bars between rebalancing
    ) -> Dict:
        """
        Multi-asset backtest with periodic rebalancing.
        """
        assets = list(asset_prices.keys())
        n_assets = len(assets)
        min_len = min(len(v) for v in asset_prices.values())

        # Align all price series
        prices_matrix = np.column_stack([asset_prices[a][:min_len] for a in assets])
        returns_matrix = np.diff(prices_matrix, axis=0) / prices_matrix[:-1]
        n_bars = returns_matrix.shape[0]

        # Normalize weights
        weights = np.array([target_weights.get(a, 1.0 / n_assets) for a in assets])
        weights = weights / np.sum(weights)

        # Simulate
        portfolio_value = self.initial_capital
        current_weights = weights.copy()
        portfolio_values = [portfolio_value]
        rebalance_count = 0

        for t in range(n_bars):
            # Portfolio return
            port_return = np.sum(current_weights * returns_matrix[t])
            portfolio_value *= (1 + port_return)
            portfolio_values.append(portfolio_value)

            # Drift weights
            current_weights *= (1 + returns_matrix[t])
            current_weights /= np.sum(current_weights) + 1e-15

            # Rebalance
            if (t + 1) % rebalance_frequency == 0:
                current_weights = weights.copy()
                rebalance_count += 1

        portfolio_values = np.array(portfolio_values)
        portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]

        # Metrics
        total_return = (portfolio_values[-1] - self.initial_capital) / self.initial_capital
        sharpe = np.mean(portfolio_returns) / (np.std(portfolio_returns) + 1e-15) * np.sqrt(252)
        running_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (running_max - portfolio_values) / (running_max + 1e-15)
        max_dd = np.max(drawdowns)

        # Per-asset contribution
        asset_contributions = {}
        for i, asset in enumerate(assets):
            asset_ret = np.sum(weights[i] * returns_matrix[:, i])
            asset_contributions[asset] = {
                "weight": float(weights[i]),
                "contribution": float(asset_ret),
                "individual_return": float(np.sum(returns_matrix[:, i])),
            }

        return {
            "total_return": float(total_return),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_dd),
            "final_value": float(portfolio_values[-1]),
            "n_rebalances": rebalance_count,
            "n_bars": n_bars,
            "n_assets": n_assets,
            "asset_contributions": asset_contributions,
            "portfolio_values": portfolio_values[::max(1, len(portfolio_values) // 100)].tolist(),
        }
