"""
Backtest Results Storage
========================
Stores and retrieves backtest results as validated artifacts.
Fixes Error #7: No backtested results stored.
"""

import json
import os
import hashlib
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts", "backtest_results")


@dataclass
class BacktestResult:
    """A single backtest run result."""
    run_id: str
    timestamp: str
    strategy_name: str
    symbol: str
    start_date: str
    end_date: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profit_factor: float
    avg_trade_return: float
    calmar_ratio: float
    sortino_ratio: float
    parameters: Dict
    regime: str = "UNKNOWN"
    alpha_signal: str = "NONE"
    kappa_at_entry: float = 0.0
    entropy_at_entry: float = 0.0
    validated: bool = False


class BacktestResultStore:
    """Persists backtest results as JSON artifacts."""

    def __init__(self, results_dir: str = None):
        self.results_dir = results_dir or RESULTS_DIR
        os.makedirs(self.results_dir, exist_ok=True)
        logger.info(f"BacktestResultStore initialized at {self.results_dir}")

    def save(self, result: BacktestResult) -> str:
        """Save a backtest result. Returns the file path."""
        filename = f"bt_{result.strategy_name}_{result.run_id}.json"
        filepath = os.path.join(self.results_dir, filename)

        data = asdict(result)
        data["_checksum"] = self._checksum(data)

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Saved backtest result: {filepath}")
        return filepath

    def load(self, run_id: str) -> Optional[BacktestResult]:
        """Load a specific backtest result by run_id."""
        for filename in os.listdir(self.results_dir):
            if run_id in filename and filename.endswith(".json"):
                filepath = os.path.join(self.results_dir, filename)
                with open(filepath, "r") as f:
                    data = json.load(f)
                data.pop("_checksum", None)
                return BacktestResult(**data)
        return None

    def list_results(self) -> List[Dict]:
        """List all stored backtest results (summary)."""
        results = []
        if not os.path.exists(self.results_dir):
            return results

        for filename in sorted(os.listdir(self.results_dir)):
            if filename.endswith(".json"):
                filepath = os.path.join(self.results_dir, filename)
                try:
                    with open(filepath, "r") as f:
                        data = json.load(f)
                    results.append({
                        "run_id": data.get("run_id"),
                        "strategy": data.get("strategy_name"),
                        "sharpe": data.get("sharpe_ratio"),
                        "return": data.get("total_return"),
                        "drawdown": data.get("max_drawdown"),
                        "timestamp": data.get("timestamp"),
                    })
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Failed to read {filename}: {e}")

        return results

    def get_best_result(self, metric: str = "sharpe_ratio") -> Optional[BacktestResult]:
        """Get the best backtest result by a given metric."""
        best = None
        best_score = float("-inf")

        for filename in os.listdir(self.results_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self.results_dir, filename)
                try:
                    with open(filepath, "r") as f:
                        data = json.load(f)
                    score = data.get(metric, float("-inf"))
                    if score > best_score:
                        best_score = score
                        data.pop("_checksum", None)
                        best = BacktestResult(**data)
                except Exception:
                    continue

        return best

    @staticmethod
    def _checksum(data: Dict) -> str:
        """Generate integrity checksum for result."""
        content = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
