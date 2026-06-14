"""
GREEK WALK-FORWARD VALIDATION
==============================

Wires the WalkForwardOptimizer to the AlphaSignalEngine, enabling
out-of-sample validation of the Greek alpha hypothesis:

  "High κ regimes with Low Entropy offer maximal Alpha"

This module:
1. Uses walk-forward windows to test alpha signals out-of-sample
2. Measures if κ/entropy signals actually predict future returns
3. Tracks signal degradation across windows (overfitting detection)
4. Produces a rigorous validation report

This was the remaining gap: "Walk-forward not yet wired to Greek framework"
"""

import numpy as np
import pandas as pd
import logging
import json
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from datetime import datetime, timedelta

from research.core.alpha_signal_engine import AlphaSignalEngine, AlphaSignal
from backtesting.walk_forward import WalkForwardOptimizer, WalkForwardWindow

logger = logging.getLogger(__name__)


@dataclass
class GreekValidationWindow:
    """Results from one walk-forward window with Greek alpha metrics."""
    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    
    # Train metrics
    train_kappa: float = 0.0
    train_entropy: float = 0.0
    train_signal: str = "HOLD"
    train_confidence: float = 0.0
    train_return: float = 0.0
    
    # Test (out-of-sample) metrics
    test_kappa: float = 0.0
    test_entropy: float = 0.0
    test_signal: str = "HOLD"
    test_confidence: float = 0.0
    test_return: float = 0.0
    
    # Validation metrics
    signal_consistent: bool = False  # Did train signal match test signal?
    kappa_stable: bool = False  # Did κ stay in same regime?
    entropy_stable: bool = False  # Did entropy stay in same regime?
    hypothesis_correct: bool = False  # Did high-κ/low-H predict positive return?
    
    # Attribution
    alpha_contribution: float = 0.0  # Return attributable to alpha signal
    noise_contribution: float = 0.0  # Return attributable to random


@dataclass
class GreekValidationReport:
    """Complete validation report across all windows."""
    total_windows: int = 0
    signal_accuracy: float = 0.0  # % of windows where signal was correct
    kappa_stability: float = 0.0  # % of windows where κ regime was stable
    entropy_stability: float = 0.0  # % of windows where entropy was stable
    hypothesis_hit_rate: float = 0.0  # % of windows where hypothesis held
    avg_train_return: float = 0.0
    avg_test_return: float = 0.0
    return_degradation: float = 0.0  # Train-to-test performance loss
    overfitting_score: float = 0.0  # 0=no overfit, 1=complete overfit
    information_ratio: float = 0.0  # OOS IR across windows
    t_statistic: float = 0.0  # t-stat of OOS returns
    p_value: float = 1.0  # p-value of OOS performance
    windows: List[GreekValidationWindow] = field(default_factory=list)


class GreekWalkForwardValidator:
    """
    Walk-forward validation specifically for the Greek Alpha Framework.
    
    Process:
    1. Split data into rolling train/test windows
    2. On each train window: run AlphaSignalEngine → get signal
    3. On each test window: measure if signal predicted correctly
    4. Aggregate: hypothesis hit rate, signal stability, overfitting score
    """

    def __init__(
        self,
        train_days: int = 252,
        test_days: int = 63,
        step_days: int = 21,
        config: Dict = None,
    ):
        self.train_days = train_days
        self.test_days = test_days
        self.step_days = step_days
        self.config = config or {}
        
        self.kappa_threshold = self.config.get("kappa_threshold", 5.0)
        self.entropy_threshold = self.config.get("entropy_threshold", 3.0)

    def validate(
        self,
        prices: pd.Series,
        volumes: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> GreekValidationReport:
        """
        Run complete Greek walk-forward validation.
        
        Args:
            prices: Full price series (at least 1 year)
            volumes: Full volume series
            benchmark_returns: Optional benchmark for excess return calculation
            
        Returns:
            GreekValidationReport with all validation metrics
        """
        logger.info("=" * 60)
        logger.info("GREEK WALK-FORWARD VALIDATION")
        logger.info(f"  Data: {len(prices)} bars")
        logger.info(f"  Train: {self.train_days}d, Test: {self.test_days}d, Step: {self.step_days}d")
        logger.info("=" * 60)

        # Generate windows
        windows = self._generate_windows(prices)
        logger.info(f"  Generated {len(windows)} validation windows")

        if not windows:
            logger.warning("  No valid windows — insufficient data")
            return GreekValidationReport()

        validation_windows: List[GreekValidationWindow] = []

        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            logger.info(f"  Window {i+1}/{len(windows)}: "
                       f"Train [{train_start.date()}→{train_end.date()}] "
                       f"Test [{test_start.date()}→{test_end.date()}]")

            # Extract train/test data
            train_prices = prices[train_start:train_end]
            train_volumes = volumes[train_start:train_end]
            test_prices = prices[test_start:test_end]
            test_volumes = volumes[test_start:test_end]

            if len(train_prices) < 50 or len(test_prices) < 10:
                logger.warning(f"    Skipping — insufficient data")
                continue

            # Run alpha engine on TRAIN data
            train_engine = AlphaSignalEngine(self.config)
            try:
                train_signal = train_engine.generate_signal(
                    train_prices, train_volumes,
                    benchmark_returns=benchmark_returns
                )
            except Exception as e:
                logger.warning(f"    Train signal failed: {e}")
                continue

            # Run alpha engine on TEST data
            test_engine = AlphaSignalEngine(self.config)
            try:
                test_signal = test_engine.generate_signal(
                    test_prices, test_volumes,
                    benchmark_returns=benchmark_returns
                )
            except Exception as e:
                logger.warning(f"    Test signal failed: {e}")
                continue

            # Calculate actual test return based on train signal
            test_returns = test_prices.pct_change().dropna()
            if train_signal.direction == "LONG":
                signal_return = float(test_returns.sum())
            elif train_signal.direction == "SHORT":
                signal_return = float(-test_returns.sum())
            else:
                signal_return = 0.0

            train_return = float(train_prices.pct_change().dropna().sum())

            # Validation checks
            signal_consistent = train_signal.direction == test_signal.direction
            kappa_stable = (
                (train_signal.kappa > self.kappa_threshold) ==
                (test_signal.kappa > self.kappa_threshold)
            )
            entropy_stable = (
                (train_signal.entropy < self.entropy_threshold) ==
                (test_signal.entropy < self.entropy_threshold)
            )

            # Hypothesis check: high κ + low entropy → positive return?
            hypothesis_predicted = (
                train_signal.kappa > self.kappa_threshold and
                train_signal.entropy < self.entropy_threshold
            )
            hypothesis_correct = (
                (hypothesis_predicted and signal_return > 0) or
                (not hypothesis_predicted and train_signal.direction == "HOLD")
            )

            window_result = GreekValidationWindow(
                window_id=i,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_kappa=train_signal.kappa,
                train_entropy=train_signal.entropy,
                train_signal=train_signal.direction,
                train_confidence=train_signal.confidence,
                train_return=train_return,
                test_kappa=test_signal.kappa,
                test_entropy=test_signal.entropy,
                test_signal=test_signal.direction,
                test_confidence=test_signal.confidence,
                test_return=signal_return,
                signal_consistent=signal_consistent,
                kappa_stable=kappa_stable,
                entropy_stable=entropy_stable,
                hypothesis_correct=hypothesis_correct,
                alpha_contribution=signal_return if train_signal.direction != "HOLD" else 0.0,
                noise_contribution=signal_return if train_signal.direction == "HOLD" else 0.0,
            )

            validation_windows.append(window_result)
            logger.info(
                f"    Train: {train_signal.direction}(κ={train_signal.kappa:.2f}, H={train_signal.entropy:.2f}) "
                f"→ Test: ret={signal_return:.4f}, consistent={signal_consistent}, "
                f"hypothesis={hypothesis_correct}"
            )

        # Aggregate results
        report = self._aggregate_results(validation_windows)
        self._log_report(report)
        return report

    def _generate_windows(
        self, prices: pd.Series
    ) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """Generate rolling train/test windows."""
        windows = []
        
        if not hasattr(prices.index, 'date'):
            # Numeric index — convert
            start_idx = 0
            total = len(prices)
            window_size = self.train_days + self.test_days
            
            while start_idx + window_size <= total:
                train_start = prices.index[start_idx]
                train_end_idx = start_idx + self.train_days - 1
                test_start_idx = start_idx + self.train_days
                test_end_idx = min(start_idx + window_size - 1, total - 1)
                
                if isinstance(train_start, (datetime, pd.Timestamp)):
                    windows.append((
                        prices.index[start_idx],
                        prices.index[train_end_idx],
                        prices.index[test_start_idx],
                        prices.index[test_end_idx],
                    ))
                else:
                    # Create datetime proxies
                    base = datetime(2020, 1, 1)
                    windows.append((
                        base + timedelta(days=start_idx),
                        base + timedelta(days=train_end_idx),
                        base + timedelta(days=test_start_idx),
                        base + timedelta(days=test_end_idx),
                    ))
                
                start_idx += self.step_days
        else:
            start_date = prices.index[0]
            end_date = prices.index[-1]
            current = start_date
            
            while current + timedelta(days=self.train_days + self.test_days) <= end_date:
                train_start = current
                train_end = current + timedelta(days=self.train_days)
                test_start = train_end + timedelta(days=1)
                test_end = test_start + timedelta(days=self.test_days)
                
                windows.append((train_start, train_end, test_start, test_end))
                current += timedelta(days=self.step_days)

        return windows

    def _aggregate_results(
        self, windows: List[GreekValidationWindow]
    ) -> GreekValidationReport:
        """Aggregate window results into a report."""
        if not windows:
            return GreekValidationReport()

        n = len(windows)
        
        signal_accuracy = sum(1 for w in windows if w.hypothesis_correct) / n
        kappa_stability = sum(1 for w in windows if w.kappa_stable) / n
        entropy_stability = sum(1 for w in windows if w.entropy_stable) / n
        hypothesis_hit_rate = sum(
            1 for w in windows 
            if w.train_kappa > self.kappa_threshold 
            and w.train_entropy < self.entropy_threshold
            and w.test_return > 0
        ) / max(1, sum(
            1 for w in windows
            if w.train_kappa > self.kappa_threshold
            and w.train_entropy < self.entropy_threshold
        ))

        avg_train = np.mean([w.train_return for w in windows])
        avg_test = np.mean([w.test_return for w in windows])
        
        # Return degradation
        if abs(avg_train) > 1e-10:
            degradation = (avg_train - avg_test) / abs(avg_train)
        else:
            degradation = 0.0

        # Overfitting score (0 = no overfit, 1 = complete overfit)
        test_returns = [w.test_return for w in windows]
        overfitting_score = max(0.0, min(1.0, degradation))

        # OOS t-test
        from scipy import stats as sp_stats
        if len(test_returns) >= 3:
            t_stat, p_value = sp_stats.ttest_1samp(test_returns, 0.0)
        else:
            t_stat, p_value = 0.0, 1.0

        # OOS Information Ratio
        test_arr = np.array(test_returns)
        if np.std(test_arr) > 1e-10:
            ir = np.mean(test_arr) / np.std(test_arr) * np.sqrt(252 / self.test_days)
        else:
            ir = 0.0

        return GreekValidationReport(
            total_windows=n,
            signal_accuracy=signal_accuracy,
            kappa_stability=kappa_stability,
            entropy_stability=entropy_stability,
            hypothesis_hit_rate=hypothesis_hit_rate,
            avg_train_return=avg_train,
            avg_test_return=avg_test,
            return_degradation=degradation,
            overfitting_score=overfitting_score,
            information_ratio=ir,
            t_statistic=t_stat,
            p_value=p_value,
            windows=windows,
        )

    def _log_report(self, report: GreekValidationReport):
        """Log the validation report."""
        logger.info("=" * 60)
        logger.info("GREEK WALK-FORWARD VALIDATION REPORT")
        logger.info("=" * 60)
        logger.info(f"  Windows Analyzed:     {report.total_windows}")
        logger.info(f"  Signal Accuracy:      {report.signal_accuracy:.1%}")
        logger.info(f"  κ Regime Stability:   {report.kappa_stability:.1%}")
        logger.info(f"  Entropy Stability:    {report.entropy_stability:.1%}")
        logger.info(f"  Hypothesis Hit Rate:  {report.hypothesis_hit_rate:.1%}")
        logger.info(f"  Avg Train Return:     {report.avg_train_return:.4f}")
        logger.info(f"  Avg Test Return (OOS):{report.avg_test_return:.4f}")
        logger.info(f"  Return Degradation:   {report.return_degradation:.1%}")
        logger.info(f"  Overfitting Score:    {report.overfitting_score:.2f}")
        logger.info(f"  OOS Info Ratio:       {report.information_ratio:.3f}")
        logger.info(f"  OOS t-statistic:      {report.t_statistic:.3f}")
        logger.info(f"  OOS p-value:          {report.p_value:.4f}")
        logger.info("=" * 60)

    def save_report(self, report: GreekValidationReport, filepath: str):
        """Save validation report to JSON."""
        data = {
            "total_windows": report.total_windows,
            "signal_accuracy": report.signal_accuracy,
            "kappa_stability": report.kappa_stability,
            "entropy_stability": report.entropy_stability,
            "hypothesis_hit_rate": report.hypothesis_hit_rate,
            "avg_train_return": report.avg_train_return,
            "avg_test_return": report.avg_test_return,
            "return_degradation": report.return_degradation,
            "overfitting_score": report.overfitting_score,
            "information_ratio": report.information_ratio,
            "t_statistic": report.t_statistic,
            "p_value": report.p_value,
            "windows": [
                {
                    "id": w.window_id,
                    "train_signal": w.train_signal,
                    "test_return": w.test_return,
                    "train_kappa": w.train_kappa,
                    "test_kappa": w.test_kappa,
                    "hypothesis_correct": w.hypothesis_correct,
                    "signal_consistent": w.signal_consistent,
                }
                for w in report.windows
            ],
        }
        
        content = json.dumps(data, indent=2, default=str)
        checksum = hashlib.sha256(content.encode()).hexdigest()[:16]
        data["checksum"] = checksum
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Report saved to {filepath} (checksum: {checksum})")
