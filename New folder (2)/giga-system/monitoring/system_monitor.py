"""
MONITORING & ALERTING — System Health, Performance Tracking, Alerts
====================================================================

Addresses Missing Concept 7.3: No Prometheus, no Grafana, no alerting.
Provides a lightweight monitoring system with metrics collection and alerting.
"""

import numpy as np
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """Single metric data point."""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class Alert:
    """Alert triggered by monitoring."""
    level: str  # "info", "warning", "critical"
    metric: str
    message: str
    value: float
    threshold: float
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False


class MetricsCollector:
    """
    Lightweight Prometheus-style metrics collector.
    Supports counters, gauges, and histograms.
    """

    def __init__(self, max_points: int = 10000):
        self.counters: Dict[str, float] = {}
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = {}
        self.time_series: Dict[str, deque] = {}
        self.max_points = max_points

    def counter_inc(self, name: str, value: float = 1.0, labels: Optional[Dict] = None):
        """Increment a counter."""
        key = self._make_key(name, labels)
        self.counters[key] = self.counters.get(key, 0) + value
        self._record(name, self.counters[key], labels)

    def gauge_set(self, name: str, value: float, labels: Optional[Dict] = None):
        """Set a gauge value."""
        key = self._make_key(name, labels)
        self.gauges[key] = value
        self._record(name, value, labels)

    def histogram_observe(self, name: str, value: float, labels: Optional[Dict] = None):
        """Observe a value for histogram."""
        key = self._make_key(name, labels)
        if key not in self.histograms:
            self.histograms[key] = []
        self.histograms[key].append(value)
        # Keep last N observations
        if len(self.histograms[key]) > self.max_points:
            self.histograms[key] = self.histograms[key][-self.max_points:]
        self._record(name, value, labels)

    def _record(self, name: str, value: float, labels: Optional[Dict] = None):
        """Record to time series."""
        if name not in self.time_series:
            self.time_series[name] = deque(maxlen=self.max_points)
        self.time_series[name].append(MetricPoint(name, value, datetime.now(), labels or {}))

    def get_stats(self, name: str) -> Dict:
        """Get statistics for a metric."""
        key = name
        if key in self.histograms:
            vals = self.histograms[key]
            return {
                "count": len(vals),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "p50": float(np.percentile(vals, 50)),
                "p95": float(np.percentile(vals, 95)),
                "p99": float(np.percentile(vals, 99)),
            }
        elif key in self.gauges:
            return {"current": self.gauges[key]}
        elif key in self.counters:
            return {"total": self.counters[key]}
        return {}

    @staticmethod
    def _make_key(name: str, labels: Optional[Dict] = None) -> str:
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def export_prometheus_format(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []
        for name, val in self.counters.items():
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {val}")
        for name, val in self.gauges.items():
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {val}")
        for name, vals in self.histograms.items():
            lines.append(f"# TYPE {name} histogram")
            lines.append(f"{name}_count {len(vals)}")
            lines.append(f"{name}_sum {sum(vals)}")
        return "\n".join(lines)


class AlertManager:
    """
    Alert system for monitoring thresholds.
    """

    def __init__(self):
        self.rules: List[Dict] = []
        self.alerts: List[Alert] = []
        self.callbacks: List[Callable] = []

    def add_rule(self, metric: str, operator: str, threshold: float,
                 level: str = "warning", message: str = ""):
        """Add an alerting rule."""
        self.rules.append({
            "metric": metric,
            "operator": operator,  # ">", "<", ">=", "<=", "=="
            "threshold": threshold,
            "level": level,
            "message": message or f"{metric} {operator} {threshold}",
        })

    def add_callback(self, fn: Callable):
        """Add a callback for when alerts fire."""
        self.callbacks.append(fn)

    def check(self, metrics: Dict[str, float]) -> List[Alert]:
        """Check all rules against current metrics."""
        fired = []
        for rule in self.rules:
            metric_name = rule["metric"]
            if metric_name not in metrics:
                continue

            value = metrics[metric_name]
            threshold = rule["threshold"]
            triggered = False

            if rule["operator"] == ">" and value > threshold:
                triggered = True
            elif rule["operator"] == "<" and value < threshold:
                triggered = True
            elif rule["operator"] == ">=" and value >= threshold:
                triggered = True
            elif rule["operator"] == "<=" and value <= threshold:
                triggered = True
            elif rule["operator"] == "==" and value == threshold:
                triggered = True

            if triggered:
                alert = Alert(
                    level=rule["level"], metric=metric_name,
                    message=rule["message"], value=value,
                    threshold=threshold,
                )
                fired.append(alert)
                self.alerts.append(alert)
                logger.warning(f"ALERT [{alert.level}]: {alert.message} (value={value})")

                for cb in self.callbacks:
                    try:
                        cb(alert)
                    except Exception:
                        pass

        return fired

    def get_unacknowledged(self) -> List[Alert]:
        return [a for a in self.alerts if not a.acknowledged]

    def acknowledge_all(self):
        for a in self.alerts:
            a.acknowledged = True


class SystemMonitor:
    """
    Full system monitoring combining metrics + alerts.
    """

    def __init__(self):
        self.metrics = MetricsCollector()
        self.alerts = AlertManager()
        self._setup_default_rules()

    def _setup_default_rules(self):
        """Setup default alerting rules."""
        self.alerts.add_rule("drawdown_pct", ">", 10.0, "warning", "Drawdown exceeds 10%")
        self.alerts.add_rule("drawdown_pct", ">", 20.0, "critical", "Drawdown exceeds 20%!")
        self.alerts.add_rule("sharpe_ratio", "<", 0.5, "warning", "Sharpe ratio below 0.5")
        self.alerts.add_rule("latency_ms", ">", 100, "warning", "Execution latency > 100ms")
        self.alerts.add_rule("error_rate", ">", 0.05, "critical", "Error rate above 5%")
        self.alerts.add_rule("position_count", ">", 20, "warning", "Too many open positions")

    def record_trade(self, pnl: float, latency_ms: float, strategy: str = ""):
        """Record a trade execution."""
        self.metrics.counter_inc("trades_total", labels={"strategy": strategy})
        self.metrics.histogram_observe("trade_pnl", pnl)
        self.metrics.histogram_observe("latency_ms", latency_ms)
        if pnl > 0:
            self.metrics.counter_inc("winning_trades")
        else:
            self.metrics.counter_inc("losing_trades")

    def record_portfolio_state(self, equity: float, drawdown_pct: float,
                                 n_positions: int, sharpe: float):
        """Record portfolio state."""
        self.metrics.gauge_set("equity", equity)
        self.metrics.gauge_set("drawdown_pct", drawdown_pct)
        self.metrics.gauge_set("position_count", n_positions)
        self.metrics.gauge_set("sharpe_ratio", sharpe)

        # Check alerts
        current = {
            "drawdown_pct": drawdown_pct,
            "sharpe_ratio": sharpe,
            "position_count": n_positions,
        }
        self.alerts.check(current)

    def get_dashboard_data(self) -> Dict:
        """Get data for monitoring dashboard."""
        return {
            "counters": dict(self.metrics.counters),
            "gauges": dict(self.metrics.gauges),
            "histogram_stats": {
                name: self.metrics.get_stats(name)
                for name in self.metrics.histograms
            },
            "active_alerts": len(self.alerts.get_unacknowledged()),
            "total_alerts": len(self.alerts.alerts),
        }
