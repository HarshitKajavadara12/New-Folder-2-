"""
Latency Monitor
Real-time latency tracking and alerting for HFT systems

Features:
- Microsecond-precision latency measurement
- Multi-component latency tracking
- Context manager support for easy timing
- Degradation detection
- Performance alerts
"""

import time
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
from contextlib import contextmanager


class LatencyComponent(Enum):
    """Latency measurement points."""
    DATA_FEED = "data_feed"
    SIGNAL_GENERATION = "signal_generation"
    RISK_CHECK = "risk_check"
    ORDER_SUBMISSION = "order_submission"
    VENUE_ROUTING = "venue_routing"
    FILL_PROCESSING = "fill_processing"
    TOTAL_LOOP = "total_loop"


@dataclass
class LatencyMeasurement:
    """Single latency measurement."""
    component: LatencyComponent
    latency_us: float
    timestamp: float
    metadata: Dict = field(default_factory=dict)


@dataclass
class LatencyStats:
    """Aggregated latency statistics."""
    component: LatencyComponent
    count: int
    mean_us: float
    median_us: float
    p95_us: float
    p99_us: float
    max_us: float
    std_us: float


class LatencyMonitor:
    """
    Real-time latency monitoring system.
    
    Tracks latency across all system components and detects
    performance degradation that could impact HFT profitability.
    """
    
    def __init__(
        self,
        window_size: int = 10000,
        alert_threshold_us: Dict[LatencyComponent, float] = None
    ):
        """
        Initialize latency monitor.
        
        Args:
            window_size: Number of recent measurements to track
            alert_threshold_us: Alert thresholds by component (microseconds)
        """
        self.window_size = window_size
        
        # Default alert thresholds
        self.alert_threshold_us = alert_threshold_us or {
            LatencyComponent.DATA_FEED: 1000,        # 1ms
            LatencyComponent.SIGNAL_GENERATION: 500,  # 500μs
            LatencyComponent.RISK_CHECK: 100,         # 100μs
            LatencyComponent.ORDER_SUBMISSION: 200,   # 200μs
            LatencyComponent.VENUE_ROUTING: 500,      # 500μs
            LatencyComponent.FILL_PROCESSING: 300,    # 300μs
            LatencyComponent.TOTAL_LOOP: 5000,        # 5ms
        }
        
        # Storage for measurements
        self.measurements: Dict[LatencyComponent, deque] = {
            component: deque(maxlen=window_size)
            for component in LatencyComponent
        }
        
        # Active timers (for in-progress measurements)
        self.active_timers: Dict[str, float] = {}
        
        # Alert tracking
        self.alerts: List[Dict] = []
        self.alert_counts: Dict[LatencyComponent, int] = {
            component: 0 for component in LatencyComponent
        }
        
    def start_timer(self, timer_id: str) -> None:
        """
        Start latency timer.
        
        Args:
            timer_id: Unique identifier for this measurement
        """
        self.active_timers[timer_id] = time.perf_counter()
    
    def stop_timer(
        self,
        timer_id: str,
        component: LatencyComponent,
        metadata: Optional[Dict] = None
    ) -> float:
        """
        Stop timer and record measurement.
        
        Args:
            timer_id: Timer identifier from start_timer
            component: System component being measured
            metadata: Optional metadata (order size, symbol, etc.)
            
        Returns:
            Measured latency in microseconds
        """
        if timer_id not in self.active_timers:
            raise ValueError(f"Timer '{timer_id}' not found")
        
        # Calculate latency
        start_time = self.active_timers.pop(timer_id)
        latency_s = time.perf_counter() - start_time
        latency_us = latency_s * 1e6
        
        # Record measurement
        measurement = LatencyMeasurement(
            component=component,
            latency_us=latency_us,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        
        self.measurements[component].append(measurement)
        
        # Check for alerts
        self._check_alert(component, latency_us)
        
        return latency_us
    
    @contextmanager
    def track(self, component: LatencyComponent, metadata: Optional[Dict] = None):
        """
        Context manager for latency tracking.
        
        Usage:
            with monitor.track(LatencyComponent.RISK_CHECK):
                perform_risk_check()
        
        Args:
            component: System component being measured
            metadata: Optional metadata
            
        Yields:
            None (latency is recorded on exit)
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            latency_us = (time.perf_counter() - start) * 1e6
            self.record_latency(component, latency_us, metadata)
    
    def record_latency(
        self,
        component: LatencyComponent,
        latency_us: float,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Directly record latency measurement.
        
        Args:
            component: System component
            latency_us: Measured latency (microseconds)
            metadata: Optional metadata
        """
        measurement = LatencyMeasurement(
            component=component,
            latency_us=latency_us,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        
        self.measurements[component].append(measurement)
        self._check_alert(component, latency_us)
    
    def _check_alert(self, component: LatencyComponent, latency_us: float) -> None:
        """Check if latency exceeds threshold and generate alert."""
        threshold = self.alert_threshold_us.get(component)
        
        if threshold and latency_us > threshold:
            alert = {
                'timestamp': time.time(),
                'component': component.value,
                'latency_us': latency_us,
                'threshold_us': threshold,
                'severity': 'HIGH' if latency_us > threshold * 2 else 'MEDIUM'
            }
            
            self.alerts.append(alert)
            self.alert_counts[component] += 1
    
    def get_stats(
        self,
        component: Optional[LatencyComponent] = None
    ) -> Dict[LatencyComponent, LatencyStats]:
        """
        Get latency statistics.
        
        Args:
            component: Specific component (None for all)
            
        Returns:
            Dictionary of latency statistics
        """
        components = [component] if component else list(LatencyComponent)
        stats = {}
        
        for comp in components:
            measurements = self.measurements[comp]
            
            if not measurements:
                continue
            
            latencies = [m.latency_us for m in measurements]
            
            stats[comp] = LatencyStats(
                component=comp,
                count=len(latencies),
                mean_us=np.mean(latencies),
                median_us=np.median(latencies),
                p95_us=np.percentile(latencies, 95),
                p99_us=np.percentile(latencies, 99),
                max_us=np.max(latencies),
                std_us=np.std(latencies)
            )
        
        return stats
    
    def get_recent_alerts(self, n: int = 10) -> List[Dict]:
        """Get N most recent alerts."""
        return self.alerts[-n:]
    
    def get_alert_summary(self) -> Dict:
        """Get summary of alerts by component."""
        return {
            component.value: count
            for component, count in self.alert_counts.items()
            if count > 0
        }
    
    def detect_degradation(
        self,
        component: LatencyComponent,
        lookback_recent: int = 100,
        lookback_baseline: int = 1000
    ) -> Dict:
        """
        Detect latency degradation by comparing recent vs baseline.
        
        Args:
            component: Component to check
            lookback_recent: Recent measurements to analyze
            lookback_baseline: Baseline measurements for comparison
            
        Returns:
            Degradation analysis results
        """
        measurements = list(self.measurements[component])
        
        if len(measurements) < lookback_baseline:
            return {'degradation_detected': False, 'reason': 'insufficient_data'}
        
        # Recent performance
        recent = [m.latency_us for m in measurements[-lookback_recent:]]
        recent_mean = np.mean(recent)
        recent_p95 = np.percentile(recent, 95)
        
        # Baseline performance
        baseline = [m.latency_us for m in measurements[-lookback_baseline:-lookback_recent]]
        baseline_mean = np.mean(baseline)
        baseline_p95 = np.percentile(baseline, 95)
        
        # Degradation metrics
        mean_increase_pct = ((recent_mean - baseline_mean) / baseline_mean) * 100
        p95_increase_pct = ((recent_p95 - baseline_p95) / baseline_p95) * 100
        
        # Detection thresholds
        degradation_detected = (
            mean_increase_pct > 20 or  # 20% increase in mean
            p95_increase_pct > 30      # 30% increase in p95
        )
        
        return {
            'degradation_detected': degradation_detected,
            'component': component.value,
            'recent_mean_us': recent_mean,
            'baseline_mean_us': baseline_mean,
            'mean_increase_pct': mean_increase_pct,
            'recent_p95_us': recent_p95,
            'baseline_p95_us': baseline_p95,
            'p95_increase_pct': p95_increase_pct
        }
    
    def reset(self) -> None:
        """Reset all measurements and alerts."""
        for component in LatencyComponent:
            self.measurements[component].clear()
            self.alert_counts[component] = 0
        
        self.alerts.clear()
        self.active_timers.clear()


# Demo
if __name__ == "__main__":
    print("=" * 70)
    print("LATENCY MONITOR DEMO")
    print("=" * 70)
    
    # Initialize monitor
    monitor = LatencyMonitor(window_size=1000)
    
    print("\nSimulating HFT system latencies...\n")
    
    # Simulate 500 trading loops
    np.random.seed(42)
    
    for i in range(500):
        # Simulate each component with realistic latencies
        
        # Data feed (varies more)
        data_latency = np.random.gamma(2, 300)
        monitor.record_latency(LatencyComponent.DATA_FEED, data_latency)
        
        # Signal generation (fast)
        signal_latency = np.random.gamma(2, 150)
        monitor.record_latency(LatencyComponent.SIGNAL_GENERATION, signal_latency)
        
        # Risk check (very fast, target <100μs)
        risk_latency = np.random.gamma(2, 35)
        monitor.record_latency(LatencyComponent.RISK_CHECK, risk_latency)
        
        # Order submission
        order_latency = np.random.gamma(2, 80)
        monitor.record_latency(LatencyComponent.ORDER_SUBMISSION, order_latency)
        
        # Venue routing
        venue_latency = np.random.gamma(2, 200)
        monitor.record_latency(LatencyComponent.VENUE_ROUTING, venue_latency)
        
        # Total loop
        total_latency = (
            data_latency + signal_latency + risk_latency +
            order_latency + venue_latency
        )
        monitor.record_latency(LatencyComponent.TOTAL_LOOP, total_latency)
        
        # Introduce some degradation after iteration 300
        if i > 300:
            # Simulate degradation (network issue, etc.)
            extra_latency = np.random.gamma(2, 500) if np.random.random() < 0.1 else 0
            monitor.record_latency(
                LatencyComponent.DATA_FEED,
                data_latency + extra_latency
            )
    
    # Display statistics
    print("=" * 70)
    print("LATENCY STATISTICS")
    print("=" * 70)
    
    stats = monitor.get_stats()
    
    print(f"\n{'Component':<25s} {'Mean (μs)':<12s} {'P95 (μs)':<12s} {'P99 (μs)':<12s} {'Max (μs)':<12s}")
    print("-" * 70)
    
    for component, stat in stats.items():
        print(
            f"{component.value:<25s} "
            f"{stat.mean_us:<12.1f} "
            f"{stat.p95_us:<12.1f} "
            f"{stat.p99_us:<12.1f} "
            f"{stat.max_us:<12.1f}"
        )
    
    # Check for degradation
    print("\n" + "=" * 70)
    print("DEGRADATION DETECTION")
    print("=" * 70)
    
    for component in [LatencyComponent.DATA_FEED, LatencyComponent.TOTAL_LOOP]:
        degradation = monitor.detect_degradation(component)
        
        print(f"\n{component.value}:")
        print(f"  Degradation detected: {degradation['degradation_detected']}")
        if degradation.get('mean_increase_pct'):
            print(f"  Mean increase: {degradation['mean_increase_pct']:.1f}%")
            print(f"  P95 increase: {degradation['p95_increase_pct']:.1f}%")
    
    # Alert summary
    print("\n" + "=" * 70)
    print("ALERT SUMMARY")
    print("=" * 70)
    
    alert_summary = monitor.get_alert_summary()
    if alert_summary:
        for component, count in alert_summary.items():
            threshold = monitor.alert_threshold_us[LatencyComponent(component)]
            print(f"  {component}: {count} alerts (threshold: {threshold:.0f} μs)")
    else:
        print("  No alerts generated  ")
    
    print("\n  Latency monitoring operational!")
