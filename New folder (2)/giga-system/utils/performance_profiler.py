"""
GIGA SYSTEM - Performance Profiler
Greek Intelligence for Global Analysis

High-precision performance profiling tools for quantitative finance operations.
Measures execution time, memory usage, and algorithmic efficiency to ensure
sub-millisecond response times for critical calculations.

Key Features:
- Function execution timing with nanosecond precision
- Memory usage tracking with detailed analysis
- Bottleneck identification for optimization
- Performance regression testing
- Comparative benchmarking
- Production-safe profiling with minimal overhead

Performance Targets:
- Profiling overhead: <0.001ms
- Memory tracking precision: <1MB accuracy
- Real-time performance monitoring
"""

import time
import functools
import threading
from typing import Dict, List, Optional, Callable, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
import warnings
import json

try:
    import psutil
    import py_spy
    ADVANCED_PROFILING = True
except ImportError:
    ADVANCED_PROFILING = False

try:
    import memory_profiler
    MEMORY_PROFILING = True
except ImportError:
    MEMORY_PROFILING = False

import numpy as np


@dataclass
class PerformanceMetrics:
    """Container for performance measurement results."""
    
    function_name: str
    execution_time_ms: float
    memory_usage_mb: Optional[float] = None
    cpu_percent: Optional[float] = None
    call_count: int = 1
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Statistical metrics for multiple runs
    min_time_ms: Optional[float] = None
    max_time_ms: Optional[float] = None
    mean_time_ms: Optional[float] = None
    std_time_ms: Optional[float] = None
    
    # Additional context
    parameters: Optional[Dict[str, Any]] = None
    return_value_size: Optional[int] = None
    error_occurred: bool = False
    error_message: Optional[str] = None
    
    def update_stats(self, new_time_ms: float):
        """Update statistical metrics with new measurement."""
        if self.min_time_ms is None:
            self.min_time_ms = new_time_ms
            self.max_time_ms = new_time_ms
            self.mean_time_ms = new_time_ms
            self.std_time_ms = 0.0
        else:
            self.min_time_ms = min(self.min_time_ms, new_time_ms)
            self.max_time_ms = max(self.max_time_ms, new_time_ms)
            
            # Update running statistics
            old_mean = self.mean_time_ms
            self.call_count += 1
            self.mean_time_ms = old_mean + (new_time_ms - old_mean) / self.call_count
            
            # Update standard deviation (Welford's algorithm)
            if self.call_count > 1:
                self.std_time_ms = np.sqrt(
                    ((self.call_count - 2) * (self.std_time_ms ** 2) + 
                     (new_time_ms - old_mean) * (new_time_ms - self.mean_time_ms)) / 
                    (self.call_count - 1)
                )


class PerformanceProfiler:
    """
    High-precision performance profiler for quantitative finance operations.
    
    Provides detailed timing, memory, and CPU usage analysis with
    minimal overhead for production use.
    """
    
    def __init__(self, 
                 enable_memory_tracking: bool = True,
                 enable_cpu_tracking: bool = True,
                 history_size: int = 1000):
        """
        Initialize performance profiler.
        
        Args:
            enable_memory_tracking: Track memory usage
            enable_cpu_tracking: Track CPU usage
            history_size: Number of measurements to keep in history
        """
        self.enable_memory_tracking = enable_memory_tracking and MEMORY_PROFILING
        self.enable_cpu_tracking = enable_cpu_tracking and ADVANCED_PROFILING
        self.history_size = history_size
        
        # Performance metrics storage
        self.metrics_history: Dict[str, List[PerformanceMetrics]] = {}
        self.active_measurements: Dict[int, Tuple[str, float, float]] = {}
        
        # Threading support
        self.lock = threading.Lock()
        
        # System baseline for relative measurements
        self._establish_baseline()
    
    def _establish_baseline(self):
        """Establish system performance baseline."""
        if self.enable_cpu_tracking:
            self.baseline_cpu = psutil.cpu_percent(interval=0.1)
        else:
            self.baseline_cpu = 0.0
        
        if self.enable_memory_tracking:
            process = psutil.Process()
            self.baseline_memory_mb = process.memory_info().rss / 1024 / 1024
        else:
            self.baseline_memory_mb = 0.0
    
    def start_measurement(self, function_name: str) -> int:
        """
        Start performance measurement for a function.
        
        Args:
            function_name: Name of function being measured
            
        Returns:
            Measurement ID for stopping measurement
        """
        measurement_id = threading.get_ident()
        start_time = time.perf_counter()
        
        # Get initial memory if tracking enabled
        if self.enable_memory_tracking:
            try:
                process = psutil.Process()
                start_memory = process.memory_info().rss / 1024 / 1024
            except:
                start_memory = 0.0
        else:
            start_memory = 0.0
        
        with self.lock:
            self.active_measurements[measurement_id] = (
                function_name, start_time, start_memory
            )
        
        return measurement_id
    
    def stop_measurement(self, 
                        measurement_id: int,
                        parameters: Optional[Dict[str, Any]] = None,
                        return_value: Any = None) -> PerformanceMetrics:
        """
        Stop performance measurement and calculate metrics.
        
        Args:
            measurement_id: ID returned by start_measurement
            parameters: Function parameters for context
            return_value: Function return value for size analysis
            
        Returns:
            PerformanceMetrics with detailed timing and resource usage
        """
        end_time = time.perf_counter()
        
        # Get final memory if tracking enabled
        if self.enable_memory_tracking:
            try:
                process = psutil.Process()
                end_memory = process.memory_info().rss / 1024 / 1024
            except:
                end_memory = 0.0
        else:
            end_memory = 0.0
        
        # Get CPU usage if tracking enabled
        if self.enable_cpu_tracking:
            try:
                cpu_percent = psutil.cpu_percent(interval=None)
            except:
                cpu_percent = 0.0
        else:
            cpu_percent = 0.0
        
        with self.lock:
            if measurement_id not in self.active_measurements:
                warnings.warn(f"Measurement ID {measurement_id} not found")
                return PerformanceMetrics("unknown", 0.0)
            
            function_name, start_time, start_memory = self.active_measurements.pop(measurement_id)
        
        # Calculate metrics
        execution_time_ms = (end_time - start_time) * 1000
        memory_delta_mb = end_memory - start_memory if self.enable_memory_tracking else None
        
        # Analyze return value size
        return_size = None
        if return_value is not None:
            try:
                if hasattr(return_value, '__len__'):
                    return_size = len(return_value)
                elif hasattr(return_value, 'nbytes'):  # NumPy arrays
                    return_size = return_value.nbytes
            except:
                pass
        
        # Create metrics object
        metrics = PerformanceMetrics(
            function_name=function_name,
            execution_time_ms=execution_time_ms,
            memory_usage_mb=memory_delta_mb,
            cpu_percent=cpu_percent,
            parameters=parameters,
            return_value_size=return_size
        )
        
        # Store in history
        self._add_to_history(metrics)
        
        return metrics
    
    def _add_to_history(self, metrics: PerformanceMetrics):
        """Add metrics to historical data with size limits."""
        function_name = metrics.function_name
        
        if function_name not in self.metrics_history:
            self.metrics_history[function_name] = []
        
        history = self.metrics_history[function_name]
        history.append(metrics)
        
        # Maintain history size limit
        if len(history) > self.history_size:
            history.pop(0)
    
    def get_function_stats(self, function_name: str) -> Optional[Dict[str, Any]]:
        """
        Get statistical summary for a specific function.
        
        Args:
            function_name: Name of function to analyze
            
        Returns:
            Dictionary with statistical metrics
        """
        if function_name not in self.metrics_history:
            return None
        
        history = self.metrics_history[function_name]
        if not history:
            return None
        
        execution_times = [m.execution_time_ms for m in history]
        memory_usages = [m.memory_usage_mb for m in history if m.memory_usage_mb is not None]
        
        stats = {
            'call_count': len(history),
            'mean_time_ms': np.mean(execution_times),
            'median_time_ms': np.median(execution_times),
            'min_time_ms': np.min(execution_times),
            'max_time_ms': np.max(execution_times),
            'std_time_ms': np.std(execution_times),
            'p95_time_ms': np.percentile(execution_times, 95),
            'p99_time_ms': np.percentile(execution_times, 99)
        }
        
        if memory_usages:
            stats.update({
                'mean_memory_mb': np.mean(memory_usages),
                'max_memory_mb': np.max(memory_usages),
                'total_memory_mb': np.sum([m for m in memory_usages if m > 0])
            })
        
        return stats
    
    def get_top_functions(self, 
                         metric: str = 'mean_time_ms', 
                         limit: int = 10) -> List[Tuple[str, float]]:
        """
        Get top functions by specified metric.
        
        Args:
            metric: Metric to sort by ('mean_time_ms', 'call_count', etc.)
            limit: Number of functions to return
            
        Returns:
            List of (function_name, metric_value) tuples
        """
        function_metrics = []
        
        for function_name in self.metrics_history:
            stats = self.get_function_stats(function_name)
            if stats and metric in stats:
                function_metrics.append((function_name, stats[metric]))
        
        # Sort by metric value (descending)
        function_metrics.sort(key=lambda x: x[1], reverse=True)
        
        return function_metrics[:limit]
    
    def clear_history(self, function_name: Optional[str] = None):
        """
        Clear performance history.
        
        Args:
            function_name: Specific function to clear (None for all)
        """
        with self.lock:
            if function_name:
                self.metrics_history.pop(function_name, None)
            else:
                self.metrics_history.clear()
    
    def export_metrics(self, filepath: Optional[Path] = None) -> str:
        """
        Export performance metrics to JSON format.
        
        Args:
            filepath: Output file path (None for string return)
            
        Returns:
            JSON string of metrics data
        """
        export_data = {}
        
        for function_name, history in self.metrics_history.items():
            stats = self.get_function_stats(function_name)
            export_data[function_name] = {
                'statistics': stats,
                'recent_calls': [
                    {
                        'timestamp': m.timestamp.isoformat(),
                        'execution_time_ms': m.execution_time_ms,
                        'memory_usage_mb': m.memory_usage_mb,
                        'parameters': str(m.parameters) if m.parameters else None
                    }
                    for m in history[-10:]  # Last 10 calls
                ]
            }
        
        json_data = json.dumps(export_data, indent=2)
        
        if filepath:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w') as f:
                f.write(json_data)
        
        return json_data


# Global profiler instance (lazy initialized)
_global_profiler: Optional[PerformanceProfiler] = None

def get_profiler() -> PerformanceProfiler:
    """Get or create the global profiler instance (Lazy Initialization)."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler()
    return _global_profiler


def profile_function(func: Optional[Callable] = None, 
                    include_params: bool = False,
                    include_return: bool = False):
    """
    Decorator for automatic function profiling.
    
    Args:
        func: Function to profile (when used as @profile_function)
        include_params: Include function parameters in metrics
        include_return: Include return value analysis
        
    Returns:
        Decorated function or decorator
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            function_name = f"{f.__module__}.{f.__name__}"
            profiler = get_profiler()
            
            # Prepare parameters for profiling
            params = None
            if include_params:
                params = {
                    'args_count': len(args),
                    'kwargs_keys': list(kwargs.keys()),
                    'args_types': [type(arg).__name__ for arg in args[:3]]  # First 3 args only
                }
            
            # Start measurement
            measurement_id = profiler.start_measurement(function_name)
            
            try:
                # Execute function
                result = f(*args, **kwargs)
                
                # Stop measurement
                metrics = profiler.stop_measurement(
                    measurement_id, 
                    parameters=params,
                    return_value=result if include_return else None
                )
                
                return result
                
            except Exception as e:
                # Record error and re-raise
                metrics = profiler.stop_measurement(measurement_id, parameters=params)
                metrics.error_occurred = True
                metrics.error_message = str(e)
                raise
        
        return wrapper
    
    # Handle both @profile_function and @profile_function() syntax
    if func is None:
        return decorator
    else:
        return decorator(func)


def benchmark_function(func: Callable, 
                      iterations: int = 100,
                      warmup_iterations: int = 10,
                      *args, **kwargs) -> Dict[str, Any]:
    """
    Benchmark a function with multiple iterations for statistical analysis.
    
    Args:
        func: Function to benchmark
        iterations: Number of benchmark iterations
        warmup_iterations: Number of warmup iterations (not counted)
        *args, **kwargs: Arguments to pass to function
        
    Returns:
        Dictionary with detailed benchmark results
    """
    function_name = f"{func.__module__}.{func.__name__}"
    
    # Warmup iterations
    for _ in range(warmup_iterations):
        try:
            func(*args, **kwargs)
        except:
            pass  # Ignore warmup errors
    
    # Benchmark iterations
    execution_times = []
    memory_usages = []
    errors = 0
    profiler = get_profiler()
    
    for i in range(iterations):
        measurement_id = profiler.start_measurement(f"{function_name}_benchmark")
        
        try:
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            
            execution_time_ms = (end_time - start_time) * 1000
            execution_times.append(execution_time_ms)
            
            # Get memory usage if available
            if profiler.enable_memory_tracking:
                try:
                    process = psutil.Process()
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    memory_usages.append(memory_mb)
                except:
                    pass
            
        except Exception as e:
            errors += 1
            warnings.warn(f"Benchmark iteration {i} failed: {e}")
        finally:
            profiler.stop_measurement(measurement_id)
    
    if not execution_times:
        return {"error": "All benchmark iterations failed"}
    
    # Calculate statistics
    times_array = np.array(execution_times)
    
    benchmark_results = {
        'function_name': function_name,
        'iterations': len(execution_times),
        'warmup_iterations': warmup_iterations,
        'errors': errors,
        'execution_times': {
            'mean_ms': np.mean(times_array),
            'median_ms': np.median(times_array),
            'min_ms': np.min(times_array),
            'max_ms': np.max(times_array),
            'std_ms': np.std(times_array),
            'p95_ms': np.percentile(times_array, 95),
            'p99_ms': np.percentile(times_array, 99),
            'operations_per_second': 1000 / np.mean(times_array)
        }
    }
    
    if memory_usages:
        memory_array = np.array(memory_usages)
        benchmark_results['memory_usage'] = {
            'mean_mb': np.mean(memory_array),
            'max_mb': np.max(memory_array),
            'std_mb': np.std(memory_array)
        }
    
    return benchmark_results


def compare_functions(functions: List[Tuple[str, Callable]], 
                     iterations: int = 100,
                     *args, **kwargs) -> Dict[str, Any]:
    """
    Compare performance of multiple functions with same interface.
    
    Args:
        functions: List of (name, function) tuples to compare
        iterations: Number of iterations per function
        *args, **kwargs: Arguments to pass to all functions
        
    Returns:
        Comparison results with rankings
    """
    results = {}
    
    for name, func in functions:
        print(f"Benchmarking {name}...")
        benchmark_result = benchmark_function(func, iterations, *args, **kwargs)
        results[name] = benchmark_result
    
    # Create comparison summary
    comparison = {
        'functions': results,
        'ranking': {
            'by_speed': [],
            'by_memory': []
        }
    }
    
    # Rank by speed (mean execution time)
    speed_ranking = sorted(
        results.items(),
        key=lambda x: x[1].get('execution_times', {}).get('mean_ms', float('inf'))
    )
    comparison['ranking']['by_speed'] = [(name, data['execution_times']['mean_ms']) 
                                       for name, data in speed_ranking 
                                       if 'execution_times' in data]
    
    # Rank by memory usage (if available)
    memory_ranking = sorted(
        [(name, data) for name, data in results.items() if 'memory_usage' in data],
        key=lambda x: x[1]['memory_usage']['mean_mb']
    )
    comparison['ranking']['by_memory'] = [(name, data['memory_usage']['mean_mb'])
                                        for name, data in memory_ranking]
    
    return comparison


class ProfileBlock:
    """Context manager for profiling code blocks."""
    
    def __init__(self, block_name: str, profiler: Optional[PerformanceProfiler] = None):
        self.block_name = block_name
        self.profiler = profiler or get_profiler()
        self.measurement_id = None
    
    def __enter__(self):
        self.measurement_id = self.profiler.start_measurement(self.block_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.measurement_id is not None:
            metrics = self.profiler.stop_measurement(self.measurement_id)
            if exc_type is not None:
                metrics.error_occurred = True
                metrics.error_message = str(exc_val)


# Usage examples and testing
if __name__ == "__main__":
    # Example 1: Function decorator
    @profile_function(include_params=True)
    def black_scholes_call(S, K, r, sigma, T):
        """Example Black-Scholes calculation."""
        import math
        d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
        d2 = d1 - sigma*math.sqrt(T)
        
        # Simulate some computation time
        time.sleep(0.001)
        
        from scipy.stats import norm
        return S*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2)
    
    # Test function profiling
    print("Testing function profiling...")
    result = black_scholes_call(100, 100, 0.05, 0.2, 1.0)
    print(f"Option price: {result:.4f}")
    
    # Example 2: Context manager
    print("\nTesting context manager...")
    with ProfileBlock("monte_carlo_simulation"):
        # Bootstrap Monte Carlo from real market data
        try:
            from data.realtime_manager import get_data_manager
            import datetime as dt
            
            dm = get_data_manager()
            end_date = dt.datetime.now()
            start_date = end_date - dt.timedelta(days=252)
            
            spy_data = dm.get_historical_data_sync('SPY', start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), '1d')
            paths = spy_data['close'].pct_change().dropna().values
            paths = np.random.choice(paths, size=10000, replace=True)
            result = np.mean(np.maximum(paths, 0))
        except Exception as e:
            print(f"Real data unavailable: {e}")
            print("Skipping Monte Carlo test")
            pass
    
    if 'result' in locals():
        print(f"Monte Carlo result: {result:.4f}")
    
    # Example 3: Benchmarking
    print("\nBenchmarking function...")
    
    def simple_calculation(n):
        return sum(range(n))
    
    benchmark_results = benchmark_function(simple_calculation, iterations=50, n=1000)
    print(f"Benchmark results: {benchmark_results['execution_times']['mean_ms']:.3f}ms average")
    
    # Example 4: Function comparison
    print("\nComparing functions...")
    
    def method_a(data):
        return sum(data)
    
    def method_b(data):
        return np.sum(data)
    
    test_data = list(range(1000))
    
    comparison_results = compare_functions([
        ("Python sum", method_a),
        ("NumPy sum", method_b)
    ], iterations=100, data=test_data)
    
    print("Speed ranking:")
    for i, (name, time_ms) in enumerate(comparison_results['ranking']['by_speed'], 1):
        print(f"  {i}. {name}: {time_ms:.3f}ms")
    
    # Show profiler statistics
    print("\nProfiler statistics:")
    profiler = get_profiler()
    for func_name in profiler.metrics_history:
        stats = profiler.get_function_stats(func_name)
        if stats:
            print(f"  {func_name}: {stats['mean_time_ms']:.3f}ms avg, {stats['call_count']} calls")
