"""
Walk-Forward Backtesting
Rolling window optimization with out-of-sample validation

Features:
- In-sample parameter optimization
- Out-of-sample performance testing
- Rolling window analysis
- Overfitting detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable, Optional
from dataclasses import dataclass
import time
import os
import logging
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardWindow:
    """Single walk-forward analysis window."""
    window_id: int
    
    # Data windows
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    
    # Optimized parameters
    optimal_params: Dict = None
    
    # Performance metrics
    train_sharpe: float = 0.0
    test_sharpe: float = 0.0
    train_return: float = 0.0
    test_return: float = 0.0
    train_drawdown: float = 0.0
    test_drawdown: float = 0.0
    
    # Overfitting indicators
    sharpe_degradation: float = 0.0
    return_degradation: float = 0.0
    
    def calculate_degradation(self) -> None:
        """Calculate performance degradation from train to test."""
        self.sharpe_degradation = (
            (self.train_sharpe - self.test_sharpe) / (abs(self.train_sharpe) + 1e-6)
        )
        self.return_degradation = (
            (self.train_return - self.test_return) / (abs(self.train_return) + 1e-6)
        )


class WalkForwardOptimizer:
    """
    Walk-forward analysis framework.
    
    Methodology:
    1. Divide data into rolling windows (train + test)
    2. Optimize parameters on training window
    3. Test parameters on out-of-sample window
    4. Roll forward and repeat
    5. Aggregate results and detect overfitting
    """
    
    def __init__(
        self,
        train_days: int = 252,
        test_days: int = 63,
        step_days: int = 21,
        optimization_metric: str = "sharpe"
    ):
        """
        Initialize walk-forward optimizer.
        
        Args:
            train_days: Training window size (trading days)
            test_days: Testing window size (trading days)
            step_days: Window step size (trading days)
            optimization_metric: Metric to optimize ("sharpe", "return", "sortino")
        """
        self.train_days = train_days
        self.test_days = test_days
        self.step_days = step_days
        self.optimization_metric = optimization_metric
        
        self.windows: List[WalkForwardWindow] = []
        self.results_df = None
        
    def run_walk_forward(
        self,
        data: pd.DataFrame,
        strategy_func: Callable,
        param_grid: Dict[str, List],
        min_trades: int = 10,
        n_jobs: int = -1
    ) -> pd.DataFrame:
        """
        Run complete walk-forward analysis with optional parallelization.
        
        Args:
            data: Price/market data with datetime index
            strategy_func: Trading strategy function(data, params) -> returns
            param_grid: Parameter search space
            min_trades: Minimum trades required per window
            n_jobs: Number of parallel workers (-1 = all CPUs, 1 = sequential)
            
        Returns:
            DataFrame with window results
        """
        logger.info(f"Starting walk-forward analysis...")
        logger.info(f"  Train: {self.train_days} days, Test: {self.test_days} days")
        logger.info(f"  Step: {self.step_days} days")
        
        # Generate windows
        windows = self._generate_windows(data)
        logger.info(f"  Generated {len(windows)} windows")
        
        # Determine worker count
        if n_jobs == -1:
            n_workers = min(os.cpu_count() or 1, len(windows))
        elif n_jobs == 1:
            n_workers = 1
        else:
            n_workers = min(n_jobs, len(windows))
        
        if n_workers <= 1 or len(windows) <= 2:
            # Sequential execution (small workloads or explicit single-thread)
            self._run_sequential(windows, data, strategy_func, param_grid, min_trades)
        else:
            # Parallel execution using ThreadPoolExecutor
            # (ProcessPoolExecutor requires picklable strategy_func which lambdas/closures break)
            logger.info(f"  Running {len(windows)} windows on {n_workers} threads")
            self._run_parallel(windows, data, strategy_func, param_grid, min_trades, n_workers)
        
        # Create results DataFrame
        self.results_df = self._create_results_df()
        
        return self.results_df
    
    def _run_sequential(self, windows, data, strategy_func, param_grid, min_trades):
        """Process windows sequentially."""
        for i, window in enumerate(windows):
            result = self._process_single_window(
                window, data, strategy_func, param_grid, min_trades
            )
            if result is not None:
                self.windows.append(result)
                logger.info(
                    f"  Window {i+1}/{len(windows)}: "
                    f"Train Sharpe={result.train_sharpe:.3f}, "
                    f"Test Sharpe={result.test_sharpe:.3f}, "
                    f"Degradation={result.sharpe_degradation*100:.1f}%"
                )
    
    def _run_parallel(self, windows, data, strategy_func, param_grid, min_trades, n_workers):
        """Process windows in parallel using ThreadPoolExecutor."""
        completed = 0
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_to_idx = {
                executor.submit(
                    self._process_single_window,
                    window, data, strategy_func, param_grid, min_trades
                ): i
                for i, window in enumerate(windows)
            }
            
            results_by_idx = {}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    if result is not None:
                        results_by_idx[idx] = result
                        completed += 1
                        logger.info(
                            f"  Window {completed}/{len(windows)} done: "
                            f"Test Sharpe={result.test_sharpe:.3f}"
                        )
                except Exception as e:
                    logger.warning(f"  Window {idx+1} failed: {e}")
            
            # Maintain order
            for idx in sorted(results_by_idx.keys()):
                self.windows.append(results_by_idx[idx])
    
    def _process_single_window(self, window, data, strategy_func, param_grid, min_trades):
        """Process a single walk-forward window. Returns populated window or None."""
        train_data = data[window.train_start:window.train_end]
        test_data = data[window.test_start:window.test_end]
        
        # Threshold 0.65 accounts for calendar→trading-day ratio (~0.69 for
        # US equities). timedelta(days=N) spans ~0.69*N trading days.
        if len(train_data) < self.train_days * 0.65 or len(test_data) < self.test_days * 0.45:
            return None
        
        # Optimize on training data
        optimal_params, train_metrics = self._optimize_parameters(
            train_data, strategy_func, param_grid, min_trades
        )
        
        if optimal_params is None:
            return None
        
        window.optimal_params = optimal_params
        window.train_sharpe = train_metrics['sharpe']
        window.train_return = train_metrics['total_return']
        window.train_drawdown = train_metrics['max_drawdown']
        
        # Test on out-of-sample data
        test_metrics = self._evaluate_strategy(
            test_data, strategy_func, optimal_params, min_trades
        )
        
        if test_metrics is None:
            return None
        
        window.test_sharpe = test_metrics['sharpe']
        window.test_return = test_metrics['total_return']
        window.test_drawdown = test_metrics['max_drawdown']
        window.calculate_degradation()
        
        return window
    
    def _generate_windows(self, data: pd.DataFrame) -> List[WalkForwardWindow]:
        """Generate rolling windows."""
        windows = []
        window_id = 0
        
        start_date = data.index[0]
        end_date = data.index[-1]
        
        current_date = start_date
        
        while current_date + timedelta(days=self.train_days + self.test_days) <= end_date:
            train_start = current_date
            train_end = current_date + timedelta(days=self.train_days)
            test_start = train_end + timedelta(days=1)
            test_end = test_start + timedelta(days=self.test_days)
            
            window = WalkForwardWindow(
                window_id=window_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end
            )
            
            windows.append(window)
            window_id += 1
            
            # Step forward
            current_date += timedelta(days=self.step_days)
        
        return windows
    
    def _optimize_parameters(
        self,
        data: pd.DataFrame,
        strategy_func: Callable,
        param_grid: Dict[str, List],
        min_trades: int
    ) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Optimize strategy parameters via grid search.
        
        Returns:
            (optimal_params, metrics) or (None, None) if failed
        """
        best_metric = -np.inf
        best_params = None
        best_metrics = None
        
        # Generate parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        # Grid search (simplified - full version would use itertools.product)
        from itertools import product
        
        for param_combo in product(*param_values):
            params = dict(zip(param_names, param_combo))
            
            # Evaluate strategy
            metrics = self._evaluate_strategy(data, strategy_func, params, min_trades)
            
            if metrics is None:
                continue
            
            # Check if best
            current_metric = metrics.get(self.optimization_metric, -np.inf)
            if current_metric > best_metric:
                best_metric = current_metric
                best_params = params
                best_metrics = metrics
        
        return best_params, best_metrics
    
    def _evaluate_strategy(
        self,
        data: pd.DataFrame,
        strategy_func: Callable,
        params: Dict,
        min_trades: int
    ) -> Optional[Dict]:
        """
        Evaluate strategy with given parameters.
        
        Returns:
            Performance metrics or None if failed
        """
        try:
            # Run strategy
            returns = strategy_func(data, params)
            
            if returns is None or len(returns) < min_trades:
                return None
            
            # Calculate metrics
            total_return = returns.sum()
            sharpe = returns.mean() / (returns.std() + 1e-6) * np.sqrt(252)
            
            # Sortino ratio (downside deviation)
            downside_returns = returns[returns < 0]
            sortino = (
                returns.mean() / (downside_returns.std() + 1e-6) * np.sqrt(252)
                if len(downside_returns) > 0 else 0
            )
            
            # Maximum drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            return {
                'total_return': total_return,
                'sharpe': sharpe,
                'sortino': sortino,
                'max_drawdown': max_drawdown,
                'num_trades': len(returns)
            }
            
        except Exception as e:
            return None
    
    def _create_results_df(self) -> pd.DataFrame:
        """Create results DataFrame from windows."""
        if not self.windows:
            return pd.DataFrame()
        
        records = []
        for window in self.windows:
            records.append({
                'window_id': window.window_id,
                'train_start': window.train_start,
                'train_end': window.train_end,
                'test_start': window.test_start,
                'test_end': window.test_end,
                'train_sharpe': window.train_sharpe,
                'test_sharpe': window.test_sharpe,
                'train_return': window.train_return,
                'test_return': window.test_return,
                'sharpe_degradation': window.sharpe_degradation,
                'return_degradation': window.return_degradation,
                'params': str(window.optimal_params)
            })
        
        return pd.DataFrame(records)
    
    def get_summary(self) -> Dict:
        """Get walk-forward analysis summary."""
        if self.results_df is None or len(self.results_df) == 0:
            return {}
        
        df = self.results_df
        
        return {
            'num_windows': len(df),
            'avg_train_sharpe': df['train_sharpe'].mean(),
            'avg_test_sharpe': df['test_sharpe'].mean(),
            'avg_train_return': df['train_return'].mean(),
            'avg_test_return': df['test_return'].mean(),
            'avg_sharpe_degradation': df['sharpe_degradation'].mean(),
            'sharpe_stability': df['test_sharpe'].std(),
            'profitable_windows': (df['test_return'] > 0).sum(),
            'overfitting_detected': df['sharpe_degradation'].mean() > 0.3
        }


# Demo
if __name__ == "__main__":
    print("=" * 70)
    print("WALK-FORWARD BACKTESTING DEMO")
    print("=" * 70)
    
    # Generate synthetic price data (2 years)
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', '2023-12-31', freq='D')
    prices = 100 * (1 + np.random.randn(len(dates)).cumsum() * 0.01)
    data = pd.DataFrame({'close': prices}, index=dates)
    
    print(f"\nGenerated {len(data)} days of price data")
    print(f"Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    
    # Define simple moving average strategy
    def ma_crossover_strategy(data: pd.DataFrame, params: Dict) -> pd.Series:
        """Simple MA crossover strategy."""
        fast_ma = data['close'].rolling(params['fast_ma']).mean()
        slow_ma = data['close'].rolling(params['slow_ma']).mean()
        
        # Generate signals
        signal = (fast_ma > slow_ma).astype(int).diff()
        
        # Calculate returns
        returns = data['close'].pct_change() * signal.shift(1)
        
        return returns.dropna()
    
    # Parameter grid
    param_grid = {
        'fast_ma': [10, 20, 30],
        'slow_ma': [50, 100, 150]
    }
    
    # Run walk-forward analysis
    print("\nRunning walk-forward analysis...")
    optimizer = WalkForwardOptimizer(
        train_days=120,  # 4 months train
        test_days=30,    # 1 month test
        step_days=30     # Roll monthly
    )
    
    results = optimizer.run_walk_forward(
        data=data,
        strategy_func=ma_crossover_strategy,
        param_grid=param_grid,
        min_trades=5
    )
    
    # Display summary
    print("\n" + "=" * 70)
    print("WALK-FORWARD RESULTS")
    print("=" * 70)
    
    summary = optimizer.get_summary()
    print(f"\nAnalysis Summary:")
    print(f"  Windows Analyzed: {summary['num_windows']}")
    print(f"  Avg Train Sharpe: {summary['avg_train_sharpe']:.3f}")
    print(f"  Avg Test Sharpe: {summary['avg_test_sharpe']:.3f}")
    print(f"  Avg Sharpe Degradation: {summary['avg_sharpe_degradation']*100:.1f}%")
    print(f"  Sharpe Stability (std): {summary['sharpe_stability']:.3f}")
    print(f"  Profitable Windows: {summary['profitable_windows']}/{summary['num_windows']}")
    print(f"  Overfitting Detected: {' ️  YES' if summary['overfitting_detected'] else '  NO'}")
    
    print("\nWindow Details:")
    print(results[['window_id', 'train_sharpe', 'test_sharpe', 'sharpe_degradation']].to_string(index=False))
    
    print("\n  Walk-forward analysis complete!")
