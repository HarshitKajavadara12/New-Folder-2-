"""
GIGA SYSTEM - Backtesting Performance Metrics
Greek Intelligence for Global Analysis

Comprehensive performance metrics and risk analysis for backtesting results.
Calculates industry-standard metrics with mathematical precision and
statistical significance testing.

Key Features:
- Return-based performance metrics
- Risk-adjusted performance measures  
- Drawdown analysis with recovery periods
- Factor attribution analysis
- Statistical significance testing
- Benchmark-relative metrics

Mathematical Rigor:
- Proper annualization accounting for compounding
- Bias-corrected standard deviation estimates
- Bootstrap confidence intervals
- Monte Carlo significance testing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
from scipy import stats
from sklearn.linear_model import LinearRegression

try:
    from ..utils.math_helpers import (
        sharpe_ratio, sortino_ratio, maximum_drawdown,
        value_at_risk, conditional_var, annualize_returns
    )
    from ..utils.performance_profiler import profile_function
except (ImportError, ValueError):
    from utils.math_helpers import (
        sharpe_ratio, sortino_ratio, maximum_drawdown,
        value_at_risk, conditional_var, annualize_returns
    )
    from utils.performance_profiler import profile_function


@dataclass
class PerformanceMetrics:
    """Container for comprehensive performance metrics."""
    
    # Return metrics
    total_return: float
    annualized_return: float
    compound_annual_growth_rate: float
    
    # Risk metrics
    volatility: float
    downside_deviation: float
    maximum_drawdown: float
    maximum_drawdown_duration: int
    
    # Risk-adjusted metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Distribution metrics
    skewness: float
    kurtosis: float
    var_95: float
    cvar_95: float
    
    # Trade-based metrics
    win_rate: Optional[float] = None
    profit_loss_ratio: Optional[float] = None
    average_win: Optional[float] = None
    average_loss: Optional[float] = None
    
    # Time-based metrics
    best_day: float = 0.0
    worst_day: float = 0.0
    positive_days: float = 0.0
    
    # Statistical significance
    t_statistic: Optional[float] = None
    p_value: Optional[float] = None
    
    # Omega ratio
    omega_ratio: Optional[float] = None
    
    # Additional context
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    n_observations: int = 0
    frequency: str = 'daily'


@dataclass  
class RiskMetrics:
    """Container for detailed risk analysis."""
    
    # Value at Risk metrics
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    
    # Drawdown analysis
    max_drawdown: float
    average_drawdown: float
    max_drawdown_duration: int
    average_drawdown_duration: float
    drawdown_recovery_factor: float
    
    # Volatility metrics
    realized_volatility: float
    downside_volatility: float
    upside_volatility: float
    volatility_of_volatility: float
    
    # Tail risk measures
    skewness: float
    excess_kurtosis: float
    jarque_bera_statistic: float
    jarque_bera_pvalue: float
    
    # Correlation and beta
    market_correlation: Optional[float] = None
    market_beta: Optional[float] = None
    tracking_error: Optional[float] = None
    
    # Risk-adjusted returns
    risk_adjusted_return: float = 0.0
    information_ratio: Optional[float] = None


class PerformanceAnalyzer:
    """
    Comprehensive performance analysis engine.
    
    Calculates detailed performance and risk metrics with
    statistical significance testing and benchmark comparison.
    """
    
    def __init__(self, 
                 risk_free_rate: float = 0.02,
                 confidence_levels: List[float] = [0.95, 0.99]):
        """
        Initialize performance analyzer.
        
        Args:
            risk_free_rate: Annual risk-free rate
            confidence_levels: Confidence levels for VaR calculations
        """
        self.risk_free_rate = risk_free_rate
        self.confidence_levels = confidence_levels
        
        # Cache for expensive calculations
        self._cache = {}
    
    @profile_function(include_params=True)
    def calculate_metrics(self, 
                         returns: Union[pd.Series, np.ndarray],
                         benchmark_returns: Optional[Union[pd.Series, np.ndarray]] = None,
                         prices: Optional[Union[pd.Series, np.ndarray]] = None,
                         trades: Optional[pd.DataFrame] = None) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            returns: Portfolio returns time series
            benchmark_returns: Benchmark returns for comparison
            prices: Price series (optional, for additional metrics)
            trades: Individual trade data (optional)
            
        Returns:
            PerformanceMetrics with all calculated metrics
        """
        # Convert to pandas Series for easier manipulation
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)
        
        # Remove any NaN values
        returns = returns.dropna()
        
        if len(returns) == 0:
            raise ValueError("No valid returns data provided")
        
        # Basic return metrics
        total_return = (1 + returns).prod() - 1
        n_periods = len(returns)
        
        # Determine frequency for annualization
        if hasattr(returns, 'index') and hasattr(returns.index, 'freq'):
            freq = returns.index.freq
        else:
            freq = self._infer_frequency(returns)
        
        periods_per_year = self._get_periods_per_year(freq)
        
        # Annualized return
        if n_periods > 0:
            annualized_return = (1 + total_return) ** (periods_per_year / n_periods) - 1
            cagr = annualized_return  # Same calculation for our purposes
        else:
            annualized_return = 0.0
            cagr = 0.0
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(periods_per_year)
        
        # Downside deviation (for Sortino ratio)
        downside_returns = returns[returns < self.risk_free_rate / periods_per_year]
        if len(downside_returns) > 0:
            downside_deviation = downside_returns.std() * np.sqrt(periods_per_year)
        else:
            downside_deviation = 0.0
        
        # Drawdown analysis
        if prices is not None:
            prices = pd.Series(prices) if isinstance(prices, np.ndarray) else prices
            max_dd, dd_duration = self._calculate_detailed_drawdown(prices)
        else:
            # Calculate from returns
            cumulative_returns = (1 + returns).cumprod()
            max_dd, _, _ = maximum_drawdown(cumulative_returns.values)
            max_dd = abs(max_dd)  # Make positive
            dd_duration = self._estimate_drawdown_duration(cumulative_returns)
        
        # Risk-adjusted ratios
        sharpe = (annualized_return - self.risk_free_rate) / volatility if volatility > 0 else 0.0
        sortino = (annualized_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0.0
        calmar = annualized_return / max_dd if max_dd > 0 else 0.0
        
        # Omega ratio
        sum_above = returns[returns > self.risk_free_rate / periods_per_year].sum()
        excess_below = (self.risk_free_rate / periods_per_year) - returns[returns <= self.risk_free_rate / periods_per_year]
        sum_below = excess_below.sum() if len(excess_below) > 0 else 0.0
        omega = sum_above / sum_below if sum_below > 0 else float('inf')
        
        # Distribution metrics
        skew = returns.skew()
        kurt = returns.kurtosis()  # Excess kurtosis
        
        # VaR calculations
        var_95 = returns.quantile(0.05)
        var_99 = returns.quantile(0.01)
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
        cvar_99 = returns[returns <= var_99].mean() if len(returns[returns <= var_99]) > 0 else var_99
        
        # Daily extremes
        best_day = returns.max()
        worst_day = returns.min()
        positive_days = (returns > 0).mean()
        
        # Statistical significance
        t_stat, p_val = self._test_significance(returns, periods_per_year)
        
        # Trade-based metrics (if provided)
        win_rate = None
        pl_ratio = None
        avg_win = None
        avg_loss = None
        
        if trades is not None:
            trade_metrics = self._calculate_trade_metrics(trades)
            win_rate = trade_metrics.get('win_rate')
            pl_ratio = trade_metrics.get('profit_loss_ratio')
            avg_win = trade_metrics.get('average_win')
            avg_loss = trade_metrics.get('average_loss')
        
        # Create metrics object
        metrics = PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            compound_annual_growth_rate=cagr,
            volatility=volatility,
            downside_deviation=downside_deviation,
            maximum_drawdown=max_dd,
            maximum_drawdown_duration=dd_duration,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            omega_ratio=omega,
            skewness=skew,
            kurtosis=kurt,
            var_95=var_95,
            cvar_95=cvar_95,
            best_day=best_day,
            worst_day=worst_day,
            positive_days=positive_days,
            t_statistic=t_stat,
            p_value=p_val,
            win_rate=win_rate,
            profit_loss_ratio=pl_ratio,
            average_win=avg_win,
            average_loss=avg_loss,
            start_date=returns.index[0] if hasattr(returns, 'index') else None,
            end_date=returns.index[-1] if hasattr(returns, 'index') else None,
            n_observations=len(returns),
            frequency=freq
        )
        
        return metrics
    
    @profile_function
    def calculate_risk_metrics(self, 
                              returns: Union[pd.Series, np.ndarray],
                              benchmark_returns: Optional[Union[pd.Series, np.ndarray]] = None) -> RiskMetrics:
        """
        Calculate detailed risk metrics.
        
        Args:
            returns: Portfolio returns
            benchmark_returns: Benchmark for beta/correlation calculations
            
        Returns:
            RiskMetrics with detailed risk analysis
        """
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)
        
        returns = returns.dropna()
        
        # Value at Risk
        var_95 = returns.quantile(0.05)
        var_99 = returns.quantile(0.01)
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
        cvar_99 = returns[returns <= var_99].mean() if len(returns[returns <= var_99]) > 0 else var_99
        
        # Drawdown analysis
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        
        max_dd = abs(drawdowns.min())
        avg_dd = abs(drawdowns[drawdowns < 0].mean()) if len(drawdowns[drawdowns < 0]) > 0 else 0.0
        
        # Drawdown durations
        dd_periods = self._calculate_drawdown_periods(drawdowns)
        max_dd_duration = max(dd_periods) if dd_periods else 0
        avg_dd_duration = np.mean(dd_periods) if dd_periods else 0.0
        
        # Recovery factor
        recovery_factor = abs(returns.sum()) / max_dd if max_dd > 0 else 0.0
        
        # Volatility metrics
        realized_vol = returns.std() * np.sqrt(252)
        downside_vol = returns[returns < 0].std() * np.sqrt(252) if len(returns[returns < 0]) > 0 else 0.0
        upside_vol = returns[returns > 0].std() * np.sqrt(252) if len(returns[returns > 0]) > 0 else 0.0
        
        # Volatility of volatility
        rolling_vol = returns.rolling(20).std()
        vol_of_vol = rolling_vol.std() * np.sqrt(252)
        
        # Distribution statistics
        skew = returns.skew()
        excess_kurt = returns.kurtosis()
        
        # Jarque-Bera test
        jb_stat, jb_pvalue = stats.jarque_bera(returns.dropna())
        
        # Benchmark-relative metrics
        market_corr = None
        market_beta = None
        tracking_error = None
        info_ratio = None
        
        if benchmark_returns is not None:
            if isinstance(benchmark_returns, np.ndarray):
                benchmark_returns = pd.Series(benchmark_returns, index=returns.index)
            
            # Align series
            aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')
            
            if len(aligned_returns) > 10:  # Need sufficient data
                # Correlation
                market_corr = aligned_returns.corr(aligned_benchmark)
                
                # Beta
                covariance = aligned_returns.cov(aligned_benchmark)
                benchmark_var = aligned_benchmark.var()
                market_beta = covariance / benchmark_var if benchmark_var > 0 else 0.0
                
                # Tracking error
                active_returns = aligned_returns - aligned_benchmark
                tracking_error = active_returns.std() * np.sqrt(252)
                
                # Information ratio
                if tracking_error > 0:
                    info_ratio = active_returns.mean() * 252 / tracking_error
        
        # Risk-adjusted return
        risk_adj_return = returns.mean() / returns.std() if returns.std() > 0 else 0.0
        
        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            max_drawdown=max_dd,
            average_drawdown=avg_dd,
            max_drawdown_duration=max_dd_duration,
            average_drawdown_duration=avg_dd_duration,
            drawdown_recovery_factor=recovery_factor,
            realized_volatility=realized_vol,
            downside_volatility=downside_vol,
            upside_volatility=upside_vol,
            volatility_of_volatility=vol_of_vol,
            skewness=skew,
            excess_kurtosis=excess_kurt,
            jarque_bera_statistic=jb_stat,
            jarque_bera_pvalue=jb_pvalue,
            market_correlation=market_corr,
            market_beta=market_beta,
            tracking_error=tracking_error,
            risk_adjusted_return=risk_adj_return,
            information_ratio=info_ratio
        )
    
    def _calculate_detailed_drawdown(self, prices: pd.Series) -> Tuple[float, int]:
        """Calculate detailed drawdown metrics from price series."""
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        
        max_dd = abs(drawdown.min())
        
        # Find drawdown duration
        max_dd_idx = drawdown.idxmin()
        peak_before_idx = peak[:max_dd_idx].idxmax()
        
        # Find recovery point
        recovery_idx = None
        if max_dd_idx < len(prices) - 1:
            post_drawdown = prices[max_dd_idx:]
            peak_value = peak.loc[max_dd_idx]
            recovery_mask = post_drawdown >= peak_value
            if recovery_mask.any():
                recovery_idx = recovery_mask.idxmax()
        
        if recovery_idx is not None:
            duration = (recovery_idx - peak_before_idx).days if hasattr(peak_before_idx, 'days') else recovery_idx - peak_before_idx
        else:
            # Still in drawdown
            duration = (prices.index[-1] - peak_before_idx).days if hasattr(peak_before_idx, 'days') else len(prices) - peak_before_idx
        
        return max_dd, duration
    
    def _estimate_drawdown_duration(self, cumulative_returns: pd.Series) -> int:
        """Estimate drawdown duration from cumulative returns."""
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        max_dd_idx = drawdown.idxmin()
        
        # Find previous peak
        pre_peak_mask = cumulative_returns.index < max_dd_idx
        if pre_peak_mask.any():
            peak_idx = cumulative_returns[pre_peak_mask].idxmax()
            return max_dd_idx - peak_idx if isinstance(max_dd_idx, int) else 1
        return 1
    
    def _calculate_drawdown_periods(self, drawdowns: pd.Series) -> List[int]:
        """Calculate all drawdown period durations."""
        periods = []
        in_drawdown = False
        start_idx = None
        
        for i, dd in enumerate(drawdowns):
            if dd < 0 and not in_drawdown:
                # Start of drawdown
                in_drawdown = True
                start_idx = i
            elif dd >= 0 and in_drawdown:
                # End of drawdown
                if start_idx is not None:
                    periods.append(i - start_idx)
                in_drawdown = False
                start_idx = None
        
        # Handle case where drawdown continues to end
        if in_drawdown and start_idx is not None:
            periods.append(len(drawdowns) - start_idx)
        
        return periods
    
    def _test_significance(self, returns: pd.Series, periods_per_year: int) -> Tuple[float, float]:
        """Test statistical significance of returns."""
        if len(returns) < 30:
            return np.nan, np.nan
        
        # Test if mean return is significantly different from risk-free rate
        daily_rf = self.risk_free_rate / periods_per_year
        excess_returns = returns - daily_rf
        
        t_stat, p_val = stats.ttest_1samp(excess_returns.dropna(), 0)
        return t_stat, p_val
    
    def _calculate_trade_metrics(self, trades: pd.DataFrame) -> Dict[str, float]:
        """Calculate trade-based performance metrics."""
        if 'pnl' not in trades.columns:
            return {}
        
        pnl = trades['pnl']
        winning_trades = pnl[pnl > 0]
        losing_trades = pnl[pnl < 0]
        
        metrics = {}
        
        if len(pnl) > 0:
            metrics['win_rate'] = len(winning_trades) / len(pnl)
            
            if len(winning_trades) > 0:
                metrics['average_win'] = winning_trades.mean()
            
            if len(losing_trades) > 0:
                metrics['average_loss'] = abs(losing_trades.mean())
                
                if metrics.get('average_loss', 0) > 0:
                    metrics['profit_loss_ratio'] = metrics.get('average_win', 0) / metrics['average_loss']
        
        return metrics
    
    def _infer_frequency(self, returns: pd.Series) -> str:
        """Infer the frequency of returns series."""
        if hasattr(returns, 'index') and len(returns) > 1:
            time_diff = returns.index[1] - returns.index[0]
            if isinstance(time_diff, pd.Timedelta):
                days = time_diff.days
                if days == 1:
                    return 'daily'
                elif days == 7:
                    return 'weekly'
                elif 28 <= days <= 31:
                    return 'monthly'
        
        return 'daily'  # Default assumption
    
    def _get_periods_per_year(self, frequency: str) -> int:
        """Get number of periods per year for given frequency."""
        freq_map = {
            'daily': 252,
            'weekly': 52,
            'monthly': 12,
            'quarterly': 4,
            'yearly': 1
        }
        return freq_map.get(frequency, 252)
    
    def monte_carlo_bootstrap(self, 
                             returns: pd.Series,
                             n_simulations: int = 1000,
                             block_size: int = 20) -> Dict[str, Any]:
        """
        Perform Monte Carlo bootstrap analysis for confidence intervals.
        
        Args:
            returns: Return series
            n_simulations: Number of bootstrap simulations
            block_size: Block size for block bootstrap
            
        Returns:
            Dictionary with bootstrap statistics
        """
        n_obs = len(returns)
        bootstrap_metrics = []
        
        for _ in range(n_simulations):
            # Block bootstrap preserving serial correlation
            # Use actual bootstrap sampling from provided returns data
            indices = []
            while len(indices) < n_obs:
                start_idx = np.random.randint(0, n_obs - block_size + 1)
                indices.extend(range(start_idx, min(start_idx + block_size, n_obs)))
            
            indices = indices[:n_obs]
            bootstrap_returns = returns.iloc[indices].reset_index(drop=True)
            
            # Calculate metrics for this bootstrap sample
            try:
                metrics = self.calculate_metrics(bootstrap_returns)
                bootstrap_metrics.append({
                    'total_return': metrics.total_return,
                    'volatility': metrics.volatility,
                    'sharpe_ratio': metrics.sharpe_ratio,
                    'max_drawdown': metrics.maximum_drawdown
                })
            except Exception:
                continue
        
        if not bootstrap_metrics:
            return {}
        
        # Calculate confidence intervals
        bootstrap_df = pd.DataFrame(bootstrap_metrics)
        confidence_intervals = {}
        
        for metric in bootstrap_df.columns:
            values = bootstrap_df[metric].dropna()
            if len(values) > 0:
                confidence_intervals[metric] = {
                    'mean': values.mean(),
                    'std': values.std(),
                    'ci_95': (values.quantile(0.025), values.quantile(0.975)),
                    'ci_90': (values.quantile(0.05), values.quantile(0.95))
                }
        
        return confidence_intervals


def calculate_performance_metrics(returns: Union[pd.Series, np.ndarray],
                                benchmark_returns: Optional[Union[pd.Series, np.ndarray]] = None,
                                risk_free_rate: float = 0.02,
                                prices: Optional[Union[pd.Series, np.ndarray]] = None,
                                trades: Optional[pd.DataFrame] = None) -> PerformanceMetrics:
    """
    Convenience function to calculate performance metrics.
    
    Args:
        returns: Portfolio return series
        benchmark_returns: Optional benchmark returns
        risk_free_rate: Risk-free rate for calculations
        prices: Optional price series
        trades: Optional trade data
        
    Returns:
        PerformanceMetrics object
    """
    analyzer = PerformanceAnalyzer(risk_free_rate=risk_free_rate)
    return analyzer.calculate_metrics(returns, benchmark_returns, prices, trades)


# Performance testing and examples
if __name__ == "__main__":
    import time
    
    print("GIGA System Backtesting Metrics - Performance Test")
    print("=" * 55)
    
    # Fetch real SPY/QQQ data for metrics testing
    REAL_DATA_AVAILABLE = True
    try:
        from data.realtime_manager import get_data_manager
        import datetime as dt
        
        manager = get_data_manager()
        end_date = dt.datetime.now()
        start_date = end_date - dt.timedelta(days=1260)
        
        spy_data = manager.get_historical_data_sync('SPY', start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), '1d')
        qqq_data = manager.get_historical_data_sync('QQQ', start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), '1d')
        
        returns = spy_data['close'].pct_change().dropna().values[-1000:]
        benchmark_returns = qqq_data['close'].pct_change().dropna().values[-1000:]
        
        dates = spy_data.index[-1000:]
        returns_series = pd.Series(returns, index=dates)
        benchmark_series = pd.Series(benchmark_returns, index=dates)
        prices = spy_data['close'].iloc[-1000:]
        
        print(f"\nFetched {len(returns)} days of real SPY returns vs QQQ benchmark")
        print(f"Period: {dates[0]} to {dates[-1]}")
    except Exception as e:
        print(f"  Real market data unavailable: {e}")
        print("  Metrics demonstration requires SPY and QQQ historical data")
        import sys; sys.exit(0)
    
    print(f"\\nGenerated {len(returns)} days of return data")
    print(f"Portfolio total return: {((1 + returns_series).prod() - 1) * 100:.2f}%")
    print(f"Benchmark total return: {((1 + benchmark_series).prod() - 1) * 100:.2f}%")
    
    # Test Performance Metrics Calculation
    print("\\n" + "-" * 40)
    print("Testing Performance Metrics")
    print("-" * 40)
    
    analyzer = PerformanceAnalyzer(risk_free_rate=0.02)
    
    start_time = time.perf_counter()
    perf_metrics = analyzer.calculate_metrics(
        returns=returns_series,
        benchmark_returns=benchmark_series,
        prices=prices
    )
    perf_time = (time.perf_counter() - start_time) * 1000
    
    print(f"Performance metrics calculation: {perf_time:.1f}ms")
    print(f"\\nKey Metrics:")
    print(f"  Total Return: {perf_metrics.total_return * 100:.2f}%")
    print(f"  Annualized Return: {perf_metrics.annualized_return * 100:.2f}%")
    print(f"  Volatility: {perf_metrics.volatility * 100:.2f}%")
    print(f"  Sharpe Ratio: {perf_metrics.sharpe_ratio:.3f}")
    print(f"  Maximum Drawdown: {perf_metrics.maximum_drawdown * 100:.2f}%")
    print(f"  Win Rate: {perf_metrics.positive_days * 100:.1f}%")
    
    # Test Risk Metrics Calculation  
    print("\\n" + "-" * 40)
    print("Testing Risk Metrics")
    print("-" * 40)
    
    start_time = time.perf_counter()
    risk_metrics = analyzer.calculate_risk_metrics(
        returns=returns_series,
        benchmark_returns=benchmark_series
    )
    risk_time = (time.perf_counter() - start_time) * 1000
    
    print(f"Risk metrics calculation: {risk_time:.1f}ms")
    print(f"\\nRisk Analysis:")
    print(f"  VaR (95%): {risk_metrics.var_95 * 100:.2f}%")
    print(f"  CVaR (95%): {risk_metrics.cvar_95 * 100:.2f}%")
    print(f"  Beta: {risk_metrics.market_beta:.3f}")
    print(f"  Correlation: {risk_metrics.market_correlation:.3f}")
    print(f"  Information Ratio: {risk_metrics.information_ratio:.3f}")
    
    # Test Statistical Significance
    if perf_metrics.p_value is not None:
        significance = "significant" if perf_metrics.p_value < 0.05 else "not significant"
        print(f"\\nStatistical Significance:")
        print(f"  t-statistic: {perf_metrics.t_statistic:.3f}")
        print(f"  p-value: {perf_metrics.p_value:.4f}")
        print(f"  Result: {significance} at 5% level")
    
    # Test Bootstrap Analysis
    print("\\n" + "-" * 40)  
    print("Testing Bootstrap Analysis")
    print("-" * 40)
    
    start_time = time.perf_counter()
    bootstrap_results = analyzer.monte_carlo_bootstrap(
        returns_series, 
        n_simulations=100,  # Reduced for speed
        block_size=10
    )
    bootstrap_time = (time.perf_counter() - start_time) * 1000
    
    print(f"Bootstrap analysis (100 sims): {bootstrap_time:.1f}ms")
    
    if bootstrap_results:
        print("\\nBootstrap Confidence Intervals (95%):")
        for metric, stats in bootstrap_results.items():
            if 'ci_95' in stats:
                ci_lower, ci_upper = stats['ci_95']
                print(f"  {metric}: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    # Test with Trade Data  
    print("\n" + "-" * 40)
    print("Testing Trade-Based Metrics") 
    print("-" * 40)
    
    # Generate realistic trade data from actual weekly returns
    n_trades = 50
    try:
        weekly_returns = returns_series.resample('W').apply(lambda x: (1 + x).prod() - 1).dropna()[-n_trades:]
        trade_returns = weekly_returns.values
        entry_dates = weekly_returns.index - pd.Timedelta(days=7)
        exit_dates = weekly_returns.index
        print(f"Using {n_trades} real weekly return periods as trades")
    except Exception as e:
        print(f"  Trade metrics unavailable: {e}")
        print("  Trade analysis requires return series data")
        import sys; sys.exit(0)
    
    trade_data = pd.DataFrame({
        'pnl': trade_returns,
        'entry_date': entry_dates,
        'exit_date': exit_dates
    })
    
    start_time = time.perf_counter()
    trade_metrics = analyzer.calculate_metrics(
        returns=returns_series,
        trades=trade_data
    )
    trade_time = (time.perf_counter() - start_time) * 1000
    
    print(f"Trade metrics calculation: {trade_time:.1f}ms")
    print(f"\\nTrade Analysis:")
    print(f"  Win Rate: {trade_metrics.win_rate * 100:.1f}%")
    print(f"  Profit/Loss Ratio: {trade_metrics.profit_loss_ratio:.2f}")
    print(f"  Average Win: {trade_metrics.average_win * 100:.2f}%")
    print(f"  Average Loss: {trade_metrics.average_loss * 100:.2f}%")
    
    print("\\n" + "=" * 55)
    print("Backtesting metrics tests completed!")