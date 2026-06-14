"""
GIGA SYSTEM - Performance Analytics
Comprehensive backtesting performance metrics and attribution
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from scipy import stats as scipy_stats


@dataclass
class PerformanceMetrics:
    """Complete performance metrics."""
    # Returns
    total_return: float
    annualized_return: float
    cagr: float
    
    # Risk
    volatility: float
    annualized_volatility: float
    downside_volatility: float
    max_drawdown: float
    avg_drawdown: float
    max_drawdown_duration: int
    
    # Risk-adjusted
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float
    information_ratio: float
    
    # Distribution
    skewness: float
    kurtosis: float
    var_95: float
    cvar_95: float
    
    # Trading
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    expectancy: float
    
    # Other
    beta: float
    alpha: float
    r_squared: float


class PerformanceAnalyzer:
    """
    Comprehensive performance analysis toolkit.
    
    Mathematical Framework:
    - Modern Portfolio Theory metrics
    - Risk decomposition
    - Factor attribution
    - Distribution analysis
    """
    
    def __init__(self, risk_free_rate: float = 0.02, 
                 periods_per_year: int = 252):
        """
        Initialize performance analyzer.
        
        Parameters
        ----------
        risk_free_rate : float
            Annual risk-free rate.
        periods_per_year : int
            Trading periods per year (252 for daily).
        """
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
        self.rf_period = risk_free_rate / periods_per_year
    
    # =========================================================================
    # RETURN METRICS
    # =========================================================================
    
    def calculate_returns(self, equity_curve: np.ndarray) -> np.ndarray:
        """Calculate period returns from equity curve."""
        returns = np.diff(equity_curve) / equity_curve[:-1]
        return returns
    
    def calculate_log_returns(self, equity_curve: np.ndarray) -> np.ndarray:
        """Calculate log returns."""
        return np.diff(np.log(equity_curve))
    
    def total_return(self, equity_curve: np.ndarray) -> float:
        """Calculate total return."""
        return (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
    
    def annualized_return(self, equity_curve: np.ndarray) -> float:
        """
        Calculate annualized return.
        
        CAGR = (V_final / V_initial)^(1/years) - 1
        """
        total = self.total_return(equity_curve)
        n_periods = len(equity_curve)
        years = n_periods / self.periods_per_year
        
        if years <= 0:
            return 0.0
        
        return (1 + total) ** (1 / years) - 1
    
    # =========================================================================
    # RISK METRICS
    # =========================================================================
    
    def volatility(self, returns: np.ndarray, annualize: bool = True) -> float:
        """
        Calculate volatility.
        
        σ = std(returns) * √(periods_per_year)
        """
        vol = np.std(returns, ddof=1)
        
        if annualize:
            vol *= np.sqrt(self.periods_per_year)
        
        return vol
    
    def downside_volatility(self, returns: np.ndarray, 
                           target: float = 0.0,
                           annualize: bool = True) -> float:
        """
        Calculate downside volatility.
        
        Only considers returns below target.
        """
        downside = returns[returns < target]
        
        if len(downside) == 0:
            return 0.0
        
        vol = np.std(downside, ddof=1)
        
        if annualize:
            vol *= np.sqrt(self.periods_per_year)
        
        return vol
    
    def max_drawdown(self, equity_curve: np.ndarray) -> Tuple[float, int, int]:
        """
        Calculate maximum drawdown.
        
        Returns
        -------
        tuple
            (max_dd, peak_idx, trough_idx)
        """
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        
        max_dd = np.min(drawdown)
        trough_idx = np.argmin(drawdown)
        peak_idx = np.argmax(equity_curve[:trough_idx+1])
        
        return max_dd, peak_idx, trough_idx
    
    def drawdown_series(self, equity_curve: np.ndarray) -> np.ndarray:
        """Calculate drawdown series."""
        peak = np.maximum.accumulate(equity_curve)
        return (equity_curve - peak) / peak
    
    def max_drawdown_duration(self, equity_curve: np.ndarray) -> int:
        """
        Calculate maximum drawdown duration.
        
        Number of periods from peak to recovery.
        """
        peak = equity_curve[0]
        max_duration = 0
        current_duration = 0
        
        for value in equity_curve:
            if value >= peak:
                peak = value
                current_duration = 0
            else:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
        
        return max_duration
    
    # =========================================================================
    # RISK-ADJUSTED METRICS
    # =========================================================================
    
    def sharpe_ratio(self, returns: np.ndarray) -> float:
        """
        Calculate Sharpe Ratio.
        
        SR = (E[R] - Rf) / σ
        """
        excess_returns = returns - self.rf_period
        
        mean_excess = np.mean(excess_returns)
        std = np.std(excess_returns, ddof=1)
        
        if std == 0:
            return 0.0
        
        # Annualize
        sharpe = mean_excess / std * np.sqrt(self.periods_per_year)
        
        return sharpe
    
    def sortino_ratio(self, returns: np.ndarray, 
                     target: float = 0.0) -> float:
        """
        Calculate Sortino Ratio.
        
        Sortino = (E[R] - target) / σ_downside
        """
        excess = returns - target
        
        mean_excess = np.mean(excess)
        downside_std = self.downside_volatility(returns, target, annualize=False)
        
        if downside_std == 0:
            return 0.0
        
        sortino = mean_excess / downside_std * np.sqrt(self.periods_per_year)
        
        return sortino
    
    def calmar_ratio(self, equity_curve: np.ndarray) -> float:
        """
        Calculate Calmar Ratio.
        
        Calmar = CAGR / |Max Drawdown|
        """
        ann_ret = self.annualized_return(equity_curve)
        max_dd, _, _ = self.max_drawdown(equity_curve)
        
        if max_dd == 0:
            return 0.0
        
        return ann_ret / abs(max_dd)
    
    def omega_ratio(self, returns: np.ndarray, 
                   threshold: float = 0.0) -> float:
        """
        Calculate Omega Ratio.
        
        Omega = ∫(1-F(r))dr / ∫F(r)dr for r > threshold
        
        Where F is the CDF of returns.
        """
        above = returns[returns > threshold] - threshold
        below = threshold - returns[returns <= threshold]
        
        if np.sum(below) == 0:
            return float('inf')
        
        return np.sum(above) / np.sum(below)
    
    def information_ratio(self, returns: np.ndarray, 
                         benchmark_returns: np.ndarray) -> float:
        """
        Calculate Information Ratio.
        
        IR = E[R - B] / σ(R - B)
        
        Where B is benchmark returns.
        """
        active_returns = returns - benchmark_returns
        
        mean_active = np.mean(active_returns)
        std_active = np.std(active_returns, ddof=1)
        
        if std_active == 0:
            return 0.0
        
        return mean_active / std_active * np.sqrt(self.periods_per_year)
    
    # =========================================================================
    # DISTRIBUTION METRICS
    # =========================================================================
    
    def skewness(self, returns: np.ndarray) -> float:
        """
        Calculate skewness of returns.
        
        Positive skew = more extreme positive returns
        Negative skew = more extreme negative returns (tail risk)
        """
        return float(scipy_stats.skew(returns))
    
    def kurtosis(self, returns: np.ndarray) -> float:
        """
        Calculate excess kurtosis.
        
        >0 = fat tails (more extreme events)
        <0 = thin tails
        """
        return float(scipy_stats.kurtosis(returns))
    
    def var(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """
        Calculate Value at Risk.
        
        VaR = quantile at (1-confidence) level
        
        Parameters
        ----------
        returns : np.ndarray
            Return series.
        confidence : float
            Confidence level.
        
        Returns
        -------
        float
            VaR (as positive number representing loss).
        """
        return -np.percentile(returns, (1 - confidence) * 100)
    
    def cvar(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall).
        
        CVaR = E[R | R < VaR]
        
        Average loss when VaR is exceeded.
        """
        var = self.var(returns, confidence)
        tail = returns[returns < -var]
        
        if len(tail) == 0:
            return var
        
        return -np.mean(tail)
    
    # =========================================================================
    # TRADING METRICS
    # =========================================================================
    
    def calculate_trade_stats(self, trades: List[Dict]) -> Dict[str, float]:
        """
        Calculate trade-level statistics.
        
        Parameters
        ----------
        trades : list
            List of trade dictionaries with 'pnl' field.
        
        Returns
        -------
        dict
            Trade statistics.
        """
        if not trades:
            return {
                'win_rate': 0,
                'profit_factor': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'expectancy': 0,
                'total_trades': 0
            }
        
        pnls = np.array([t.get('pnl', 0) for t in trades])
        
        winners = pnls[pnls > 0]
        losers = pnls[pnls < 0]
        
        n_trades = len(pnls)
        n_winners = len(winners)
        n_losers = len(losers)
        
        win_rate = n_winners / n_trades if n_trades > 0 else 0
        
        gross_profit = np.sum(winners) if len(winners) > 0 else 0
        gross_loss = abs(np.sum(losers)) if len(losers) > 0 else 0
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        avg_win = np.mean(winners) if len(winners) > 0 else 0
        avg_loss = np.mean(losers) if len(losers) > 0 else 0
        
        # Expectancy = P(win) * avg_win + P(loss) * avg_loss
        expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
        
        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'expectancy': expectancy,
            'total_trades': n_trades,
            'n_winners': n_winners,
            'n_losers': n_losers,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss
        }
    
    # =========================================================================
    # FACTOR METRICS
    # =========================================================================
    
    def beta(self, returns: np.ndarray, 
            benchmark_returns: np.ndarray) -> float:
        """
        Calculate beta (market sensitivity).
        
        β = Cov(R, B) / Var(B)
        """
        if len(returns) != len(benchmark_returns):
            raise ValueError("Returns must have same length")
        
        covariance = np.cov(returns, benchmark_returns)[0, 1]
        benchmark_var = np.var(benchmark_returns, ddof=1)
        
        if benchmark_var == 0:
            return 0.0
        
        return covariance / benchmark_var
    
    def alpha(self, returns: np.ndarray,
             benchmark_returns: np.ndarray) -> float:
        """
        Calculate Jensen's Alpha.
        
        α = E[R] - Rf - β(E[B] - Rf)
        """
        beta = self.beta(returns, benchmark_returns)
        
        expected_return = np.mean(returns) * self.periods_per_year
        expected_benchmark = np.mean(benchmark_returns) * self.periods_per_year
        
        alpha = expected_return - self.risk_free_rate - beta * (expected_benchmark - self.risk_free_rate)
        
        return alpha
    
    def r_squared(self, returns: np.ndarray,
                 benchmark_returns: np.ndarray) -> float:
        """
        Calculate R-squared (how much variance explained by benchmark).
        """
        correlation = np.corrcoef(returns, benchmark_returns)[0, 1]
        return correlation ** 2
    
    # =========================================================================
    # COMPREHENSIVE ANALYSIS
    # =========================================================================
    
    def analyze(self, equity_curve: np.ndarray,
               benchmark: Optional[np.ndarray] = None,
               trades: Optional[List[Dict]] = None) -> PerformanceMetrics:
        """
        Comprehensive performance analysis.
        
        Parameters
        ----------
        equity_curve : np.ndarray
            Portfolio equity values.
        benchmark : np.ndarray, optional
            Benchmark equity curve.
        trades : list, optional
            List of trade dictionaries.
        
        Returns
        -------
        PerformanceMetrics
            Complete performance metrics.
        """
        returns = self.calculate_returns(equity_curve)
        
        # Calculate benchmark returns if provided
        if benchmark is not None:
            bench_returns = self.calculate_returns(benchmark)
        else:
            # Use zero as benchmark (absolute returns)
            bench_returns = np.zeros_like(returns)
        
        # Calculate all metrics
        max_dd, _, _ = self.max_drawdown(equity_curve)
        dd_series = self.drawdown_series(equity_curve)
        
        # Trade stats
        if trades:
            trade_stats = self.calculate_trade_stats(trades)
        else:
            trade_stats = {
                'win_rate': 0, 'profit_factor': 0, 
                'avg_win': 0, 'avg_loss': 0, 'expectancy': 0
            }
        
        return PerformanceMetrics(
            # Returns
            total_return=self.total_return(equity_curve),
            annualized_return=self.annualized_return(equity_curve),
            cagr=self.annualized_return(equity_curve),
            
            # Risk
            volatility=self.volatility(returns, annualize=False),
            annualized_volatility=self.volatility(returns, annualize=True),
            downside_volatility=self.downside_volatility(returns),
            max_drawdown=max_dd,
            avg_drawdown=np.mean(dd_series[dd_series < 0]) if np.any(dd_series < 0) else 0,
            max_drawdown_duration=self.max_drawdown_duration(equity_curve),
            
            # Risk-adjusted
            sharpe_ratio=self.sharpe_ratio(returns),
            sortino_ratio=self.sortino_ratio(returns),
            calmar_ratio=self.calmar_ratio(equity_curve),
            omega_ratio=self.omega_ratio(returns),
            information_ratio=self.information_ratio(returns, bench_returns),
            
            # Distribution
            skewness=self.skewness(returns),
            kurtosis=self.kurtosis(returns),
            var_95=self.var(returns, 0.95),
            cvar_95=self.cvar(returns, 0.95),
            
            # Trading
            win_rate=trade_stats['win_rate'],
            profit_factor=trade_stats['profit_factor'],
            avg_win=trade_stats['avg_win'],
            avg_loss=trade_stats['avg_loss'],
            expectancy=trade_stats['expectancy'],
            
            # Factor
            beta=self.beta(returns, bench_returns),
            alpha=self.alpha(returns, bench_returns),
            r_squared=self.r_squared(returns, bench_returns)
        )
    
    def generate_report(self, metrics: PerformanceMetrics) -> str:
        """Generate text performance report."""
        report = """
                                                                
                     PERFORMANCE REPORT                         
                                                                
  RETURNS                                                       
    Total Return:        {total_return:>12.2%}                         
    Annualized Return:   {ann_return:>12.2%}                         
    CAGR:                {cagr:>12.2%}                         
                                                                
  RISK                                                          
    Volatility (Ann.):   {volatility:>12.2%}                         
    Downside Vol:        {down_vol:>12.2%}                         
    Max Drawdown:        {max_dd:>12.2%}                         
    Max DD Duration:     {dd_dur:>12} periods                   
                                                                
  RISK-ADJUSTED                                                 
    Sharpe Ratio:        {sharpe:>12.2f}                               
    Sortino Ratio:       {sortino:>12.2f}                               
    Calmar Ratio:        {calmar:>12.2f}                               
    Omega Ratio:         {omega:>12.2f}                               
                                                                
  DISTRIBUTION                                                  
    Skewness:            {skew:>12.2f}                               
    Kurtosis:            {kurt:>12.2f}                               
    VaR (95%):           {var:>12.2%}                         
    CVaR (95%):          {cvar:>12.2%}                         
                                                                
  TRADING                                                       
    Win Rate:            {win_rate:>12.2%}                         
    Profit Factor:       {pf:>12.2f}                               
    Expectancy:          {expect:>12.2f}                               
                                                                
  FACTOR ANALYSIS                                               
    Beta:                {beta:>12.2f}                               
    Alpha (Ann.):        {alpha:>12.2%}                         
    R-Squared:           {r2:>12.2%}                         
                                                                
        """.format(
            total_return=metrics.total_return,
            ann_return=metrics.annualized_return,
            cagr=metrics.cagr,
            volatility=metrics.annualized_volatility,
            down_vol=metrics.downside_volatility,
            max_dd=metrics.max_drawdown,
            dd_dur=metrics.max_drawdown_duration,
            sharpe=metrics.sharpe_ratio,
            sortino=metrics.sortino_ratio,
            calmar=metrics.calmar_ratio,
            omega=metrics.omega_ratio,
            skew=metrics.skewness,
            kurt=metrics.kurtosis,
            var=metrics.var_95,
            cvar=metrics.cvar_95,
            win_rate=metrics.win_rate,
            pf=metrics.profit_factor,
            expect=metrics.expectancy,
            beta=metrics.beta,
            alpha=metrics.alpha,
            r2=metrics.r_squared
        )
        
        return report


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    import numpy as np
    
    print("=" * 60)
    print("PERFORMANCE ANALYTICS TEST")
    print("=" * 60)
    
    # Use REAL market data instead of synthetic
    try:
        from data.realtime_manager import get_data_manager
        dm = get_data_manager()
        
        spy_df = dm.get_historical_data_sync('SPY', '2021-01-01', '2024-12-31')
        qqq_df = dm.get_historical_data_sync('QQQ', '2021-01-01', '2024-12-31')
        
        if not spy_df.empty and not qqq_df.empty:
            daily_returns = spy_df['close'].pct_change().dropna().values
            equity_curve = 1_000_000 * np.cumprod(1 + daily_returns)
            
            benchmark_returns = qqq_df['close'].pct_change().dropna().values
            benchmark = 1_000_000 * np.cumprod(1 + benchmark_returns)
            
            print("  Using REAL market data (SPY vs QQQ)")
        else:
            raise Exception("No data")
    except Exception as e:
        print(f"  Real market data unavailable: {e}")
        print("  Performance analysis requires SPY and QQQ historical data")
        import sys; sys.exit(0)
    
    # Analyze
    analyzer = PerformanceAnalyzer(risk_free_rate=0.02)
    metrics = analyzer.analyze(equity_curve, benchmark)
    
    # Generate report
    report = analyzer.generate_report(metrics)
    print(report)
