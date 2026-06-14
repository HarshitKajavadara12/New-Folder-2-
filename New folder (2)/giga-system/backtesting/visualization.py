"""
GIGA SYSTEM - Backtest Visualization
Interactive charts for backtest analysis using Plotly
"""
from __future__ import annotations   # lazy annotation evaluation – fixes go.Figure NameError

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class BacktestVisualizer:
    """
    Backtest visualization toolkit.
    
    Features:
    - Equity curve plots
    - Drawdown analysis
    - Rolling metrics
    - Trade analysis
    - Risk decomposition
    """
    
    def __init__(self, template: str = "plotly_dark"):
        """
        Initialize visualizer.
        
        Parameters
        ----------
        template : str
            Plotly template ("plotly_dark", "plotly_white", etc.)
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly required for visualization")
        
        self.template = template
        self.colors = {
            'primary': '#00D4AA',
            'secondary': '#FF6B6B',
            'tertiary': '#4ECDC4',
            'background': '#1a1a2e',
            'positive': '#00ff88',
            'negative': '#ff4444'
        }
    
    # =========================================================================
    # EQUITY CURVE
    # =========================================================================
    
    def plot_equity_curve(self, 
                         timestamps: List[datetime],
                         equity: np.ndarray,
                         benchmark: Optional[np.ndarray] = None,
                         title: str = "Portfolio Equity") -> go.Figure:
        """
        Plot equity curve with optional benchmark.
        
        Parameters
        ----------
        timestamps : list
            List of timestamps.
        equity : np.ndarray
            Portfolio equity values.
        benchmark : np.ndarray, optional
            Benchmark equity values.
        title : str
            Chart title.
        
        Returns
        -------
        go.Figure
            Plotly figure.
        """
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3],
            subplot_titles=('Equity Curve', 'Drawdown')
        )
        
        # Equity curve
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=equity,
                name='Portfolio',
                line=dict(color=self.colors['primary'], width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 212, 170, 0.1)'
            ),
            row=1, col=1
        )
        
        # Benchmark
        if benchmark is not None:
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=benchmark,
                    name='Benchmark',
                    line=dict(color=self.colors['secondary'], width=1, dash='dash')
                ),
                row=1, col=1
            )
        
        # Drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak * 100
        
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=drawdown,
                name='Drawdown',
                line=dict(color=self.colors['negative'], width=1),
                fill='tozeroy',
                fillcolor='rgba(255, 68, 68, 0.3)'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title=title,
            template=self.template,
            height=600,
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        
        fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        
        return fig
    
    # =========================================================================
    # RETURNS ANALYSIS
    # =========================================================================
    
    def plot_returns_distribution(self,
                                  returns: np.ndarray,
                                  title: str = "Returns Distribution") -> go.Figure:
        """
        Plot returns distribution with statistics.
        
        Parameters
        ----------
        returns : np.ndarray
            Period returns.
        title : str
            Chart title.
        
        Returns
        -------
        go.Figure
            Plotly figure.
        """
        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.6, 0.4],
            subplot_titles=('Return Distribution', 'Monthly Returns')
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(
                x=returns * 100,
                nbinsx=50,
                name='Returns',
                marker_color=self.colors['primary'],
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Add normal distribution overlay
        mean = np.mean(returns) * 100
        std = np.std(returns) * 100
        x_range = np.linspace(mean - 4*std, mean + 4*std, 100)
        normal_dist = (1/(std * np.sqrt(2*np.pi))) * np.exp(-0.5*((x_range-mean)/std)**2)
        
        # Scale to histogram
        hist_scale = len(returns) * (x_range[1] - x_range[0])
        
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=normal_dist * hist_scale,
                name='Normal',
                line=dict(color=self.colors['secondary'], width=2)
            ),
            row=1, col=1
        )
        
        # Add VaR line
        var_95 = np.percentile(returns, 5) * 100
        fig.add_vline(
            x=var_95,
            line_dash="dash",
            line_color=self.colors['negative'],
            annotation_text=f"VaR 95%: {var_95:.2f}%",
            row=1, col=1
        )
        
        # Cumulative returns by month (simplified)
        n_months = len(returns) // 21  # Approximate
        if n_months > 0:
            monthly_returns = []
            for i in range(n_months):
                start = i * 21
                end = min((i + 1) * 21, len(returns))
                monthly_ret = np.prod(1 + returns[start:end]) - 1
                monthly_returns.append(monthly_ret * 100)
            
            colors = [self.colors['positive'] if r > 0 else self.colors['negative'] 
                     for r in monthly_returns]
            
            fig.add_trace(
                go.Bar(
                    x=list(range(1, len(monthly_returns) + 1)),
                    y=monthly_returns,
                    name='Monthly Returns',
                    marker_color=colors
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            title=title,
            template=self.template,
            height=400,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Return (%)", row=1, col=1)
        fig.update_xaxes(title_text="Month", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_yaxes(title_text="Return (%)", row=1, col=2)
        
        return fig
    
    # =========================================================================
    # ROLLING METRICS
    # =========================================================================
    
    def plot_rolling_metrics(self,
                            timestamps: List[datetime],
                            returns: np.ndarray,
                            window: int = 63,
                            title: str = "Rolling Metrics") -> go.Figure:
        """
        Plot rolling Sharpe, volatility, etc.
        
        Parameters
        ----------
        timestamps : list
            Timestamps.
        returns : np.ndarray
            Returns array.
        window : int
            Rolling window (63 = ~3 months).
        title : str
            Chart title.
        
        Returns
        -------
        go.Figure
            Plotly figure.
        """
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Rolling Sharpe (3M)', 'Rolling Volatility (3M)', 
                           'Rolling Beta (3M)')
        )
        
        # Calculate rolling metrics
        n = len(returns)
        if n < window:
            return fig
        
        rolling_sharpe = []
        rolling_vol = []
        rolling_return = []
        
        for i in range(window, n):
            window_returns = returns[i-window:i]
            
            # Sharpe
            mean_ret = np.mean(window_returns) * 252
            std_ret = np.std(window_returns) * np.sqrt(252)
            sharpe = mean_ret / std_ret if std_ret > 0 else 0
            rolling_sharpe.append(sharpe)
            
            # Volatility
            rolling_vol.append(std_ret)
            
            # Return
            rolling_return.append(mean_ret)
        
        plot_timestamps = timestamps[window:] if len(timestamps) > window else timestamps
        
        # Rolling Sharpe
        fig.add_trace(
            go.Scatter(
                x=plot_timestamps,
                y=rolling_sharpe,
                name='Rolling Sharpe',
                line=dict(color=self.colors['primary'], width=2)
            ),
            row=1, col=1
        )
        fig.add_hline(y=0, line_dash='dash', line_color='gray', row=1, col=1)
        fig.add_hline(y=1, line_dash='dot', line_color=self.colors['positive'], row=1, col=1)
        
        # Rolling Volatility
        fig.add_trace(
            go.Scatter(
                x=plot_timestamps,
                y=[v * 100 for v in rolling_vol],
                name='Rolling Vol',
                line=dict(color=self.colors['secondary'], width=2),
                fill='tozeroy',
                fillcolor='rgba(255, 107, 107, 0.2)'
            ),
            row=2, col=1
        )
        
        # Rolling Return
        fig.add_trace(
            go.Scatter(
                x=plot_timestamps,
                y=[r * 100 for r in rolling_return],
                name='Rolling Return',
                line=dict(color=self.colors['tertiary'], width=2)
            ),
            row=3, col=1
        )
        fig.add_hline(y=0, line_dash='dash', line_color='gray', row=3, col=1)
        
        fig.update_layout(
            title=title,
            template=self.template,
            height=700,
            showlegend=True
        )
        
        fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=1)
        fig.update_yaxes(title_text="Volatility (%)", row=2, col=1)
        fig.update_yaxes(title_text="Return (%)", row=3, col=1)
        
        return fig
    
    # =========================================================================
    # TRADE ANALYSIS
    # =========================================================================
    
    def plot_trade_analysis(self,
                           trades: List[Dict],
                           title: str = "Trade Analysis") -> go.Figure:
        """
        Plot trade-level analysis.
        
        Parameters
        ----------
        trades : list
            List of trade dictionaries with 'pnl', 'timestamp', etc.
        title : str
            Chart title.
        
        Returns
        -------
        go.Figure
            Plotly figure.
        """
        if not trades:
            return go.Figure()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Cumulative P&L', 'P&L by Trade',
                           'Win/Loss Distribution', 'Trade Duration')
        )
        
        pnls = [t.get('pnl', 0) for t in trades]
        cumulative_pnl = np.cumsum(pnls)
        
        # Cumulative P&L
        fig.add_trace(
            go.Scatter(
                x=list(range(len(cumulative_pnl))),
                y=cumulative_pnl,
                name='Cumulative P&L',
                line=dict(color=self.colors['primary'], width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 212, 170, 0.2)'
            ),
            row=1, col=1
        )
        
        # P&L by trade
        colors = [self.colors['positive'] if p > 0 else self.colors['negative'] 
                 for p in pnls]
        
        fig.add_trace(
            go.Bar(
                x=list(range(len(pnls))),
                y=pnls,
                name='Trade P&L',
                marker_color=colors
            ),
            row=1, col=2
        )
        
        # Win/Loss distribution
        winners = [p for p in pnls if p > 0]
        losers = [abs(p) for p in pnls if p < 0]
        
        fig.add_trace(
            go.Box(
                y=winners,
                name='Winners',
                marker_color=self.colors['positive']
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Box(
                y=losers,
                name='Losers',
                marker_color=self.colors['negative']
            ),
            row=2, col=1
        )
        
        # Trade duration (if available)
        durations = [t.get('duration', 1) for t in trades]
        fig.add_trace(
            go.Histogram(
                x=durations,
                name='Duration',
                marker_color=self.colors['tertiary']
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=title,
            template=self.template,
            height=600,
            showlegend=True
        )
        
        return fig
    
    # =========================================================================
    # COMPREHENSIVE DASHBOARD
    # =========================================================================
    
    def create_dashboard(self,
                        timestamps: List[datetime],
                        equity: np.ndarray,
                        returns: np.ndarray,
                        trades: Optional[List[Dict]] = None,
                        benchmark: Optional[np.ndarray] = None) -> go.Figure:
        """
        Create comprehensive backtest dashboard.
        
        Parameters
        ----------
        timestamps : list
            Timestamps.
        equity : np.ndarray
            Equity curve.
        returns : np.ndarray
            Returns array.
        trades : list, optional
            Trade list.
        benchmark : np.ndarray, optional
            Benchmark equity.
        
        Returns
        -------
        go.Figure
            Dashboard figure.
        """
        fig = make_subplots(
            rows=3, cols=2,
            row_heights=[0.4, 0.3, 0.3],
            column_widths=[0.6, 0.4],
            specs=[
                [{"colspan": 2}, None],
                [{}, {}],
                [{}, {}]
            ],
            subplot_titles=(
                'Portfolio Performance',
                'Drawdown', 'Returns Distribution',
                'Rolling Sharpe', 'Monthly Returns'
            ),
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        # Row 1: Equity curve (full width)
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=equity,
                name='Portfolio',
                line=dict(color=self.colors['primary'], width=2)
            ),
            row=1, col=1
        )
        
        if benchmark is not None:
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=benchmark,
                    name='Benchmark',
                    line=dict(color=self.colors['secondary'], width=1, dash='dash')
                ),
                row=1, col=1
            )
        
        # Row 2, Col 1: Drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak * 100
        
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=drawdown,
                name='Drawdown',
                line=dict(color=self.colors['negative'], width=1),
                fill='tozeroy',
                fillcolor='rgba(255, 68, 68, 0.3)'
            ),
            row=2, col=1
        )
        
        # Row 2, Col 2: Returns distribution
        fig.add_trace(
            go.Histogram(
                x=returns * 100,
                nbinsx=30,
                name='Returns',
                marker_color=self.colors['primary'],
                opacity=0.7
            ),
            row=2, col=2
        )
        
        # Row 3, Col 1: Rolling Sharpe
        window = min(63, len(returns) // 2)
        if window > 10:
            rolling_sharpe = []
            for i in range(window, len(returns)):
                w_ret = returns[i-window:i]
                sharpe = np.mean(w_ret) / np.std(w_ret) * np.sqrt(252) if np.std(w_ret) > 0 else 0
                rolling_sharpe.append(sharpe)
            
            fig.add_trace(
                go.Scatter(
                    x=timestamps[window:],
                    y=rolling_sharpe,
                    name='Rolling Sharpe',
                    line=dict(color=self.colors['tertiary'], width=2)
                ),
                row=3, col=1
            )
            fig.add_hline(y=0, line_dash='dash', line_color='gray', row=3, col=1)
        
        # Row 3, Col 2: Monthly returns
        n_months = len(returns) // 21
        if n_months > 1:
            monthly = []
            for i in range(n_months):
                start = i * 21
                end = min((i + 1) * 21, len(returns))
                monthly.append((np.prod(1 + returns[start:end]) - 1) * 100)
            
            colors = [self.colors['positive'] if m > 0 else self.colors['negative'] 
                     for m in monthly]
            
            fig.add_trace(
                go.Bar(
                    x=list(range(1, len(monthly) + 1)),
                    y=monthly,
                    name='Monthly',
                    marker_color=colors
                ),
                row=3, col=2
            )
        
        # Layout
        fig.update_layout(
            title=dict(
                text='<b>GIGA SYSTEM - Backtest Analysis Dashboard</b>',
                font=dict(size=20)
            ),
            template=self.template,
            height=900,
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            )
        )
        
        # Axis labels
        fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=2)
        fig.update_yaxes(title_text="Sharpe Ratio", row=3, col=1)
        fig.update_yaxes(title_text="Return (%)", row=3, col=2)
        
        fig.update_xaxes(title_text="Return (%)", row=2, col=2)
        fig.update_xaxes(title_text="Month", row=3, col=2)
        
        return fig


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    import numpy as np
    from datetime import datetime, timedelta

    def _run_demo():
        print("=" * 60)
        print("BACKTEST VISUALIZATION TEST")
        print("=" * 60)

        if not PLOTLY_AVAILABLE:
            print("Plotly not available. Install with: pip install plotly")
            return

        try:
            from data.realtime_manager import get_data_manager
            dm = get_data_manager()

            hist_df = dm.get_historical_data_sync('SPY', '2022-01-01', '2024-12-31')

            if hist_df.empty:
                raise Exception("No data")

            timestamps = hist_df['timestamp'].tolist()
            returns = hist_df['close'].pct_change().dropna().values
            equity = 1_000_000 * np.cumprod(1 + returns)

            bench_df = dm.get_historical_data_sync('QQQ', '2022-01-01', '2024-12-31')
            bench_returns = bench_df['close'].pct_change().dropna().values
            benchmark = 1_000_000 * np.cumprod(1 + bench_returns)

            print("  Using REAL market data (SPY vs QQQ)")
        except Exception as e:
            print(f"  Real market data unavailable: {e}")
            print("  Backtest visualization requires SPY and QQQ historical data")
            return

        viz = BacktestVisualizer()
        fig = viz.create_dashboard(
            timestamps=timestamps,
            equity=equity,
            returns=returns,
            benchmark=benchmark
        )
        fig.write_html("backtest_dashboard.html")
        print("Dashboard saved to backtest_dashboard.html")

        equity_fig = viz.plot_equity_curve(timestamps, equity, benchmark)
        returns_fig = viz.plot_returns_distribution(returns)
        rolling_fig = viz.plot_rolling_metrics(timestamps, returns)
        print("Individual plots created successfully")

    _run_demo()