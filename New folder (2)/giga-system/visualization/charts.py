"""
GIGA SYSTEM - Interactive Charts
Greek Intelligence for Global Analysis

Advanced interactive charting components for financial data visualization.
Built on Plotly for high-performance, interactive charts with real-time updates.

Key Features:
- Real-time candlestick charts with volume
- Options Greeks surface plots
- Portfolio performance attribution charts
- Risk heatmaps and correlation matrices
- Quantum algorithm visualization
- 3D portfolio optimization surfaces

Chart Types:
- Market data: OHLCV, indicators, patterns
- Options: Payoff diagrams, Greeks surfaces, volatility smiles
- Portfolio: Allocation, performance, risk attribution
- Backtesting: Equity curves, drawdown analysis
- Quantum: Circuit diagrams, amplitude plots, optimization landscapes
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import warnings

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.figure_factory as ff
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available - charts will not be generated")

try:
    from ..utils.math_helpers import black_scholes_call, black_scholes_put
except ImportError:
    # Fallback math functions
    def black_scholes_call(S, K, T, r, sigma):
        return max(S - K, 0)  # Simplified
    
    def black_scholes_put(S, K, T, r, sigma):
        return max(K - S, 0)  # Simplified


# =============================================================================
# COLOR CONFIGURATION
# =============================================================================

CHART_COLORS = {
    'up': '#00ff88',
    'down': '#ff4444',
    'primary': '#00D4AA',
    'secondary': '#FF6B6B',
    'grid': '#333333',
    'background': '#0d1117',
    'paper': '#161b22'
}

DEFAULT_LAYOUT = {
    'template': 'plotly_dark',
    'paper_bgcolor': CHART_COLORS['paper'],
    'plot_bgcolor': CHART_COLORS['background'],
    'font': {'color': '#ffffff'},
    'margin': dict(l=50, r=50, t=50, b=50)
}


# =============================================================================
# PRICE CHARTS
# =============================================================================

def candlestick_chart(df: pd.DataFrame,
                     title: str = "Price Chart",
                     show_volume: bool = True,
                     indicators: Optional[List[Dict]] = None,
                     height: int = 600) -> go.Figure:
    """
    Create candlestick chart with optional indicators.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with columns: open, high, low, close, volume.
    title : str
        Chart title.
    show_volume : bool
        Whether to show volume bars.
    indicators : list, optional
        List of indicator dicts with 'name', 'data', 'color'.
    height : int
        Chart height.
    
    Returns
    -------
    go.Figure
        Plotly figure.
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly required for candlestick charts")
    
    row_heights = [0.7, 0.3] if show_volume else [1.0]
    
    fig = make_subplots(
        rows=2 if show_volume else 1,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights
    )
    
    # Candlestick
    colors = np.where(df['close'] >= df['open'], 
                     CHART_COLORS['up'], CHART_COLORS['down'])
    
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price',
        increasing_line_color=CHART_COLORS['up'],
        decreasing_line_color=CHART_COLORS['down']
    ), row=1, col=1)
    
    # Add indicators
    if indicators:
        for ind in indicators:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=ind['data'],
                name=ind['name'],
                line=dict(color=ind.get('color', CHART_COLORS['primary']),
                         width=ind.get('width', 1))
            ), row=1, col=1)
    
    # Volume bars
    if show_volume and 'volume' in df.columns:
        fig.add_trace(go.Bar(
            x=df.index,
            y=df['volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.5
        ), row=2, col=1)
    
    fig.update_layout(
        **DEFAULT_LAYOUT,
        height=height,
        title=title,
        xaxis_rangeslider_visible=False,
        showlegend=True
    )
    
    return fig


def multi_asset_chart(prices: Dict[str, pd.Series],
                     normalize: bool = True,
                     title: str = "Multi-Asset Performance",
                     height: int = 500) -> go.Figure:
    """
    Create multi-asset comparison chart.
    
    Parameters
    ----------
    prices : dict
        Dict of symbol -> price series.
    normalize : bool
        Normalize to percentage returns.
    title : str
        Chart title.
    height : int
        Chart height.
    
    Returns
    -------
    go.Figure
        Plotly figure.
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly required")
    
    fig = go.Figure()
    
    colors = ['#00D4AA', '#FF6B6B', '#4ECDC4', '#FFE66D', '#9B59B6',
              '#3498DB', '#E74C3C', '#2ECC71', '#F39C12', '#1ABC9C']
    
    for i, (symbol, series) in enumerate(prices.items()):
        if normalize:
            data = (series / series.iloc[0] - 1) * 100
            yaxis = 'Return (%)'
        else:
            data = series
            yaxis = 'Price'
        
        fig.add_trace(go.Scatter(
            x=series.index,
            y=data,
            name=symbol,
            line=dict(color=colors[i % len(colors)], width=2)
        ))
    
    fig.update_layout(
        **DEFAULT_LAYOUT,
        height=height,
        title=title,
        yaxis_title=yaxis,
        hovermode='x unified'
    )
    
    return fig


# =============================================================================
# OPTIONS CHARTS
# =============================================================================

def volatility_surface(strikes: np.ndarray,
                      maturities: np.ndarray,
                      ivs: np.ndarray,
                      title: str = "Implied Volatility Surface",
                      height: int = 600) -> go.Figure:
    """
    Create 3D volatility surface.
    
    Parameters
    ----------
    strikes : np.ndarray
        Strike prices.
    maturities : np.ndarray
        Time to maturity.
    ivs : np.ndarray
        Implied volatilities (2D matrix).
    title : str
        Chart title.
    height : int
        Chart height.
    
    Returns
    -------
    go.Figure
        Plotly figure.
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly required")
    
    fig = go.Figure(data=[go.Surface(
        x=strikes,
        y=maturities,
        z=ivs * 100,
        colorscale='Viridis',
        colorbar=dict(title='IV (%)')
    )])
    
    fig.update_layout(
        **DEFAULT_LAYOUT,
        height=height,
        title=title,
        scene=dict(
            xaxis_title='Strike',
            yaxis_title='Maturity (days)',
            zaxis_title='IV (%)'
        )
    )
    
    return fig


def volatility_smile(strikes: np.ndarray,
                    ivs: np.ndarray,
                    spot: float,
                    title: str = "Volatility Smile",
                    height: int = 400) -> go.Figure:
    """
    Create volatility smile chart.
    
    Parameters
    ----------
    strikes : np.ndarray
        Strike prices.
    ivs : np.ndarray
        Implied volatilities.
    spot : float
        Current spot price.
    title : str
        Chart title.
    height : int
        Chart height.
    
    Returns
    -------
    go.Figure
        Plotly figure.
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly required")
    
    moneyness = strikes / spot
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=moneyness,
        y=ivs * 100,
        mode='lines+markers',
        name='IV',
        line=dict(color=CHART_COLORS['primary'], width=2)
    ))
    
    fig.add_vline(x=1.0, line_dash='dash', 
                 line_color=CHART_COLORS['secondary'],
                 annotation_text='ATM')
    
    fig.update_layout(
        **DEFAULT_LAYOUT,
        height=height,
        title=title,
        xaxis_title='Moneyness (K/S)',
        yaxis_title='Implied Volatility (%)'
    )
    
    return fig


def greeks_chart(strikes: np.ndarray,
                greeks: Dict[str, np.ndarray],
                spot: float,
                height: int = 500) -> go.Figure:
    """
    Create Greeks multi-panel chart.
    
    Parameters
    ----------
    strikes : np.ndarray
        Strike prices.
    greeks : dict
        Dict with 'delta', 'gamma', 'theta', 'vega'.
    spot : float
        Current spot price.
    height : int
        Chart height.
    
    Returns
    -------
    go.Figure
        Plotly figure.
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly required")
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Delta (Δ)', 'Gamma (Γ)', 'Theta (Θ)', 'Vega (V)']
    )
    
    greek_configs = [
        ('delta', 1, 1, '#00D4AA'),
        ('gamma', 1, 2, '#FF6B6B'),
        ('theta', 2, 1, '#4ECDC4'),
        ('vega', 2, 2, '#FFE66D')
    ]
    
    for name, row, col, color in greek_configs:
        if name in greeks:
            fig.add_trace(go.Scatter(
                x=strikes,
                y=greeks[name],
                name=name.capitalize(),
                line=dict(color=color, width=2)
            ), row=row, col=col)
            
            # ATM line
            fig.add_vline(x=spot, line_dash='dot', 
                         line_color='white', opacity=0.3,
                         row=row, col=col)
    
    fig.update_layout(
        **DEFAULT_LAYOUT,
        height=height,
        title='Option Greeks Analysis',
        showlegend=False
    )
    
    return fig


def payoff_diagram(strikes: np.ndarray,
                  payoffs: List[Dict],
                  spot_range: Tuple[float, float] = None,
                  height: int = 400) -> go.Figure:
    """
    Create options payoff diagram.
    
    Parameters
    ----------
    strikes : np.ndarray
        Strike price range.
    payoffs : list
        List of payoff dicts with 'name', 'values', 'color'.
    spot_range : tuple, optional
        (min, max) spot price range.
    height : int
        Chart height.
    
    Returns
    -------
    go.Figure
        Plotly figure.
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly required")
    
    fig = go.Figure()
    
    for payoff in payoffs:
        fig.add_trace(go.Scatter(
            x=strikes,
            y=payoff['values'],
            name=payoff.get('name', 'Payoff'),
            line=dict(color=payoff.get('color', CHART_COLORS['primary']), width=2),
            fill='tozeroy' if payoff.get('fill', False) else None
        ))
    
    fig.add_hline(y=0, line_dash='dash', line_color='white', opacity=0.3)
    
    fig.update_layout(
        **DEFAULT_LAYOUT,
        height=height,
        title='Payoff Diagram',
        xaxis_title='Spot Price at Expiry',
        yaxis_title='P&L'
    )
    
    return fig


# =============================================================================
# RISK CHARTS
# =============================================================================

def var_chart(returns: np.ndarray,
             var_levels: Dict[str, float],
             title: str = "Value at Risk Analysis",
             height: int = 400) -> go.Figure:
    """
    Create VaR visualization chart.
    
    Parameters
    ----------
    returns : np.ndarray
        Historical returns.
    var_levels : dict
        Dict with confidence levels and VaR values.
    title : str
        Chart title.
    height : int
        Chart height.
    
    Returns
    -------
    go.Figure
        Plotly figure.
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly required")
    
    fig = go.Figure()
    
    # Histogram
    fig.add_trace(go.Histogram(
        x=returns * 100,
        nbinsx=50,
        name='Returns',
        marker_color=CHART_COLORS['primary'],
        opacity=0.6
    ))
    
    # VaR lines
    colors = ['#FFE66D', '#FF6B6B', '#FF0000']
    for i, (level, var_val) in enumerate(var_levels.items()):
        fig.add_vline(
            x=-var_val * 100,
            line_dash='dash',
            line_color=colors[i % len(colors)],
            annotation_text=f'{level}: {-var_val*100:.2f}%'
        )
    
    fig.update_layout(
        **DEFAULT_LAYOUT,
        height=height,
        title=title,
        xaxis_title='Return (%)',
        yaxis_title='Frequency',
        bargap=0.1
    )
    
    return fig


def risk_decomposition_chart(risk_contrib: Dict[str, float],
                            title: str = "Risk Contribution",
                            height: int = 400) -> go.Figure:
    """
    Create risk decomposition chart.
    
    Parameters
    ----------
    risk_contrib : dict
        Asset -> risk contribution percentage.
    title : str
        Chart title.
    height : int
        Chart height.
    
    Returns
    -------
    go.Figure
        Plotly figure.
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly required")
    
    assets = list(risk_contrib.keys())
    values = list(risk_contrib.values())
    
    colors = ['#00D4AA', '#FF6B6B', '#4ECDC4', '#FFE66D', '#9B59B6',
              '#3498DB', '#E74C3C', '#2ECC71', '#F39C12', '#1ABC9C']
    
    fig = go.Figure(data=[go.Bar(
        x=assets,
        y=[v * 100 for v in values],
        marker_color=colors[:len(assets)]
    )])
    
    fig.update_layout(
        **DEFAULT_LAYOUT,
        height=height,
        title=title,
        yaxis_title='Risk Contribution (%)'
    )
    
    return fig


# =============================================================================
# PORTFOLIO CHARTS
# =============================================================================

def efficient_frontier(portfolios: List[Dict],
                      optimal: Optional[Dict] = None,
                      height: int = 500) -> go.Figure:
    """
    Create efficient frontier chart.
    
    Parameters
    ----------
    portfolios : list
        List of portfolio dicts with 'return', 'risk', 'sharpe'.
    optimal : dict, optional
        Optimal portfolio point.
    height : int
        Chart height.
    
    Returns
    -------
    go.Figure
        Plotly figure.
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly required")
    
    risks = [p['risk'] * 100 for p in portfolios]
    returns = [p['return'] * 100 for p in portfolios]
    sharpes = [p.get('sharpe', 0) for p in portfolios]
    
    fig = go.Figure()
    
    # Efficient frontier
    fig.add_trace(go.Scatter(
        x=risks,
        y=returns,
        mode='markers',
        marker=dict(
            size=8,
            color=sharpes,
            colorscale='Viridis',
            colorbar=dict(title='Sharpe')
        ),
        name='Portfolios'
    ))
    
    # Optimal portfolio
    if optimal:
        fig.add_trace(go.Scatter(
            x=[optimal['risk'] * 100],
            y=[optimal['return'] * 100],
            mode='markers',
            marker=dict(size=15, color=CHART_COLORS['secondary'], symbol='star'),
            name='Optimal'
        ))
    
    fig.update_layout(
        **DEFAULT_LAYOUT,
        height=height,
        title='Efficient Frontier',
        xaxis_title='Risk (Volatility %)',
        yaxis_title='Expected Return (%)'
    )
    
    return fig


def weights_timeline(dates: List[datetime],
                    weights_history: List[Dict[str, float]],
                    height: int = 400) -> go.Figure:
    """
    Create portfolio weights timeline.
    
    Parameters
    ----------
    dates : list
        Rebalancing dates.
    weights_history : list
        List of weight dicts at each date.
    height : int
        Chart height.
    
    Returns
    -------
    go.Figure
        Plotly figure.
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly required")
    
    # Get all assets
    all_assets = set()
    for w in weights_history:
        all_assets.update(w.keys())
    all_assets = sorted(all_assets)
    
    fig = go.Figure()
    
    colors = ['#00D4AA', '#FF6B6B', '#4ECDC4', '#FFE66D', '#9B59B6',
              '#3498DB', '#E74C3C', '#2ECC71', '#F39C12', '#1ABC9C']
    
    for i, asset in enumerate(all_assets):
        values = [w.get(asset, 0) * 100 for w in weights_history]
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=values,
            name=asset,
            mode='lines',
            line=dict(color=colors[i % len(colors)], width=2),
            stackgroup='one',
            groupnorm='percent'
        ))
    
    fig.update_layout(
        **DEFAULT_LAYOUT,
        height=height,
        title='Portfolio Weights Over Time',
        yaxis_title='Weight (%)',
        hovermode='x unified'
    )
    
    return fig


# =============================================================================
# BACKTEST CHARTS
# =============================================================================

def backtest_results_chart(timestamps: List[datetime],
                          equity: np.ndarray,
                          benchmark: Optional[np.ndarray] = None,
                          trades: Optional[List[Dict]] = None,
                          height: int = 700) -> go.Figure:
    """
    Create comprehensive backtest results chart.
    
    Parameters
    ----------
    timestamps : list
        List of timestamps.
    equity : np.ndarray
        Equity curve.
    benchmark : np.ndarray, optional
        Benchmark equity.
    trades : list, optional
        List of trade dicts with 'timestamp', 'side', 'pnl'.
    height : int
        Chart height.
    
    Returns
    -------
    go.Figure
        Plotly figure.
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly required")
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=['Equity Curve', 'Drawdown', 'Daily Returns']
    )
    
    # Equity curve
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=equity,
        name='Portfolio',
        line=dict(color=CHART_COLORS['primary'], width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 212, 170, 0.1)'
    ), row=1, col=1)
    
    if benchmark is not None:
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=benchmark,
            name='Benchmark',
            line=dict(color=CHART_COLORS['secondary'], width=1, dash='dash')
        ), row=1, col=1)
    
    # Trade markers
    if trades:
        buy_times = [t['timestamp'] for t in trades if t['side'] == 'buy']
        buy_values = [equity[timestamps.index(t['timestamp'])] 
                     for t in trades if t['side'] == 'buy' and t['timestamp'] in timestamps]
        
        sell_times = [t['timestamp'] for t in trades if t['side'] == 'sell']
        sell_values = [equity[timestamps.index(t['timestamp'])] 
                      for t in trades if t['side'] == 'sell' and t['timestamp'] in timestamps]
        
        if buy_times:
            fig.add_trace(go.Scatter(
                x=buy_times[:len(buy_values)],
                y=buy_values,
                mode='markers',
                marker=dict(symbol='triangle-up', size=10, color=CHART_COLORS['up']),
                name='Buy'
            ), row=1, col=1)
        
        if sell_times:
            fig.add_trace(go.Scatter(
                x=sell_times[:len(sell_values)],
                y=sell_values,
                mode='markers',
                marker=dict(symbol='triangle-down', size=10, color=CHART_COLORS['down']),
                name='Sell'
            ), row=1, col=1)
    
    # Drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak * 100
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=drawdown,
        name='Drawdown',
        line=dict(color=CHART_COLORS['down'], width=1),
        fill='tozeroy',
        fillcolor='rgba(255, 68, 68, 0.3)',
        showlegend=False
    ), row=2, col=1)
    
    # Daily returns
    returns = np.diff(equity) / equity[:-1] * 100
    colors = [CHART_COLORS['up'] if r >= 0 else CHART_COLORS['down'] for r in returns]
    
    fig.add_trace(go.Bar(
        x=timestamps[1:],
        y=returns,
        name='Returns',
        marker_color=colors,
        showlegend=False
    ), row=3, col=1)
    
    fig.update_layout(
        **DEFAULT_LAYOUT,
        height=height,
        title='Backtest Results',
        hovermode='x unified'
    )
    
    return fig


def monthly_returns_heatmap(returns: pd.Series,
                           title: str = "Monthly Returns Heatmap",
                           height: int = 400) -> go.Figure:
    """
    Create monthly returns heatmap.
    
    Parameters
    ----------
    returns : pd.Series
        Daily returns with datetime index.
    title : str
        Chart title.
    height : int
        Chart height.
    
    Returns
    -------
    go.Figure
        Plotly figure.
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly required")
    
    # Aggregate to monthly
    monthly = returns.groupby([returns.index.year, returns.index.month]).sum()
    
    # Pivot to matrix
    years = sorted(set(returns.index.year))
    months = list(range(1, 13))
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    z = np.zeros((len(years), 12))
    for (year, month), ret in monthly.items():
        y_idx = years.index(year)
        m_idx = month - 1
        z[y_idx, m_idx] = ret * 100
    
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=month_names,
        y=[str(y) for y in years],
        colorscale='RdYlGn',
        zmid=0,
        colorbar=dict(title='Return (%)')
    ))
    
    # Add annotations
    for i in range(len(years)):
        for j in range(12):
            fig.add_annotation(
                x=month_names[j],
                y=str(years[i]),
                text=f'{z[i,j]:.1f}%',
                showarrow=False,
                font=dict(color='white' if abs(z[i,j]) > 5 else 'black', size=10)
            )
    
    fig.update_layout(
        **DEFAULT_LAYOUT,
        height=height,
        title=title
    )
    
    return fig


# =============================================================================
# QUANTUM CHARTS
# =============================================================================

def quantum_circuit_diagram(circuit_data: Dict,
                           height: int = 400) -> go.Figure:
    """
    Create quantum circuit visualization.
    
    Parameters
    ----------
    circuit_data : dict
        Circuit information with 'qubits', 'gates'.
    height : int
        Chart height.
    
    Returns
    -------
    go.Figure
        Plotly figure.
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly required")
    
    n_qubits = circuit_data.get('qubits', 4)
    gates = circuit_data.get('gates', [])
    
    fig = go.Figure()
    
    # Draw qubit lines
    for i in range(n_qubits):
        fig.add_trace(go.Scatter(
            x=[0, 10],
            y=[i, i],
            mode='lines',
            line=dict(color='white', width=1),
            showlegend=False
        ))
        
        # Qubit labels
        fig.add_annotation(
            x=-0.5, y=i,
            text=f'q{i}',
            showarrow=False,
            font=dict(color='white')
        )
    
    # Draw gates
    for gate in gates:
        qubit = gate.get('qubit', 0)
        position = gate.get('position', 0)
        gate_type = gate.get('type', 'H')
        
        fig.add_shape(
            type='rect',
            x0=position - 0.3, x1=position + 0.3,
            y0=qubit - 0.3, y1=qubit + 0.3,
            fillcolor=CHART_COLORS['primary'],
            line=dict(color='white')
        )
        
        fig.add_annotation(
            x=position, y=qubit,
            text=gate_type,
            showarrow=False,
            font=dict(color='white', size=12)
        )
    
    fig.update_layout(
        **DEFAULT_LAYOUT,
        height=height,
        title='Quantum Circuit',
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, showticklabels=False)
    )
    
    return fig


def quantum_probability_chart(states: List[str],
                             probabilities: np.ndarray,
                             title: str = "Quantum State Probabilities",
                             height: int = 400) -> go.Figure:
    """
    Create quantum state probability bar chart.
    
    Parameters
    ----------
    states : list
        State labels (e.g., '|00>', '|01>').
    probabilities : np.ndarray
        State probabilities.
    title : str
        Chart title.
    height : int
        Chart height.
    
    Returns
    -------
    go.Figure
        Plotly figure.
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly required")
    
    fig = go.Figure(data=[go.Bar(
        x=states,
        y=probabilities * 100,
        marker_color=CHART_COLORS['primary']
    )])
    
    fig.update_layout(
        **DEFAULT_LAYOUT,
        height=height,
        title=title,
        xaxis_title='Quantum State',
        yaxis_title='Probability (%)'
    )
    
    return fig
