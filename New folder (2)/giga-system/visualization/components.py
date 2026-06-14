"""
GIGA SYSTEM - UI Components
Greek Intelligence for Global Analysis

Reusable UI components for the Streamlit application.
Provides consistent styling and interactive elements across all pages.

Key Components:
- Metric cards with performance indicators
- Data input forms with validation
- Interactive tables with sorting/filtering
- Progress indicators for long-running operations
- Alert and notification components
- Loading spinners and status indicators

Styling Features:
- Consistent dark theme
- Responsive design elements
- Professional financial color scheme
- Interactive hover effects
- Animated state transitions
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import warnings
import time

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available for advanced visualizations")


# =============================================================================
# COLOR SCHEMES
# =============================================================================

COLORS = {
    'primary': '#00D4AA',
    'secondary': '#FF6B6B',
    'tertiary': '#4ECDC4',
    'positive': '#00ff88',
    'negative': '#ff4444',
    'warning': '#ffaa00',
    'background': '#1a1a2e',
    'card': '#16213e',
    'text': '#ffffff',
    'text_muted': '#888888'
}


# =============================================================================
# METRIC CARDS
# =============================================================================

def metric_card(label: str, value: str, delta: Optional[str] = None,
               delta_color: str = "normal") -> None:
    """
    Display styled metric card.
    
    Parameters
    ----------
    label : str
        Metric label.
    value : str
        Metric value.
    delta : str, optional
        Delta value.
    delta_color : str
        "normal", "inverse", or "off".
    """
    st.metric(label=label, value=value, delta=delta, delta_color=delta_color)


def metric_row(metrics: List[Dict[str, Any]], columns: int = 4) -> None:
    """
    Display row of metric cards.
    
    Parameters
    ----------
    metrics : list
        List of metric dicts with 'label', 'value', 'delta'.
    columns : int
        Number of columns.
    """
    cols = st.columns(columns)
    
    for i, metric in enumerate(metrics):
        with cols[i % columns]:
            st.metric(
                label=metric.get('label', ''),
                value=metric.get('value', ''),
                delta=metric.get('delta', None)
            )


# =============================================================================
# DATA TABLES
# =============================================================================

def styled_dataframe(df: pd.DataFrame, 
                    height: int = 400,
                    highlight_columns: Optional[List[str]] = None,
                    format_dict: Optional[Dict] = None) -> None:
    """
    Display styled dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data to display.
    height : int
        Table height.
    highlight_columns : list, optional
        Columns to highlight positive/negative.
    format_dict : dict, optional
        Column format specifications.
    """
    styled = df.style
    
    if format_dict:
        styled = styled.format(format_dict)
    
    if highlight_columns:
        def color_negative_red(val):
            if isinstance(val, (int, float)):
                color = COLORS['negative'] if val < 0 else COLORS['positive']
                return f'color: {color}'
            return ''
        
        for col in highlight_columns:
            if col in df.columns:
                styled = styled.applymap(color_negative_red, subset=[col])
    
    st.dataframe(styled, height=height, use_container_width=True)


def trade_table(trades: List[Dict]) -> None:
    """Display trade table with styling."""
    df = pd.DataFrame(trades)
    
    if 'pnl' in df.columns:
        df['pnl'] = df['pnl'].apply(
            lambda x: f"<span style='color: {COLORS['positive'] if x >= 0 else COLORS['negative']}'>${x:+,.2f}</span>"
        )
    
    st.write(df.to_html(escape=False), unsafe_allow_html=True)


# =============================================================================
# CHARTS
# =============================================================================

def equity_chart(timestamps: List[datetime], 
                equity: np.ndarray,
                benchmark: Optional[np.ndarray] = None,
                height: int = 400) -> None:
    """
    Display equity curve chart.
    
    Parameters
    ----------
    timestamps : list
        List of timestamps.
    equity : np.ndarray
        Equity values.
    benchmark : np.ndarray, optional
        Benchmark values.
    height : int
        Chart height.
    """
    if PLOTLY_AVAILABLE:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=equity,
            name='Portfolio',
            line=dict(color=COLORS['primary'], width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 212, 170, 0.1)'
        ))
        
        if benchmark is not None:
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=benchmark,
                name='Benchmark',
                line=dict(color=COLORS['secondary'], width=1, dash='dash')
            ))
        
        fig.update_layout(
            template='plotly_dark',
            height=height,
            margin=dict(l=0, r=0, t=30, b=0),
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        chart_data = pd.DataFrame({'Equity': equity}, index=timestamps)
        if benchmark is not None:
            chart_data['Benchmark'] = benchmark
        st.line_chart(chart_data, height=height)


def drawdown_chart(timestamps: List[datetime],
                  equity: np.ndarray,
                  height: int = 200) -> None:
    """Display drawdown chart."""
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak * 100
    
    if PLOTLY_AVAILABLE:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=drawdown,
            name='Drawdown',
            line=dict(color=COLORS['negative'], width=1),
            fill='tozeroy',
            fillcolor='rgba(255, 68, 68, 0.3)'
        ))
        
        fig.update_layout(
            template='plotly_dark',
            height=height,
            margin=dict(l=0, r=0, t=0, b=0),
            yaxis_title='Drawdown (%)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        chart_data = pd.DataFrame({'Drawdown': drawdown}, index=timestamps)
        st.area_chart(chart_data, height=height)


def allocation_pie(weights: Dict[str, float], height: int = 300) -> None:
    """Display allocation pie chart."""
    if PLOTLY_AVAILABLE:
        fig = go.Figure(data=[go.Pie(
            labels=list(weights.keys()),
            values=list(weights.values()),
            hole=0.4,
            marker_colors=[COLORS['primary'], COLORS['secondary'], 
                          COLORS['tertiary'], COLORS['warning'], '#9b59b6']
        )])
        
        fig.update_layout(
            template='plotly_dark',
            height=height,
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        df = pd.DataFrame({'Weight': weights.values()}, index=weights.keys())
        st.bar_chart(df, height=height)


def returns_histogram(returns: np.ndarray, 
                     var_95: Optional[float] = None,
                     height: int = 300) -> None:
    """Display returns distribution histogram."""
    if PLOTLY_AVAILABLE:
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=returns * 100,
            nbinsx=50,
            marker_color=COLORS['primary'],
            opacity=0.7
        ))
        
        if var_95 is not None:
            fig.add_vline(
                x=-var_95 * 100,
                line_dash='dash',
                line_color=COLORS['negative'],
                annotation_text=f'VaR 95%: {-var_95*100:.2f}%'
            )
        
        fig.update_layout(
            template='plotly_dark',
            height=height,
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis_title='Return (%)',
            yaxis_title='Frequency'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        df = pd.DataFrame({'Returns': returns * 100})
        st.bar_chart(df.value_counts(bins=30).sort_index(), height=height)


def correlation_heatmap(corr_matrix: pd.DataFrame, height: int = 400) -> None:
    """Display correlation heatmap."""
    if PLOTLY_AVAILABLE:
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdYlGn',
            zmin=-1, zmax=1
        ))
        
        fig.update_layout(
            template='plotly_dark',
            height=height,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.dataframe(
            corr_matrix.style.background_gradient(cmap='RdYlGn', vmin=-1, vmax=1),
            height=height
        )


# =============================================================================
# INPUT COMPONENTS
# =============================================================================

def symbol_selector(label: str = "Select Symbols",
                   default: List[str] = None,
                   key: str = None) -> List[str]:
    """
    Multi-select symbol input.
    
    Parameters
    ----------
    label : str
        Input label.
    default : list
        Default selected symbols.
    key : str
        Streamlit key.
    
    Returns
    -------
    list
        Selected symbols.
    """
    common_symbols = [
        'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NVDA', 'TSLA',
        'JPM', 'V', 'JNJ', 'WMT', 'PG', 'UNH', 'HD', 'MA'
    ]
    
    return st.multiselect(
        label,
        options=common_symbols,
        default=default or common_symbols[:5],
        key=key
    )


def date_range_selector(default_start: datetime = None,
                       default_end: datetime = None,
                       key: str = None) -> Tuple[datetime, datetime]:
    """
    Date range selector.
    
    Returns
    -------
    tuple
        (start_date, end_date)
    """
    col1, col2 = st.columns(2)
    
    with col1:
        start = st.date_input(
            "Start Date",
            value=default_start or datetime(2022, 1, 1),
            key=f"{key}_start" if key else None
        )
    
    with col2:
        end = st.date_input(
            "End Date",
            value=default_end or datetime.now(),
            key=f"{key}_end" if key else None
        )
    
    return start, end


def risk_parameters_input() -> Dict[str, float]:
    """
    Risk parameters input form.
    
    Returns
    -------
    dict
        Risk parameters.
    """
    col1, col2, col3 = st.columns(3)
    
    with col1:
        confidence = st.slider("Confidence Level", 0.90, 0.99, 0.95, 0.01)
    
    with col2:
        horizon = st.number_input("Horizon (days)", 1, 30, 10)
    
    with col3:
        risk_free = st.number_input("Risk-Free Rate (%)", 0.0, 10.0, 2.0, 0.1)
    
    return {
        'confidence': confidence,
        'horizon': horizon,
        'risk_free_rate': risk_free / 100
    }


def optimization_parameters_input() -> Dict[str, Any]:
    """
    Portfolio optimization parameters.
    
    Returns
    -------
    dict
        Optimization parameters.
    """
    method = st.selectbox(
        "Optimization Method",
        ["Mean-Variance", "Risk Parity", "Maximum Sharpe", 
         "Minimum Volatility", "Black-Litterman", "Quantum QAOA"]
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        risk_aversion = st.slider("Risk Aversion", 0.0, 1.0, 0.5, 0.1)
        min_weight = st.number_input("Min Weight", 0.0, 0.5, 0.0, 0.01)
    
    with col2:
        max_weight = st.number_input("Max Weight", 0.1, 1.0, 0.25, 0.01)
        use_quantum = st.checkbox("Use Quantum", value=False)
    
    return {
        'method': method,
        'risk_aversion': risk_aversion,
        'min_weight': min_weight,
        'max_weight': max_weight,
        'use_quantum': use_quantum
    }


# =============================================================================
# STATUS INDICATORS
# =============================================================================

def status_indicator(status: str, label: str = None) -> None:
    """
    Display status indicator.
    
    Parameters
    ----------
    status : str
        "success", "warning", "error", "info".
    label : str
        Status label.
    """
    icons = {
        'success': ' ',
        'warning': ' ',
        'error': ' ',
        'info': ' '
    }
    
    icon = icons.get(status, ' ')
    st.write(f"{icon} {label or status.capitalize()}")


def progress_bar(value: float, label: str = None) -> None:
    """
    Display progress bar with label.
    
    Parameters
    ----------
    value : float
        Progress value (0-1).
    label : str
        Progress label.
    """
    if label:
        st.write(label)
    st.progress(value)


def alert_box(message: str, alert_type: str = "info") -> None:
    """
    Display alert box.
    
    Parameters
    ----------
    message : str
        Alert message.
    alert_type : str
        "info", "warning", "error", "success".
    """
    if alert_type == "info":
        st.info(message)
    elif alert_type == "warning":
        st.warning(message)
    elif alert_type == "error":
        st.error(message)
    elif alert_type == "success":
        st.success(message)


# =============================================================================
# LAYOUT HELPERS
# =============================================================================

def card(title: str, content_func: callable) -> None:
    """
    Display content in styled card.
    
    Parameters
    ----------
    title : str
        Card title.
    content_func : callable
        Function to render content.
    """
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 10px;
        padding: 1.5rem;
        border: 1px solid #00D4AA33;
        margin-bottom: 1rem;
    ">
        <h4 style="color: #00D4AA; margin-bottom: 1rem;">{title}</h4>
    </div>
    """, unsafe_allow_html=True)
    
    content_func()


def section_header(title: str, subtitle: str = None) -> None:
    """Display section header with optional subtitle."""
    st.markdown(f"## {title}")
    if subtitle:
        st.markdown(f"*{subtitle}*")
    st.markdown("---")


def empty_state(message: str, icon: str = " ") -> None:
    """Display empty state message."""
    st.markdown(f"""
    <div style="
        text-align: center;
        padding: 3rem;
        color: #888;
    ">
        <h1>{icon}</h1>
        <p>{message}</p>
    </div>
    """, unsafe_allow_html=True)
