"""
GIGA SYSTEM - Visualization Package
Streamlit dashboard and Plotly visualizations
"""

from .components import (
    # Color schemes
    COLORS,
    
    # Metric displays
    metric_card,
    metric_row,
    
    # Data tables
    styled_dataframe,
    trade_table,
    
    # Charts
    equity_chart,
    drawdown_chart,
    allocation_pie,
    returns_histogram,
    correlation_heatmap,
    
    # Input components
    symbol_selector,
    date_range_selector,
    risk_parameters_input,
    optimization_parameters_input,
    
    # Status indicators
    status_indicator,
    progress_bar,
    alert_box,
    
    # Layout helpers
    card,
    section_header,
    empty_state
)

from .charts import (
    # Price charts
    candlestick_chart,
    multi_asset_chart,
    
    # Options charts
    volatility_surface,
    volatility_smile,
    greeks_chart,
    payoff_diagram,
    
    # Risk charts
    var_chart,
    risk_decomposition_chart,
    
    # Portfolio charts
    efficient_frontier,
    weights_timeline,
    
    # Backtest charts
    backtest_results_chart,
    monthly_returns_heatmap,
    
    # Quantum charts
    quantum_circuit_diagram,
    quantum_probability_chart
)


__all__ = [
    # Components
    'COLORS',
    'metric_card',
    'metric_row',
    'styled_dataframe',
    'trade_table',
    'equity_chart',
    'drawdown_chart',
    'allocation_pie',
    'returns_histogram',
    'correlation_heatmap',
    'symbol_selector',
    'date_range_selector',
    'risk_parameters_input',
    'optimization_parameters_input',
    'status_indicator',
    'progress_bar',
    'alert_box',
    'card',
    'section_header',
    'empty_state',
    
    # Charts
    'candlestick_chart',
    'multi_asset_chart',
    'volatility_surface',
    'volatility_smile',
    'greeks_chart',
    'payoff_diagram',
    'var_chart',
    'risk_decomposition_chart',
    'efficient_frontier',
    'weights_timeline',
    'backtest_results_chart',
    'monthly_returns_heatmap',
    'quantum_circuit_diagram',
    'quantum_probability_chart'
]
