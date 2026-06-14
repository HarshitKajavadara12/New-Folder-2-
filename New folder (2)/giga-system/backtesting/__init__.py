"""
GIGA SYSTEM - Backtesting Module
Event-driven backtesting with performance analytics
"""

from .engine import (
    BacktestEngine,
    ExecutionSimulator,
    Portfolio,
    Event,
    EventType,
    Order,
    OrderType,
    OrderStatus,
    Fill,
    create_data_iterator
)

from .performance import (
    PerformanceAnalyzer,
    PerformanceMetrics
)

from .visualization import (
    BacktestVisualizer
)

__all__ = [
    # Engine
    'BacktestEngine',
    'ExecutionSimulator',
    'Portfolio',
    'Event',
    'EventType',
    'Order',
    'OrderType',
    'OrderStatus',
    'Fill',
    'create_data_iterator',
    
    # Performance
    'PerformanceAnalyzer',
    'PerformanceMetrics',
    
    # Visualization
    'BacktestVisualizer',
]
