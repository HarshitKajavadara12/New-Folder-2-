"""
GIGA SYSTEM - Bridge Package
Python-R integration and data interfaces
"""

from .r_bridge import (
    RSession,
    get_r_session,
    R_AVAILABLE,
    # Time Series
    fit_arima,
    forecast_arima,
    test_stationarity,
    # Risk Models
    fit_garch,
    fit_egarch,
    fit_copula,
    fit_gpd,
    # Econometrics
    test_cointegration,
    granger_causality,
    fit_var,
    # Portfolio
    mean_variance_optimize,
    risk_parity,
    black_litterman,
    # Regime
    fit_hmm,
    markov_regime_switching,
    # Performance
    calculate_performance_metrics,
    drawdown_analysis,
    # Correlation
    fit_dcc,
    tail_dependence,
)

from .data_bridge import (
    MarketData,
    DataBridge,
    StreamingDataSource,
    SimulatedTickStream,
    POLARS_AVAILABLE,
    DUCKDB_AVAILABLE,
)

__all__ = [
    # R Bridge
    'RSession',
    'get_r_session',
    'R_AVAILABLE',
    'fit_arima',
    'forecast_arima',
    'test_stationarity',
    'fit_garch',
    'fit_egarch',
    'fit_copula',
    'fit_gpd',
    'test_cointegration',
    'granger_causality',
    'fit_var',
    'mean_variance_optimize',
    'risk_parity',
    'black_litterman',
    'fit_hmm',
    'markov_regime_switching',
    'calculate_performance_metrics',
    'drawdown_analysis',
    'fit_dcc',
    'tail_dependence',
    # Data Bridge
    'MarketData',
    'DataBridge',
    'StreamingDataSource',
    'SimulatedTickStream',
    'POLARS_AVAILABLE',
    'DUCKDB_AVAILABLE',
]
