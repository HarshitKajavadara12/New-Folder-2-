"""
GIGA SYSTEM - Data Package
Market data handling, database, and technical indicators
"""

from .market_data import (
    OHLCV,
    MarketDataLoader,
)

from .database import (
    DatabaseManager,
    DUCKDB_AVAILABLE,
)

from .indicators import (
    # Moving Averages
    sma,
    ema,
    wma,
    dema,
    tema,
    # Momentum
    rsi,
    macd,
    stochastic,
    williams_r,
    momentum,
    roc,
    # Volatility
    atr,
    bollinger_bands,
    keltner_channels,
    # Trend
    adx,
    supertrend,
    # Volume
    obv,
    vwap,
    mfi,
)

__all__ = [
    # Market Data
    'OHLCV',
    'MarketDataLoader',
    # Database
    'DatabaseManager',
    'DUCKDB_AVAILABLE',
    # Moving Averages
    'sma',
    'ema',
    'wma',
    'dema',
    'tema',
    # Momentum Indicators
    'rsi',
    'macd',
    'stochastic',
    'williams_r',
    'momentum',
    'roc',
    # Volatility Indicators
    'atr',
    'bollinger_bands',
    'keltner_channels',
    # Trend Indicators
    'adx',
    'supertrend',
    # Volume Indicators
    'obv',
    'vwap',
    'mfi',
]
