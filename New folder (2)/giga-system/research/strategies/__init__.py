"""
GIGA SYSTEM - Strategies Module
Quantitative trading strategies: pairs, momentum, options, market making
"""

#  ️ PHASE 2 WARNING: PROMOTION GATE MISSING
# All strategies exported here are treated equally by the system.
# There is currently NO validation layer distinguishing "Experimental" from "Live-Ready".
# Use with extreme caution.
# TODO: Implement StrategyManifest for validation status.

from .base import (
    Strategy,
    Signal,
    Side,
    Position,
    Trade,
    PositionSizer,
    KellyCriterionSizer
)

from .pairs_trading import (
    PairsTradingStrategy,
    PairStats
)

from .momentum import (
    TrendFollowingStrategy,
    BreakoutStrategy,
    TrendState,
    MomentumState
)

from .options_strategies import (
    DeltaHedgingStrategy,
    VolatilityArbitrageStrategy,
    IronCondorStrategy,
    OptionType,
    OptionContract,
    OptionQuote
)

from .market_making import (
    AvellanedaStoikovMM,
    SimpleSpreadMM,
    OrderBook,
    Quote
)

__all__ = [
    # Base classes
    'Strategy',
    'Signal',
    'Side',
    'Position',
    'Trade',
    'PositionSizer',
    'KellyCriterionSizer',
    
    # Pairs trading
    'PairsTradingStrategy',
    'PairStats',
    
    # Momentum
    'TrendFollowingStrategy',
    'BreakoutStrategy',
    'TrendState',
    'MomentumState',
    
    # Options
    'DeltaHedgingStrategy',
    'VolatilityArbitrageStrategy',
    'IronCondorStrategy',
    'OptionType',
    'OptionContract',
    'OptionQuote',
    
    # Market Making
    'AvellanedaStoikovMM',
    'SimpleSpreadMM',
    'OrderBook',
    'Quote',
]
