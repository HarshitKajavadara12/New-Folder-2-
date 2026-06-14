import sys
sys.path.insert(0, r'c:\Users\HARSHIT\Desktop\New folder\New folder (2)\giga-system')

modules = [
    'research.quantum.portfolio_quantum',
    'backtesting.engine',
    'backtesting.advanced_backtesting',
    'research.core.risk_metrics',
    'research.ml.regime_detection',
    'research.ml.volatility_forecast',
    'data.market_data',
    'utils.performance_profiler',
    'research.core.implied_volatility',
    'research.core.volatility_surface',
    'research.core.greeks_hedging',
    'research.core.greek_mathematics',
    'optimization.ai_optimizer',
    'research.strategies.momentum',
    'research.strategies.pairs_trading',
    'backtesting.metrics',
    'backtesting.walk_forward',
    'visualization.charts',
    'visualization.correlation_heatmap',
]
for m in modules:
    try:
        mod = __import__(m, fromlist=[''])
        attrs = [a for a in dir(mod) if not a.startswith('_')][:8]
        print(f'OK  {m}  => {attrs}')
    except Exception as e:
        print(f'ERR {m}  => {type(e).__name__}: {e}')
