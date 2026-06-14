# GIGA SYSTEM - API Reference

## Core Mathematics

### Black-Scholes Module

```python
from giga_system.core import black_scholes_price, BlackScholesModel
```

#### `black_scholes_price(S, K, T, r, sigma, option_type='call')`

Calculate Black-Scholes option price.

**Parameters:**
- `S` (float): Spot price
- `K` (float): Strike price  
- `T` (float): Time to expiry in years
- `r` (float): Risk-free rate
- `sigma` (float): Volatility
- `option_type` (str): 'call' or 'put'

**Returns:**
- `float`: Option price

**Example:**
```python
price = black_scholes_price(100, 100, 0.25, 0.05, 0.2, 'call')
# Returns: 5.876...
```

#### `BlackScholesModel`

Full Black-Scholes model with Greeks.

```python
bs = BlackScholesModel(S=100, K=100, T=0.25, r=0.05, sigma=0.2)

# Prices
call_price = bs.call_price()
put_price = bs.put_price()

# Greeks
delta = bs.delta('call')
gamma = bs.gamma()
theta = bs.theta('call')
vega = bs.vega()
rho = bs.rho('call')

# Vectorized
prices = np.array([100, 101, 102])
bs_vec = BlackScholesModel(S=prices, K=100, T=0.25, r=0.05, sigma=0.2)
call_prices = bs_vec.call_price()  # Returns array
```

---

### Greeks Module

```python
from giga_system.core import OptionGreeks
```

#### `OptionGreeks`

Calculate all option sensitivities.

**Methods:**

| Method | Description | Formula |
|--------|-------------|---------|
| `delta(S, K, T, r, sigma, type)` | First derivative w.r.t. S | $\frac{\partial V}{\partial S}$ |
| `gamma(S, K, T, r, sigma)` | Second derivative w.r.t. S | $\frac{\partial^2 V}{\partial S^2}$ |
| `theta(S, K, T, r, sigma, type)` | Time decay | $\frac{\partial V}{\partial t}$ |
| `vega(S, K, T, r, sigma)` | Volatility sensitivity | $\frac{\partial V}{\partial \sigma}$ |
| `rho(S, K, T, r, sigma, type)` | Rate sensitivity | $\frac{\partial V}{\partial r}$ |
| `vanna(S, K, T, r, sigma)` | Cross derivative | $\frac{\partial^2 V}{\partial S \partial \sigma}$ |
| `volga(S, K, T, r, sigma)` | Second vol derivative | $\frac{\partial^2 V}{\partial \sigma^2}$ |
| `charm(S, K, T, r, sigma, type)` | Delta decay | $\frac{\partial \Delta}{\partial t}$ |

**Example:**
```python
greeks = OptionGreeks()

# Individual Greeks
delta = greeks.delta(100, 100, 0.25, 0.05, 0.2, 'call')
gamma = greeks.gamma(100, 100, 0.25, 0.05, 0.2)

# All Greeks at once
all_greeks = greeks.all_greeks(100, 100, 0.25, 0.05, 0.2, 'call')
# Returns: {'delta': 0.56, 'gamma': 0.04, 'theta': -0.02, 'vega': 0.19, 'rho': 0.12}
```

---

### Monte Carlo Module

```python
from giga_system.core import MonteCarloEngine, PathGenerator
```

#### `PathGenerator`

Generate asset price paths.

**Methods:**

| Method | Model | SDE |
|--------|-------|-----|
| `gbm()` | Geometric Brownian Motion | $dS = \mu S dt + \sigma S dW$ |
| `heston()` | Heston Stochastic Vol | $dS = \mu S dt + \sqrt{v} S dW_S$ |
| `merton_jump()` | Jump Diffusion | $dS = \mu S dt + \sigma S dW + J dN$ |

**Example:**
```python
pg = PathGenerator(
    S0=100,
    r=0.05,
    sigma=0.2,
    T=1.0,
    n_steps=252,
    n_paths=10000
)

# Generate GBM paths
paths = pg.gbm()  # Shape: (10000, 252)

# Generate Heston paths
paths, vol_paths = pg.heston(v0=0.04, kappa=2, theta=0.04, xi=0.3, rho=-0.7)
```

#### `MonteCarloEngine`

Price options using Monte Carlo.

```python
mc = MonteCarloEngine(n_paths=100000)

# European option
price = mc.price_european(S=100, K=100, T=0.25, r=0.05, sigma=0.2, option_type='call')

# Asian option
price = mc.price_asian(S=100, K=100, T=1.0, r=0.05, sigma=0.2, n_steps=252)

# Barrier option
price = mc.price_barrier(S=100, K=100, T=0.5, r=0.05, sigma=0.2, 
                        barrier=120, barrier_type='up-and-out')
```

---

### Risk Metrics Module

```python
from giga_system.core import RiskMetrics
```

#### `RiskMetrics`

Calculate risk measures.

**Methods:**

| Method | Description |
|--------|-------------|
| `var_historical(returns, confidence, horizon)` | Historical VaR |
| `var_parametric(returns, confidence, horizon)` | Parametric VaR |
| `var_monte_carlo(returns, confidence, horizon, n_sims)` | Monte Carlo VaR |
| `cvar(returns, confidence)` | Conditional VaR (Expected Shortfall) |
| `max_drawdown(returns)` | Maximum drawdown |
| `sharpe_ratio(returns, rf_rate)` | Sharpe ratio |
| `sortino_ratio(returns, rf_rate)` | Sortino ratio |
| `calmar_ratio(returns)` | Calmar ratio |

**Example:**
```python
risk = RiskMetrics()

returns = np.random.randn(252) * 0.02  # Daily returns

# Value at Risk
var_95 = risk.var_historical(returns, confidence=0.95, horizon=10)
var_99 = risk.var_parametric(returns, confidence=0.99, horizon=1)

# Expected Shortfall
cvar = risk.cvar(returns, confidence=0.95)

# Risk-adjusted returns
sharpe = risk.sharpe_ratio(returns, rf_rate=0.02/252)
sortino = risk.sortino_ratio(returns, rf_rate=0.02/252)
```

---

## R Analytics Bridge

### R Bridge Module

```python
from giga_system.bridge import RBridge
```

#### `RBridge`

Interface to R analytics.

**Methods:**

```python
r = RBridge()

# Time Series
garch_result = r.fit_garch(returns, p=1, q=1)
forecast = r.forecast_garch(garch_result, horizon=10)

# Risk Modeling
copula = r.fit_copula(returns_matrix, family='gaussian')
tail_risk = r.evt_analysis(returns, threshold=0.05)

# Regime Detection
hmm = r.fit_hmm(returns, n_states=2)
regimes = r.predict_regime(hmm, returns)

# Econometrics
coint = r.test_cointegration(series1, series2)
granger = r.granger_causality(series1, series2, max_lag=5)
```

---

## Trading Strategies

### Base Strategy

```python
from giga_system.strategies import Strategy, Signal, Position
```

#### `Strategy`

Abstract base class for strategies.

```python
class MyStrategy(Strategy):
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        # Implement signal logic
        pass
    
    def position_size(self, signal: Signal, portfolio_value: float) -> float:
        # Implement position sizing
        pass
```

### Pairs Trading

```python
from giga_system.strategies import PairsFinder, StatArbStrategy
```

#### `PairsFinder`

Find cointegrated pairs.

```python
finder = PairsFinder()

# Find pairs
pairs = finder.find_pairs(prices_df, significance=0.05)

# Get spread
spread = finder.calculate_spread(pair, prices_df)

# Half-life
halflife = finder.halflife(spread)
```

#### `StatArbStrategy`

Statistical arbitrage strategy.

```python
strat = StatArbStrategy(
    lookback=60,
    entry_zscore=2.0,
    exit_zscore=0.5,
    stop_zscore=4.0
)

signals = strat.generate_signals(spread_data)
```

### Momentum Strategy

```python
from giga_system.strategies import MomentumStrategy, TrendFollowing
```

```python
momentum = MomentumStrategy(
    lookback=20,
    threshold=0.02,
    risk_per_trade=0.01
)

signals = momentum.generate_signals(prices)
```

---

## Backtesting

### Backtest Engine

```python
from giga_system.backtesting import BacktestEngine, PerformanceAnalyzer
```

#### `BacktestEngine`

Event-driven backtesting.

```python
engine = BacktestEngine(
    initial_capital=100000,
    commission=0.001,
    slippage=0.0005
)

# Run backtest
results = engine.run(
    strategy=my_strategy,
    data=market_data,
    start_date='2022-01-01',
    end_date='2023-12-31'
)

# Get equity curve
equity = results.equity_curve

# Get trades
trades = results.trades
```

#### `PerformanceAnalyzer`

Analyze backtest results.

```python
analyzer = PerformanceAnalyzer(results)

# Get metrics
metrics = analyzer.calculate_metrics()
# Returns: {'sharpe': 1.5, 'sortino': 2.1, 'max_dd': -0.15, ...}

# Attribution
attribution = analyzer.factor_attribution()

# Monthly returns
monthly = analyzer.monthly_returns()
```

---

## Quantum Computing

### Quantum Optimizer

```python
from giga_system.quantum import QuantumOptimizer, QuantumPortfolio
```

#### `QuantumOptimizer`

QAOA/VQE optimization.

```python
optimizer = QuantumOptimizer(
    n_qubits=6,
    depth=3,
    optimizer='COBYLA'
)

# Solve QUBO
Q = np.random.randn(6, 6)
Q = (Q + Q.T) / 2  # Symmetric

result = optimizer.solve_qaoa(Q, shots=1024)
# Returns: {'solution': [0, 1, 1, 0, 1, 0], 'cost': -3.2, 'probability': 0.35}
```

#### `QuantumPortfolio`

Quantum portfolio optimization.

```python
qp = QuantumPortfolio(
    expected_returns=returns,
    covariance=cov_matrix,
    risk_aversion=0.5
)

# Optimize
weights = qp.optimize(method='qaoa', budget=3)
```

---

## Visualization

### Dashboard

```bash
streamlit run visualization/app.py
```

### Chart Functions

```python
from giga_system.visualization import (
    candlestick_chart,
    volatility_surface,
    efficient_frontier,
    backtest_results_chart
)
```

```python
# Candlestick
fig = candlestick_chart(ohlcv_df, title='AAPL', show_volume=True)

# Vol surface
fig = volatility_surface(strikes, maturities, ivs)

# Efficient frontier
fig = efficient_frontier(portfolios, optimal_portfolio)
```

---

## Data Types

### Signal

```python
@dataclass
class Signal:
    timestamp: datetime
    symbol: str
    direction: int  # 1=long, -1=short, 0=flat
    strength: float  # 0-1
    metadata: Dict[str, Any]
```

### Position

```python
@dataclass
class Position:
    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
```

### Trade

```python
@dataclass
class Trade:
    timestamp: datetime
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    commission: float
    pnl: float
```

---

## Error Handling

```python
from giga_system.exceptions import (
    GIGAError,
    DataError,
    CalculationError,
    StrategyError,
    QuantumError
)

try:
    result = risky_calculation()
except CalculationError as e:
    logger.error(f"Calculation failed: {e}")
```

---

## Logging

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('giga_system')

# Debug calculations
logger.debug(f"Black-Scholes: S={S}, K={K}, price={price}")

# Info
logger.info(f"Strategy generated {len(signals)} signals")

# Warnings
logger.warning(f"Low liquidity detected for {symbol}")
```

---

## Performance Tips

1. **Use Numba JIT**: Enable with `@njit` decorator
2. **Vectorize**: Use NumPy arrays instead of loops
3. **Cache**: Enable caching for repeated calculations
4. **Parallel**: Use `parallel=True` in Numba
5. **DuckDB**: Use for large dataset queries

```python
# Fast Greeks calculation
from giga_system.core.greeks import greeks_vectorized

# Calculate Greeks for 1M options
S = np.random.uniform(90, 110, 1_000_000)
greeks = greeks_vectorized(S, K=100, T=0.25, r=0.05, sigma=0.2)
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2024-01 | Initial release |
| 1.1.0 | 2024-03 | Added quantum module |
| 1.2.0 | 2024-06 | Added market making |

---

*GIGA SYSTEM - Quality of code, quality of system, quality of architecture*
