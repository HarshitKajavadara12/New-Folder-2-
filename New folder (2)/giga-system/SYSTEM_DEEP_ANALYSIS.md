# GIGA System — Complete Deep Analysis
> **Greek Intelligence for Global Analysis**  
> Full system map: surface → depth, every file, every module, every function class, every data flow.  
> Generated: 2026-03-07

---

## 1. SYSTEM IDENTITY

| Property | Value |
|----------|-------|
| **Full Name** | GIGA — Greek Intelligence for Global Analysis |
| **Type** | Institutional Quantitative Finance Platform |
| **Purpose** | End-to-end quant research → backtesting → live trading pipeline |
| **Language** | Python 3.12, with R bridge (via rpy2) |
| **DB** | DuckDB (`giga_system.duckdb`) |
| **Frontend** | Streamlit (visualization/app.py) |
| **Total Python files** | ~80 |
| **Estimated total lines** | ~38,000+ |
| **Total classes** | ~200+ |
| **Total functions (module-level)** | ~300+ |

---

## 2. TOP-LEVEL DIRECTORY MAP

```
giga-system/
│
├── ENTRY POINTS
│   ├── launch_giga_system.py       ← master launcher (demo / app / status / setup / test)
│   ├── run_system_pipeline.py      ← headless pipeline runner
│   ├── run_greek_research_lab.py   ← Greek math research mode
│   ├── QUICK_START.py              ← minimal quick demo
│   ├── giga_full_report.py         ← full institutional report (10 modules, EXIT:0)
│   └── demo_complete_system.py     ← demonstration pipeline
│
├── RESEARCH LAYER (core math / ML / quantum / strategies)
│   └── research/
│       ├── core/        ← pure math: BS pricing, Greeks, MC, IV, risk, alpha
│       ├── ml/          ← ML: features, regime detection, vol forecast
│       ├── quantum/     ← quantum: portfolio, MC, ML, optimizer, hybrid
│       ├── strategies/  ← strategy logic: market making, momentum, options, pairs
│       └── r_analytics/ ← R statistical analytics (via bridge)
│
├── BACKTESTING LAYER
│   └── backtesting/
│       ├── engine.py           ← event-driven engine (10 classes)
│       ├── metrics.py          ← performance analytics (3 classes, 50+ metrics)
│       ├── benchmark.py        ← SPY/index benchmarking
│       ├── performance.py      ← performance tracker
│       ├── visualization.py    ← plotly charts for backtest
│       ├── walk_forward.py     ← walk-forward validation
│       ├── advanced_backtesting.py ← portfolio-level backtest
│       ├── validator.py        ← strategy validation
│       └── result_store.py     ← DuckDB result persistence
│
├── DATA LAYER
│   └── data/
│       ├── market_data.py      ← yfinance + data fetching (2 classes)
│       ├── database.py         ← DuckDB ORM layer (2 classes)
│       ├── database_layer.py   ← abstract DB layer
│       ├── indicators.py       ← 200+ technical indicators (19 functions)
│       ├── preprocessing.py    ← normalization / feature prep
│       ├── storage_manager.py  ← parquet / CSV / DuckDB storage
│       ├── multi_exchange.py   ← multi-exchange data aggregation
│       └── live/
│           ├── binance_ws_feed.py  ← Binance WebSocket feed
│           └── market_stream.py   ← generic streaming
│
├── EXECUTION LAYER
│   └── execution/
│       ├── execution_engine.py ← core execution (3 classes)
│       ├── order_manager.py    ← order lifecycle mgmt (6 classes)
│       ├── smart_router.py     ← smart order routing (5 classes)
│       ├── order_router.py     ← basic routing
│       ├── binance_executor.py ← Binance CEX execution
│       ├── latency_monitor.py  ← sub-ms latency tracking (4 classes)
│       └── instructions.py     ← order instructions (3 classes)
│
├── BRIDGE LAYER (research ↔ live)
│   └── bridge/
│       ├── research_live_bridge.py ← research → live signal bridge (7 classes)
│       ├── data_bridge.py          ← data pipeline bridge (4 classes)
│       ├── model_wrapper.py        ← ML model wrappers (4 classes)
│       ├── r_bridge.py             ← Python ↔ R bridge (1 class, 20 funcs)
│       ├── rpy2_interface.py       ← rpy2 R interface
│       └── data_converter.py       ← data format conversion
│
├── OPTIMIZATION LAYER
│   └── optimization/
│       ├── ai_optimizer.py         ← AI-driven param optimizer (1 class)
│       └── quantum_validation.py   ← quantum circuit validation (6 classes)
│
├── RISK LAYER
│   └── risk/
│       ├── session_guard.py        ← live session risk guard (1 class)
│       └── strategy_breaker.py     ← circuit breaker (2 classes)
│
├── VISUALIZATION LAYER
│   └── visualization/
│       ├── app.py                  ← Streamlit app (23 functions, 1539 lines)
│       ├── charts.py               ← financial charts (14 functions)
│       ├── components.py           ← UI components (19 functions)
│       ├── quantum_visualizer.py   ← quantum circuit viz (1 class)
│       ├── risk_dashboard.py       ← risk dashboard (1 class)
│       ├── statistical_plots.py    ← stat plots (1 class)
│       ├── pnl_attribution.py      ← P&L attribution (1 class)
│       ├── correlation_heatmap.py  ← correlation viz (1 class)
│       ├── greeks_dashboard.py     ← Greeks viz (1 class)
│       ├── education_mode.py       ← educational UI (1 class)
│       ├── education_viz.py        ← educational charts (5 classes)
│       ├── observer_app.py         ← observer pattern UI
│       └── pages/
│           ├── backtest_page.py    ← backtest UI page
│           ├── options_page.py     ← options UI page
│           ├── portfolio_page.py   ← portfolio UI page
│           └── quantum_page.py     ← quantum UI page
│
├── UTILS LAYER (cross-cutting)
│   └── utils/
│       ├── math_helpers.py         ← 31 math utility functions
│       ├── config_loader.py        ← TOML config loader (1 class, 4 funcs)
│       ├── logger.py               ← structured logging (4 classes, 9 funcs)
│       ├── validators.py           ← input validation (1 class, 8 funcs)
│       ├── performance_profiler.py ← profiling (3 classes, 4 funcs)
│       ├── alerting.py             ← alert system (1 class)
│       ├── rate_limiter.py         ← API rate limiting (2 classes)
│       └── retry.py                ← retry logic (2 classes, 2 funcs)
│
├── SUPPORT LAYERS
│   ├── brain/
│   │   └── state_machine.py        ← system state machine (3 classes)
│   ├── monitoring/
│   │   └── system_monitor.py       ← system health (5 classes)
│   ├── observer/
│   │   └── observer.py             ← observer pattern (1 class)
│   ├── feedback/
│   │   └── adaptive_engine.py      ← adaptive feedback (4 classes)
│   ├── reducer/
│   │   └── reducer.py              ← state reducer (1 class)
│   ├── session/
│   │   └── session_controller.py   ← session management
│   ├── account/
│   │   └── live_account.py         ← live account (1 class)
│   └── artifacts/
│       └── definitions.py          ← system-wide enums/dataclasses (5 classes)
│
└── CONFIG & INFRA
    ├── config/
    │   ├── system_config.toml
    │   ├── database_config.toml
    │   ├── models_config.toml
    │   └── strategies_config.toml
    ├── docker/
    │   ├── Dockerfile
    │   └── docker-compose.yml
    ├── tests/
    │   ├── test_account.py
    │   ├── test_greeks.py
    │   ├── test_risk.py
    │   ├── test_routing.py
    │   └── test_utils.py
    └── requirements.txt / pyproject.toml / setup.py
```

---

## 3. RESEARCH LAYER — COMPLETE MODULE BREAKDOWN

### 3.1 `research/core/` — Pure Mathematical Finance

| File | Classes | Functions | Lines | What It Does |
|------|---------|-----------|-------|--------------|
| `black_scholes.py` | 0 | 12 | 429 | Numba JIT BS call/put/dividend pricing; put-call parity; d1/d2 |
| `greeks.py` | 1 | 19 | 662 | All 5 Greeks (Δ,Γ,ν,Θ,ρ) + charm, speed, vanna, volga; Numba JIT |
| `monte_carlo.py` | 0 | 12 | 730 | GBM path simulation, antithetic variates, MC pricing, VaR/CVaR |
| `implied_volatility.py` | 0 | 10 | 562 | Newton-Raphson IV, bisection IV, IV surface construction |
| `risk_metrics.py` | 0 | 13 | 737 | VaR (historical/param), CVaR, drawdown, IR, Sharpe, Sortino, Calmar |
| `greek_mathematics.py` | 10 | 0 | 1086 | Deep Greek math: Taylor expansion, PDE derivations, sensitivity analysis |
| `greeks_hedging.py` | 7 | 0 | 368 | Delta hedge, gamma scalp, vega trade, multi-leg strategy builder |
| `alpha_factor_library.py` | 13 | 0 | 500 | 13 alpha factor classes: momentum, value, quality, technical |
| `alpha_signal_engine.py` | 3 | 0 | 548 | Signal generation engine, alpha combination, signal scoring |
| `volatility_surface.py` | 5 | 0 | 290 | Vol surface fitting, smile modeling, term-structure |
| `binomial_tree.py` | 0 | 10 | 623 | American/European option pricing via CRR binomial trees |
| `cross_sectional_alpha.py` | 3 | 0 | 479 | Cross-sectional factor models, z-scoring, neutralization |
| `microstructure_alpha.py` | 7 | 0 | 479 | Order flow, bid-ask spread, Kyle's lambda, price impact |
| `options_data_feed.py` | 3 | 0 | 585 | Options chain parser, IV surface from market data |
| `domain_data_connector.py` | 4 | 0 | 399 | Alternative data: macro, sentiment, earnings connectors |
| `information_geometry.py` | 1 | 0 | 62 | Shannon entropy, Fisher information, statistical manifolds |
| `market_state_space.py` | 5 | 0 | 100 | Market state enums and state-space representation |
| `stochastic_models.py` | 2 | 0 | 77 | Heston, SABR stochastic volatility models |
| `time_asymmetry.py` | 2 | 0 | 69 | Time-series asymmetry measures |
| `greek_response.py` | 2 | 0 | 220 | Portfolio-level Greek response / PnL attribution |
| `greek_walk_forward.py` | 3 | 0 | 423 | Walk-forward validation for Greek-based strategies |

**Core math total: ~21 files, ~9,300 lines**

---

### 3.2 `research/ml/` — Machine Learning

| File | Classes | Functions | Lines | What It Does |
|------|---------|-----------|-------|--------------|
| `feature_engineering.py` | 1 | 1 | 235 | 200+ features: price, vol, momentum, microstructure, seasonality |
| `regime_detection.py` | 3 | 0 | 362 | `RegimeDetector` (GMM), `MarketState` enum, HMM fallback |
| `volatility_forecast.py` | 5 | 1 | 406 | `VolatilityForecaster`: EWMA + GARCH + HAR ensemble; `VolForecast` result |

**Key classes:**
- `FeatureEngineer` — builds feature matrix from OHLCV
- `RegimeDetector(window_size, n_regimes)` — `.retrain()`, `.detect(returns)` → `MarketState`
- `VolatilityForecaster(ewma_weight, garch_weight, har_weight)` — `.fit(series).forecast()` → `.daily_vol`
- `MarketState` — TRENDING_UP | TRENDING_DOWN | HIGH_VOL | RANGING | UNCERTAIN

---

### 3.3 `research/quantum/` — Quantum Computing Suite

| File | Classes | Functions | Lines | What It Does |
|------|---------|-----------|-------|--------------|
| `portfolio_quantum.py` | 3 | 0 | 640 | `QuantumPortfolioOptimizer`: max Sharpe / min variance via DE+SLSQP |
| `quantum_optimizer.py` | 4 | 0 | 571 | QUBO formulation, `QuantumOptimizer`, classical annealing |
| `quantum_monte_carlo.py` | 3 | 2 | 706 | Quantum amplitude estimation speedup for MC |
| `quantum_ml.py` | 7 | 2 | 996 | QSVM, VQC, Quantum Neural Networks, QPCA |
| `hybrid_algorithms.py` | 4 | 1 | 915 | QAOA, VQE, hybrid quantum-classical neural nets |
| `risk_quantum.py` | 4 | 0 | 681 | Quantum VaR, CVaR, stress testing with quantum speedup |

**Key classes:**
- `QuantumPortfolioOptimizer(n_assets, risk_free_rate)` — `.maximum_sharpe(mu, cov)`, `.minimum_variance(mu, cov)`, `.risk_parity(cov)`
- `PortfolioConstraints` — weight bounds, sector limits, turnover constraints
- `QuantumOptimizer` — QUBO solver with DE+SLSQP backend
- `QuantumML` — quantum kernel SVM, variational classifier
- `HybridAlgorithm` — QAOA for combinatorial optimization

---

### 3.4 `research/strategies/` — Trading Strategy Library

| File | Classes | Functions | Lines | What It Does |
|------|---------|-----------|-------|--------------|
| `base.py` | 12 | 1 | 598 | Abstract `Strategy`, `Signal`, `Position`, `Trade`, `Side`, `OrderType` |
| `market_making.py` | 4 | 0 | 768 | Avellaneda-Stoikov MM, inventory skew, spread optimization |
| `momentum.py` | 5 | 0 | 954 | Cross-sectional momentum, time-series momentum, dual momentum |
| `options_strategies.py` | 6 | 0 | 1032 | Delta hedge, vol arb, iron condor, straddle, strangle, butterfly |
| `pairs_trading.py` | 2 | 0 | 669 | Cointegration pairs, Kalman filter spread, Ornstein-Uhlenbeck |
| `adaptive_params.py` | 2 | 0 | 371 | Adaptive parameter tuning, Bayesian optimization of strategy params |

**All strategies inherit from `Strategy(ABC)` and produce `Signal` objects.**

---

## 4. BACKTESTING LAYER — COMPLETE BREAKDOWN

| File | Classes | Lines | What It Does |
|------|---------|-------|--------------|
| `engine.py` | 10 | 775 | Event-driven engine: `EventType`, `OrderType`, `Order`, `Fill`, `BacktestEngine`, `Portfolio` |
| `metrics.py` | 3 | 784 | `PerformanceAnalyzer` (50+ metrics), `RollingMetrics`, `TradeAnalytics` |
| `benchmark.py` | 2 | 781 | SPY/index benchmark comparison, alpha/beta decomposition |
| `performance.py` | 2 | 645 | Real-time performance tracker, equity curve, drawdown tracking |
| `visualization.py` | 1 | 691 | Plotly backtest charts: equity, drawdown, rolling Sharpe, returns distribution |
| `walk_forward.py` | 2 | 464 | `WalkForwardOptimizer`, `WalkForwardResult` — in-sample/out-of-sample splits |
| `advanced_backtesting.py` | 5 | 441 | Portfolio-level multi-asset backtest, position sizing, rebalancing |
| `validator.py` | 2 | 203 | Strategy validation: overfitting checks, regime robustness |
| `result_store.py` | 2 | 130 | DuckDB result persistence |

**Event flow in engine:**  
`BAR → Strategy.on_bar() → Signal → Order → Fill → Portfolio.update()`

**`PerformanceAnalyzer` metrics:**
Sharpe, Sortino, Calmar, Omega, Max Drawdown, MaxDD Duration, VaR 95/99, CVaR,  
Alpha, Beta, R², Information Ratio, Hit Rate, Avg Win, Avg Loss, Profit Factor,  
Best/Worst Day, Skewness, Kurtosis, Consecutive Win/Loss

---

## 5. DATA LAYER — COMPLETE BREAKDOWN

| File | Classes | Functions | Lines | What It Does |
|------|---------|-----------|-------|--------------|
| `market_data.py` | 2 | 0 | 624 | `MarketDataFetcher` (yfinance, OHLCV, multi-ticker), `DataCache` |
| `database.py` | 2 | 0 | 665 | `GigaDatabase` (DuckDB ORM), `QueryBuilder` |
| `database_layer.py` | 2 | 0 | 212 | Abstract DB layer + connection pooling |
| `indicators.py` | 0 | 19 | 774 | 200+ indicators: SMA/EMA/VWAP/RSI/MACD/BB/ATR/OBV/Stoch/etc |
| `preprocessing.py` | 1 | 2 | 652 | `DataPreprocessor`: clean, normalize, winsorize, fill gaps |
| `storage_manager.py` | 1 | 3 | 708 | `StorageManager`: parquet/CSV/DuckDB read-write |
| `multi_exchange.py` | 3 | 0 | 189 | Multi-exchange data aggregation (crypto: Binance/Coinbase/Kraken) |
| `live/binance_ws_feed.py` | 1 | 0 | 158 | Binance WebSocket OHLCV + trade feed |
| `live/market_stream.py` | 2 | 0 | 137 | Generic async market data stream |

**200+ indicators from `data/indicators.py`:**
- Trend: SMA, EMA, DEMA, TEMA, KAMA, VWAP, Parabolic SAR
- Momentum: RSI, MACD, Stochastic, Williams %R, CCI, ROC, MFI
- Volatility: ATR, Bollinger Bands, Keltner Channels, Donchian
- Volume: OBV, VWAP, Chaikin, Force Index, Ease of Movement
- Statistical: Z-score, Hurst exponent, autocorrelation, skew/kurt

---

## 6. EXECUTION LAYER — COMPLETE BREAKDOWN

| File | Classes | Lines | What It Does |
|------|---------|-------|--------------|
| `execution_engine.py` | 3 | 363 | `ExecutionEngine`, `ExecutionContext`, `LiveExecutor` — top-level orchestrator |
| `order_manager.py` | 6 | 475 | Order lifecycle: create, submit, amend, cancel, fill, expire |
| `smart_router.py` | 5 | 356 | Smart order routing: TWAP, VWAP, POV, IS, liquidity-seeking |
| `order_router.py` | 1 | 94 | Basic exchange routing |
| `binance_executor.py` | 1 | 394 | Binance REST API executor (spot + futures) |
| `latency_monitor.py` | 4 | 426 | Sub-ms latency tracking, order-to-fill timing, SLA alerts |
| `instructions.py` | 3 | 89 | `OrderInstruction`, `ExecutionInstruction`, `HedgeInstruction` |

**Order types supported:** Market, Limit, Stop, Stop-Limit, Iceberg, TWAP, VWAP

---

## 7. BRIDGE LAYER — COMPLETE BREAKDOWN

| File | Classes | Functions | Lines | What It Does |
|------|---------|-----------|-------|--------------|
| `research_live_bridge.py` | 7 | 0 | 479 | Converts research `Signal` → live `Order`; manages position state |
| `data_bridge.py` | 4 | 0 | 705 | Pipelines data between research, backtesting, and live layers |
| `model_wrapper.py` | 4 | 3 | 635 | Wraps ML models (sklearn/TF/PyTorch) with unified predict interface |
| `r_bridge.py` | 1 | 20 | 681 | Python↔R bridge: call R functions from Python via subprocess |
| `rpy2_interface.py` | 1 | 3 | 335 | rpy2-based R interface (faster than subprocess) |
| `data_converter.py` | 1 | 4 | 368 | Converts between pandas/polars/numpy/DuckDB formats |

---

## 8. VISUALIZATION LAYER — COMPLETE BREAKDOWN

| File | Functions/Class | Lines | What It Does |
|------|----------------|-------|--------------|
| `app.py` | 23 functions | 1539 | Main Streamlit app: routing, layout, sidebar, page dispatch |
| `charts.py` | 14 functions | 987 | Plotly OHLCV, vol surface, P&L, equity curves, Greek profiles |
| `components.py` | 19 functions | 572 | Reusable Streamlit widgets, metric cards, tables, alerts |
| `quantum_visualizer.py` | 1 class | 1221 | Quantum circuit diagrams, Bloch sphere, QAOA landscape |
| `risk_dashboard.py` | 1 class | 878 | Real-time risk: VaR, CVaR, Greeks, exposure heatmaps |
| `statistical_plots.py` | 1 class | 940 | Return distributions, QQ-plots, ACF/PACF, rolling stats |
| `pnl_attribution.py` | 1 class | 746 | P&L attribution by factor, strategy, time |
| `correlation_heatmap.py` | 1 class | 926 | Dynamic correlation matrices, rolling correlation |
| `greeks_dashboard.py` | 1 class | 631 | Real-time Greek surfaces (3D), delta/gamma/vega profiles |
| `education_mode.py` | 1 class | 987 | Educational mode: interactive BS, Greeks intuition |
| `education_viz.py` | 5 classes | 745 | Educational visualizations: payoff diagrams, scenario analysis |
| `pages/backtest_page.py` | 3 functions | 571 | Backtest results UI: strategy comparison, drawdown, metrics |
| `pages/options_page.py` | 1 function | 450 | Options pricing UI: BS, Greeks, IV surface |
| `pages/portfolio_page.py` | 1 function | 591 | Portfolio optimization UI: efficient frontier, allocations |
| `pages/quantum_page.py` | 1 function | 630 | Quantum results UI: QAOA circuit, optimization landscape |

**Visualization total: ~14 files, ~11,400 lines — the largest single layer**

---

## 9. UTILS LAYER — COMPLETE BREAKDOWN

| File | Classes | Functions | Lines | What It Does |
|------|---------|-----------|-------|--------------|
| `math_helpers.py` | 0 | 31 | 889 | Cholesky, matrix ops, fast rolling stats, numerical integration |
| `config_loader.py` | 1 | 4 | 538 | TOML config loader, environment overrides, secrets management |
| `logger.py` | 4 | 9 | 415 | Structured logging: `GigaLogger`, `TradeLogger`, `RiskLogger`, `PerformanceLogger` |
| `validators.py` | 1 | 8 | 526 | Input validation: price, vol, weights, portfolio, order |
| `performance_profiler.py` | 3 | 4 | 689 | Sub-ms profiling, `@profile` decorator, flamegraph export |
| `alerting.py` | 1 | 0 | 150 | Alert system: email/Slack/webhook notifications |
| `rate_limiter.py` | 2 | 0 | 125 | Token bucket + sliding window rate limiters |
| `retry.py` | 2 | 2 | 280 | Exponential backoff retry decorator |

---

## 10. SUPPORT / INFRASTRUCTURE LAYERS

### `brain/state_machine.py` (256 lines, 3 classes)
- `SystemState` enum — INITIALIZING, RESEARCH, BACKTESTING, LIVE, ERROR, SHUTDOWN
- `StateMachine` — manages system-level state transitions
- `SystemBrain` — orchestrates state with callbacks

### `monitoring/system_monitor.py` (259 lines, 5 classes)
- `SystemMonitor` — CPU/memory/latency health checks
- `HealthStatus`, `ComponentHealth`, `AlertManager`, `MetricsCollector`

### `observer/observer.py` (218 lines, 1 class)
- `Observer` — publish/subscribe event pattern (strategy signals → multiple consumers)

### `feedback/adaptive_engine.py` (184 lines, 4 classes)
- `AdaptiveEngine` — reinforcement-learning-style param adaptation
- Adapts strategy params based on live P&L feedback

### `reducer/reducer.py` (260 lines, 1 class)
- `StateReducer` — immutable state management (Redux-pattern) for system state

### `risk/session_guard.py` (206 lines, 1 class)
- `SessionGuard` — live session risk limits: max loss, max drawdown, position limits

### `risk/strategy_breaker.py` (192 lines, 2 classes)
- `CircuitBreaker` — stops strategy on consecutive losses / drawdown threshold
- `StrategyBreaker` — per-strategy kill switch

### `optimization/ai_optimizer.py` (128 lines, 1 class)
- `AIOptimizer` — Bayesian / genetic optimization of strategy hyperparameters

### `optimization/quantum_validation.py` (476 lines, 6 classes)
- `QuantumCircuitValidator`, `QUBOValidator`, `QuantumResultValidator`
- Validates quantum circuit correctness and QUBO solution fidelity

---

## 11. CONFIGURATION

```toml
# config/system_config.toml
[system]
environment = "production"
log_level = "INFO"
db_path = "giga_system.duckdb"

[risk]
max_position_size = 0.25
max_drawdown = -0.20
var_confidence = 0.99

[execution]
default_slippage_bps = 5
latency_target_us = 100

# config/strategies_config.toml
[market_making]
spread_bps = 10
inventory_limit = 1000

[momentum]
lookback_days = 252
rebalance_freq = "monthly"
```

---

## 12. COMPLETE DATA FLOW

```
MARKET DATA SOURCES
    │
    ├── yfinance (historical OHLCV)          ─── data/market_data.py
    ├── Binance WebSocket (live ticks)        ─── data/live/binance_ws_feed.py
    ├── Multi-exchange aggregation            ─── data/multi_exchange.py
    └── Alternative data (domain connector)  ─── research/core/domain_data_connector.py
    │
    ▼
DATA PIPELINE
    │
    ├── Preprocess / normalize                ─── data/preprocessing.py
    ├── Compute 200+ indicators               ─── data/indicators.py
    ├── Feature engineering                   ─── research/ml/feature_engineering.py
    └── Store to DuckDB/parquet               ─── data/storage_manager.py + database.py
    │
    ▼
RESEARCH LAYER (offline / batch)
    │
    ├── OPTIONS PRICING
    │   ├── Black-Scholes (Numba JIT)         ─── research/core/black_scholes.py
    │   ├── Binomial Tree (American)          ─── research/core/binomial_tree.py
    │   ├── Monte Carlo                       ─── research/core/monte_carlo.py
    │   └── Implied Volatility (Newton/bisect)─── research/core/implied_volatility.py
    │
    ├── GREEKS ANALYTICS
    │   ├── All 5 Greeks (Numba JIT)          ─── research/core/greeks.py
    │   ├── Greek Math (PDE, Taylor)          ─── research/core/greek_mathematics.py
    │   ├── Hedging Engine                    ─── research/core/greeks_hedging.py
    │   └── Vol Surface                       ─── research/core/volatility_surface.py
    │
    ├── PORTFOLIO OPTIMIZATION
    │   ├── Quantum (QUBO / DE+SLSQP)         ─── research/quantum/portfolio_quantum.py
    │   ├── Classical Markowitz               ─── research/quantum/quantum_optimizer.py
    │   ├── AI Optimizer                      ─── optimization/ai_optimizer.py
    │   └── Risk Parity / Min-Var             ─── research/quantum/portfolio_quantum.py
    │
    ├── RISK ANALYTICS
    │   ├── VaR / CVaR / Drawdown / IR        ─── research/core/risk_metrics.py
    │   ├── Quantum Risk                      ─── research/quantum/risk_quantum.py
    │   └── Session Guard                     ─── risk/session_guard.py
    │
    ├── ALPHA SIGNALS
    │   ├── Alpha Factor Library (13 factors) ─── research/core/alpha_factor_library.py
    │   ├── Alpha Signal Engine               ─── research/core/alpha_signal_engine.py
    │   ├── Cross-Sectional Alpha             ─── research/core/cross_sectional_alpha.py
    │   └── Microstructure Alpha              ─── research/core/microstructure_alpha.py
    │
    ├── ML / AI
    │   ├── Regime Detection (GMM)            ─── research/ml/regime_detection.py
    │   ├── Volatility Forecast (EWMA+GARCH+HAR)─ research/ml/volatility_forecast.py
    │   ├── Quantum ML (QSVM, VQC)            ─── research/quantum/quantum_ml.py
    │   └── Hybrid Quantum-Classical          ─── research/quantum/hybrid_algorithms.py
    │
    └── STRATEGIES
        ├── Market Making (Avellaneda-Stoikov) ─── research/strategies/market_making.py
        ├── Momentum (XS / TS / Dual)         ─── research/strategies/momentum.py
        ├── Options (delta hedge / vol arb)   ─── research/strategies/options_strategies.py
        ├── Pairs Trading (cointegration)     ─── research/strategies/pairs_trading.py
        └── Adaptive Params                   ─── research/strategies/adaptive_params.py
    │
    ▼
BACKTESTING LAYER
    │
    ├── Event-Driven Engine                   ─── backtesting/engine.py
    │   └── BAR → Signal → Order → Fill → Portfolio
    ├── Performance Analytics (50+ metrics)   ─── backtesting/metrics.py
    ├── Benchmark Comparison (SPY)            ─── backtesting/benchmark.py
    ├── Walk-Forward Validation               ─── backtesting/walk_forward.py
    ├── Advanced Portfolio Backtest           ─── backtesting/advanced_backtesting.py
    └── Store Results                         ─── backtesting/result_store.py (DuckDB)
    │
    ▼
BRIDGE LAYER (research → live)
    │
    ├── Research→Live Bridge                  ─── bridge/research_live_bridge.py
    ├── Model Wrapper                         ─── bridge/model_wrapper.py
    └── Data Bridge                           ─── bridge/data_bridge.py
    │
    ▼
LIVE EXECUTION LAYER
    │
    ├── Execution Engine                      ─── execution/execution_engine.py
    ├── Order Manager                         ─── execution/order_manager.py
    ├── Smart Router (TWAP/VWAP/POV)          ─── execution/smart_router.py
    ├── Binance Executor                      ─── execution/binance_executor.py
    └── Latency Monitor                       ─── execution/latency_monitor.py
    │
    ▼
RISK / GUARDS (wrap entire live layer)
    ├── Session Guard                         ─── risk/session_guard.py
    └── Circuit Breaker                       ─── risk/strategy_breaker.py
    │
    ▼
MONITORING / FEEDBACK
    ├── System Monitor                        ─── monitoring/system_monitor.py
    ├── Adaptive Engine                       ─── feedback/adaptive_engine.py
    ├── State Machine                         ─── brain/state_machine.py
    └── Observer                             ─── observer/observer.py
    │
    ▼
VISUALIZATION (Streamlit app, port 8501)
    ├── app.py → routes to 4 pages
    ├── Portfolio Page (efficient frontier, holding weights)
    ├── Options Page (pricing, Greeks, IV surface)
    ├── Backtest Page (strategy comparison, metrics)
    └── Quantum Page (circuit diagrams, QAOA landscape)
```

---

## 13. CLASS HIERARCHY SUMMARY

```
Strategy (ABC)
├── MarketMakingStrategy          (market_making.py)
├── MomentumStrategy              (momentum.py)
│   ├── CrossSectionalMomentum
│   ├── TimeSeriesMomentum
│   └── DualMomentum
├── OptionsStrategy               (options_strategies.py)
│   ├── DeltaHedgeStrategy
│   ├── VolArbitrageStrategy
│   └── IronCondorStrategy
├── PairsTradingStrategy          (pairs_trading.py)
└── AdaptiveStrategy              (adaptive_params.py)

PortfolioOptimizer (base)
└── QuantumPortfolioOptimizer     (portfolio_quantum.py)
    ├── .maximum_sharpe(mu, cov)
    ├── .minimum_variance(mu, cov)
    └── .risk_parity(cov)

BacktestEngine
├── EventQueue
├── DataHandler
├── Portfolio
├── OrderManager
└── ExecutionHandler

PerformanceAnalyzer
├── .calculate_metrics(returns)
├── .rolling_metrics(returns, window)
└── .trade_analytics(trades)

RegimeDetector
└── .retrain(returns) → fits GMM
└── .detect(returns) → MarketState

VolatilityForecaster
├── .fit(series) → self
└── .forecast() → VolForecast(.daily_vol)

GreeksHedgingEngine
├── .compute_position_greeks(legs, S) → OptionGreeks
├── .delta_hedge(legs, S) → HedgeAction
└── .gamma_scalp_signal(legs, S, rv, iv) → Dict
```

---

## 14. MODULES USED IN `giga_full_report.py` (10/12 working)

| # | Module | Import Path | Status | Used For |
|---|--------|-------------|--------|----------|
| 1 | Black-Scholes | `research.core.black_scholes` | ✅ | Option pricing (Numba JIT) |
| 2 | Greeks | `research.core.greeks` | ✅ | Δ/Γ/ν/Θ/ρ validation |
| 3 | Monte Carlo | `research.core.monte_carlo` | ✅ | 10k GBM paths |
| 4 | Perf Analyzer | `backtesting.metrics.PerformanceAnalyzer` | ✅ | 50+ backtest metrics |
| 5 | Quantum Optimizer | `research.quantum.portfolio_quantum.QuantumPortfolioOptimizer` | ✅ | Max Sharpe (QUBO/DE+SLSQP) |
| 6 | Risk Metrics | `research.core.risk_metrics` | ✅ | VaR/CVaR/IR/drawdown |
| 7 | Regime Detector | `research.ml.regime_detection.RegimeDetector` | ✅ | Regime classification |
| 8 | Vol Forecast | `research.ml.volatility_forecast.VolatilityForecaster` | ✅ | EWMA+GARCH+HAR vol |
| 9 | Implied Vol | `research.core.implied_volatility` | ✅ | IV bisection round-trip |
| 10 | Greeks Hedging | `research.core.greeks_hedging.GreeksHedgingEngine` | ✅ | Delta hedge |
| — | Backtesting Engine | `backtesting.engine.BacktestEngine` | ❌ | Not yet wired (usable) |
| — | Walk Forward | `backtesting.walk_forward.WalkForwardOptimizer` | ❌ | Not yet wired (usable) |
| — | Alpha Signals | `research.core.alpha_signal_engine` | ❌ | Not yet wired |
| — | Microstructure | `research.core.microstructure_alpha` | ❌ | Not yet wired |
| — | Options Strategies | `research.strategies.options_strategies` | ❌ | Not yet wired |
| — | Market Making | `research.strategies.market_making` | ❌ | Not yet wired |
| — | R Bridge | `bridge.r_bridge` | ❌ | Needs R installed |

---

## 15. WHAT IS MISSING / NOT YET INTEGRATED

### Missing from `giga_full_report.py` (available but unused):

1. **`backtesting.engine.BacktestEngine`** — full event-driven backtest (currently using a vectorized approximation instead)
2. **`backtesting.walk_forward.WalkForwardOptimizer`** — in-sample / out-of-sample splits
3. **`research.core.alpha_factor_library`** — 13 alpha factors (momentum, value, quality)
4. **`research.core.alpha_signal_engine`** — signal scoring and combination
5. **`research.core.microstructure_alpha`** — order-flow, Kyle's lambda
6. **`research.core.cross_sectional_alpha`** — cross-sectional factor model
7. **`research.core.binomial_tree`** — American option pricing
8. **`research.core.volatility_surface`** — full vol surface fitting
9. **`research.core.greek_mathematics`** — deep Taylor/PDE Greek analysis
10. **`research.core.information_geometry`** — Shannon entropy, Fisher info
11. **`research.strategies.*`** — 6 full strategies (market making, momentum, options, pairs)
12. **`research.quantum.quantum_ml`** — QSVM, VQC, QNN
13. **`research.quantum.hybrid_algorithms`** — QAOA, VQE
14. **`research.quantum.quantum_monte_carlo`** — quantum amplitude estimation
15. **`research.quantum.risk_quantum`** — quantum VaR
16. **`optimization.quantum_validation`** — QUBO validation
17. **`bridge.research_live_bridge`** — strategy → live order bridge
18. **`bridge.model_wrapper`** — ML model deployment
19. **`feedback.adaptive_engine`** — RL-style param adaptation
20. **`brain.state_machine`** — system state management
21. **`data.indicators`** — 200+ technical indicators
22. **`data.preprocessing`** — data cleaning pipeline
23. **`execution.*`** — all execution modules
24. **`risk.*`** — session guard, circuit breaker
25. **`visualization.*`** — Streamlit app (separate launch)

---

## 16. METRICS SUMMARY

| Category | Count |
|----------|-------|
| Python files | ~82 |
| Total lines of code | ~38,000+ |
| Module-level functions | ~300+ |
| Classes | ~200+ |
| Independent entry points | 6 |
| Config files (TOML) | 4 |
| Test files | 5 |
| Supported exchanges | Binance (live), any (demo) |
| Technical indicators | 200+ |
| Alpha factors | 13 |
| Strategy types | 6 |
| Risk metrics | 50+ |
| Visualization dashboards | 4 (Streamlit pages) |
| Visualization chart types | 30+ |

---

## 17. DEPENDENCY MAP (key imports between layers)

```
giga_full_report.py
    └── imports from: research.core.* + research.ml.* + research.quantum.* + backtesting.metrics

backtesting/engine.py
    └── imports: research.strategies.base

research/strategies/*
    └── imports: research.core.black_scholes, research.core.greeks, research.core.risk_metrics

bridge/research_live_bridge.py
    └── imports: research.strategies.base, execution.order_manager, data.market_data

execution/execution_engine.py
    └── imports: execution.order_manager, execution.smart_router, utils.logger

visualization/app.py
    └── imports: visualization.pages.*, visualization.charts, research.core.*, backtesting.*

data/market_data.py
    └── imports: yfinance, data.database, utils.config_loader

utils/* (no internal imports — pure utilities)
```

---
*Analysis complete. System has ~38,000 lines across 82 Python files, organized into 10 functional layers. Core math and visualization are the largest layers (~9,300 and ~11,400 lines respectively).*
