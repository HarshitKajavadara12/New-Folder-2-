# GIGA SYSTEM — Comprehensive Code Research Report

**Project:** giga-system — Greek Intelligence for Global Analysis  
**Architecture:** Research → Reducer → Live execution pipeline with Observer audit, Feedback learning, Monitoring alerts, and Air-Gap validation  
**Total Files Audited:** 52 Python files across 11 directories  
**Total Lines of Code (estimated):** ~22,000+

---

## Table of Contents

1. [backtesting/ (10 files)](#1-backtesting-10-files)
2. [feedback/ (2 files)](#2-feedback-2-files)
3. [monitoring/ (2 files)](#3-monitoring-2-files)
4. [observer/ (2 files)](#4-observer-2-files)
5. [optimization/ (3 files)](#5-optimization-3-files)
6. [reducer/ (2 files)](#6-reducer-2-files)
7. [session/ (1 file)](#7-session-1-file)
8. [live/ (3 files)](#8-live-3-files)
9. [utils/ (9 files)](#9-utils-9-files)
10. [artifacts/ (1 file)](#10-artifacts-1-file)
11. [visualization/ (17 files)](#11-visualization-17-files)

---

## 1. backtesting/ (10 files)

### 1.1 `backtesting/__init__.py` — 45 lines

**Purpose:** Package exports for the backtesting module.

**Exports:**
- `BacktestEngine`, `ExecutionSimulator`, `Portfolio` — from `engine`
- `Event`, `EventType`, `Order`, `OrderType`, `OrderStatus`, `Fill` — data classes from `engine`
- `create_data_iterator` — from `engine`
- `PerformanceAnalyzer`, `PerformanceMetrics` — from `metrics`
- `BacktestVisualizer` — from `visualization`

**Expected Behavior:** Acts as the public API surface for all backtesting functionality. All imports are wrapped in try/except with graceful fallback.

---

### 1.2 `backtesting/engine.py` — 776 lines

**Purpose:** Core event-driven backtesting engine with realistic execution simulation.

**Enums:**
- `EventType` — `MARKET_DATA`, `SIGNAL`, `ORDER`, `FILL`, `RISK_CHECK`
- `OrderType` — `MARKET`, `LIMIT`, `STOP`, `STOP_LIMIT`
- `OrderStatus` — `PENDING`, `SUBMITTED`, `FILLED`, `PARTIALLY_FILLED`, `CANCELLED`, `REJECTED`

**Dataclasses:**
- `Event(type: EventType, timestamp: datetime, data: Dict)`
- `Order(symbol: str, side: str, quantity: float, order_type: OrderType, price: Optional[float], status: OrderStatus, timestamp: datetime, fill_price: Optional[float], fill_quantity: Optional[float], commission: float)`
- `Fill(order: Order, fill_price: float, fill_quantity: float, commission: float, slippage: float, timestamp: datetime)`
- `PortfolioPosition(symbol: str, quantity: float, avg_cost: float, market_value: float, unrealized_pnl: float, realized_pnl: float)`

**Classes:**

#### `ExecutionSimulator`
- `__init__(self, slippage_pct=0.0001, commission_per_share=0.005, commission_minimum=1.0, fill_probability=0.98, partial_fill_prob=0.1)`
- `simulate_fill(self, order: Order, market_data: Dict) → Optional[Fill]` — Applies random slippage (normal distribution), calculates commission as `max(qty * per_share, minimum)`, handles partial fills
- `calculate_slippage(self, price, side) → float` — Gaussian slippage model
- `calculate_commission(self, quantity, price) → float`

#### `Portfolio`
- `__init__(self, initial_cash=1_000_000)`
- Properties: `cash`, `positions: Dict[str, PortfolioPosition]`, `equity_curve: List[float]`, `trade_history: List[Dict]`
- `update_position(self, fill: Fill)` — Updates positions, tracks realized P&L on close
- `update_market_values(self, market_data: Dict)` — Mark-to-market
- `get_total_equity(self) → float` — Cash + sum of market values
- `get_position_weight(self, symbol) → float`

#### `BacktestEngine`
- `__init__(self, initial_capital=1_000_000, commission_rate=0.001, slippage_rate=0.0005)`
- `run(self, data: pd.DataFrame, strategy_func: Callable, **kwargs) → Dict` — Main event loop: iterates over data rows, calls `strategy_func(row, positions, equity)`, executes returned signals
- `_execute_signal(self, signal: Dict, row: pd.Series)` — Creates Order, simulates Fill, updates Portfolio
- `get_results(self) → Dict` — Returns equity curve, trades, positions, metrics

**Functions:**
- `create_data_iterator(df: pd.DataFrame, chunk_size=1) → Iterator` — Yields data in chunks for streaming backtest

**Constants/Defaults:**
- Initial capital: $1,000,000
- Slippage: 0.01% (1 bps)
- Commission per share: $0.005
- Commission minimum: $1.00
- Fill probability: 98%
- Partial fill probability: 10%

**Expected Behavior:** Processes market data events sequentially, generates signals via strategy function, simulates realistic execution with slippage/commission/partial fills, tracks equity curve and trade history.

---

### 1.3 `backtesting/metrics.py` — 784 lines

**Purpose:** Comprehensive performance metrics calculation with statistical rigor.

**Dataclasses:**
- `PerformanceMetrics` — Fields: `total_return, cagr, annual_volatility, sharpe_ratio, sortino_ratio, calmar_ratio, max_drawdown, max_drawdown_duration, var_95, var_99, cvar_95, cvar_99, skewness, kurtosis, win_rate, profit_factor, avg_win, avg_loss, t_statistic, p_value, num_trades`
- `RiskMetrics` — Fields: `beta, alpha, r_squared, tracking_error, information_ratio, treynor_ratio`

**Classes:**

#### `PerformanceAnalyzer`
- `__init__(self, risk_free_rate=0.02, confidence_levels=[0.95, 0.99], periods_per_year=252)`
- `calculate_returns(self, equity_curve) → np.ndarray`
- `calculate_sharpe_ratio(self, returns) → float` — `(mean * √252 - rf) / (std * √252)`
- `calculate_sortino_ratio(self, returns) → float` — Uses downside deviation only
- `calculate_calmar_ratio(self, returns) → float` — CAGR / |MaxDD|
- `calculate_omega_ratio(self, returns, threshold=0.0) → float` — ∫(gains above threshold) / ∫(losses below threshold)
- `calculate_var(self, returns, confidence=0.95, method='historical') → float` — Historical or parametric (Gaussian) VaR
- `calculate_cvar(self, returns, confidence=0.95) → float` — Mean of losses beyond VaR
- `calculate_max_drawdown(self, equity_curve) → Tuple[float, int]` — Returns (max_dd, duration_in_periods)
- `calculate_t_statistic(self, returns) → Tuple[float, float]` — t-stat and p-value for mean != 0
- `monte_carlo_bootstrap(self, returns, n_simulations=1000, block_size=20) → Dict` — Block bootstrap with confidence intervals for Sharpe, return, volatility
- `generate_report(self, equity_curve, benchmark=None) → Dict` — Complete performance report including risk metrics if benchmark provided

**Expected Behavior:** Produces a comprehensive performance report from an equity curve, including all standard ratios, risk metrics, statistical tests, and bootstrap confidence intervals.

---

### 1.4 `backtesting/performance.py` — 646 lines

**Purpose:** Alternative performance analyzer with additional metrics.

**Dataclasses:**
- `PerformanceMetrics` — Includes `beta, alpha, r_squared, win_rate, avg_win_loss_ratio, expectancy, kelly_criterion`

**Classes:**

#### `PerformanceAnalyzer`
- `__init__(self, risk_free_rate=0.02, periods_per_year=252)`
- All standard ratio calculations (Sharpe, Sortino, Calmar, Omega, Information Ratio)
- `calculate_beta(self, returns, benchmark_returns) → float` — Covariance / benchmark variance
- `calculate_alpha(self, returns, benchmark_returns) → float` — Jensen's alpha
- `calculate_kelly_criterion(self, returns) → float` — Optimal bet sizing: `mean/var`
- `calculate_information_ratio(self, returns, benchmark_returns) → float` — Active return / tracking error
- `generate_report(self, equity_curve, trades=None, benchmark=None) → PerformanceMetrics`

**Expected Behavior:** Provides an alternative metrics engine with Kelly criterion and expectancy calculations not in `metrics.py`.

---

### 1.5 `backtesting/advanced_backtesting.py` — ~290 lines

**Purpose:** Specialized backtesting strategies for options and Greek-aware trading.

**Dataclasses:**
- `BacktestSummary(strategy, total_return, sharpe, max_drawdown, trades, win_rate)`
- `BacktestComparison` — JSON-serializable container for comparing multiple backtests

**Classes:**

#### `GreekAwareBacktester`
- `__init__(self, kappa_threshold=5.0, entropy_threshold=3.5, position_size_pct=2.0)`
- `run(self, data, signal_func) → BacktestSummary` — Enters trades only when Greek signal κ > threshold AND entropy < threshold; sizes positions at `position_size_pct`% of equity

#### `OptionsBacktester`
- Strategies:
  - **Straddle**: `entry_iv_percentile=25` (buy straddle when IV is cheap)
  - **Delta Hedge**: `rehedge_frequency=1` (daily delta rebalancing)
- `run_straddle_backtest(self, data) → BacktestSummary`
- `run_delta_hedge_backtest(self, data) → BacktestSummary`

#### `MultiAssetBacktester`
- `__init__(self, rebalance_frequency=20)` — Rebalances every 20 trading days
- `run(self, multi_asset_data, weight_func) → BacktestSummary`

**Expected Behavior:** Extends the base engine with options-specific (straddle, delta-hedging) and cross-asset rebalancing backtests.

---

### 1.6 `backtesting/benchmark.py` — 781 lines

**Purpose:** Benchmark comparison and performance attribution.

**Dataclasses:**
- `BenchmarkComparison(alpha, beta, r_squared, tracking_error, information_ratio, up_capture, down_capture)`

**Classes:**

#### `BenchmarkAnalyzer`
- `__init__(self, risk_free_rate=0.02)`
- `brinson_hood_beebower(self, portfolio_returns, benchmark_returns, portfolio_weights, benchmark_weights) → Dict` — BHB attribution: allocation effect, selection effect, interaction effect
- `style_analysis(self, returns, factor_returns) → Dict` — Constrained quadratic optimization (SLSQP) to find style weights summing to 1.0 with non-negativity constraint
- `rolling_performance_comparison(self, portfolio_returns, benchmark_returns, window=252) → pd.DataFrame` — Rolling alpha, beta, Sharpe, tracking error
- `multi_benchmark_analysis(self, portfolio_returns, benchmark_dict) → pd.DataFrame` — Compares portfolio against multiple benchmarks simultaneously
- `up_down_capture(self, portfolio_returns, benchmark_returns) → Tuple[float, float]` — Capture ratios for up and down markets

**Expected Behavior:** Full performance attribution suite: BHB decomposition, style analysis, capture ratios, rolling comparisons.

---

### 1.7 `backtesting/result_store.py` — ~120 lines

**Purpose:** Persistent storage for backtest results with integrity verification.

**Dataclasses:**
- `BacktestResult(strategy_name, params, metrics, equity_curve, trades, timestamp, checksum)` — `checksum` is SHA-256 of serialized content

**Classes:**

#### `BacktestResultStore`
- `__init__(self, storage_dir='artifacts/backtest_results')`
- `save(self, result: BacktestResult) → str` — Saves as JSON, returns file path
- `load(self, result_id: str) → BacktestResult` — Loads and verifies SHA-256 checksum
- `list_results(self, strategy_name=None) → List[str]`
- `get_best_result(self, metric='sharpe_ratio', strategy_name=None) → BacktestResult` — Returns result with highest value of specified metric

**Expected Behavior:** Provides tamper-evident storage of backtest results. If checksum verification fails on load, raises IntegrityError.

---

### 1.8 `backtesting/validator.py` — ~175 lines

**Purpose:** Pipeline validation for trading artifacts (signals, contexts).

**Constants:**
```python
PARAM_BOUNDS = {
    'confidence': (0, 1),
    'strength': (-1, 1),
    'direction': (-1, 1),
    'delta': (-1, 1),
    'gamma': (0, 100),
    'iv': (0, 10),
    'weight': (0, 1)
}
```

**Classes:**

#### `ValidationPipeline`
- `validate(self, artifact) → List[ValidationResult]` — Runs 6 checks:
  1. **Determinism check** — Same input produces same output
  2. **Context validation** — Required fields present (regime, horizon)
  3. **NaN/Inf check** — No NaN or Inf in any numeric field
  4. **Bounds check** — All values within PARAM_BOUNDS
  5. **Staleness check** — Artifact not older than max_age (default 300s)
  6. **Type check** — Artifact is correct class instance

**Expected Behavior:** Used in the Air-Gap between Research and Live to ensure no invalid artifacts reach execution.

---

### 1.9 `backtesting/visualization.py` — 690 lines

**Purpose:** Plotly-based backtest result visualization.

**Constants:**
```python
# Template and colors
template = "plotly_dark"
colors = {
    'primary': '#00D4AA',
    'secondary': '#FF6B6B',
    'positive': '#00ff88',
    'negative': '#ff4444'
}
```

**Classes:**

#### `BacktestVisualizer`
- `__init__(self, template='plotly_dark')`
- `plot_equity_curve(self, timestamps, equity, benchmark=None) → go.Figure` — Portfolio equity with optional benchmark overlay, fill-under-curve
- `plot_returns_distribution(self, returns, bins=50) → go.Figure` — Histogram + fitted normal, VaR/CVaR lines
- `plot_rolling_metrics(self, returns, window=63) → go.Figure` — 3-row subplot: rolling Sharpe, rolling volatility, rolling beta
- `plot_trade_analysis(self, trades: List[Dict]) → go.Figure` — Trade P&L scatter, win/loss coloring
- `create_dashboard(self, results: Dict) → go.Figure` — 3×2 composite: equity + returns distribution + rolling Sharpe + drawdown + trade scatter + monthly heatmap

**Expected Behavior:** Generates publication-quality dark-themed interactive charts from backtest results.

---

### 1.10 `backtesting/walk_forward.py` — ~360 lines

**Purpose:** Walk-forward optimization with overfitting detection.

**Dataclasses:**
- `WalkForwardWindow(train_start, train_end, test_start, test_end, train_sharpe, test_sharpe, optimal_params)`

**Classes:**

#### `WalkForwardOptimizer`
- `__init__(self, train_days=252, test_days=63, step_days=21)` — 1 year train, 3 months test, 1 month step
- `generate_windows(self, data) → List[WalkForwardWindow]`
- `optimize_window(self, train_data, param_grid, strategy_func) → Dict` — Grid search over param combinations, returns best by Sharpe
- `run(self, data, param_grid, strategy_func) → List[WalkForwardWindow]` — Full walk-forward with results per window
- `detect_overfitting(self, windows) → bool` — Returns True if average `sharpe_degradation = 1 - test_sharpe/train_sharpe > 0.3`
- `aggregate_results(self, windows) → Dict` — Combined metrics across all out-of-sample periods

**Expected Behavior:** Prevents overfitting by validating strategy performance across rolling out-of-sample windows. Flags overfitting when test Sharpe degrades >30% from train.

---

## 2. feedback/ (2 files)

### 2.1 `feedback/__init__.py` — ~5 lines

Exports `AdaptiveEngine`.

---

### 2.2 `feedback/adaptive_engine.py` — ~155 lines

**Purpose:** Adaptive position sizing and risk limit adjustment based on recent performance.

**Constants:**
```python
LOSS_CUT_FACTOR = 0.95      # Shrink limits by 5% on loss
GAIN_GROW_FACTOR = 1.03     # Grow limits by 3% on gain
MIN_LIMIT_RATIO = 0.30      # Minimum 30% of initial limit
MAX_LIMIT_RATIO = 3.0       # Maximum 300% of initial limit
LOOKBACK_WINDOW = 20        # 20-period moving average
COOLDOWN_TRADES = 5         # Trades before next adjustment
```

**Enums:**
- `CapitalRegime` — `SEED` (<$50K), `GROWTH` ($50K-$500K), `SCALE` ($500K-$5M), `INSTITUTION` (>$5M)

**Classes:**

#### `AdaptiveEngine`
- `__init__(self, initial_limits: Dict)`
- `adjust_limits(self, recent_pnl: List[float]) → Dict` — After each COOLDOWN_TRADES batch, if mean P&L < 0 → multiply all limits by LOSS_CUT_FACTOR; if > 0 → multiply by GAIN_GROW_FACTOR; clamp within [MIN_LIMIT_RATIO, MAX_LIMIT_RATIO] × initial

#### `PositionSizer`
- `__init__(self, base_unit=0.1, k_factor=10000)`
- `calculate_size(self, equity: float, regime: CapitalRegime) → float`
- **Formula:** `size = base_unit × log10(1 + equity/K) × regime_multiplier`
- Regime multipliers: SEED=0.5, GROWTH=1.0, SCALE=1.5, INSTITUTION=2.0

**Expected Behavior:** Dynamically adjusts position sizes and risk limits based on rolling P&L performance — shrinking on drawdowns, growing during profitable periods, with hard min/max caps.

---

## 3. monitoring/ (2 files)

### 3.1 `monitoring/__init__.py` — ~5 lines

Exports `MetricsCollector`, `AlertManager`.

---

### 3.2 `monitoring/system_monitor.py` — ~230 lines

**Purpose:** System monitoring with Prometheus-compatible metrics export and threshold-based alerting.

**Classes:**

#### `MetricsCollector`
- `__init__(self)`
- `record_metric(self, name: str, value: float, labels: Dict = None)`
- `get_metric(self, name: str) → List[Tuple[datetime, float]]`
- `export_prometheus(self) → str` — Outputs Prometheus text format

#### `AlertManager`
- `__init__(self)`
- Default alert rules:
  ```python
  drawdown > 10%    → WARNING
  drawdown > 20%    → CRITICAL
  sharpe < 0.5      → WARNING
  latency > 100ms   → WARNING
  error_rate > 5%   → CRITICAL
  position_count > 20 → WARNING
  ```
- `check_alerts(self, metrics: Dict) → List[Alert]` — Evaluates all rules against current metrics
- `send_alert(self, alert: Alert)` — Dispatches via configured channels (log, email, webhook)

**Expected Behavior:** Continuously monitors system health metrics, generates alerts when thresholds are breached, and exports metrics in Prometheus format for Grafana dashboards.

---

## 4. observer/ (2 files)

### 4.1 `observer/__init__.py` — ~5 lines

Exports `Observer`.

---

### 4.2 `observer/observer.py` — ~180 lines

**Purpose:** Read-only audit observer that logs all system events without modifying state.

**Classes:**

#### `Observer`
- `__init__(self, log_dir='observer/')`
- Internal state: `total_signals=0`, `executed_signals=0`, `cumulative_pnl=0.0`, `avg_latency_ms=0.0`
- Uses `asyncio.Queue(maxsize=10000)` for non-blocking event processing
- `RotatingFileHandler` — 10MB per file, 5 backup files
- `observe_signal(self, signal: Dict)` — Records signal arrival, increments counters
- `observe_execution(self, execution: Dict)` — Records fill, updates P&L and latency
- `observe_risk_event(self, event: Dict)` — Records risk breaches
- `get_state(self) → Dict` — Returns current observer state as dict
- `save_state(self)` — Writes state to `observer/state.json`
- `_process_queue(self)` — Async consumer that writes events to `observer/events.log`

**Expected Behavior:** Sits as a passive auditor of all trading activity. Never modifies signals or execution — only records. State persisted to JSON for the observer_app dashboard to read.

---

## 5. optimization/ (3 files)

### 5.1 `optimization/__init__.py` — ~5 lines

Exports `AIOptimizer`, `QuantumBackendTester`.

---

### 5.2 `optimization/ai_optimizer.py` — ~115 lines

**Purpose:** AI-driven signal optimization with reward-based retraining.

**Classes:**

#### `AIOptimizer`
- `__init__(self, model=None)`
- `compute_reward(self, pnl: float, risk_metric: float) → float` — Reward function balancing P&L and risk
- `optimize_signal(self, signal: Dict, recent_performance: List[float]) → Dict`
  - If `compute_reward(mean_pnl, risk) < -200.0` → triggers model retraining
  - `min_retrain_interval = 300 seconds` (5-minute cooldown between retrains)
  - Adjusts signal confidence and direction based on optimization

**Expected Behavior:** Monitors live signal quality and retrains the underlying ML model when cumulative reward drops below -200 (severe underperformance).

---

### 5.3 `optimization/quantum_validation.py` — ~370 lines

**Purpose:** Quantum computing validation and benchmarking.

**Classes:**

#### `QuantumBackendTester`
- Tests QAOA circuit execution on different backends
- Validates qubit fidelity and gate accuracy

#### `QuantumAdvantageBenchmark`
- `__init__(self, assets=[2, 3, 4, 5, 6, 8])`
- Benchmarks quantum vs classical portfolio optimization across problem sizes
- Measures wall clock time, solution quality, and approximation ratio

#### `QuantumErrorMitigation`
- Implements Zero Noise Extrapolation (ZNE) with Richardson extrapolation
- Reduces circuit noise effects on results

#### `QuantumFeatureMap`
- Supports angle encoding and IQP (Instantaneous Quantum Polynomial) encoding
- Maps classical financial data to quantum Hilbert space

**Expected Behavior:** Validates quantum computing components before production use — ensures backends work correctly, measures actual quantum advantage, and applies error mitigation.

---

## 6. reducer/ (2 files)

### 6.1 `reducer/__init__.py` — ~5 lines

Exports `DecisionReducer`.

---

### 6.2 `reducer/reducer.py` — ~220 lines

**Purpose:** Aggregates signals from multiple strategies into a single trading decision.

**Constants:**
```python
strategy_weights = {
    'Momentum': 1.0,
    'MarketMaking': 0.5,
    'PairsTrading': 0.8,
    'VolArb': 0.7
}
min_confidence = 0.6
exit_confidence = 0.4
```

**Classes:**

#### `DecisionReducer`
- `__init__(self, strategy_weights: Dict, min_confidence=0.6, exit_confidence=0.4)`
- `reduce(self, signals: List[Dict]) → Optional[Dict]` — Weighted vote aggregation:
  1. Weight each signal by `strategy_weights[signal.strategy]`
  2. Direction = weighted average of all directions
  3. Confidence = weighted average of all confidences
  4. **Regime adaptation:**
     - `HIGH_VOL` → `min_confidence = 0.8`
     - `CRASH` → `min_confidence = 0.95`
  5. **Signal age decay:** Signals lose 5% confidence per tick of age
  6. **EXIT priority:** If any signal has `exit=True` with confidence > exit_confidence, output EXIT regardless
  7. If final confidence < min_confidence → no trade (returns None)

**Expected Behavior:** Merges 4+ concurrent strategy signals into a unified decision. Conservative in volatile markets (higher confidence required), aggressive in calm markets.

---

## 7. session/ (1 file)

### 7.1 `session/session_controller.py` — ~35 lines

**Purpose:** Trading session lifecycle management.

**Constants:**
```python
MAX_SESSION_DURATION = 36000  # 10 hours in seconds
```

**Classes:**

#### `SessionGuard`
- `__init__(self, max_loss=500.0)`
- `check_session(self, elapsed_seconds: float, cumulative_pnl: float) → bool`
  - Returns False (halt trading) if `elapsed_seconds > 36000` OR `cumulative_pnl < -max_loss`
- `start_session(self) → datetime` — Records session start time
- `end_session(self) → Dict` — Returns session summary

**Expected Behavior:** Hard safety limits: automatically kills trading after 10 hours or $500 cumulative loss, whichever comes first.

---

## 8. live/ (3 files)

### 8.1 `live/__init__.py` — ~5 lines

Exports `RealTimeDataStream`, `RealTimeDataManager`.

---

### 8.2 `live/stream/streaming.py` — 784 lines

**Purpose:** Real-time market data streaming from multiple providers.

**Dataclasses:**
- `Quote(symbol, bid, ask, bid_size, ask_size, timestamp)`
- `Trade(symbol, price, size, timestamp, conditions)`
- `OrderBookSnapshot(symbol, bids: List[Tuple], asks: List[Tuple], timestamp)`

**Classes:**

#### `RealTimeDataStream`
- `__init__(self, provider='yahoo', api_key=None, buffer_size=10000)`
- Supported providers: `alpaca`, `polygon`, `iex`, `yahoo`, `binance`
- `connect(self) → None` — Establishes WebSocket connection with exponential backoff (max 60s cap)
- `subscribe(self, symbols: List[str], data_types=['quotes', 'trades'])`
- `on_quote(self, callback: Callable)` — Register quote event handler
- `on_trade(self, callback: Callable)` — Register trade event handler
- `_reconnect(self)` — Exponential backoff: `delay = min(2^attempt, 60)` seconds
- Internal buffer: `asyncio.Queue(maxsize=buffer_size)`

#### `HistoricalDataFetcher`
- `fetch(self, symbol, start, end, interval='1d') → pd.DataFrame`
- `fetch_multiple(self, symbols, start, end) → Dict[str, pd.DataFrame]`

**Expected Behavior:** Connects to market data WebSocket feeds, buffers incoming data, and dispatches to registered callbacks. Auto-reconnects on disconnection with exponential backoff.

---

### 8.3 `live/stream/realtime_manager.py` — 661 lines

**Purpose:** Singleton data manager for real-time and historical market data.

**Classes:**

#### `MarketDataConfig`
- Stores provider settings, API keys, cache parameters

#### `RealTimeDataManager` (Singleton)
- `__init__(self)` — Singleton pattern via `_instance` class variable
- `get_realtime_quote(self, symbol) → Dict` — Live bid/ask/last
- `get_realtime_price(self, symbol) → float` — Last traded price
- `get_realtime_prices(self, symbols: List[str]) → Dict[str, float]` — Batch pricing
- `get_historical_data(self, symbol, start, end, interval='1d') → pd.DataFrame` — With caching (TTL=300s)
- `get_historical_data_sync(self, symbol, start, end, interval='1d') → pd.DataFrame` — Synchronous wrapper
- `get_portfolio_data_sync(self, symbols, start, end) → Dict[str, pd.DataFrame]`
- `calculate_correlation_matrix(self, symbols, start, end) → pd.DataFrame`
- Cache: `Dict[str, Tuple[datetime, pd.DataFrame]]` with 300-second TTL

**BUG FIX #8:** Removed Phase 11 jitter injection that was corrupting live data feeds.

**Expected Behavior:** Provides a unified API for all market data access throughout the system. Caches historical data to avoid redundant API calls.

---

## 9. utils/ (9 files)

### 9.1 `utils/__init__.py` — 34 lines

Exports all utility modules: `alerting`, `config_loader`, `logger`, `math_helpers`, `performance_profiler`, `rate_limiter`, `retry`, `validators`.

---

### 9.2 `utils/alerting.py` — ~140 lines

**Purpose:** Multi-channel alerting (Telegram + Discord).

**Classes:**

#### `AlertManager`
- `__init__(self)` — Reads env vars: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `DISCORD_WEBHOOK_URL`
- `send_telegram(self, message: str)` — Posts via Telegram Bot API (non-blocking daemon thread)
- `send_discord(self, message: str)` — Posts via Discord webhook (non-blocking daemon thread)
- `send_alert(self, message: str, level='INFO')` — Sends to all configured channels
- `send_trade_alert(self, trade: Dict)` — Formatted trade notification

**Expected Behavior:** Fires alerts to Telegram and Discord in background threads so they never block the trading loop.

---

### 9.3 `utils/config_loader.py` — 538 lines

**Purpose:** TOML configuration loading with environment variable substitution, encryption, and hot-reload.

**Classes:**

#### `ConfigManager`
- `__init__(self, config_path: str = 'config/settings.toml')`
- **Environment variable substitution:** `${VAR_NAME:default_value}` syntax
- **Encryption:** Values prefixed with `ENC:` are decrypted using `cryptography.fernet.Fernet`
- **Hot reload:** Watches config file mtime, auto-reloads on change
- `get(self, key: str, default=None)` — Dot-notation access: `get('trading.risk_free_rate')`
- `set(self, key: str, value)`
- `encrypt_value(self, value: str) → str` — Returns `ENC:xxx` string
- **Default configuration values:**
  ```python
  black_scholes.risk_free_rate = 0.05
  monte_carlo.simulations = 10000
  ```

**Expected Behavior:** Loads TOML config with recursive env var expansion, supports encrypted secrets, and hot-reloads when file changes on disk.

---

### 9.4 `utils/logger.py` — ~300 lines

**Purpose:** Structured logging with specialized formatters and context management.

**Classes:**

#### `GigaFormatter`
- Custom log format with timestamps, level, module, and message

#### `JsonFormatter`
- JSON-structured log output for log aggregation systems

#### `LogContext`
- Context manager that adds fields to all log messages within scope

**Functions:**
- `setup_logging(level='INFO', use_loguru=True) → Logger` — Prefers loguru if available, falls back to standard logging
- `log_performance(func_name, duration_ms, metrics)` — Structured performance log
- `log_greek_calculation(greek_name, inputs, result, duration_ms)` — Domain-specific Greek calc log
- `log_trade_execution(trade: Dict)` — Trade execution audit log
- `log_strategy_performance(strategy_name, metrics: Dict)` — Strategy P&L log

**Expected Behavior:** Provides consistent, structured logging across the entire system. Greek calcukations, trades, and performance all use specialized log formats for easy filtering.

---

### 9.5 `utils/math_helpers.py` — 889 lines

**Purpose:** Optimized numerical computation functions with JIT compilation.

**JIT-compiled functions (numba):**
- `fast_exp(x)`, `fast_log(x)`, `fast_sqrt(x)` — Thin wrappers with numba JIT
- `fast_mean(arr)`, `fast_std(arr)`, `fast_skewness(arr)`, `fast_kurtosis(arr)` — Statistics without numpy overhead

**Functions (no JIT):**
- `normal_cdf(x) → float` — Abramowitz-Stegun approximation (max error ~7.5e-8)
- `normal_pdf(x) → float` — Standard Gaussian PDF
- `black_scholes_call(S, K, T, r, sigma) → float` — Analytical BS call price
- `black_scholes_put(S, K, T, r, sigma) → float` — Via put-call parity
- `delta_call(S, K, T, r, sigma) → float` — N(d1)
- `delta_put(S, K, T, r, sigma) → float` — N(d1) - 1
- `gamma(S, K, T, r, sigma) → float` — n(d1) / (S × σ × √T)
- `theta_call(S, K, T, r, sigma) → float` — Time decay per day
- `vega(S, K, T, r, sigma) → float` — Per 1% vol change
- `value_at_risk(returns, confidence=0.95, method='historical') → float` — Historical or parametric VaR
- `conditional_var(returns, confidence=0.95) → float` — Mean of tail losses
- `maximum_drawdown(equity_curve) → float`
- `sharpe_ratio(returns, risk_free=0.02) → float`
- `sortino_ratio(returns, risk_free=0.02) → float`
- `efficient_frontier_point(returns, cov_matrix, target_return, short_selling=False) → np.ndarray` — SLSQP optimization for minimum variance at target return
- `tangent_portfolio(returns, cov_matrix, risk_free_rate=0.02) → np.ndarray` — Analytical solution: `C^-1 × (μ - rf)`, with singular matrix fallback to pseudo-inverse
- `cubic_spline_interpolation(x, y, x_new) → np.ndarray`
- `implied_volatility_smile(market_prices, strikes, S, T, r) → np.ndarray` — Brent root-finding method (scipy.optimize.brentq), σ range [0.01%, 500%]
- `correlation_matrix_to_covariance(corr, volatilities) → np.ndarray`
- `nearest_positive_definite(matrix) → np.ndarray` — Eigenvalue clipping (min eigenvalue = 1e-8), symmetric reconstruction
- `annualize_returns(returns, periods_per_year=252) → float`
- `compound_annual_growth_rate(equity_curve) → float`

**`__main__` block:** Runs performance benchmarks on real SPY data — times BS pricing, Greek calculations, VaR computation.

**Expected Behavior:** High-performance math library. JIT functions are 10-100x faster than pure Python equivalents. All BS/Greek implementations use analytical formulas for speed.

---

### 9.6 `utils/performance_profiler.py` — 690 lines

**Purpose:** Function-level performance profiling with memory tracking.

**Classes:**

#### `PerformanceMetrics`
- Uses Welford's online algorithm for running mean/variance without storing all values
- Fields: `count, mean, m2, min_val, max_val`

#### `PerformanceProfiler`
- `__init__(self, enable_memory=False, enable_cpu=False, history_size=1000)`
- `start(self, name: str)` — Begin timing a named operation
- `stop(self, name: str) → float` — End timing, return duration in ms
- `get_stats(self, name: str) → Dict` — Returns mean, std, min, max, count, p95, p99
- `report(self) → pd.DataFrame` — Full report across all profiled operations

**Functions:**
- `profile_function(func)` — Decorator that profiles every call to `func`
- `benchmark_function(func, *args, n_runs=100) → Dict` — Returns: mean, median, min, max, std, p95, p99, ops_per_sec, plus optional memory stats (peak_memory_mb, avg_memory_mb)
- `compare_functions(funcs: List[Callable], *args) → pd.DataFrame` — Ranks functions by speed and memory usage

#### `ProfileBlock`
- Context manager: `with ProfileBlock("operation_name") as pb:`
- Automatically logs duration on exit

**Expected Behavior:** Provides detailed performance insights without modifying function code. Used throughout the system to identify bottlenecks — especially in Greeks calculation hot paths.

---

### 9.7 `utils/rate_limiter.py` — ~120 lines

**Purpose:** API rate limiting with two algorithms.

**Classes:**

#### `TokenBucketLimiter`
- `__init__(self, rate=5.0, burst=10, timeout=30.0)`
- `acquire(self) → bool` — Blocks until token available or timeout
- `try_acquire(self) → bool` — Non-blocking check
- Token refill: `rate` tokens per second, up to `burst` max

#### `SlidingWindowLimiter`
- `__init__(self, max_requests=1200, window=60.0)`
- Weight-based: each request can consume multiple units
- `acquire(self, weight=1) → bool`
- Sliding window: removes entries older than `window` seconds

**Expected Behavior:** Prevents exceeding API rate limits. Token bucket for bursty patterns (WebSocket reconnects), sliding window for steady-state limits (historical data requests).

---

### 9.8 `utils/retry.py` — ~240 lines

**Purpose:** Retry logic, circuit breaker, and state persistence.

**Functions:**
- `retry(max_attempts=3, backoff=2.0, max_delay=60.0, exceptions=(Exception,))` — Decorator with exponential backoff
- `async_retry(max_attempts=3, backoff=2.0, max_delay=60.0)` — Async version

**Classes:**

#### `CircuitBreaker`
- `__init__(self, failure_threshold=5, reset_timeout=60.0)`
- States: CLOSED (normal) → OPEN (failing) → HALF_OPEN (testing)
- Opens after 5 consecutive failures
- After 60s, transitions to HALF_OPEN and allows one test call
- If test succeeds → CLOSED; if fails → OPEN again

#### `StatePersistence`
- `save(self, state: Dict, filepath: str)` — Atomic write via temp file + rename (prevents corruption on crash)
- `load(self, filepath: str) → Dict`

**Expected Behavior:** Makes all external calls (API, WebSocket, database) resilient. Circuit breaker prevents cascade failures by stopping calls to repeatedly failing services.

---

### 9.9 `utils/validators.py` — 527 lines

**Purpose:** Input data validation using polars DataFrames.

**Classes:**

#### `ValidationResult`
- `is_valid: bool`, `errors: List[str]`, `warnings: List[str]`

**Functions:**
- `validate_price_data(df: pl.DataFrame) → ValidationResult` — Checks:
  - Required columns: open, high, low, close, volume
  - OHLCV consistency: high ≥ low, close within [low, high], open within [low, high]
  - No negative prices or volumes
  - Chronological ordering
  
- `validate_returns(returns: pl.Series, min_observations=100, max_return_threshold=0.5) → ValidationResult` — Checks:
  - Minimum 100 data points
  - No single return > 50% (likely error)
  - Warns on high skewness (>2) or kurtosis (>10)

- `validate_greeks(greeks: Dict) → ValidationResult` — Cross-Greek validation:
  - Delta: [-1, 1]
  - Gamma: [0, ∞)
  - Theta: typically negative
  - Vega: [0, ∞)
  - Put-call parity check on delta

- `validate_portfolio(weights: Dict, positions: Dict) → ValidationResult` — Checks:
  - Weights sum to ~1.0 (tolerance 0.01)
  - `max_single_position = 0.1` (10%)
  - `max_sector_weight = 0.3` (30%)
  - `max_gross_exposure = 2.0` (200%)

- `validate_option_chain(chain: pd.DataFrame) → ValidationResult` — Checks:
  - Strike monotonicity
  - Bid ≤ Ask
  - No negative prices
  - Calendar spread arbitrage (later expiry ≥ earlier expiry)

**Expected Behavior:** Comprehensive input validation before any data enters the pipeline. Uses polars for high-performance validation of large datasets.

---

## 10. artifacts/ (1 file)

### 10.1 `artifacts/definitions.py` — 127 lines

**Purpose:** Core artifact type definitions for the Research → Live pipeline.

**Enums:**
- `MarketRegime` — `UNKNOWN`, `LOW_VOL_BULL`, `LOW_VOL_BEAR`, `HIGH_VOL_CRASH`, `HIGH_VOL_RALLY`, `SIDEWAYS_CHOP`
- `TimeHorizon` — `HFT` (<1ms), `INTRA_MINUTE` (1 min), `INTRA_HOUR` (1 hour), `DAILY` (1 day)

**Dataclasses:**
- `Context(regime: MarketRegime, horizon: TimeHorizon, asset_class: str, constraints: Dict, valid_until: datetime)`
- `Artifact(id: UUID4, name: str, version: str, created_at: datetime, context: Context, content: Dict)`
  - `validate(self) → bool` — Checks required fields present and context not expired
  - `to_dict(self) → Dict` — Serialization with datetime ISO format
- `SignalArtifact(Artifact)` — Adds: `direction: float [-1, 1]`, `strength: float [0, 1]`, `confidence: float [0, 1]`
  - Validates ranges in `__post_init__`

**Expected Behavior:** Defines the canonical data structures that flow through the Research → Reducer → Live pipeline. All signals are wrapped in SignalArtifact with market context.

---

## 11. visualization/ (17 files)

### 11.1 `visualization/__init__.py` — 110 lines

**Purpose:** Package exports.

**Exports from `components`:**
- `COLORS`, `metric_card`, `metric_row`, `styled_dataframe`, `trade_table`, `equity_chart`, `drawdown_chart`, `allocation_pie`, `returns_histogram`, `correlation_heatmap`, `symbol_selector`, `date_range_selector`, `risk_parameters_input`, `optimization_parameters_input`, `status_indicator`, `progress_bar`, `alert_box`, `card`, `section_header`, `empty_state`

**Exports from `charts`:**
- `candlestick_chart`, `multi_asset_chart`, `volatility_surface`, `volatility_smile`, `greeks_chart`, `payoff_diagram`, `var_chart`, `risk_decomposition_chart`, `efficient_frontier`, `weights_timeline`, `backtest_results_chart`, `monthly_returns_heatmap`, `quantum_circuit_diagram`, `quantum_probability_chart`

---

### 11.2 `visualization/app.py` — 1540 lines

**Purpose:** Main Streamlit application integrating all ~70 modules.

**Configuration:**
```python
st.set_page_config(
    page_title="GIGA System - Financial Analysis Platform",
    layout="wide"
)
```

**PHASE 2 WARNING:** "This application imports EVERYTHING directly, creating a monolithic failure point. Violates Air-Gap."

**Navigation:** 30+ pages organized by category:
- **Dashboard**: Main dashboard
- **Portfolio/Trading**: Portfolio, Strategy Analysis
- **Risk/Analytics**: Risk Management
- **Market Data**: Real-time Monitor, Storage Manager, Technical Indicators
- **Quantum**: Quantum Computing, Quantum Monte Carlo, Quantum Risk, Quantum ML
- **ML**: Regime Detection, Volatility Forecast, Feature Engineering
- **Education**: Education Mode, Greeks Dashboard
- **Standalone Dashboards**: Risk Dashboard, Greeks Dashboard, P&L Attribution, Correlation Heatmap, Statistical Plots, Quantum Visualizer
- **Settings**: General, API Keys, Data Sources

**Render Functions (inline):**
- `render_dashboard()` — 5-column metrics (Portfolio Value, Daily P&L, Sharpe, VaR, Active Positions), equity chart (real SPY/QQQ data normalized to $1M), asset allocation, recent trades, top performers, risk alerts
- `render_portfolio()` — 3 tabs: Current Holdings (7 stocks with P&L), Optimization (6 methods including Quantum QAOA), Analytics (real correlation matrix)
- `render_strategy()` — Active strategies (Pairs/Momentum/MeanRev/MM), Backtest runner, Strategy configuration
- `render_risk()` — VaR/CVaR/Vol/Beta, Greeks exposure table, sector exposure, stress testing (6 historical scenarios)
- `render_quantum()` — QAOA/VQE/Grover, Monte Carlo VaR, Amplitude Estimation, classical vs quantum comparison
- `render_settings()` — Theme/currency/risk-free rate, API keys (password fields), data source config, R integration path
- `render_pairs_trading_demo()` — Cointegration-based pairs trading interface
- `render_momentum_demo()` — Momentum/trend following strategy demo
- `render_market_making_demo()` — Bid-ask spread management, inventory skew
- `render_performance_metrics()` — Sharpe/Sortino/MaxDD/WinRate display
- `render_benchmark_comparison()` — Alpha/Beta vs benchmark
- `render_realtime_monitor()` — Live streaming data monitor
- `render_storage_manager()` — Data storage stats, query interface, cache maintenance
- `render_technical_indicators()` — SMA/EMA/MACD/RSI/Bollinger indicator selection
- `render_quantum_monte_carlo()` — QMC demonstration with speedup metrics
- `render_quantum_risk()` — Quantum VaR comparison
- `render_quantum_ml()` — QSVM/VQC/QNN model training
- `render_regime_detection()` — HMM/Clustering regime identification
- `render_volatility_forecast()` — GARCH/EGARCH/LSTM volatility models
- `render_feature_engineering()` — Feature generation pipeline
- `render_math_helpers()` — Math function browser & perf test
- `render_system_profiler()` — System performance metrics and function profiling

**Expected Behavior:** Central hub routing to all system features. Each render function tries to import its module with graceful fallback if unavailable.

---

### 11.3 `visualization/charts.py` — 988 lines

**Purpose:** Reusable Plotly chart functions for financial data.

**Constants:**
```python
CHART_COLORS = {
    'up': '#00ff88',
    'down': '#ff4444',
    'primary': '#00D4AA',
    'secondary': '#FF6B6B'
}
DEFAULT_LAYOUT = {
    'template': 'plotly_dark',
    'paper_bgcolor': '#161b22',
    'plot_bgcolor': '#0d1117'
}
```

**Functions:**
- `candlestick_chart(df, show_volume=True, indicators=None) → go.Figure` — OHLCV candlestick with optional volume subplot and indicator overlays
- `multi_asset_chart(prices: Dict, normalize=True) → go.Figure` — Multi-line chart, optional base-100 normalization
- `volatility_surface(strikes, maturities, ivs) → go.Figure` — 3D Plotly Surface of implied volatility
- `volatility_smile(strikes, ivs, spot) → go.Figure` — 2D IV smile with ATM/moneyness markers
- `greeks_chart(strikes, greeks: Dict, spot) → go.Figure` — 2×2 subplots: Delta, Gamma, Theta, Vega with spot price vertical line
- `payoff_diagram(strikes, payoffs) → go.Figure` — Option strategy payoff with breakeven shading
- `var_chart(returns, var_levels: Dict) → go.Figure` — Return histogram with VaR/CVaR vertical lines
- `risk_decomposition_chart(risk_contributions: Dict) → go.Figure` — Horizontal bar chart of risk contribution by asset
- `efficient_frontier(portfolios: List[Dict], optimal: Dict) → go.Figure` — Scatter plot colored by Sharpe, star marker for optimal
- `weights_timeline(dates, weights_history: Dict) → go.Figure` — Stacked area chart of portfolio weight evolution
- `backtest_results_chart(timestamps, equity, benchmark, trades) → go.Figure` — 3-row subplot: equity curve + drawdown + daily returns
- `monthly_returns_heatmap(returns: pd.Series) → go.Figure` — Year × Month heatmap with RdYlGn colorscale, annotated %
- `quantum_circuit_diagram(circuit_data: Dict) → go.Figure` — Gate visualization with colored boxes and CNOT connections
- `quantum_probability_chart(states: List[str], probabilities: List[float]) → go.Figure` — Bar chart of quantum state measurement outcomes

**Expected Behavior:** All charts use dark theme and consistent color palette. Functions are stateless — each returns a Plotly Figure that the caller passes to `st.plotly_chart()`.

---

### 11.4 `visualization/components.py` — 573 lines

**Purpose:** Reusable Streamlit UI components.

**Constants:**
```python
COLORS = {
    'primary': '#00D4AA',
    'secondary': '#FF6B6B',
    'tertiary': '#4ECDC4',
    'positive': '#00ff88',
    'negative': '#ff4444',
    'warning': '#ffaa00'
}
```

**Functions:**

*Metric Display:*
- `metric_card(title, value, delta=None, icon=None)` — Styled metric card with CSS
- `metric_row(metrics: List[Dict], columns=4)` — Grid of metric cards

*Data Tables:*
- `styled_dataframe(df, highlight_columns=None, format_dict=None)` — Styled pandas DataFrame with conditional formatting
- `trade_table(trades: List[Dict])` — Formatted trade history table

*Charts:*
- `equity_chart(timestamps, equity, benchmark=None)` — Simple equity line chart
- `drawdown_chart(timestamps, drawdown)` — Filled drawdown area chart
- `allocation_pie(weights: Dict)` — Donut chart of portfolio weights
- `returns_histogram(returns, var_95=None)` — Histogram with optional VaR line
- `correlation_heatmap(corr_matrix: pd.DataFrame)` — Annotated heatmap

*Inputs:*
- `symbol_selector(default_symbols=['AAPL','GOOGL','MSFT',...])` — Multiselect with 15 common symbols
- `date_range_selector(default_start='2022-01-01')` — Date input pair
- `risk_parameters_input()` — Confidence level (0.95), horizon (10 days), risk-free rate (2%)
- `optimization_parameters_input()` — Method selection, risk aversion, min/max weight, quantum toggle

*Status:*
- `status_indicator(status: str, message: str)` — Green/yellow/red status dot
- `progress_bar(value: float, label: str)` — Styled progress bar
- `alert_box(message, level='info')` — Colored alert box (info/warning/error/success)

*Layout:*
- `card(title, content_func)` — Styled container with title header
- `section_header(title, subtitle=None)` — Section divider with optional description
- `empty_state(message, icon='📊')` — Placeholder for empty data states

---

### 11.5 `visualization/correlation_heatmap.py` — 926 lines

**Purpose:** Interactive correlation analysis dashboard.

**Classes:**

#### `CorrelationHeatmap`
- `__init__(self)` — Default 26 assets: tech (AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, NFLX, AMD, INTC), finance (JPM, BAC, GS, MS, WFC), ETFs (SPY, QQQ, IWM, DIA), commodities (GLD, SLV, USO), bonds (TLT, IEF, AGG), crypto (BTC-USD)

**Methods:**
- `load_real_correlation_data(self, symbols, start_date, end_date) → Tuple[pd.DataFrame, Dict]` — Fetches real data from `get_data_manager()`
- `generate_sample_correlation_data(self, n_assets=10) → ...` — **DEPRECATED**, kept for fallback
- `calculate_correlation_matrix(self, returns_df, method='pearson') → pd.DataFrame` — Supports pearson/spearman/kendall
- `calculate_rolling_correlations(self, returns_df, window=60) → pd.DataFrame`
- `perform_hierarchical_clustering(self, corr_matrix) → List[int]` — scipy Ward linkage + squareform distance → reordered indices
- `create_correlation_heatmap(self, corr_matrix, cluster=False) → go.Figure` — Annotated heatmap, optional clustering reorder
- `create_correlation_network(self, corr_matrix, threshold=0.5) → go.Figure` — networkx spring layout, edges colored by sign (red=positive, blue=negative), width = |correlation| × 5
- `create_time_varying_correlation(self, returns_df, pair) → go.Figure` — 2-row: rolling correlation + normalized prices
- `create_correlation_distribution(self, corr_matrix) → go.Figure` — Histogram + box plot of off-diagonal correlations
- `calculate_correlation_statistics(self, corr_matrix) → Dict` — Mean/median/std of correlations, top 5 most correlated pairs
- `run_dashboard(self)` — 5 tabs: Heatmap, Network, Time-Varying, Distribution, Statistics. Sidebar: method selection, symbol selection, date range, clustering toggle. CSV export buttons.

---

### 11.6 `visualization/education_mode.py` — 987 lines

**Purpose:** Interactive educational tutorials for quantitative finance concepts.

**Classes:**

#### `EducationMode`
- Session state tracking: `completed_modules`, `quiz_scores`, `progress`

**Methods:**
- `black_scholes_tutorial(self)` — Interactive BS pricing:
  - LaTeX formula display
  - Sliders for S, K, T, r, σ
  - Price sensitivity chart with payoff overlay
  - Quiz (3 questions: price effect of vol increase, intrinsic value, time value)
  
- `greeks_tutorial(self)` — Interactive Greeks calculator:
  - Numerical differentiation to verify analytical Greeks
  - Scenario analysis: price shock, vol shock, time decay, rate shock
  - Shows predicted vs actual change with accuracy percentage
  
- `portfolio_optimization_tutorial(self)` — Real data portfolio optimization:
  - Fetches real data from `get_data_manager()` for asset characteristics
  - Weight sliders for 5 assets
  - Risk-return scatter plot
  - Diversification benefit: `1 - portfolio_vol / weighted_avg_vol`
  - Quiz on diversification concepts
  
- `monte_carlo_tutorial(self)` — Bootstrap Monte Carlo simulation:
  - Uses real SPY historical returns
  - Bootstrap sampling (not parametric)
  - Up to 50 sample paths + mean + 5th/95th percentile bands
  - Final price distribution histogram
  - VaR calculation, probability of profit
  
- `run_dashboard(self)` — 4 tutorial tabs with sidebar progress tracking
  - Progress bar per module
  - Completion certificate (congratulations message) when all modules finished

---

### 11.7 `visualization/education_viz.py` — 746 lines

**Purpose:** Advanced educational visualizations mapping ancient Greek mathematics to modern quant trading.

**Classes:**

#### `GreekMathTutorial`
- 5 lessons:
  1. **Eudoxus Exhaustion → κ estimation** — Demonstrates OLS κ series estimation from price data
  2. **Pythagoras Harmony → Correlation** — Musical ratios → price correlation ratios
  3. **Archimedes Volume → Market Impact** — Displaced volume → `impact = √(V/ADV)` market impact model
  4. **Euclid GCD → Lot Sizing** — GCD algorithm for optimal lot sizing
  5. **Thales Proportionality → Pair Trading** — Proportional reasoning → z-score pair trading
- Each lesson includes interactive exercises with generated data

#### `RiskSurfaceGenerator`
- `generate_delta_surface(self) → Tuple[np.ndarray, np.ndarray, np.ndarray]` — 50×50 grid over strike × maturity
- `generate_gamma_surface(self)` — Same grid
- `generate_vega_surface(self)` — Same grid
- Default params: spot=100, r=0.05, σ=0.25

#### `DomainTimeline`
- 5 domains with signals derived from price data:
  1. **State-Space Ω** — Log-return z-score
  2. **Variational** — Gradient magnitude
  3. **Stochastic** — Local volatility
  4. **Time Asymmetry** — Skewness of returns
  5. **Information Geometry** — Entropy of return distribution
- All signals normalized to [0, 1]

#### `RegimeProgressionDashboard`
- 4 regimes derived from volatility percentiles: Low Vol, Normal, High Vol, Crisis
- Transition matrix (4×4) computed from consecutive regime pairs
- Capital allocations per regime:
  ```
  Low Vol:  Equity=80%, Hedges=5%,  Cash=15%
  Normal:   Equity=60%, Hedges=15%, Cash=25%
  High Vol: Equity=30%, Hedges=30%, Cash=40%
  Crisis:   Equity=10%, Hedges=40%, Cash=50%
  ```
- Duration tracking per regime

#### `GreekVsModernComparison`
- 4 comparisons with dual implementations:
  1. **MR Estimation**: Bisection method (ancient) vs MLE (modern)
  2. **Risk Measurement**: Archimedean displacement vs VaR
  3. **Portfolio Construction**: Pythagorean harmony vs Markowitz optimization (SLSQP)
  4. **Signal Generation**: 5-domain composite signal vs simple momentum

---

### 11.8 `visualization/greeks_dashboard.py` — 631 lines

**Purpose:** Interactive 3D Greeks visualization dashboard.

**Classes:**

#### `GreeksDashboard`
- Default params: `spot=100, strike=100, T=0.25, r=0.05, vol=0.20`
- Greek colorscales: `delta=viridis, gamma=plasma, theta=inferno, vega=cividis, rho=turbo`

**Methods:**
- `create_parameter_controls(self)` — Sidebar sliders for all BS parameters
- `generate_3d_surface_data(self, greek_name, params, resolution=50)` — Meshgrid over spot price × volatility
- `create_3d_surface_plot(self, greek_name, params) → go.Figure` — 3D Plotly Surface
- `create_greek_comparison_chart(self, params) → go.Figure` — 2×2 subplots: Delta/Gamma/Theta/Vega across spot price range with current spot vertical line
- `create_heatmap_visualization(self, params) → go.Figure` — Delta heatmap over spot × time-to-expiry
- `display_greek_metrics(self, params)` — Greek values with profiled calculation times (ms)
- `display_option_value(self, params)` — Call/Put price, intrinsic value, time value breakdown
- `run_dashboard(self)` — 4 tabs: 3D Surfaces, Greek Comparison, Heatmap, Advanced Analysis
  - Advanced tab includes LaTeX formulas for all Greeks + interpretation guide
- `main()` — Streamlit page config + CSS + dashboard instantiation

---

### 11.9 `visualization/observer_app.py` — 55 lines

**Purpose:** Minimal observer monitoring dashboard.

**Logic:**
- Reads `observer/state.json` → displays metrics: total_signals, executed_signals, cumulative_pnl, avg_latency_ms
- Reads `observer/events.log` → displays last 20 events (signal arrivals, execution confirmations)
- Streamlit page with auto-refresh capability

**Expected Behavior:** Lightweight dashboard for operations monitoring — shows what the Observer has recorded.

---

### 11.10 `visualization/pnl_attribution.py` — 746 lines

**Purpose:** P&L attribution analysis with waterfall charts.

**Constants:**
```python
colors = {
    'positive': '#2E8B57',
    'negative': '#DC143C',
    'neutral': '#4682B4',
    'total': '#FF6347',
    'benchmark': '#708090'
}
```

**Classes:**

#### `PnLAttributionDashboard`
- 6 P&L components: stock, options, FX, interest, transaction costs, funding costs

**Methods:**
- `load_real_pnl_data(self, symbols, start, end) → pd.DataFrame` — Asset-level P&L decomposition, transaction costs at 0.001% (1 bps), SPY benchmark
- `generate_sample_pnl_data(self, num_days=252) → pd.DataFrame` — **DEPRECATED**
- `create_waterfall_chart(self, attribution: Dict, title) → go.Figure` — go.Waterfall for period attribution
- `create_daily_waterfall(self, df, date_idx) → go.Figure` — Daily P&L decomposition
- `create_cumulative_pnl_chart(self, df) → go.Figure` — 8 traces: 6 components + total + benchmark
- `create_daily_pnl_chart(self, df) → go.Figure` — Stacked bars with total line on secondary y-axis
- `create_rolling_attribution_chart(self, df, window=20) → go.Figure` — Rolling mean + rolling volatility per component
- `create_performance_metrics_table(self, df) → pd.DataFrame` — Per-component: total, mean daily, std, Sharpe, max DD, win rate, skewness, kurtosis
- `create_correlation_analysis(self, df) → go.Figure` — 8×8 correlation heatmap of P&L components
- `run_dashboard(self)` — 5 tabs: Waterfall Charts, Time Series (cumulative/daily + alpha + rolling benchmark correlation), Rolling Analysis, Performance Metrics (with CSV download), Correlation Analysis (with key insights)

---

### 11.11 `visualization/quantum_visualizer.py` — 1221 lines

**Purpose:** Quantum computing visualization and interactive demos.

**Classes:**

#### `QuantumVisualizer`

**Methods:**
- `create_bloch_sphere(self, theta, phi) → go.Figure` — 3D unit sphere with state vector, coordinate axes, |0⟩/|1⟩/|+⟩/|+i⟩ labels
- `create_quantum_circuit(self, circuit_type) → go.Figure` — Gate diagrams:
  - QAOA: 4 qubits, H → RZ → CNOT → RX
  - VQE: 3 qubits, RY → CNOT → RZ
  - QMC: 5 qubits, H → P → CU → QFT
  - Color-coded gate boxes with CNOT connection lines
- `create_quantum_algorithm_flow(self, algorithm) → go.Figure` — Directed flowcharts for QAOA/VQE/QMC with node→edge steps
- `create_quantum_optimization_landscape(self, problem_type) → go.Figure` — 3D surfaces with multiple local minima:
  - Portfolio: `x² + y² + 0.5sin(5x)cos(5y)` + perturbations
  - Risk: quartic polynomial with oscillatory terms
  - Monte Carlo: Gaussian × quantum oscillation
  - Includes quantum optimization spiral path and global minimum marker
- `create_quantum_advantage_comparison(self) → go.Figure` — Classical O(2^n) vs Fault-Tolerant Quantum O(n²) vs NISQ O(n^1.5) scaling. Yellow region marks quantum advantage zone.
- `quantum_portfolio_demo(self)` — Full interactive QAOA portfolio optimization:
  - Sliders: num_assets (3-8), risk_tolerance, QAOA layers (1-5), return target, iterations (10-100)
  - Fetches REAL asset data from `get_data_manager()` for returns/volatilities/correlations
  - Simulated optimization with progress bar
  - Displays: return, risk, Sharpe, convergence plot, weights bar chart, QAOA circuit
- `quantum_risk_demo(self)` — Interactive VQE risk analysis:
  - VaR confidence level, time horizon, market stress, VQE depth, qubits, shots
  - Classical vs quantum VaR comparison
  - Risk distribution histogram with VaR lines
  - Quantum-enhanced correlation heatmap
- `quantum_state_demo(self)` — Bloch sphere visualization:
  - Theta/phi sliders
  - State amplitudes α/β, probabilities, purity, coherence
  - Bloch sphere 3D plot
  - State vector bar chart (real/imaginary components)
  - Financial interpretation (bull/bear probability, market uncertainty)
- `run_dashboard(self)` — Navigation: Portfolio Optimization, Risk Analysis, Quantum States, Algorithm Comparison, Optimization Landscapes
  - Quantum system status sidebar: simulator active, 127 qubits, 100μs coherence, 99.5% gate fidelity
  - Algorithm complexity comparison table
  - Educational resources (books, papers, platforms)
  - Quantum glossary (10 key terms)
  - Performance metrics footer: 15 circuits, 1247 optimization runs, 156 avg gates, 94.2% success rate

---

### 11.12 `visualization/risk_dashboard.py` — 878 lines

**Purpose:** Real-time risk monitoring dashboard.

**Constants:**
```python
risk_limits = {
    'var_95': 100000,       # $100K daily VaR limit
    'var_99': 200000,       # $200K daily VaR limit
    'volatility': 0.25,     # 25% annualized vol limit
    'concentration': 0.20,  # 20% max single position
    'leverage': 3.0,        # 3x max leverage
    'correlation': 0.80     # 80% max correlation
}
risk_colors = {
    'low': '#2E8B57',
    'medium': '#FFA500',
    'high': '#DC143C',
    'extreme': '#8B0000'
}
```

**Classes:**

#### `RiskDashboard`

**Methods:**
- `load_real_portfolio_data(self, symbols, start, end) → Tuple[pd.DataFrame, Dict]` — From `get_data_manager()`
- `generate_sample_portfolio_data(self) → ...` — **DEPRECATED**, tries real data with fallback
- `calculate_portfolio_risk_metrics(self, returns_df, weights) → Dict` — Returns: var_95, var_99, cvar_95, cvar_99, volatility, max_drawdown, sharpe_ratio, skewness, kurtosis, herfindahl_index, max_weight, plus calculation times in ms
- `create_var_gauge_chart(self, var_value, limit, title) → go.Figure` — Gauge with 4 zones: green (0-50%), yellow (50-75%), orange (75-100%), red (>100% of limit)
- `create_risk_decomposition_chart(self, returns_df, weights, assets) → go.Figure` — Marginal VaR approach, sorted horizontal bar chart
- `create_risk_time_series(self, returns_df, weights, window=30) → go.Figure` — 3-row: rolling volatility, rolling VaR 95%, rolling VaR 99%
- `create_stress_testing_scenarios(self, returns_df, weights) → go.Figure` — 6 scenarios:
  1. Market Crash: -20%
  2. Volatility Spike: 2× current vol
  3. Correlation Breakdown: all correlations → 0.9
  4. Interest Rate Shock: sudden +5%
  5. Liquidity Crisis: -15%
  6. Black Swan: -30%
- `create_monte_carlo_risk_simulation(self, returns_df, weights) → go.Figure` — Simulates 10,000 portfolio return scenarios, P&L distribution with VaR lines, expected P&L indicator
- `run_dashboard(self)` — 5 tabs: Risk Gauges (VaR/Vol gauges + detailed metrics table), Risk Decomposition (chart + composition table), Time Series (rolling window slider), Stress Testing (6 scenarios + descriptions), Monte Carlo (1K/5K/10K simulations)
  - Sidebar: VaR limit input, volatility limit slider, asset multiselect (15 options, 10 default), date range, real data toggle
  - Data source indicator: green for real, yellow for synthetic

---

### 11.13 `visualization/statistical_plots.py` — 940 lines

**Purpose:** Statistical analysis plots with optional R ggplot2 integration.

**Classes:**

#### `StatisticalPlots`
- `__init__(self)` — Checks for rpy2/R availability, 4 color palettes

**Methods:**
- `load_real_financial_data(self, symbol, start, end) → pd.DataFrame` — From `get_data_manager()`: returns, realized_vol (20-day rolling), log_returns, momentum (5d/20d)
- `generate_sample_financial_data(self, n_points=1000) → pd.DataFrame` — **DEPRECATED**: 3 volatility regimes, mean-reversion, 5% fat tail probability, volatility clustering
- `create_distribution_analysis(self, data, title) → go.Figure` — 4-panel:
  1. Histogram + fitted normal PDF overlay
  2. Q-Q plot (scipy.stats.probplot with theoretical quantiles)
  3. Box plot with quartile labels
  4. Empirical CDF vs normal CDF
- `create_regression_analysis(self, x, y, title) → go.Figure` — 4-panel:
  1. Scatter plot + OLS regression line
  2. Residuals vs fitted values
  3. Residuals distribution histogram
  4. Cook's distance influence plot (identifies high-leverage observations)
- `create_time_series_decomposition(self, series, title) → go.Figure` — 4-row:
  1. Original series
  2. Trend (moving average)
  3. Seasonal component (original - trend)
  4. Residual
- `create_correlation_analysis(self, data) → go.Figure` — 2-panel:
  1. Correlation heatmap (RdBu colorscale, annotated)
  2. Correlation network (circular layout, edges for |corr| > 0.5)
- `create_risk_return_analysis(self, returns_data) → go.Figure` — Scatter plot: annualized vol vs return, marker size = Sharpe ratio, color = Sharpe, approximate efficient frontier line
- `run_dashboard(self)` — 5 tabs: Distribution, Regression, Time Series, Correlation, Risk-Return
  - Sidebar: real data toggle, symbol input, date range, data point slider
  - Data summary metrics: points, mean return, volatility, Sharpe
  - Distribution tab: variable picker + Shapiro-Wilk and Kolmogorov-Smirnov normality tests
  - Regression tab: X/Y variable selectors
  - Time Series tab: Jarque-Bera test, autocorrelation, volatility clustering measure
  - Correlation tab: variable multiselect, styled correlation matrix
  - Risk-Return tab: per-asset metrics table (return, vol, Sharpe, skew, kurtosis, max DD)
  - Export: CSV download of summary statistics

---

### 11.14 `visualization/pages/backtest_page.py` — 572 lines

**Purpose:** Standalone backtesting page with strategy selection.

**Functions:**
- `render_backtest_page()` — Main page:
  - Sidebar: Strategy selection (6 options: Momentum, Mean Reversion, Pairs Trading, Trend Following, Market Making, Options Selling), asset multiselect (7 stocks), date range, initial capital ($10K-$10M, default $100K), commission (0-100 bps, default 10), slippage (0-100 bps, default 5), strategy-specific parameters (lookback, threshold/z-score)
  - Fetches real data from `get_data_manager().get_portfolio_data_sync()`
  - Normalized price preview chart
  
- `run_simulated_backtest(prices_df, strategy, capital, commission, slippage, lookback) → Dict` — Simplistic signal generation:
  - Momentum: `signal = mean/std > 0.5 → long, < -0.5 → short`
  - Mean Reversion: `z-score > 2 → short, < -2 → long`
  - Others: random signals (placeholder)
  - Returns: equity array, returns, drawdown, trades list, metrics dict (total_return, annual_return, annual_vol, sharpe, sortino=sharpe×1.2 simplified, max_drawdown, calmar, win_rate, num_trades, profit_factor)

- `display_backtest_results(results, prices_df, initial_capital)` — 4 tabs:
  1. **Equity Curve**: 2-row (portfolio + benchmark equity, daily returns bar)
  2. **Drawdown**: Drawdown chart + max DD line + statistics (max DD, avg DD, DD duration)
  3. **Trades**: Trade history dataframe (last 50), trade distribution by asset (pie), buy/sell distribution (bar)
  4. **Statistics**: 3-column layout (Performance/Risk/Trading metrics), monthly returns heatmap

---

### 11.15 `visualization/pages/options_page.py` — 500 lines

**Purpose:** Options pricing and Greeks analysis page.

**Functions:**
- `render_options_page()` — 4 tabs:

  **Tab 1 - Pricing:**
  - Sidebar: real-time price fetch toggle, symbol input, BS parameters (spot, strike, T, r, σ, dividend), option type
  - Call and put prices with intrinsic/time value breakdown
  - Put-call parity verification: C + Ke^(-rT) = P + S (shows parity error)
  
  **Tab 2 - Greeks:**
  - All 5 Greeks (Δ, Γ, Θ, V, ρ) with inline BS formulas
  - Uses OptionGreeks class if available, else inline scipy.stats calculations
  - Greeks across strikes chart (50 points from 0.7K to 1.3K): 2×2 subplot
  
  **Tab 3 - Volatility Surface:**
  - Synthetic vol surface with skew + term structure + smile:
    - `skew = 0.1 × (1 - K/S)` — Higher IV for OTM puts
    - `term_structure = 0.02 × ln(T/30)` — Upward sloping
    - `smile = 0.05 × (K/S - 1)²` — U-shape
  - 3D surface across 20 strikes × 7 maturities (7d to 365d)
  - 2D volatility smile with maturity slider
  
  **Tab 4 - Payoff Analysis:**
  - 7 strategies: Long Call, Long Put, Bull Call Spread, Bear Put Spread, Long Straddle, Iron Condor, Butterfly
  - Payoff diagram with breakeven points, current spot marker
  - Metrics: max profit, max loss, breakeven price, risk/reward ratio

---

### 11.16 `visualization/pages/portfolio_page.py` — 592 lines

**Purpose:** Portfolio optimization and analysis page.

**Functions:**
- `render_portfolio_page()` — 4 tabs:

  **Tab 1 - Optimization:**
  - Sidebar: 10 assets, 6 optimization methods (Mean-Variance, Risk Parity, Maximum Sharpe, Minimum Volatility, Black-Litterman, Quantum QAOA), risk aversion slider, min/max weight
  - Real data from `get_data_manager()` — annualized returns, covariance matrix
  - Optimization implementations:
    - Mean-Variance: Equal weight + Sharpe-based tilts scaled by risk aversion
    - Risk Parity: Inverse volatility weighting
    - Maximum Sharpe: Softmax of Sharpe ratios
    - Minimum Volatility: Inverse variance weighting
  - Displays: allocation pie chart, weights table, portfolio metrics (return, vol, Sharpe, max weight), efficient frontier scatter (100 random portfolios + optimal star)
  
  **Tab 2 - Analysis:**
  - Normalized price chart, correlation matrix heatmap, summary statistics table (ann. return, ann. vol, Sharpe, max DD, skewness, kurtosis)
  
  **Tab 3 - Risk:**
  - VaR 95%/99%, CVaR 95%, max daily loss
  - Return distribution with VaR lines
  - Marginal risk contribution (bar chart + table)
  
  **Tab 4 - Quantum:**
  - QUBO formulation: min Σ Q_ij x_i x_j
  - Quantum parameter controls (qubits, depth, shots, solver selection)
  - Runs simulated optimization → classical vs quantum weight comparison, quantum Sharpe improvement, quantum circuit visualization

---

### 11.17 `visualization/pages/quantum_page.py` — 631 lines

**Purpose:** Quantum computing experimentation lab.

**Functions:**
- `render_quantum_page()` — 4 tabs:

  **Tab 1 - QAOA:**
  - Sidebar: backend (Simulator/IBM/Local), qubits (2-16), depth (1-10), shots (100-10K), optimizer (COBYLA/SPSA/SLSQP/L-BFGS-B), max iterations
  - Problem types: Portfolio Selection (QUBO), Max-Cut, TSP Subset, Custom QUBO
  - QAOA circuit visualization: H gates (green) → ZZ cost layer (red connections) → Rx mixer (teal) → Measurement (yellow)
  - Simulated optimization with progress bar: 10 iterations, cost convergence plot
  - Solution distribution: top 16 measurement outcomes bar chart
  - Best solution: bitstring, cost value, probability
  
  **Tab 2 - VQE:**
  - Ansatz selection: RY (Hardware Efficient), UCCSD, Two-Local, Custom
  - Entanglement: linear, circular, full, SCA
  - Rotation gates: RX, RY, RZ (multiselect)
  - Portfolio Hamiltonian: λΣσ_ij Z_i Z_j - μΣr_i Z_i
  - Ansatz circuit visualization
  - Energy convergence: exponential decay toward ground state (-2.0), error percentage
  
  **Tab 3 - Portfolio:**
  - Asset multiselect (6 options), budget constraint slider
  - QAOA/VQE/QA solver selection, risk aversion
  - Real asset statistics from `get_data_manager()` (returns, volatilities, Sharpe)
  - Quantum optimization results: selected assets, weights, return, vol, Sharpe
  
  **Tab 4 - Risk:**
  - VaR confidence, time horizon, Monte Carlo scenarios
  - Quantum speedup display: Classical O(N) vs Quantum O(√N)
  - Real SPY bootstrap VaR/CVaR + histogram visualization

---

## Summary Statistics

| Directory | Files | Total Lines | Key Classes |
|-----------|-------|-------------|-------------|
| backtesting/ | 10 | ~4,600 | BacktestEngine, ExecutionSimulator, Portfolio, PerformanceAnalyzer, WalkForwardOptimizer, BenchmarkAnalyzer |
| feedback/ | 2 | ~160 | AdaptiveEngine, PositionSizer |
| monitoring/ | 2 | ~235 | MetricsCollector, AlertManager |
| observer/ | 2 | ~185 | Observer |
| optimization/ | 3 | ~490 | AIOptimizer, QuantumBackendTester, QuantumAdvantageBenchmark |
| reducer/ | 2 | ~225 | DecisionReducer |
| session/ | 1 | ~35 | SessionGuard |
| live/ | 3 | ~1,450 | RealTimeDataStream, RealTimeDataManager, HistoricalDataFetcher |
| utils/ | 9 | ~3,480 | ConfigManager, PerformanceProfiler, CircuitBreaker, TokenBucketLimiter |
| artifacts/ | 1 | 127 | MarketRegime, TimeHorizon, Context, Artifact, SignalArtifact |
| visualization/ | 17 | ~11,500 | CorrelationHeatmap, EducationMode, QuantumVisualizer, RiskDashboard, StatisticalPlots, PnLAttributionDashboard, GreeksDashboard |
| **TOTAL** | **52** | **~22,500** | |

## Architectural Observations

1. **Air-Gap Violation:** `app.py` imports everything directly, creating a monolithic coupling point. The intended Research→Reducer→Live pipeline with validation gates is bypassed in the UI.

2. **Real Data Transition:** All visualization modules mark `generate_sample_*()` as **DEPRECATED** in favor of `load_real_*()` using `get_data_manager()`. This is a Phase 2 migration.

3. **Consistent Design Language:** All visualizations use `plotly_dark` template with `#00D4AA` primary color. Components are reusable via `charts.py` and `components.py`.

4. **Quantum Computing:** Extensive but primarily simulated — QAOA/VQE circuits are visualized and their optimization progress is simulated with classical fallbacks. Real quantum execution would require Qiskit + IBM Quantum access.

5. **Regime Adaptivity:** The reducer, feedback engine, and education modules all implement regime-dependent behavior: tighter risk limits in HIGH_VOL/CRASH, position sizing via Capital Regime.

6. **Safety Layers:**
   - SessionGuard: 10-hour / $500 loss hard limits
   - ValidationPipeline: 6-check artifact validation
   - CircuitBreaker: 5-failure threshold with 60s recovery
   - Rate limiting: Token bucket (5/s burst 10) + sliding window (1200/min)
   - Observer: Read-only audit trail

7. **Performance Optimization:** JIT-compiled math helpers (numba), Welford's algorithm for streaming statistics, polars for high-throughput validation, caching with TTL for market data.
