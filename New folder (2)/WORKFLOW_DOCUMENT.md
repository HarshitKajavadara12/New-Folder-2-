# GIGA-SYSTEM — WORKFLOW DOCUMENT

## Document Purpose

This document defines the **complete operational workflows** for the GIGA-SYSTEM quantitative crypto/options trading platform. It maps every step-by-step process, decision gate, data transformation, and operational procedure across the system's 2-Pipeline Air-Gap Architecture — from research hypothesis generation through live trade execution, risk management, and continuous adaptation.

---

## Table of Contents

1. [Master Workflow Overview](#1-master-workflow-overview)
2. [Workflow 1: System Initialization](#2-workflow-1-system-initialization)
3. [Workflow 2: Research Pipeline Execution](#3-workflow-2-research-pipeline-execution)
4. [Workflow 3: 5-Domain Greek Alpha Analysis](#4-workflow-3-5-domain-greek-alpha-analysis)
5. [Workflow 4: Alpha Signal Generation](#5-workflow-4-alpha-signal-generation)
6. [Workflow 5: Factor Research & Validation](#6-workflow-5-factor-research--validation)
7. [Workflow 6: Machine Learning Pipeline](#7-workflow-6-machine-learning-pipeline)
8. [Workflow 7: Quantum Computing Pipeline](#8-workflow-7-quantum-computing-pipeline)
9. [Workflow 8: Backtesting & Walk-Forward](#9-workflow-8-backtesting--walk-forward)
10. [Workflow 9: Bridge Artifact Generation](#10-workflow-9-bridge-artifact-generation)
11. [Workflow 10: Live System Boot](#11-workflow-10-live-system-boot)
12. [Workflow 11: on_tick() Hot Path](#12-workflow-11-on_tick-hot-path)
13. [Workflow 12: Order Execution Lifecycle](#13-workflow-12-order-execution-lifecycle)
14. [Workflow 13: State Machine Transitions](#14-workflow-13-state-machine-transitions)
15. [Workflow 14: Risk Management Cascade](#15-workflow-14-risk-management-cascade)
16. [Workflow 15: Feedback & Adaptation Loop](#16-workflow-15-feedback--adaptation-loop)
17. [Workflow 16: Monitoring & Alerting](#17-workflow-16-monitoring--alerting)
18. [Workflow 17: R Analytics Integration](#18-workflow-17-r-analytics-integration)
19. [Workflow 18: Data Ingestion & Storage](#19-workflow-18-data-ingestion--storage)
20. [Workflow 19: Strategy Decision Reduction](#20-workflow-19-strategy-decision-reduction)
21. [Workflow 20: Emergency Shutdown](#21-workflow-20-emergency-shutdown)
22. [Workflow 21: Visualization & Dashboards](#22-workflow-21-visualization--dashboards)
23. [Workflow 22: CI/CD & Deployment](#23-workflow-22-cicd--deployment)
24. [Workflow 23: Full 2-Pipeline End-to-End](#24-workflow-23-full-2-pipeline-end-to-end)
25. [Cross-Workflow Dependencies](#25-cross-workflow-dependencies)
26. [Workflow Verification Matrix](#26-workflow-verification-matrix)

---

## 1. Master Workflow Overview

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                        GIGA-SYSTEM MASTER WORKFLOW                                │
│                                                                                  │
│  PIPELINE 1: RESEARCH                    PIPELINE 2: LIVE                        │
│  ┌──────────────────────────┐            ┌──────────────────────────┐            │
│  │ WF2: Research Pipeline   │            │ WF10: Live System Boot   │            │
│  │   ├── WF3: 5-Domain      │   TOML     │   ├── WF11: on_tick()    │            │
│  │   ├── WF4: Alpha Signal  │ ========>  │   ├── WF12: Order Exec   │            │
│  │   ├── WF5: Factor Res.   │  Bridge    │   ├── WF13: State Machine│            │
│  │   ├── WF6: ML Pipeline   │            │   ├── WF14: Risk Cascade │            │
│  │   ├── WF7: Quantum       │            │   ├── WF15: Feedback     │            │
│  │   ├── WF8: Backtesting   │            │   └── WF16: Monitoring   │            │
│  │   └── WF9: Bridge Gen    │            └──────────────────────────┘            │
│  └──────────────────────────┘                                                    │
│                                                                                  │
│  SUPPORT WORKFLOWS:                                                              │
│  WF1: Init | WF17: R Analytics | WF18: Data Ingestion | WF19: Decision Reduce   │
│  WF20: Emergency Shutdown | WF21: Visualization | WF22: CI/CD | WF23: End-to-End│
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Workflow 1: System Initialization

**Entry Points**: `QUICK_START.py`, `demo_complete_system.py`, `run_system_pipeline.py`, `launch_giga_system.py`

### Step-by-Step

| Step | Action | Component | File |
|------|--------|-----------|------|
| 1.1 | Parse command-line arguments | argparse | `run_system_pipeline.py` |
| 1.2 | Load TOML configuration | `ConfigManager` | `utils/config_loader.py` |
| 1.3 | Initialize logging system | `GigaFormatter`, `LogContext` | `utils/logger.py` |
| 1.4 | Initialize DuckDB database | `TimeSeriesDatabase` | `data/database.py` |
| 1.5 | Create database schema | Schema DDL | `data/database_layer.py` |
| 1.6 | Initialize rate limiters | `TokenBucketLimiter` | `utils/rate_limiter.py` |
| 1.7 | Check module health | Import verification | `scripts/health_check.py` |
| 1.8 | Select mode (research/live/full) | CLI `--mode` flag | `run_system_pipeline.py` |

### Decision Gate

```
--mode research  ──► Pipeline 1 only (WF2)
--mode live      ──► Pipeline 2 only (WF10) — requires existing bridge artifacts
--mode full      ──► Pipeline 1 → Bridge → Pipeline 2 (WF23)
```

### Initialization Flow

```
CLI Arguments ──► ConfigManager.load()
                       │
                       ├── system_config.toml    (global settings)
                       ├── database_config.toml  (DuckDB settings)
                       ├── models_config.toml    (ML hyperparams)
                       └── strategies_config.toml (THE Bridge Artifact)
                              │
                              ▼
                    Logger + Database + RateLimiter initialized
                              │
                              ▼
                    Mode Selection → Route to Pipeline 1 or 2
```

---

## 3. Workflow 2: Research Pipeline Execution

**Entry**: `run_system_pipeline.py --mode research` or `demo_complete_system.py`

### Step-by-Step

| Step | Action | Component | Output |
|------|--------|-----------|--------|
| 2.1 | Load historical data | `DataBridge`, `MarketDataLoader` | OHLCV DataFrame |
| 2.2 | Preprocess data | `preprocessing.py` (Polars) | Cleaned/imputed data |
| 2.3 | Compute indicators | `indicators.py` (Numba JIT) | SMA/EMA/RSI/MACD/ATR/BB |
| 2.4 | Run 5-Domain analysis | `AlphaSignalEngine` | Alpha signals (WF3) |
| 2.5 | Detect market regimes | `RegimeDetector` (GMM/HMM) | Regime labels |
| 2.6 | Forecast volatility | `VolatilityForecaster` | Vol forecasts |
| 2.7 | Run factor research | `AlphaFactorLibrary` | Factor values + z-scores (WF5) |
| 2.8 | Walk-forward validation | `GreekWalkForwardValidator` | Validation report (WF8) |
| 2.9 | Run backtests | `BacktestEngine` | Performance metrics (WF8) |
| 2.10 | Quantum optimization | `QuantumOptimizer` | Optimal weights (WF7) |
| 2.11 | Decision reduction | `DecisionReducer` | Final verdict (WF19) |
| 2.12 | Generate bridge artifacts | `TOMLGenerator` | `strategies_config.toml` (WF9) |

### Research Pipeline Flow

```
DataBridge.load() ──► Polars Preprocessing ──► Numba Indicators
         │
         ▼
  AlphaSignalEngine.generate_signal()
         │
    ┌────┼────────────────────────────────┐
    ▼    ▼              ▼                 ▼
Domain1  Domain2     Domain3           Domain4-5
(Regime) (Greeks)   (O-U κ fit)       (Ergodicity/Entropy)
    │    │              │                 │
    └────┼──────────────┼─────────────────┘
         ▼
  RegimeDetector + VolatilityForecaster + FactorLibrary
         │
         ▼
  WalkForwardValidator.run_validation()
         │
         ▼
  BacktestEngine.run() ──► PerformanceMetrics
         │
         ▼
  QuantumOptimizer.optimize() (if available)
         │
         ▼
  DecisionReducer.reduce() ──► TOMLGenerator.generate()
         │
         ▼
  strategies_config.toml  ◄── FROZEN BRIDGE ARTIFACT
```

---

## 4. Workflow 3: 5-Domain Greek Alpha Analysis

**Central Orchestrator**: `AlphaSignalEngine.generate_signal()` (`research/core/alpha_signal_engine.py`)

### Step-by-Step

| Step | Domain | Action | Component | Output |
|------|--------|--------|-----------|--------|
| 3.1 | Ω,Λ | Classify market state | `StateSpaceOmega.classify_state()` | `MarketState(vol, trend, liquidity, risk_score)` |
| 3.2 | Δ,Γ,Θ | Calculate variational sensitivity | `VariationalAnalyzer.calculate_delta()` | `SensitivityProfile(delta, gamma, theta, vega)` |
| 3.3 | μ,σ,κ | Fit Ornstein-Uhlenbeck | `StochasticModeler.fit_ornstein_uhlenbeck()` | `StochasticParams(mu, sigma, kappa)` |
| 3.4 | τ,ε | Check ergodicity | `TimeAsymmetryAnalyzer.check_ergodicity()` | `ErgodicityResult(tau, epsilon, is_ergodic)` |
| 3.5 | Η,Φ | Compute information geometry | `InformationGeometer.calculate_market_entropy()` | Shannon entropy, KL divergence |
| 3.6 | ALL | Generate composite signal | `AlphaSignalEngine.generate_signal()` | `AlphaSignal(direction, strength, confidence, regime)` |

### Domain Analysis Flow

```
prices[] ──► Domain 1: StateSpaceOmega
                  │
                  ├── vol_regime: VolatilityRegime (LOW/NORMAL/HIGH/EXTREME)
                  ├── trend_regime: TrendRegime (STRONG_DOWN...STRONG_UP)
                  └── liquidity_regime: LiquidityRegime (ILLIQUID/NORMAL/LIQUID)
                  
prices[] ──► Domain 2: VariationalAnalyzer
                  │
                  ├── delta: price sensitivity (log-diff first derivative)
                  ├── gamma: delta convexity (second derivative)
                  ├── theta: time decay rate
                  └── vega: volatility sensitivity

prices[] ──► Domain 3: StochasticModeler
                  │
                  ├── mu: drift parameter
                  ├── sigma: diffusion parameter
                  └── kappa: mean-reversion speed ◄── KEY HYPOTHESIS VARIABLE

prices[] ──► Domain 4: TimeAsymmetryAnalyzer
                  │
                  ├── tau: relaxation time
                  ├── epsilon: ergodic distance
                  └── is_ergodic: bool (time-avg ≈ ensemble-avg?)

prices[] ──► Domain 5: InformationGeometer
                  │
                  ├── shannon_entropy: H(X) = -Σ p(x) log p(x)
                  ├── market_entropy: binned return entropy
                  └── kl_divergence: D_KL(P || Q) ◄── KEY HYPOTHESIS VARIABLE
```

### Core Hypothesis Check

```python
# HIGH κ (fast mean-reversion) + LOW entropy (predictable) = ALPHA
if kappa > kappa_threshold and entropy < entropy_threshold:
    signal = STRONG_SIGNAL  # High confidence trade
else:
    signal = WEAK_SIGNAL    # Reduced confidence or no trade
```

---

## 5. Workflow 4: Alpha Signal Generation

**Component**: `AlphaSignalEngine` (`research/core/alpha_signal_engine.py` — 549 lines)

### Step-by-Step

| Step | Action | Input | Output |
|------|--------|-------|--------|
| 4.1 | Initialize 5-domain analyzers | Config | Domain objects |
| 4.2 | Feed price array to all domains | `prices: np.ndarray` | 5 domain results |
| 4.3 | Apply domain weights | Configurable weights | Weighted scores |
| 4.4 | Calculate composite direction | Weighted aggregation | BUY/SELL/HOLD |
| 4.5 | Calculate signal strength | Normalized (0–1) | Float |
| 4.6 | Calculate confidence | Signal consistency | Float |
| 4.7 | Track alpha decay | `AlphaDecayTracker` | Decayed signal |
| 4.8 | Package result | — | `AlphaSignal` dataclass |

### Signal Processing

```
5 Domain Results ──► Weight Application (configurable per domain)
                              │
                              ▼
                    Composite Score = Σ(weight_i × domain_result_i)
                              │
                    ┌─────────┼──────────┐
                    ▼         ▼          ▼
               direction   strength   confidence
               (BUY/SELL)  (0.0–1.0)  (0.0–1.0)
                    │         │          │
                    └─────────┼──────────┘
                              ▼
                    AlphaDecayTracker.track()
                              │
                              ▼
                    AlphaSignal(direction, strength, confidence, regime)
```

---

## 6. Workflow 5: Factor Research & Validation

**Components**: `AlphaFactorLibrary` + `GreekWalkForwardValidator`

### Step-by-Step

| Step | Action | Component | Output |
|------|--------|-----------|--------|
| 5.1 | Compute KappaAlpha | `KappaAlpha.compute()` | κ factor value + z-score |
| 5.2 | Compute EntropyAlpha | `EntropyAlpha.compute()` | Entropy factor + z-score |
| 5.3 | Compute GammaAlpha | `GammaAlpha.compute()` | Gamma exposure factor |
| 5.4 | Compute ErgodicityAlpha | `ErgodicityAlpha.compute()` | Ergodicity factor |
| 5.5 | Compute MomentumAlpha | `MomentumAlpha.compute()` | Momentum factor |
| 5.6 | Compute VolRegimeAlpha | `VolRegimeAlpha.compute()` | Vol regime factor |
| 5.7 | Hypothesis testing | `HypothesisTest` | p-values, significance |
| 5.8 | Factor combination | `AlphaCombiner.combine()` | Multi-factor composite |
| 5.9 | Cross-asset analysis | `CrossAssetGreekAlpha` | Cross-asset correlations |
| 5.10 | Factor attribution | `AlphaAttribution` | Factor decomposition |
| 5.11 | Walk-forward validation | `GreekWalkForwardValidator.run_validation()` | IS/OOS performance |

### Walk-Forward Validation Flow

```
Historical Data ──► Split into Rolling Windows
                         │
                    ┌─────┤
                    ▼     ▼
              Window_1  Window_2  ... Window_N
              [IS|OOS]  [IS|OOS]     [IS|OOS]
                    │     │           │
                    ▼     ▼           ▼
              Optimize  Optimize    Optimize    (In-Sample)
                    │     │           │
                    ▼     ▼           ▼
              Validate  Validate    Validate    (Out-of-Sample)
                    │     │           │
                    └─────┼───────────┘
                          ▼
              GreekValidationReport
              ├── IS/OOS Sharpe ratio comparison
              ├── α decay rate across windows
              ├── Overfitting detection (IS >> OOS)
              └── Statistical significance
```

---

## 7. Workflow 6: Machine Learning Pipeline

### Step-by-Step

| Step | Action | Component | Output |
|------|--------|-----------|--------|
| 6.1 | Generate features | `FeatureEngine` | Feature matrix (price/vol/tech/volume/regime) |
| 6.2 | Train regime model | `RegimeDetector` (GMM) | Regime labels (2-4 states) |
| 6.3 | Train HMM | `RegimeDetector` (HMM) | Hidden state sequence |
| 6.4 | Forecast EWMA vol | `EWMAVolModel.fit()` | EWMA vol forecast |
| 6.5 | Forecast GARCH vol | `GARCH11Model.fit()` | GARCH(1,1) forecast |
| 6.6 | Forecast HAR vol | `HARModel.fit()` | HAR realized vol forecast |
| 6.7 | Ensemble vol forecast | `VolatilityForecaster` | Combined vol estimate |
| 6.8 | Feed to AlphaSignalEngine | Integration | Enhanced alpha signals |

### ML Flow

```
Raw OHLCV ──► FeatureEngine
                   │
              ┌────┼────┐
              ▼    ▼    ▼
           Price  Vol  Technical  Volume  Regime
           feats  feats feats    feats   feats
              │    │    │         │       │
              └────┼────┼─────────┼───────┘
                   ▼
            Full Feature Matrix
                   │
         ┌─────────┼─────────┐
         ▼         ▼         ▼
  RegimeDetector  VolForecaster  AlphaFactorLibrary
  (GMM/HMM)     (EWMA/GARCH/HAR)
         │         │              │
         ▼         ▼              ▼
  Regime Labels  Vol Forecast   Factor Scores
         │         │              │
         └─────────┼──────────────┘
                   ▼
         AlphaSignalEngine (enhanced)
```

---

## 8. Workflow 7: Quantum Computing Pipeline

### Step-by-Step

| Step | Action | Component | Fallback |
|------|--------|-----------|----------|
| 7.1 | Check Qiskit availability | try/except import | Classical scipy.optimize |
| 7.2 | Define portfolio problem | `QuantumOptimizer` | Mean-variance classical |
| 7.3 | Build QAOA circuit | `QuantumApproximateOptimization` | Skip |
| 7.4 | Run VQE optimization | `VariationalQuantumEigensolver` | Classical eigen |
| 7.5 | Quantum Monte Carlo pricing | `QuantumMonteCarlo.estimate()` | Classical MC |
| 7.6 | Quantum SVM classification | `QuantumSupportVectorMachine` | Classical SVM |
| 7.7 | VQC regime detection | `VariationalQuantumClassifier` | Classical RF |
| 7.8 | Quantum risk analysis | `QuantumRiskAnalyzer` | Classical VaR |
| 7.9 | Error mitigation (ZNE) | `QuantumErrorMitigation` | Raw results |
| 7.10 | Backend validation | `QuantumBackendTester` | Simulator check |

### Graceful Degradation

```
try:
    import qiskit
    result = QuantumOptimizer.optimize(portfolio)  # QAOA/VQE
except ImportError:
    result = classical_scipy_optimize(portfolio)   # Fallback
```

**All quantum imports are wrapped in try/except with classical fallbacks.**

---

## 9. Workflow 8: Backtesting & Walk-Forward

### Step-by-Step

| Step | Action | Component | Output |
|------|--------|-----------|--------|
| 8.1 | Initialize BacktestEngine | Config + Strategy | Engine instance |
| 8.2 | Load historical data | `DataBridge` | OHLCV data |
| 8.3 | Event-driven simulation | `BacktestEngine.run()` | Trade events |
| 8.4 | Fill simulation | `ExecutionSimulator` | Filled orders |
| 8.5 | Portfolio tracking | `Portfolio` | Positions, cash, equity |
| 8.6 | Calculate metrics | `PerformanceAnalyzer` | Sharpe, sortino, max DD |
| 8.7 | Bootstrap CI | `PerformanceMetrics` | Confidence intervals |
| 8.8 | Benchmark comparison | `BenchmarkAnalyzer` | Relative metrics |
| 8.9 | Store results | `BacktestResultStore` | JSON + checksum |
| 8.10 | Air-gap validation | `AirGapValidator` | Determinism/NaN/range checks |
| 8.11 | Walk-forward | `WalkForwardOptimizer` | Rolling window OOS |
| 8.12 | Greek-aware backtest | `GreekAwareBacktester` | Greek-adjusted metrics |
| 8.13 | Visualize results | `BacktestVisualizer` | Plotly equity curves |

### Backtesting Event Loop

```
for each bar in historical_data:
    │
    ├── Strategy.on_bar(bar)
    │       │
    │       └── Signal(side, strength) or None
    │
    ├── if Signal:
    │       │
    │       └── ExecutionSimulator.fill(order)
    │               │
    │               └── Fill(price, size, commission)
    │
    ├── Portfolio.update(fill)
    │       │
    │       ├── Update positions
    │       ├── Calculate equity
    │       └── Track drawdown
    │
    └── PerformanceMetrics.update()
            │
            └── Rolling Sharpe, P&L, exposure
```

### Air-Gap Validation Checks

```
AirGapValidator:
  ├── Determinism: same input → same output (hash comparison)
  ├── NaN Guard: no NaN/Inf in signals or prices
  ├── Staleness: no stale data (timestamp freshness)
  ├── Range: all values within expected bounds
  └── Consistency: portfolio balance == cash + positions_value
```

---

## 10. Workflow 9: Bridge Artifact Generation

**Component**: `TOMLGenerator` (`bridge/research_live_bridge.py` — 480 lines)

### Step-by-Step

| Step | Action | Component | Output |
|------|--------|-----------|--------|
| 9.1 | Collect research results | All Pipeline 1 outputs | Aggregated data |
| 9.2 | Extract regime parameters | Research analysis | kappa_threshold, entropy_threshold |
| 9.3 | Extract execution parameters | Backtest optimization | max_position_size, stop/take profit |
| 9.4 | Extract risk parameters | Risk analysis | max_drawdown, daily_loss_limit |
| 9.5 | Generate TOML | `TOMLGenerator.generate()` | `strategies_config.toml` |
| 9.6 | Version artifact | `TOMLVersionManager.version()` | Versioned file |
| 9.7 | Store in artifact store | `ResearchArtifactStore` | Persisted artifact |
| 9.8 | Validate artifact | Schema validation | Verified TOML |

### Bridge Artifact Generation

```
Research Results ──► TOMLGenerator.generate()
                          │
                          ├── [regime_params]
                          │     kappa_threshold = 0.5
                          │     entropy_threshold = 2.0
                          │
                          ├── [execution_params]
                          │     max_position_size = 0.1
                          │     stop_loss_pct = 0.02
                          │
                          └── [risk_params]
                                max_drawdown_pct = 0.05
                                daily_loss_limit = 500.0
                          │
                          ▼
              TOMLVersionManager.version()   ──► strategies_config.toml
                          │
                          ▼
              ResearchArtifactStore.persist() ──► Artifact DB
```

---

## 11. Workflow 10: Live System Boot

**Entry**: `launch_giga_system.py` (264 lines)

### Step-by-Step

| Step | Action | Component | Verification |
|------|--------|-----------|-------------|
| 10.1 | Load TOML bridge config | `ConfigManager` | File exists + valid schema |
| 10.2 | Create LiveAccount | `LiveAccount($10K)` | Balance initialized |
| 10.3 | Create SessionController | `SessionController(10h)` | Timer started |
| 10.4 | Create SessionGuard | `SessionGuard` | Kill switch armed |
| 10.5 | Create BinanceExecutor | `BinanceExecutor(paper=True)` | Paper mode forced |
| 10.6 | Create MarketStream | `MarketStream("BTCUSDT")` | Binance connection OK |
| 10.7 | Create LiveMomentumStrategy | `LiveMomentumStrategy(config)` | Strategy loaded |
| 10.8 | Create VariationalAnalyzer | `VariationalAnalyzer()` | Domain 2 ready |
| 10.9 | Create StateMachineBrain | `StateMachineBrain()` | State = BOOT → IDLE |
| 10.10 | Initialize rolling buffer | 50-price window | Empty, filling |
| 10.11 | Enter main loop | `while True: on_tick()` | Loop started |

### Boot Sequence

```
TOML Load ──► LiveAccount($10K) ──► SessionController(10h)
                                         │
                                         ▼
  BinanceExecutor(paper=True) ◄── SessionGuard(armed)
         │
         ▼
  MarketStream("BTCUSDT") ──► Connection verified
         │
         ▼
  LiveMomentumStrategy(config) + VariationalAnalyzer()
         │
         ▼
  StateMachineBrain: BOOT → IDLE
         │
         ▼
  while True:
      on_tick()  ──► [WF11]
```

---

## 12. Workflow 11: on_tick() Hot Path

**The core live trading loop**. Runs every tick (1-5 seconds).

### Step-by-Step

| Step | Action | Component | Gate |
|------|--------|-----------|------|
| 11.1 | Session guard check | `SessionGuard.check()` | FAIL → HALT |
| 11.2 | Session time check | `SessionController.heartbeat()` | >10h → STOP |
| 11.3 | Fetch market data | `MarketStream.get_ticker()` | Rate-limited |
| 11.4 | Update rolling buffer | `prices.append(price)` | Need 50 prices |
| 11.5 | Calculate Greeks | `VariationalAnalyzer.calculate_delta(prices)` | NaN/Inf guard |
| 11.6 | Load regime params | From TOML bridge config | Static params |
| 11.7 | Update strategy | `LiveMomentumStrategy.update(price, greeks)` | Signal or None |
| 11.8 | Validate against kappa | Check `kappa > threshold` | FAIL → skip trade |
| 11.9 | Calculate position size | From TOML `max_position_size` | Size check |
| 11.10 | Construct order | `ExecutionInstruction(side, size, price)` | Valid instruction |
| 11.11 | Execute order | `BinanceExecutor.execute_order()` | Paper or live fill |
| 11.12 | Update LiveAccount | `LiveAccount.update_pnl()` | PnL tracked |
| 11.13 | Update state machine | `StateMachineBrain.transition()` | FLAT↔LONG↔SHORT |
| 11.14 | Check cooldown | Cooldown window | In cooldown → skip |
| 11.15 | Log to Observer | `Observer.log_event()` | Non-blocking async |
| 11.16 | Sleep interval | `time.sleep(interval)` | Next tick |

### on_tick() Decision Tree

```
on_tick() START
    │
    ├── SessionGuard.check() ──► FAIL? ──► HALT system
    │
    ├── SessionController.heartbeat() ──► >10h? ──► STOP
    │
    ├── MarketStream.get_ticker() ──► price
    │
    ├── prices.append(price) ──► len(prices) < 50? ──► SKIP (filling buffer)
    │
    ├── VariationalAnalyzer.calculate_delta(prices[-50:])
    │       │
    │       └── NaN/Inf guard ──► invalid? ──► SKIP
    │
    ├── Load regime_params from TOML
    │       │
    │       ├── kappa_threshold
    │       └── entropy_threshold
    │
    ├── LiveMomentumStrategy.update(price, delta) ──► signal
    │       │
    │       └── None? ──► SKIP (no trade)
    │
    ├── Check kappa > threshold ──► FAIL? ──► SKIP
    │
    ├── PositionSizing (max_position_size from TOML)
    │
    ├── StateMachineBrain:
    │       │
    │       ├── FLAT + BUY signal ──► ENTRY(LONG)
    │       ├── FLAT + SELL signal ──► ENTRY(SHORT)
    │       ├── LONG + EXIT signal ──► EXIT
    │       ├── SHORT + EXIT signal ──► EXIT
    │       ├── IN_COOLDOWN ──► SKIP
    │       └── HALTED ──► STOP
    │
    ├── BinanceExecutor.execute_order(instruction)
    │       │
    │       └── Paper mode: simulated fill
    │
    ├── LiveAccount.update_pnl(fill)
    │       │
    │       └── Weighted-average cost, unrealized/realized PnL
    │
    └── Observer.log_event(state)
         │
         └── Async queue → events.log + state.json
```

---

## 13. Workflow 12: Order Execution Lifecycle

### Step-by-Step

| Step | Action | Component | Output |
|------|--------|-----------|--------|
| 12.1 | Signal received | Strategy | `Signal(side, strength, confidence)` |
| 12.2 | Route to exchange | `OrderRouter` | Venue selection |
| 12.3 | Smart routing | `SmartOrderRouter` | Optimal venue + slippage model |
| 12.4 | Order slicing | `SlicingEngine` | TWAP/VWAP child orders |
| 12.5 | Exposure check | `ExposureGovernor` | Within limits? |
| 12.6 | Create order | `OrderManager` | `Order(type, side, size, price)` |
| 12.7 | Execute | `BinanceExecutor.execute_order()` | Fill result |
| 12.8 | Record latency | `LatencyMonitor.record()` | Microsecond measurement |
| 12.9 | Update portfolio | `LiveAccount.add_trade()` | Position updated |
| 12.10 | Log event | `Observer.log_event()` | Trade logged |

### Order Lifecycle State Machine

```
CREATED ──► SUBMITTED ──► PARTIAL_FILL ──► FILLED
    │           │              │              │
    │           ▼              ▼              ▼
    │       REJECTED      CANCELLED      COMPLETED
    │           │
    └───────────┘
         ▼
      EXPIRED
```

### Execution Modes

```
Paper Mode (default):
  BinanceExecutor(paper=True)
  ├── Simulated fill at market price
  ├── Configurable slippage
  └── No real API calls

Live Mode (explicit opt-in):
  BinanceExecutor(paper=False)
  ├── Real ccxt API calls to Binance
  ├── Actual order placement
  └── Real fill confirmation
```

---

## 14. Workflow 13: State Machine Transitions

**Component**: `StateMachineBrain` (`brain/state_machine.py` — 257 lines)

### Transition Table

| From State | Event | To State | Action |
|------------|-------|----------|--------|
| BOOT | init_complete | IDLE | System ready |
| IDLE | signal_received | ANALYZING | Begin analysis |
| ANALYZING | entry_signal | ENTRY | Place entry order |
| ANALYZING | no_signal | IDLE | Return to idle |
| ENTRY | order_filled | IN_POSITION | Track position |
| ENTRY | order_rejected | IDLE | Log rejection |
| IN_POSITION | exit_signal | EXIT | Place exit order |
| IN_POSITION | stop_loss_hit | EXIT | Emergency exit |
| EXIT | order_filled | COOLDOWN | Start cooldown |
| COOLDOWN | cooldown_expired | IDLE | Ready for next trade |
| ANY | emergency | HALTED | System shutdown |
| ANY | session_expired | HALTED | 10h limit reached |

### Position State (Phase 14)

```
Position State (within on_tick):
  FLAT ──► BUY signal ──► LONG (set stop_loss, take_profit)
  FLAT ──► SELL signal ──► SHORT (set stop_loss, take_profit)
  LONG ──► EXIT signal ──► FLAT (cooldown_window starts)
  SHORT ──► EXIT signal ──► FLAT (cooldown_window starts)
  ANY ──► IN_COOLDOWN ──► wait(cooldown_seconds) ──► FLAT
```

---

## 15. Workflow 14: Risk Management Cascade

### Multi-Level Risk Hierarchy

```
Level 1: SESSION GUARD (GLOBAL KILL SWITCH)
         SessionGuard.check()
              │
              ├── Drawdown > max_drawdown_pct? ──► HALT
              ├── Daily loss > daily_loss_limit? ──► HALT
              ├── Order rate > max_orders/min? ──► THROTTLE
              ├── Emergency flag set? ──► HALT
              └── Session > 10h? ──► HALT
              │
              ▼ (PASS)
Level 2: STRATEGY BREAKER (PER-STRATEGY CIRCUIT BREAKER)
         StrategyBreaker.check()
              │
              ├── Consecutive losses > N? ──► TRIP (cooldown)
              ├── Daily loss > strategy_cap? ──► TRIP
              └── Strategy-specific limits? ──► TRIP
              │
              ▼ (PASS)
Level 3: EXPOSURE GOVERNOR (POSITION-LEVEL)
         ExposureGovernor.check()
              │
              ├── Position size > max_position? ──► REJECT
              ├── Leverage > max_leverage? ──► REJECT
              ├── Margin insufficient? ──► REJECT
              └── Near liquidation? ──► REDUCE
              │
              ▼ (PASS)
Level 4: AIR-GAP VALIDATOR (DATA-LEVEL)
         AirGapValidator.validate()
              │
              ├── NaN/Inf in data? ──► REJECT
              ├── Stale data? ──► REJECT
              ├── Out of range? ──► REJECT
              └── Non-deterministic? ──► REJECT
              │
              ▼ (PASS)
         ORDER PROCEEDS TO EXECUTION
```

### Risk Decision Steps

| Step | Check | Action on Fail | Recovery |
|------|-------|----------------|----------|
| 14.1 | `SessionGuard.check()` | HALT entire system | Manual restart required |
| 14.2 | `SessionController.is_valid()` | STOP session | New session next day |
| 14.3 | `StrategyBreaker.check()` | TRIP strategy | Auto-reset after cooldown |
| 14.4 | `ExposureGovernor.check()` | REJECT order | Reduce and retry |
| 14.5 | NaN/Inf guard | SKIP tick | Continue on next tick |
| 14.6 | Kappa threshold | SKIP trade | Wait for next signal |
| 14.7 | Position cooldown | SKIP entry | Wait for cooldown expiry |

---

## 16. Workflow 15: Feedback & Adaptation Loop

### Step-by-Step

| Step | Action | Component | Effect |
|------|--------|-----------|--------|
| 15.1 | Receive trade result | PnL from LiveAccount | Realized P&L |
| 15.2 | Classify result | Win/loss | Binary classification |
| 15.3 | Asymmetric adaptation | `AdaptiveEngine` | Cut 5% on loss, grow 3% on gain |
| 15.4 | Capital regime update | `CapitalRegimeEngine` | Regime-based capital allocation |
| 15.5 | Position size adjustment | `PositionSizer` | Adaptive sizing |
| 15.6 | AI feedback loop | `AIOptimizer.feedback_loop()` | Reward → parameter adjustment |
| 15.7 | Parameter optimization | `AdaptiveParameterOptimizer` | Online learning update |
| 15.8 | Apply updated params | Next `on_tick()` cycle | Parameters active |

### Adaptation Flow

```
Trade PnL ──► AdaptiveEngine
                  │
                  ├── Win: risk_multiplier *= 1.03  (grow 3%)
                  │
                  └── Loss: risk_multiplier *= 0.95  (cut 5%)
                  │
                  ▼
          CapitalRegimeEngine
                  │
                  ├── AGGRESSIVE: 80% capital deployed
                  ├── NORMAL: 50% capital deployed
                  └── DEFENSIVE: 20% capital deployed
                  │
                  ▼
          AIOptimizer.feedback_loop()
                  │
                  ├── Signal quality ──► Adjust signal weights
                  ├── Fill quality ──► Adjust execution params
                  └── PnL trajectory ──► Adjust risk params
                  │
                  ▼
          Updated Parameters ──► Next on_tick() uses new values
```

---

## 17. Workflow 16: Monitoring & Alerting

### Step-by-Step

| Step | Action | Component | Output |
|------|--------|-----------|--------|
| 16.1 | Collect metrics | `MetricsCollector` | Prometheus-style metrics |
| 16.2 | Check alert rules | `AlertManager` | Alert if threshold breached |
| 16.3 | Send alerts | `alerting.py` | Telegram + Discord (non-blocking) |
| 16.4 | Observer logging | `Observer` | state.json + events.log |
| 16.5 | Performance profiling | `PerformanceProfiler` | Nanosecond timing |
| 16.6 | System health | `SystemMonitor` | CPU/memory/latency |

### Monitoring Flow

```
on_tick() ──► Observer (async queue, 10K buffer)
    │              │
    │              ├── state.json (current trading state)
    │              │     ├── position_state: FLAT/LONG/SHORT
    │              │     ├── current_pnl: $XXX
    │              │     ├── last_signal: BUY/SELL/HOLD
    │              │     └── uptime: HH:MM:SS
    │              │
    │              └── events.log (time-series event log)
    │                    ├── [TRADE] BUY 0.001 BTC @ 45000
    │                    ├── [SIGNAL] domain_2_delta=0.023
    │                    └── [RISK] session_guard_check=PASS
    │
    ├── PerformanceProfiler
    │       └── on_tick latency: 2.3ms (p99: 5.1ms)
    │
    ├── SystemMonitor
    │       ├── CPU: 23%
    │       ├── Memory: 512MB / 2GB
    │       └── DuckDB connections: 3/10
    │
    └── AlertManager
            ├── Drawdown > 3%? ──► Telegram alert
            ├── Latency > 100ms? ──► Discord alert
            └── Error rate > 5%? ──► Both channels
```

---

## 18. Workflow 17: R Analytics Integration

### Step-by-Step

| Step | Action | Component | Output |
|------|--------|-----------|--------|
| 17.1 | Initialize R session | `RBridge` → `RSession` | R process active |
| 17.2 | Convert data | `DataConverter` (Polars → R) | R data.frame |
| 17.3 | Run GARCH model | `GARCHModel` (via rugarch) | Vol forecast |
| 17.4 | Run ARIMA model | `ARIMAModel` | Time-series forecast |
| 17.5 | Run cointegration | `CointegrationTest` (Johansen) | Cointegration results |
| 17.6 | Run econometrics | `econometrics.R` | ADF, PP, Granger tests |
| 17.7 | Convert results back | `DataConverter` (R → Polars) | Python-native results |
| 17.8 | Clean up R session | `RBridge.close()` | Memory freed |

### R Bridge Architecture

```
Python ──► DataConverter.to_r_dataframe(polars_df)
               │
               ▼
           RBridge.execute("source('econometrics.R')")
               │
               ├── GARCHModel.fit() ──► rugarch::ugarchfit
               ├── ARIMAModel.fit() ──► forecast::auto.arima
               └── CointegrationTest.test() ──► urca::ca.jo
               │
               ▼
           DataConverter.from_r_dataframe(r_result)
               │
               ▼
           Python ModelResults
```

**Graceful degradation**: If rpy2 is not installed, all R analytics skip with warning.

---

## 19. Workflow 18: Data Ingestion & Storage

### Step-by-Step

| Step | Action | Component | Output |
|------|--------|-----------|--------|
| 18.1 | Select data source | Config-driven | Source identifier |
| 18.2 | Fetch raw data | `MarketDataLoader` | Raw OHLCV |
| 18.3 | Rate limit API calls | `TokenBucketRateLimiter` | Throttled requests |
| 18.4 | Preprocess with Polars | `preprocessing.py` | Clean data |
| 18.5 | Validate data | `validators.py` | Validation report |
| 18.6 | Compute indicators | `indicators.py` (Numba JIT) | Technical indicators |
| 18.7 | Store in DuckDB | `TimeSeriesDatabase.insert()` | Persisted to OLAP |
| 18.8 | Multi-exchange check | `multi_exchange.py` | Price triangulation |

### Data Source Selection

```
Config ──► Source Selection:
              │
              ├── CSV/Parquet (bulk historical)
              ├── yfinance (daily OHLCV)
              ├── Alpha Vantage (intraday)
              ├── ccxt (crypto exchange multi-source)
              └── Binance REST/WS (real-time)
              │
              ▼
         MarketDataLoader.load()
              │
              ▼
         Polars preprocessing (clean, impute, validate)
              │
              ▼
         Numba JIT indicators (SMA/EMA/RSI/MACD/ATR/BB)
              │
              ▼
         DuckDB TimeSeriesDatabase
              │
              ├── ohlcv table
              ├── signals table
              ├── trades table
              └── snapshots table
```

---

## 20. Workflow 19: Strategy Decision Reduction

**Component**: `DecisionReducer` (`reducer/reducer.py` — 261 lines)

### Step-by-Step

| Step | Action | Component | Output |
|------|--------|-----------|--------|
| 19.1 | Collect strategy signals | All active strategies | Signal array |
| 19.2 | Apply weights | Configurable per strategy | Weighted signals |
| 19.3 | Aggregate votes | `DecisionReducer.reduce()` | Weighted sum |
| 19.4 | Apply threshold | Signal threshold | BUY/SELL/HOLD |
| 19.5 | Build instruction | `ExecutionInstruction` | Final action |

### Reduction Flow

```
MomentumStrategy.signal() ──── (weight: 0.4) ──┐
PairsTradingStrategy.signal()── (weight: 0.3) ──┤
MarketMakingStrategy.signal()── (weight: 0.2) ──┼──► DecisionReducer.reduce()
OptionsStrategy.signal() ────── (weight: 0.1) ──┘         │
                                                            ▼
                                                  Weighted Sum = Σ(w_i × s_i)
                                                            │
                                                    ┌───────┼───────┐
                                                    ▼       ▼       ▼
                                                  > +0.3  ±0.3    < -0.3
                                                   BUY    HOLD    SELL
                                                            │
                                                            ▼
                                              ExecutionInstruction(action, size, price)
```

---

## 21. Workflow 20: Emergency Shutdown

### Trigger Conditions

| Trigger | Source | Detection |
|---------|--------|-----------|
| Max drawdown breached | `SessionGuard` | PnL monitoring |
| Daily loss limit hit | `SessionGuard` | Cumulative daily PnL |
| Session time expired | `SessionController` | 10h hard limit |
| API connection lost | `BinanceExecutor` | Connection error |
| Critical error | Any component | Unhandled exception |
| Manual shutdown | Operator | Emergency flag |

### Shutdown Sequence

| Step | Action | Component | Effect |
|------|--------|-----------|--------|
| 20.1 | Trigger detected | Any trigger above | Shutdown initiated |
| 20.2 | State → HALTED | `StateMachineBrain` | FSM halted |
| 20.3 | Cancel all open orders | `OrderManager` | Orders cancelled |
| 20.4 | Close all positions | `BinanceExecutor` | Market close orders |
| 20.5 | Update LiveAccount | `LiveAccount` | Final PnL calculated |
| 20.6 | Persist state | `Observer` | Final state.json |
| 20.7 | Send alerts | `AlertManager` | Telegram + Discord emergency |
| 20.8 | Log shutdown | `Logger` | Shutdown reason logged |
| 20.9 | Exit process | `sys.exit()` | Clean exit |

### Emergency Shutdown Flow

```
TRIGGER ──► SessionGuard.emergency_halt()
                │
                ├── StateMachineBrain.transition(HALTED)
                │
                ├── OrderManager.cancel_all()
                │
                ├── BinanceExecutor.close_all_positions()
                │       │
                │       └── Market orders to close all
                │
                ├── LiveAccount.finalize()
                │       │
                │       └── Final PnL: $XXX
                │
                ├── Observer.log_event("EMERGENCY_SHUTDOWN")
                │
                ├── AlertManager.send_critical("System halted: {reason}")
                │       │
                │       ├── Telegram: "⚠️ GIGA-SYSTEM HALTED"
                │       └── Discord: "⚠️ GIGA-SYSTEM HALTED"
                │
                └── sys.exit(1)
```

---

## 22. Workflow 21: Visualization & Dashboards

### Step-by-Step

| Step | Action | Component | Output |
|------|--------|-----------|--------|
| 21.1 | Launch Streamlit | `visualization/app.py` | Web dashboard |
| 21.2 | Greeks dashboard | `greeks_dashboard.py` | 3D surface plots |
| 21.3 | Risk dashboard | `risk_dashboard.py` | VaR/CVaR gauges |
| 21.4 | Quantum visualizer | `quantum_visualizer.py` | Circuit diagrams |
| 21.5 | Education mode | `education_mode.py` | Interactive tutorials |
| 21.6 | P&L attribution | `pnl_attribution.py` | Waterfall charts |
| 21.7 | Correlation heatmap | `correlation_heatmap.py` | Clustering viz |
| 21.8 | Observer dashboard | `observer_app.py` | Real-time state monitor |

### Dashboard Architecture

```
streamlit run visualization/app.py
       │
       ├── Page: Greeks ──► 3D Δ/Γ/Θ/V surfaces (Plotly)
       ├── Page: Risk ──► VaR/CVaR gauges, MC stress test
       ├── Page: Quantum ──► Circuit diagrams, Bloch sphere
       ├── Page: Education ──► Interactive BS/Greeks tutorials
       ├── Page: P&L ──► Waterfall attribution, factor decomposition
       ├── Page: Correlation ──► Hierarchical clustering, network graph
       └── Page: Observer ──► state.json + events.log monitor
```

---

## 23. Workflow 22: CI/CD & Deployment

### CI/CD Pipeline (`scripts/ci_cd_pipeline.py` — 253 lines)

| Step | Tool | Action |
|------|------|--------|
| 22.1 | flake8/black | Lint Python code |
| 22.2 | mypy | Type checking |
| 22.3 | pytest | Unit tests (test_greeks, test_risk, test_routing, test_account, test_utils) |
| 22.4 | Integration | Integration tests |
| 22.5 | health_check.py | Full module import verification |
| 22.6 | Docker | `docker-compose build` |
| 22.7 | Deploy | Container deployment |

### CI/CD Flow

```
git push ──► .github/workflows/ci.yml
                   │
                   ├── Step 1: Lint (flake8, black)
                   ├── Step 2: Type Check (mypy)
                   ├── Step 3: Unit Tests (pytest)
                   ├── Step 4: Integration Tests
                   ├── Step 5: Health Check (all imports)
                   ├── Step 6: Docker Build
                   └── Step 7: Deploy (if main branch)
```

---

## 24. Workflow 23: Full 2-Pipeline End-to-End

**Entry**: `run_system_pipeline.py --mode full`

### Step-by-Step

| Step | Phase | Action | Component |
|------|-------|--------|-----------|
| 23.1 | Research | Load historical data | `DataBridge` |
| 23.2 | Research | Run 5-domain analysis | `AlphaSignalEngine` |
| 23.3 | Research | Run ML regime detection | `RegimeDetector` |
| 23.4 | Research | Run backtesting | `BacktestEngine` |
| 23.5 | Research | Optimize parameters | Walk-forward |
| 23.6 | Bridge | Generate TOML artifact | `TOMLGenerator` |
| 23.7 | Bridge | Version artifact | `TOMLVersionManager` |
| 23.8 | Bridge | Validate artifact | Schema check |
| 23.9 | Live | Verify bridge exists | File check |
| 23.10 | Live | Load bridge config | `TOMLParameterReader` |
| 23.11 | Live | Boot live system | `launch_giga_system.py` |
| 23.12 | Live | Enter on_tick() loop | Main trading loop |

### End-to-End Flow

```
run_system_pipeline.py --mode full
         │
         ├── PIPELINE 1: run_pipeline_1_research()
         │       │
         │       ├── Greek research lab
         │       ├── 5-Domain analysis
         │       ├── ML regime detection
         │       ├── Backtesting
         │       ├── Parameter optimization
         │       └── Generate strategies_config.toml
         │
         ├── BRIDGE VALIDATION
         │       │
         │       ├── File exists?
         │       ├── Schema valid?
         │       └── Parameters in range?
         │
         └── PIPELINE 2: run_pipeline_2_live()
                 │
                 ├── Load bridge config
                 ├── Boot live system
                 └── Enter on_tick() loop
                         │
                         └── [Runs until session expires or emergency halt]
```

---

## 25. Cross-Workflow Dependencies

| Workflow | Depends On | Produces For |
|----------|-----------|-------------|
| WF1 (Init) | — | All workflows |
| WF2 (Research) | WF1, WF3, WF4, WF5, WF6, WF7, WF8, WF18 | WF9 |
| WF3 (5-Domain) | WF18 (data) | WF4, WF5 |
| WF4 (Alpha Signal) | WF3 | WF5, WF2 |
| WF5 (Factor Research) | WF3, WF4 | WF8, WF9 |
| WF6 (ML Pipeline) | WF18 (data) | WF2, WF4 |
| WF7 (Quantum) | WF2 results | WF9 |
| WF8 (Backtesting) | WF2, WF5, WF11 (strategy) | WF9 |
| WF9 (Bridge Gen) | WF2 | WF10 |
| WF10 (Live Boot) | WF9 (TOML artifact) | WF11 |
| WF11 (on_tick) | WF10, WF14, WF12, WF13 | WF15, WF16 |
| WF12 (Order Exec) | WF11 | WF15 |
| WF13 (State Machine) | WF11 | WF12 |
| WF14 (Risk Cascade) | WF11 | WF20 (if triggered) |
| WF15 (Feedback) | WF12 results | WF11 (next cycle) |
| WF16 (Monitoring) | WF11 | WF20 (if alert) |
| WF17 (R Analytics) | WF18 (data) | WF2 |
| WF18 (Data Ingestion) | WF1 | WF2, WF3, WF6, WF10 |
| WF19 (Decision Reduce) | Multiple strategies | WF11 |
| WF20 (Emergency) | WF14, WF16 | System halt |
| WF21 (Visualization) | WF2, WF11 | User dashboards |
| WF22 (CI/CD) | Git push | Deployment |
| WF23 (End-to-End) | All workflows | Complete run |

---

## 26. Workflow Verification Matrix

| Workflow | Key Components | Verification Method | Expected Result |
|----------|---------------|---------------------|-----------------|
| WF1 | ConfigManager, Logger, Database | Config loads, DB initializes | No errors |
| WF2 | AlphaSignalEngine, DataBridge | Research pipeline completes | TOML generated |
| WF3 | 5 Domain analyzers | All 5 domains produce results | 5 domain outputs |
| WF4 | AlphaSignalEngine | generate_signal() returns AlphaSignal | Valid signal |
| WF5 | AlphaFactorLibrary, WalkForward | Factors compute, WF validates | Reports |
| WF6 | RegimeDetector, VolForecaster | Models train, predict | Labels + forecasts |
| WF7 | QuantumOptimizer | Optimize or fallback | Weights |
| WF8 | BacktestEngine, AirGapValidator | Backtest completes, validation passes | Metrics |
| WF9 | TOMLGenerator | TOML file created with valid schema | File exists |
| WF10 | LiveAccount, BinanceExecutor | All components initialized | System booted |
| WF11 | on_tick() hot path | Tick processed within latency target | Trade or skip |
| WF12 | OrderManager, BinanceExecutor | Order lifecycle completes | Fill or reject |
| WF13 | StateMachineBrain | State transitions valid | No invalid states |
| WF14 | SessionGuard, StrategyBreaker | Risk checks pass or halt | Protected |
| WF15 | AdaptiveEngine, AIOptimizer | Parameters adjusted | Updated params |
| WF16 | Observer, SystemMonitor | Events logged, metrics collected | Logs + metrics |
| WF17 | RBridge, ModelWrapper | R models execute or skip | Results or skip |
| WF18 | MarketDataLoader, DuckDB | Data loaded, stored, validated | Clean data |
| WF19 | DecisionReducer | Weighted vote produces verdict | BUY/SELL/HOLD |
| WF20 | SessionGuard emergency | System halts gracefully | Clean shutdown |
| WF21 | Streamlit dashboards | Pages render | Visual output |
| WF22 | CI/CD pipeline | All checks pass | Green build |
| WF23 | Full pipeline | Research → Bridge → Live | End-to-end run |

---

## Document Signature

- **System**: GIGA-SYSTEM
- **Total Workflows**: 23
- **Architecture**: 2-Pipeline Air-Gap (Research → TOML Bridge → Live)
- **Core Hypothesis**: High κ + Low Entropy = Maximal Alpha
- **Hot Path**: on_tick() — 16 steps per tick
- **Risk Levels**: 4-level cascade (Session → Strategy → Exposure → Data)
- **Generated**: Workflow Document v1.0
