# GIGA-SYSTEM — PIPELINE DOCUMENT

## Document Purpose

This document defines the **complete data and execution pipeline** for the GIGA-SYSTEM quantitative crypto/options trading platform. It maps every component, class, module, data flow, and integration point across the system's **2-Pipeline Air-Gap Architecture** — from raw market data ingestion through research-driven alpha generation, bridge artifact serialization, live execution, risk management, and continuous adaptation.

**System Identity**: GIGA-SYSTEM is a production-grade algorithmic trading system built on a **5-Domain Greek Alpha Framework** with the core hypothesis: **"High κ (mean-reversion speed) regimes with Low Entropy offer maximal Alpha."**

---

## Table of Contents

1. [Architecture Overview — 2-Pipeline Air-Gap](#1-architecture-overview--2-pipeline-air-gap)
2. [Pipeline 1: RESEARCH (Offline Truth)](#2-pipeline-1-research-offline-truth)
3. [The Bridge: TOML Artifact Boundary](#3-the-bridge-toml-artifact-boundary)
4. [Pipeline 2: LIVE (Reality Execution)](#4-pipeline-2-live-reality-execution)
5. [5-Domain Greek Alpha Framework](#5-5-domain-greek-alpha-framework)
6. [Data Ingestion Pipeline](#6-data-ingestion-pipeline)
7. [Research Core Pipeline](#7-research-core-pipeline)
8. [Quantitative Pricing Pipeline](#8-quantitative-pricing-pipeline)
9. [Machine Learning Pipeline](#9-machine-learning-pipeline)
10. [Quantum Computing Pipeline](#10-quantum-computing-pipeline)
11. [Strategy Pipeline](#11-strategy-pipeline)
12. [R Analytics Pipeline](#12-r-analytics-pipeline)
13. [Backtesting Pipeline](#13-backtesting-pipeline)
14. [Execution Pipeline](#14-execution-pipeline)
15. [Risk Management Pipeline](#15-risk-management-pipeline)
16. [State Machine Pipeline](#16-state-machine-pipeline)
17. [Monitoring & Observability Pipeline](#17-monitoring--observability-pipeline)
18. [Feedback & Adaptation Pipeline](#18-feedback--adaptation-pipeline)
19. [Visualization Pipeline](#19-visualization-pipeline)
20. [Configuration Pipeline](#20-configuration-pipeline)
21. [CI/CD & Testing Pipeline](#21-cicd--testing-pipeline)
22. [Complete Module Registry](#22-complete-module-registry)
23. [Complete Class Registry](#23-complete-class-registry)
24. [Performance Engineering](#24-performance-engineering)
25. [Design Patterns](#25-design-patterns)

---

## 1. Architecture Overview — 2-Pipeline Air-Gap

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        GIGA-SYSTEM ARCHITECTURE                              │
│                                                                              │
│  ┌─────────────────────────────┐    AIR-GAP    ┌──────────────────────────┐  │
│  │   PIPELINE 1: RESEARCH      │    (TOML)     │   PIPELINE 2: LIVE       │  │
│  │   (Offline Truth)           │ ═══════════>  │   (Reality Execution)    │  │
│  │                             │  Frozen       │                          │  │
│  │  • 5-Domain Alpha           │  Artifacts    │  • MarketStream          │  │
│  │  • ML Regime Detection      │               │  • BinanceExecutor       │  │
│  │  • Quantum Optimization     │  strategies_  │  • LiveMomentumStrategy  │  │
│  │  • Walk-Forward Validation  │  config.toml  │  • VariationalAnalyzer   │  │
│  │  • Backtesting Engine       │               │  • SessionGuard          │  │
│  │  • R Statistical Models     │               │  • TradingStateMachine   │  │
│  │  • Factor Library           │               │  • LiveAccount           │  │
│  └─────────────────────────────┘               └──────────────────────────┘  │
│                                                                              │
│  Entry Points:                                                               │
│  • Research: demo_complete_system.py, run_greek_research_lab.py              │
│  • Full:     run_system_pipeline.py --mode full                              │
│  • Live:     launch_giga_system.py (THE authorized live runner)              │
│  • Safe:     QUICK_START.py (research only, no live)                         │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Air-Gap Principle
Research code **never** calls execution code directly. The only data that crosses from Pipeline 1 to Pipeline 2 is a **frozen TOML configuration file** (`config/strategies_config.toml`) containing regime parameters, execution thresholds, and risk limits. This ensures:
- No look-ahead bias leakage
- Deterministic reproducibility
- Clear separation of truth-seeking (research) from money-at-risk (live)

### Entry Points

| File | Lines | Purpose | Pipeline |
|------|-------|---------|----------|
| `launch_giga_system.py` | 264 | THE live execution runner (Phase 13) | Pipeline 2 |
| `run_system_pipeline.py` | 125 | 2-Pipeline connector (`--mode research\|live\|full`) | Both |
| `demo_complete_system.py` | 168 | Research/demo mode, generates bridge artifacts | Pipeline 1 |
| `run_greek_research_lab.py` | 202 | Phase 12 Greek Alpha demonstration | Pipeline 1 |
| `QUICK_START.py` | 120 | Safe research pipeline entry | Pipeline 1 |

---

## 2. Pipeline 1: RESEARCH (Offline Truth)

Pipeline 1 runs entirely offline. It ingests historical data, applies the 5-Domain Greek Alpha Framework, trains ML models, runs quantum optimizations, backtests strategies, and produces **frozen TOML artifacts** as output.

```
Historical Data ──► 5-Domain Analysis ──► Alpha Signal Engine ──► Factor Library
       │                                         │                      │
       ▼                                         ▼                      ▼
  ML Regime Detection              Walk-Forward Validation      Backtest Engine
       │                                         │                      │
       ▼                                         ▼                      ▼
  Quantum Optimization ──► Decision Reducer ──► TOML Bridge Artifacts
```

### Research Components

| Package | Files | Key Classes | Purpose |
|---------|-------|-------------|---------|
| `research/core/` | 21 | `AlphaSignalEngine`, `VariationalAnalyzer`, `StateSpaceOmega`, `StochasticModeler`, `TimeAsymmetryAnalyzer`, `InformationGeometer` | 5-Domain Greek Alpha Framework |
| `research/ml/` | 3 | `FeatureEngine`, `RegimeDetector`, `VolatilityForecaster` | ML regime detection & vol forecasting |
| `research/quantum/` | 6 | `QuantumOptimizer`, `QuantumMonteCarlo`, `QuantumML`, `QuantumPortfolio` | Quantum portfolio optimization |
| `research/strategies/` | 6 | `Strategy(ABC)`, `MomentumStrategy`, `PairsTradingStrategy`, `MarketMakingStrategy`, `OptionsStrategy` | Strategy definitions |
| `research/r_analytics/` | 7 | R scripts: GARCH, ARIMA, VAR | Statistical econometrics via R |
| `backtesting/` | 9 | `BacktestEngine`, `WalkForwardOptimizer`, `AirGapValidator` | Event-driven backtesting |

### Research Pipeline Flow

1. **Data Ingestion** → `DataBridge` loads CSV/Parquet/DuckDB/API data
2. **Preprocessing** → `preprocessing.py` (Polars) cleans, imputes, validates OHLCV
3. **Feature Engineering** → `FeatureEngine` generates price/vol/technical/volume/regime features
4. **5-Domain Analysis** → `AlphaSignalEngine.generate_signal()` orchestrates all 5 domains
5. **Regime Detection** → `RegimeDetector` (GMM/HMM) classifies market state
6. **Volatility Forecasting** → `VolatilityForecaster` (EWMA/GARCH/HAR)
7. **Factor Research** → `AlphaFactorLibrary` computes Kappa/Entropy/Gamma/Ergodicity/Momentum/VolRegime factors
8. **Walk-Forward Validation** → `GreekWalkForwardValidator.run_validation()` validates alpha hypothesis
9. **Backtesting** → `BacktestEngine` runs event-driven strategy backtests
10. **Quantum Optimization** → `QuantumOptimizer` (QAOA/VQE) optimizes portfolio weights
11. **Artifact Generation** → `TOMLGenerator` freezes parameters into `strategies_config.toml`

---

## 3. The Bridge: TOML Artifact Boundary

The bridge is the critical **air-gap** between research and live. It consists of:

### Bridge Components

| File | Lines | Key Classes | Purpose |
|------|-------|-------------|---------|
| `bridge/research_live_bridge.py` | 480 | `TOMLGenerator`, `TOMLVersionManager`, `TOMLParameterReader`, `ResearchArtifactStore`, `ResearchLiveComparator` | TOML generation, versioning, comparison |
| `bridge/data_bridge.py` | 706 | `DataBridge`, `StreamingDataSource`, `SimulatedTickStream` | Unified data interface |
| `bridge/r_bridge.py` | 682 | `RBridge`, `RSession` | R↔Python session management |
| `bridge/rpy2_interface.py` | 335 | `RInterface` | Low-level R↔Python interface |
| `bridge/model_wrapper.py` | 635 | `GARCHModel`, `ARIMAModel`, `CointegrationTest` | R model wrappers |
| `bridge/data_converter.py` | 368 | `DataConverter` | Polars/NumPy/Pandas ↔ R data.frame |

### TOML Artifact Structure
```toml
[regime_params]
kappa_threshold = 0.5
entropy_threshold = 2.0
volatility_regime = "NORMAL"

[execution_params]
max_position_size = 0.1
stop_loss_pct = 0.02
take_profit_pct = 0.04
cooldown_seconds = 300

[risk_params]
max_drawdown_pct = 0.05
daily_loss_limit = 500.0
max_leverage = 3.0
```

### Bridge Flow
```
Research Output ──► TOMLGenerator.generate() ──► strategies_config.toml
                                                        │
                    TOMLVersionManager.version()         │
                                                        ▼
                    TOMLParameterReader.load() ◄── Live System Reads
```

---

## 4. Pipeline 2: LIVE (Reality Execution)

Pipeline 2 is the live trading loop. It reads frozen TOML artifacts, connects to Binance, and executes the `on_tick()` hot path.

### Live Architecture (`launch_giga_system.py`)

```
MarketStream("BTCUSDT")  ──►  on_tick() Hot Path:
       │                           │
       ▼                           ├── 1. SessionGuard.check()        (kill switch)
   LiveAccount($10K)               ├── 2. Regime Selection            (from TOML config)
       │                           ├── 3. VariationalAnalyzer.delta() (Greek calculation)
       ▼                           ├── 4. LiveMomentumStrategy.update()(strategy logic)
   SessionController               ├── 5. Risk Validation             (kappa threshold)
   (10h hard limit)                ├── 6. Position Sizing              (from TOML)
       │                           ├── 7. Order Construction           (BUY/SELL/HOLD)
       ▼                           ├── 8. BinanceExecutor.execute()    (paper/live)
   TradingStateMachine             └── 9. State Machine Update         (FLAT/LONG/SHORT)
   (FSM: BOOT→...→HALTED)
```

### Live Components

| File | Lines | Key Classes | Role |
|------|-------|-------------|------|
| `launch_giga_system.py` | 264 | — | Main loop, `on_tick()` orchestrator |
| `data/live/market_stream.py` | 138 | `MarketStream`, `TokenBucketRateLimiter` | Binance REST ticker feed |
| `data/live/binance_ws_feed.py` | 159 | `BinanceWebSocketFeed` | Binance WebSocket aggTrade |
| `execution/binance_executor.py` | 395 | `BinanceExecutor` | Paper/live order execution via ccxt |
| `account/live_account.py` | 290 | `LiveAccount` | Real PnL, margin, leverage, liquidation |
| `session/session_controller.py` | 38 | `SessionController` | 10h session hard limit, heartbeat |
| `risk/session_guard.py` | 207 | `SessionGuard` | Global drawdown kill switch |
| `brain/state_machine.py` | 257 | `TradingStateMachine` (`StateMachineBrain`) | FSM: BOOT→IDLE→ANALYZING→ENTRY→IN_POSITION→EXIT→COOLDOWN→HALTED |
| `research/strategies/momentum.py` | 955 | `LiveMomentumStrategy` | Live strategy implementation |
| `research/core/greek_response.py` | 221 | `VariationalAnalyzer` | Real-time Greek calculation |

### on_tick() Hot Path Detail
1. **Session Guard** → `SessionGuard.check()` — drawdown limit, rate limiting, emergency flag
2. **Data Fetch** → `MarketStream.get_ticker()` — Binance REST price with token-bucket rate limiter
3. **Rolling Window** → 50-price rolling buffer for Greek calculation
4. **Greek Calculation** → `VariationalAnalyzer.calculate_delta()` on rolling window (NaN/Inf guard)
5. **Regime Selection** → Load `regime_params` from TOML (kappa_threshold, entropy_threshold)
6. **Strategy Update** → `LiveMomentumStrategy.update(price, greeks)` → signal generation
7. **Risk Validation** → Check kappa threshold, position sizing from TOML, exposure limits
8. **Order Construction** → `ExecutionInstruction(BUY/SELL/HOLD, size, price)`
9. **Execution** → `BinanceExecutor.execute_order()` — paper mode forced by default
10. **State Transition** → `StateMachineBrain` transitions: FLAT↔LONG↔SHORT with cooldown
11. **PnL Update** → `LiveAccount.update_pnl()` — weighted-average cost, realized/unrealized

---

## 5. 5-Domain Greek Alpha Framework

The core intellectual engine of GIGA-SYSTEM. Maps five mathematical domains to generate alpha signals.

### Domain Architecture

```
Domain 1 (Ω,Λ): Market State Space     ──► vol/trend/liquidity regimes
Domain 2 (Δ,Γ,Θ): Variational Sensitivity ──► delta/gamma/theta response
Domain 3 (μ,σ,κ): Stochastic Models    ──► O-U mean-reversion speed (κ)
Domain 4 (τ,ε): Time Asymmetry         ──► ergodicity/relaxation-time
Domain 5 (Η,Φ): Information Geometry   ──► Shannon entropy, KL divergence
                           │
                           ▼
              AlphaSignalEngine.generate_signal()
                           │
                           ▼
              AlphaSignal(direction, strength, confidence, regime)
```

### Domain Implementation

| Domain | File | Class | Key Methods | Lines |
|--------|------|-------|-------------|-------|
| 1 (Ω,Λ) | `research/core/market_state_space.py` | `StateSpaceOmega` | `classify_state()`, `get_transition_matrix()` | 100 |
| 2 (Δ,Γ,Θ) | `research/core/greek_response.py` | `VariationalAnalyzer` | `calculate_delta()`, `calculate_gamma()`, `calculate_theta()`, `calculate_vega()` | 221 |
| 3 (μ,σ,κ) | `research/core/stochastic_models.py` | `StochasticModeler` | `fit_ornstein_uhlenbeck()` | 84 |
| 4 (τ,ε) | `research/core/time_asymmetry.py` | `TimeAsymmetryAnalyzer` | `check_ergodicity()`, `calculate_relaxation_time()` | 73 |
| 5 (Η,Φ) | `research/core/information_geometry.py` | `InformationGeometer` | `calculate_shannon_entropy()`, `calculate_market_entropy()`, `calculate_kl_divergence()` | 67 |

### Central Orchestrator

| File | Class | Lines | Purpose |
|------|-------|-------|---------|
| `research/core/alpha_signal_engine.py` | `AlphaSignalEngine` | 549 | Orchestrates all 5 domains → `generate_signal()` → `AlphaSignal` |
| `research/core/alpha_signal_engine.py` | `AlphaDecayTracker` | — | Half-life alpha signal decay tracking |

### Core Hypothesis
> **"High κ (mean-reversion speed) in Ornstein-Uhlenbeck regimes combined with Low Shannon Entropy (Domain 5) produces predictable, exploitable alpha — maximal when markets revert quickly in low-information-uncertainty environments."**

---

## 6. Data Ingestion Pipeline

### Components

| File | Lines | Key Classes | Purpose |
|------|-------|-------------|---------|
| `data/market_data.py` | 625 | `MarketDataLoader` | Multi-source: CSV, yfinance, Alpha Vantage, ccxt |
| `data/database.py` | 666 | `TimeSeriesDatabase` | DuckDB OLAP with thread-safe connection pool |
| `data/database_layer.py` | 213 | — | DuckDB schema: ohlcv, signals, trades, snapshots |
| `data/indicators.py` | 775 | — | Numba JIT indicators: SMA, EMA, RSI, MACD, ATR, Bollinger |
| `data/preprocessing.py` | 652 | — | Polars-based financial data cleaning, imputation |
| `data/multi_exchange.py` | 190 | — | ccxt multi-exchange price triangulation |
| `data/storage_manager.py` | 708 | — | DuckDB storage with SQL injection protection |
| `data/live/market_stream.py` | 138 | `MarketStream`, `TokenBucketRateLimiter` | Binance REST ticker (rate-limited) |
| `data/live/binance_ws_feed.py` | 159 | `BinanceWebSocketFeed` | Binance WebSocket aggTrade (SSL, auto-reconnect) |
| `live/stream/streaming.py` | 784 | `StreamingManager`, `RealTimeDataStream` | Multi-provider WebSocket (Alpaca/Polygon/IEX/Binance/Yahoo) |
| `live/stream/realtime_manager.py` | 661 | `RealtimeManager`, `RealTimeDataManager` | Unified real-time + historical data |

### Data Flow

```
External Sources                    Internal Storage
┌────────────┐                     ┌──────────────┐
│ Binance WS │──aggTrade──────────►│              │
│ Binance REST│──ticker────────────►│  DuckDB      │
│ yfinance   │──OHLCV─────────────►│  (OLAP)      │
│ Alpha Vantage│──daily────────────►│              │
│ ccxt       │──multi-exchange─────►│  Tables:     │
│ CSV/Parquet│──bulk───────────────►│  · ohlcv     │
└────────────┘                     │  · signals   │
       │                           │  · trades    │
       ▼                           │  · snapshots │
  Polars Preprocessing             └──────────────┘
  (clean, impute, validate)               │
       │                                  ▼
       ▼                           Numba JIT Indicators
  TokenBucket Rate Limiter         (SMA/EMA/RSI/MACD/ATR/BB)
```

---

## 7. Research Core Pipeline

### Alpha Factor Library (`research/core/alpha_factor_library.py` — 501 lines)

| Class | Purpose |
|-------|---------|
| `AlphaFactor` (ABC) | Abstract base with `compute()` method |
| `KappaAlpha` | Mean-reversion speed factor (Domain 3) |
| `EntropyAlpha` | Information entropy factor (Domain 5) |
| `GammaAlpha` | Options gamma exposure factor |
| `ErgodicityAlpha` | Time-average vs ensemble factor (Domain 4) |
| `MomentumAlpha` | Price momentum factor |
| `VolRegimeAlpha` | Volatility regime factor (Domain 1) |
| `HypothesisTest` | Statistical significance testing |
| `AlphaResearchPipeline` | End-to-end research pipeline |
| `AlphaCombiner` | Multi-factor combination |
| `CrossAssetGreekAlpha` | Cross-asset alpha analysis |
| `AlphaAttribution` | Factor attribution analysis |

### Additional Core Components

| File | Lines | Class | Purpose |
|------|-------|-------|---------|
| `research/core/cross_sectional_alpha.py` | 480 | `CrossSectionalAlphaEngine` | Multi-asset alpha: rank, pairs, cross-sectional signals |
| `research/core/microstructure_alpha.py` | 480 | `MicrostructureAlphaEngine` | VPIN, order flow, Kyle's lambda, Amihud, Roll spread |
| `research/core/options_data_feed.py` | 586 | `OptionsDataFeed` | Multi-source options data (CBOE, yfinance, Deribit, CSV) |
| `research/core/domain_data_connector.py` | 400 | `DomainDataConnector` | Real market data → 5-domain analysis |
| `research/core/greek_mathematics.py` | 1087 | `GreekMathematics` | 10 ancient-Greek math concepts mapped to modern quant |
| `research/core/greek_walk_forward.py` | 424 | `GreekWalkForwardValidator` | Walk-forward alpha hypothesis validation |

### Greek Mathematics Modules (10 Ancient-to-Modern Mappings)

| Class | Ancient Concept | Modern Application |
|-------|----------------|-------------------|
| `EuclideanOrderSizer` | Euclidean distance | Distance-based position sizing |
| `ArchimedeanRebalancer` | Exhaustion convergence | Portfolio rebalancing |
| `EudoxianConvergence` | Eudoxus proportionality | Iterative κ estimation |
| `PythagoreanHarmony` | Harmonic mean | Portfolio balance |
| `ApolloniusCurvature` | Conic sections | Curvature-based gamma detection |
| `ZenoConvergence` | Zeno's paradox | Convergence series for mean-reversion |
| `PlatonicSymmetry` | Platonic solids | Ideal portfolio symmetry |
| `ThalesProportionality` | Thales' theorem | Price ratio analysis |
| `HipparchusTrigonometry` | Trigonometry | Cyclic pattern detection |
| `DiophantineOptimizer` | Diophantine equations | Integer constraint optimization |

---

## 8. Quantitative Pricing Pipeline

### Options Pricing Engine

| File | Lines | Class | Purpose |
|------|-------|-------|---------|
| `research/core/greeks.py` | 650 | `GreeksCalculator` | Full Greeks (Δ/Γ/Θ/V/ρ) with Numba JIT — <0.005ms |
| `research/core/black_scholes.py` | 425 | `BlackScholesCalculator` | BS pricing with Numba JIT — <0.001ms |
| `research/core/implied_volatility.py` | 563 | `ImpliedVolatilitySolver` | Newton-Raphson IV solver (Numba JIT) |
| `research/core/monte_carlo.py` | 731 | `MonteCarloEngine` | GBM simulation with variance reduction |
| `research/core/binomial_tree.py` | 624 | `BinomialTree` | CRR binomial model (European + American) |
| `research/core/risk_metrics.py` | 738 | `RiskMetrics` | VaR (historical/parametric/MC), CVaR |
| `research/core/volatility_surface.py` | 291 | `VolatilitySurface` | SVI parameterization, term structure, skew |
| `research/core/greeks_hedging.py` | 369 | `GreeksHedgingEngine`, `OptionsStrategyBuilder` | Delta hedging, iron condors, strangles |

### Pricing Flow
```
Market Data ──► ImpliedVolatilitySolver ──► VolatilitySurface (SVI fit)
                       │                          │
                       ▼                          ▼
              BlackScholesCalculator       MonteCarloEngine (GBM)
                       │                          │
                       ▼                          ▼
               GreeksCalculator            BinomialTree (CRR)
                       │                          │
                       ▼                          ▼
              GreeksHedgingEngine ────► RiskMetrics (VaR/CVaR)
```

---

## 9. Machine Learning Pipeline

### Components

| File | Lines | Class | Purpose |
|------|-------|-------|---------|
| `research/ml/feature_engineering.py` | 236 | `FeatureEngine` | Price/vol/technical/volume/regime feature generation |
| `research/ml/regime_detection.py` | 363 | `RegimeDetector` | GMM/HMM regime classification |
| `research/ml/volatility_forecast.py` | 407 | `VolatilityForecaster` | EWMA, GARCH(1,1), HAR models |

### Sub-models in Volatility Forecasting

| Class | Model | Purpose |
|-------|-------|---------|
| `EWMAVolModel` | Exponentially Weighted Moving Average | Fast, adaptive vol estimate |
| `GARCH11Model` | GARCH(1,1) | Parametric conditional volatility |
| `HARModel` | Heterogeneous AR | Multi-horizon realized vol (daily/weekly/monthly) |

### ML Flow
```
Raw OHLCV ──► FeatureEngine ──► Feature Matrix
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
           RegimeDetector   VolatilityForecaster   AlphaFactorLibrary
           (GMM/HMM)       (EWMA/GARCH/HAR)       (6 alpha factors)
                    │                │                │
                    └────────────────┼────────────────┘
                                     ▼
                           AlphaSignalEngine.generate_signal()
```

---

## 10. Quantum Computing Pipeline

### Components

| File | Lines | Class | Purpose |
|------|-------|-------|---------|
| `research/quantum/quantum_optimizer.py` | 569 | `QuantumOptimizer` | QAOA/VQE portfolio optimization (Qiskit) |
| `research/quantum/quantum_monte_carlo.py` | 706 | `QuantumMonteCarlo`, `AmplitudeEstimation` | Quantum amplitude estimation for option pricing |
| `research/quantum/quantum_ml.py` | 996 | `QuantumSupportVectorMachine`, `VariationalQuantumClassifier`, `QuantumFeatureMap`, `QuantumMLPipeline` | Quantum SVM, VQC |
| `research/quantum/portfolio_quantum.py` | 641 | `QuantumPortfolioOptimizer` | Mean-Variance/Risk Parity/CVaR via quantum |
| `research/quantum/hybrid_algorithms.py` | 915 | `QuantumClassicalNeuralNetwork`, `QuantumApproximateOptimization`, `VariationalQuantumEigensolver` | QCNN, QAOA, VQE hybrid |
| `research/quantum/risk_quantum.py` | 682 | `QuantumRiskAnalyzer`, `QuantumAmplitudeEstimation` | Quantum risk with amplitude estimation |
| `optimization/quantum_validation.py` | 477 | `QuantumBackendTester`, `QuantumAdvantageBenchmark`, `QuantumErrorMitigation` | Quantum backend testing, ZNE error mitigation |

### Quantum Flow
```
Portfolio Problem ──► QuantumOptimizer (QAOA/VQE) ──► Optimal Weights
                            │
Option Pricing ──► QuantumMonteCarlo (Amplitude Est.) ──► Quantum Price
                            │
Classification ──► QuantumMLPipeline (QSVM/VQC) ──► Regime Labels
                            │
Risk Analysis ──► QuantumRiskAnalyzer ──► Quantum VaR/CVaR

Note: All quantum imports wrapped in try/except — classical fallbacks active
```

---

## 11. Strategy Pipeline

### Strategy Hierarchy

```
Strategy (ABC)  ── base.py (599 lines)
    ├── IndicatorStrategy
    ├── MultiAssetStrategy
    ├── MomentumStrategy (955 lines)
    │       ├── TrendFollowingStrategy
    │       ├── BreakoutStrategy
    │       └── LiveMomentumStrategy  ◄── Used in Pipeline 2
    ├── PairsTradingStrategy (670 lines) — Engle-Granger cointegration
    ├── MarketMakingStrategy (769 lines)
    │       ├── AvellanedaStoikovMM
    │       └── SimpleSpreadMM
    ├── OptionsStrategy (1033 lines)
    │       ├── DeltaHedgingStrategy
    │       ├── VolatilityArbitrageStrategy
    │       └── IronCondorStrategy
    └── AdaptiveParameterOptimizer (372 lines) — Online learning
```

### Position Sizing

| Class | File | Approach |
|-------|------|----------|
| `PositionSizer` (ABC) | `research/strategies/base.py` | Abstract |
| `FixedFractionSizer` | `research/strategies/base.py` | Fixed % of capital |
| `KellyCriterionSizer` | `research/strategies/base.py` | Kelly criterion optimal |
| `PositionSizer` | `feedback/adaptive_engine.py` | Asymmetric adaptive sizing |

### Signal → Order Flow
```
Strategy.generate_signal() ──► Signal(side, strength, confidence)
         │
         ▼
DecisionReducer.reduce() ──► Weighted vote across strategies
         │
         ▼
ExecutionInstruction(BUY/SELL/HOLD, size, price, stop, target)
```

---

## 12. R Analytics Pipeline

### R Scripts (`research/r_analytics/`)

| File | Purpose |
|------|---------|
| `correlation_analysis.R` | DCC-GARCH, wavelet correlation, regime-dependent correlation |
| `econometrics.R` | ADF/PP unit root, Johansen cointegration, Granger causality |
| `performance_analytics.R` | PerformanceAnalytics-based Sharpe/Sortino/calmar/omega |
| `portfolio_optimization.R` | Mean-variance, risk parity, Black-Litterman, HRP |
| `regime_detection.R` | MS-AR, HMM, changepoint detection |
| `risk_modeling.R` | rugarch GARCH family, EVT (GEV/GPD), copula dependence |
| `timeseries_models.R` | ARIMA, VAR, DLM state-space, STL decomposition |

### R↔Python Bridge

```
Python (research) ──► RBridge (rpy2 session) ──► R Scripts
         │                    │
         ▼                    ▼
  DataConverter        RInterface (low-level)
  (Polars↔R df)              │
                              ▼
                    ModelWrapper (GARCH/ARIMA/VAR)
```

---

## 13. Backtesting Pipeline

### Components

| File | Lines | Class | Purpose |
|------|-------|-------|---------|
| `backtesting/engine.py` | 776 | `BacktestEngine`, `ExecutionSimulator`, `Portfolio` | Event-driven backtest engine |
| `backtesting/walk_forward.py` | 465 | `WalkForwardOptimizer` | Rolling window with overfitting detection |
| `backtesting/metrics.py` | 784 | `PerformanceMetrics`, `PerformanceAnalyzer` | Comprehensive metrics with bootstrap CI |
| `backtesting/performance.py` | 646 | `PerformanceMetrics`, `PerformanceAnalyzer` | MPT metrics, factor attribution |
| `backtesting/benchmark.py` | 781 | `BenchmarkComparison`, `BenchmarkAnalyzer` | Multi-benchmark comparison |
| `backtesting/result_store.py` | 131 | `BacktestResultStore` | JSON persistence with checksum |
| `backtesting/validator.py` | 204 | `ValidationPipeline`, `AirGapValidator` | Determinism, NaN guard, staleness, range checks |
| `backtesting/visualization.py` | 690 | `BacktestVisualizer` | Plotly equity curves, drawdown, rolling metrics |
| `backtesting/advanced_backtesting.py` | 442 | `GreekAwareBacktester`, `OptionsBacktester`, `MultiAssetBacktester` | Greek-aware, options, multi-asset backtesting |

### Backtesting Flow
```
Strategy + HistoricalData ──► BacktestEngine (event-driven)
         │                          │
         ▼                          ▼
  WalkForwardOptimizer       ExecutionSimulator
  (rolling windows)          (fill simulation)
         │                          │
         ▼                          ▼
  PerformanceMetrics ◄──── Portfolio (positions, cash)
         │
         ▼
  BenchmarkComparison ──► BacktestResultStore (JSON + checksum)
         │
         ▼
  AirGapValidator ──► Determinism check ──► BacktestVisualizer
```

---

## 14. Execution Pipeline

### Components

| File | Lines | Class | Purpose |
|------|-------|-------|---------|
| `execution/execution_engine.py` | 364 | `ExecutionEngine` | HFT fill simulator with "Phase 11 chaos mode" |
| `execution/binance_executor.py` | 395 | `BinanceExecutor` | Paper/live switchable via ccxt |
| `execution/order_manager.py` | 476 | `OrderManager`, `ExposureGovernor` | HFT microsecond order lifecycle |
| `execution/order_router.py` | 95 | `OrderRouter` | Signal-to-exchange routing |
| `execution/smart_router.py` | 357 | `SmartOrderRouter`, `SlicingEngine` | Optimal venue selection + order slicing |
| `execution/latency_monitor.py` | 427 | `LatencyMonitor` | Microsecond-precision multi-component |
| `execution/instructions.py` | 90 | `ExecutionInstruction` | Deterministic execution commands |

### Execution Flow
```
Signal ──► OrderRouter ──► SmartOrderRouter (venue selection)
                │                   │
                ▼                   ▼
        ExposureGovernor     SlicingEngine (TWAP/VWAP)
        (global limits)            │
                │                  ▼
                ▼           OrderManager (lifecycle)
        ExecutionInstruction       │
                │                  ▼
                └──────────► BinanceExecutor.execute_order()
                                   │
                             LatencyMonitor.record()
```

### Fill Models (`ExecutionEngine`)

| Model | Description |
|-------|-------------|
| `INSTANT` | Immediate fill at market price |
| `SLIPPAGE` | Realistic slippage simulation |
| `PARTIAL` | Partial fills with market impact |
| `CHAOS` | Phase 11 chaos mode — random delays, rejects, partial fills |

---

## 15. Risk Management Pipeline

### Components

| File | Lines | Class | Purpose |
|------|-------|-------|---------|
| `risk/session_guard.py` | 207 | `SessionGuard` | Global kill switch: drawdown, rate limiting, emergency shutdown |
| `risk/strategy_breaker.py` | 193 | `StrategyBreaker`, `StrategyBreakerManager` | Per-strategy circuit breaker: consecutive losses, daily cap, cooldown |
| `research/core/risk_metrics.py` | 738 | `RiskMetrics` | VaR (historical/parametric/MC), CVaR, max drawdown |

### Risk Hierarchy
```
Level 1: SessionGuard (GLOBAL)
    ├── Max drawdown check (e.g., 5%)
    ├── Daily loss limit (e.g., $500)
    ├── Rate limiter (orders/minute)
    ├── Emergency shutdown flag
    └── 10h session hard limit (SessionController)

Level 2: StrategyBreaker (PER-STRATEGY)
    ├── Consecutive loss counter
    ├── Daily loss cap per strategy
    ├── Cooldown period after trip
    └── Auto-reset on new session

Level 3: ExposureGovernor (POSITION-LEVEL)
    ├── Max position size
    ├── Max leverage
    ├── Margin requirements
    └── Liquidation price monitoring

Level 4: AirGapValidator (DATA-LEVEL)
    ├── Determinism check
    ├── NaN/Inf guard
    ├── Staleness detection
    └── Range validation
```

---

## 16. State Machine Pipeline

### TradingStateMachine (`brain/state_machine.py` — 257 lines)

```
                    ┌────────────┐
                    │    BOOT    │
                    └─────┬──────┘
                          │ initialization complete
                          ▼
                    ┌────────────┐
            ┌──────│    IDLE    │◄──────────────────┐
            │      └─────┬──────┘                    │
            │            │ signal received            │ cooldown expired
            ▼            ▼                            │
     ┌────────────┐  ┌────────────┐          ┌────────────┐
     │   HALTED   │  │ ANALYZING  │          │  COOLDOWN   │
     └────────────┘  └─────┬──────┘          └──────┬─────┘
            ▲              │ entry signal            │
            │              ▼                         │
            │       ┌────────────┐                   │
            │       │   ENTRY    │                   │
            │       └─────┬──────┘                   │
            │             │ filled                   │
            │             ▼                          │
            │       ┌────────────┐                   │
            │       │IN_POSITION │                   │
            │       └─────┬──────┘                   │
            │             │ exit signal              │
            │             ▼                          │
            │       ┌────────────┐                   │
            └───────│    EXIT    │───────────────────┘
                    └────────────┘
```

### States

| State | Description |
|-------|-------------|
| `BOOT` | System initialization, loading configs |
| `IDLE` | Waiting for signals, no open positions |
| `ANALYZING` | Processing incoming data, domain analysis |
| `ENTRY` | Placing entry orders |
| `IN_POSITION` | Active position, monitoring PnL |
| `EXIT` | Closing position, placing exit orders |
| `COOLDOWN` | Post-trade cooldown window |
| `HALTED` | Emergency stop, risk limit breached |

### Decision Reducer (`reducer/reducer.py` — 261 lines)

```
Strategy_1.signal() ──┐
Strategy_2.signal() ──┼──► DecisionReducer.reduce()
Strategy_3.signal() ──┘         │
                                ▼
                    Weighted Vote Aggregation
                    (configurable weights)
                                │
                                ▼
                    Final Verdict: BUY/SELL/HOLD
```

---

## 17. Monitoring & Observability Pipeline

### Components

| File | Lines | Class | Purpose |
|------|-------|-------|---------|
| `monitoring/system_monitor.py` | 260 | `SystemMonitor`, `MetricsCollector`, `AlertManager` | Prometheus-style metrics + alerting |
| `observer/observer.py` | 219 | `Observer` | Non-blocking async logging (read-only witness) |
| `utils/logger.py` | 415 | `GigaFormatter`, `PerformanceFilter`, `JsonFormatter`, `LogContext` | Colored console + JSON structured + rotation |
| `utils/performance_profiler.py` | 690 | `PerformanceProfiler`, `ProfileBlock` | Nanosecond precision with memory tracking |
| `utils/alerting.py` | 151 | `AlertManager` | Telegram + Discord alerts (non-blocking) |

### Observability Flow
```
on_tick() ──► Observer (async queue, 10K buffer)
    │                │
    │                ├── state.json (current state)
    │                └── events.log (event history, 10MB rotation)
    │
    ├──► PerformanceProfiler (nanosecond timing)
    │         └── memory/CPU tracking
    │
    ├──► SystemMonitor (Prometheus metrics)
    │         └── MetricsCollector → AlertManager
    │
    └──► Logger (colored console + JSON file + rotation)
              └── Telegram/Discord alerts on critical events
```

---

## 18. Feedback & Adaptation Pipeline

### Components

| File | Lines | Class | Purpose |
|------|-------|-------|---------|
| `feedback/adaptive_engine.py` | 185 | `AdaptiveEngine` | Asymmetric risk adaptation (cut 5% on loss, grow 3% on gain) |
| `feedback/adaptive_engine.py` | — | `CapitalRegimeEngine` | Capital allocation regime management |
| `feedback/adaptive_engine.py` | — | `PositionSizer` | Adaptive position sizing |
| `optimization/ai_optimizer.py` | 129 | `AIOptimizer` | Signal→Trade→PnL→Reward→Adjust feedback loop |
| `research/strategies/adaptive_params.py` | 372 | `AdaptiveParameterOptimizer` | Online learning parameter adaptation |

### Adaptation Flow
```
Trade Result (PnL) ──► AdaptiveEngine
         │                   │
         │          ┌────────┼────────┐
         │          ▼        ▼        ▼
         │    CapitalRegime  PositionSizer  Risk Multiplier
         │    Engine         (adaptive)     (asymmetric)
         │          │        │              │
         └──────────┼────────┼──────────────┘
                    ▼
              AIOptimizer.feedback_loop()
                    │
                    ▼
              Updated Parameters ──► Next on_tick()
```

---

## 19. Visualization Pipeline

### Components

| File | Lines | Purpose |
|------|-------|---------|
| `visualization/app.py` | 1540 | Main Streamlit dashboard (all modules integrated) |
| `visualization/greeks_dashboard.py` | 631 | 3D Greek surfaces (Δ/Γ/Θ/V over strike×maturity) |
| `visualization/risk_dashboard.py` | 878 | Real-time VaR/CVaR, Monte Carlo stress testing |
| `visualization/quantum_visualizer.py` | 1221 | Quantum circuit diagrams, Bloch sphere, QAOA/VQE |
| `visualization/education_mode.py` | 987 | Interactive BS/Greeks/portfolio tutorials |
| `visualization/education_viz.py` | 746 | Ancient-Greek-to-modern-quant, 3D risk surfaces |
| `visualization/pnl_attribution.py` | 746 | Waterfall P&L attribution, factor decomposition |
| `visualization/correlation_heatmap.py` | 926 | Hierarchical clustering, network visualization |
| `visualization/observer_app.py` | 70 | Minimal observer dashboard (reads state.json) |
| `visualization/statistical_plots.py` | — | Statistical distribution plots |
| `visualization/components.py` | — | Reusable Streamlit components |
| `visualization/charts.py` | — | Chart helpers |

### Visualization Architecture
```
Streamlit App (app.py) ──► Multi-page dashboard
    ├── Greeks Dashboard ──► 3D surface plots (Plotly)
    ├── Risk Dashboard ──► VaR/CVaR gauges, MC histograms
    ├── Quantum Visualizer ──► Circuit diagrams, Bloch sphere
    ├── Education Mode ──► Interactive BS tutorials
    ├── P&L Attribution ──► Waterfall charts
    ├── Correlation Heatmap ──► Hierarchical clustering
    └── Observer App ──► Real-time state.json monitor
```

---

## 20. Configuration Pipeline

### TOML Configuration Files

| File | Purpose |
|------|---------|
| `config/system_config.toml` | Global system settings |
| `config/strategies_config.toml` | **THE Bridge Artifact** — regime params, execution params, risk limits |
| `config/models_config.toml` | ML model hyperparameters |
| `config/database_config.toml` | DuckDB connection settings |

### Configuration Manager (`utils/config_loader.py` — 538 lines)

| Feature | Description |
|---------|-------------|
| `ConfigManager` | TOML loading with environment variable substitution |
| Encryption | Sensitive config encryption support |
| Hot Reload | Runtime config refresh without restart |
| Validation | Schema validation for config files |
| Env Substitution | `${ENV_VAR}` syntax in TOML values |

### Artifacts (`artifacts/definitions.py` — 127 lines)

| Class | Purpose |
|-------|---------|
| `MarketRegime(Enum)` | BULL/BEAR/SIDEWAYS/VOLATILE |
| `TimeHorizon(Enum)` | TICK/MINUTE/HOUR/DAILY/WEEKLY |
| `Context` | Execution context (regime, horizon, confidence) |
| `Artifact` | Base artifact with metadata |
| `SignalArtifact` | Signal artifact with direction/strength |

---

## 21. CI/CD & Testing Pipeline

### CI/CD (`scripts/ci_cd_pipeline.py` — 253 lines)

```
Lint (flake8/black) ──► Type Check (mypy) ──► Unit Tests (pytest)
         │                                          │
         ▼                                          ▼
  Integration Tests ──► Health Check ──► Docker Build ──► Deploy
```

### Health Check (`scripts/health_check.py` — 135 lines)

Full import health check across all modules — verifies every package imports cleanly.

### Test Suite

| File | Lines | Coverage |
|------|-------|----------|
| `tests/test_greeks.py` | 128 | BS pricing, put-call parity, delta approximation |
| `tests/test_risk.py` | 106 | CircuitBreaker, StrategyBreaker, SlidingWindowLimiter |
| `tests/test_routing.py` | — | Order routing logic |
| `tests/test_account.py` | — | LiveAccount PnL, margin |
| `tests/test_utils.py` | — | Utility functions |

### GitHub Actions (`.github/workflows/ci.yml`)

Automated CI pipeline on push/PR.

---

## 22. Complete Module Registry

### Total: 144 Python files across 20+ packages

| Package | Files | Total Lines | Primary Purpose |
|---------|-------|-------------|-----------------|
| Root | 7 | ~1,261 | Entry points, system orchestration |
| `research/core/` | 21 | ~8,495 | 5-Domain Greek Alpha Framework |
| `research/ml/` | 3 | ~1,006 | ML regime detection & vol forecasting |
| `research/quantum/` | 6 | ~4,509 | Quantum computing (Qiskit) |
| `research/strategies/` | 6 | ~4,398 | Trading strategy implementations |
| `research/r_analytics/` | 7 | — | R statistical scripts |
| `data/` | 7 | ~4,429 | Data ingestion & storage |
| `data/live/` | 2 | ~297 | Real-time data feeds |
| `execution/` | 7 | ~2,204 | Order execution & routing |
| `brain/` | 1 | ~257 | FSM state machine |
| `reducer/` | 1 | ~261 | Decision aggregation |
| `risk/` | 2 | ~400 | Risk management & circuit breakers |
| `account/` | 1 | ~290 | Live account management |
| `session/` | 1 | ~38 | Session control |
| `observer/` | 1 | ~219 | Read-only observation |
| `feedback/` | 1 | ~185 | Adaptive parameters |
| `optimization/` | 2 | ~606 | AI optimizer + quantum validation |
| `monitoring/` | 1 | ~260 | Prometheus-style monitoring |
| `bridge/` | 6 | ~3,206 | Research↔Live bridge + R bridge |
| `live/stream/` | 2 | ~1,445 | WebSocket streaming |
| `backtesting/` | 9 | ~4,919 | Event-driven backtesting |
| `utils/` | 8 | ~3,617 | Utilities, profiling, alerting |
| `visualization/` | 13 | ~8,745+ | Streamlit dashboards |
| `artifacts/` | 1 | ~127 | Artifact definitions |
| `tests/` | 5+ | ~234+ | Unit tests |
| `scripts/` | 4 | ~388+ | CI/CD, health check |
| `config/` | 4 | — | TOML configuration |

**Estimated Total**: ~51,000+ lines of Python/R code

---

## 23. Complete Class Registry

### 250+ Classes Across All Packages

#### research/core/ — 5-Domain Framework
| Class | File | Purpose |
|-------|------|---------|
| `VolatilityRegime(Enum)` | market_state_space.py | LOW/NORMAL/HIGH/EXTREME |
| `TrendRegime(Enum)` | market_state_space.py | STRONG_DOWN…STRONG_UP |
| `LiquidityRegime(Enum)` | market_state_space.py | ILLIQUID/NORMAL/LIQUID |
| `MarketState` | market_state_space.py | Dataclass: vol/trend/liquidity + risk_score |
| `StateSpaceOmega` | market_state_space.py | Domain 1: `classify_state()`, `get_transition_matrix()` |
| `SensitivityProfile` | greek_response.py | Dataclass: delta/gamma/theta/vega |
| `VariationalAnalyzer` | greek_response.py | Domain 2: `calculate_delta()`, `calculate_gamma()`, `calculate_theta()` |
| `StochasticParams` | stochastic_models.py | Dataclass: mu/sigma/kappa |
| `StochasticModeler` | stochastic_models.py | Domain 3: `fit_ornstein_uhlenbeck()` |
| `ErgodicityResult` | time_asymmetry.py | Dataclass: tau/epsilon/is_ergodic |
| `TimeAsymmetryAnalyzer` | time_asymmetry.py | Domain 4: `check_ergodicity()`, `calculate_relaxation_time()` |
| `InformationGeometer` | information_geometry.py | Domain 5: `calculate_shannon_entropy()`, `calculate_kl_divergence()` |
| `AlphaSignal` | alpha_signal_engine.py | Dataclass: direction/strength/confidence/regime |
| `AlphaDecayTracker` | alpha_signal_engine.py | Half-life alpha decay |
| `AlphaSignalEngine` | alpha_signal_engine.py | Central: `generate_signal()` from 5 domains |
| `AlphaFactor(ABC)` | alpha_factor_library.py | Abstract `compute()` |
| `KappaAlpha` | alpha_factor_library.py | Mean-reversion speed alpha |
| `EntropyAlpha` | alpha_factor_library.py | Information entropy alpha |
| `GammaAlpha` | alpha_factor_library.py | Gamma exposure alpha |
| `ErgodicityAlpha` | alpha_factor_library.py | Ergodicity alpha |
| `MomentumAlpha` | alpha_factor_library.py | Price momentum alpha |
| `VolRegimeAlpha` | alpha_factor_library.py | Volatility regime alpha |
| `HypothesisTest` | alpha_factor_library.py | Statistical significance |
| `AlphaResearchPipeline` | alpha_factor_library.py | End-to-end factor research |
| `AlphaCombiner` | alpha_factor_library.py | Multi-factor combination |
| `CrossAssetGreekAlpha` | alpha_factor_library.py | Cross-asset analysis |
| `AlphaAttribution` | alpha_factor_library.py | Factor attribution |
| `GreekResult` | greeks.py | Numba JIT Greeks result |
| `VolatilitySurface` | volatility_surface.py | SVI: `fit_svi()`, `get_implied_vol()` |
| `BSGreeksCalculator` | greeks_hedging.py | Full BS Greeks suite |
| `GreeksHedgingEngine` | greeks_hedging.py | Delta hedging |
| `OptionsStrategyBuilder` | greeks_hedging.py | Iron condors, strangles |
| `EuclideanOrderSizer` | greek_mathematics.py | Distance-based sizing |
| `ArchimedeanRebalancer` | greek_mathematics.py | Rebalancing |
| `EudoxianConvergence` | greek_mathematics.py | Iterative κ estimation |
| `PythagoreanHarmony` | greek_mathematics.py | Portfolio balance |
| `ApolloniusCurvature` | greek_mathematics.py | Gamma detection |
| `ZenoConvergence` | greek_mathematics.py | Mean-reversion series |
| `PlatonicSymmetry` | greek_mathematics.py | Portfolio symmetry |
| `ThalesProportionality` | greek_mathematics.py | Price ratio analysis |
| `HipparchusTrigonometry` | greek_mathematics.py | Cyclic patterns |
| `DiophantineOptimizer` | greek_mathematics.py | Integer optimization |
| `GreekWalkForwardValidator` | greek_walk_forward.py | Walk-forward validation |
| `CrossSectionalAlphaEngine` | cross_sectional_alpha.py | Multi-asset alpha |
| `VPINCalculator` | microstructure_alpha.py | Volume-Synchronized PIN |
| `OrderFlowImbalance` | microstructure_alpha.py | Buy/sell flow |
| `KyleLambda` | microstructure_alpha.py | Price impact |
| `AmihudIlliquidity` | microstructure_alpha.py | Illiquidity ratio |
| `RollSpread` | microstructure_alpha.py | Bid-ask estimator |
| `MicrostructureAlphaEngine` | microstructure_alpha.py | Microstructure orchestrator |
| `OptionsDataFeed` | options_data_feed.py | Multi-source options data |
| `DomainDataConnector` | domain_data_connector.py | Real data → 5 domains |

#### research/ml/
| Class | File | Purpose |
|-------|------|---------|
| `FeatureEngine` | feature_engineering.py | Feature generation |
| `RegimeDetector` | regime_detection.py | GMM/HMM regime classification |
| `VolatilityForecaster` | volatility_forecast.py | EWMA/GARCH/HAR forecasting |
| `EWMAVolModel` | volatility_forecast.py | EWMA volatility |
| `GARCH11Model` | volatility_forecast.py | GARCH(1,1) |
| `HARModel` | volatility_forecast.py | Heterogeneous AR |

#### research/quantum/
| Class | File | Purpose |
|-------|------|---------|
| `QuantumOptimizer` | quantum_optimizer.py | QAOA/VQE optimization |
| `QuantumMonteCarlo` | quantum_monte_carlo.py | Quantum amplitude estimation |
| `QuantumSupportVectorMachine` | quantum_ml.py | Quantum SVM |
| `VariationalQuantumClassifier` | quantum_ml.py | VQC classification |
| `QuantumMLPipeline` | quantum_ml.py | Quantum ML pipeline |
| `QuantumPortfolioOptimizer` | portfolio_quantum.py | Quantum portfolio opt |
| `QuantumClassicalNeuralNetwork` | hybrid_algorithms.py | QCNN hybrid |
| `QuantumRiskAnalyzer` | risk_quantum.py | Quantum risk analysis |

#### research/strategies/
| Class | File | Purpose |
|-------|------|---------|
| `Strategy(ABC)` | base.py | Abstract strategy base |
| `Signal`, `Order`, `Position`, `Trade` | base.py | Core trading primitives |
| `FixedFractionSizer` | base.py | Fixed fraction sizing |
| `KellyCriterionSizer` | base.py | Kelly criterion |
| `TrendFollowingStrategy` | momentum.py | Trend following |
| `BreakoutStrategy` | momentum.py | Breakout detection |
| `LiveMomentumStrategy` | momentum.py | **Live trading strategy** |
| `PairsTradingStrategy` | pairs_trading.py | Cointegration pairs |
| `AvellanedaStoikovMM` | market_making.py | Avellaneda-Stoikov MM |
| `DeltaHedgingStrategy` | options_strategies.py | Delta hedging |
| `VolatilityArbitrageStrategy` | options_strategies.py | Vol arb |
| `IronCondorStrategy` | options_strategies.py | Iron condor |
| `AdaptiveParameterOptimizer` | adaptive_params.py | Online learning |

#### execution/
| Class | File | Purpose |
|-------|------|---------|
| `ExecutionEngine` | execution_engine.py | HFT fill simulator |
| `BinanceExecutor` | binance_executor.py | Paper/live via ccxt |
| `OrderManager` | order_manager.py | Order lifecycle |
| `ExposureGovernor` | order_manager.py | Global exposure limits |
| `OrderRouter` | order_router.py | Signal→exchange routing |
| `SmartOrderRouter` | smart_router.py | Optimal venue selection |
| `SlicingEngine` | smart_router.py | Order slicing (TWAP/VWAP) |
| `LatencyMonitor` | latency_monitor.py | Microsecond precision |
| `ExecutionInstruction` | instructions.py | Deterministic commands |

#### brain/ + reducer/ + risk/ + account/ + session/ + observer/ + feedback/ + optimization/
| Class | File | Purpose |
|-------|------|---------|
| `StateMachineBrain` | brain/state_machine.py | FSM: BOOT→…→HALTED |
| `DecisionReducer` | reducer/reducer.py | Weighted vote aggregation |
| `SessionGuard` | risk/session_guard.py | Global kill switch |
| `StrategyBreaker` | risk/strategy_breaker.py | Per-strategy circuit breaker |
| `LiveAccount` | account/live_account.py | Real PnL, margin, leverage |
| `SessionController` | session/session_controller.py | 10h hard limit |
| `Observer` | observer/observer.py | Non-blocking async logging |
| `AdaptiveEngine` | feedback/adaptive_engine.py | Asymmetric adaptation |
| `CapitalRegimeEngine` | feedback/adaptive_engine.py | Capital regime management |
| `AIOptimizer` | optimization/ai_optimizer.py | Feedback-driven optimization |

#### bridge/
| Class | File | Purpose |
|-------|------|---------|
| `DataBridge` | data_bridge.py | Unified data interface |
| `TOMLGenerator` | research_live_bridge.py | TOML artifact generation |
| `TOMLVersionManager` | research_live_bridge.py | TOML versioning |
| `TOMLParameterReader` | research_live_bridge.py | TOML loading |
| `ResearchArtifactStore` | research_live_bridge.py | Artifact persistence |
| `ResearchLiveComparator` | research_live_bridge.py | Research vs live comparison |
| `RBridge` | r_bridge.py | R session management |
| `RInterface` | rpy2_interface.py | Low-level R↔Python |
| `DataConverter` | data_converter.py | Cross-language data conversion |

#### data/ + live/stream/
| Class | File | Purpose |
|-------|------|---------|
| `MarketDataLoader` | data/market_data.py | Multi-source data |
| `TimeSeriesDatabase` | data/database.py | DuckDB OLAP |
| `MarketStream` | data/live/market_stream.py | Binance REST ticker |
| `BinanceWebSocketFeed` | data/live/binance_ws_feed.py | Binance WS aggTrade |
| `StreamingManager` | live/stream/streaming.py | Multi-provider WS |
| `RealtimeManager` | live/stream/realtime_manager.py | Unified RT manager |

#### backtesting/
| Class | File | Purpose |
|-------|------|---------|
| `BacktestEngine` | engine.py | Event-driven engine |
| `WalkForwardOptimizer` | walk_forward.py | Rolling window validation |
| `PerformanceAnalyzer` | metrics.py | Bootstrap CI metrics |
| `BenchmarkAnalyzer` | benchmark.py | Multi-benchmark |
| `BacktestResultStore` | result_store.py | JSON + checksum |
| `ValidationPipeline` | validator.py | Air-gap validation |
| `BacktestVisualizer` | visualization.py | Plotly charts |
| `GreekAwareBacktester` | advanced_backtesting.py | Greek-aware backtesting |

#### utils/
| Class | File | Purpose |
|-------|------|---------|
| `ConfigManager` | config_loader.py | TOML + env vars |
| `PerformanceProfiler` | performance_profiler.py | Nanosecond profiling |
| `TokenBucketLimiter` | rate_limiter.py | Token-bucket rate limit |
| `SlidingWindowLimiter` | rate_limiter.py | Sliding-window rate limit |
| `CircuitBreaker` | retry.py | Circuit breaker pattern |
| `AlertManager` | alerting.py | Telegram + Discord |

---

## 24. Performance Engineering

| Technology | Application | Performance Target |
|------------|-------------|-------------------|
| **Numba JIT** | BS pricing, Greeks, indicators | <0.001ms (BS), <0.005ms (Greeks) |
| **DuckDB** | Time-series OLAP storage | Columnar, thread-safe pool |
| **Polars** | Data preprocessing | Zero-copy DataFrames |
| **perf_counter_ns()** | Latency monitoring | Nanosecond precision |
| **Token-bucket** | API rate limiting | Thread-safe, configurable |
| **Async queue** | Observer logging | Non-blocking, 10K buffer |
| **Numba vectorize** | SMA/EMA/RSI/MACD | Vectorized indicator computation |

---

## 25. Design Patterns

| Pattern | Implementation | Purpose |
|---------|---------------|---------|
| **Air-Gap Architecture** | TOML-only bridge between research and live | No look-ahead bias, deterministic |
| **Finite State Machine** | `StateMachineBrain` (7 states) | Explicit trading state transitions |
| **Circuit Breaker** | `StrategyBreaker` + `CircuitBreaker` | Fault isolation, auto-recovery |
| **Asymmetric Adaptation** | `AdaptiveEngine` (cut 5% loss, grow 3% gain) | Conservative risk management |
| **Graceful Degradation** | try/except on quantum/R imports | Classical fallbacks always available |
| **Observer Pattern** | `Observer` (read-only, async) | Non-intrusive system monitoring |
| **Strategy Pattern** | `Strategy(ABC)` → multiple implementations | Pluggable strategy framework |
| **Factory Pattern** | `ExecutionEngine` fill models | Configurable fill simulation |
| **Token Bucket** | `TokenBucketRateLimiter` | API call rate management |
| **Weighted Voting** | `DecisionReducer` | Multi-strategy signal aggregation |

---

## Document Signature

- **System**: GIGA-SYSTEM
- **Architecture**: 2-Pipeline Air-Gap (Research → TOML Bridge → Live)
- **Core Hypothesis**: High κ + Low Entropy = Maximal Alpha
- **Framework**: 5-Domain Greek Alpha Framework
- **Total Files**: 144 Python + 7 R + 4 TOML + Docker/CI
- **Total Lines**: ~51,000+ estimated
- **Total Classes**: 250+
- **Generated**: Pipeline Document v1.0
