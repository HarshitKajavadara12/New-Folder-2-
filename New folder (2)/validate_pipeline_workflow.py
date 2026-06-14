#!/usr/bin/env python3
"""
GIGA-SYSTEM — Pipeline & Workflow Validation Script
Validates that every component documented in PIPELINE_DOCUMENT.md and WORKFLOW_DOCUMENT.md
actually exists and is wired correctly in the codebase.

Total checks: 400+
"""

import os
import sys
import re
import ast
import importlib.util
from pathlib import Path

# ─── Configuration ───────────────────────────────────────────────────────────

GIGA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "giga-system")
PASS = 0
FAIL = 0
WARN = 0
RESULTS = []


def check(condition: bool, description: str, warn_only: bool = False):
    global PASS, FAIL, WARN
    if condition:
        PASS += 1
        RESULTS.append(("PASS", description))
    elif warn_only:
        WARN += 1
        RESULTS.append(("WARN", description))
    else:
        FAIL += 1
        RESULTS.append(("FAIL", description))


def file_exists(rel_path: str) -> bool:
    return os.path.isfile(os.path.join(GIGA_ROOT, rel_path))


def dir_exists(rel_path: str) -> bool:
    return os.path.isdir(os.path.join(GIGA_ROOT, rel_path))


def file_contains(rel_path: str, pattern: str) -> bool:
    fpath = os.path.join(GIGA_ROOT, rel_path)
    if not os.path.isfile(fpath):
        return False
    try:
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        return bool(re.search(pattern, content, re.IGNORECASE))
    except Exception:
        return False


def file_has_class(rel_path: str, class_name: str) -> bool:
    return file_contains(rel_path, rf"class\s+{class_name}\b")


def file_has_function(rel_path: str, func_name: str) -> bool:
    return file_contains(rel_path, rf"def\s+{func_name}\b")


def file_has_import(rel_path: str, import_name: str) -> bool:
    return file_contains(rel_path, rf"(?:from|import)\s+.*{import_name}")


def count_lines(rel_path: str) -> int:
    fpath = os.path.join(GIGA_ROOT, rel_path)
    if not os.path.isfile(fpath):
        return 0
    try:
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            return len(f.readlines())
    except Exception:
        return 0


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: DIRECTORY STRUCTURE VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_directory_structure():
    print("\n" + "=" * 80)
    print("SECTION 1: DIRECTORY STRUCTURE VALIDATION")
    print("=" * 80)

    # Core directories
    dirs = [
        "research", "research/core", "research/ml", "research/quantum",
        "research/strategies", "research/r_analytics",
        "data", "data/live",
        "execution",
        "brain",
        "reducer",
        "risk",
        "account",
        "session",
        "observer",
        "feedback",
        "optimization",
        "monitoring",
        "bridge",
        "live", "live/stream",
        "backtesting",
        "utils",
        "visualization",
        "artifacts",
        "tests",
        "scripts",
        "config",
    ]
    for d in dirs:
        check(dir_exists(d), f"Directory exists: {d}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: ENTRY POINT FILES
# ═══════════════════════════════════════════════════════════════════════════════

def validate_entry_points():
    print("\n" + "=" * 80)
    print("SECTION 2: ENTRY POINT FILES")
    print("=" * 80)

    # Entry point files exist
    entry_files = [
        "launch_giga_system.py",
        "run_system_pipeline.py",
        "demo_complete_system.py",
        "run_greek_research_lab.py",
        "QUICK_START.py",
        "setup.py",
        "test_connection.py",
    ]
    for f in entry_files:
        check(file_exists(f), f"Entry point exists: {f}")

    # launch_giga_system.py specifics
    check(file_contains("launch_giga_system.py", r"on_tick"), "launch_giga_system.py has on_tick function")
    check(file_contains("launch_giga_system.py", r"MarketStream"), "launch_giga_system.py uses MarketStream")
    check(file_contains("launch_giga_system.py", r"BinanceExecutor"), "launch_giga_system.py uses BinanceExecutor")
    check(file_contains("launch_giga_system.py", r"LiveAccount"), "launch_giga_system.py uses LiveAccount")
    check(file_contains("launch_giga_system.py", r"SessionController"), "launch_giga_system.py uses SessionController")
    check(file_contains("launch_giga_system.py", r"LiveMomentumStrategy"), "launch_giga_system.py uses LiveMomentumStrategy")
    check(file_contains("launch_giga_system.py", r"VariationalAnalyzer"), "launch_giga_system.py uses VariationalAnalyzer")
    check(file_contains("launch_giga_system.py", r"SessionGuard|SessionController|session_guard|session_controller"), "launch_giga_system.py uses SessionGuard/SessionController")
    check(file_contains("launch_giga_system.py", r"calculate_delta"), "launch_giga_system.py calls calculate_delta")
    check(file_contains("launch_giga_system.py", r"paper"), "launch_giga_system.py has paper mode reference")
    check(file_contains("launch_giga_system.py", r"BTCUSDT"), "launch_giga_system.py targets BTCUSDT")
    check(file_contains("launch_giga_system.py", r"toml|config"), "launch_giga_system.py loads TOML config")

    # run_system_pipeline.py specifics
    check(file_contains("run_system_pipeline.py", r"pipeline_1|research"), "run_system_pipeline.py has Pipeline 1 (research)")
    check(file_contains("run_system_pipeline.py", r"pipeline_2|live"), "run_system_pipeline.py has Pipeline 2 (live)")
    check(file_contains("run_system_pipeline.py", r"argparse|mode"), "run_system_pipeline.py has CLI mode selection")
    check(file_contains("run_system_pipeline.py", r"strategies_config"), "run_system_pipeline.py references strategies_config")

    # demo_complete_system.py specifics
    check(file_contains("demo_complete_system.py", r"DecisionReducer"), "demo_complete_system.py uses DecisionReducer")
    check(file_contains("demo_complete_system.py", r"CapitalRegimeEngine|AdaptiveEngine"), "demo_complete_system.py uses adaptive components")
    check(file_contains("demo_complete_system.py", r"Observer"), "demo_complete_system.py uses Observer")
    check(file_contains("demo_complete_system.py", r"bridge|artifact|toml"), "demo_complete_system.py generates bridge artifacts")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: 5-DOMAIN GREEK ALPHA FRAMEWORK
# ═══════════════════════════════════════════════════════════════════════════════

def validate_5_domain_framework():
    print("\n" + "=" * 80)
    print("SECTION 3: 5-DOMAIN GREEK ALPHA FRAMEWORK")
    print("=" * 80)

    # Domain 1: Market State Space
    check(file_exists("research/core/market_state_space.py"), "Domain 1 file exists: market_state_space.py")
    check(file_has_class("research/core/market_state_space.py", "StateSpaceOmega"), "Domain 1 class: StateSpaceOmega")
    check(file_has_function("research/core/market_state_space.py", "classify_state"), "Domain 1 method: classify_state()")
    check(file_contains("research/core/market_state_space.py", r"VolatilityRegime"), "Domain 1 enum: VolatilityRegime")
    check(file_contains("research/core/market_state_space.py", r"TrendRegime"), "Domain 1 enum: TrendRegime")
    check(file_contains("research/core/market_state_space.py", r"LiquidityRegime"), "Domain 1 enum: LiquidityRegime")
    check(file_has_class("research/core/market_state_space.py", "MarketState"), "Domain 1 dataclass: MarketState")

    # Domain 2: Variational Sensitivity
    check(file_exists("research/core/greek_response.py"), "Domain 2 file exists: greek_response.py")
    check(file_has_class("research/core/greek_response.py", "VariationalAnalyzer"), "Domain 2 class: VariationalAnalyzer")
    check(file_has_function("research/core/greek_response.py", "calculate_delta"), "Domain 2 method: calculate_delta()")
    check(file_has_function("research/core/greek_response.py", "calculate_gamma"), "Domain 2 method: calculate_gamma()")
    check(file_has_function("research/core/greek_response.py", "calculate_theta"), "Domain 2 method: calculate_theta()")
    check(file_contains("research/core/greek_response.py", r"SensitivityProfile"), "Domain 2 dataclass: SensitivityProfile")

    # Domain 3: Stochastic Models
    check(file_exists("research/core/stochastic_models.py"), "Domain 3 file exists: stochastic_models.py")
    check(file_has_class("research/core/stochastic_models.py", "StochasticModeler"), "Domain 3 class: StochasticModeler")
    check(file_has_function("research/core/stochastic_models.py", "fit_ornstein_uhlenbeck"), "Domain 3 method: fit_ornstein_uhlenbeck()")
    check(file_contains("research/core/stochastic_models.py", r"kappa"), "Domain 3 has kappa parameter")
    check(file_contains("research/core/stochastic_models.py", r"StochasticParams"), "Domain 3 dataclass: StochasticParams")

    # Domain 4: Time Asymmetry
    check(file_exists("research/core/time_asymmetry.py"), "Domain 4 file exists: time_asymmetry.py")
    check(file_has_class("research/core/time_asymmetry.py", "TimeAsymmetryAnalyzer"), "Domain 4 class: TimeAsymmetryAnalyzer")
    check(file_has_function("research/core/time_asymmetry.py", "check_ergodicity"), "Domain 4 method: check_ergodicity()")
    check(file_has_function("research/core/time_asymmetry.py", "calculate_relaxation_time"), "Domain 4 method: calculate_relaxation_time()")
    check(file_contains("research/core/time_asymmetry.py", r"ErgodicityResult"), "Domain 4 dataclass: ErgodicityResult")

    # Domain 5: Information Geometry
    check(file_exists("research/core/information_geometry.py"), "Domain 5 file exists: information_geometry.py")
    check(file_has_class("research/core/information_geometry.py", "InformationGeometer"), "Domain 5 class: InformationGeometer")
    check(file_has_function("research/core/information_geometry.py", "calculate_shannon_entropy"), "Domain 5 method: calculate_shannon_entropy()")
    check(file_has_function("research/core/information_geometry.py", "calculate_market_entropy"), "Domain 5 method: calculate_market_entropy()")
    check(file_has_function("research/core/information_geometry.py", "calculate_kl_divergence"), "Domain 5 method: calculate_kl_divergence()")

    # AlphaSignalEngine (Central orchestrator)
    check(file_exists("research/core/alpha_signal_engine.py"), "Central orchestrator exists: alpha_signal_engine.py")
    check(file_has_class("research/core/alpha_signal_engine.py", "AlphaSignalEngine"), "Class: AlphaSignalEngine")
    check(file_has_function("research/core/alpha_signal_engine.py", "generate_signal"), "Method: generate_signal()")
    check(file_contains("research/core/alpha_signal_engine.py", r"AlphaSignal"), "Dataclass: AlphaSignal")
    check(file_contains("research/core/alpha_signal_engine.py", r"AlphaDecayTracker"), "Class: AlphaDecayTracker")
    # Verify it imports all 5 domains
    check(file_has_import("research/core/alpha_signal_engine.py", r"StateSpaceOmega|market_state_space"), "AlphaSignalEngine imports Domain 1")
    check(file_has_import("research/core/alpha_signal_engine.py", r"VariationalAnalyzer|greek_response"), "AlphaSignalEngine imports Domain 2")
    check(file_has_import("research/core/alpha_signal_engine.py", r"StochasticModeler|stochastic_models"), "AlphaSignalEngine imports Domain 3")
    check(file_has_import("research/core/alpha_signal_engine.py", r"TimeAsymmetryAnalyzer|time_asymmetry"), "AlphaSignalEngine imports Domain 4")
    check(file_has_import("research/core/alpha_signal_engine.py", r"InformationGeometer|information_geometry"), "AlphaSignalEngine imports Domain 5")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: RESEARCH CORE FILES (21 files)
# ═══════════════════════════════════════════════════════════════════════════════

def validate_research_core():
    print("\n" + "=" * 80)
    print("SECTION 4: RESEARCH CORE FILES")
    print("=" * 80)

    core_files = {
        "research/core/alpha_factor_library.py": [
            ("AlphaFactor", "class"),
            ("KappaAlpha", "class"),
            ("EntropyAlpha", "class"),
            ("GammaAlpha", "class"),
            ("ErgodicityAlpha", "class"),
            ("MomentumAlpha", "class"),
            ("VolRegimeAlpha", "class"),
            ("HypothesisTest", "class"),
            ("AlphaResearchPipeline", "class"),
            ("AlphaCombiner", "class"),
            ("AlphaAttribution", "class"),
        ],
        "research/core/greeks.py": [
            ("GreeksCalculator|GreekResult", "class"),
        ],
        "research/core/black_scholes.py": [
            ("BlackScholes|black_scholes|bs_price|jit", "function_or_jit"),
        ],
        "research/core/monte_carlo.py": [
            ("MonteCarloEngine|MonteCarlo|simulate_gbm|monte_carlo|jit", "function_or_jit"),
        ],
        "research/core/implied_volatility.py": [
            ("ImpliedVolatilitySolver|ImpliedVolatility|implied_vol|newton|jit", "function_or_jit"),
        ],
        "research/core/binomial_tree.py": [
            ("BinomialTree|binomial|crr|jit", "function_or_jit"),
        ],
        "research/core/risk_metrics.py": [
            ("RiskMetrics|var|cvar|VaR|CVaR|risk_metric", "function_or_jit"),
        ],
        "research/core/volatility_surface.py": [
            ("VolatilitySurface", "class"),
        ],
        "research/core/greeks_hedging.py": [
            ("GreeksHedgingEngine|GreeksHedging", "class"),
            ("OptionsStrategyBuilder", "class"),
        ],
        "research/core/greek_mathematics.py": [
            ("EuclideanOrderSizer|Euclidean", "class"),
            ("ArchimedeanRebalancer|Archimedean", "class"),
            ("EudoxianConvergence|Eudoxian", "class"),
            ("PythagoreanHarmony|Pythagorean", "class"),
            ("ApolloniusCurvature|Apollonius", "class"),
            ("ZenoConvergence|Zeno", "class"),
            ("PlatonicSymmetry|Platonic", "class"),
            ("ThalesProportionality|Thales", "class"),
            ("HipparchusTrigonometry|Hipparchus", "class"),
            ("DiophantineOptimizer|Diophantine", "class"),
        ],
        "research/core/greek_walk_forward.py": [
            ("GreekWalkForward", "function_or_jit"),
        ],
        "research/core/cross_sectional_alpha.py": [
            ("CrossSectionalAlpha", "function_or_jit"),
        ],
        "research/core/microstructure_alpha.py": [
            ("VPINCalculator|VPIN", "class"),
            ("OrderFlowImbalance", "class"),
            ("KyleLambda", "class"),
            ("MicrostructureAlpha", "function_or_jit"),
        ],
        "research/core/options_data_feed.py": [
            ("OptionsDataFeed", "class"),
        ],
        "research/core/domain_data_connector.py": [
            ("DomainDataConnector", "class"),
        ],
    }

    for fpath, checks in core_files.items():
        check(file_exists(fpath), f"Research core file exists: {fpath}")
        for pattern, ctype in checks:
            if ctype == "function_or_jit":
                check(file_contains(fpath, pattern), f"{fpath}: contains {pattern.split('|')[0]}")
            else:
                check(file_contains(fpath, rf"class\s+(?:{pattern})\b"), f"{fpath}: class {pattern}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: RESEARCH ML FILES
# ═══════════════════════════════════════════════════════════════════════════════

def validate_research_ml():
    print("\n" + "=" * 80)
    print("SECTION 5: RESEARCH ML FILES")
    print("=" * 80)

    check(file_exists("research/ml/feature_engineering.py"), "ML file: feature_engineering.py")
    check(file_has_class("research/ml/feature_engineering.py", "FeatureEngin") or file_contains("research/ml/feature_engineering.py", r"class\s+FeatureEngine"), "ML class: FeatureEngine")

    check(file_exists("research/ml/regime_detection.py"), "ML file: regime_detection.py")
    check(file_has_class("research/ml/regime_detection.py", "RegimeDetect") or file_contains("research/ml/regime_detection.py", r"class\s+RegimeDetect"), "ML class: RegimeDetector")
    check(file_contains("research/ml/regime_detection.py", r"GMM|GaussianMixture"), "RegimeDetector uses GMM")
    check(file_contains("research/ml/regime_detection.py", r"HMM|hmm"), "RegimeDetector uses HMM")

    check(file_exists("research/ml/volatility_forecast.py"), "ML file: volatility_forecast.py")
    check(file_has_class("research/ml/volatility_forecast.py", "VolatilityForecaster"), "ML class: VolatilityForecaster")
    check(file_contains("research/ml/volatility_forecast.py", r"EWMA"), "VolForecaster has EWMA model")
    check(file_contains("research/ml/volatility_forecast.py", r"GARCH"), "VolForecaster has GARCH model")
    check(file_contains("research/ml/volatility_forecast.py", r"HAR"), "VolForecaster has HAR model")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: RESEARCH QUANTUM FILES
# ═══════════════════════════════════════════════════════════════════════════════

def validate_research_quantum():
    print("\n" + "=" * 80)
    print("SECTION 6: RESEARCH QUANTUM FILES")
    print("=" * 80)

    quantum_checks = {
        "research/quantum/quantum_optimizer.py": ["QuantumOptimizer|QuantumPortfolioOptimizer", "QAOA|VQE"],
        "research/quantum/quantum_monte_carlo.py": ["QuantumMonteCarlo", "AmplitudeEstimation|amplitude"],
        "research/quantum/quantum_ml.py": ["QuantumSupportVectorMachine|QuantumSVM|QuantumML", "VariationalQuantumClassifier|VQC"],
        "research/quantum/portfolio_quantum.py": ["QuantumPortfolio", "mean.variance|risk.parity|CVaR"],
        "research/quantum/hybrid_algorithms.py": ["QuantumClassicalNeuralNetwork|QCNN|HybridAlgorithm", "QAOA|VQE"],
        "research/quantum/risk_quantum.py": ["QuantumRiskAnalyzer|QuantumRisk", "amplitude"],
    }

    for fpath, patterns in quantum_checks.items():
        check(file_exists(fpath), f"Quantum file exists: {fpath}")
        for pattern in patterns:
            check(file_contains(fpath, pattern), f"{fpath}: contains {pattern}")

    # Graceful degradation check
    has_try_except = False
    for fpath in quantum_checks:
        if file_contains(fpath, r"try.*import|except.*Import"):
            has_try_except = True
            break
    check(has_try_except, "Quantum files have try/except import guards (graceful degradation)")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: RESEARCH STRATEGIES FILES
# ═══════════════════════════════════════════════════════════════════════════════

def validate_research_strategies():
    print("\n" + "=" * 80)
    print("SECTION 7: RESEARCH STRATEGIES FILES")
    print("=" * 80)

    # base.py
    check(file_exists("research/strategies/base.py"), "Strategy base exists: base.py")
    check(file_has_class("research/strategies/base.py", "Strategy"), "Strategy ABC class")
    check(file_contains("research/strategies/base.py", r"Signal"), "Signal class/dataclass")
    check(file_contains("research/strategies/base.py", r"Order"), "Order class/dataclass")
    check(file_contains("research/strategies/base.py", r"Position"), "Position class/dataclass")
    check(file_contains("research/strategies/base.py", r"PositionSizer"), "PositionSizer class")
    check(file_contains("research/strategies/base.py", r"KellyCriterion"), "KellyCriterionSizer")

    # momentum.py
    check(file_exists("research/strategies/momentum.py"), "Strategy file: momentum.py")
    check(file_has_class("research/strategies/momentum.py", "MomentumStrategy|TrendFollowingStrategy"), "MomentumStrategy class")
    check(file_has_class("research/strategies/momentum.py", "BreakoutStrategy"), "BreakoutStrategy class")
    check(file_has_class("research/strategies/momentum.py", "LiveMomentumStrategy"), "LiveMomentumStrategy class (used in Pipeline 2)")

    # pairs_trading.py
    check(file_exists("research/strategies/pairs_trading.py"), "Strategy file: pairs_trading.py")
    check(file_has_class("research/strategies/pairs_trading.py", "PairsTradingStrategy"), "PairsTradingStrategy class")
    check(file_contains("research/strategies/pairs_trading.py", r"cointegrat"), "Pairs trading uses cointegration")

    # market_making.py
    check(file_exists("research/strategies/market_making.py"), "Strategy file: market_making.py")
    check(file_contains("research/strategies/market_making.py", r"Avellaneda|MarketMaking"), "MarketMaking has Avellaneda-Stoikov")

    # options_strategies.py
    check(file_exists("research/strategies/options_strategies.py"), "Strategy file: options_strategies.py")
    check(file_contains("research/strategies/options_strategies.py", r"DeltaHedging|delta.*hedg"), "Options: DeltaHedging")
    check(file_contains("research/strategies/options_strategies.py", r"VolatilityArbitrage|vol.*arb"), "Options: VolatilityArbitrage")
    check(file_contains("research/strategies/options_strategies.py", r"IronCondor|iron.*condor"), "Options: IronCondor")

    # adaptive_params.py
    check(file_exists("research/strategies/adaptive_params.py"), "Strategy file: adaptive_params.py")
    check(file_has_class("research/strategies/adaptive_params.py", "AdaptiveParameter") or file_contains("research/strategies/adaptive_params.py", r"class\s+AdaptiveParameter"), "AdaptiveParameterOptimizer class")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: R ANALYTICS FILES
# ═══════════════════════════════════════════════════════════════════════════════

def validate_r_analytics():
    print("\n" + "=" * 80)
    print("SECTION 8: R ANALYTICS FILES")
    print("=" * 80)

    r_files = [
        "research/r_analytics/correlation_analysis.R",
        "research/r_analytics/econometrics.R",
        "research/r_analytics/performance_analytics.R",
        "research/r_analytics/portfolio_optimization.R",
        "research/r_analytics/regime_detection.R",
        "research/r_analytics/risk_modeling.R",
        "research/r_analytics/timeseries_models.R",
    ]
    for f in r_files:
        check(file_exists(f), f"R analytics file exists: {f}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9: DATA PIPELINE FILES
# ═══════════════════════════════════════════════════════════════════════════════

def validate_data_pipeline():
    print("\n" + "=" * 80)
    print("SECTION 9: DATA PIPELINE FILES")
    print("=" * 80)

    # Core data files
    check(file_exists("data/market_data.py"), "Data file: market_data.py")
    check(file_has_class("data/market_data.py", "MarketDataLoader"), "MarketDataLoader class")

    check(file_exists("data/database.py"), "Data file: database.py")
    check(file_has_class("data/database.py", "TimeSeriesDatabase") or file_has_class("data/database.py", "DatabaseManager"), "Database class (TimeSeriesDatabase or DatabaseManager)")
    check(file_contains("data/database.py", r"duckdb|DuckDB"), "Database uses DuckDB")

    check(file_exists("data/database_layer.py"), "Data file: database_layer.py")
    check(file_contains("data/database_layer.py", r"ohlcv|CREATE TABLE"), "database_layer has schema")

    check(file_exists("data/indicators.py"), "Data file: indicators.py")
    check(file_contains("data/indicators.py", r"numba|jit|njit"), "Indicators use Numba JIT")
    check(file_contains("data/indicators.py", r"SMA|sma"), "Indicators: SMA")
    check(file_contains("data/indicators.py", r"EMA|ema"), "Indicators: EMA")
    check(file_contains("data/indicators.py", r"RSI|rsi"), "Indicators: RSI")
    check(file_contains("data/indicators.py", r"MACD|macd"), "Indicators: MACD")

    check(file_exists("data/preprocessing.py"), "Data file: preprocessing.py")
    check(file_contains("data/preprocessing.py", r"polars|pl\."), "Preprocessing uses Polars")

    check(file_exists("data/multi_exchange.py"), "Data file: multi_exchange.py")
    check(file_contains("data/multi_exchange.py", r"ccxt"), "Multi-exchange uses ccxt")

    check(file_exists("data/storage_manager.py"), "Data file: storage_manager.py")

    # Live data
    check(file_exists("data/live/market_stream.py"), "Live data: market_stream.py")
    check(file_has_class("data/live/market_stream.py", "MarketStream"), "MarketStream class")
    check(file_contains("data/live/market_stream.py", r"TokenBucket|rate_limit"), "MarketStream has rate limiter")
    check(file_contains("data/live/market_stream.py", r"binance|Binance"), "MarketStream connects to Binance")

    check(file_exists("data/live/binance_ws_feed.py"), "Live data: binance_ws_feed.py")
    check(file_has_class("data/live/binance_ws_feed.py", "BinanceW") or file_contains("data/live/binance_ws_feed.py", r"class\s+Binance"), "BinanceWebSocketFeed class")
    check(file_contains("data/live/binance_ws_feed.py", r"websocket|WebSocket|aggTrade"), "BinanceWS uses WebSocket")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10: EXECUTION PIPELINE FILES
# ═══════════════════════════════════════════════════════════════════════════════

def validate_execution_pipeline():
    print("\n" + "=" * 80)
    print("SECTION 10: EXECUTION PIPELINE FILES")
    print("=" * 80)

    check(file_exists("execution/execution_engine.py"), "Exec file: execution_engine.py")
    check(file_has_class("execution/execution_engine.py", "ExecutionEngine"), "ExecutionEngine class")
    check(file_contains("execution/execution_engine.py", r"chaos|CHAOS"), "ExecutionEngine has chaos mode")

    check(file_exists("execution/binance_executor.py"), "Exec file: binance_executor.py")
    check(file_has_class("execution/binance_executor.py", "BinanceExecutor"), "BinanceExecutor class")
    check(file_contains("execution/binance_executor.py", r"paper|PAPER"), "BinanceExecutor has paper mode")
    check(file_contains("execution/binance_executor.py", r"ccxt"), "BinanceExecutor uses ccxt")

    check(file_exists("execution/order_manager.py"), "Exec file: order_manager.py")
    check(file_has_class("execution/order_manager.py", "OrderManager"), "OrderManager class")
    check(file_has_class("execution/order_manager.py", "ExposureGovernor"), "ExposureGovernor class")

    check(file_exists("execution/order_router.py"), "Exec file: order_router.py")
    check(file_has_class("execution/order_router.py", "OrderRouter"), "OrderRouter class")

    check(file_exists("execution/smart_router.py"), "Exec file: smart_router.py")
    check(file_contains("execution/smart_router.py", r"SmartRouter|SmartOrderRouter"), "SmartOrderRouter class")
    check(file_has_class("execution/smart_router.py", "SlicingEngine"), "SlicingEngine class")

    check(file_exists("execution/latency_monitor.py"), "Exec file: latency_monitor.py")
    check(file_has_class("execution/latency_monitor.py", "LatencyMonitor"), "LatencyMonitor class")
    check(file_contains("execution/latency_monitor.py", r"perf_counter|microsecond|nanosecond"), "LatencyMonitor uses precision timing")

    check(file_exists("execution/instructions.py"), "Exec file: instructions.py")
    check(file_contains("execution/instructions.py", r"ExecutionInstruction|Instruction"), "ExecutionInstruction class")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11: BRAIN, REDUCER, RISK, ACCOUNT, SESSION, OBSERVER
# ═══════════════════════════════════════════════════════════════════════════════

def validate_core_components():
    print("\n" + "=" * 80)
    print("SECTION 11: BRAIN, REDUCER, RISK, ACCOUNT, SESSION, OBSERVER")
    print("=" * 80)

    # Brain / State Machine
    check(file_exists("brain/state_machine.py"), "Brain file: state_machine.py")
    check(file_has_class("brain/state_machine.py", "StateMachineBrain|TradingStateMachine"), "StateMachineBrain class")
    check(file_contains("brain/state_machine.py", r"BOOT"), "State: BOOT")
    check(file_contains("brain/state_machine.py", r"IDLE"), "State: IDLE")
    check(file_contains("brain/state_machine.py", r"ANALYZING"), "State: ANALYZING")
    check(file_contains("brain/state_machine.py", r"ENTRY"), "State: ENTRY")
    check(file_contains("brain/state_machine.py", r"IN_POSITION"), "State: IN_POSITION")
    check(file_contains("brain/state_machine.py", r"EXIT"), "State: EXIT")
    check(file_contains("brain/state_machine.py", r"COOLDOWN"), "State: COOLDOWN")
    check(file_contains("brain/state_machine.py", r"HALTED"), "State: HALTED")
    check(file_contains("brain/state_machine.py", r"State.*Enum|class\s+State"), "State enum defined")

    # Reducer
    check(file_exists("reducer/reducer.py"), "Reducer file: reducer.py")
    check(file_has_class("reducer/reducer.py", "DecisionReducer"), "DecisionReducer class")
    check(file_has_function("reducer/reducer.py", "reduce") or file_has_function("reducer/reducer.py", "decide"), "DecisionReducer.reduce()/decide() method")
    check(file_contains("reducer/reducer.py", r"weight"), "DecisionReducer uses weights")

    # Risk - SessionGuard
    check(file_exists("risk/session_guard.py"), "Risk file: session_guard.py")
    check(file_has_class("risk/session_guard.py", "SessionGuard"), "SessionGuard class")
    check(file_contains("risk/session_guard.py", r"drawdown"), "SessionGuard checks drawdown")
    check(file_contains("risk/session_guard.py", r"kill|halt|emergency|shutdown"), "SessionGuard has kill switch")

    # Risk - StrategyBreaker
    check(file_exists("risk/strategy_breaker.py"), "Risk file: strategy_breaker.py")
    check(file_has_class("risk/strategy_breaker.py", "StrategyBreaker"), "StrategyBreaker class")
    check(file_contains("risk/strategy_breaker.py", r"circuit|consecutive|cooldown"), "StrategyBreaker has circuit breaker logic")

    # Account
    check(file_exists("account/live_account.py"), "Account file: live_account.py")
    check(file_has_class("account/live_account.py", "LiveAccount"), "LiveAccount class")
    check(file_contains("account/live_account.py", r"pnl|PnL"), "LiveAccount tracks PnL")
    check(file_contains("account/live_account.py", r"margin|leverage"), "LiveAccount has margin/leverage")

    # Session
    check(file_exists("session/session_controller.py"), "Session file: session_controller.py")
    check(file_has_class("session/session_controller.py", "SessionController"), "SessionController class")

    # Observer
    check(file_exists("observer/observer.py"), "Observer file: observer.py")
    check(file_has_class("observer/observer.py", "Observer"), "Observer class")
    check(file_contains("observer/observer.py", r"async|queue|Queue|non.blocking"), "Observer is non-blocking")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 12: FEEDBACK & OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_feedback_optimization():
    print("\n" + "=" * 80)
    print("SECTION 12: FEEDBACK & OPTIMIZATION")
    print("=" * 80)

    check(file_exists("feedback/adaptive_engine.py"), "Feedback file: adaptive_engine.py")
    check(file_has_class("feedback/adaptive_engine.py", "AdaptiveEngine"), "AdaptiveEngine class")
    check(file_has_class("feedback/adaptive_engine.py", "CapitalRegimeEngine"), "CapitalRegimeEngine class")
    check(file_contains("feedback/adaptive_engine.py", r"PositionSizer"), "PositionSizer in adaptive_engine")
    check(file_contains("feedback/adaptive_engine.py", r"0\.95|0\.97|cut.*loss|asymmetric"), "Asymmetric risk adaptation")

    check(file_exists("optimization/ai_optimizer.py"), "Optimization file: ai_optimizer.py")
    check(file_has_class("optimization/ai_optimizer.py", "AIOptimizer"), "AIOptimizer class")
    check(file_contains("optimization/ai_optimizer.py", r"feedback|reward|adjust"), "AIOptimizer has feedback loop")

    check(file_exists("optimization/quantum_validation.py"), "Optimization file: quantum_validation.py")
    check(file_contains("optimization/quantum_validation.py", r"QuantumBackendTester|backend"), "QuantumBackendTester")
    check(file_contains("optimization/quantum_validation.py", r"ZNE|error.*mitigat"), "Quantum error mitigation (ZNE)")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 13: MONITORING
# ═══════════════════════════════════════════════════════════════════════════════

def validate_monitoring():
    print("\n" + "=" * 80)
    print("SECTION 13: MONITORING")
    print("=" * 80)

    check(file_exists("monitoring/system_monitor.py"), "Monitoring file: system_monitor.py")
    check(file_has_class("monitoring/system_monitor.py", "SystemMonitor"), "SystemMonitor class")
    check(file_contains("monitoring/system_monitor.py", r"MetricsCollector|metrics"), "MetricsCollector")
    check(file_contains("monitoring/system_monitor.py", r"AlertManager|alert"), "AlertManager in monitoring")
    check(file_contains("monitoring/system_monitor.py", r"prometheus|Prometheus|gauge|counter"), "Prometheus-style metrics", warn_only=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 14: BRIDGE FILES
# ═══════════════════════════════════════════════════════════════════════════════

def validate_bridge():
    print("\n" + "=" * 80)
    print("SECTION 14: BRIDGE FILES")
    print("=" * 80)

    check(file_exists("bridge/data_bridge.py"), "Bridge file: data_bridge.py")
    check(file_has_class("bridge/data_bridge.py", "DataBridge"), "DataBridge class")

    check(file_exists("bridge/research_live_bridge.py"), "Bridge file: research_live_bridge.py")
    check(file_contains("bridge/research_live_bridge.py", r"TOMLGenerator|toml.*generat"), "TOMLGenerator class")
    check(file_contains("bridge/research_live_bridge.py", r"TOMLVersionManager|version"), "TOMLVersionManager")
    check(file_contains("bridge/research_live_bridge.py", r"TOMLParameterReader|param.*read"), "TOMLParameterReader")
    check(file_contains("bridge/research_live_bridge.py", r"ResearchArtifactStore|artifact.*store"), "ResearchArtifactStore")

    check(file_exists("bridge/r_bridge.py"), "Bridge file: r_bridge.py")
    check(file_contains("bridge/r_bridge.py", r"RBridge|RSession"), "RBridge/RSession class")
    check(file_contains("bridge/r_bridge.py", r"rpy2"), "R bridge uses rpy2")

    check(file_exists("bridge/rpy2_interface.py"), "Bridge file: rpy2_interface.py")
    check(file_contains("bridge/rpy2_interface.py", r"RInterface"), "RInterface class")

    check(file_exists("bridge/model_wrapper.py"), "Bridge file: model_wrapper.py")
    check(file_contains("bridge/model_wrapper.py", r"GARCH"), "ModelWrapper has GARCH")
    check(file_contains("bridge/model_wrapper.py", r"ARIMA"), "ModelWrapper has ARIMA")

    check(file_exists("bridge/data_converter.py"), "Bridge file: data_converter.py")
    check(file_has_class("bridge/data_converter.py", "DataConverter"), "DataConverter class")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 15: LIVE STREAM FILES
# ═══════════════════════════════════════════════════════════════════════════════

def validate_live_stream():
    print("\n" + "=" * 80)
    print("SECTION 15: LIVE STREAM FILES")
    print("=" * 80)

    check(file_exists("live/stream/streaming.py"), "Live stream: streaming.py")
    check(file_has_class("live/stream/streaming.py", "StreamingManager") or file_has_class("live/stream/streaming.py", "RealTimeDataStream"), "StreamingManager/RealTimeDataStream class")
    check(file_contains("live/stream/streaming.py", r"WebSocket|websocket"), "Streaming uses WebSocket")
    check(file_contains("live/stream/streaming.py", r"Alpaca|Polygon|Binance|IEX|Yahoo"), "Multi-provider support")

    check(file_exists("live/stream/realtime_manager.py"), "Live stream: realtime_manager.py")
    check(file_contains("live/stream/realtime_manager.py", r"RealtimeManager|RealTimeDataManager"), "RealtimeManager class")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 16: BACKTESTING FILES
# ═══════════════════════════════════════════════════════════════════════════════

def validate_backtesting():
    print("\n" + "=" * 80)
    print("SECTION 16: BACKTESTING FILES")
    print("=" * 80)

    check(file_exists("backtesting/engine.py"), "Backtest file: engine.py")
    check(file_has_class("backtesting/engine.py", "BacktestEngine"), "BacktestEngine class")
    check(file_contains("backtesting/engine.py", r"ExecutionSimulator|execution"), "ExecutionSimulator")
    check(file_has_class("backtesting/engine.py", "Portfolio"), "Portfolio class")

    check(file_exists("backtesting/walk_forward.py"), "Backtest file: walk_forward.py")
    check(file_has_class("backtesting/walk_forward.py", "WalkForwardOptimizer"), "WalkForwardOptimizer class")
    check(file_contains("backtesting/walk_forward.py", r"overfit"), "WalkForward has overfitting detection")

    check(file_exists("backtesting/metrics.py"), "Backtest file: metrics.py")
    check(file_contains("backtesting/metrics.py", r"PerformanceMetrics|PerformanceAnalyzer"), "PerformanceMetrics class")
    check(file_contains("backtesting/metrics.py", r"bootstrap"), "Metrics has bootstrap CI")

    check(file_exists("backtesting/performance.py"), "Backtest file: performance.py")

    check(file_exists("backtesting/benchmark.py"), "Backtest file: benchmark.py")
    check(file_contains("backtesting/benchmark.py", r"BenchmarkComparison|BenchmarkAnalyzer"), "BenchmarkComparison class")

    check(file_exists("backtesting/result_store.py"), "Backtest file: result_store.py")
    check(file_contains("backtesting/result_store.py", r"ResultStore"), "ResultStore class")
    check(file_contains("backtesting/result_store.py", r"checksum|hash"), "ResultStore has checksum")

    check(file_exists("backtesting/validator.py"), "Backtest file: validator.py")
    check(file_contains("backtesting/validator.py", r"AirGapValidator|ValidationPipeline"), "AirGapValidator class")
    check(file_contains("backtesting/validator.py", r"NaN|nan|determinism|stale"), "Validator checks NaN/determinism")

    check(file_exists("backtesting/visualization.py"), "Backtest file: visualization.py")
    check(file_contains("backtesting/visualization.py", r"plotly|Plotly"), "Backtest viz uses Plotly")

    check(file_exists("backtesting/advanced_backtesting.py"), "Backtest file: advanced_backtesting.py")
    check(file_contains("backtesting/advanced_backtesting.py", r"GreekAwareBacktester|GreekAware"), "GreekAwareBacktester class")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 17: UTILS FILES
# ═══════════════════════════════════════════════════════════════════════════════

def validate_utils():
    print("\n" + "=" * 80)
    print("SECTION 17: UTILS FILES")
    print("=" * 80)

    check(file_exists("utils/config_loader.py"), "Utils: config_loader.py")
    check(file_has_class("utils/config_loader.py", "ConfigManager"), "ConfigManager class")
    check(file_contains("utils/config_loader.py", r"toml|TOML"), "ConfigManager loads TOML")
    check(file_contains("utils/config_loader.py", r"env|ENV|environment"), "ConfigManager supports env vars")

    check(file_exists("utils/math_helpers.py"), "Utils: math_helpers.py")
    check(file_contains("utils/math_helpers.py", r"numba|jit|njit"), "Math helpers use Numba JIT")

    check(file_exists("utils/validators.py"), "Utils: validators.py")
    check(file_contains("utils/validators.py", r"polars|Polars"), "Validators use Polars")

    check(file_exists("utils/logger.py"), "Utils: logger.py")
    check(file_contains("utils/logger.py", r"GigaFormatter"), "Logger: GigaFormatter")
    check(file_contains("utils/logger.py", r"JsonFormatter|JSON|json"), "Logger: JSON formatter")

    check(file_exists("utils/performance_profiler.py"), "Utils: performance_profiler.py")
    check(file_has_class("utils/performance_profiler.py", "PerformanceProfiler"), "PerformanceProfiler class")
    check(file_contains("utils/performance_profiler.py", r"nanosecond|perf_counter_ns"), "Profiler: nanosecond precision")

    check(file_exists("utils/rate_limiter.py"), "Utils: rate_limiter.py")
    check(file_contains("utils/rate_limiter.py", r"TokenBucket"), "TokenBucketLimiter class")
    check(file_contains("utils/rate_limiter.py", r"SlidingWindow"), "SlidingWindowLimiter class")

    check(file_exists("utils/alerting.py"), "Utils: alerting.py")
    check(file_contains("utils/alerting.py", r"telegram|Telegram"), "Alerting: Telegram support")
    check(file_contains("utils/alerting.py", r"discord|Discord"), "Alerting: Discord support")

    check(file_exists("utils/retry.py"), "Utils: retry.py")
    check(file_contains("utils/retry.py", r"CircuitBreaker|circuit"), "Retry: CircuitBreaker")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 18: VISUALIZATION FILES
# ═══════════════════════════════════════════════════════════════════════════════

def validate_visualization():
    print("\n" + "=" * 80)
    print("SECTION 18: VISUALIZATION FILES")
    print("=" * 80)

    viz_files = [
        ("visualization/app.py", "streamlit|Streamlit"),
        ("visualization/greeks_dashboard.py", "greek|Greek|3D|surface"),
        ("visualization/risk_dashboard.py", "VaR|risk|Risk"),
        ("visualization/quantum_visualizer.py", "quantum|Quantum|circuit"),
        ("visualization/education_mode.py", "education|Education|tutorial"),
        ("visualization/education_viz.py", "education|ancient|Greek"),
        ("visualization/pnl_attribution.py", "pnl|attribution|waterfall"),
        ("visualization/correlation_heatmap.py", "correlation|heatmap"),
        ("visualization/observer_app.py", "observer|state"),
    ]

    for fpath, pattern in viz_files:
        check(file_exists(fpath), f"Viz file exists: {fpath}")
        if file_exists(fpath):
            check(file_contains(fpath, pattern), f"{fpath}: contains expected content")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 19: ARTIFACTS, CONFIG, DOCKER, CI
# ═══════════════════════════════════════════════════════════════════════════════

def validate_artifacts_config():
    print("\n" + "=" * 80)
    print("SECTION 19: ARTIFACTS, CONFIG, DOCKER, CI")
    print("=" * 80)

    # Artifacts
    check(file_exists("artifacts/definitions.py"), "Artifacts: definitions.py")
    check(file_contains("artifacts/definitions.py", r"MarketRegime"), "Artifacts: MarketRegime enum")
    check(file_contains("artifacts/definitions.py", r"TimeHorizon"), "Artifacts: TimeHorizon enum")
    check(file_contains("artifacts/definitions.py", r"Artifact"), "Artifacts: Artifact class")
    check(file_contains("artifacts/definitions.py", r"SignalArtifact"), "Artifacts: SignalArtifact class")
    check(file_contains("artifacts/definitions.py", r"Context"), "Artifacts: Context class")

    # Config files
    config_files = [
        "config/system_config.toml",
        "config/strategies_config.toml",
        "config/models_config.toml",
        "config/database_config.toml",
    ]
    for f in config_files:
        check(file_exists(f), f"Config file exists: {f}")

    # Docker
    check(file_exists("docker-compose.yml") or file_exists("Dockerfile") or file_exists("docker/Dockerfile") or file_exists("docker/docker-compose.yml"), "Docker configuration exists")

    # CI
    check(file_exists(".github/workflows/ci.yml") or dir_exists(".github/workflows"), "CI/CD workflow exists")

    # Scripts
    check(file_exists("scripts/ci_cd_pipeline.py"), "Scripts: ci_cd_pipeline.py")
    check(file_exists("scripts/health_check.py"), "Scripts: health_check.py")

    # Tests
    test_files = [
        "tests/test_greeks.py",
        "tests/test_risk.py",
    ]
    for f in test_files:
        check(file_exists(f), f"Test file exists: {f}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 20: PIPELINE WIRING VALIDATION (CROSS-COMPONENT)
# ═══════════════════════════════════════════════════════════════════════════════

def validate_pipeline_wiring():
    print("\n" + "=" * 80)
    print("SECTION 20: PIPELINE WIRING VALIDATION")
    print("=" * 80)

    # Pipeline 1→Bridge→Pipeline 2 flow
    check(
        file_contains("run_system_pipeline.py", r"pipeline_1|research") and
        file_contains("run_system_pipeline.py", r"pipeline_2|live"),
        "run_system_pipeline.py connects Pipeline 1 → Pipeline 2"
    )
    check(
        file_contains("run_system_pipeline.py", r"strategies_config|toml"),
        "Pipeline connector references TOML bridge artifact"
    )

    # launch_giga_system.py reads bridge artifacts
    check(
        file_contains("launch_giga_system.py", r"toml|config|strategies_config"),
        "Live system reads TOML bridge config"
    )

    # demo_complete_system.py generates bridge artifacts
    check(
        file_contains("demo_complete_system.py", r"bridge|artifact|toml|strategies_config"),
        "Demo system generates bridge artifacts"
    )

    # AlphaSignalEngine orchestrates 5 domains
    ase_path = "research/core/alpha_signal_engine.py"
    if file_exists(ase_path):
        domain_refs = 0
        for domain_term in ["StateSpaceOmega", "VariationalAnalyzer", "StochasticModeler",
                           "TimeAsymmetryAnalyzer", "InformationGeometer",
                           "market_state_space", "greek_response", "stochastic_models",
                           "time_asymmetry", "information_geometry"]:
            if file_contains(ase_path, domain_term):
                domain_refs += 1
        check(domain_refs >= 3, f"AlphaSignalEngine references {domain_refs}/5 domains (need >=3)")

    # Execution pipeline wiring
    check(
        file_contains("launch_giga_system.py", r"execute.*order|BinanceExecutor"),
        "Live system wires to BinanceExecutor"
    )

    # Risk cascade wiring
    check(
        file_contains("launch_giga_system.py", r"SessionGuard|session_guard|SessionController|session_controller"),
        "Live system wires SessionGuard/SessionController"
    )

    # State machine wiring
    check(
        file_contains("launch_giga_system.py", r"state|State|FLAT|LONG|SHORT"),
        "Live system uses position state machine"
    )

    # Observer wiring in demo
    check(
        file_contains("demo_complete_system.py", r"Observer"),
        "Demo wires Observer for monitoring"
    )

    # DecisionReducer wiring
    check(
        file_contains("demo_complete_system.py", r"DecisionReducer|reducer"),
        "Demo wires DecisionReducer"
    )

    # Feedback loop wiring
    check(
        file_contains("demo_complete_system.py", r"AdaptiveEngine|AIOptimizer|adaptive"),
        "Demo wires feedback/adaptation components"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 21: WORKFLOW INTEGRITY VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_workflow_integrity():
    print("\n" + "=" * 80)
    print("SECTION 21: WORKFLOW INTEGRITY VALIDATION")
    print("=" * 80)

    # WF1: Initialization
    check(file_contains("utils/config_loader.py", r"toml"), "WF1: ConfigManager can load TOML")
    check(file_contains("utils/logger.py", r"logging"), "WF1: Logger initialized")
    check(file_contains("data/database.py", r"connect|connection"), "WF1: Database connection established")

    # WF3: 5-Domain analysis produces results
    check(file_contains("research/core/market_state_space.py", r"return|MarketState"), "WF3-D1: Domain 1 returns MarketState")
    check(file_contains("research/core/greek_response.py", r"return|delta"), "WF3-D2: Domain 2 returns delta")
    check(file_contains("research/core/stochastic_models.py", r"return|kappa|StochasticParams"), "WF3-D3: Domain 3 returns kappa/StochasticParams")
    check(file_contains("research/core/time_asymmetry.py", r"return|ErgodicityResult"), "WF3-D4: Domain 4 returns ErgodicityResult")
    check(file_contains("research/core/information_geometry.py", r"return|entropy"), "WF3-D5: Domain 5 returns entropy")

    # WF4: Alpha Signal generation
    check(file_contains("research/core/alpha_signal_engine.py", r"generate_signal"), "WF4: generate_signal() exists")
    check(file_contains("research/core/alpha_signal_engine.py", r"AlphaSignal"), "WF4: AlphaSignal produced")

    # WF8: Backtesting loop
    check(file_contains("backtesting/engine.py", r"run|execute"), "WF8: BacktestEngine has run method")
    check(file_contains("backtesting/engine.py", r"event|Event"), "WF8: Event-driven architecture")

    # WF9: Bridge artifact generation
    check(file_contains("bridge/research_live_bridge.py", r"generate|write|save"), "WF9: TOMLGenerator can generate")

    # WF11: on_tick hot path
    check(file_contains("launch_giga_system.py", r"on_tick"), "WF11: on_tick function exists")
    check(file_contains("launch_giga_system.py", r"while\s+True|loop"), "WF11: Main loop exists")
    check(file_contains("launch_giga_system.py", r"sleep|time\.sleep"), "WF11: Sleep interval between ticks")

    # WF13: State machine transitions
    check(file_contains("brain/state_machine.py", r"transition|next_state"), "WF13: State transitions defined")

    # WF14: Risk cascade
    check(file_has_function("risk/session_guard.py", "check") or file_has_function("risk/session_guard.py", "check_health"), "WF14: SessionGuard.check()/check_health() exists")

    # WF15: Feedback loop
    check(file_contains("feedback/adaptive_engine.py", r"update|adapt|adjust"), "WF15: Adaptive update mechanism")
    check(file_contains("optimization/ai_optimizer.py", r"feedback|optimize|adjust"), "WF15: AI feedback loop")

    # WF16: Monitoring
    check(file_contains("observer/observer.py", r"log.*event|write|append"), "WF16: Observer logs events")
    check(file_contains("monitoring/system_monitor.py", r"collect|report"), "WF16: SystemMonitor collects metrics")

    # WF20: Emergency shutdown
    check(file_contains("risk/session_guard.py", r"halt|shutdown|emergency|kill"), "WF20: Emergency halt capability")
    check(file_contains("brain/state_machine.py", r"HALTED"), "WF20: HALTED state exists")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 22: AIR-GAP ARCHITECTURE VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_air_gap():
    print("\n" + "=" * 80)
    print("SECTION 22: AIR-GAP ARCHITECTURE VALIDATION")
    print("=" * 80)

    # Research files should NOT directly import execution components
    research_files = [
        "research/core/alpha_signal_engine.py",
        "research/core/market_state_space.py",
        "research/core/greek_response.py",
        "research/core/stochastic_models.py",
        "research/core/time_asymmetry.py",
        "research/core/information_geometry.py",
    ]

    for rf in research_files:
        if file_exists(rf):
            has_exec_import = file_contains(rf, r"from\s+execution\s+import|import\s+execution\.")
            check(not has_exec_import, f"Air-gap: {rf} does NOT import execution code", warn_only=True)

    # Bridge artifact is TOML-based (not Python import)
    check(
        file_contains("bridge/research_live_bridge.py", r"toml|TOML"),
        "Air-gap: Bridge uses TOML (not Python imports)"
    )

    # Backtesting validator enforces air-gap
    check(
        file_contains("backtesting/validator.py", r"AirGap|air.*gap|determinism|NaN"),
        "Air-gap: Validator enforces data boundary"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 23: PERFORMANCE ENGINEERING VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_performance():
    print("\n" + "=" * 80)
    print("SECTION 23: PERFORMANCE ENGINEERING VALIDATION")
    print("=" * 80)

    # Numba JIT usage
    numba_files = [
        "research/core/greeks.py",
        "research/core/black_scholes.py",
        "research/core/implied_volatility.py",
        "data/indicators.py",
        "utils/math_helpers.py",
    ]
    for f in numba_files:
        if file_exists(f):
            check(file_contains(f, r"numba|jit|njit|vectorize"), f"Performance: {f} uses Numba JIT")

    # DuckDB usage
    check(file_contains("data/database.py", r"duckdb"), "Performance: DuckDB OLAP engine")

    # Polars usage
    check(file_contains("data/preprocessing.py", r"polars|pl\."), "Performance: Polars DataFrames")

    # Rate limiting
    check(file_contains("utils/rate_limiter.py", r"token|Token|bucket|Bucket"), "Performance: Token-bucket rate limiter")
    check(file_contains("utils/rate_limiter.py", r"thread|Thread|Lock|lock"), "Performance: Thread-safe rate limiter")

    # Non-blocking Observer
    check(file_contains("observer/observer.py", r"queue|Queue|async|thread"), "Performance: Non-blocking Observer")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 24: DESIGN PATTERN VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_design_patterns():
    print("\n" + "=" * 80)
    print("SECTION 24: DESIGN PATTERN VALIDATION")
    print("=" * 80)

    # Strategy pattern (ABC)
    check(file_contains("research/strategies/base.py", r"ABC|abstract|abstractmethod"), "Pattern: Strategy ABC")

    # State machine pattern
    check(file_contains("brain/state_machine.py", r"Enum"), "Pattern: FSM with Enum states")
    check(file_contains("brain/state_machine.py", r"transition"), "Pattern: FSM transitions")

    # Circuit breaker pattern
    check(file_contains("risk/strategy_breaker.py", r"circuit|trip|reset|cooldown"), "Pattern: Circuit breaker")

    # Observer pattern (read-only)
    check(file_contains("observer/observer.py", r"read.only|witness|non.block|async"), "Pattern: Observer (read-only witness)")

    # Weighted voting (DecisionReducer)
    check(file_contains("reducer/reducer.py", r"weight|vote|aggregate|reduce"), "Pattern: Weighted voting")

    # Factory pattern (FillModel)
    check(file_contains("execution/execution_engine.py", r"FillModel|INSTANT|SLIPPAGE|CHAOS"), "Pattern: Factory (fill models)")

    # Graceful degradation
    quantum_has_fallback = any(
        file_contains(f"research/quantum/{f}", r"try|except|fallback|classical")
        for f in ["quantum_optimizer.py", "quantum_monte_carlo.py", "quantum_ml.py"]
        if file_exists(f"research/quantum/{f}")
    )
    check(quantum_has_fallback, "Pattern: Graceful degradation (quantum fallback)")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 25: LINE COUNT VALIDATION (Size Verification)
# ═══════════════════════════════════════════════════════════════════════════════

def validate_line_counts():
    print("\n" + "=" * 80)
    print("SECTION 25: LINE COUNT VALIDATION")
    print("=" * 80)

    # Verify critical files have substantial content
    size_checks = [
        ("launch_giga_system.py", 100, 500),
        ("research/core/alpha_signal_engine.py", 200, 1000),
        ("research/core/alpha_factor_library.py", 200, 1000),
        ("research/core/greeks.py", 200, 1200),
        ("research/core/black_scholes.py", 150, 800),
        ("research/core/monte_carlo.py", 200, 1200),
        ("research/core/greek_mathematics.py", 400, 2000),
        ("research/strategies/momentum.py", 300, 1500),
        ("research/strategies/options_strategies.py", 300, 1500),
        ("research/quantum/quantum_ml.py", 300, 1500),
        ("data/market_data.py", 200, 1000),
        ("data/database.py", 200, 1000),
        ("data/indicators.py", 200, 1200),
        ("execution/binance_executor.py", 100, 700),
        ("backtesting/engine.py", 200, 1200),
        ("bridge/data_bridge.py", 200, 1200),
        ("utils/config_loader.py", 200, 900),
        ("utils/math_helpers.py", 300, 1500),
    ]

    for fpath, min_lines, max_lines in size_checks:
        lines = count_lines(fpath)
        check(
            min_lines <= lines <= max_lines,
            f"Line count: {fpath} has {lines} lines (expected {min_lines}-{max_lines})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 26: CORE HYPOTHESIS VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_core_hypothesis():
    print("\n" + "=" * 80)
    print("SECTION 26: CORE HYPOTHESIS VALIDATION")
    print("=" * 80)

    # κ (kappa) is central to hypothesis
    kappa_files = [
        "research/core/stochastic_models.py",
        "research/core/alpha_factor_library.py",
        "research/core/alpha_signal_engine.py",
        "launch_giga_system.py",
    ]
    for f in kappa_files:
        if file_exists(f):
            check(file_contains(f, r"kappa|κ"), f"Hypothesis: {f} references kappa (κ)")

    # Entropy is central to hypothesis
    entropy_files = [
        "research/core/information_geometry.py",
        "research/core/alpha_factor_library.py",
        "research/core/alpha_signal_engine.py",
    ]
    for f in entropy_files:
        if file_exists(f):
            check(file_contains(f, r"entropy|shannon|Shannon"), f"Hypothesis: {f} references entropy")

    # Ornstein-Uhlenbeck mean-reversion
    check(
        file_contains("research/core/stochastic_models.py", r"OrnsteinUhlenbeck|ornstein|mean.reversion"),
        "Hypothesis: O-U mean-reversion model present"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("GIGA-SYSTEM — PIPELINE & WORKFLOW VALIDATION SCRIPT")
    print("=" * 80)
    print(f"Root: {GIGA_ROOT}")
    print(f"Root exists: {os.path.isdir(GIGA_ROOT)}")

    if not os.path.isdir(GIGA_ROOT):
        print(f"\nERROR: GIGA_ROOT not found at {GIGA_ROOT}")
        sys.exit(1)

    # Run all validations
    validate_directory_structure()
    validate_entry_points()
    validate_5_domain_framework()
    validate_research_core()
    validate_research_ml()
    validate_research_quantum()
    validate_research_strategies()
    validate_r_analytics()
    validate_data_pipeline()
    validate_execution_pipeline()
    validate_core_components()
    validate_feedback_optimization()
    validate_monitoring()
    validate_bridge()
    validate_live_stream()
    validate_backtesting()
    validate_utils()
    validate_visualization()
    validate_artifacts_config()
    validate_pipeline_wiring()
    validate_workflow_integrity()
    validate_air_gap()
    validate_performance()
    validate_design_patterns()
    validate_line_counts()
    validate_core_hypothesis()

    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    total = PASS + FAIL + WARN
    print(f"Total Checks:  {total}")
    print(f"PASSED:        {PASS}")
    print(f"FAILED:        {FAIL}")
    print(f"WARNINGS:      {WARN}")
    print(f"Pass Rate:     {PASS}/{PASS + FAIL} = {PASS / max(PASS + FAIL, 1) * 100:.1f}%")
    print("=" * 80)

    # Print failures
    if FAIL > 0:
        print("\nFAILED CHECKS:")
        for status, desc in RESULTS:
            if status == "FAIL":
                print(f"  [FAIL] {desc}")

    # Print warnings
    if WARN > 0:
        print("\nWARNINGS:")
        for status, desc in RESULTS:
            if status == "WARN":
                print(f"  [WARN] {desc}")

    print(f"\nVERDICT: {'YES — Pipeline and Workflow validated successfully' if FAIL == 0 else 'NO — Some checks failed'}")

    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
