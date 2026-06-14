#!/usr/bin/env python3
"""
GIGA SYSTEM — Full System Health Check
Verifies all components, runs quick tests, reports status.
"""

import sys
import os
import importlib
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_import(module_path: str) -> bool:
    """Try to import a module, return success."""
    try:
        importlib.import_module(module_path)
        return True
    except Exception as e:
        print(f"  FAIL: {module_path} → {e}")
        return False


def main():
    print("=" * 60)
    print("  GIGA SYSTEM — Health Check")
    print("=" * 60)

    modules = {
        "Research Core": [
            "research.core.greeks",
            "research.core.black_scholes",
            "research.core.monte_carlo",
            "research.core.implied_volatility",
            "research.core.risk_metrics",
            "research.core.binomial_tree",
            "research.core.greek_response",
            "research.core.market_state_space",
            "research.core.stochastic_models",
            "research.core.time_asymmetry",
            "research.core.information_geometry",
            "research.core.alpha_signal_engine",
            "research.core.greek_mathematics",
            "research.core.cross_sectional_alpha",
            "research.core.greek_walk_forward",
            "research.core.microstructure_alpha",
            "research.core.options_data_feed",
            "research.core.domain_data_connector",
            "research.core.volatility_surface",
            "research.core.greeks_hedging",
            "research.core.alpha_factor_library",
        ],
        "Research ML": [
            "research.ml.feature_engineering",
            "research.ml.regime_detection",
            "research.ml.volatility_forecast",
        ],
        "Research Strategies": [
            "research.strategies.base",
            "research.strategies.momentum",
            "research.strategies.market_making",
            "research.strategies.options_strategies",
            "research.strategies.pairs_trading",
        ],
        "Brain & Reducer": [
            "brain.state_machine",
            "reducer.reducer",
        ],
        "Execution": [
            "execution.execution_engine",
            "execution.order_manager",
            "execution.smart_router",
            "execution.order_router",
            "execution.latency_monitor",
        ],
        "Risk & Account": [
            "risk.session_guard",
            "risk.strategy_breaker",
            "account.live_account",
        ],
        "Backtesting": [
            "backtesting.engine",
            "backtesting.walk_forward",
            "backtesting.metrics",
            "backtesting.result_store",
            "backtesting.advanced_backtesting",
        ],
        "Feedback & Observer": [
            "feedback.adaptive_engine",
            "observer.observer",
        ],
        "Bridge": [
            "bridge.research_live_bridge",
        ],
        "Optimization": [
            "optimization.quantum_validation",
        ],
        "Monitoring": [
            "monitoring.system_monitor",
        ],
        "Data Layer": [
            "data.database_layer",
            "data.multi_exchange",
        ],
        "CI/CD": [
            "scripts.ci_cd_pipeline",
        ],
        "Visualization & Education": [
            "visualization.education_viz",
        ],
    }

    total = 0
    passed = 0

    for category, mods in modules.items():
        print(f"\n[{category}]")
        for mod in mods:
            total += 1
            if check_import(mod):
                print(f"  OK: {mod}")
                passed += 1

    print("\n" + "=" * 60)
    print(f"  Result: {passed}/{total} modules loaded successfully")
    print("=" * 60)

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
