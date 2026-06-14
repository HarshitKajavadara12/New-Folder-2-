"""
PHASE 12: GREEK RESEARCH LAB
=============================
Demonstration of the 5-Domain Greek Alpha Framework.
Generates an Alpha Artifact from pure mathematical analysis.

Usage:
    python run_greek_research_lab.py
"""

import numpy as np
import pandas as pd
import sys
import os
import logging

logger = logging.getLogger(__name__)

# Ensure import paths worked
sys.path.insert(0, os.getcwd())

from research.core.market_state_space import StateSpaceOmega, MarketState
from research.core.greek_response import VariationalAnalyzer
from research.core.stochastic_models import StochasticModeler
from research.core.time_asymmetry import TimeAsymmetryAnalyzer
from research.core.information_geometry import InformationGeometer
from research.core.alpha_signal_engine import AlphaSignalEngine

def generate_synthetic_data(n=1000):
    """Generate fake market data for research demo (fallback when no real data)."""
    np.random.seed(42)
    
    # Simulate a regime-switching process
    returns = np.random.normal(0, 0.01, n) 
    # Add a shock
    returns[500:550] *= 5 
    # Add a trend
    returns[800:] += 0.002
    
    price = 100 * (1 + returns).cumprod()
    volume = np.random.lognormal(10, 1, n)
    
    return pd.Series(price), pd.Series(volume)


def load_real_data():
    """Attempt to load real market data via yfinance or from stored files."""
    # Try loading from stored data first
    stored_paths = [
        "data_samples/btc_daily.csv",
        "data_samples/market_data.csv",
    ]
    for path in stored_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, parse_dates=True, index_col=0)
                if 'close' in df.columns and 'volume' in df.columns:
                    logger.info(f"Loaded real data from {path}")
                    return pd.Series(df['close'].values), pd.Series(df['volume'].values)
                elif 'Close' in df.columns and 'Volume' in df.columns:
                    logger.info(f"Loaded real data from {path}")
                    return pd.Series(df['Close'].values), pd.Series(df['Volume'].values)
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")
    
    # Try yfinance
    try:
        import yfinance as yf
        ticker = yf.Ticker("BTC-USD")
        hist = ticker.history(period="2y")
        if len(hist) > 100:
            logger.info("Loaded real BTC-USD data from yfinance")
            return pd.Series(hist['Close'].values), pd.Series(hist['Volume'].values)
    except ImportError:
        logger.info("yfinance not installed, using synthetic data")
    except Exception as e:
        logger.warning(f"yfinance fetch failed: {e}")
    
    return None, None

def run_lab():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')
    logger.info("="*60)
    logger.info("  GREEK RESEARCH LAB (PHASE 12)")
    logger.info("  Analysis of Structural Alpha")
    logger.info("="*60)

    # 0. Load Data — prefer real data, fall back to synthetic
    price, volume = load_real_data()
    data_source = "Real Market Data"
    if price is None:
        price, volume = generate_synthetic_data()
        data_source = "Synthetic (fallback)"
    
    returns = price.pct_change().dropna()
    logger.info(f"[DATA] Loaded {len(price)} candles ({data_source})")

    # 1. State Space Analysis (Ω, Λ)
    logger.info("[DOMAIN 1] State Space Topology (Ω)")
    omega = StateSpaceOmega()
    # Feed history
    for i in range(50, len(price), 50):
        window_ret = returns[i-50:i]
        window_vol = volume[i-50:i]
        state = omega.classify_state(window_ret, window_vol)
        omega.record_observation(state)
        
    logger.info(f"   Observed States: {len(omega.states)}")
    transitions = omega.get_lambda_matrix()
    logger.info(f"   Transition Matrix Keys: {list(transitions.keys())}")

    # 2. Variational Analysis (Δ, Γ, Θ)
    logger.info("[DOMAIN 2] Variational Sensitivity (Δ, Γ)")
    va = VariationalAnalyzer()
    # Mock PnL (Momentum strategy: Long if ret > 0)
    pnl = (returns.shift(-1) * np.sign(returns)).cumsum().fillna(0)
    
    sensitivity = va.analyze_convexity(pnl, price.iloc[1:])
    logger.info(f"   Delta (Directionality): {sensitivity.delta:.4f}")
    logger.info(f"   Gamma (Convexity):      {sensitivity.gamma:.4f}")
    logger.info(f"   Theta (Decay):          {sensitivity.theta:.4f}")

    # 3. Stochastic Modeling (μ, σ, κ)
    logger.info("[DOMAIN 3] Stochastic Parameters (μ, σ, κ)")
    sm = StochasticModeler()
    params = sm.fit_ornstein_uhlenbeck(price.iloc[-100:])
    logger.info(f"   Drift (μ):      {params.mu:.4f}")
    logger.info(f"   Volatility (σ): {params.sigma:.4f}")
    logger.info(f"   Mean Rev (κ):   {params.kappa:.4f}")

    # 4. Ergodicity (τ, Time Asymmetry)
    logger.info("[DOMAIN 4] Time Asymmetry (Ergodicity GAP)")
    ta = TimeAsymmetryAnalyzer()
    erg = ta.check_ergodicity(returns)
    logger.info(f"   Ensemble Avg: {erg.ensemble_average:.6f}")
    logger.info(f"   Time Avg:     {erg.time_average:.6f}")
    logger.info(f"   Ergodic?      {erg.is_ergodic}")
    logger.info(f"   Kelly Frac:   {erg.kelly_fraction:.2f}")

    # 5. Information (Entropy)
    logger.info("[DOMAIN 5] Information Geometry (Η)")
    ig = InformationGeometer()
    entropy_val = ig.calculate_market_entropy(price)
    logger.info(f"   Market Entropy (Η): {entropy_val:.4f} bits")

    # =====================================================================
    # 6. ALPHA SIGNAL ENGINE — The missing link
    # =====================================================================
    logger.info("="*60)
    logger.info("  ALPHA SIGNAL ENGINE — Wiring 5 Domains to Trading")
    logger.info("="*60)
    
    alpha_engine = AlphaSignalEngine()
    signal = alpha_engine.generate_signal(
        prices=price,
        volumes=volume,
        pnl_series=pnl,
    )
    
    logger.info(f"   Direction:  {signal.direction}")
    logger.info(f"   Confidence: {signal.confidence:.4f}")
    logger.info(f"   κ={signal.kappa:.4f}, H={signal.entropy:.4f}")
    logger.info(f"   Kelly:      {signal.kelly_fraction:.4f}")
    logger.info(f"   p-value:    {signal.p_value:.6f}")
    logger.info(f"   IR:         {signal.information_ratio:.4f}")
    logger.info(f"   Decay Rate: {signal.alpha_decay_rate:.4f}")
    logger.info(f"   Factors:    {signal.factors}")
    logger.info(f"   Reason:     {signal.reason}")

    # Store results as artifact
    report = alpha_engine.get_alpha_report()
    
    # Save backtest results artifact
    results_path = "artifacts/alpha_analysis_results.json"
    try:
        import json
        os.makedirs("artifacts", exist_ok=True)
        with open(results_path, "w") as f:
            # Convert non-serializable types
            serializable_report = {}
            for k, v in report.items():
                if isinstance(v, (float, int, str, bool, dict, list)):
                    serializable_report[k] = v
                else:
                    serializable_report[k] = str(v)
            json.dump(serializable_report, f, indent=2, default=str)
        logger.info(f"   Saved alpha analysis to {results_path}")
    except Exception as e:
        logger.warning(f"   Failed to save results: {e}")

    logger.info("="*60)
    logger.info("  HYPOTHESIS ARTIFACT GENERATED")
    logger.info("  Hypothesis: 'High κ regimes with Low Entropy offer maximal Alpha'")
    logger.info(f"  Signal: {signal.direction} (conf={signal.confidence:.2f})")
    logger.info("  Action: Submit to Reducer via Alpha Signal Engine")
    logger.info("="*60)
    
    return signal, report

if __name__ == "__main__":
    run_lab()
