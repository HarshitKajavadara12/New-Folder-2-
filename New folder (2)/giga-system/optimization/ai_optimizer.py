"""
GIGA SYSTEM - AI Optimizer
Adaptive parameter tuning and ML model retraining pipeline.

Responsibilities:
1. Compute risk-adjusted reward for each trade cycle
2. Adaptive parameter tuning via feedback loop
3. Trigger ML model retraining when performance degrades
4. Track optimization history for analysis
"""

import logging
import time
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

from feedback.adaptive_engine import AdaptiveEngine
import research.ml.feature_engineering as feature_eng
import research.ml.volatility_forecast as vol_forecast

logger = logging.getLogger(__name__)

# Attempt optional regime detection import
try:
    import research.ml.regime_detection as regime_detect
    HAS_REGIME = True
except (ImportError, ModuleNotFoundError):
    HAS_REGIME = False


class AIOptimizer:
    """
    AI-driven optimizer that closes the feedback loop:
    Signal -> Trade -> P&L -> Reward -> Adjust Parameters -> Better Signals
    """
    
    def __init__(self, adaptive_engine: AdaptiveEngine, config: Dict[str, Any]):
        self.adaptive_engine = adaptive_engine
        self.config = config
        
        # Reward tracking
        self.reward_history: List[float] = []
        self.retrain_count = 0
        self.last_retrain_time = 0.0
        self.min_retrain_interval = config.get("min_retrain_interval_sec", 300)
        
        # Performance tracking
        self.optimization_history: List[Dict] = []
    
    def compute_reward(self, pnl: float, risk_metric: float) -> float:
        """Compute risk-adjusted reward."""
        risk_penalty = self.config.get("risk_penalty", 0.1)
        reward = pnl - risk_penalty * risk_metric
        self.reward_history.append(reward)
        return reward

    def optimize_signal(self, signal, pnl, confidence, strategy_params, risk_metric,
                       market_data=None):
        """
        Main optimization loop: evaluate performance and adapt.
        
        Returns:
            (updated_params, reward, params_changed)
        """
        # Step 1: Evaluate feedback
        feed_metric = self.adaptive_engine.evaluate_signal(signal, pnl, confidence)

        # Step 2: Compute reward
        reward = self.compute_reward(pnl, risk_metric)

        # Step 3: Adaptive parameter tuning
        updated_params, changed = self.adaptive_engine.adjust_parameters(strategy_params)

        # Step 4: Conditional retrain trigger with cooldown
        reward_threshold = self.config.get("reward_threshold", -200.0)
        now = time.time()
        
        if reward < reward_threshold and (now - self.last_retrain_time) > self.min_retrain_interval:
            logger.warning(
                f"[AI OPTIMIZER] Reward {reward:.2f} < threshold {reward_threshold}. "
                f"Triggering ML retraining..."
            )
            self.retrain_models(market_data)
        
        # Step 5: Record
        self.optimization_history.append({
            'timestamp': now, 'reward': reward, 'pnl': pnl,
            'risk_metric': risk_metric, 'params_changed': changed,
        })
        
        return updated_params, reward, changed

    def retrain_models(self, market_data=None):
        """Retrain all ML models. Respects cooldown."""
        logger.info("[AI OPTIMIZER] Starting retraining sequence...")
        self.retrain_count += 1
        self.last_retrain_time = time.time()
        
        try:
            feature_eng.update_features(market_data)
        except Exception as e:
            logger.error(f"[AI OPTIMIZER] Feature engineering failed: {e}")
        
        if HAS_REGIME:
            try:
                regime_detect.retrain(market_data)
            except Exception as e:
                logger.error(f"[AI OPTIMIZER] Regime detection failed: {e}")
        
        try:
            vol_forecast.retrain(market_data)
        except Exception as e:
            logger.error(f"[AI OPTIMIZER] Volatility forecast failed: {e}")
        
        logger.info(f"[AI OPTIMIZER] Retraining complete (total: {self.retrain_count})")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Return optimizer performance metrics."""
        if not self.reward_history:
            return {'status': 'no_data'}
        rewards = np.array(self.reward_history)
        return {
            'total_optimizations': len(self.optimization_history),
            'total_retrains': self.retrain_count,
            'mean_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'positive_pct': float(np.mean(rewards > 0)),
        }
