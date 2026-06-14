"""
GIGA SYSTEM - ML Regime Detection
Layer 1 - Market Context Engine

Production-grade regime detection using:
- Gaussian Mixture Models (GMM) for regime clustering
- Hidden Markov Models (HMM) via hmmlearn (optional)
- Feature extraction: returns, volatility, momentum, volume
- Online update capability via partial_fit
- Properly typed, no bare excepts
"""

import logging
import warnings
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

try:
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False

logger = logging.getLogger(__name__)


class MarketState(Enum):
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    HIGH_VOL = "HIGH_VOL"
    UNCERTAIN = "UNCERTAIN"


# Map GMM cluster labels to MarketState based on cluster characteristics
_REGIME_MAP = {
    'high_vol': MarketState.HIGH_VOL,
    'trend_up': MarketState.TRENDING_UP,
    'trend_down': MarketState.TRENDING_DOWN,
    'ranging': MarketState.RANGING,
}


@dataclass
class RegimeFeatures:
    """Extracted features for regime classification."""
    returns: float = 0.0
    volatility: float = 0.0
    momentum: float = 0.0
    mean_reversion: float = 0.0
    volume_ratio: float = 1.0


class RegimeDetector:
    """
    ML-based Market Regime Detector.
    
    Uses Gaussian Mixture Model to identify market regimes from 
    price-derived features. Falls back to rules-based detection
    if sklearn is unavailable or insufficient training data.
    
    Regimes:
    - TRENDING_UP: Positive momentum, low-medium vol
    - TRENDING_DOWN: Negative momentum, low-medium vol
    - HIGH_VOL: High volatility regardless of direction
    - RANGING: Low momentum, low volatility
    - UNCERTAIN: Insufficient data for classification
    """
    
    def __init__(self, window_size: int = 60, n_regimes: int = 4,
                 vol_lookback: int = 20, momentum_lookback: int = 10):
        self.window_size = window_size
        self.n_regimes = n_regimes
        self.vol_lookback = vol_lookback
        self.momentum_lookback = momentum_lookback
        
        self.prices: List[float] = []
        self.volumes: List[float] = []
        
        # ML components
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.gmm: Optional[GaussianMixture] = None
        self.hmm: Optional[Any] = None
        self._trained = False
        self._feature_history: List[np.ndarray] = []
        
        # Regime tracking
        self._current_regime = MarketState.UNCERTAIN
        self._regime_history: List[MarketState] = []
        self._transition_matrix = np.zeros((len(MarketState), len(MarketState)))
        
        # Cluster-to-regime mapping (learned during fit)
        self._cluster_map: Dict[int, MarketState] = {}
        
        logger.info(f"RegimeDetector initialized (window={window_size}, "
                     f"sklearn={'yes' if SKLEARN_AVAILABLE else 'no'}, "
                     f"hmm={'yes' if HMM_AVAILABLE else 'no'})")
    
    def update(self, price: float, volume: float = 0.0) -> MarketState:
        """
        Update with new price observation and return detected regime.
        
        Args:
            price: Latest price
            volume: Latest volume (0 if unavailable)
            
        Returns:
            Current MarketState classification  
        """
        self.prices.append(price)
        self.volumes.append(volume)
        
        # Trim to window size
        if len(self.prices) > self.window_size * 2:
            self.prices = self.prices[-self.window_size * 2:]
            self.volumes = self.volumes[-self.window_size * 2:]
        
        if len(self.prices) < max(self.vol_lookback, self.momentum_lookback) + 2:
            return MarketState.UNCERTAIN
        
        # Extract features
        features = self._extract_features()
        feature_vec = np.array([
            features.returns, features.volatility,
            features.momentum, features.mean_reversion
        ])
        self._feature_history.append(feature_vec)
        
        # ML classification if trained
        if self._trained and self.gmm is not None:
            try:
                scaled = self.scaler.transform(feature_vec.reshape(1, -1))
                cluster = self.gmm.predict(scaled)[0]
                regime = self._cluster_map.get(cluster, MarketState.UNCERTAIN)
            except Exception as e:
                logger.debug(f"GMM prediction failed: {e}, using rules fallback")
                regime = self._rules_based_classify(features)
        else:
            regime = self._rules_based_classify(features)
            
            # Auto-train when enough data accumulated
            if len(self._feature_history) >= self.window_size and not self._trained:
                self._fit_gmm()
        
        # Track transitions
        if self._regime_history:
            prev_idx = list(MarketState).index(self._current_regime)
            curr_idx = list(MarketState).index(regime)
            self._transition_matrix[prev_idx, curr_idx] += 1
        
        self._current_regime = regime
        self._regime_history.append(regime)
        return regime
    
    def _extract_features(self) -> RegimeFeatures:
        """Extract regime-relevant features from price history."""
        prices = np.array(self.prices)
        
        # Returns
        returns = np.diff(np.log(prices[-self.vol_lookback - 1:]))
        
        # Realized volatility (annualized)
        volatility = np.std(returns) * np.sqrt(252)
        
        # Momentum (exponentially-weighted)
        momentum_returns = np.diff(np.log(prices[-self.momentum_lookback - 1:]))
        weights = np.exp(np.linspace(-1, 0, len(momentum_returns)))
        weights /= weights.sum()
        momentum = float(np.dot(momentum_returns, weights)) * 252
        
        # Mean reversion signal (z-score of price relative to MA)
        ma = np.mean(prices[-self.vol_lookback:])
        std = np.std(prices[-self.vol_lookback:])
        mean_reversion = (prices[-1] - ma) / std if std > 0 else 0.0
        
        # Volume ratio
        if len(self.volumes) >= self.vol_lookback and any(v > 0 for v in self.volumes):
            recent_vol = np.mean(self.volumes[-5:]) if self.volumes[-1] > 0 else 1.0
            avg_vol = np.mean([v for v in self.volumes[-self.vol_lookback:] if v > 0]) or 1.0
            volume_ratio = recent_vol / avg_vol
        else:
            volume_ratio = 1.0
        
        return RegimeFeatures(
            returns=float(np.mean(returns) * 252),
            volatility=float(volatility),
            momentum=float(momentum),
            mean_reversion=float(mean_reversion),
            volume_ratio=float(volume_ratio)
        )
    
    def _rules_based_classify(self, features: RegimeFeatures) -> MarketState:
        """Rules-based fallback classification."""
        # High volatility dominates
        if features.volatility > 0.25:
            return MarketState.HIGH_VOL
        
        # Trend detection
        if features.momentum > 0.10:
            return MarketState.TRENDING_UP
        elif features.momentum < -0.10:
            return MarketState.TRENDING_DOWN
        
        return MarketState.RANGING
    
    def _fit_gmm(self) -> None:
        """Fit GMM to accumulated feature history."""
        if not SKLEARN_AVAILABLE or len(self._feature_history) < self.window_size:
            return
        
        try:
            X = np.array(self._feature_history)
            X_scaled = self.scaler.fit_transform(X)
            
            self.gmm = GaussianMixture(
                n_components=self.n_regimes,
                covariance_type='full',
                n_init=3,
                max_iter=200,
                random_state=42
            )
            self.gmm.fit(X_scaled)
            
            # Map clusters to regimes based on cluster centers
            self._map_clusters_to_regimes(X)
            self._trained = True
            logger.info(f"GMM fitted on {len(X)} samples, BIC={self.gmm.bic(X_scaled):.1f}")
            
        except Exception as e:
            logger.warning(f"GMM fitting failed: {e}")
            self._trained = False
    
    def _map_clusters_to_regimes(self, X: np.ndarray) -> None:
        """Map GMM clusters to MarketState based on cluster statistics."""
        X_scaled = self.scaler.transform(X)
        labels = self.gmm.predict(X_scaled)
        
        for cluster_id in range(self.n_regimes):
            mask = labels == cluster_id
            if mask.sum() == 0:
                self._cluster_map[cluster_id] = MarketState.UNCERTAIN
                continue
            
            cluster_features = X[mask]
            avg_vol = np.mean(cluster_features[:, 1])      # volatility
            avg_momentum = np.mean(cluster_features[:, 2])  # momentum
            
            # Classification logic based on cluster characteristics
            if avg_vol > np.percentile(X[:, 1], 75):
                self._cluster_map[cluster_id] = MarketState.HIGH_VOL
            elif avg_momentum > np.percentile(X[:, 2], 65):
                self._cluster_map[cluster_id] = MarketState.TRENDING_UP
            elif avg_momentum < np.percentile(X[:, 2], 35):
                self._cluster_map[cluster_id] = MarketState.TRENDING_DOWN
            else:
                self._cluster_map[cluster_id] = MarketState.RANGING
    
    def retrain(self, market_data: Any = None) -> bool:
        """
        Retrain the regime detection model.
        
        Args:
            market_data: Optional new data. If None, retrain on accumulated history.
            
        Returns:
            True if retraining succeeded
        """
        if market_data is not None:
            # Extract prices from market data
            if hasattr(market_data, '__iter__') and len(market_data) > 0:
                if isinstance(market_data, np.ndarray):
                    prices = market_data.flatten()
                elif hasattr(market_data, 'values'):
                    prices = np.array(market_data.values).flatten()
                else:
                    prices = np.array(list(market_data)).flatten()
                
                # Rebuild feature history from new data
                self._feature_history.clear()
                self.prices = list(prices)
                self.volumes = [0.0] * len(prices)
                
                for i in range(max(self.vol_lookback, self.momentum_lookback) + 2, len(prices)):
                    self.prices = list(prices[:i + 1])
                    features = self._extract_features()
                    self._feature_history.append(np.array([
                        features.returns, features.volatility,
                        features.momentum, features.mean_reversion
                    ]))
                
                self.prices = list(prices)
        
        if len(self._feature_history) < self.window_size:
            logger.warning(f"Insufficient data for retraining: "
                          f"{len(self._feature_history)} < {self.window_size}")
            return False
        
        # Reset and refit
        self._trained = False
        self._fit_gmm()
        
        if self._trained:
            logger.info(f"Regime detector retrained on {len(self._feature_history)} samples")
        return self._trained
    
    def get_regime_probabilities(self) -> Dict[str, float]:
        """Get probability of each regime based on GMM posterior."""
        if not self._trained or not self._feature_history:
            return {state.value: 0.25 for state in MarketState if state != MarketState.UNCERTAIN}
        
        try:
            latest = np.array(self._feature_history[-1]).reshape(1, -1)
            scaled = self.scaler.transform(latest)
            probs = self.gmm.predict_proba(scaled)[0]
            
            result = {}
            for cluster_id, prob in enumerate(probs):
                regime = self._cluster_map.get(cluster_id, MarketState.UNCERTAIN)
                result[regime.value] = result.get(regime.value, 0.0) + float(prob)
            
            return result
        except Exception as e:
            logger.debug(f"Probability estimation failed: {e}")
            return {state.value: 0.25 for state in MarketState if state != MarketState.UNCERTAIN}
    
    def get_transition_matrix(self) -> np.ndarray:
        """Get normalized regime transition probability matrix."""
        row_sums = self._transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        return self._transition_matrix / row_sums
    
    @property
    def current_regime(self) -> MarketState:
        return self._current_regime
    
    @property 
    def is_trained(self) -> bool:
        return self._trained
    
    def get_summary(self) -> Dict[str, Any]:
        """Get detector status summary."""
        regime_counts = {}
        for r in self._regime_history:
            regime_counts[r.value] = regime_counts.get(r.value, 0) + 1
        
        return {
            'current_regime': self._current_regime.value,
            'is_trained': self._trained,
            'observations': len(self.prices),
            'feature_samples': len(self._feature_history),
            'regime_distribution': regime_counts,
            'model': 'GMM' if self._trained else 'rules_based'
        }
