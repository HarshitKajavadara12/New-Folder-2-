"""
GIGA SYSTEM - Feature Engineering Module
Transforms raw market data into ML-ready features.

Feature Categories:
1. Price-based: Returns, log-returns, momentum, mean-reversion
2. Volatility: Realized vol, EWMA vol, Parkinson, Garman-Klass
3. Microstructure: Spread, order imbalance, VPIN
4. Technical: RSI, MACD cross, Bollinger %B, ATR
5. Cross-asset: Correlation regime, sector momentum
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class FeatureEngine:
    """
    Production feature engineering pipeline.
    Converts raw OHLCV + market data into ML-ready feature matrices.
    """
    
    def __init__(self, lookback: int = 60, vol_window: int = 20):
        self.lookback = lookback
        self.vol_window = vol_window
        self.feature_names: List[str] = []
        self._fitted = False
    
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all features from OHLCV DataFrame.
        
        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
            
        Returns:
            DataFrame with computed features (NaN rows dropped)
        """
        if df is None or len(df) < self.lookback:
            logger.warning(f"Insufficient data: {len(df) if df is not None else 0} rows < {self.lookback}")
            return pd.DataFrame()
        
        features = pd.DataFrame(index=df.index)
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values if 'volume' in df.columns else np.ones(len(close))
        
        # === PRICE-BASED FEATURES ===
        features['returns_1'] = df['close'].pct_change(1)
        features['returns_5'] = df['close'].pct_change(5)
        features['returns_10'] = df['close'].pct_change(10)
        features['returns_20'] = df['close'].pct_change(20)
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Momentum
        features['momentum_10'] = df['close'] / df['close'].shift(10) - 1.0
        features['momentum_20'] = df['close'] / df['close'].shift(20) - 1.0
        
        # Mean reversion signal
        sma_20 = df['close'].rolling(20).mean()
        features['mean_reversion'] = (df['close'] - sma_20) / sma_20
        
        # === VOLATILITY FEATURES ===
        features['realized_vol_10'] = features['log_returns'].rolling(10).std() * np.sqrt(252)
        features['realized_vol_20'] = features['log_returns'].rolling(20).std() * np.sqrt(252)
        
        # EWMA volatility
        features['ewma_vol'] = features['log_returns'].ewm(span=self.vol_window).std() * np.sqrt(252)
        
        # Parkinson volatility (uses high-low range)
        hl_ratio = np.log(df['high'] / df['low'])
        features['parkinson_vol'] = np.sqrt(
            (1.0 / (4.0 * np.log(2))) * (hl_ratio ** 2).rolling(self.vol_window).mean() * 252
        )
        
        # Garman-Klass volatility
        features['garman_klass_vol'] = self._garman_klass(df, self.vol_window)
        
        # Vol-of-vol (volatility clustering)
        features['vol_of_vol'] = features['realized_vol_20'].rolling(20).std()
        
        # === TECHNICAL FEATURES ===
        features['rsi_14'] = self._rsi(close, 14)
        features['rsi_7'] = self._rsi(close, 7)
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9).mean()
        features['macd_histogram'] = macd_line - signal_line
        features['macd_cross'] = np.sign(features['macd_histogram']) - np.sign(features['macd_histogram'].shift(1))
        
        # Bollinger Bands %B
        sma = df['close'].rolling(20).mean()
        std = df['close'].rolling(20).std()
        features['bb_pct_b'] = (df['close'] - (sma - 2 * std)) / (4 * std)
        features['bb_width'] = (4 * std) / sma  # Normalized bandwidth
        
        # ATR (Average True Range)
        features['atr_14'] = self._atr(df, 14)
        features['atr_ratio'] = features['atr_14'] / df['close']  # Normalized ATR
        
        # === VOLUME FEATURES ===
        features['volume_sma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        features['volume_change'] = df['volume'].pct_change()
        
        # On-Balance Volume trend
        obv = (np.sign(features['returns_1'].fillna(0)) * df['volume']).cumsum()
        features['obv_slope'] = obv.rolling(10).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else 0, raw=False
        )
        
        # === REGIME FEATURES ===
        # Trend strength (ADX proxy)
        features['trend_strength'] = abs(features['momentum_20']) / features['realized_vol_20'].clip(lower=1e-6)
        
        # High-low range as % of price
        features['hl_range_pct'] = (df['high'] - df['low']) / df['close']
        
        # === CROSS-SECTIONAL FEATURES ===
        # Price distance from 52-week high/low
        if len(df) >= 252:
            features['dist_52w_high'] = df['close'] / df['high'].rolling(252).max() - 1.0
            features['dist_52w_low'] = df['close'] / df['low'].rolling(252).min() - 1.0

        # Gap (open vs prev close)
        features['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)

        # Intraday range relative to volatility
        features['range_to_vol'] = features['hl_range_pct'] / features['realized_vol_10'].clip(lower=1e-6)

        # Store feature names
        self.feature_names = features.columns.tolist()
        self._fitted = True
        
        return features.dropna()
    
    def _rsi(self, prices: np.ndarray, period: int) -> pd.Series:
        """Relative Strength Index."""
        deltas = np.diff(prices, prepend=prices[0])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = pd.Series(gains).rolling(period).mean()
        avg_loss = pd.Series(losses).rolling(period).mean()
        
        rs = avg_gain / avg_loss.clip(lower=1e-10)
        return 100 - (100 / (1 + rs))
    
    def _atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Average True Range."""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(period).mean()
    
    def _garman_klass(self, df: pd.DataFrame, window: int) -> pd.Series:
        """Garman-Klass volatility estimator."""
        log_hl = np.log(df['high'] / df['low'])
        log_co = np.log(df['close'] / df['open'])
        gk = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
        return np.sqrt(gk.rolling(window).mean() * 252)
    
    def get_feature_names(self) -> List[str]:
        """Return list of computed feature names."""
        return self.feature_names.copy()

    def normalize_features(self, features: pd.DataFrame, method: str = 'zscore') -> pd.DataFrame:
        """
        Normalize features for ML model input.
        
        Args:
            features: Raw feature DataFrame from compute_features()
            method: 'zscore' (default), 'minmax', or 'robust'
            
        Returns:
            Normalized feature DataFrame
        """
        if method == 'zscore':
            return (features - features.mean()) / features.std().clip(lower=1e-10)
        elif method == 'minmax':
            fmin = features.min()
            fmax = features.max()
            return (features - fmin) / (fmax - fmin).clip(lower=1e-10)
        elif method == 'robust':
            median = features.median()
            iqr = features.quantile(0.75) - features.quantile(0.25)
            return (features - median) / iqr.clip(lower=1e-10)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'zscore', 'minmax', or 'robust'")

    def feature_importance(self, features: pd.DataFrame, target: pd.Series) -> pd.Series:
        """
        Rank features by absolute correlation with target variable.
        
        Args:
            features: Feature DataFrame
            target: Target variable (e.g., forward returns)
            
        Returns:
            Series of feature importances sorted descending
        """
        aligned_features, aligned_target = features.align(target, join='inner', axis=0)
        correlations = aligned_features.corrwith(aligned_target).abs()
        return correlations.sort_values(ascending=False)


def update_features(data) -> Optional[pd.DataFrame]:
    """
    Legacy API: Compute features from data.
    Compatible with ai_optimizer.py calls.
    
    Args:
        data: DataFrame with OHLCV data, or None for no-op
        
    Returns:
        Feature DataFrame or None
    """
    if data is None:
        logger.info("[ML] Feature engine initialized — no data provided")
        return None
    
    engine = FeatureEngine()
    features = engine.compute_features(data)
    logger.info(f"[ML] Features computed: {len(features)} rows x {len(engine.feature_names)} features")
    return features

