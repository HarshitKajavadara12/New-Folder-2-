"""
DOMAIN 5: INFORMATION & ENTROPY (Η, Φ)
Greek Concepts:
- Η (Eta) → Entropy (Uncertainty / Disorder)
- Φ (Phi) → Information Flow (Surprise)

Implementation:
Using Shannon Entropy and KL Divergence to measure information content.
Alpha = Trading before information entropy is maximized (State resolution).
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy

class InformationGeometer:
    """
    Measures the information dynamics of the market.
    """
    
    def calculate_shannon_entropy(self, value_counts: pd.Series) -> float:
        """
        Calculate Η (Eta): Shannon Entropy of a distribution.
        H = -sum(p * log(p))
        """
        probabilities = value_counts / value_counts.sum()
        return entropy(probabilities, base=2)
    
    def calculate_market_entropy(self, price_series: pd.Series, bins: int = 20) -> float:
        """
        Discretize returns and calculate entropy.
        High Entropy = Random Walk (Efficient).
        Low Entropy = Structured (Predictable).
        """
        returns = price_series.pct_change().dropna()
        if len(returns) == 0: return 0.0
        
        hist, bin_edges = np.histogram(returns, bins=bins, density=True)
        # Add small epsilon to avoid log(0)
        hist = hist + 1e-10
        
        return entropy(hist, base=2)

    def calculate_kl_divergence(self, p: np.array, q: np.array) -> float:
        """
        Calculate Kullback-Leibler Divergence (Φ - Information Gain).
        Measures surprise of distribution P relative to prior Q.
        """
        return entropy(p, qk=q)

    def measure_feature_compression(self, signals: pd.DataFrame) -> float:
        """
        Measure how much information signals contain vs random noise.
        """
        # PCA-based entropy estimation could go here
        # Simplified: Mean correlation
        corr_matrix = signals.corr().abs()
        avg_corr = corr_matrix.mean().mean()
        
        # If avg_corr is 1, Entropy is 0 (Redundant).
        # If avg_corr is 0, Entropy is Max (Independent).
        return 1.0 - avg_corr
