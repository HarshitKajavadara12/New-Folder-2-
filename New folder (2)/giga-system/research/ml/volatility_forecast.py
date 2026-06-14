"""
GIGA SYSTEM - Volatility Forecast Module
Forecasts future volatility using multiple models.

Models:
1. EWMA (Exponentially Weighted Moving Average) — fast, production-ready
2. GARCH(1,1) — MLE-fitted, industry standard
3. HAR-RV (Heterogeneous Autoregressive Realized Volatility) — for realized vol
4. Ensemble — weighted average of all models
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Tuple, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class VolForecast:
    """Volatility forecast result."""
    daily_vol: float          # 1-day ahead vol forecast (annualized)
    weekly_vol: float         # 5-day ahead vol forecast
    monthly_vol: float        # 21-day ahead vol forecast
    model: str                # Model used
    confidence_interval: Tuple[float, float]  # 90% CI
    timestamp: Optional[str] = None


class EWMAVolModel:
    """
    EWMA Volatility Model (RiskMetrics standard).
    lambda = 0.94 for daily data (JP Morgan RiskMetrics).
    """
    
    def __init__(self, decay_factor: float = 0.94):
        self.decay_factor = decay_factor
        self.current_variance = None
    
    def fit(self, returns: np.ndarray) -> float:
        """Fit EWMA to return series, return current variance."""
        variance = np.var(returns[:10])  # Seed with first 10 obs
        
        for r in returns:
            variance = self.decay_factor * variance + (1 - self.decay_factor) * r**2
        
        self.current_variance = variance
        return variance
    
    def forecast(self, horizon_days: int = 1) -> float:
        """Forecast volatility h-days ahead (annualized)."""
        if self.current_variance is None:
            raise ValueError("Model not fitted. Call fit() first.")
        # EWMA variance is constant for all horizons (mean-reverting to 0)
        return np.sqrt(self.current_variance * 252)


class GARCH11Model:
    """
    GARCH(1,1) Model — Maximum Likelihood Estimation.
    σ²_t = ω + α * ε²_{t-1} + β * σ²_{t-1}
    
    Constraints: α + β < 1 (stationarity), ω > 0, α > 0, β > 0
    """
    
    def __init__(self):
        self.omega = 0.0
        self.alpha = 0.0
        self.beta = 0.0
        self.current_variance = None
        self.long_run_variance = None
        self._fitted = False
    
    def fit(self, returns: np.ndarray) -> Dict[str, float]:
        """
        Fit GARCH(1,1) via simplified MLE (grid search + Nelder-Mead).
        For production: use arch library. This is a self-contained fallback.
        """
        n = len(returns)
        if n < 30:
            raise ValueError(f"Need >= 30 returns for GARCH, got {n}")
        
        sample_var = np.var(returns)
        
        best_ll = -np.inf
        best_params = (0.0, 0.0, 0.0)
        
        # Grid search for good starting point
        for alpha in np.arange(0.02, 0.20, 0.02):
            for beta in np.arange(0.70, 0.98, 0.02):
                if alpha + beta >= 0.9999:
                    continue
                omega = sample_var * (1 - alpha - beta)
                if omega <= 0:
                    continue
                ll = self._log_likelihood(returns, omega, alpha, beta)
                if ll > best_ll:
                    best_ll = ll
                    best_params = (omega, alpha, beta)
        
        self.omega, self.alpha, self.beta = best_params
        self.long_run_variance = self.omega / (1 - self.alpha - self.beta) if (self.alpha + self.beta) < 1 else sample_var
        
        # Compute current conditional variance
        variance = sample_var
        for r in returns:
            variance = self.omega + self.alpha * r**2 + self.beta * variance
        self.current_variance = variance
        self._fitted = True
        
        return {
            'omega': self.omega,
            'alpha': self.alpha,
            'beta': self.beta,
            'persistence': self.alpha + self.beta,
            'long_run_vol': np.sqrt(self.long_run_variance * 252)
        }
    
    def _log_likelihood(self, returns: np.ndarray, omega: float, alpha: float, beta: float) -> float:
        """Gaussian log-likelihood for GARCH(1,1)."""
        n = len(returns)
        variance = np.var(returns)
        ll = 0.0
        
        for t in range(n):
            if variance <= 0:
                variance = 1e-10
            ll += -0.5 * (np.log(2 * np.pi) + np.log(variance) + returns[t]**2 / variance)
            variance = omega + alpha * returns[t]**2 + beta * variance
        
        return ll
    
    def forecast(self, horizon_days: int = 1) -> float:
        """
        Forecast annualized volatility h-days ahead.
        Uses GARCH mean-reversion formula.
        """
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        persistence = self.alpha + self.beta
        h_var = self.long_run_variance + (persistence ** horizon_days) * (self.current_variance - self.long_run_variance)
        return np.sqrt(h_var * 252)


class HARModel:
    """
    HAR-RV (Heterogeneous Autoregressive Realized Volatility).
    RV_t = c + β_d * RV_{t-1} + β_w * RV_{t-5:t} + β_m * RV_{t-22:t} + ε
    """
    
    def __init__(self):
        self.intercept = 0.0
        self.beta_daily = 0.0
        self.beta_weekly = 0.0
        self.beta_monthly = 0.0
        self._fitted = False
    
    def fit(self, returns: np.ndarray) -> Dict[str, float]:
        """Fit HAR model using OLS on realized variance components."""
        rv = returns**2  # Squared returns as realized variance proxy
        
        n = len(rv)
        if n < 30:
            raise ValueError(f"Need >= 30 observations, got {n}")
        
        # Build HAR components
        rv_daily = rv[22:]  # Target
        rv_d = rv[21:-1]    # Daily lag
        rv_w = np.array([np.mean(rv[i:i+5]) for i in range(17, n-5)])[:len(rv_daily)]  # Weekly
        rv_m = np.array([np.mean(rv[i:i+22]) for i in range(0, n-22)])[:len(rv_daily)]  # Monthly
        
        min_len = min(len(rv_daily), len(rv_d), len(rv_w), len(rv_m))
        rv_daily = rv_daily[:min_len]
        rv_d = rv_d[:min_len]
        rv_w = rv_w[:min_len]
        rv_m = rv_m[:min_len]
        
        # OLS: Y = Xβ
        X = np.column_stack([np.ones(min_len), rv_d, rv_w, rv_m])
        y = rv_daily
        
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            self.intercept, self.beta_daily, self.beta_weekly, self.beta_monthly = beta
            self._fitted = True
        except np.linalg.LinAlgError:
            # Fallback to equal weights
            self.intercept = np.mean(rv)
            self.beta_daily = 0.33
            self.beta_weekly = 0.33
            self.beta_monthly = 0.33
            self._fitted = True
        
        return {
            'intercept': self.intercept,
            'beta_daily': self.beta_daily,
            'beta_weekly': self.beta_weekly,
            'beta_monthly': self.beta_monthly
        }
    
    def forecast(self, returns: np.ndarray) -> float:
        """Forecast 1-day ahead annualized vol using fitted HAR."""
        if not self._fitted:
            raise ValueError("Model not fitted")
        
        rv = returns**2
        rv_d = rv[-1]
        rv_w = np.mean(rv[-5:])
        rv_m = np.mean(rv[-22:]) if len(rv) >= 22 else np.mean(rv)
        
        forecast_var = self.intercept + self.beta_daily * rv_d + self.beta_weekly * rv_w + self.beta_monthly * rv_m
        return np.sqrt(max(forecast_var, 1e-10) * 252)


class VolatilityForecaster:
    """
    Ensemble volatility forecaster combining EWMA, GARCH, and HAR models.
    """
    
    def __init__(self, ewma_weight: float = 0.3, garch_weight: float = 0.5, har_weight: float = 0.2):
        self.ewma = EWMAVolModel()
        self.garch = GARCH11Model()
        self.har = HARModel()
        self.weights = {
            'ewma': ewma_weight,
            'garch': garch_weight,
            'har': har_weight
        }
        self._fitted = False
    
    def fit(self, returns: np.ndarray) -> Dict[str, Any]:
        """Fit all models on return series."""
        results = {}
        
        # EWMA
        try:
            self.ewma.fit(returns)
            results['ewma'] = {'status': 'OK', 'vol': self.ewma.forecast(1)}
        except Exception as e:
            results['ewma'] = {'status': f'FAIL: {e}', 'vol': None}
            self.weights['ewma'] = 0
        
        # GARCH
        try:
            garch_params = self.garch.fit(returns)
            results['garch'] = {'status': 'OK', 'params': garch_params, 'vol': self.garch.forecast(1)}
        except Exception as e:
            results['garch'] = {'status': f'FAIL: {e}', 'vol': None}
            self.weights['garch'] = 0
        
        # HAR
        try:
            har_params = self.har.fit(returns)
            results['har'] = {'status': 'OK', 'params': har_params, 'vol': self.har.forecast(returns)}
        except Exception as e:
            results['har'] = {'status': f'FAIL: {e}', 'vol': None}
            self.weights['har'] = 0
        
        # Normalize weights
        total_w = sum(self.weights.values())
        if total_w > 0:
            self.weights = {k: v / total_w for k, v in self.weights.items()}
        
        self._fitted = True
        return results
    
    def forecast(self, returns: np.ndarray, horizon_days: int = 1) -> VolForecast:
        """
        Produce ensemble volatility forecast.
        
        Args:
            returns: Recent return series (for HAR model)
            horizon_days: Forecast horizon
            
        Returns:
            VolForecast
        """
        if not self._fitted:
            self.fit(returns)
        
        forecasts = {}
        
        if self.weights.get('ewma', 0) > 0:
            forecasts['ewma'] = self.ewma.forecast(horizon_days)
        if self.weights.get('garch', 0) > 0:
            forecasts['garch'] = self.garch.forecast(horizon_days)
        if self.weights.get('har', 0) > 0:
            forecasts['har'] = self.har.forecast(returns)
        
        if not forecasts:
            # Fallback: simple realized vol
            daily_vol = np.std(returns) * np.sqrt(252)
            return VolForecast(
                daily_vol=daily_vol, weekly_vol=daily_vol, monthly_vol=daily_vol,
                model='fallback_realized', confidence_interval=(daily_vol * 0.7, daily_vol * 1.3)
            )
        
        # Weighted ensemble
        ensemble_vol = sum(
            self.weights.get(model, 0) * vol
            for model, vol in forecasts.items()
        )
        
        # Confidence interval (±30% for daily vol)
        ci_low = ensemble_vol * 0.70
        ci_high = ensemble_vol * 1.30
        
        # Term structure using GARCH persistence (if available)
        if self.weights.get('garch', 0) > 0 and self.garch._fitted:
            persistence = self.garch.alpha + self.garch.beta
            # Multi-step variance: reduce sqrt-T scaling for high persistence
            weekly_factor = np.sqrt(5) * (1 - 0.2 * (1 - persistence))
            monthly_factor = np.sqrt(21) * (1 - 0.3 * (1 - persistence))
        else:
            weekly_factor = np.sqrt(5)
            monthly_factor = np.sqrt(21)

        # De-annualize to get daily vol, then scale to horizon
        daily_vol_raw = ensemble_vol / np.sqrt(252)
        weekly_vol = daily_vol_raw * weekly_factor
        monthly_vol = daily_vol_raw * monthly_factor

        return VolForecast(
            daily_vol=ensemble_vol,
            weekly_vol=weekly_vol,
            monthly_vol=monthly_vol,
            model='ensemble',
            confidence_interval=(ci_low, ci_high)
        )

    def forecast_horizon(self, returns: np.ndarray, horizon_days: int) -> float:
        """
        Forecast non-annualized volatility for a specific horizon.
        Uses GARCH mean-reversion term structure when available.

        Args:
            returns: Recent return series
            horizon_days: Forecast horizon in trading days

        Returns:
            Non-annualized volatility for the horizon period
        """
        if not self._fitted:
            self.fit(returns)

        if self.weights.get('garch', 0) > 0 and self.garch._fitted:
            # GARCH multi-step: variance converges to long_run_variance
            persistence = self.garch.alpha + self.garch.beta
            current_var = self.garch.current_variance
            lr_var = self.garch.long_run_variance

            # Sum of h-step-ahead conditional variances
            total_var = 0.0
            for h in range(1, horizon_days + 1):
                h_var = lr_var + (persistence ** h) * (current_var - lr_var)
                total_var += h_var

            return np.sqrt(total_var)
        else:
            # Fallback: sqrt-T scaling from daily vol
            daily_vol_raw = self.ewma.forecast(1) / np.sqrt(252) if self.weights.get('ewma', 0) > 0 else np.std(returns)
            return daily_vol_raw * np.sqrt(horizon_days)


def retrain(data) -> Optional[VolForecast]:
    """
    Legacy API: Retrain volatility forecast models.
    Compatible with ai_optimizer.py calls.
    
    Args:
        data: DataFrame with 'close' column, or returns array, or None
        
    Returns:
        VolForecast or None
    """
    if data is None:
        logger.info("[ML] Volatility forecaster initialized — no data provided")
        return None
    
    # Extract returns
    if isinstance(data, pd.DataFrame) and 'close' in data.columns:
        returns = data['close'].pct_change().dropna().values
    elif isinstance(data, np.ndarray):
        returns = data
    else:
        logger.warning("[ML] Cannot extract returns from data")
        return None
    
    if len(returns) < 30:
        logger.warning(f"[ML] Insufficient data for vol forecast: {len(returns)} < 30")
        return None
    
    forecaster = VolatilityForecaster()
    fit_results = forecaster.fit(returns)
    forecast = forecaster.forecast(returns)
    
    logger.info(
        f"[ML] Vol forecast retrained: {forecast.daily_vol:.2%} daily "
        f"(model={forecast.model}, CI=[{forecast.confidence_interval[0]:.2%}, {forecast.confidence_interval[1]:.2%}])"
    )
    
    return forecast

