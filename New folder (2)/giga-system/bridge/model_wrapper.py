"""
GIGA SYSTEM - R Model Wrappers
Greek Intelligence for Global Analysis

Python wrappers for R statistical models used in quantitative finance.
Provides clean, Pythonic interfaces to powerful R packages like rugarch,
forecast, vars, etc.

Key Models:
- GARCH: Volatility forecasting (rugarch package)
- ARIMA: Time series forecasting (forecast package)  
- VAR: Vector autoregression (vars package)
- Cointegration: Statistical arbitrage (urca package)
- Copulas: Tail dependence modeling (copula package)

Design Philosophy:
- Pythonic API with R computational power
- Automatic parameter validation
- Error handling with meaningful messages
- Results returned as Python objects
"""

#  ️ PHASE 2 WARNING: AIR-GAP VIOLATION
# Wrappers that make Research code look like Live code are an Anti-Pattern.
# "Making X look like Y so Z can use it directly" violates the separation of concerns.
# This code belongs in the RESEARCH domain only.
# DO NOT IMPORT IN EXECUTION ENGINE.

from typing import Any, Dict, List, Optional, Tuple, Union
import warnings
from dataclasses import dataclass

import numpy as np
import polars as pl

from .rpy2_interface import r_interface, check_r_availability
from .data_converter import data_converter

# Python-only fallbacks when R is unavailable
try:
    from statsmodels.tsa.arima.model import ARIMA as StatsARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False


@dataclass
class ModelResults:
    """Container for model fitting results."""
    success: bool
    parameters: Dict[str, Any]
    fitted_values: Optional[np.ndarray] = None
    residuals: Optional[np.ndarray] = None
    log_likelihood: Optional[float] = None
    aic: Optional[float] = None
    bic: Optional[float] = None
    forecasts: Optional[np.ndarray] = None
    forecast_errors: Optional[np.ndarray] = None


class GARCHModel:
    """
    Python wrapper for R's rugarch package.
    
    Provides GARCH family models for volatility forecasting:
    - GARCH(1,1): Standard volatility model
    - EGARCH: Exponential GARCH (asymmetric effects)
    - GJR-GARCH: Glosten-Jagannathan-Runkle GARCH
    - APARCH: Asymmetric Power ARCH
    """
    
    def __init__(self, model_type: str = "sGARCH", distribution: str = "norm"):
        """
        Initialize GARCH model specification.
        
        Args:
            model_type: GARCH variant ("sGARCH", "eGARCH", "gjrGARCH", "apARCH")
            distribution: Error distribution ("norm", "std", "sstd", "ged")
        """
        self.model_type = model_type
        self.distribution = distribution
        self.fitted = False
        self.r_model_name = "garch_model"
        self._use_python_fallback = False
        self._python_model = None
        
        if not check_r_availability():
            if ARCH_AVAILABLE:
                warnings.warn("R not available, using Python 'arch' package as fallback for GARCH")
                self._use_python_fallback = True
            else:
                raise RuntimeError(
                    "R interface not available and 'arch' package not installed. "
                    "Install with: pip install arch"
                )
        else:
            # Ensure rugarch package is loaded
            success = r_interface.ensure_packages(["rugarch"])
            if not success:
                if ARCH_AVAILABLE:
                    warnings.warn("rugarch R package not available, using Python 'arch' fallback")
                    self._use_python_fallback = True
                else:
                    raise RuntimeError("Failed to load rugarch package and no Python fallback available")
    
    def fit(self, 
            returns: Union[np.ndarray, pl.DataFrame],
            p: int = 1,
            q: int = 1) -> ModelResults:
        """
        Fit GARCH model to return data.
        
        Args:
            returns: Financial return data
            p: ARCH order (lags in conditional variance equation)
            q: GARCH order (lags in conditional variance equation)
            
        Returns:
            ModelResults: Fitted model results
        """
        try:
            # Prepare data
            if isinstance(returns, pl.DataFrame):
                returns_array = returns.to_numpy()[:, -1]  # Last column assumed to be returns
            else:
                returns_array = returns.flatten()
            
            if self._use_python_fallback:
                return self._fit_python_garch(returns_array, p, q)
            
            # Convert to R
            data_converter.numpy_to_r_matrix(returns_array, "garch_returns")
            
            # Create GARCH specification in R
            spec_code = f"""
            garch_spec <- ugarchspec(
                variance.model = list(
                    model = "{self.model_type}",
                    garchOrder = c({p}, {q})
                ),
                mean.model = list(armaOrder = c(0, 0)),
                distribution.model = "{self.distribution}"
            )
            """
            
            r_interface.execute_r_code(spec_code)
            
            # Fit the model
            fit_code = f"""
            {self.r_model_name} <- ugarchfit(
                spec = garch_spec,
                data = garch_returns,
                solver = "hybrid"
            )
            """
            
            r_interface.execute_r_code(fit_code)
            
            # Extract results
            results = self._extract_garch_results()
            results.success = True
            self.fitted = True
            
            return results
            
        except Exception as e:
            warnings.warn(f"GARCH model fitting failed: {e}")
            return ModelResults(success=False, parameters={})
    
    def _fit_python_garch(self, returns_array: np.ndarray, p: int, q: int) -> ModelResults:
        """Fit GARCH model using Python 'arch' package as fallback."""
        try:
            # Map distribution names from R to arch
            dist_map = {"norm": "normal", "std": "t", "sstd": "skewt", "ged": "ged"}
            dist = dist_map.get(self.distribution, "normal")
            
            # Map model type from R to arch
            vol_map = {"sGARCH": "GARCH", "eGARCH": "EGARCH", "gjrGARCH": "GJR-GARCH"}
            vol = vol_map.get(self.model_type, "GARCH")
            
            am = arch_model(returns_array * 100, vol=vol, p=p, q=q, dist=dist)
            res = am.fit(disp="off")
            self._python_model = res
            self.fitted = True
            
            return ModelResults(
                success=True,
                parameters=dict(res.params),
                fitted_values=res.conditional_volatility / 100,
                residuals=res.resid / 100,
                log_likelihood=res.loglikelihood,
                aic=res.aic,
                bic=res.bic,
            )
        except Exception as e:
            warnings.warn(f"Python GARCH fallback failed: {e}")
            return ModelResults(success=False, parameters={})
    
    def forecast(self, 
                n_ahead: int = 1,
                n_roll: int = 0) -> Optional[Dict[str, np.ndarray]]:
        """
        Generate volatility forecasts from fitted model.
        
        Args:
            n_ahead: Number of periods to forecast
            n_roll: Number of rolling forecasts
            
        Returns:
            Dict with forecasted volatility and confidence intervals
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before forecasting")
        
        try:
            # Generate forecasts in R
            forecast_code = f"""
            garch_forecast <- ugarchforecast(
                {self.r_model_name},
                n.ahead = {n_ahead},
                n.roll = {n_roll}
            )
            forecast_sigma <- sigma(garch_forecast)
            forecast_mean <- fitted(garch_forecast)
            """
            
            r_interface.execute_r_code(forecast_code)
            
            # Extract forecast results
            forecasted_vol = data_converter.r_matrix_to_numpy("forecast_sigma")
            forecasted_mean = data_converter.r_matrix_to_numpy("forecast_mean")
            
            return {
                "volatility": forecasted_vol,
                "mean": forecasted_mean,
                "n_ahead": n_ahead
            }
            
        except Exception as e:
            warnings.warn(f"GARCH forecasting failed: {e}")
            return None
    
    def _extract_garch_results(self) -> ModelResults:
        """Extract fitted model results from R."""
        try:
            # Get model coefficients
            r_interface.execute_r_code("garch_coef <- coef(garch_model)")
            coefficients = data_converter.get_r_model_results("garch_model", ["coef"])
            
            # Get fitted values and residuals
            r_interface.execute_r_code("garch_fitted <- fitted(garch_model)")
            r_interface.execute_r_code("garch_residuals <- residuals(garch_model)")
            
            fitted_values = data_converter.r_matrix_to_numpy("garch_fitted")
            residuals = data_converter.r_matrix_to_numpy("garch_residuals")
            
            # Information criteria
            r_interface.execute_r_code("garch_loglik <- likelihood(garch_model)")
            r_interface.execute_r_code("garch_aic <- infocriteria(garch_model)[1]")
            r_interface.execute_r_code("garch_bic <- infocriteria(garch_model)[2]")
            
            log_likelihood = float(r_interface.r_to_numpy("garch_loglik"))
            aic = float(r_interface.r_to_numpy("garch_aic"))
            bic = float(r_interface.r_to_numpy("garch_bic"))
            
            return ModelResults(
                success=True,
                parameters=coefficients,
                fitted_values=fitted_values,
                residuals=residuals,
                log_likelihood=log_likelihood,
                aic=aic,
                bic=bic
            )
            
        except Exception as e:
            warnings.warn(f"Failed to extract GARCH results: {e}")
            return ModelResults(success=False, parameters={})


class ARIMAModel:
    """
    Python wrapper for R's forecast package ARIMA models.
    
    Provides comprehensive ARIMA modeling:
    - Automatic order selection
    - Seasonal ARIMA (SARIMA)
    - Forecasting with confidence intervals
    - Model diagnostics
    """
    
    def __init__(self, seasonal: bool = False):
        """
        Initialize ARIMA model.
        
        Args:
            seasonal: Whether to use seasonal ARIMA (SARIMA)
        """
        self.seasonal = seasonal
        self.fitted = False
        self.r_model_name = "arima_model"
        self._use_python_fallback = False
        self._python_model = None
        
        if not check_r_availability():
            if STATSMODELS_AVAILABLE:
                warnings.warn("R not available, using statsmodels as fallback for ARIMA")
                self._use_python_fallback = True
            else:
                raise RuntimeError(
                    "R interface not available and statsmodels not installed. "
                    "Install with: pip install statsmodels"
                )
        else:
            # Ensure forecast package is loaded
            success = r_interface.ensure_packages(["forecast"])
            if not success:
                if STATSMODELS_AVAILABLE:
                    warnings.warn("forecast R package not available, using statsmodels fallback")
                    self._use_python_fallback = True
                else:
                    raise RuntimeError("Failed to load forecast package and no Python fallback available")
    
    def fit(self, 
           data: Union[np.ndarray, pl.DataFrame],
           p: Optional[int] = None,
           d: Optional[int] = None,
           q: Optional[int] = None) -> ModelResults:
        """
        Fit ARIMA model with optional automatic order selection.
        
        Args:
            data: Time series data
            p: AR order (None for automatic selection)
            d: Differencing order (None for automatic selection)  
            q: MA order (None for automatic selection)
            
        Returns:
            ModelResults: Fitted model results
        """
        try:
            # Prepare data
            if isinstance(data, pl.DataFrame):
                ts_data = data.to_numpy()[:, -1]
            else:
                ts_data = data.flatten()
            
            if self._use_python_fallback:
                return self._fit_python_arima(ts_data, p, d, q)
            
            data_converter.numpy_to_r_matrix(ts_data, "arima_data")
            
            # Fit model (automatic or manual order)
            if all(param is not None for param in [p, d, q]):
                # Manual order specification
                fit_code = f"""
                {self.r_model_name} <- Arima(
                    arima_data,
                    order = c({p}, {d}, {q})
                )
                """
            else:
                # Automatic order selection
                fit_code = f"""
                {self.r_model_name} <- auto.arima(
                    arima_data,
                    seasonal = {str(self.seasonal).upper()},
                    stepwise = FALSE,
                    approximation = FALSE
                )
                """
            
            r_interface.execute_r_code(fit_code)
            
            # Extract results
            results = self._extract_arima_results()
            results.success = True
            self.fitted = True
            
            return results
            
        except Exception as e:
            warnings.warn(f"ARIMA model fitting failed: {e}")
            return ModelResults(success=False, parameters={})
    
    def _fit_python_arima(self, ts_data: np.ndarray, 
                          p: Optional[int], d: Optional[int], q: Optional[int]) -> ModelResults:
        """Fit ARIMA model using statsmodels as fallback."""
        try:
            # Default order if not specified
            order = (p or 1, d or 0, q or 0)
            
            model = StatsARIMA(ts_data, order=order)
            res = model.fit()
            self._python_model = res
            self.fitted = True
            
            return ModelResults(
                success=True,
                parameters=dict(zip(res.param_names, res.params)),
                fitted_values=res.fittedvalues,
                residuals=res.resid,
                log_likelihood=res.llf,
                aic=res.aic,
                bic=res.bic,
            )
        except Exception as e:
            warnings.warn(f"Python ARIMA fallback failed: {e}")
            return ModelResults(success=False, parameters={})
    
    def forecast(self, h: int = 1, level: List[float] = [80, 95]) -> Optional[Dict[str, Any]]:
        """
        Generate forecasts from fitted ARIMA model.
        
        Args:
            h: Forecast horizon
            level: Confidence levels for intervals
            
        Returns:
            Dict with forecasts and confidence intervals
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before forecasting")
        
        try:
            level_str = "c(" + ",".join(map(str, level)) + ")"
            
            forecast_code = f"""
            arima_forecast <- forecast({self.r_model_name}, h = {h}, level = {level_str})
            forecast_mean <- as.numeric(arima_forecast$mean)
            forecast_lower <- as.numeric(arima_forecast$lower)
            forecast_upper <- as.numeric(arima_forecast$upper)
            """
            
            r_interface.execute_r_code(forecast_code)
            
            # Extract forecasts
            forecasts = data_converter.r_matrix_to_numpy("forecast_mean")
            lower_bounds = data_converter.r_matrix_to_numpy("forecast_lower")
            upper_bounds = data_converter.r_matrix_to_numpy("forecast_upper")
            
            return {
                "forecasts": forecasts,
                "lower": lower_bounds,
                "upper": upper_bounds,
                "confidence_levels": level,
                "horizon": h
            }
            
        except Exception as e:
            warnings.warn(f"ARIMA forecasting failed: {e}")
            return None
    
    def _extract_arima_results(self) -> ModelResults:
        """Extract ARIMA model results."""
        try:
            # Get model information
            results = data_converter.get_r_model_results("arima_model", 
                                                       ["coef", "fitted", "residuals"])
            
            # Information criteria
            r_interface.execute_r_code("arima_aic <- AIC(arima_model)")
            r_interface.execute_r_code("arima_bic <- BIC(arima_model)")
            
            aic = float(data_converter.r_matrix_to_numpy("arima_aic"))
            bic = float(data_converter.r_matrix_to_numpy("arima_bic"))
            
            return ModelResults(
                success=True,
                parameters=results.get("coef", {}),
                fitted_values=results.get("fitted"),
                residuals=results.get("residuals"),
                aic=aic,
                bic=bic
            )
            
        except Exception as e:
            warnings.warn(f"Failed to extract ARIMA results: {e}")
            return ModelResults(success=False, parameters={})


class CointegrationTest:
    """
    Python wrapper for R's urca package cointegration tests.
    
    Essential for pairs trading and statistical arbitrage:
    - Johansen test: Multiple cointegration relationships
    - Engle-Granger test: Simple two-variable case
    - Phillips-Ouliaris test: Alternative approach
    """
    
    def __init__(self):
        """Initialize cointegration testing."""
        if not check_r_availability():
            raise RuntimeError("R interface not available")
        
        # Ensure urca package is loaded
        success = r_interface.ensure_packages(["urca"])
        if not success:
            raise RuntimeError("Failed to load urca package")
    
    def johansen_test(self, 
                     data: Union[np.ndarray, pl.DataFrame],
                     test_type: str = "trace",
                     k: int = 2) -> Dict[str, Any]:
        """
        Perform Johansen cointegration test.
        
        Args:
            data: Multivariate time series (each column is a variable)
            test_type: "trace" or "eigen"
            k: Number of lags
            
        Returns:
            Dict with test results
        """
        try:
            # Prepare data
            if isinstance(data, pl.DataFrame):
                test_data = data.to_numpy()
            else:
                test_data = data
            
            data_converter.numpy_to_r_matrix(test_data, "coint_data")
            
            # Run Johansen test
            test_code = f"""
            johansen_result <- ca.jo(
                coint_data,
                type = "{test_type}",
                K = {k},
                ecdet = "const"
            )
            test_statistic <- johansen_result@teststat
            critical_values <- johansen_result@cval
            """
            
            r_interface.execute_r_code(test_code)
            
            # Extract results
            test_stat = data_converter.r_matrix_to_numpy("test_statistic")
            crit_vals = data_converter.r_matrix_to_numpy("critical_values")
            
            return {
                "test_statistic": test_stat,
                "critical_values": crit_vals,
                "test_type": test_type,
                "conclusion": self._interpret_johansen(test_stat, crit_vals)
            }
            
        except Exception as e:
            warnings.warn(f"Johansen test failed: {e}")
            return {"error": str(e)}
    
    def engle_granger_test(self, 
                          y: np.ndarray, 
                          x: np.ndarray) -> Dict[str, Any]:
        """
        Perform Engle-Granger two-step cointegration test.
        
        Args:
            y: Dependent variable time series
            x: Independent variable time series
            
        Returns:
            Dict with test results
        """
        try:
            # Send data to R
            data_converter.numpy_to_r_matrix(y, "y_series")
            data_converter.numpy_to_r_matrix(x, "x_series")
            
            # Run Engle-Granger test
            test_code = """
            eg_result <- ca.po(cbind(y_series, x_series), demean = "constant")
            eg_statistic <- eg_result@teststat
            eg_cval <- eg_result@cval
            """
            
            r_interface.execute_r_code(test_code)
            
            # Extract results
            test_stat = float(data_converter.r_matrix_to_numpy("eg_statistic"))
            crit_vals = data_converter.r_matrix_to_numpy("eg_cval")
            
            return {
                "test_statistic": test_stat,
                "critical_values": crit_vals,
                "conclusion": "Cointegrated" if test_stat < crit_vals[1] else "Not cointegrated"
            }
            
        except Exception as e:
            warnings.warn(f"Engle-Granger test failed: {e}")
            return {"error": str(e)}
    
    def _interpret_johansen(self, test_stat: np.ndarray, crit_vals: np.ndarray) -> str:
        """Interpret Johansen test results."""
        # Simple interpretation: compare first test statistic with 5% critical value
        if len(test_stat) > 0 and len(crit_vals) > 0:
            if test_stat[0] > crit_vals[0, 1]:  # 5% critical value
                return "Evidence of cointegration"
            else:
                return "No evidence of cointegration"
        return "Unable to interpret results"


# Convenience functions for quick model usage
def fit_garch(returns: Union[np.ndarray, pl.DataFrame], **kwargs) -> ModelResults:
    """Quick GARCH model fitting."""
    model = GARCHModel(**kwargs)
    return model.fit(returns)


def fit_arima(data: Union[np.ndarray, pl.DataFrame], **kwargs) -> ModelResults:
    """Quick ARIMA model fitting."""
    model = ARIMAModel(**kwargs)
    return model.fit(data)


def test_cointegration(data: Union[np.ndarray, pl.DataFrame], method: str = "johansen") -> Dict[str, Any]:
    """Quick cointegration testing."""
    tester = CointegrationTest()
    
    if method == "johansen":
        return tester.johansen_test(data)
    elif method == "engle-granger" and data.shape[1] == 2:
        return tester.engle_granger_test(data[:, 0], data[:, 1])
    else:
        raise ValueError("Unsupported method or incorrect data dimensions")