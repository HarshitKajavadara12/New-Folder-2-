"""
GIGA SYSTEM - Python-R Bridge
Seamless integration between Python and R analytics

Uses rpy2 for native R integration within Python
"""

import numpy as np
from typing import Dict, List, Any, Optional, Union
import os

# R integration via rpy2
try:
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri, pandas2ri
    from rpy2.robjects.packages import importr
    from rpy2.robjects.conversion import localconverter
    
    # Activate automatic numpy/pandas conversion
    numpy2ri.activate()
    pandas2ri.activate()
    
    R_AVAILABLE = True
except (ImportError, OSError, Exception):
    R_AVAILABLE = False
    ro = None


# =============================================================================
# R SESSION MANAGEMENT
# =============================================================================

class RSession:
    """Manages R session and provides interface to R functions."""
    
    def __init__(self, r_scripts_path: Optional[str] = None):
        """
        Initialize R session and load scripts.
        
        Parameters
        ----------
        r_scripts_path : str, optional
            Path to directory containing R scripts.
            If None, uses default 'r_analytics' directory.
        """
        if not R_AVAILABLE:
            raise ImportError("rpy2 is required for R integration")
        
        self.r = ro.r
        self._loaded_scripts = []
        
        # Default path
        if r_scripts_path is None:
            r_scripts_path = os.path.join(
                os.path.dirname(__file__), '..', 'r_analytics'
            )
        self.r_scripts_path = os.path.abspath(r_scripts_path)
        
        # Load required R packages
        self._load_packages()
        
        # Load custom R scripts
        self._load_scripts()
    
    def _load_packages(self):
        """Load required R packages."""
        packages = [
            'forecast',
            'rugarch',
            'copula',
            'evd',
            'PortfolioAnalytics',
            'PerformanceAnalytics',
            'vars',
            'urca',
            'depmixS4',
            'rmgarch'
        ]
        
        for pkg in packages:
            try:
                importr(pkg)
            except Exception as e:
                print(f"Warning: Could not load R package '{pkg}': {e}")
    
    def _load_scripts(self):
        """Load custom R scripts from r_analytics directory.
        
        Note: These scripts are optional. If not found, the corresponding
        R functions must be called through R packages directly (e.g.,
        rugarch::ugarchfit, forecast::auto.arima) rather than through
        custom wrapper scripts.
        """
        if not os.path.exists(self.r_scripts_path):
            print(f"Warning: R scripts directory not found: {self.r_scripts_path}")
            print("  R functions should be called through R packages directly.")
            return
        
        scripts = [
            'timeseries_models.R',
            'risk_modeling.R',
            'econometrics.R',
            'portfolio_optimization.R',
            'performance_analytics.R',
            'regime_detection.R',
            'correlation_analysis.R'
        ]
        
        missing_scripts = []
        for script in scripts:
            script_path = os.path.join(self.r_scripts_path, script)
            if os.path.exists(script_path):
                try:
                    self.r.source(script_path)
                    self._loaded_scripts.append(script)
                except Exception as e:
                    print(f"Warning: Could not load R script '{script}': {e}")
            else:
                missing_scripts.append(script)
        
        if missing_scripts:
            print(f"Warning: {len(missing_scripts)} R script(s) not found: {', '.join(missing_scripts)}")
            print("  These functions must be called through R packages directly.")
    
    def call(self, func_name: str, **kwargs) -> Any:
        """
        Call an R function with keyword arguments.
        
        Parameters
        ----------
        func_name : str
            Name of R function to call.
        **kwargs
            Arguments to pass to R function.
        
        Returns
        -------
        Any
            Result from R function, converted to Python types.
        """
        r_func = self.r[func_name]
        
        # Convert arguments to R types
        r_args = {}
        for key, value in kwargs.items():
            if isinstance(value, np.ndarray):
                r_args[key] = ro.FloatVector(value.flatten())
            elif isinstance(value, list):
                r_args[key] = ro.FloatVector(value)
            else:
                r_args[key] = value
        
        # Call function
        result = r_func(**r_args)
        
        # Convert result to Python
        return self._convert_r_to_python(result)
    
    def _convert_r_to_python(self, r_obj) -> Any:
        """Convert R object to Python types."""
        with localconverter(ro.default_converter + numpy2ri.converter):
            if isinstance(r_obj, ro.vectors.ListVector):
                # Named list -> dict
                return {
                    str(name): self._convert_r_to_python(r_obj.rx2(name))
                    for name in r_obj.names
                }
            elif isinstance(r_obj, ro.vectors.FloatVector):
                arr = np.array(r_obj)
                return arr.item() if arr.size == 1 else arr
            elif isinstance(r_obj, ro.vectors.IntVector):
                arr = np.array(r_obj, dtype=int)
                return arr.item() if arr.size == 1 else arr
            elif isinstance(r_obj, ro.vectors.StrVector):
                return list(r_obj) if len(r_obj) > 1 else str(r_obj[0])
            elif isinstance(r_obj, ro.vectors.BoolVector):
                arr = np.array(r_obj, dtype=bool)
                return arr.item() if arr.size == 1 else arr
            elif isinstance(r_obj, ro.vectors.Matrix):
                return np.array(r_obj)
            else:
                try:
                    return np.array(r_obj)
                except Exception:
                    return r_obj


# Global R session (lazy initialization)
_r_session: Optional[RSession] = None


def get_r_session() -> RSession:
    """Get or create the global R session."""
    global _r_session
    if _r_session is None:
        _r_session = RSession()
    return _r_session


# =============================================================================
# TIME SERIES MODELS (R)
# =============================================================================

def fit_arima(returns: np.ndarray, max_p: int = 5, max_q: int = 5, 
              criterion: str = "aic") -> Dict[str, Any]:
    """
    Fit ARIMA model using R's auto.arima.
    
    Parameters
    ----------
    returns : np.ndarray
        Return series.
    max_p : int
        Maximum AR order.
    max_q : int
        Maximum MA order.
    criterion : str
        Information criterion ("aic" or "bic").
    
    Returns
    -------
    dict
        Fitted model results with order, AIC, forecasts.
    """
    r = get_r_session()
    return r.call('fit_arima', returns=returns, max_p=max_p, 
                  max_q=max_q, criterion=criterion)


def forecast_arima(model_result: Dict, horizon: int = 10, 
                   confidence_level: float = 0.95) -> Dict[str, Any]:
    """
    Forecast using fitted ARIMA model.
    
    Parameters
    ----------
    model_result : dict
        Result from fit_arima.
    horizon : int
        Forecast horizon.
    confidence_level : float
        Confidence interval level.
    
    Returns
    -------
    dict
        Point forecasts and confidence intervals.
    """
    r = get_r_session()
    return r.call('forecast_arima', model=model_result, horizon=horizon,
                  confidence_level=confidence_level)


def test_stationarity(data: np.ndarray) -> Dict[str, Any]:
    """
    Test for stationarity using ADF, PP, and KPSS tests.
    
    Parameters
    ----------
    data : np.ndarray
        Time series data.
    
    Returns
    -------
    dict
        Test results with p-values and recommendation.
    """
    r = get_r_session()
    return r.call('test_stationarity', data=data)


# =============================================================================
# RISK MODELS (R)
# =============================================================================

def fit_garch(returns: np.ndarray, p: int = 1, q: int = 1, 
              dist: str = "std") -> Dict[str, Any]:
    """
    Fit GARCH(p,q) model using R's rugarch.
    
    Parameters
    ----------
    returns : np.ndarray
        Return series.
    p : int
        ARCH order.
    q : int
        GARCH order.
    dist : str
        Innovation distribution ("norm", "std", "ged").
    
    Returns
    -------
    dict
        Model coefficients, persistence, conditional volatility.
    """
    r = get_r_session()
    return r.call('fit_garch', returns=returns, p=p, q=q, dist=dist)


def fit_egarch(returns: np.ndarray, dist: str = "std") -> Dict[str, Any]:
    """
    Fit EGARCH model for asymmetric volatility.
    
    Parameters
    ----------
    returns : np.ndarray
        Return series.
    dist : str
        Innovation distribution.
    
    Returns
    -------
    dict
        Model with asymmetry coefficient (leverage effect).
    """
    r = get_r_session()
    return r.call('fit_egarch', returns=returns, dist=dist)


def fit_copula(returns1: np.ndarray, returns2: np.ndarray, 
               family: str = "t") -> Dict[str, Any]:
    """
    Fit copula model to capture dependence structure.
    
    Parameters
    ----------
    returns1 : np.ndarray
        First asset returns.
    returns2 : np.ndarray
        Second asset returns.
    family : str
        Copula family ("gaussian", "t", "clayton", "gumbel", "frank").
    
    Returns
    -------
    dict
        Copula parameters and tail dependence.
    """
    r = get_r_session()
    return r.call('fit_copula', returns1=returns1, returns2=returns2, 
                  family=family)


def fit_gpd(losses: np.ndarray, threshold: float = 0.95) -> Dict[str, Any]:
    """
    Fit Generalized Pareto Distribution for tail risk.
    
    Parameters
    ----------
    losses : np.ndarray
        Loss series (positive = loss).
    threshold : float
        Threshold for exceedances (quantile if < 1).
    
    Returns
    -------
    dict
        GPD parameters, VaR, CVaR estimates.
    """
    r = get_r_session()
    return r.call('fit_gpd', losses=losses, threshold=threshold)


# =============================================================================
# ECONOMETRICS (R)
# =============================================================================

def test_cointegration(y: np.ndarray, x: np.ndarray) -> Dict[str, Any]:
    """
    Test for cointegration using Engle-Granger method.
    
    Parameters
    ----------
    y : np.ndarray
        Dependent series.
    x : np.ndarray
        Independent series.
    
    Returns
    -------
    dict
        Cointegration test results, hedge ratio, half-life.
    """
    r = get_r_session()
    return r.call('test_cointegration', y=y, x=x)


def granger_causality(y: np.ndarray, x: np.ndarray, 
                      max_lag: int = 5) -> Dict[str, Any]:
    """
    Test Granger causality between two series.
    
    Parameters
    ----------
    y : np.ndarray
        Target series (effect).
    x : np.ndarray
        Potential cause series.
    max_lag : int
        Maximum lag to test.
    
    Returns
    -------
    dict
        Causality test results by lag.
    """
    r = get_r_session()
    return r.call('granger_causality', y=y, x=x, max_lag=max_lag)


def fit_var(data: np.ndarray, p: Union[int, str] = "auto") -> Dict[str, Any]:
    """
    Fit Vector Autoregression model.
    
    Parameters
    ----------
    data : np.ndarray
        Matrix of time series (T x N).
    p : int or "auto"
        VAR lag order.
    
    Returns
    -------
    dict
        VAR model, stability diagnostics, forecasts.
    """
    r = get_r_session()
    return r.call('fit_var', data=data, p=p)


# =============================================================================
# PORTFOLIO OPTIMIZATION (R)
# =============================================================================

def mean_variance_optimize(returns: np.ndarray, target_return: Optional[float] = None,
                           short_selling: bool = False) -> Dict[str, Any]:
    """
    Markowitz mean-variance optimization.
    
    Parameters
    ----------
    returns : np.ndarray
        Asset returns matrix (T x N).
    target_return : float, optional
        Target portfolio return (if None, maximizes Sharpe).
    short_selling : bool
        Allow short selling.
    
    Returns
    -------
    dict
        Optimal weights, expected return, volatility, Sharpe ratio.
    """
    r = get_r_session()
    kwargs = {'returns': returns, 'short_selling': short_selling}
    if target_return is not None:
        kwargs['target_return'] = target_return
    return r.call('mean_variance_optimize', **kwargs)


def risk_parity(returns: np.ndarray) -> Dict[str, Any]:
    """
    Risk parity (equal risk contribution) portfolio.
    
    Parameters
    ----------
    returns : np.ndarray
        Asset returns matrix.
    
    Returns
    -------
    dict
        Weights and risk contributions.
    """
    r = get_r_session()
    return r.call('risk_parity', returns=returns)


def black_litterman(returns: np.ndarray, views_P: Optional[np.ndarray] = None,
                    views_Q: Optional[np.ndarray] = None,
                    views_confidence: float = 0.5) -> Dict[str, Any]:
    """
    Black-Litterman portfolio optimization.
    
    Parameters
    ----------
    returns : np.ndarray
        Asset returns matrix.
    views_P : np.ndarray, optional
        Pick matrix (which assets views are about).
    views_Q : np.ndarray, optional
        View values (expected returns).
    views_confidence : float
        Confidence in views (0-1).
    
    Returns
    -------
    dict
        Optimal weights incorporating views.
    """
    r = get_r_session()
    kwargs = {'returns': returns, 'views_confidence': views_confidence}
    if views_P is not None:
        kwargs['views_P'] = views_P
    if views_Q is not None:
        kwargs['views_Q'] = views_Q
    return r.call('black_litterman', **kwargs)


# =============================================================================
# REGIME DETECTION (R)
# =============================================================================

def fit_hmm(returns: np.ndarray, n_states: int = 2) -> Dict[str, Any]:
    """
    Fit Hidden Markov Model for regime detection.
    
    Parameters
    ----------
    returns : np.ndarray
        Return series.
    n_states : int
        Number of hidden states/regimes.
    
    Returns
    -------
    dict
        State assignments, transition matrix, regime statistics.
    """
    r = get_r_session()
    return r.call('fit_hmm', returns=returns, n_states=n_states)


def markov_regime_switching(returns: np.ndarray) -> Dict[str, Any]:
    """
    Fit bull/bear market regime switching model.
    
    Parameters
    ----------
    returns : np.ndarray
        Return series.
    
    Returns
    -------
    dict
        Bull/bear regimes, transition probabilities, duration.
    """
    r = get_r_session()
    return r.call('markov_regime_switching', returns=returns)


# =============================================================================
# PERFORMANCE ANALYTICS (R)
# =============================================================================

def calculate_performance_metrics(returns: np.ndarray, 
                                  risk_free: float = 0.02) -> Dict[str, Any]:
    """
    Calculate comprehensive performance metrics.
    
    Parameters
    ----------
    returns : np.ndarray
        Return series.
    risk_free : float
        Annualized risk-free rate.
    
    Returns
    -------
    dict
        Return metrics, Sharpe, Sortino, drawdowns, etc.
    """
    r = get_r_session()
    return r.call('calculate_return_metrics', returns=returns, 
                  risk_free=risk_free)


def drawdown_analysis(returns: np.ndarray, top_n: int = 5) -> Dict[str, Any]:
    """
    Comprehensive drawdown analysis.
    
    Parameters
    ----------
    returns : np.ndarray
        Return series.
    top_n : int
        Number of top drawdowns to report.
    
    Returns
    -------
    dict
        Max drawdown, average drawdown, drawdown periods.
    """
    r = get_r_session()
    return r.call('drawdown_analysis', returns=returns, top_n=top_n)


# =============================================================================
# CORRELATION ANALYSIS (R)
# =============================================================================

def fit_dcc(returns: np.ndarray) -> Dict[str, Any]:
    """
    Fit DCC-GARCH model for time-varying correlations.
    
    Parameters
    ----------
    returns : np.ndarray
        Asset returns matrix.
    
    Returns
    -------
    dict
        Time-varying correlations and covariances.
    """
    r = get_r_session()
    return r.call('fit_dcc', returns=returns)


def tail_dependence(x: np.ndarray, y: np.ndarray, 
                    quantile: float = 0.05) -> Dict[str, Any]:
    """
    Estimate tail dependence coefficients.
    
    Parameters
    ----------
    x : np.ndarray
        First return series.
    y : np.ndarray
        Second return series.
    quantile : float
        Tail quantile.
    
    Returns
    -------
    dict
        Lower and upper tail dependence.
    """
    r = get_r_session()
    return r.call('tail_dependence', x=x, y=y, quantile=quantile)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    if R_AVAILABLE:
        # Test R integration
        print("Testing Python-R Bridge...")
        
        # Generate sample data
        np.random.seed(42)
        returns = np.random.normal(0.0004, 0.015, 500)
        
        # Test stationarity
        print("\n1. Stationarity Test:")
        stat = test_stationarity(returns)
        print(f"   Is stationary: {stat.get('consensus_stationary', 'N/A')}")
        
        # Test GARCH
        print("\n2. GARCH(1,1) Model:")
        try:
            garch = fit_garch(returns)
            print(f"   Persistence: {garch.get('persistence', 'N/A'):.4f}")
            print(f"   Half-life: {garch.get('half_life', 'N/A'):.1f} days")
        except Exception as e:
            print(f"   Error: {e}")
        
        # Test HMM
        print("\n3. Hidden Markov Model:")
        try:
            hmm = fit_hmm(returns, n_states=2)
            print(f"   Current state: {hmm.get('current_state', 'N/A')}")
        except Exception as e:
            print(f"   Error: {e}")
        
        print("\nPython-R Bridge test complete!")
    else:
        print("R integration not available. Install rpy2 to enable.")
