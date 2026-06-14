"""
GIGA SYSTEM - Python-R Interface Bridge
Greek Intelligence for Global Analysis

This module provides seamless integration between Python and R environments
using rpy2, enabling the best of both worlds: Python's speed + R's statistics.

Core Functionality:
- Initialize R environment with required packages
- Execute R code from Python with error handling
- Convert data between Python (Polars/NumPy) and R formats
- Provide Pythonic wrappers for R statistical models

Performance Targets:
- Data conversion: <10ms for 100K rows
- R model fitting: <100ms for GARCH/ARIMA
- Memory efficiency: Zero-copy when possible
"""

#  ️ PHASE 2 WARNING: AIR-GAP VIOLATION
# This module represents a "Bridge Trap" (Failure Category 1).
# Research code (R) should NOT be run inside Live Python processes.
# This module is strictly for RESEARCH/BACKTESTING environments.
# DO NOT DEPLOY TO LIVE EXECUTION.

import os
import sys
import warnings
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import numpy as np
import polars as pl
import pandas as pd

# Suppress R warnings that are not critical
warnings.filterwarnings("ignore", category=UserWarning, module="rpy2")

# Configure R_HOME - prefer environment variable, then auto-detect
if 'R_HOME' not in os.environ:
    # Try common R installation paths
    _r_paths = [
        r'C:\Program Files\R',
        r'C:\Program Files (x86)\R',
        '/usr/lib/R',
        '/usr/local/lib/R',
    ]
    for _base in _r_paths:
        if os.path.isdir(_base):
            # Find latest R version in directory
            try:
                _versions = sorted(os.listdir(_base), reverse=True)
                for _v in _versions:
                    _candidate = os.path.join(_base, _v)
                    if os.path.isdir(_candidate) and os.path.exists(os.path.join(_candidate, 'bin')):
                        os.environ['R_HOME'] = _candidate
                        break
            except OSError:
                pass
            if 'R_HOME' in os.environ:
                break

try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri, numpy2ri
    from rpy2.robjects.packages import importr, isinstalled
    from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
    import rpy2.rinterface as rinterface
    
    # Suppress R output to keep Python console clean
    rpy2_logger.setLevel(40)  # Only show errors
    
    R_AVAILABLE = True
except (ImportError, OSError, Exception) as e:
    R_AVAILABLE = False
    ro = None


class RInterface:
    """
    Main interface class for Python-R communication.
    
    Handles R environment initialization, package management,
    data conversion, and execution of R scripts.
    """
    
    def __init__(self):
        """Initialize R environment and load required packages."""
        self.r_available = R_AVAILABLE
        self.r_packages = {}
        self.initialized = False
        
        if self.r_available:
            self._initialize_r()
    
    def _initialize_r(self):
        """Initialize R environment and load essential packages."""
        try:
            # Activate pandas and numpy conversion
            pandas2ri.activate()
            numpy2ri.activate()
            
            # Set R options for better performance
            ro.r("""
            options(warn = -1)  # Suppress warnings
            options(digits = 10)  # High precision
            """)
            
            # Load base R packages
            self._load_r_package('base')
            self._load_r_package('stats')
            self._load_r_package('utils')
            
            self.initialized = True
            print("R interface initialized successfully")
            
        except Exception as e:
            print(f"Failed to initialize R interface: {e}")
            self.r_available = False
    
    def _load_r_package(self, package_name: str) -> bool:
        """
        Load an R package, installing if necessary.
        
        Args:
            package_name: Name of R package to load
            
        Returns:
            bool: True if package loaded successfully
        """
        if not self.r_available:
            return False
        
        try:
            # Check if package is installed
            if not isinstalled(package_name):
                print(f"Installing R package: {package_name}")
                utils = importr('utils')
                utils.install_packages(package_name)
            
            # Load the package
            self.r_packages[package_name] = importr(package_name)
            return True
            
        except Exception as e:
            print(f"Failed to load R package {package_name}: {e}")
            return False
    
    def ensure_packages(self, packages: List[str]) -> bool:
        """
        Ensure all required R packages are loaded.
        
        Args:
            packages: List of required R package names
            
        Returns:
            bool: True if all packages loaded successfully
        """
        if not self.r_available:
            return False
        
        success = True
        for package in packages:
            if package not in self.r_packages:
                if not self._load_r_package(package):
                    success = False
        
        return success
    
    def execute_r_code(self, r_code: str) -> Any:
        """
        Execute R code and return results.
        
        Args:
            r_code: R code string to execute
            
        Returns:
            Results from R execution (converted to Python objects)
        """
        if not self.r_available:
            raise RuntimeError("R interface not available")
        
        try:
            result = ro.r(r_code)
            return result
        except Exception as e:
            raise RuntimeError(f"R execution failed: {e}")
    
    def execute_r_script(self, script_path: Union[str, Path]) -> Any:
        """
        Execute an R script file.
        
        Args:
            script_path: Path to R script file
            
        Returns:
            Results from R script execution
        """
        if not self.r_available:
            raise RuntimeError("R interface not available")
        
        script_path = Path(script_path)
        if not script_path.exists():
            raise FileNotFoundError(f"R script not found: {script_path}")
        
        return self.execute_r_code(f'source("{script_path}")')
    
    def polars_to_r(self, df: pl.DataFrame, name: str = "data") -> None:
        """
        Convert Polars DataFrame to R data.frame.
        
        Args:
            df: Polars DataFrame
            name: Variable name in R environment
        """
        if not self.r_available:
            raise RuntimeError("R interface not available")
        
        # Convert Polars to Pandas first (rpy2 doesn't support Polars directly)
        pandas_df = df.to_pandas()
        
        # Convert to R
        ro.globalenv[name] = pandas_df
    
    def r_to_polars(self, r_var_name: str) -> pl.DataFrame:
        """
        Convert R data.frame to Polars DataFrame.
        
        Args:
            r_var_name: Name of R variable containing data.frame
            
        Returns:
            pl.DataFrame: Converted data
        """
        if not self.r_available:
            raise RuntimeError("R interface not available")
        
        # Get R data.frame
        r_df = ro.globalenv[r_var_name]
        
        # Convert to Pandas first
        pandas_df = pandas2ri.rpy2py(r_df)
        
        # Convert to Polars
        return pl.from_pandas(pandas_df)
    
    def numpy_to_r(self, array: np.ndarray, name: str = "data") -> None:
        """
        Convert NumPy array to R vector/matrix.
        
        Args:
            array: NumPy array
            name: Variable name in R environment
        """
        if not self.r_available:
            raise RuntimeError("R interface not available")
        
        ro.globalenv[name] = array
    
    def r_to_numpy(self, r_var_name: str) -> np.ndarray:
        """
        Convert R vector/matrix to NumPy array.
        
        Args:
            r_var_name: Name of R variable
            
        Returns:
            np.ndarray: Converted array
        """
        if not self.r_available:
            raise RuntimeError("R interface not available")
        
        r_obj = ro.globalenv[r_var_name]
        return numpy2ri.rpy2py(r_obj)


# Global R interface instance
r_interface = RInterface()


def check_r_availability() -> bool:
    """Check if R interface is available and working."""
    return r_interface.r_available and r_interface.initialized


def get_r_version() -> Optional[str]:
    """Get R version string."""
    if not check_r_availability():
        return None
    
    try:
        version = r_interface.execute_r_code("R.version.string")
        return str(version[0])
    except Exception:
        return None


def install_required_r_packages():
    """Install all R packages required by GIGA System."""
    if not check_r_availability():
        print("R interface not available - cannot install packages")
        return False
    
    required_packages = [
        'forecast',      # ARIMA, exponential smoothing
        'rugarch',       # GARCH models (volatility)
        'vars',          # Vector AutoRegression
        'urca',          # Unit root tests, cointegration
        'PerformanceAnalytics',  # Sharpe, Sortino, drawdowns
        'quantmod',      # Financial data, indicators
        'TTR',           # Technical Trading Rules
        'MASS',          # Multivariate statistics
        'copula',        # Copula models (tail dependence)
        'evd',           # Extreme Value Distributions
    ]
    
    print("Installing required R packages...")
    success = r_interface.ensure_packages(required_packages)
    
    if success:
        print("All R packages installed successfully")
    else:
        print("Some R packages failed to install")
    
    return success


# Initialize on import
if __name__ == "__main__":
    # Test R interface
    if check_r_availability():
        print(f"R Version: {get_r_version()}")
        print("R interface is working correctly")
    else:
        print("R interface is not available")