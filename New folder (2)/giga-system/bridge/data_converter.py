"""
GIGA SYSTEM - Data Format Converter
Greek Intelligence for Global Analysis

Efficient conversion between Python data formats (Polars, NumPy, Pandas)
and R data formats (data.frame, matrix, vector) with zero-copy optimization.

Key Features:
- Polars ↔ R data.frame conversion
- NumPy ↔ R matrix conversion  
- Memory-efficient operations (minimal copying)
- Type preservation and validation
- Error handling with informative messages

Performance Targets:
- 100K rows: <10ms conversion time
- Memory overhead: <2x original data size
- Type fidelity: 100% preservation for numeric data
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import warnings

import numpy as np
import polars as pl
import pandas as pd

from .rpy2_interface import r_interface, check_r_availability

# Type aliases for clarity
PolarsDF = pl.DataFrame
PandasDF = pd.DataFrame
NumpyArray = np.ndarray


class DataConverter:
    """
    High-performance data converter between Python and R formats.
    
    Optimized for quantitative finance data (time series, price matrices,
    return calculations) with focus on numerical precision and speed.
    """
    
    def __init__(self):
        """Initialize converter with R interface check."""
        self.r_available = check_r_availability()
        if not self.r_available:
            warnings.warn("R interface not available - conversions will fail")
    
    def polars_to_r_dataframe(self, 
                             df: PolarsDF, 
                             r_name: str = "giga_data",
                             preserve_types: bool = True) -> bool:
        """
        Convert Polars DataFrame to R data.frame with type preservation.
        
        Args:
            df: Source Polars DataFrame
            r_name: Variable name in R global environment
            preserve_types: Maintain data types during conversion
            
        Returns:
            bool: Success status
        """
        if not self.r_available:
            return False
        
        try:
            # Type optimization for financial data
            if preserve_types:
                df = self._optimize_financial_types(df)
            
            # Convert Polars -> Pandas -> R (most reliable path)
            pandas_df = df.to_pandas()
            
            # Handle financial data specifics
            pandas_df = self._prepare_pandas_for_r(pandas_df)
            
            # Send to R environment
            r_interface.polars_to_r(df, r_name)
            
            return True
            
        except Exception as e:
            warnings.warn(f"Polars to R conversion failed: {e}")
            return False
    
    def r_dataframe_to_polars(self, 
                             r_name: str, 
                             optimize_types: bool = True) -> Optional[PolarsDF]:
        """
        Convert R data.frame to Polars DataFrame with type optimization.
        
        Args:
            r_name: R variable name containing data.frame
            optimize_types: Apply financial data type optimizations
            
        Returns:
            PolarsDF or None: Converted DataFrame or None if failed
        """
        if not self.r_available:
            return None
        
        try:
            # Get data from R
            polars_df = r_interface.r_to_polars(r_name)
            
            # Apply financial data optimizations
            if optimize_types:
                polars_df = self._optimize_financial_types(polars_df)
            
            return polars_df
            
        except Exception as e:
            warnings.warn(f"R to Polars conversion failed: {e}")
            return None
    
    def numpy_to_r_matrix(self, 
                         array: NumpyArray, 
                         r_name: str = "giga_matrix") -> bool:
        """
        Convert NumPy array to R matrix with dimension preservation.
        
        Args:
            array: Source NumPy array
            r_name: Variable name in R environment
            
        Returns:
            bool: Success status
        """
        if not self.r_available:
            return False
        
        try:
            # Validate array for financial data
            if not self._validate_financial_array(array):
                warnings.warn("Array validation failed - proceeding anyway")
            
            # Send to R
            r_interface.numpy_to_r(array, r_name)
            
            return True
            
        except Exception as e:
            warnings.warn(f"NumPy to R conversion failed: {e}")
            return False
    
    def r_matrix_to_numpy(self, r_name: str) -> Optional[NumpyArray]:
        """
        Convert R matrix/vector to NumPy array.
        
        Args:
            r_name: R variable name
            
        Returns:
            NumpyArray or None: Converted array or None if failed
        """
        if not self.r_available:
            return None
        
        try:
            array = r_interface.r_to_numpy(r_name)
            return array
            
        except Exception as e:
            warnings.warn(f"R to NumPy conversion failed: {e}")
            return None
    
    def prepare_returns_for_garch(self, 
                                 returns: Union[PolarsDF, NumpyArray],
                                 r_name: str = "returns_data") -> bool:
        """
        Prepare return data specifically for GARCH modeling in R.
        
        Handles common issues:
        - Remove NaN/infinite values
        - Ensure proper time series structure
        - Scale returns if necessary
        
        Args:
            returns: Return data (DataFrame with time index or 1D array)
            r_name: R variable name for the prepared data
            
        Returns:
            bool: Success status
        """
        if not self.r_available:
            return False
        
        try:
            if isinstance(returns, pl.DataFrame):
                # Polars DataFrame case
                cleaned_returns = (returns
                    .drop_nulls()
                    .filter(pl.col(returns.columns[1]).is_finite())  # Assume price column is second
                )
                return self.polars_to_r_dataframe(cleaned_returns, r_name)
                
            elif isinstance(returns, np.ndarray):
                # NumPy array case
                cleaned_returns = returns[np.isfinite(returns)]
                return self.numpy_to_r_matrix(cleaned_returns.reshape(-1, 1), r_name)
            
            else:
                warnings.warn("Unsupported data type for GARCH preparation")
                return False
                
        except Exception as e:
            warnings.warn(f"GARCH data preparation failed: {e}")
            return False
    
    def get_r_model_results(self, 
                           model_name: str,
                           extract_params: List[str] = None) -> Dict[str, Any]:
        """
        Extract results from R model objects (GARCH, ARIMA, etc.).
        
        Args:
            model_name: R variable name containing fitted model
            extract_params: List of parameters to extract
            
        Returns:
            Dict with extracted model results
        """
        if not self.r_available:
            return {}
        
        try:
            results = {}
            
            # Common extractions for financial models
            if extract_params is None:
                extract_params = ['coef', 'fitted', 'residuals', 'logLik']
            
            for param in extract_params:
                try:
                    r_code = f"{param}({model_name})"
                    result = r_interface.execute_r_code(r_code)
                    
                    # Convert to Python objects
                    if hasattr(result, '__len__') and len(result) > 1:
                        results[param] = np.array(result)
                    else:
                        results[param] = float(result[0]) if len(result) == 1 else result
                        
                except Exception as e:
                    warnings.warn(f"Failed to extract {param}: {e}")
                    continue
            
            return results
            
        except Exception as e:
            warnings.warn(f"Model results extraction failed: {e}")
            return {}
    
    def _optimize_financial_types(self, df: PolarsDF) -> PolarsDF:
        """
        Optimize data types for financial time series data.
        
        - Dates -> proper datetime
        - Prices -> Float64 (high precision)
        - Volume -> Int64
        - Returns -> Float64
        """
        try:
            # Common financial column patterns
            datetime_patterns = ['date', 'time', 'timestamp']
            price_patterns = ['price', 'close', 'open', 'high', 'low', 'bid', 'ask']
            volume_patterns = ['volume', 'size', 'quantity']
            return_patterns = ['return', 'pnl', 'change']
            
            for col in df.columns:
                col_lower = col.lower()
                
                # DateTime columns
                if any(pattern in col_lower for pattern in datetime_patterns):
                    try:
                        df = df.with_columns(pl.col(col).str.strptime(pl.Datetime))
                    except Exception as e:
                        warnings.warn(f"DateTime conversion failed for column '{col}': {e}")
                
                # Price columns (high precision)
                elif any(pattern in col_lower for pattern in price_patterns):
                    df = df.with_columns(pl.col(col).cast(pl.Float64))
                
                # Volume columns (integers)
                elif any(pattern in col_lower for pattern in volume_patterns):
                    try:
                        df = df.with_columns(pl.col(col).cast(pl.Int64))
                    except Exception as e:
                        warnings.warn(f"Int64 cast failed for column '{col}', using Float64: {e}")
                        df = df.with_columns(pl.col(col).cast(pl.Float64))
                
                # Return columns (high precision)
                elif any(pattern in col_lower for pattern in return_patterns):
                    df = df.with_columns(pl.col(col).cast(pl.Float64))
            
            return df
            
        except Exception as e:
            warnings.warn(f"Type optimization failed: {e}")
            return df
    
    def _prepare_pandas_for_r(self, df: PandasDF) -> PandasDF:
        """
        Prepare Pandas DataFrame for R conversion.
        
        Handles R-specific requirements:
        - Column names (no spaces, special chars)
        - Index handling
        - Missing values
        """
        # Clean column names for R compatibility
        df.columns = [col.replace(' ', '_').replace('-', '_').replace('.', '_') 
                     for col in df.columns]
        
        # Handle datetime index
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
        
        return df
    
    def _validate_financial_array(self, array: NumpyArray) -> bool:
        """
        Validate NumPy array for financial data quality.
        
        Checks:
        - No infinite values
        - Reasonable value ranges
        - Non-empty
        """
        if array.size == 0:
            return False
        
        if not np.isfinite(array).all():
            warnings.warn("Array contains infinite or NaN values")
            return False
        
        # Basic range checks for financial data
        if array.dtype == np.float64:
            if np.abs(array).max() > 1e6:  # Very large values might be problematic
                warnings.warn("Array contains unusually large values")
        
        return True


# Global converter instance
data_converter = DataConverter()


def polars_to_r(df: PolarsDF, r_name: str = "data") -> bool:
    """Convenience function for Polars to R conversion."""
    return data_converter.polars_to_r_dataframe(df, r_name)


def r_to_polars(r_name: str) -> Optional[PolarsDF]:
    """Convenience function for R to Polars conversion."""
    return data_converter.r_dataframe_to_polars(r_name)


def numpy_to_r(array: NumpyArray, r_name: str = "data") -> bool:
    """Convenience function for NumPy to R conversion."""
    return data_converter.numpy_to_r_matrix(array, r_name)


def r_to_numpy(r_name: str) -> Optional[NumpyArray]:
    """Convenience function for R to NumPy conversion."""
    return data_converter.r_matrix_to_numpy(r_name)