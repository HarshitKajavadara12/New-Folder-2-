"""
GIGA SYSTEM - Data Validators
Greek Intelligence for Global Analysis

Comprehensive validation functions for financial data quality.
Ensures data integrity across all system components with
performance-aware validation that doesn't slow down operations.

Key Features:
- Price data validation (OHLCV consistency)
- Return series validation (outliers, stationarity)
- Greek values validation (theoretical bounds)
- Portfolio validation (position limits, exposure)
- Real-time validation with minimal overhead

Performance Targets:
- Validate 100K rows in <100ms
- Memory efficient streaming validation
- Early exit on critical errors
"""

from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
import warnings

import numpy as np
import polars as pl
from scipy import stats


class ValidationResult:
    """Container for validation results."""
    
    def __init__(self, valid: bool = True, errors: List[str] = None, warnings: List[str] = None):
        self.valid = valid
        self.errors = errors or []
        self.warnings = warnings or []
    
    def add_error(self, message: str):
        """Add error message."""
        self.errors.append(message)
        self.valid = False
    
    def add_warning(self, message: str):
        """Add warning message."""
        self.warnings.append(message)
    
    def __bool__(self):
        """Return validation status."""
        return self.valid
    
    def summary(self) -> str:
        """Get validation summary."""
        if self.valid:
            result = "  Validation passed"
        else:
            result = "  Validation failed"
        
        if self.errors:
            result += f"\nErrors ({len(self.errors)}):\n"
            result += "\n".join([f"  - {err}" for err in self.errors])
        
        if self.warnings:
            result += f"\nWarnings ({len(self.warnings)}):\n"
            result += "\n".join([f"  - {warn}" for warn in self.warnings])
        
        return result


def validate_price_data(df: pl.DataFrame, 
                       required_columns: List[str] = None,
                       check_ohlc_consistency: bool = True,
                       check_volume: bool = True,
                       check_timestamps: bool = True) -> ValidationResult:
    """
    Validate market price data for common issues.
    
    Args:
        df: Price data DataFrame
        required_columns: Required column names
        check_ohlc_consistency: Validate OHLC relationships
        check_volume: Validate volume data
        check_timestamps: Check timestamp consistency
        
    Returns:
        ValidationResult with detailed feedback
    """
    result = ValidationResult()
    
    if df.is_empty():
        result.add_error("DataFrame is empty")
        return result
    
    # Check required columns
    if required_columns is None:
        required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        result.add_error(f"Missing required columns: {missing_columns}")
        return result
    
    # Check for null values
    null_counts = df.null_count()
    for col in required_columns:
        null_count = null_counts.select(col).item()
        if null_count > 0:
            result.add_error(f"Column '{col}' has {null_count} null values")
    
    # Price validation
    price_columns = ["open", "high", "low", "close"]
    for col in price_columns:
        if col in df.columns:
            # Check for non-positive prices
            negative_count = df.filter(pl.col(col) <= 0).height
            if negative_count > 0:
                result.add_error(f"Column '{col}' has {negative_count} non-positive values")
            
            # Check for extremely large values (potential data errors)
            if df[col].max() > 1e6:
                result.add_warning(f"Column '{col}' has very large values (>1M)")
    
    # OHLC consistency checks
    if check_ohlc_consistency and all(col in df.columns for col in price_columns):
        # High should be >= Open, Close
        high_low_violations = df.filter(
            (pl.col("high") < pl.col("open")) | 
            (pl.col("high") < pl.col("close"))
        ).height
        
        if high_low_violations > 0:
            result.add_error(f"Found {high_low_violations} rows where High < Open or High < Close")
        
        # Low should be <= Open, Close
        low_violations = df.filter(
            (pl.col("low") > pl.col("open")) | 
            (pl.col("low") > pl.col("close"))
        ).height
        
        if low_violations > 0:
            result.add_error(f"Found {low_violations} rows where Low > Open or Low > Close")
        
        # High should be >= Low
        high_low_inconsistent = df.filter(pl.col("high") < pl.col("low")).height
        if high_low_inconsistent > 0:
            result.add_error(f"Found {high_low_inconsistent} rows where High < Low")
    
    # Volume validation
    if check_volume and "volume" in df.columns:
        negative_volume = df.filter(pl.col("volume") < 0).height
        if negative_volume > 0:
            result.add_error(f"Found {negative_volume} rows with negative volume")
        
        # Check for zero volume (might be suspicious)
        zero_volume = df.filter(pl.col("volume") == 0).height
        if zero_volume > df.height * 0.1:  # More than 10% zero volume
            result.add_warning(f"High proportion of zero volume rows: {zero_volume}/{df.height}")
    
    # Timestamp validation
    if check_timestamps and "timestamp" in df.columns:
        # Check for duplicate timestamps
        unique_timestamps = df.select("timestamp").unique().height
        if unique_timestamps != df.height:
            duplicates = df.height - unique_timestamps
            result.add_error(f"Found {duplicates} duplicate timestamps")
        
        # Check timestamp ordering
        if not df.select("timestamp").is_sorted():
            result.add_warning("Timestamps are not sorted")
    
    return result


def validate_returns(returns: Union[np.ndarray, pl.DataFrame, pl.Series],
                    min_observations: int = 100,
                    max_return_threshold: float = 0.5,
                    check_stationarity: bool = True) -> ValidationResult:
    """
    Validate return series for statistical properties.
    
    Args:
        returns: Return series data
        min_observations: Minimum number of observations required
        max_return_threshold: Maximum reasonable return (50% default)
        check_stationarity: Check for stationarity
        
    Returns:
        ValidationResult with statistical tests
    """
    result = ValidationResult()
    
    # Convert to numpy array for analysis
    if isinstance(returns, pl.DataFrame):
        returns_array = returns.to_numpy()[:, -1]  # Last column
    elif isinstance(returns, pl.Series):
        returns_array = returns.to_numpy()
    else:
        returns_array = returns
    
    # Remove NaN values
    returns_clean = returns_array[~np.isnan(returns_array)]
    
    if len(returns_clean) == 0:
        result.add_error("All return values are NaN")
        return result
    
    # Check minimum observations
    if len(returns_clean) < min_observations:
        result.add_error(f"Insufficient observations: {len(returns_clean)} < {min_observations}")
    
    # Check for infinite values
    inf_count = np.sum(~np.isfinite(returns_clean))
    if inf_count > 0:
        result.add_error(f"Found {inf_count} infinite return values")
    
    # Check for extreme returns
    extreme_returns = np.sum(np.abs(returns_clean) > max_return_threshold)
    if extreme_returns > 0:
        result.add_warning(f"Found {extreme_returns} extreme returns (>{max_return_threshold*100}%)")
    
    # Statistical properties
    if len(returns_clean) > 10:
        mean_return = np.mean(returns_clean)
        std_return = np.std(returns_clean)
        skewness = stats.skew(returns_clean)
        kurtosis = stats.kurtosis(returns_clean)
        
        # Check for reasonable statistics
        if abs(mean_return) > 0.01:  # Daily return > 1%
            result.add_warning(f"Very high mean return: {mean_return:.4f}")
        
        if std_return > 0.1:  # Daily volatility > 10%
            result.add_warning(f"Very high volatility: {std_return:.4f}")
        
        # Check for excessive skewness/kurtosis
        if abs(skewness) > 5:
            result.add_warning(f"Extreme skewness: {skewness:.2f}")
        
        if kurtosis > 20:
            result.add_warning(f"Extreme kurtosis: {kurtosis:.2f}")
    
    # Stationarity test (simplified)
    if check_stationarity and len(returns_clean) > 50:
        try:
            # Simple stationarity check using rolling statistics
            window = min(20, len(returns_clean) // 4)
            rolling_mean = pd.Series(returns_clean).rolling(window).mean()
            rolling_std = pd.Series(returns_clean).rolling(window).std()
            
            # Check if rolling statistics are relatively stable
            mean_stability = np.std(rolling_mean.dropna()) / np.abs(np.mean(rolling_mean.dropna()))
            std_stability = np.std(rolling_std.dropna()) / np.mean(rolling_std.dropna())
            
            if mean_stability > 2.0:
                result.add_warning("Returns may not be stationary (unstable mean)")
            
            if std_stability > 1.0:
                result.add_warning("Returns may not be stationary (unstable variance)")
                
        except Exception as e:
            result.add_warning(f"Stationarity test failed: {e}")
    
    return result


def validate_greeks(greeks: Dict[str, float],
                   option_type: str = "call",
                   time_to_expiry: float = None,
                   moneyness: float = None) -> ValidationResult:
    """
    Validate Greek values for theoretical consistency.
    
    Args:
        greeks: Dictionary of Greek values
        option_type: "call" or "put"
        time_to_expiry: Time to expiry (for bounds checking)
        moneyness: S/K ratio (for bounds checking)
        
    Returns:
        ValidationResult with Greek validation
    """
    result = ValidationResult()
    
    # Required Greeks
    required_greeks = ["delta", "gamma", "theta", "vega", "rho"]
    missing_greeks = [g for g in required_greeks if g not in greeks]
    if missing_greeks:
        result.add_error(f"Missing Greeks: {missing_greeks}")
    
    # Delta validation
    if "delta" in greeks:
        delta = greeks["delta"]
        
        if option_type.lower() == "call":
            if not (0 <= delta <= 1):
                result.add_error(f"Call delta out of bounds [0,1]: {delta}")
        elif option_type.lower() == "put":
            if not (-1 <= delta <= 0):
                result.add_error(f"Put delta out of bounds [-1,0]: {delta}")
    
    # Gamma validation
    if "gamma" in greeks:
        gamma = greeks["gamma"]
        
        if gamma < 0:
            result.add_error(f"Gamma cannot be negative: {gamma}")
        
        if gamma > 10:
            result.add_warning(f"Very high gamma value: {gamma}")
    
    # Theta validation
    if "theta" in greeks:
        theta = greeks["theta"]
        
        # For long options, theta should be negative (time decay)
        if theta > 0:
            result.add_warning(f"Positive theta unusual for long options: {theta}")
        
        if abs(theta) > 5:
            result.add_warning(f"Very high theta magnitude: {theta}")
    
    # Vega validation
    if "vega" in greeks:
        vega = greeks["vega"]
        
        if vega < 0:
            result.add_error(f"Vega cannot be negative: {vega}")
        
        if vega > 100:
            result.add_warning(f"Very high vega value: {vega}")
    
    # Rho validation
    if "rho" in greeks:
        rho = greeks["rho"]
        
        if abs(rho) > 100:
            result.add_warning(f"Very high rho magnitude: {abs(rho)}")
    
    # Cross-Greek validation
    if "gamma" in greeks and "vega" in greeks and time_to_expiry:
        # Gamma and Vega relationship
        gamma = greeks["gamma"]
        vega = greeks["vega"]
        
        # For ATM options, there's a relationship between gamma and vega
        if moneyness and 0.9 <= moneyness <= 1.1:  # Near ATM
            expected_gamma_vega_ratio = 1 / (time_to_expiry * 365)  # Rough approximation
            actual_ratio = gamma / vega if vega != 0 else 0
            
            if abs(actual_ratio - expected_gamma_vega_ratio) > expected_gamma_vega_ratio * 2:
                result.add_warning("Gamma-Vega relationship seems inconsistent")
    
    return result


def validate_portfolio(positions: pl.DataFrame,
                      max_single_position: float = 0.1,
                      max_sector_exposure: float = 0.3,
                      max_gross_exposure: float = 2.0) -> ValidationResult:
    """
    Validate portfolio positions for risk management.
    
    Args:
        positions: Portfolio positions DataFrame
        max_single_position: Maximum single position size (as fraction)
        max_sector_exposure: Maximum sector exposure
        max_gross_exposure: Maximum gross exposure
        
    Returns:
        ValidationResult with risk checks
    """
    result = ValidationResult()
    
    if positions.is_empty():
        result.add_error("Portfolio is empty")
        return result
    
    # Required columns
    required_columns = ["symbol", "quantity", "current_price"]
    missing_columns = [col for col in required_columns if col not in positions.columns]
    if missing_columns:
        result.add_error(f"Missing required columns: {missing_columns}")
        return result
    
    # Calculate position values
    try:
        positions_with_value = positions.with_columns([
            (pl.col("quantity") * pl.col("current_price")).alias("position_value")
        ])
        
        total_portfolio_value = positions_with_value["position_value"].abs().sum()
        
        if total_portfolio_value == 0:
            result.add_error("Total portfolio value is zero")
            return result
        
        # Check individual position sizes
        position_weights = positions_with_value.with_columns([
            (pl.col("position_value").abs() / total_portfolio_value).alias("weight")
        ])
        
        max_position_weight = position_weights["weight"].max()
        if max_position_weight > max_single_position:
            result.add_warning(f"Large single position: {max_position_weight:.1%} > {max_single_position:.1%}")
        
        # Check gross exposure
        gross_exposure = total_portfolio_value / positions_with_value["position_value"].sum()
        if abs(gross_exposure) > max_gross_exposure:
            result.add_warning(f"High gross exposure: {gross_exposure:.1f}x > {max_gross_exposure}x")
        
        # Check for concentration risk
        unique_symbols = positions["symbol"].unique().len()
        if unique_symbols < 5:
            result.add_warning(f"Low diversification: only {unique_symbols} unique positions")
        
    except Exception as e:
        result.add_error(f"Portfolio validation failed: {e}")
    
    return result


def validate_option_chain(options: pl.DataFrame) -> ValidationResult:
    """
    Validate options chain data for consistency.
    
    Args:
        options: Options chain DataFrame
        
    Returns:
        ValidationResult with options-specific checks
    """
    result = ValidationResult()
    
    if options.is_empty():
        result.add_error("Options chain is empty")
        return result
    
    # Required columns for options
    required_columns = ["strike", "expiry", "option_type", "price", "implied_volatility"]
    missing_columns = [col for col in required_columns if col not in options.columns]
    if missing_columns:
        result.add_error(f"Missing required columns: {missing_columns}")
        return result
    
    # Validate option types
    valid_option_types = {"CALL", "PUT", "call", "put"}
    invalid_types = options.filter(
        ~pl.col("option_type").is_in(valid_option_types)
    ).height
    
    if invalid_types > 0:
        result.add_error(f"Found {invalid_types} rows with invalid option_type")
    
    # Validate strikes are positive
    negative_strikes = options.filter(pl.col("strike") <= 0).height
    if negative_strikes > 0:
        result.add_error(f"Found {negative_strikes} non-positive strike prices")
    
    # Validate implied volatility
    if "implied_volatility" in options.columns:
        negative_iv = options.filter(pl.col("implied_volatility") <= 0).height
        if negative_iv > 0:
            result.add_error(f"Found {negative_iv} non-positive implied volatilities")
        
        extreme_iv = options.filter(pl.col("implied_volatility") > 5).height  # 500%
        if extreme_iv > 0:
            result.add_warning(f"Found {extreme_iv} extremely high implied volatilities (>500%)")
    
    # Check for reasonable Greeks bounds if present
    greek_columns = ["delta", "gamma", "theta", "vega", "rho"]
    for greek in greek_columns:
        if greek in options.columns:
            greek_result = validate_greeks(
                {greek: options[greek].mean()},  # Use mean for validation
                "call"  # Assume call for validation
            )
            if not greek_result.valid:
                result.add_warning(f"Greek validation issues for {greek}")
    
    return result


# Convenience functions
def quick_validate_price_data(df: pl.DataFrame) -> bool:
    """Quick validation of price data - returns True/False."""
    return validate_price_data(df).valid


def quick_validate_returns(returns: np.ndarray) -> bool:
    """Quick validation of return series - returns True/False."""
    return validate_returns(returns).valid


def quick_validate_greeks(greeks: Dict[str, float]) -> bool:
    """Quick validation of Greek values - returns True/False."""
    return validate_greeks(greeks).valid


# Usage examples
if __name__ == "__main__":
    # Example price data validation
    sample_data = pl.DataFrame({
        "timestamp": ["2023-01-01", "2023-01-02", "2023-01-03"],
        "open": [100.0, 101.0, 102.0],
        "high": [105.0, 106.0, 107.0],
        "low": [98.0, 99.0, 100.0],
        "close": [103.0, 104.0, 105.0],
        "volume": [1000, 1500, 2000]
    })
    
    price_result = validate_price_data(sample_data)
    print("Price validation:")
    print(price_result.summary())
    
    # Example Greek validation
    sample_greeks = {
        "delta": 0.6,
        "gamma": 0.05,
        "theta": -0.02,
        "vega": 0.15,
        "rho": 0.08
    }
    
    greek_result = validate_greeks(sample_greeks, "call")
    print("\nGreek validation:")
    print(greek_result.summary())
