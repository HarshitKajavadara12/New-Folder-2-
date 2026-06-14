"""
GIGA SYSTEM - Data Bridge
Unified data interface for multiple sources (CSV, Parquet, DuckDB, APIs)
"""

import numpy as np
from typing import Dict, List, Any, Optional, Union, Iterator
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    import pandas as pd

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False


@dataclass
class MarketData:
    """Standardized market data container."""
    symbol: str
    timestamps: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    
    @property
    def returns(self) -> np.ndarray:
        """Calculate simple returns."""
        return np.diff(self.close) / self.close[:-1]
    
    @property
    def log_returns(self) -> np.ndarray:
        """Calculate log returns."""
        return np.diff(np.log(self.close))
    
    def __len__(self) -> int:
        return len(self.close)


class DataBridge:
    """
    Unified data interface supporting multiple backends.
    
    Supported backends:
    - CSV files
    - Parquet files (columnar, efficient)
    - DuckDB (SQL analytics)
    - In-memory (Polars/Pandas)
    
    Features:
    - Lazy loading for large datasets
    - Automatic type inference
    - Data validation
    - Caching
    """
    
    def __init__(self, data_dir: str = "./data", cache_size: int = 100):
        """
        Initialize DataBridge.
        
        Parameters
        ----------
        data_dir : str
            Base directory for data files.
        cache_size : int
            Maximum number of datasets to cache in memory.
        """
        self.data_dir = Path(data_dir)
        self.cache_size = cache_size
        self._cache: Dict[str, MarketData] = {}
        
        # Create data directory if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize DuckDB connection if available
        self.db: Optional[duckdb.DuckDBPyConnection] = None
        if DUCKDB_AVAILABLE:
            self.db = duckdb.connect(str(self.data_dir / "giga.duckdb"))
            self._init_database()
    
    def _init_database(self):
        """Initialize DuckDB tables."""
        if self.db is None:
            return
        
        # Create OHLCV table
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv (
                symbol VARCHAR,
                timestamp TIMESTAMP,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume BIGINT,
                PRIMARY KEY (symbol, timestamp)
            )
        """)
        
        # Create index
        self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol 
            ON ohlcv (symbol)
        """)
    
    # =========================================================================
    # DATA LOADING
    # =========================================================================
    
    def load_csv(self, filepath: str, symbol: str = "UNKNOWN",
                 date_column: str = "date", 
                 parse_dates: bool = True) -> MarketData:
        """
        Load data from CSV file.
        
        Parameters
        ----------
        filepath : str
            Path to CSV file.
        symbol : str
            Symbol name for the data.
        date_column : str
            Name of date/timestamp column.
        parse_dates : bool
            Whether to parse dates.
        
        Returns
        -------
        MarketData
            Loaded and parsed market data.
        """
        filepath = Path(filepath)
        
        if POLARS_AVAILABLE:
            df = pl.read_csv(str(filepath), try_parse_dates=parse_dates)
            
            # Standardize column names
            df = df.rename({c: c.lower() for c in df.columns})
            
            return MarketData(
                symbol=symbol,
                timestamps=df[date_column].to_numpy() if date_column.lower() in df.columns else np.arange(len(df)),
                open=df["open"].to_numpy(),
                high=df["high"].to_numpy(),
                low=df["low"].to_numpy(),
                close=df["close"].to_numpy(),
                volume=df["volume"].to_numpy() if "volume" in df.columns else np.zeros(len(df))
            )
        else:
            df = pd.read_csv(filepath, parse_dates=[date_column] if parse_dates else None)
            df.columns = [c.lower() for c in df.columns]
            
            return MarketData(
                symbol=symbol,
                timestamps=df[date_column].values if date_column.lower() in df.columns else np.arange(len(df)),
                open=df["open"].values,
                high=df["high"].values,
                low=df["low"].values,
                close=df["close"].values,
                volume=df["volume"].values if "volume" in df.columns else np.zeros(len(df))
            )
    
    def load_parquet(self, filepath: str, symbol: str = "UNKNOWN") -> MarketData:
        """
        Load data from Parquet file (efficient columnar format).
        
        Parameters
        ----------
        filepath : str
            Path to Parquet file.
        symbol : str
            Symbol name.
        
        Returns
        -------
        MarketData
            Loaded market data.
        """
        filepath = Path(filepath)
        
        if POLARS_AVAILABLE:
            df = pl.read_parquet(str(filepath))
            df = df.rename({c: c.lower() for c in df.columns})
            
            return MarketData(
                symbol=symbol,
                timestamps=df["timestamp"].to_numpy() if "timestamp" in df.columns else np.arange(len(df)),
                open=df["open"].to_numpy(),
                high=df["high"].to_numpy(),
                low=df["low"].to_numpy(),
                close=df["close"].to_numpy(),
                volume=df["volume"].to_numpy() if "volume" in df.columns else np.zeros(len(df))
            )
        else:
            df = pd.read_parquet(filepath)
            df.columns = [c.lower() for c in df.columns]
            
            return MarketData(
                symbol=symbol,
                timestamps=df["timestamp"].values if "timestamp" in df.columns else np.arange(len(df)),
                open=df["open"].values,
                high=df["high"].values,
                low=df["low"].values,
                close=df["close"].values,
                volume=df["volume"].values if "volume" in df.columns else np.zeros(len(df))
            )
    
    def load_auto(self, filepath: str, symbol: str = "UNKNOWN") -> MarketData:
        """
        Auto-detect file format and load data.
        
        Supports: .csv, .parquet, .json
        
        Parameters
        ----------
        filepath : str
            Path to data file.
        symbol : str
            Symbol name.
        
        Returns
        -------
        MarketData
            Loaded market data.
        """
        filepath = Path(filepath)
        ext = filepath.suffix.lower()
        
        if ext == '.csv':
            return self.load_csv(str(filepath), symbol)
        elif ext in ('.parquet', '.pq'):
            return self.load_parquet(str(filepath), symbol)
        elif ext == '.json':
            return self._load_json(str(filepath), symbol)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Use .csv, .parquet, or .json")

    def _load_json(self, filepath: str, symbol: str = "UNKNOWN") -> MarketData:
        """Load data from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            # List of OHLCV records
            timestamps = np.array([r.get('timestamp', r.get('date', i)) for i, r in enumerate(data)])
            opens = np.array([r.get('open', 0.0) for r in data])
            highs = np.array([r.get('high', 0.0) for r in data])
            lows = np.array([r.get('low', 0.0) for r in data])
            closes = np.array([r.get('close', 0.0) for r in data])
            volumes = np.array([r.get('volume', 0) for r in data])
        elif isinstance(data, dict):
            timestamps = np.array(data.get('timestamps', data.get('dates', [])))
            opens = np.array(data.get('open', []))
            highs = np.array(data.get('high', []))
            lows = np.array(data.get('low', []))
            closes = np.array(data.get('close', []))
            volumes = np.array(data.get('volume', np.zeros(len(closes))))
        else:
            raise ValueError("JSON must contain a list of records or a dict of arrays")
        
        return MarketData(
            symbol=symbol,
            timestamps=timestamps,
            open=opens,
            high=highs,
            low=lows,
            close=closes,
            volume=volumes
        )

    # =========================================================================
    # DATABASE OPERATIONS
    # =========================================================================
    
    def save_to_db(self, data: MarketData):
        """
        Save market data to DuckDB.
        
        Parameters
        ----------
        data : MarketData
            Market data to save.
        """
        if self.db is None:
            raise RuntimeError("DuckDB not available")
        
        # Create temporary table from data
        values = list(zip(
            [data.symbol] * len(data),
            data.timestamps,
            data.open,
            data.high,
            data.low,
            data.close,
            data.volume
        ))
        
        # Insert with upsert
        self.db.executemany("""
            INSERT OR REPLACE INTO ohlcv 
            (symbol, timestamp, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, values)
    
    def load_from_db(self, symbol: str, 
                     start_date: Optional[datetime] = None,
                     end_date: Optional[datetime] = None) -> MarketData:
        """
        Load market data from DuckDB.
        
        Parameters
        ----------
        symbol : str
            Symbol to load.
        start_date : datetime, optional
            Start date filter.
        end_date : datetime, optional
            End date filter.
        
        Returns
        -------
        MarketData
            Loaded market data.
        """
        if self.db is None:
            raise RuntimeError("DuckDB not available")
        
        query = "SELECT * FROM ohlcv WHERE symbol = ?"
        params = [symbol]
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
        
        query += " ORDER BY timestamp"
        
        result = self.db.execute(query, params).fetchall()
        
        if not result:
            raise ValueError(f"No data found for symbol: {symbol}")
        
        # Unpack results
        symbols, timestamps, opens, highs, lows, closes, volumes = zip(*result)
        
        return MarketData(
            symbol=symbol,
            timestamps=np.array(timestamps),
            open=np.array(opens),
            high=np.array(highs),
            low=np.array(lows),
            close=np.array(closes),
            volume=np.array(volumes)
        )
    
    def query(self, sql: str, params: Optional[list] = None) -> Any:
        """
        Execute SQL query on DuckDB.
        
        Parameters
        ----------
        sql : str
            SQL query string (use ? for parameters).
        params : list, optional
            Query parameters for safe parameterized queries.
        
        Returns
        -------
        Any
            Query results.
        """
        if self.db is None:
            raise RuntimeError("DuckDB not available")
        if params:
            return self.db.execute(sql, params).fetchall()
        return self.db.execute(sql).fetchall()
    
    # =========================================================================
    # DATA GENERATION (for testing)
    # =========================================================================
    
    @staticmethod
    def generate_synthetic(symbol: str = "SYN", 
                          n_days: int = 252,
                          start_price: float = 100.0,
                          volatility: float = 0.02,
                          drift: float = 0.0001) -> MarketData:
        """
        Generate synthetic OHLCV data for TESTING ONLY.
        
        **WARNING:** This method generates fake data using Geometric Brownian Motion.
        For production use, fetch real market data using:
        - DataBridge.from_yfinance()
        - DataBridge.from_csv()
        - DataBridge.from_parquet()
        
        Uses Geometric Brownian Motion:
        dS = μ*S*dt + σ*S*dW
        
        Parameters
        ----------
        symbol : str
            Symbol name.
        n_days : int
            Number of trading days.
        start_price : float
            Initial price.
        volatility : float
            Daily volatility (σ).
        drift : float
            Daily drift (μ).
        
        Returns
        -------
        MarketData
            Generated synthetic data.
        """
        np.random.seed(42)  # For reproducibility
        
        # Generate returns
        returns = np.random.normal(drift, volatility, n_days)
        
        # Generate prices
        close = start_price * np.cumprod(1 + returns)
        close = np.insert(close, 0, start_price)[:-1]
        
        # Generate OHLC from close
        # High is close + random increment
        # Low is close - random decrement
        # Open is between previous close and current
        high_factor = np.random.uniform(1.0, 1.02, n_days)
        low_factor = np.random.uniform(0.98, 1.0, n_days)
        
        high = close * high_factor
        low = close * low_factor
        open_prices = np.roll(close, 1) * np.random.uniform(0.99, 1.01, n_days)
        open_prices[0] = start_price
        
        # Generate volume (log-normal)
        volume = np.random.lognormal(15, 0.5, n_days).astype(int)
        
        # Generate timestamps
        base_date = datetime(2023, 1, 1)
        timestamps = np.array([
            base_date + timedelta(days=i) 
            for i in range(n_days)
        ])
        
        return MarketData(
            symbol=symbol,
            timestamps=timestamps,
            open=open_prices,
            high=high,
            low=low,
            close=close,
            volume=volume
        )
    
    # =========================================================================
    # DATA VALIDATION
    # =========================================================================
    
    @staticmethod
    def validate(data: MarketData) -> Dict[str, Any]:
        """
        Validate market data quality.
        
        Checks:
        - No missing values
        - High >= Close >= Low
        - High >= Open >= Low
        - No negative prices
        - Volume >= 0
        
        Parameters
        ----------
        data : MarketData
            Data to validate.
        
        Returns
        -------
        dict
            Validation results with issues found.
        """
        issues = []
        
        # Check for NaN
        for name, arr in [("open", data.open), ("high", data.high), 
                          ("low", data.low), ("close", data.close)]:
            if np.any(np.isnan(arr)):
                issues.append(f"NaN values in {name}")
        
        # Check OHLC relationships
        if np.any(data.high < data.close):
            issues.append("High < Close violations")
        if np.any(data.low > data.close):
            issues.append("Low > Close violations")
        if np.any(data.high < data.open):
            issues.append("High < Open violations")
        if np.any(data.low > data.open):
            issues.append("Low > Open violations")
        if np.any(data.high < data.low):
            issues.append("High < Low violations")
        
        # Check for negative values
        if np.any(data.close <= 0):
            issues.append("Negative or zero prices")
        if np.any(data.volume < 0):
            issues.append("Negative volume")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "n_observations": len(data),
            "date_range": (data.timestamps[0], data.timestamps[-1]) if len(data) > 0 else None,
            "price_range": (float(np.min(data.close)), float(np.max(data.close)))
        }
    
    # =========================================================================
    # CACHING
    # =========================================================================
    
    def get_cached(self, symbol: str) -> Optional[MarketData]:
        """Get data from cache if available."""
        return self._cache.get(symbol)
    
    def cache(self, data: MarketData):
        """Add data to cache, evicting oldest if full."""
        if len(self._cache) >= self.cache_size:
            # Remove oldest entry
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[data.symbol] = data
    
    def clear_cache(self):
        """Clear all cached data."""
        self._cache.clear()

    def close(self):
        """Close database connection and release resources."""
        if self.db is not None:
            self.db.close()
            self.db = None
        self.clear_cache()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# =============================================================================
# STREAMING DATA INTERFACE
# =============================================================================

class StreamingDataSource:
    """
    Interface for streaming market data.
    
    Subclass this to implement specific data sources:
    - WebSocket feeds
    - Message queues (Kafka, RabbitMQ)
    - Simulated tick data
    """
    
    def __init__(self, symbol: str):
        """
        Initialize streaming source.
        
        Parameters
        ----------
        symbol : str
            Symbol to stream.
        """
        self.symbol = symbol
        self._running = False
    
    def start(self):
        """Start streaming."""
        self._running = True
    
    def stop(self):
        """Stop streaming."""
        self._running = False
    
    def ticks(self) -> Iterator[Dict[str, Any]]:
        """
        Yield tick data.
        
        Yields
        ------
        dict
            Tick data with timestamp, price, volume.
        """
        raise NotImplementedError("Subclass must implement ticks()")


class SimulatedTickStream(StreamingDataSource):
    """
    Simulated tick data generator for testing.
    
    Generates random ticks following GBM dynamics.
    """
    
    def __init__(self, symbol: str, tick_rate: float = 10.0,
                 start_price: float = 100.0, volatility: float = 0.0001):
        """
        Initialize simulated stream.
        
        Parameters
        ----------
        symbol : str
            Symbol name.
        tick_rate : float
            Ticks per second.
        start_price : float
            Initial price.
        volatility : float
            Per-tick volatility.
        """
        super().__init__(symbol)
        self.tick_rate = tick_rate
        self.current_price = start_price
        self.volatility = volatility
    
    def ticks(self) -> Iterator[Dict[str, Any]]:
        """Generate simulated ticks."""
        import time
        
        while self._running:
            # GBM price update
            ret = np.random.normal(0, self.volatility)
            self.current_price *= (1 + ret)
            
            # Random volume
            volume = int(np.random.exponential(100))
            
            yield {
                "symbol": self.symbol,
                "timestamp": datetime.now(),
                "price": self.current_price,
                "volume": volume,
                "side": "buy" if ret > 0 else "sell"
            }
            
            time.sleep(1.0 / self.tick_rate)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Create data bridge
    bridge = DataBridge(data_dir="./test_data")
    
    # Generate synthetic data
    print("Generating synthetic data...")
    data = DataBridge.generate_synthetic(
        symbol="TEST",
        n_days=252,
        start_price=100.0,
        volatility=0.02
    )
    
    print(f"Generated {len(data)} days of data")
    print(f"Returns - Mean: {np.mean(data.returns)*252:.2%}, "
          f"Vol: {np.std(data.returns)*np.sqrt(252):.2%}")
    
    # Validate data
    print("\nValidating data...")
    validation = DataBridge.validate(data)
    print(f"Valid: {validation['valid']}")
    print(f"Price range: ${validation['price_range'][0]:.2f} - ${validation['price_range'][1]:.2f}")
    
    # Cache data
    bridge.cache(data)
    print(f"\nCached: {bridge.get_cached('TEST') is not None}")
    
    # Save to database if DuckDB available
    if DUCKDB_AVAILABLE:
        print("\nSaving to DuckDB...")
        bridge.save_to_db(data)
        
        # Query
        loaded = bridge.load_from_db("TEST")
        print(f"Loaded {len(loaded)} records from database")
    
    print("\nData bridge test complete!")
