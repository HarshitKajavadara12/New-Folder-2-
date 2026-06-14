"""
GIGA SYSTEM - Market Data Loader
Multi-source data acquisition with caching and validation
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import hashlib

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    import pandas as pd


@dataclass
class OHLCV:
    """OHLCV data container with computed properties."""
    symbol: str
    timestamps: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    
    @property
    def n(self) -> int:
        return len(self.close)
    
    @property
    def returns(self) -> np.ndarray:
        """Simple returns: (P_t - P_{t-1}) / P_{t-1}"""
        return np.diff(self.close) / self.close[:-1]
    
    @property
    def log_returns(self) -> np.ndarray:
        """Log returns: ln(P_t / P_{t-1})"""
        return np.diff(np.log(self.close))
    
    @property
    def typical_price(self) -> np.ndarray:
        """Typical price: (H + L + C) / 3"""
        return (self.high + self.low + self.close) / 3
    
    @property
    def vwap(self) -> np.ndarray:
        """Volume-weighted average price."""
        return np.cumsum(self.typical_price * self.volume) / np.cumsum(self.volume)
    
    @property
    def dollar_volume(self) -> np.ndarray:
        """Dollar volume: Price * Volume"""
        return self.close * self.volume
    
    @property
    def intraday_range(self) -> np.ndarray:
        """Intraday range: (High - Low) / Close"""
        return (self.high - self.low) / self.close
    
    @property
    def overnight_return(self) -> np.ndarray:
        """Overnight return: Open_t / Close_{t-1} - 1"""
        return self.open[1:] / self.close[:-1] - 1
    
    @property
    def intraday_return(self) -> np.ndarray:
        """Intraday return: Close / Open - 1"""
        return self.close / self.open - 1


class MarketDataLoader:
    """
    Market data loader with multi-source support.
    
    Sources:
    - Local CSV/Parquet files
    - Yahoo Finance (yfinance)
    - Alpha Vantage API
    - Crypto exchanges via ccxt (Binance, etc.)
    - Custom data providers
    
    Features:
    - Automatic caching (disk + memory)
    - Data validation
    - Resampling (daily, weekly, monthly)
    - Corporate action adjustments
    """
    
    def __init__(self, cache_dir: str = "./cache", 
                 api_keys: Optional[Dict[str, str]] = None):
        """
        Initialize loader.
        
        Parameters
        ----------
        cache_dir : str
            Directory for cached data.
        api_keys : dict, optional
            API keys for data providers.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.api_keys = api_keys or {}
        self._memory_cache: Dict[str, OHLCV] = {}
    
    # =========================================================================
    # DATA LOADING
    # =========================================================================
    
    def load(self, symbol: str, 
             start: Optional[str] = None, 
             end: Optional[str] = None,
             source: str = "auto") -> OHLCV:
        """
        Load market data for symbol.
        
        Parameters
        ----------
        symbol : str
            Ticker symbol.
        start : str, optional
            Start date (YYYY-MM-DD).
        end : str, optional
            End date (YYYY-MM-DD).
        source : str
            Data source: "auto", "yahoo", "alpha_vantage", "local".
        
        Returns
        -------
        OHLCV
            Loaded market data.
        """
        # Check memory cache
        cache_key = self._cache_key(symbol, start, end)
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]
        
        # Check disk cache
        cached = self._load_from_cache(cache_key)
        if cached is not None:
            self._memory_cache[cache_key] = cached
            return cached
        
        # Load from source
        if source == "auto":
            source = self._detect_source(symbol)
        
        if source == "yahoo":
            data = self._load_yahoo(symbol, start, end)
        elif source == "alpha_vantage":
            data = self._load_alpha_vantage(symbol, start, end)
        elif source == "local":
            data = self._load_local(symbol, start, end)
        elif source == "crypto":
            data = self._load_crypto(symbol, start, end)
        else:
            raise ValueError(f"Unknown source: {source}")
        
        # Validate
        self._validate(data)
        
        # Cache
        self._save_to_cache(cache_key, data)
        self._memory_cache[cache_key] = data
        
        return data
    
    def _load_yahoo(self, symbol: str, start: Optional[str], 
                    end: Optional[str]) -> OHLCV:
        """Load from Yahoo Finance."""
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance required: pip install yfinance")
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end)
        
        if len(df) == 0:
            raise ValueError(f"No data found for {symbol}")
        
        return OHLCV(
            symbol=symbol,
            timestamps=df.index.to_numpy(),
            open=df['Open'].to_numpy(),
            high=df['High'].to_numpy(),
            low=df['Low'].to_numpy(),
            close=df['Close'].to_numpy(),
            volume=df['Volume'].to_numpy()
        )
    
    def _load_alpha_vantage(self, symbol: str, start: Optional[str],
                            end: Optional[str]) -> OHLCV:
        """Load from Alpha Vantage API."""
        api_key = self.api_keys.get('alpha_vantage')
        if not api_key:
            raise ValueError("Alpha Vantage API key required")
        
        try:
            import requests
        except ImportError:
            raise ImportError("requests required: pip install requests")
        
        url = f"https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "outputsize": "full",
            "apikey": api_key
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if "Time Series (Daily)" not in data:
            raise ValueError(f"Error fetching {symbol}: {data.get('Note', 'Unknown error')}")
        
        ts = data["Time Series (Daily)"]
        
        dates = []
        opens, highs, lows, closes, volumes = [], [], [], [], []
        
        for date, values in sorted(ts.items()):
            dates.append(datetime.strptime(date, "%Y-%m-%d"))
            opens.append(float(values["1. open"]))
            highs.append(float(values["2. high"]))
            lows.append(float(values["3. low"]))
            closes.append(float(values["5. adjusted close"]))
            volumes.append(int(values["6. volume"]))
        
        return OHLCV(
            symbol=symbol,
            timestamps=np.array(dates),
            open=np.array(opens),
            high=np.array(highs),
            low=np.array(lows),
            close=np.array(closes),
            volume=np.array(volumes)
        )
    
    def _load_local(self, symbol: str, start: Optional[str],
                    end: Optional[str]) -> OHLCV:
        """Load from local file."""
        # Search for file
        for ext in ['.parquet', '.csv', '.csv.gz']:
            filepath = self.cache_dir / f"{symbol}{ext}"
            if filepath.exists():
                return self._load_file(filepath, symbol)
        
        raise FileNotFoundError(f"No local data found for {symbol}")
    
    def _load_file(self, filepath: Path, symbol: str) -> OHLCV:
        """Load from file (CSV or Parquet)."""
        if filepath.suffix == '.parquet':
            if POLARS_AVAILABLE:
                df = pl.read_parquet(str(filepath))
            else:
                df = pd.read_parquet(filepath)
        else:
            if POLARS_AVAILABLE:
                df = pl.read_csv(str(filepath), try_parse_dates=True)
            else:
                df = pd.read_csv(filepath, parse_dates=['date', 'timestamp', 'Date', 'Timestamp'])
        
        # Standardize columns
        if POLARS_AVAILABLE:
            cols = {c: c.lower() for c in df.columns}
            df = df.rename(cols)
            
            date_col = 'date' if 'date' in df.columns else 'timestamp'
            
            return OHLCV(
                symbol=symbol,
                timestamps=df[date_col].to_numpy(),
                open=df['open'].to_numpy(),
                high=df['high'].to_numpy(),
                low=df['low'].to_numpy(),
                close=df['close'].to_numpy(),
                volume=df['volume'].to_numpy() if 'volume' in df.columns else np.zeros(len(df))
            )
        else:
            df.columns = [c.lower() for c in df.columns]
            date_col = 'date' if 'date' in df.columns else 'timestamp'
            
            return OHLCV(
                symbol=symbol,
                timestamps=df[date_col].values,
                open=df['open'].values,
                high=df['high'].values,
                low=df['low'].values,
                close=df['close'].values,
                volume=df['volume'].values if 'volume' in df.columns else np.zeros(len(df))
            )
    
    def _load_crypto(self, symbol: str, start: Optional[str],
                     end: Optional[str]) -> OHLCV:
        """
        Load crypto OHLCV data via ccxt (Binance by default).
        
        Supports pagination to fetch full history.
        Normalizes symbol format (BTCUSDT → BTC/USDT).
        """
        try:
            import ccxt
        except ImportError:
            raise ImportError("ccxt required for crypto data: pip install ccxt")
        
        exchange_id = self.api_keys.get('crypto_exchange', 'binance')
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({
            'apiKey': self.api_keys.get('crypto_api_key', ''),
            'secret': self.api_keys.get('crypto_api_secret', ''),
            'enableRateLimit': True,
        })
        
        # Normalize symbol: accept both "BTCUSDT" and "BTC/USDT"
        if '/' not in symbol:
            for quote in ['USDT', 'USD', 'BUSD', 'USDC', 'BTC', 'ETH']:
                if symbol.upper().endswith(quote):
                    base = symbol.upper()[:-len(quote)]
                    symbol = f"{base}/{quote}"
                    break
        
        since = None
        if start:
            since = int(datetime.strptime(start, "%Y-%m-%d").timestamp() * 1000)
        
        timeframe = self.api_keys.get('crypto_timeframe', '1d')
        
        all_ohlcv = []
        limit = 1000
        fetch_since = since
        
        while True:
            batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe,
                                         since=fetch_since, limit=limit)
            if not batch:
                break
            all_ohlcv.extend(batch)
            fetch_since = batch[-1][0] + 1  # next ms after last candle
            
            # Respect end date
            if end:
                end_ms = int(datetime.strptime(end, "%Y-%m-%d").timestamp() * 1000)
                if fetch_since >= end_ms:
                    break
            
            if len(batch) < limit:
                break  # No more data available
        
        if not all_ohlcv:
            raise ValueError(f"No crypto data found for {symbol}")
        
        arr = np.array(all_ohlcv)
        
        # Filter by end date
        if end:
            end_ms = int(datetime.strptime(end, "%Y-%m-%d").timestamp() * 1000)
            arr = arr[arr[:, 0] <= end_ms]
        
        timestamps = np.array([
            datetime.utcfromtimestamp(ts / 1000) for ts in arr[:, 0]
        ])
        
        return OHLCV(
            symbol=symbol,
            timestamps=timestamps,
            open=arr[:, 1].astype(float),
            high=arr[:, 2].astype(float),
            low=arr[:, 3].astype(float),
            close=arr[:, 4].astype(float),
            volume=arr[:, 5].astype(float),
        )
    
    def _detect_source(self, symbol: str) -> str:
        """Detect best data source for symbol."""
        # Check local first
        for ext in ['.parquet', '.csv']:
            if (self.cache_dir / f"{symbol}{ext}").exists():
                return "local"
        
        # Detect crypto symbols (contain '/' or common crypto suffixes)
        crypto_quotes = ('USDT', 'USD', 'BUSD', 'USDC')
        if '/' in symbol or symbol.upper().endswith(crypto_quotes):
            return "crypto"
        
        # Default to Yahoo
        return "yahoo"
    
    # =========================================================================
    # CACHING
    # =========================================================================
    
    def _cache_key(self, symbol: str, start: Optional[str], 
                   end: Optional[str]) -> str:
        """Generate cache key."""
        key = f"{symbol}_{start or 'all'}_{end or 'latest'}"
        return hashlib.md5(key.encode()).hexdigest()[:16]
    
    def _load_from_cache(self, cache_key: str) -> Optional[OHLCV]:
        """Load from disk cache."""
        cache_file = self.cache_dir / f"{cache_key}.npz"
        meta_file = self.cache_dir / f"{cache_key}.json"
        
        if not cache_file.exists() or not meta_file.exists():
            return None
        
        # Check cache freshness (1 day for daily data)
        cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        if cache_age > timedelta(days=1):
            return None
        
        # Load data
        with np.load(cache_file) as npz:
            with open(meta_file) as f:
                meta = json.load(f)
            
            return OHLCV(
                symbol=meta['symbol'],
                timestamps=npz['timestamps'],
                open=npz['open'],
                high=npz['high'],
                low=npz['low'],
                close=npz['close'],
                volume=npz['volume']
            )
    
    def _save_to_cache(self, cache_key: str, data: OHLCV):
        """Save to disk cache."""
        cache_file = self.cache_dir / f"{cache_key}.npz"
        meta_file = self.cache_dir / f"{cache_key}.json"
        
        np.savez_compressed(
            cache_file,
            timestamps=data.timestamps,
            open=data.open,
            high=data.high,
            low=data.low,
            close=data.close,
            volume=data.volume
        )
        
        with open(meta_file, 'w') as f:
            json.dump({'symbol': data.symbol, 'n': data.n}, f)
    
    # =========================================================================
    # VALIDATION
    # =========================================================================
    
    def _validate(self, data: OHLCV):
        """Validate data quality."""
        # Check for NaN
        for name, arr in [('open', data.open), ('high', data.high),
                          ('low', data.low), ('close', data.close)]:
            nan_count = np.sum(np.isnan(arr))
            if nan_count > 0:
                logger.warning(f"{nan_count} NaN values in {name}")
        
        # Check OHLC consistency
        violations = np.sum(data.high < data.low)
        if violations > 0:
            logger.warning(f"{violations} High < Low violations")
    
    # =========================================================================
    # DATA TRANSFORMATION
    # =========================================================================
    
    def resample(self, data: OHLCV, period: str = 'W') -> OHLCV:
        """
        Resample data to different frequency.
        
        Parameters
        ----------
        data : OHLCV
            Input data.
        period : str
            Target period: 'W' (weekly), 'M' (monthly), 'Q' (quarterly).
        
        Returns
        -------
        OHLCV
            Resampled data.
        """
        if POLARS_AVAILABLE:
            df = pl.DataFrame({
                'timestamp': data.timestamps,
                'open': data.open,
                'high': data.high,
                'low': data.low,
                'close': data.close,
                'volume': data.volume
            })
            
            # Group by period
            resampled = df.group_by_dynamic('timestamp', every=period).agg([
                pl.col('open').first(),
                pl.col('high').max(),
                pl.col('low').min(),
                pl.col('close').last(),
                pl.col('volume').sum()
            ])
            
            return OHLCV(
                symbol=data.symbol,
                timestamps=resampled['timestamp'].to_numpy(),
                open=resampled['open'].to_numpy(),
                high=resampled['high'].to_numpy(),
                low=resampled['low'].to_numpy(),
                close=resampled['close'].to_numpy(),
                volume=resampled['volume'].to_numpy()
            )
        else:
            df = pd.DataFrame({
                'timestamp': data.timestamps,
                'open': data.open,
                'high': data.high,
                'low': data.low,
                'close': data.close,
                'volume': data.volume
            })
            df.set_index('timestamp', inplace=True)
            
            resampled = df.resample(period).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            return OHLCV(
                symbol=data.symbol,
                timestamps=resampled.index.to_numpy(),
                open=resampled['open'].values,
                high=resampled['high'].values,
                low=resampled['low'].values,
                close=resampled['close'].values,
                volume=resampled['volume'].values
            )
    
    def align(self, *datasets: OHLCV) -> Tuple[OHLCV, ...]:
        """
        Align multiple datasets to common timestamps.
        
        Parameters
        ----------
        *datasets : OHLCV
            Multiple datasets to align.
        
        Returns
        -------
        tuple
            Aligned datasets.
        """
        # Find common dates
        common_dates = set(datasets[0].timestamps)
        for d in datasets[1:]:
            common_dates &= set(d.timestamps)
        common_dates = sorted(common_dates)
        
        aligned = []
        for data in datasets:
            mask = np.isin(data.timestamps, common_dates)
            aligned.append(OHLCV(
                symbol=data.symbol,
                timestamps=data.timestamps[mask],
                open=data.open[mask],
                high=data.high[mask],
                low=data.low[mask],
                close=data.close[mask],
                volume=data.volume[mask]
            ))
        
        return tuple(aligned)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Initialize loader
    loader = MarketDataLoader(cache_dir="./cache")
    
    # Generate synthetic data for testing
    from bridge.data_bridge import DataBridge
    
    print("Generating synthetic data...")
    synthetic = DataBridge.generate_synthetic("TEST", n_days=500)
    
    # Convert to OHLCV
    data = OHLCV(
        symbol="TEST",
        timestamps=synthetic.timestamps,
        open=synthetic.open,
        high=synthetic.high,
        low=synthetic.low,
        close=synthetic.close,
        volume=synthetic.volume
    )
    
    print(f"\nLoaded {data.n} observations")
    print(f"Date range: {data.timestamps[0]} to {data.timestamps[-1]}")
    
    # Calculate properties
    print(f"\nReturns:")
    print(f"  Mean daily: {np.mean(data.returns)*100:.4f}%")
    print(f"  Annualized: {np.mean(data.returns)*252*100:.2f}%")
    print(f"  Volatility: {np.std(data.returns)*np.sqrt(252)*100:.2f}%")
    
    print(f"\nTypical price range: ${np.min(data.typical_price):.2f} - ${np.max(data.typical_price):.2f}")
    print(f"Avg intraday range: {np.mean(data.intraday_range)*100:.2f}%")
    
    print("\nMarket data loader test complete!")
