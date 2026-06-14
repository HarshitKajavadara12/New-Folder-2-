"""
GIGA SYSTEM - Database Manager
DuckDB-based analytics database with time-series optimizations
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime, timedelta
from contextlib import contextmanager
import threading
from queue import Queue, Empty
import logging

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    logging.getLogger(__name__).warning("DuckDB not available. Install with: pip install duckdb")

logger = logging.getLogger(__name__)


class ConnectionPool:
    """Thread-safe DuckDB connection pool.
    
    Pre-creates N connections and provides checkout/checkin semantics.
    Prevents concurrent access conflicts on a single connection.
    """
    
    def __init__(self, db_path: str, pool_size: int = 5, read_only: bool = False):
        self._db_path = db_path
        self._read_only = read_only
        self._pool: Queue = Queue(maxsize=pool_size)
        self._pool_size = pool_size
        self._lock = threading.Lock()
        self._created = 0
        
        # Pre-populate pool
        for _ in range(pool_size):
            conn = self._create_connection()
            self._pool.put(conn)
            self._created += 1
    
    def _create_connection(self) -> 'duckdb.DuckDBPyConnection':
        """Create a new DuckDB connection."""
        if self._read_only:
            return duckdb.connect(self._db_path, read_only=True)
        return duckdb.connect(self._db_path)
    
    @contextmanager
    def connection(self, timeout: float = 30.0):
        """Checkout a connection from the pool, auto-return on exit.
        
        Usage:
            with pool.connection() as conn:
                conn.execute("SELECT ...")
        """
        conn = None
        try:
            conn = self._pool.get(timeout=timeout)
            yield conn
        except Empty:
            raise TimeoutError(
                f"Could not get DB connection within {timeout}s. "
                f"Pool size: {self._pool_size}, all checked out."
            )
        finally:
            if conn is not None:
                try:
                    self._pool.put_nowait(conn)
                except Exception:
                    # Pool full (shouldn't happen), close the extra connection
                    try:
                        conn.close()
                    except Exception:
                        pass
    
    def close_all(self):
        """Close all connections in the pool."""
        closed = 0
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                conn.close()
                closed += 1
            except Empty:
                break
        logger.info(f"Connection pool closed ({closed} connections)")


class DatabaseManager:
    """
    DuckDB-based analytics database.
    
    Features:
    - Columnar storage (100x faster for analytics)
    - Time-series optimized queries
    - Automatic partitioning
    - SQL interface
    - In-memory or persistent modes
    
    Why DuckDB for Quant:
    - OLAP optimized (perfect for backtesting)
    - Vectorized query execution
    - Zero-copy integration with NumPy/Polars
    - No server required (embedded)
    """
    
    def __init__(self, db_path: str = ":memory:", read_only: bool = False,
                 pool_size: int = 5):
        """
        Initialize database with connection pool.
        
        Parameters
        ----------
        db_path : str
            Path to database file, or ":memory:" for in-memory.
        read_only : bool
            Open in read-only mode.
        pool_size : int
            Number of connections in the pool (default 5).
        """
        if not DUCKDB_AVAILABLE:
            raise ImportError("DuckDB required: pip install duckdb")
        
        self.db_path = db_path
        self.read_only = read_only
        self._pool: Optional[ConnectionPool] = None
        # Keep a dedicated connection for schema init and backward compat
        self._conn: Optional[duckdb.DuckDBPyConnection] = None
        
        self._connect(pool_size)
        self._init_schema()
    
    def _connect(self, pool_size: int = 5):
        """Establish database connection pool."""
        self._pool = ConnectionPool(self.db_path, pool_size, self.read_only)
        # Keep one direct connection for backward compatibility
        if self.read_only:
            self._conn = duckdb.connect(self.db_path, read_only=True)
        else:
            self._conn = duckdb.connect(self.db_path)
    
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool (thread-safe).
        
        Usage:
            with db.get_connection() as conn:
                result = conn.execute("SELECT ...").fetchall()
        """
        with self._pool.connection() as conn:
            yield conn
    
    def _init_schema(self):
        """Initialize database schema."""
        if self.read_only:
            return
        
        # OHLCV table (main market data)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv (
                symbol VARCHAR NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                open DOUBLE NOT NULL,
                high DOUBLE NOT NULL,
                low DOUBLE NOT NULL,
                close DOUBLE NOT NULL,
                volume BIGINT NOT NULL,
                PRIMARY KEY (symbol, timestamp)
            )
        """)
        
        # Trades table (tick data)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id BIGINT,
                symbol VARCHAR NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                price DOUBLE NOT NULL,
                volume DOUBLE NOT NULL,
                side VARCHAR,  -- 'buy', 'sell', or NULL
                PRIMARY KEY (symbol, timestamp, id)
            )
        """)
        
        # Orders table (order book snapshots)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS orderbook (
                symbol VARCHAR NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                side VARCHAR NOT NULL,  -- 'bid' or 'ask'
                price DOUBLE NOT NULL,
                quantity DOUBLE NOT NULL,
                level INTEGER NOT NULL,
                PRIMARY KEY (symbol, timestamp, side, level)
            )
        """)
        
        # Backtests table
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS backtests (
                id VARCHAR PRIMARY KEY,
                strategy VARCHAR NOT NULL,
                start_date TIMESTAMP,
                end_date TIMESTAMP,
                initial_capital DOUBLE,
                final_capital DOUBLE,
                total_return DOUBLE,
                sharpe_ratio DOUBLE,
                max_drawdown DOUBLE,
                n_trades INTEGER,
                win_rate DOUBLE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                params JSON
            )
        """)
        
        # Trade log table
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS trade_log (
                backtest_id VARCHAR,
                timestamp TIMESTAMP,
                symbol VARCHAR,
                side VARCHAR,
                quantity DOUBLE,
                price DOUBLE,
                commission DOUBLE,
                pnl DOUBLE,
                FOREIGN KEY (backtest_id) REFERENCES backtests(id)
            )
        """)
        
        # Create indexes
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol 
            ON ohlcv (symbol)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ohlcv_timestamp 
            ON ohlcv (timestamp)
        """)
    
    @contextmanager
    def transaction(self):
        """Context manager for transactions."""
        self._conn.execute("BEGIN TRANSACTION")
        try:
            yield
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
    
    # =========================================================================
    # OHLCV OPERATIONS
    # =========================================================================
    
    def insert_ohlcv(self, symbol: str, timestamps: np.ndarray,
                     open: np.ndarray, high: np.ndarray, 
                     low: np.ndarray, close: np.ndarray,
                     volume: np.ndarray, replace: bool = True):
        """
        Insert OHLCV data.
        
        Parameters
        ----------
        symbol : str
            Ticker symbol.
        timestamps : np.ndarray
            Array of timestamps.
        open, high, low, close, volume : np.ndarray
            Price/volume arrays.
        replace : bool
            Replace existing records on conflict.
        """
        n = len(timestamps)
        
        # Build values list
        data = list(zip(
            [symbol] * n,
            timestamps.tolist(),
            open.tolist(),
            high.tolist(),
            low.tolist(),
            close.tolist(),
            volume.tolist()
        ))
        
        if replace:
            self._conn.executemany("""
                INSERT OR REPLACE INTO ohlcv 
                (symbol, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, data)
        else:
            self._conn.executemany("""
                INSERT INTO ohlcv 
                (symbol, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, data)
    
    def get_ohlcv(self, symbol: str, 
                  start: Optional[datetime] = None,
                  end: Optional[datetime] = None) -> Dict[str, np.ndarray]:
        """
        Get OHLCV data for symbol.
        
        Parameters
        ----------
        symbol : str
            Ticker symbol.
        start : datetime, optional
            Start date.
        end : datetime, optional
            End date.
        
        Returns
        -------
        dict
            Dictionary with timestamp, open, high, low, close, volume arrays.
        """
        query = "SELECT * FROM ohlcv WHERE symbol = ?"
        params = [symbol]
        
        if start:
            query += " AND timestamp >= ?"
            params.append(start)
        if end:
            query += " AND timestamp <= ?"
            params.append(end)
        
        query += " ORDER BY timestamp"
        
        result = self._conn.execute(query, params).fetchall()
        
        if not result:
            return None
        
        symbols, timestamps, opens, highs, lows, closes, volumes = zip(*result)
        
        return {
            'timestamp': np.array(timestamps),
            'open': np.array(opens),
            'high': np.array(highs),
            'low': np.array(lows),
            'close': np.array(closes),
            'volume': np.array(volumes)
        }
    
    def get_symbols(self) -> List[str]:
        """Get list of all symbols in database."""
        result = self._conn.execute("SELECT DISTINCT symbol FROM ohlcv").fetchall()
        return [r[0] for r in result]
    
    def get_date_range(self, symbol: str) -> Tuple[datetime, datetime]:
        """Get date range for symbol."""
        result = self._conn.execute("""
            SELECT MIN(timestamp), MAX(timestamp) 
            FROM ohlcv WHERE symbol = ?
        """, [symbol]).fetchone()
        return result[0], result[1]
    
    # =========================================================================
    # ANALYTICS QUERIES
    # =========================================================================
    
    def calculate_returns(self, symbol: str, 
                         start: Optional[datetime] = None,
                         end: Optional[datetime] = None) -> np.ndarray:
        """Calculate returns using SQL window functions."""
        result = self._conn.execute("""
            SELECT 
                timestamp,
                (close - LAG(close) OVER (ORDER BY timestamp)) 
                    / LAG(close) OVER (ORDER BY timestamp) AS return
            FROM ohlcv
            WHERE symbol = ?
              AND timestamp >= COALESCE(?, '1900-01-01')
              AND timestamp <= COALESCE(?, '2100-01-01')
            ORDER BY timestamp
        """, [symbol, start, end]).fetchall()
        
        return np.array([r[1] for r in result if r[1] is not None])
    
    def calculate_volatility(self, symbol: str, window: int = 21) -> np.ndarray:
        """Calculate rolling volatility using SQL."""
        result = self._conn.execute(f"""
            WITH returns AS (
                SELECT 
                    timestamp,
                    (close - LAG(close) OVER (ORDER BY timestamp)) 
                        / LAG(close) OVER (ORDER BY timestamp) AS ret
                FROM ohlcv
                WHERE symbol = ?
            )
            SELECT 
                timestamp,
                STDDEV(ret) OVER (
                    ORDER BY timestamp 
                    ROWS BETWEEN {window-1} PRECEDING AND CURRENT ROW
                ) * SQRT(252) AS vol
            FROM returns
            ORDER BY timestamp
        """, [symbol]).fetchall()
        
        return np.array([r[1] for r in result if r[1] is not None])
    
    def correlation_matrix(self, symbols: List[str],
                          start: Optional[datetime] = None,
                          end: Optional[datetime] = None) -> np.ndarray:
        """Calculate correlation matrix between symbols."""
        returns_data = {}
        
        for symbol in symbols:
            returns = self.calculate_returns(symbol, start, end)
            returns_data[symbol] = returns
        
        # Align lengths
        min_len = min(len(r) for r in returns_data.values())
        aligned = np.array([returns_data[s][:min_len] for s in symbols])
        
        return np.corrcoef(aligned)
    
    def get_summary_stats(self, symbol: str) -> Dict[str, float]:
        """Get summary statistics for symbol."""
        result = self._conn.execute("""
            WITH price_data AS (
                SELECT 
                    close,
                    (close - LAG(close) OVER (ORDER BY timestamp)) 
                        / LAG(close) OVER (ORDER BY timestamp) AS ret,
                    volume
                FROM ohlcv
                WHERE symbol = ?
            )
            SELECT 
                COUNT(*) as n_days,
                AVG(close) as avg_price,
                MIN(close) as min_price,
                MAX(close) as max_price,
                AVG(ret) * 252 as annual_return,
                STDDEV(ret) * SQRT(252) as annual_vol,
                AVG(volume) as avg_volume
            FROM price_data
        """, [symbol]).fetchone()
        
        return {
            'n_days': result[0],
            'avg_price': result[1],
            'min_price': result[2],
            'max_price': result[3],
            'annual_return': result[4],
            'annual_volatility': result[5],
            'avg_volume': result[6]
        }
    
    # =========================================================================
    # BACKTEST STORAGE
    # =========================================================================
    
    def save_backtest(self, backtest_id: str, strategy: str,
                      metrics: Dict[str, float], trades: List[Dict],
                      params: Optional[Dict] = None):
        """
        Save backtest results.
        
        Parameters
        ----------
        backtest_id : str
            Unique identifier for backtest.
        strategy : str
            Strategy name.
        metrics : dict
            Performance metrics.
        trades : list
            List of trade dictionaries.
        params : dict, optional
            Strategy parameters.
        """
        import json
        
        # Insert backtest record
        self._conn.execute("""
            INSERT INTO backtests 
            (id, strategy, start_date, end_date, initial_capital, 
             final_capital, total_return, sharpe_ratio, max_drawdown, 
             n_trades, win_rate, params)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            backtest_id,
            strategy,
            metrics.get('start_date'),
            metrics.get('end_date'),
            metrics.get('initial_capital'),
            metrics.get('final_capital'),
            metrics.get('total_return'),
            metrics.get('sharpe_ratio'),
            metrics.get('max_drawdown'),
            metrics.get('n_trades'),
            metrics.get('win_rate'),
            json.dumps(params) if params else None
        ])
        
        # Insert trades
        trade_data = [
            (backtest_id, t.get('timestamp'), t.get('symbol'), t.get('side'),
             t.get('quantity'), t.get('price'), t.get('commission', 0), t.get('pnl', 0))
            for t in trades
        ]
        
        self._conn.executemany("""
            INSERT INTO trade_log 
            (backtest_id, timestamp, symbol, side, quantity, price, commission, pnl)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, trade_data)
    
    def get_backtest(self, backtest_id: str) -> Dict[str, Any]:
        """Get backtest results by ID."""
        result = self._conn.execute("""
            SELECT * FROM backtests WHERE id = ?
        """, [backtest_id]).fetchone()
        
        if not result:
            return None
        
        columns = ['id', 'strategy', 'start_date', 'end_date', 'initial_capital',
                   'final_capital', 'total_return', 'sharpe_ratio', 'max_drawdown',
                   'n_trades', 'win_rate', 'created_at', 'params']
        
        return dict(zip(columns, result))
    
    def list_backtests(self, strategy: Optional[str] = None) -> List[Dict]:
        """List all backtests, optionally filtered by strategy."""
        query = "SELECT id, strategy, total_return, sharpe_ratio, created_at FROM backtests"
        params = []
        
        if strategy:
            query += " WHERE strategy = ?"
            params.append(strategy)
        
        query += " ORDER BY created_at DESC"
        
        results = self._conn.execute(query, params).fetchall()
        
        return [
            {'id': r[0], 'strategy': r[1], 'total_return': r[2], 
             'sharpe_ratio': r[3], 'created_at': r[4]}
            for r in results
        ]
    
    # =========================================================================
    # RAW SQL INTERFACE
    # =========================================================================
    
    def execute(self, sql: str, params: Optional[List] = None) -> Any:
        """Execute raw SQL query."""
        if params:
            return self._conn.execute(sql, params).fetchall()
        return self._conn.execute(sql).fetchall()
    
    def execute_df(self, sql: str, params: Optional[List] = None):
        """Execute query and return as DataFrame (Polars or Pandas)."""
        if params:
            return self._conn.execute(sql, params).df()
        return self._conn.execute(sql).df()
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def vacuum(self):
        """Optimize database storage."""
        self._conn.execute("VACUUM")
    
    def close(self):
        """Close database connection pool and direct connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
        if self._pool:
            self._pool.close_all()
            self._pool = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    @property
    def info(self) -> Dict[str, Any]:
        """Get database info."""
        tables = self._conn.execute("""
            SELECT table_name, 
                   (SELECT COUNT(*) FROM information_schema.columns 
                    WHERE table_name = t.table_name) as n_columns
            FROM information_schema.tables t
            WHERE table_schema = 'main'
        """).fetchall()
        
        return {
            'path': self.db_path,
            'tables': {t[0]: {'columns': t[1]} for t in tables}
        }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    if DUCKDB_AVAILABLE:
        # Create in-memory database
        db = DatabaseManager(":memory:")
        
        print("Database initialized")
        print(f"Tables: {list(db.info['tables'].keys())}")
        
        # Generate synthetic data
        np.random.seed(42)
        n = 252
        
        timestamps = np.array([
            datetime(2023, 1, 1) + timedelta(days=i) 
            for i in range(n)
        ])
        close = 100 * np.cumprod(1 + np.random.normal(0.0004, 0.02, n))
        
        # Insert data
        db.insert_ohlcv(
            symbol="TEST",
            timestamps=timestamps,
            open=close * 0.999,
            high=close * 1.01,
            low=close * 0.99,
            close=close,
            volume=np.random.randint(1000000, 10000000, n)
        )
        
        print(f"\nInserted {n} records for TEST")
        
        # Query data
        data = db.get_ohlcv("TEST")
        print(f"Retrieved {len(data['close'])} records")
        
        # Calculate returns
        returns = db.calculate_returns("TEST")
        print(f"\nReturns calculated: {len(returns)} observations")
        print(f"Mean daily return: {np.mean(returns)*100:.4f}%")
        
        # Get summary stats
        stats = db.get_summary_stats("TEST")
        print(f"\nSummary Statistics:")
        for key, value in stats.items():
            if value is not None:
                print(f"  {key}: {value:.4f}")
        
        # Clean up
        db.close()
        print("\nDatabase test complete!")
    else:
        print("DuckDB not available. Install with: pip install duckdb")
