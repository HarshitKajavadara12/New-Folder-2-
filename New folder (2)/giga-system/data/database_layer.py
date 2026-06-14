"""
DATABASE LAYER — Time-Series & OLAP Storage
=============================================

Addresses Missing Concept 7.4: No proper OLAP or time-series database.
Provides DuckDB-based OLAP and time-series storage with Parquet integration.
"""

import numpy as np
import logging
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


@dataclass
class TimeSeriesRecord:
    """A single time-series record."""
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    metadata: Optional[Dict] = None


class TimeSeriesDB:
    """
    Time-series database using DuckDB for OLAP queries.
    Falls back to in-memory storage if DuckDB unavailable.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or str(
            Path(__file__).parent.parent / "giga_system.duckdb"
        )
        self._conn = None
        self._memory_store: Dict[str, List[Dict]] = {}

        if DUCKDB_AVAILABLE:
            try:
                self._conn = duckdb.connect(self.db_path)
                self._init_tables()
            except Exception as e:
                logger.warning(f"DuckDB init failed: {e}, using memory store")
                self._conn = None

    def _init_tables(self):
        """Initialize database tables."""
        if not self._conn:
            return
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv (
                timestamp TIMESTAMP,
                symbol VARCHAR,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                timestamp TIMESTAMP,
                symbol VARCHAR,
                direction VARCHAR,
                confidence DOUBLE,
                source VARCHAR,
                metadata VARCHAR
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                timestamp TIMESTAMP,
                symbol VARCHAR,
                side VARCHAR,
                quantity DOUBLE,
                price DOUBLE,
                pnl DOUBLE,
                strategy VARCHAR
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                timestamp TIMESTAMP,
                equity DOUBLE,
                drawdown DOUBLE,
                sharpe DOUBLE,
                n_positions INTEGER,
                metadata VARCHAR
            )
        """)

    def insert_ohlcv(self, records: List[TimeSeriesRecord]):
        """Insert OHLCV data."""
        if self._conn:
            for r in records:
                self._conn.execute(
                    "INSERT INTO ohlcv VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [r.timestamp, r.symbol, r.open, r.high, r.low, r.close, r.volume]
                )
        else:
            for r in records:
                if r.symbol not in self._memory_store:
                    self._memory_store[r.symbol] = []
                self._memory_store[r.symbol].append({
                    "timestamp": r.timestamp.isoformat(),
                    "open": r.open, "high": r.high, "low": r.low,
                    "close": r.close, "volume": r.volume,
                })

    def insert_signal(self, timestamp: datetime, symbol: str, direction: str,
                      confidence: float, source: str, metadata: Optional[Dict] = None):
        """Insert a trading signal."""
        if self._conn:
            self._conn.execute(
                "INSERT INTO signals VALUES (?, ?, ?, ?, ?, ?)",
                [timestamp, symbol, direction, confidence, source,
                 json.dumps(metadata or {})]
            )
        else:
            key = "signals"
            if key not in self._memory_store:
                self._memory_store[key] = []
            self._memory_store[key].append({
                "timestamp": timestamp.isoformat(), "symbol": symbol,
                "direction": direction, "confidence": confidence,
            })

    def insert_trade(self, timestamp: datetime, symbol: str, side: str,
                     quantity: float, price: float, pnl: float, strategy: str = ""):
        """Insert a trade record."""
        if self._conn:
            self._conn.execute(
                "INSERT INTO trades VALUES (?, ?, ?, ?, ?, ?, ?)",
                [timestamp, symbol, side, quantity, price, pnl, strategy]
            )

    def query_ohlcv(self, symbol: str, start: Optional[datetime] = None,
                    end: Optional[datetime] = None, limit: int = 1000) -> List[Dict]:
        """Query OHLCV data."""
        if self._conn:
            query = "SELECT * FROM ohlcv WHERE symbol = ?"
            params = [symbol]
            if start:
                query += " AND timestamp >= ?"
                params.append(start)
            if end:
                query += " AND timestamp <= ?"
                params.append(end)
            query += f" ORDER BY timestamp DESC LIMIT {limit}"

            result = self._conn.execute(query, params).fetchall()
            columns = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
            return [dict(zip(columns, row)) for row in result]
        else:
            return self._memory_store.get(symbol, [])[:limit]

    def query_performance(self, period_days: int = 30) -> Dict:
        """Query aggregate performance metrics."""
        if self._conn:
            try:
                result = self._conn.execute("""
                    SELECT 
                        COUNT(*) as total_trades,
                        SUM(pnl) as total_pnl,
                        AVG(pnl) as avg_pnl,
                        SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades
                    FROM trades
                    WHERE timestamp > CURRENT_TIMESTAMP - INTERVAL ? DAY
                """, [period_days]).fetchone()
                return {
                    "total_trades": result[0] or 0,
                    "total_pnl": float(result[1] or 0),
                    "avg_pnl": float(result[2] or 0),
                    "winning_trades": result[3] or 0,
                    "win_rate": (result[3] or 0) / max(result[0] or 1, 1),
                }
            except Exception:
                return {"total_trades": 0, "total_pnl": 0.0}
        return {"total_trades": 0, "total_pnl": 0.0}

    def export_to_parquet(self, table: str, output_path: str):
        """Export a table to Parquet format."""
        if self._conn and PANDAS_AVAILABLE:
            df = self._conn.execute(f"SELECT * FROM {table}").fetchdf()
            df.to_parquet(output_path)
            logger.info(f"Exported {table} to {output_path}")

    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
