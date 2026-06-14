"""
GIGA SYSTEM - Real-Time Market Data Streaming
==============================================

High-frequency real-time market data streaming with sub-millisecond latency.
Production-grade implementation for HFT-level data ingestion.

Supported Data Sources:
- Alpaca Data API (Real-time stocks, crypto)
- Polygon.io (Real-time stocks, options, forex)
- IEX Cloud (Real-time quotes)
- Interactive Brokers TWS API (Multi-asset real-time)
- Binance WebSocket (Crypto real-time)
- Yahoo Finance WebSocket (Free tier)

Performance Targets:
- WebSocket latency: <5ms
- Message processing: <1ms
- Data validation: <0.1ms
- Memory efficient: Streaming with backpressure
"""

import asyncio
import websockets
import json
import time
import logging
from typing import Dict, List, Optional, Callable, Any, AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading
import queue

logger = logging.getLogger(__name__)

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False


@dataclass
class Quote:
    """Real-time quote data structure."""
    symbol: str
    timestamp: float  # Unix timestamp in seconds
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    last: float
    last_size: int
    volume: int = 0
    
    @property
    def mid(self) -> float:
        """Mid price."""
        return (self.bid + self.ask) / 2
    
    @property
    def spread(self) -> float:
        """Bid-ask spread."""
        return self.ask - self.bid
    
    @property
    def spread_bps(self) -> float:
        """Spread in basis points."""
        mid = self.mid
        if mid == 0:
            return 0.0
        return (self.spread / mid) * 10000
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp),
            'bid': self.bid,
            'ask': self.ask,
            'bid_size': self.bid_size,
            'ask_size': self.ask_size,
            'last': self.last,
            'last_size': self.last_size,
            'volume': self.volume,
            'mid': self.mid,
            'spread': self.spread,
            'spread_bps': self.spread_bps
        }


@dataclass
class Trade:
    """Real-time trade data structure."""
    symbol: str
    timestamp: float
    price: float
    size: int
    conditions: List[str] = field(default_factory=list)
    exchange: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp),
            'price': self.price,
            'size': self.size,
            'conditions': self.conditions,
            'exchange': self.exchange
        }


@dataclass
class OrderBookSnapshot:
    """Real-time order book snapshot."""
    symbol: str
    timestamp: float
    bids: List[tuple]  # [(price, size), ...]
    asks: List[tuple]  # [(price, size), ...]
    
    @property
    def best_bid(self) -> float:
        """Best bid price."""
        return self.bids[0][0] if self.bids else 0.0
    
    @property
    def best_ask(self) -> float:
        """Best ask price."""
        return self.asks[0][0] if self.asks else 0.0
    
    @property
    def mid_price(self) -> float:
        """Mid price."""
        return (self.best_bid + self.best_ask) / 2
    
    def get_depth(self, levels: int = 5) -> Dict:
        """Get order book depth."""
        return {
            'bids': self.bids[:levels],
            'asks': self.asks[:levels],
            'bid_volume': sum(size for _, size in self.bids[:levels]),
            'ask_volume': sum(size for _, size in self.asks[:levels])
        }


class RealTimeDataStream:
    """
    Real-time market data streaming client.
    
    High-performance async WebSocket client with:
    - Automatic reconnection
    - Message buffering with backpressure
    - Sub-millisecond processing
    - Thread-safe operation
    """
    
    def __init__(self, provider: str = 'alpaca', api_key: Optional[str] = None,
                 api_secret: Optional[str] = None, buffer_size: int = 10000):
        """
        Initialize real-time data stream.
        
        Parameters
        ----------
        provider : str
            Data provider: 'alpaca', 'polygon', 'iex', 'yahoo', 'binance'
        api_key : str, optional
            API key for authenticated providers
        api_secret : str, optional
            API secret for authenticated providers
        buffer_size : int
            Maximum buffer size for messages
        """
        self.provider = provider.lower()
        self.api_key = api_key
        self.api_secret = api_secret
        self.buffer_size = buffer_size
        
        # WebSocket connection
        self.ws = None
        self._connected_event = asyncio.Event()
        self.running = False
        
        # Subscription tracking for reconnect
        self._subscribed_symbols: List[str] = []
        self._subscribed_channels: List[str] = []
        
        # Message buffer (thread-safe)
        self.message_queue = queue.Queue(maxsize=buffer_size)
        
        # Callbacks
        self.quote_callbacks: List[Callable] = []
        self.trade_callbacks: List[Callable] = []
        self.orderbook_callbacks: List[Callable] = []
        
        # Statistics
        self.stats = {
            'messages_received': 0,
            'messages_processed': 0,
            'messages_dropped': 0,
            'latency_avg': 0.0,
            'latency_p99': 0.0
        }
        self.latencies = deque(maxlen=1000)
        
        # Configuration per provider
        self._setup_provider()
    
    def _setup_provider(self):
        """Setup provider-specific configuration."""
        if self.provider == 'alpaca':
            self.ws_url = 'wss://stream.data.alpaca.markets/v2/iex'
            self.auth_required = True
        
        elif self.provider == 'polygon':
            self.ws_url = f'wss://socket.polygon.io/stocks'
            self.auth_required = True
        
        elif self.provider == 'iex':
            self.ws_url = 'wss://cloud.iexapis.com/stable/stocksUSNoUTP'
            self.auth_required = True
        
        elif self.provider == 'yahoo':
            self.ws_url = 'wss://streamer.finance.yahoo.com'
            self.auth_required = False
        
        elif self.provider == 'binance':
            self.ws_url = 'wss://stream.binance.com:9443/ws'
            self.auth_required = False
        
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    async def connect(self):
        """Establish WebSocket connection."""
        try:
            self.ws = await websockets.connect(
                self.ws_url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5
            )
            self._connected_event.set()
            
            # Authenticate if required
            if self.auth_required:
                await self._authenticate()
            
            logger.info(f"Connected to {self.provider} real-time stream")
            
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self._connected_event.clear()
            raise
    
    async def _authenticate(self):
        """Authenticate with provider."""
        if self.provider == 'alpaca':
            auth_msg = {
                "action": "auth",
                "key": self.api_key,
                "secret": self.api_secret
            }
            await self.ws.send(json.dumps(auth_msg))
            response = await self.ws.recv()
            auth_response = json.loads(response)
            
            if isinstance(auth_response, list) and len(auth_response) > 0:
                if auth_response[0].get('T') != 'success':
                    raise ValueError(f"Authentication failed: {auth_response}")
            elif isinstance(auth_response, dict) and auth_response.get('T') != 'success':
                raise ValueError(f"Authentication failed: {auth_response}")
        
        elif self.provider == 'polygon':
            auth_msg = {"action": "auth", "params": self.api_key}
            await self.ws.send(json.dumps(auth_msg))
            response = await self.ws.recv()
            auth_response = json.loads(response)
            
            if isinstance(auth_response, list) and len(auth_response) > 0:
                if auth_response[0].get('status') != 'auth_success':
                    raise ValueError(f"Authentication failed: {auth_response}")
            elif isinstance(auth_response, dict) and auth_response.get('status') != 'auth_success':
                raise ValueError(f"Authentication failed: {auth_response}")
    
    async def subscribe(self, symbols: List[str], channels: List[str] = ['quotes', 'trades']):
        """
        Subscribe to real-time data for symbols.
        
        Parameters
        ----------
        symbols : list of str
            List of symbols to subscribe
        channels : list of str
            Data channels: 'quotes', 'trades', 'orderbook'
        """
        if not self._connected_event.is_set():
            await self.connect()
        
        # Track subscriptions for reconnect
        self._subscribed_symbols = list(symbols)
        self._subscribed_channels = list(channels)
        
        if self.provider == 'alpaca':
            sub_msg = {
                "action": "subscribe",
                "quotes": symbols if 'quotes' in channels else [],
                "trades": symbols if 'trades' in channels else []
            }
            await self.ws.send(json.dumps(sub_msg))
        
        elif self.provider == 'polygon':
            streams = []
            if 'quotes' in channels:
                streams.extend([f"Q.{s}" for s in symbols])
            if 'trades' in channels:
                streams.extend([f"T.{s}" for s in symbols])
            
            sub_msg = {"action": "subscribe", "params": ",".join(streams)}
            await self.ws.send(json.dumps(sub_msg))
        
        elif self.provider == 'yahoo':
            sub_msg = {"subscribe": symbols}
            await self.ws.send(json.dumps(sub_msg))
        
        elif self.provider == 'binance':
            # Binance uses different URL format
            streams = [f"{s.lower()}@aggTrade" for s in symbols]
            combined_url = f"wss://stream.binance.com:9443/stream?streams={'/'.join(streams)}"
            try:
                if self.ws:
                    await self.ws.close()
            except Exception:
                pass
            self.ws = await websockets.connect(
                combined_url, ping_interval=20, ping_timeout=10, close_timeout=5
            )
        
        logger.info(f"Subscribed to {len(symbols)} symbols on {channels}")
    
    async def stream_quotes(self, symbols: List[str]) -> AsyncIterator[Quote]:
        """
        Stream real-time quotes.
        
        Parameters
        ----------
        symbols : list of str
            Symbols to stream
        
        Yields
        ------
        Quote
            Real-time quote data
        """
        await self.subscribe(symbols, channels=['quotes'])
        
        async for message in self._receive_messages():
            if message.get('type') == 'quote':
                yield self._parse_quote(message)
    
    async def stream_trades(self, symbols: List[str]) -> AsyncIterator[Trade]:
        """
        Stream real-time trades.
        
        Parameters
        ----------
        symbols : list of str
            Symbols to stream
        
        Yields
        ------
        Trade
            Real-time trade data
        """
        await self.subscribe(symbols, channels=['trades'])
        
        async for message in self._receive_messages():
            if message.get('type') == 'trade':
                yield self._parse_trade(message)
    
    async def _receive_messages(self) -> AsyncIterator[Dict]:
        """Receive and parse WebSocket messages."""
        try:
            async for raw_message in self.ws:
                receive_time = time.time()
                
                try:
                    message = json.loads(raw_message)
                    
                    # Parse based on provider format
                    parsed_messages = self._parse_message(message)
                    
                    for parsed_msg in parsed_messages:
                        # Calculate latency (batch numpy stats every 100 msgs)
                        if 'timestamp' in parsed_msg:
                            latency = (receive_time - parsed_msg['timestamp']) * 1000
                            self.latencies.append(latency)
                        
                        self.stats['messages_received'] += 1
                        self.stats['messages_processed'] += 1
                        
                        # Update numpy stats periodically (not every message)
                        if self.stats['messages_received'] % 100 == 0 and len(self.latencies) > 0:
                            lat_arr = np.array(self.latencies)
                            self.stats['latency_avg'] = float(np.mean(lat_arr))
                            self.stats['latency_p99'] = float(np.percentile(lat_arr, 99))
                        
                        # Invoke registered callbacks
                        msg_type = parsed_msg.get('type')
                        if msg_type == 'quote':
                            quote = self._parse_quote(parsed_msg)
                            for cb in self.quote_callbacks:
                                try:
                                    cb(quote)
                                except Exception as cb_err:
                                    logger.warning(f"Quote callback error: {cb_err}")
                        elif msg_type == 'trade':
                            trade = self._parse_trade(parsed_msg)
                            for cb in self.trade_callbacks:
                                try:
                                    cb(trade)
                                except Exception as cb_err:
                                    logger.warning(f"Trade callback error: {cb_err}")
                        
                        yield parsed_msg
                
                except json.JSONDecodeError:
                    self.stats['messages_dropped'] += 1
                    continue
                except Exception as e:
                    self.stats['messages_dropped'] += 1
                    logger.warning(f"Message parsing error: {e}")
                    continue
        
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Connection closed, reconnecting...")
            await self.reconnect()
    
    def _parse_message(self, message: Any) -> List[Dict]:
        """Parse message based on provider format."""
        parsed = []
        
        if self.provider == 'alpaca':
            # Alpaca returns list of messages
            if isinstance(message, list):
                for msg in message:
                    msg_type = msg.get('T')
                    if msg_type == 'q':  # Quote
                        # Alpaca 't' is RFC-3339 string, not nanoseconds
                        ts_raw = msg.get('t', '')
                        try:
                            ts = datetime.fromisoformat(ts_raw.replace('Z', '+00:00')).timestamp()
                        except (ValueError, AttributeError):
                            ts = time.time()
                        parsed.append({
                            'type': 'quote',
                            'symbol': msg.get('S'),
                            'timestamp': ts,
                            'bid': msg.get('bp'),
                            'ask': msg.get('ap'),
                            'bid_size': msg.get('bs'),
                            'ask_size': msg.get('as')
                        })
                    elif msg_type == 't':  # Trade
                        ts_raw = msg.get('t', '')
                        try:
                            ts = datetime.fromisoformat(ts_raw.replace('Z', '+00:00')).timestamp()
                        except (ValueError, AttributeError):
                            ts = time.time()
                        parsed.append({
                            'type': 'trade',
                            'symbol': msg.get('S'),
                            'timestamp': ts,
                            'price': msg.get('p'),
                            'size': msg.get('s')
                        })
        
        elif self.provider == 'polygon':
            # Polygon format
            if isinstance(message, list):
                for msg in message:
                    ev = msg.get('ev')
                    if ev == 'Q':  # Quote
                        parsed.append({
                            'type': 'quote',
                            'symbol': msg.get('sym'),
                            'timestamp': msg.get('t') / 1000,  # Milliseconds
                            'bid': msg.get('bp'),
                            'ask': msg.get('ap'),
                            'bid_size': msg.get('bs'),
                            'ask_size': msg.get('as')
                        })
                    elif ev == 'T':  # Trade
                        parsed.append({
                            'type': 'trade',
                            'symbol': msg.get('sym'),
                            'timestamp': msg.get('t') / 1000,
                            'price': msg.get('p'),
                            'size': msg.get('s')
                        })
        
        elif self.provider == 'binance':
            # Binance format
            if 'data' in message:
                data = message['data']
                parsed.append({
                    'type': 'trade',
                    'symbol': data.get('s'),
                    'timestamp': data.get('T') / 1000,  # Milliseconds
                    'price': float(data.get('p')),
                    'size': float(data.get('q'))
                })
        
        elif self.provider == 'yahoo':
            # Yahoo Finance format — map price to bid/ask for proper Quote
            if message.get('id'):
                price = message.get('price', 0.0)
                parsed.append({
                    'type': 'quote',
                    'symbol': message.get('id'),
                    'timestamp': time.time(),
                    'bid': price,
                    'ask': price,
                    'bid_size': 0,
                    'ask_size': 0,
                    'last': price,
                    'volume': message.get('dayVolume', 0)
                })
        
        return parsed
    
    def _parse_quote(self, message: Dict) -> Quote:
        """Parse quote message."""
        return Quote(
            symbol=message['symbol'],
            timestamp=message['timestamp'],
            bid=message.get('bid', 0.0),
            ask=message.get('ask', 0.0),
            bid_size=message.get('bid_size', 0),
            ask_size=message.get('ask_size', 0),
            last=message.get('last', 0.0),
            last_size=message.get('last_size', 0),
            volume=message.get('volume', 0)
        )
    
    def _parse_trade(self, message: Dict) -> Trade:
        """Parse trade message."""
        return Trade(
            symbol=message['symbol'],
            timestamp=message['timestamp'],
            price=message['price'],
            size=message['size'],
            conditions=message.get('conditions', []),
            exchange=message.get('exchange', '')
        )
    
    async def reconnect(self, max_retries: int = 5):
        """Reconnect with exponential backoff and re-subscribe."""
        self._connected_event.clear()
        for attempt in range(max_retries):
            try:
                delay = min(2 ** attempt, 60)  # Cap at 60s
                await asyncio.sleep(delay)
                await self.connect()
                # Re-subscribe to previously subscribed streams
                if self._subscribed_symbols:
                    await self.subscribe(self._subscribed_symbols, self._subscribed_channels)
                logger.info(f"Reconnected on attempt {attempt + 1}")
                return
            except Exception as e:
                logger.warning(f"Reconnection attempt {attempt + 1} failed: {e}")
        
        raise ConnectionError("Failed to reconnect after maximum retries")
    
    def add_quote_callback(self, callback: Callable[[Quote], None]):
        """Add callback for quote updates."""
        self.quote_callbacks.append(callback)
    
    def add_trade_callback(self, callback: Callable[[Trade], None]):
        """Add callback for trade updates."""
        self.trade_callbacks.append(callback)
    
    def get_stats(self) -> Dict:
        """Get streaming statistics."""
        return {
            **self.stats,
            'connected': self._connected_event.is_set(),
            'buffer_size': self.message_queue.qsize(),
            'latency_avg_ms': self.stats['latency_avg'],
            'latency_p99_ms': self.stats['latency_p99']
        }
    
    @property
    def connected(self) -> bool:
        """Whether the WebSocket is connected (thread-safe)."""
        return self._connected_event.is_set()
    
    async def close(self):
        """Close WebSocket connection."""
        if self.ws:
            await self.ws.close()
            self._connected_event.clear()
            logger.info("Disconnected from real-time stream")


class HistoricalDataFetcher:
    """
    High-performance historical data fetcher.
    
    For backtesting with real historical data.
    Uses multiple data sources for redundancy.
    """
    
    def __init__(self, provider: str = 'alpaca', api_key: Optional[str] = None,
                 api_secret: Optional[str] = None):
        """
        Initialize historical data fetcher.
        
        Parameters
        ----------
        provider : str
            Data provider: 'alpaca', 'polygon', 'yahoo'
        api_key : str, optional
            API key for authenticated providers
        api_secret : str, optional
            API secret for authenticated providers
        """
        self.provider = provider
        self.api_key = api_key
        self.api_secret = api_secret
    
    async def fetch_bars(self, symbol: str, timeframe: str, start: str, end: str,
                        limit: int = 10000) -> pd.DataFrame:
        """
        Fetch historical OHLCV bars.
        
        Parameters
        ----------
        symbol : str
            Symbol to fetch
        timeframe : str
            Timeframe: '1Min', '5Min', '1Hour', '1Day'
        start : str
            Start date (ISO format)
        end : str
            End date (ISO format)
        limit : int
            Maximum bars to fetch
        
        Returns
        -------
        pd.DataFrame
            Historical OHLCV data
        """
        if self.provider == 'alpaca':
            return await self._fetch_alpaca_bars(symbol, timeframe, start, end, limit)
        elif self.provider == 'polygon':
            return await self._fetch_polygon_bars(symbol, timeframe, start, end, limit)
        elif self.provider == 'yahoo':
            return await self._fetch_yahoo_bars(symbol, timeframe, start, end)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    async def _fetch_alpaca_bars(self, symbol, timeframe, start, end, limit):
        """Fetch from Alpaca (async-safe)."""
        import requests
        
        def _do_fetch():
            url = f"https://data.alpaca.markets/v2/stocks/{symbol}/bars"
            headers = {
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.api_secret or ""
            }
            params = {
                "timeframe": timeframe,
                "start": start,
                "end": end,
                "limit": limit
            }
            response = requests.get(url, headers=headers, params=params)
            return response.json()
        
        # Run sync IO in thread to avoid blocking event loop
        data = await asyncio.get_event_loop().run_in_executor(None, _do_fetch)
        
        if 'bars' in data:
            df = pd.DataFrame(data['bars'])
            df['timestamp'] = pd.to_datetime(df['t'])
            df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        return pd.DataFrame()

    async def _fetch_polygon_bars(self, symbol, timeframe, start, end, limit):
        """Fetch from Polygon.io (async-safe)."""
        import requests
        
        tf_map = {'1Min': '1/minute', '5Min': '5/minute', '1Hour': '1/hour', '1Day': '1/day'}
        agg_path = tf_map.get(timeframe, '1/day')
        
        def _do_fetch():
            url = (f"https://api.polygon.io/v2/aggs/ticker/{symbol}"
                   f"/range/{agg_path}/{start}/{end}")
            params = {"apiKey": self.api_key, "limit": limit, "sort": "asc"}
            response = requests.get(url, params=params)
            return response.json()
        
        data = await asyncio.get_event_loop().run_in_executor(None, _do_fetch)
        
        if 'results' in data and data['results']:
            df = pd.DataFrame(data['results'])
            df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
            df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        return pd.DataFrame()
    
    async def _fetch_yahoo_bars(self, symbol, timeframe, start, end):
        """Fetch from Yahoo Finance."""
        try:
            import yfinance as yf
            
            # Convert timeframe to yfinance format
            interval_map = {
                '1Min': '1m',
                '5Min': '5m',
                '1Hour': '1h',
                '1Day': '1d'
            }
            interval = interval_map.get(timeframe, '1d')
            
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start, end=end, interval=interval)
            df = df.reset_index()
            df = df.rename(columns={'Date': 'timestamp', 'Open': 'open', 'High': 'high',
                                   'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
            
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        except ImportError:
            raise ImportError("yfinance required: pip install yfinance")


# Example usage
async def example_streaming():
    """Example of real-time streaming usage."""
    
    # Initialize stream (use your API keys)
    stream = RealTimeDataStream(
        provider='alpaca',
        api_key='YOUR_API_KEY',
        api_secret='YOUR_API_SECRET'
    )
    
    # Stream real-time quotes
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    
    try:
        async for quote in stream.stream_quotes(symbols):
            logger.info(f"{quote.symbol}: ${quote.mid:.2f} | Spread: {quote.spread_bps:.2f} bps | Latency: {stream.stats['latency_avg']:.2f}ms")
            
            # Process quote (e.g., trading logic)
            if quote.spread_bps < 5:  # Tight spread
                logger.info(f"Tight spread on {quote.symbol}, consider trading")
    
    finally:
        await stream.close()


if __name__ == "__main__":
    # Run example
    asyncio.run(example_streaming())
