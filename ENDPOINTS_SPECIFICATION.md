# Chimera Trading System v2.0 - Data Endpoints Specification

## Overview

This document specifies the exact data endpoints and formats required for integrating external data sources with the Chimera Trading System. Use this specification to connect your own data feeds.

## Endpoint Interface Summary

### Required Methods

| Method | Purpose | Input | Output |
|--------|---------|-------|--------|
| `connect()` | Establish connection | credentials dict | bool |
| `disconnect()` | Close connection | none | none |
| `subscribe_level2()` | Level 2 market data | symbol, callback | bool |
| `subscribe_trades()` | Trade executions | symbol, callback | bool |
| `subscribe_quotes()` | Best bid/ask | symbol, callback | bool |
| `subscribe_bars()` | OHLCV bars | symbol, timeframe, callback | bool |
| `subscribe_news()` | News/events | symbols list, callback | bool |
| `unsubscribe()` | Remove subscription | symbol, data_type | bool |
| `get_historical_bars()` | Historical OHLCV | symbol, timeframe, start, end | List[BarUpdate] |
| `get_historical_trades()` | Historical trades | symbol, start, end | List[TradeUpdate] |
| `is_connected()` | Connection status | none | bool |
| `get_supported_symbols()` | Available symbols | none | List[str] |
| `get_market_status()` | Market hours info | symbol | dict |

## Data Format Specifications

### 1. Level 2 Market Data

**Endpoint:** `subscribe_level2(symbol: str, callback: Callable)`

**Expected Data Format:**
```python
{
    "symbol": "AAPL",                    # Required: Trading symbol
    "timestamp": 1640995200000000,       # Required: Microseconds since epoch
    "side": "bid",                       # Required: "bid" or "ask"
    "price": 150.25,                     # Required: Price level (float)
    "size": 1000.0,                      # Required: Size at level (float)
    "level": 0,                          # Required: 0=best, 1=second, etc.
    "operation": "update",               # Required: "add", "update", "delete"
    "order_count": 5,                    # Optional: Number of orders
    "exchange": "NASDAQ"                 # Optional: Exchange identifier
}
```

**Callback Signature:**
```python
def level2_callback(update: Level2Update) -> None:
    # Process Level 2 update
    pass
```

**Update Frequency:** Real-time (as market data changes)

### 2. Trade Data

**Endpoint:** `subscribe_trades(symbol: str, callback: Callable)`

**Expected Data Format:**
```python
{
    "symbol": "AAPL",                    # Required: Trading symbol
    "timestamp": 1640995200000000,       # Required: Microseconds since epoch
    "price": 150.25,                     # Required: Trade price (float)
    "size": 100.0,                       # Required: Trade size (float)
    "side": "buy",                       # Required: "buy" or "sell"
    "trade_id": "T123456789",            # Required: Unique trade ID
    "exchange": "NASDAQ",                # Optional: Exchange identifier
    "conditions": ["Regular"]            # Optional: Trade conditions
}
```

**Callback Signature:**
```python
def trade_callback(update: TradeUpdate) -> None:
    # Process trade update
    pass
```

**Update Frequency:** Real-time (every trade execution)

### 3. Quote Data (Best Bid/Ask)

**Endpoint:** `subscribe_quotes(symbol: str, callback: Callable)`

**Expected Data Format:**
```python
{
    "symbol": "AAPL",                    # Required: Trading symbol
    "timestamp": 1640995200000000,       # Required: Microseconds since epoch
    "bid_price": 150.20,                 # Required: Best bid price (float)
    "bid_size": 500.0,                   # Required: Best bid size (float)
    "ask_price": 150.25,                 # Required: Best ask price (float)
    "ask_size": 300.0,                   # Required: Best ask size (float)
    "exchange": "NASDAQ"                 # Optional: Exchange identifier
}
```

**Callback Signature:**
```python
def quote_callback(update: QuoteUpdate) -> None:
    # Process quote update
    pass
```

**Update Frequency:** Real-time (when best bid/ask changes)

### 4. Bar Data (OHLCV)

**Endpoint:** `subscribe_bars(symbol: str, timeframe: str, callback: Callable)`

**Expected Data Format:**
```python
{
    "symbol": "AAPL",                    # Required: Trading symbol
    "timestamp": 1640995200000000,       # Required: Bar start time (microseconds)
    "timeframe": "1m",                   # Required: "1s", "1m", "5m", "1h", "1d"
    "open_price": 150.00,                # Required: Opening price (float)
    "high_price": 150.50,                # Required: High price (float)
    "low_price": 149.80,                 # Required: Low price (float)
    "close_price": 150.25,               # Required: Closing price (float)
    "volume": 10000.0,                   # Required: Volume (float)
    "vwap": 150.15,                      # Optional: Volume weighted avg price
    "trade_count": 150                   # Optional: Number of trades
}
```

**Supported Timeframes:**
- `"1s"` - 1 second
- `"5s"` - 5 seconds
- `"10s"` - 10 seconds
- `"30s"` - 30 seconds
- `"1m"` - 1 minute
- `"5m"` - 5 minutes
- `"15m"` - 15 minutes
- `"30m"` - 30 minutes
- `"1h"` - 1 hour
- `"4h"` - 4 hours
- `"1d"` - 1 day

**Callback Signature:**
```python
def bar_callback(update: BarUpdate) -> None:
    # Process bar update
    pass
```

**Update Frequency:** Based on timeframe (e.g., every minute for 1m bars)

### 5. News/Events Data

**Endpoint:** `subscribe_news(symbols: List[str], callback: Callable)`

**Expected Data Format:**
```python
{
    "timestamp": 1640995200000000,       # Required: Event time (microseconds)
    "headline": "Apple Reports Q4 Earnings", # Required: News headline
    "summary": "Apple Inc. reported...",  # Required: News summary/content
    "symbols": ["AAPL"],                 # Required: Affected symbols list
    "sentiment": "positive",             # Optional: "positive", "negative", "neutral"
    "importance": 8,                     # Optional: 1-10 importance scale
    "source": "Reuters"                  # Optional: News source
}
```

**Callback Signature:**
```python
def news_callback(update: NewsUpdate) -> None:
    # Process news update
    pass
```

**Update Frequency:** As news events occur

## Historical Data Endpoints

### Historical Bars

**Endpoint:** `get_historical_bars(symbol: str, timeframe: str, start_time: int, end_time: int)`

**Parameters:**
- `symbol`: Trading symbol (e.g., "AAPL")
- `timeframe`: Bar timeframe (e.g., "1m", "1h", "1d")
- `start_time`: Start timestamp in microseconds since epoch
- `end_time`: End timestamp in microseconds since epoch

**Return Format:**
```python
[
    {
        "symbol": "AAPL",
        "timestamp": 1640995200000000,
        "timeframe": "1m",
        "open_price": 150.00,
        "high_price": 150.50,
        "low_price": 149.80,
        "close_price": 150.25,
        "volume": 10000.0,
        "vwap": 150.15,
        "trade_count": 150
    },
    # ... more bars
]
```

### Historical Trades

**Endpoint:** `get_historical_trades(symbol: str, start_time: int, end_time: int)`

**Parameters:**
- `symbol`: Trading symbol (e.g., "AAPL")
- `start_time`: Start timestamp in microseconds since epoch
- `end_time`: End timestamp in microseconds since epoch

**Return Format:**
```python
[
    {
        "symbol": "AAPL",
        "timestamp": 1640995200000000,
        "price": 150.25,
        "size": 100.0,
        "side": "buy",
        "trade_id": "T123456789",
        "exchange": "NASDAQ",
        "conditions": ["Regular"]
    },
    # ... more trades
]
```

## Connection and Authentication

### Connection Method

**Endpoint:** `connect(credentials: Optional[Dict[str, Any]])`

**Expected Credentials Format:**

**Interactive Brokers:**
```python
{
    "host": "127.0.0.1",               # IB Gateway/TWS host
    "port": 7497,                      # Port (7497=TWS, 4001=Gateway)
    "client_id": 1                     # Unique client ID
}
```

**Alpaca:**
```python
{
    "api_key": "your_api_key",         # Alpaca API key
    "secret_key": "your_secret_key",   # Alpaca secret key
    "base_url": "https://paper-api.alpaca.markets"  # Paper or live URL
}
```

**Polygon.io:**
```python
{
    "api_key": "your_polygon_api_key"  # Polygon.io API key
}
```

**TD Ameritrade:**
```python
{
    "client_id": "your_client_id",     # TD Ameritrade client ID
    "refresh_token": "your_token",     # OAuth refresh token
    "redirect_uri": "your_redirect"    # OAuth redirect URI
}
```

**Custom WebSocket:**
```python
{
    "url": "wss://your-feed.com/stream",  # WebSocket URL
    "api_key": "your_api_key",            # Authentication key
    "headers": {                          # Additional headers
        "Authorization": "Bearer token"
    }
}
```

### Status Methods

**Connection Status:** `is_connected() -> bool`
- Returns `True` if connected and ready
- Returns `False` if disconnected or error

**Market Status:** `get_market_status(symbol: str) -> Dict[str, Any]`
```python
{
    "is_open": True,                    # Market currently open
    "next_open": 1640995200000000,      # Next market open (microseconds)
    "next_close": 1641081600000000,     # Next market close (microseconds)
    "timezone": "America/New_York"      # Market timezone
}
```

**Supported Symbols:** `get_supported_symbols() -> List[str]`
```python
["AAPL", "NVDA", "TSLA", "SPY", "QQQ", ...]
```

## Error Handling

### Connection Errors
- Return `False` from `connect()` method
- Log error details for debugging
- Implement automatic retry logic

### Subscription Errors
- Return `False` from subscription methods
- Handle rate limiting gracefully
- Provide meaningful error messages

### Data Quality Issues
- Validate all required fields
- Handle missing or invalid data
- Implement data sanitization

## Performance Requirements

### Latency Targets
- **Level 2 Updates**: < 1ms processing time
- **Trade Updates**: < 1ms processing time
- **Quote Updates**: < 1ms processing time
- **Bar Updates**: < 10ms processing time

### Throughput Targets
- **Level 2**: 1000+ updates/second per symbol
- **Trades**: 100+ trades/second per symbol
- **Quotes**: 100+ quotes/second per symbol
- **Bars**: Real-time as bars complete

### Memory Usage
- Efficient data structures
- Automatic cleanup of old data
- Configurable retention policies

## Testing Your Implementation

### Unit Tests
```python
def test_connection():
    source = YourDataSource()
    result = source.connect(test_credentials)
    assert result == True
    assert source.is_connected() == True

def test_level2_subscription():
    source = YourDataSource()
    source.connect(test_credentials)
    
    updates = []
    def callback(update):
        updates.append(update)
    
    result = source.subscribe_level2("AAPL", callback)
    assert result == True
    
    # Wait for updates and verify format
    time.sleep(5)
    assert len(updates) > 0
    assert isinstance(updates[0], Level2Update)
```

### Integration Tests
```python
# Run with the trading system
from main import ChimeraTradingSystem
from data import register_data_source, set_active_data_source

# Register your source
register_data_source('your_source', YourDataSource())
set_active_data_source('your_source')

# Test with trading system
system = ChimeraTradingSystem()
system.run(['AAPL'], duration=60)
```

## Example Implementation Template

```python
from data.data_interface import DataSourceInterface, Level2Update
import websocket
import json
import threading

class YourDataSource(DataSourceInterface):
    def __init__(self):
        self.ws = None
        self.callbacks = {}
        self.is_connected_flag = False
        self.credentials = None
    
    def connect(self, credentials=None):
        """Implement your connection logic"""
        self.credentials = credentials
        
        try:
            # Example: WebSocket connection
            ws_url = f"wss://your-api.com/stream?key={credentials['api_key']}"
            self.ws = websocket.WebSocketApp(
                ws_url,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=self._on_open
            )
            
            # Start WebSocket in background
            ws_thread = threading.Thread(target=self.ws.run_forever)
            ws_thread.daemon = True
            ws_thread.start()
            
            # Wait for connection
            time.sleep(2)
            return self.is_connected_flag
            
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
    
    def _on_open(self, ws):
        """WebSocket opened"""
        self.is_connected_flag = True
        print("Connected to data source")
    
    def _on_message(self, ws, message):
        """Process incoming data"""
        try:
            data = json.loads(message)
            
            # Convert to standard format
            if data['type'] == 'level2':
                update = Level2Update(
                    symbol=data['symbol'],
                    timestamp=int(data['timestamp'] * 1_000_000),
                    side=data['side'],
                    price=float(data['price']),
                    size=float(data['size']),
                    level=int(data['level']),
                    operation=data['operation']
                )
                
                # Call registered callbacks
                self._call_callbacks(data['symbol'], 'level2', update)
                
        except Exception as e:
            print(f"Message processing error: {e}")
    
    def subscribe_level2(self, symbol, callback):
        """Subscribe to Level 2 data"""
        self._add_callback(symbol, 'level2', callback)
        
        # Send subscription message
        subscribe_msg = {
            'action': 'subscribe',
            'type': 'level2',
            'symbol': symbol
        }
        self.ws.send(json.dumps(subscribe_msg))
        return True
    
    def _add_callback(self, symbol, data_type, callback):
        """Add callback for symbol/data_type"""
        if symbol not in self.callbacks:
            self.callbacks[symbol] = {}
        if data_type not in self.callbacks[symbol]:
            self.callbacks[symbol][data_type] = []
        self.callbacks[symbol][data_type].append(callback)
    
    def _call_callbacks(self, symbol, data_type, update):
        """Call all registered callbacks"""
        if symbol in self.callbacks and data_type in self.callbacks[symbol]:
            for callback in self.callbacks[symbol][data_type]:
                try:
                    callback(update)
                except Exception as e:
                    print(f"Callback error: {e}")
    
    # Implement remaining methods...
    def disconnect(self):
        if self.ws:
            self.ws.close()
        self.is_connected_flag = False
    
    def is_connected(self):
        return self.is_connected_flag
    
    # ... implement other required methods
```

## Quick Start Checklist

- [ ] Implement `DataSourceInterface` abstract methods
- [ ] Handle connection and authentication
- [ ] Convert your data format to standard format
- [ ] Implement real-time subscriptions
- [ ] Add historical data methods
- [ ] Test with mock data first
- [ ] Validate data quality and timing
- [ ] Register with the trading system
- [ ] Run integration tests
- [ ] Deploy and monitor

---

**Ready to integrate? Use this specification to connect your data source to the Chimera Trading System!**

