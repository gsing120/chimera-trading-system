# Data Source Integration Guide

## Overview

The Chimera Trading System v2.0 provides a standardized interface for integrating external market data sources. The system currently works with mock data but can be easily extended to support real-time data from brokers and data providers.

## Data Interface Architecture

### Core Components

1. **DataSourceInterface**: Abstract base class defining the standard interface
2. **Data Update Classes**: Standardized data structures for all market data types
3. **DataSourceRegistry**: Central registry for managing multiple data sources
4. **Adapter Pattern**: Seamless integration with existing system components

### Supported Data Types

#### Level 2 Market Data
```python
Level2Update(
    symbol: str,           # Trading symbol (e.g., 'AAPL')
    timestamp: int,        # Microseconds since epoch
    side: str,            # 'bid' or 'ask'
    price: float,         # Price level
    size: float,          # Size at price level
    level: int,           # 0 = best, 1 = second level, etc.
    operation: str,       # 'add', 'update', 'delete'
    order_count: int,     # Number of orders (optional)
    exchange: str         # Exchange identifier (optional)
)
```

#### Trade Data
```python
TradeUpdate(
    symbol: str,          # Trading symbol
    timestamp: int,       # Microseconds since epoch
    price: float,         # Trade price
    size: float,          # Trade size
    side: str,           # 'buy' or 'sell'
    trade_id: str,       # Unique trade identifier
    exchange: str,       # Exchange identifier (optional)
    conditions: List[str] # Trade conditions/flags (optional)
)
```

#### Quote Data
```python
QuoteUpdate(
    symbol: str,          # Trading symbol
    timestamp: int,       # Microseconds since epoch
    bid_price: float,     # Best bid price
    bid_size: float,      # Best bid size
    ask_price: float,     # Best ask price
    ask_size: float,      # Best ask size
    exchange: str         # Exchange identifier (optional)
)
```

#### Bar Data (OHLCV)
```python
BarUpdate(
    symbol: str,          # Trading symbol
    timestamp: int,       # Bar start time (microseconds)
    timeframe: str,       # '1m', '5m', '1h', '1d', etc.
    open_price: float,    # Opening price
    high_price: float,    # High price
    low_price: float,     # Low price
    close_price: float,   # Closing price
    volume: float,        # Volume
    vwap: float,         # Volume weighted average price (optional)
    trade_count: int      # Number of trades (optional)
)
```

#### News/Events
```python
NewsUpdate(
    timestamp: int,       # Microseconds since epoch
    headline: str,        # News headline
    summary: str,         # News summary
    symbols: List[str],   # Affected symbols
    sentiment: str,       # 'positive', 'negative', 'neutral' (optional)
    importance: int,      # 1-10 scale (optional)
    source: str          # News source (optional)
)
```

## Integration Examples

### 1. Interactive Brokers (IBKR) Integration

```python
from data.data_interface import IBKRDataSource, register_data_source

# Create IBKR data source
ibkr_source = IBKRDataSource()

# Connect with credentials
credentials = {
    'host': '127.0.0.1',
    'port': 7497,  # TWS port
    'client_id': 1
}

if ibkr_source.connect(credentials):
    # Register the source
    register_data_source('ibkr', ibkr_source)
    
    # Set as active source
    set_active_data_source('ibkr')
    
    # Subscribe to data
    def handle_level2(update: Level2Update):
        print(f"Level 2: {update.symbol} {update.side} {update.price} x {update.size}")
    
    ibkr_source.subscribe_level2('AAPL', handle_level2)
```

### 2. Alpaca Integration

```python
from data.data_interface import AlpacaDataSource, register_data_source

# Create Alpaca data source
alpaca_source = AlpacaDataSource()

# Connect with API credentials
credentials = {
    'api_key': 'your_alpaca_api_key',
    'secret_key': 'your_alpaca_secret_key',
    'base_url': 'https://paper-api.alpaca.markets'  # Paper trading
}

if alpaca_source.connect(credentials):
    register_data_source('alpaca', alpaca_source)
    set_active_data_source('alpaca')
    
    # Subscribe to trades
    def handle_trades(update: TradeUpdate):
        print(f"Trade: {update.symbol} {update.price} x {update.size}")
    
    alpaca_source.subscribe_trades('AAPL', handle_trades)
```

### 3. Polygon.io Integration

```python
from data.data_interface import PolygonDataSource, register_data_source

# Create Polygon data source
polygon_source = PolygonDataSource()

# Connect with API key
credentials = {
    'api_key': 'your_polygon_api_key'
}

if polygon_source.connect(credentials):
    register_data_source('polygon', polygon_source)
    set_active_data_source('polygon')
    
    # Subscribe to quotes
    def handle_quotes(update: QuoteUpdate):
        print(f"Quote: {update.symbol} {update.bid_price}/{update.ask_price}")
    
    polygon_source.subscribe_quotes('AAPL', handle_quotes)
```

### 4. Custom Data Source

```python
from data.data_interface import DataSourceInterface, Level2Update
import websocket
import json

class CustomDataSource(DataSourceInterface):
    def __init__(self):
        self.ws = None
        self.callbacks = {}
        self.is_connected_flag = False
    
    def connect(self, credentials=None):
        """Connect to your custom data feed"""
        try:
            # Example WebSocket connection
            self.ws = websocket.WebSocketApp(
                "wss://your-data-feed.com/stream",
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )
            
            # Start WebSocket in background thread
            import threading
            ws_thread = threading.Thread(target=self.ws.run_forever)
            ws_thread.daemon = True
            ws_thread.start()
            
            self.is_connected_flag = True
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
    
    def _on_message(self, ws, message):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            
            # Convert your data format to standard format
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
                symbol = data['symbol']
                if symbol in self.callbacks and 'level2' in self.callbacks[symbol]:
                    for callback in self.callbacks[symbol]['level2']:
                        callback(update)
                        
        except Exception as e:
            print(f"Message processing error: {e}")
    
    def subscribe_level2(self, symbol, callback):
        """Subscribe to Level 2 data"""
        if symbol not in self.callbacks:
            self.callbacks[symbol] = {}
        if 'level2' not in self.callbacks[symbol]:
            self.callbacks[symbol]['level2'] = []
        
        self.callbacks[symbol]['level2'].append(callback)
        
        # Send subscription message to your data feed
        subscribe_msg = {
            'action': 'subscribe',
            'type': 'level2',
            'symbol': symbol
        }
        self.ws.send(json.dumps(subscribe_msg))
        return True
    
    # Implement other required methods...
    def disconnect(self):
        if self.ws:
            self.ws.close()
        self.is_connected_flag = False
    
    def is_connected(self):
        return self.is_connected_flag
    
    # ... implement remaining abstract methods

# Register your custom source
custom_source = CustomDataSource()
register_data_source('custom', custom_source)
```

## System Integration

### Using Data Sources with the Trading System

```python
from main import ChimeraTradingSystem
from data import set_active_data_source

# Initialize trading system
system = ChimeraTradingSystem()

# Set your data source
set_active_data_source('ibkr')  # or 'alpaca', 'polygon', 'custom'

# Run the system
system.run(['AAPL', 'NVDA'], duration=3600)
```

### Switching Data Sources

```python
from data import list_data_sources, get_active_data_source, set_active_data_source

# List available sources
print("Available sources:", list_data_sources())

# Check current source
current = get_active_data_source()
print(f"Current source: {current}")

# Switch to different source
set_active_data_source('polygon')
```

## Data Source Requirements

### Required Dependencies by Source

**Interactive Brokers:**
```bash
pip install ibapi
```

**Alpaca:**
```bash
pip install alpaca-trade-api
```

**Polygon.io:**
```bash
pip install polygon-api-client
```

**TD Ameritrade:**
```bash
pip install tda-api
```

### Authentication Requirements

**IBKR:**
- TWS or IB Gateway running
- Valid IB account
- API permissions enabled

**Alpaca:**
- Alpaca account (paper or live)
- API key and secret key
- Market data subscription (for real-time data)

**Polygon.io:**
- Polygon.io account
- API key
- Subscription plan for real-time data

## Performance Considerations

### Latency Optimization
- Use WebSocket connections for real-time data
- Implement connection pooling for multiple symbols
- Buffer updates to reduce callback overhead
- Use separate threads for data processing

### Memory Management
- Implement data retention policies
- Use circular buffers for high-frequency data
- Clean up old subscriptions automatically
- Monitor memory usage in production

### Error Handling
- Implement automatic reconnection logic
- Handle rate limiting gracefully
- Log all connection issues
- Provide fallback data sources

## Testing Your Integration

### Unit Tests
```python
import unittest
from data.data_interface import Level2Update

class TestCustomDataSource(unittest.TestCase):
    def setUp(self):
        self.source = CustomDataSource()
    
    def test_connection(self):
        result = self.source.connect({'api_key': 'test'})
        self.assertTrue(result)
    
    def test_subscription(self):
        updates = []
        def callback(update):
            updates.append(update)
        
        result = self.source.subscribe_level2('AAPL', callback)
        self.assertTrue(result)
        
        # Simulate data update
        # ... test logic
```

### Integration Tests
```python
from tests.test_integration import TestSystemIntegration

# Run with your data source
test = TestSystemIntegration()
test.setUp()

# Set your data source
set_active_data_source('your_source')

# Run integration tests
test.test_complete_system_simulation()
```

## Production Deployment

### Configuration Management
```python
# config.json
{
    "data_source": "ibkr",
    "credentials": {
        "host": "127.0.0.1",
        "port": 7497,
        "client_id": 1
    },
    "symbols": ["AAPL", "NVDA", "TSLA"],
    "backup_source": "polygon"
}
```

### Monitoring and Alerts
- Monitor connection status
- Track data latency and gaps
- Alert on subscription failures
- Log performance metrics

### Backup and Failover
- Configure backup data sources
- Implement automatic failover
- Store critical data locally
- Maintain data continuity

## Support and Troubleshooting

### Common Issues

**Connection Failures:**
- Check network connectivity
- Verify credentials
- Confirm API permissions
- Check rate limits

**Data Quality Issues:**
- Validate timestamp formats
- Check symbol mappings
- Verify data completeness
- Monitor for duplicates

**Performance Issues:**
- Optimize callback functions
- Reduce subscription frequency
- Use data filtering
- Implement caching

### Getting Help

1. Check the data source documentation
2. Review system logs for errors
3. Test with mock data first
4. Contact data provider support
5. Submit issues to the project repository

## Future Enhancements

### Planned Features
- Automatic data source discovery
- Smart failover mechanisms
- Data quality scoring
- Real-time performance monitoring
- Cloud-based data aggregation

### Contributing
- Submit new data source implementations
- Improve existing adapters
- Add performance optimizations
- Enhance error handling
- Write comprehensive tests

---

**Ready to integrate your data source? Start with the mock adapter and gradually replace with real data feeds!**

