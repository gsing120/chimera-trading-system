"""
Chimera Trading System v2.0 - Data Interface Specification
Clear endpoints and data formats for external data source integration
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import time


class DataSourceType(Enum):
    """Types of data sources"""
    MOCK = "mock"
    IBKR = "ibkr"
    TD_AMERITRADE = "td_ameritrade"
    ALPACA = "alpaca"
    POLYGON = "polygon"
    CUSTOM = "custom"


@dataclass
class Level2Update:
    """Level 2 market data update"""
    symbol: str
    timestamp: int  # Microseconds since epoch
    side: str  # 'bid' or 'ask'
    price: float
    size: float
    level: int  # 0 = best bid/ask, 1 = second level, etc.
    operation: str  # 'add', 'update', 'delete'
    order_count: Optional[int] = None
    exchange: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'side': self.side,
            'price': self.price,
            'size': self.size,
            'level': self.level,
            'operation': self.operation,
            'order_count': self.order_count,
            'exchange': self.exchange
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Level2Update':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class TradeUpdate:
    """Trade execution update"""
    symbol: str
    timestamp: int  # Microseconds since epoch
    price: float
    size: float
    side: str  # 'buy' or 'sell'
    trade_id: str
    exchange: Optional[str] = None
    conditions: Optional[List[str]] = None  # Trade conditions/flags
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'price': self.price,
            'size': self.size,
            'side': self.side,
            'trade_id': self.trade_id,
            'exchange': self.exchange,
            'conditions': self.conditions
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradeUpdate':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class QuoteUpdate:
    """Best bid/ask quote update"""
    symbol: str
    timestamp: int  # Microseconds since epoch
    bid_price: float
    bid_size: float
    ask_price: float
    ask_size: float
    exchange: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'bid_price': self.bid_price,
            'bid_size': self.bid_size,
            'ask_price': self.ask_price,
            'ask_size': self.ask_size,
            'exchange': self.exchange
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QuoteUpdate':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class BarUpdate:
    """OHLCV bar update"""
    symbol: str
    timestamp: int  # Microseconds since epoch
    timeframe: str  # '1m', '5m', '1h', '1d', etc.
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    vwap: Optional[float] = None
    trade_count: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'timeframe': self.timeframe,
            'open_price': self.open_price,
            'high_price': self.high_price,
            'low_price': self.low_price,
            'close_price': self.close_price,
            'volume': self.volume,
            'vwap': self.vwap,
            'trade_count': self.trade_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BarUpdate':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class NewsUpdate:
    """News/event update"""
    timestamp: int  # Microseconds since epoch
    headline: str
    summary: str
    symbols: List[str]  # Affected symbols
    sentiment: Optional[str] = None  # 'positive', 'negative', 'neutral'
    importance: Optional[int] = None  # 1-10 scale
    source: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'timestamp': self.timestamp,
            'headline': self.headline,
            'summary': self.summary,
            'symbols': self.symbols,
            'sentiment': self.sentiment,
            'importance': self.importance,
            'source': self.source
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NewsUpdate':
        """Create from dictionary"""
        return cls(**data)


class DataSourceInterface(ABC):
    """
    Abstract interface for market data sources
    Implement this interface to connect your own data source
    """
    
    @abstractmethod
    def connect(self, credentials: Optional[Dict[str, Any]] = None) -> bool:
        """
        Connect to the data source
        
        Args:
            credentials: Authentication credentials (API keys, tokens, etc.)
            
        Returns:
            bool: True if connection successful
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the data source"""
        pass
    
    @abstractmethod
    def subscribe_level2(self, symbol: str, callback: Callable[[Level2Update], None]) -> bool:
        """
        Subscribe to Level 2 market data
        
        Args:
            symbol: Trading symbol (e.g., 'AAPL', 'TSLA')
            callback: Function to call when Level 2 update received
            
        Returns:
            bool: True if subscription successful
        """
        pass
    
    @abstractmethod
    def subscribe_trades(self, symbol: str, callback: Callable[[TradeUpdate], None]) -> bool:
        """
        Subscribe to trade data
        
        Args:
            symbol: Trading symbol
            callback: Function to call when trade update received
            
        Returns:
            bool: True if subscription successful
        """
        pass
    
    @abstractmethod
    def subscribe_quotes(self, symbol: str, callback: Callable[[QuoteUpdate], None]) -> bool:
        """
        Subscribe to quote data (best bid/ask)
        
        Args:
            symbol: Trading symbol
            callback: Function to call when quote update received
            
        Returns:
            bool: True if subscription successful
        """
        pass
    
    @abstractmethod
    def subscribe_bars(self, symbol: str, timeframe: str, 
                      callback: Callable[[BarUpdate], None]) -> bool:
        """
        Subscribe to OHLCV bar data
        
        Args:
            symbol: Trading symbol
            timeframe: Bar timeframe ('1m', '5m', '1h', etc.)
            callback: Function to call when bar update received
            
        Returns:
            bool: True if subscription successful
        """
        pass
    
    @abstractmethod
    def subscribe_news(self, symbols: List[str], 
                      callback: Callable[[NewsUpdate], None]) -> bool:
        """
        Subscribe to news/events
        
        Args:
            symbols: List of symbols to monitor
            callback: Function to call when news update received
            
        Returns:
            bool: True if subscription successful
        """
        pass
    
    @abstractmethod
    def unsubscribe(self, symbol: str, data_type: str) -> bool:
        """
        Unsubscribe from data
        
        Args:
            symbol: Trading symbol
            data_type: Type of data ('level2', 'trades', 'quotes', 'bars', 'news')
            
        Returns:
            bool: True if unsubscription successful
        """
        pass
    
    @abstractmethod
    def get_historical_bars(self, symbol: str, timeframe: str, 
                           start_time: int, end_time: int) -> List[BarUpdate]:
        """
        Get historical bar data
        
        Args:
            symbol: Trading symbol
            timeframe: Bar timeframe
            start_time: Start timestamp (microseconds)
            end_time: End timestamp (microseconds)
            
        Returns:
            List[BarUpdate]: Historical bars
        """
        pass
    
    @abstractmethod
    def get_historical_trades(self, symbol: str, 
                             start_time: int, end_time: int) -> List[TradeUpdate]:
        """
        Get historical trade data
        
        Args:
            symbol: Trading symbol
            start_time: Start timestamp (microseconds)
            end_time: End timestamp (microseconds)
            
        Returns:
            List[TradeUpdate]: Historical trades
        """
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected to data source"""
        pass
    
    @abstractmethod
    def get_supported_symbols(self) -> List[str]:
        """Get list of supported trading symbols"""
        pass
    
    @abstractmethod
    def get_market_status(self, symbol: str) -> Dict[str, Any]:
        """
        Get market status for symbol
        
        Returns:
            Dict with keys: 'is_open', 'next_open', 'next_close', 'timezone'
        """
        pass


class DataSourceRegistry:
    """
    Registry for managing multiple data sources
    """
    
    def __init__(self):
        self._sources: Dict[str, DataSourceInterface] = {}
        self._active_source: Optional[str] = None
    
    def register_source(self, name: str, source: DataSourceInterface) -> None:
        """Register a data source"""
        self._sources[name] = source
    
    def set_active_source(self, name: str) -> bool:
        """Set the active data source"""
        if name in self._sources:
            self._active_source = name
            return True
        return False
    
    def get_active_source(self) -> Optional[DataSourceInterface]:
        """Get the currently active data source"""
        if self._active_source and self._active_source in self._sources:
            return self._sources[self._active_source]
        return None
    
    def get_source(self, name: str) -> Optional[DataSourceInterface]:
        """Get a specific data source by name"""
        return self._sources.get(name)
    
    def list_sources(self) -> List[str]:
        """List all registered data sources"""
        return list(self._sources.keys())


# Example implementation templates for popular data sources

class IBKRDataSource(DataSourceInterface):
    """
    Interactive Brokers data source implementation template
    
    To implement:
    1. Install ibapi: pip install ibapi
    2. Implement connection logic using IB Gateway/TWS
    3. Handle IB's specific data formats and convert to our standard format
    """
    
    def __init__(self):
        self.client = None  # IB API client
        self.is_connected_flag = False
    
    def connect(self, credentials: Optional[Dict[str, Any]] = None) -> bool:
        """
        Connect to IB Gateway/TWS
        
        Expected credentials:
        {
            'host': '127.0.0.1',
            'port': 7497,  # 7497 for TWS, 4001 for IB Gateway
            'client_id': 1
        }
        """
        # TODO: Implement IB connection
        # from ibapi.client import EClient
        # from ibapi.wrapper import EWrapper
        # self.client = IBClient(self)
        # self.client.connect(host, port, client_id)
        return False
    
    def disconnect(self) -> None:
        """Disconnect from IB"""
        # TODO: Implement IB disconnection
        pass
    
    def subscribe_level2(self, symbol: str, callback: Callable[[Level2Update], None]) -> bool:
        """Subscribe to IB Level 2 data"""
        # TODO: Implement IB Level 2 subscription
        # Use reqMktDepth() and handle updateMktDepth() callbacks
        return False
    
    def subscribe_trades(self, symbol: str, callback: Callable[[TradeUpdate], None]) -> bool:
        """Subscribe to IB trade data"""
        # TODO: Implement IB trade subscription
        # Use reqTickByTickData() with tickType="Last"
        return False
    
    def subscribe_quotes(self, symbol: str, callback: Callable[[QuoteUpdate], None]) -> bool:
        """Subscribe to IB quote data"""
        # TODO: Implement IB quote subscription
        # Use reqMktData() and handle tickPrice() callbacks
        return False
    
    def subscribe_bars(self, symbol: str, timeframe: str, 
                      callback: Callable[[BarUpdate], None]) -> bool:
        """Subscribe to IB bar data"""
        # TODO: Implement IB bar subscription
        # Use reqRealTimeBars() for real-time bars
        return False
    
    def subscribe_news(self, symbols: List[str], 
                      callback: Callable[[NewsUpdate], None]) -> bool:
        """Subscribe to IB news"""
        # TODO: Implement IB news subscription
        # Use reqNewsProviders() and reqNewsArticle()
        return False
    
    def unsubscribe(self, symbol: str, data_type: str) -> bool:
        """Unsubscribe from IB data"""
        # TODO: Implement IB unsubscription
        return False
    
    def get_historical_bars(self, symbol: str, timeframe: str, 
                           start_time: int, end_time: int) -> List[BarUpdate]:
        """Get IB historical bars"""
        # TODO: Implement IB historical data request
        # Use reqHistoricalData()
        return []
    
    def get_historical_trades(self, symbol: str, 
                             start_time: int, end_time: int) -> List[TradeUpdate]:
        """Get IB historical trades"""
        # TODO: Implement IB historical tick data
        # Use reqHistoricalTicks()
        return []
    
    def is_connected(self) -> bool:
        """Check IB connection status"""
        return self.is_connected_flag
    
    def get_supported_symbols(self) -> List[str]:
        """Get IB supported symbols"""
        # TODO: Implement symbol lookup
        return []
    
    def get_market_status(self, symbol: str) -> Dict[str, Any]:
        """Get IB market status"""
        # TODO: Implement market hours check
        return {'is_open': False, 'next_open': None, 'next_close': None, 'timezone': 'UTC'}


class AlpacaDataSource(DataSourceInterface):
    """
    Alpaca data source implementation template
    
    To implement:
    1. Install alpaca-trade-api: pip install alpaca-trade-api
    2. Get API keys from Alpaca
    3. Implement real-time and historical data feeds
    """
    
    def __init__(self):
        self.api = None
        self.stream = None
        self.is_connected_flag = False
    
    def connect(self, credentials: Optional[Dict[str, Any]] = None) -> bool:
        """
        Connect to Alpaca
        
        Expected credentials:
        {
            'api_key': 'your_api_key',
            'secret_key': 'your_secret_key',
            'base_url': 'https://paper-api.alpaca.markets'  # or live URL
        }
        """
        # TODO: Implement Alpaca connection
        # import alpaca_trade_api as tradeapi
        # self.api = tradeapi.REST(api_key, secret_key, base_url)
        return False
    
    def disconnect(self) -> None:
        """Disconnect from Alpaca"""
        # TODO: Implement Alpaca disconnection
        pass
    
    def subscribe_level2(self, symbol: str, callback: Callable[[Level2Update], None]) -> bool:
        """Subscribe to Alpaca Level 2 data (if available)"""
        # Note: Alpaca may not have full Level 2 data
        return False
    
    def subscribe_trades(self, symbol: str, callback: Callable[[TradeUpdate], None]) -> bool:
        """Subscribe to Alpaca trade data"""
        # TODO: Implement Alpaca trade stream
        # Use stream.subscribe_trades()
        return False
    
    def subscribe_quotes(self, symbol: str, callback: Callable[[QuoteUpdate], None]) -> bool:
        """Subscribe to Alpaca quote data"""
        # TODO: Implement Alpaca quote stream
        # Use stream.subscribe_quotes()
        return False
    
    def subscribe_bars(self, symbol: str, timeframe: str, 
                      callback: Callable[[BarUpdate], None]) -> bool:
        """Subscribe to Alpaca bar data"""
        # TODO: Implement Alpaca bar stream
        # Use stream.subscribe_bars()
        return False
    
    def subscribe_news(self, symbols: List[str], 
                      callback: Callable[[NewsUpdate], None]) -> bool:
        """Subscribe to Alpaca news"""
        # TODO: Implement Alpaca news stream
        # Use stream.subscribe_news()
        return False
    
    def unsubscribe(self, symbol: str, data_type: str) -> bool:
        """Unsubscribe from Alpaca data"""
        # TODO: Implement Alpaca unsubscription
        return False
    
    def get_historical_bars(self, symbol: str, timeframe: str, 
                           start_time: int, end_time: int) -> List[BarUpdate]:
        """Get Alpaca historical bars"""
        # TODO: Implement Alpaca historical bars
        # Use api.get_bars()
        return []
    
    def get_historical_trades(self, symbol: str, 
                             start_time: int, end_time: int) -> List[TradeUpdate]:
        """Get Alpaca historical trades"""
        # TODO: Implement Alpaca historical trades
        # Use api.get_trades()
        return []
    
    def is_connected(self) -> bool:
        """Check Alpaca connection status"""
        return self.is_connected_flag
    
    def get_supported_symbols(self) -> List[str]:
        """Get Alpaca supported symbols"""
        # TODO: Implement Alpaca asset lookup
        return []
    
    def get_market_status(self, symbol: str) -> Dict[str, Any]:
        """Get Alpaca market status"""
        # TODO: Implement Alpaca market calendar
        return {'is_open': False, 'next_open': None, 'next_close': None, 'timezone': 'UTC'}


class PolygonDataSource(DataSourceInterface):
    """
    Polygon.io data source implementation template
    
    To implement:
    1. Install polygon-api-client: pip install polygon-api-client
    2. Get API key from Polygon.io
    3. Implement real-time and historical data feeds
    """
    
    def __init__(self):
        self.client = None
        self.websocket = None
        self.is_connected_flag = False
    
    def connect(self, credentials: Optional[Dict[str, Any]] = None) -> bool:
        """
        Connect to Polygon.io
        
        Expected credentials:
        {
            'api_key': 'your_polygon_api_key'
        }
        """
        # TODO: Implement Polygon connection
        # from polygon import RESTClient, WebSocketClient
        # self.client = RESTClient(api_key)
        # self.websocket = WebSocketClient(api_key)
        return False
    
    def disconnect(self) -> None:
        """Disconnect from Polygon"""
        # TODO: Implement Polygon disconnection
        pass
    
    def subscribe_level2(self, symbol: str, callback: Callable[[Level2Update], None]) -> bool:
        """Subscribe to Polygon Level 2 data"""
        # TODO: Implement Polygon Level 2 subscription
        # Use websocket.subscribe_level2()
        return False
    
    def subscribe_trades(self, symbol: str, callback: Callable[[TradeUpdate], None]) -> bool:
        """Subscribe to Polygon trade data"""
        # TODO: Implement Polygon trade stream
        # Use websocket.subscribe_trades()
        return False
    
    def subscribe_quotes(self, symbol: str, callback: Callable[[QuoteUpdate], None]) -> bool:
        """Subscribe to Polygon quote data"""
        # TODO: Implement Polygon quote stream
        # Use websocket.subscribe_quotes()
        return False
    
    def subscribe_bars(self, symbol: str, timeframe: str, 
                      callback: Callable[[BarUpdate], None]) -> bool:
        """Subscribe to Polygon bar data"""
        # TODO: Implement Polygon bar stream
        # Use websocket.subscribe_bars()
        return False
    
    def subscribe_news(self, symbols: List[str], 
                      callback: Callable[[NewsUpdate], None]) -> bool:
        """Subscribe to Polygon news"""
        # TODO: Implement Polygon news stream
        return False
    
    def unsubscribe(self, symbol: str, data_type: str) -> bool:
        """Unsubscribe from Polygon data"""
        # TODO: Implement Polygon unsubscription
        return False
    
    def get_historical_bars(self, symbol: str, timeframe: str, 
                           start_time: int, end_time: int) -> List[BarUpdate]:
        """Get Polygon historical bars"""
        # TODO: Implement Polygon historical bars
        # Use client.get_aggs()
        return []
    
    def get_historical_trades(self, symbol: str, 
                             start_time: int, end_time: int) -> List[TradeUpdate]:
        """Get Polygon historical trades"""
        # TODO: Implement Polygon historical trades
        # Use client.get_trades()
        return []
    
    def is_connected(self) -> bool:
        """Check Polygon connection status"""
        return self.is_connected_flag
    
    def get_supported_symbols(self) -> List[str]:
        """Get Polygon supported symbols"""
        # TODO: Implement Polygon ticker lookup
        return []
    
    def get_market_status(self, symbol: str) -> Dict[str, Any]:
        """Get Polygon market status"""
        # TODO: Implement Polygon market status
        return {'is_open': False, 'next_open': None, 'next_close': None, 'timezone': 'UTC'}


# Global registry instance
data_source_registry = DataSourceRegistry()


def register_data_source(name: str, source: DataSourceInterface) -> None:
    """Register a data source globally"""
    data_source_registry.register_source(name, source)


def get_data_source(name: str) -> Optional[DataSourceInterface]:
    """Get a registered data source"""
    return data_source_registry.get_source(name)


def set_active_data_source(name: str) -> bool:
    """Set the active data source"""
    return data_source_registry.set_active_source(name)


def get_active_data_source() -> Optional[DataSourceInterface]:
    """Get the currently active data source"""
    return data_source_registry.get_active_source()


def list_data_sources() -> List[str]:
    """List all registered data sources"""
    return data_source_registry.list_sources()


# Utility functions for data conversion

def timestamp_to_microseconds(timestamp: float) -> int:
    """Convert timestamp to microseconds since epoch"""
    return int(timestamp * 1_000_000)


def microseconds_to_timestamp(microseconds: int) -> float:
    """Convert microseconds since epoch to timestamp"""
    return microseconds / 1_000_000


def format_symbol(symbol: str, source_type: DataSourceType) -> str:
    """Format symbol for specific data source"""
    if source_type == DataSourceType.IBKR:
        # IB uses specific contract formats
        return symbol.upper()
    elif source_type == DataSourceType.ALPACA:
        # Alpaca uses simple symbols
        return symbol.upper()
    elif source_type == DataSourceType.POLYGON:
        # Polygon uses specific formats
        return symbol.upper()
    else:
        return symbol.upper()


def validate_update(update: Any) -> bool:
    """Validate data update format"""
    required_fields = {
        Level2Update: ['symbol', 'timestamp', 'side', 'price', 'size', 'level', 'operation'],
        TradeUpdate: ['symbol', 'timestamp', 'price', 'size', 'side', 'trade_id'],
        QuoteUpdate: ['symbol', 'timestamp', 'bid_price', 'bid_size', 'ask_price', 'ask_size'],
        BarUpdate: ['symbol', 'timestamp', 'timeframe', 'open_price', 'high_price', 'low_price', 'close_price', 'volume'],
        NewsUpdate: ['timestamp', 'headline', 'summary', 'symbols']
    }
    
    update_type = type(update)
    if update_type not in required_fields:
        return False
    
    for field in required_fields[update_type]:
        if not hasattr(update, field) or getattr(update, field) is None:
            return False
    
    return True

