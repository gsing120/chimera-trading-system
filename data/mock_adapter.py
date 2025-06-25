"""
Data Source Adapter
Adapts the existing mock data generator to the new data interface
"""

import time
import threading
from typing import Dict, List, Optional, Any, Callable

from .data_interface import (
    DataSourceInterface, Level2Update, TradeUpdate, QuoteUpdate, 
    BarUpdate, NewsUpdate, DataSourceType
)
from .mock_data_generator import MockDataGenerator
from core.data_orchestrator import DataOrchestrator


class MockDataSourceAdapter(DataSourceInterface):
    """
    Adapter that wraps the existing mock data generator
    to implement the standard data source interface
    """
    
    def __init__(self):
        self.generators: Dict[str, MockDataGenerator] = {}
        self.data_orchestrator: Optional[DataOrchestrator] = None
        self.is_connected_flag = False
        self.subscriptions: Dict[str, Dict[str, List[Callable]]] = {}
        self._lock = threading.RLock()
        
    def connect(self, credentials: Optional[Dict[str, Any]] = None) -> bool:
        """Connect to mock data source"""
        with self._lock:
            if not self.is_connected_flag:
                # Initialize data orchestrator for mock data
                self.data_orchestrator = DataOrchestrator("mock_data.db")
                self.data_orchestrator.start()
                self.is_connected_flag = True
                print("✓ Connected to mock data source")
            return True
    
    def disconnect(self) -> None:
        """Disconnect from mock data source"""
        with self._lock:
            if self.is_connected_flag:
                # Stop all generators
                for generator in self.generators.values():
                    generator.stop_generation()
                
                # Stop data orchestrator
                if self.data_orchestrator:
                    self.data_orchestrator.stop()
                
                self.generators.clear()
                self.subscriptions.clear()
                self.is_connected_flag = False
                print("✓ Disconnected from mock data source")
    
    def subscribe_level2(self, symbol: str, callback: Callable[[Level2Update], None]) -> bool:
        """Subscribe to Level 2 market data"""
        with self._lock:
            if not self.is_connected_flag:
                return False
            
            # Initialize subscription tracking
            if symbol not in self.subscriptions:
                self.subscriptions[symbol] = {}
            if 'level2' not in self.subscriptions[symbol]:
                self.subscriptions[symbol]['level2'] = []
            
            self.subscriptions[symbol]['level2'].append(callback)
            
            # Create generator if needed
            if symbol not in self.generators:
                self._create_generator(symbol)
            
            # Set up callback wrapper
            def level2_wrapper(update):
                if update.update_type == 'level2':
                    level2_update = Level2Update(
                        symbol=update.symbol,
                        timestamp=update.timestamp,
                        side=update.data.get('side', 'bid'),
                        price=update.data.get('price', 0.0),
                        size=update.data.get('size', 0.0),
                        level=update.data.get('level', 0),
                        operation=update.data.get('operation', 'update'),
                        order_count=update.data.get('order_count'),
                        exchange='MOCK'
                    )
                    callback(level2_update)
            
            # Subscribe to data orchestrator updates
            self.data_orchestrator.subscribe_to_updates(level2_wrapper, [symbol])
            
            print(f"✓ Subscribed to Level 2 data for {symbol}")
            return True
    
    def subscribe_trades(self, symbol: str, callback: Callable[[TradeUpdate], None]) -> bool:
        """Subscribe to trade data"""
        with self._lock:
            if not self.is_connected_flag:
                return False
            
            # Initialize subscription tracking
            if symbol not in self.subscriptions:
                self.subscriptions[symbol] = {}
            if 'trades' not in self.subscriptions[symbol]:
                self.subscriptions[symbol]['trades'] = []
            
            self.subscriptions[symbol]['trades'].append(callback)
            
            # Create generator if needed
            if symbol not in self.generators:
                self._create_generator(symbol)
            
            # Set up callback wrapper
            def trade_wrapper(update):
                if update.update_type == 'trade':
                    trade_update = TradeUpdate(
                        symbol=update.symbol,
                        timestamp=update.timestamp,
                        price=update.data.get('price', 0.0),
                        size=update.data.get('size', 0.0),
                        side=update.data.get('side', 'buy'),
                        trade_id=update.data.get('trade_id', ''),
                        exchange='MOCK'
                    )
                    callback(trade_update)
            
            # Subscribe to data orchestrator updates
            self.data_orchestrator.subscribe_to_updates(trade_wrapper, [symbol])
            
            print(f"✓ Subscribed to trade data for {symbol}")
            return True
    
    def subscribe_quotes(self, symbol: str, callback: Callable[[QuoteUpdate], None]) -> bool:
        """Subscribe to quote data"""
        with self._lock:
            if not self.is_connected_flag:
                return False
            
            # Initialize subscription tracking
            if symbol not in self.subscriptions:
                self.subscriptions[symbol] = {}
            if 'quotes' not in self.subscriptions[symbol]:
                self.subscriptions[symbol]['quotes'] = []
            
            self.subscriptions[symbol]['quotes'].append(callback)
            
            # Create generator if needed
            if symbol not in self.generators:
                self._create_generator(symbol)
            
            # Set up callback wrapper to generate quotes from order book
            def quote_wrapper(update):
                if update.update_type == 'level2':
                    # Get current order book state
                    order_book = self.data_orchestrator.get_order_book(symbol)
                    if order_book:
                        best_bid = order_book.get_best_bid()
                        best_ask = order_book.get_best_ask()
                        
                        if best_bid and best_ask:
                            quote_update = QuoteUpdate(
                                symbol=symbol,
                                timestamp=update.timestamp,
                                bid_price=best_bid.price,
                                bid_size=best_bid.size,
                                ask_price=best_ask.price,
                                ask_size=best_ask.size,
                                exchange='MOCK'
                            )
                            callback(quote_update)
            
            # Subscribe to data orchestrator updates
            self.data_orchestrator.subscribe_to_updates(quote_wrapper, [symbol])
            
            print(f"✓ Subscribed to quote data for {symbol}")
            return True
    
    def subscribe_bars(self, symbol: str, timeframe: str, 
                      callback: Callable[[BarUpdate], None]) -> bool:
        """Subscribe to bar data"""
        with self._lock:
            if not self.is_connected_flag:
                return False
            
            # Initialize subscription tracking
            if symbol not in self.subscriptions:
                self.subscriptions[symbol] = {}
            if 'bars' not in self.subscriptions[symbol]:
                self.subscriptions[symbol]['bars'] = []
            
            self.subscriptions[symbol]['bars'].append(callback)
            
            # Create generator if needed
            if symbol not in self.generators:
                self._create_generator(symbol)
            
            # For mock data, generate bars periodically
            def generate_bars():
                """Generate mock bars periodically"""
                timeframe_seconds = self._parse_timeframe(timeframe)
                
                while self.is_connected_flag and symbol in self.subscriptions:
                    if 'bars' in self.subscriptions[symbol]:
                        # Generate mock bar data
                        generator = self.generators.get(symbol)
                        if generator:
                            current_price = generator.current_price
                            
                            bar_update = BarUpdate(
                                symbol=symbol,
                                timestamp=int(time.time() * 1_000_000),
                                timeframe=timeframe,
                                open_price=current_price * 0.999,
                                high_price=current_price * 1.002,
                                low_price=current_price * 0.998,
                                close_price=current_price,
                                volume=1000.0,
                                vwap=current_price,
                                trade_count=10
                            )
                            
                            for cb in self.subscriptions[symbol]['bars']:
                                cb(bar_update)
                    
                    time.sleep(timeframe_seconds)
            
            # Start bar generation thread
            bar_thread = threading.Thread(target=generate_bars, daemon=True)
            bar_thread.start()
            
            print(f"✓ Subscribed to {timeframe} bars for {symbol}")
            return True
    
    def subscribe_news(self, symbols: List[str], 
                      callback: Callable[[NewsUpdate], None]) -> bool:
        """Subscribe to news data"""
        with self._lock:
            if not self.is_connected_flag:
                return False
            
            # For mock data, generate periodic news events
            def generate_news():
                """Generate mock news events"""
                news_events = [
                    "Company reports strong quarterly earnings",
                    "New product launch announced",
                    "Regulatory approval received",
                    "Partnership agreement signed",
                    "Market volatility expected"
                ]
                
                while self.is_connected_flag:
                    import random
                    
                    news_update = NewsUpdate(
                        timestamp=int(time.time() * 1_000_000),
                        headline=random.choice(news_events),
                        summary="Mock news event for testing purposes",
                        symbols=symbols,
                        sentiment=random.choice(['positive', 'negative', 'neutral']),
                        importance=random.randint(1, 10),
                        source='MOCK_NEWS'
                    )
                    
                    callback(news_update)
                    time.sleep(300)  # News every 5 minutes
            
            # Start news generation thread
            news_thread = threading.Thread(target=generate_news, daemon=True)
            news_thread.start()
            
            print(f"✓ Subscribed to news for {symbols}")
            return True
    
    def unsubscribe(self, symbol: str, data_type: str) -> bool:
        """Unsubscribe from data"""
        with self._lock:
            if symbol in self.subscriptions and data_type in self.subscriptions[symbol]:
                del self.subscriptions[symbol][data_type]
                
                # If no more subscriptions for this symbol, stop generator
                if not self.subscriptions[symbol]:
                    if symbol in self.generators:
                        self.generators[symbol].stop_generation()
                        del self.generators[symbol]
                    del self.subscriptions[symbol]
                
                print(f"✓ Unsubscribed from {data_type} data for {symbol}")
                return True
            return False
    
    def get_historical_bars(self, symbol: str, timeframe: str, 
                           start_time: int, end_time: int) -> List[BarUpdate]:
        """Get historical bar data"""
        # Generate mock historical bars
        bars = []
        timeframe_seconds = self._parse_timeframe(timeframe)
        
        current_time = start_time
        base_price = 100.0
        
        while current_time < end_time:
            # Generate mock OHLCV data
            open_price = base_price
            high_price = open_price * (1 + 0.01)
            low_price = open_price * (1 - 0.01)
            close_price = open_price + (high_price - low_price) * 0.5
            
            bar = BarUpdate(
                symbol=symbol,
                timestamp=current_time,
                timeframe=timeframe,
                open_price=open_price,
                high_price=high_price,
                low_price=low_price,
                close_price=close_price,
                volume=1000.0,
                vwap=close_price,
                trade_count=10
            )
            bars.append(bar)
            
            current_time += timeframe_seconds * 1_000_000  # Convert to microseconds
            base_price = close_price
        
        return bars
    
    def get_historical_trades(self, symbol: str, 
                             start_time: int, end_time: int) -> List[TradeUpdate]:
        """Get historical trade data"""
        # Generate mock historical trades
        trades = []
        current_time = start_time
        base_price = 100.0
        
        trade_count = 0
        while current_time < end_time and trade_count < 1000:  # Limit for demo
            trade = TradeUpdate(
                symbol=symbol,
                timestamp=current_time,
                price=base_price + (trade_count % 10 - 5) * 0.01,
                size=100.0,
                side='buy' if trade_count % 2 == 0 else 'sell',
                trade_id=f"mock_trade_{trade_count}",
                exchange='MOCK'
            )
            trades.append(trade)
            
            current_time += 1_000_000  # 1 second intervals
            trade_count += 1
        
        return trades
    
    def is_connected(self) -> bool:
        """Check connection status"""
        return self.is_connected_flag
    
    def get_supported_symbols(self) -> List[str]:
        """Get supported symbols"""
        return ['MOCK', 'AAPL', 'NVDA', 'TSLA', 'SPY', 'QQQ', 'DEMO']
    
    def get_market_status(self, symbol: str) -> Dict[str, Any]:
        """Get market status"""
        return {
            'is_open': True,  # Mock market is always open
            'next_open': None,
            'next_close': None,
            'timezone': 'UTC'
        }
    
    def _create_generator(self, symbol: str) -> None:
        """Create and start a mock data generator for symbol"""
        if symbol not in self.generators:
            base_prices = {
                'AAPL': 150.0,
                'NVDA': 400.0,
                'TSLA': 200.0,
                'SPY': 450.0,
                'QQQ': 350.0,
                'MOCK': 100.0,
                'DEMO': 100.0
            }
            
            base_price = base_prices.get(symbol, 100.0)
            generator = MockDataGenerator(symbol, base_price)
            
            # Start generation
            generator.start_generation(self.data_orchestrator, 50)  # 50 updates/sec
            self.generators[symbol] = generator
    
    def _parse_timeframe(self, timeframe: str) -> int:
        """Parse timeframe string to seconds"""
        timeframe_map = {
            '1s': 1,
            '5s': 5,
            '10s': 10,
            '30s': 30,
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '30m': 1800,
            '1h': 3600,
            '4h': 14400,
            '1d': 86400
        }
        return timeframe_map.get(timeframe, 60)  # Default to 1 minute


# Register the mock data source
from .data_interface import register_data_source

mock_source = MockDataSourceAdapter()
register_data_source('mock', mock_source)

