"""
IBKR Data Adapter - Production Ready
Connects to Interactive Brokers TWS/Gateway with correct data types
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime
import threading
import time

try:
    from ib_insync import IB, Stock, Contract, TickData, MktDepthData
    from ib_insync.objects import BarData, Trade as IBTrade
    IBKR_AVAILABLE = True
except ImportError:
    IBKR_AVAILABLE = False
    print("WARNING: ib_insync not installed. Install with: pip install ib_insync")

from data.data_interface import DataSourceInterface, Level2Update, TradeData, QuoteData
from core.order_book import OrderBook, PriceLevel

@dataclass
class IBKRConfig:
    """IBKR Connection Configuration"""
    host: str = "127.0.0.1"
    port: int = 7497  # 7497 for paper trading, 7496 for live
    client_id: int = 1
    timeout: int = 10
    readonly: bool = True

class IBKRAdapter(DataSourceInterface):
    """
    Interactive Brokers Data Adapter
    Provides real-time Level 2 market data from IBKR TWS/Gateway
    """
    
    def __init__(self, config: IBKRConfig = None):
        if not IBKR_AVAILABLE:
            raise ImportError("ib_insync package required. Install with: pip install ib_insync")
        
        self.config = config or IBKRConfig()
        self.ib = IB()
        self.connected = False
        self.subscriptions: Dict[str, Dict] = {}
        self.contracts: Dict[str, Contract] = {}
        self.order_books: Dict[str, OrderBook] = {}
        self.callbacks: Dict[str, List[Callable]] = {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    async def connect(self) -> bool:
        """Connect to IBKR TWS/Gateway"""
        try:
            await self.ib.connectAsync(
                host=self.config.host,
                port=self.config.port,
                clientId=self.config.client_id,
                timeout=self.config.timeout,
                readonly=self.config.readonly
            )
            self.connected = True
            self.logger.info(f"Connected to IBKR at {self.config.host}:{self.config.port}")
            
            # Setup event handlers
            self.ib.pendingTickersEvent += self._on_ticker_update
            self.ib.updateEvent += self._on_market_depth_update
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to IBKR: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from IBKR"""
        if self.connected:
            self.ib.disconnect()
            self.connected = False
            self.logger.info("Disconnected from IBKR")
    
    def subscribe_level2(self, symbol: str, callback: Callable[[Level2Update], None]):
        """
        Subscribe to Level 2 market data for a symbol
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'MSFT')
            callback: Function to call with Level2Update data
        """
        if not self.connected:
            raise ConnectionError("Not connected to IBKR. Call connect() first.")
        
        # Create contract for the symbol
        contract = Stock(symbol, 'SMART', 'USD')
        self.contracts[symbol] = contract
        
        # Initialize order book
        self.order_books[symbol] = OrderBook(symbol)
        
        # Store callback
        if symbol not in self.callbacks:
            self.callbacks[symbol] = []
        self.callbacks[symbol].append(callback)
        
        try:
            # Request market depth (Level 2 data)
            self.ib.reqMktDepth(contract, numRows=10, isSmartDepth=True)
            
            # Request ticker data for trades and quotes
            ticker = self.ib.reqMktData(contract, '', False, False)
            
            self.subscriptions[symbol] = {
                'contract': contract,
                'ticker': ticker,
                'active': True
            }
            
            self.logger.info(f"Subscribed to Level 2 data for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Failed to subscribe to {symbol}: {e}")
            raise
    
    def subscribe_trades(self, symbol: str, callback: Callable[[TradeData], None]):
        """Subscribe to trade data for a symbol"""
        # Trade data comes through the same ticker subscription
        if symbol in self.subscriptions:
            # Add trade callback to existing subscription
            if 'trade_callbacks' not in self.subscriptions[symbol]:
                self.subscriptions[symbol]['trade_callbacks'] = []
            self.subscriptions[symbol]['trade_callbacks'].append(callback)
        else:
            # Create new subscription
            self.subscribe_level2(symbol, lambda x: None)  # Dummy callback
            self.subscriptions[symbol]['trade_callbacks'] = [callback]
    
    def subscribe_quotes(self, symbol: str, callback: Callable[[QuoteData], None]):
        """Subscribe to quote data for a symbol"""
        # Quote data comes through the same ticker subscription
        if symbol in self.subscriptions:
            # Add quote callback to existing subscription
            if 'quote_callbacks' not in self.subscriptions[symbol]:
                self.subscriptions[symbol]['quote_callbacks'] = []
            self.subscriptions[symbol]['quote_callbacks'].append(callback)
        else:
            # Create new subscription
            self.subscribe_level2(symbol, lambda x: None)  # Dummy callback
            self.subscriptions[symbol]['quote_callbacks'] = [callback]
    
    def unsubscribe(self, symbol: str):
        """Unsubscribe from all data for a symbol"""
        if symbol in self.subscriptions:
            contract = self.subscriptions[symbol]['contract']
            
            # Cancel market depth
            self.ib.cancelMktDepth(contract, isSmartDepth=True)
            
            # Cancel ticker
            self.ib.cancelMktData(contract)
            
            # Clean up
            del self.subscriptions[symbol]
            if symbol in self.callbacks:
                del self.callbacks[symbol]
            if symbol in self.order_books:
                del self.order_books[symbol]
            
            self.logger.info(f"Unsubscribed from {symbol}")
    
    def _on_market_depth_update(self, ticker, update):
        """Handle market depth (Level 2) updates from IBKR"""
        if hasattr(update, 'domBids') or hasattr(update, 'domAsks'):
            symbol = ticker.contract.symbol
            
            if symbol not in self.order_books:
                return
            
            order_book = self.order_books[symbol]
            
            # Update bids
            if hasattr(update, 'domBids') and update.domBids:
                for bid in update.domBids:
                    if bid.price > 0 and bid.size >= 0:
                        if bid.size == 0:
                            # Remove level
                            order_book.remove_bid(bid.price)
                        else:
                            # Update level
                            order_book.update_bid(bid.price, bid.size)
            
            # Update asks
            if hasattr(update, 'domAsks') and update.domAsks:
                for ask in update.domAsks:
                    if ask.price > 0 and ask.size >= 0:
                        if ask.size == 0:
                            # Remove level
                            order_book.remove_ask(ask.price)
                        else:
                            # Update level
                            order_book.update_ask(ask.price, ask.size)
            
            # Create Level2Update and notify callbacks
            level2_update = Level2Update(
                symbol=symbol,
                bids=[(level.price, level.size) for level in order_book.get_bids()[:10]],
                asks=[(level.price, level.size) for level in order_book.get_asks()[:10]],
                timestamp=int(time.time() * 1000)
            )
            
            # Notify all callbacks
            if symbol in self.callbacks:
                for callback in self.callbacks[symbol]:
                    try:
                        callback(level2_update)
                    except Exception as e:
                        self.logger.error(f"Error in callback for {symbol}: {e}")
    
    def _on_ticker_update(self, tickers):
        """Handle ticker updates (trades and quotes) from IBKR"""
        for ticker in tickers:
            symbol = ticker.contract.symbol
            
            if symbol not in self.subscriptions:
                continue
            
            subscription = self.subscriptions[symbol]
            
            # Handle trade data
            if hasattr(ticker, 'last') and ticker.last > 0:
                trade_data = TradeData(
                    symbol=symbol,
                    price=float(ticker.last),
                    size=int(ticker.lastSize) if ticker.lastSize else 0,
                    timestamp=int(time.time() * 1000),
                    side='BUY' if ticker.lastSize > 0 else 'SELL'  # Simplified
                )
                
                # Notify trade callbacks
                if 'trade_callbacks' in subscription:
                    for callback in subscription['trade_callbacks']:
                        try:
                            callback(trade_data)
                        except Exception as e:
                            self.logger.error(f"Error in trade callback for {symbol}: {e}")
            
            # Handle quote data
            if hasattr(ticker, 'bid') and hasattr(ticker, 'ask'):
                if ticker.bid > 0 and ticker.ask > 0:
                    quote_data = QuoteData(
                        symbol=symbol,
                        bid_price=float(ticker.bid),
                        ask_price=float(ticker.ask),
                        bid_size=int(ticker.bidSize) if ticker.bidSize else 0,
                        ask_size=int(ticker.askSize) if ticker.askSize else 0,
                        timestamp=int(time.time() * 1000)
                    )
                    
                    # Notify quote callbacks
                    if 'quote_callbacks' in subscription:
                        for callback in subscription['quote_callbacks']:
                            try:
                                callback(quote_data)
                            except Exception as e:
                                self.logger.error(f"Error in quote callback for {symbol}: {e}")
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols (basic implementation)"""
        # In a real implementation, this would query IBKR for available contracts
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'SPY', 'QQQ']
    
    def is_connected(self) -> bool:
        """Check if connected to IBKR"""
        return self.connected and self.ib.isConnected()
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get detailed connection status"""
        return {
            'connected': self.is_connected(),
            'host': self.config.host,
            'port': self.config.port,
            'client_id': self.config.client_id,
            'subscriptions': len(self.subscriptions),
            'symbols': list(self.subscriptions.keys())
        }

# Async wrapper for easier integration
class AsyncIBKRAdapter:
    """Async wrapper for IBKR adapter"""
    
    def __init__(self, config: IBKRConfig = None):
        self.adapter = IBKRAdapter(config)
        self.loop = None
        self.thread = None
        
    def start(self):
        """Start the async event loop in a separate thread"""
        def run_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self.adapter.connect())
            self.loop.run_forever()
        
        self.thread = threading.Thread(target=run_loop, daemon=True)
        self.thread.start()
        
        # Wait for connection
        timeout = 10
        start_time = time.time()
        while not self.adapter.is_connected() and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        
        return self.adapter.is_connected()
    
    def stop(self):
        """Stop the adapter and event loop"""
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.adapter:
            self.adapter.disconnect()
    
    def __getattr__(self, name):
        """Delegate all other methods to the adapter"""
        return getattr(self.adapter, name)

# Factory function for easy instantiation
def create_ibkr_adapter(host: str = "127.0.0.1", port: int = 7497, client_id: int = 1) -> AsyncIBKRAdapter:
    """
    Create and configure IBKR adapter
    
    Args:
        host: IBKR TWS/Gateway host (default: 127.0.0.1)
        port: IBKR TWS/Gateway port (7497 for paper, 7496 for live)
        client_id: Unique client ID for this connection
    
    Returns:
        Configured IBKR adapter
    """
    config = IBKRConfig(host=host, port=port, client_id=client_id)
    return AsyncIBKRAdapter(config)

