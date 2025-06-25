"""
Order Book Management System
Handles Level 2 market data with high-performance operations
"""

import time
import heapq
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import threading


@dataclass
class OrderBookLevel:
    """Represents a single price level in the order book"""
    price: float
    size: int
    orders: int
    timestamp: int
    side: str  # 'bid' or 'ask'
    
    def __post_init__(self):
        if self.side not in ['bid', 'ask']:
            raise ValueError("Side must be 'bid' or 'ask'")


@dataclass
class Trade:
    """Represents an executed trade"""
    price: float
    size: int
    timestamp: int
    side: str  # 'buy' or 'sell' (aggressor side)
    trade_id: str


class OrderBook:
    """
    High-performance order book implementation
    Optimized for real-time Level 2 data processing
    """
    
    def __init__(self, symbol: str, max_levels: int = 100):
        self.symbol = symbol
        self.max_levels = max_levels
        
        # Use heaps for efficient price-ordered access
        self._bids = []  # Max heap (negative prices for max behavior)
        self._asks = []  # Min heap
        
        # Price -> Level mapping for O(1) updates
        self._bid_levels: Dict[float, OrderBookLevel] = {}
        self._ask_levels: Dict[float, OrderBookLevel] = {}
        
        # Historical data for analysis
        self._trade_history = deque(maxlen=1000)
        self._book_snapshots = deque(maxlen=100)
        
        # Metrics
        self.last_update_time = 0
        self.update_count = 0
        
        # Thread safety
        self._lock = threading.RLock()
    
    def update_level(self, price: float, size: int, orders: int, side: str) -> None:
        """Update a single price level"""
        with self._lock:
            timestamp = int(time.time() * 1000000)  # microseconds
            
            if side == 'bid':
                if size == 0:
                    # Remove level
                    if price in self._bid_levels:
                        del self._bid_levels[price]
                        # Remove from heap (will be filtered out during access)
                else:
                    # Add or update level
                    level = OrderBookLevel(price, size, orders, timestamp, side)
                    self._bid_levels[price] = level
                    heapq.heappush(self._bids, -price)  # Negative for max heap
                    
            elif side == 'ask':
                if size == 0:
                    # Remove level
                    if price in self._ask_levels:
                        del self._ask_levels[price]
                else:
                    # Add or update level
                    level = OrderBookLevel(price, size, orders, timestamp, side)
                    self._ask_levels[price] = level
                    heapq.heappush(self._asks, price)
            
            self.last_update_time = timestamp
            self.update_count += 1
            
            # Clean up heaps periodically
            if self.update_count % 1000 == 0:
                self._cleanup_heaps()
    
    def add_trade(self, price: float, size: int, side: str, trade_id: str = None) -> None:
        """Add a trade to the history"""
        with self._lock:
            timestamp = int(time.time() * 1000000)
            if trade_id is None:
                trade_id = f"{self.symbol}_{timestamp}_{len(self._trade_history)}"
            
            trade = Trade(price, size, timestamp, side, trade_id)
            self._trade_history.append(trade)
    
    def get_best_bid(self) -> Optional[OrderBookLevel]:
        """Get the best bid (highest price)"""
        with self._lock:
            while self._bids:
                price = -heapq.heappop(self._bids)
                if price in self._bid_levels:
                    heapq.heappush(self._bids, -price)  # Put it back
                    return self._bid_levels[price]
            return None
    
    def get_best_ask(self) -> Optional[OrderBookLevel]:
        """Get the best ask (lowest price)"""
        with self._lock:
            while self._asks:
                price = heapq.heappop(self._asks)
                if price in self._ask_levels:
                    heapq.heappush(self._asks, price)  # Put it back
                    return self._ask_levels[price]
            return None
    
    def get_spread(self) -> Optional[float]:
        """Get the bid-ask spread"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid and best_ask:
            return best_ask.price - best_bid.price
        return None
    
    def get_mid_price(self) -> Optional[float]:
        """Get the mid price"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid and best_ask:
            return (best_bid.price + best_ask.price) / 2.0
        return None
    
    def get_bids(self, levels: int = None) -> List[OrderBookLevel]:
        """Get bid levels sorted by price (highest first)"""
        if levels is None:
            levels = self.max_levels
            
        with self._lock:
            sorted_prices = sorted(self._bid_levels.keys(), reverse=True)
            return [self._bid_levels[price] for price in sorted_prices[:levels]]
    
    def get_asks(self, levels: int = None) -> List[OrderBookLevel]:
        """Get ask levels sorted by price (lowest first)"""
        if levels is None:
            levels = self.max_levels
            
        with self._lock:
            sorted_prices = sorted(self._ask_levels.keys())
            return [self._ask_levels[price] for price in sorted_prices[:levels]]
    
    def get_depth(self, side: str, levels: int = 10) -> float:
        """Calculate total size for given number of levels"""
        with self._lock:
            if side == 'bid':
                bid_levels = self.get_bids(levels)
                return sum(level.size for level in bid_levels)
            elif side == 'ask':
                ask_levels = self.get_asks(levels)
                return sum(level.size for level in ask_levels)
            else:
                raise ValueError("Side must be 'bid' or 'ask'")
    
    def get_imbalance(self, levels: int = 5) -> float:
        """Calculate order book imbalance"""
        bid_depth = self.get_depth('bid', levels)
        ask_depth = self.get_depth('ask', levels)
        
        if bid_depth + ask_depth == 0:
            return 0.0
        
        return (bid_depth - ask_depth) / (bid_depth + ask_depth)
    
    def get_recent_trades(self, count: int = 100) -> List[Trade]:
        """Get recent trades"""
        with self._lock:
            return list(self._trade_history)[-count:]
    
    def get_volume_at_price(self, price: float, side: str) -> int:
        """Get volume at specific price level"""
        with self._lock:
            if side == 'bid' and price in self._bid_levels:
                return self._bid_levels[price].size
            elif side == 'ask' and price in self._ask_levels:
                return self._ask_levels[price].size
            return 0
    
    def calculate_liquidity_density(self, price_range: float = 0.01) -> Dict[str, float]:
        """Calculate liquidity density around current price"""
        mid_price = self.get_mid_price()
        if not mid_price:
            return {'bid_density': 0.0, 'ask_density': 0.0}
        
        with self._lock:
            # Calculate density in price range around mid
            lower_bound = mid_price - price_range / 2
            upper_bound = mid_price + price_range / 2
            
            bid_volume = sum(
                level.size for price, level in self._bid_levels.items()
                if lower_bound <= price <= upper_bound
            )
            
            ask_volume = sum(
                level.size for price, level in self._ask_levels.items()
                if lower_bound <= price <= upper_bound
            )
            
            return {
                'bid_density': bid_volume / price_range if price_range > 0 else 0.0,
                'ask_density': ask_volume / price_range if price_range > 0 else 0.0
            }
    
    def take_snapshot(self) -> Dict:
        """Take a snapshot of current order book state"""
        with self._lock:
            snapshot = {
                'timestamp': int(time.time() * 1000000),
                'symbol': self.symbol,
                'bids': self.get_bids(20),
                'asks': self.get_asks(20),
                'best_bid': self.get_best_bid(),
                'best_ask': self.get_best_ask(),
                'spread': self.get_spread(),
                'mid_price': self.get_mid_price(),
                'imbalance': self.get_imbalance()
            }
            
            self._book_snapshots.append(snapshot)
            return snapshot
    
    def _cleanup_heaps(self) -> None:
        """Clean up heaps by removing invalid entries"""
        # Clean bid heap
        valid_bids = []
        while self._bids:
            price = -heapq.heappop(self._bids)
            if price in self._bid_levels:
                valid_bids.append(-price)
        self._bids = valid_bids
        heapq.heapify(self._bids)
        
        # Clean ask heap
        valid_asks = []
        while self._asks:
            price = heapq.heappop(self._asks)
            if price in self._ask_levels:
                valid_asks.append(price)
        self._asks = valid_asks
        heapq.heapify(self._asks)
    
    def get_statistics(self) -> Dict:
        """Get order book statistics"""
        with self._lock:
            return {
                'symbol': self.symbol,
                'bid_levels': len(self._bid_levels),
                'ask_levels': len(self._ask_levels),
                'total_trades': len(self._trade_history),
                'update_count': self.update_count,
                'last_update': self.last_update_time,
                'spread': self.get_spread(),
                'mid_price': self.get_mid_price(),
                'imbalance': self.get_imbalance()
            }


class OrderBookManager:
    """Manages multiple order books for different instruments"""
    
    def __init__(self):
        self._books: Dict[str, OrderBook] = {}
        self._lock = threading.RLock()
    
    def get_or_create_book(self, symbol: str) -> OrderBook:
        """Get existing order book or create new one"""
        with self._lock:
            if symbol not in self._books:
                self._books[symbol] = OrderBook(symbol)
            return self._books[symbol]
    
    def get_book(self, symbol: str) -> Optional[OrderBook]:
        """Get order book for symbol"""
        with self._lock:
            return self._books.get(symbol)
    
    def get_all_symbols(self) -> List[str]:
        """Get all tracked symbols"""
        with self._lock:
            return list(self._books.keys())
    
    def get_all_books(self) -> Dict[str, OrderBook]:
        """Get all order books"""
        with self._lock:
            return self._books.copy()
    
    def remove_book(self, symbol: str) -> bool:
        """Remove order book for symbol"""
        with self._lock:
            if symbol in self._books:
                del self._books[symbol]
                return True
            return False

