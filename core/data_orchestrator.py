"""
Data Orchestrator - Multi-Instrument Data Management
Handles real-time data ingestion and distribution
"""

import time
import json
import threading
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Any
from queue import Queue, Empty
import sqlite3
import os

from .order_book import OrderBookManager, OrderBook


@dataclass
class MarketDataUpdate:
    """Represents a market data update"""
    symbol: str
    update_type: str  # 'level2', 'trade', 'quote'
    timestamp: int
    data: Dict[str, Any]


@dataclass
class SubscriptionConfig:
    """Configuration for market data subscription"""
    symbol: str
    data_types: List[str]  # ['level2', 'trades', 'quotes']
    enabled: bool = True


class DataOrchestrator:
    """
    Central data management system for multi-instrument trading
    Handles data ingestion, processing, and distribution
    """
    
    def __init__(self, db_path: str = "market_data.db"):
        self.db_path = db_path
        self.order_book_manager = OrderBookManager()
        
        # Data queues and processing
        self._data_queue = Queue(maxsize=10000)
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        
        # Configuration
        self._subscriptions: Dict[str, SubscriptionConfig] = {}
        self._running = False
        
        # Threading
        self._processor_thread = None
        self._lock = threading.RLock()
        
        # Statistics
        self._stats = {
            'messages_processed': 0,
            'messages_dropped': 0,
            'last_update_time': 0,
            'processing_rate': 0.0
        }
        
        # Initialize database
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize SQLite database for data storage"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    update_type TEXT NOT NULL,
                    data TEXT NOT NULL
                )
            ''')
            
            # Create indexes separately
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp 
                ON market_data(symbol, timestamp)
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    price REAL NOT NULL,
                    size REAL NOT NULL,
                    side TEXT NOT NULL,
                    trade_id TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_trades_symbol_timestamp 
                ON trades(symbol, timestamp)
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS order_book_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    snapshot TEXT NOT NULL
                )
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_snapshots_symbol_timestamp 
                ON order_book_snapshots(symbol, timestamp)
            ''')
            
            conn.commit()
    
    def add_subscription(self, config: SubscriptionConfig) -> None:
        """Add a new market data subscription"""
        with self._lock:
            self._subscriptions[config.symbol] = config
            
            # Create order book if needed
            if 'level2' in config.data_types:
                self.order_book_manager.get_or_create_book(config.symbol)
    
    def remove_subscription(self, symbol: str) -> None:
        """Remove market data subscription"""
        with self._lock:
            if symbol in self._subscriptions:
                del self._subscriptions[symbol]
                self.order_book_manager.remove_book(symbol)
    
    def subscribe_to_updates(self, callback: Callable[[MarketDataUpdate], None], 
                           symbols: List[str] = None) -> None:
        """Subscribe to market data updates"""
        with self._lock:
            if symbols is None:
                symbols = ['*']  # Subscribe to all
            
            for symbol in symbols:
                self._subscribers[symbol].append(callback)
    
    def start(self) -> None:
        """Start the data orchestrator"""
        with self._lock:
            if self._running:
                return
            
            self._running = True
            self._processor_thread = threading.Thread(
                target=self._process_data_loop,
                daemon=True
            )
            self._processor_thread.start()
    
    def stop(self) -> None:
        """Stop the data orchestrator"""
        with self._lock:
            self._running = False
            
            if self._processor_thread:
                self._processor_thread.join(timeout=5.0)
    
    def push_market_data(self, symbol: str, update_type: str, data: Dict[str, Any]) -> bool:
        """Push market data update to processing queue"""
        if not self._running:
            return False
        
        timestamp = int(time.time() * 1000000)
        update = MarketDataUpdate(symbol, update_type, timestamp, data)
        
        try:
            self._data_queue.put_nowait(update)
            return True
        except:
            self._stats['messages_dropped'] += 1
            return False
    
    def push_level2_update(self, symbol: str, price: float, size: int, 
                          orders: int, side: str) -> bool:
        """Push Level 2 order book update"""
        data = {
            'price': price,
            'size': size,
            'orders': orders,
            'side': side
        }
        return self.push_market_data(symbol, 'level2', data)
    
    def push_trade(self, symbol: str, price: float, size: int, side: str,
                   trade_id: str = None) -> bool:
        """Push trade data"""
        data = {
            'price': price,
            'size': size,
            'side': side,
            'trade_id': trade_id
        }
        return self.push_market_data(symbol, 'trade', data)
    
    def _process_data_loop(self) -> None:
        """Main data processing loop"""
        last_stats_update = time.time()
        messages_in_interval = 0
        
        while self._running:
            try:
                # Get update with timeout
                update = self._data_queue.get(timeout=0.1)
                
                # Process the update
                self._process_update(update)
                
                # Update statistics
                self._stats['messages_processed'] += 1
                self._stats['last_update_time'] = update.timestamp
                messages_in_interval += 1
                
                # Calculate processing rate
                current_time = time.time()
                if current_time - last_stats_update >= 1.0:
                    self._stats['processing_rate'] = messages_in_interval / (current_time - last_stats_update)
                    messages_in_interval = 0
                    last_stats_update = current_time
                
            except Empty:
                continue
            except Exception as e:
                print(f"Error processing market data: {e}")
                continue
    
    def _process_update(self, update: MarketDataUpdate) -> None:
        """Process a single market data update"""
        # Check if we have subscription for this symbol
        if update.symbol not in self._subscriptions:
            return
        
        config = self._subscriptions[update.symbol]
        if not config.enabled or update.update_type not in config.data_types:
            return
        
        # Process based on update type
        if update.update_type == 'level2':
            self._process_level2_update(update)
        elif update.update_type == 'trade':
            self._process_trade_update(update)
        
        # Store in database
        self._store_update(update)
        
        # Notify subscribers
        self._notify_subscribers(update)
    
    def _process_level2_update(self, update: MarketDataUpdate) -> None:
        """Process Level 2 order book update"""
        order_book = self.order_book_manager.get_book(update.symbol)
        if not order_book:
            return
        
        data = update.data
        order_book.update_level(
            price=data['price'],
            size=data['size'],
            orders=data['orders'],
            side=data['side']
        )
    
    def _process_trade_update(self, update: MarketDataUpdate) -> None:
        """Process trade update"""
        order_book = self.order_book_manager.get_book(update.symbol)
        if not order_book:
            return
        
        data = update.data
        order_book.add_trade(
            price=data['price'],
            size=data['size'],
            side=data['side'],
            trade_id=data.get('trade_id')
        )
    
    def _store_update(self, update: MarketDataUpdate) -> None:
        """Store update in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if update.update_type == 'trade':
                    # Store in trades table
                    data = update.data
                    cursor.execute('''
                        INSERT INTO trades (symbol, timestamp, price, size, side, trade_id)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        update.symbol,
                        update.timestamp,
                        data['price'],
                        data['size'],
                        data['side'],
                        data.get('trade_id')
                    ))
                
                # Store in general market_data table
                cursor.execute('''
                    INSERT INTO market_data (symbol, timestamp, update_type, data)
                    VALUES (?, ?, ?, ?)
                ''', (
                    update.symbol,
                    update.timestamp,
                    update.update_type,
                    json.dumps(update.data)
                ))
                
                conn.commit()
        except Exception as e:
            print(f"Error storing market data: {e}")
    
    def _notify_subscribers(self, update: MarketDataUpdate) -> None:
        """Notify all subscribers of the update"""
        # Notify symbol-specific subscribers
        for callback in self._subscribers.get(update.symbol, []):
            try:
                callback(update)
            except Exception as e:
                print(f"Error in subscriber callback: {e}")
        
        # Notify global subscribers
        for callback in self._subscribers.get('*', []):
            try:
                callback(update)
            except Exception as e:
                print(f"Error in global subscriber callback: {e}")
    
    def get_order_book(self, symbol: str) -> Optional[OrderBook]:
        """Get order book for symbol"""
        return self.order_book_manager.get_book(symbol)
    
    def get_all_symbols(self) -> List[str]:
        """Get all subscribed symbols"""
        with self._lock:
            return list(self._subscriptions.keys())
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get orchestrator statistics"""
        with self._lock:
            stats = self._stats.copy()
            stats['active_subscriptions'] = len(self._subscriptions)
            stats['queue_size'] = self._data_queue.qsize()
            stats['running'] = self._running
            return stats
    
    def get_historical_data(self, symbol: str, start_time: int = None, 
                           end_time: int = None, limit: int = 1000) -> List[Dict]:
        """Get historical market data"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            query = '''
                SELECT timestamp, update_type, data
                FROM market_data
                WHERE symbol = ?
            '''
            params = [symbol]
            
            if start_time:
                query += ' AND timestamp >= ?'
                params.append(start_time)
            
            if end_time:
                query += ' AND timestamp <= ?'
                params.append(end_time)
            
            query += ' ORDER BY timestamp DESC LIMIT ?'
            params.append(limit)
            
            cursor.execute(query, params)
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'timestamp': row[0],
                    'update_type': row[1],
                    'data': json.loads(row[2])
                })
            
            return results
    
    def get_historical_trades(self, symbol: str, start_time: int = None,
                             end_time: int = None, limit: int = 1000) -> List[Dict]:
        """Get historical trade data"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            query = '''
                SELECT timestamp, price, size, side, trade_id
                FROM trades
                WHERE symbol = ?
            '''
            params = [symbol]
            
            if start_time:
                query += ' AND timestamp >= ?'
                params.append(start_time)
            
            if end_time:
                query += ' AND timestamp <= ?'
                params.append(end_time)
            
            query += ' ORDER BY timestamp DESC LIMIT ?'
            params.append(limit)
            
            cursor.execute(query, params)
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'timestamp': row[0],
                    'price': row[1],
                    'size': row[2],
                    'side': row[3],
                    'trade_id': row[4]
                })
            
            return results
    
    def save_order_book_snapshot(self, symbol: str) -> bool:
        """Save current order book snapshot to database"""
        order_book = self.get_order_book(symbol)
        if not order_book:
            return False
        
        try:
            snapshot = order_book.take_snapshot()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO order_book_snapshots (symbol, timestamp, snapshot)
                    VALUES (?, ?, ?)
                ''', (
                    symbol,
                    snapshot['timestamp'],
                    json.dumps(snapshot, default=str)
                ))
                conn.commit()
            
            return True
        except Exception as e:
            print(f"Error saving order book snapshot: {e}")
            return False
    
    def cleanup_old_data(self, days_to_keep: int = 7) -> None:
        """Clean up old data from database"""
        cutoff_time = int((time.time() - days_to_keep * 24 * 3600) * 1000000)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('DELETE FROM market_data WHERE timestamp < ?', (cutoff_time,))
                cursor.execute('DELETE FROM trades WHERE timestamp < ?', (cutoff_time,))
                cursor.execute('DELETE FROM order_book_snapshots WHERE timestamp < ?', (cutoff_time,))
                
                conn.commit()
                
                # Vacuum to reclaim space
                cursor.execute('VACUUM')
                
        except Exception as e:
            print(f"Error cleaning up old data: {e}")

