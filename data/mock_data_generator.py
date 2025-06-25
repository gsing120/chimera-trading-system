"""
Mock Level 2 Data Generator
Generates realistic market microstructure data for testing
"""

import time
import random
import math
import threading
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import json

from core.data_orchestrator import DataOrchestrator, SubscriptionConfig


class MarketRegime(Enum):
    """Market regime types"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGE_BOUND = "range_bound"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    NEWS_DRIVEN = "news_driven"


class OrderType(Enum):
    """Order types for simulation"""
    MARKET_BUY = "market_buy"
    MARKET_SELL = "market_sell"
    LIMIT_BUY = "limit_buy"
    LIMIT_SELL = "limit_sell"
    ICEBERG_BUY = "iceberg_buy"
    ICEBERG_SELL = "iceberg_sell"
    SWEEP_BUY = "sweep_buy"
    SWEEP_SELL = "sweep_sell"


@dataclass
class MarketScenario:
    """Defines a market scenario for simulation"""
    name: str
    regime: MarketRegime
    duration_seconds: int
    base_price: float
    volatility: float
    trend_strength: float
    liquidity_density: float
    iceberg_probability: float
    sweep_probability: float
    news_events: List[Dict[str, Any]]


@dataclass
class SimulatedOrder:
    """Represents a simulated order"""
    order_id: str
    order_type: OrderType
    price: float
    size: int
    side: str  # 'bid' or 'ask'
    timestamp: int
    is_iceberg: bool = False
    parent_size: int = 0
    remaining_size: int = 0


class MockDataGenerator:
    """
    Advanced mock Level 2 data generator
    Simulates realistic market microstructure patterns
    """
    
    def __init__(self, symbol: str = "MOCK", base_price: float = 100.0):
        self.symbol = symbol
        self.base_price = base_price
        self.current_price = base_price
        
        # Market state
        self.current_regime = MarketRegime.RANGE_BOUND
        self.volatility = 0.001  # 0.1%
        self.trend_strength = 0.0
        self.liquidity_density = 1.0
        
        # Order book simulation
        self._bid_levels: Dict[float, int] = {}
        self._ask_levels: Dict[float, int] = {}
        self._active_orders: Dict[str, SimulatedOrder] = {}
        self._order_counter = 0
        
        # Iceberg orders
        self._iceberg_orders: Dict[str, Dict] = {}
        
        # Market patterns
        self._absorption_zones: List[Tuple[float, float]] = []  # (price, strength)
        self._sweep_targets: List[float] = []
        
        # Statistics
        self._stats = {
            'orders_generated': 0,
            'trades_generated': 0,
            'icebergs_created': 0,
            'sweeps_executed': 0
        }
        
        # Threading
        self._running = False
        self._generation_thread = None
        self._lock = threading.RLock()
        
        # Callbacks
        self._data_callbacks: List[Callable] = []
        
        # Initialize order book
        self._initialize_order_book()
    
    def _initialize_order_book(self) -> None:
        """Initialize the order book with base liquidity"""
        tick_size = 0.01
        spread = 0.02
        
        # Create bid levels
        for i in range(20):
            price = round(self.current_price - spread/2 - i * tick_size, 2)
            size = random.randint(100, 1000)
            self._bid_levels[price] = size
        
        # Create ask levels
        for i in range(20):
            price = round(self.current_price + spread/2 + i * tick_size, 2)
            size = random.randint(100, 1000)
            self._ask_levels[price] = size
    
    def add_data_callback(self, callback: Callable[[str, str, Dict], None]) -> None:
        """Add callback for generated data"""
        self._data_callbacks.append(callback)
    
    def set_scenario(self, scenario: MarketScenario) -> None:
        """Set market scenario"""
        with self._lock:
            self.current_regime = scenario.regime
            self.base_price = scenario.base_price
            self.volatility = scenario.volatility
            self.trend_strength = scenario.trend_strength
            self.liquidity_density = scenario.liquidity_density
            
            # Set up scenario-specific patterns
            self._setup_scenario_patterns(scenario)
    
    def _setup_scenario_patterns(self, scenario: MarketScenario) -> None:
        """Set up patterns for specific scenario"""
        # Create absorption zones
        self._absorption_zones = []
        if scenario.regime in [MarketRegime.RANGE_BOUND, MarketRegime.LOW_VOLATILITY]:
            # Add strong absorption zones at key levels
            for i in range(3):
                price = scenario.base_price + random.uniform(-0.5, 0.5)
                strength = random.uniform(0.7, 0.9)
                self._absorption_zones.append((price, strength))
        
        # Create sweep targets
        self._sweep_targets = []
        if scenario.sweep_probability > 0.3:
            # Add obvious highs/lows for sweeping
            for i in range(2):
                if random.random() < 0.5:
                    target = scenario.base_price + random.uniform(0.2, 0.8)
                else:
                    target = scenario.base_price - random.uniform(0.2, 0.8)
                self._sweep_targets.append(target)
    
    def start_generation(self, data_orchestrator: DataOrchestrator, 
                        updates_per_second: int = 100) -> None:
        """Start generating mock data"""
        with self._lock:
            if self._running:
                return
            
            self._running = True
            self._data_orchestrator = data_orchestrator
            self._updates_per_second = updates_per_second
            
            # Add subscription
            config = SubscriptionConfig(
                symbol=self.symbol,
                data_types=['level2', 'trades']
            )
            data_orchestrator.add_subscription(config)
            
            # Start generation thread
            self._generation_thread = threading.Thread(
                target=self._generation_loop,
                daemon=True
            )
            self._generation_thread.start()
    
    def stop_generation(self) -> None:
        """Stop generating mock data"""
        with self._lock:
            self._running = False
            
            if self._generation_thread:
                self._generation_thread.join(timeout=5.0)
    
    def _generation_loop(self) -> None:
        """Main data generation loop"""
        update_interval = 1.0 / self._updates_per_second
        
        while self._running:
            start_time = time.time()
            
            try:
                # Generate market updates
                self._generate_market_update()
                
                # Sleep for remaining time
                elapsed = time.time() - start_time
                sleep_time = max(0, update_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except Exception as e:
                print(f"Error in data generation: {e}")
                time.sleep(0.1)
    
    def _generate_market_update(self) -> None:
        """Generate a single market update"""
        with self._lock:
            # Decide what type of update to generate
            update_type = self._choose_update_type()
            
            if update_type == 'price_move':
                self._generate_price_movement()
            elif update_type == 'liquidity_update':
                self._generate_liquidity_update()
            elif update_type == 'trade':
                self._generate_trade()
            elif update_type == 'iceberg':
                self._generate_iceberg_activity()
            elif update_type == 'sweep':
                self._generate_liquidity_sweep()
            elif update_type == 'absorption':
                self._generate_absorption_event()
    
    def _choose_update_type(self) -> str:
        """Choose what type of update to generate"""
        # Probabilities based on current regime
        if self.current_regime == MarketRegime.TRENDING_UP:
            weights = {
                'price_move': 0.3,
                'liquidity_update': 0.2,
                'trade': 0.3,
                'iceberg': 0.1,
                'sweep': 0.08,
                'absorption': 0.02
            }
        elif self.current_regime == MarketRegime.TRENDING_DOWN:
            weights = {
                'price_move': 0.3,
                'liquidity_update': 0.2,
                'trade': 0.3,
                'iceberg': 0.1,
                'sweep': 0.08,
                'absorption': 0.02
            }
        elif self.current_regime == MarketRegime.RANGE_BOUND:
            weights = {
                'price_move': 0.15,
                'liquidity_update': 0.3,
                'trade': 0.25,
                'iceberg': 0.15,
                'sweep': 0.05,
                'absorption': 0.1
            }
        elif self.current_regime == MarketRegime.HIGH_VOLATILITY:
            weights = {
                'price_move': 0.4,
                'liquidity_update': 0.15,
                'trade': 0.35,
                'iceberg': 0.05,
                'sweep': 0.04,
                'absorption': 0.01
            }
        else:  # LOW_VOLATILITY
            weights = {
                'price_move': 0.1,
                'liquidity_update': 0.4,
                'trade': 0.2,
                'iceberg': 0.2,
                'sweep': 0.02,
                'absorption': 0.08
            }
        
        # Choose based on weights
        rand = random.random()
        cumulative = 0.0
        for update_type, weight in weights.items():
            cumulative += weight
            if rand <= cumulative:
                return update_type
        
        return 'liquidity_update'
    
    def _generate_price_movement(self) -> None:
        """Generate price movement based on regime"""
        # Calculate price change
        if self.current_regime == MarketRegime.TRENDING_UP:
            drift = self.trend_strength * 0.0001
            noise = random.gauss(0, self.volatility)
            price_change = drift + noise
        elif self.current_regime == MarketRegime.TRENDING_DOWN:
            drift = -self.trend_strength * 0.0001
            noise = random.gauss(0, self.volatility)
            price_change = drift + noise
        else:
            # Mean reverting for range-bound
            mean_reversion = -0.1 * (self.current_price - self.base_price) / self.base_price
            noise = random.gauss(0, self.volatility)
            price_change = mean_reversion + noise
        
        # Update current price
        self.current_price = max(0.01, self.current_price + price_change)
        
        # Update order book levels accordingly
        self._update_order_book_for_price_change(price_change)
    
    def _update_order_book_for_price_change(self, price_change: float) -> None:
        """Update order book levels for price change"""
        # Shift existing levels
        new_bid_levels = {}
        new_ask_levels = {}
        
        for price, size in self._bid_levels.items():
            new_price = round(price + price_change, 2)
            new_bid_levels[new_price] = size
        
        for price, size in self._ask_levels.items():
            new_price = round(price + price_change, 2)
            new_ask_levels[new_price] = size
        
        self._bid_levels = new_bid_levels
        self._ask_levels = new_ask_levels
        
        # Send updates for changed levels
        self._send_order_book_updates()
    
    def _generate_liquidity_update(self) -> None:
        """Generate order book liquidity updates"""
        # Choose to update bid or ask
        side = 'bid' if random.random() < 0.5 else 'ask'
        levels = self._bid_levels if side == 'bid' else self._ask_levels
        
        if not levels:
            return
        
        # Choose a price level to update
        price = random.choice(list(levels.keys()))
        
        # Decide on update type
        if random.random() < 0.3:
            # Remove level
            new_size = 0
            if price in levels:
                del levels[price]
        else:
            # Update size
            current_size = levels.get(price, 0)
            if random.random() < 0.6:
                # Increase size
                new_size = current_size + random.randint(100, 500)
            else:
                # Decrease size
                new_size = max(0, current_size - random.randint(50, 200))
            
            if new_size > 0:
                levels[price] = new_size
            elif price in levels:
                del levels[price]
        
        # Send update
        self._send_level2_update(price, new_size, random.randint(1, 5), side)
    
    def _generate_trade(self) -> None:
        """Generate a trade execution"""
        # Choose side (buy or sell aggressor)
        side = 'buy' if random.random() < 0.5 else 'sell'
        
        # Get best price
        if side == 'buy':
            # Buying hits the ask
            if not self._ask_levels:
                return
            price = min(self._ask_levels.keys())
            available_size = self._ask_levels[price]
        else:
            # Selling hits the bid
            if not self._bid_levels:
                return
            price = max(self._bid_levels.keys())
            available_size = self._bid_levels[price]
        
        # Determine trade size
        max_size = min(available_size, 1000)
        trade_size = random.randint(1, max_size)
        
        # Execute trade
        self._execute_trade(price, trade_size, side)
        
        # Update order book
        if side == 'buy':
            self._ask_levels[price] -= trade_size
            if self._ask_levels[price] <= 0:
                del self._ask_levels[price]
        else:
            self._bid_levels[price] -= trade_size
            if self._bid_levels[price] <= 0:
                del self._bid_levels[price]
        
        # Send order book update
        remaining_size = self._ask_levels.get(price, 0) if side == 'buy' else self._bid_levels.get(price, 0)
        book_side = 'ask' if side == 'buy' else 'bid'
        self._send_level2_update(price, remaining_size, random.randint(1, 3), book_side)
    
    def _generate_iceberg_activity(self) -> None:
        """Generate iceberg order activity"""
        # Check if we should create new iceberg or update existing
        if random.random() < 0.3 and len(self._iceberg_orders) < 3:
            self._create_iceberg_order()
        else:
            self._update_iceberg_orders()
    
    def _create_iceberg_order(self) -> None:
        """Create a new iceberg order"""
        side = 'bid' if random.random() < 0.5 else 'ask'
        
        # Choose price near best
        if side == 'bid' and self._bid_levels:
            best_bid = max(self._bid_levels.keys())
            price = round(best_bid - random.uniform(0, 0.05), 2)
        elif side == 'ask' and self._ask_levels:
            best_ask = min(self._ask_levels.keys())
            price = round(best_ask + random.uniform(0, 0.05), 2)
        else:
            price = round(self.current_price + random.uniform(-0.1, 0.1), 2)
        
        # Create iceberg
        total_size = random.randint(5000, 20000)
        visible_size = random.randint(100, 500)
        
        iceberg_id = f"iceberg_{self._order_counter}"
        self._order_counter += 1
        
        self._iceberg_orders[iceberg_id] = {
            'price': price,
            'side': side,
            'total_size': total_size,
            'visible_size': visible_size,
            'remaining_size': total_size,
            'last_replenish': time.time()
        }
        
        # Add to order book
        if side == 'bid':
            self._bid_levels[price] = visible_size
        else:
            self._ask_levels[price] = visible_size
        
        # Send update
        self._send_level2_update(price, visible_size, 1, side)
        
        self._stats['icebergs_created'] += 1
    
    def _update_iceberg_orders(self) -> None:
        """Update existing iceberg orders"""
        for iceberg_id, iceberg in list(self._iceberg_orders.items()):
            # Simulate partial fills and replenishment
            if random.random() < 0.4:  # 40% chance of activity
                price = iceberg['price']
                side = iceberg['side']
                visible_size = iceberg['visible_size']
                
                # Simulate partial fill
                fill_size = random.randint(10, min(visible_size, 100))
                iceberg['remaining_size'] -= fill_size
                
                if iceberg['remaining_size'] <= 0:
                    # Iceberg exhausted
                    del self._iceberg_orders[iceberg_id]
                    if side == 'bid':
                        self._bid_levels.pop(price, None)
                    else:
                        self._ask_levels.pop(price, None)
                    self._send_level2_update(price, 0, 0, side)
                else:
                    # Replenish visible size
                    new_visible = min(iceberg['visible_size'], iceberg['remaining_size'])
                    if side == 'bid':
                        self._bid_levels[price] = new_visible
                    else:
                        self._ask_levels[price] = new_visible
                    
                    self._send_level2_update(price, new_visible, 1, side)
                    iceberg['last_replenish'] = time.time()
    
    def _generate_liquidity_sweep(self) -> None:
        """Generate a liquidity sweep event"""
        if not self._sweep_targets or random.random() > 0.1:
            return
        
        target_price = random.choice(self._sweep_targets)
        current_price = self.current_price
        
        # Determine sweep direction
        if target_price > current_price:
            # Sweep up (buy aggression)
            side = 'buy'
            levels_to_sweep = [p for p in self._ask_levels.keys() if p <= target_price]
        else:
            # Sweep down (sell aggression)
            side = 'sell'
            levels_to_sweep = [p for p in self._bid_levels.keys() if p >= target_price]
        
        if not levels_to_sweep:
            return
        
        # Execute sweep
        total_volume = 0
        for price in sorted(levels_to_sweep):
            if side == 'buy':
                size = self._ask_levels.get(price, 0)
                if size > 0:
                    trade_size = random.randint(size // 2, size)
                    self._execute_trade(price, trade_size, side)
                    self._ask_levels[price] -= trade_size
                    if self._ask_levels[price] <= 0:
                        del self._ask_levels[price]
                    total_volume += trade_size
            else:
                size = self._bid_levels.get(price, 0)
                if size > 0:
                    trade_size = random.randint(size // 2, size)
                    self._execute_trade(price, trade_size, side)
                    self._bid_levels[price] -= trade_size
                    if self._bid_levels[price] <= 0:
                        del self._bid_levels[price]
                    total_volume += trade_size
        
        # Update current price
        self.current_price = target_price
        
        # Remove used target
        self._sweep_targets.remove(target_price)
        
        self._stats['sweeps_executed'] += 1
    
    def _generate_absorption_event(self) -> None:
        """Generate an absorption event"""
        if not self._absorption_zones:
            return
        
        zone_price, strength = random.choice(self._absorption_zones)
        
        # Check if price is near absorption zone
        if abs(self.current_price - zone_price) / self.current_price > 0.01:
            return
        
        # Generate aggressive orders that get absorbed
        side = 'buy' if random.random() < 0.5 else 'sell'
        volume = random.randint(1000, 5000)
        
        # Execute multiple trades at the zone without significant price movement
        for _ in range(random.randint(3, 8)):
            trade_size = random.randint(100, 500)
            price_noise = random.uniform(-0.005, 0.005)
            trade_price = zone_price + price_noise
            
            self._execute_trade(trade_price, trade_size, side)
            
            # Small delay between trades
            time.sleep(0.001)
        
        # Replenish liquidity at the zone (showing absorption)
        if side == 'buy':
            # Replenish ask liquidity
            self._ask_levels[zone_price] = self._ask_levels.get(zone_price, 0) + random.randint(500, 1500)
            self._send_level2_update(zone_price, self._ask_levels[zone_price], random.randint(2, 5), 'ask')
        else:
            # Replenish bid liquidity
            self._bid_levels[zone_price] = self._bid_levels.get(zone_price, 0) + random.randint(500, 1500)
            self._send_level2_update(zone_price, self._bid_levels[zone_price], random.randint(2, 5), 'bid')
    
    def _execute_trade(self, price: float, size: int, side: str) -> None:
        """Execute a trade and send trade data"""
        trade_id = f"trade_{self._stats['trades_generated']}"
        
        # Send trade data
        self._send_trade_data(price, size, side, trade_id)
        
        self._stats['trades_generated'] += 1
    
    def _send_level2_update(self, price: float, size: int, orders: int, side: str) -> None:
        """Send Level 2 order book update"""
        if hasattr(self, '_data_orchestrator'):
            self._data_orchestrator.push_level2_update(
                symbol=self.symbol,
                price=price,
                size=size,
                orders=orders,
                side=side
            )
        
        # Call callbacks
        for callback in self._data_callbacks:
            callback(self.symbol, 'level2', {
                'price': price,
                'size': size,
                'orders': orders,
                'side': side
            })
        
        self._stats['orders_generated'] += 1
    
    def _send_trade_data(self, price: float, size: int, side: str, trade_id: str) -> None:
        """Send trade data"""
        if hasattr(self, '_data_orchestrator'):
            self._data_orchestrator.push_trade(
                symbol=self.symbol,
                price=price,
                size=size,
                side=side,
                trade_id=trade_id
            )
        
        # Call callbacks
        for callback in self._data_callbacks:
            callback(self.symbol, 'trade', {
                'price': price,
                'size': size,
                'side': side,
                'trade_id': trade_id
            })
    
    def _send_order_book_updates(self) -> None:
        """Send all order book level updates"""
        # Send bid updates
        for price, size in list(self._bid_levels.items())[:10]:  # Top 10 levels
            self._send_level2_update(price, size, random.randint(1, 3), 'bid')
        
        # Send ask updates
        for price, size in list(self._ask_levels.items())[:10]:  # Top 10 levels
            self._send_level2_update(price, size, random.randint(1, 3), 'ask')
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current generator state"""
        with self._lock:
            return {
                'symbol': self.symbol,
                'current_price': self.current_price,
                'regime': self.current_regime.value,
                'volatility': self.volatility,
                'trend_strength': self.trend_strength,
                'bid_levels': len(self._bid_levels),
                'ask_levels': len(self._ask_levels),
                'active_icebergs': len(self._iceberg_orders),
                'absorption_zones': len(self._absorption_zones),
                'sweep_targets': len(self._sweep_targets),
                'statistics': self._stats.copy()
            }
    
    def create_test_scenario(self, scenario_name: str) -> MarketScenario:
        """Create predefined test scenarios"""
        scenarios = {
            'liquidity_sweep_test': MarketScenario(
                name='Liquidity Sweep Test',
                regime=MarketRegime.TRENDING_UP,
                duration_seconds=300,
                base_price=100.0,
                volatility=0.002,
                trend_strength=0.8,
                liquidity_density=0.7,
                iceberg_probability=0.3,
                sweep_probability=0.8,
                news_events=[]
            ),
            'absorption_test': MarketScenario(
                name='Absorption Test',
                regime=MarketRegime.RANGE_BOUND,
                duration_seconds=300,
                base_price=100.0,
                volatility=0.001,
                trend_strength=0.1,
                liquidity_density=1.2,
                iceberg_probability=0.6,
                sweep_probability=0.2,
                news_events=[]
            ),
            'iceberg_test': MarketScenario(
                name='Iceberg Test',
                regime=MarketRegime.LOW_VOLATILITY,
                duration_seconds=300,
                base_price=100.0,
                volatility=0.0005,
                trend_strength=0.2,
                liquidity_density=1.5,
                iceberg_probability=0.9,
                sweep_probability=0.1,
                news_events=[]
            ),
            'high_volatility_test': MarketScenario(
                name='High Volatility Test',
                regime=MarketRegime.HIGH_VOLATILITY,
                duration_seconds=300,
                base_price=100.0,
                volatility=0.005,
                trend_strength=0.6,
                liquidity_density=0.5,
                iceberg_probability=0.1,
                sweep_probability=0.4,
                news_events=[]
            )
        }
        
        return scenarios.get(scenario_name, scenarios['absorption_test'])


    def get_order_book(self) -> 'OrderBook':
        """Get current order book state"""
        from core.order_book import OrderBook
        
        # Create OrderBook instance
        order_book = OrderBook(self.symbol)
        
        # Initialize order book if empty
        if not self._bid_levels or not self._ask_levels:
            self._initialize_order_book()
        
        # Add bid levels
        for price in sorted(self._bid_levels.keys(), reverse=True)[:10]:
            size = self._bid_levels[price]
            if size > 0:
                order_book.update_level(price, size, 1, 'bid')
        
        # Add ask levels  
        for price in sorted(self._ask_levels.keys())[:10]:
            size = self._ask_levels[price]
            if size > 0:
                order_book.update_level(price, size, 1, 'ask')
        
        return order_book
    
    def generate_order_book(self, symbol: str = None) -> 'OrderBook':
        """Generate a fresh order book for the given symbol"""
        if symbol and symbol != self.symbol:
            # Create a new generator for different symbol
            temp_gen = MockDataGenerator(symbol, self.base_price + random.uniform(-5, 5))
            return temp_gen.get_order_book()
        else:
            return self.get_order_book()

