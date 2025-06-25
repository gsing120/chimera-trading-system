"""
Feature Engineering Engine
Computes market microstructure features from order book and trade data
"""

import time
import math
import statistics
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import threading

from .order_book import OrderBook, Trade
from .data_orchestrator import MarketDataUpdate


@dataclass
class MarketFeatures:
    """Container for all market features"""
    symbol: str
    timestamp: int
    
    # Price Features
    mid_price: float
    spread: float
    spread_bps: float
    
    # Liquidity Features
    bid_liquidity_density: float
    ask_liquidity_density: float
    total_bid_depth: float
    total_ask_depth: float
    depth_imbalance: float
    
    # Flow Features
    aggressive_buy_volume: float
    aggressive_sell_volume: float
    net_flow: float
    flow_imbalance: float
    
    # Microstructure Features
    tick_direction: int
    price_impact: float
    effective_spread: float
    realized_spread: float
    
    # Technical Features
    vwap: float
    vwap_deviation: float
    volume_profile_poc: float  # Point of Control
    hvn_distance: float        # High Volume Node distance
    lvn_distance: float        # Low Volume Node distance
    
    # Order Book Features
    order_book_imbalance: float
    weighted_mid_price: float
    microprice: float
    
    # Volatility Features
    realized_volatility: float
    price_acceleration: float
    
    # Advanced Features
    absorption_strength: float
    iceberg_probability: float
    sweep_probability: float


class VolumeProfile:
    """Volume profile calculator"""
    
    def __init__(self, price_bucket_size: float = 0.01):
        self.price_bucket_size = price_bucket_size
        self.volume_by_price: Dict[float, float] = defaultdict(float)
        self.total_volume = 0.0
    
    def add_trade(self, price: float, volume: float) -> None:
        """Add trade to volume profile"""
        bucket = round(price / self.price_bucket_size) * self.price_bucket_size
        self.volume_by_price[bucket] += volume
        self.total_volume += volume
    
    def get_poc(self) -> float:
        """Get Point of Control (price with highest volume)"""
        if not self.volume_by_price:
            return 0.0
        return max(self.volume_by_price.items(), key=lambda x: x[1])[0]
    
    def get_value_area(self, percentage: float = 0.7) -> Tuple[float, float]:
        """Get value area (price range containing X% of volume)"""
        if not self.volume_by_price:
            return 0.0, 0.0
        
        target_volume = self.total_volume * percentage
        sorted_prices = sorted(self.volume_by_price.items(), key=lambda x: x[1], reverse=True)
        
        accumulated_volume = 0.0
        prices_in_area = []
        
        for price, volume in sorted_prices:
            accumulated_volume += volume
            prices_in_area.append(price)
            if accumulated_volume >= target_volume:
                break
        
        return min(prices_in_area), max(prices_in_area)
    
    def is_hvn(self, price: float, threshold: float = 1.5) -> bool:
        """Check if price is a High Volume Node"""
        bucket = round(price / self.price_bucket_size) * self.price_bucket_size
        if bucket not in self.volume_by_price:
            return False
        
        avg_volume = self.total_volume / len(self.volume_by_price) if self.volume_by_price else 0
        return self.volume_by_price[bucket] >= avg_volume * threshold
    
    def is_lvn(self, price: float, threshold: float = 0.5) -> bool:
        """Check if price is a Low Volume Node"""
        bucket = round(price / self.price_bucket_size) * self.price_bucket_size
        if bucket not in self.volume_by_price:
            return True
        
        avg_volume = self.total_volume / len(self.volume_by_price) if self.volume_by_price else 0
        return self.volume_by_price[bucket] <= avg_volume * threshold


class FeatureEngine:
    """
    Advanced feature engineering for market microstructure analysis
    """
    
    def __init__(self, lookback_periods: Dict[str, int] = None):
        if lookback_periods is None:
            lookback_periods = {
                'short': 100,    # 100 ticks
                'medium': 500,   # 500 ticks
                'long': 1000     # 1000 ticks
            }
        
        self.lookback_periods = lookback_periods
        
        # Feature history storage
        self._feature_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.lookback_periods['long'])
        )
        
        # Trade flow tracking
        self._trade_flow: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.lookback_periods['medium'])
        )
        
        # Price history
        self._price_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.lookback_periods['long'])
        )
        
        # Volume profiles
        self._volume_profiles: Dict[str, VolumeProfile] = {}
        
        # VWAP calculation
        self._vwap_data: Dict[str, Dict] = defaultdict(lambda: {
            'price_volume_sum': 0.0,
            'volume_sum': 0.0,
            'trades': deque(maxlen=self.lookback_periods['medium'])
        })
        
        # Thread safety
        self._lock = threading.RLock()
    
    def update_features(self, order_book: OrderBook) -> MarketFeatures:
        """Calculate all features for given order book"""
        with self._lock:
            symbol = order_book.symbol
            timestamp = int(time.time() * 1000000)
            
            # Get basic order book data
            best_bid = order_book.get_best_bid()
            best_ask = order_book.get_best_ask()
            
            if not best_bid or not best_ask:
                return self._create_empty_features(symbol, timestamp)
            
            # Calculate features
            features = MarketFeatures(
                symbol=symbol,
                timestamp=timestamp,
                
                # Price features
                mid_price=order_book.get_mid_price() or 0.0,
                spread=order_book.get_spread() or 0.0,
                spread_bps=self._calculate_spread_bps(order_book),
                
                # Liquidity features
                bid_liquidity_density=self._calculate_liquidity_density(order_book, 'bid'),
                ask_liquidity_density=self._calculate_liquidity_density(order_book, 'ask'),
                total_bid_depth=order_book.get_depth('bid', 10),
                total_ask_depth=order_book.get_depth('ask', 10),
                depth_imbalance=order_book.get_imbalance(5),
                
                # Flow features
                aggressive_buy_volume=self._calculate_aggressive_volume(order_book, 'buy'),
                aggressive_sell_volume=self._calculate_aggressive_volume(order_book, 'sell'),
                net_flow=self._calculate_net_flow(order_book),
                flow_imbalance=self._calculate_flow_imbalance(order_book),
                
                # Microstructure features
                tick_direction=self._calculate_tick_direction(order_book),
                price_impact=self._calculate_price_impact(order_book),
                effective_spread=self._calculate_effective_spread(order_book),
                realized_spread=self._calculate_realized_spread(order_book),
                
                # Technical features
                vwap=self._calculate_vwap(order_book),
                vwap_deviation=self._calculate_vwap_deviation(order_book),
                volume_profile_poc=self._calculate_volume_profile_poc(symbol),
                hvn_distance=self._calculate_hvn_distance(order_book),
                lvn_distance=self._calculate_lvn_distance(order_book),
                
                # Order book features
                order_book_imbalance=order_book.get_imbalance(3),
                weighted_mid_price=self._calculate_weighted_mid_price(order_book),
                microprice=self._calculate_microprice(order_book),
                
                # Volatility features
                realized_volatility=self._calculate_realized_volatility(order_book),
                price_acceleration=self._calculate_price_acceleration(order_book),
                
                # Advanced features
                absorption_strength=self._calculate_absorption_strength(order_book),
                iceberg_probability=self._calculate_iceberg_probability(order_book),
                sweep_probability=self._calculate_sweep_probability(order_book)
            )
            
            # Store features in history
            self._feature_history[symbol].append(features)
            
            # Update price history
            if features.mid_price > 0:
                self._price_history[symbol].append((timestamp, features.mid_price))
            
            return features
    
    def _create_empty_features(self, symbol: str, timestamp: int) -> MarketFeatures:
        """Create empty features when order book is invalid"""
        return MarketFeatures(
            symbol=symbol,
            timestamp=timestamp,
            mid_price=0.0, spread=0.0, spread_bps=0.0,
            bid_liquidity_density=0.0, ask_liquidity_density=0.0,
            total_bid_depth=0.0, total_ask_depth=0.0, depth_imbalance=0.0,
            aggressive_buy_volume=0.0, aggressive_sell_volume=0.0,
            net_flow=0.0, flow_imbalance=0.0,
            tick_direction=0, price_impact=0.0,
            effective_spread=0.0, realized_spread=0.0,
            vwap=0.0, vwap_deviation=0.0, volume_profile_poc=0.0,
            hvn_distance=0.0, lvn_distance=0.0,
            order_book_imbalance=0.0, weighted_mid_price=0.0, microprice=0.0,
            realized_volatility=0.0, price_acceleration=0.0,
            absorption_strength=0.0, iceberg_probability=0.0, sweep_probability=0.0
        )
    
    def _calculate_spread_bps(self, order_book: OrderBook) -> float:
        """Calculate spread in basis points"""
        spread = order_book.get_spread()
        mid_price = order_book.get_mid_price()
        
        if not spread or not mid_price or mid_price == 0:
            return 0.0
        
        return (spread / mid_price) * 10000  # Convert to basis points
    
    def _calculate_liquidity_density(self, order_book: OrderBook, side: str) -> float:
        """Calculate liquidity density for given side"""
        density_data = order_book.calculate_liquidity_density(0.01)
        return density_data.get(f'{side}_density', 0.0)
    
    def _calculate_aggressive_volume(self, order_book: OrderBook, side: str) -> float:
        """Calculate aggressive volume for given side"""
        recent_trades = order_book.get_recent_trades(50)
        
        volume = 0.0
        for trade in recent_trades:
            if trade.side == side:
                volume += trade.size
        
        return volume
    
    def _calculate_net_flow(self, order_book: OrderBook) -> float:
        """Calculate net order flow"""
        buy_volume = self._calculate_aggressive_volume(order_book, 'buy')
        sell_volume = self._calculate_aggressive_volume(order_book, 'sell')
        return buy_volume - sell_volume
    
    def _calculate_flow_imbalance(self, order_book: OrderBook) -> float:
        """Calculate flow imbalance ratio"""
        buy_volume = self._calculate_aggressive_volume(order_book, 'buy')
        sell_volume = self._calculate_aggressive_volume(order_book, 'sell')
        
        total_volume = buy_volume + sell_volume
        if total_volume == 0:
            return 0.0
        
        return (buy_volume - sell_volume) / total_volume
    
    def _calculate_tick_direction(self, order_book: OrderBook) -> int:
        """Calculate tick direction (-1, 0, 1)"""
        symbol = order_book.symbol
        price_history = list(self._price_history[symbol])
        
        if len(price_history) < 2:
            return 0
        
        current_price = price_history[-1][1]
        previous_price = price_history[-2][1]
        
        if current_price > previous_price:
            return 1
        elif current_price < previous_price:
            return -1
        else:
            return 0
    
    def _calculate_price_impact(self, order_book: OrderBook) -> float:
        """Calculate price impact of recent trades"""
        recent_trades = order_book.get_recent_trades(10)
        if len(recent_trades) < 2:
            return 0.0
        
        # Calculate average price impact
        impacts = []
        for i in range(1, len(recent_trades)):
            price_change = abs(recent_trades[i].price - recent_trades[i-1].price)
            impacts.append(price_change)
        
        return statistics.mean(impacts) if impacts else 0.0
    
    def _calculate_effective_spread(self, order_book: OrderBook) -> float:
        """Calculate effective spread"""
        recent_trades = order_book.get_recent_trades(10)
        mid_price = order_book.get_mid_price()
        
        if not recent_trades or not mid_price:
            return 0.0
        
        effective_spreads = []
        for trade in recent_trades:
            effective_spread = 2 * abs(trade.price - mid_price)
            effective_spreads.append(effective_spread)
        
        return statistics.mean(effective_spreads) if effective_spreads else 0.0
    
    def _calculate_realized_spread(self, order_book: OrderBook) -> float:
        """Calculate realized spread"""
        # Simplified implementation
        return self._calculate_effective_spread(order_book) * 0.5
    
    def _calculate_vwap(self, order_book: OrderBook) -> float:
        """Calculate Volume Weighted Average Price"""
        symbol = order_book.symbol
        vwap_data = self._vwap_data[symbol]
        
        # Update VWAP with recent trades
        recent_trades = order_book.get_recent_trades(10)
        for trade in recent_trades:
            # Check if trade is already processed
            trade_key = f"{trade.timestamp}_{trade.trade_id}"
            if trade_key not in [t.get('key') for t in vwap_data['trades']]:
                vwap_data['price_volume_sum'] += trade.price * trade.size
                vwap_data['volume_sum'] += trade.size
                vwap_data['trades'].append({
                    'key': trade_key,
                    'price': trade.price,
                    'size': trade.size
                })
        
        if vwap_data['volume_sum'] == 0:
            return order_book.get_mid_price() or 0.0
        
        return vwap_data['price_volume_sum'] / vwap_data['volume_sum']
    
    def _calculate_vwap_deviation(self, order_book: OrderBook) -> float:
        """Calculate deviation from VWAP"""
        vwap = self._calculate_vwap(order_book)
        mid_price = order_book.get_mid_price()
        
        if not vwap or not mid_price or vwap == 0:
            return 0.0
        
        return (mid_price - vwap) / vwap
    
    def _calculate_volume_profile_poc(self, symbol: str) -> float:
        """Calculate Point of Control from volume profile"""
        if symbol not in self._volume_profiles:
            self._volume_profiles[symbol] = VolumeProfile()
        
        return self._volume_profiles[symbol].get_poc()
    
    def _calculate_hvn_distance(self, order_book: OrderBook) -> float:
        """Calculate distance to nearest High Volume Node"""
        symbol = order_book.symbol
        mid_price = order_book.get_mid_price()
        
        if not mid_price or symbol not in self._volume_profiles:
            return 0.0
        
        volume_profile = self._volume_profiles[symbol]
        
        # Find nearest HVN
        min_distance = float('inf')
        for price in volume_profile.volume_by_price.keys():
            if volume_profile.is_hvn(price):
                distance = abs(price - mid_price)
                min_distance = min(min_distance, distance)
        
        return min_distance if min_distance != float('inf') else 0.0
    
    def _calculate_lvn_distance(self, order_book: OrderBook) -> float:
        """Calculate distance to nearest Low Volume Node"""
        symbol = order_book.symbol
        mid_price = order_book.get_mid_price()
        
        if not mid_price or symbol not in self._volume_profiles:
            return 0.0
        
        volume_profile = self._volume_profiles[symbol]
        
        # Find nearest LVN
        min_distance = float('inf')
        for price in volume_profile.volume_by_price.keys():
            if volume_profile.is_lvn(price):
                distance = abs(price - mid_price)
                min_distance = min(min_distance, distance)
        
        return min_distance if min_distance != float('inf') else 0.0
    
    def _calculate_weighted_mid_price(self, order_book: OrderBook) -> float:
        """Calculate size-weighted mid price"""
        best_bid = order_book.get_best_bid()
        best_ask = order_book.get_best_ask()
        
        if not best_bid or not best_ask:
            return 0.0
        
        total_size = best_bid.size + best_ask.size
        if total_size == 0:
            return (best_bid.price + best_ask.price) / 2
        
        return (best_bid.price * best_ask.size + best_ask.price * best_bid.size) / total_size
    
    def _calculate_microprice(self, order_book: OrderBook) -> float:
        """Calculate microprice (probability-weighted mid price)"""
        best_bid = order_book.get_best_bid()
        best_ask = order_book.get_best_ask()
        
        if not best_bid or not best_ask:
            return 0.0
        
        total_size = best_bid.size + best_ask.size
        if total_size == 0:
            return (best_bid.price + best_ask.price) / 2
        
        # Simple microprice calculation
        bid_prob = best_bid.size / total_size
        ask_prob = best_ask.size / total_size
        
        return best_bid.price * ask_prob + best_ask.price * bid_prob
    
    def _calculate_realized_volatility(self, order_book: OrderBook) -> float:
        """Calculate realized volatility"""
        symbol = order_book.symbol
        price_history = list(self._price_history[symbol])
        
        if len(price_history) < 10:
            return 0.0
        
        # Calculate log returns
        returns = []
        for i in range(1, len(price_history)):
            if price_history[i-1][1] > 0 and price_history[i][1] > 0:
                ret = math.log(price_history[i][1] / price_history[i-1][1])
                returns.append(ret)
        
        if len(returns) < 2:
            return 0.0
        
        # Calculate standard deviation of returns
        return statistics.stdev(returns) if len(returns) > 1 else 0.0
    
    def _calculate_price_acceleration(self, order_book: OrderBook) -> float:
        """Calculate price acceleration (second derivative)"""
        symbol = order_book.symbol
        price_history = list(self._price_history[symbol])
        
        if len(price_history) < 3:
            return 0.0
        
        # Calculate second derivative
        p1 = price_history[-3][1]
        p2 = price_history[-2][1]
        p3 = price_history[-1][1]
        
        if p1 > 0 and p2 > 0 and p3 > 0:
            return (p3 - 2*p2 + p1)
        
        return 0.0
    
    def _calculate_absorption_strength(self, order_book: OrderBook) -> float:
        """Calculate absorption strength at current levels"""
        # Simplified absorption calculation
        recent_trades = order_book.get_recent_trades(20)
        mid_price = order_book.get_mid_price()
        
        if not recent_trades or not mid_price:
            return 0.0
        
        # Count trades near current price that didn't move price significantly
        absorption_trades = 0
        for trade in recent_trades:
            if abs(trade.price - mid_price) / mid_price < 0.001:  # Within 0.1%
                absorption_trades += 1
        
        return absorption_trades / len(recent_trades) if recent_trades else 0.0
    
    def _calculate_iceberg_probability(self, order_book: OrderBook) -> float:
        """Calculate probability of iceberg orders"""
        # Look for consistent replenishment patterns
        best_bid = order_book.get_best_bid()
        best_ask = order_book.get_best_ask()
        
        if not best_bid or not best_ask:
            return 0.0
        
        # Simplified iceberg detection
        # Check if there's consistent size at best levels
        bid_consistency = 1.0 if best_bid.size > 100 else 0.5
        ask_consistency = 1.0 if best_ask.size > 100 else 0.5
        
        return (bid_consistency + ask_consistency) / 2
    
    def _calculate_sweep_probability(self, order_book: OrderBook) -> float:
        """Calculate probability of liquidity sweep"""
        # Look for patterns indicating potential sweep
        imbalance = order_book.get_imbalance(5)
        recent_trades = order_book.get_recent_trades(10)
        
        if not recent_trades:
            return 0.0
        
        # Check for increasing aggression
        recent_volume = sum(trade.size for trade in recent_trades[-5:])
        earlier_volume = sum(trade.size for trade in recent_trades[:5]) if len(recent_trades) >= 10 else 0
        
        volume_acceleration = recent_volume / earlier_volume if earlier_volume > 0 else 1.0
        
        # Combine imbalance and volume acceleration
        sweep_score = abs(imbalance) * min(volume_acceleration, 3.0) / 3.0
        
        return min(sweep_score, 1.0)
    
    def update_with_trade(self, symbol: str, trade: Trade) -> None:
        """Update volume profile with new trade"""
        with self._lock:
            if symbol not in self._volume_profiles:
                self._volume_profiles[symbol] = VolumeProfile()
            
            self._volume_profiles[symbol].add_trade(trade.price, trade.size)
    
    def get_feature_history(self, symbol: str, count: int = 100) -> List[MarketFeatures]:
        """Get historical features for symbol"""
        with self._lock:
            history = list(self._feature_history[symbol])
            return history[-count:] if count else history
    
    def get_statistics(self, symbol: str) -> Dict[str, Any]:
        """Get feature engine statistics"""
        with self._lock:
            return {
                'symbol': symbol,
                'feature_history_length': len(self._feature_history[symbol]),
                'price_history_length': len(self._price_history[symbol]),
                'vwap_trades_count': len(self._vwap_data[symbol]['trades']),
                'volume_profile_buckets': len(self._volume_profiles.get(symbol, VolumeProfile()).volume_by_price)
            }

