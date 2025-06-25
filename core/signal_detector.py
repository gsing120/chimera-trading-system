"""
Signal Detection System
Implements Bookmap-based trading strategies and signal generation
"""

import time
import math
import statistics
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import threading
from enum import Enum

from .order_book import OrderBook, Trade
from .feature_engine import MarketFeatures, FeatureEngine


class SignalType(Enum):
    """Types of trading signals"""
    LIQUIDITY_SWEEP_REVERSAL = "liquidity_sweep_reversal"
    STACKED_ABSORPTION_REVERSAL = "stacked_absorption_reversal"
    ICEBERG_DEFENSE_ENTRY = "iceberg_defense_entry"
    VACUUM_ENTRY = "vacuum_entry"
    MEAN_REVERSION_FADE = "mean_reversion_fade"
    MOMENTUM_CONTINUATION = "momentum_continuation"
    ORDER_BOOK_IMBALANCE = "order_book_imbalance"


class SignalStrength(Enum):
    """Signal strength levels"""
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4


@dataclass
class TradingSignal:
    """Represents a trading signal"""
    symbol: str
    signal_type: SignalType
    direction: str  # 'long' or 'short'
    strength: SignalStrength
    confidence: float  # 0.0 to 1.0
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    timestamp: int
    features: MarketFeatures
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary"""
        return {
            'symbol': self.symbol,
            'signal_type': self.signal_type.value,
            'direction': self.direction,
            'strength': self.strength.value,
            'confidence': self.confidence,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'position_size': self.position_size,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }


@dataclass
class AbsorptionEvent:
    """Represents an absorption event"""
    price: float
    volume_absorbed: float
    timestamp: int
    side: str  # 'bid' or 'ask'
    strength: float


@dataclass
class LiquiditySweep:
    """Represents a liquidity sweep event"""
    sweep_price: float
    target_price: float
    volume: float
    timestamp: int
    direction: str  # 'up' or 'down'
    trapped_volume: float


class SignalDetector:
    """
    Advanced signal detection system implementing Bookmap strategies
    """
    
    def __init__(self, feature_engine: FeatureEngine):
        self.feature_engine = feature_engine
        
        # Signal history
        self._signal_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Pattern detection state
        self._absorption_events: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self._sweep_events: Dict[str, deque] = defaultdict(lambda: deque(maxlen=20))
        self._iceberg_levels: Dict[str, Dict[float, Dict]] = defaultdict(dict)
        
        # Configuration parameters
        self.config = {
            'min_absorption_volume': 1000,
            'absorption_price_tolerance': 0.001,  # 0.1%
            'sweep_volume_threshold': 2000,
            'iceberg_replenishment_threshold': 0.8,
            'imbalance_threshold': 0.3,
            'confluence_weight': 1.5,
            'min_signal_confidence': 0.6
        }
        
        # Thread safety
        self._lock = threading.RLock()
    
    def detect_signals(self, order_book: OrderBook, features: MarketFeatures) -> List[TradingSignal]:
        """Main signal detection method"""
        with self._lock:
            signals = []
            
            # Update internal state
            self._update_absorption_events(order_book, features)
            self._update_sweep_events(order_book, features)
            self._update_iceberg_levels(order_book, features)
            
            # Detect different signal types
            signals.extend(self._detect_liquidity_sweep_reversal(order_book, features))
            signals.extend(self._detect_stacked_absorption_reversal(order_book, features))
            signals.extend(self._detect_iceberg_defense_entry(order_book, features))
            signals.extend(self._detect_vacuum_entry(order_book, features))
            signals.extend(self._detect_mean_reversion_fade(order_book, features))
            signals.extend(self._detect_momentum_continuation(order_book, features))
            signals.extend(self._detect_order_book_imbalance(order_book, features))
            
            # Filter signals by confidence
            filtered_signals = [
                signal for signal in signals 
                if signal.confidence >= self.config['min_signal_confidence']
            ]
            
            # Store signals in history
            for signal in filtered_signals:
                self._signal_history[signal.symbol].append(signal)
            
            return filtered_signals
    
    def _detect_liquidity_sweep_reversal(self, order_book: OrderBook, features: MarketFeatures) -> List[TradingSignal]:
        """Detect liquidity sweep reversal patterns"""
        signals = []
        symbol = order_book.symbol
        
        recent_sweeps = list(self._sweep_events[symbol])[-3:]  # Last 3 sweeps
        if not recent_sweeps:
            return signals
        
        latest_sweep = recent_sweeps[-1]
        current_time = int(time.time() * 1000000)
        
        # Check if sweep is recent (within last 30 seconds)
        if current_time - latest_sweep.timestamp > 30_000_000:
            return signals
        
        # Look for absorption after sweep
        recent_absorptions = [
            event for event in self._absorption_events[symbol]
            if event.timestamp > latest_sweep.timestamp
        ]
        
        if not recent_absorptions:
            return signals
        
        # Check for reversal conditions
        mid_price = features.mid_price
        sweep_direction = latest_sweep.direction
        
        # Determine signal direction (opposite to sweep)
        signal_direction = 'short' if sweep_direction == 'up' else 'long'
        
        # Calculate confidence based on absorption strength and volume
        absorption_strength = sum(event.strength for event in recent_absorptions) / len(recent_absorptions)
        volume_ratio = latest_sweep.trapped_volume / latest_sweep.volume if latest_sweep.volume > 0 else 0
        
        confidence = min(0.9, absorption_strength * 0.4 + volume_ratio * 0.3 + 0.3)
        
        if confidence >= self.config['min_signal_confidence']:
            # Calculate entry, stop, and target levels
            if signal_direction == 'long':
                entry_price = mid_price
                stop_loss = latest_sweep.sweep_price * 0.999  # Below sweep low
                take_profit = latest_sweep.target_price * 1.002  # Above target
            else:
                entry_price = mid_price
                stop_loss = latest_sweep.sweep_price * 1.001  # Above sweep high
                take_profit = latest_sweep.target_price * 0.998  # Below target
            
            signal = TradingSignal(
                symbol=symbol,
                signal_type=SignalType.LIQUIDITY_SWEEP_REVERSAL,
                direction=signal_direction,
                strength=SignalStrength.STRONG,
                confidence=confidence,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=self._calculate_position_size(entry_price, stop_loss),
                timestamp=current_time,
                features=features,
                metadata={
                    'sweep_price': latest_sweep.sweep_price,
                    'sweep_volume': latest_sweep.volume,
                    'trapped_volume': latest_sweep.trapped_volume,
                    'absorption_events': len(recent_absorptions)
                }
            )
            signals.append(signal)
        
        return signals
    
    def _detect_stacked_absorption_reversal(self, order_book: OrderBook, features: MarketFeatures) -> List[TradingSignal]:
        """Detect stacked absorption reversal patterns"""
        signals = []
        symbol = order_book.symbol
        
        # Look for multiple absorption events at similar price levels
        recent_absorptions = list(self._absorption_events[symbol])[-10:]
        if len(recent_absorptions) < 2:
            return signals
        
        # Group absorptions by price level
        price_groups = defaultdict(list)
        for event in recent_absorptions:
            price_key = round(event.price / (features.mid_price * 0.001)) * (features.mid_price * 0.001)
            price_groups[price_key].append(event)
        
        # Find groups with multiple absorptions
        for price_level, events in price_groups.items():
            if len(events) >= 2:
                # Check if absorptions are getting stronger (stacked)
                events.sort(key=lambda x: x.timestamp)
                
                if len(events) >= 2 and events[-1].strength > events[-2].strength:
                    # Determine direction based on absorption side
                    absorption_side = events[-1].side
                    signal_direction = 'long' if absorption_side == 'ask' else 'short'
                    
                    # Calculate confidence
                    total_volume = sum(event.volume_absorbed for event in events)
                    avg_strength = sum(event.strength for event in events) / len(events)
                    
                    confidence = min(0.9, avg_strength * 0.5 + min(total_volume / 5000, 1.0) * 0.3 + 0.2)
                    
                    if confidence >= self.config['min_signal_confidence']:
                        entry_price = features.mid_price
                        
                        if signal_direction == 'long':
                            stop_loss = price_level * 0.998
                            take_profit = price_level * 1.005
                        else:
                            stop_loss = price_level * 1.002
                            take_profit = price_level * 0.995
                        
                        signal = TradingSignal(
                            symbol=symbol,
                            signal_type=SignalType.STACKED_ABSORPTION_REVERSAL,
                            direction=signal_direction,
                            strength=SignalStrength.MODERATE,
                            confidence=confidence,
                            entry_price=entry_price,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            position_size=self._calculate_position_size(entry_price, stop_loss),
                            timestamp=int(time.time() * 1000000),
                            features=features,
                            metadata={
                                'absorption_level': price_level,
                                'absorption_count': len(events),
                                'total_volume_absorbed': total_volume
                            }
                        )
                        signals.append(signal)
        
        return signals
    
    def _detect_iceberg_defense_entry(self, order_book: OrderBook, features: MarketFeatures) -> List[TradingSignal]:
        """Detect iceberg defense entry opportunities"""
        signals = []
        symbol = order_book.symbol
        
        # Check for known iceberg levels
        iceberg_levels = self._iceberg_levels[symbol]
        if not iceberg_levels:
            return signals
        
        mid_price = features.mid_price
        current_time = int(time.time() * 1000000)
        
        for price_level, iceberg_data in iceberg_levels.items():
            # Check if price is testing the iceberg level
            distance_to_level = abs(mid_price - price_level) / mid_price
            
            if distance_to_level <= 0.002:  # Within 0.2%
                # Check if iceberg is still active
                last_replenishment = iceberg_data.get('last_replenishment', 0)
                if current_time - last_replenishment < 60_000_000:  # Within 60 seconds
                    
                    side = iceberg_data.get('side', 'unknown')
                    signal_direction = 'long' if side == 'bid' else 'short'
                    
                    # Calculate confidence based on iceberg strength
                    replenishment_count = iceberg_data.get('replenishment_count', 0)
                    avg_size = iceberg_data.get('avg_size', 0)
                    
                    confidence = min(0.85, 
                                   min(replenishment_count / 10, 1.0) * 0.4 + 
                                   min(avg_size / 1000, 1.0) * 0.3 + 0.3)
                    
                    if confidence >= self.config['min_signal_confidence']:
                        entry_price = mid_price
                        
                        if signal_direction == 'long':
                            stop_loss = price_level * 0.997
                            take_profit = price_level * 1.003
                        else:
                            stop_loss = price_level * 1.003
                            take_profit = price_level * 0.997
                        
                        signal = TradingSignal(
                            symbol=symbol,
                            signal_type=SignalType.ICEBERG_DEFENSE_ENTRY,
                            direction=signal_direction,
                            strength=SignalStrength.MODERATE,
                            confidence=confidence,
                            entry_price=entry_price,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            position_size=self._calculate_position_size(entry_price, stop_loss),
                            timestamp=current_time,
                            features=features,
                            metadata={
                                'iceberg_level': price_level,
                                'replenishment_count': replenishment_count,
                                'avg_iceberg_size': avg_size
                            }
                        )
                        signals.append(signal)
        
        return signals
    
    def _detect_vacuum_entry(self, order_book: OrderBook, features: MarketFeatures) -> List[TradingSignal]:
        """Detect vacuum entry opportunities (LVN exploitation)"""
        signals = []
        
        # Check if we're breaking through a significant level into a vacuum
        if features.lvn_distance < 0.001 and features.flow_imbalance > 0.4:
            # Strong momentum into low volume area
            signal_direction = 'long' if features.net_flow > 0 else 'short'
            
            # Calculate confidence based on momentum and volume
            momentum_strength = abs(features.flow_imbalance)
            volume_acceleration = features.aggressive_buy_volume + features.aggressive_sell_volume
            
            confidence = min(0.8, momentum_strength * 0.5 + min(volume_acceleration / 2000, 1.0) * 0.3 + 0.2)
            
            if confidence >= self.config['min_signal_confidence']:
                entry_price = features.mid_price
                
                if signal_direction == 'long':
                    stop_loss = entry_price * 0.995
                    take_profit = entry_price * 1.008  # Larger target for vacuum moves
                else:
                    stop_loss = entry_price * 1.005
                    take_profit = entry_price * 0.992
                
                signal = TradingSignal(
                    symbol=order_book.symbol,
                    signal_type=SignalType.VACUUM_ENTRY,
                    direction=signal_direction,
                    strength=SignalStrength.STRONG,
                    confidence=confidence,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    position_size=self._calculate_position_size(entry_price, stop_loss),
                    timestamp=int(time.time() * 1000000),
                    features=features,
                    metadata={
                        'lvn_distance': features.lvn_distance,
                        'flow_imbalance': features.flow_imbalance,
                        'volume_acceleration': volume_acceleration
                    }
                )
                signals.append(signal)
        
        return signals
    
    def _detect_mean_reversion_fade(self, order_book: OrderBook, features: MarketFeatures) -> List[TradingSignal]:
        """Detect mean reversion fade opportunities"""
        signals = []
        
        # Check for extreme VWAP deviation with absorption
        if abs(features.vwap_deviation) > 0.005 and features.absorption_strength > 0.6:
            # Price is far from VWAP and showing absorption
            signal_direction = 'short' if features.vwap_deviation > 0 else 'long'
            
            # Calculate confidence
            deviation_strength = min(abs(features.vwap_deviation) / 0.01, 1.0)
            confidence = min(0.8, deviation_strength * 0.4 + features.absorption_strength * 0.4 + 0.2)
            
            if confidence >= self.config['min_signal_confidence']:
                entry_price = features.mid_price
                
                # Target back towards VWAP
                if signal_direction == 'long':
                    stop_loss = entry_price * 0.997
                    take_profit = features.vwap * 1.001
                else:
                    stop_loss = entry_price * 1.003
                    take_profit = features.vwap * 0.999
                
                signal = TradingSignal(
                    symbol=order_book.symbol,
                    signal_type=SignalType.MEAN_REVERSION_FADE,
                    direction=signal_direction,
                    strength=SignalStrength.MODERATE,
                    confidence=confidence,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    position_size=self._calculate_position_size(entry_price, stop_loss),
                    timestamp=int(time.time() * 1000000),
                    features=features,
                    metadata={
                        'vwap_deviation': features.vwap_deviation,
                        'absorption_strength': features.absorption_strength,
                        'target_vwap': features.vwap
                    }
                )
                signals.append(signal)
        
        return signals
    
    def _detect_momentum_continuation(self, order_book: OrderBook, features: MarketFeatures) -> List[TradingSignal]:
        """Detect momentum continuation opportunities"""
        signals = []
        
        # Check for pullback to HVN with renewed momentum
        if (features.hvn_distance < 0.002 and 
            abs(features.flow_imbalance) > 0.3 and 
            features.price_acceleration > 0):
            
            signal_direction = 'long' if features.flow_imbalance > 0 else 'short'
            
            # Calculate confidence
            momentum_strength = abs(features.flow_imbalance)
            acceleration_factor = min(features.price_acceleration / 0.01, 1.0)
            
            confidence = min(0.75, momentum_strength * 0.4 + acceleration_factor * 0.3 + 0.3)
            
            if confidence >= self.config['min_signal_confidence']:
                entry_price = features.mid_price
                
                if signal_direction == 'long':
                    stop_loss = features.volume_profile_poc * 0.998
                    take_profit = entry_price * 1.006
                else:
                    stop_loss = features.volume_profile_poc * 1.002
                    take_profit = entry_price * 0.994
                
                signal = TradingSignal(
                    symbol=order_book.symbol,
                    signal_type=SignalType.MOMENTUM_CONTINUATION,
                    direction=signal_direction,
                    strength=SignalStrength.MODERATE,
                    confidence=confidence,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    position_size=self._calculate_position_size(entry_price, stop_loss),
                    timestamp=int(time.time() * 1000000),
                    features=features,
                    metadata={
                        'hvn_distance': features.hvn_distance,
                        'flow_imbalance': features.flow_imbalance,
                        'price_acceleration': features.price_acceleration
                    }
                )
                signals.append(signal)
        
        return signals
    
    def _detect_order_book_imbalance(self, order_book: OrderBook, features: MarketFeatures) -> List[TradingSignal]:
        """Detect order book imbalance signals"""
        signals = []
        
        # Check for extreme order book imbalance
        if abs(features.order_book_imbalance) > self.config['imbalance_threshold']:
            signal_direction = 'long' if features.order_book_imbalance > 0 else 'short'
            
            # Calculate confidence based on imbalance strength and flow
            imbalance_strength = abs(features.order_book_imbalance)
            flow_confirmation = abs(features.flow_imbalance) > 0.2
            
            confidence = min(0.7, imbalance_strength * 0.5 + (0.3 if flow_confirmation else 0.1) + 0.2)
            
            if confidence >= self.config['min_signal_confidence']:
                entry_price = features.mid_price
                
                if signal_direction == 'long':
                    stop_loss = entry_price * 0.998
                    take_profit = entry_price * 1.004
                else:
                    stop_loss = entry_price * 1.002
                    take_profit = entry_price * 0.996
                
                signal = TradingSignal(
                    symbol=order_book.symbol,
                    signal_type=SignalType.ORDER_BOOK_IMBALANCE,
                    direction=signal_direction,
                    strength=SignalStrength.WEAK,
                    confidence=confidence,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    position_size=self._calculate_position_size(entry_price, stop_loss),
                    timestamp=int(time.time() * 1000000),
                    features=features,
                    metadata={
                        'order_book_imbalance': features.order_book_imbalance,
                        'flow_confirmation': flow_confirmation
                    }
                )
                signals.append(signal)
        
        return signals
    
    def _update_absorption_events(self, order_book: OrderBook, features: MarketFeatures) -> None:
        """Update absorption event tracking"""
        # Detect new absorption events
        if features.absorption_strength > 0.7:
            absorption_event = AbsorptionEvent(
                price=features.mid_price,
                volume_absorbed=features.aggressive_buy_volume + features.aggressive_sell_volume,
                timestamp=int(time.time() * 1000000),
                side='bid' if features.flow_imbalance < 0 else 'ask',
                strength=features.absorption_strength
            )
            self._absorption_events[order_book.symbol].append(absorption_event)
    
    def _update_sweep_events(self, order_book: OrderBook, features: MarketFeatures) -> None:
        """Update liquidity sweep event tracking"""
        # Simplified sweep detection
        if features.sweep_probability > 0.8:
            recent_trades = order_book.get_recent_trades(20)
            if recent_trades:
                total_volume = sum(trade.size for trade in recent_trades)
                
                if total_volume > self.config['sweep_volume_threshold']:
                    sweep_event = LiquiditySweep(
                        sweep_price=features.mid_price,
                        target_price=features.volume_profile_poc,
                        volume=total_volume,
                        timestamp=int(time.time() * 1000000),
                        direction='up' if features.net_flow > 0 else 'down',
                        trapped_volume=total_volume * 0.3  # Estimate
                    )
                    self._sweep_events[order_book.symbol].append(sweep_event)
    
    def _update_iceberg_levels(self, order_book: OrderBook, features: MarketFeatures) -> None:
        """Update iceberg level tracking"""
        if features.iceberg_probability > 0.7:
            best_bid = order_book.get_best_bid()
            best_ask = order_book.get_best_ask()
            
            current_time = int(time.time() * 1000000)
            
            # Track bid iceberg
            if best_bid and best_bid.size > 100:
                price = best_bid.price
                if price not in self._iceberg_levels[order_book.symbol]:
                    self._iceberg_levels[order_book.symbol][price] = {
                        'side': 'bid',
                        'first_seen': current_time,
                        'replenishment_count': 0,
                        'avg_size': 0,
                        'last_replenishment': current_time
                    }
                
                iceberg_data = self._iceberg_levels[order_book.symbol][price]
                iceberg_data['replenishment_count'] += 1
                iceberg_data['avg_size'] = (iceberg_data['avg_size'] + best_bid.size) / 2
                iceberg_data['last_replenishment'] = current_time
            
            # Track ask iceberg
            if best_ask and best_ask.size > 100:
                price = best_ask.price
                if price not in self._iceberg_levels[order_book.symbol]:
                    self._iceberg_levels[order_book.symbol][price] = {
                        'side': 'ask',
                        'first_seen': current_time,
                        'replenishment_count': 0,
                        'avg_size': 0,
                        'last_replenishment': current_time
                    }
                
                iceberg_data = self._iceberg_levels[order_book.symbol][price]
                iceberg_data['replenishment_count'] += 1
                iceberg_data['avg_size'] = (iceberg_data['avg_size'] + best_ask.size) / 2
                iceberg_data['last_replenishment'] = current_time
    
    def _calculate_position_size(self, entry_price: float, stop_loss: float, 
                                risk_per_trade: float = 0.01) -> float:
        """Calculate position size based on risk management"""
        if entry_price <= 0 or stop_loss <= 0:
            return 0.0
        
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share == 0:
            return 0.0
        
        # Assume $10,000 account size for calculation
        account_size = 10000.0
        risk_amount = account_size * risk_per_trade
        
        position_size = risk_amount / risk_per_share
        return round(position_size, 2)
    
    def get_signal_history(self, symbol: str, count: int = 50) -> List[TradingSignal]:
        """Get signal history for symbol"""
        with self._lock:
            history = list(self._signal_history[symbol])
            return history[-count:] if count else history
    
    def get_statistics(self, symbol: str) -> Dict[str, Any]:
        """Get signal detector statistics"""
        with self._lock:
            return {
                'symbol': symbol,
                'total_signals': len(self._signal_history[symbol]),
                'absorption_events': len(self._absorption_events[symbol]),
                'sweep_events': len(self._sweep_events[symbol]),
                'iceberg_levels': len(self._iceberg_levels[symbol]),
                'config': self.config.copy()
            }

