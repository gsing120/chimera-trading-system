"""
Market Simulator
Comprehensive market simulation with multiple scenarios and instruments
"""

import time
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import json

from .mock_data_generator import MockDataGenerator, MarketScenario, MarketRegime
from core.data_orchestrator import DataOrchestrator


@dataclass
class SimulationConfig:
    """Configuration for market simulation"""
    instruments: List[str]
    duration_seconds: int
    updates_per_second: int
    scenarios: Dict[str, MarketScenario]
    correlation_matrix: Dict[str, Dict[str, float]]
    enable_cross_asset_effects: bool = True
    enable_news_events: bool = True
    save_data: bool = True


class EventType(Enum):
    """Types of market events"""
    NEWS_RELEASE = "news_release"
    EARNINGS = "earnings"
    FED_ANNOUNCEMENT = "fed_announcement"
    MARKET_OPEN = "market_open"
    MARKET_CLOSE = "market_close"
    VOLATILITY_SPIKE = "volatility_spike"


@dataclass
class MarketEvent:
    """Represents a market event"""
    event_type: EventType
    timestamp: int
    affected_instruments: List[str]
    impact_strength: float
    duration_seconds: int
    metadata: Dict[str, Any]


class MarketSimulator:
    """
    Comprehensive market simulator for testing trading systems
    """
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        
        # Data orchestrator
        self.data_orchestrator = DataOrchestrator("simulation_data.db")
        
        # Mock data generators for each instrument
        self.generators: Dict[str, MockDataGenerator] = {}
        
        # Market events
        self.scheduled_events: List[MarketEvent] = []
        self.event_callbacks: List[Callable[[MarketEvent], None]] = []
        
        # Simulation state
        self.start_time = 0
        self.current_time = 0
        self.is_running = False
        
        # Statistics
        self.stats = {
            'total_updates': 0,
            'total_trades': 0,
            'events_triggered': 0,
            'instruments_simulated': len(config.instruments)
        }
        
        # Threading
        self._simulation_thread = None
        self._event_thread = None
        self._lock = threading.RLock()
        
        # Initialize generators
        self._initialize_generators()
        
        # Schedule events
        self._schedule_events()
    
    def _initialize_generators(self) -> None:
        """Initialize mock data generators for each instrument"""
        base_prices = {
            'AAPL': 150.0,
            'NVDA': 400.0,
            'TSLA': 200.0,
            'SPY': 450.0,
            'QQQ': 350.0,
            'MOCK': 100.0
        }
        
        for instrument in self.config.instruments:
            base_price = base_prices.get(instrument, 100.0)
            generator = MockDataGenerator(instrument, base_price)
            
            # Set scenario if specified
            if instrument in self.config.scenarios:
                generator.set_scenario(self.config.scenarios[instrument])
            
            self.generators[instrument] = generator
    
    def _schedule_events(self) -> None:
        """Schedule market events during simulation"""
        if not self.config.enable_news_events:
            return
        
        # Schedule some random events
        event_times = [
            self.config.duration_seconds * 0.2,  # 20% through
            self.config.duration_seconds * 0.5,  # 50% through
            self.config.duration_seconds * 0.8   # 80% through
        ]
        
        for i, event_time in enumerate(event_times):
            event = MarketEvent(
                event_type=EventType.NEWS_RELEASE,
                timestamp=int(event_time * 1000000),  # Convert to microseconds
                affected_instruments=self.config.instruments[:2],  # Affect first 2 instruments
                impact_strength=0.5 + i * 0.2,
                duration_seconds=60,
                metadata={'news_type': 'earnings', 'sentiment': 'positive' if i % 2 == 0 else 'negative'}
            )
            self.scheduled_events.append(event)
    
    def add_event_callback(self, callback: Callable[[MarketEvent], None]) -> None:
        """Add callback for market events"""
        self.event_callbacks.append(callback)
    
    def start_simulation(self) -> None:
        """Start the market simulation"""
        with self._lock:
            if self.is_running:
                return
            
            self.is_running = True
            self.start_time = int(time.time() * 1000000)
            self.current_time = self.start_time
            
            # Start data orchestrator
            self.data_orchestrator.start()
            
            # Start generators
            for generator in self.generators.values():
                generator.start_generation(self.data_orchestrator, self.config.updates_per_second)
            
            # Start simulation threads
            self._simulation_thread = threading.Thread(target=self._simulation_loop, daemon=True)
            self._event_thread = threading.Thread(target=self._event_loop, daemon=True)
            
            self._simulation_thread.start()
            self._event_thread.start()
            
            print(f"Market simulation started with {len(self.config.instruments)} instruments")
    
    def stop_simulation(self) -> None:
        """Stop the market simulation"""
        with self._lock:
            if not self.is_running:
                return
            
            self.is_running = False
            
            # Stop generators
            for generator in self.generators.values():
                generator.stop_generation()
            
            # Stop data orchestrator
            self.data_orchestrator.stop()
            
            # Wait for threads
            if self._simulation_thread:
                self._simulation_thread.join(timeout=5.0)
            if self._event_thread:
                self._event_thread.join(timeout=5.0)
            
            print("Market simulation stopped")
    
    def _simulation_loop(self) -> None:
        """Main simulation loop"""
        while self.is_running:
            current_real_time = int(time.time() * 1000000)
            self.current_time = current_real_time
            
            # Check if simulation duration is complete
            elapsed_time = (current_real_time - self.start_time) / 1000000  # Convert to seconds
            if elapsed_time >= self.config.duration_seconds:
                print(f"Simulation completed after {elapsed_time:.1f} seconds")
                self.stop_simulation()
                break
            
            # Apply cross-asset effects if enabled
            if self.config.enable_cross_asset_effects:
                self._apply_cross_asset_effects()
            
            # Update statistics
            self._update_statistics()
            
            time.sleep(1.0)  # Update every second
    
    def _event_loop(self) -> None:
        """Event processing loop"""
        while self.is_running:
            current_time = self.current_time
            
            # Check for scheduled events
            for event in list(self.scheduled_events):
                if current_time >= self.start_time + event.timestamp:
                    self._trigger_event(event)
                    self.scheduled_events.remove(event)
            
            time.sleep(0.1)  # Check events every 100ms
    
    def _apply_cross_asset_effects(self) -> None:
        """Apply cross-asset correlation effects"""
        if not self.config.correlation_matrix:
            return
        
        # Get current states of all generators
        states = {}
        for instrument, generator in self.generators.items():
            states[instrument] = generator.get_current_state()
        
        # Apply correlations
        for instrument1, correlations in self.config.correlation_matrix.items():
            if instrument1 not in self.generators:
                continue
            
            generator1 = self.generators[instrument1]
            
            for instrument2, correlation in correlations.items():
                if instrument2 not in states or instrument1 == instrument2:
                    continue
                
                # Apply correlation effect
                state2 = states[instrument2]
                price_change_2 = state2['current_price'] - state2.get('previous_price', state2['current_price'])
                
                if abs(price_change_2) > 0.001:  # Significant price change
                    # Adjust volatility and trend based on correlation
                    correlation_effect = correlation * 0.1  # Scale down the effect
                    
                    if correlation > 0.5:  # Positive correlation
                        generator1.volatility = min(0.01, generator1.volatility + abs(correlation_effect))
                        if price_change_2 > 0:
                            generator1.trend_strength = min(1.0, generator1.trend_strength + correlation_effect)
                        else:
                            generator1.trend_strength = max(-1.0, generator1.trend_strength - correlation_effect)
                    elif correlation < -0.5:  # Negative correlation
                        generator1.volatility = min(0.01, generator1.volatility + abs(correlation_effect))
                        if price_change_2 > 0:
                            generator1.trend_strength = max(-1.0, generator1.trend_strength - correlation_effect)
                        else:
                            generator1.trend_strength = min(1.0, generator1.trend_strength + correlation_effect)
    
    def _trigger_event(self, event: MarketEvent) -> None:
        """Trigger a market event"""
        print(f"Triggering event: {event.event_type.value} affecting {event.affected_instruments}")
        
        # Apply event effects to affected instruments
        for instrument in event.affected_instruments:
            if instrument in self.generators:
                generator = self.generators[instrument]
                
                # Modify generator parameters based on event
                if event.event_type == EventType.NEWS_RELEASE:
                    # Increase volatility and trend strength
                    generator.volatility = min(0.01, generator.volatility * (1 + event.impact_strength))
                    
                    sentiment = event.metadata.get('sentiment', 'neutral')
                    if sentiment == 'positive':
                        generator.trend_strength = min(1.0, generator.trend_strength + event.impact_strength)
                    elif sentiment == 'negative':
                        generator.trend_strength = max(-1.0, generator.trend_strength - event.impact_strength)
                
                elif event.event_type == EventType.VOLATILITY_SPIKE:
                    generator.volatility = min(0.02, generator.volatility * 3.0)
                    generator.current_regime = MarketRegime.HIGH_VOLATILITY
        
        # Notify callbacks
        for callback in self.event_callbacks:
            try:
                callback(event)
            except Exception as e:
                print(f"Error in event callback: {e}")
        
        self.stats['events_triggered'] += 1
        
        # Schedule event end (return to normal)
        end_time = event.timestamp + event.duration_seconds * 1000000
        end_event = MarketEvent(
            event_type=EventType.MARKET_CLOSE,  # Reuse as "event end"
            timestamp=end_time,
            affected_instruments=event.affected_instruments,
            impact_strength=-event.impact_strength,  # Reverse effect
            duration_seconds=0,
            metadata={'original_event': event.event_type.value}
        )
        self.scheduled_events.append(end_event)
    
    def _update_statistics(self) -> None:
        """Update simulation statistics"""
        total_updates = 0
        total_trades = 0
        
        for generator in self.generators.values():
            state = generator.get_current_state()
            stats = state['statistics']
            total_updates += stats['orders_generated']
            total_trades += stats['trades_generated']
        
        self.stats['total_updates'] = total_updates
        self.stats['total_trades'] = total_trades
    
    def get_simulation_status(self) -> Dict[str, Any]:
        """Get current simulation status"""
        with self._lock:
            elapsed_time = (self.current_time - self.start_time) / 1000000 if self.start_time > 0 else 0
            
            status = {
                'is_running': self.is_running,
                'elapsed_time': elapsed_time,
                'remaining_time': max(0, self.config.duration_seconds - elapsed_time),
                'progress_percent': min(100, (elapsed_time / self.config.duration_seconds) * 100),
                'statistics': self.stats.copy(),
                'instruments': {}
            }
            
            # Add instrument-specific status
            for instrument, generator in self.generators.items():
                status['instruments'][instrument] = generator.get_current_state()
            
            return status
    
    def get_order_book(self, instrument: str):
        """Get order book for specific instrument"""
        return self.data_orchestrator.get_order_book(instrument)
    
    def get_historical_data(self, instrument: str, limit: int = 1000) -> List[Dict]:
        """Get historical data for instrument"""
        return self.data_orchestrator.get_historical_data(instrument, limit=limit)
    
    def get_historical_trades(self, instrument: str, limit: int = 1000) -> List[Dict]:
        """Get historical trades for instrument"""
        return self.data_orchestrator.get_historical_trades(instrument, limit=limit)
    
    def inject_custom_event(self, event: MarketEvent) -> None:
        """Inject a custom market event"""
        with self._lock:
            # Adjust timestamp to be relative to simulation start
            event.timestamp = (self.current_time - self.start_time) + event.timestamp
            self.scheduled_events.append(event)
            print(f"Custom event scheduled: {event.event_type.value}")
    
    def create_test_simulation(self, test_name: str) -> 'SimulationConfig':
        """Create predefined test simulation configurations"""
        test_configs = {
            'single_instrument_test': SimulationConfig(
                instruments=['MOCK'],
                duration_seconds=300,  # 5 minutes
                updates_per_second=50,
                scenarios={
                    'MOCK': MockDataGenerator('MOCK').create_test_scenario('absorption_test')
                },
                correlation_matrix={},
                enable_cross_asset_effects=False,
                enable_news_events=True
            ),
            
            'multi_instrument_test': SimulationConfig(
                instruments=['AAPL', 'NVDA', 'SPY'],
                duration_seconds=600,  # 10 minutes
                updates_per_second=30,
                scenarios={
                    'AAPL': MockDataGenerator('AAPL').create_test_scenario('liquidity_sweep_test'),
                    'NVDA': MockDataGenerator('NVDA').create_test_scenario('iceberg_test'),
                    'SPY': MockDataGenerator('SPY').create_test_scenario('high_volatility_test')
                },
                correlation_matrix={
                    'AAPL': {'NVDA': 0.7, 'SPY': 0.8},
                    'NVDA': {'AAPL': 0.7, 'SPY': 0.6},
                    'SPY': {'AAPL': 0.8, 'NVDA': 0.6}
                },
                enable_cross_asset_effects=True,
                enable_news_events=True
            ),
            
            'stress_test': SimulationConfig(
                instruments=['MOCK1', 'MOCK2', 'MOCK3', 'MOCK4', 'MOCK5'],
                duration_seconds=1800,  # 30 minutes
                updates_per_second=100,
                scenarios={},  # Use default scenarios
                correlation_matrix={},
                enable_cross_asset_effects=False,
                enable_news_events=False
            )
        }
        
        return test_configs.get(test_name, test_configs['single_instrument_test'])
    
    def export_simulation_data(self, filename: str) -> bool:
        """Export simulation data to file"""
        try:
            data = {
                'config': {
                    'instruments': self.config.instruments,
                    'duration_seconds': self.config.duration_seconds,
                    'updates_per_second': self.config.updates_per_second
                },
                'statistics': self.stats,
                'status': self.get_simulation_status()
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            return True
        except Exception as e:
            print(f"Error exporting simulation data: {e}")
            return False

