#!/usr/bin/env python3
"""
Chimera Trading System v2.0 - Main Application
Complete algorithmic trading system with ML enhancement

Usage:
    python main.py [command] [options]

Commands:
    run         - Start the complete trading system
    test        - Run integration tests
    demo        - Run demonstration with mock data
    optimize    - Run genetic algorithm optimization
    backtest    - Run backtesting on historical data
    
Options:
    --symbol SYMBOL     - Trading symbol (default: MOCK)
    --duration SECONDS  - Simulation duration (default: 300)
    --updates-per-sec N - Market updates per second (default: 50)
    --no-ml            - Disable ML components
    --no-gui           - Run in headless mode
    --config FILE      - Configuration file path
    --help             - Show this help message
"""

import sys
import os
import argparse
import time
import threading
import signal
import json
from typing import Dict, List, Any, Optional

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import system components
from core import (
    DataOrchestrator, FeatureEngine, SignalDetector, SubscriptionConfig
)
from data import (
    MockDataGenerator, MarketSimulator, SimulationConfig, MarketScenario
)
from ml import (
    MarketRegimeDetector, SignalClassifier, RLExitAgent, GeneticOptimizer
)
from core.signal_detector import SignalType
from tests.test_integration import run_integration_tests


class ChimeraTradingSystem:
    """
    Main trading system orchestrator
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_running = False
        self.shutdown_event = threading.Event()
        
        # Core components
        self.data_orchestrator = None
        self.feature_engine = None
        self.signal_detector = None
        
        # ML components
        self.regime_detector = None
        self.signal_classifier = None
        self.rl_agents: Dict[str, RLExitAgent] = {}
        self.genetic_optimizer = None
        
        # Market simulation
        self.market_simulator = None
        
        # Statistics
        self.stats = {
            'start_time': 0,
            'signals_generated': 0,
            'trades_executed': 0,
            'ml_predictions': 0,
            'regime_changes': 0
        }
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\nReceived signal {signum}, shutting down gracefully...")
        self.shutdown()
    
    def initialize(self) -> bool:
        """Initialize all system components"""
        try:
            print("Initializing Chimera Trading System...")
            
            # Initialize core components
            db_path = self.config.get('database_path', 'chimera_market_data.db')
            self.data_orchestrator = DataOrchestrator(db_path)
            self.feature_engine = FeatureEngine()
            self.signal_detector = SignalDetector(self.feature_engine)
            
            # Initialize ML components if enabled
            if not self.config.get('no_ml', False):
                print("Initializing ML components...")
                self.regime_detector = MarketRegimeDetector()
                self.signal_classifier = SignalClassifier()
                self.genetic_optimizer = GeneticOptimizer()
                
                # Set up genetic optimizer backtest function
                self.genetic_optimizer.set_backtest_function(
                    self.genetic_optimizer.create_simple_backtest_function()
                )
            
            print("✓ System initialization completed")
            return True
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            return False
    
    def start_market_simulation(self) -> bool:
        """Start market data simulation"""
        try:
            symbols = self.config.get('symbols', ['MOCK'])
            duration = self.config.get('duration', 300)
            updates_per_sec = self.config.get('updates_per_second', 50)
            
            print(f"Starting market simulation for {symbols}")
            print(f"Duration: {duration}s, Updates/sec: {updates_per_sec}")
            
            # Create simulation config
            sim_config = SimulationConfig(
                instruments=symbols,
                duration_seconds=duration,
                updates_per_second=updates_per_sec,
                scenarios={},
                correlation_matrix={},
                enable_cross_asset_effects=len(symbols) > 1,
                enable_news_events=True
            )
            
            # Create and start simulator
            self.market_simulator = MarketSimulator(sim_config)
            
            # Subscribe to market events
            self.market_simulator.add_event_callback(self._handle_market_event)
            
            # Start simulation
            self.market_simulator.start_simulation()
            
            print("✓ Market simulation started")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start market simulation: {e}")
            return False
    
    def start_trading_engine(self) -> bool:
        """Start the main trading engine"""
        try:
            print("Starting trading engine...")
            
            # Start data orchestrator
            self.data_orchestrator.start()
            
            # Subscribe to symbols
            symbols = self.config.get('symbols', ['MOCK'])
            for symbol in symbols:
                config = SubscriptionConfig(
                    symbol=symbol,
                    data_types=['level2', 'trades']
                )
                self.data_orchestrator.add_subscription(config)
                
                # Create RL exit agent for each symbol
                if not self.config.get('no_ml', False):
                    self.rl_agents[symbol] = RLExitAgent(symbol)
            
            # Subscribe to data updates
            self.data_orchestrator.subscribe_to_updates(
                self._handle_market_update, symbols
            )
            
            print("✓ Trading engine started")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start trading engine: {e}")
            return False
    
    def _handle_market_update(self, update):
        """Handle incoming market data updates"""
        try:
            if update.update_type == 'level2':
                # Get order book
                order_book = self.data_orchestrator.get_order_book(update.symbol)
                if not order_book:
                    return
                
                # Calculate features
                features = self.feature_engine.update_features(order_book)
                
                # Detect market regime (if ML enabled)
                regime = None
                if self.regime_detector:
                    regime = self.regime_detector.update_features(update.symbol, features)
                    if regime:
                        self.stats['regime_changes'] += 1
                
                # Detect trading signals
                signals = self.signal_detector.detect_signals(order_book, features)
                
                for signal in signals:
                    self.stats['signals_generated'] += 1
                    self._process_trading_signal(signal, regime)
                    
        except Exception as e:
            print(f"Error handling market update: {e}")
    
    def _process_trading_signal(self, signal, regime=None):
        """Process a detected trading signal"""
        try:
            print(f"📊 Signal: {signal.signal_type.value} | {signal.direction} | "
                  f"Confidence: {signal.confidence:.3f} | Price: {signal.entry_price:.4f}")
            
            # Get ML probability if available
            if self.signal_classifier:
                prediction = self.signal_classifier.predict_signal_probability(signal, regime)
                self.stats['ml_predictions'] += 1
                
                print(f"🤖 ML Probability: {prediction.ml_probability:.3f} | "
                      f"Regime Adjusted: {prediction.regime_adjusted_probability:.3f}")
                
                # Only proceed if ML probability is above threshold
                ml_threshold = self.config.get('ml_threshold', 0.6)
                if prediction.regime_adjusted_probability < ml_threshold:
                    print(f"⚠️  Signal rejected - ML probability below threshold ({ml_threshold})")
                    return
            
            # Simulate trade execution
            self._simulate_trade_execution(signal)
            
        except Exception as e:
            print(f"Error processing signal: {e}")
    
    def _simulate_trade_execution(self, signal):
        """Simulate trade execution (placeholder for real execution)"""
        try:
            # This is where real trade execution would happen
            # For now, we just simulate and track statistics
            
            print(f"💰 TRADE EXECUTED: {signal.direction.upper()} {signal.symbol} @ {signal.entry_price:.4f}")
            
            self.stats['trades_executed'] += 1
            
            # Simulate position management with RL agent if available
            if signal.symbol in self.rl_agents:
                # This would be called periodically for open positions
                # For demo purposes, we'll just show it's available
                rl_agent = self.rl_agents[signal.symbol]
                print(f"🧠 RL Exit Agent ready for position management")
            
        except Exception as e:
            print(f"Error simulating trade execution: {e}")
    
    def _handle_market_event(self, event):
        """Handle market events from simulator"""
        print(f"📰 Market Event: {event.event_type.value} | "
              f"Impact: {event.impact_strength:.2f} | "
              f"Instruments: {event.affected_instruments}")
    
    def run_main_loop(self):
        """Run the main trading loop"""
        print("\n" + "="*60)
        print("🚀 CHIMERA TRADING SYSTEM v2.0 - RUNNING")
        print("="*60)
        
        self.is_running = True
        self.stats['start_time'] = time.time()
        
        try:
            # Main loop
            while self.is_running and not self.shutdown_event.is_set():
                # Print periodic status
                if int(time.time()) % 30 == 0:  # Every 30 seconds
                    self._print_status()
                
                # Check if simulation is complete
                if self.market_simulator:
                    status = self.market_simulator.get_simulation_status()
                    if not status['is_running']:
                        print("\n📊 Market simulation completed")
                        break
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
        except Exception as e:
            print(f"\n❌ Error in main loop: {e}")
        finally:
            self.shutdown()
    
    def _print_status(self):
        """Print current system status"""
        elapsed = time.time() - self.stats['start_time']
        
        print(f"\n📈 Status Update (Runtime: {elapsed:.0f}s)")
        print(f"   Signals Generated: {self.stats['signals_generated']}")
        print(f"   Trades Executed: {self.stats['trades_executed']}")
        print(f"   ML Predictions: {self.stats['ml_predictions']}")
        print(f"   Regime Changes: {self.stats['regime_changes']}")
        
        # Get simulation status if available
        if self.market_simulator:
            sim_status = self.market_simulator.get_simulation_status()
            print(f"   Simulation Progress: {sim_status['progress_percent']:.1f}%")
    
    def shutdown(self):
        """Shutdown the system gracefully"""
        if not self.is_running:
            return
        
        print("\n🔄 Shutting down system...")
        self.is_running = False
        self.shutdown_event.set()
        
        # Stop market simulator
        if self.market_simulator:
            self.market_simulator.stop_simulation()
        
        # Stop data orchestrator
        if self.data_orchestrator:
            self.data_orchestrator.stop()
        
        # Save ML models
        if self.regime_detector:
            self.regime_detector._save_model()
        
        for rl_agent in self.rl_agents.values():
            rl_agent.save_model()
        
        print("✓ System shutdown completed")
    
    def run_optimization(self, symbol: str, signal_type: SignalType, generations: int = 10):
        """Run genetic algorithm optimization"""
        if not self.genetic_optimizer:
            print("❌ Genetic optimizer not available (ML disabled)")
            return
        
        print(f"\n🧬 Starting genetic optimization for {symbol} {signal_type.value}")
        
        try:
            result = self.genetic_optimizer.run_optimization(symbol, signal_type, generations)
            
            print(f"\n✓ Optimization completed!")
            print(f"Best fitness: {result['final_stats']['best_fitness']:.4f}")
            
            return result
            
        except Exception as e:
            print(f"❌ Optimization failed: {e}")
            return None


def load_config(config_file: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from file or use defaults"""
    default_config = {
        'symbols': ['MOCK'],
        'duration': 300,
        'updates_per_second': 50,
        'no_ml': False,
        'no_gui': True,
        'ml_threshold': 0.6,
        'database_path': 'chimera_market_data.db'
    }
    
    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
            default_config.update(file_config)
            print(f"✓ Loaded configuration from {config_file}")
        except Exception as e:
            print(f"⚠️  Failed to load config file: {e}, using defaults")
    
    return default_config


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Chimera Trading System v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('command', nargs='?', default='run',
                       choices=['run', 'test', 'demo', 'optimize', 'backtest'],
                       help='Command to execute')
    
    parser.add_argument('--symbol', default='MOCK',
                       help='Trading symbol (default: MOCK)')
    
    parser.add_argument('--duration', type=int, default=300,
                       help='Simulation duration in seconds (default: 300)')
    
    parser.add_argument('--updates-per-sec', type=int, default=50,
                       help='Market updates per second (default: 50)')
    
    parser.add_argument('--no-ml', action='store_true',
                       help='Disable ML components')
    
    parser.add_argument('--no-gui', action='store_true', default=True,
                       help='Run in headless mode')
    
    parser.add_argument('--config',
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with command line arguments
    config.update({
        'symbols': [args.symbol],
        'duration': args.duration,
        'updates_per_second': args.updates_per_sec,
        'no_ml': args.no_ml,
        'no_gui': args.no_gui
    })
    
    # Execute command
    if args.command == 'test':
        print("Running integration tests...")
        success = run_integration_tests()
        sys.exit(0 if success else 1)
    
    elif args.command == 'demo':
        print("Running demonstration mode...")
        config['duration'] = 60  # Short demo
        config['symbols'] = ['DEMO']
    
    elif args.command == 'optimize':
        print("Running genetic optimization...")
        system = ChimeraTradingSystem(config)
        if system.initialize():
            system.run_optimization(args.symbol, SignalType.LIQUIDITY_SWEEP_REVERSAL)
        sys.exit(0)
    
    elif args.command == 'backtest':
        print("Backtesting mode not yet implemented")
        sys.exit(1)
    
    # Default: run the trading system
    system = ChimeraTradingSystem(config)
    
    if not system.initialize():
        print("❌ Failed to initialize system")
        sys.exit(1)
    
    if not system.start_market_simulation():
        print("❌ Failed to start market simulation")
        sys.exit(1)
    
    if not system.start_trading_engine():
        print("❌ Failed to start trading engine")
        sys.exit(1)
    
    # Run main loop
    system.run_main_loop()
    
    print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()

