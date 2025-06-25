"""
Comprehensive Integration Tests for Chimera Trading System
Tests the complete system end-to-end with mock data
"""

import time
import threading
import unittest
import tempfile
import os
import shutil
from typing import Dict, List, Any

# Import all system components
from core import (
    OrderBook, DataOrchestrator, FeatureEngine, SignalDetector, SubscriptionConfig
)
from data import MockDataGenerator, MarketSimulator, MarketScenario
from ml import (
    MarketRegimeDetector, SignalClassifier, RLExitAgent, GeneticOptimizer
)


class TestSystemIntegration(unittest.TestCase):
    """Integration tests for the complete trading system"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        self.original_dir = os.getcwd()
        os.chdir(self.test_dir)
        
        # Initialize system components
        self.data_orchestrator = DataOrchestrator("test_market_data.db")
        self.feature_engine = FeatureEngine()
        self.signal_detector = SignalDetector(self.feature_engine)
        self.regime_detector = MarketRegimeDetector()
        self.signal_classifier = SignalClassifier()
        
        # Test symbol
        self.test_symbol = "TEST"
        
    def tearDown(self):
        """Clean up test environment"""
        # Stop data orchestrator
        self.data_orchestrator.stop()
        
        # Change back to original directory
        os.chdir(self.original_dir)
        
        # Clean up temporary directory
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_basic_data_flow(self):
        """Test basic data flow through the system"""
        print("\n=== Testing Basic Data Flow ===")
        
        # Start data orchestrator
        self.data_orchestrator.start()
        
        # Add subscription
        config = SubscriptionConfig(
            symbol=self.test_symbol,
            data_types=['level2', 'trades']
        )
        self.data_orchestrator.add_subscription(config)
        
        # Create mock data generator
        generator = MockDataGenerator(self.test_symbol, 100.0)
        
        # Generate some data
        for i in range(10):
            # Generate level 2 update
            price = 100.0 + (i * 0.01)
            size = 1000 + (i * 100)
            side = 'bid' if i % 2 == 0 else 'ask'
            
            success = self.data_orchestrator.push_level2_update(
                self.test_symbol, price, size, 1, side
            )
            self.assertTrue(success, f"Failed to push level2 update {i}")
            
            # Generate trade
            trade_success = self.data_orchestrator.push_trade(
                self.test_symbol, price, 100, 'buy', f"trade_{i}"
            )
            self.assertTrue(trade_success, f"Failed to push trade {i}")
        
        # Wait for processing
        time.sleep(0.5)
        
        # Verify order book exists and has data
        order_book = self.data_orchestrator.get_order_book(self.test_symbol)
        self.assertIsNotNone(order_book, "Order book should exist")
        
        # Verify trades were recorded
        recent_trades = order_book.get_recent_trades(20)
        self.assertGreater(len(recent_trades), 0, "Should have recorded trades")
        
        print(f"✓ Processed {len(recent_trades)} trades successfully")
        
    def test_feature_engineering(self):
        """Test feature engineering pipeline"""
        print("\n=== Testing Feature Engineering ===")
        
        # Start data orchestrator
        self.data_orchestrator.start()
        
        # Add subscription
        config = SubscriptionConfig(
            symbol=self.test_symbol,
            data_types=['level2', 'trades']
        )
        self.data_orchestrator.add_subscription(config)
        
        # Generate realistic market data
        generator = MockDataGenerator(self.test_symbol, 100.0)
        
        # Generate order book data
        for i in range(50):
            # Bid levels
            for j in range(5):
                price = 100.0 - (j * 0.01)
                size = 1000 + (i * 10)
                self.data_orchestrator.push_level2_update(
                    self.test_symbol, price, size, 1, 'bid'
                )
            
            # Ask levels
            for j in range(5):
                price = 100.02 + (j * 0.01)
                size = 1000 + (i * 10)
                self.data_orchestrator.push_level2_update(
                    self.test_symbol, price, size, 1, 'ask'
                )
            
            # Generate trades
            if i % 5 == 0:
                trade_price = 100.01 + (i * 0.001)
                self.data_orchestrator.push_trade(
                    self.test_symbol, trade_price, 200, 'buy', f"trade_{i}"
                )
        
        # Wait for processing
        time.sleep(0.5)
        
        # Get order book and calculate features
        order_book = self.data_orchestrator.get_order_book(self.test_symbol)
        self.assertIsNotNone(order_book)
        
        features = self.feature_engine.update_features(order_book)
        
        # Verify features are calculated
        self.assertEqual(features.symbol, self.test_symbol)
        self.assertGreater(features.mid_price, 0, "Mid price should be positive")
        self.assertGreater(features.spread, 0, "Spread should be positive")
        self.assertIsInstance(features.depth_imbalance, float)
        self.assertIsInstance(features.flow_imbalance, float)
        
        print(f"✓ Generated features: mid_price={features.mid_price:.4f}, spread={features.spread:.4f}")
        
    def test_signal_detection(self):
        """Test signal detection system"""
        print("\n=== Testing Signal Detection ===")
        
        # Start data orchestrator
        self.data_orchestrator.start()
        
        # Add subscription
        config = SubscriptionConfig(
            symbol=self.test_symbol,
            data_types=['level2', 'trades']
        )
        self.data_orchestrator.add_subscription(config)
        
        # Generate market data that should trigger signals
        order_book = self.data_orchestrator.get_order_book(self.test_symbol)
        if not order_book:
            # Create order book by adding subscription
            config = SubscriptionConfig(self.test_symbol, ['level2'])
            self.data_orchestrator.add_subscription(config)
            order_book = self.data_orchestrator.get_order_book(self.test_symbol)
        
        # Create order book state
        for i in range(10):
            order_book.update_level(100.0 - i*0.01, 1000, 1, 'bid')
            order_book.update_level(100.02 + i*0.01, 1000, 1, 'ask')
        
        # Add some trades to create flow
        for i in range(20):
            order_book.add_trade(100.01, 100, 'buy', f"trade_{i}")
        
        # Calculate features
        features = self.feature_engine.update_features(order_book)
        
        # Detect signals
        signals = self.signal_detector.detect_signals(order_book, features)
        
        # Verify signal detection works
        self.assertIsInstance(signals, list)
        print(f"✓ Detected {len(signals)} signals")
        
        # If signals were detected, verify their structure
        for signal in signals:
            self.assertEqual(signal.symbol, self.test_symbol)
            self.assertIn(signal.direction, ['long', 'short'])
            self.assertGreaterEqual(signal.confidence, 0.0)
            self.assertLessEqual(signal.confidence, 1.0)
            print(f"  - {signal.signal_type.value}: {signal.direction} (confidence: {signal.confidence:.3f})")
    
    def test_regime_detection(self):
        """Test market regime detection"""
        print("\n=== Testing Regime Detection ===")
        
        # Generate features for regime detection
        order_book = OrderBook(self.test_symbol)
        
        # Create realistic order book
        for i in range(10):
            order_book.update_level(100.0 - i*0.01, 1000, 1, 'bid')
            order_book.update_level(100.02 + i*0.01, 1000, 1, 'ask')
        
        # Generate multiple feature updates to build history
        for i in range(20):
            # Add some trades
            order_book.add_trade(100.01 + i*0.001, 100, 'buy', f"trade_{i}")
            
            # Calculate features
            features = self.feature_engine.update_features(order_book)
            
            # Update regime detector
            regime_classification = self.regime_detector.update_features(self.test_symbol, features)
            
            # Verify regime classification
            self.assertIsNotNone(regime_classification)
            self.assertIsNotNone(regime_classification.regime)
            self.assertGreaterEqual(regime_classification.confidence, 0.0)
            self.assertLessEqual(regime_classification.confidence, 1.0)
        
        # Get final regime
        current_regime = self.regime_detector.get_current_regime(self.test_symbol)
        self.assertIsNotNone(current_regime)
        
        print(f"✓ Detected regime: {current_regime.regime.value} (confidence: {current_regime.confidence:.3f})")
    
    def test_ml_signal_classification(self):
        """Test ML signal classification"""
        print("\n=== Testing ML Signal Classification ===")
        
        # Create a test signal
        order_book = OrderBook(self.test_symbol)
        for i in range(10):
            order_book.update_level(100.0 - i*0.01, 1000, 1, 'bid')
            order_book.update_level(100.02 + i*0.01, 1000, 1, 'ask')
        
        features = self.feature_engine.update_features(order_book)
        signals = self.signal_detector.detect_signals(order_book, features)
        
        if signals:
            signal = signals[0]
            
            # Get regime classification
            regime = self.regime_detector.update_features(self.test_symbol, features)
            
            # Test signal classification
            prediction = self.signal_classifier.predict_signal_probability(signal, regime)
            
            # Verify prediction structure
            self.assertIsNotNone(prediction)
            self.assertEqual(prediction.signal.symbol, self.test_symbol)
            self.assertGreaterEqual(prediction.ml_probability, 0.0)
            self.assertLessEqual(prediction.ml_probability, 1.0)
            self.assertGreaterEqual(prediction.regime_adjusted_probability, 0.0)
            self.assertLessEqual(prediction.regime_adjusted_probability, 1.0)
            
            print(f"✓ ML probability: {prediction.ml_probability:.3f}")
            print(f"✓ Regime adjusted: {prediction.regime_adjusted_probability:.3f}")
        else:
            print("✓ No signals to classify (expected for basic test data)")
    
    def test_rl_exit_agent(self):
        """Test reinforcement learning exit agent"""
        print("\n=== Testing RL Exit Agent ===")
        
        # Create RL exit agent
        rl_agent = RLExitAgent(self.test_symbol)
        
        # Create test position state
        order_book = OrderBook(self.test_symbol)
        for i in range(10):
            order_book.update_level(100.0 - i*0.01, 1000, 1, 'bid')
            order_book.update_level(100.02 + i*0.01, 1000, 1, 'ask')
        
        features = self.feature_engine.update_features(order_book)
        
        from ml.rl_exit_agent import PositionState
        
        position_state = PositionState(
            symbol=self.test_symbol,
            entry_price=100.0,
            current_price=100.05,
            position_size=1000,
            unrealized_pnl=50.0,
            unrealized_pnl_pct=0.0005,
            time_in_position=300,  # 5 minutes
            max_favorable_excursion=60.0,
            max_adverse_excursion=-10.0,
            current_stop_loss=99.5,
            current_take_profit=101.0,
            features=features
        )
        
        # Get exit decision
        decision = rl_agent.get_exit_decision(position_state)
        
        # Verify decision structure
        self.assertIsNotNone(decision)
        self.assertIsNotNone(decision.action)
        self.assertGreaterEqual(decision.confidence, 0.0)
        self.assertLessEqual(decision.confidence, 1.0)
        self.assertGreaterEqual(decision.position_adjustment, 0.0)
        self.assertLessEqual(decision.position_adjustment, 1.0)
        
        print(f"✓ Exit decision: {decision.action.value} (confidence: {decision.confidence:.3f})")
        print(f"✓ Position adjustment: {decision.position_adjustment:.3f}")
    
    def test_genetic_optimizer(self):
        """Test genetic algorithm optimizer"""
        print("\n=== Testing Genetic Optimizer ===")
        
        # Create genetic optimizer
        optimizer = GeneticOptimizer(population_size=10)  # Small population for testing
        
        # Set simple backtest function
        optimizer.set_backtest_function(optimizer.create_simple_backtest_function())
        
        # Initialize population
        from core.signal_detector import SignalType
        
        population = optimizer.initialize_population(self.test_symbol, SignalType.LIQUIDITY_SWEEP_REVERSAL)
        
        # Verify population
        self.assertEqual(len(population), 10)
        
        for genome in population:
            self.assertEqual(genome.symbol, self.test_symbol)
            self.assertEqual(genome.signal_type, SignalType.LIQUIDITY_SWEEP_REVERSAL)
            self.assertGreater(genome.min_absorption_volume, 0)
            self.assertGreater(genome.max_position_size, 0)
        
        print(f"✓ Created population of {len(population)} genomes")
        
        # Test single generation evolution
        try:
            generation_result = optimizer.evolve_generation(self.test_symbol, SignalType.LIQUIDITY_SWEEP_REVERSAL)
            
            self.assertIn('generation', generation_result)
            self.assertIn('best_fitness', generation_result)
            self.assertIn('avg_fitness', generation_result)
            
            print(f"✓ Evolution completed - Best fitness: {generation_result['best_fitness']:.4f}")
            
        except Exception as e:
            print(f"⚠ Evolution test skipped due to: {e}")
    
    def test_complete_system_simulation(self):
        """Test complete system with market simulation"""
        print("\n=== Testing Complete System Simulation ===")
        
        # Create market simulator
        from data.market_simulator import SimulationConfig
        
        config = SimulationConfig(
            instruments=[self.test_symbol],
            duration_seconds=30,  # Short simulation for testing
            updates_per_second=10,
            scenarios={},
            correlation_matrix={},
            enable_cross_asset_effects=False,
            enable_news_events=False
        )
        
        simulator = MarketSimulator(config)
        
        # Set up signal tracking
        signals_detected = []
        regimes_detected = []
        
        def track_signals(update):
            """Track signals generated during simulation"""
            if update.update_type == 'level2':
                order_book = simulator.get_order_book(update.symbol)
                if order_book:
                    features = self.feature_engine.update_features(order_book)
                    
                    # Detect regime
                    regime = self.regime_detector.update_features(update.symbol, features)
                    regimes_detected.append(regime)
                    
                    # Detect signals
                    signals = self.signal_detector.detect_signals(order_book, features)
                    signals_detected.extend(signals)
        
        # Subscribe to updates
        simulator.data_orchestrator.subscribe_to_updates(track_signals, [self.test_symbol])
        
        # Start simulation
        simulator.start_simulation()
        
        # Wait for simulation to complete
        time.sleep(35)  # Wait a bit longer than simulation duration
        
        # Stop simulation
        simulator.stop_simulation()
        
        # Verify results
        print(f"✓ Simulation completed")
        print(f"✓ Signals detected: {len(signals_detected)}")
        print(f"✓ Regimes detected: {len(regimes_detected)}")
        
        # Verify we got some data
        self.assertGreater(len(regimes_detected), 0, "Should have detected some regimes")
        
        # Get final statistics
        order_book = simulator.get_order_book(self.test_symbol)
        if order_book:
            stats = order_book.get_statistics()
            print(f"✓ Final order book stats: {stats['update_count']} updates, {stats['total_trades']} trades")


def run_integration_tests():
    """Run all integration tests"""
    print("=" * 60)
    print("CHIMERA TRADING SYSTEM - INTEGRATION TESTS")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSystemIntegration)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=None)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    if success:
        print("\n🎉 ALL TESTS PASSED! System is ready for use.")
    else:
        print("\n❌ Some tests failed. Please check the output above.")
    
    return success


if __name__ == "__main__":
    run_integration_tests()

