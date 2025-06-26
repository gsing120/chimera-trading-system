#!/usr/bin/env python3
"""
Real IBKR Integration Test for Chimera Trading System
Tests actual connection to IBKR Gateway and data flow
"""

import sys
import os
import time
import threading
from typing import Dict, Any

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import ChimeraTradingSystem
from data.ibkr_adapter import create_ibkr_adapter


class RealIntegrationTest:
    """Test real IBKR integration without simulations"""
    
    def __init__(self):
        self.test_results = {}
        self.connection_established = False
        self.data_received = False
        
    def test_ibkr_connection(self) -> bool:
        """Test IBKR Gateway connection"""
        print("🔍 Testing IBKR Gateway connection...")
        
        try:
            # Create IBKR adapter
            adapter = create_ibkr_adapter(
                host='127.0.0.1',
                port=4002,  # Gateway paper trading port
                client_id=1
            )
            
            # Test connection
            if adapter.start():
                print("✅ IBKR Gateway connection successful")
                self.connection_established = True
                
                # Test connection status
                status = adapter.get_connection_status()
                print(f"📊 Connection Status: {status}")
                
                # Clean up
                adapter.stop()
                return True
            else:
                print("❌ IBKR Gateway connection failed")
                return False
                
        except Exception as e:
            print(f"❌ IBKR connection test failed: {e}")
            return False
    
    def test_data_subscription(self) -> bool:
        """Test real data subscription"""
        print("🔍 Testing real data subscription...")
        
        try:
            # Create adapter
            adapter = create_ibkr_adapter()
            
            if not adapter.start():
                print("❌ Could not establish connection for data test")
                return False
            
            # Set up data callback
            data_count = 0
            
            def data_callback(update):
                nonlocal data_count
                data_count += 1
                if data_count == 1:
                    print(f"✅ First data update received: {update.symbol}")
                    self.data_received = True
            
            # Subscribe to test symbol
            test_symbol = 'AAPL'
            print(f"📡 Subscribing to {test_symbol} data...")
            adapter.subscribe_level2(test_symbol, data_callback)
            
            # Wait for data
            timeout = 30
            start_time = time.time()
            
            while not self.data_received and (time.time() - start_time) < timeout:
                time.sleep(1)
            
            # Clean up
            adapter.unsubscribe(test_symbol)
            adapter.stop()
            
            if self.data_received:
                print(f"✅ Real data subscription successful ({data_count} updates)")
                return True
            else:
                print("❌ No data received within timeout")
                return False
                
        except Exception as e:
            print(f"❌ Data subscription test failed: {e}")
            return False
    
    def test_chimera_system_integration(self) -> bool:
        """Test full Chimera system with IBKR"""
        print("🔍 Testing Chimera system integration...")
        
        try:
            # Configuration for real IBKR
            config = {
                'symbols': ['AAPL'],
                'ibkr_host': '127.0.0.1',
                'ibkr_port': 4002,
                'ibkr_client_id': 1,
                'no_ml': True,  # Disable ML for faster testing
                'database_path': './test_chimera.db'
            }
            
            # Create system
            system = ChimeraTradingSystem(config)
            
            # Initialize
            if not system.initialize():
                print("❌ System initialization failed")
                return False
            
            print("✅ Chimera system initialized")
            
            # Test IBKR connection
            if not system.start_ibkr_connection():
                print("❌ IBKR connection in system failed")
                return False
            
            print("✅ IBKR connection in system successful")
            
            # Let it run briefly to test data flow
            print("⏳ Testing data flow for 10 seconds...")
            time.sleep(10)
            
            # Check statistics
            stats = system.stats
            print(f"📊 System Statistics: {stats}")
            
            # Shutdown
            system.shutdown()
            print("✅ System shutdown successful")
            
            return True
            
        except Exception as e:
            print(f"❌ Chimera system integration test failed: {e}")
            return False
    
    def test_configuration_validation(self) -> bool:
        """Test configuration validation"""
        print("🔍 Testing configuration validation...")
        
        try:
            # Check environment file
            env_path = '.env.example'
            if os.path.exists(env_path):
                with open(env_path, 'r') as f:
                    content = f.read()
                    
                # Check for required IBKR settings
                required_settings = [
                    'DATA_SOURCE=ibkr',
                    'IBKR_PORT=4002',
                    'TWS_USERID=isht1430',
                    'TRADING_MODE=paper'
                ]
                
                for setting in required_settings:
                    if setting in content:
                        print(f"✅ Found: {setting}")
                    else:
                        print(f"❌ Missing: {setting}")
                        return False
                
                # Check no mock data references
                forbidden_settings = [
                    'MOCK_',
                    'mock',
                    'simulation'
                ]
                
                for forbidden in forbidden_settings:
                    if forbidden.lower() in content.lower():
                        print(f"⚠️  Found forbidden setting: {forbidden}")
                
                print("✅ Configuration validation passed")
                return True
            else:
                print("❌ Environment file not found")
                return False
                
        except Exception as e:
            print(f"❌ Configuration validation failed: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all integration tests"""
        print("🚀 Starting Real IBKR Integration Tests")
        print("=" * 50)
        
        tests = [
            ("Configuration Validation", self.test_configuration_validation),
            ("IBKR Connection", self.test_ibkr_connection),
            ("Data Subscription", self.test_data_subscription),
            ("Chimera System Integration", self.test_chimera_system_integration)
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            print(f"\n🧪 Running: {test_name}")
            print("-" * 30)
            
            try:
                result = test_func()
                results[test_name] = result
                
                if result:
                    print(f"✅ {test_name}: PASSED")
                else:
                    print(f"❌ {test_name}: FAILED")
                    
            except Exception as e:
                print(f"💥 {test_name}: ERROR - {e}")
                results[test_name] = False
        
        # Summary
        print("\n" + "=" * 50)
        print("📋 TEST SUMMARY")
        print("=" * 50)
        
        passed = sum(1 for r in results.values() if r)
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name}: {status}")
        
        print(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 ALL TESTS PASSED - System ready for deployment!")
        else:
            print("⚠️  Some tests failed - Check IBKR Gateway connection")
        
        return results


if __name__ == "__main__":
    test = RealIntegrationTest()
    results = test.run_all_tests()
    
    # Exit with appropriate code
    all_passed = all(results.values())
    sys.exit(0 if all_passed else 1)

