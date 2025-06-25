import os
import sys
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Add parent directories to path for importing trading system
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room

# Import trading system components
try:
    from core.data_orchestrator import DataOrchestrator
    from core.feature_engine import FeatureEngine
    from core.signal_detector import SignalDetector
    from data.mock_data_generator import MockDataGenerator
    from data.market_simulator import MarketSimulator
    from ml.regime_detector import MarketRegimeDetector
    from ml.signal_classifier import SignalClassifier
    from ml.rl_exit_agent import RLExitAgent
    TRADING_SYSTEM_AVAILABLE = True
    print("✓ Trading system components imported successfully")
except ImportError as e:
    print(f"Warning: Could not import trading system components: {e}")
    print("Dashboard will run in mock mode")
    TRADING_SYSTEM_AVAILABLE = False

app = Flask(__name__)
app.config['SECRET_KEY'] = 'chimera_dashboard_secret_key_2024'

# Enable CORS for all routes
CORS(app, origins="*")

# Initialize SocketIO for real-time communication
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global variables for trading system integration
trading_system_active = False
data_orchestrator = None
feature_engine = None
signal_detector = None
mock_data_generator = None
market_simulator = None
ml_components = {}

system_stats = {
    'status': 'stopped',
    'uptime': 0,
    'start_time': None,
    'signals_generated': 0,
    'trades_executed': 0,
    'ml_predictions': 0,
    'regime_changes': 0,
    'active_symbols': ['AAPL', 'NVDA', 'TSLA', 'SPY'],
    'performance': {
        'total_pnl': 0.0,
        'daily_pnl': 0.0,
        'sharpe_ratio': 0.0,
        'max_drawdown': 0.0,
        'win_rate': 0.0
    },
    'risk': {
        'current_exposure': 0.0,
        'var_95': 2500.0,
        'position_count': 0,
        'leverage': 1.0
    }
}

# Real-time data storage
real_time_data = {
    'order_books': {},
    'recent_signals': [],
    'recent_trades': [],
    'market_data': {},
    'system_alerts': []
}

def initialize_trading_system():
    """Initialize the trading system components with real integration"""
    global trading_system_active, data_orchestrator, feature_engine, signal_detector
    global mock_data_generator, market_simulator, ml_components
    
    if not TRADING_SYSTEM_AVAILABLE:
        print("Trading system components not available, using mock mode")
        return False
    
    try:
        print("🔧 Initializing trading system components...")
        
        # Initialize data orchestrator
        data_orchestrator = DataOrchestrator("dashboard_market_data.db")
        print("✓ Data orchestrator initialized")
        
        # Initialize feature engine
        feature_engine = FeatureEngine()
        print("✓ Feature engine initialized")
        
        # Initialize signal detector
        signal_detector = SignalDetector(feature_engine)
        print("✓ Signal detector initialized")
        
        # Initialize mock data generator
        mock_data_generator = MockDataGenerator()
        print("✓ Mock data generator initialized")
        
        # Note: MarketSimulator requires config, skipping for now
        # market_simulator = MarketSimulator()
        # print("✓ Market simulator initialized")
        
        # Initialize ML components (simplified for dashboard integration)
        ml_components = {
            'regime_detector': MarketRegimeDetector(),
            'signal_classifier': SignalClassifier(),
            # 'rl_exit_agent': RLExitAgent('AAPL')  # Requires symbol, skip for now
        }
        print("✓ ML components initialized")
        
        trading_system_active = True
        print("✅ Trading system initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize trading system: {e}")
        import traceback
        traceback.print_exc()
        return False

def start_trading_system():
    """Start the trading system and begin generating data"""
    global system_stats, real_time_data
    
    if not trading_system_active:
        if not initialize_trading_system():
            return False
    
    try:
        system_stats['status'] = 'running'
        system_stats['start_time'] = time.time()
        
        # Start mock data generation
        if mock_data_generator:
            # Generate initial market data for multiple symbols
            for symbol in system_stats['active_symbols']:
                order_book = mock_data_generator.generate_order_book(symbol)
                real_time_data['order_books'][symbol] = order_book
        
        print("✅ Trading system started successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to start trading system: {e}")
        system_stats['status'] = 'error'
        return False

def stop_trading_system():
    """Stop the trading system"""
    global system_stats
    
    system_stats['status'] = 'stopped'
    system_stats['start_time'] = None
    system_stats['uptime'] = 0
    
    print("🛑 Trading system stopped")
    return True

def update_system_stats():
    """Update system statistics with real data from trading system"""
    global system_stats, real_time_data
    
    if system_stats['start_time']:
        system_stats['uptime'] = int(time.time() - system_stats['start_time'])
    
    if trading_system_active and system_stats['status'] == 'running':
        try:
            # Generate new signals using the signal detector
            if signal_detector and mock_data_generator:
                # Generate market data for signal detection
                for symbol in system_stats['active_symbols'][:2]:  # Limit to 2 symbols for performance
                    # Generate new order book data
                    order_book = mock_data_generator.generate_order_book(symbol)
                    real_time_data['order_books'][symbol] = order_book
                    
                    # Extract features with correct method name
                    if feature_engine:
                        features = feature_engine.update_features(order_book)
                        
                        # Detect signals with correct parameter order
                        signals = signal_detector.detect_signals(order_book, features)
                        
                        # Add new signals to recent signals
                        for signal in signals:
                            signal_data = {
                                'symbol': symbol,
                                'type': signal.signal_type.value if hasattr(signal.signal_type, 'value') else str(signal.signal_type),
                                'price': signal.entry_price,  # Use correct attribute name
                                'confidence': signal.confidence,
                                'direction': signal.direction,
                                'timestamp': time.time() * 1000
                            }
                            real_time_data['recent_signals'].append(signal_data)
                            system_stats['signals_generated'] += 1
                        
                        # Keep only recent signals (last 50)
                        real_time_data['recent_signals'] = real_time_data['recent_signals'][-50:]
            
            # Update performance metrics with realistic simulation
            import random
            
            # Simulate trading performance
            if system_stats['signals_generated'] > 0:
                # Simulate some trades based on signals
                if random.random() > 0.8:  # 20% chance of new trade
                    trade_pnl = random.uniform(-50, 100)  # Random P&L
                    system_stats['performance']['daily_pnl'] += trade_pnl
                    system_stats['performance']['total_pnl'] += trade_pnl
                    system_stats['trades_executed'] += 1
                    
                    # Add trade to recent trades
                    trade_data = {
                        'symbol': random.choice(system_stats['active_symbols']),
                        'side': random.choice(['buy', 'sell']),
                        'price': round(150 + random.uniform(-10, 10), 2),
                        'size': random.randint(100, 1000),
                        'pnl': trade_pnl,
                        'timestamp': time.time() * 1000
                    }
                    real_time_data['recent_trades'].append(trade_data)
                    real_time_data['recent_trades'] = real_time_data['recent_trades'][-100:]
            
            # Update win rate
            if system_stats['trades_executed'] > 0:
                profitable_trades = len([t for t in real_time_data['recent_trades'] if t.get('pnl', 0) > 0])
                system_stats['performance']['win_rate'] = profitable_trades / min(system_stats['trades_executed'], len(real_time_data['recent_trades']))
            
            # Update other metrics
            system_stats['performance']['sharpe_ratio'] = min(system_stats['performance']['win_rate'] * 2, 3.0)
            system_stats['performance']['max_drawdown'] = abs(min(0, system_stats['performance']['daily_pnl'] / 1000))
            
            # Update risk metrics
            system_stats['risk']['position_count'] = min(system_stats['trades_executed'], 10)
            system_stats['risk']['current_exposure'] = min(abs(system_stats['performance']['daily_pnl']) / 10000, 0.8)
            
            # ML predictions
            if random.random() > 0.9:  # 10% chance of new ML prediction
                system_stats['ml_predictions'] += 1
                
        except Exception as e:
            print(f"Error updating system stats: {e}")

def start_system_monitoring():
    """Start background monitoring of the trading system"""
    def monitor_loop():
        while True:
            try:
                if system_stats['status'] == 'running':
                    # Update system statistics with real data
                    update_system_stats()
                    
                    # Emit real-time updates to connected clients
                    socketio.emit('system:status', system_stats)
                    socketio.emit('trading:data', get_trading_data())
                    socketio.emit('performance:metrics', get_performance_metrics())
                    socketio.emit('risk:metrics', get_risk_metrics())
                
                time.sleep(2)  # Update every 2 seconds
                
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(5)
    
    monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
    monitor_thread.start()
    print("📊 System monitoring started")
def get_trading_data():
    """Get current trading data"""
    try:
        # Convert order books to serializable format
        serializable_order_books = {}
        for symbol, order_book in real_time_data['order_books'].items():
            if order_book:
                serializable_order_books[symbol] = {
                    'symbol': symbol,
                    'bids': [[level.price, level.size] for level in order_book.get_bids()[:10]],
                    'asks': [[level.price, level.size] for level in order_book.get_asks()[:10]],
                    'mid_price': order_book.get_mid_price(),
                    'spread': order_book.get_spread(),
                    'timestamp': int(time.time() * 1000)
                }
        
        return {
            'signals': real_time_data['recent_signals'][-10:],  # Last 10 signals
            'trades': real_time_data['recent_trades'][-20:],    # Last 20 trades
            'order_books': serializable_order_books,
            'timestamp': int(time.time() * 1000)
        }
    except Exception as e:
        print(f"Error in get_trading_data: {e}")
        return {
            'signals': [],
            'trades': [],
            'order_books': {},
            'timestamp': int(time.time() * 1000),
            'error': str(e)
        }

def get_performance_metrics():
    """Get performance metrics"""
    return {
        'total_pnl': system_stats['performance']['total_pnl'],
        'daily_pnl': system_stats['performance']['daily_pnl'],
        'sharpe_ratio': system_stats['performance']['sharpe_ratio'],
        'max_drawdown': system_stats['performance']['max_drawdown'],
        'win_rate': system_stats['performance']['win_rate'],
        'timestamp': int(time.time() * 1000)
    }

def get_risk_metrics():
    """Get risk metrics"""
    return {
        'current_exposure': system_stats['risk']['current_exposure'],
        'var_95': system_stats['risk']['var_95'],
        'position_count': system_stats['risk']['position_count'],
        'leverage': system_stats['risk']['leverage'],
        'timestamp': int(time.time() * 1000)
    }

# API Routes

@app.route('/api/system/status')
def get_system_status():
    """Get current system status"""
    return jsonify(system_stats)

@app.route('/api/system/start', methods=['POST'])
def start_system():
    """Start the trading system"""
    try:
        if start_trading_system():
            return jsonify({
                'success': True,
                'message': 'Trading system started successfully',
                'status': system_stats
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to start trading system'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error starting system: {str(e)}'
        }), 500

@app.route('/api/system/stop', methods=['POST'])
def stop_system():
    """Stop the trading system"""
    try:
        if stop_trading_system():
            return jsonify({
                'success': True,
                'message': 'Trading system stopped successfully',
                'status': system_stats
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to stop trading system'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error stopping system: {str(e)}'
        }), 500

@app.route('/api/trading/data')
def get_trading_data_endpoint():
    """Get current trading data"""
    return jsonify(get_trading_data())

@app.route('/api/performance/metrics')
def get_performance_metrics_endpoint():
    """Get performance metrics"""
    return jsonify(get_performance_metrics())

@app.route('/api/risk/metrics')
def get_risk_metrics_endpoint():
    """Get risk metrics"""
    return jsonify(get_risk_metrics())

# WebSocket Events

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print(f"Client connected: {request.sid}")
    emit('system:status', system_stats)

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print(f"Client disconnected: {request.sid}")

@socketio.on('start_system')
def handle_start_system():
    """Handle start system request via WebSocket"""
    if start_trading_system():
        emit('system:status', system_stats, broadcast=True)
        emit('message', {'type': 'success', 'text': 'Trading system started'}, broadcast=True)
    else:
        emit('message', {'type': 'error', 'text': 'Failed to start trading system'})

@socketio.on('stop_system')
def handle_stop_system():
    """Handle stop system request via WebSocket"""
    if stop_trading_system():
        emit('system:status', system_stats, broadcast=True)
        emit('message', {'type': 'success', 'text': 'Trading system stopped'}, broadcast=True)
    else:
        emit('message', {'type': 'error', 'text': 'Failed to stop trading system'})

if __name__ == '__main__':
    print("🚀 Starting Chimera Trading Dashboard API...")
    
    # Initialize trading system on startup
    initialize_trading_system()
    
    # Start system monitoring
    start_system_monitoring()
    
    print("📊 Dashboard API running on http://localhost:5000")
    print("🔌 WebSocket endpoint: ws://localhost:5000")
    
    # Run the Flask-SocketIO app
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
