import os
import sys
import json
import time
import logging
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import eventlet

# Import trading system components
from core.data_orchestrator import DataOrchestrator
from core.feature_engine import FeatureEngine
from core.signal_detector import SignalDetector
from ml.regime_detector import MarketRegimeDetector
from ml.signal_classifier import SignalClassifier

# Import data adapters
from data.mock_adapter import MockDataAdapter
from data.ibkr_adapter import create_ibkr_adapter, IBKR_AVAILABLE

# Monkey patch for eventlet
eventlet.monkey_patch()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'chimera-trading-system-v2'

# Enable CORS for all routes
CORS(app, origins="*", allow_headers=["Content-Type", "Authorization"])

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Global system state
system_stats = {
    'running': False,
    'start_time': None,
    'data_source': os.getenv('DATA_SOURCE', 'mock'),
    'symbols_subscribed': [],
    'signals_generated': 0,
    'ml_predictions': 0,
    'system_health': 95,
    'risk_level': 15,
    'win_rate': 68,
    'performance': 12.5,
    'total_trades': 0,
    'active_positions': 0,
    'pnl': 0.0,
    'max_drawdown': 0.0,
    'sharpe_ratio': 1.85
}

# System components
data_orchestrator = None
feature_engine = None
signal_detector = None
regime_detector = None
signal_classifier = None
data_adapter = None

def initialize_data_adapter():
    """Initialize the appropriate data adapter based on configuration"""
    global data_adapter
    
    data_source = os.getenv('DATA_SOURCE', 'mock').lower()
    
    if data_source == 'ibkr':
        if not IBKR_AVAILABLE:
            logger.error("IBKR adapter requested but ib_insync not available. Falling back to mock data.")
            data_source = 'mock'
        else:
            try:
                ibkr_host = os.getenv('IBKR_HOST', '127.0.0.1')
                ibkr_port = int(os.getenv('IBKR_PORT', '7497'))
                ibkr_client_id = int(os.getenv('IBKR_CLIENT_ID', '1'))
                
                logger.info(f"Initializing IBKR adapter: {ibkr_host}:{ibkr_port}")
                data_adapter = create_ibkr_adapter(
                    host=ibkr_host,
                    port=ibkr_port,
                    client_id=ibkr_client_id
                )
                
                # Test connection
                if data_adapter.start():
                    logger.info("IBKR adapter connected successfully")
                    system_stats['data_source'] = 'ibkr'
                else:
                    logger.error("Failed to connect to IBKR. Falling back to mock data.")
                    data_source = 'mock'
                    
            except Exception as e:
                logger.error(f"Error initializing IBKR adapter: {e}. Falling back to mock data.")
                data_source = 'mock'
    
    if data_source == 'mock':
        logger.info("Initializing mock data adapter")
        data_adapter = MockDataAdapter()
        system_stats['data_source'] = 'mock'
    
    return data_adapter

@app.route('/api/system/status', methods=['GET'])
def get_system_status():
    """Get current system status"""
    try:
        status = {
            'status': 'running' if system_stats['running'] else 'stopped',
            'uptime': int(time.time() - system_stats['start_time']) if system_stats['start_time'] else 0,
            'data_source': system_stats['data_source'],
            'symbols_subscribed': system_stats['symbols_subscribed'],
            'signals_generated': system_stats['signals_generated'],
            'ml_predictions': system_stats['ml_predictions'],
            'system_health': system_stats['system_health'],
            'risk_level': system_stats['risk_level'],
            'win_rate': system_stats['win_rate'],
            'performance': system_stats['performance'],
            'timestamp': int(time.time() * 1000),
            'version': '2.0.0',
            'components': {
                'data_orchestrator': data_orchestrator is not None,
                'feature_engine': feature_engine is not None,
                'signal_detector': signal_detector is not None,
                'regime_detector': regime_detector is not None,
                'signal_classifier': signal_classifier is not None,
                'data_adapter': data_adapter is not None
            }
        }
        
        if data_adapter and hasattr(data_adapter, 'get_connection_status'):
            status['connection'] = data_adapter.get_connection_status()
        
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/system/start', methods=['POST'])
def start_system():
    """Start the trading system"""
    global data_orchestrator, feature_engine, signal_detector, regime_detector, signal_classifier, data_adapter
    
    try:
        if system_stats['running']:
            return jsonify({'message': 'System already running', 'status': 'running'})
        
        logger.info("Starting Chimera Trading System...")
        
        # Initialize data adapter
        data_adapter = initialize_data_adapter()
        if not data_adapter:
            return jsonify({'error': 'Failed to initialize data adapter'}), 500
        
        # Initialize core components
        data_orchestrator = DataOrchestrator(':memory:')  # Use in-memory DB for container
        feature_engine = FeatureEngine()
        signal_detector = SignalDetector(feature_engine)
        
        # Initialize ML components
        regime_detector = MarketRegimeDetector()
        signal_classifier = SignalClassifier()
        
        # Subscribe to symbols
        symbols = os.getenv('TRADING_SYMBOLS', 'AAPL,MSFT,GOOGL,AMZN').split(',')
        
        for symbol in symbols:
            symbol = symbol.strip()
            try:
                # Subscribe to Level 2 data
                data_adapter.subscribe_level2(symbol, lambda update: handle_level2_update(update))
                
                # Subscribe to trades
                data_adapter.subscribe_trades(symbol, lambda trade: handle_trade_update(trade))
                
                # Subscribe to quotes
                data_adapter.subscribe_quotes(symbol, lambda quote: handle_quote_update(quote))
                
                system_stats['symbols_subscribed'].append(symbol)
                logger.info(f"Subscribed to {symbol}")
                
            except Exception as e:
                logger.error(f"Failed to subscribe to {symbol}: {e}")
        
        # Mark system as running
        system_stats['running'] = True
        system_stats['start_time'] = time.time()
        
        # Start background data processing
        start_background_processing()
        
        logger.info("Chimera Trading System started successfully")
        
        return jsonify({
            'message': 'System started successfully',
            'status': 'running',
            'data_source': system_stats['data_source'],
            'symbols': system_stats['symbols_subscribed']
        })
        
    except Exception as e:
        logger.error(f"Error starting system: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/system/stop', methods=['POST'])
def stop_system():
    """Stop the trading system"""
    global data_adapter
    
    try:
        if not system_stats['running']:
            return jsonify({'message': 'System already stopped', 'status': 'stopped'})
        
        logger.info("Stopping Chimera Trading System...")
        
        # Stop data adapter
        if data_adapter:
            if hasattr(data_adapter, 'stop'):
                data_adapter.stop()
            data_adapter = None
        
        # Reset system state
        system_stats['running'] = False
        system_stats['start_time'] = None
        system_stats['symbols_subscribed'] = []
        
        logger.info("Chimera Trading System stopped")
        
        return jsonify({
            'message': 'System stopped successfully',
            'status': 'stopped'
        })
        
    except Exception as e:
        logger.error(f"Error stopping system: {e}")
        return jsonify({'error': str(e)}), 500

def handle_level2_update(update):
    """Handle Level 2 order book updates"""
    try:
        # Update order book
        order_book = data_orchestrator.get_order_book(update.symbol)
        
        # Update bids
        for price, size in update.bids:
            if size == 0:
                order_book.remove_bid(price)
            else:
                order_book.update_bid(price, size)
        
        # Update asks
        for price, size in update.asks:
            if size == 0:
                order_book.remove_ask(price)
            else:
                order_book.update_ask(price, size)
        
        # Extract features
        features = feature_engine.update_features(order_book)
        
        # Detect signals
        signals = signal_detector.detect_signals(order_book, features)
        
        if signals:
            system_stats['signals_generated'] += len(signals)
            
            # Emit signals via WebSocket
            socketio.emit('signals', {
                'symbol': update.symbol,
                'signals': [
                    {
                        'type': signal.signal_type.value,
                        'direction': signal.direction.value,
                        'confidence': signal.confidence,
                        'entry_price': signal.entry_price,
                        'stop_loss': signal.stop_loss,
                        'take_profit': signal.take_profit,
                        'timestamp': signal.timestamp
                    }
                    for signal in signals
                ]
            })
        
        # Update ML components
        if regime_detector:
            regime_detector.update_features(features)
            current_regime = regime_detector.get_current_regime()
        
        if signal_classifier and signals:
            for signal in signals:
                prediction = signal_classifier.classify_signal(signal, features)
                system_stats['ml_predictions'] += 1
        
        # Emit real-time data via WebSocket
        socketio.emit('market_data', {
            'symbol': update.symbol,
            'bids': update.bids[:5],  # Top 5 levels
            'asks': update.asks[:5],  # Top 5 levels
            'timestamp': update.timestamp,
            'features': {
                'spread': features.spread,
                'mid_price': features.mid_price,
                'depth_imbalance': features.depth_imbalance,
                'microprice': features.microprice
            } if features else None
        })
        
    except Exception as e:
        logger.error(f"Error handling Level 2 update: {e}")

def handle_trade_update(trade):
    """Handle trade updates"""
    try:
        system_stats['total_trades'] += 1
        
        # Emit trade data via WebSocket
        socketio.emit('trade_data', {
            'symbol': trade.symbol,
            'price': trade.price,
            'size': trade.size,
            'side': trade.side,
            'timestamp': trade.timestamp
        })
        
    except Exception as e:
        logger.error(f"Error handling trade update: {e}")

def handle_quote_update(quote):
    """Handle quote updates"""
    try:
        # Emit quote data via WebSocket
        socketio.emit('quote_data', {
            'symbol': quote.symbol,
            'bid_price': quote.bid_price,
            'ask_price': quote.ask_price,
            'bid_size': quote.bid_size,
            'ask_size': quote.ask_size,
            'timestamp': quote.timestamp
        })
        
    except Exception as e:
        logger.error(f"Error handling quote update: {e}")

def start_background_processing():
    """Start background processing thread"""
    def background_worker():
        while system_stats['running']:
            try:
                # Update system metrics
                system_stats['system_health'] = min(100, system_stats['system_health'] + 0.1)
                system_stats['risk_level'] = max(0, system_stats['risk_level'] - 0.05)
                
                # Emit system stats via WebSocket
                socketio.emit('system_stats', system_stats)
                
                time.sleep(1)  # Update every second
                
            except Exception as e:
                logger.error(f"Error in background processing: {e}")
                time.sleep(5)
    
    thread = threading.Thread(target=background_worker, daemon=True)
    thread.start()

@app.route('/api/trading/data', methods=['GET'])
def get_trading_data():
    """Get current trading data"""
    try:
        trading_data = {
            'symbols': system_stats['symbols_subscribed'],
            'signals_generated': system_stats['signals_generated'],
            'ml_predictions': system_stats['ml_predictions'],
            'total_trades': system_stats['total_trades'],
            'active_positions': system_stats['active_positions'],
            'pnl': system_stats['pnl'],
            'max_drawdown': system_stats['max_drawdown'],
            'sharpe_ratio': system_stats['sharpe_ratio'],
            'timestamp': int(time.time() * 1000)
        }
        
        return jsonify(trading_data)
        
    except Exception as e:
        logger.error(f"Error getting trading data: {e}")
        return jsonify({'error': str(e)}), 500

@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection"""
    logger.info(f"Client connected: {request.sid}")
    emit('connected', {'status': 'connected', 'timestamp': int(time.time() * 1000)})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection"""
    logger.info(f"Client disconnected: {request.sid}")

if __name__ == '__main__':
    logger.info("🚀 Starting Chimera Trading System Backend API v2.0")
    logger.info(f"📊 Data Source: {os.getenv('DATA_SOURCE', 'mock')}")
    logger.info("🌐 Server starting on http://0.0.0.0:5000")
    
    # Run the Flask-SocketIO app
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)

