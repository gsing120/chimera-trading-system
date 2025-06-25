# 🚀 Chimera Trading System v2.0

**Advanced Algorithmic Trading System with Machine Learning and Real-Time Dashboard**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![React](https://img.shields.io/badge/React-18+-61DAFB.svg)](https://reactjs.org)
[![Flask](https://img.shields.io/badge/Flask-3.1+-000000.svg)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Live Data Integration](#live-data-integration)
- [API Documentation](#api-documentation)
- [Dashboard Features](#dashboard-features)
- [Machine Learning Components](#machine-learning-components)
- [Trading Strategies](#trading-strategies)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

Chimera Trading System v2.0 is a sophisticated algorithmic trading platform that combines advanced market microstructure analysis, machine learning, and real-time visualization. Inspired by professional trading tools like Bookmap, it provides institutional-grade trading capabilities with a modern, intuitive interface.

### Key Highlights

- 🧠 **Advanced ML Pipeline**: Regime detection, signal classification, and reinforcement learning
- 📊 **Professional Dashboard**: Real-time visualization with Bookmap-style order book analysis
- ⚡ **High-Performance**: Optimized for low-latency trading with efficient data structures
- 🔌 **Multi-Broker Support**: Ready for IBKR, DXFeed, MooMoo, and other data sources
- 🛡️ **Risk Management**: Comprehensive risk controls and position management
- 📈 **Market Microstructure**: Advanced order flow analysis and liquidity detection

## ✨ Features

### 🎯 Core Trading Features
- **Multi-Strategy Signal Detection**: Liquidity sweeps, absorption patterns, iceberg orders
- **Advanced Order Book Analysis**: Depth imbalance, microprice calculation, flow analysis
- **Machine Learning Integration**: Market regime detection and signal classification
- **Real-Time Risk Management**: Position sizing, VaR calculation, drawdown monitoring
- **Performance Analytics**: Sharpe ratio, win rate, P&L tracking

### 📊 Dashboard Features
- **Real-Time Gauges**: System health, risk level, win rate, performance metrics
- **Interactive Charts**: Equity curves, signal frequency, order book heatmaps
- **Order Book Visualization**: Bookmap-style depth visualization with intensity colors
- **WebSocket Streaming**: Live data updates with sub-second latency
- **Dark Theme Interface**: Professional trading environment

### 🤖 Machine Learning Components
- **Market Regime Detector**: Identifies trending, ranging, and volatile market conditions
- **Signal Classifier**: ML-based signal validation and confidence scoring
- **Reinforcement Learning Exit Agent**: Optimizes trade exits using Q-learning
- **Genetic Algorithm Optimizer**: Evolves trading parameters automatically

### 🔌 Data Integration
- **Mock Data Generator**: Realistic Level 2 order book simulation for testing
- **Multi-Source Support**: IBKR, DXFeed, MooMoo, Yahoo Finance integration ready
- **Standardized Interface**: Easy switching between data providers
- **Real-Time Processing**: Efficient handling of high-frequency market data

## 🚀 Quick Start

### One-Command Setup
```bash
git clone https://github.com/yourusername/chimera-trading-system.git
cd chimera-trading-system
./start_complete_system.sh
```

### Access the System
- **Dashboard**: http://localhost:5173
- **API**: http://localhost:5000

### Start Trading
1. Open the dashboard
2. Click "Start System" button
3. Monitor real-time signals and order flow

## 🏗️ System Architecture

```
chimera_trading_system/
├── core/                    # Core trading engine
│   ├── data_orchestrator.py # Multi-instrument data management
│   ├── feature_engine.py    # Market microstructure features
│   ├── order_book.py        # Order book data structures
│   └── signal_detector.py   # Trading signal detection
├── ml/                      # Machine learning components
│   ├── regime_detector.py   # Market regime classification
│   ├── signal_classifier.py # ML signal validation
│   ├── rl_exit_agent.py     # Reinforcement learning exits
│   └── genetic_optimizer.py # Parameter optimization
├── data/                    # Data management
│   ├── mock_data_generator.py # Realistic market simulation
│   ├── data_interface.py    # Standardized data interface
│   └── market_simulator.py  # Advanced market simulation
├── dashboard_api/           # Backend API
│   └── src/main.py         # Flask API with WebSocket support
├── dashboard_frontend/      # React dashboard
│   └── src/                # Modern trading interface
├── tests/                   # Comprehensive test suite
├── config/                  # Configuration files
└── docs/                   # Documentation
```

## 📦 Installation

### Prerequisites
- Python 3.11+
- Node.js 20+
- Git

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/chimera-trading-system.git
cd chimera-trading-system
```

### Step 2: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Setup Backend API
```bash
cd dashboard_api
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 4: Setup Frontend Dashboard
```bash
cd ../dashboard_frontend
npm install  # or: pnpm install
```

## 🎮 Usage

### Manual Startup

#### Terminal 1: Backend API
```bash
cd dashboard_api
source venv/bin/activate
python src/main.py
```

#### Terminal 2: Frontend Dashboard
```bash
cd dashboard_frontend
npm run dev --host
```

### Automated Startup
```bash
./start_complete_system.sh
```

### API Commands
```bash
# Start trading system
curl -X POST http://localhost:5000/api/system/start

# Check system status
curl http://localhost:5000/api/system/status

# Get trading data
curl http://localhost:5000/api/trading/data

# Stop trading system
curl -X POST http://localhost:5000/api/system/stop
```

## 🔌 Live Data Integration

The system supports multiple live data sources through a standardized interface.

### Interactive Brokers (IBKR)

#### Installation
```bash
pip install ib-insync ibapi
```

#### Configuration
```python
# In data/ibkr_adapter.py
from data.data_interface import DataSourceInterface
import ib_insync as ib

class IBKRAdapter(DataSourceInterface):
    def __init__(self, host='127.0.0.1', port=7497, client_id=1):
        self.ib = ib.IB()
        self.ib.connect(host, port, clientId=client_id)
    
    def subscribe_level2(self, symbol: str, callback):
        contract = ib.Stock(symbol, 'SMART', 'USD')
        self.ib.reqMktDepth(contract, 10)
        # Implementation details...
```

#### Setup
1. Install TWS or IB Gateway
2. Enable API connections (Configure → API → Settings)
3. Set socket port to 7497 (paper) or 7496 (live)
4. Update configuration:
```python
# In main.py
from data.ibkr_adapter import IBKRAdapter
data_source = IBKRAdapter(host='127.0.0.1', port=7497)
```

### DXFeed Integration

#### Installation
```bash
pip install dxfeed
```

#### Configuration
```python
# In data/dxfeed_adapter.py
from data.data_interface import DataSourceInterface
import dxfeed as dx

class DXFeedAdapter(DataSourceInterface):
    def __init__(self, endpoint='demo.dxfeed.com:7300'):
        self.endpoint = endpoint
        self.subscription = dx.create_subscription('Quote', 'Trade')
    
    def subscribe_level2(self, symbol: str, callback):
        self.subscription.add_symbols([symbol])
        # Implementation details...
```

#### Setup
1. Get DXFeed credentials from https://dxfeed.com
2. Set environment variables:
```bash
export DXFEED_USER="your_username"
export DXFEED_PASSWORD="your_password"
```

### MooMoo (Futu) Integration

#### Installation
```bash
pip install futu-api
```

#### Configuration
```python
# In data/moomoo_adapter.py
from data.data_interface import DataSourceInterface
from futu import OpenQuoteContext, RET_OK

class MooMooAdapter(DataSourceInterface):
    def __init__(self, host='127.0.0.1', port=11111):
        self.quote_ctx = OpenQuoteContext(host=host, port=port)
    
    def subscribe_level2(self, symbol: str, callback):
        ret, data = self.quote_ctx.subscribe(symbol, ['ORDER_BOOK'])
        # Implementation details...
```

#### Setup
1. Download MooMoo OpenAPI from https://www.futunn.com/download/openAPI
2. Install and configure the client
3. Update configuration with your credentials

### Yahoo Finance (Free Alternative)

#### Installation
```bash
pip install yfinance
```

#### Configuration
```python
# In data/yahoo_adapter.py
from data.data_interface import DataSourceInterface
import yfinance as yf

class YahooAdapter(DataSourceInterface):
    def subscribe_level2(self, symbol: str, callback):
        ticker = yf.Ticker(symbol)
        # Note: Yahoo Finance doesn't provide Level 2 data
        # This adapter provides basic OHLCV data
```

### Switching Data Sources

```python
# In main.py or configuration
from data.mock_adapter import MockAdapter
from data.ibkr_adapter import IBKRAdapter
from data.dxfeed_adapter import DXFeedAdapter

# Choose your data source
# data_source = MockAdapter()  # For testing
# data_source = IBKRAdapter()  # For IBKR
data_source = DXFeedAdapter()  # For DXFeed

# Register with the system
data_orchestrator.set_data_source(data_source)
```

## 📚 API Documentation

### System Endpoints

#### GET /api/system/status
Returns current system status and statistics.

**Response:**
```json
{
  "status": "running",
  "uptime": 3600,
  "signals_generated": 42,
  "trades_executed": 15,
  "active_symbols": ["AAPL", "NVDA", "TSLA", "SPY"],
  "performance": {
    "total_pnl": 1250.50,
    "daily_pnl": 125.75,
    "win_rate": 0.68,
    "sharpe_ratio": 1.85
  }
}
```

#### POST /api/system/start
Starts the trading system.

#### POST /api/system/stop
Stops the trading system.

### Trading Data Endpoints

#### GET /api/trading/data
Returns real-time trading data including signals, trades, and order books.

**Response:**
```json
{
  "signals": [
    {
      "symbol": "AAPL",
      "type": "VACUUM_ENTRY",
      "price": 150.25,
      "confidence": 0.85,
      "direction": "LONG",
      "timestamp": 1640995200000
    }
  ],
  "order_books": {
    "AAPL": {
      "bids": [[150.24, 500], [150.23, 300]],
      "asks": [[150.26, 400], [150.27, 600]],
      "mid_price": 150.25,
      "spread": 0.02
    }
  }
}
```

### Performance Endpoints

#### GET /api/performance/metrics
Returns performance metrics and analytics.

#### GET /api/risk/metrics
Returns risk management metrics.

### WebSocket Events

The system provides real-time updates via WebSocket:

```javascript
const socket = io('http://localhost:5000');

socket.on('trading_update', (data) => {
  console.log('New trading data:', data);
});

socket.on('signal_generated', (signal) => {
  console.log('New signal:', signal);
});
```

## 📊 Dashboard Features

### Real-Time Gauges
- **System Health**: Overall system performance (0-100%)
- **Risk Level**: Current risk exposure (0-100%)
- **Win Rate**: Percentage of profitable trades
- **Performance**: Sharpe ratio and other metrics

### Interactive Charts
- **Equity Curve**: Real-time P&L visualization
- **Signal Frequency**: Trading signal distribution over time
- **Order Book Heatmap**: Bookmap-style depth visualization

### Order Book Visualization
- **Depth Ladder**: Real-time bid/ask levels
- **Intensity Colors**: Volume-based color coding
- **Spread Analysis**: Real-time spread monitoring
- **Flow Indicators**: Order flow direction and strength

### Navigation Panels
- **Overview**: System dashboard and key metrics
- **Trading Monitor**: Order book and signal analysis
- **Performance**: Detailed performance analytics
- **Risk Management**: Risk controls and monitoring
- **Configuration**: System settings and parameters

## 🤖 Machine Learning Components

### Market Regime Detector
Classifies market conditions into distinct regimes:
- **Trending**: Strong directional movement
- **Ranging**: Sideways price action
- **Volatile**: High volatility periods

```python
from ml.regime_detector import MarketRegimeDetector

detector = MarketRegimeDetector()
regime = detector.get_current_regime(features)
print(f"Current regime: {regime}")  # TRENDING, RANGING, or VOLATILE
```

### Signal Classifier
ML-based validation of trading signals:
- **Feature Engineering**: Technical indicators and market microstructure
- **Classification**: Signal quality assessment
- **Confidence Scoring**: Probability-based signal strength

```python
from ml.signal_classifier import SignalClassifier

classifier = SignalClassifier()
confidence = classifier.classify_signal(signal, features)
print(f"Signal confidence: {confidence:.2f}")
```

### Reinforcement Learning Exit Agent
Optimizes trade exits using Q-learning:
- **State Representation**: Market conditions and position status
- **Action Space**: Hold, partial exit, full exit
- **Reward Function**: Risk-adjusted returns

```python
from ml.rl_exit_agent import RLExitAgent

agent = RLExitAgent()
action = agent.get_action(position_state, market_features)
```

### Genetic Algorithm Optimizer
Evolves trading parameters automatically:
- **Parameter Optimization**: Strategy parameters and thresholds
- **Multi-Objective**: Balances return, risk, and drawdown
- **Adaptive**: Continuously evolves with market conditions

```python
from ml.genetic_optimizer import GeneticOptimizer

optimizer = GeneticOptimizer()
best_params = optimizer.optimize(strategy, historical_data)
```

## 📈 Trading Strategies

### Liquidity Sweep Detection
Identifies when large orders sweep through multiple price levels:
- **Volume Threshold**: Configurable minimum volume
- **Price Impact**: Measures market impact
- **Time Window**: Sweep detection timeframe

### Absorption Pattern Recognition
Detects when large hidden orders absorb market flow:
- **Order Flow Analysis**: Bid/ask flow imbalance
- **Volume Profile**: Unusual volume at specific levels
- **Price Rejection**: Failed breakout patterns

### Iceberg Order Detection
Identifies large hidden orders being executed in small pieces:
- **Replenishment Patterns**: Order level refilling
- **Size Consistency**: Similar order sizes
- **Time Intervals**: Regular execution timing

### Mean Reversion Strategies
Exploits temporary price dislocations:
- **Statistical Arbitrage**: Price deviation from fair value
- **Bollinger Band**: Overbought/oversold conditions
- **RSI Divergence**: Momentum divergence patterns

## ⚙️ Configuration

### System Configuration
```python
# config/system_config.py
SYSTEM_CONFIG = {
    'max_positions': 10,
    'max_risk_per_trade': 0.02,
    'max_portfolio_risk': 0.10,
    'data_update_interval': 0.1,  # seconds
    'signal_confidence_threshold': 0.6
}
```

### Trading Parameters
```python
# config/trading_config.py
TRADING_CONFIG = {
    'liquidity_sweep': {
        'min_volume': 10000,
        'price_impact_threshold': 0.001,
        'time_window': 5  # seconds
    },
    'absorption': {
        'flow_imbalance_threshold': 0.7,
        'volume_threshold': 5000,
        'rejection_count': 3
    },
    'iceberg': {
        'replenishment_threshold': 0.8,
        'size_consistency': 0.9,
        'time_interval_tolerance': 2  # seconds
    }
}
```

### Risk Management
```python
# config/risk_config.py
RISK_CONFIG = {
    'max_drawdown': 0.15,
    'var_confidence': 0.95,
    'position_sizing': 'kelly',
    'stop_loss_pct': 0.02,
    'take_profit_ratio': 2.0
}
```

## 🔧 Troubleshooting

### Common Issues

#### Port Already in Use
```bash
# Check what's using the port
netstat -tlnp | grep 5000

# Kill the process
sudo lsof -ti:5000 | xargs kill -9
```

#### Python Dependencies
```bash
# Reinstall requirements
pip install --force-reinstall -r requirements.txt

# Check specific packages
pip show flask numpy pandas scikit-learn
```

#### Frontend Build Issues
```bash
# Clear npm cache
npm cache clean --force

# Delete node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

#### Database Issues
```bash
# Reset database
rm -f *.db
python -c "from core.data_orchestrator import DataOrchestrator; DataOrchestrator('trading_system.db')"
```

### Performance Optimization

#### Memory Usage
```bash
# Monitor memory usage
python -c "
import psutil
import os
process = psutil.Process(os.getpid())
print(f'Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB')
"
```

#### CPU Optimization
```bash
# Check CPU usage
top -p $(pgrep -f "python.*main.py")

# Profile performance
python -m cProfile -o profile.prof main.py demo
```

### Logging and Debugging

#### Enable Debug Mode
```python
# In main.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### View System Logs
```bash
# Backend logs
tail -f dashboard_api/logs/system.log

# Frontend logs (browser console)
# Open browser developer tools (F12)
```

## 🤝 Contributing

We welcome contributions to the Chimera Trading System! Please follow these guidelines:

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Add tests for new functionality
5. Run the test suite: `python -m pytest tests/`
6. Commit your changes: `git commit -m 'Add amazing feature'`
7. Push to the branch: `git push origin feature/amazing-feature`
8. Open a Pull Request

### Code Style
- Follow PEP 8 for Python code
- Use ESLint configuration for JavaScript/React
- Add docstrings for all functions and classes
- Include type hints where appropriate

### Testing
- Write unit tests for new features
- Ensure all tests pass before submitting
- Include integration tests for complex features
- Test with both mock and live data when possible

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by professional trading platforms like Bookmap
- Built with modern web technologies and machine learning frameworks
- Thanks to the open-source community for excellent libraries and tools

## 📞 Support

For support, questions, or feature requests:
- Open an issue on GitHub
- Check the documentation in the `docs/` directory
- Review the troubleshooting section above

---

**⚠️ Disclaimer**: This software is for educational and research purposes only. Trading involves substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results. Use at your own risk.

