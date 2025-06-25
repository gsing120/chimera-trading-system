# Chimera Trading System v2.0 - Complete Documentation

## 🎯 Overview

The Chimera Trading System v2.0 is a state-of-the-art algorithmic trading platform that combines advanced machine learning, real-time market data processing, and professional-grade visualization. The system includes:

### 🏗️ Core Trading Engine
- **Multi-instrument order book management** with Level 2 data processing
- **Advanced signal detection** using Bookmap-inspired strategies
- **Machine learning components** including regime detection and signal classification
- **Risk management** with real-time monitoring and controls
- **Mock data generation** for testing and development

### 📊 Professional Dashboard
- **Real-time visualization** with dark theme and modern UI
- **Order book heatmap** (Bookmap-style) with intensity-based coloring
- **Performance gauges** showing system health, risk, win rate, and Sharpe ratio
- **Interactive charts** for equity curve and signal frequency
- **WebSocket integration** for real-time data streaming

## 🚀 Quick Start

### Single Command Startup
```bash
cd chimera_trading_system
./start_complete_system.sh
```

This will start both the trading system backend and dashboard frontend automatically.

### Manual Startup
If you prefer to start components separately:

1. **Start Backend API:**
```bash
cd dashboard_api
source venv/bin/activate
python src/main.py
```

2. **Start Frontend Dashboard:**
```bash
cd dashboard_frontend
pnpm run dev --host
```

3. **Start Trading System:**
```bash
python main.py demo
```

## 🎮 Using the Dashboard

### Navigation
- **Overview**: Main dashboard with key metrics and gauges
- **Trading Monitor**: Real-time order book and signal feed
- **Performance**: Detailed analytics and strategy performance
- **Risk Management**: Portfolio risk monitoring and controls
- **Configuration**: System settings and parameters

### Key Features

#### 📈 Real-time Gauges
- **System Health**: Overall system status (0-100%)
- **Risk Level**: Current portfolio risk exposure
- **Win Rate**: Trading success percentage
- **Performance**: Sharpe ratio indicator

#### 📊 Order Book Heatmap
- **Bid/Ask Visualization**: Color-coded order intensity
- **Real-time Updates**: Live market data simulation
- **Spread Monitoring**: Current bid-ask spread
- **Mid Price Display**: Real-time price calculation

#### 🎯 Signal Detection
- **Live Signal Feed**: Real-time trading signals
- **Strategy Attribution**: Signal source identification
- **Confidence Levels**: ML-based signal strength
- **Execution Tracking**: Trade execution monitoring

## 🔧 System Architecture

### Backend Components
```
chimera_trading_system/
├── core/                    # Core trading engine
│   ├── order_book.py       # Order book management
│   ├── data_orchestrator.py # Multi-instrument data handling
│   ├── feature_engine.py   # Market feature extraction
│   └── signal_detector.py  # Trading signal generation
├── ml/                     # Machine learning components
│   ├── regime_detector.py  # Market regime classification
│   ├── signal_classifier.py # Signal quality assessment
│   ├── rl_exit_agent.py    # Reinforcement learning exits
│   └── genetic_optimizer.py # Strategy optimization
├── data/                   # Data management
│   ├── mock_data_generator.py # Level 2 data simulation
│   ├── market_simulator.py    # Market condition simulation
│   └── data_interface.py      # External data integration
└── dashboard_api/          # REST API and WebSocket server
    └── src/main.py         # Flask application
```

### Frontend Components
```
dashboard_frontend/
├── src/
│   ├── components/         # React components
│   │   ├── Sidebar.jsx    # Navigation sidebar
│   │   ├── Header.jsx     # Top header with controls
│   │   ├── OverviewDashboard.jsx # Main dashboard
│   │   ├── TradingMonitor.jsx    # Order book visualization
│   │   └── ...            # Other dashboard components
│   └── App.jsx            # Main application
└── public/                # Static assets
```

## 🔌 Data Integration

The system is designed to work with external data sources. See the integration guides:

- **[DATA_INTEGRATION_GUIDE.md](DATA_INTEGRATION_GUIDE.md)** - Complete integration instructions
- **[ENDPOINTS_SPECIFICATION.md](ENDPOINTS_SPECIFICATION.md)** - Data format specifications

### Supported Data Sources
- Interactive Brokers (IBKR)
- Alpaca Markets
- Polygon.io
- Custom data feeds

## 📊 Performance Metrics

### Trading Metrics
- **Total P&L**: Cumulative profit/loss
- **Daily P&L**: Current day performance
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades

### System Metrics
- **Signal Generation Rate**: Signals per hour
- **Execution Latency**: Order processing speed
- **Fill Rate**: Successful order execution percentage
- **System Uptime**: Operational availability

## 🛡️ Risk Management

### Built-in Controls
- **Position Limits**: Maximum position sizes
- **Exposure Limits**: Portfolio concentration limits
- **Loss Limits**: Daily/total loss thresholds
- **VaR Monitoring**: Value at Risk calculations

### Real-time Monitoring
- **Risk Gauges**: Visual risk level indicators
- **Alert System**: Automated risk notifications
- **Emergency Stops**: Automatic system shutdown triggers

## 🧠 Machine Learning Features

### Regime Detection
- **Market State Classification**: Trending, ranging, volatile
- **Adaptive Parameters**: Strategy adjustment based on regime
- **Confidence Scoring**: Regime prediction reliability

### Signal Classification
- **Quality Assessment**: ML-based signal filtering
- **Feature Engineering**: Advanced market microstructure features
- **Ensemble Methods**: Multiple model combination

### Reinforcement Learning
- **Exit Optimization**: RL-based position exit timing
- **Dynamic Risk Adjustment**: Adaptive risk parameters
- **Continuous Learning**: Online model updates

## 🔧 Configuration

### System Parameters
- **Trading Symbols**: Instruments to monitor
- **Risk Limits**: Maximum exposure levels
- **ML Models**: Enable/disable specific models
- **Data Sources**: Configure data providers

### Dashboard Settings
- **Update Frequency**: Real-time refresh rates
- **Display Options**: Chart types and timeframes
- **Alert Preferences**: Notification settings

## 🧪 Testing

### Mock Data Testing
The system includes comprehensive mock data generation for testing:

```bash
# Run integration tests
python main.py test

# Run demo mode
python main.py demo

# Generate mock Level 2 data
python -c "from data.mock_data_generator import MockDataGenerator; MockDataGenerator().generate_sample_data()"
```

### Performance Testing
- **Latency Testing**: Measure system response times
- **Load Testing**: Stress test with high data volumes
- **Accuracy Testing**: Validate signal generation quality

## 📦 Deployment

### Local Development
- Use the provided startup scripts
- All dependencies are included
- No external services required

### Production Deployment
- Deploy backend using Flask production server
- Deploy frontend using static hosting
- Configure real data sources
- Set up monitoring and alerting

## 🔍 Troubleshooting

### Common Issues
1. **Port Conflicts**: Ensure ports 5000 and 5173 are available
2. **Dependencies**: Run `pip install -r requirements.txt` if needed
3. **Data Connection**: Check data source configuration
4. **Performance**: Monitor system resources during operation

### Debug Mode
Enable debug logging:
```bash
export CHIMERA_DEBUG=1
python main.py demo
```

## 📞 Support

### Documentation
- **System Architecture**: [DASHBOARD_ARCHITECTURE.md](dashboard/DASHBOARD_ARCHITECTURE.md)
- **API Reference**: [ENDPOINTS_SPECIFICATION.md](ENDPOINTS_SPECIFICATION.md)
- **Integration Guide**: [DATA_INTEGRATION_GUIDE.md](DATA_INTEGRATION_GUIDE.md)

### Features
- **Self-contained**: No external dependencies required
- **Cross-platform**: Works on Windows, macOS, and Linux
- **Scalable**: Designed for high-frequency trading
- **Extensible**: Easy to add new strategies and data sources

---

**Chimera Trading System v2.0** - Professional algorithmic trading platform with state-of-the-art visualization and machine learning capabilities.

