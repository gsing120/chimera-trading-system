# 🚀 Chimera Trading System v2.0 - Deployment Guide

## 📋 Repository Information

**GitHub Repository**: https://github.com/gsing120/chimera-trading-system  
**Clone URL**: `git clone https://github.com/gsing120/chimera-trading-system.git`

## ⚡ Quick Deployment

### 1. Clone and Setup
```bash
git clone https://github.com/gsing120/chimera-trading-system.git
cd chimera-trading-system
```

### 2. One-Command Start
```bash
./start_complete_system.sh
```

### 3. Access System
- **Dashboard**: http://localhost:5173
- **API**: http://localhost:5000

## 🔧 Manual Setup (Alternative)

### Prerequisites
- Python 3.11+
- Node.js 20+
- Git

### Step-by-Step Installation
```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Setup backend
cd dashboard_api
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Setup frontend
cd ../dashboard_frontend
npm install  # or: pnpm install

# 4. Start backend (Terminal 1)
cd ../dashboard_api
source venv/bin/activate
python src/main.py

# 5. Start frontend (Terminal 2)
cd ../dashboard_frontend
npm run dev --host
```

## 🎯 System Verification

### 1. Check System Status
```bash
curl http://localhost:5000/api/system/status
```

### 2. Start Trading System
```bash
curl -X POST http://localhost:5000/api/system/start
```

### 3. Verify Data Flow
```bash
curl http://localhost:5000/api/trading/data
```

### 4. Dashboard Features
- ✅ Real-time gauges updating
- ✅ Live charts showing data
- ✅ WebSocket connection active
- ✅ Order book visualization working

## 🔌 Live Data Integration

### Interactive Brokers (IBKR)
```bash
# Install IBKR dependencies
pip install ib-insync ibapi

# Configure in data/ibkr_adapter.py
# Set TWS/Gateway connection details
```

### DXFeed
```bash
# Install DXFeed dependencies
pip install dxfeed

# Set API credentials
export DXFEED_USER="your_username"
export DXFEED_PASSWORD="your_password"
```

### MooMoo
```bash
# Install MooMoo dependencies
pip install futu-api

# Configure OpenAPI client
# Download from https://www.futunn.com/download/openAPI
```

## 📊 System Features Verified

### ✅ Core Trading Engine
- [x] Mock data generation working
- [x] Order book management functional
- [x] Feature engineering extracting 25+ features
- [x] Signal detection identifying multiple patterns
- [x] ML components operational

### ✅ Dashboard Interface
- [x] Real-time gauges (System Health, Risk, Win Rate, Performance)
- [x] Interactive charts (Equity Curve, Signal Frequency)
- [x] Order book visualization ready
- [x] WebSocket streaming active
- [x] Dark theme professional interface

### ✅ API Endpoints
- [x] System status and control
- [x] Trading data retrieval
- [x] Performance metrics
- [x] Risk management data
- [x] WebSocket real-time updates

### ✅ Machine Learning Pipeline
- [x] Market regime detection
- [x] Signal classification
- [x] Reinforcement learning exit agent
- [x] Genetic algorithm optimizer

## 🛡️ Security and Configuration

### Environment Variables
Create `.env` file for sensitive data:
```bash
# IBKR Configuration
IBKR_ACCOUNT=your_account_number
IBKR_HOST=127.0.0.1
IBKR_PORT=7497

# DXFeed Configuration
DXFEED_USER=your_username
DXFEED_PASSWORD=your_password

# MooMoo Configuration
MOOMOO_HOST=127.0.0.1
MOOMOO_PORT=11111
```

### Risk Management Settings
```python
# config/risk_config.py
RISK_CONFIG = {
    'max_drawdown': 0.15,
    'max_position_size': 0.02,
    'max_portfolio_risk': 0.10,
    'stop_loss_pct': 0.02,
    'take_profit_ratio': 2.0
}
```

## 🔍 Troubleshooting

### Common Issues
1. **Port conflicts**: Use `netstat -tlnp | grep 5000` to check
2. **Dependencies**: Run `pip install --force-reinstall -r requirements.txt`
3. **Database**: Delete `*.db` files and restart system
4. **Frontend**: Clear npm cache with `npm cache clean --force`

### Performance Monitoring
```bash
# Check memory usage
python -c "import psutil; print(f'Memory: {psutil.virtual_memory().percent}%')"

# Monitor CPU usage
top -p $(pgrep -f "python.*main.py")
```

## 📈 Production Deployment

### Docker Deployment (Optional)
```dockerfile
# Dockerfile example
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "dashboard_api/src/main.py"]
```

### Cloud Deployment
- **AWS**: Use EC2 with security groups for ports 5000, 5173
- **Google Cloud**: Deploy on Compute Engine
- **Azure**: Use Virtual Machines with proper networking

## 🎯 Next Steps

1. **Configure Live Data**: Choose and setup your preferred data source
2. **Customize Strategies**: Modify signal detection parameters
3. **Risk Management**: Adjust risk settings for your trading style
4. **Backtesting**: Use historical data to validate strategies
5. **Paper Trading**: Test with paper trading before live deployment

## 📞 Support

- **GitHub Issues**: https://github.com/gsing120/chimera-trading-system/issues
- **Documentation**: See README.md and commands.txt
- **Commands Reference**: See commands.txt for all available commands

---

**⚠️ Important**: This system is for educational purposes. Always test thoroughly before using with real money. Trading involves substantial risk of loss.

