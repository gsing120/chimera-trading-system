# Chimera Trading System - Real IBKR Integration Analysis

## Current State Analysis

### Simulation Components to Remove:
1. **data/mock_data_generator.py** - Mock Level 2 data generator
2. **data/mock_adapter.py** - Mock data adapter
3. **data/market_simulator.py** - Market simulation engine
4. **Dockerfile.mockdata** - Mock data container
5. All mock data references in main.py

### IBKR Integration Issues Found:
1. **Wrong Port**: Using 7497 (TWS) instead of 4002 (Gateway paper) / 4001 (Gateway live)
2. **Missing Gateway Integration**: No connection to our containerized Gateway
3. **Simulation Dependencies**: System defaults to mock data

### Data Format Requirements:
Based on the code analysis, Chimera expects:

#### Level 2 Data Format:
```python
@dataclass
class Level2Update:
    symbol: str
    bids: List[Tuple[float, int]]  # [(price, size), ...]
    asks: List[Tuple[float, int]]  # [(price, size), ...]
    timestamp: int  # milliseconds
```

#### Trade Data Format:
```python
@dataclass
class TradeData:
    symbol: str
    price: float
    size: int
    timestamp: int
    side: str  # 'BUY' or 'SELL'
```

#### Quote Data Format:
```python
@dataclass
class QuoteData:
    symbol: str
    bid_price: float
    ask_price: float
    bid_size: int
    ask_size: int
    timestamp: int
```

## Required Changes:

### 1. Remove All Simulation Components
- Delete mock data files
- Remove simulation references from main.py
- Update configuration to force IBKR only

### 2. Fix IBKR Configuration
- Change port from 7497 to 4002 (Gateway paper)
- Update connection parameters for Gateway
- Integrate with our containerized Gateway

### 3. Create Unified Container
- Combine IBKR Gateway container with Chimera system
- Ensure proper startup sequence
- Configure networking between components

### 4. Update Environment Configuration
- Remove mock data options
- Set IBKR as only data source
- Configure Gateway connection parameters

## Implementation Plan:

1. **Remove Simulations**: Delete all mock/simulation files
2. **Fix IBKR Adapter**: Update ports and connection logic
3. **Create Unified Dockerfile**: Combine Gateway + Chimera
4. **Update Configuration**: Remove mock options, set IBKR defaults
5. **Test Integration**: Verify real data flow

