"""
Chimera Trading System v2.0
Advanced Algorithmic Trading System with Machine Learning Enhancement

A comprehensive trading system that combines:
- Real-time market data processing
- Advanced order book analysis
- Machine learning signal classification
- Market regime detection
- Reinforcement learning exit strategies
- Genetic algorithm optimization

Usage:
    python main.py run --symbol AAPL --duration 300
    python main.py test
    python main.py demo
"""

__version__ = "2.0.0"
__author__ = "Chimera Trading Systems"
__description__ = "Advanced Algorithmic Trading System with ML Enhancement"

# Import main components for easy access
from .core import (
    OrderBook,
    DataOrchestrator, 
    FeatureEngine,
    SignalDetector,
    SubscriptionConfig
)

from .data import (
    MockDataGenerator,
    MarketSimulator,
    SimulationConfig
)

from .ml import (
    MarketRegimeDetector,
    SignalClassifier,
    RLExitAgent,
    GeneticOptimizer
)

__all__ = [
    # Core components
    'OrderBook',
    'DataOrchestrator',
    'FeatureEngine', 
    'SignalDetector',
    'SubscriptionConfig',
    
    # Data components
    'MockDataGenerator',
    'MarketSimulator',
    'SimulationConfig',
    
    # ML components
    'MarketRegimeDetector',
    'SignalClassifier',
    'RLExitAgent',
    'GeneticOptimizer'
]

