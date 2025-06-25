"""
Chimera Trading System v2.0 - Core Module
A comprehensive algorithmic trading system with ML capabilities
"""

__version__ = "2.0.0"
__author__ = "Chimera Trading System"

from .order_book import OrderBook, OrderBookLevel
from .data_orchestrator import DataOrchestrator, SubscriptionConfig
from .feature_engine import FeatureEngine, MarketFeatures
from .signal_detector import SignalDetector, TradingSignal

__all__ = [
    'OrderBook',
    'OrderBookLevel', 
    'DataOrchestrator',
    'SubscriptionConfig',
    'FeatureEngine',
    'MarketFeatures',
    'SignalDetector',
    'TradingSignal'
]

