"""
Chimera Trading System v2.0 - Machine Learning Module
Advanced ML components for trading signal enhancement
"""

from .regime_detector import MarketRegimeDetector, RegimeState
from .signal_classifier import SignalClassifier, ModelPerformance
from .rl_exit_agent import RLExitAgent, ExitAction
from .genetic_optimizer import GeneticOptimizer, StrategyGenome

__all__ = [
    'MarketRegimeDetector',
    'RegimeState',
    'SignalClassifier', 
    'ModelPerformance',
    'RLExitAgent',
    'ExitAction',
    'GeneticOptimizer',
    'StrategyGenome'
]

