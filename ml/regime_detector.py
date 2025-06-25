"""
Market Regime Detection System
Classifies market environment for adaptive trading strategies
"""

import time
import math
import statistics
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import threading
import pickle
import os

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

from core.feature_engine import MarketFeatures


class RegimeState(Enum):
    """Market regime states"""
    HIGH_VOLATILITY_TRENDING = "high_volatility_trending"
    LOW_VOLATILITY_RANGEBOUND = "low_volatility_rangebound"
    POST_NEWS_DRIFT = "post_news_drift"
    RISK_OFF_CONTRACTION = "risk_off_contraction"
    MOMENTUM_BREAKOUT = "momentum_breakout"
    MEAN_REVERSION = "mean_reversion"


@dataclass
class RegimeFeatures:
    """Features for regime classification"""
    volatility_short: float
    volatility_long: float
    volatility_ratio: float
    trend_strength: float
    volume_acceleration: float
    price_momentum: float
    correlation_breakdown: float
    vix_level: float
    vix_change: float
    spread_widening: float
    flow_imbalance_persistence: float
    absorption_frequency: float


@dataclass
class RegimeClassification:
    """Result of regime classification"""
    regime: RegimeState
    confidence: float
    probabilities: Dict[RegimeState, float]
    timestamp: int
    features: RegimeFeatures


class MarketRegimeDetector:
    """
    Advanced market regime detection using machine learning
    """
    
    def __init__(self, lookback_periods: Dict[str, int] = None):
        if lookback_periods is None:
            lookback_periods = {
                'short': 50,
                'medium': 200,
                'long': 500
            }
        
        self.lookback_periods = lookback_periods
        
        # ML Model
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Feature history
        self._feature_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.lookback_periods['long'])
        )
        
        # Regime history
        self._regime_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=100)
        )
        
        # Macro indicators (simulated)
        self._macro_indicators = {
            'vix': deque(maxlen=100),
            'yield_curve': deque(maxlen=100),
            'dollar_index': deque(maxlen=100)
        }
        
        # Model persistence
        self.model_path = "regime_model.pkl"
        self.scaler_path = "regime_scaler.pkl"
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Load existing model if available
        self._load_model()
    
    def update_features(self, symbol: str, features: MarketFeatures) -> RegimeClassification:
        """Update with new features and classify regime"""
        with self._lock:
            # Store features
            self._feature_history[symbol].append(features)
            
            # Update macro indicators (simulated)
            self._update_macro_indicators(features)
            
            # Calculate regime features
            regime_features = self._calculate_regime_features(symbol, features)
            
            # Classify regime
            classification = self._classify_regime(regime_features)
            
            # Store classification
            self._regime_history[symbol].append(classification)
            
            return classification
    
    def _update_macro_indicators(self, features: MarketFeatures) -> None:
        """Update simulated macro indicators"""
        # Simulate VIX based on volatility
        vix_value = min(100, max(10, features.realized_volatility * 100 + 20))
        self._macro_indicators['vix'].append(vix_value)
        
        # Simulate yield curve (simplified)
        yield_curve = 2.0 + features.vwap_deviation * 10
        self._macro_indicators['yield_curve'].append(yield_curve)
        
        # Simulate dollar index
        dollar_index = 100 + features.flow_imbalance * 5
        self._macro_indicators['dollar_index'].append(dollar_index)
    
    def _calculate_regime_features(self, symbol: str, current_features: MarketFeatures) -> RegimeFeatures:
        """Calculate features for regime classification"""
        history = list(self._feature_history[symbol])
        
        if len(history) < self.lookback_periods['short']:
            # Not enough history, return default features
            return self._get_default_regime_features()
        
        # Calculate volatility measures
        short_volatilities = [f.realized_volatility for f in history[-self.lookback_periods['short']:]]
        long_volatilities = [f.realized_volatility for f in history[-self.lookback_periods['long']:] if f.realized_volatility > 0]
        
        volatility_short = statistics.mean(short_volatilities) if short_volatilities else 0.0
        volatility_long = statistics.mean(long_volatilities) if long_volatilities else 0.0
        volatility_ratio = volatility_short / volatility_long if volatility_long > 0 else 1.0
        
        # Calculate trend strength
        prices = [f.mid_price for f in history[-self.lookback_periods['medium']:] if f.mid_price > 0]
        if len(prices) >= 2:
            price_changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
            positive_changes = sum(1 for change in price_changes if change > 0)
            trend_strength = (positive_changes / len(price_changes)) * 2 - 1  # Scale to [-1, 1]
        else:
            trend_strength = 0.0
        
        # Calculate volume acceleration
        volumes = [f.aggressive_buy_volume + f.aggressive_sell_volume for f in history[-20:]]
        if len(volumes) >= 10:
            recent_volume = statistics.mean(volumes[-5:])
            earlier_volume = statistics.mean(volumes[:5])
            volume_acceleration = (recent_volume / earlier_volume - 1) if earlier_volume > 0 else 0.0
        else:
            volume_acceleration = 0.0
        
        # Calculate price momentum
        if len(prices) >= 10:
            price_momentum = (prices[-1] - prices[-10]) / prices[-10] if prices[-10] > 0 else 0.0
        else:
            price_momentum = 0.0
        
        # Calculate correlation breakdown (simplified)
        flow_imbalances = [f.flow_imbalance for f in history[-50:]]
        if len(flow_imbalances) >= 10:
            correlation_breakdown = statistics.stdev(flow_imbalances)
        else:
            correlation_breakdown = 0.0
        
        # Get macro indicators
        vix_values = list(self._macro_indicators['vix'])
        vix_level = vix_values[-1] if vix_values else 20.0
        vix_change = (vix_values[-1] - vix_values[-5]) if len(vix_values) >= 5 else 0.0
        
        # Calculate spread widening
        spreads = [f.spread_bps for f in history[-20:] if f.spread_bps > 0]
        if len(spreads) >= 10:
            recent_spread = statistics.mean(spreads[-5:])
            earlier_spread = statistics.mean(spreads[:5])
            spread_widening = (recent_spread / earlier_spread - 1) if earlier_spread > 0 else 0.0
        else:
            spread_widening = 0.0
        
        # Calculate flow imbalance persistence
        flow_imbalances_recent = [f.flow_imbalance for f in history[-10:]]
        if flow_imbalances_recent:
            # Check if flow imbalance maintains same direction
            positive_count = sum(1 for x in flow_imbalances_recent if x > 0)
            flow_imbalance_persistence = abs(positive_count / len(flow_imbalances_recent) * 2 - 1)
        else:
            flow_imbalance_persistence = 0.0
        
        # Calculate absorption frequency
        absorption_events = [f.absorption_strength for f in history[-50:] if f.absorption_strength > 0.5]
        absorption_frequency = len(absorption_events) / 50 if len(history) >= 50 else 0.0
        
        return RegimeFeatures(
            volatility_short=volatility_short,
            volatility_long=volatility_long,
            volatility_ratio=volatility_ratio,
            trend_strength=trend_strength,
            volume_acceleration=volume_acceleration,
            price_momentum=price_momentum,
            correlation_breakdown=correlation_breakdown,
            vix_level=vix_level,
            vix_change=vix_change,
            spread_widening=spread_widening,
            flow_imbalance_persistence=flow_imbalance_persistence,
            absorption_frequency=absorption_frequency
        )
    
    def _get_default_regime_features(self) -> RegimeFeatures:
        """Get default regime features when insufficient history"""
        return RegimeFeatures(
            volatility_short=0.01,
            volatility_long=0.01,
            volatility_ratio=1.0,
            trend_strength=0.0,
            volume_acceleration=0.0,
            price_momentum=0.0,
            correlation_breakdown=0.0,
            vix_level=20.0,
            vix_change=0.0,
            spread_widening=0.0,
            flow_imbalance_persistence=0.0,
            absorption_frequency=0.0
        )
    
    def _classify_regime(self, regime_features: RegimeFeatures) -> RegimeClassification:
        """Classify market regime using ML model or rules"""
        timestamp = int(time.time() * 1000000)
        
        if self.is_trained:
            # Use ML model
            feature_vector = self._features_to_vector(regime_features)
            feature_vector_scaled = self.scaler.transform([feature_vector])
            
            # Get probabilities for each regime
            probabilities = self.model.predict_proba(feature_vector_scaled)[0]
            regime_classes = self.model.classes_
            
            # Create probability dictionary
            prob_dict = {}
            for i, regime_class in enumerate(regime_classes):
                regime_state = RegimeState(regime_class)
                prob_dict[regime_state] = probabilities[i]
            
            # Get best regime
            best_regime_idx = np.argmax(probabilities)
            best_regime = RegimeState(regime_classes[best_regime_idx])
            confidence = probabilities[best_regime_idx]
            
        else:
            # Use rule-based classification
            best_regime, confidence, prob_dict = self._rule_based_classification(regime_features)
        
        return RegimeClassification(
            regime=best_regime,
            confidence=confidence,
            probabilities=prob_dict,
            timestamp=timestamp,
            features=regime_features
        )
    
    def _rule_based_classification(self, features: RegimeFeatures) -> Tuple[RegimeState, float, Dict[RegimeState, float]]:
        """Rule-based regime classification as fallback"""
        scores = {}
        
        # High Volatility Trending
        vol_score = min(1.0, features.volatility_ratio * 2)
        trend_score = abs(features.trend_strength)
        scores[RegimeState.HIGH_VOLATILITY_TRENDING] = (vol_score * 0.6 + trend_score * 0.4)
        
        # Low Volatility Range Bound
        low_vol_score = max(0, 1 - features.volatility_ratio)
        range_score = max(0, 1 - abs(features.trend_strength))
        absorption_score = features.absorption_frequency
        scores[RegimeState.LOW_VOLATILITY_RANGEBOUND] = (low_vol_score * 0.4 + range_score * 0.4 + absorption_score * 0.2)
        
        # Post News Drift
        vol_change_score = min(1.0, abs(features.vix_change) / 5)
        momentum_score = min(1.0, abs(features.price_momentum) * 10)
        scores[RegimeState.POST_NEWS_DRIFT] = (vol_change_score * 0.5 + momentum_score * 0.5)
        
        # Risk Off Contraction
        vix_score = min(1.0, max(0, (features.vix_level - 20) / 30))
        spread_score = min(1.0, max(0, features.spread_widening * 5))
        scores[RegimeState.RISK_OFF_CONTRACTION] = (vix_score * 0.6 + spread_score * 0.4)
        
        # Momentum Breakout
        momentum_score = min(1.0, abs(features.price_momentum) * 20)
        volume_score = min(1.0, max(0, features.volume_acceleration))
        persistence_score = features.flow_imbalance_persistence
        scores[RegimeState.MOMENTUM_BREAKOUT] = (momentum_score * 0.4 + volume_score * 0.3 + persistence_score * 0.3)
        
        # Mean Reversion
        reversion_score = max(0, 1 - abs(features.trend_strength))
        absorption_score = features.absorption_frequency
        scores[RegimeState.MEAN_REVERSION] = (reversion_score * 0.6 + absorption_score * 0.4)
        
        # Normalize scores to probabilities
        total_score = sum(scores.values())
        if total_score > 0:
            probabilities = {regime: score / total_score for regime, score in scores.items()}
        else:
            # Default equal probabilities
            probabilities = {regime: 1.0 / len(RegimeState) for regime in RegimeState}
        
        # Get best regime
        best_regime = max(probabilities.items(), key=lambda x: x[1])[0]
        confidence = probabilities[best_regime]
        
        return best_regime, confidence, probabilities
    
    def train_model(self, training_data: List[Tuple[RegimeFeatures, RegimeState]]) -> Dict[str, float]:
        """Train the regime classification model"""
        if len(training_data) < 50:
            print("Insufficient training data for regime model")
            return {'error': 'insufficient_data'}
        
        # Prepare training data
        X = []
        y = []
        
        for features, regime in training_data:
            feature_vector = self._features_to_vector(features)
            X.append(feature_vector)
            y.append(regime.value)
        
        X = np.array(X)
        y = np.array(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        # Calculate training accuracy
        train_predictions = self.model.predict(X_scaled)
        accuracy = np.mean(train_predictions == y)
        
        # Save model
        self._save_model()
        
        return {
            'accuracy': accuracy,
            'training_samples': len(training_data),
            'feature_importance': dict(zip(self._get_feature_names(), self.model.feature_importances_))
        }
    
    def _features_to_vector(self, features: RegimeFeatures) -> List[float]:
        """Convert regime features to feature vector"""
        return [
            features.volatility_short,
            features.volatility_long,
            features.volatility_ratio,
            features.trend_strength,
            features.volume_acceleration,
            features.price_momentum,
            features.correlation_breakdown,
            features.vix_level,
            features.vix_change,
            features.spread_widening,
            features.flow_imbalance_persistence,
            features.absorption_frequency
        ]
    
    def _get_feature_names(self) -> List[str]:
        """Get feature names for model interpretation"""
        return [
            'volatility_short',
            'volatility_long',
            'volatility_ratio',
            'trend_strength',
            'volume_acceleration',
            'price_momentum',
            'correlation_breakdown',
            'vix_level',
            'vix_change',
            'spread_widening',
            'flow_imbalance_persistence',
            'absorption_frequency'
        ]
    
    def _save_model(self) -> None:
        """Save trained model to disk"""
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
                
        except Exception as e:
            print(f"Error saving regime model: {e}")
    
    def _load_model(self) -> None:
        """Load trained model from disk"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                
                self.is_trained = True
                print("Regime detection model loaded successfully")
                
        except Exception as e:
            print(f"Error loading regime model: {e}")
            self.is_trained = False
    
    def get_current_regime(self, symbol: str) -> Optional[RegimeClassification]:
        """Get current regime for symbol"""
        with self._lock:
            history = list(self._regime_history[symbol])
            return history[-1] if history else None
    
    def get_regime_history(self, symbol: str, count: int = 50) -> List[RegimeClassification]:
        """Get regime history for symbol"""
        with self._lock:
            history = list(self._regime_history[symbol])
            return history[-count:] if count else history
    
    def get_regime_stability(self, symbol: str, lookback: int = 10) -> float:
        """Calculate regime stability (how often regime changes)"""
        with self._lock:
            history = list(self._regime_history[symbol])
            
            if len(history) < lookback:
                return 1.0  # Assume stable if insufficient data
            
            recent_regimes = [classification.regime for classification in history[-lookback:]]
            unique_regimes = len(set(recent_regimes))
            
            # Stability is inverse of regime diversity
            stability = 1.0 - (unique_regimes - 1) / (lookback - 1) if lookback > 1 else 1.0
            return max(0.0, stability)
    
    def get_statistics(self, symbol: str) -> Dict[str, Any]:
        """Get regime detector statistics"""
        with self._lock:
            current_regime = self.get_current_regime(symbol)
            stability = self.get_regime_stability(symbol)
            
            return {
                'symbol': symbol,
                'is_trained': self.is_trained,
                'current_regime': current_regime.regime.value if current_regime else None,
                'current_confidence': current_regime.confidence if current_regime else 0.0,
                'regime_stability': stability,
                'feature_history_length': len(self._feature_history[symbol]),
                'regime_history_length': len(self._regime_history[symbol])
            }

