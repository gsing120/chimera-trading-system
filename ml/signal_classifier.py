"""
Signal Classification System
ML-based signal probability scoring and validation
"""

import time
import math
import statistics
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import threading
import pickle
import os

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import numpy as np

from core.signal_detector import TradingSignal, SignalType
from core.feature_engine import MarketFeatures
from .regime_detector import RegimeClassification, RegimeState


@dataclass
class ModelPerformance:
    """Model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    cross_val_score: float
    feature_importance: Dict[str, float]
    training_samples: int


@dataclass
class SignalPrediction:
    """ML prediction for a trading signal"""
    signal: TradingSignal
    ml_probability: float
    regime_adjusted_probability: float
    confidence_interval: Tuple[float, float]
    feature_contributions: Dict[str, float]
    timestamp: int


class SignalClassifier:
    """
    ML-based signal classification and probability scoring
    """
    
    def __init__(self):
        # Separate models for each signal type
        self.models: Dict[SignalType, GradientBoostingClassifier] = {}
        self.scalers: Dict[SignalType, StandardScaler] = {}
        self.trained_models: set = set()
        
        # Training data storage
        self.training_data: Dict[SignalType, List[Tuple[List[float], bool]]] = defaultdict(list)
        
        # Performance tracking
        self.performance_metrics: Dict[SignalType, ModelPerformance] = {}
        
        # Signal outcome tracking
        self.signal_outcomes: Dict[str, Dict] = {}  # signal_id -> outcome data
        
        # Model persistence
        self.model_dir = "signal_models"
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize models
        self._initialize_models()
        
        # Load existing models
        self._load_models()
    
    def _initialize_models(self) -> None:
        """Initialize ML models for each signal type"""
        for signal_type in SignalType:
            self.models[signal_type] = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            self.scalers[signal_type] = StandardScaler()
    
    def predict_signal_probability(self, signal: TradingSignal, 
                                 regime: Optional[RegimeClassification] = None) -> SignalPrediction:
        """Predict probability of signal success"""
        with self._lock:
            # Extract features from signal
            feature_vector = self._extract_signal_features(signal, regime)
            
            # Get base ML probability
            ml_probability = self._get_ml_probability(signal.signal_type, feature_vector)
            
            # Adjust for market regime
            regime_adjusted_probability = self._adjust_for_regime(
                ml_probability, signal.signal_type, regime
            )
            
            # Calculate confidence interval
            confidence_interval = self._calculate_confidence_interval(
                regime_adjusted_probability, signal.signal_type
            )
            
            # Calculate feature contributions
            feature_contributions = self._calculate_feature_contributions(
                signal.signal_type, feature_vector
            )
            
            return SignalPrediction(
                signal=signal,
                ml_probability=ml_probability,
                regime_adjusted_probability=regime_adjusted_probability,
                confidence_interval=confidence_interval,
                feature_contributions=feature_contributions,
                timestamp=int(time.time() * 1000000)
            )
    
    def _extract_signal_features(self, signal: TradingSignal, 
                                regime: Optional[RegimeClassification] = None) -> List[float]:
        """Extract features from signal for ML prediction"""
        features = signal.features
        
        # Base signal features
        feature_vector = [
            signal.confidence,
            signal.strength.value,
            features.spread_bps,
            features.depth_imbalance,
            features.flow_imbalance,
            features.absorption_strength,
            features.iceberg_probability,
            features.sweep_probability,
            features.vwap_deviation,
            features.realized_volatility,
            features.price_acceleration,
            features.order_book_imbalance,
            features.bid_liquidity_density,
            features.ask_liquidity_density,
            features.hvn_distance,
            features.lvn_distance
        ]
        
        # Add regime features if available
        if regime:
            regime_features = [
                regime.confidence,
                regime.features.volatility_ratio,
                regime.features.trend_strength,
                regime.features.volume_acceleration,
                regime.features.vix_level,
                regime.features.flow_imbalance_persistence
            ]
            feature_vector.extend(regime_features)
        else:
            # Add default regime features
            feature_vector.extend([0.5, 1.0, 0.0, 0.0, 20.0, 0.0])
        
        # Add signal-specific features
        signal_specific = self._get_signal_specific_features(signal)
        feature_vector.extend(signal_specific)
        
        return feature_vector
    
    def _get_signal_specific_features(self, signal: TradingSignal) -> List[float]:
        """Get signal-type specific features"""
        metadata = signal.metadata
        
        if signal.signal_type == SignalType.LIQUIDITY_SWEEP_REVERSAL:
            return [
                metadata.get('sweep_volume', 0) / 10000,  # Normalized
                metadata.get('trapped_volume', 0) / 10000,
                metadata.get('absorption_events', 0) / 10
            ]
        
        elif signal.signal_type == SignalType.STACKED_ABSORPTION_REVERSAL:
            return [
                metadata.get('absorption_count', 0) / 10,
                metadata.get('total_volume_absorbed', 0) / 10000,
                0.0  # Placeholder
            ]
        
        elif signal.signal_type == SignalType.ICEBERG_DEFENSE_ENTRY:
            return [
                metadata.get('replenishment_count', 0) / 20,
                metadata.get('avg_iceberg_size', 0) / 1000,
                0.0  # Placeholder
            ]
        
        elif signal.signal_type == SignalType.VACUUM_ENTRY:
            return [
                metadata.get('volume_acceleration', 0) / 5000,
                signal.features.lvn_distance * 1000,  # Scale up
                abs(signal.features.flow_imbalance)
            ]
        
        elif signal.signal_type == SignalType.MEAN_REVERSION_FADE:
            return [
                abs(metadata.get('vwap_deviation', 0)) * 100,
                metadata.get('absorption_strength', 0),
                0.0  # Placeholder
            ]
        
        else:
            # Default features for other signal types
            return [0.0, 0.0, 0.0]
    
    def _get_ml_probability(self, signal_type: SignalType, feature_vector: List[float]) -> float:
        """Get ML model probability for signal type"""
        if signal_type not in self.trained_models:
            # Use rule-based probability if model not trained
            return self._rule_based_probability(signal_type, feature_vector)
        
        try:
            # Scale features
            feature_array = np.array([feature_vector])
            scaled_features = self.scalers[signal_type].transform(feature_array)
            
            # Get probability
            probabilities = self.models[signal_type].predict_proba(scaled_features)[0]
            
            # Return probability of success (class 1)
            if len(probabilities) > 1:
                return probabilities[1]
            else:
                return probabilities[0]
                
        except Exception as e:
            print(f"Error in ML prediction for {signal_type}: {e}")
            return self._rule_based_probability(signal_type, feature_vector)
    
    def _rule_based_probability(self, signal_type: SignalType, feature_vector: List[float]) -> float:
        """Rule-based probability as fallback"""
        # Extract key features (first 16 are base features)
        if len(feature_vector) < 16:
            return 0.5
        
        confidence = feature_vector[0]
        strength = feature_vector[1]
        flow_imbalance = abs(feature_vector[4])
        absorption_strength = feature_vector[5]
        volatility = feature_vector[9]
        
        # Base probability from signal confidence and strength
        base_prob = (confidence * 0.6 + (strength / 4) * 0.4)
        
        # Adjust based on signal type
        if signal_type == SignalType.LIQUIDITY_SWEEP_REVERSAL:
            # Higher probability with strong absorption after sweep
            prob = base_prob * (1 + absorption_strength * 0.3)
        
        elif signal_type == SignalType.STACKED_ABSORPTION_REVERSAL:
            # Higher probability with strong absorption
            prob = base_prob * (1 + absorption_strength * 0.5)
        
        elif signal_type == SignalType.ICEBERG_DEFENSE_ENTRY:
            # Moderate probability, depends on market conditions
            prob = base_prob * 0.8
        
        elif signal_type == SignalType.VACUUM_ENTRY:
            # Higher probability with strong flow imbalance
            prob = base_prob * (1 + flow_imbalance * 0.4)
        
        elif signal_type == SignalType.MEAN_REVERSION_FADE:
            # Higher probability in low volatility
            vol_factor = max(0.5, 1 - volatility * 10)
            prob = base_prob * vol_factor
        
        else:
            prob = base_prob
        
        return min(0.95, max(0.05, prob))
    
    def _adjust_for_regime(self, base_probability: float, signal_type: SignalType,
                          regime: Optional[RegimeClassification]) -> float:
        """Adjust probability based on market regime"""
        if not regime:
            return base_probability
        
        regime_state = regime.regime
        regime_confidence = regime.confidence
        
        # Regime adjustment factors
        adjustments = {
            SignalType.LIQUIDITY_SWEEP_REVERSAL: {
                RegimeState.HIGH_VOLATILITY_TRENDING: 1.2,
                RegimeState.LOW_VOLATILITY_RANGEBOUND: 0.8,
                RegimeState.POST_NEWS_DRIFT: 1.1,
                RegimeState.RISK_OFF_CONTRACTION: 0.9,
                RegimeState.MOMENTUM_BREAKOUT: 0.7,
                RegimeState.MEAN_REVERSION: 1.3
            },
            SignalType.STACKED_ABSORPTION_REVERSAL: {
                RegimeState.HIGH_VOLATILITY_TRENDING: 0.9,
                RegimeState.LOW_VOLATILITY_RANGEBOUND: 1.3,
                RegimeState.POST_NEWS_DRIFT: 0.8,
                RegimeState.RISK_OFF_CONTRACTION: 1.1,
                RegimeState.MOMENTUM_BREAKOUT: 0.7,
                RegimeState.MEAN_REVERSION: 1.4
            },
            SignalType.VACUUM_ENTRY: {
                RegimeState.HIGH_VOLATILITY_TRENDING: 1.3,
                RegimeState.LOW_VOLATILITY_RANGEBOUND: 0.7,
                RegimeState.POST_NEWS_DRIFT: 1.2,
                RegimeState.RISK_OFF_CONTRACTION: 0.8,
                RegimeState.MOMENTUM_BREAKOUT: 1.4,
                RegimeState.MEAN_REVERSION: 0.6
            },
            SignalType.MEAN_REVERSION_FADE: {
                RegimeState.HIGH_VOLATILITY_TRENDING: 0.8,
                RegimeState.LOW_VOLATILITY_RANGEBOUND: 1.2,
                RegimeState.POST_NEWS_DRIFT: 0.9,
                RegimeState.RISK_OFF_CONTRACTION: 1.1,
                RegimeState.MOMENTUM_BREAKOUT: 0.6,
                RegimeState.MEAN_REVERSION: 1.5
            }
        }
        
        # Get adjustment factor
        signal_adjustments = adjustments.get(signal_type, {})
        adjustment_factor = signal_adjustments.get(regime_state, 1.0)
        
        # Weight adjustment by regime confidence
        weighted_adjustment = 1.0 + (adjustment_factor - 1.0) * regime_confidence
        
        # Apply adjustment
        adjusted_probability = base_probability * weighted_adjustment
        
        return min(0.95, max(0.05, adjusted_probability))
    
    def _calculate_confidence_interval(self, probability: float, 
                                     signal_type: SignalType) -> Tuple[float, float]:
        """Calculate confidence interval for probability"""
        # Use model performance to estimate uncertainty
        if signal_type in self.performance_metrics:
            accuracy = self.performance_metrics[signal_type].accuracy
            uncertainty = (1 - accuracy) * 0.5  # Scale uncertainty
        else:
            uncertainty = 0.2  # Default uncertainty
        
        lower_bound = max(0.0, probability - uncertainty)
        upper_bound = min(1.0, probability + uncertainty)
        
        return (lower_bound, upper_bound)
    
    def _calculate_feature_contributions(self, signal_type: SignalType,
                                       feature_vector: List[float]) -> Dict[str, float]:
        """Calculate feature contributions to prediction"""
        if signal_type not in self.trained_models:
            return {}
        
        try:
            # Get feature importance from trained model
            feature_importance = self.models[signal_type].feature_importances_
            feature_names = self._get_feature_names()
            
            # Calculate contributions (importance * feature_value)
            contributions = {}
            for i, (name, importance) in enumerate(zip(feature_names, feature_importance)):
                if i < len(feature_vector):
                    contributions[name] = importance * abs(feature_vector[i])
            
            return contributions
            
        except Exception as e:
            print(f"Error calculating feature contributions: {e}")
            return {}
    
    def _get_feature_names(self) -> List[str]:
        """Get feature names for interpretation"""
        return [
            'signal_confidence',
            'signal_strength',
            'spread_bps',
            'depth_imbalance',
            'flow_imbalance',
            'absorption_strength',
            'iceberg_probability',
            'sweep_probability',
            'vwap_deviation',
            'realized_volatility',
            'price_acceleration',
            'order_book_imbalance',
            'bid_liquidity_density',
            'ask_liquidity_density',
            'hvn_distance',
            'lvn_distance',
            'regime_confidence',
            'volatility_ratio',
            'trend_strength',
            'volume_acceleration',
            'vix_level',
            'flow_persistence',
            'signal_specific_1',
            'signal_specific_2',
            'signal_specific_3'
        ]
    
    def add_training_sample(self, signal: TradingSignal, outcome: bool,
                           regime: Optional[RegimeClassification] = None) -> None:
        """Add training sample for model improvement"""
        with self._lock:
            feature_vector = self._extract_signal_features(signal, regime)
            self.training_data[signal.signal_type].append((feature_vector, outcome))
            
            # Store signal outcome for tracking
            signal_id = f"{signal.symbol}_{signal.timestamp}"
            self.signal_outcomes[signal_id] = {
                'signal': signal,
                'outcome': outcome,
                'regime': regime,
                'timestamp': int(time.time() * 1000000)
            }
    
    def train_models(self, min_samples: int = 50) -> Dict[SignalType, ModelPerformance]:
        """Train ML models for signal types with sufficient data"""
        results = {}
        
        with self._lock:
            for signal_type, training_samples in self.training_data.items():
                if len(training_samples) >= min_samples:
                    performance = self._train_single_model(signal_type, training_samples)
                    results[signal_type] = performance
                    self.performance_metrics[signal_type] = performance
                    self.trained_models.add(signal_type)
                    
                    # Save model
                    self._save_model(signal_type)
        
        return results
    
    def _train_single_model(self, signal_type: SignalType,
                           training_samples: List[Tuple[List[float], bool]]) -> ModelPerformance:
        """Train a single model for specific signal type"""
        # Prepare data
        X = np.array([sample[0] for sample in training_samples])
        y = np.array([sample[1] for sample in training_samples])
        
        # Scale features
        X_scaled = self.scalers[signal_type].fit_transform(X)
        
        # Train model
        self.models[signal_type].fit(X_scaled, y)
        
        # Calculate performance metrics
        train_predictions = self.models[signal_type].predict(X_scaled)
        accuracy = np.mean(train_predictions == y)
        
        # Calculate precision, recall, F1
        true_positives = np.sum((train_predictions == 1) & (y == 1))
        false_positives = np.sum((train_predictions == 1) & (y == 0))
        false_negatives = np.sum((train_predictions == 0) & (y == 1))
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Cross-validation score
        try:
            cv_scores = cross_val_score(self.models[signal_type], X_scaled, y, cv=5)
            cv_score = np.mean(cv_scores)
        except:
            cv_score = accuracy
        
        # Feature importance
        feature_names = self._get_feature_names()
        feature_importance = dict(zip(feature_names, self.models[signal_type].feature_importances_))
        
        return ModelPerformance(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            cross_val_score=cv_score,
            feature_importance=feature_importance,
            training_samples=len(training_samples)
        )
    
    def _save_model(self, signal_type: SignalType) -> None:
        """Save trained model to disk"""
        try:
            model_path = os.path.join(self.model_dir, f"{signal_type.value}_model.pkl")
            scaler_path = os.path.join(self.model_dir, f"{signal_type.value}_scaler.pkl")
            
            with open(model_path, 'wb') as f:
                pickle.dump(self.models[signal_type], f)
            
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scalers[signal_type], f)
                
        except Exception as e:
            print(f"Error saving model for {signal_type}: {e}")
    
    def _load_models(self) -> None:
        """Load trained models from disk"""
        for signal_type in SignalType:
            try:
                model_path = os.path.join(self.model_dir, f"{signal_type.value}_model.pkl")
                scaler_path = os.path.join(self.model_dir, f"{signal_type.value}_scaler.pkl")
                
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    with open(model_path, 'rb') as f:
                        self.models[signal_type] = pickle.load(f)
                    
                    with open(scaler_path, 'rb') as f:
                        self.scalers[signal_type] = pickle.load(f)
                    
                    self.trained_models.add(signal_type)
                    
            except Exception as e:
                print(f"Error loading model for {signal_type}: {e}")
    
    def get_model_performance(self, signal_type: SignalType) -> Optional[ModelPerformance]:
        """Get performance metrics for specific signal type"""
        return self.performance_metrics.get(signal_type)
    
    def get_training_data_stats(self) -> Dict[SignalType, Dict[str, int]]:
        """Get training data statistics"""
        stats = {}
        
        with self._lock:
            for signal_type, samples in self.training_data.items():
                positive_samples = sum(1 for _, outcome in samples if outcome)
                negative_samples = len(samples) - positive_samples
                
                stats[signal_type] = {
                    'total_samples': len(samples),
                    'positive_samples': positive_samples,
                    'negative_samples': negative_samples,
                    'is_trained': signal_type in self.trained_models
                }
        
        return stats
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get overall classifier statistics"""
        with self._lock:
            return {
                'trained_models': len(self.trained_models),
                'total_signal_types': len(SignalType),
                'total_training_samples': sum(len(samples) for samples in self.training_data.values()),
                'total_signal_outcomes': len(self.signal_outcomes),
                'training_data_stats': self.get_training_data_stats()
            }

