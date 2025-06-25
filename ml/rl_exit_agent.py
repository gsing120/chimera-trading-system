"""
Reinforcement Learning Exit Agent
Q-learning based exit strategy optimization
"""

import time
import math
import random
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import threading
import pickle
import os
import numpy as np

from core.feature_engine import MarketFeatures


class ExitAction(Enum):
    """Possible exit actions"""
    HOLD = "hold"
    PARTIAL_EXIT_25 = "partial_exit_25"
    PARTIAL_EXIT_50 = "partial_exit_50"
    PARTIAL_EXIT_75 = "partial_exit_75"
    FULL_EXIT = "full_exit"
    TRAIL_STOP = "trail_stop"
    TIGHTEN_STOP = "tighten_stop"


@dataclass
class PositionState:
    """Current position state for RL agent"""
    symbol: str
    entry_price: float
    current_price: float
    position_size: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    time_in_position: int  # seconds
    max_favorable_excursion: float
    max_adverse_excursion: float
    current_stop_loss: float
    current_take_profit: float
    features: MarketFeatures


@dataclass
class ExitDecision:
    """RL agent exit decision"""
    action: ExitAction
    confidence: float
    expected_reward: float
    position_adjustment: float  # How much to exit (0.0 to 1.0)
    new_stop_loss: Optional[float]
    reasoning: str
    timestamp: int


class RLExitAgent:
    """
    Q-learning based exit strategy agent
    Learns optimal exit timing for different market conditions
    """
    
    def __init__(self, symbol: str, learning_rate: float = 0.1, 
                 discount_factor: float = 0.95, epsilon: float = 0.1):
        self.symbol = symbol
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon  # Exploration rate
        
        # Q-table: state -> action -> Q-value
        self.q_table: Dict[str, Dict[ExitAction, float]] = defaultdict(lambda: defaultdict(float))
        
        # Experience replay buffer
        self.experience_buffer = deque(maxlen=10000)
        
        # State discretization parameters
        self.state_bins = {
            'pnl_pct': [-0.1, -0.05, -0.02, -0.01, 0, 0.01, 0.02, 0.05, 0.1],
            'time_in_position': [0, 60, 300, 900, 1800, 3600],  # seconds
            'volatility': [0, 0.005, 0.01, 0.02, 0.05],
            'trend_strength': [-1, -0.5, -0.1, 0.1, 0.5, 1],
            'flow_imbalance': [-0.5, -0.2, -0.1, 0.1, 0.2, 0.5]
        }
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.action_counts = defaultdict(int)
        self.reward_history = deque(maxlen=1000)
        
        # Model persistence
        self.model_path = f"rl_exit_agent_{symbol}.pkl"
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Load existing model
        self._load_model()
    
    def get_exit_decision(self, position_state: PositionState) -> ExitDecision:
        """Get exit decision for current position state"""
        with self._lock:
            # Discretize state
            state_key = self._discretize_state(position_state)
            
            # Choose action using epsilon-greedy policy
            if random.random() < self.epsilon:
                # Exploration: random action
                action = random.choice(list(ExitAction))
            else:
                # Exploitation: best known action
                action = self._get_best_action(state_key)
            
            # Calculate expected reward
            expected_reward = self.q_table[state_key][action]
            
            # Calculate confidence based on Q-value and experience
            confidence = self._calculate_confidence(state_key, action)
            
            # Determine position adjustment
            position_adjustment = self._get_position_adjustment(action)
            
            # Calculate new stop loss if applicable
            new_stop_loss = self._calculate_new_stop_loss(action, position_state)
            
            # Generate reasoning
            reasoning = self._generate_reasoning(action, position_state, expected_reward)
            
            # Record action
            self.action_counts[action] += 1
            
            return ExitDecision(
                action=action,
                confidence=confidence,
                expected_reward=expected_reward,
                position_adjustment=position_adjustment,
                new_stop_loss=new_stop_loss,
                reasoning=reasoning,
                timestamp=int(time.time() * 1000000)
            )
    
    def _discretize_state(self, position_state: PositionState) -> str:
        """Convert continuous state to discrete state key"""
        # Discretize PnL percentage
        pnl_bin = self._discretize_value(position_state.unrealized_pnl_pct, self.state_bins['pnl_pct'])
        
        # Discretize time in position
        time_bin = self._discretize_value(position_state.time_in_position, self.state_bins['time_in_position'])
        
        # Discretize volatility
        vol_bin = self._discretize_value(position_state.features.realized_volatility, self.state_bins['volatility'])
        
        # Discretize trend strength (using price acceleration as proxy)
        trend_bin = self._discretize_value(position_state.features.price_acceleration, self.state_bins['trend_strength'])
        
        # Discretize flow imbalance
        flow_bin = self._discretize_value(position_state.features.flow_imbalance, self.state_bins['flow_imbalance'])
        
        # Create state key
        state_key = f"{pnl_bin}_{time_bin}_{vol_bin}_{trend_bin}_{flow_bin}"
        
        return state_key
    
    def _discretize_value(self, value: float, bins: List[float]) -> int:
        """Discretize a continuous value into bins"""
        for i, threshold in enumerate(bins):
            if value <= threshold:
                return i
        return len(bins)
    
    def _get_best_action(self, state_key: str) -> ExitAction:
        """Get best action for given state"""
        if state_key not in self.q_table:
            return ExitAction.HOLD  # Default action
        
        q_values = self.q_table[state_key]
        if not q_values:
            return ExitAction.HOLD
        
        return max(q_values.items(), key=lambda x: x[1])[0]
    
    def _calculate_confidence(self, state_key: str, action: ExitAction) -> float:
        """Calculate confidence in action based on experience"""
        if state_key not in self.q_table:
            return 0.1  # Low confidence for unseen states
        
        q_values = self.q_table[state_key]
        if action not in q_values:
            return 0.1
        
        # Confidence based on Q-value magnitude and relative ranking
        action_q = q_values[action]
        max_q = max(q_values.values()) if q_values else 0
        min_q = min(q_values.values()) if q_values else 0
        
        if max_q == min_q:
            return 0.5
        
        # Normalize Q-value to [0, 1]
        normalized_q = (action_q - min_q) / (max_q - min_q)
        
        # Add experience factor
        experience_factor = min(1.0, len(q_values) / len(ExitAction))
        
        confidence = normalized_q * 0.7 + experience_factor * 0.3
        
        return min(0.95, max(0.05, confidence))
    
    def _get_position_adjustment(self, action: ExitAction) -> float:
        """Get position adjustment amount for action"""
        adjustments = {
            ExitAction.HOLD: 0.0,
            ExitAction.PARTIAL_EXIT_25: 0.25,
            ExitAction.PARTIAL_EXIT_50: 0.50,
            ExitAction.PARTIAL_EXIT_75: 0.75,
            ExitAction.FULL_EXIT: 1.0,
            ExitAction.TRAIL_STOP: 0.0,
            ExitAction.TIGHTEN_STOP: 0.0
        }
        
        return adjustments.get(action, 0.0)
    
    def _calculate_new_stop_loss(self, action: ExitAction, position_state: PositionState) -> Optional[float]:
        """Calculate new stop loss for stop-related actions"""
        if action == ExitAction.TRAIL_STOP:
            # Trail stop to breakeven or small profit
            if position_state.unrealized_pnl_pct > 0.01:  # If up more than 1%
                return position_state.entry_price * 1.002  # 0.2% profit
            else:
                return position_state.entry_price * 0.999  # Small loss
        
        elif action == ExitAction.TIGHTEN_STOP:
            # Tighten stop loss
            current_stop = position_state.current_stop_loss
            entry_price = position_state.entry_price
            current_price = position_state.current_price
            
            if current_price > entry_price:  # Long position in profit
                # Move stop closer to current price
                new_stop = current_price * 0.995  # 0.5% below current
                return max(new_stop, current_stop)  # Don't move stop down
            else:  # Long position in loss
                # Tighten stop slightly
                return current_stop * 1.001
        
        return None
    
    def _generate_reasoning(self, action: ExitAction, position_state: PositionState, 
                          expected_reward: float) -> str:
        """Generate human-readable reasoning for action"""
        pnl_pct = position_state.unrealized_pnl_pct * 100
        time_mins = position_state.time_in_position / 60
        
        base_reason = f"PnL: {pnl_pct:.2f}%, Time: {time_mins:.1f}min, Expected reward: {expected_reward:.3f}"
        
        if action == ExitAction.HOLD:
            return f"Hold position. {base_reason}"
        elif action == ExitAction.FULL_EXIT:
            return f"Full exit recommended. {base_reason}"
        elif action.value.startswith('partial_exit'):
            pct = action.value.split('_')[-1]
            return f"Partial exit ({pct}%) recommended. {base_reason}"
        elif action == ExitAction.TRAIL_STOP:
            return f"Trail stop to protect profits. {base_reason}"
        elif action == ExitAction.TIGHTEN_STOP:
            return f"Tighten stop loss. {base_reason}"
        else:
            return f"Action: {action.value}. {base_reason}"
    
    def update_q_value(self, state_key: str, action: ExitAction, reward: float, 
                      next_state_key: Optional[str] = None) -> None:
        """Update Q-value using Q-learning algorithm"""
        with self._lock:
            # Current Q-value
            current_q = self.q_table[state_key][action]
            
            # Calculate target Q-value
            if next_state_key and next_state_key in self.q_table:
                # Get maximum Q-value for next state
                next_q_values = self.q_table[next_state_key]
                max_next_q = max(next_q_values.values()) if next_q_values else 0
            else:
                max_next_q = 0
            
            # Q-learning update
            target_q = reward + self.discount_factor * max_next_q
            new_q = current_q + self.learning_rate * (target_q - current_q)
            
            # Update Q-table
            self.q_table[state_key][action] = new_q
            
            # Store experience
            experience = {
                'state': state_key,
                'action': action,
                'reward': reward,
                'next_state': next_state_key,
                'timestamp': int(time.time() * 1000000)
            }
            self.experience_buffer.append(experience)
            
            # Track reward
            self.reward_history.append(reward)
    
    def calculate_reward(self, position_state_before: PositionState, 
                        position_state_after: PositionState, 
                        action: ExitAction) -> float:
        """Calculate reward for taken action"""
        # Base reward from PnL change
        pnl_change = position_state_after.unrealized_pnl - position_state_before.unrealized_pnl
        base_reward = pnl_change / abs(position_state_before.entry_price) * 100  # Normalize
        
        # Time penalty (encourage faster decisions)
        time_penalty = -0.001 * (position_state_after.time_in_position - position_state_before.time_in_position) / 60
        
        # Risk adjustment
        risk_adjustment = 0.0
        if action == ExitAction.FULL_EXIT:
            # Reward for taking profit or cutting losses at right time
            if position_state_before.unrealized_pnl_pct > 0.02:  # Good profit
                risk_adjustment = 0.1
            elif position_state_before.unrealized_pnl_pct < -0.02:  # Cut losses
                risk_adjustment = 0.05
        
        elif action in [ExitAction.TRAIL_STOP, ExitAction.TIGHTEN_STOP]:
            # Small reward for risk management
            risk_adjustment = 0.02
        
        # Volatility adjustment
        vol_adjustment = -position_state_after.features.realized_volatility * 10
        
        total_reward = base_reward + time_penalty + risk_adjustment + vol_adjustment
        
        return total_reward
    
    def replay_experience(self, batch_size: int = 32) -> None:
        """Replay experiences for additional learning"""
        if len(self.experience_buffer) < batch_size:
            return
        
        with self._lock:
            # Sample random batch
            batch = random.sample(list(self.experience_buffer), batch_size)
            
            for experience in batch:
                state = experience['state']
                action = experience['action']
                reward = experience['reward']
                next_state = experience['next_state']
                
                # Re-update Q-value with current parameters
                self.update_q_value(state, action, reward, next_state)
    
    def decay_epsilon(self, decay_rate: float = 0.995, min_epsilon: float = 0.01) -> None:
        """Decay exploration rate over time"""
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        with self._lock:
            if not self.reward_history:
                return {'error': 'no_data'}
            
            avg_reward = sum(self.reward_history) / len(self.reward_history)
            recent_avg_reward = sum(list(self.reward_history)[-100:]) / min(100, len(self.reward_history))
            
            return {
                'symbol': self.symbol,
                'total_experiences': len(self.experience_buffer),
                'total_states': len(self.q_table),
                'avg_reward': avg_reward,
                'recent_avg_reward': recent_avg_reward,
                'epsilon': self.epsilon,
                'action_distribution': dict(self.action_counts),
                'reward_trend': list(self.reward_history)[-20:]  # Last 20 rewards
            }
    
    def get_q_table_summary(self) -> Dict[str, Any]:
        """Get summary of Q-table for analysis"""
        with self._lock:
            if not self.q_table:
                return {'error': 'no_data'}
            
            # Calculate statistics
            all_q_values = []
            action_preferences = defaultdict(int)
            
            for state, actions in self.q_table.items():
                for action, q_value in actions.items():
                    all_q_values.append(q_value)
                
                # Count best actions
                if actions:
                    best_action = max(actions.items(), key=lambda x: x[1])[0]
                    action_preferences[best_action] += 1
            
            return {
                'total_states': len(self.q_table),
                'total_q_values': len(all_q_values),
                'avg_q_value': sum(all_q_values) / len(all_q_values) if all_q_values else 0,
                'max_q_value': max(all_q_values) if all_q_values else 0,
                'min_q_value': min(all_q_values) if all_q_values else 0,
                'preferred_actions': dict(action_preferences)
            }
    
    def _save_model(self) -> None:
        """Save Q-table and parameters to disk"""
        try:
            model_data = {
                'q_table': dict(self.q_table),
                'epsilon': self.epsilon,
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'action_counts': dict(self.action_counts),
                'performance_history': list(self.performance_history),
                'reward_history': list(self.reward_history)
            }
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
                
        except Exception as e:
            print(f"Error saving RL model: {e}")
    
    def _load_model(self) -> None:
        """Load Q-table and parameters from disk"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                # Restore Q-table
                loaded_q_table = model_data.get('q_table', {})
                for state, actions in loaded_q_table.items():
                    for action_str, q_value in actions.items():
                        action = ExitAction(action_str)
                        self.q_table[state][action] = q_value
                
                # Restore parameters
                self.epsilon = model_data.get('epsilon', self.epsilon)
                self.learning_rate = model_data.get('learning_rate', self.learning_rate)
                self.discount_factor = model_data.get('discount_factor', self.discount_factor)
                
                # Restore statistics
                self.action_counts.update(model_data.get('action_counts', {}))
                self.performance_history.extend(model_data.get('performance_history', []))
                self.reward_history.extend(model_data.get('reward_history', []))
                
                print(f"RL exit agent model loaded for {self.symbol}")
                
        except Exception as e:
            print(f"Error loading RL model: {e}")
    
    def save_model(self) -> None:
        """Public method to save model"""
        self._save_model()
    
    def reset_learning(self) -> None:
        """Reset learning state (for testing or retraining)"""
        with self._lock:
            self.q_table.clear()
            self.experience_buffer.clear()
            self.action_counts.clear()
            self.performance_history.clear()
            self.reward_history.clear()
            self.epsilon = 0.1  # Reset exploration rate

