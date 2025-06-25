"""
Genetic Algorithm Optimizer
Evolves trading strategy parameters for optimal performance
"""

import time
import random
import math
import statistics
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any, Callable
import threading
import pickle
import os
import copy

from core.signal_detector import SignalType


@dataclass
class StrategyGenome:
    """Represents a strategy's genetic parameters"""
    signal_type: SignalType
    symbol: str
    
    # Signal detection parameters
    min_absorption_volume: float
    absorption_price_tolerance: float
    sweep_volume_threshold: float
    iceberg_replenishment_threshold: float
    imbalance_threshold: float
    confluence_weight: float
    min_signal_confidence: float
    
    # Risk management parameters
    max_position_size: float
    stop_loss_multiplier: float
    take_profit_multiplier: float
    risk_per_trade: float
    
    # Timing parameters
    entry_delay_ms: int
    exit_delay_ms: int
    max_hold_time_minutes: int
    
    # Market condition filters
    min_volatility: float
    max_volatility: float
    min_liquidity: float
    trend_filter_strength: float
    
    # Performance tracking
    fitness_score: float = 0.0
    trades_count: int = 0
    win_rate: float = 0.0
    avg_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert genome to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyGenome':
        """Create genome from dictionary"""
        return cls(**data)


@dataclass
class BacktestResult:
    """Results from strategy backtesting"""
    genome: StrategyGenome
    total_return: float
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    avg_trade_duration: float
    fitness_score: float


class GeneticOptimizer:
    """
    Genetic Algorithm for optimizing trading strategy parameters
    """
    
    def __init__(self, population_size: int = 50, mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8, elitism_rate: float = 0.2):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_rate = elitism_rate
        
        # Population management
        self.populations: Dict[str, List[StrategyGenome]] = {}  # symbol -> population
        self.generation_count: Dict[str, int] = defaultdict(int)
        self.best_genomes: Dict[str, StrategyGenome] = {}
        
        # Evolution history
        self.evolution_history: Dict[str, List[Dict]] = defaultdict(list)
        
        # Parameter bounds for mutation
        self.parameter_bounds = {
            'min_absorption_volume': (500, 5000),
            'absorption_price_tolerance': (0.0005, 0.005),
            'sweep_volume_threshold': (1000, 10000),
            'iceberg_replenishment_threshold': (0.5, 0.95),
            'imbalance_threshold': (0.1, 0.5),
            'confluence_weight': (1.0, 3.0),
            'min_signal_confidence': (0.3, 0.8),
            'max_position_size': (0.01, 0.1),
            'stop_loss_multiplier': (0.5, 3.0),
            'take_profit_multiplier': (1.0, 5.0),
            'risk_per_trade': (0.005, 0.02),
            'entry_delay_ms': (0, 1000),
            'exit_delay_ms': (0, 2000),
            'max_hold_time_minutes': (5, 120),
            'min_volatility': (0.0001, 0.01),
            'max_volatility': (0.01, 0.1),
            'min_liquidity': (100, 2000),
            'trend_filter_strength': (0.0, 1.0)
        }
        
        # Backtesting function
        self.backtest_function: Optional[Callable[[StrategyGenome], BacktestResult]] = None
        
        # Model persistence
        self.model_dir = "genetic_models"
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Load existing populations
        self._load_populations()
    
    def set_backtest_function(self, backtest_func: Callable[[StrategyGenome], BacktestResult]) -> None:
        """Set the backtesting function for fitness evaluation"""
        self.backtest_function = backtest_func
    
    def initialize_population(self, symbol: str, signal_type: SignalType) -> List[StrategyGenome]:
        """Initialize random population for symbol and signal type"""
        with self._lock:
            population = []
            
            for _ in range(self.population_size):
                genome = self._create_random_genome(symbol, signal_type)
                population.append(genome)
            
            key = f"{symbol}_{signal_type.value}"
            self.populations[key] = population
            self.generation_count[key] = 0
            
            return population
    
    def _create_random_genome(self, symbol: str, signal_type: SignalType) -> StrategyGenome:
        """Create a random genome within parameter bounds"""
        genome = StrategyGenome(
            signal_type=signal_type,
            symbol=symbol,
            min_absorption_volume=random.uniform(*self.parameter_bounds['min_absorption_volume']),
            absorption_price_tolerance=random.uniform(*self.parameter_bounds['absorption_price_tolerance']),
            sweep_volume_threshold=random.uniform(*self.parameter_bounds['sweep_volume_threshold']),
            iceberg_replenishment_threshold=random.uniform(*self.parameter_bounds['iceberg_replenishment_threshold']),
            imbalance_threshold=random.uniform(*self.parameter_bounds['imbalance_threshold']),
            confluence_weight=random.uniform(*self.parameter_bounds['confluence_weight']),
            min_signal_confidence=random.uniform(*self.parameter_bounds['min_signal_confidence']),
            max_position_size=random.uniform(*self.parameter_bounds['max_position_size']),
            stop_loss_multiplier=random.uniform(*self.parameter_bounds['stop_loss_multiplier']),
            take_profit_multiplier=random.uniform(*self.parameter_bounds['take_profit_multiplier']),
            risk_per_trade=random.uniform(*self.parameter_bounds['risk_per_trade']),
            entry_delay_ms=random.randint(*[int(x) for x in self.parameter_bounds['entry_delay_ms']]),
            exit_delay_ms=random.randint(*[int(x) for x in self.parameter_bounds['exit_delay_ms']]),
            max_hold_time_minutes=random.randint(*[int(x) for x in self.parameter_bounds['max_hold_time_minutes']]),
            min_volatility=random.uniform(*self.parameter_bounds['min_volatility']),
            max_volatility=random.uniform(*self.parameter_bounds['max_volatility']),
            min_liquidity=random.uniform(*self.parameter_bounds['min_liquidity']),
            trend_filter_strength=random.uniform(*self.parameter_bounds['trend_filter_strength'])
        )
        
        return genome
    
    def evolve_generation(self, symbol: str, signal_type: SignalType) -> Dict[str, Any]:
        """Evolve one generation of the population"""
        if not self.backtest_function:
            raise ValueError("Backtest function not set")
        
        with self._lock:
            key = f"{symbol}_{signal_type.value}"
            
            if key not in self.populations:
                self.initialize_population(symbol, signal_type)
            
            population = self.populations[key]
            
            # Evaluate fitness for all genomes
            print(f"Evaluating fitness for {len(population)} genomes...")
            fitness_results = []
            
            for i, genome in enumerate(population):
                try:
                    result = self.backtest_function(genome)
                    genome.fitness_score = result.fitness_score
                    genome.trades_count = result.total_trades
                    genome.win_rate = result.win_rate
                    genome.avg_return = result.total_return
                    genome.sharpe_ratio = result.sharpe_ratio
                    genome.max_drawdown = result.max_drawdown
                    
                    fitness_results.append(result)
                    
                except Exception as e:
                    print(f"Error evaluating genome {i}: {e}")
                    genome.fitness_score = -1.0  # Penalty for failed evaluation
            
            # Sort population by fitness
            population.sort(key=lambda g: g.fitness_score, reverse=True)
            
            # Track best genome
            if population[0].fitness_score > self.best_genomes.get(key, StrategyGenome(signal_type, symbol, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)).fitness_score:
                self.best_genomes[key] = copy.deepcopy(population[0])
            
            # Create next generation
            new_population = self._create_next_generation(population)
            self.populations[key] = new_population
            self.generation_count[key] += 1
            
            # Record evolution statistics
            generation_stats = {
                'generation': self.generation_count[key],
                'best_fitness': population[0].fitness_score,
                'avg_fitness': sum(g.fitness_score for g in population) / len(population),
                'worst_fitness': population[-1].fitness_score,
                'best_genome': population[0].to_dict(),
                'timestamp': int(time.time() * 1000000)
            }
            
            self.evolution_history[key].append(generation_stats)
            
            # Save population
            self._save_population(key)
            
            return generation_stats
    
    def _create_next_generation(self, population: List[StrategyGenome]) -> List[StrategyGenome]:
        """Create next generation using selection, crossover, and mutation"""
        new_population = []
        
        # Elitism: keep best genomes
        elite_count = int(len(population) * self.elitism_rate)
        new_population.extend(copy.deepcopy(population[:elite_count]))
        
        # Generate rest through crossover and mutation
        while len(new_population) < self.population_size:
            # Selection
            parent1 = self._tournament_selection(population)
            parent2 = self._tournament_selection(population)
            
            # Crossover
            if random.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
            
            # Mutation
            if random.random() < self.mutation_rate:
                child1 = self._mutate(child1)
            if random.random() < self.mutation_rate:
                child2 = self._mutate(child2)
            
            new_population.extend([child1, child2])
        
        # Trim to exact population size
        return new_population[:self.population_size]
    
    def _tournament_selection(self, population: List[StrategyGenome], 
                            tournament_size: int = 3) -> StrategyGenome:
        """Select genome using tournament selection"""
        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=lambda g: g.fitness_score)
    
    def _crossover(self, parent1: StrategyGenome, parent2: StrategyGenome) -> Tuple[StrategyGenome, StrategyGenome]:
        """Create two children through crossover"""
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
        
        # Get all numeric parameters
        numeric_params = [
            'min_absorption_volume', 'absorption_price_tolerance', 'sweep_volume_threshold',
            'iceberg_replenishment_threshold', 'imbalance_threshold', 'confluence_weight',
            'min_signal_confidence', 'max_position_size', 'stop_loss_multiplier',
            'take_profit_multiplier', 'risk_per_trade', 'entry_delay_ms', 'exit_delay_ms',
            'max_hold_time_minutes', 'min_volatility', 'max_volatility', 'min_liquidity',
            'trend_filter_strength'
        ]
        
        # Uniform crossover: randomly swap parameters
        for param in numeric_params:
            if random.random() < 0.5:
                # Swap parameter values
                val1 = getattr(child1, param)
                val2 = getattr(child2, param)
                setattr(child1, param, val2)
                setattr(child2, param, val1)
        
        # Reset fitness scores
        child1.fitness_score = 0.0
        child2.fitness_score = 0.0
        
        return child1, child2
    
    def _mutate(self, genome: StrategyGenome) -> StrategyGenome:
        """Mutate genome parameters"""
        mutated = copy.deepcopy(genome)
        
        # Choose random parameter to mutate
        numeric_params = [
            'min_absorption_volume', 'absorption_price_tolerance', 'sweep_volume_threshold',
            'iceberg_replenishment_threshold', 'imbalance_threshold', 'confluence_weight',
            'min_signal_confidence', 'max_position_size', 'stop_loss_multiplier',
            'take_profit_multiplier', 'risk_per_trade', 'entry_delay_ms', 'exit_delay_ms',
            'max_hold_time_minutes', 'min_volatility', 'max_volatility', 'min_liquidity',
            'trend_filter_strength'
        ]
        
        # Mutate 1-3 parameters
        params_to_mutate = random.sample(numeric_params, random.randint(1, 3))
        
        for param in params_to_mutate:
            current_value = getattr(mutated, param)
            bounds = self.parameter_bounds[param]
            
            # Gaussian mutation with 10% standard deviation
            mutation_strength = (bounds[1] - bounds[0]) * 0.1
            new_value = current_value + random.gauss(0, mutation_strength)
            
            # Clamp to bounds
            new_value = max(bounds[0], min(bounds[1], new_value))
            
            # Handle integer parameters
            if param in ['entry_delay_ms', 'exit_delay_ms', 'max_hold_time_minutes']:
                new_value = int(new_value)
            
            setattr(mutated, param, new_value)
        
        # Reset fitness score
        mutated.fitness_score = 0.0
        
        return mutated
    
    def get_best_genome(self, symbol: str, signal_type: SignalType) -> Optional[StrategyGenome]:
        """Get best genome for symbol and signal type"""
        key = f"{symbol}_{signal_type.value}"
        return self.best_genomes.get(key)
    
    def get_population_stats(self, symbol: str, signal_type: SignalType) -> Dict[str, Any]:
        """Get population statistics"""
        with self._lock:
            key = f"{symbol}_{signal_type.value}"
            
            if key not in self.populations:
                return {'error': 'no_population'}
            
            population = self.populations[key]
            fitness_scores = [g.fitness_score for g in population]
            
            return {
                'symbol': symbol,
                'signal_type': signal_type.value,
                'generation': self.generation_count[key],
                'population_size': len(population),
                'best_fitness': max(fitness_scores) if fitness_scores else 0,
                'avg_fitness': sum(fitness_scores) / len(fitness_scores) if fitness_scores else 0,
                'worst_fitness': min(fitness_scores) if fitness_scores else 0,
                'fitness_std': statistics.stdev(fitness_scores) if len(fitness_scores) > 1 else 0,
                'best_genome': self.best_genomes.get(key, {})
            }
    
    def get_evolution_history(self, symbol: str, signal_type: SignalType) -> List[Dict]:
        """Get evolution history for analysis"""
        key = f"{symbol}_{signal_type.value}"
        return self.evolution_history.get(key, [])
    
    def run_optimization(self, symbol: str, signal_type: SignalType, 
                        generations: int = 20) -> Dict[str, Any]:
        """Run complete optimization for specified generations"""
        if not self.backtest_function:
            raise ValueError("Backtest function not set")
        
        print(f"Starting genetic optimization for {symbol} {signal_type.value}")
        print(f"Running {generations} generations with population size {self.population_size}")
        
        results = []
        
        for generation in range(generations):
            print(f"\nGeneration {generation + 1}/{generations}")
            
            try:
                generation_result = self.evolve_generation(symbol, signal_type)
                results.append(generation_result)
                
                print(f"Best fitness: {generation_result['best_fitness']:.4f}")
                print(f"Avg fitness: {generation_result['avg_fitness']:.4f}")
                
            except Exception as e:
                print(f"Error in generation {generation + 1}: {e}")
                break
        
        # Final results
        best_genome = self.get_best_genome(symbol, signal_type)
        final_stats = self.get_population_stats(symbol, signal_type)
        
        return {
            'symbol': symbol,
            'signal_type': signal_type.value,
            'generations_completed': len(results),
            'best_genome': best_genome.to_dict() if best_genome else None,
            'final_stats': final_stats,
            'evolution_history': results
        }
    
    def _save_population(self, key: str) -> None:
        """Save population to disk"""
        try:
            population_data = {
                'population': [genome.to_dict() for genome in self.populations[key]],
                'generation_count': self.generation_count[key],
                'best_genome': self.best_genomes.get(key, {}).to_dict() if key in self.best_genomes else None,
                'evolution_history': self.evolution_history[key]
            }
            
            filename = os.path.join(self.model_dir, f"population_{key}.pkl")
            with open(filename, 'wb') as f:
                pickle.dump(population_data, f)
                
        except Exception as e:
            print(f"Error saving population {key}: {e}")
    
    def _load_populations(self) -> None:
        """Load populations from disk"""
        try:
            for filename in os.listdir(self.model_dir):
                if filename.startswith('population_') and filename.endswith('.pkl'):
                    key = filename[11:-4]  # Remove 'population_' and '.pkl'
                    
                    filepath = os.path.join(self.model_dir, filename)
                    with open(filepath, 'rb') as f:
                        population_data = pickle.load(f)
                    
                    # Restore population
                    population = []
                    for genome_dict in population_data['population']:
                        genome = StrategyGenome.from_dict(genome_dict)
                        population.append(genome)
                    
                    self.populations[key] = population
                    self.generation_count[key] = population_data['generation_count']
                    
                    # Restore best genome
                    if population_data['best_genome']:
                        self.best_genomes[key] = StrategyGenome.from_dict(population_data['best_genome'])
                    
                    # Restore evolution history
                    self.evolution_history[key] = population_data['evolution_history']
                    
                    print(f"Loaded population {key} with {len(population)} genomes")
                    
        except Exception as e:
            print(f"Error loading populations: {e}")
    
    def create_simple_backtest_function(self) -> Callable[[StrategyGenome], BacktestResult]:
        """Create a simple backtesting function for testing"""
        def simple_backtest(genome: StrategyGenome) -> BacktestResult:
            # Simulate random trading results based on genome parameters
            # This is a placeholder - real implementation would use historical data
            
            # Simulate trades
            num_trades = random.randint(10, 100)
            wins = 0
            total_return = 0.0
            returns = []
            
            for _ in range(num_trades):
                # Simulate trade outcome based on genome quality
                base_win_prob = 0.5
                
                # Adjust win probability based on parameters
                if genome.min_signal_confidence > 0.6:
                    base_win_prob += 0.1
                if genome.stop_loss_multiplier < 2.0:
                    base_win_prob += 0.05
                if genome.take_profit_multiplier > 2.0:
                    base_win_prob += 0.05
                
                if random.random() < base_win_prob:
                    # Winning trade
                    trade_return = random.uniform(0.005, 0.02) * genome.take_profit_multiplier
                    wins += 1
                else:
                    # Losing trade
                    trade_return = -random.uniform(0.005, 0.015) * genome.stop_loss_multiplier
                
                total_return += trade_return
                returns.append(trade_return)
            
            # Calculate metrics
            win_rate = wins / num_trades if num_trades > 0 else 0
            avg_return = total_return / num_trades if num_trades > 0 else 0
            
            # Calculate Sharpe ratio (simplified)
            if len(returns) > 1:
                return_std = statistics.stdev(returns)
                sharpe_ratio = avg_return / return_std if return_std > 0 else 0
            else:
                sharpe_ratio = 0
            
            # Calculate max drawdown (simplified)
            cumulative_returns = []
            cumulative = 0
            for ret in returns:
                cumulative += ret
                cumulative_returns.append(cumulative)
            
            max_drawdown = 0
            peak = 0
            for cum_ret in cumulative_returns:
                if cum_ret > peak:
                    peak = cum_ret
                drawdown = peak - cum_ret
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
            
            # Calculate profit factor
            gross_profit = sum(ret for ret in returns if ret > 0)
            gross_loss = abs(sum(ret for ret in returns if ret < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 1.0
            
            # Calculate fitness score (combination of metrics)
            fitness_score = (
                total_return * 0.4 +
                win_rate * 0.2 +
                sharpe_ratio * 0.2 +
                profit_factor * 0.1 -
                max_drawdown * 0.1
            )
            
            return BacktestResult(
                genome=genome,
                total_return=total_return,
                win_rate=win_rate,
                profit_factor=profit_factor,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                total_trades=num_trades,
                avg_trade_duration=random.uniform(5, 60),  # minutes
                fitness_score=fitness_score
            )
        
        return simple_backtest
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get overall optimizer statistics"""
        with self._lock:
            return {
                'total_populations': len(self.populations),
                'total_generations': sum(self.generation_count.values()),
                'population_size': self.population_size,
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate,
                'elitism_rate': self.elitism_rate,
                'active_optimizations': list(self.populations.keys()),
                'best_genomes_count': len(self.best_genomes)
            }

