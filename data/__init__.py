"""
Chimera Trading System v2.0 - Data Module
Market data generation, simulation, and interface management
"""

from .mock_data_generator import MockDataGenerator, MarketScenario, MarketRegime
from .market_simulator import MarketSimulator, SimulationConfig
from .data_interface import (
    DataSourceInterface, DataSourceType, DataSourceRegistry,
    Level2Update, TradeUpdate, QuoteUpdate, BarUpdate, NewsUpdate,
    IBKRDataSource, AlpacaDataSource, PolygonDataSource,
    register_data_source, get_data_source, set_active_data_source,
    get_active_data_source, list_data_sources
)
from .mock_adapter import MockDataSourceAdapter

__all__ = [
    # Mock data generation
    'MockDataGenerator',
    'MarketScenario', 
    'MarketRegime',
    'MarketSimulator',
    'SimulationConfig',
    
    # Data interface
    'DataSourceInterface',
    'DataSourceType',
    'DataSourceRegistry',
    'Level2Update',
    'TradeUpdate', 
    'QuoteUpdate',
    'BarUpdate',
    'NewsUpdate',
    
    # Data source implementations
    'IBKRDataSource',
    'AlpacaDataSource', 
    'PolygonDataSource',
    'MockDataSourceAdapter',
    
    # Registry functions
    'register_data_source',
    'get_data_source',
    'set_active_data_source', 
    'get_active_data_source',
    'list_data_sources'
]