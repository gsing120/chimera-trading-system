"""
Chimera Trading System v2.0 - Data Module
Real IBKR data integration only - NO SIMULATIONS
"""

from .data_interface import (
    DataSourceInterface, DataSourceType, DataSourceRegistry,
    Level2Update, TradeUpdate, QuoteUpdate, BarUpdate, NewsUpdate,
    IBKRDataSource, AlpacaDataSource, PolygonDataSource,
    register_data_source, get_data_source, set_active_data_source,
    get_active_data_source, list_data_sources
)
from .ibkr_adapter import IBKRAdapter, create_ibkr_adapter

__all__ = [
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
    'IBKRAdapter',
    'create_ibkr_adapter',
    
    # Registry functions
    'register_data_source',
    'get_data_source',
    'set_active_data_source', 
    'get_active_data_source',
    'list_data_sources'
]