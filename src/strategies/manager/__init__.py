"""
Strategy Manager Package for the Algorithmic Trading System

This package provides functionality for managing trading strategies,
including strategy loading, signal aggregation, performance tracking,
and optimization.
"""

from src.strategies.manager.core import StrategyManager
from src.strategies.manager.loader import load_strategy_class, get_available_strategies
from src.strategies.manager.signal_aggregator import aggregate_signals, create_aggregated_signal
from src.strategies.manager.performance_tracker import (
    update_strategy_performance,
    update_strategy_weight,
    save_performance_data,
    load_performance_data,
    get_strategy_performance_summary
)
from src.strategies.manager.optimization import (
    adjust_strategy_weight,
    optimize_strategy_weights,
    select_top_strategies,
    get_recommended_adjustments
)

__all__ = [
    # Core
    'StrategyManager',
    
    # Loader
    'load_strategy_class',
    'get_available_strategies',
    
    # Signal Aggregator
    'aggregate_signals',
    'create_aggregated_signal',
    
    # Performance Tracking
    'update_strategy_performance',
    'update_strategy_weight',
    'save_performance_data',
    'load_performance_data',
    'get_strategy_performance_summary',
    
    # Optimization
    'adjust_strategy_weight',
    'optimize_strategy_weights',
    'select_top_strategies',
    'get_recommended_adjustments'
] 