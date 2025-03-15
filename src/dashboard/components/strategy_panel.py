"""
Strategy Panel Module

This module re-exports the strategy panel components from the modular structure.
"""

# Re-export from modular components
from src.dashboard.components.strategy.panel import create_strategy_panel
from src.dashboard.components.strategy.performance_view import (
    create_strategy_performance_graph,
    create_strategy_comparison_graph,
    render_top_strategies_card,
    create_detailed_performance_breakdown,
    create_market_condition_performance,
    create_strategy_correlation_matrix
)
from src.dashboard.components.strategy.signals_view import (
    render_recent_signals_table,
    format_signal_data
)
from src.dashboard.components.strategy.callbacks import (
    register_strategy_callbacks,
    create_strategy_activation_controls
)

# Define exported functions
__all__ = [
    'create_strategy_panel',
    'create_strategy_performance_graph',
    'create_strategy_comparison_graph',
    'render_top_strategies_card',
    'render_recent_signals_table',
    'format_signal_data',
    'register_strategy_callbacks',
    'create_detailed_performance_breakdown',
    'create_market_condition_performance',
    'create_strategy_correlation_matrix',
    'create_strategy_activation_controls'
] 