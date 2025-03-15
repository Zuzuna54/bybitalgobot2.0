"""
Strategy Panel Components

This package provides components for the strategy panel in the dashboard,
including strategy performance, recent signals, and strategy comparisons.
"""

from src.dashboard.components.strategy.panel import create_strategy_panel
from src.dashboard.components.strategy.performance_view import (
    create_strategy_performance_graph, 
    render_top_strategies_card,
    create_strategy_comparison_graph
)
from src.dashboard.components.strategy.signals_view import render_recent_signals_table
from src.dashboard.components.strategy.callbacks import register_strategy_callbacks

__all__ = [
    'create_strategy_panel',
    'create_strategy_performance_graph',
    'render_top_strategies_card',
    'render_recent_signals_table',
    'create_strategy_comparison_graph',
    'register_strategy_callbacks'
] 