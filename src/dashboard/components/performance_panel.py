"""
Performance Panel Component for the Trading Dashboard

This module is maintained for backward compatibility and re-exports
the functionality from the new modular performance panel components.

For new code, please use the modular imports directly from:
src.dashboard.components.performance
"""

# Re-export from modular structure
from src.dashboard.components.performance import (
    create_performance_panel,
    render_metrics_card,
    register_performance_callbacks,
    create_equity_curve_chart,
    create_drawdown_chart,
    create_return_distribution_chart,
    create_daily_performance_graph,
)

# For backward compatibility
__all__ = [
    "create_performance_panel",
    "render_metrics_card",
    "register_performance_callbacks",
    "create_equity_curve_chart",
    "create_drawdown_chart",
    "create_return_distribution_chart",
    "create_daily_performance_graph",
]
