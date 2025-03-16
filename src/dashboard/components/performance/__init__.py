"""
Performance Panel Components Package for the Trading Dashboard

This package provides components for displaying performance metrics,
equity curves, and other performance-related data.
"""

from src.dashboard.components.performance.panel import create_performance_panel
from src.dashboard.components.performance.metrics import render_metrics_card
from src.dashboard.components.performance.callbacks import (
    register_performance_callbacks,
)
from src.dashboard.services.chart_service import (
    create_equity_curve_chart,
    create_drawdown_chart,
    create_return_distribution_chart,
    create_daily_performance_graph,
)

__all__ = [
    # Main panel
    "create_performance_panel",
    # Components
    "render_metrics_card",
    # Charts
    "create_equity_curve_chart",
    "create_drawdown_chart",
    "create_return_distribution_chart",
    "create_daily_performance_graph",
    # Callbacks
    "register_performance_callbacks",
]
