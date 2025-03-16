"""
Orderbook Panel Components Package for the Trading Dashboard

This package provides components for displaying and analyzing orderbook data,
including visualizations, data processing, and dashboard callbacks.
"""

from src.dashboard.components.orderbook.panel import create_orderbook_panel
from src.dashboard.services.chart_service import (
    render_imbalance_indicator,
    render_liquidity_ratio,
    create_orderbook_depth_chart,
    render_support_resistance_levels,
    render_execution_recommendations,
)
from src.dashboard.components.orderbook.data_processing import (
    calculate_orderbook_imbalance,
    calculate_liquidity_ratio,
    identify_support_resistance_levels,
    generate_execution_recommendations,
)
from src.dashboard.components.orderbook.callbacks import register_orderbook_callbacks

__all__ = [
    # Main panel
    "create_orderbook_panel",
    # Visualization components
    "render_imbalance_indicator",
    "render_liquidity_ratio",
    "create_orderbook_depth_chart",
    "render_support_resistance_levels",
    "render_execution_recommendations",
    # Data processing
    "calculate_orderbook_imbalance",
    "calculate_liquidity_ratio",
    "identify_support_resistance_levels",
    "generate_execution_recommendations",
    # Callbacks
    "register_orderbook_callbacks",
]
