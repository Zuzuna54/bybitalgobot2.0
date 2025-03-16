"""
Order Book Panel Component for the Trading Dashboard

This module provides visualization components for displaying order book data,
liquidity analysis, and order execution recommendations.

This file has been refactored into smaller components in the src/dashboard/components/orderbook/ package
and now re-exports the necessary components from there.
"""

# Re-export the main panel function
from src.dashboard.components.orderbook.panel import create_orderbook_panel

# Re-export the callbacks registration function
from src.dashboard.components.orderbook.callbacks import register_orderbook_callbacks

# Re-export visualization components directly from chart_service
from src.dashboard.services.chart_service import (
    render_imbalance_indicator,
    render_liquidity_ratio,
    create_orderbook_depth_chart,
    render_support_resistance_levels,
    render_execution_recommendations,
)

# Define the public API
__all__ = [
    "create_orderbook_panel",
    "register_orderbook_callbacks",
    "render_imbalance_indicator",
    "render_liquidity_ratio",
    "create_orderbook_depth_chart",
    "render_support_resistance_levels",
    "render_execution_recommendations",
]
