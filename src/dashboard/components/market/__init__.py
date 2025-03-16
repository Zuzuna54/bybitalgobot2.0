"""
Market Panel Components Package for the Trading Dashboard

This package provides components for displaying market data and related visualizations.
"""

from src.dashboard.components.market.panel import create_market_panel
from src.dashboard.components.market.callbacks import register_market_callbacks

__all__ = [
    # Main panel
    "create_market_panel",
    # Callbacks
    "register_market_callbacks",
]
