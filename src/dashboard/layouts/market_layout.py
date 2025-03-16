"""
Market Layout Module

This module provides the layout for the market data section of the dashboard.
"""

from typing import Dict, Any
import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc

from src.dashboard.components.market.panel import create_market_panel


def create_market_layout() -> html.Div:
    """
    Create the market data tab layout.

    Returns:
        Dash component for the market tab
    """
    return html.Div(
        id="market-layout",
        className="dashboard-layout",
        children=[
            # Market panel
            create_market_panel()
        ],
    )


def register_market_layout_callbacks(app: dash.Dash) -> None:
    """
    Register callbacks specific to the market layout.

    Args:
        app: The Dash application instance
    """
    # Add any layout-specific callbacks here
    pass
