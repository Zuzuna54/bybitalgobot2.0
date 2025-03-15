"""
Market Panel Component

This module provides components for displaying market data.
"""

from typing import Dict, Any, List, Optional
import dash
from dash import html, dcc
from loguru import logger


def create_market_panel() -> html.Div:
    """
    Create the market panel component.
    
    Returns:
        Dash component for the market panel
    """
    return html.Div(
        id="market-panel",
        className="dashboard-panel",
        children=[
            # Header with controls
            html.Div(
                className="panel-header",
                children=[
                    html.H2("Market Data"),
                    html.Div(
                        className="panel-controls",
                        children=[
                            # Symbol selector
                            html.Div(
                                className="control-group",
                                children=[
                                    html.Label("Symbol:"),
                                    dcc.Dropdown(
                                        id="market-symbol-dropdown",
                                        options=[
                                            {"label": "BTC/USD", "value": "BTC/USD"},
                                            {"label": "ETH/USD", "value": "ETH/USD"},
                                            {"label": "BNB/USD", "value": "BNB/USD"},
                                            {"label": "SOL/USD", "value": "SOL/USD"},
                                            {"label": "ADA/USD", "value": "ADA/USD"},
                                            {"label": "XRP/USD", "value": "XRP/USD"}
                                        ],
                                        value="BTC/USD",
                                        clearable=False
                                    )
                                ]
                            ),
                            # Timeframe selector
                            html.Div(
                                className="control-group",
                                children=[
                                    html.Label("Timeframe:"),
                                    dcc.Dropdown(
                                        id="market-timeframe-dropdown",
                                        options=[
                                            {"label": "1 Day", "value": "1d"},
                                            {"label": "1 Week", "value": "1w"},
                                            {"label": "1 Month", "value": "1m"},
                                            {"label": "3 Months", "value": "3m"},
                                            {"label": "6 Months", "value": "6m"},
                                            {"label": "1 Year", "value": "1y"},
                                            {"label": "All", "value": "all"}
                                        ],
                                        value="1d",
                                        clearable=False
                                    )
                                ]
                            )
                        ]
                    )
                ]
            ),
            
            # Market data content
            html.Div(
                className="panel-content",
                children=[
                    # Price and stats cards
                    html.Div(
                        className="market-cards",
                        children=[
                            # Price card
                            html.Div(
                                id="market-price-card",
                                className="market-card price-card",
                                children=html.Div("Loading price data...")
                            ),
                            # Stats card
                            html.Div(
                                id="market-stats-card",
                                className="market-card stats-card",
                                children=html.Div("Loading market statistics...")
                            )
                        ]
                    ),
                    
                    # Chart container
                    html.Div(
                        id="market-chart-container",
                        className="market-chart-container",
                        children=html.Div("Loading chart data...")
                    )
                ]
            ),
            
            # Interval component for auto-refresh
            dcc.Interval(
                id="market-interval-component",
                interval=30 * 1000,  # 30 seconds
                n_intervals=0
            )
        ]
    )


def create_market_card(title: str, value: str, change: Optional[float] = None) -> html.Div:
    """
    Create a market data card component.
    
    Args:
        title: Card title
        value: Main value to display
        change: Optional change value (with color coding)
        
    Returns:
        Dash component for the market card
    """
    children = [
        html.H4(title),
        html.H2(value)
    ]
    
    if change is not None:
        children.append(
            html.Div(
                className="change-indicator",
                children=[
                    html.Span(
                        f"{change:.2f}%",
                        style={
                            "color": "green" if change >= 0 else "red",
                            "font-weight": "bold"
                        }
                    ),
                    html.Span(" (24h)", style={"color": "gray"})
                ]
            )
        )
    
    return html.Div(
        className="market-card",
        children=children
    ) 