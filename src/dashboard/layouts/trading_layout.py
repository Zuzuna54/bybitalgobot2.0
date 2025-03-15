"""
Trading Panel Layout Module

This module provides the layout for the trading panel in the dashboard.
It displays current trades, order history, and trading controls.
"""

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from datetime import datetime
from loguru import logger


def create_trading_panel() -> html.Div:
    """
    Create the trading panel layout.
    
    Returns:
        Dash HTML Div containing the trading panel layout
    """
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H3("Active Trades"),
                html.Div(id="active-trades-table", className="data-table"),
                html.Hr(),
                html.H3("Order History"),
                html.Div(id="order-history-table", className="data-table"),
            ], md=8),
            dbc.Col([
                html.H3("Trading Controls"),
                html.Div([
                    dbc.Card([
                        dbc.CardHeader("System Controls"),
                        dbc.CardBody([
                            dbc.Button("Start Trading", id="start-system-button", color="success", className="me-2"),
                            dbc.Button("Pause Trading", id="pause-system-button", color="warning", className="me-2"),
                            dbc.Button("Stop Trading", id="stop-system-button", color="danger"),
                        ]),
                    ]),
                    html.Br(),
                    dbc.Card([
                        dbc.CardHeader("Paper Trading Controls"),
                        dbc.CardBody([
                            dbc.Button(
                                "Reset Paper Trading", 
                                id="reset-paper-trading-button", 
                                color="info"
                            ),
                            html.Div(id="paper-trading-status", className="mt-3"),
                        ]),
                    ]),
                    html.Br(),
                    dbc.Card([
                        dbc.CardHeader("Manual Trading"),
                        dbc.CardBody([
                            dbc.Label("Symbol"),
                            dbc.Input(id="manual-trade-symbol", type="text", placeholder="BTCUSDT"),
                            dbc.Label("Side"),
                            dbc.RadioItems(
                                id="manual-trade-side",
                                options=[
                                    {"label": "Buy", "value": "buy"},
                                    {"label": "Sell", "value": "sell"},
                                ],
                                value="buy",
                                inline=True,
                            ),
                            dbc.Label("Quantity"),
                            dbc.Input(id="manual-trade-quantity", type="number", placeholder="0.001"),
                            dbc.Button(
                                "Place Order", 
                                id="manual-trade-button", 
                                color="primary", 
                                className="mt-3",
                                disabled=True
                            ),
                            html.Div("Manual trading is disabled in this version", className="text-muted mt-2"),
                        ]),
                    ]),
                ]),
            ], md=4),
        ]),
    ])


def register_trading_callbacks(app: dash.Dash) -> None:
    """
    Register callbacks for the trading panel.
    
    Args:
        app: The Dash application instance
    """
    @app.callback(
        dash.Output("active-trades-table", "children"),
        [dash.Input("data-update-interval", "n_intervals")]
    )
    def update_active_trades(n_intervals):
        """Update the active trades table with current trading data."""
        # This would be populated with actual trade data
        return html.P("No active trades")
    
    @app.callback(
        dash.Output("order-history-table", "children"),
        [dash.Input("data-update-interval", "n_intervals")]
    )
    def update_order_history(n_intervals):
        """Update the order history table with historical trading data."""
        # This would be populated with actual order history data
        return html.P("No order history") 