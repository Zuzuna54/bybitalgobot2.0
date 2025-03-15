"""
Trading Panel Layout

This module provides the main layout for the trading panel in the dashboard.
"""

from typing import Dict, Any, List
from dash import html, dcc
import dash_bootstrap_components as dbc


def create_trading_panel() -> html.Div:
    """
    Create the trading panel layout.
    
    Returns:
        Dash HTML Div containing the trading panel
    """
    return html.Div([
        html.H2("Trading Activity", className="panel-header"),
        
        dbc.Tabs([
            dbc.Tab(label="Active Trades", tab_id="active-trades-tab", children=[
                html.Div(id="active-trades-content")
            ]),
            dbc.Tab(label="Pending Orders", tab_id="pending-orders-tab", children=[
                html.Div(id="pending-orders-content")
            ]),
            dbc.Tab(label="Trade History", tab_id="trade-history-tab", children=[
                html.Div(id="trade-history-content")
            ])
        ], id="trading-tabs", active_tab="active-trades-tab"),
        
        html.Br(),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("P&L by Symbol"),
                    dbc.CardBody([
                        dcc.Graph(id="pnl-by-symbol-graph")
                    ])
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Win/Loss by Strategy"),
                    dbc.CardBody([
                        dcc.Graph(id="win-loss-by-strategy-graph")
                    ])
                ])
            ], width=6)
        ]),
        
        dcc.Interval(
            id="trading-update-interval",
            interval=5 * 1000,  # 5 seconds in milliseconds
            n_intervals=0
        )
    ], id="trading-panel") 