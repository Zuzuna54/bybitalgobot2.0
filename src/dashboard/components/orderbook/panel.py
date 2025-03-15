"""
Orderbook Panel Layout Component for the Trading Dashboard

This module provides the main layout for the orderbook panel component.
"""

from typing import Dict, Any
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

def create_orderbook_panel() -> html.Div:
    """
    Create the order book panel layout.
    
    Returns:
        Dash HTML Div containing the order book panel
    """
    return html.Div([
        html.H2("Order Book Analysis", className="panel-header"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Symbol"),
                    dbc.CardBody([
                        dcc.Dropdown(
                            id="orderbook-symbol-dropdown",
                            placeholder="Select symbol",
                            value=None
                        )
                    ])
                ])
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Order Book Imbalance"),
                    dbc.CardBody([
                        html.Div(id="orderbook-imbalance-indicator")
                    ])
                ])
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        "Buy/Sell Ratio ",
                        html.Span(id="liquidity-data-freshness-indicator", className="data-freshness-indicator")
                    ]),
                    dbc.CardBody([
                        html.Div(id="liquidity-ratio-indicator")
                    ])
                ])
            ], width=4)
        ]),
        
        html.Br(),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        "Order Book Visualization ",
                        html.Span(id="orderbook-data-freshness-indicator", className="data-freshness-indicator")
                    ]),
                    dbc.CardBody([
                        dcc.Graph(id="orderbook-depth-graph")
                    ])
                ])
            ], width=12)
        ]),
        
        html.Br(),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        "Support & Resistance Levels ",
                        html.Span(id="support-resistance-freshness-indicator", className="data-freshness-indicator")
                    ]),
                    dbc.CardBody([
                        html.Div(id="support-resistance-content")
                    ])
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Execution Recommendations"),
                    dbc.CardBody([
                        html.Div(id="execution-recommendations-content")
                    ])
                ])
            ], width=6)
        ]),
        
        html.Br(),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Data Connection Status"),
                    dbc.CardBody([
                        html.Div([
                            html.Div([
                                html.Span("WebSocket Status: ", className="connection-status-label"),
                                html.Span(id="websocket-status-badge", className="connection-status-badge")
                            ], className="mb-2"),
                            html.Div([
                                html.Span("Last Update: ", className="connection-status-label"),
                                html.Span(id="orderbook-last-update-time", className="last-update-time")
                            ]),
                            html.Div(id="ws-connection-details", className="mt-3")
                        ])
                    ])
                ])
            ], width=12)
        ]),
        
        dcc.Interval(
            id="orderbook-update-interval",
            interval=1 * 1000,  # 1 second in milliseconds
            n_intervals=0
        ),
        
        # Store for available symbols
        dcc.Store(id="available-symbols-store"),
        
        # Store for WebSocket status
        dcc.Store(id="websocket-status-store"),
        
        # Store for data freshness status
        dcc.Store(id="data-freshness-store")
    ], id="orderbook-panel") 