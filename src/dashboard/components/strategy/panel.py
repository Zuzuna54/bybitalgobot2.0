"""
Strategy Panel Layout

This module provides the main layout for the strategy panel in the dashboard.
"""

from typing import Dict, Any, List
from dash import html, dcc
import dash_bootstrap_components as dbc


def create_strategy_panel() -> html.Div:
    """
    Create the strategy panel layout.
    
    Returns:
        Dash HTML Div containing the strategy panel
    """
    return html.Div([
        html.H2("Strategy Analysis", className="panel-header"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Strategy Performance"),
                    dbc.CardBody([
                        dcc.Graph(id="strategy-performance-graph")
                    ])
                ])
            ], width=8),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Top Strategies"),
                    dbc.CardBody(id="top-strategies-content")
                ])
            ], width=4)
        ]),
        
        html.Br(),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Recent Signals"),
                    dbc.CardBody(id="recent-signals-content")
                ])
            ], width=12)
        ]),
        
        html.Br(),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Strategy Comparison"),
                    dbc.CardBody([
                        dcc.Dropdown(
                            id="strategy-comparison-dropdown",
                            multi=True,
                            placeholder="Select strategies to compare"
                        ),
                        dcc.Graph(id="strategy-comparison-graph")
                    ])
                ])
            ], width=12)
        ]),
        
        html.Br(),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Strategy Correlation Matrix"),
                    dbc.CardBody([
                        dcc.Graph(id="strategy-correlation-matrix")
                    ])
                ])
            ], width=12)
        ]),
        
        html.Br(),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Detailed Performance Analysis"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label("Select Strategy:"),
                                dcc.Dropdown(
                                    id="strategy-detail-dropdown",
                                    placeholder="Select a strategy"
                                ),
                            ], width=12),
                        ]),
                        html.Br(),
                        dbc.Tabs([
                            dbc.Tab([
                                dcc.Graph(id="detailed-performance-breakdown")
                            ], label="Performance by Timeframe"),
                            dbc.Tab([
                                dcc.Graph(id="market-condition-performance")
                            ], label="Market Condition Analysis")
                        ])
                    ])
                ])
            ], width=12)
        ]),
        
        html.Br(),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Strategy Activation Controls"),
                    dbc.CardBody(id="strategy-activation-controls")
                ])
            ], width=12)
        ]),
        
        # Store for strategy names
        dcc.Store(id="strategy-names-store"),
        
        dcc.Interval(
            id="strategy-update-interval",
            interval=60 * 1000,  # 60 seconds in milliseconds
            n_intervals=0
        )
    ], id="strategy-panel") 