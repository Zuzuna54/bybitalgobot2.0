"""
Performance Panel Component for the Trading Dashboard

This module provides the main layout for the performance panel.
"""

from dash import dcc, html
import dash_bootstrap_components as dbc


def create_performance_panel() -> html.Div:
    """
    Create the performance panel layout.

    Returns:
        Dash HTML Div containing the performance panel
    """
    return html.Div(
        [
            html.H2("Performance Metrics", className="panel-header"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader("Key Metrics"),
                                    dbc.CardBody(id="performance-metrics-card"),
                                ]
                            )
                        ],
                        width=5,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader("Equity Curve"),
                                    dbc.CardBody([dcc.Graph(id="equity-curve-graph")]),
                                ]
                            )
                        ],
                        width=7,
                    ),
                ]
            ),
            html.Br(),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader("Drawdown Analysis"),
                                    dbc.CardBody([dcc.Graph(id="drawdown-graph")]),
                                ]
                            )
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader("Return Distribution"),
                                    dbc.CardBody(
                                        [dcc.Graph(id="return-distribution-graph")]
                                    ),
                                ]
                            )
                        ],
                        width=6,
                    ),
                ]
            ),
            html.Br(),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader("Daily Performance"),
                                    dbc.CardBody(
                                        [dcc.Graph(id="daily-performance-graph")]
                                    ),
                                ]
                            )
                        ],
                        width=12,
                    )
                ]
            ),
            dcc.Interval(
                id="performance-update-interval",
                interval=30 * 1000,  # 30 seconds in milliseconds
                n_intervals=0,
            ),
        ],
        id="performance-panel",
    )
