"""
Performance Metrics Display Components

This module provides components for displaying performance metrics.
"""

from typing import Dict, Any
from dash import html
import dash_bootstrap_components as dbc


def render_metrics_card(metrics: Dict[str, Any]) -> html.Div:
    """
    Render the performance metrics card content.

    Args:
        metrics: Dictionary containing performance metrics

    Returns:
        HTML Div with formatted metrics
    """
    if not metrics:
        return html.Div("No performance data available")

    # Format metrics for display
    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H4(
                                f"{metrics.get('total_return', 0):.2f}%",
                                className="metric-value",
                            ),
                            html.P("Total Return", className="metric-label"),
                        ],
                        width=4,
                    ),
                    dbc.Col(
                        [
                            html.H4(
                                f"{metrics.get('win_rate', 0) * 100:.1f}%",
                                className="metric-value",
                            ),
                            html.P("Win Rate", className="metric-label"),
                        ],
                        width=4,
                    ),
                    dbc.Col(
                        [
                            html.H4(
                                f"{metrics.get('profit_factor', 0):.2f}",
                                className="metric-value",
                            ),
                            html.P("Profit Factor", className="metric-label"),
                        ],
                        width=4,
                    ),
                ]
            ),
            html.Br(),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H4(
                                f"{metrics.get('max_drawdown', 0):.2f}%",
                                className="metric-value-secondary",
                            ),
                            html.P("Max Drawdown", className="metric-label"),
                        ],
                        width=4,
                    ),
                    dbc.Col(
                        [
                            html.H4(
                                f"{metrics.get('sharpe_ratio', 0):.2f}",
                                className="metric-value-secondary",
                            ),
                            html.P("Sharpe Ratio", className="metric-label"),
                        ],
                        width=4,
                    ),
                    dbc.Col(
                        [
                            html.H4(
                                f"{metrics.get('risk_reward_ratio', 0):.2f}",
                                className="metric-value-secondary",
                            ),
                            html.P("Risk/Reward", className="metric-label"),
                        ],
                        width=4,
                    ),
                ]
            ),
            html.Br(),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H4(
                                f"{metrics.get('total_trades', 0)}",
                                className="metric-value-tertiary",
                            ),
                            html.P("Total Trades", className="metric-label"),
                        ],
                        width=4,
                    ),
                    dbc.Col(
                        [
                            html.H4(
                                f"{metrics.get('avg_trade_duration', '0d')}",
                                className="metric-value-tertiary",
                            ),
                            html.P("Avg Duration", className="metric-label"),
                        ],
                        width=4,
                    ),
                    dbc.Col(
                        [
                            html.H4(
                                f"${metrics.get('pnl_per_day', 0):.2f}",
                                className="metric-value-tertiary",
                            ),
                            html.P("Daily P&L", className="metric-label"),
                        ],
                        width=4,
                    ),
                ]
            ),
        ]
    )
