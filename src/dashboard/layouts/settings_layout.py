"""
Settings Panel Layout Module

This module provides the layout for the settings panel in the dashboard.
It allows configuration of risk parameters, system settings, and debugging options.
"""

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from loguru import logger
from typing import Dict, Any, Optional
from src.dashboard.router.callback_registry import callback_registrar


def create_settings_panel() -> html.Div:
    """
    Create the settings panel layout.

    Returns:
        Dash HTML Div containing the settings panel layout
    """
    # Create risk management settings card
    risk_settings = dbc.Card(
        [
            dbc.CardHeader("Risk Management Settings"),
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("Position Size (% of Balance)"),
                                    dcc.Slider(
                                        id="position-size-slider",
                                        min=0.1,
                                        max=10,
                                        step=0.1,
                                        value=2.0,
                                        marks={i: f"{i}%" for i in range(0, 11, 2)},
                                    ),
                                    html.Div(
                                        id="position-size-output",
                                        className="slider-output",
                                    ),
                                ],
                                md=6,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Max Daily Drawdown (%)"),
                                    dcc.Slider(
                                        id="max-drawdown-slider",
                                        min=0.5,
                                        max=5,
                                        step=0.5,
                                        value=2.0,
                                        marks={i: f"{i}%" for i in range(0, 6, 1)},
                                    ),
                                    html.Div(
                                        id="max-drawdown-output",
                                        className="slider-output",
                                    ),
                                ],
                                md=6,
                            ),
                        ]
                    ),
                    html.Hr(),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("Default Leverage"),
                                    dbc.Input(
                                        id="default-leverage-input",
                                        type="number",
                                        value=2,
                                    ),
                                ],
                                md=3,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Max Open Positions"),
                                    dbc.Input(
                                        id="max-positions-input", type="number", value=5
                                    ),
                                ],
                                md=3,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Stop Loss (ATR Multiplier)"),
                                    dbc.Input(
                                        id="stop-loss-atr-input",
                                        type="number",
                                        value=2.0,
                                    ),
                                ],
                                md=3,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Risk/Reward Ratio"),
                                    dbc.Input(
                                        id="risk-reward-input", type="number", value=2.0
                                    ),
                                ],
                                md=3,
                            ),
                        ]
                    ),
                    html.Hr(),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("Use Trailing Stop"),
                                    dbc.Checklist(
                                        id="trailing-stop-toggle",
                                        options=[{"label": "Enabled", "value": 1}],
                                        value=[1],
                                        switch=True,
                                    ),
                                ],
                                md=6,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Circuit Breaker (consecutive losses)"),
                                    dbc.Input(
                                        id="circuit-breaker-input",
                                        type="number",
                                        value=3,
                                    ),
                                ],
                                md=6,
                            ),
                        ]
                    ),
                    html.Hr(),
                    dbc.Button(
                        "Save Risk Parameters",
                        id="save-risk-parameters-button",
                        color="primary",
                        className="mt-2",
                    ),
                ]
            ),
        ]
    )

    # Create system information card
    system_info = dbc.Card(
        [
            dbc.CardHeader("System Information"),
            dbc.CardBody(
                [
                    # This will be populated by a callback
                    html.Div(id="system-status-display"),
                    # WebSocket status
                    html.Div(
                        [
                            html.H5("WebSocket Status", className="mt-3"),
                            html.Div(id="websocket-status-display"),
                        ]
                    ),
                    # Data freshness
                    html.Div(
                        [
                            html.H5("Data Freshness", className="mt-3"),
                            html.Div(id="data-freshness-display"),
                        ]
                    ),
                ]
            ),
        ]
    )

    # Create debug tools card (for development only)
    debug_tools = dbc.Card(
        [
            dbc.CardHeader("Debug Tools"),
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label("Test Notifications"),
                                    dbc.Input(
                                        id="debug-notification-message",
                                        type="text",
                                        placeholder="Notification message",
                                        value="This is a test notification",
                                    ),
                                ],
                                md=6,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Notification Type"),
                                    dbc.Select(
                                        id="debug-notification-type",
                                        options=[
                                            {"label": "Info", "value": "info"},
                                            {"label": "Success", "value": "success"},
                                            {"label": "Warning", "value": "warning"},
                                            {"label": "Error", "value": "error"},
                                            {"label": "Trade", "value": "trade"},
                                        ],
                                        value="info",
                                    ),
                                ],
                                md=3,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label("Duration (ms)"),
                                    dbc.Input(
                                        id="debug-notification-duration",
                                        type="number",
                                        min=1000,
                                        max=10000,
                                        step=1000,
                                        value=3000,
                                    ),
                                ],
                                md=3,
                            ),
                        ]
                    ),
                    dbc.Button(
                        "Show Notification",
                        id="debug-show-notification",
                        color="info",
                        className="mt-2 me-2",
                    ),
                    dbc.Button(
                        "Simulate Error",
                        id="debug-simulate-error",
                        color="danger",
                        className="mt-2 me-2",
                    ),
                    dbc.Button(
                        "Test Loading State",
                        id="debug-loading-test",
                        color="warning",
                        className="mt-2 me-2",
                    ),
                ]
            ),
        ]
    )

    # Assemble the complete panel
    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            risk_settings,
                        ],
                        md=8,
                    ),
                    dbc.Col(
                        [
                            system_info,
                            html.Br(),
                            debug_tools,
                        ],
                        md=4,
                    ),
                ]
            ),
        ]
    )


@callback_registrar(name="settings_layout")
def register_settings_callbacks(
    app: dash.Dash, data_service: Optional[Any] = None, **kwargs
) -> None:
    """
    Register callbacks for the settings layout.

    Args:
        app: The Dash application instance
        data_service: Optional data service instance
        **kwargs: Additional keyword arguments
    """
    logger.debug("Registering settings layout callbacks")

    @app.callback(
        dash.Output("position-size-output", "children"),
        [dash.Input("position-size-slider", "value")],
    )
    def update_position_size_output(value):
        """Update the position size output display."""
        return f"Position Size: {value}% of balance"

    @app.callback(
        dash.Output("max-drawdown-output", "children"),
        [dash.Input("max-drawdown-slider", "value")],
    )
    def update_max_drawdown_output(value):
        """Update the max drawdown output display."""
        return f"Max Daily Drawdown: {value}%"

    @app.callback(
        [
            dash.Output("risk-parameters-result", "data"),
            dash.Output("risk-parameters-save-alert", "is_open"),
            dash.Output("add-notification-trigger", "data", allow_duplicate=True),
        ],
        [dash.Input("save-risk-parameters-button", "n_clicks")],
        [
            dash.State("position-size-slider", "value"),
            dash.State("max-drawdown-slider", "value"),
            dash.State("default-leverage-input", "value"),
            dash.State("max-positions-input", "value"),
            dash.State("stop-loss-atr-input", "value"),
            dash.State("risk-reward-input", "value"),
            dash.State("trailing-stop-toggle", "value"),
            dash.State("circuit-breaker-input", "value"),
        ],
        prevent_initial_call=True,
    )
    def save_risk_parameters(
        n_clicks,
        position_size,
        max_drawdown,
        default_leverage,
        max_positions,
        stop_loss_atr,
        risk_reward,
        use_trailing_stop,
        circuit_breaker,
    ):
        """
        Save risk management parameters.

        Args:
            n_clicks: Button click count
            position_size: Max position size as percentage of balance
            max_drawdown: Maximum daily drawdown percentage
            default_leverage: Default leverage value
            max_positions: Maximum number of open positions
            stop_loss_atr: Stop loss ATR multiplier
            risk_reward: Risk/reward ratio
            use_trailing_stop: Whether to use trailing stops
            circuit_breaker: Number of consecutive losses to trigger circuit breaker

        Returns:
            Risk parameters data, alert visibility, and notification
        """
        # In actual implementation, this would call the data provider's set_risk_parameters method

        # For now, just return the parameters
        risk_params = {
            "max_position_size_percent": position_size,
            "max_daily_drawdown_percent": max_drawdown,
            "default_leverage": default_leverage,
            "max_open_positions": max_positions,
            "stop_loss_atr_multiplier": stop_loss_atr,
            "take_profit_risk_reward_ratio": risk_reward,
            "use_trailing_stop": bool(use_trailing_stop),
            "circuit_breaker_consecutive_losses": circuit_breaker,
        }

        # Create notification
        notification = {
            "type": "success",
            "message": "Risk parameters saved successfully",
            "header": "Settings Updated",
            "duration": 4000,
        }

        return risk_params, True, notification

    @app.callback(
        dash.Output("websocket-status-display", "children"),
        [dash.Input("status-update-interval", "n_intervals")],
    )
    def update_websocket_status(n_intervals):
        """
        Update the WebSocket status display.

        Args:
            n_intervals: Update interval counter

        Returns:
            WebSocket status display component
        """
        # For demonstration, using static data
        ws_status = {
            "market_data": {"connected": True, "last_message": "2 seconds ago"},
            "trades": {"connected": True, "last_message": "10 seconds ago"},
            "orderbook": {"connected": False, "last_message": "N/A"},
        }

        rows = []
        for name, status in ws_status.items():
            status_badge = dbc.Badge(
                "Connected" if status["connected"] else "Disconnected",
                color="success" if status["connected"] else "danger",
                className="me-1",
            )

            rows.append(
                html.Tr(
                    [
                        html.Td(name.replace("_", " ").title()),
                        html.Td(status_badge),
                        html.Td(status["last_message"]),
                    ]
                )
            )

        return dbc.Table(
            [
                html.Thead(
                    html.Tr(
                        [
                            html.Th("Stream"),
                            html.Th("Status"),
                            html.Th("Last Message"),
                        ]
                    )
                ),
                html.Tbody(rows),
            ],
            bordered=True,
            size="sm",
        )

    @app.callback(
        dash.Output("data-freshness-display", "children"),
        [dash.Input("status-update-interval", "n_intervals")],
    )
    def update_data_freshness(n_intervals):
        """
        Update the data freshness display.

        Args:
            n_intervals: Update interval counter

        Returns:
            Data freshness display component
        """
        # For demonstration, using static data
        data_freshness = {
            "market_data": {"fresh": True, "updated": "5 seconds ago"},
            "orderbook": {"fresh": True, "updated": "2 seconds ago"},
            "account": {"fresh": True, "updated": "30 seconds ago"},
            "trades": {"fresh": False, "updated": "5 minutes ago"},
        }

        rows = []
        for name, info in data_freshness.items():
            status_badge = dbc.Badge(
                "Fresh" if info["fresh"] else "Stale",
                color="success" if info["fresh"] else "warning",
                className="me-1",
            )

            rows.append(
                html.Tr(
                    [
                        html.Td(name.replace("_", " ").title()),
                        html.Td(status_badge),
                        html.Td(info["updated"]),
                    ]
                )
            )

        return dbc.Table(
            [
                html.Thead(
                    html.Tr(
                        [
                            html.Th("Data Type"),
                            html.Th("Status"),
                            html.Th("Last Updated"),
                        ]
                    )
                ),
                html.Tbody(rows),
            ],
            bordered=True,
            size="sm",
        )
