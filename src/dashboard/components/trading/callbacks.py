"""
Trading Panel Callbacks

This module provides dashboard callbacks and visualization functions for the trading panel.
"""

from typing import Dict, Any, List, Callable, Optional
import dash
import pandas as pd
from dash.dependencies import Input, Output
from dash import html
from loguru import logger

from src.dashboard.router.callback_registry import callback_registrar
from src.dashboard.components.trading.position_display import (
    render_active_trades_table,
    render_trade_history_table,
)
from src.dashboard.components.trading.order_manager import render_pending_orders_table
from src.dashboard.services.chart_service import (
    create_pnl_by_symbol_graph,
    create_win_loss_by_strategy_graph,
)


@callback_registrar(name="trading")
def register_trading_callbacks(
    app: dash.Dash, data_service: Optional[Any] = None, **kwargs
) -> None:
    """
    Register all trading panel callbacks.

    Args:
        app: Dash application instance
        data_service: Data service instance
        **kwargs: Additional keyword arguments
    """
    logger.debug("Registering trading callbacks")

    # Get the trade data function from kwargs or data_service
    get_trade_data_func = kwargs.get("get_trade_data_func")
    if not get_trade_data_func and data_service:
        get_trade_data_func = getattr(data_service, "get_trade_data", None)

    if not get_trade_data_func:
        logger.warning("No trade data function provided, using empty function")
        get_trade_data_func = lambda: {}

    @app.callback(
        [
            Output("active-trades-content", "children"),
            Output("pending-orders-content", "children"),
            Output("trade-history-content", "children"),
            Output("pnl-by-symbol-graph", "figure"),
            Output("win-loss-by-strategy-graph", "figure"),
        ],
        [Input("trading-update-interval", "n_intervals")],
    )
    def update_trading_panel(n_intervals):
        """
        Update the trading panel components.

        Args:
            n_intervals: Number of interval updates

        Returns:
            Tuple of active trades content, pending orders content, trade history content,
            PnL by symbol graph, and win/loss by strategy graph
        """
        try:
            # Get trade data
            trade_data = get_trade_data_func()

            if not trade_data:
                return (
                    html.Div(
                        "No active trades data available", className="no-data-message"
                    ),
                    html.Div(
                        "No pending orders data available", className="no-data-message"
                    ),
                    html.Div(
                        "No trade history data available", className="no-data-message"
                    ),
                    create_pnl_by_symbol_graph([]),
                    create_win_loss_by_strategy_graph([]),
                )

            # Extract data components
            active_trades = trade_data.get("active_trades", [])
            pending_orders = trade_data.get("pending_orders", [])
            trade_history = trade_data.get("trade_history", [])

            # Render components
            active_trades_table = render_active_trades_table(active_trades)
            pending_orders_table = render_pending_orders_table(pending_orders)
            trade_history_table = render_trade_history_table(trade_history)

            # Create visualizations
            pnl_graph = create_pnl_by_symbol_graph(active_trades)
            win_loss_graph = create_win_loss_by_strategy_graph(trade_history)

            return (
                active_trades_table,
                pending_orders_table,
                trade_history_table,
                pnl_graph,
                win_loss_graph,
            )
        except Exception as e:
            logger.error(f"Error updating trading panel: {str(e)}")
            return (
                html.Div(
                    f"Error loading active trades: {str(e)}", className="error-message"
                ),
                html.Div(
                    f"Error loading pending orders: {str(e)}", className="error-message"
                ),
                html.Div(
                    f"Error loading trade history: {str(e)}", className="error-message"
                ),
                create_pnl_by_symbol_graph([]),
                create_win_loss_by_strategy_graph([]),
            )
