"""
Trading Panel Callbacks

This module provides dashboard callbacks and visualization functions for the trading panel.
"""

from typing import Dict, Any, List, Callable
import dash
import pandas as pd
from dash.dependencies import Input, Output
from dash import html

from src.dashboard.components.trading.position_display import render_active_trades_table, render_trade_history_table
from src.dashboard.components.trading.order_manager import render_pending_orders_table
from src.dashboard.services.chart_service import create_pnl_by_symbol_graph, create_win_loss_by_strategy_graph


def register_trading_callbacks(app: dash.Dash, get_trade_data_func: Callable) -> None:
    """
    Register all trading panel callbacks.
    
    Args:
        app: Dash application instance
        get_trade_data_func: Function to retrieve trade data
    """
    @app.callback(
        [
            Output("active-trades-content", "children"),
            Output("pending-orders-content", "children"),
            Output("trade-history-content", "children"),
            Output("pnl-by-symbol-graph", "figure"),
            Output("win-loss-by-strategy-graph", "figure")
        ],
        [Input("trading-update-interval", "n_intervals")]
    )
    def update_trading_panel(n_intervals):
        # Get trade data
        trade_data = get_trade_data_func()
        
        if not trade_data:
            return (
                html.Div("No active trades data available", className="no-data-message"),
                html.Div("No pending orders data available", className="no-data-message"),
                html.Div("No trade history data available", className="no-data-message"),
                create_pnl_by_symbol_graph([]),
                create_win_loss_by_strategy_graph([])
            )
        
        # Extract data components
        active_trades = trade_data.get('active_trades', [])
        pending_orders = trade_data.get('pending_orders', [])
        trade_history = trade_data.get('trade_history', [])
        
        # Create components
        active_trades_table = render_active_trades_table(active_trades)
        pending_orders_table = render_pending_orders_table(pending_orders)
        trade_history_table = render_trade_history_table(trade_history)
        
        # Create charts using the centralized chart service
        pnl_chart = create_pnl_by_symbol_graph(trade_history)
        win_loss_chart = create_win_loss_by_strategy_graph(trade_history)
        
        return (
            active_trades_table,
            pending_orders_table,
            trade_history_table,
            pnl_chart,
            win_loss_chart
        ) 