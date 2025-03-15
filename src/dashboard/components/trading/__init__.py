"""
Trading Panel Components

This package provides components for the trading panel in the dashboard,
including active trades, pending orders, and trade history visualizations.
"""

from src.dashboard.components.trading.panel import create_trading_panel
from src.dashboard.components.trading.order_manager import render_pending_orders_table
from src.dashboard.components.trading.position_display import render_active_trades_table, render_trade_history_table
from src.dashboard.components.trading.callbacks import register_trading_callbacks, create_pnl_by_symbol_graph, create_win_loss_by_strategy_graph

__all__ = [
    'create_trading_panel',
    'render_active_trades_table',
    'render_pending_orders_table',
    'render_trade_history_table',
    'create_pnl_by_symbol_graph',
    'create_win_loss_by_strategy_graph',
    'register_trading_callbacks'
] 