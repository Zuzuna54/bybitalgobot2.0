"""
Trading Panel Component for the Trading Dashboard

This module is maintained for backward compatibility and re-exports
the functionality from the new modular trading panel components.

For new code, please use the modular imports directly from:
src.dashboard.components.trading
"""

# Re-export from modular structure
from src.dashboard.components.trading import (
    create_trading_panel,
    render_active_trades_table,
    render_pending_orders_table,
    render_trade_history_table,
    create_pnl_by_symbol_graph,
    create_win_loss_by_strategy_graph,
    register_trading_callbacks
)

# For backward compatibility
__all__ = [
    'create_trading_panel',
    'render_active_trades_table',
    'render_pending_orders_table',
    'render_trade_history_table',
    'create_pnl_by_symbol_graph',
    'create_win_loss_by_strategy_graph',
    'register_trading_callbacks'
] 