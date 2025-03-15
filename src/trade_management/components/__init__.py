"""
Trade Management Components for the Algorithmic Trading System

This package contains modular components for managing trades,
including order handling, position tracking, trade lifecycle management,
and execution reporting.
"""

from src.trade_management.components.order_handler import (
    OrderType,
    OrderSide,
    create_market_order,
    create_limit_order,
    create_stop_order,
    create_take_profit_order,
    update_stop_loss_order,
    cancel_order,
    get_order_status
)

from src.trade_management.components.position_tracker import (
    Trade,
    TradeStatus,
    update_position_market_data,
    calculate_unrealized_profit_pct,
    get_active_positions_summary
)

from src.trade_management.components.trade_lifecycle import (
    process_trading_signal,
    execute_trade_entry,
    close_trade_at_market
)

from src.trade_management.components.execution_report import (
    get_trade_summary,
    get_trade_history_dataframe,
    save_trade_history,
    get_trades_by_symbol,
    get_trades_by_strategy,
    get_performance_by_strategy
)

__all__ = [
    # Order handler
    'OrderType',
    'OrderSide',
    'create_market_order',
    'create_limit_order',
    'create_stop_order',
    'create_take_profit_order',
    'update_stop_loss_order',
    'cancel_order',
    'get_order_status',
    
    # Position tracker
    'Trade',
    'TradeStatus',
    'update_position_market_data',
    'calculate_unrealized_profit_pct',
    'get_active_positions_summary',
    
    # Trade lifecycle
    'process_trading_signal',
    'execute_trade_entry',
    'close_trade_at_market',
    
    # Execution report
    'get_trade_summary',
    'get_trade_history_dataframe',
    'save_trade_history',
    'get_trades_by_symbol',
    'get_trades_by_strategy',
    'get_performance_by_strategy'
] 