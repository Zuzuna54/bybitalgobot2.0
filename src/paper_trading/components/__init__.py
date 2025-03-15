"""
Paper Trading Components for the Algorithmic Trading System

This package contains the modular components used by the paper trading simulator.
"""

from src.paper_trading.components.order_processor import (
    process_pending_orders,
    process_limit_order,
    process_stop_order,
    process_take_profit_order,
    calculate_execution_price
)

from src.paper_trading.components.position_manager import (
    update_positions,
    close_position,
    update_stop_loss_in_trade_history,
    calculate_total_equity
)

from src.paper_trading.components.execution_engine import (
    execute_paper_trade,
    get_market_data,
    process_strategy_signals
)

from src.paper_trading.components.state_manager import (
    save_state,
    load_state,
    json_serializer,
    get_summary,
    compare_to_backtest
)

__all__ = [
    # Order processor
    'process_pending_orders',
    'process_limit_order',
    'process_stop_order',
    'process_take_profit_order',
    'calculate_execution_price',
    
    # Position manager
    'update_positions',
    'close_position',
    'update_stop_loss_in_trade_history',
    'calculate_total_equity',
    
    # Execution engine
    'execute_paper_trade',
    'get_market_data',
    'process_strategy_signals',
    
    # State manager
    'save_state',
    'load_state',
    'json_serializer',
    'get_summary',
    'compare_to_backtest'
] 