"""
Position management functionality for the paper trading simulator.

This module provides functions for managing and tracking positions in the paper trading system.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from loguru import logger


def update_positions(
    active_positions: Dict[str, Dict[str, Any]],
    get_current_price_func,
    close_position_func,
    risk_manager
) -> None:
    """
    Update all active positions with current prices and check for exit conditions.

    Args:
        active_positions: Dictionary of active positions by symbol
        get_current_price_func: Function to get current price for a symbol
        close_position_func: Function to close a position
        risk_manager: Risk manager instance for calculating trailing stops
    """
    symbols = list(active_positions.keys())
    
    for symbol in symbols:
        position = active_positions[symbol]
        current_price = get_current_price_func(symbol)
        
        if not current_price:
            continue
        
        is_long = position["type"] == "long"
        entry_price = position["entry_price"]
        
        # Update highest/lowest prices
        if is_long and (position["highest_price"] is None or current_price > position["highest_price"]):
            position["highest_price"] = current_price
        elif not is_long and (position["lowest_price"] is None or current_price < position["lowest_price"]):
            position["lowest_price"] = current_price
        
        # Calculate unrealized PnL
        price_diff = current_price - entry_price if is_long else entry_price - current_price
        unrealized_pnl = price_diff * position["position_size"]
        position["unrealized_pnl"] = unrealized_pnl
        
        # Check for stop loss hit
        if (is_long and current_price <= position["stop_loss"]) or (not is_long and current_price >= position["stop_loss"]):
            close_position_func(symbol, position["stop_loss"], "stop_loss")
            continue
        
        # Check for take profit hit
        if (is_long and current_price >= position["take_profit"]) or (not is_long and current_price <= position["take_profit"]):
            close_position_func(symbol, position["take_profit"], "take_profit")
            continue
        
        # Check if trailing stop should be used
        unrealized_pnl_pct = price_diff / entry_price * 100
        
        if risk_manager.should_use_trailing_stop(unrealized_pnl_pct):
            # Calculate new trailing stop
            new_stop = risk_manager.calculate_trailing_stop(
                current_price=current_price,
                entry_price=entry_price,
                is_long=is_long,
                highest_price=position.get("highest_price"),
                lowest_price=position.get("lowest_price")
            )
            
            # Only move stop in the favorable direction
            if (is_long and new_stop > position["stop_loss"]) or (not is_long and new_stop < position["stop_loss"]):
                position["stop_loss"] = new_stop
                
                # Update the same trade in trade history (this will be done by caller in new implementation)
                update_stop_loss_in_trade_history(position["id"], new_stop, None)


def close_position(
    symbol: str,
    exit_price: float,
    exit_reason: str,
    active_positions: Dict[str, Dict[str, Any]],
    trade_history: List[Dict[str, Any]],
    commission_rate: float,
    risk_manager,
    performance_tracker,
    strategy_manager
) -> Optional[float]:
    """
    Close a position and calculate results.
    
    Args:
        symbol: Symbol to close
        exit_price: Exit price
        exit_reason: Reason for exiting the position
        active_positions: Dictionary of active positions
        trade_history: List of all trades
        commission_rate: Commission rate as a decimal
        risk_manager: Risk manager instance
        performance_tracker: Performance tracker instance
        strategy_manager: Strategy manager instance
    
    Returns:
        Updated account balance or None if position does not exist
    """
    if symbol not in active_positions:
        return None
    
    position = active_positions[symbol]
    is_long = position["type"] == "long"
    
    # Calculate profit/loss
    price_diff = exit_price - position["entry_price"] if is_long else position["entry_price"] - exit_price
    pnl = price_diff * position["position_size"]
    pnl_pct = (price_diff / position["entry_price"]) * 100
    
    # Calculate commission
    commission = exit_price * position["position_size"] * commission_rate
    total_commission = position["commission_paid"] + commission
    
    # Net PnL after commission
    net_pnl = pnl - total_commission
    
    # Calculate new balance (position value minus commission)
    new_balance = position["position_size"] * exit_price - commission
    
    # Update trade history
    for trade in trade_history:
        if trade["id"] == position["id"]:
            trade["exit_time"] = datetime.now()
            trade["exit_price"] = exit_price
            trade["exit_reason"] = exit_reason
            trade["pnl"] = pnl
            trade["pnl_pct"] = pnl_pct
            trade["commission"] = total_commission
            trade["net_pnl"] = net_pnl
            trade["status"] = "closed"
            break
    
    logger.info(
        f"Closed {position['type']} paper position for {symbol} at {exit_price} "
        f"({exit_reason}): PnL = {net_pnl:.2f} ({pnl_pct:.2f}%)"
    )
    
    # Update risk manager
    risk_manager.update_trade_result({
        "realized_pnl": net_pnl,
        "status": "closed"
    })
    
    # Update performance tracker
    performance_tracker.add_trade({
        "id": position["id"],
        "symbol": symbol,
        "side": "Buy" if is_long else "Sell",
        "entry_price": position["entry_price"],
        "exit_price": exit_price,
        "position_size": position["position_size"],
        "entry_time": position["entry_time"],
        "exit_time": datetime.now(),
        "realized_pnl": net_pnl,
        "realized_pnl_percent": pnl_pct,
        "strategy_name": position["strategy_name"],
        "exit_reason": exit_reason,
        "status": "closed"
    })
    
    # Update strategy performance
    strategy_manager.update_strategy_performance({
        "strategy_name": position["strategy_name"],
        "realized_pnl": net_pnl,
        "realized_pnl_percent": pnl_pct,
        "exit_reason": exit_reason
    })
    
    # Remove from active positions
    del active_positions[symbol]
    
    return new_balance


def update_stop_loss_in_trade_history(
    trade_id: str,
    new_stop_loss: float,
    trade_history: List[Dict[str, Any]]
) -> None:
    """
    Update the stop loss price in trade history.

    Args:
        trade_id: ID of the trade to update
        new_stop_loss: New stop loss price
        trade_history: List of all trades (can be None if used outside of this module)
    """
    if not trade_history:
        return
        
    for trade in trade_history:
        if trade["id"] == trade_id:
            trade["stop_loss"] = new_stop_loss
            break


def calculate_total_equity(
    current_balance: float,
    active_positions: Dict[str, Dict[str, Any]]
) -> float:
    """
    Calculate total equity including unrealized PnL.

    Args:
        current_balance: Current account balance
        active_positions: Dictionary of active positions

    Returns:
        Total equity (balance + unrealized PnL)
    """
    # Calculate unrealized PnL from open positions
    unrealized_pnl = sum(
        position.get("unrealized_pnl", 0) for position in active_positions.values()
    )
    
    # Calculate total equity
    return current_balance + unrealized_pnl 