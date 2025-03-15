"""
Order processing functionality for the paper trading simulator.

This module provides functions for processing various types of orders in the paper trading system.
"""

from typing import Dict, Any, Optional
from datetime import datetime
from loguru import logger


def process_pending_orders(
    pending_orders: Dict[str, Dict[str, Any]], 
    get_current_price_func, 
    close_position_func,
    execute_order_func
) -> None:
    """
    Process any pending limit, stop loss, or take profit orders.
    
    Args:
        pending_orders: Dictionary of pending orders by ID
        get_current_price_func: Function to get current price for a symbol
        close_position_func: Function to close a position
        execute_order_func: Function to execute an order
    """
    pending_order_ids = list(pending_orders.keys())
    
    for order_id in pending_order_ids:
        order = pending_orders[order_id]
        symbol = order["symbol"]
        is_long = order["type"] == "long"
        
        current_price = get_current_price_func(symbol)
        if not current_price:
            continue
        
        # Process different order types
        if order["order_type"] == "limit":
            process_limit_order(order, current_price, pending_orders, execute_order_func)
        elif order["order_type"] == "stop_loss":
            process_stop_order(order, current_price, pending_orders, close_position_func)
        elif order["order_type"] == "take_profit":
            process_take_profit_order(order, current_price, pending_orders, close_position_func)


def process_limit_order(
    order: Dict[str, Any], 
    current_price: float, 
    pending_orders: Dict[str, Dict[str, Any]], 
    execute_order_func
) -> None:
    """
    Process a pending limit order.
    
    Args:
        order: Order details
        current_price: Current market price
        pending_orders: Dictionary of pending orders
        execute_order_func: Function to execute an order
    """
    is_long = order["type"] == "long"
    limit_price = order["price"]
    
    # Check if limit order should be executed
    if (is_long and current_price <= limit_price) or (not is_long and current_price >= limit_price):
        # Execute the limit order
        execute_order_func(order, limit_price)
        
        # Remove from pending orders
        del pending_orders[order["id"]]


def process_stop_order(
    order: Dict[str, Any], 
    current_price: float, 
    pending_orders: Dict[str, Dict[str, Any]], 
    close_position_func
) -> None:
    """
    Process a pending stop loss order.
    
    Args:
        order: Order details
        current_price: Current market price
        pending_orders: Dictionary of pending orders
        close_position_func: Function to close a position
    """
    is_long = order["type"] == "long"
    stop_price = order["price"]
    
    # Check if stop loss should be triggered
    if (is_long and current_price <= stop_price) or (not is_long and current_price >= stop_price):
        # Execute the stop loss
        close_position_func(order["symbol"], current_price, "stop_loss")
        
        # Remove from pending orders
        del pending_orders[order["id"]]


def process_take_profit_order(
    order: Dict[str, Any], 
    current_price: float, 
    pending_orders: Dict[str, Dict[str, Any]], 
    close_position_func
) -> None:
    """
    Process a pending take profit order.
    
    Args:
        order: Order details
        current_price: Current market price
        pending_orders: Dictionary of pending orders
        close_position_func: Function to close a position
    """
    is_long = order["type"] == "long"
    take_profit_price = order["price"]
    
    # Check if take profit should be triggered
    if (is_long and current_price >= take_profit_price) or (not is_long and current_price <= take_profit_price):
        # Execute the take profit
        close_position_func(order["symbol"], current_price, "take_profit")
        
        # Remove from pending orders
        del pending_orders[order["id"]]


def calculate_execution_price(symbol: str, is_long: bool, slippage: float, get_current_price_func) -> float:
    """
    Calculate execution price with slippage.
    
    Args:
        symbol: Trading pair symbol
        is_long: Whether the trade is long or short
        slippage: Slippage amount as a decimal
        get_current_price_func: Function to get current price for a symbol
        
    Returns:
        Execution price with slippage
    """
    current_price = get_current_price_func(symbol)
    
    if not current_price:
        logger.error(f"Cannot get current price for {symbol}")
        return 0.0
    
    # Apply slippage
    if is_long:
        execution_price = current_price * (1 + slippage)
    else:
        execution_price = current_price * (1 - slippage)
    
    return execution_price 