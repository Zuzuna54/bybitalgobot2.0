"""
Order handling functionality for the Algorithmic Trading System.

This module provides functions and classes for handling different types of orders,
including market, limit, stop, and take profit orders.
"""

from typing import Dict, Any, Optional
from enum import Enum
import time
from datetime import datetime
from loguru import logger

from src.api.bybit_client import BybitClient


class OrderType(Enum):
    """Order types for trade execution."""
    MARKET = "Market"
    LIMIT = "Limit"
    STOP_MARKET = "StopMarket"
    STOP_LIMIT = "StopLimit"
    TAKE_PROFIT_MARKET = "TakeProfitMarket"
    TAKE_PROFIT_LIMIT = "TakeProfitLimit"


class OrderSide(Enum):
    """Order side - buy or sell."""
    BUY = "Buy"
    SELL = "Sell"


def create_market_order(
    api_client: BybitClient,
    symbol: str,
    side: OrderSide,
    qty: float,
    reduce_only: bool = False,
    simulate: bool = False
) -> Dict[str, Any]:
    """
    Create a market order.
    
    Args:
        api_client: Bybit API client
        symbol: Trading pair symbol
        side: Order side (BUY or SELL)
        qty: Order quantity
        reduce_only: Whether this order should only reduce position
        simulate: Whether to simulate the order instead of placing it
        
    Returns:
        Order result dictionary
    """
    if simulate:
        order_id = f"sim-{symbol}-{side.value}-{datetime.now().timestamp()}"
        return {
            "orderId": order_id,
            "symbol": symbol,
            "side": side.value,
            "price": None,
            "qty": qty,
            "orderStatus": "Filled",
            "timeInForce": "GTC",
            "orderType": OrderType.MARKET.value,
            "reduceOnly": reduce_only
        }
    
    try:
        order_result = api_client.place_order(
            symbol=symbol,
            side=side.value,
            order_type=OrderType.MARKET.value,
            qty=qty,
            price=None,  # Market order doesn't need price
            reduce_only=reduce_only,
            time_in_force="GTC"
        )
        
        logger.info(f"Market order placed: {order_result}")
        return order_result
    
    except Exception as e:
        logger.error(f"Error creating market order: {e}")
        return {
            "error": str(e),
            "symbol": symbol,
            "side": side.value,
            "qty": qty,
            "orderStatus": "Rejected"
        }


def create_limit_order(
    api_client: BybitClient,
    symbol: str,
    side: OrderSide,
    qty: float,
    price: float,
    reduce_only: bool = False,
    post_only: bool = False,
    simulate: bool = False
) -> Dict[str, Any]:
    """
    Create a limit order.
    
    Args:
        api_client: Bybit API client
        symbol: Trading pair symbol
        side: Order side (BUY or SELL)
        qty: Order quantity
        price: Limit price
        reduce_only: Whether this order should only reduce position
        post_only: Whether this order should be post-only
        simulate: Whether to simulate the order instead of placing it
        
    Returns:
        Order result dictionary
    """
    if simulate:
        order_id = f"sim-{symbol}-{side.value}-{datetime.now().timestamp()}"
        return {
            "orderId": order_id,
            "symbol": symbol,
            "side": side.value,
            "price": price,
            "qty": qty,
            "orderStatus": "New",
            "timeInForce": "PostOnly" if post_only else "GTC",
            "orderType": OrderType.LIMIT.value,
            "reduceOnly": reduce_only
        }
    
    try:
        time_in_force = "PostOnly" if post_only else "GTC"
        
        order_result = api_client.place_order(
            symbol=symbol,
            side=side.value,
            order_type=OrderType.LIMIT.value,
            qty=qty,
            price=price,
            reduce_only=reduce_only,
            time_in_force=time_in_force
        )
        
        logger.info(f"Limit order placed: {order_result}")
        return order_result
    
    except Exception as e:
        logger.error(f"Error creating limit order: {e}")
        return {
            "error": str(e),
            "symbol": symbol,
            "side": side.value,
            "price": price,
            "qty": qty,
            "orderStatus": "Rejected"
        }


def create_stop_order(
    api_client: BybitClient,
    symbol: str,
    side: OrderSide,
    qty: float,
    stop_price: float,
    reduce_only: bool = True,
    simulate: bool = False
) -> Dict[str, Any]:
    """
    Create a stop market order.
    
    Args:
        api_client: Bybit API client
        symbol: Trading pair symbol
        side: Order side (BUY or SELL)
        qty: Order quantity
        stop_price: Stop trigger price
        reduce_only: Whether this order should only reduce position
        simulate: Whether to simulate the order instead of placing it
        
    Returns:
        Order result dictionary
    """
    if simulate:
        order_id = f"sim-{symbol}-{side.value}-stop-{datetime.now().timestamp()}"
        return {
            "orderId": order_id,
            "symbol": symbol,
            "side": side.value,
            "stopPx": stop_price,
            "qty": qty,
            "orderStatus": "New",
            "timeInForce": "GTC",
            "orderType": OrderType.STOP_MARKET.value,
            "reduceOnly": reduce_only
        }
    
    try:
        order_result = api_client.place_order(
            symbol=symbol,
            side=side.value,
            order_type=OrderType.STOP_MARKET.value,
            qty=qty,
            price=None,
            stop_px=stop_price,
            reduce_only=reduce_only,
            time_in_force="GTC"
        )
        
        logger.info(f"Stop order placed: {order_result}")
        return order_result
    
    except Exception as e:
        logger.error(f"Error creating stop order: {e}")
        return {
            "error": str(e),
            "symbol": symbol,
            "side": side.value,
            "stopPx": stop_price,
            "qty": qty,
            "orderStatus": "Rejected"
        }


def create_take_profit_order(
    api_client: BybitClient,
    symbol: str,
    side: OrderSide,
    qty: float,
    take_profit_price: float,
    reduce_only: bool = True,
    simulate: bool = False
) -> Dict[str, Any]:
    """
    Create a take profit market order.
    
    Args:
        api_client: Bybit API client
        symbol: Trading pair symbol
        side: Order side (BUY or SELL)
        qty: Order quantity
        take_profit_price: Take profit trigger price
        reduce_only: Whether this order should only reduce position
        simulate: Whether to simulate the order instead of placing it
        
    Returns:
        Order result dictionary
    """
    if simulate:
        order_id = f"sim-{symbol}-{side.value}-tp-{datetime.now().timestamp()}"
        return {
            "orderId": order_id,
            "symbol": symbol,
            "side": side.value,
            "stopPx": take_profit_price,
            "qty": qty,
            "orderStatus": "New",
            "timeInForce": "GTC",
            "orderType": OrderType.TAKE_PROFIT_MARKET.value,
            "reduceOnly": reduce_only
        }
    
    try:
        order_result = api_client.place_order(
            symbol=symbol,
            side=side.value,
            order_type=OrderType.TAKE_PROFIT_MARKET.value,
            qty=qty,
            price=None,
            stop_px=take_profit_price,
            reduce_only=reduce_only,
            time_in_force="GTC"
        )
        
        logger.info(f"Take profit order placed: {order_result}")
        return order_result
    
    except Exception as e:
        logger.error(f"Error creating take profit order: {e}")
        return {
            "error": str(e),
            "symbol": symbol,
            "side": side.value,
            "stopPx": take_profit_price,
            "qty": qty,
            "orderStatus": "Rejected"
        }


def update_stop_loss_order(trade, api_client: BybitClient, simulate: bool = False) -> bool:
    """
    Update stop loss order for a trade.
    
    Args:
        trade: Trade object
        api_client: Bybit API client
        simulate: Whether to simulate the update instead of placing it
        
    Returns:
        True if update was successful, False otherwise
    """
    if simulate:
        # No need to update orders in simulation mode
        return True
    
    try:
        # Find existing stop loss order ID
        sl_order_id = None
        for order_id, order_data in trade.orders.items():
            if (
                order_data.get("orderType") == OrderType.STOP_MARKET.value
                and order_data.get("orderStatus") not in ["Filled", "Canceled"]
            ):
                sl_order_id = order_id
                break
        
        if sl_order_id:
            # Cancel existing stop loss order
            cancel_result = cancel_order(api_client, trade.symbol, sl_order_id)
            
            # Place new stop loss order
            sl_side = OrderSide.SELL if trade.side == OrderSide.BUY else OrderSide.BUY
            new_sl_order_result = create_stop_order(
                api_client=api_client,
                symbol=trade.symbol,
                side=sl_side,
                qty=trade.position_size,
                stop_price=trade.stop_loss_price,
                reduce_only=True,
                simulate=False
            )
            
            logger.info(f"New stop loss order placed: {new_sl_order_result}")
            
            if "orderId" in new_sl_order_result:
                new_sl_order_id = new_sl_order_result["orderId"]
                trade.update_order(new_sl_order_id, new_sl_order_result)
                return True
        
        return False
    
    except Exception as e:
        logger.error(f"Error updating stop loss order: {e}")
        return False


def cancel_order(api_client: BybitClient, symbol: str, order_id: str) -> Dict[str, Any]:
    """
    Cancel an order.
    
    Args:
        api_client: Bybit API client
        symbol: Trading pair symbol
        order_id: Order ID
        
    Returns:
        Cancel result dictionary
    """
    try:
        cancel_result = api_client.cancel_order(symbol, order_id)
        logger.info(f"Order canceled: {cancel_result}")
        return cancel_result
    
    except Exception as e:
        logger.error(f"Error canceling order: {e}")
        return {
            "error": str(e),
            "symbol": symbol,
            "orderId": order_id,
            "status": "Failed"
        }


def get_order_status(api_client: BybitClient, symbol: str, order_id: str) -> Dict[str, Any]:
    """
    Get order status.
    
    Args:
        api_client: Bybit API client
        symbol: Trading pair symbol
        order_id: Order ID
        
    Returns:
        Order status dictionary
    """
    try:
        order_status = api_client.get_order_status(symbol, order_id)
        return order_status
    
    except Exception as e:
        logger.error(f"Error getting order status: {e}")
        return {
            "error": str(e),
            "symbol": symbol,
            "orderId": order_id,
            "orderStatus": "Unknown"
        } 