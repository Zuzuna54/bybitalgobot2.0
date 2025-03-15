"""
Order Service for Bybit API

This module provides functionality for managing orders on Bybit,
including creating, canceling, and querying orders.
"""

from typing import Dict, Any, Optional, List, Union
from loguru import logger

from src.api.bybit.core.connection import ConnectionManager
from src.api.bybit.core.error_handling import with_error_handling
from src.api.bybit.core.rate_limiting import rate_limited
from src.api.bybit.core.api_client import make_request


class OrderService:
    """
    Service for managing orders on Bybit.
    """
    
    def __init__(self, connection_manager: ConnectionManager):
        """
        Initialize the order service.
        
        Args:
            connection_manager: Connection manager instance
        """
        self.connection_manager = connection_manager
    
    @rate_limited('order')
    @with_error_handling
    def place_order(
        self, 
        category: str,
        symbol: str,
        side: str,
        order_type: str,
        qty: Union[str, float],
        price: Optional[Union[str, float]] = None,
        time_in_force: str = "GTC",
        position_idx: Optional[int] = None,
        order_link_id: Optional[str] = None,
        reduce_only: Optional[bool] = None,
        stop_loss: Optional[Union[str, float]] = None,
        take_profit: Optional[Union[str, float]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Place a new order.
        
        Args:
            category: Product type (spot, linear, inverse)
            symbol: Trading pair symbol
            side: Order side (Buy, Sell)
            order_type: Order type (Limit, Market)
            qty: Order quantity
            price: Order price, required for limit orders
            time_in_force: Time in force (GTC, IOC, FOK, PostOnly)
            position_idx: Position index for hedge mode
            order_link_id: Custom order ID
            reduce_only: Whether to close position only
            stop_loss: Stop loss price
            take_profit: Take profit price
            **kwargs: Additional order parameters
            
        Returns:
            Order placement result
        """
        endpoint = "/v5/order/create"
        
        # Prepare the order parameters
        params = {
            "category": category,
            "symbol": symbol,
            "side": side,
            "orderType": order_type,
            "qty": str(qty),
            "timeInForce": time_in_force
        }
        
        # Add optional parameters
        if price is not None:
            params["price"] = str(price)
        
        if position_idx is not None:
            params["positionIdx"] = position_idx
            
        if order_link_id is not None:
            params["orderLinkId"] = order_link_id
            
        if reduce_only is not None:
            params["reduceOnly"] = reduce_only
            
        if stop_loss is not None:
            params["stopLoss"] = str(stop_loss)
            
        if take_profit is not None:
            params["takeProfit"] = str(take_profit)
            
        # Add any additional parameters
        params.update(kwargs)
        
        # Make the request
        response = make_request(
            connection_manager=self.connection_manager,
            method="POST",
            endpoint=endpoint,
            params=params,
            auth_required=True
        )
        
        return response.get('result', {})
    
    @rate_limited('order')
    @with_error_handling
    def cancel_order(
        self, 
        category: str,
        symbol: str,
        order_id: Optional[str] = None,
        order_link_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Cancel an existing order.
        
        Args:
            category: Product type (spot, linear, inverse)
            symbol: Trading pair symbol
            order_id: Order ID
            order_link_id: Custom order ID
            
        Returns:
            Order cancellation result
        """
        endpoint = "/v5/order/cancel"
        
        # Prepare the parameters
        params = {
            "category": category,
            "symbol": symbol
        }
        
        # Either order_id or order_link_id must be provided
        if order_id is not None:
            params["orderId"] = order_id
        elif order_link_id is not None:
            params["orderLinkId"] = order_link_id
        else:
            raise ValueError("Either order_id or order_link_id must be provided")
        
        # Make the request
        response = make_request(
            connection_manager=self.connection_manager,
            method="POST",
            endpoint=endpoint,
            params=params,
            auth_required=True
        )
        
        return response.get('result', {})
    
    @rate_limited('order')
    @with_error_handling
    def cancel_all_orders(
        self, 
        category: str,
        symbol: Optional[str] = None,
        base_coin: Optional[str] = None,
        settle_coin: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Cancel all active orders.
        
        Args:
            category: Product type (spot, linear, inverse)
            symbol: Trading pair symbol
            base_coin: Base coin
            settle_coin: Settle coin
            
        Returns:
            Order cancellation results
        """
        endpoint = "/v5/order/cancel-all"
        
        # Prepare the parameters
        params = {
            "category": category
        }
        
        # Add optional parameters
        if symbol is not None:
            params["symbol"] = symbol
            
        if base_coin is not None:
            params["baseCoin"] = base_coin
            
        if settle_coin is not None:
            params["settleCoin"] = settle_coin
        
        # Make the request
        response = make_request(
            connection_manager=self.connection_manager,
            method="POST",
            endpoint=endpoint,
            params=params,
            auth_required=True
        )
        
        return response.get('result', {})
    
    @rate_limited('order')
    @with_error_handling
    def get_active_orders(
        self, 
        category: str,
        symbol: Optional[str] = None,
        base_coin: Optional[str] = None,
        settle_coin: Optional[str] = None,
        order_id: Optional[str] = None,
        order_link_id: Optional[str] = None,
        order_filter: Optional[str] = None,
        limit: int = 50,
        cursor: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get active orders.
        
        Args:
            category: Product type (spot, linear, inverse)
            symbol: Trading pair symbol
            base_coin: Base coin
            settle_coin: Settle coin
            order_id: Order ID
            order_link_id: Custom order ID
            order_filter: Order filter
            limit: Result limit
            cursor: Cursor for pagination
            
        Returns:
            Active orders data
        """
        endpoint = "/v5/order/realtime"
        
        # Prepare the parameters
        params = {
            "category": category,
            "limit": limit
        }
        
        # Add optional parameters
        if symbol is not None:
            params["symbol"] = symbol
            
        if base_coin is not None:
            params["baseCoin"] = base_coin
            
        if settle_coin is not None:
            params["settleCoin"] = settle_coin
            
        if order_id is not None:
            params["orderId"] = order_id
            
        if order_link_id is not None:
            params["orderLinkId"] = order_link_id
            
        if order_filter is not None:
            params["orderFilter"] = order_filter
            
        if cursor is not None:
            params["cursor"] = cursor
        
        # Make the request
        response = make_request(
            connection_manager=self.connection_manager,
            method="GET",
            endpoint=endpoint,
            params=params,
            auth_required=True
        )
        
        return response.get('result', {})
    
    @rate_limited('order')
    @with_error_handling
    def get_order_history(
        self, 
        category: str,
        symbol: Optional[str] = None,
        base_coin: Optional[str] = None,
        settle_coin: Optional[str] = None,
        order_id: Optional[str] = None,
        order_link_id: Optional[str] = None,
        order_filter: Optional[str] = None,
        order_status: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 50,
        cursor: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get order history.
        
        Args:
            category: Product type (spot, linear, inverse)
            symbol: Trading pair symbol
            base_coin: Base coin
            settle_coin: Settle coin
            order_id: Order ID
            order_link_id: Custom order ID
            order_filter: Order filter
            order_status: Order status
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds
            limit: Result limit
            cursor: Cursor for pagination
            
        Returns:
            Order history data
        """
        endpoint = "/v5/order/history"
        
        # Prepare the parameters
        params = {
            "category": category,
            "limit": limit
        }
        
        # Add optional parameters
        if symbol is not None:
            params["symbol"] = symbol
            
        if base_coin is not None:
            params["baseCoin"] = base_coin
            
        if settle_coin is not None:
            params["settleCoin"] = settle_coin
            
        if order_id is not None:
            params["orderId"] = order_id
            
        if order_link_id is not None:
            params["orderLinkId"] = order_link_id
            
        if order_filter is not None:
            params["orderFilter"] = order_filter
            
        if order_status is not None:
            params["orderStatus"] = order_status
            
        if start_time is not None:
            params["startTime"] = start_time
            
        if end_time is not None:
            params["endTime"] = end_time
            
        if cursor is not None:
            params["cursor"] = cursor
        
        # Make the request
        response = make_request(
            connection_manager=self.connection_manager,
            method="GET",
            endpoint=endpoint,
            params=params,
            auth_required=True
        )
        
        return response.get('result', {})
    
    @rate_limited('order')
    @with_error_handling
    def amend_order(
        self, 
        category: str,
        symbol: str,
        order_id: Optional[str] = None,
        order_link_id: Optional[str] = None,
        price: Optional[Union[str, float]] = None,
        qty: Optional[Union[str, float]] = None,
        take_profit: Optional[Union[str, float]] = None,
        stop_loss: Optional[Union[str, float]] = None,
        tp_trigger_by: Optional[str] = None,
        sl_trigger_by: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Amend an existing order.
        
        Args:
            category: Product type (spot, linear, inverse)
            symbol: Trading pair symbol
            order_id: Order ID
            order_link_id: Custom order ID
            price: New order price
            qty: New order quantity
            take_profit: New take profit price
            stop_loss: New stop loss price
            tp_trigger_by: Take profit trigger by
            sl_trigger_by: Stop loss trigger by
            **kwargs: Additional parameters
            
        Returns:
            Order amendment result
        """
        endpoint = "/v5/order/amend"
        
        # Prepare the parameters
        params = {
            "category": category,
            "symbol": symbol
        }
        
        # Either order_id or order_link_id must be provided
        if order_id is not None:
            params["orderId"] = order_id
        elif order_link_id is not None:
            params["orderLinkId"] = order_link_id
        else:
            raise ValueError("Either order_id or order_link_id must be provided")
        
        # Add optional parameters
        if price is not None:
            params["price"] = str(price)
            
        if qty is not None:
            params["qty"] = str(qty)
            
        if take_profit is not None:
            params["takeProfit"] = str(take_profit)
            
        if stop_loss is not None:
            params["stopLoss"] = str(stop_loss)
            
        if tp_trigger_by is not None:
            params["tpTriggerBy"] = tp_trigger_by
            
        if sl_trigger_by is not None:
            params["slTriggerBy"] = sl_trigger_by
            
        # Add any additional parameters
        params.update(kwargs)
        
        # Make the request
        response = make_request(
            connection_manager=self.connection_manager,
            method="POST",
            endpoint=endpoint,
            params=params,
            auth_required=True
        )
        
        return response.get('result', {})
        
    @rate_limited('order')
    @with_error_handling
    def get_execution_list(
        self, 
        category: str,
        symbol: Optional[str] = None,
        order_id: Optional[str] = None,
        order_link_id: Optional[str] = None,
        base_coin: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        exec_type: Optional[str] = None,
        limit: int = 50,
        cursor: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get trade execution history.
        
        Args:
            category: Product type (spot, linear, inverse)
            symbol: Trading pair symbol
            order_id: Order ID
            order_link_id: Custom order ID
            base_coin: Base coin
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds
            exec_type: Execution type
            limit: Result limit
            cursor: Cursor for pagination
            
        Returns:
            Execution history data
        """
        endpoint = "/v5/execution/list"
        
        # Prepare the parameters
        params = {
            "category": category,
            "limit": limit
        }
        
        # Add optional parameters
        if symbol is not None:
            params["symbol"] = symbol
            
        if order_id is not None:
            params["orderId"] = order_id
            
        if order_link_id is not None:
            params["orderLinkId"] = order_link_id
            
        if base_coin is not None:
            params["baseCoin"] = base_coin
            
        if start_time is not None:
            params["startTime"] = start_time
            
        if end_time is not None:
            params["endTime"] = end_time
            
        if exec_type is not None:
            params["execType"] = exec_type
            
        if cursor is not None:
            params["cursor"] = cursor
        
        # Make the request
        response = make_request(
            connection_manager=self.connection_manager,
            method="GET",
            endpoint=endpoint,
            params=params,
            auth_required=True
        )
        
        return response.get('result', {})