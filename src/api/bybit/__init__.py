"""
Bybit API Client for the Algorithmic Trading System

This package provides functionality for interacting with the Bybit exchange API,
including market data retrieval, account management, order management, and
WebSocket functionality.
"""

from src.api.bybit.client import BybitClient
from src.api.bybit.core.connection import ConnectionManager
from src.api.bybit.core.api_client import make_request
import functools
import warnings

# Create a global client for backward compatibility
_default_client = BybitClient(testnet=False)

# -------------------------------------------------------------------
# Backward compatibility layer - Market data functions
# -------------------------------------------------------------------

def get_tickers(base_url=None, category="linear", symbol=None):
    """Backward compatible wrapper for market.get_tickers"""
    warnings.warn(
        "This function is deprecated. Please use BybitClient().market.get_tickers() instead.",
        DeprecationWarning, 
        stacklevel=2
    )
    return _default_client.market.get_tickers(category=category, symbol=symbol)

def get_orderbook(base_url=None, symbol=None, limit=50, category="linear"):
    """Backward compatible wrapper for market.get_orderbook"""
    warnings.warn(
        "This function is deprecated. Please use BybitClient().market.get_orderbook() instead.",
        DeprecationWarning, 
        stacklevel=2
    )
    return _default_client.market.get_orderbook(symbol=symbol, limit=limit, category=category)

def get_klines(base_url=None, symbol=None, interval=None, start_time=None, end_time=None, limit=200, category="linear"):
    """Backward compatible wrapper for market.get_klines"""
    warnings.warn(
        "This function is deprecated. Please use BybitClient().market.get_klines() instead.",
        DeprecationWarning, 
        stacklevel=2
    )
    return _default_client.market.get_klines(
        symbol=symbol, 
        interval=interval, 
        start_time=start_time, 
        end_time=end_time, 
        limit=limit, 
        category=category
    )

def get_trades(base_url=None, symbol=None, limit=50, category="linear"):
    """Backward compatible wrapper for market.get_trades"""
    warnings.warn(
        "This function is deprecated. Please use BybitClient().market.get_trades() instead.",
        DeprecationWarning, 
        stacklevel=2
    )
    return _default_client.market.get_trades(symbol=symbol, limit=limit, category=category)

def get_instruments_info(base_url=None, category="linear", symbol=None, status=None):
    """Backward compatible wrapper for market.get_instruments_info"""
    warnings.warn(
        "This function is deprecated. Please use BybitClient().market.get_instruments_info() instead.",
        DeprecationWarning, 
        stacklevel=2
    )
    return _default_client.market.get_instruments_info(category=category, symbol=symbol, status=status)

# -------------------------------------------------------------------
# Backward compatibility layer - Account functions
# -------------------------------------------------------------------

def get_wallet_balance(base_url=None, api_key=None, api_secret=None, account_type="UNIFIED", coin=None):
    """Backward compatible wrapper for account.get_wallet_balance"""
    warnings.warn(
        "This function is deprecated. Please use BybitClient().account.get_wallet_balance() instead.",
        DeprecationWarning, 
        stacklevel=2
    )
    client = _default_client
    if api_key and api_secret:
        client = BybitClient(api_key=api_key, api_secret=api_secret)
    return client.account.get_wallet_balance(account_type=account_type, coin=coin)

def get_positions(base_url=None, api_key=None, api_secret=None, category="linear", symbol=None, settle_coin=None):
    """Backward compatible wrapper for account.get_positions"""
    warnings.warn(
        "This function is deprecated. Please use BybitClient().account.get_positions() instead.",
        DeprecationWarning, 
        stacklevel=2
    )
    client = _default_client
    if api_key and api_secret:
        client = BybitClient(api_key=api_key, api_secret=api_secret)
    return client.account.get_positions(category=category, symbol=symbol, settle_coin=settle_coin)

# -------------------------------------------------------------------
# Backward compatibility layer - Order functions
# -------------------------------------------------------------------

def place_order(base_url=None, api_key=None, api_secret=None, symbol=None, side=None, order_type=None, 
                qty=None, price=None, time_in_force="GTC", reduce_only=False, close_on_trigger=False, 
                stop_loss=None, take_profit=None, **kwargs):
    """Backward compatible wrapper for order.place_order"""
    warnings.warn(
        "This function is deprecated. Please use BybitClient().order.place_order() instead.",
        DeprecationWarning, 
        stacklevel=2
    )
    client = _default_client
    if api_key and api_secret:
        client = BybitClient(api_key=api_key, api_secret=api_secret)
    
    return client.order.place_order(
        category="linear",  # Default to linear for backward compatibility
        symbol=symbol,
        side=side,
        order_type=order_type,
        qty=qty,
        price=price,
        time_in_force=time_in_force,
        reduce_only=reduce_only,
        stop_loss=stop_loss,
        take_profit=take_profit,
        **kwargs
    )

def cancel_order(base_url=None, api_key=None, api_secret=None, symbol=None, order_id=None, order_link_id=None):
    """Backward compatible wrapper for order.cancel_order"""
    warnings.warn(
        "This function is deprecated. Please use BybitClient().order.cancel_order() instead.",
        DeprecationWarning, 
        stacklevel=2
    )
    client = _default_client
    if api_key and api_secret:
        client = BybitClient(api_key=api_key, api_secret=api_secret)
    
    return client.order.cancel_order(
        category="linear",  # Default to linear for backward compatibility
        symbol=symbol,
        order_id=order_id,
        order_link_id=order_link_id
    )

def get_active_orders(base_url=None, api_key=None, api_secret=None, symbol=None, order_id=None, 
                       order_link_id=None, category="linear", limit=50, **kwargs):
    """Backward compatible wrapper for order.get_active_orders"""
    warnings.warn(
        "This function is deprecated. Please use BybitClient().order.get_active_orders() instead.",
        DeprecationWarning, 
        stacklevel=2
    )
    client = _default_client
    if api_key and api_secret:
        client = BybitClient(api_key=api_key, api_secret=api_secret)
    
    return client.order.get_active_orders(
        category=category,
        symbol=symbol,
        order_id=order_id,
        order_link_id=order_link_id,
        limit=limit
    )

def get_order_history(base_url=None, api_key=None, api_secret=None, symbol=None, category="linear", limit=50, **kwargs):
    """Backward compatible wrapper for order.get_order_history"""
    warnings.warn(
        "This function is deprecated. Please use BybitClient().order.get_order_history() instead.",
        DeprecationWarning, 
        stacklevel=2
    )
    client = _default_client
    if api_key and api_secret:
        client = BybitClient(api_key=api_key, api_secret=api_secret)
    
    return client.order.get_order_history(
        category=category,
        symbol=symbol,
        limit=limit
)

__all__ = [
    'BybitClient',
    'ConnectionManager',
    'make_request',
    # Market data functions
    'get_tickers',
    'get_orderbook',
    'get_klines',
    'get_trades',
    'get_instruments_info',
    # Account functions
    'get_wallet_balance',
    'get_positions',
    # Order functions
    'place_order',
    'cancel_order',
    'get_active_orders',
    'get_order_history',
] 