"""
Bybit API Client for the Algorithmic Trading System

This package provides functionality for interacting with the Bybit exchange API,
including market data retrieval, account management, order management, and
WebSocket functionality.
"""

from src.api.bybit.client import BybitClient
from src.api.bybit.core.connection import ConnectionManager
from src.api.bybit.core.api_client import make_request
from src.api.bybit.services.data_service import DataService
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
        stacklevel=2,
    )
    return _default_client.market.get_tickers(category=category, symbol=symbol)


def get_orderbook(base_url=None, symbol=None, limit=50, category="linear"):
    """Backward compatible wrapper for market.get_orderbook"""
    warnings.warn(
        "This function is deprecated. Please use BybitClient().market.get_orderbook() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _default_client.market.get_orderbook(
        symbol=symbol, limit=limit, category=category
    )


def get_klines(
    base_url=None,
    symbol=None,
    interval=None,
    start_time=None,
    end_time=None,
    limit=200,
    category="linear",
):
    """Backward compatible wrapper for market.get_klines"""
    warnings.warn(
        "This function is deprecated. Please use BybitClient().market.get_klines() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if not symbol or not interval:
        raise ValueError("Symbol and interval are required parameters")

    return _default_client.market.get_klines(
        symbol=symbol,
        interval=interval,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
        category=category,
    )


def get_trades(base_url=None, symbol=None, limit=50, category="linear"):
    """Backward compatible wrapper for market.get_trades"""
    warnings.warn(
        "This function is deprecated. Please use BybitClient().market.get_trades() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _default_client.market.get_trades(
        symbol=symbol, limit=limit, category=category
    )


def get_instruments_info(base_url=None, category="linear", symbol=None, status=None):
    """Backward compatible wrapper for market.get_instruments_info"""
    warnings.warn(
        "This function is deprecated. Please use BybitClient().market.get_instruments_info() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _default_client.market.get_instruments_info(
        category=category, symbol=symbol, status=status
    )


# -------------------------------------------------------------------
# Backward compatibility layer - Enhanced data service functions
# -------------------------------------------------------------------


def fetch_historical_klines(
    symbol=None, interval=None, start_time=None, end_time=None, use_cache=True
):
    """Backward compatible wrapper for data.fetch_historical_klines"""
    warnings.warn(
        "This function is deprecated. Please use BybitClient().data.fetch_historical_klines() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if not symbol or not interval:
        raise ValueError("Symbol and interval are required parameters")

    return _default_client.data.fetch_historical_klines(
        symbol=symbol,
        interval=interval,
        start_time=start_time,
        end_time=end_time,
        use_cache=use_cache,
    )


def start_ticker_stream(symbol=None, callback=None):
    """Backward compatible wrapper for data.start_ticker_stream"""
    warnings.warn(
        "This function is deprecated. Please use BybitClient().data.start_ticker_stream() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if not symbol:
        raise ValueError("Symbol is a required parameter")

    return _default_client.data.start_ticker_stream(symbol=symbol, callback=callback)


def start_klines_stream(symbol=None, interval=None, callback=None):
    """Backward compatible wrapper for data.start_klines_stream"""
    warnings.warn(
        "This function is deprecated. Please use BybitClient().data.start_klines_stream() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if not symbol or not interval:
        raise ValueError("Symbol and interval are required parameters")

    return _default_client.data.start_klines_stream(
        symbol=symbol, interval=interval, callback=callback
    )


def start_orderbook_stream(symbol=None, depth="50", callback=None):
    """Backward compatible wrapper for data.start_orderbook_stream"""
    warnings.warn(
        "This function is deprecated. Please use BybitClient().data.start_orderbook_stream() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if not symbol:
        raise ValueError("Symbol is a required parameter")

    return _default_client.data.start_orderbook_stream(
        symbol=symbol, depth=depth, callback=callback
    )


def get_current_price(symbol=None):
    """Backward compatible wrapper for data.get_current_price"""
    warnings.warn(
        "This function is deprecated. Please use BybitClient().data.get_current_price() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if not symbol:
        raise ValueError("Symbol is a required parameter")

    return _default_client.data.get_current_price(symbol=symbol)


# -------------------------------------------------------------------
# Backward compatibility layer - Account functions
# -------------------------------------------------------------------


def get_wallet_balance(
    base_url=None, api_key=None, api_secret=None, account_type="UNIFIED", coin=None
):
    """Backward compatible wrapper for account.get_wallet_balance"""
    warnings.warn(
        "This function is deprecated. Please use BybitClient().account.get_wallet_balance() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    client = _get_client_with_auth(base_url, api_key, api_secret)
    return client.account.get_wallet_balance(account_type=account_type, coin=coin)


def get_positions(
    base_url=None,
    api_key=None,
    api_secret=None,
    category="linear",
    symbol=None,
    settle_coin=None,
):
    """Backward compatible wrapper for account.get_positions"""
    warnings.warn(
        "This function is deprecated. Please use BybitClient().account.get_positions() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    client = _get_client_with_auth(base_url, api_key, api_secret)

    kwargs = {}
    if symbol:
        kwargs["symbol"] = symbol
    if settle_coin:
        kwargs["settle_coin"] = settle_coin

    return client.account.get_positions(category=category, **kwargs)


# -------------------------------------------------------------------
# Backward compatibility layer - Order functions
# -------------------------------------------------------------------


def place_order(
    base_url=None,
    api_key=None,
    api_secret=None,
    symbol=None,
    side=None,
    order_type=None,
    qty=None,
    price=None,
    time_in_force="GTC",
    reduce_only=False,
    close_on_trigger=False,
    stop_loss=None,
    take_profit=None,
    **kwargs
):
    """Backward compatible wrapper for order.place_order"""
    warnings.warn(
        "This function is deprecated. Please use BybitClient().order.place_order() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Validate required parameters
    if not all([symbol, side, order_type, qty]):
        raise ValueError("symbol, side, order_type, and qty are required parameters")

    client = _get_client_with_auth(base_url, api_key, api_secret)

    # Build parameters
    params = {
        "symbol": symbol,
        "side": side,
        "order_type": order_type,
        "qty": qty,
        "time_in_force": time_in_force,
        "reduce_only": reduce_only,
        "close_on_trigger": close_on_trigger,
    }

    # Add optional parameters
    if price:
        params["price"] = price
    if stop_loss:
        params["stop_loss"] = stop_loss
    if take_profit:
        params["take_profit"] = take_profit

    # Add any extra parameters
    params.update(kwargs)

    return client.order.place_order(**params)


def cancel_order(
    base_url=None,
    api_key=None,
    api_secret=None,
    symbol=None,
    order_id=None,
    order_link_id=None,
):
    """Backward compatible wrapper for order.cancel_order"""
    warnings.warn(
        "This function is deprecated. Please use BybitClient().order.cancel_order() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Validate required parameters
    if not symbol or (not order_id and not order_link_id):
        raise ValueError(
            "symbol and either order_id or order_link_id are required parameters"
        )

    client = _get_client_with_auth(base_url, api_key, api_secret)

    params = {"symbol": symbol}
    if order_id:
        params["order_id"] = order_id
    if order_link_id:
        params["order_link_id"] = order_link_id

    return client.order.cancel_order(**params)


def get_active_orders(
    base_url=None,
    api_key=None,
    api_secret=None,
    symbol=None,
    order_id=None,
    order_link_id=None,
    category="linear",
    limit=50,
    **kwargs
):
    """Backward compatible wrapper for order.get_active_orders"""
    warnings.warn(
        "This function is deprecated. Please use BybitClient().order.get_active_orders() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    client = _get_client_with_auth(base_url, api_key, api_secret)

    params = {"category": category, "limit": limit}
    if symbol:
        params["symbol"] = symbol
    if order_id:
        params["order_id"] = order_id
    if order_link_id:
        params["order_link_id"] = order_link_id

    # Add any extra parameters
    params.update(kwargs)

    return client.order.get_active_orders(**params)


def get_order_history(
    base_url=None,
    api_key=None,
    api_secret=None,
    symbol=None,
    category="linear",
    limit=50,
    **kwargs
):
    """Backward compatible wrapper for order.get_order_history"""
    warnings.warn(
        "This function is deprecated. Please use BybitClient().order.get_order_history() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    client = _get_client_with_auth(base_url, api_key, api_secret)

    params = {"category": category, "limit": limit}
    if symbol:
        params["symbol"] = symbol

    # Add any extra parameters
    params.update(kwargs)

    return client.order.get_order_history(**params)


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------


def _get_client_with_auth(base_url=None, api_key=None, api_secret=None):
    """Get a client with authentication configured."""
    if api_key and api_secret:
        client = BybitClient(api_key=api_key, api_secret=api_secret)
        return client
    return _default_client


__all__ = [
    "BybitClient",
    "ConnectionManager",
    "make_request",
    # Market data functions
    "get_tickers",
    "get_orderbook",
    "get_klines",
    "get_trades",
    "get_instruments_info",
    # Enhanced data service functions
    "fetch_historical_klines",
    "start_ticker_stream",
    "start_klines_stream",
    "start_orderbook_stream",
    "get_current_price",
    # Account functions
    "get_wallet_balance",
    "get_positions",
    # Order functions
    "place_order",
    "cancel_order",
    "get_active_orders",
    "get_order_history",
]
