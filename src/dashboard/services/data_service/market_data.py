"""
Market Data Module

This module provides functions to retrieve and process market-related data
for the dashboard, including orderbook and price data.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loguru import logger


def _initialize_market_data(service):
    """
    Initialize market data storage with sample data for standalone mode.

    Args:
        service: DashboardDataService instance
    """
    # Initialize orderbook data for popular symbols
    service._orderbook_data = {
        "BTCUSDT": _generate_sample_orderbook("BTCUSDT"),
        "ETHUSDT": _generate_sample_orderbook("ETHUSDT"),
        "SOLUSDT": _generate_sample_orderbook("SOLUSDT"),
    }

    # Initialize market data cache
    service._market_data_cache = {
        "BTCUSDT": _generate_sample_market_data("BTCUSDT"),
        "ETHUSDT": _generate_sample_market_data("ETHUSDT"),
        "SOLUSDT": _generate_sample_market_data("SOLUSDT"),
    }

    # Set initial update timestamps
    service._data_updated_at["orderbook"] = datetime.now()
    service._data_updated_at["market"] = datetime.now()

    logger.debug("Initialized sample market data")


def _update_orderbook_data(service, symbol: Optional[str] = None):
    """
    Update orderbook data for the specified symbol or all symbols.

    Args:
        service: DashboardDataService instance
        symbol: Optional symbol to update, or None to update all
    """
    if service.standalone_mode:
        # In standalone mode, regenerate sample data
        if symbol:
            if symbol in service._orderbook_data:
                service._orderbook_data[symbol] = _generate_sample_orderbook(symbol)
        else:
            # Update all symbols
            for sym in service._orderbook_data.keys():
                service._orderbook_data[sym] = _generate_sample_orderbook(sym)

        # Update timestamp
        service._data_updated_at["orderbook"] = datetime.now()
        return

    try:
        # Get data from API client or orderbook analyzer
        if service.api_client or service.orderbook_analyzer:
            logger.debug(f"Updating orderbook data for symbol: {symbol or 'all'}")

            symbols_to_update = (
                [symbol] if symbol else list(service._orderbook_data.keys())
            )

            for sym in symbols_to_update:
                if service.orderbook_analyzer:
                    orderbook = service.orderbook_analyzer.get_orderbook(sym)
                elif service.api_client:
                    orderbook = service.api_client.market.get_orderbook(symbol=sym)
                else:
                    continue

                if orderbook:
                    service._orderbook_data[sym] = orderbook

        # Update timestamp
        service._data_updated_at["orderbook"] = datetime.now()

    except Exception as e:
        logger.error(f"Error updating orderbook data: {str(e)}")
        # Keep using existing data if update fails


def _update_market_data(service, symbol: Optional[str] = None):
    """
    Update market data for the specified symbol or all symbols.

    Args:
        service: DashboardDataService instance
        symbol: Optional symbol to update, or None to update all
    """
    if service.standalone_mode:
        # In standalone mode, regenerate sample data
        if symbol:
            if symbol in service._market_data_cache:
                service._market_data_cache[symbol] = _generate_sample_market_data(
                    symbol
                )
        else:
            # Update all symbols
            for sym in service._market_data_cache.keys():
                service._market_data_cache[sym] = _generate_sample_market_data(sym)

        # Update timestamp
        service._data_updated_at["market"] = datetime.now()
        return

    try:
        # Get data from API client or market data service
        if service.api_client or service.market_data:
            logger.debug(f"Updating market data for symbol: {symbol or 'all'}")

            symbols_to_update = (
                [symbol] if symbol else list(service._market_data_cache.keys())
            )

            for sym in symbols_to_update:
                market_data = {}

                # Get ticker data
                if service.api_client:
                    ticker = service.api_client.market.get_tickers(symbol=sym)
                    if ticker and "result" in ticker and ticker["result"]:
                        ticker_data = ticker["result"]
                        market_data["price_data"] = {
                            "symbol": sym,
                            "current_price": float(ticker_data.get("last_price", 0)),
                            "price_change": float(ticker_data.get("price_24h_pcnt", 0)),
                            "price_change_pct": float(
                                ticker_data.get("price_24h_pcnt", 0)
                            )
                            * 100,
                            "high_24h": float(ticker_data.get("high_price_24h", 0)),
                            "low_24h": float(ticker_data.get("low_price_24h", 0)),
                            "volume_24h": float(ticker_data.get("volume_24h", 0)),
                        }

                # Get candle data
                candles = None
                if service.market_data:
                    candles = service.market_data.get_candles(
                        sym, timeframe="1h", limit=100
                    )
                elif service.api_client:
                    candles_response = service.api_client.market.get_klines(
                        symbol=sym, interval="60", limit=100
                    )
                    if (
                        candles_response
                        and "result" in candles_response
                        and candles_response["result"]
                    ):
                        candles_data = candles_response["result"]["list"]
                        candles = pd.DataFrame(
                            candles_data,
                            columns=[
                                "timestamp",
                                "open",
                                "high",
                                "low",
                                "close",
                                "volume",
                                "turnover",
                            ],
                        )
                        candles["timestamp"] = pd.to_datetime(
                            candles["timestamp"], unit="ms"
                        )
                        for col in [
                            "open",
                            "high",
                            "low",
                            "close",
                            "volume",
                            "turnover",
                        ]:
                            candles[col] = pd.to_numeric(candles[col])

                if candles is not None:
                    market_data["candle_data"] = candles

                # Get market stats
                market_data["market_stats"] = {
                    "symbol": sym,
                    "24h_volume": market_data.get("price_data", {}).get(
                        "volume_24h", 0
                    ),
                    "24h_high": market_data.get("price_data", {}).get("high_24h", 0),
                    "24h_low": market_data.get("price_data", {}).get("low_24h", 0),
                }

                service._market_data_cache[sym] = market_data

        # Update timestamp
        service._data_updated_at["market"] = datetime.now()

    except Exception as e:
        logger.error(f"Error updating market data: {str(e)}")
        # Keep using existing data if update fails


def get_orderbook_data(
    self, symbol: Optional[str] = None, depth: int = 50
) -> Dict[str, Any]:
    """
    Get orderbook data for the specified symbol.

    Args:
        symbol: Symbol to get orderbook for, or None to get all symbols
        depth: Maximum number of price levels to include

    Returns:
        Dictionary with orderbook data
    """
    # If no symbol specified, return data for all symbols
    if symbol is None:
        return self._orderbook_data

    # Get orderbook for specified symbol
    if symbol in self._orderbook_data:
        orderbook = self._orderbook_data[symbol]

        # Apply depth limit if specified
        if depth > 0 and "bids" in orderbook and "asks" in orderbook:
            limited_orderbook = orderbook.copy()
            limited_orderbook["bids"] = orderbook["bids"][:depth]
            limited_orderbook["asks"] = orderbook["asks"][:depth]
            return limited_orderbook

        return orderbook

    # If symbol not found, return empty orderbook
    return {"symbol": symbol, "bids": [], "asks": []}


def get_market_data(self, symbol: Optional[str] = None) -> Dict[str, Any]:
    """
    Get market data for the specified symbol.

    Args:
        symbol: Symbol to get market data for, or None to get all symbols

    Returns:
        Dictionary with market data
    """
    # If no symbol specified, return empty data
    if symbol is None:
        return {}

    # Get market data for specified symbol
    if symbol in self._market_data_cache:
        return self._market_data_cache[symbol]

    # If symbol not found, return empty data
    return {}


def _generate_sample_orderbook(symbol: str) -> Dict[str, Any]:
    """
    Generate a sample orderbook for a symbol.

    Args:
        symbol: Symbol to generate orderbook for

    Returns:
        Dictionary with sample orderbook data
    """
    # Base price depends on the symbol
    if symbol == "BTCUSDT":
        base_price = 50000
        price_precision = 1
        size_precision = 3
    elif symbol == "ETHUSDT":
        base_price = 3000
        price_precision = 2
        size_precision = 3
    elif symbol == "SOLUSDT":
        base_price = 100
        price_precision = 3
        size_precision = 1
    else:
        base_price = 100
        price_precision = 3
        size_precision = 2

    # Generate bids (buy orders) - sorted by price descending
    bids = []
    current_price = base_price * (
        1 - np.random.uniform(0.001, 0.002)
    )  # Start slightly below base price

    for i in range(100):
        price = round(current_price - (i * base_price * 0.0005), price_precision)
        size = round(np.random.uniform(0.1, 5.0), size_precision)

        # Larger sizes near the top of the book
        if i < 5:
            size *= 3

        bids.append([price, size])

    # Generate asks (sell orders) - sorted by price ascending
    asks = []
    current_price = base_price * (
        1 + np.random.uniform(0.001, 0.002)
    )  # Start slightly above base price

    for i in range(100):
        price = round(current_price + (i * base_price * 0.0005), price_precision)
        size = round(np.random.uniform(0.1, 5.0), size_precision)

        # Larger sizes near the top of the book
        if i < 5:
            size *= 3

        asks.append([price, size])

    # Create orderbook
    orderbook = {
        "symbol": symbol,
        "bids": bids,
        "asks": asks,
        "timestamp": datetime.now().timestamp() * 1000,  # Milliseconds timestamp
    }

    return orderbook


def _generate_sample_market_data(symbol: str) -> Dict[str, Any]:
    """
    Generate sample market data for a symbol.

    Args:
        symbol: Symbol to generate market data for

    Returns:
        Dictionary with sample market data
    """
    # Base price depends on the symbol
    if symbol == "BTCUSDT":
        base_price = 50000
        price_range = 1000
        daily_volume = 10000
    elif symbol == "ETHUSDT":
        base_price = 3000
        price_range = 100
        daily_volume = 20000
    elif symbol == "SOLUSDT":
        base_price = 100
        price_range = 5
        daily_volume = 30000
    else:
        base_price = 100
        price_range = 5
        daily_volume = 15000

    # Generate price data
    price_change_pct = np.random.uniform(-5, 5)  # -5% to +5%
    price_change = base_price * price_change_pct / 100
    current_price = base_price + price_change

    price_data = {
        "symbol": symbol,
        "current_price": current_price,
        "price_change": price_change,
        "price_change_pct": price_change_pct,
        "high_24h": current_price * (1 + np.random.uniform(0.01, 0.05)),
        "low_24h": current_price * (1 - np.random.uniform(0.01, 0.05)),
        "volume_24h": daily_volume * (1 + np.random.uniform(-0.2, 0.2)),
    }

    # Generate candle data
    candle_data = _generate_sample_candles(symbol, base_price, price_range)

    # Generate market stats
    market_stats = {
        "symbol": symbol,
        "24h_volume": price_data["volume_24h"],
        "24h_high": price_data["high_24h"],
        "24h_low": price_data["low_24h"],
        "open_interest": daily_volume * 2 * (1 + np.random.uniform(-0.1, 0.1)),
        "funding_rate": np.random.uniform(-0.01, 0.01),
        "index_price": current_price * (1 + np.random.uniform(-0.001, 0.001)),
    }

    # Combine all data
    market_data = {
        "price_data": price_data,
        "candle_data": candle_data,
        "market_stats": market_stats,
    }

    return market_data


def _generate_sample_candles(
    symbol: str, base_price: float, price_range: float
) -> pd.DataFrame:
    """
    Generate sample OHLCV candle data for a symbol.

    Args:
        symbol: Symbol to generate candles for
        base_price: Base price for the symbol
        price_range: Approximate price range for movement

    Returns:
        DataFrame with sample candle data
    """
    # Generate 100 hourly candles (about 4 days of data)
    candle_count = 100
    end_time = datetime.now()

    # Calculate start time
    start_time = end_time - timedelta(hours=candle_count)

    # Create timestamps
    timestamps = [start_time + timedelta(hours=i) for i in range(candle_count)]

    # Generate price movement with some trend and volatility
    trend = np.random.choice([-1, 1]) * 0.0001  # Small upward or downward trend
    volatility = price_range / base_price / 10  # Scale volatility based on price range

    # Start with current price and work backward
    prices = [base_price]

    for i in range(1, candle_count):
        # Random walk with drift
        prev_price = prices[-1]
        change = prev_price * (trend + np.random.normal(0, volatility))
        new_price = prev_price + change
        prices.append(new_price)

    # Reverse to get chronological order
    prices.reverse()

    # Generate OHLCV data
    data = []

    for i, timestamp in enumerate(timestamps):
        price = prices[i]

        # Generate candle with reasonable high, low, open, close
        intrabar_volatility = price * 0.005  # 0.5% intrabar volatility

        open_price = price * (1 + np.random.normal(0, 0.002))
        close_price = price * (1 + np.random.normal(0, 0.002))
        high_price = max(open_price, close_price) + abs(
            np.random.normal(0, intrabar_volatility)
        )
        low_price = min(open_price, close_price) - abs(
            np.random.normal(0, intrabar_volatility)
        )

        # Volume proportional to volatility
        volume = (
            base_price
            * (20 + 10 * abs((high_price - low_price) / price))
            * (1 + np.random.uniform(-0.5, 1.0))
        )

        data.append(
            {
                "timestamp": timestamp,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume,
            }
        )

    # Create DataFrame
    df = pd.DataFrame(data)

    return df
