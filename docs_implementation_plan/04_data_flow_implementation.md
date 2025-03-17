# Data Flow Implementation Details

This document provides detailed implementation instructions for ensuring efficient, reliable data movement throughout the trading system, from market data acquisition to performance reporting.

## Table of Contents

1. [Market Data Flow](#1-market-data-flow)
   1. [Real-time Data Acquisition](#11-real-time-data-acquisition)
   2. [Data Distribution System](#12-data-distribution-system)
   3. [Historical Data Management](#13-historical-data-management)
2. [Strategy Signal Flow](#2-strategy-signal-flow)
   1. [Signal Generation Pipeline](#21-signal-generation-pipeline)
   2. [Signal Aggregation System](#22-signal-aggregation-system)
   3. [Decision Making Pipeline](#23-decision-making-pipeline)
3. [Order and Execution Flow](#3-order-and-execution-flow)
   1. [Order Creation Pipeline](#31-order-creation-pipeline)
   2. [Order Execution Tracking](#32-order-execution-tracking)
   3. [Execution Feedback Loop](#33-execution-feedback-loop)
4. [Position and Risk Flow](#4-position-and-risk-flow)
   1. [Position Tracking System](#41-position-tracking-system)
   2. [Risk Metrics Distribution](#42-risk-metrics-distribution)
   3. [Exposure Management](#43-exposure-management)
5. [Performance Data Flow](#5-performance-data-flow)
   1. [Trade Recording System](#51-trade-recording-system)
   2. [Performance Metrics Calculation](#52-performance-metrics-calculation)
   3. [Reporting and Visualization](#53-reporting-and-visualization)

## Introduction

The data flow implementation phase focuses on ensuring all data within the trading system moves efficiently and reliably between components. This document details the specific implementation steps needed to create a robust data pipeline that handles market data, strategy signals, orders, positions, risk metrics, and performance data.

Key objectives for this implementation phase:

1. Establish reliable real-time and historical market data flow to strategies
2. Create a robust strategy signal flow that supports multi-strategy environments
3. Implement comprehensive order and execution tracking
4. Ensure accurate position tracking and risk assessment
5. Build a performance data pipeline for analysis and visualization

Throughout this document, we provide code snippets, implementation guidance, and validation checks to ensure proper functionality of all data flows within the system.

## 1. Market Data Flow

The market data flow establishes how price and market information is acquired, processed, and distributed to various components of the trading system.

### 1.1 Real-time Data Acquisition

#### Implementation Details

1. **Complete WebSocket Integration for Real-time Data**:

```python
# In src/api/bybit/core/websocket_manager.py, enhance WebSocket integration
import json
import threading
import time
import websocket
from typing import Dict, Any, List, Callable, Optional
from enum import Enum
from loguru import logger

class WebSocketChannels(Enum):
    """WebSocket channel types."""
    ORDERBOOK = "orderbook"
    TRADE = "publicTrade"
    KLINE = "kline"
    TICKERS = "tickers"
    POSITION = "position"
    EXECUTION = "execution"
    ORDER = "order"
    WALLET = "wallet"

class WebSocketManager:
    """Manages WebSocket connections for real-time data."""

    def __init__(self, api_key: str = "", api_secret: str = "", use_testnet: bool = True):
        """
        Initialize the WebSocket manager.

        Args:
            api_key: API key for authenticated channels
            api_secret: API secret for authenticated channels
            use_testnet: Whether to use testnet URLs
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.use_testnet = use_testnet

        # Base WebSocket URLs
        self._base_public_url = "wss://stream-testnet.bybit.com/v5/public" if use_testnet else "wss://stream.bybit.com/v5/public"
        self._base_private_url = "wss://stream-testnet.bybit.com/v5/private" if use_testnet else "wss://stream.bybit.com/v5/private"

        # Active connections
        self._connections = {}
        self._callbacks = {}
        self._subscription_info = {}

        # Thread management
        self._threads = {}
        self._running = {}
        self._lock = threading.RLock()

        logger.info("WebSocket manager initialized")

    def subscribe_public(self, channel_type: WebSocketChannels, symbols: List[str],
                        callback: Callable[[Dict[str, Any]], None]) -> str:
        """
        Subscribe to a public channel for multiple symbols.

        Args:
            channel_type: Channel type
            symbols: List of symbols
            callback: Callback function for received data

        Returns:
            Connection ID
        """
        # Validate inputs
        if not symbols or not callback:
            logger.error("Invalid subscription parameters")
            return ""

        # Create connection ID
        conn_id = f"public_{channel_type.value}_{'-'.join(symbols)}"

        with self._lock:
            # Check if already subscribed
            if conn_id in self._connections:
                logger.info(f"Already subscribed to {channel_type.value} for {symbols}")
                self._callbacks[conn_id].append(callback)
                return conn_id

            # Create subscription arguments
            args = []
            for symbol in symbols:
                args.append(f"{channel_type.value}.{symbol}")

            # Create subscription message
            sub_message = {
                "op": "subscribe",
                "args": args
            }

            # Store connection info
            self._subscription_info[conn_id] = {
                "type": "public",
                "channel": channel_type.value,
                "symbols": symbols,
                "subscription_message": sub_message
            }
            self._callbacks[conn_id] = [callback]

            # Initialize the connection
            self._init_connection(conn_id, self._base_public_url, sub_message)

        logger.info(f"Subscribed to {channel_type.value} for {symbols}")
        return conn_id

    def subscribe_private(self, channel_type: WebSocketChannels,
                         callback: Callable[[Dict[str, Any]], None]) -> str:
        """
        Subscribe to a private channel.

        Args:
            channel_type: Channel type
            callback: Callback function for received data

        Returns:
            Connection ID
        """
        # Validate API credentials for private channels
        if not self.api_key or not self.api_secret:
            logger.error("API credentials required for private channel subscription")
            return ""

        # Create connection ID
        conn_id = f"private_{channel_type.value}"

        with self._lock:
            # Check if already subscribed
            if conn_id in self._connections:
                logger.info(f"Already subscribed to private {channel_type.value}")
                self._callbacks[conn_id].append(callback)
                return conn_id

            # Create subscription message
            sub_message = {
                "op": "subscribe",
                "args": [channel_type.value]
            }

            # Store connection info
            self._subscription_info[conn_id] = {
                "type": "private",
                "channel": channel_type.value,
                "subscription_message": sub_message
            }
            self._callbacks[conn_id] = [callback]

            # Initialize the connection with authentication
            self._init_connection(conn_id, self._base_private_url, sub_message, True)

        logger.info(f"Subscribed to private {channel_type.value}")
        return conn_id

    def _init_connection(self, conn_id: str, base_url: str, sub_message: Dict[str, Any],
                         authenticate: bool = False) -> None:
        """
        Initialize a WebSocket connection.

        Args:
            conn_id: Connection ID
            base_url: Base WebSocket URL
            sub_message: Subscription message
            authenticate: Whether to authenticate the connection
        """
        def on_message(ws, message):
            """Handle received messages."""
            try:
                data = json.loads(message)

                # Forward to all registered callbacks
                for callback in self._callbacks.get(conn_id, []):
                    try:
                        callback(data)
                    except Exception as e:
                        logger.error(f"Error in callback for {conn_id}: {str(e)}")

            except json.JSONDecodeError:
                logger.error(f"Failed to parse WebSocket message: {message}")
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {str(e)}")

        def on_error(ws, error):
            """Handle connection errors."""
            logger.error(f"WebSocket error for {conn_id}: {str(error)}")

        def on_close(ws, close_status_code, close_msg):
            """Handle connection close."""
            logger.info(f"WebSocket connection closed for {conn_id}: {close_status_code} - {close_msg}")

            # Attempt to reconnect if this was an unexpected closure
            with self._lock:
                if conn_id in self._running and self._running[conn_id]:
                    logger.info(f"Attempting to reconnect {conn_id} in 5 seconds...")
                    time.sleep(5)
                    self._reconnect(conn_id)

        def on_open(ws):
            """Handle connection open."""
            logger.info(f"WebSocket connection established for {conn_id}")

            # Send subscription message
            ws.send(json.dumps(sub_message))
            logger.debug(f"Sent subscription message for {conn_id}: {sub_message}")

        # Create WebSocket app
        ws = websocket.WebSocketApp(
            base_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )

        # Store connection
        with self._lock:
            self._connections[conn_id] = ws
            self._running[conn_id] = True

            # Start the WebSocket in a separate thread
            thread = threading.Thread(
                target=self._run_websocket,
                args=(conn_id, ws),
                daemon=True
            )
            self._threads[conn_id] = thread
            thread.start()

    def _run_websocket(self, conn_id: str, ws) -> None:
        """
        Run the WebSocket connection.

        Args:
            conn_id: Connection ID
            ws: WebSocket app
        """
        while self._running.get(conn_id, False):
            try:
                ws.run_forever(ping_interval=30, ping_timeout=10)

                # If we get here, the connection has closed
                logger.warning(f"WebSocket connection {conn_id} has closed")

                # If still supposed to be running, reconnect
                if self._running.get(conn_id, False):
                    logger.info(f"Attempting to reconnect {conn_id} in 5 seconds...")
                    time.sleep(5)
                    self._reconnect(conn_id)
                    break

            except Exception as e:
                logger.error(f"Error in WebSocket loop for {conn_id}: {str(e)}")
                time.sleep(5)

    def _reconnect(self, conn_id: str) -> None:
        """
        Reconnect a WebSocket connection.

        Args:
            conn_id: Connection ID
        """
        with self._lock:
            if conn_id not in self._subscription_info:
                logger.error(f"Cannot reconnect unknown connection: {conn_id}")
                return

            info = self._subscription_info[conn_id]

            if info["type"] == "public":
                base_url = self._base_public_url
                authenticate = False
            else:
                base_url = self._base_private_url
                authenticate = True

            # Re-initialize the connection
            self._init_connection(conn_id, base_url, info["subscription_message"], authenticate)
            logger.info(f"Reconnected {conn_id}")

    def unsubscribe(self, conn_id: str) -> bool:
        """
        Unsubscribe from a channel.

        Args:
            conn_id: Connection ID

        Returns:
            Success flag
        """
        with self._lock:
            if conn_id not in self._connections:
                logger.warning(f"Cannot unsubscribe from unknown connection: {conn_id}")
                return False

            # Set the connection to not running
            self._running[conn_id] = False

            # Close the WebSocket
            try:
                self._connections[conn_id].close()
            except Exception as e:
                logger.error(f"Error closing WebSocket for {conn_id}: {str(e)}")

            # Clean up resources
            if conn_id in self._threads and self._threads[conn_id].is_alive():
                self._threads[conn_id].join(timeout=1.0)

            # Remove connection info
            del self._connections[conn_id]
            del self._callbacks[conn_id]
            del self._subscription_info[conn_id]
            del self._running[conn_id]
            del self._threads[conn_id]

            logger.info(f"Unsubscribed from {conn_id}")
            return True

    def close_all(self) -> None:
        """Close all WebSocket connections."""
        logger.info("Closing all WebSocket connections")

        with self._lock:
            for conn_id in list(self._connections.keys()):
                self.unsubscribe(conn_id)

        logger.info("All WebSocket connections closed")
```

2. **Implement Efficient Subscription Management**:

```python
# In src/market_data/manager.py, implement subscription management
class MarketDataManager:
    """Manages market data acquisition and distribution."""

    def __init__(self, api_client, use_websocket=True):
        """
        Initialize the market data manager.

        Args:
            api_client: API client
            use_websocket: Whether to use WebSocket for real-time data
        """
        self.api_client = api_client
        self.use_websocket = use_websocket

        # Initialize WebSocket manager if needed
        if self.use_websocket:
            self.websocket_manager = WebSocketManager(
                api_key=api_client.connection_manager.api_key,
                api_secret=api_client.connection_manager.api_secret,
                use_testnet=api_client.connection_manager.testnet
            )
        else:
            self.websocket_manager = None

        # Data storage
        self.active_subscriptions = {}  # Maps symbol to subscription info
        self.market_data_cache = {}     # Cached market data by symbol and type
        self.kline_cache = {}           # Cached klines by symbol and timeframe
        self.orderbook_cache = {}       # Cached orderbook data by symbol

        # Callbacks for data updates
        self.market_data_callbacks = {}

        # Lock for thread safety
        self.lock = threading.RLock()

        logger.info("Market data manager initialized")

    def subscribe_klines(self, symbol: str, interval: str, callback: Callable[[Dict[str, Any]], None]) -> str:
        """
        Subscribe to kline (candlestick) updates.

        Args:
            symbol: Symbol to subscribe to
            interval: Kline interval (e.g., "1h", "15m")
            callback: Callback for data updates

        Returns:
            Subscription ID
        """
        if not self.use_websocket or not self.websocket_manager:
            logger.warning("WebSocket not enabled, cannot subscribe to klines")
            return ""

        # Format the subscription key
        sub_key = f"kline_{symbol}_{interval}"

        with self.lock:
            # Check if already subscribed
            if sub_key in self.active_subscriptions:
                logger.info(f"Already subscribed to {sub_key}")

                # Add the callback
                if sub_key not in self.market_data_callbacks:
                    self.market_data_callbacks[sub_key] = []
                self.market_data_callbacks[sub_key].append(callback)

                return self.active_subscriptions[sub_key]

            # Create the WebSocket callback
            def ws_callback(data):
                # Process the kline data
                if "data" in data and isinstance(data["data"], list):
                    for kline_data in data["data"]:
                        try:
                            processed_kline = self._process_kline_data(kline_data, symbol, interval)

                            # Update the cache
                            self._update_kline_cache(symbol, interval, processed_kline)

                            # Notify callbacks
                            if sub_key in self.market_data_callbacks:
                                for cb in self.market_data_callbacks[sub_key]:
                                    try:
                                        cb(processed_kline)
                                    except Exception as e:
                                        logger.error(f"Error in kline callback: {str(e)}")

                        except Exception as e:
                            logger.error(f"Error processing kline data: {str(e)}")

            # Subscribe to the WebSocket channel
            conn_id = self.websocket_manager.subscribe_public(
                WebSocketChannels.KLINE,
                [symbol],
                ws_callback
            )

            if not conn_id:
                logger.error(f"Failed to subscribe to klines for {symbol} {interval}")
                return ""

            # Store the subscription
            self.active_subscriptions[sub_key] = conn_id

            # Initialize callback list
            if sub_key not in self.market_data_callbacks:
                self.market_data_callbacks[sub_key] = []
            self.market_data_callbacks[sub_key].append(callback)

            logger.info(f"Subscribed to klines for {symbol} {interval}")
            return conn_id

    def subscribe_orderbook(self, symbol: str, callback: Callable[[Dict[str, Any]], None]) -> str:
        """
        Subscribe to orderbook updates.

        Args:
            symbol: Symbol to subscribe to
            callback: Callback for data updates

        Returns:
            Subscription ID
        """
        if not self.use_websocket or not self.websocket_manager:
            logger.warning("WebSocket not enabled, cannot subscribe to orderbook")
            return ""

        # Format the subscription key
        sub_key = f"orderbook_{symbol}"

        with self.lock:
            # Check if already subscribed
            if sub_key in self.active_subscriptions:
                logger.info(f"Already subscribed to {sub_key}")

                # Add the callback
                if sub_key not in self.market_data_callbacks:
                    self.market_data_callbacks[sub_key] = []
                self.market_data_callbacks[sub_key].append(callback)

                return self.active_subscriptions[sub_key]

            # Create the WebSocket callback
            def ws_callback(data):
                # Process the orderbook data
                if "data" in data and isinstance(data["data"], dict):
                    try:
                        processed_orderbook = self._process_orderbook_data(data["data"], symbol)

                        # Update the cache
                        self._update_orderbook_cache(symbol, processed_orderbook)

                        # Notify callbacks
                        if sub_key in self.market_data_callbacks:
                            for cb in self.market_data_callbacks[sub_key]:
                                try:
                                    cb(processed_orderbook)
                                except Exception as e:
                                    logger.error(f"Error in orderbook callback: {str(e)}")

                    except Exception as e:
                        logger.error(f"Error processing orderbook data: {str(e)}")

            # Subscribe to the WebSocket channel
            conn_id = self.websocket_manager.subscribe_public(
                WebSocketChannels.ORDERBOOK,
                [symbol],
                ws_callback
            )

            if not conn_id:
                logger.error(f"Failed to subscribe to orderbook for {symbol}")
                return ""

            # Store the subscription
            self.active_subscriptions[sub_key] = conn_id

            # Initialize callback list
            if sub_key not in self.market_data_callbacks:
                self.market_data_callbacks[sub_key] = []
            self.market_data_callbacks[sub_key].append(callback)

            logger.info(f"Subscribed to orderbook for {symbol}")
            return conn_id

    def unsubscribe(self, sub_key: str) -> bool:
        """
        Unsubscribe from a data channel.

        Args:
            sub_key: Subscription key

        Returns:
            Success flag
        """
        with self.lock:
            if sub_key not in self.active_subscriptions:
                logger.warning(f"Cannot unsubscribe from unknown subscription: {sub_key}")
                return False

            conn_id = self.active_subscriptions[sub_key]

            # Unsubscribe from the WebSocket
            if self.websocket_manager:
                success = self.websocket_manager.unsubscribe(conn_id)
                if not success:
                    logger.warning(f"Failed to unsubscribe from WebSocket for {sub_key}")
                    return False

            # Clean up resources
            del self.active_subscriptions[sub_key]
            if sub_key in self.market_data_callbacks:
                del self.market_data_callbacks[sub_key]

            logger.info(f"Unsubscribed from {sub_key}")
            return True

    def _process_kline_data(self, data: Dict[str, Any], symbol: str, interval: str) -> Dict[str, Any]:
        """
        Process raw kline data from WebSocket.

        Args:
            data: Raw kline data
            symbol: Symbol
            interval: Interval

        Returns:
            Processed kline data
        """
        # Extract and transform the data
        processed = {
            "symbol": symbol,
            "interval": interval,
            "open_time": data.get("start", 0),
            "open": float(data.get("open", 0)),
            "high": float(data.get("high", 0)),
            "low": float(data.get("low", 0)),
            "close": float(data.get("close", 0)),
            "volume": float(data.get("volume", 0)),
            "close_time": data.get("end", 0),
            "confirmed": data.get("confirm", False)
        }

        return processed

    def _update_kline_cache(self, symbol: str, interval: str, kline: Dict[str, Any]) -> None:
        """
        Update the kline cache.

        Args:
            symbol: Symbol
            interval: Interval
            kline: Kline data
        """
        # Initialize caches if needed
        if symbol not in self.kline_cache:
            self.kline_cache[symbol] = {}

        if interval not in self.kline_cache[symbol]:
            self.kline_cache[symbol][interval] = []

        # Check if we need to update an existing kline or add a new one
        updated = False
        for i, existing_kline in enumerate(self.kline_cache[symbol][interval]):
            if existing_kline["open_time"] == kline["open_time"]:
                self.kline_cache[symbol][interval][i] = kline
                updated = True
                break

        # Add new kline if not updated
        if not updated:
            self.kline_cache[symbol][interval].append(kline)

            # Sort by open time
            self.kline_cache[symbol][interval] = sorted(
                self.kline_cache[symbol][interval],
                key=lambda k: k["open_time"]
            )

            # Limit cache size
            max_klines = 500  # Adjust as needed
            if len(self.kline_cache[symbol][interval]) > max_klines:
                self.kline_cache[symbol][interval] = self.kline_cache[symbol][interval][-max_klines:]
```

3. **Add Data Validation and Normalization**:

```python
# In src/market_data/processor.py, implement data validation and normalization
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger

class MarketDataProcessor:
    """Processes and validates market data."""

    @staticmethod
    def validate_klines(klines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate kline data.

        Args:
            klines: List of kline data

        Returns:
            Validated kline data
        """
        if not isinstance(klines, list):
            logger.error(f"Invalid klines data type: {type(klines)}")
            return []

        validated_klines = []

        for kline in klines:
            if not isinstance(kline, dict):
                logger.warning(f"Invalid kline type: {type(kline)}")
                continue

            # Check for required fields
            required_fields = ["open_time", "open", "high", "low", "close", "volume"]
            missing_fields = [field for field in required_fields if field not in kline]

            if missing_fields:
                logger.warning(f"Kline missing required fields: {missing_fields}")
                continue

            # Validate numeric fields
            numeric_fields = ["open", "high", "low", "close", "volume"]
            valid_numeric = True

            for field in numeric_fields:
                try:
                    kline[field] = float(kline[field])
                except (ValueError, TypeError):
                    logger.warning(f"Invalid value for {field}: {kline.get(field)}")
                    valid_numeric = False
                    break

            if not valid_numeric:
                continue

            # Validate timestamp
            try:
                kline["open_time"] = int(kline["open_time"])
            except (ValueError, TypeError):
                logger.warning(f"Invalid timestamp: {kline.get('open_time')}")
                continue

            # Validate price consistency
            if not (kline["low"] <= kline["open"] <= kline["high"] and
                   kline["low"] <= kline["close"] <= kline["high"]):
                logger.warning(f"Inconsistent price data: {kline}")
                continue

            # Add validated kline
            validated_klines.append(kline)

        return validated_klines

    @staticmethod
    def normalize_klines(klines: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Normalize kline data to a pandas DataFrame.

        Args:
            klines: List of kline data

        Returns:
            Normalized DataFrame
        """
        if not klines:
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(klines)

        # Ensure timestamp columns are datetime
        if "open_time" in df.columns:
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")

        if "close_time" in df.columns:
            df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

        # Set index to timestamp
        if "open_time" in df.columns:
            df.set_index("open_time", inplace=True)

        # Ensure all numeric columns are float
        numeric_columns = ["open", "high", "low", "close", "volume"]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Sort by index
        df.sort_index(inplace=True)

        return df

    @staticmethod
    def validate_orderbook(orderbook: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Validate orderbook data.

        Args:
            orderbook: Orderbook data

        Returns:
            Validated orderbook data or None if invalid
        """
        if not isinstance(orderbook, dict):
            logger.error(f"Invalid orderbook data type: {type(orderbook)}")
            return None

        # Check for required fields
        required_fields = ["s", "ts", "bids", "asks"]
        missing_fields = [field for field in required_fields if field not in orderbook]

        if missing_fields:
            logger.warning(f"Orderbook missing required fields: {missing_fields}")
            return None

        # Validate bid and ask arrays
        if not isinstance(orderbook["bids"], list) or not isinstance(orderbook["asks"], list):
            logger.warning("Invalid bid or ask data type")
            return None

        # Normalize the orderbook
        validated = {
            "symbol": orderbook["s"],
            "timestamp": int(orderbook["ts"]),
            "bids": [],
            "asks": []
        }

        # Validate and transform bids
        for bid in orderbook["bids"]:
            if len(bid) < 2:
                continue

            try:
                price = float(bid[0])
                quantity = float(bid[1])
                validated["bids"].append([price, quantity])
            except (ValueError, TypeError):
                logger.warning(f"Invalid bid data: {bid}")

        # Validate and transform asks
        for ask in orderbook["asks"]:
            if len(ask) < 2:
                continue

            try:
                price = float(ask[0])
                quantity = float(ask[1])
                validated["asks"].append([price, quantity])
            except (ValueError, TypeError):
                logger.warning(f"Invalid ask data: {ask}")

        # Sort bids and asks
        validated["bids"] = sorted(validated["bids"], key=lambda x: x[0], reverse=True)
        validated["asks"] = sorted(validated["asks"], key=lambda x: x[0])

        return validated
```

The real-time data acquisition implementation includes three key components:

1. **WebSocket Integration** - A comprehensive WebSocket manager to handle real-time data streams from Bybit
2. **Subscription Management** - Robust subscription handling for different data types with callbacks
3. **Data Validation** - Thorough data validation and normalization to ensure consistency and reliability

These components work together to provide a robust foundation for real-time market data acquisition that will feed into the trading strategies and other system components.

### 1.2 Data Distribution System

#### Implementation Details

1. **Implement a Publish-Subscribe System for Market Data**:

```python
# In src/market_data/pubsub.py, implement a publish-subscribe system
from typing import Dict, Any, List, Callable, Optional, Set
import threading
from enum import Enum
from loguru import logger

class DataType(Enum):
    """Types of market data."""
    KLINE = "kline"
    ORDERBOOK = "orderbook"
    TICKER = "ticker"
    TRADE = "trade"
    FUNDING = "funding"

class MarketDataPubSub:
    """Publish-subscribe system for market data distribution."""

    def __init__(self):
        """Initialize the publish-subscribe system."""
        # Subscribers by data type, symbol, and interval
        self._subscribers = {}

        # Lock for thread safety
        self._lock = threading.RLock()

        logger.info("Market data publish-subscribe system initialized")

    def subscribe(self, data_type: DataType, symbol: str, interval: Optional[str] = None,
                 callback: Callable[[Dict[str, Any]], None]) -> str:
        """
        Subscribe to market data updates.

        Args:
            data_type: Type of data
            symbol: Symbol
            interval: Interval (for klines)
            callback: Callback for data updates

        Returns:
            Subscription ID
        """
        with self._lock:
            # Create subscription ID
            sub_id = f"{data_type.value}_{symbol}"
            if interval:
                sub_id += f"_{interval}"
            sub_id += f"_{id(callback)}"

            # Initialize subscription paths if needed
            if data_type.value not in self._subscribers:
                self._subscribers[data_type.value] = {}

            if symbol not in self._subscribers[data_type.value]:
                self._subscribers[data_type.value][symbol] = {}

            if interval not in self._subscribers[data_type.value][symbol]:
                self._subscribers[data_type.value][symbol][interval] = {}

            # Add the subscriber
            self._subscribers[data_type.value][symbol][interval][sub_id] = callback

            logger.debug(f"Added subscriber {sub_id} for {data_type.value} {symbol} {interval}")
            return sub_id

    def unsubscribe(self, sub_id: str) -> bool:
        """
        Unsubscribe from market data updates.

        Args:
            sub_id: Subscription ID

        Returns:
            Success flag
        """
        # Parse the subscription ID
        parts = sub_id.split('_')
        if len(parts) < 3:
            logger.warning(f"Invalid subscription ID format: {sub_id}")
            return False

        data_type = parts[0]
        symbol = parts[1]
        interval = parts[2] if len(parts) > 3 else None

        with self._lock:
            try:
                # Remove the subscriber
                if (data_type in self._subscribers and
                    symbol in self._subscribers[data_type] and
                    interval in self._subscribers[data_type][symbol] and
                    sub_id in self._subscribers[data_type][symbol][interval]):

                    del self._subscribers[data_type][symbol][interval][sub_id]

                    # Clean up empty containers
                    if not self._subscribers[data_type][symbol][interval]:
                        del self._subscribers[data_type][symbol][interval]

                    if not self._subscribers[data_type][symbol]:
                        del self._subscribers[data_type][symbol]

                    if not self._subscribers[data_type]:
                        del self._subscribers[data_type]

                    logger.debug(f"Removed subscriber {sub_id}")
                    return True
                else:
                    logger.warning(f"Subscription not found: {sub_id}")
                    return False
            except Exception as e:
                logger.error(f"Error unsubscribing {sub_id}: {str(e)}")
                return False

    def publish(self, data_type: DataType, symbol: str, data: Dict[str, Any],
               interval: Optional[str] = None) -> int:
        """
        Publish market data update to subscribers.

        Args:
            data_type: Type of data
            symbol: Symbol
            data: Data to publish
            interval: Interval (for klines)

        Returns:
            Number of subscribers notified
        """
        notify_count = 0

        with self._lock:
            # Check if we have subscribers for this data type
            if data_type.value not in self._subscribers:
                return 0

            # Check if we have subscribers for this symbol
            if symbol not in self._subscribers[data_type.value]:
                return 0

            # Notify subscribers for this specific interval
            if interval in self._subscribers[data_type.value][symbol]:
                subscribers = self._subscribers[data_type.value][symbol][interval]

                for sub_id, callback in list(subscribers.items()):
                    try:
                        callback(data)
                        notify_count += 1
                    except Exception as e:
                        logger.error(f"Error in subscriber callback {sub_id}: {str(e)}")

            # Also notify subscribers for 'None' interval (all intervals)
            if None in self._subscribers[data_type.value][symbol]:
                subscribers = self._subscribers[data_type.value][symbol][None]

                for sub_id, callback in list(subscribers.items()):
                    try:
                        callback(data)
                        notify_count += 1
                    except Exception as e:
                        logger.error(f"Error in subscriber callback {sub_id}: {str(e)}")

        return notify_count

    def get_subscription_count(self, data_type: Optional[DataType] = None,
                              symbol: Optional[str] = None,
                              interval: Optional[str] = None) -> int:
        """
        Get count of subscribers matching filters.

        Args:
            data_type: Type of data (optional)
            symbol: Symbol (optional)
            interval: Interval (optional)

        Returns:
            Subscriber count
        """
        count = 0

        with self._lock:
            # Filter by data type
            data_types = [data_type.value] if data_type else list(self._subscribers.keys())

            for dt in data_types:
                if dt not in self._subscribers:
                    continue

                # Filter by symbol
                symbols = [symbol] if symbol else list(self._subscribers[dt].keys())

                for sym in symbols:
                    if sym not in self._subscribers[dt]:
                        continue

                    # Filter by interval
                    intervals = [interval] if interval is not None else list(self._subscribers[dt][sym].keys())

                    for intv in intervals:
                        if intv not in self._subscribers[dt][sym]:
                            continue

                        count += len(self._subscribers[dt][sym][intv])

        return count

    def get_active_subscriptions(self) -> Dict[str, List[str]]:
        """
        Get a summary of active subscriptions.

        Returns:
            Dictionary of data types and symbols
        """
        result = {}

        with self._lock:
            for data_type in self._subscribers:
                result[data_type] = []

                for symbol in self._subscribers[data_type]:
                    for interval in self._subscribers[data_type][symbol]:
                        sub_count = len(self._subscribers[data_type][symbol][interval])

                        if interval:
                            result[data_type].append(f"{symbol}_{interval} ({sub_count} subscribers)")
                        else:
                            result[data_type].append(f"{symbol} ({sub_count} subscribers)")

        return result
```

2. **Create Efficient Data Routing to Interested Components**:

```python
# In src/market_data/manager.py, enhance the manager to use the publish-subscribe system
from src.market_data.pubsub import MarketDataPubSub, DataType

class MarketDataManager:
    # Add to the existing class

    def __init__(self, api_client, use_websocket=True):
        # Existing initialization code

        # Add publish-subscribe system
        self.pubsub = MarketDataPubSub()

        # Connect WebSocket callbacks to publish-subscribe
        self._setup_pubsub_integration()

    def _setup_pubsub_integration(self):
        """Connect WebSocket callbacks with publish-subscribe system."""
        if not self.use_websocket or not self.websocket_manager:
            return

        # Save the original methods
        original_subscribe_klines = self.subscribe_klines
        original_subscribe_orderbook = self.subscribe_orderbook

        # Enhance the subscribe_klines method
        def enhanced_subscribe_klines(symbol, interval, callback):
            # Create a wrapper callback that publishes to our pubsub
            def ws_callback(data):
                # Publish to pubsub system
                self.pubsub.publish(DataType.KLINE, symbol, data, interval)

                # Call the original callback
                callback(data)

            # Call the original method with our wrapper
            return original_subscribe_klines(symbol, interval, ws_callback)

        # Enhance the subscribe_orderbook method
        def enhanced_subscribe_orderbook(symbol, callback):
            # Create a wrapper callback that publishes to our pubsub
            def ws_callback(data):
                # Publish to pubsub system
                self.pubsub.publish(DataType.ORDERBOOK, symbol, data)

                # Call the original callback
                callback(data)

            # Call the original method with our wrapper
            return original_subscribe_orderbook(symbol, ws_callback)

        # Replace the methods
        self.subscribe_klines = enhanced_subscribe_klines
        self.subscribe_orderbook = enhanced_subscribe_orderbook

    def add_market_data_listener(self, data_type: DataType, symbol: str,
                                callback: Callable[[Dict[str, Any]], None],
                                interval: Optional[str] = None) -> str:
        """
        Add a market data listener using the pubsub system.

        Args:
            data_type: Type of data
            symbol: Symbol
            callback: Callback function
            interval: Interval (for klines)

        Returns:
            Subscription ID
        """
        return self.pubsub.subscribe(data_type, symbol, interval, callback)

    def remove_market_data_listener(self, subscription_id: str) -> bool:
        """
        Remove a market data listener.

        Args:
            subscription_id: Subscription ID

        Returns:
            Success flag
        """
        return self.pubsub.unsubscribe(subscription_id)

    def get_market_data_listeners(self) -> Dict[str, List[str]]:
        """
        Get a summary of active market data listeners.

        Returns:
            Dictionary of active listeners
        """
        return self.pubsub.get_active_subscriptions()
```

3. **Add Data Transformation for Different Consumer Needs**:

```python
# In src/market_data/transformer.py, implement data transformation utilities
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger

class MarketDataTransformer:
    """Transform market data for different consumer needs."""

    @staticmethod
    def klines_to_dataframe(klines: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert kline data to a pandas DataFrame.

        Args:
            klines: List of kline data

        Returns:
            DataFrame with kline data
        """
        if not klines:
            return pd.DataFrame()

        # Create DataFrame
        df = pd.DataFrame(klines)

        # Convert timestamp to datetime
        if "open_time" in df.columns:
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
            df.set_index("open_time", inplace=True)

        # Ensure numeric columns
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col])

        # Sort by index
        df.sort_index(inplace=True)

        return df

    @staticmethod
    def klines_to_ohlcv(klines: List[Dict[str, Any]]) -> Dict[str, List[float]]:
        """
        Convert kline data to OHLCV format.

        Args:
            klines: List of kline data

        Returns:
            Dictionary with OHLCV arrays
        """
        if not klines:
            return {
                "timestamps": [],
                "open": [],
                "high": [],
                "low": [],
                "close": [],
                "volume": []
            }

        # Sort klines by timestamp
        sorted_klines = sorted(klines, key=lambda k: k["open_time"])

        # Extract arrays
        result = {
            "timestamps": [k["open_time"] for k in sorted_klines],
            "open": [float(k["open"]) for k in sorted_klines],
            "high": [float(k["high"]) for k in sorted_klines],
            "low": [float(k["low"]) for k in sorted_klines],
            "close": [float(k["close"]) for k in sorted_klines],
            "volume": [float(k["volume"]) for k in sorted_klines]
        }

        return result

    @staticmethod
    def orderbook_to_arrays(orderbook: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Convert orderbook data to arrays.

        Args:
            orderbook: Orderbook data

        Returns:
            Dictionary with bid/ask arrays
        """
        if not orderbook or "bids" not in orderbook or "asks" not in orderbook:
            return {
                "bid_prices": np.array([]),
                "bid_sizes": np.array([]),
                "ask_prices": np.array([]),
                "ask_sizes": np.array([])
            }

        # Extract arrays
        bid_prices = np.array([float(bid[0]) for bid in orderbook["bids"]])
        bid_sizes = np.array([float(bid[1]) for bid in orderbook["bids"]])
        ask_prices = np.array([float(ask[0]) for ask in orderbook["asks"]])
        ask_sizes = np.array([float(ask[1]) for ask in orderbook["asks"]])

        return {
            "bid_prices": bid_prices,
            "bid_sizes": bid_sizes,
            "ask_prices": ask_prices,
            "ask_sizes": ask_sizes
        }

    @staticmethod
    def calculate_mid_price(orderbook: Dict[str, Any]) -> Optional[float]:
        """
        Calculate mid price from orderbook.

        Args:
            orderbook: Orderbook data

        Returns:
            Mid price or None if not available
        """
        if not orderbook or "bids" not in orderbook or "asks" not in orderbook:
            return None

        # Get best bid and ask
        if not orderbook["bids"] or not orderbook["asks"]:
            return None

        best_bid = float(orderbook["bids"][0][0])
        best_ask = float(orderbook["asks"][0][0])

        # Calculate mid price
        mid_price = (best_bid + best_ask) / 2

        return mid_price

    @staticmethod
    def calculate_vwap(klines: List[Dict[str, Any]]) -> Optional[float]:
        """
        Calculate volume-weighted average price from klines.

        Args:
            klines: List of kline data

        Returns:
            VWAP or None if not available
        """
        if not klines:
            return None

        total_volume = 0
        volume_times_price = 0

        for kline in klines:
            try:
                typical_price = (float(kline["high"]) + float(kline["low"]) + float(kline["close"])) / 3
                volume = float(kline["volume"])

                volume_times_price += typical_price * volume
                total_volume += volume
            except (ValueError, TypeError, KeyError):
                pass

        if total_volume == 0:
            return None

        vwap = volume_times_price / total_volume

        return vwap
```

The data distribution system implementation includes three key components:

1. **Publish-Subscribe System** - A flexible system for distributing market data to interested components
2. **Efficient Data Routing** - Integration with the WebSocket system to ensure data flows to subscribers
3. **Data Transformation** - Utilities for transforming data to different formats needed by consumers

This system ensures that market data can flow efficiently to all components that need it, with appropriate transformations for their specific needs.

### 1.3 Historical Data Management

#### Implementation Details

1. **Implement Historical Data Storage and Retrieval**:

```python
# In src/market_data/storage.py, implement historical data storage
from typing import Dict, Any, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
import h5py
from loguru import logger

class MarketDataStorage:
    """Storage for historical market data."""

    def __init__(self, base_dir: str = "data/market_data"):
        """
        Initialize the market data storage.

        Args:
            base_dir: Base directory for data storage
        """
        self.base_dir = base_dir

        # Ensure directory exists
        os.makedirs(base_dir, exist_ok=True)

        # Create subdirectories
        self.kline_dir = os.path.join(base_dir, "klines")
        self.orderbook_dir = os.path.join(base_dir, "orderbook")
        self.ticker_dir = os.path.join(base_dir, "tickers")
        self.trade_dir = os.path.join(base_dir, "trades")

        # Ensure subdirectories exist
        for directory in [self.kline_dir, self.orderbook_dir, self.ticker_dir, self.trade_dir]:
            os.makedirs(directory, exist_ok=True)

        logger.info(f"Market data storage initialized at {base_dir}")

    def store_klines(self, symbol: str, interval: str, klines: List[Dict[str, Any]]) -> bool:
        """
        Store kline data.

        Args:
            symbol: Symbol
            interval: Interval
            klines: List of kline data

        Returns:
            Success flag
        """
        if not klines:
            logger.warning(f"No klines to store for {symbol} {interval}")
            return False

        try:
            # Create symbol directory
            symbol_dir = os.path.join(self.kline_dir, symbol)
            os.makedirs(symbol_dir, exist_ok=True)

            # Create file path
            file_path = os.path.join(symbol_dir, f"{interval}.h5")

            # Convert to DataFrame
            df = pd.DataFrame(klines)

            # Ensure timestamp column is present
            if "open_time" not in df.columns:
                logger.error(f"Missing open_time in klines for {symbol} {interval}")
                return False

            # Set timestamp as index
            df["open_time"] = pd.to_numeric(df["open_time"])
            df.set_index("open_time", inplace=True)

            # Sort by timestamp
            df.sort_index(inplace=True)

            # Convert numeric columns
            numeric_cols = ["open", "high", "low", "close", "volume"]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col])

            # Check if file exists
            if os.path.exists(file_path):
                # Read existing data
                existing_df = pd.read_hdf(file_path, key="klines")

                # Combine with new data
                combined_df = pd.concat([existing_df, df])

                # Remove duplicates
                combined_df = combined_df[~combined_df.index.duplicated(keep="last")]

                # Sort by timestamp
                combined_df.sort_index(inplace=True)

                # Write back to file
                combined_df.to_hdf(file_path, key="klines", mode="w")
            else:
                # Write new file
                df.to_hdf(file_path, key="klines", mode="w")

            logger.info(f"Stored {len(klines)} klines for {symbol} {interval}")
            return True

        except Exception as e:
            logger.error(f"Error storing klines for {symbol} {interval}: {str(e)}")
            return False

    def get_klines(self, symbol: str, interval: str,
                  start_time: Optional[int] = None,
                  end_time: Optional[int] = None,
                  limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get kline data.

        Args:
            symbol: Symbol
            interval: Interval
            start_time: Start time in milliseconds (optional)
            end_time: End time in milliseconds (optional)
            limit: Maximum number of klines (optional)

        Returns:
            List of kline data
        """
        try:
            # Create file path
            file_path = os.path.join(self.kline_dir, symbol, f"{interval}.h5")

            # Check if file exists
            if not os.path.exists(file_path):
                logger.warning(f"No kline data found for {symbol} {interval}")
                return []

            # Read data
            df = pd.read_hdf(file_path, key="klines")

            # Apply time filters
            if start_time is not None:
                df = df[df.index >= start_time]

            if end_time is not None:
                df = df[df.index <= end_time]

            # Apply limit
            if limit is not None and limit > 0:
                df = df.tail(limit)

            # Convert to list of dictionaries
            result = []
            for idx, row in df.iterrows():
                kline = dict(row)
                kline["open_time"] = idx
                result.append(kline)

            logger.debug(f"Retrieved {len(result)} klines for {symbol} {interval}")
            return result

        except Exception as e:
            logger.error(f"Error retrieving klines for {symbol} {interval}: {str(e)}")
            return []

    def get_klines_as_dataframe(self, symbol: str, interval: str,
                               start_time: Optional[int] = None,
                               end_time: Optional[int] = None,
                               limit: Optional[int] = None) -> pd.DataFrame:
        """
        Get kline data as a pandas DataFrame.

        Args:
            symbol: Symbol
            interval: Interval
            start_time: Start time in milliseconds (optional)
            end_time: End time in milliseconds (optional)
            limit: Maximum number of klines (optional)

        Returns:
            DataFrame with kline data
        """
        try:
            # Create file path
            file_path = os.path.join(self.kline_dir, symbol, f"{interval}.h5")

            # Check if file exists
            if not os.path.exists(file_path):
                logger.warning(f"No kline data found for {symbol} {interval}")
                return pd.DataFrame()

            # Read data
            df = pd.read_hdf(file_path, key="klines")

            # Apply time filters
            if start_time is not None:
                df = df[df.index >= start_time]

            if end_time is not None:
                df = df[df.index <= end_time]

            # Apply limit
            if limit is not None and limit > 0:
                df = df.tail(limit)

            logger.debug(f"Retrieved {len(df)} klines as DataFrame for {symbol} {interval}")
            return df

        except Exception as e:
            logger.error(f"Error retrieving klines as DataFrame for {symbol} {interval}: {str(e)}")
            return pd.DataFrame()

    def store_orderbook_snapshot(self, symbol: str, timestamp: int, orderbook: Dict[str, Any]) -> bool:
        """
        Store orderbook snapshot.

        Args:
            symbol: Symbol
            timestamp: Timestamp in milliseconds
            orderbook: Orderbook data

        Returns:
            Success flag
        """
        if not orderbook or "bids" not in orderbook or "asks" not in orderbook:
            logger.warning(f"Invalid orderbook data for {symbol}")
            return False

        try:
            # Create symbol directory
            symbol_dir = os.path.join(self.orderbook_dir, symbol)
            os.makedirs(symbol_dir, exist_ok=True)

            # Create date directory (organize by date for easier management)
            date_str = datetime.fromtimestamp(timestamp / 1000).strftime("%Y-%m-%d")
            date_dir = os.path.join(symbol_dir, date_str)
            os.makedirs(date_dir, exist_ok=True)

            # Create file name with timestamp
            file_name = f"{timestamp}.json"
            file_path = os.path.join(date_dir, file_name)

            # Store as JSON
            with open(file_path, "w") as f:
                json.dump(orderbook, f)

            logger.debug(f"Stored orderbook snapshot for {symbol} at {timestamp}")
            return True

        except Exception as e:
            logger.error(f"Error storing orderbook snapshot for {symbol}: {str(e)}")
            return False

    def get_orderbook_snapshot(self, symbol: str, timestamp: int) -> Optional[Dict[str, Any]]:
        """
        Get orderbook snapshot closest to the timestamp.

        Args:
            symbol: Symbol
            timestamp: Timestamp in milliseconds

        Returns:
            Orderbook data or None if not found
        """
        try:
            # Create symbol directory path
            symbol_dir = os.path.join(self.orderbook_dir, symbol)
            if not os.path.exists(symbol_dir):
                logger.warning(f"No orderbook data found for {symbol}")
                return None

            # Get date string
            date_str = datetime.fromtimestamp(timestamp / 1000).strftime("%Y-%m-%d")
            date_dir = os.path.join(symbol_dir, date_str)

            if not os.path.exists(date_dir):
                logger.warning(f"No orderbook data found for {symbol} on {date_str}")
                return None

            # Get list of files
            files = os.listdir(date_dir)

            # Find closest timestamp
            closest_ts = None
            min_diff = float("inf")

            for file_name in files:
                if not file_name.endswith(".json"):
                    continue

                try:
                    file_ts = int(file_name.split(".")[0])
                    diff = abs(file_ts - timestamp)

                    if diff < min_diff:
                        min_diff = diff
                        closest_ts = file_ts
                except ValueError:
                    continue

            if closest_ts is None:
                logger.warning(f"No valid orderbook snapshot found for {symbol} near {timestamp}")
                return None

            # Read the file
            file_path = os.path.join(date_dir, f"{closest_ts}.json")
            with open(file_path, "r") as f:
                orderbook = json.load(f)

            logger.debug(f"Retrieved orderbook snapshot for {symbol} at {closest_ts} (requested {timestamp})")
            return orderbook

        except Exception as e:
            logger.error(f"Error retrieving orderbook snapshot for {symbol}: {str(e)}")
            return None

    def get_available_symbols(self, data_type: str = "klines") -> List[str]:
        """
        Get list of available symbols.

        Args:
            data_type: Type of data ("klines", "orderbook", "tickers", "trades")

        Returns:
            List of symbol strings
        """
        try:
            # Get appropriate directory
            if data_type == "klines":
                data_dir = self.kline_dir
            elif data_type == "orderbook":
                data_dir = self.orderbook_dir
            elif data_type == "tickers":
                data_dir = self.ticker_dir
            elif data_type == "trades":
                data_dir = self.trade_dir
            else:
                logger.warning(f"Unknown data type: {data_type}")
                return []

            # Check if directory exists
            if not os.path.exists(data_dir):
                return []

            # Get list of subdirectories (symbols)
            symbols = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

            return symbols

        except Exception as e:
            logger.error(f"Error getting available symbols for {data_type}: {str(e)}")
            return []

    def get_available_intervals(self, symbol: str) -> List[str]:
        """
        Get list of available intervals for klines.

        Args:
            symbol: Symbol

        Returns:
            List of interval strings
        """
        try:
            # Create symbol directory path
            symbol_dir = os.path.join(self.kline_dir, symbol)
            if not os.path.exists(symbol_dir):
                return []

            # Get list of files
            files = os.listdir(symbol_dir)

            # Extract intervals from file names
            intervals = [f.split(".")[0] for f in files if f.endswith(".h5")]

            return intervals

        except Exception as e:
            logger.error(f"Error getting available intervals for {symbol}: {str(e)}")
            return []

    def get_data_date_range(self, symbol: str, interval: str) -> Tuple[Optional[int], Optional[int]]:
        """
        Get date range for kline data.

        Args:
            symbol: Symbol
            interval: Interval

        Returns:
            Tuple of (start_time, end_time) in milliseconds, or (None, None) if not available
        """
        try:
            # Create file path
            file_path = os.path.join(self.kline_dir, symbol, f"{interval}.h5")

            # Check if file exists
            if not os.path.exists(file_path):
                return (None, None)

            # Read data
            df = pd.read_hdf(file_path, key="klines")

            # Get start and end time
            if len(df) == 0:
                return (None, None)

            start_time = df.index.min()
            end_time = df.index.max()

            return (int(start_time), int(end_time))

        except Exception as e:
            logger.error(f"Error getting date range for {symbol} {interval}: {str(e)}")
            return (None, None)
```

2. **Add Efficient Data Caching Mechanisms**:

```python
# In src/market_data/cache.py, implement data caching
from typing import Dict, Any, List, Optional, Union, Callable
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import time
from loguru import logger

class MarketDataCache:
    """Cache for market data to reduce storage access and API calls."""

    def __init__(self, max_cache_size: int = 1000, expiry_time: int = 3600):
        """
        Initialize the market data cache.

        Args:
            max_cache_size: Maximum number of items in cache
            expiry_time: Cache expiry time in seconds
        """
        self.max_cache_size = max_cache_size
        self.expiry_time = expiry_time

        # Caches for different data types
        self.kline_cache = {}  # Symbol -> interval -> DataFrame
        self.orderbook_cache = {}  # Symbol -> timestamp -> orderbook

        # Cache metadata
        self.access_times = {}  # Cache key -> last access time
        self.insert_times = {}  # Cache key -> insert time

        # Lock for thread safety
        self.lock = threading.RLock()

        # Start cache cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()

        logger.info(f"Market data cache initialized with size {max_cache_size}, expiry {expiry_time}s")

    def get_klines(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        """
        Get klines from cache.

        Args:
            symbol: Symbol
            interval: Interval

        Returns:
            DataFrame with klines or None if not in cache
        """
        with self.lock:
            # Create cache key
            cache_key = f"klines_{symbol}_{interval}"

            # Check if in cache
            if (symbol in self.kline_cache and
                interval in self.kline_cache[symbol]):

                # Update access time
                self.access_times[cache_key] = time.time()

                return self.kline_cache[symbol][interval]

            return None

    def set_klines(self, symbol: str, interval: str, klines: pd.DataFrame) -> None:
        """
        Set klines in cache.

        Args:
            symbol: Symbol
            interval: Interval
            klines: DataFrame with klines
        """
        with self.lock:
            # Create cache key
            cache_key = f"klines_{symbol}_{interval}"

            # Initialize caches if needed
            if symbol not in self.kline_cache:
                self.kline_cache[symbol] = {}

            # Store in cache
            self.kline_cache[symbol][interval] = klines

            # Update metadata
            current_time = time.time()
            self.access_times[cache_key] = current_time
            self.insert_times[cache_key] = current_time

            # Check cache size
            self._enforce_cache_size()

    def get_orderbook(self, symbol: str, timestamp: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Get orderbook from cache.

        Args:
            symbol: Symbol
            timestamp: Timestamp or None for latest

        Returns:
            Orderbook data or None if not in cache
        """
        with self.lock:
            # Check if symbol is in cache
            if symbol not in self.orderbook_cache:
                return None

            # If timestamp is None, get latest
            if timestamp is None:
                if not self.orderbook_cache[symbol]:
                    return None

                # Get latest timestamp
                latest_ts = max(self.orderbook_cache[symbol].keys())
                cache_key = f"orderbook_{symbol}_{latest_ts}"

                # Update access time
                self.access_times[cache_key] = time.time()

                return self.orderbook_cache[symbol][latest_ts]

            # Check if timestamp is in cache
            if timestamp in self.orderbook_cache[symbol]:
                cache_key = f"orderbook_{symbol}_{timestamp}"

                # Update access time
                self.access_times[cache_key] = time.time()

                return self.orderbook_cache[symbol][timestamp]

            # Find closest timestamp
            timestamps = list(self.orderbook_cache[symbol].keys())
            if not timestamps:
                return None

            # Find closest timestamp
            closest_ts = min(timestamps, key=lambda ts: abs(ts - timestamp))

            # Check if within reasonable range (10 seconds)
            if abs(closest_ts - timestamp) > 10000:
                return None

            cache_key = f"orderbook_{symbol}_{closest_ts}"

            # Update access time
            self.access_times[cache_key] = time.time()

            return self.orderbook_cache[symbol][closest_ts]

    def set_orderbook(self, symbol: str, timestamp: int, orderbook: Dict[str, Any]) -> None:
        """
        Set orderbook in cache.

        Args:
            symbol: Symbol
            timestamp: Timestamp
            orderbook: Orderbook data
        """
        with self.lock:
            # Create cache key
            cache_key = f"orderbook_{symbol}_{timestamp}"

            # Initialize caches if needed
            if symbol not in self.orderbook_cache:
                self.orderbook_cache[symbol] = {}

            # Store in cache
            self.orderbook_cache[symbol][timestamp] = orderbook

            # Update metadata
            current_time = time.time()
            self.access_times[cache_key] = current_time
            self.insert_times[cache_key] = current_time

            # Check cache size
            self._enforce_cache_size()

    def _enforce_cache_size(self) -> None:
        """Enforce maximum cache size by removing least recently used items."""
        # Count total cache items
        kline_count = sum(len(intervals) for intervals in self.kline_cache.values())
        orderbook_count = sum(len(timestamps) for timestamps in self.orderbook_cache.values())

        total_count = kline_count + orderbook_count

        # Check if we need to remove items
        if total_count <= self.max_cache_size:
            return

        # Calculate how many items to remove
        to_remove = total_count - self.max_cache_size

        # Get sorted list of cache keys by access time
        sorted_keys = sorted(self.access_times.items(), key=lambda x: x[1])

        # Remove oldest accessed items
        for i in range(min(to_remove, len(sorted_keys))):
            cache_key = sorted_keys[i][0]

            # Parse cache key
            parts = cache_key.split("_")
            data_type = parts[0]
            symbol = parts[1]

            if data_type == "klines":
                interval = parts[2]

                if symbol in self.kline_cache and interval in self.kline_cache[symbol]:
                    del self.kline_cache[symbol][interval]
                    del self.access_times[cache_key]
                    del self.insert_times[cache_key]

                    # Clean up empty dictionaries
                    if not self.kline_cache[symbol]:
                        del self.kline_cache[symbol]

            elif data_type == "orderbook":
                timestamp = int(parts[2])

                if symbol in self.orderbook_cache and timestamp in self.orderbook_cache[symbol]:
                    del self.orderbook_cache[symbol][timestamp]
                    del self.access_times[cache_key]
                    del self.insert_times[cache_key]

                    # Clean up empty dictionaries
                    if not self.orderbook_cache[symbol]:
                        del self.orderbook_cache[symbol]

    def _cleanup_loop(self) -> None:
        """Background loop for cache cleanup."""
        while True:
            try:
                # Sleep for a while
                time.sleep(60)

                # Clean up expired items
                self._cleanup_expired()

            except Exception as e:
                logger.error(f"Error in cache cleanup loop: {str(e)}")

    def _cleanup_expired(self) -> None:
        """Remove expired items from cache."""
        with self.lock:
            current_time = time.time()
            expired_keys = []

            # Find expired keys
            for cache_key, insert_time in self.insert_times.items():
                if current_time - insert_time > self.expiry_time:
                    expired_keys.append(cache_key)

            # Remove expired items
            for cache_key in expired_keys:
                # Parse cache key
                parts = cache_key.split("_")
                data_type = parts[0]
                symbol = parts[1]

                if data_type == "klines":
                    interval = parts[2]

                    if symbol in self.kline_cache and interval in self.kline_cache[symbol]:
                        del self.kline_cache[symbol][interval]

                    # Clean up empty dictionaries
                    if symbol in self.kline_cache and not self.kline_cache[symbol]:
                        del self.kline_cache[symbol]

                elif data_type == "orderbook":
                    timestamp = int(parts[2])

                    if symbol in self.orderbook_cache and timestamp in self.orderbook_cache[symbol]:
                        del self.orderbook_cache[symbol][timestamp]

                    # Clean up empty dictionaries
                    if symbol in self.orderbook_cache and not self.orderbook_cache[symbol]:
                        del self.orderbook_cache[symbol]

                # Remove from metadata
                del self.access_times[cache_key]
                del self.insert_times[cache_key]

            if expired_keys:
                logger.debug(f"Removed {len(expired_keys)} expired items from cache")
```

3. **Create Data Backfilling for Historical Analysis**:

```python
# In src/market_data/backfill.py, implement data backfilling
from typing import Dict, Any, List, Optional, Union, Callable
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from loguru import logger

class MarketDataBackfill:
    """Utility for backfilling historical market data."""

    def __init__(self, api_client, storage):
        """
        Initialize the backfill utility.

        Args:
            api_client: API client
            storage: Market data storage
        """
        self.api_client = api_client
        self.storage = storage

        logger.info("Market data backfill utility initialized")

    def backfill_klines(self, symbol: str, interval: str,
                       start_time: Optional[int] = None,
                       end_time: Optional[int] = None,
                       max_requests: int = 100,
                       request_delay: float = 0.5) -> bool:
        """
        Backfill kline data.

        Args:
            symbol: Symbol
            interval: Interval
            start_time: Start time in milliseconds (optional)
            end_time: End time in milliseconds (optional)
            max_requests: Maximum number of API requests
            request_delay: Delay between requests in seconds

        Returns:
            Success flag
        """
        try:
            # Set default end time if not provided
            if end_time is None:
                end_time = int(time.time() * 1000)

            # Check existing data
            existing_start, existing_end = self.storage.get_data_date_range(symbol, interval)

            # If we have existing data, adjust start/end times
            if existing_start is not None and existing_end is not None:
                logger.info(f"Existing data for {symbol} {interval}: {existing_start} to {existing_end}")

                # If requested start time is before existing start, backfill earlier data
                if start_time is not None and start_time < existing_start:
                    end_time = existing_start

                # If requested end time is after existing end, backfill later data
                elif end_time > existing_end:
                    start_time = existing_end

                # Otherwise, we already have the requested data
                else:
                    logger.info(f"Already have data for {symbol} {interval} from {start_time} to {end_time}")
                    return True

            # Calculate number of klines per request
            limit = 1000  # Maximum limit for Bybit API

            # If no start time, calculate based on interval and limit
            if start_time is None:
                start_time = self._calculate_start_time(end_time, interval, limit)

            logger.info(f"Backfilling klines for {symbol} {interval} from {start_time} to {end_time}")

            # Initialize loop variables
            current_end = end_time
            request_count = 0
            total_klines = 0

            while True:
                # Check if we've reached max requests
                if request_count >= max_requests:
                    logger.warning(f"Reached maximum requests ({max_requests}) for backfill")
                    break

                # Check if we've backfilled to start time
                if current_end <= start_time:
                    logger.info(f"Completed backfill to start time {start_time}")
                    break

                # Make API request
                klines = self.api_client.market.get_kline(
                    category="linear",
                    symbol=symbol,
                    interval=interval,
                    limit=limit,
                    end=current_end
                )

                # Increment request count
                request_count += 1

                # Process response
                if "result" not in klines or "list" not in klines["result"]:
                    logger.error(f"Invalid response for klines: {klines}")
                    return False

                # Extract kline data
                kline_list = klines["result"]["list"]

                # If no klines returned, we're done
                if not kline_list:
                    logger.info(f"No more klines available before {current_end}")
                    break

                # Process klines
                processed_klines = []

                for kline in kline_list:
                    # Bybit API returns klines in format [timestamp, open, high, low, close, volume, ...]
                    processed = {
                        "open_time": int(kline[0]),
                        "open": float(kline[1]),
                        "high": float(kline[2]),
                        "low": float(kline[3]),
                        "close": float(kline[4]),
                        "volume": float(kline[5])
                    }

                    processed_klines.append(processed)

                # Store klines
                if processed_klines:
                    success = self.storage.store_klines(symbol, interval, processed_klines)
                    if not success:
                        logger.error(f"Failed to store klines for {symbol} {interval}")
                        return False

                    # Update counts
                    total_klines += len(processed_klines)

                    # Update current end time for next request
                    current_end = min(kline[0] for kline in kline_list)

                    logger.debug(f"Stored {len(processed_klines)} klines, new end time: {current_end}")

                # Delay between requests
                time.sleep(request_delay)

            logger.info(f"Backfill complete, fetched {total_klines} klines in {request_count} requests")
            return True

        except Exception as e:
            logger.error(f"Error backfilling klines for {symbol} {interval}: {str(e)}")
            return False

    def _calculate_start_time(self, end_time: int, interval: str, limit: int) -> int:
        """
        Calculate start time based on interval and limit.

        Args:
            end_time: End time in milliseconds
            interval: Interval
            limit: Number of klines

        Returns:
            Start time in milliseconds
        """
        # Map interval to milliseconds
        interval_ms = {
            "1m": 60 * 1000,
            "3m": 3 * 60 * 1000,
            "5m": 5 * 60 * 1000,
            "15m": 15 * 60 * 1000,
            "30m": 30 * 60 * 1000,
            "1h": 60 * 60 * 1000,
            "2h": 2 * 60 * 60 * 1000,
            "4h": 4 * 60 * 60 * 1000,
            "6h": 6 * 60 * 60 * 1000,
            "12h": 12 * 60 * 60 * 1000,
            "1d": 24 * 60 * 60 * 1000,
            "1w": 7 * 24 * 60 * 60 * 1000,
            "1M": 30 * 24 * 60 * 60 * 1000
        }

        # Get interval in milliseconds
        if interval not in interval_ms:
            # Default to 1 day
            interval_ms_value = 24 * 60 * 60 * 1000
        else:
            interval_ms_value = interval_ms[interval]

        # Calculate start time
        start_time = end_time - (limit * interval_ms_value)

        return start_time

    def backfill_all_symbols(self, intervals: List[str],
                            days_back: int = 30,
                            max_requests_per_symbol: int = 50,
                            request_delay: float = 0.5) -> Dict[str, bool]:
        """
        Backfill kline data for all available symbols.

        Args:
            intervals: List of intervals
            days_back: Number of days to backfill
            max_requests_per_symbol: Maximum requests per symbol
            request_delay: Delay between requests in seconds

        Returns:
            Dictionary of symbol to success flag
        """
        try:
            # Get list of available symbols
            symbols_info = self.api_client.market.get_tickers(category="linear")

            if "result" not in symbols_info or "list" not in symbols_info["result"]:
                logger.error(f"Invalid response for symbols: {symbols_info}")
                return {}

            # Extract symbols
            symbols = [ticker["symbol"] for ticker in symbols_info["result"]["list"]]

            # Calculate end time (now)
            end_time = int(time.time() * 1000)

            # Calculate start time
            start_time = end_time - (days_back * 24 * 60 * 60 * 1000)

            # Backfill each symbol
            results = {}

            for symbol in symbols:
                symbol_success = True

                for interval in intervals:
                    logger.info(f"Backfilling {symbol} {interval}")

                    success = self.backfill_klines(
                        symbol=symbol,
                        interval=interval,
                        start_time=start_time,
                        end_time=end_time,
                        max_requests=max_requests_per_symbol,
                        request_delay=request_delay
                    )

                    if not success:
                        symbol_success = False

                results[symbol] = symbol_success

            return results

        except Exception as e:
            logger.error(f"Error backfilling all symbols: {str(e)}")
            return {}
```

The historical data management implementation includes three key components:

1. **Data Storage and Retrieval** - Efficient storage and retrieval of historical market data
2. **Data Caching** - Memory caching to reduce storage access and improve performance
3. **Data Backfilling** - Utilities for acquiring and managing historical data for analysis

These components work together to provide a robust historical data management system that supports both real-time trading and historical analysis.
