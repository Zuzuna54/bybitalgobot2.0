"""
WebSocket Service for Bybit API

This module provides functionality for handling WebSocket connections to Bybit,
including public and private channels.
"""

import json
import hmac
import time
import hashlib
import uuid
from typing import Dict, Any, Optional, List, Callable, Union, Tuple
import threading
import websocket
from loguru import logger
import queue

from src.api.bybit.core.connection import ConnectionManager
from src.api.bybit.core.error_handling import (
    with_error_handling,
    BybitAuthenticationError,
)
from src.api.bybit.core.rate_limiting import rate_limited


class WebSocketService:
    """
    Service for handling WebSocket connections to Bybit.
    """

    def __init__(self, connection_manager: ConnectionManager):
        """
        Initialize the WebSocket service.

        Args:
            connection_manager: Connection manager instance
        """
        self.connection_manager = connection_manager
        self.public_ws = None
        self.private_ws = None
        self.public_callbacks = {}
        self.private_callbacks = {}
        self.running = False
        self.auth_credentials = None
        self.public_ping_thread = None
        self.private_ping_thread = None
        self.public_ws_thread = None
        self.private_ws_thread = None
        self.auth_failed = False
        self.auth_retry_count = 0
        self.max_auth_retries = 3

        # Message queue for storing received messages
        self.message_queue = queue.Queue()

        # Maximum queue size
        self.max_queue_size = 100

    def set_auth_credentials(self, api_key: str, api_secret: str) -> None:
        """
        Set authentication credentials for private WebSocket channels.

        Args:
            api_key: Bybit API key
            api_secret: Bybit API secret
        """
        if not api_key or not api_secret:
            logger.warning(
                "Empty API credentials provided for WebSocket authentication"
            )
            return

        # Validate API key format (basic check)
        if not isinstance(api_key, str) or len(api_key.strip()) < 5:
            logger.error("Invalid API key format for WebSocket authentication")
            return

        # Validate API secret format (basic check)
        if not isinstance(api_secret, str) or len(api_secret.strip()) < 5:
            logger.error("Invalid API secret format for WebSocket authentication")
            return

        # Update credentials
        self.auth_credentials = (api_key.strip(), api_secret.strip())
        self.auth_failed = False
        self.auth_retry_count = 0

        logger.info(
            f"WebSocket authentication credentials updated. API key: {api_key[:4]}***"
        )

        # Reconnect private WebSocket with new credentials if already running
        if self.running and self.private_ws:
            logger.info("Reconnecting private WebSocket with new credentials")
            self._connect_private()

    def _generate_signature(self, expires: int) -> str:
        """
        Generate signature for authentication.

        Args:
            expires: Expiry timestamp

        Returns:
            Authentication signature
        """
        if not self.auth_credentials:
            raise ValueError("Authentication credentials not set")

        api_key, api_secret = self.auth_credentials

        signature = hmac.new(
            bytes(api_secret, "utf-8"),
            bytes(f"GET/realtime{expires}", "utf-8"),
            digestmod=hashlib.sha256,
        ).hexdigest()

        return signature

    def _authenticate(self, ws) -> None:
        """
        Authenticate to the private WebSocket channel.

        Args:
            ws: WebSocket connection
        """
        if not self.auth_credentials:
            logger.warning(
                "Authentication credentials not set, cannot authenticate WebSocket"
            )
            return

        if self.auth_failed and self.auth_retry_count >= self.max_auth_retries:
            logger.warning(
                f"WebSocket authentication has already failed {self.auth_retry_count} times, not retrying"
            )
            return

        api_key, _ = self.auth_credentials
        expires = int((time.time() + 10) * 1000)

        try:
            signature = self._generate_signature(expires)
            auth_message = json.dumps(
                {"op": "auth", "args": [api_key, expires, signature]}
            )
            ws.send(auth_message)
            logger.info("WebSocket authentication sent")
        except Exception as e:
            logger.error(f"Error during WebSocket authentication: {e}")
            self.auth_failed = True
            self.auth_retry_count += 1

    def _on_message(self, ws, message) -> None:
        """
        Handle incoming WebSocket messages.

        Args:
            ws: WebSocket instance
            message: Message received from WebSocket
        """
        try:
            # Parse the message
            data = json.loads(message)
            logger.debug(f"Received WebSocket message: {data}")

            # Add to message queue (with size limit)
            if self.message_queue.qsize() < self.max_queue_size:
                self.message_queue.put(data)
            else:
                # Remove oldest message if queue is full
                try:
                    self.message_queue.get_nowait()
                    self.message_queue.put(data)
                except queue.Empty:
                    pass

            # Handle ping/pong
            if "op" in data and data["op"] == "ping":
                pong_message = json.dumps({"op": "pong"})
                ws.send(pong_message)
                return

            # Handle subscribed confirmation
            if "op" in data and data["op"] == "subscribe":
                logger.info(f"Successfully subscribed to {data.get('args', [])}")
                return

            # Handle authentication response
            if "op" in data and data["op"] == "auth":
                if data.get("success"):
                    logger.info("WebSocket authentication successful")
                    self.auth_failed = False
                    self.auth_retry_count = 0
                else:
                    error_msg = data.get("ret_msg", "Unknown error")
                    logger.error(f"WebSocket authentication failed: {error_msg}")
                    self.auth_failed = True
                    self.auth_retry_count += 1

                    # Notify the connection manager of authentication failure
                    if hasattr(self.connection_manager, "invalidate_credentials"):
                        self.connection_manager.invalidate_credentials()

                    # Try to reconnect with exponential backoff if not exceeding max retries
                    if self.auth_retry_count < self.max_auth_retries and self.running:
                        retry_delay = 2**self.auth_retry_count  # Exponential backoff
                        logger.info(
                            f"Will retry WebSocket authentication in {retry_delay} seconds"
                        )
                        threading.Timer(retry_delay, self._connect_private).start()
                return

            # Handle error messages
            if "success" in data and not data["success"]:
                error_msg = data.get("ret_msg", "Unknown error")
                logger.error(f"WebSocket error: {error_msg}")

                # Check for authentication errors
                if "auth" in error_msg.lower() or "permission" in error_msg.lower():
                    logger.error(
                        "Possible authentication issue detected in WebSocket message"
                    )
                    self.auth_failed = True
                    self.auth_retry_count += 1

                    # Notify the connection manager of authentication failure
                    if hasattr(self.connection_manager, "invalidate_credentials"):
                        self.connection_manager.invalidate_credentials()
                return

            # Determine the channel type and dispatch to appropriate callback
            if "topic" in data:
                topic = data["topic"]
                logger.debug(f"Processing message for topic: {topic}")

                # Find and call the registered callbacks
                if ws == self.public_ws and topic in self.public_callbacks:
                    for callback in self.public_callbacks.get(topic, []):
                        try:
                            callback(data)
                        except Exception as e:
                            logger.error(f"Error in callback for {topic}: {e}")
                elif ws == self.private_ws and topic in self.private_callbacks:
                    for callback in self.private_callbacks.get(topic, []):
                        try:
                            callback(data)
                        except Exception as e:
                            logger.error(f"Error in callback for {topic}: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in WebSocket message: {e}")
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")

    def _on_error(self, ws, error) -> None:
        """
        Handle WebSocket errors.

        Args:
            ws: WebSocket connection
            error: Error occurred
        """
        logger.error(f"WebSocket error: {error}")

        # Check for authentication errors
        if isinstance(error, str) and (
            "auth" in error.lower()
            or "permission" in error.lower()
            or "unauthorized" in error.lower()
        ):
            logger.error("Authentication error in WebSocket connection")
            self.auth_failed = True
            self.auth_retry_count += 1

            # Notify the connection manager of authentication failure
            if hasattr(self.connection_manager, "invalidate_credentials"):
                self.connection_manager.invalidate_credentials()

    def _on_close(self, ws, close_status_code, close_msg) -> None:
        """
        Handle WebSocket connection close.

        Args:
            ws: WebSocket connection
            close_status_code: Close status code
            close_msg: Close message
        """
        logger.info(f"WebSocket closed: {close_status_code} - {close_msg}")

        # Check if we need to reconnect
        if self.running:
            # For private WebSocket, check if we have valid auth
            if ws == self.private_ws:
                if self.auth_failed and self.auth_retry_count >= self.max_auth_retries:
                    logger.warning(
                        "Not reconnecting private WebSocket due to authentication failures"
                    )
                    return

            logger.info("Attempting to reconnect WebSocket")
            reconnect_delay = 1  # Start with 1 second

            if ws == self.public_ws:
                # Use exponential backoff for reconnection
                threading.Timer(reconnect_delay, self._connect_public).start()
            elif ws == self.private_ws:
                # Use exponential backoff with longer delay for private connections
                reconnect_delay = 2 ** min(
                    self.auth_retry_count, 3
                )  # Cap at 2^3 = 8 seconds
                threading.Timer(reconnect_delay, self._connect_private).start()

    def _on_open(self, ws) -> None:
        """
        Handle WebSocket connection open.

        Args:
            ws: WebSocket connection
        """
        logger.info("WebSocket connection opened")

        # If this is the private WebSocket, authenticate
        if ws == self.private_ws and self.auth_credentials:
            self._authenticate(ws)

    def _send_ping(self, ws, interval: int) -> None:
        """
        Send periodic ping messages to keep the connection alive.

        Args:
            ws: WebSocket connection
            interval: Ping interval in seconds
        """
        while self.running and ws.sock and ws.sock.connected:
            try:
                ping_message = json.dumps({"op": "ping"})
                ws.send(ping_message)
                logger.debug("Sent WebSocket ping")
            except Exception as e:
                logger.error(f"Error sending ping: {e}")
                break

            # Sleep for the specified interval
            time.sleep(interval)

    def _connect_public(self) -> None:
        """
        Connect to the public WebSocket channel.
        """
        # Check if websocket module is available
        try:
            import websocket
        except ImportError:
            logger.error(
                "websocket-client module not installed. Install it with 'pip install websocket-client'"
            )
            raise ImportError("websocket-client module not installed")

        # Close existing connection if any
        if self.public_ws:
            try:
                self.public_ws.close()
            except Exception as e:
                logger.error(f"Error closing existing public WebSocket: {e}")

        # Get the WebSocket URL from connection manager
        ws_url = self.connection_manager.get_websocket_url(public=True)
        logger.debug(f"Connecting to public WebSocket URL: {ws_url}")

        # Create new WebSocket connection
        self.public_ws = websocket.WebSocketApp(
            ws_url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )

        # Start WebSocket in a separate thread
        self.public_ws_thread = threading.Thread(
            target=self.public_ws.run_forever,
            kwargs={"ping_interval": 30, "ping_timeout": 10},
        )
        self.public_ws_thread.daemon = True
        self.public_ws_thread.start()

        # Start ping thread
        self.public_ping_thread = threading.Thread(
            target=self._send_ping, args=(self.public_ws, 20)
        )
        self.public_ping_thread.daemon = True
        self.public_ping_thread.start()

        logger.debug("Public WebSocket connection started")

    def _connect_private(self) -> None:
        """
        Connect to the private WebSocket channel.
        """
        # Check if websocket module is available
        try:
            import websocket
        except ImportError:
            logger.error(
                "websocket-client module not installed. Install it with 'pip install websocket-client'"
            )
            raise ImportError("websocket-client module not installed")

        # Skip if no API key is provided
        if (
            not self.connection_manager.api_key
            or not self.connection_manager.api_secret
            or (
                hasattr(self.connection_manager, "is_authenticated")
                and not self.connection_manager.is_authenticated
            )
        ):
            logger.warning(
                "No valid API credentials available, skipping private WebSocket connection"
            )
            return

        # Skip if too many auth failures
        if self.auth_failed and self.auth_retry_count >= self.max_auth_retries:
            logger.warning(
                f"WebSocket authentication has failed {self.auth_retry_count} times, skipping private connection"
            )
            return

        # Close existing connection if any
        if self.private_ws:
            try:
                self.private_ws.close()
            except Exception as e:
                logger.error(f"Error closing existing private WebSocket: {e}")

        # Get the WebSocket URL from connection manager
        ws_url = self.connection_manager.get_websocket_url(public=False)
        logger.debug(f"Connecting to private WebSocket URL: {ws_url}")

        # Create new WebSocket connection
        self.private_ws = websocket.WebSocketApp(
            ws_url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )

        # Start WebSocket in a separate thread
        self.private_ws_thread = threading.Thread(
            target=self.private_ws.run_forever,
            kwargs={"ping_interval": 30, "ping_timeout": 10},
        )
        self.private_ws_thread.daemon = True
        self.private_ws_thread.start()

        # Start ping thread
        self.private_ping_thread = threading.Thread(
            target=self._send_ping, args=(self.private_ws, 20)
        )
        self.private_ping_thread.daemon = True
        self.private_ping_thread.start()

        logger.debug("Private WebSocket connection started")

    def start(self) -> None:
        """
        Start the WebSocket connections.
        """
        # Only start if not already running
        if not self.running:
            self.running = True
            try:
                self._connect_public()
                if self.auth_credentials or (
                    self.connection_manager.api_key
                    and self.connection_manager.api_secret
                ):
                    self._connect_private()
                logger.info("WebSocket connections started")
            except Exception as e:
                logger.error(f"Error starting WebSocket connections: {e}")
                self.running = False
                raise

    def stop(self) -> None:
        """
        Stop the WebSocket connections.
        """
        if not self.running:
            logger.debug("WebSocket connections already stopped")
            return

        logger.info("Stopping WebSocket connections")
        self.running = False

        # Close connections
        if self.public_ws:
            try:
                self.public_ws.close()
            except Exception as e:
                logger.error(f"Error closing public WebSocket: {e}")

        if self.private_ws:
            try:
                self.private_ws.close()
            except Exception as e:
                logger.error(f"Error closing private WebSocket: {e}")

        # Wait for threads to terminate
        threads_to_join = [
            (self.public_ws_thread, "public WebSocket thread"),
            (self.private_ws_thread, "private WebSocket thread"),
            (self.public_ping_thread, "public ping thread"),
            (self.private_ping_thread, "private ping thread"),
        ]

        for thread, name in threads_to_join:
            if thread and thread.is_alive():
                logger.debug(f"Waiting for {name} to terminate")
                thread.join(timeout=1)
                if thread.is_alive():
                    logger.warning(f"{name} did not terminate within timeout")

        # Reset thread references
        self.public_ws_thread = None
        self.private_ws_thread = None
        self.public_ping_thread = None
        self.private_ping_thread = None
        self.public_ws = None
        self.private_ws = None

        logger.info("WebSocket connections stopped")

    def subscribe_public(
        self, topics: List[str], callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """
        Subscribe to public WebSocket topics.

        Args:
            topics: List of topics to subscribe to
            callback: Callback function to handle messages
        """
        if not self.public_ws:
            logger.warning("Public WebSocket not connected")
            return

        # Register callbacks
        for topic in topics:
            if topic not in self.public_callbacks:
                self.public_callbacks[topic] = []
            self.public_callbacks[topic].append(callback)

        # Send subscription message
        subscription = {"op": "subscribe", "args": topics}
        self.public_ws.send(json.dumps(subscription))
        logger.info(f"Subscribed to public topics: {topics}")

    def subscribe_private(
        self, topics: List[str], callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """
        Subscribe to private WebSocket topics.

        Args:
            topics: List of topics to subscribe to
            callback: Callback function to handle messages
        """
        if not self.private_ws:
            logger.warning("Private WebSocket not connected")
            return

        # Register callbacks
        for topic in topics:
            if topic not in self.private_callbacks:
                self.private_callbacks[topic] = []
            self.private_callbacks[topic].append(callback)

        # Send subscription message
        subscription = {"op": "subscribe", "args": topics}
        self.private_ws.send(json.dumps(subscription))
        logger.info(f"Subscribed to private topics: {topics}")

    def unsubscribe(self, topics: List[str]) -> None:
        """
        Unsubscribe from WebSocket topics.

        Args:
            topics: List of topics to unsubscribe from
        """
        # Send unsubscribe message to both WebSockets
        unsubscription = {"op": "unsubscribe", "args": topics}

        if self.public_ws:
            self.public_ws.send(json.dumps(unsubscription))

        if self.private_ws:
            self.private_ws.send(json.dumps(unsubscription))

        # Remove callbacks
        for topic in topics:
            if topic in self.public_callbacks:
                del self.public_callbacks[topic]
            if topic in self.private_callbacks:
                del self.private_callbacks[topic]

        logger.info(f"Unsubscribed from topics: {topics}")

    def subscribe_to_ticker(self, symbol: str, callback: Callable) -> None:
        """
        Subscribe to ticker updates for a symbol.

        Args:
            symbol: Symbol to subscribe to (e.g., "BTCUSDT")
            callback: Callback function to execute when a message is received
        """
        # Format the topic according to Bybit V5 API
        topic = f"tickers.{symbol}"

        # Register callback
        if topic not in self.public_callbacks:
            self.public_callbacks[topic] = []
        self.public_callbacks[topic].append(callback)

        if self.public_ws and self.public_ws.sock and self.public_ws.sock.connected:
            # Format the subscription message according to Bybit V5 API
            subscription = {"op": "subscribe", "args": [topic]}
            self.public_ws.send(json.dumps(subscription))
            logger.info(f"Subscribed to ticker for {symbol}")
        else:
            logger.warning(
                f"Cannot subscribe to {symbol} ticker: WebSocket not connected"
            )
