"""
Bybit API Client for the Algorithmic Trading System

This module provides a client for interacting with the Bybit API,
including authentication, account management, order execution, and market data.
"""

from typing import Dict, Any, Optional, Union, List, Callable
from loguru import logger
import warnings

from src.api.bybit.core.connection import ConnectionManager
from src.api.bybit.core.rate_limiting import initialize_rate_limiters
from src.api.bybit.services.account_service import AccountService
from src.api.bybit.services.order_service import OrderService
from src.api.bybit.services.market_service import MarketDataService
from src.api.bybit.services.websocket_service import WebSocketService
from src.api.bybit.services.data_service import DataService


class BybitClient:
    """
    Client for interacting with the Bybit API.
    """

    def __init__(
        self,
        testnet: bool = True,
        api_key: str = "",
        api_secret: str = "",
        recv_window: int = 5000,
        data_dir: str = "data",
    ):
        """
        Initialize the Bybit client.

        Args:
            testnet: Whether to use the testnet API
            api_key: Bybit API key
            api_secret: Bybit API secret
            recv_window: Receive window for requests (milliseconds)
            data_dir: Directory to store market data
        """
        # Initialize connection manager
        self.connection_manager = ConnectionManager(
            testnet=testnet,
            api_key=api_key,
            api_secret=api_secret,
            recv_window=recv_window,
        )

        # Initialize rate limiters
        initialize_rate_limiters()

        # Initialize services
        self.account = AccountService(self.connection_manager)
        self.order = OrderService(self.connection_manager)
        self.market = MarketDataService(self.connection_manager)
        self.websocket = WebSocketService(self.connection_manager)

        # Initialize data service (enhanced market data with caching and persistence)
        self.data = DataService(
            connection_manager=self.connection_manager,
            market_service=self.market,
            websocket_service=self.websocket,
            data_dir=data_dir,
        )

        # Set authentication if credentials provided
        if api_key and api_secret:
            self.set_auth(api_key, api_secret)

    def set_auth(self, api_key: str, api_secret: str) -> None:
        """
        Set authentication credentials.

        Args:
            api_key: Bybit API key
            api_secret: Bybit API secret
        """
        self.connection_manager.set_auth_credentials(api_key, api_secret)
        self.websocket.set_auth_credentials(api_key, api_secret)

    def use_testnet(self, testnet: bool = True) -> None:
        """
        Set whether to use the testnet API.

        Args:
            testnet: Whether to use the testnet API
        """
        self.connection_manager.set_testnet(testnet)

    def get_server_time(self) -> int:
        """
        Get the server time.

        Returns:
            Server time in milliseconds
        """
        return self.connection_manager.get_server_time()

    def verify_credentials(self) -> bool:
        """
        Verify that the API credentials are valid.

        Returns:
            True if credentials are valid, False otherwise
        """
        return self.connection_manager.verify_credentials()
