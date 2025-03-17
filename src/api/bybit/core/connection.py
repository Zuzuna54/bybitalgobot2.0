"""
Connection Management for Bybit API

This module provides functionality for managing API connections,
including URL generation and connection settings.
"""

from typing import Dict, Any, Optional, Tuple
import time
import hmac
import hashlib
import json
import requests
from loguru import logger
from urllib.parse import urlencode


class ConnectionManager:
    """
    Manages connections to the Bybit API.
    """

    def __init__(
        self,
        testnet: bool = False,
        api_key: str = "",
        api_secret: str = "",
        recv_window: int = 5000,
    ):
        """
        Initialize the connection manager.

        Args:
            testnet: Whether to use the testnet API
            api_key: Bybit API key
            api_secret: Bybit API secret
            recv_window: Receive window for requests (milliseconds)
        """
        self.testnet = testnet
        self.api_key = api_key
        self.api_secret = api_secret
        self.recv_window = recv_window
        self.base_url = self._get_base_url()
        self.ws_url = self._get_ws_url()
        self.is_authenticated = False
        self.auth_error_count = 0
        self.max_auth_errors = 3

    def _get_base_url(self) -> str:
        """
        Get the base URL for REST API calls.

        Returns:
            Base URL for the Bybit API
        """
        return (
            "https://api-testnet.bybit.com" if self.testnet else "https://api.bybit.com"
        )

    def _get_ws_url(self) -> str:
        """
        Get the WebSocket URL.

        Returns:
            WebSocket URL for the Bybit API
        """
        return (
            "wss://stream-testnet.bybit.com"
            if self.testnet
            else "wss://stream.bybit.com"
        )

    def get_websocket_url(self, public: bool = True, category: str = "linear") -> str:
        """
        Get the appropriate WebSocket URL based on whether it's public or private.

        Args:
            public: Whether to use public or private endpoint
            category: Product category (linear, inverse, spot) - only for public

        Returns:
            WebSocket URL
        """
        if public:
            return f"{self.ws_url}/v5/public/{category}"
        else:
            return f"{self.ws_url}/v5/private"

    def set_testnet(self, testnet: bool) -> None:
        """
        Set whether to use the testnet API.

        Args:
            testnet: Whether to use the testnet API
        """
        if self.testnet != testnet:
            self.testnet = testnet
            self.base_url = self._get_base_url()
            self.ws_url = self._get_ws_url()
            logger.info(f"API mode changed to {'testnet' if testnet else 'mainnet'}")

    def set_auth_credentials(self, api_key: str, api_secret: str) -> None:
        """
        Set or update authentication credentials.

        Args:
            api_key: Bybit API key
            api_secret: Bybit API secret
        """
        if not api_key or not api_secret:
            logger.warning("Empty API credentials provided")
            return

        # Validate API key format (basic check)
        if not isinstance(api_key, str) or len(api_key.strip()) < 5:
            logger.error("Invalid API key format")
            return

        # Validate API secret format (basic check)
        if not isinstance(api_secret, str) or len(api_secret.strip()) < 5:
            logger.error("Invalid API secret format")
            return

        # Store credentials
        self.api_key = api_key.strip()
        self.api_secret = api_secret.strip()
        self.is_authenticated = True
        self.auth_error_count = 0

        logger.info(f"API credentials updated. API key: {self.api_key[:4]}***")

    def invalidate_credentials(self) -> None:
        """
        Invalidate the current authentication credentials after repeated failures.
        This helps prevent further failed API calls with known bad credentials.
        """
        self.auth_error_count += 1
        logger.warning(
            f"Authentication error count: {self.auth_error_count}/{self.max_auth_errors}"
        )

        if self.auth_error_count >= self.max_auth_errors:
            logger.error("Too many authentication failures, invalidating credentials")
            self.is_authenticated = False
            # Don't clear the credentials in case they need to be debugged

    def get_server_time(self) -> int:
        """
        Get the server time.

        Returns:
            Server time in milliseconds
        """
        # Use the V5 server time endpoint
        endpoint = "/v5/market/time"
        url = f"{self.base_url}{endpoint}"

        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                # The V5 API returns time in milliseconds directly
                return int(data.get("result", {}).get("timeSecond", 0)) * 1000
            else:
                logger.error(
                    f"Failed to get server time: {response.status_code} - {response.text}"
                )
        except Exception as e:
            logger.error(f"Error getting server time: {e}")

        # Fallback to local time
        return int(time.time() * 1000)

    def get_signature(self, params: Dict[str, Any], timestamp: int) -> str:
        """
        Generate HMAC signature for API authentication following Bybit V5 API requirements.

        Args:
            params: Request parameters
            timestamp: Current timestamp in milliseconds

        Returns:
            HMAC signature
        """
        # FIXED: Corrected signature calculation for Bybit V5 API
        logger.debug(f"Generating signature with timestamp: {timestamp}")
        logger.debug(f"API key being used: {self.api_key[:4]}***")
        logger.debug(f"API secret length: {len(self.api_secret)} chars")
        logger.debug(f"Params for signature: {params}")

        # Create the signature string according to V5 API documentation
        signature_string = str(timestamp) + self.api_key + str(self.recv_window)

        # Add params based on whether it's a GET or POST request
        query_string = ""
        if params:
            if isinstance(params, dict):
                # For GET requests: append URL-encoded query string
                sorted_params = dict(sorted(params.items()))
                query_string = urlencode(sorted_params)
                signature_string += query_string
                logger.debug(f"Query string for GET request: {query_string}")
            elif isinstance(params, str):
                # For POST requests with string payload
                signature_string += params
                logger.debug(f"String payload for POST request: {params[:100]}...")
            else:
                # For POST requests with JSON payload
                json_payload = json.dumps(params)
                signature_string += json_payload
                logger.debug(f"JSON payload for POST request: {json_payload[:100]}...")

        logger.debug(f"Full signature string: {signature_string}")

        # Generate HMAC-SHA256 signature
        signature = hmac.new(
            bytes(self.api_secret, "utf-8"),
            bytes(signature_string, "utf-8"),
            digestmod=hashlib.sha256,
        ).hexdigest()

        logger.debug(f"Generated signature: {signature}")

        return signature

    def verify_credentials(self) -> bool:
        """
        Verify that the API credentials are valid.

        Returns:
            True if credentials are valid, False otherwise
        """
        if not self.api_key or not self.api_secret:
            logger.warning("No API credentials provided to verify")
            return False

        if not self.is_authenticated:
            logger.warning("Credentials are currently marked as invalid")
            return False

        logger.debug(f"Verifying credentials with API key: {self.api_key[:4]}***")

        # Use wallet balance endpoint to verify credentials
        endpoint = "/v5/account/wallet-balance"
        url = f"{self.base_url}{endpoint}"

        timestamp = int(time.time() * 1000)
        params = {"accountType": "UNIFIED", "coin": "USDT"}

        # Generate signature
        signature = self.get_signature(params, timestamp)
        logger.debug(f"Generated signature: {signature}")

        # Set headers
        headers = {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-TIMESTAMP": str(timestamp),
            "X-BAPI-SIGN": signature,
            "X-BAPI-RECV-WINDOW": str(self.recv_window),
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        logger.debug(f"Request URL: {url}")
        logger.debug(f"Request headers: {headers}")
        logger.debug(f"Request params: {params}")

        try:
            response = requests.get(url, headers=headers, params=params)
            logger.debug(f"Response status: {response.status_code}")
            logger.debug(f"Response text: {response.text}")

            if response.status_code != 200:
                logger.warning(
                    f"API credentials verification failed with status {response.status_code}: {response.text}"
                )
                self.invalidate_credentials()
                return False

            data = response.json()

            if data.get("retCode") == 0:
                logger.info("API credentials verified successfully")
                self.auth_error_count = 0  # Reset error count on success
                return True
            else:
                error_msg = data.get("retMsg", "Unknown error")
                logger.warning(f"API credentials verification failed: {error_msg}")
                self.invalidate_credentials()
                return False

        except Exception as e:
            logger.error(f"Error verifying API credentials: {e}")
            self.invalidate_credentials()
            return False

    def get_rest_endpoint(self, endpoint: str) -> str:
        """
        Construct a full REST API endpoint URL.

        Args:
            endpoint: API endpoint path

        Returns:
            Full URL for the API endpoint
        """
        # Ensure endpoint starts with a slash
        if not endpoint.startswith("/"):
            endpoint = f"/{endpoint}"
        return f"{self.base_url}{endpoint}"
