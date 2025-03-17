"""
Core API Client Functionality for Bybit Exchange

This module provides core functionality for making HTTP requests to the Bybit API,
integrated with error handling and rate limiting.
"""

import json
import time
import hmac
import hashlib
import requests
from typing import Dict, Any, Optional, Union
from urllib.parse import urlencode
from loguru import logger

from src.api.bybit.core.error_handling import process_response, with_error_handling
from src.api.bybit.core.rate_limiting import rate_limited
from src.api.bybit.core.connection import ConnectionManager


def generate_signature(
    api_key: str, api_secret: str, params: Dict[str, Any] = None
) -> Dict[str, str]:
    """
    Generate authentication headers for Bybit API requests.

    Args:
        api_key: Bybit API key
        api_secret: Bybit API secret
        params: Request parameters

    Returns:
        Dictionary of authentication headers
    """
    if params is None:
        params = {}

    timestamp = int(time.time() * 1000)
    params_with_auth = {**params, "api_key": api_key, "timestamp": timestamp}

    # Sort parameters by key
    sorted_params = {k: params_with_auth[k] for k in sorted(params_with_auth)}
    query_string = urlencode(sorted_params)

    # Generate signature
    signature = hmac.new(
        api_secret.encode("utf-8"), query_string.encode("utf-8"), hashlib.sha256
    ).hexdigest()

    # Add signature to params
    params_with_auth["sign"] = signature

    return params_with_auth


@with_error_handling
@rate_limited()
def make_request(
    connection_manager: ConnectionManager,
    method: str,
    endpoint: str,
    params: Optional[Dict[str, Any]] = None,
    auth_required: bool = False,
    retry_count: int = 3,
    retry_delay: float = 1.0,
) -> Dict[str, Any]:
    """
    Make a request to the Bybit API with error handling and rate limiting.

    Args:
        connection_manager: Connection manager instance
        method: HTTP method (GET, POST, etc.)
        endpoint: API endpoint
        params: Request parameters
        auth_required: Whether authentication is required for this request
        retry_count: Number of times to retry on failure
        retry_delay: Delay between retries (in seconds)

    Returns:
        API response data

    Raises:
        BybitAPIError: If the request fails or returns an error
    """
    # Build full URL
    url = connection_manager.get_rest_endpoint(endpoint)

    # Initialize request parameters
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    request_params = None
    json_data = None

    # Prepare request parameters
    if params is None:
        params = {}

    # Add authentication if needed
    if auth_required:
        if not connection_manager.api_key or not connection_manager.api_secret:
            raise ValueError("Authentication required but no API credentials provided")

        timestamp = int(time.time() * 1000)

        # Generate signature using connection manager
        signature = connection_manager.get_signature(params, timestamp)

        # Add authentication headers
        headers["X-BAPI-API-KEY"] = connection_manager.api_key
        headers["X-BAPI-TIMESTAMP"] = str(timestamp)
        headers["X-BAPI-SIGN"] = signature
        headers["X-BAPI-RECV-WINDOW"] = str(connection_manager.recv_window)

    # Prepare the request data
    if method.upper() == "GET":
        request_params = params
    else:
        json_data = params

    # Log the request (without sensitive data)
    logger.debug(f"API Request: {method} {url}")

    # Make the request with retries
    remaining_retries = retry_count
    response_json = {}

    while remaining_retries > 0:
        try:
            # Make the request
            response = requests.request(
                method=method.upper(),
                url=url,
                params=request_params,
                json=json_data,
                headers=headers,
                timeout=10,
            )

            # Better handling for authentication errors (401)
            if response.status_code == 401:
                logger.error(f"Authentication error (401) when calling {url}")
                logger.debug(f"Request headers: {headers}")
                logger.debug(
                    f"API key: {connection_manager.api_key[:4]}*** Secret: {'Present' if connection_manager.api_secret else 'Missing'}"
                )

                # Try to log the response body if possible
                try:
                    response_json = response.json()
                    logger.error(f"API authentication error response: {response_json}")
                except Exception:
                    # If JSON parsing fails, log the raw text (if any)
                    logger.error(
                        f"API authentication error response (raw): {response.text}"
                    )
                    # For an empty response body return a custom error structure
                    if not response.text:
                        return {
                            "retCode": 401,
                            "retMsg": "Authentication failed - empty response received",
                            "result": {},
                        }

            # Parse the response
            try:
                response_json = response.json()
            except Exception as e:
                logger.error(f"Failed to parse JSON response: {e}")
                # Return a structured error for empty responses
                if not response.text:
                    return {
                        "retCode": response.status_code,
                        "retMsg": f"Failed to parse empty response: {str(e)}",
                        "result": {},
                    }
                # Try to return the raw text if JSON parsing fails
                return {
                    "retCode": response.status_code,
                    "retMsg": f"Failed to parse response: {str(e)}",
                    "result": {"raw_text": response.text[:1000]},  # Limit the size
                }

            # Process the response to handle any errors
            return process_response(response_json)

        except requests.exceptions.RequestException as e:
            # Handle request exceptions
            logger.warning(f"API request failed: {str(e)}")
            remaining_retries -= 1

            if remaining_retries > 0:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                # Increase delay for next retry (exponential backoff)
                retry_delay *= 2
            else:
                logger.error(f"API request failed after {retry_count} retries")
                raise

    # This should never happen, but just in case
    return response_json


def verify_signature(
    api_key: str, api_secret: str, received_signature: str, params: Dict[str, Any]
) -> bool:
    """
    Verify a signature received from Bybit.

    Args:
        api_key: Bybit API key
        api_secret: Bybit API secret
        received_signature: Signature to verify
        params: Request parameters

    Returns:
        True if signature is valid, False otherwise
    """
    # Generate the expected signature from the parameters
    auth_params = generate_signature(api_key, api_secret, params)
    expected_signature = auth_params.get("sign", "")

    # Compare the signatures
    return received_signature == expected_signature


def is_api_key_valid(
    connection_manager: ConnectionManager, api_key: str, api_secret: str
) -> bool:
    """
    Check if an API key is valid.

    Args:
        connection_manager: Connection manager instance
        api_key: Bybit API key
        api_secret: Bybit API secret

    Returns:
        True if the key is valid, False otherwise
    """
    try:
        # Try to get wallet balance to check if the API key is valid
        endpoint = "/v5/account/wallet-balance"
        params = {"accountType": "UNIFIED"}

        response = make_request(
            connection_manager=connection_manager,
            method="GET",
            endpoint=endpoint,
            params=params,
            auth_required=True,
        )

        # If we get a response without an error, the key is valid
        return True
    except Exception as e:
        logger.warning(f"API key validation failed: {str(e)}")
        return False
