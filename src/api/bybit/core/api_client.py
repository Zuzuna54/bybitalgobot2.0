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

from src.api.bybit.core.error_handling import (
    process_response,
    with_error_handling,
    BybitAuthenticationError,
    BybitAPIError,
    BybitRateLimitError,
)
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
        BybitAuthenticationError: If authentication fails
        BybitRateLimitError: If rate limit is exceeded
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
            logger.error("Authentication required but no API credentials provided")
            raise BybitAuthenticationError(
                "Authentication required but no API credentials provided"
            )

        # Check if credentials are already known to be invalid
        if (
            hasattr(connection_manager, "is_authenticated")
            and not connection_manager.is_authenticated
        ):
            logger.error("Authentication required but credentials are invalid")
            raise BybitAuthenticationError(
                "Authentication required but credentials are invalid"
            )

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

    # Make the request with retries and exponential backoff
    remaining_retries = retry_count
    current_delay = retry_delay
    response_json = {}

    while remaining_retries >= 0:
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

            # Handle authentication errors explicitly (401, 403)
            if response.status_code in (401, 403):
                logger.error(
                    f"Authentication error ({response.status_code}) when calling {url}"
                )
                logger.debug(f"Request headers: {headers}")
                logger.debug(
                    f"API key: {connection_manager.api_key[:4]}*** Secret: {'Present' if connection_manager.api_secret else 'Missing'}"
                )

                # Try to log the response body if possible
                try:
                    response_json = response.json()
                    logger.error(f"API authentication error response: {response_json}")

                    # Invalidate credentials to prevent further failed attempts
                    if hasattr(connection_manager, "invalidate_credentials"):
                        connection_manager.invalidate_credentials()

                    # Get the specific error message if available
                    error_msg = response_json.get("retMsg", "Authentication failed")
                    raise BybitAuthenticationError(error_msg, response_json)

                except json.JSONDecodeError:
                    # If JSON parsing fails, log the raw text (if any)
                    logger.error(
                        f"API authentication error response (raw): {response.text}"
                    )
                    # For an empty response body return a custom error structure
                    error_msg = "Authentication failed - empty response received"
                    if hasattr(connection_manager, "invalidate_credentials"):
                        connection_manager.invalidate_credentials()
                    raise BybitAuthenticationError(error_msg)

            # Check for other HTTP errors
            if response.status_code >= 400:
                logger.error(
                    f"HTTP error {response.status_code} when calling {url}: {response.text}"
                )
                # Don't retry for client errors (4xx) except rate limit
                if response.status_code == 429:
                    # Rate limit error, can retry
                    retry_after = float(
                        response.headers.get("Retry-After", current_delay)
                    )
                    logger.warning(f"Rate limit exceeded, retry after {retry_after}s")
                    raise BybitRateLimitError(
                        f"Rate limit exceeded, retry after {retry_after}s", retry_after
                    )
                elif 400 <= response.status_code < 500 and response.status_code != 408:
                    # Client error, don't retry
                    raise BybitAPIError(
                        response.status_code, f"Client error: {response.text}"
                    )

            # Parse the response
            try:
                response_json = response.json()
            except json.JSONDecodeError as e:
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

            # Process the response to handle any API-level errors
            processed_response = process_response(response_json)

            # If we got here, the request was successful
            return processed_response

        except (BybitAuthenticationError, BybitAPIError) as e:
            # Don't retry authentication errors or specific API errors
            logger.error(f"API error: {e}")
            raise

        except BybitRateLimitError as e:
            # For rate limit errors, use the provided retry-after value
            wait_time = e.retry_after if e.retry_after else current_delay
            if remaining_retries > 0:
                logger.warning(
                    f"Rate limit error, retrying in {wait_time}s ({remaining_retries} retries left)"
                )
                time.sleep(wait_time)
                remaining_retries -= 1
                # Increase delay for next retry (exponential backoff)
                current_delay = min(current_delay * 2, 60)  # Cap at 60 seconds
            else:
                logger.error("Rate limit error, no retries left")
                raise

        except requests.exceptions.RequestException as e:
            # Network-related errors can be retried
            if remaining_retries > 0:
                logger.warning(
                    f"Request error: {str(e)}, retrying in {current_delay}s ({remaining_retries} retries left)"
                )
                time.sleep(current_delay)
                remaining_retries -= 1
                # Increase delay for next retry (exponential backoff)
                current_delay = min(current_delay * 2, 60)  # Cap at 60 seconds
            else:
                logger.error(f"Request error after all retries: {str(e)}")
                raise BybitAPIError(
                    0, f"Request failed after {retry_count} retries: {str(e)}"
                )

        except Exception as e:
            # Unexpected errors
            logger.error(f"Unexpected error during API request: {str(e)}")
            if remaining_retries > 0:
                logger.warning(
                    f"Retrying in {current_delay}s ({remaining_retries} retries left)"
                )
                time.sleep(current_delay)
                remaining_retries -= 1
                # Increase delay for next retry (exponential backoff)
                current_delay = min(current_delay * 2, 60)  # Cap at 60 seconds
            else:
                logger.error(f"Unexpected error after all retries: {str(e)}")
                raise BybitAPIError(
                    0, f"Unexpected error after {retry_count} retries: {str(e)}"
                )

    # This should never happen due to the exception handling above
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
        # Save original credentials to restore after test
        original_key = connection_manager.api_key
        original_secret = connection_manager.api_secret

        # Temporarily set the credentials to test
        connection_manager.set_auth_credentials(api_key, api_secret)

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
    finally:
        # Restore original credentials
        connection_manager.set_auth_credentials(original_key, original_secret)
