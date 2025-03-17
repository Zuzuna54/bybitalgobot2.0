"""
Error Handling for Bybit API

This module provides functionality for handling and processing API errors,
including error classification, retry logic, and exception management.
"""

from typing import Dict, Any, Optional, Callable, TypeVar, Generic
import time
from loguru import logger
import requests
import json


class BybitAPIError(Exception):
    """Base exception for Bybit API errors."""

    def __init__(self, status_code: int, message: str, data: Any = None):
        """
        Initialize a Bybit API error.

        Args:
            status_code: HTTP status code
            message: Error message
            data: Additional error data
        """
        self.status_code = status_code
        self.message = message
        self.data = data
        super().__init__(f"Bybit API Error ({status_code}): {message}")


class BybitRateLimitError(BybitAPIError):
    """Exception for rate limit errors."""

    def __init__(
        self, message: str, retry_after: Optional[float] = None, data: Any = None
    ):
        """
        Initialize a rate limit error.

        Args:
            message: Error message
            retry_after: Time in seconds to wait before retrying
            data: Additional error data
        """
        self.retry_after = retry_after
        super().__init__(429, message, data)


class BybitAuthenticationError(BybitAPIError):
    """Exception for authentication errors."""

    def __init__(self, message: str, data: Any = None):
        """
        Initialize an authentication error.

        Args:
            message: Error message
            data: Additional error data
        """
        super().__init__(401, message, data)


class BybitNetworkError(BybitAPIError):
    """Exception for network-related errors."""

    def __init__(self, message: str, original_error: Optional[Exception] = None):
        """
        Initialize a network error.

        Args:
            message: Error message
            original_error: The original exception that caused this error
        """
        self.original_error = original_error
        super().__init__(0, message)


T = TypeVar("T")


def process_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process an API response and raise appropriate exceptions for errors.

    Args:
        response: The API response dictionary

    Returns:
        The response data if successful

    Raises:
        BybitAPIError: If the response contains an error
        BybitAuthenticationError: If the response indicates an authentication error
        BybitRateLimitError: If the response indicates a rate limit error
    """
    # Check if the response contains an error
    if response.get("retCode") != 0:
        error_code = response.get("retCode")
        error_message = response.get("retMsg", "Unknown error")

        # Log the error details
        logger.debug(f"API error detected: Code={error_code}, Message={error_message}")

        # Handle authentication errors (expanded list of error codes)
        if error_code in (10001, 10002, 10003, 10004, 10005, 10006, 10010):
            logger.error(f"Authentication error: {error_message}")
            # Attempt to get connection manager to invalidate credentials if possible
            connection_manager = None
            for arg in list(response.get("args", [])):
                if hasattr(arg, "invalidate_credentials"):
                    connection_manager = arg
                    break

            if connection_manager:
                logger.debug("Notifying connection manager of authentication failure")
                connection_manager.invalidate_credentials()

            raise BybitAuthenticationError(error_message, response)

        # Handle rate limit errors
        elif error_code in (10006, 10018, 10029, 10030):
            retry_after = None
            # Try to parse retry-after information if available
            if "retry_after" in response:
                retry_after = float(response.get("retry_after", 1.0))

            logger.warning(
                f"Rate limit error: {error_message}, retry after: {retry_after}s"
            )
            raise BybitRateLimitError(error_message, retry_after, response)

        # Handle general API errors
        else:
            logger.error(f"API error: {error_code} - {error_message}")
            raise BybitAPIError(error_code, error_message, response)

    return response


def with_error_handling(func):
    """
    Decorator for handling Bybit API errors.

    Args:
        func: Function to decorate

    Returns:
        Decorated function
    """

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except BybitAPIError:
            # Already handled, just re-raise
            raise
        except requests.exceptions.HTTPError as e:
            # Handle HTTP errors
            status_code = e.response.status_code if hasattr(e, "response") else 0

            if status_code in (401, 403):
                # Authentication error
                error_msg = f"Authentication failed: {str(e)}"
                logger.error(error_msg)

                # Try to extract more diagnostic info
                response = getattr(e, "response", None)
                if response:
                    try:
                        error_data = response.json()
                        logger.error(f"Authentication error details: {error_data}")

                        # Try to invalidate credentials if connection_manager is available
                        for arg in args:
                            if hasattr(arg, "invalidate_credentials"):
                                logger.debug(
                                    "Notifying connection manager of authentication failure"
                                )
                                arg.invalidate_credentials()
                                break

                        return BybitAuthenticationError(error_msg, error_data)
                    except json.JSONDecodeError:
                        logger.error(
                            f"Authentication error raw response: {response.text}"
                        )

                raise BybitAuthenticationError(error_msg)
            elif status_code == 429:
                # Rate limit error
                error_msg = f"Rate limit exceeded: {str(e)}"
                logger.warning(error_msg)

                # Try to get retry-after header
                retry_after = None
                response = getattr(e, "response", None)
                if response and "Retry-After" in response.headers:
                    retry_after = float(response.headers["Retry-After"])

                raise BybitRateLimitError(error_msg, retry_after)
            else:
                # Other HTTP errors
                error_msg = f"HTTP error {status_code}: {str(e)}"
                logger.error(error_msg)
                raise BybitAPIError(status_code, error_msg)
        except requests.exceptions.RequestException as e:
            # Handle network errors
            error_msg = f"Network error: {str(e)}"
            logger.error(error_msg)
            raise BybitNetworkError(error_msg, e)
        except json.JSONDecodeError as e:
            # Handle JSON decode errors
            error_msg = f"JSON decode error: {str(e)}"
            logger.error(error_msg)
            raise BybitNetworkError(error_msg, e)
        except Exception as e:
            # Handle all other errors
            logger.error(f"Unexpected error in API call: {str(e)}")
            raise BybitAPIError(0, f"Unexpected error: {str(e)}")

    return wrapper
