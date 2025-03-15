"""
Error Handling for Bybit API

This module provides functionality for handling and processing API errors,
including error classification, retry logic, and exception management.
"""

from typing import Dict, Any, Optional, Callable, TypeVar, Generic
import time
from loguru import logger


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
    
    def __init__(self, message: str, retry_after: Optional[float] = None, data: Any = None):
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


T = TypeVar('T')

def process_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process an API response and raise appropriate exceptions for errors.
    
    Args:
        response: The API response dictionary
        
    Returns:
        The response data if successful
        
    Raises:
        BybitAPIError: If the response contains an error
    """
    # Check if the response contains an error
    if response.get('retCode') != 0:
        error_code = response.get('retCode')
        error_message = response.get('retMsg', 'Unknown error')
        
        # Handle rate limit errors
        if error_code == 10006 or error_code == 10018:
            retry_after = None
            # Try to parse retry-after header or use default backoff
            raise BybitRateLimitError(error_message, retry_after, response)
        
        # Handle authentication errors
        elif error_code in (10001, 10002, 10003, 10004):
            raise BybitAuthenticationError(error_message, response)
        
        # Handle other errors
        else:
            raise BybitAPIError(error_code, error_message, response)
    
    return response


def with_error_handling(func):
    """
    Decorator for functions that make API calls to handle common errors.
    
    Args:
        func: The function to decorate
        
    Returns:
        Decorated function with error handling
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except BybitRateLimitError as e:
            logger.warning(f"Rate limit exceeded: {e.message}")
            if e.retry_after:
                logger.info(f"Retrying after {e.retry_after} seconds")
                time.sleep(e.retry_after)
                return func(*args, **kwargs)
            raise
        except BybitAuthenticationError as e:
            logger.error(f"Authentication error: {e.message}")
            raise
        except BybitAPIError as e:
            logger.error(f"API error: {e.message}")
            raise
        except Exception as e:
            logger.exception(f"Unexpected error in API call: {str(e)}")
            raise BybitNetworkError(f"Network error: {str(e)}", e)
    
    return wrapper