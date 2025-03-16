"""
Error handling utilities for the dashboard
"""

import functools
import traceback
from typing import Callable, Any
from loguru import logger


def with_error_handling(callback_func: Callable) -> Callable:
    """
    Decorator to add error handling to dashboard callbacks.

    This decorator wraps a callback function and catches any exceptions,
    logging them and returning a fallback value to prevent the dashboard
    from crashing.

    Args:
        callback_func: The callback function to wrap

    Returns:
        Wrapped function with error handling
    """

    @functools.wraps(callback_func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return callback_func(*args, **kwargs)
        except Exception as e:
            # Log the error with traceback
            error_msg = f"Error in callback {callback_func.__name__}: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())

            # Return a fallback value based on the callback's return type annotation
            return_type = callback_func.__annotations__.get("return")

            if return_type is None:
                return None
            elif hasattr(return_type, "__origin__") and return_type.__origin__ is list:
                return []
            elif hasattr(return_type, "__origin__") and return_type.__origin__ is dict:
                return {}
            elif return_type is str:
                return f"Error: {str(e)}"
            elif return_type is bool:
                return False
            elif return_type is int or return_type is float:
                return 0
            else:
                return None

    return wrapper


def log_error(error: Exception, context: str = "") -> None:
    """
    Log an error with context information.

    Args:
        error: The exception to log
        context: Additional context information
    """
    error_msg = f"Error{' in ' + context if context else ''}: {str(error)}"
    logger.error(error_msg)
    logger.error(traceback.format_exc())
