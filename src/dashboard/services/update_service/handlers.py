"""
Update Handlers Module

This module provides functions for registering and managing update handlers (callbacks).
"""

from typing import Callable, Dict, List, Optional
import threading
from loguru import logger


# Global registry for update handlers
# This allows registration outside of UpdateService instance
_update_handlers: Dict[str, List[Callable]] = {}
_handlers_lock = threading.Lock()


def register_update_handler(data_type: str, callback: Callable) -> None:
    """
    Register a callback function to be called when data is updated.

    Args:
        data_type: Type of data to register for
        callback: Function to call when data is updated
    """
    with _handlers_lock:
        if data_type not in _update_handlers:
            _update_handlers[data_type] = []

        if callback not in _update_handlers[data_type]:
            _update_handlers[data_type].append(callback)
            logger.debug(f"Registered update handler for {data_type}")
        else:
            logger.debug(f"Handler already registered for {data_type}")


def unregister_update_handler(data_type: str, callback: Callable) -> None:
    """
    Unregister a callback function from data updates.

    Args:
        data_type: Type of data to unregister from
        callback: Function to unregister
    """
    with _handlers_lock:
        if data_type in _update_handlers and callback in _update_handlers[data_type]:
            _update_handlers[data_type].remove(callback)
            logger.debug(f"Unregistered update handler for {data_type}")
        else:
            logger.debug(f"Handler not found for {data_type}")


def trigger_update(data_type: str, update_service=None) -> None:
    """
    Trigger an update for a specific data type.

    This will force an update regardless of the normal update interval.

    Args:
        data_type: Type of data to update
        update_service: Optional UpdateService instance to use
    """
    if update_service:
        try:
            # Process the update in the service
            update_service._process_update(data_type)
            logger.debug(f"Triggered update for {data_type}")
        except Exception as e:
            logger.error(f"Error triggering update for {data_type}: {str(e)}")
    else:
        # Just call the registered handlers
        _notify_handlers(data_type)


def _notify_handlers(data_type: str) -> None:
    """
    Notify registered handlers about a data update.

    Args:
        data_type: Type of data that was updated
    """
    with _handlers_lock:
        if data_type not in _update_handlers:
            return

        handlers = _update_handlers[data_type].copy()

    # Call each handler (outside the lock)
    for handler in handlers:
        try:
            handler()
        except Exception as e:
            logger.error(f"Error in update handler for {data_type}: {str(e)}")


def get_registered_handlers() -> Dict[str, int]:
    """
    Get information about registered handlers.

    Returns:
        Dictionary with data types and number of handlers
    """
    with _handlers_lock:
        return {
            data_type: len(handlers) for data_type, handlers in _update_handlers.items()
        }
