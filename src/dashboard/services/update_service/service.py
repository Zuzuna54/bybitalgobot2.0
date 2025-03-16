"""
Update Service Module

This module contains the core UpdateService class for managing real-time updates to the dashboard.
"""

from typing import Dict, Any, Callable, Optional, List
import threading
import time
from datetime import datetime, timedelta
from loguru import logger

from src.dashboard.services.update_service.utils import (
    is_update_due,
    get_next_update_time,
)
from src.dashboard.services.update_service.config import DEFAULT_UPDATE_INTERVALS


class UpdateService:
    """Service for managing real-time updates to the dashboard."""

    def __init__(self, data_service=None, update_intervals=None):
        """
        Initialize the update service.

        Args:
            data_service: The data service providing data for updates
            update_intervals: Dictionary of update intervals in seconds for each data type
        """
        self.data_service = data_service

        # Update intervals in seconds - use defaults or provided values
        self.update_intervals = update_intervals or DEFAULT_UPDATE_INTERVALS.copy()

        # Last update times
        self.last_updates = {data_type: None for data_type in self.update_intervals}

        # Update callbacks
        self.update_callbacks: Dict[str, List[Callable]] = {
            data_type: [] for data_type in self.update_intervals
        }

        # Running state
        self._running = False
        self._update_thread = None
        self._start_time = None

    def start(self) -> None:
        """Start the update service."""
        if self._running:
            logger.warning("Update service is already running")
            return

        logger.info("Starting update service")
        self._running = True
        self._start_time = datetime.now()

        # Start update thread
        self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self._update_thread.start()

    def stop(self) -> None:
        """Stop the update service."""
        if not self._running:
            logger.warning("Update service is not running")
            return

        logger.info("Stopping update service")
        self._running = False

        # Wait for update thread to finish
        if self._update_thread and self._update_thread.is_alive():
            self._update_thread.join(timeout=1.0)

    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the update service.

        Returns:
            Dictionary with update service status information
        """
        return {
            "running": self._running,
            "start_time": self._start_time.isoformat() if self._start_time else None,
            "update_intervals": self.update_intervals,
            "last_updates": {
                data_type: update_time.isoformat() if update_time else None
                for data_type, update_time in self.last_updates.items()
            },
            "next_updates": {
                data_type: (
                    self.get_next_update_time(data_type).isoformat()
                    if self.get_next_update_time(data_type)
                    else None
                )
                for data_type in self.update_intervals
            },
            "callback_counts": {
                data_type: len(callbacks)
                for data_type, callbacks in self.update_callbacks.items()
            },
        }

    def _update_loop(self) -> None:
        """Main update loop that runs in a separate thread."""
        logger.debug("Update loop started")

        while self._running:
            # Check all data types for updates
            for data_type in self.update_intervals:
                try:
                    self._check_and_update(data_type)
                except Exception as e:
                    logger.error(f"Error in update loop for {data_type}: {str(e)}")

            # Sleep a short time to avoid high CPU usage
            time.sleep(0.1)

    def _check_and_update(self, data_type: str) -> None:
        """
        Check if an update is due for the data type and perform it if needed.

        Args:
            data_type: Type of data to check for update
        """
        # Skip if data type is not in the update intervals
        if data_type not in self.update_intervals:
            return

        # Check if update is due
        if self.is_update_due(data_type):
            logger.debug(f"Update due for {data_type}")

            # Process the update
            self._process_update(data_type)

    def _process_update(self, data_type: str) -> None:
        """
        Process an update for the specified data type.

        Args:
            data_type: Type of data to update
        """
        # Update the data using the data service if available
        if self.data_service:
            if hasattr(self.data_service, "update_all_data") and data_type == "all":
                self.data_service.update_all_data()
            elif hasattr(self.data_service, f"_update_{data_type}_data"):
                # Call the appropriate update method
                update_method = getattr(self.data_service, f"_update_{data_type}_data")
                update_method()

        # Update the last update time
        self.last_updates[data_type] = datetime.now()

        # Notify callbacks
        self._notify_callbacks(data_type)

    def _notify_callbacks(self, data_type: str) -> None:
        """
        Notify registered callbacks about a data update.

        Args:
            data_type: Type of data that was updated
        """
        # Skip if data type doesn't have any callbacks
        if data_type not in self.update_callbacks:
            return

        # Call each registered callback
        for callback in self.update_callbacks[data_type]:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in update callback for {data_type}: {str(e)}")

    def get_last_update_time(self, data_type: str) -> Optional[datetime]:
        """
        Get the time of the last update for a data type.

        Args:
            data_type: Type of data to get last update time for

        Returns:
            Datetime of last update or None if not updated yet
        """
        return self.last_updates.get(data_type)

    def get_next_update_time(self, data_type: str) -> Optional[datetime]:
        """
        Get the time of the next scheduled update for a data type.

        Args:
            data_type: Type of data to get next update time for

        Returns:
            Datetime of next update or None if not applicable
        """
        return get_next_update_time(
            self.get_last_update_time(data_type),
            self.update_intervals.get(data_type, 0),
        )

    def is_update_due(self, data_type: str) -> bool:
        """
        Check if an update is due for a data type.

        Args:
            data_type: Type of data to check

        Returns:
            True if an update is due, False otherwise
        """
        return is_update_due(
            self.get_last_update_time(data_type),
            self.update_intervals.get(data_type, 0),
        )
