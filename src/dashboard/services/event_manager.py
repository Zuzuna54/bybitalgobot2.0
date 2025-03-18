"""
Event Manager Module

This module implements an event-based system for real-time data updates in the dashboard.
It manages event publishing, subscribing, and dispatching to enable efficient
communication between dashboard components.
"""

import threading
from typing import Dict, Any, List, Callable, Optional
from enum import Enum
import queue
import time
from datetime import datetime
from loguru import logger


class EventType(Enum):
    """Types of events in the system."""

    MARKET_DATA_UPDATE = "market_data_update"
    TRADE_UPDATE = "trade_update"
    POSITION_UPDATE = "position_update"
    PERFORMANCE_UPDATE = "performance_update"
    STRATEGY_UPDATE = "strategy_update"
    SYSTEM_STATUS_UPDATE = "system_status_update"


class EventManager:
    """Manager for system events and callbacks."""

    def __init__(self):
        """Initialize the event manager."""
        self._subscribers = {}
        self._event_queue = queue.Queue()
        self._running = False
        self._thread = None
        self._lock = threading.RLock()
        self._last_event_times = {}

    def subscribe(
        self, event_type: EventType, callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """
        Subscribe to an event.

        Args:
            event_type: Type of event
            callback: Function to call when event occurs
        """
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []

            if callback not in self._subscribers[event_type]:
                self._subscribers[event_type].append(callback)
                logger.debug(f"Subscribed to event: {event_type.value}")
            else:
                logger.debug(f"Already subscribed to event: {event_type.value}")

    def unsubscribe(
        self, event_type: EventType, callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """
        Unsubscribe from an event.

        Args:
            event_type: Type of event
            callback: Function to unsubscribe
        """
        with self._lock:
            if event_type in self._subscribers:
                if callback in self._subscribers[event_type]:
                    self._subscribers[event_type].remove(callback)
                    logger.debug(f"Unsubscribed from event: {event_type.value}")
                else:
                    logger.debug(f"Callback not found for event: {event_type.value}")

    def publish(
        self, event_type: EventType, data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Publish an event.

        Args:
            event_type: Type of event
            data: Event data
        """
        current_time = time.time()
        event = {"type": event_type, "data": data or {}, "timestamp": current_time}

        # Update last event time for this type
        self._last_event_times[event_type] = current_time

        # Add event to the queue
        self._event_queue.put(event)
        logger.debug(f"Published event: {event_type.value}")

    def start(self) -> None:
        """Start the event processing thread."""
        with self._lock:
            if self._running:
                logger.warning("Event manager is already running")
                return

            logger.info("Starting event manager")
            self._running = True
            self._thread = threading.Thread(target=self._process_events, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        """Stop the event processing thread."""
        with self._lock:
            if not self._running:
                logger.warning("Event manager is not running")
                return

            logger.info("Stopping event manager")
            self._running = False

            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=1.0)
                if self._thread.is_alive():
                    logger.warning("Event manager thread did not terminate gracefully")

    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the event manager.

        Returns:
            Dictionary with event manager status information
        """
        with self._lock:
            status = {
                "running": self._running,
                "queue_size": self._event_queue.qsize(),
                "subscriber_counts": {
                    event_type.value: len(subscribers)
                    for event_type, subscribers in self._subscribers.items()
                },
                "last_event_times": {
                    event_type.value: timestamp
                    for event_type, timestamp in self._last_event_times.items()
                },
            }
            return status

    def _process_events(self) -> None:
        """Process events from the queue."""
        logger.debug("Event processing thread started")

        while self._running:
            try:
                # Get event with timeout to allow for stopping
                event = self._event_queue.get(timeout=0.1)

                # Process the event
                event_type = event["type"]
                event_data = event["data"]

                # Notify subscribers
                self._notify_subscribers(event_type, event_data)

                # Mark as done
                self._event_queue.task_done()

            except queue.Empty:
                pass  # No events to process
            except Exception as e:
                logger.error(f"Error processing events: {str(e)}")

        logger.debug("Event processing thread stopped")

    def _notify_subscribers(
        self, event_type: EventType, event_data: Dict[str, Any]
    ) -> None:
        """
        Notify subscribers of an event.

        Args:
            event_type: Type of event
            event_data: Event data
        """
        # Get a copy of subscribers to avoid issues if list changes during iteration
        with self._lock:
            subscribers = self._subscribers.get(event_type, []).copy()

        # Call each subscriber
        for callback in subscribers:
            try:
                callback(event_data)
            except Exception as e:
                logger.error(
                    f"Error in event callback for {event_type.value}: {str(e)}"
                )


# Create singleton instance
event_manager = EventManager()
