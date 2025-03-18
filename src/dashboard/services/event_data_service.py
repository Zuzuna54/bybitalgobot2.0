"""
Event-based Data Service Module

This module integrates the EventManager with the DashboardDataService to provide
event-based real-time data updates to dashboard components.
"""

from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from loguru import logger

from src.dashboard.services.data_service.base import DashboardDataService
from src.dashboard.services.event_manager import EventManager, EventType, event_manager


class EventDataService:
    """Service that integrates EventManager with DashboardDataService for event-based updates."""

    def __init__(
        self,
        data_service: DashboardDataService,
        event_mgr: Optional[EventManager] = None,
    ):
        """
        Initialize the event data service.

        Args:
            data_service: The dashboard data service instance
            event_mgr: Event manager instance (uses the singleton if not provided)
        """
        self.data_service = data_service
        self.event_manager = event_mgr or event_manager
        self._subscribers = {}
        self._initialized = False

        # Map data types to event types
        self.data_to_event_map = {
            "performance": EventType.PERFORMANCE_UPDATE,
            "trades": EventType.TRADE_UPDATE,
            "positions": EventType.POSITION_UPDATE,
            "market": EventType.MARKET_DATA_UPDATE,
            "orderbook": EventType.MARKET_DATA_UPDATE,  # Both use the same event type
            "strategy": EventType.STRATEGY_UPDATE,
            "system": EventType.SYSTEM_STATUS_UPDATE,
        }

        # Map event types to data update methods
        self.event_to_update_method = {
            EventType.PERFORMANCE_UPDATE: self._update_performance_data,
            EventType.TRADE_UPDATE: self._update_trade_data,
            EventType.POSITION_UPDATE: self._update_trade_data,  # Same handler
            EventType.MARKET_DATA_UPDATE: self._update_market_data,
            EventType.STRATEGY_UPDATE: self._update_strategy_data,
            EventType.SYSTEM_STATUS_UPDATE: self._update_system_status,
        }

    def initialize(self):
        """Initialize the event data service and set up event subscriptions."""
        if self._initialized:
            logger.warning("Event data service already initialized")
            return

        # Subscribe to all event types
        for event_type in EventType:
            self.event_manager.subscribe(
                event_type, lambda data, et=event_type: self._handle_event(et, data)
            )

        self._initialized = True
        logger.info("Event data service initialized and subscribed to events")

    def shutdown(self):
        """Shutdown the event data service and clean up event subscriptions."""
        if not self._initialized:
            logger.warning("Event data service not initialized")
            return

        # Unsubscribe from event types (this is simplified and would need proper implementation)
        # Proper implementation would need to store references to the specific lambda functions
        # that were subscribed, to unsubscribe them later.
        self._initialized = False
        logger.info("Event data service shut down")

    def _handle_event(self, event_type: EventType, event_data: Dict[str, Any]):
        """
        Handle an event by calling the appropriate update method.

        Args:
            event_type: Type of event
            event_data: Event data
        """
        try:
            # Call the appropriate update method for this event type
            update_method = self.event_to_update_method.get(event_type)
            if update_method:
                update_method(event_data)
                logger.debug(f"Handled event: {event_type.value}")
            else:
                logger.warning(f"No update method for event type: {event_type.value}")
        except Exception as e:
            logger.error(f"Error handling event {event_type.value}: {str(e)}")

    def _update_performance_data(self, event_data: Dict[str, Any]):
        """
        Update performance data based on event.

        Args:
            event_data: Performance event data
        """
        try:
            # If event has full data, use it directly
            if event_data and "metrics" in event_data:
                # Update the data service with provided data
                self.data_service._performance_data = event_data["metrics"]
                self.data_service._increment_data_version("performance")
            else:
                # Otherwise trigger data service to fetch data
                if hasattr(self.data_service, "_update_performance_data"):
                    self.data_service._update_performance_data()
        except Exception as e:
            logger.error(f"Error updating performance data: {str(e)}")

    def _update_trade_data(self, event_data: Dict[str, Any]):
        """
        Update trade and position data based on event.

        Args:
            event_data: Trade event data
        """
        try:
            # If event contains trade information, process it
            if event_data and ("trade" in event_data or "position" in event_data):
                # Process specific trade/position updates
                if "trade" in event_data:
                    # Update specific trade in data service
                    self._process_trade_update(event_data["trade"])
                if "position" in event_data:
                    # Update specific position in data service
                    self._process_position_update(event_data["position"])
            else:
                # Otherwise trigger data service to fetch all trade data
                if hasattr(self.data_service, "_update_trade_data"):
                    self.data_service._update_trade_data()
        except Exception as e:
            logger.error(f"Error updating trade data: {str(e)}")

    def _process_trade_update(self, trade_data: Dict[str, Any]):
        """
        Process a specific trade update.

        Args:
            trade_data: Trade data to update
        """
        # Implementation depends on the specific structure of trade data
        # For now, just trigger a full update
        if hasattr(self.data_service, "_update_trade_data"):
            self.data_service._update_trade_data()
        self.data_service._increment_data_version("trades")

    def _process_position_update(self, position_data: Dict[str, Any]):
        """
        Process a specific position update.

        Args:
            position_data: Position data to update
        """
        # Implementation depends on the specific structure of position data
        # For now, just trigger a full update
        if hasattr(self.data_service, "_update_trade_data"):
            self.data_service._update_trade_data()
        self.data_service._increment_data_version("trades")

    def _update_market_data(self, event_data: Dict[str, Any]):
        """
        Update market data based on event.

        Args:
            event_data: Market data event
        """
        try:
            # Check if we have specific symbol data
            if event_data and "symbol" in event_data:
                symbol = event_data["symbol"]

                # Handle orderbook updates
                if "orderbook" in event_data:
                    self._process_orderbook_update(symbol, event_data["orderbook"])

                # Handle market data updates
                if "market_data" in event_data:
                    self._process_market_data_update(symbol, event_data["market_data"])
            else:
                # Trigger full updates
                if hasattr(self.data_service, "_update_market_data"):
                    self.data_service._update_market_data()
                if hasattr(self.data_service, "_update_orderbook_data"):
                    self.data_service._update_orderbook_data()
        except Exception as e:
            logger.error(f"Error updating market data: {str(e)}")

    def _process_orderbook_update(self, symbol: str, orderbook_data: Dict[str, Any]):
        """
        Process a specific orderbook update.

        Args:
            symbol: Symbol for the orderbook
            orderbook_data: Orderbook data to update
        """
        # Update the orderbook data in the data service
        if hasattr(self.data_service, "_orderbook_data"):
            self.data_service._orderbook_data[symbol] = orderbook_data
            self.data_service._increment_data_version("orderbook")

    def _process_market_data_update(self, symbol: str, market_data: Dict[str, Any]):
        """
        Process a specific market data update.

        Args:
            symbol: Symbol for the market data
            market_data: Market data to update
        """
        # Update the market data in the data service
        if hasattr(self.data_service, "_market_data_cache"):
            self.data_service._market_data_cache[symbol] = market_data
            self.data_service._increment_data_version("market")

    def _update_strategy_data(self, event_data: Dict[str, Any]):
        """
        Update strategy data based on event.

        Args:
            event_data: Strategy event data
        """
        try:
            # If event contains strategy ID, update specific strategy
            if event_data and "strategy_id" in event_data:
                strategy_id = event_data["strategy_id"]
                # Update specific strategy in data service
                self._process_strategy_update(strategy_id, event_data)
            else:
                # Trigger full strategy data update
                if hasattr(self.data_service, "_update_strategy_data"):
                    self.data_service._update_strategy_data()
        except Exception as e:
            logger.error(f"Error updating strategy data: {str(e)}")

    def _process_strategy_update(self, strategy_id: str, strategy_data: Dict[str, Any]):
        """
        Process a specific strategy update.

        Args:
            strategy_id: ID of the strategy to update
            strategy_data: Strategy data to update
        """
        # Update the strategy data in the data service
        if hasattr(self.data_service, "_strategy_data") and strategy_id:
            # Depending on structure, we might need to merge with existing data
            # For now, simplified implementation:
            if strategy_id in self.data_service._strategy_data:
                # Update existing strategy data
                self.data_service._strategy_data[strategy_id].update(
                    strategy_data.get("data", {})
                )
            else:
                # Add new strategy data
                self.data_service._strategy_data[strategy_id] = strategy_data.get(
                    "data", {}
                )

            self.data_service._increment_data_version("strategy")

    def _update_system_status(self, event_data: Dict[str, Any]):
        """
        Update system status based on event.

        Args:
            event_data: System status event data
        """
        try:
            # Update system status in data service
            if event_data:
                # If we have status data in the event, use it
                if hasattr(self.data_service, "_system_status"):
                    self.data_service._system_status = event_data
                    self.data_service._increment_data_version("system")
            else:
                # Trigger data service to fetch system status
                if hasattr(self.data_service, "_update_system_status"):
                    self.data_service._update_system_status()
        except Exception as e:
            logger.error(f"Error updating system status: {str(e)}")

    def publish_data_update(
        self, data_type: str, data: Optional[Dict[str, Any]] = None
    ):
        """
        Publish a data update event.

        Args:
            data_type: Type of data being updated
            data: The data to publish (optional)
        """
        # Map data type to event type
        event_type = self.data_to_event_map.get(data_type)
        if not event_type:
            logger.warning(f"No event type mapping for data type: {data_type}")
            return

        # Create the event data
        event_data = data or {}

        # Add timestamp if not present
        if "timestamp" not in event_data:
            event_data["timestamp"] = datetime.now().isoformat()

        # Add data type
        event_data["data_type"] = data_type

        # Publish the event
        self.event_manager.publish(event_type, event_data)
        logger.debug(f"Published {data_type} update event")

    def register_data_callback(
        self, data_type: str, callback: Callable[[Dict[str, Any]], None]
    ):
        """
        Register a callback for data updates.

        Args:
            data_type: Type of data to register for
            callback: Function to call when data is updated
        """
        # Map data type to event type
        event_type = self.data_to_event_map.get(data_type)
        if not event_type:
            logger.warning(f"No event type mapping for data type: {data_type}")
            return

        # Subscribe to the event
        self.event_manager.subscribe(event_type, callback)
        logger.debug(f"Registered callback for {data_type} updates")

    def unregister_data_callback(
        self, data_type: str, callback: Callable[[Dict[str, Any]], None]
    ):
        """
        Unregister a callback for data updates.

        Args:
            data_type: Type of data to unregister from
            callback: Function to unregister
        """
        # Map data type to event type
        event_type = self.data_to_event_map.get(data_type)
        if not event_type:
            logger.warning(f"No event type mapping for data type: {data_type}")
            return

        # Unsubscribe from the event
        self.event_manager.unsubscribe(event_type, callback)
        logger.debug(f"Unregistered callback for {data_type} updates")
