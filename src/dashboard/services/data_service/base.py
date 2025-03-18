"""
Base Data Service Module

This module provides the core data service class and utility functions
for dashboard data access.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from loguru import logger
import threading

from src.dashboard.utils.transformers import data_transformer
from src.dashboard.services.component_registry import ComponentRegistry

# Import function implementations from specific modules
from src.dashboard.services.data_service.performance_data import (
    _initialize_performance_data,
    _update_performance_data,
    get_performance_data,
)

from src.dashboard.services.data_service.trade_data import (
    _initialize_trade_data,
    _update_trade_data,
    get_trade_data,
)

from src.dashboard.services.data_service.system_data import (
    _initialize_system_data,
    _update_system_status,
    get_system_status,
    get_data_freshness,
)

from src.dashboard.services.data_service.market_data import (
    _initialize_market_data,
    _update_orderbook_data,
    _update_market_data,
    get_orderbook_data,
    get_market_data,
)

from src.dashboard.services.data_service.strategy_data import (
    _initialize_strategy_data,
    _update_strategy_data,
    get_strategy_data,
)


class DashboardDataService:
    """Service for retrieving and processing data for the dashboard."""

    def __init__(
        self,
        component_registry=None,
        api_client=None,
        trade_manager=None,
        performance_tracker=None,
        risk_manager=None,
        strategy_manager=None,
        market_data=None,
        paper_trading=None,
        orderbook_analyzer=None,
    ):
        """
        Initialize the data service.

        Args:
            component_registry: Component registry instance (optional)
            api_client: Bybit API client
            trade_manager: Trade manager instance
            performance_tracker: Performance tracker instance
            risk_manager: Risk manager instance
            strategy_manager: Strategy manager instance
            market_data: Market data instance
            paper_trading: Paper trading simulator instance
            orderbook_analyzer: Orderbook analyzer instance
        """
        # Initialize thread-safe lock for data access
        self._lock = threading.RLock()

        # If component registry is provided, use it to get components
        if component_registry and isinstance(component_registry, ComponentRegistry):
            self.component_registry = component_registry
            self.api_client = component_registry.get("api_client")
            self.trade_manager = component_registry.get("trade_manager")
            self.performance_tracker = component_registry.get("performance_tracker")
            self.risk_manager = component_registry.get("risk_manager")
            self.strategy_manager = component_registry.get("strategy_manager")
            self.market_data = component_registry.get("market_data")
            self.paper_trading = component_registry.get("paper_trading")
            self.orderbook_analyzer = component_registry.get("orderbook_analyzer")
        else:
            # Use direct component references if no registry is provided
            self.component_registry = None
            self.api_client = api_client
            self.trade_manager = trade_manager
            self.performance_tracker = performance_tracker
            self.risk_manager = risk_manager
            self.strategy_manager = strategy_manager
            self.market_data = market_data
            self.paper_trading = paper_trading
            self.orderbook_analyzer = orderbook_analyzer

            # Create a component registry if needed for later use
            if any([api_client, trade_manager, performance_tracker]):
                self.component_registry = ComponentRegistry()
                # Register all components
                components = {}

                # Only add components that exist
                if api_client:
                    components["api_client"] = api_client
                if trade_manager:
                    components["trade_manager"] = trade_manager
                if performance_tracker:
                    components["performance_tracker"] = performance_tracker
                if risk_manager:
                    components["risk_manager"] = risk_manager
                if strategy_manager:
                    components["strategy_manager"] = strategy_manager
                if market_data:
                    components["market_data"] = market_data
                if paper_trading:
                    components["paper_trading"] = paper_trading
                if orderbook_analyzer:
                    components["orderbook_analyzer"] = orderbook_analyzer

                self.component_registry.register_many(components)

        # Validate components
        self.is_valid, self.missing_components = self._validate_components(
            self.component_registry
        )

        # Set update interval
        self.update_interval_sec = 1.0  # Default update interval
        self.last_update_time = {}

        # Initialize data storage
        self._performance_data = {}
        self._trade_data = {}
        self._orderbook_data = {}
        self._strategy_data = {}
        self._market_data_cache = {}

        # Data versioning for efficient updates
        self._data_versions = {
            "performance": 0,
            "trades": 0,
            "orderbook": 0,
            "strategy": 0,
            "market": 0,
            "system": 0,
        }

        # Data freshness tracking
        self._data_updated_at = {
            "performance": None,
            "trades": None,
            "orderbook": None,
            "strategy": None,
            "market": None,
            "system": None,
        }

        # System status
        self._system_running = False
        self._system_start_time = None
        self._system_mode = "Stopped"

        # Set operating mode
        self.is_standalone = not self.is_valid

        # Initialize in standalone mode if required components are missing
        if self.is_standalone:
            logger.warning("Running in standalone mode with limited functionality")
            self._initialize_standalone_mode()
        else:
            logger.info("Running in integrated mode with live data")
            self.update_all_data()  # Initialize with real data

    def _validate_components(self, component_registry=None):
        """
        Validate that all required components are registered.

        Args:
            component_registry: Component registry to validate. If None, uses self.component_registry.

        Returns:
            Tuple of (is_valid, missing_components)
        """
        required_components = [
            "api_client",
            "trade_manager",
            "performance_tracker",
            "strategy_manager",
        ]

        missing_components = []

        # Use provided registry or fall back to self.component_registry
        registry = component_registry or self.component_registry

        if registry:
            # Check using the registry
            for component_name in required_components:
                component = registry.get(component_name)
                if component is None:
                    missing_components.append(component_name)
                else:
                    # Validate component capabilities
                    try:
                        if component_name == "performance_tracker" and not hasattr(
                            component, "get_performance_metrics"
                        ):
                            logger.warning(
                                f"Component {component_name} is missing required method: get_performance_metrics"
                            )
                            missing_components.append(
                                f"{component_name} (missing methods)"
                            )
                        elif component_name == "trade_manager" and not hasattr(
                            component, "get_active_positions"
                        ):
                            logger.warning(
                                f"Component {component_name} is missing required method: get_active_positions"
                            )
                            missing_components.append(
                                f"{component_name} (missing methods)"
                            )
                        elif component_name == "strategy_manager" and not hasattr(
                            component, "get_strategies"
                        ):
                            logger.warning(
                                f"Component {component_name} is missing required method: get_strategies"
                            )
                            missing_components.append(
                                f"{component_name} (missing methods)"
                            )
                    except Exception as e:
                        logger.warning(
                            f"Error validating component {component_name}: {str(e)}"
                        )
                        missing_components.append(
                            f"{component_name} (validation error)"
                        )
        else:
            # Check direct references
            component_dict = {
                "api_client": self.api_client,
                "trade_manager": self.trade_manager,
                "performance_tracker": self.performance_tracker,
                "strategy_manager": self.strategy_manager,
            }

            for component_name, component in component_dict.items():
                if component is None:
                    missing_components.append(component_name)
                else:
                    # Validate component capabilities
                    try:
                        if component_name == "performance_tracker" and not hasattr(
                            component, "get_performance_metrics"
                        ):
                            logger.warning(
                                f"Component {component_name} is missing required method: get_performance_metrics"
                            )
                            missing_components.append(
                                f"{component_name} (missing methods)"
                            )
                        elif component_name == "trade_manager" and not hasattr(
                            component, "get_active_positions"
                        ):
                            logger.warning(
                                f"Component {component_name} is missing required method: get_active_positions"
                            )
                            missing_components.append(
                                f"{component_name} (missing methods)"
                            )
                        elif component_name == "strategy_manager" and not hasattr(
                            component, "get_strategies"
                        ):
                            logger.warning(
                                f"Component {component_name} is missing required method: get_strategies"
                            )
                            missing_components.append(
                                f"{component_name} (missing methods)"
                            )
                    except Exception as e:
                        logger.warning(
                            f"Error validating component {component_name}: {str(e)}"
                        )
                        missing_components.append(
                            f"{component_name} (validation error)"
                        )

        if missing_components:
            logger.warning(
                f"Missing required components: {', '.join(missing_components)}"
            )
            return False, missing_components

        return True, []

    @property
    def standalone_mode(self) -> bool:
        """
        Check if the dashboard is running in standalone mode.

        Returns:
            True if running in standalone mode, False otherwise
        """
        return self.is_standalone

    def _initialize_standalone_mode(self):
        """Initialize with sample data for standalone dashboard mode."""
        logger.info("Initializing dashboard in standalone mode with sample data")

        # Initialize sample data
        _initialize_performance_data(self)
        _initialize_trade_data(self)
        _initialize_system_data(self)
        _initialize_market_data(self)
        _initialize_strategy_data(self)

    def _initialize_data(self):
        """Initialize data storage."""
        # Data storage
        self._performance_data = {}
        self._trade_data = {}
        self._orderbook_data = {}
        self._strategy_data = {}
        self._market_data_cache = {}

        # Data versioning
        self._data_versions = {
            "performance": 0,
            "trades": 0,
            "orderbook": 0,
            "strategy": 0,
            "market": 0,
            "system": 0,
        }

        # Data freshness tracking
        self._data_updated_at = {
            "performance": None,
            "trades": None,
            "orderbook": None,
            "strategy": None,
            "market": None,
            "system": None,
        }

    def get_data_version(self, data_type):
        """
        Get the current version of a data type.

        Args:
            data_type: Type of data ('performance', 'trades', etc.)

        Returns:
            Current version number
        """
        with self._lock:
            return self._data_versions.get(data_type, 0)

    def _increment_data_version(self, data_type):
        """
        Increment the version of a data type.

        Args:
            data_type: Type of data ('performance', 'trades', etc.)
        """
        with self._lock:
            self._data_versions[data_type] = self._data_versions.get(data_type, 0) + 1
            self._data_updated_at[data_type] = datetime.now()

    def is_data_stale(self, data_type, max_age_sec=60):
        """
        Check if data is considered stale.

        Args:
            data_type: Type of data ('performance', 'trades', etc.)
            max_age_sec: Maximum age in seconds before data is considered stale

        Returns:
            True if data is stale or doesn't exist, False otherwise
        """
        with self._lock:
            last_updated = self._data_updated_at.get(data_type)
            if last_updated is None:
                return True

            age_sec = (datetime.now() - last_updated).total_seconds()
            return age_sec > max_age_sec

    def get_data_age(self, data_type):
        """
        Get the age of data in seconds.

        Args:
            data_type: Type of data ('performance', 'trades', etc.)

        Returns:
            Age in seconds or None if data doesn't exist
        """
        with self._lock:
            last_updated = self._data_updated_at.get(data_type)
            if last_updated is None:
                return None

            return (datetime.now() - last_updated).total_seconds()

    def update_all_data(self):
        """Update all dashboard data at once."""
        logger.debug("Updating all dashboard data")

        with self._lock:
            # Track current time to avoid redundant updates
            current_time = datetime.now()

            # Only update if enough time has passed since last update
            if (
                self.last_update_time.get("performance") is None
                or (
                    current_time - self.last_update_time.get("performance")
                ).total_seconds()
                > self.update_interval_sec
            ):
                _update_performance_data(self)
                self.last_update_time["performance"] = current_time

            if (
                self.last_update_time.get("trades") is None
                or (current_time - self.last_update_time.get("trades")).total_seconds()
                > self.update_interval_sec
            ):
                _update_trade_data(self)
                self.last_update_time["trades"] = current_time

            if (
                self.last_update_time.get("system") is None
                or (current_time - self.last_update_time.get("system")).total_seconds()
                > self.update_interval_sec
            ):
                _update_system_status(self)
                self.last_update_time["system"] = current_time

            if (
                self.last_update_time.get("orderbook") is None
                or (
                    current_time - self.last_update_time.get("orderbook")
                ).total_seconds()
                > self.update_interval_sec
            ):
                _update_orderbook_data(self)
                self.last_update_time["orderbook"] = current_time

            if (
                self.last_update_time.get("market") is None
                or (current_time - self.last_update_time.get("market")).total_seconds()
                > self.update_interval_sec
            ):
                _update_market_data(self)
                self.last_update_time["market"] = current_time

            if (
                self.last_update_time.get("strategy") is None
                or (
                    current_time - self.last_update_time.get("strategy")
                ).total_seconds()
                > self.update_interval_sec
            ):
                _update_strategy_data(self)
                self.last_update_time["strategy"] = current_time

    # Re-export methods from specific modules
    get_performance_data = get_performance_data
    get_trade_data = get_trade_data
    get_system_status = get_system_status
    get_data_freshness = get_data_freshness
    get_orderbook_data = get_orderbook_data
    get_market_data = get_market_data
    get_strategy_data = get_strategy_data
