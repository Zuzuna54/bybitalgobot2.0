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
        api_client=None,
        trade_manager=None,
        performance_tracker=None,
        risk_manager=None,
        strategy_manager=None,
        market_data=None,
        paper_trading=None,
        orderbook_analyzer=None,
        component_registry=None,
    ):
        """
        Initialize the data service.

        Args:
            api_client: Bybit API client
            trade_manager: Trade manager instance
            performance_tracker: Performance tracker instance
            risk_manager: Risk manager instance
            strategy_manager: Strategy manager instance
            market_data: Market data instance
            paper_trading: Paper trading simulator instance
            orderbook_analyzer: Orderbook analyzer instance
            component_registry: Component registry instance (optional)
        """
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
            if all([api_client, trade_manager, performance_tracker]):
                self.component_registry = ComponentRegistry()
                # Register all components
                components = {
                    "api_client": api_client,
                    "trade_manager": trade_manager,
                    "performance_tracker": performance_tracker,
                    "risk_manager": risk_manager,
                    "strategy_manager": strategy_manager,
                    "market_data": market_data,
                }
                # Add optional components if available
                if paper_trading:
                    components["paper_trading"] = paper_trading
                if orderbook_analyzer:
                    components["orderbook_analyzer"] = orderbook_analyzer

                self.component_registry.register_many(components)

        # Validate components
        self.is_valid, self.missing_components = self._validate_components(
            self.component_registry
        )

        # Data storage
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
                if not registry.is_registered(component_name):
                    missing_components.append(component_name)
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
        return self._data_versions.get(data_type, 0)

    def _increment_data_version(self, data_type):
        """
        Increment the version of a data type.

        Args:
            data_type: Type of data ('performance', 'trades', etc.)
        """
        self._data_versions[data_type] = self._data_versions.get(data_type, 0) + 1
        self._data_updated_at[data_type] = datetime.now()

    def update_all_data(self):
        """Update all dashboard data at once."""
        logger.debug("Updating all dashboard data")

        _update_performance_data(self)
        _update_trade_data(self)
        _update_system_status(self)
        _update_orderbook_data(self)
        _update_market_data(self)
        _update_strategy_data(self)

    # Re-export methods from specific modules
    get_performance_data = get_performance_data
    get_trade_data = get_trade_data
    get_system_status = get_system_status
    get_data_freshness = get_data_freshness
    get_orderbook_data = get_orderbook_data
    get_market_data = get_market_data
    get_strategy_data = get_strategy_data
