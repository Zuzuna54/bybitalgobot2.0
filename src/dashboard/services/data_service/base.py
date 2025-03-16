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
        """
        self.api_client = api_client
        self.trade_manager = trade_manager
        self.performance_tracker = performance_tracker
        self.risk_manager = risk_manager
        self.strategy_manager = strategy_manager
        self.market_data = market_data
        self.paper_trading = paper_trading
        self.orderbook_analyzer = orderbook_analyzer

        # Data storage
        self._performance_data = {}
        self._trade_data = {}
        self._orderbook_data = {}
        self._strategy_data = {}
        self._market_data_cache = {}

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

        # Initialize in standalone mode if components are missing
        if (
            self.api_client is None
            or self.trade_manager is None
            or self.performance_tracker is None
        ):
            self._initialize_standalone_mode()

    @property
    def standalone_mode(self) -> bool:
        """
        Check if the dashboard is running in standalone mode.

        Returns:
            True if running in standalone mode, False otherwise
        """
        return (
            self.api_client is None
            or self.trade_manager is None
            or self.performance_tracker is None
        )

    def _initialize_standalone_mode(self):
        """Initialize with sample data for standalone dashboard mode."""
        logger.info("Initializing dashboard in standalone mode with sample data")

        # Initialize sample data
        _initialize_performance_data(self)
        _initialize_trade_data(self)
        _initialize_system_data(self)
        _initialize_market_data(self)
        _initialize_strategy_data(self)

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
