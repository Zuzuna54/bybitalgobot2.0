"""
Dashboard Data Service Package

This package provides centralized data retrieval and processing for the dashboard.
It contains modules for accessing various types of trading system data.
"""

# Re-export DashboardDataService for backward compatibility
from src.dashboard.services.data_service.base import DashboardDataService

# Re-export methods from specific modules
from src.dashboard.services.data_service.performance_data import (
    get_performance_data,
)

from src.dashboard.services.data_service.trade_data import (
    get_trade_data,
)

from src.dashboard.services.data_service.system_data import (
    get_system_status,
    get_data_freshness,
)

from src.dashboard.services.data_service.market_data import (
    get_orderbook_data,
    get_market_data,
)

from src.dashboard.services.data_service.strategy_data import (
    get_strategy_data,
)

# Define all public exports from this package
__all__ = [
    # Base class
    "DashboardDataService",
    # Performance data
    "get_performance_data",
    # Trade data
    "get_trade_data",
    # System data
    "get_system_status",
    "get_data_freshness",
    # Market data
    "get_orderbook_data",
    "get_market_data",
    # Strategy data
    "get_strategy_data",
]
