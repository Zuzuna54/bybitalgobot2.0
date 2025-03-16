"""
Dashboard Data Service Module (DEPRECATED)

This module is being refactored into smaller, more maintainable modules.
Please use the new modules in the src/dashboard/services/data_service/ package:

- base.py: Core data service class and utility functions
- performance_data.py: Performance data retrieval and processing
- trade_data.py: Trade data retrieval and processing
- system_data.py: System status and configuration data
- market_data.py: Market and orderbook data retrieval
- strategy_data.py: Strategy data retrieval and processing

This file is kept for backward compatibility but will be removed in a future version.
"""

import warnings

# Add deprecation warning
warnings.warn(
    "The data_service.py module is deprecated and will be removed in a future version. "
    "Please use the data_service package instead. "
    "Import from src.dashboard.services.data_service directly, which now refers to the package.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export all data service functions from the new package
from src.dashboard.services.data_service import *
