"""
Dashboard Chart Service Module (DEPRECATED)

This module is being refactored into smaller, more maintainable modules.
Please use the new modules in the src/dashboard/services/chart_service/ package:

- base.py: Base chart utilities and theme
- performance_charts.py: Performance visualization charts
- market_charts.py: Market data visualization charts
- orderbook_charts.py: Order book visualization charts
- strategy_charts.py: Strategy performance visualization charts
- component_renderers.py: UI component renderers

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

# Re-export all chart functions from the new modules
from src.dashboard.services.chart_service import *

# The original implementation is kept below for backward compatibility
# but will be removed in a future version.

# ... existing code ...
