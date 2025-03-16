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

# Re-export all chart functions from the new modules
from src.dashboard.services.chart_service import *

# The original implementation is kept below for backward compatibility
# but will be removed in a future version.

# ... existing code ...
