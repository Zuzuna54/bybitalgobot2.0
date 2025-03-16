"""
Chart Service Package

This package provides chart generation functionality for the dashboard.
It contains modules for creating various types of charts and visualizations,
organized by functionality.
"""

# Re-export all chart functions for backward compatibility
from src.dashboard.services.chart_service.base import (
    apply_chart_theme,
    create_empty_chart,
    create_empty_sparkline,
    filter_data_by_time_range,
    CHART_THEME,
)

from src.dashboard.services.chart_service.performance_charts import (
    create_return_sparkline,
    create_equity_curve_chart,
    create_return_distribution_chart,
    create_drawdown_chart,
    create_strategy_performance_chart,
    create_trade_win_loss_chart,
    create_daily_performance_graph,
)

from src.dashboard.services.chart_service.market_charts import (
    create_custom_indicator_chart,
    create_candlestick_chart,
)

from src.dashboard.services.chart_service.orderbook_charts import (
    create_orderbook_depth_chart,
    create_orderbook_heatmap,
    create_orderbook_imbalance_chart,
    create_liquidity_profile_chart,
    create_orderbook_depth_graph,  # Alias for backward compatibility
)

from src.dashboard.services.chart_service.strategy_charts import (
    create_strategy_performance_graph,
    create_strategy_comparison_graph,
    create_detailed_performance_breakdown,
    create_market_condition_performance,
    create_strategy_correlation_matrix,
)

from src.dashboard.services.chart_service.component_renderers import (
    render_imbalance_indicator,
    render_liquidity_ratio,
    render_support_resistance_levels,
    render_level_strength_indicator,
    create_level_confluence_chart,
    render_execution_recommendations,
)

# Define all public exports from this package
__all__ = [
    # Base
    "CHART_THEME",
    "apply_chart_theme",
    "create_empty_chart",
    "create_empty_sparkline",
    "filter_data_by_time_range",
    # Performance charts
    "create_return_sparkline",
    "create_equity_curve_chart",
    "create_return_distribution_chart",
    "create_drawdown_chart",
    "create_strategy_performance_chart",
    "create_trade_win_loss_chart",
    "create_daily_performance_graph",
    # Market charts
    "create_custom_indicator_chart",
    "create_candlestick_chart",
    # Orderbook charts
    "create_orderbook_depth_chart",
    "create_orderbook_heatmap",
    "create_orderbook_imbalance_chart",
    "create_liquidity_profile_chart",
    "create_orderbook_depth_graph",  # Alias for backward compatibility
    # Strategy charts
    "create_strategy_performance_graph",
    "create_strategy_comparison_graph",
    "create_detailed_performance_breakdown",
    "create_market_condition_performance",
    "create_strategy_correlation_matrix",
    # Component renderers
    "render_imbalance_indicator",
    "render_liquidity_ratio",
    "render_support_resistance_levels",
    "render_level_strength_indicator",
    "create_level_confluence_chart",
    "render_execution_recommendations",
]
