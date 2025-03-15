"""
Order Book Analysis Module for the Algorithmic Trading System

This package provides functionality for analyzing order book data to optimize trade entries
and exits based on market depth information.
"""

from src.trade_execution.orderbook.analyzer import OrderBookAnalyzer
from src.trade_execution.orderbook.depth_analysis import (
    get_significant_levels,
    calculate_market_impact,
    get_optimal_trade_size,
    get_optimal_limit_price
)
from src.trade_execution.orderbook.liquidity import (
    analyze_liquidity,
    should_split_order,
    calculate_execution_quality
)
from src.trade_execution.orderbook.signals import (
    detect_order_book_imbalance,
    generate_entry_signal,
    generate_exit_signal
)
from src.trade_execution.orderbook.visualization import (
    visualize_orderbook,
    plot_liquidity_profile,
    plot_order_book_depth,
    create_order_book_heatmap,
    visualize_imbalance,
    plot_price_levels,
    save_visualization
)

__all__ = [
    'OrderBookAnalyzer',
    'get_significant_levels',
    'calculate_market_impact',
    'get_optimal_trade_size',
    'get_optimal_limit_price',
    'analyze_liquidity',
    'should_split_order',
    'detect_order_book_imbalance',
    'generate_entry_signal',
    'generate_exit_signal',
    'calculate_execution_quality',
    'visualize_orderbook',
    'plot_liquidity_profile',
    'plot_order_book_depth',
    'create_order_book_heatmap',
    'visualize_imbalance',
    'plot_price_levels',
    'save_visualization'
] 