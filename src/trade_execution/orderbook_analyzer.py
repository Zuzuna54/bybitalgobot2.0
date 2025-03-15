"""
Order Book Analyzer

This module is maintained for backward compatibility and re-exports
the functionality from the new modular orderbook package.

For new code, please use the modular imports directly from:
src.trade_execution.orderbook
"""

# Re-export from modular structure
from src.trade_execution.orderbook import (
    OrderBookAnalyzer,
    get_significant_levels,
    calculate_market_impact,
    get_optimal_trade_size,
    get_optimal_limit_price,
    detect_order_book_imbalance,
    should_split_order,
    calculate_execution_quality,
    analyze_liquidity,
    generate_entry_signal,
    generate_exit_signal,
    
    # Visualization functions
    plot_order_book_depth,
    create_order_book_heatmap,
    visualize_imbalance,
    plot_price_levels,
    save_visualization
)

# For backward compatibility
__all__ = [
    'OrderBookAnalyzer',
    'get_significant_levels',
    'calculate_market_impact',
    'get_optimal_trade_size',
    'get_optimal_limit_price',
    'detect_order_book_imbalance',
    'should_split_order',
    'calculate_execution_quality',
    'analyze_liquidity',
    'generate_entry_signal',
    'generate_exit_signal',
    'plot_order_book_depth',
    'create_order_book_heatmap',
    'visualize_imbalance',
    'plot_price_levels',
    'save_visualization'
] 