"""
Performance Components Package

This package contains the components for the performance tracking system.
"""

from src.performance.components.metrics_calculator import (
    PerformanceMetrics,
    calculate_metrics,
    calculate_daily_summary,
    calculate_strategy_comparison
)

from src.performance.components.report_generator import (
    generate_performance_summary,
    generate_full_performance_report,
    save_performance_report
)

from src.performance.components.visualization import (
    generate_equity_curve_chart,
    generate_profit_distribution_chart,
    generate_strategy_comparison_chart,
    generate_daily_pnl_chart,
    generate_drawdown_chart,
    generate_performance_dashboard
)

from src.performance.components.storage import (
    save_metrics_to_json,
    load_metrics_from_json,
    save_trades_to_csv,
    load_trades_from_csv,
    save_daily_summary_to_csv,
    load_daily_summary_from_csv,
    save_equity_curve_to_csv,
    load_equity_curve_from_csv,
    create_timestamped_filename
)

from src.performance.components.tracker import PerformanceTracker

__all__ = [
    # Metrics
    'PerformanceMetrics',
    'calculate_metrics',
    'calculate_daily_summary',
    'calculate_strategy_comparison',
    
    # Reporting
    'generate_performance_summary',
    'generate_full_performance_report',
    'save_performance_report',
    
    # Visualization
    'generate_equity_curve_chart',
    'generate_profit_distribution_chart',
    'generate_strategy_comparison_chart',
    'generate_daily_pnl_chart',
    'generate_drawdown_chart',
    'generate_performance_dashboard',
    
    # Storage
    'save_metrics_to_json',
    'load_metrics_from_json',
    'save_trades_to_csv',
    'load_trades_from_csv',
    'save_daily_summary_to_csv',
    'load_daily_summary_from_csv',
    'save_equity_curve_to_csv',
    'load_equity_curve_from_csv',
    'create_timestamped_filename',
    
    # Main Tracker
    'PerformanceTracker'
] 