"""
Performance Report Generation for the Algorithmic Trading System

This module provides functionality for generating performance reports
and textual summaries of trading performance.
"""

from typing import Dict, Any, List, Optional
import os
import json
import pandas as pd

from src.performance.components.metrics_calculator import PerformanceMetrics


def generate_performance_summary(metrics: PerformanceMetrics, daily_pnl: Dict[str, float] = None) -> str:
    """
    Generate a human-readable performance summary.
    
    Args:
        metrics: Performance metrics
        daily_pnl: Dictionary of daily profit/loss values
        
    Returns:
        Summary text
    """
    if daily_pnl is None:
        daily_pnl = {}
    
    summary = []
    summary.append("====== TRADING PERFORMANCE SUMMARY ======")
    summary.append(f"Total Trades: {metrics.total_trades}")
    summary.append(f"Win Rate: {metrics.win_rate:.2f}%")
    summary.append(f"Profit Factor: {metrics.profit_factor:.2f}")
    summary.append(f"Total P&L: ${metrics.total_profit_loss:.2f}")
    summary.append(f"Return: {metrics.return_percent:.2f}%")
    summary.append(f"Max Drawdown: {metrics.max_drawdown_percent:.2f}%")
    summary.append(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    summary.append(f"Risk-Reward Ratio: {metrics.risk_reward_ratio:.2f}")
    summary.append(f"Average Profit: ${metrics.average_profit:.2f}")
    summary.append(f"Average Loss: ${metrics.average_loss:.2f}")
    summary.append(f"Largest Profit: ${metrics.largest_profit:.2f}")
    summary.append(f"Largest Loss: ${metrics.largest_loss:.2f}")
    summary.append(f"Average Trade Duration: {metrics.average_trade_duration_hours:.2f} hours")
    
    if metrics.strategy_performance:
        summary.append("\n====== STRATEGY PERFORMANCE ======")
        for strategy, stats in metrics.strategy_performance.items():
            summary.append(f"\nStrategy: {strategy}")
            summary.append(f"  Trades: {stats['count']}")
            summary.append(f"  Win Rate: {stats['win_rate']:.2f}%")
            summary.append(f"  Total P&L: ${stats['total_pnl']:.2f}")
            summary.append(f"  Avg P&L: ${stats['avg_pnl']:.2f} ({stats['avg_pnl_percent']:.2f}%)")
    
    if metrics.best_day and metrics.best_day in daily_pnl:
        summary.append(f"\nBest Day: {metrics.best_day} (${daily_pnl[metrics.best_day]:.2f})")
    if metrics.worst_day and metrics.worst_day in daily_pnl:
        summary.append(f"Worst Day: {metrics.worst_day} (${daily_pnl[metrics.worst_day]:.2f})")
    
    return "\n".join(summary)


def generate_full_performance_report(
    metrics: PerformanceMetrics,
    daily_summary: pd.DataFrame,
    active_trades: Dict[str, Dict[str, Any]],
    completed_trades: List[Dict[str, Any]],
    daily_pnl: Dict[str, float] = None,
    include_trade_list: bool = True
) -> Dict[str, Any]:
    """
    Generate a comprehensive performance report.
    
    Args:
        metrics: Performance metrics
        daily_summary: Daily performance summary DataFrame
        active_trades: Dictionary of active trades
        completed_trades: List of completed trades
        daily_pnl: Dictionary of daily profit/loss values
        include_trade_list: Whether to include full trade list
    
    Returns:
        Performance report dictionary
    """
    summary = generate_performance_summary(metrics, daily_pnl)
    
    report = {
        "summary": summary,
        "metrics": metrics.to_dict(),
        "daily_summary": daily_summary.to_dict() if not daily_summary.empty else {},
        "active_trades_count": len(active_trades),
        "completed_trades_count": len(completed_trades),
    }
    
    if include_trade_list:
        report["completed_trades"] = completed_trades
        report["active_trades"] = list(active_trades.values())
    
    return report


def save_performance_report(
    metrics: PerformanceMetrics,
    completed_trades: List[Dict[str, Any]],
    equity_history: List[Dict[str, Any]],
    data_directory: str,
    daily_summary: pd.DataFrame = None,
    file_prefix: str = ""
) -> str:
    """
    Save performance report to files.
    
    Args:
        metrics: Performance metrics
        completed_trades: List of completed trades
        equity_history: Equity history data
        data_directory: Directory to save files
        daily_summary: Daily performance summary DataFrame
        file_prefix: Prefix for filenames
        
    Returns:
        Path to main performance report file
    """
    # Make sure directory exists
    os.makedirs(data_directory, exist_ok=True)
    
    # Create timestamp for files
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{file_prefix}_" if file_prefix else ""
    
    # Create base filename
    base_filename = f"{prefix}performance_{timestamp}"
    
    # Save metrics to JSON
    metrics_file = os.path.join(data_directory, f"{base_filename}_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics.to_dict(), f, indent=2)
    
    # Save trade history to CSV
    trades_df = pd.DataFrame(completed_trades)
    if not trades_df.empty:
        trades_file = os.path.join(data_directory, f"{base_filename}_trades.csv")
        trades_df.to_csv(trades_file, index=False)
    
    # Save daily summary to CSV if provided
    if daily_summary is not None and not daily_summary.empty:
        summary_file = os.path.join(data_directory, f"{base_filename}_daily_summary.csv")
        daily_summary.to_csv(summary_file)
    
    # Save equity curve to CSV
    equity_file = os.path.join(data_directory, f"{base_filename}_equity_curve.csv")
    equity_df = pd.DataFrame(equity_history)
    equity_df.to_csv(equity_file, index=False)
    
    return metrics_file 