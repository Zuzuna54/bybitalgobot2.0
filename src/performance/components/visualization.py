"""
Performance Visualization for the Algorithmic Trading System

This module provides functionality for generating charts and visualizations
related to trading performance metrics.
"""

from typing import Dict, Any, List, Optional
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from src.performance.components.metrics_calculator import PerformanceMetrics


def generate_equity_curve_chart(
    equity_history: List[Dict[str, Any]],
    output_file: str = None,
    show_plot: bool = False,
    figsize: tuple = (12, 6)
) -> None:
    """
    Generate equity curve chart.
    
    Args:
        equity_history: List of equity history points
        output_file: Path to save the chart (if None, won't save)
        show_plot: Whether to display the plot
        figsize: Figure size as (width, height) tuple
    """
    if not equity_history or len(equity_history) <= 1:
        return
    
    plt.figure(figsize=figsize)
    
    equity_df = pd.DataFrame(equity_history)
    equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"])
    
    plt.plot(equity_df["timestamp"], equity_df["total_equity"], label="Total Equity", linewidth=2)
    plt.plot(equity_df["timestamp"], equity_df["balance"], label="Balance", linewidth=1, alpha=0.7)
    
    plt.title("Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend()
    plt.grid(True)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def generate_profit_distribution_chart(
    completed_trades: List[Dict[str, Any]],
    output_file: str = None,
    show_plot: bool = False,
    figsize: tuple = (10, 6)
) -> None:
    """
    Generate profit distribution histogram.
    
    Args:
        completed_trades: List of completed trades
        output_file: Path to save the chart (if None, won't save)
        show_plot: Whether to display the plot
        figsize: Figure size as (width, height) tuple
    """
    if not completed_trades:
        return
    
    df = pd.DataFrame(completed_trades)
    
    plt.figure(figsize=figsize)
    plt.hist(df["realized_pnl"], bins=20, alpha=0.7, color="skyblue", edgecolor="black")
    plt.axvline(x=0, color="red", linestyle="--")
    plt.title("Profit Distribution")
    plt.xlabel("Profit/Loss")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def generate_strategy_comparison_chart(
    completed_trades: List[Dict[str, Any]],
    output_file: str = None,
    show_plot: bool = False,
    figsize: tuple = (10, 6)
) -> None:
    """
    Generate strategy comparison bar chart.
    
    Args:
        completed_trades: List of completed trades
        output_file: Path to save the chart (if None, won't save)
        show_plot: Whether to display the plot
        figsize: Figure size as (width, height) tuple
    """
    if not completed_trades:
        return
    
    df = pd.DataFrame(completed_trades)
    
    if "strategy_name" not in df.columns or len(df["strategy_name"].unique()) <= 1:
        return
    
    strategy_performance = df.groupby("strategy_name").agg({
        "realized_pnl": ["count", "sum", "mean"],
        "realized_pnl_percent": ["mean"]
    })
    
    strategy_performance.columns = ["_".join(col) for col in strategy_performance.columns]
    
    plt.figure(figsize=figsize)
    bars = plt.bar(strategy_performance.index, strategy_performance["realized_pnl_sum"], alpha=0.7)
    
    # Color bars based on profit/loss
    for i, bar in enumerate(bars):
        if strategy_performance["realized_pnl_sum"].iloc[i] >= 0:
            bar.set_color("green")
        else:
            bar.set_color("red")
    
    plt.title("Strategy Performance Comparison")
    plt.xlabel("Strategy")
    plt.ylabel("Total Profit/Loss")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, alpha=0.3, axis="y")
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def generate_daily_pnl_chart(
    daily_pnl: Dict[str, float],
    output_file: str = None,
    show_plot: bool = False,
    figsize: tuple = (12, 6)
) -> None:
    """
    Generate daily profit/loss bar chart.
    
    Args:
        daily_pnl: Dictionary of daily profit/loss values
        output_file: Path to save the chart (if None, won't save)
        show_plot: Whether to display the plot
        figsize: Figure size as (width, height) tuple
    """
    if not daily_pnl:
        return
    
    daily_pnl_df = pd.DataFrame(
        {"date": list(daily_pnl.keys()), "pnl": list(daily_pnl.values())}
    )
    daily_pnl_df["date"] = pd.to_datetime(daily_pnl_df["date"])
    daily_pnl_df = daily_pnl_df.sort_values("date")
    
    plt.figure(figsize=figsize)
    bars = plt.bar(daily_pnl_df["date"], daily_pnl_df["pnl"], alpha=0.7)
    
    # Color bars based on profit/loss
    for i, bar in enumerate(bars):
        if daily_pnl_df["pnl"].iloc[i] >= 0:
            bar.set_color("green")
        else:
            bar.set_color("red")
    
    plt.title("Daily Profit/Loss")
    plt.xlabel("Date")
    plt.ylabel("Profit/Loss")
    plt.grid(True, alpha=0.3, axis="y")
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def generate_drawdown_chart(
    equity_curve: List[float],
    output_file: str = None,
    show_plot: bool = False,
    figsize: tuple = (12, 6)
) -> None:
    """
    Generate drawdown chart.
    
    Args:
        equity_curve: List of equity values
        output_file: Path to save the chart (if None, won't save)
        show_plot: Whether to display the plot
        figsize: Figure size as (width, height) tuple
    """
    if not equity_curve or len(equity_curve) <= 1:
        return
    
    equity_series = pd.Series(equity_curve)
    rolling_max = equity_series.cummax()
    drawdown_series = (rolling_max - equity_series) / rolling_max * 100
    
    plt.figure(figsize=figsize)
    plt.plot(drawdown_series, color="red", linewidth=1.5)
    plt.fill_between(drawdown_series.index, drawdown_series, 0, color="red", alpha=0.3)
    
    plt.title("Drawdown Chart")
    plt.xlabel("Trade")
    plt.ylabel("Drawdown (%)")
    plt.grid(True, alpha=0.3)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def generate_performance_dashboard(
    metrics: PerformanceMetrics,
    completed_trades: List[Dict[str, Any]],
    equity_history: List[Dict[str, Any]],
    daily_pnl: Dict[str, float],
    output_directory: str,
    base_filename: str = "performance"
) -> Dict[str, str]:
    """
    Generate a complete set of performance charts.
    
    Args:
        metrics: Performance metrics
        completed_trades: List of completed trades
        equity_history: List of equity history points
        daily_pnl: Dictionary of daily profit/loss values
        output_directory: Directory to save charts
        base_filename: Base filename for charts
        
    Returns:
        Dictionary with paths to generated charts
    """
    # Create charts directory
    charts_dir = os.path.join(output_directory, "charts")
    os.makedirs(charts_dir, exist_ok=True)
    
    # Set matplotlib style
    plt.style.use('seaborn-darkgrid')
    
    # Generate charts
    chart_files = {}
    
    # Equity curve chart
    equity_file = os.path.join(charts_dir, f"{base_filename}_equity_curve.png")
    generate_equity_curve_chart(equity_history, equity_file)
    chart_files["equity_curve"] = equity_file
    
    # Profit distribution chart
    profit_file = os.path.join(charts_dir, f"{base_filename}_profit_distribution.png")
    generate_profit_distribution_chart(completed_trades, profit_file)
    chart_files["profit_distribution"] = profit_file
    
    # Strategy comparison chart
    strategy_file = os.path.join(charts_dir, f"{base_filename}_strategy_comparison.png")
    generate_strategy_comparison_chart(completed_trades, strategy_file)
    chart_files["strategy_comparison"] = strategy_file
    
    # Daily PnL chart
    daily_file = os.path.join(charts_dir, f"{base_filename}_daily_pnl.png")
    generate_daily_pnl_chart(daily_pnl, daily_file)
    chart_files["daily_pnl"] = daily_file
    
    # Drawdown chart
    drawdown_file = os.path.join(charts_dir, f"{base_filename}_drawdown.png")
    generate_drawdown_chart(metrics.equity_curve, drawdown_file)
    chart_files["drawdown"] = drawdown_file
    
    return chart_files 