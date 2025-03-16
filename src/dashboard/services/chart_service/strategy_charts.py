"""
Strategy Charts Module

This module provides chart generation functions for strategy performance visualization.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from loguru import logger

from src.dashboard.services.chart_service.base import (
    apply_chart_theme,
    create_empty_chart,
)


def create_strategy_performance_graph(strategies: List[Dict[str, Any]]) -> go.Figure:
    """
    Create a strategy performance comparison graph.

    Args:
        strategies: List of strategy data dictionaries

    Returns:
        Plotly figure object
    """
    if not strategies:
        return create_empty_chart("No Strategy Data Available")

    # Create figure
    fig = go.Figure()

    # Add a trace for each strategy
    for i, strategy in enumerate(strategies):
        strategy_name = strategy.get("name", f"Strategy {i}")
        performance_data = strategy.get("performance_history", [])

        if not performance_data:
            continue

        # Convert to DataFrame if it's a list
        if isinstance(performance_data, list):
            df = pd.DataFrame(performance_data)
            if "timestamp" not in df.columns or "value" not in df.columns:
                continue

            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")
        else:
            continue

        # Add line trace for strategy
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["value"],
                mode="lines",
                name=strategy_name,
                hovertemplate="%{x}<br>" + "Value: %{y:.2f}<br>" + "<extra></extra>",
            )
        )

    # Apply standard theme
    fig = apply_chart_theme(fig, "Strategy Performance")

    # Update layout
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Performance Value",
        hovermode="x unified",
        height=500,
    )

    return fig


def create_strategy_comparison_graph(
    strategy_performance: List[Dict[str, Any]], selected_strategies: List[str] = None
) -> go.Figure:
    """
    Create a bar chart comparing key metrics across strategies.

    Args:
        strategy_performance: List of strategy performance data
        selected_strategies: Optional list of strategy names to include

    Returns:
        Plotly figure object
    """
    if not strategy_performance:
        return create_empty_chart("No Strategy Data Available")

    # Filter selected strategies if provided
    if selected_strategies:
        strategies = [
            s
            for s in strategy_performance
            if s.get("strategy_name") in selected_strategies
        ]
    else:
        strategies = strategy_performance

    if not strategies:
        return create_empty_chart("No Selected Strategies Available")

    # Extract metrics for comparison
    strategy_names = [
        s.get("strategy_name", f"Strategy {i}") for i, s in enumerate(strategies)
    ]
    win_rates = [
        s.get("win_rate", 0) * 100 for s in strategies
    ]  # Convert to percentage
    sharpe_ratios = [s.get("sharpe_ratio", 0) for s in strategies]
    max_drawdowns = [
        s.get("max_drawdown", 0) * 100 for s in strategies
    ]  # Convert to percentage
    total_returns = [
        s.get("total_return", 0) * 100 for s in strategies
    ]  # Convert to percentage

    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Win Rate (%)",
            "Sharpe Ratio",
            "Max Drawdown (%)",
            "Total Return (%)",
        ),
    )

    # Add win rate bars
    fig.add_trace(
        go.Bar(
            x=strategy_names,
            y=win_rates,
            marker_color="rgba(50, 171, 96, 0.7)",
            name="Win Rate",
        ),
        row=1,
        col=1,
    )

    # Add sharpe ratio bars
    sharpe_colors = [
        (
            "rgba(220, 53, 69, 0.7)"
            if sr < 1
            else "rgba(255, 193, 7, 0.7)" if sr < 2 else "rgba(50, 171, 96, 0.7)"
        )
        for sr in sharpe_ratios
    ]

    fig.add_trace(
        go.Bar(
            x=strategy_names,
            y=sharpe_ratios,
            marker_color=sharpe_colors,
            name="Sharpe Ratio",
        ),
        row=1,
        col=2,
    )

    # Add max drawdown bars
    fig.add_trace(
        go.Bar(
            x=strategy_names,
            y=max_drawdowns,
            marker_color="rgba(220, 53, 69, 0.7)",
            name="Max Drawdown",
        ),
        row=2,
        col=1,
    )

    # Add total return bars
    return_colors = [
        "rgba(220, 53, 69, 0.7)" if ret < 0 else "rgba(50, 171, 96, 0.7)"
        for ret in total_returns
    ]

    fig.add_trace(
        go.Bar(
            x=strategy_names,
            y=total_returns,
            marker_color=return_colors,
            name="Total Return",
        ),
        row=2,
        col=2,
    )

    # Apply standard theme
    fig = apply_chart_theme(fig, "Strategy Comparison")

    # Update layout
    fig.update_layout(
        height=600,
        showlegend=False,
        margin=dict(t=50, b=50),
    )

    return fig


def create_detailed_performance_breakdown(
    strategy_performance: List[Dict[str, Any]], selected_strategy: str = None
) -> go.Figure:
    """
    Create a detailed performance breakdown for a selected strategy.

    Args:
        strategy_performance: List of strategy performance data
        selected_strategy: Strategy name to show breakdown for

    Returns:
        Plotly figure object
    """
    if not strategy_performance or not selected_strategy:
        return create_empty_chart("No Strategy Selected")

    # Find the selected strategy
    strategy = next(
        (
            s
            for s in strategy_performance
            if s.get("strategy_name") == selected_strategy
        ),
        None,
    )

    if not strategy:
        return create_empty_chart(f"Strategy '{selected_strategy}' Not Found")

    # Extract detailed metrics
    metrics = strategy.get("metrics", {})
    if not metrics:
        return create_empty_chart("No Metrics Available")

    # Create metrics table
    metric_names = []
    metric_values = []

    for key, value in metrics.items():
        metric_names.append(key.replace("_", " ").title())

        # Format value based on type
        if isinstance(value, float):
            if "rate" in key or "ratio" in key or "return" in key:
                metric_values.append(f"{value:.2%}")
            else:
                metric_values.append(f"{value:.4f}")
        else:
            metric_values.append(str(value))

    # Create figure with table
    fig = go.Figure()

    fig.add_trace(
        go.Table(
            header=dict(
                values=["Metric", "Value"],
                fill_color="rgb(230, 230, 230)",
                align="left",
                font=dict(size=14, color="black"),
                height=40,
            ),
            cells=dict(
                values=[metric_names, metric_values],
                fill_color="white",
                align="left",
                font=dict(size=12),
                height=30,
            ),
        )
    )

    # Apply standard theme and update layout
    fig = apply_chart_theme(fig, f"{selected_strategy} - Detailed Performance")
    fig.update_layout(
        height=len(metric_names) * 30 + 100,  # Adjust height based on number of metrics
        margin=dict(l=20, r=20, t=50, b=20),
    )

    return fig


def create_market_condition_performance(
    strategy_performance: List[Dict[str, Any]], selected_strategy: str = None
) -> go.Figure:
    """
    Create a visualization of strategy performance across different market conditions.

    Args:
        strategy_performance: List of strategy performance data
        selected_strategy: Strategy name to show

    Returns:
        Plotly figure object
    """
    if not strategy_performance or not selected_strategy:
        return create_empty_chart("No Strategy Selected")

    # Find the selected strategy
    strategy = next(
        (
            s
            for s in strategy_performance
            if s.get("strategy_name") == selected_strategy
        ),
        None,
    )

    if not strategy:
        return create_empty_chart(f"Strategy '{selected_strategy}' Not Found")

    # Extract market condition performance
    market_condition_data = strategy.get("market_condition_performance", {})
    if not market_condition_data:
        return create_empty_chart("No Market Condition Data Available")

    # Extract conditions and performance values
    conditions = []
    returns = []
    win_rates = []
    trade_counts = []

    for condition, data in market_condition_data.items():
        conditions.append(condition.replace("_", " ").title())
        returns.append(data.get("return", 0) * 100)  # Convert to percentage
        win_rates.append(data.get("win_rate", 0) * 100)  # Convert to percentage
        trade_counts.append(data.get("trade_count", 0))

    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Returns by Market Condition", "Win Rate by Market Condition"),
        specs=[[{"type": "bar"}, {"type": "bar"}]],
    )

    # Add returns by condition
    fig.add_trace(
        go.Bar(
            x=conditions,
            y=returns,
            marker_color=[
                "rgba(220, 53, 69, 0.7)" if ret < 0 else "rgba(50, 171, 96, 0.7)"
                for ret in returns
            ],
            name="Return %",
            text=trade_counts,
            textposition="auto",
            hovertemplate="%{x}<br>"
            + "Return: %{y:.2f}%<br>"
            + "Trades: %{text}<br>"
            + "<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Add win rates by condition
    fig.add_trace(
        go.Bar(
            x=conditions,
            y=win_rates,
            marker_color=[
                "rgba(220, 53, 69, 0.7)" if wr < 50 else "rgba(50, 171, 96, 0.7)"
                for wr in win_rates
            ],
            name="Win Rate",
            text=trade_counts,
            textposition="auto",
            hovertemplate="%{x}<br>"
            + "Win Rate: %{y:.2f}%<br>"
            + "Trades: %{text}<br>"
            + "<extra></extra>",
        ),
        row=1,
        col=2,
    )

    # Apply standard theme
    fig = apply_chart_theme(fig, f"{selected_strategy} - Market Condition Performance")

    # Update layout
    fig.update_layout(
        height=500,
        showlegend=False,
        xaxis=dict(tickangle=-45),
        xaxis2=dict(tickangle=-45),
    )

    return fig


def create_strategy_correlation_matrix(
    strategy_performance: List[Dict[str, Any]],
) -> go.Figure:
    """
    Create a correlation matrix heatmap for strategies.

    Args:
        strategy_performance: List of strategy performance data

    Returns:
        Plotly figure object
    """
    if not strategy_performance or len(strategy_performance) < 2:
        return create_empty_chart("Insufficient Strategy Data for Correlation")

    # Extract daily returns for each strategy
    strategy_returns = {}

    for strategy in strategy_performance:
        strategy_name = strategy.get("strategy_name", "Unknown")
        daily_returns = strategy.get("daily_returns", [])

        if daily_returns:
            # Convert to DataFrame
            df = pd.DataFrame(daily_returns)
            if "date" in df.columns and "return" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")
                strategy_returns[strategy_name] = df["return"]

    if len(strategy_returns) < 2:
        return create_empty_chart("Insufficient Daily Returns Data for Correlation")

    # Create DataFrame with all returns
    returns_df = pd.DataFrame(strategy_returns)

    # Calculate correlation matrix
    corr_matrix = returns_df.corr()

    # Create heatmap
    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale=[
                [0, "rgba(220, 53, 69, 0.7)"],  # Red for negative correlation
                [0.5, "rgba(255, 255, 255, 1)"],  # White for no correlation
                [1, "rgba(50, 171, 96, 0.7)"],  # Green for positive correlation
            ],
            zmin=-1,
            zmax=1,
            text=corr_matrix.round(2).values,
            texttemplate="%{text:.2f}",
            hovertemplate="Correlation between<br>"
            + "%{y} and %{x}:<br>"
            + "%{z:.4f}<br>"
            + "<extra></extra>",
        )
    )

    # Apply standard theme
    fig = apply_chart_theme(fig, "Strategy Correlation Matrix")

    # Update layout
    fig.update_layout(
        height=500,
        xaxis=dict(side="top"),
        margin=dict(t=50, b=50),
    )

    return fig
