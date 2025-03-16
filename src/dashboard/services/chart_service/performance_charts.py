"""
Performance Charts Module

This module provides chart generation functions for performance visualization.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from loguru import logger

from src.dashboard.services.chart_service.base import (
    apply_chart_theme,
    create_empty_chart,
    filter_data_by_time_range,
)


def create_return_sparkline(returns_data: Optional[pd.DataFrame] = None) -> go.Figure:
    """
    Create a return sparkline chart.

    Args:
        returns_data: Optional dataframe with return data. If None, sample data is used.

    Returns:
        Sparkline chart figure
    """
    fig = go.Figure()

    # If no data provided, generate sample data
    if returns_data is None or returns_data.empty:
        # Generate sample data
        x = list(range(30))
        y = [10000 * (1 + 0.0023) ** i for i in range(30)]
    else:
        # Use the provided data
        x = list(range(len(returns_data)))
        y = returns_data.values.tolist()

    # Create the figure
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(color="green", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(0,255,0,0.1)",
        )
    )

    # Update layout
    fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def create_equity_curve_chart(
    equity_data: pd.DataFrame, time_range: str = "1m"
) -> go.Figure:
    """
    Create an equity curve chart from equity data.

    Args:
        equity_data: DataFrame with equity history
        time_range: Time range to display (e.g., "1d", "1w", "1m", "3m", "all")

    Returns:
        Plotly figure object
    """
    # Handle empty data case
    if equity_data is None or equity_data.empty:
        return create_empty_chart("Equity Curve")

    # Filter data by time range
    filtered_data = filter_data_by_time_range(equity_data, time_range)

    if filtered_data.empty:
        return create_empty_chart("Equity Curve")

    # Create figure
    fig = go.Figure()

    # Add equity curve line
    fig.add_trace(
        go.Scatter(
            x=filtered_data.index,
            y=filtered_data["equity"],
            mode="lines",
            name="Equity",
            line=dict(color="#636EFA", width=2),
        )
    )

    # Add initial equity line
    initial_equity = filtered_data["equity"].iloc[0]
    fig.add_trace(
        go.Scatter(
            x=[filtered_data.index[0], filtered_data.index[-1]],
            y=[initial_equity, initial_equity],
            mode="lines",
            name="Initial Equity",
            line=dict(color="#EF553B", width=1, dash="dash"),
        )
    )

    # Apply theme
    fig = apply_chart_theme(fig, "Equity Curve")

    # Add custom hover template
    fig.update_traces(
        hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>Equity: $%{y:.2f}<extra></extra>"
    )

    return fig


def create_return_distribution_chart(
    returns_data: pd.DataFrame, time_range: str = "1m"
) -> go.Figure:
    """
    Create a return distribution chart from daily returns data.

    Args:
        returns_data: DataFrame with daily returns
        time_range: Time range to display ("1d", "1w", "1m", "3m", "all")

    Returns:
        Plotly figure object
    """
    if returns_data is None or returns_data.empty:
        return create_empty_chart("Return Distribution")

    # Filter data based on time range
    filtered_data = filter_data_by_time_range(returns_data, time_range)

    if filtered_data.empty:
        return create_empty_chart("Return Distribution")

    # Create histogram
    fig = px.histogram(
        filtered_data,
        x="return_pct",
        nbins=20,
        color_discrete_sequence=["steelblue"],
        labels={"return_pct": "Daily Return (%)"},
    )

    # Add a vertical line at zero
    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=0,
        y1=1,
        yref="paper",
        line=dict(color="red", width=2, dash="dash"),
    )

    # Update layout
    fig = apply_chart_theme(fig, "Daily Return Distribution")

    # Additional layout customization
    fig.update_layout(
        showlegend=False,
        xaxis=dict(title="Daily Return (%)"),
        yaxis=dict(title="Frequency"),
    )

    return fig


def create_drawdown_chart(
    equity_data: pd.DataFrame, time_range: str = "1m"
) -> go.Figure:
    """
    Create a drawdown chart from equity data.

    Args:
        equity_data: DataFrame with equity and drawdown history
        time_range: Time range to display ("1d", "1w", "1m", "3m", "all")

    Returns:
        Plotly figure object
    """
    if (
        equity_data is None
        or equity_data.empty
        or "drawdown_pct" not in equity_data.columns
    ):
        return create_empty_chart("Drawdown Chart")

    # Filter data by time range
    filtered_data = filter_data_by_time_range(equity_data, time_range)

    if filtered_data.empty:
        return create_empty_chart("Drawdown Chart")

    # Create figure
    fig = go.Figure()

    # Add drawdown line
    fig.add_trace(
        go.Scatter(
            x=filtered_data.index,
            y=-filtered_data["drawdown_pct"],  # Negate to show as negative values
            mode="lines",
            line=dict(color="red", width=2),
            fill="tozeroy",
            fillcolor="rgba(255,0,0,0.1)",
            name="Drawdown",
        )
    )

    # Apply theme and customize
    fig = apply_chart_theme(fig, "Drawdown Over Time")

    # Additional layout customization
    fig.update_layout(
        showlegend=False,
        xaxis=dict(title="Date"),
        yaxis=dict(
            title="Drawdown (%)",
            rangemode="nonpositive",  # Only show negative values
        ),
    )

    return fig


def create_strategy_performance_chart(strategy_data: Dict[str, Any]) -> go.Figure:
    """
    Create a strategy performance comparison chart.

    Args:
        strategy_data: Dictionary with strategy performance metrics

    Returns:
        Plotly figure object
    """
    if not strategy_data or "strategies" not in strategy_data:
        return create_empty_chart("Strategy Performance")

    strategies = strategy_data.get("strategies", [])
    if len(strategies) == 0:
        return create_empty_chart("Strategy Performance")

    # Extract strategy names and returns
    names = []
    returns = []
    trades = []
    colors = []

    for strategy in strategies:
        names.append(strategy.get("name", "Unknown"))
        returns.append(strategy.get("return_pct", 0))
        trades.append(strategy.get("trades", 0))

        # Set color based on return
        if strategy.get("return_pct", 0) >= 0:
            colors.append("green")
        else:
            colors.append("red")

    # Create figure with two subplots: returns and trade count
    fig = go.Figure()

    # Add returns bars
    fig.add_trace(go.Bar(x=names, y=returns, marker_color=colors, name="Return (%)"))

    # Add trade count as a line on a secondary axis
    fig.add_trace(
        go.Scatter(
            x=names,
            y=trades,
            mode="markers+lines",
            marker=dict(size=10),
            line=dict(color="royalblue", width=2),
            name="Trade Count",
            yaxis="y2",
        )
    )

    # Apply theme and customize
    fig = apply_chart_theme(fig, "Strategy Performance Comparison")

    # Additional layout customization
    fig.update_layout(
        xaxis=dict(title="Strategy"),
        yaxis=dict(title="Return (%)", side="left"),
        yaxis2=dict(
            title="Trade Count",
            side="right",
            overlaying="y",
            rangemode="nonnegative",
        ),
        barmode="group",
    )

    return fig


def create_trade_win_loss_chart(trade_data: List[Dict[str, Any]]) -> go.Figure:
    """
    Create a win/loss chart from trade history.

    Args:
        trade_data: List of trade dictionaries

    Returns:
        Plotly figure object
    """
    if not trade_data or len(trade_data) == 0:
        return create_empty_chart("Win/Loss Distribution")

    # Count wins and losses
    wins = 0
    losses = 0

    for trade in trade_data:
        if trade.get("profitable", False):
            wins += 1
        else:
            losses += 1

    # Create donut chart
    fig = go.Figure()

    fig.add_trace(
        go.Pie(
            values=[wins, losses],
            labels=["Profitable", "Unprofitable"],
            hole=0.6,
            marker=dict(colors=["green", "red"]),
            textinfo="label+percent",
            insidetextorientation="radial",
        )
    )

    # Add text in the middle with win/loss count
    fig.add_annotation(
        text=f"{wins}/{losses}",
        font=dict(size=20, family="Arial", color="black"),
        showarrow=False,
        x=0.5,
        y=0.5,
    )

    # Apply theme and customize
    fig = apply_chart_theme(fig, "Win/Loss Distribution")

    # Additional layout customization
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    )

    return fig


def create_daily_performance_graph(daily_summary: pd.DataFrame) -> go.Figure:
    """
    Create a graph of daily performance.

    Args:
        daily_summary: DataFrame with daily performance data

    Returns:
        Plotly figure object
    """
    if daily_summary is None or daily_summary.empty:
        return create_empty_chart("Daily Performance")

    # Make a copy to avoid modifying the original
    daily_data = daily_summary.copy()

    # Create figure with dual y-axes
    fig = go.Figure()

    # Add PnL bars
    fig.add_trace(
        go.Bar(
            x=daily_data.index,
            y=daily_data["pnl"],
            name="Daily P&L",
            marker_color=[
                "rgba(46, 204, 113, 0.8)" if pnl >= 0 else "rgba(231, 76, 60, 0.8)"
                for pnl in daily_data["pnl"]
            ],
        )
    )

    # Add trade count line
    if "trade_count" in daily_data.columns:
        fig.add_trace(
            go.Scatter(
                x=daily_data.index,
                y=daily_data["trade_count"],
                name="Trade Count",
                mode="lines+markers",
                marker=dict(size=8, color="rgba(52, 152, 219, 0.8)"),
                line=dict(width=2, color="rgba(52, 152, 219, 0.8)"),
                yaxis="y2",
            )
        )

    # Apply standard theme
    fig = apply_chart_theme(fig, "Daily Performance Summary")

    # Update layout with additional settings
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="P&L ($)",
        yaxis2=dict(
            title="Trade Count", overlaying="y", side="right", rangemode="tozero"
        ),
        height=300,
    )

    return fig
