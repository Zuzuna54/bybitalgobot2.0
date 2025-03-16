"""
Market Charts Module

This module provides chart generation functions for market data visualization.
"""

from typing import Optional
import pandas as pd
import plotly.graph_objects as go
from loguru import logger

from src.dashboard.services.chart_service.base import (
    apply_chart_theme,
    create_empty_chart,
)


def create_custom_indicator_chart(
    market_data: pd.DataFrame, indicator_name: str, symbol: str, timeframe: str
) -> go.Figure:
    """
    Create a custom indicator chart.

    Args:
        market_data: DataFrame with market data including indicators
        indicator_name: Name of the indicator to plot
        symbol: Trading symbol
        timeframe: Timeframe of the data

    Returns:
        Plotly figure object
    """
    if market_data is None or market_data.empty:
        return create_empty_chart(f"{indicator_name} - {symbol} ({timeframe})")

    if indicator_name not in market_data.columns:
        return create_empty_chart(f"{indicator_name} - {symbol} ({timeframe})")

    # Create figure with price and indicator
    fig = go.Figure()

    # Add price data as candlesticks
    fig.add_trace(
        go.Candlestick(
            x=market_data.index,
            open=market_data["open"],
            high=market_data["high"],
            low=market_data["low"],
            close=market_data["close"],
            name="Price",
            showlegend=True,
        )
    )

    # Add indicator as a line
    fig.add_trace(
        go.Scatter(
            x=market_data.index,
            y=market_data[indicator_name],
            mode="lines",
            line=dict(color="purple", width=2),
            name=indicator_name,
            yaxis="y2",
        )
    )

    # Apply theme and customize
    fig = apply_chart_theme(fig, f"{indicator_name} - {symbol} ({timeframe})")

    # Additional layout customization
    fig.update_layout(
        xaxis=dict(rangeslider=dict(visible=False), type="date"),
        yaxis=dict(title="Price", side="left"),
        yaxis2=dict(
            title=indicator_name,
            side="right",
            overlaying="y",
        ),
    )

    return fig


def create_candlestick_chart(
    candles: pd.DataFrame,
    symbol: str = "",
    title: Optional[str] = None,
    show_volume: bool = True,
    height: int = 500,
) -> go.Figure:
    """
    Create a candlestick chart with optional volume bars.

    Args:
        candles: DataFrame with OHLCV data
        symbol: Trading symbol
        title: Optional chart title
        show_volume: Whether to show volume bars
        height: Chart height in pixels

    Returns:
        Plotly figure object
    """
    if candles is None or candles.empty:
        chart_title = title if title else f"{symbol} Price Chart"
        return create_empty_chart(chart_title)

    # Create candlestick chart
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=(
                    candles["timestamp"]
                    if "timestamp" in candles.columns
                    else candles.index
                ),
                open=candles["open"],
                high=candles["high"],
                low=candles["low"],
                close=candles["close"],
                name="Price",
            )
        ]
    )

    # Add volume as bar chart on secondary y-axis
    if show_volume and "volume" in candles.columns:
        fig.add_trace(
            go.Bar(
                x=(
                    candles["timestamp"]
                    if "timestamp" in candles.columns
                    else candles.index
                ),
                y=candles["volume"],
                name="Volume",
                marker_color="rgba(128, 128, 128, 0.5)",
                yaxis="y2",
            )
        )

    # Set chart title
    chart_title = title if title else f"{symbol} Price Chart"

    # Apply standard theme
    fig = apply_chart_theme(fig, chart_title)

    # Update layout with additional settings
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Price",
        yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False),
        xaxis_rangeslider_visible=False,
        height=height,
    )

    return fig
