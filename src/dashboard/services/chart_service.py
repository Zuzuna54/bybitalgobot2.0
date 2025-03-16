"""
Dashboard Chart Service Module

This module provides chart generation functionality for the dashboard.
It contains functions for creating various charts and visualizations.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from loguru import logger
from plotly.subplots import make_subplots
import colorsys
from dash import html, dcc
import dash_bootstrap_components as dbc


# Standard chart theme configuration
CHART_THEME = {
    "template": "plotly_white",
    "font": {"family": "Arial, sans-serif", "size": 12, "color": "#444"},
    "title_font": {"family": "Arial, sans-serif", "size": 16, "color": "#333"},
    "legend": {
        "orientation": "h",
        "yanchor": "bottom",
        "y": 1.02,
        "xanchor": "right",
        "x": 1,
    },
    "colorway": [
        "#636EFA",
        "#EF553B",
        "#00CC96",
        "#AB63FA",
        "#FFA15A",
        "#19D3F3",
        "#FF6692",
        "#B6E880",
        "#FF97FF",
        "#FECB52",
    ],
    "grid": {"showgrid": True, "gridcolor": "#f0f0f0", "zeroline": False},
    "margin": {"l": 40, "r": 40, "t": 40, "b": 40},
}


def apply_chart_theme(fig: go.Figure, title: Optional[str] = None) -> go.Figure:
    """
    Apply the standard chart theme to a figure.

    Args:
        fig: The figure to apply the theme to
        title: Optional title for the chart

    Returns:
        The figure with theme applied
    """
    fig.update_layout(
        template=CHART_THEME["template"],
        font=CHART_THEME["font"],
        margin=CHART_THEME["margin"],
        colorway=CHART_THEME["colorway"],
        legend=CHART_THEME["legend"],
    )

    # Apply title if provided
    if title:
        fig.update_layout(
            title={
                "text": title,
                "font": CHART_THEME["title_font"],
                "x": 0.5,
                "xanchor": "center",
            }
        )

    # Apply grid settings
    fig.update_xaxes(
        showgrid=CHART_THEME["grid"]["showgrid"],
        gridcolor=CHART_THEME["grid"]["gridcolor"],
        zeroline=CHART_THEME["grid"]["zeroline"],
    )
    fig.update_yaxes(
        showgrid=CHART_THEME["grid"]["showgrid"],
        gridcolor=CHART_THEME["grid"]["gridcolor"],
        zeroline=CHART_THEME["grid"]["zeroline"],
    )

    return fig


def create_empty_chart(title: str) -> go.Figure:
    """
    Create an empty chart figure with a title.

    Args:
        title: The chart title

    Returns:
        Plotly figure object
    """
    fig = go.Figure()

    # Apply standard theme
    fig = apply_chart_theme(fig, title)

    # Add "no data" annotation
    fig.add_annotation(
        x=0.5,
        y=0.5,
        text="No data available",
        showarrow=False,
        font=dict(size=16, color="#888888"),
    )

    return fig


def create_empty_sparkline() -> go.Figure:
    """
    Create an empty sparkline chart.

    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    # Add a flat line
    x = list(range(10))
    y = [0] * 10
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color="gray", width=1)))
    return fig


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


def filter_data_by_time_range(data: pd.DataFrame, time_range: str) -> pd.DataFrame:
    """
    Filter data by time range.

    Args:
        data: DataFrame with datetime index
        time_range: Time range to filter by (e.g., "1d", "1w", "1m", "3m", "all")

    Returns:
        Filtered DataFrame
    """
    if time_range == "all" or data.empty:
        return data

    now = datetime.now()

    if time_range == "1d":
        start_date = now - timedelta(days=1)
    elif time_range == "1w":
        start_date = now - timedelta(weeks=1)
    elif time_range == "1m":
        start_date = now - timedelta(days=30)
    elif time_range == "3m":
        start_date = now - timedelta(days=90)
    else:
        # Default to 1 month if invalid range
        start_date = now - timedelta(days=30)

    try:
        return data[data.index >= start_date]
    except Exception as e:
        logger.error(f"Error filtering data by time range: {str(e)}")
        return data


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
    end_date = returns_data.index.max()
    if time_range == "1d":
        start_date = end_date - timedelta(days=1)
    elif time_range == "1w":
        start_date = end_date - timedelta(days=7)
    elif time_range == "1m":
        start_date = end_date - timedelta(days=30)
    elif time_range == "3m":
        start_date = end_date - timedelta(days=90)
    else:  # "all"
        start_date = returns_data.index.min()

    # Filter data
    filtered_data = returns_data.loc[start_date:end_date]

    if len(filtered_data) == 0:
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
    fig.update_layout(
        title="Daily Return Distribution",
        template="plotly_white",
        showlegend=False,
        xaxis=dict(showgrid=True, zeroline=False, title="Daily Return (%)"),
        yaxis=dict(showgrid=True, zeroline=False, title="Frequency"),
        margin=dict(l=40, r=40, t=40, b=40),
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

    # Filter data based on time range
    end_date = equity_data.index.max()
    if time_range == "1d":
        start_date = end_date - timedelta(days=1)
    elif time_range == "1w":
        start_date = end_date - timedelta(days=7)
    elif time_range == "1m":
        start_date = end_date - timedelta(days=30)
    elif time_range == "3m":
        start_date = end_date - timedelta(days=90)
    else:  # "all"
        start_date = equity_data.index.min()

    # Filter data
    filtered_data = equity_data.loc[start_date:end_date]

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

    # Update layout
    fig.update_layout(
        title="Drawdown Over Time",
        template="plotly_white",
        showlegend=False,
        xaxis=dict(showgrid=True, zeroline=False, title="Date"),
        yaxis=dict(
            showgrid=True,
            zeroline=True,
            title="Drawdown (%)",
            rangemode="nonpositive",  # Only show negative values
        ),
        margin=dict(l=40, r=40, t=40, b=40),
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

    # Update layout
    fig.update_layout(
        title="Strategy Performance Comparison",
        template="plotly_white",
        xaxis=dict(showgrid=False, zeroline=False, title="Strategy"),
        yaxis=dict(showgrid=True, zeroline=True, title="Return (%)", side="left"),
        yaxis2=dict(
            showgrid=False,
            zeroline=False,
            title="Trade Count",
            side="right",
            overlaying="y",
            rangemode="nonnegative",
        ),
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
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

    # Update layout
    fig.update_layout(
        title="Win/Loss Distribution",
        template="plotly_white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=40, b=20),
    )

    return fig


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

    # Update layout
    fig.update_layout(
        title=f"{indicator_name} - {symbol} ({timeframe})",
        template="plotly_white",
        xaxis=dict(
            showgrid=False, zeroline=False, rangeslider=dict(visible=False), type="date"
        ),
        yaxis=dict(showgrid=True, zeroline=False, title="Price", side="left"),
        yaxis2=dict(
            showgrid=False,
            zeroline=False,
            title=indicator_name,
            side="right",
            overlaying="y",
        ),
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
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


def create_orderbook_depth_chart(
    orderbook: Dict[str, Any],
    depth: int = 20,
    support_levels: Optional[List[float]] = None,
    resistance_levels: Optional[List[float]] = None,
    height: int = 500,
) -> go.Figure:
    """
    Create an orderbook depth chart visualization.

    Args:
        orderbook: Dictionary with bids and asks arrays
        depth: Number of price levels to show
        support_levels: Optional list of support price levels
        resistance_levels: Optional list of resistance price levels
        height: Chart height in pixels

    Returns:
        Plotly figure object
    """
    if not orderbook or "bids" not in orderbook or "asks" not in orderbook:
        return create_empty_chart("Orderbook Depth")

    bids = orderbook["bids"]
    asks = orderbook["asks"]

    if not bids or not asks:
        return create_empty_chart("Orderbook Depth")

    # Get current market price (midpoint between best bid and best ask)
    best_bid = bids[0][0] if bids else None
    best_ask = asks[0][0] if asks else None

    if best_bid is None or best_ask is None:
        return create_empty_chart("Orderbook Depth")

    # Calculate midpoint price
    mid_price = (best_bid + best_ask) / 2

    # Limit the number of price levels based on depth parameter
    limited_bids = bids[:depth] if len(bids) > depth else bids
    limited_asks = asks[:depth] if len(asks) > depth else asks

    # Extract prices and sizes
    bid_prices = [bid[0] for bid in limited_bids]
    bid_sizes = [bid[1] for bid in limited_bids]
    ask_prices = [ask[0] for ask in limited_asks]
    ask_sizes = [ask[1] for ask in limited_asks]

    # Calculate cumulative sizes
    bid_cumulative = np.cumsum(bid_sizes)
    ask_cumulative = np.cumsum(ask_sizes)

    # Create figure
    fig = go.Figure()

    # Add bid depth trace
    fig.add_trace(
        go.Scatter(
            x=bid_prices[::-1],  # Reverse to show highest bids on left
            y=bid_cumulative[::-1],
            mode="lines",
            name="Bids",
            line=dict(color="rgba(50, 171, 96, 0.8)", width=2),
            fill="tozeroy",
            fillcolor="rgba(50, 171, 96, 0.3)",
        )
    )

    # Add ask depth trace
    fig.add_trace(
        go.Scatter(
            x=ask_prices,
            y=ask_cumulative,
            mode="lines",
            name="Asks",
            line=dict(color="rgba(220, 53, 69, 0.8)", width=2),
            fill="tozeroy",
            fillcolor="rgba(220, 53, 69, 0.3)",
        )
    )

    # Add support levels if provided
    if support_levels:
        for level in support_levels:
            if level < best_bid * 0.7 or level > best_ask * 1.3:
                continue  # Skip levels that are too far away

            fig.add_shape(
                type="line",
                x0=level,
                y0=0,
                x1=level,
                y1=max(max(bid_cumulative), max(ask_cumulative)) * 0.9,
                line=dict(color="green", width=1.5, dash="dash"),
                name=f"Support: {level:.2f}",
            )

    # Add resistance levels if provided
    if resistance_levels:
        for level in resistance_levels:
            if level < best_bid * 0.7 or level > best_ask * 1.3:
                continue  # Skip levels that are too far away

            fig.add_shape(
                type="line",
                x0=level,
                y0=0,
                x1=level,
                y1=max(max(bid_cumulative), max(ask_cumulative)) * 0.9,
                line=dict(color="red", width=1.5, dash="dash"),
                name=f"Resistance: {level:.2f}",
            )

    # Apply standard theme
    fig = apply_chart_theme(fig, "Orderbook Depth")

    # Add current price line
    fig.add_shape(
        type="line",
        x0=mid_price,
        y0=0,
        x1=mid_price,
        y1=max(max(bid_cumulative), max(ask_cumulative)),
        line=dict(color="rgba(70, 130, 180, 1)", width=2),
        name="Current Price",
    )

    # Add midpoint annotation
    fig.add_annotation(
        x=mid_price,
        y=max(max(bid_cumulative), max(ask_cumulative)) * 0.95,
        text=f"${mid_price:.2f}",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="rgba(70, 130, 180, 1)",
        font=dict(size=12, color="rgba(70, 130, 180, 1)"),
    )

    # Update layout
    fig.update_layout(
        xaxis_title="Price",
        yaxis_title="Cumulative Size",
        height=height,
        hovermode="x unified",
    )

    # Update hover templates
    fig.update_traces(
        hovertemplate="<b>Price</b>: $%{x:.2f}<br><b>Cumulative Size</b>: %{y:.4f}<extra></extra>"
    )

    return fig


def create_orderbook_heatmap(
    orderbook: Dict[str, Any],
    height: int = 500,
    title: str = "Order Book Heatmap",
) -> go.Figure:
    """
    Create a heatmap visualization of the orderbook.

    Args:
        orderbook: Dictionary with bids and asks arrays
        height: Chart height in pixels
        title: Chart title

    Returns:
        Plotly figure object
    """
    if not orderbook or "bids" not in orderbook or "asks" not in orderbook:
        # Return empty figure if no valid data
        fig = go.Figure()
        fig.update_layout(
            title="No orderbook data available",
            annotations=[
                dict(
                    text="Missing orderbook data",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                )
            ],
        )
        return fig

    # Extract orderbook data
    bids = orderbook.get("bids", [])
    asks = orderbook.get("asks", [])

    if not bids or not asks:
        # Return empty figure if no bids or asks
        fig = go.Figure()
        fig.update_layout(
            title="Insufficient orderbook data",
            annotations=[
                dict(
                    text="No bid or ask data available",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                )
            ],
        )
        return fig

    # Process the data for the heatmap
    # We'll use price as y-axis and quantity as the intensity (z-value)
    # Each side (bid/ask) will be a separate trace

    # Sort bids (descending) and asks (ascending) by price
    bids_sorted = sorted(bids, key=lambda x: float(x[0]), reverse=True)
    asks_sorted = sorted(asks, key=lambda x: float(x[0]))

    # Limit the number of levels to visualize
    max_levels = 20
    bids_display = bids_sorted[:max_levels]
    asks_display = asks_sorted[:max_levels]

    # Extract prices and quantities
    bid_prices = [float(bid[0]) for bid in bids_display]
    bid_quantities = [float(bid[1]) for bid in bids_display]

    ask_prices = [float(ask[0]) for ask in asks_display]
    ask_quantities = [float(ask[1]) for ask in asks_display]

    # Normalize quantities for color intensity
    max_quantity = max(max(bid_quantities, default=1), max(ask_quantities, default=1))
    bid_intensities = [qty / max_quantity for qty in bid_quantities]
    ask_intensities = [qty / max_quantity for qty in ask_quantities]

    # Create the figure with separate heatmaps for bids and asks
    fig = go.Figure()

    # Add bid heatmap (green)
    fig.add_trace(
        go.Heatmap(
            y=bid_prices,
            z=[[intensity] for intensity in bid_intensities],
            colorscale=[[0, "rgba(0,100,0,0.1)"], [1, "rgba(0,255,0,0.8)"]],
            showscale=False,
            name="Bids",
            hovertemplate="Price: %{y}<br>Quantity: %{text}<extra>Bids</extra>",
            text=bid_quantities,
        )
    )

    # Add ask heatmap (red)
    fig.add_trace(
        go.Heatmap(
            y=ask_prices,
            z=[[intensity] for intensity in ask_intensities],
            colorscale=[[0, "rgba(100,0,0,0.1)"], [1, "rgba(255,0,0,0.8)"]],
            showscale=False,
            name="Asks",
            hovertemplate="Price: %{y}<br>Quantity: %{text}<extra>Asks</extra>",
            text=ask_quantities,
        )
    )

    # Add a line trace to mark the spread
    if bid_prices and ask_prices:
        best_bid = max(bid_prices)
        best_ask = min(ask_prices)
        spread = best_ask - best_bid
        spread_pct = (spread / best_bid) * 100

        mid_price = (best_bid + best_ask) / 2

        fig.add_trace(
            go.Scatter(
                x=[0, 0],
                y=[best_bid, best_ask],
                mode="lines",
                line=dict(color="gray", width=1, dash="dash"),
                name=f"Spread: {spread:.6f} ({spread_pct:.3f}%)",
                hoverinfo="name",
            )
        )

        # Add annotation for the mid price
        fig.add_annotation(
            x=0,
            y=mid_price,
            text=f"Mid: {mid_price:.6f}",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1,
        )

    # Update layout
    fig.update_layout(
        title=title,
        yaxis_title="Price",
        height=height,
        template="plotly_white",
        xaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            fixedrange=True,
        ),
        yaxis=dict(
            side="right",
            zeroline=False,
        ),
    )

    # Apply standard theme
    fig = apply_chart_theme(fig, title)

    return fig


def create_orderbook_imbalance_chart(
    orderbook: Dict[str, Any],
    depth_levels: int = 10,
    height: int = 300,
    title: str = "Order Book Imbalance",
) -> go.Figure:
    """
    Create a visualization of order book imbalance using Plotly.

    Args:
        orderbook: Order book data with bids and asks
        depth_levels: Number of price levels to consider
        height: Height of the chart in pixels
        title: Chart title

    Returns:
        Plotly figure object
    """
    # Extract bids and asks
    bids = orderbook.get("bids", [])
    asks = orderbook.get("asks", [])

    if not bids or not asks:
        return create_empty_chart("No Orderbook Data Available")

    # Limit to specified depth
    bids = bids[:depth_levels] if depth_levels < len(bids) else bids
    asks = asks[:depth_levels] if depth_levels < len(asks) else asks

    # Calculate total volume on each side
    bid_volume = sum(float(bid[1]) for bid in bids)
    ask_volume = sum(float(ask[1]) for ask in asks)

    # Calculate imbalance
    total_volume = bid_volume + ask_volume

    if total_volume == 0:
        imbalance = 0
    else:
        imbalance = (bid_volume - ask_volume) / total_volume

    # Calculate percentages for the chart
    bid_pct = (bid_volume / total_volume * 100) if total_volume > 0 else 0
    ask_pct = (ask_volume / total_volume * 100) if total_volume > 0 else 0

    # Create figure
    fig = go.Figure()

    # Add horizontal bars
    fig.add_trace(
        go.Bar(
            y=["Volume"],
            x=[ask_pct],
            name="Ask Volume",
            orientation="h",
            marker=dict(color="rgba(220, 20, 60, 0.7)"),  # Crimson for asks
            hovertemplate="Ask Volume: %{x:.1f}%<extra></extra>",
        )
    )

    fig.add_trace(
        go.Bar(
            y=["Volume"],
            x=[bid_pct],
            name="Bid Volume",
            orientation="h",
            marker=dict(color="rgba(46, 139, 87, 0.7)"),  # Sea green for bids
            hovertemplate="Bid Volume: %{x:.1f}%<extra></extra>",
        )
    )

    # Apply chart theme
    fig = apply_chart_theme(fig, title)

    # Calculate imbalance color (red for negative, green for positive)
    imbalance_color = "rgb(46, 139, 87)" if imbalance > 0 else "rgb(220, 20, 60)"

    # Add imbalance annotation
    fig.add_annotation(
        x=0.5,
        y=-0.3,
        xref="paper",
        yref="paper",
        text=f"Imbalance: {imbalance:.3f}",
        showarrow=False,
        font=dict(size=14, color=imbalance_color),
        bgcolor="rgba(255, 255, 255, 0.7)",
        bordercolor=imbalance_color,
        borderwidth=1,
        borderpad=4,
    )

    # Update layout
    fig.update_layout(
        height=height,
        barmode="stack",
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(0, 0, 0, 0.1)",
            tickvals=[0, 25, 50, 75, 100],
            ticktext=["0%", "25%", "50%", "75%", "100%"],
            range=[0, 100],
        ),
        yaxis=dict(showticklabels=False),
        legend=dict(orientation="h", y=1.1, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=40, b=80),
    )

    # Add center line
    fig.add_shape(
        type="line",
        x0=50,
        y0=0,
        x1=50,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(color="black", width=1, dash="dash"),
    )

    return fig


def create_liquidity_profile_chart(
    orderbook: Dict[str, Any],
    price_range_pct: float = 2.0,
    height: int = 500,
    title: str = "Orderbook Liquidity Profile",
) -> go.Figure:
    """
    Create a visualization of order book liquidity profile using Plotly.

    Args:
        orderbook: Order book data with bids and asks
        price_range_pct: Price range to display as percentage from mid price
        height: Height of the chart in pixels
        title: Chart title

    Returns:
        Plotly figure object
    """
    # Extract bids and asks
    bids = orderbook.get("bids", [])
    asks = orderbook.get("asks", [])

    if not bids or not asks:
        return create_empty_chart("No Orderbook Data Available")

    # Convert to DataFrame if needed
    if not isinstance(bids, pd.DataFrame):
        bids_df = pd.DataFrame(bids, columns=["price", "size"])
        asks_df = pd.DataFrame(asks, columns=["price", "size"])
    else:
        bids_df = bids.copy()
        asks_df = asks.copy()

    # Ensure numeric data types
    bids_df["price"] = pd.to_numeric(bids_df["price"])
    bids_df["size"] = pd.to_numeric(bids_df["size"])
    asks_df["price"] = pd.to_numeric(asks_df["price"])
    asks_df["size"] = pd.to_numeric(asks_df["size"])

    # Calculate mid price
    best_bid = bids_df["price"].iloc[0] if not bids_df.empty else 0
    best_ask = asks_df["price"].iloc[0] if not asks_df.empty else 0
    mid_price = (best_bid + best_ask) / 2

    # Calculate price range
    price_range = mid_price * price_range_pct / 100
    min_price = mid_price - price_range
    max_price = mid_price + price_range

    # Filter orders within price range
    bids_in_range = bids_df[bids_df["price"] >= min_price]
    asks_in_range = asks_df[asks_df["price"] <= max_price]

    # Group by price ranges for better visualization
    num_bins = 20
    bid_bins = np.linspace(min_price, mid_price, num_bins // 2 + 1)
    ask_bins = np.linspace(mid_price, max_price, num_bins // 2 + 1)

    # Group bids by price bins
    bids_in_range["bin"] = pd.cut(bids_in_range["price"], bins=bid_bins)
    bid_liquidity = bids_in_range.groupby("bin")["size"].sum().reset_index()
    bid_liquidity["mid_price"] = bid_liquidity["bin"].apply(
        lambda x: (x.left + x.right) / 2
    )

    # Group asks by price bins
    asks_in_range["bin"] = pd.cut(asks_in_range["price"], bins=ask_bins)
    ask_liquidity = asks_in_range.groupby("bin")["size"].sum().reset_index()
    ask_liquidity["mid_price"] = ask_liquidity["bin"].apply(
        lambda x: (x.left + x.right) / 2
    )

    # Create figure
    fig = go.Figure()

    # Add bid liquidity
    fig.add_trace(
        go.Bar(
            x=bid_liquidity["mid_price"],
            y=bid_liquidity["size"],
            name="Bid Liquidity",
            marker=dict(color="rgba(46, 139, 87, 0.7)"),  # Sea green for bids
            hovertemplate="Price: %{x:.2f}<br>Size: %{y:.2f}<extra></extra>",
        )
    )

    # Add ask liquidity
    fig.add_trace(
        go.Bar(
            x=ask_liquidity["mid_price"],
            y=ask_liquidity["size"],
            name="Ask Liquidity",
            marker=dict(color="rgba(220, 20, 60, 0.7)"),  # Crimson for asks
            hovertemplate="Price: %{x:.2f}<br>Size: %{y:.2f}<extra></extra>",
        )
    )

    # Add mid price line
    fig.add_shape(
        type="line",
        x0=mid_price,
        y0=0,
        x1=mid_price,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(color="black", width=1, dash="dash"),
    )

    fig.add_annotation(
        x=mid_price,
        y=1,
        text=f"Mid: {mid_price:.2f}",
        showarrow=False,
        yanchor="bottom",
        bgcolor="rgba(255, 255, 255, 0.7)",
    )

    # Apply chart theme
    fig = apply_chart_theme(fig, title)

    # Update layout
    fig.update_layout(
        height=height,
        xaxis_title="Price",
        yaxis_title="Liquidity (Size)",
        bargap=0.01,
        legend=dict(orientation="h", y=1.02, xanchor="right", x=1),
    )

    return fig


# Strategy Visualization Functions


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


# -------------------------
# ORDERBOOK VISUALIZATION FUNCTIONS
# -------------------------


# Alias for backward compatibility
def create_orderbook_depth_graph(*args, **kwargs):
    """
    DEPRECATED: Use create_orderbook_depth_chart instead.

    This function is kept for backward compatibility and calls
    create_orderbook_depth_chart.
    """
    import warnings

    warnings.warn(
        "create_orderbook_depth_graph is deprecated. "
        "Use create_orderbook_depth_chart instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return create_orderbook_depth_chart(*args, **kwargs)


def render_imbalance_indicator(imbalance: float) -> html.Div:
    """
    Render the order book imbalance indicator.

    Args:
        imbalance: Imbalance score (-1.0 to 1.0)

    Returns:
        HTML Div with imbalance visualization
    """
    if imbalance is None:
        return html.Div("No data available")

    # Calculate color based on imbalance
    if imbalance > 0:
        # Buy pressure (green)
        color = f"rgba(50, 171, 96, {min(abs(imbalance) + 0.3, 1.0)})"
        label = "Buy Pressure"
    else:
        # Sell pressure (red)
        color = f"rgba(220, 53, 69, {min(abs(imbalance) + 0.3, 1.0)})"
        label = "Sell Pressure"

    # Calculate width percentage
    width_pct = abs(imbalance) * 100

    # Create indicator components
    return html.Div(
        [
            html.H4(f"{imbalance:.2f}", className="text-center"),
            html.P(label, className="text-center"),
            html.Div(
                [
                    html.Div(
                        className="imbalance-bar",
                        style={
                            "width": f"{width_pct}%",
                            "background-color": color,
                            "height": "10px",
                            "border-radius": "5px",
                        },
                    )
                ],
                className="imbalance-container",
            ),
        ],
        className="orderbook-imbalance-indicator",
    )


def render_liquidity_ratio(buy_sell_ratio: float) -> html.Div:
    """
    Render the liquidity ratio indicator.

    Args:
        buy_sell_ratio: Ratio of buy to sell liquidity (> 1 means more buy liquidity)

    Returns:
        HTML Div with liquidity ratio visualization
    """
    if buy_sell_ratio is None:
        return html.Div("No data available")

    # Calculate percentage for the gauge display
    if buy_sell_ratio > 1:
        # More buy liquidity
        pct = min(buy_sell_ratio / 2, 1) * 50 + 50
        color = "rgba(50, 171, 96, 0.8)"  # Green
        label = "Buy Liquidity Dominance"
    else:
        # More sell liquidity
        pct = max(buy_sell_ratio, 0) * 50
        color = "rgba(220, 53, 69, 0.8)"  # Red
        label = "Sell Liquidity Dominance"

    # Create gauge indicator
    return html.Div(
        [
            html.H4(f"{buy_sell_ratio:.2f}x", className="text-center"),
            html.P(label, className="text-center"),
            html.Div(
                [
                    html.Div(
                        className="liquidity-gauge",
                        style={"transform": f"rotate({pct * 1.8}deg)"},
                    ),
                    html.Div(className="liquidity-gauge-scale"),
                ],
                className="liquidity-gauge-container",
            ),
        ],
        className="liquidity-ratio-indicator",
    )


def render_support_resistance_levels(levels_data: Dict[str, Any]) -> html.Div:
    """
    Render support and resistance levels information.

    Args:
        levels_data: Dictionary containing support and resistance level information

    Returns:
        HTML Div with support and resistance levels
    """
    if not levels_data or not any(
        [
            levels_data.get("support_levels", []),
            levels_data.get("resistance_levels", []),
        ]
    ):
        return html.Div(
            [
                html.P(
                    "No significant support or resistance levels detected",
                    className="text-center text-muted",
                )
            ]
        )

    support_levels = levels_data.get("support_levels", [])
    resistance_levels = levels_data.get("resistance_levels", [])

    # Create support levels list
    support_items = []
    for level in sorted(support_levels, key=lambda x: x.get("price", 0), reverse=True):
        price = level.get("price", 0)
        strength = level.get("strength", 0.5)
        distance = level.get("distance_pct", 0)

        support_items.append(
            dbc.ListGroupItem(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Span(
                                        f"${price:.2f}",
                                        className="fw-bold me-2",
                                    ),
                                    html.Span(
                                        f"{distance:.2f}% away",
                                        className="small text-muted",
                                    ),
                                ],
                                width=6,
                            ),
                            dbc.Col(
                                render_level_strength_indicator(
                                    strength, is_support=True
                                ),
                                width=6,
                            ),
                        ]
                    )
                ],
                className="support-level-item",
            )
        )

    # Create resistance levels list
    resistance_items = []
    for level in sorted(resistance_levels, key=lambda x: x.get("price", 0)):
        price = level.get("price", 0)
        strength = level.get("strength", 0.5)
        distance = level.get("distance_pct", 0)

        resistance_items.append(
            dbc.ListGroupItem(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Span(
                                        f"${price:.2f}",
                                        className="fw-bold me-2",
                                    ),
                                    html.Span(
                                        f"{distance:.2f}% away",
                                        className="small text-muted",
                                    ),
                                ],
                                width=6,
                            ),
                            dbc.Col(
                                render_level_strength_indicator(
                                    strength, is_support=False
                                ),
                                width=6,
                            ),
                        ]
                    )
                ],
                className="resistance-level-item",
            )
        )

    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H6("Resistance Levels", className="text-center"),
                            (
                                dbc.ListGroup(resistance_items, className="mb-3")
                                if resistance_items
                                else html.P(
                                    "No resistance detected",
                                    className="text-center text-muted small",
                                )
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            html.H6("Support Levels", className="text-center"),
                            (
                                dbc.ListGroup(support_items, className="mb-3")
                                if support_items
                                else html.P(
                                    "No support detected",
                                    className="text-center text-muted small",
                                )
                            ),
                        ],
                        md=6,
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [create_level_confluence_chart(levels_data)],
                        width=12,
                    )
                ]
            ),
        ]
    )


def render_level_strength_indicator(strength: float, is_support: bool) -> html.Div:
    """
    Render an indicator for the strength of a support/resistance level.

    Args:
        strength: Strength value (0.0 - 1.0)
        is_support: Whether this is a support (True) or resistance (False) level

    Returns:
        HTML Div with strength indicator
    """
    # Determine color based on strength
    hue = 120 * strength  # 0 = red, 120 = green
    rgb = colorsys.hsv_to_rgb(hue / 360, 0.8, 0.8)
    color = f"rgba({int(rgb[0] * 255)}, {int(rgb[1] * 255)}, {int(rgb[2] * 255)}, 0.8)"

    # Determine label
    if strength < 0.33:
        label = "Weak"
    elif strength < 0.66:
        label = "Moderate"
    else:
        label = "Strong"

    level_type = "Support" if is_support else "Resistance"

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        style={
                            "width": f"{strength * 100}%",
                            "background-color": color,
                            "height": "8px",
                            "border-radius": "4px",
                        }
                    )
                ],
                className="strength-bar",
            ),
            html.Div(
                f"{label} {level_type}",
                className="small text-end",
            ),
        ],
        className="level-strength-indicator",
    )


def create_level_confluence_chart(levels_data: Dict[str, Any]) -> dcc.Graph:
    """
    Create a chart showing support and resistance level confluence.

    Args:
        levels_data: Dictionary containing support and resistance levels information

    Returns:
        Dash Graph component with confluence visualization
    """
    # Create the figure
    fig = go.Figure()

    # Add price axis
    if "current_price" in levels_data:
        current_price = levels_data["current_price"]
    else:
        current_price = 0
        if levels_data.get("support_levels"):
            current_price = max(
                [s.get("price", 0) for s in levels_data["support_levels"]]
            )
        elif levels_data.get("resistance_levels"):
            current_price = min(
                [r.get("price", 0) for r in levels_data["resistance_levels"]]
            )

    # Set price range to +/- 10% of current price
    price_range = current_price * 0.1
    y_min = current_price - price_range
    y_max = current_price + price_range

    # Add horizontal line for current price
    fig.add_shape(
        type="line",
        x0=0,
        x1=1,
        y0=current_price,
        y1=current_price,
        line=dict(color="rgba(0, 0, 0, 0.5)", width=1, dash="dot"),
    )
    fig.add_annotation(
        x=1.01,
        y=current_price,
        text=f"${current_price:.2f}",
        showarrow=False,
        xref="paper",
        yref="y",
        font=dict(size=10),
    )

    # Add support and resistance levels
    if "support_levels" in levels_data:
        for level in levels_data["support_levels"]:
            price = level.get("price", 0)
            strength = level.get("strength", 0.5)

            # Skip levels outside our range
            if price < y_min or price > y_max:
                continue

            # Color based on strength
            hue = 120 * strength  # 0 = red, 120 = green
            rgb = colorsys.hsv_to_rgb(hue / 360, 0.8, 0.8)
            color = f"rgba({int(rgb[0] * 255)}, {int(rgb[1] * 255)}, {int(rgb[2] * 255)}, 0.8)"

            # Line width based on strength
            width = 1 + strength * 3

            # Add line
            fig.add_shape(
                type="line",
                x0=0,
                x1=1,
                y0=price,
                y1=price,
                line=dict(color=color, width=width),
            )

            # Add label
            fig.add_annotation(
                x=0,
                y=price,
                text=f"S: ${price:.2f}",
                showarrow=False,
                xanchor="left",
                xref="paper",
                yref="y",
                font=dict(size=10, color="green"),
            )

    if "resistance_levels" in levels_data:
        for level in levels_data["resistance_levels"]:
            price = level.get("price", 0)
            strength = level.get("strength", 0.5)

            # Skip levels outside our range
            if price < y_min or price > y_max:
                continue

            # Color based on strength
            hue = 120 * strength  # 0 = red, 120 = green
            rgb = colorsys.hsv_to_rgb(hue / 360, 0.8, 0.8)
            color = f"rgba({int(rgb[0] * 255)}, {int(rgb[1] * 255)}, {int(rgb[2] * 255)}, 0.8)"

            # Line width based on strength
            width = 1 + strength * 3

            # Add line
            fig.add_shape(
                type="line",
                x0=0,
                x1=1,
                y0=price,
                y1=price,
                line=dict(color=color, width=width, dash="solid"),
            )

            # Add label
            fig.add_annotation(
                x=1,
                y=price,
                text=f"R: ${price:.2f}",
                showarrow=False,
                xanchor="right",
                xref="paper",
                yref="y",
                font=dict(size=10, color="red"),
            )

    # Update layout
    fig.update_layout(
        title="Price Level Confluence",
        height=250,
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=False,
        xaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            title="Price",
            showgrid=True,
            gridcolor="rgba(0, 0, 0, 0.1)",
            range=[y_min, y_max],
        ),
        plot_bgcolor="white",
    )

    return dcc.Graph(figure=fig, config={"displayModeBar": False})


def render_execution_recommendations(
    recommendations: Dict[str, Any], orderbook: Dict[str, Any] = None
) -> html.Div:
    """
    Render execution recommendations based on orderbook analysis.

    Args:
        recommendations: Dictionary containing execution recommendations
        orderbook: Optional orderbook data for additional context

    Returns:
        HTML Div with execution recommendations
    """
    if not recommendations or not recommendations.get("actions"):
        return html.Div(
            [
                html.P(
                    "No execution recommendations available",
                    className="text-center text-muted",
                )
            ]
        )

    actions = recommendations.get("actions", [])

    # Create recommendation cards
    cards = []
    for action in actions:
        action_type = action.get("type", "").capitalize()
        description = action.get("description", "")
        confidence = action.get("confidence", 0.5)
        price = action.get("price")
        size = action.get("size")

        # Determine color based on action type
        if action_type.lower() == "buy":
            color = "success"
            icon = "arrow-up"
        elif action_type.lower() == "sell":
            color = "danger"
            icon = "arrow-down"
        else:
            color = "info"
            icon = "info-circle"

        # Determine confidence label
        if confidence < 0.33:
            conf_label = "Low Confidence"
            conf_color = "danger"
        elif confidence < 0.66:
            conf_label = "Medium Confidence"
            conf_color = "warning"
        else:
            conf_label = "High Confidence"
            conf_color = "success"

        # Create recommendation card
        card = dbc.Card(
            [
                dbc.CardHeader(
                    [
                        html.Div(
                            [
                                html.I(
                                    className=f"fas fa-{icon} me-2",
                                ),
                                html.Span(action_type, className="fw-bold"),
                            ],
                            className=f"text-{color} d-flex align-items-center",
                        )
                    ]
                ),
                dbc.CardBody(
                    [
                        html.P(description, className="recommendation-description"),
                        html.Div(
                            [
                                html.Span(
                                    f"Confidence: {confidence:.2f}",
                                    className="me-2 small",
                                ),
                                dbc.Badge(
                                    conf_label,
                                    color=conf_color,
                                    className="me-1",
                                ),
                            ],
                            className="mb-2",
                        ),
                        html.Div(
                            [
                                html.Strong(
                                    "Suggested Price: ", className="me-1 small"
                                ),
                                html.Span(
                                    f"${price:.2f}" if price else "Market",
                                    className="me-3 small",
                                ),
                                html.Strong("Size: ", className="me-1 small"),
                                html.Span(
                                    f"{size}" if size else "Default",
                                    className="small",
                                ),
                            ],
                            className="recommendation-details",
                        ),
                    ]
                ),
            ],
            className="mb-2 recommendation-card",
        )
        cards.append(card)

    return html.Div(
        [
            html.Div(
                [
                    html.P(
                        "Trading recommendations based on current market conditions",
                        className="text-muted small mb-3",
                    ),
                    *cards,
                    html.Div(
                        [
                            html.P(
                                "These recommendations are automated suggestions only and should be used as part of a broader trading strategy.",
                                className="text-muted fst-italic small",
                            )
                        ],
                        className="mt-3",
                    ),
                ]
            )
        ]
    )
