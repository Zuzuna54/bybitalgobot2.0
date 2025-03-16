"""
Orderbook Charts Module

This module provides chart generation functions for orderbook data visualization.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from loguru import logger
import warnings

from src.dashboard.services.chart_service.base import (
    apply_chart_theme,
    create_empty_chart,
)


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
        return create_empty_chart("Order Book Heatmap")

    # Extract orderbook data
    bids = orderbook.get("bids", [])
    asks = orderbook.get("asks", [])

    if not bids or not asks:
        return create_empty_chart("Insufficient orderbook data")

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

    # Apply standard theme
    fig = apply_chart_theme(fig, title)

    # Update layout
    fig.update_layout(
        yaxis_title="Price",
        height=height,
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


# Alias for backward compatibility
def create_orderbook_depth_graph(*args, **kwargs):
    """
    DEPRECATED: Use create_orderbook_depth_chart instead.

    This function is kept for backward compatibility and calls
    create_orderbook_depth_chart.
    """
    warnings.warn(
        "create_orderbook_depth_graph is deprecated. "
        "Use create_orderbook_depth_chart instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return create_orderbook_depth_chart(*args, **kwargs)
