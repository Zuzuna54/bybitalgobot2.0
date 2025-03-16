"""
Orderbook Visualization Components for the Trading Dashboard

This module provides visualization components for displaying orderbook data,
including imbalance indicators, liquidity ratios, and depth charts.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import colorsys
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import html, dcc
import dash_bootstrap_components as dbc


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
                            "backgroundColor": color,
                            "height": "20px",
                            "borderRadius": "4px",
                        },
                    )
                ],
                style={
                    "backgroundColor": "#f8f9fa",
                    "borderRadius": "4px",
                    "padding": "2px",
                    "marginTop": "10px",
                },
            ),
        ]
    )


def render_liquidity_ratio(buy_sell_ratio: float) -> html.Div:
    """
    Render the liquidity ratio indicator.

    Args:
        buy_sell_ratio: Ratio of buy to sell liquidity

    Returns:
        HTML Div with liquidity ratio visualization
    """
    if buy_sell_ratio is None:
        return html.Div("No data available")

    # Calculate descriptive label
    if buy_sell_ratio > 1.5:
        label = "Strong Buy Liquidity"
        color = "rgba(50, 171, 96, 0.8)"
    elif buy_sell_ratio > 1.1:
        label = "Moderate Buy Liquidity"
        color = "rgba(50, 171, 96, 0.6)"
    elif buy_sell_ratio > 0.9:
        label = "Balanced Liquidity"
        color = "rgba(70, 130, 180, 0.6)"
    elif buy_sell_ratio > 0.5:
        label = "Moderate Sell Liquidity"
        color = "rgba(220, 53, 69, 0.6)"
    else:
        label = "Strong Sell Liquidity"
        color = "rgba(220, 53, 69, 0.8)"

    # Create indicator components
    return html.Div(
        [
            html.H4(f"{buy_sell_ratio:.2f}", className="text-center"),
            html.P(label, className="text-center"),
            html.Div(
                [
                    html.Div(
                        style={
                            "display": "flex",
                            "height": "20px",
                            "borderRadius": "4px",
                            "overflow": "hidden",
                        },
                        children=[
                            html.Div(
                                style={
                                    "flex": str(buy_sell_ratio),
                                    "backgroundColor": "rgba(50, 171, 96, 0.8)",
                                    "height": "100%",
                                }
                            ),
                            html.Div(
                                style={
                                    "flex": "1",
                                    "backgroundColor": "rgba(220, 53, 69, 0.8)",
                                    "height": "100%",
                                }
                            ),
                        ],
                    )
                ],
                style={
                    "backgroundColor": "#f8f9fa",
                    "borderRadius": "4px",
                    "padding": "2px",
                    "marginTop": "10px",
                },
            ),
        ]
    )


def create_orderbook_depth_graph(
    orderbook: Dict[str, Any],
    support_levels: List[float] = None,
    resistance_levels: List[float] = None,
    sr_levels: Optional[Dict[str, Any]] = None,
) -> go.Figure:
    """
    Create an order book depth visualization.

    Args:
        orderbook: Order book data from the API
        support_levels: List of support level prices
        resistance_levels: List of resistance level prices
        sr_levels: Support and resistance levels data

    Returns:
        Plotly figure with orderbook depth visualization
    """
    if not orderbook or "bids" not in orderbook or "asks" not in orderbook:
        # Create empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No order book data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14),
        )
        return fig

    # Extract bids and asks
    bids = orderbook.get("bids", [])
    asks = orderbook.get("asks", [])

    if not bids or not asks:
        # Create empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="Order book is empty",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14),
        )
        return fig

    # Convert to numeric and create DataFrames
    bid_prices = [float(bid[0]) for bid in bids]
    bid_sizes = [float(bid[1]) for bid in bids]
    bid_cumulative = np.cumsum(bid_sizes)

    ask_prices = [float(ask[0]) for ask in asks]
    ask_sizes = [float(ask[1]) for ask in asks]
    ask_cumulative = np.cumsum(ask_sizes)

    # Calculate spread
    spread = min(ask_prices) - max(bid_prices)
    spread_pct = spread / ((min(ask_prices) + max(bid_prices)) / 2) * 100

    # Create figure with secondary y-axis for cumulative sizes
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add individual bid sizes as green bars
    fig.add_trace(
        go.Bar(
            x=bid_prices,
            y=bid_sizes,
            name="Bid Size",
            marker_color="rgba(50, 171, 96, 0.6)",
            marker_line_color="rgba(50, 171, 96, 1.0)",
            marker_line_width=1,
            opacity=0.7,
        )
    )

    # Add individual ask sizes as red bars
    fig.add_trace(
        go.Bar(
            x=ask_prices,
            y=ask_sizes,
            name="Ask Size",
            marker_color="rgba(220, 53, 69, 0.6)",
            marker_line_color="rgba(220, 53, 69, 1.0)",
            marker_line_width=1,
            opacity=0.7,
        )
    )

    # Add cumulative bid sizes
    fig.add_trace(
        go.Scatter(
            x=bid_prices,
            y=bid_cumulative,
            name="Cumulative Bids",
            mode="lines",
            line=dict(color="rgba(50, 171, 96, 1.0)", width=2, dash="solid"),
            fill="tozeroy",
            fillcolor="rgba(50, 171, 96, 0.1)",
        ),
        secondary_y=True,
    )

    # Add cumulative ask sizes
    fig.add_trace(
        go.Scatter(
            x=ask_prices,
            y=ask_cumulative,
            name="Cumulative Asks",
            mode="lines",
            line=dict(color="rgba(220, 53, 69, 1.0)", width=2, dash="solid"),
            fill="tozeroy",
            fillcolor="rgba(220, 53, 69, 0.1)",
        ),
        secondary_y=True,
    )

    # Use provided support/resistance levels if available, otherwise try to get from sr_levels
    if support_levels is None and sr_levels:
        support_levels = sr_levels.get("support_levels", [])

    if resistance_levels is None and sr_levels:
        resistance_levels = sr_levels.get("resistance_levels", [])

    # Add support levels as vertical lines
    if support_levels:
        for i, level in enumerate(support_levels[:3]):  # Show top 3 levels
            opacity = 0.9 - (i * 0.2)  # Decrease opacity for less significant levels
            fig.add_shape(
                type="line",
                x0=level,
                y0=0,
                x1=level,
                y1=1,
                yref="paper",
                line=dict(color=f"rgba(0, 128, 0, {opacity})", width=2, dash="dash"),
            )

            # Add annotation
            fig.add_annotation(
                x=level,
                y=0.1,
                yref="paper",
                text=f"S{i+1}",
                showarrow=False,
                font=dict(color=f"rgba(0, 128, 0, {opacity + 0.1})", size=10),
                bgcolor="rgba(255, 255, 255, 0.7)",
                bordercolor=f"rgba(0, 128, 0, {opacity})",
                borderwidth=1,
                borderpad=2,
                opacity=opacity + 0.1,
            )

    # Add resistance levels as vertical lines
    if resistance_levels:
        for i, level in enumerate(resistance_levels[:3]):  # Show top 3 levels
            opacity = 0.9 - (i * 0.2)  # Decrease opacity for less significant levels
            fig.add_shape(
                type="line",
                x0=level,
                y0=0,
                x1=level,
                y1=1,
                yref="paper",
                line=dict(color=f"rgba(220, 53, 69, {opacity})", width=2, dash="dash"),
            )

            # Add annotation
            fig.add_annotation(
                x=level,
                y=0.9,
                yref="paper",
                text=f"R{i+1}",
                showarrow=False,
                font=dict(color=f"rgba(220, 53, 69, {opacity + 0.1})", size=10),
                bgcolor="rgba(255, 255, 255, 0.7)",
                bordercolor=f"rgba(220, 53, 69, {opacity})",
                borderwidth=1,
                borderpad=2,
                opacity=opacity + 0.1,
            )

    # Add VWAP levels if available
    if sr_levels and "vwap_levels" in sr_levels:
        vwap_levels = sr_levels.get("vwap_levels", {})
        for key, value in vwap_levels.items():
            # Use different colors for different VWAP periods
            if "20" in key:
                color = "rgba(70, 130, 180, 0.8)"  # SteelBlue
            elif "50" in key:
                color = "rgba(75, 0, 130, 0.8)"  # Indigo
            else:
                color = "rgba(128, 0, 128, 0.8)"  # Purple

            fig.add_shape(
                type="line",
                x0=value,
                y0=0,
                x1=value,
                y1=1,
                yref="paper",
                line=dict(color=color, width=1.5, dash="dot"),
            )

            # Add annotation
            fig.add_annotation(
                x=value,
                y=0.5,
                yref="paper",
                text=key.upper(),
                showarrow=False,
                font=dict(color=color, size=8),
                bgcolor="rgba(255, 255, 255, 0.7)",
                bordercolor=color,
                borderwidth=1,
                borderpad=2,
                opacity=0.8,
            )

    # Add volume clusters visualization
    if sr_levels and "volume_clusters" in sr_levels:
        volume_clusters = sr_levels.get("volume_clusters", {})
        support_clusters = volume_clusters.get("support", [])
        resistance_clusters = volume_clusters.get("resistance", [])

        # Visualize volume clusters as semi-transparent rectangles
        for i, level in enumerate(support_clusters[:5]):  # Top 5 support clusters
            # Skip if already visualized as a main support level
            if support_levels and level in support_levels[:3]:
                continue

            height = max(bid_sizes) * 0.15  # Make height relative to max bid size
            y_pos = i * height * 0.3  # Stagger vertically to avoid overlap

            # Get a slightly different green hue for each cluster
            hue = 0.33 + (i * 0.02)  # Green with slight variations
            r, g, b = colorsys.hsv_to_rgb(hue, 0.6, 0.8)
            color = f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, 0.4)"

            fig.add_shape(
                type="rect",
                x0=level - (spread * 0.3),
                y0=y_pos,
                x1=level + (spread * 0.3),
                y1=y_pos + height,
                line=dict(
                    color=color,
                    width=1,
                ),
                fillcolor=color,
                opacity=0.3,
            )

        for i, level in enumerate(resistance_clusters[:5]):  # Top 5 resistance clusters
            # Skip if already visualized as a main resistance level
            if resistance_levels and level in resistance_levels[:3]:
                continue

            height = max(ask_sizes) * 0.15  # Make height relative to max ask size
            y_pos = max(ask_sizes) - (i * height * 0.3)  # Stagger vertically from top

            # Get a slightly different red hue for each cluster
            hue = 0.98 - (i * 0.02)  # Red with slight variations
            r, g, b = colorsys.hsv_to_rgb(hue, 0.6, 0.8)
            color = f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, 0.4)"

            fig.add_shape(
                type="rect",
                x0=level - (spread * 0.3),
                y0=y_pos - height,
                x1=level + (spread * 0.3),
                y1=y_pos,
                line=dict(
                    color=color,
                    width=1,
                ),
                fillcolor=color,
                opacity=0.3,
            )

    # Update layout
    fig.update_layout(
        title=f"Order Book Depth - Spread: {spread:.8f} ({spread_pct:.2f}%)",
        xaxis_title="Price",
        barmode="overlay",
        bargap=0,
        bargroupgap=0,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        margin=dict(l=10, r=10, t=50, b=30),
    )

    # Update axes
    fig.update_yaxes(title_text="Size", secondary_y=False, showgrid=True)
    fig.update_yaxes(title_text="Cumulative Size", secondary_y=True, showgrid=False)
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0, 0, 0, 0.1)")

    return fig


def render_support_resistance_levels(levels_data: Dict[str, Any]) -> html.Div:
    """
    Render support and resistance levels with enhanced visualization.

    Args:
        levels_data: Dictionary with support/resistance levels and metadata

    Returns:
        HTML Div with support/resistance visualization
    """
    if not levels_data:
        return html.Div("No support/resistance data available")

    support_levels = levels_data.get("support_levels", [])
    resistance_levels = levels_data.get("resistance_levels", [])
    vwap_levels = levels_data.get("vwap_levels", {})

    if not support_levels and not resistance_levels:
        return html.Div("No significant levels detected")

    # Create visualization
    return html.Div(
        [
            # Header
            html.Div(
                [
                    html.H5("Support & Resistance Levels", className="levels-header"),
                    html.Hr(style={"margin": "0.5rem 0"}),
                ]
            ),
            # Support levels
            html.Div(
                [
                    html.H6("Support Levels", className="level-subheader"),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        f"S{i+1}: {level:.4f}",
                                        className="support-level-badge",
                                    ),
                                    render_level_strength_indicator(
                                        strength=1.0
                                        - (i * 0.2),  # Decrease strength for each level
                                        is_support=True,
                                    ),
                                ],
                                className="level-item",
                            )
                            for i, level in enumerate(
                                support_levels[:5]
                            )  # Display top 5 levels
                        ],
                        className="levels-container",
                    ),
                ],
                className="support-levels-section",
            ),
            # Resistance levels
            html.Div(
                [
                    html.H6("Resistance Levels", className="level-subheader"),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        f"R{i+1}: {level:.4f}",
                                        className="resistance-level-badge",
                                    ),
                                    render_level_strength_indicator(
                                        strength=1.0
                                        - (i * 0.2),  # Decrease strength for each level
                                        is_support=False,
                                    ),
                                ],
                                className="level-item",
                            )
                            for i, level in enumerate(
                                resistance_levels[:5]
                            )  # Display top 5 levels
                        ],
                        className="levels-container",
                    ),
                ],
                className="resistance-levels-section",
            ),
            # VWAP levels if available
            (
                html.Div(
                    [
                        html.H6("VWAP Levels", className="level-subheader"),
                        (
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div(
                                                f"{key.upper()}: {value:.4f}",
                                                className="vwap-level-badge",
                                                style={
                                                    "borderLeftColor": get_vwap_color(
                                                        key
                                                    )
                                                },
                                            )
                                        ],
                                        className="level-item",
                                    )
                                    for key, value in vwap_levels.items()
                                ],
                                className="levels-container",
                            )
                            if vwap_levels
                            else html.Div(
                                "No VWAP data available", className="no-data-message"
                            )
                        ),
                    ],
                    className="vwap-levels-section",
                )
                if vwap_levels
                else html.Div()
            ),
            # Price action confluence
            html.Div(
                [
                    html.Hr(style={"margin": "0.5rem 0"}),
                    html.H6("Signal Strength", className="level-subheader"),
                    create_level_confluence_chart(levels_data),
                ],
                className="confluence-section",
            ),
        ],
        className="support-resistance-container",
    )


def render_level_strength_indicator(strength: float, is_support: bool) -> html.Div:
    """
    Render a visual indicator of level strength.

    Args:
        strength: Level strength (0.0-1.0)
        is_support: Whether this is a support level (True) or resistance level (False)

    Returns:
        HTML Div with strength indicator
    """
    # Determine color
    if is_support:
        color = f"rgba(0, 128, 0, {min(strength + 0.3, 1.0)})"
    else:
        color = f"rgba(220, 53, 69, {min(strength + 0.3, 1.0)})"

    # Calculate width percentage
    width_pct = strength * 100

    return html.Div(
        [
            html.Div(
                className="strength-bar",
                style={
                    "width": f"{width_pct}%",
                    "backgroundColor": color,
                    "height": "6px",
                    "borderRadius": "3px",
                },
            )
        ],
        style={
            "backgroundColor": "#f8f9fa",
            "borderRadius": "3px",
            "padding": "2px",
            "marginTop": "4px",
            "width": "100%",
        },
    )


def create_level_confluence_chart(levels_data: Dict[str, Any]) -> dcc.Graph:
    """
    Create a chart showing the confluence of different sources of support/resistance.

    Args:
        levels_data: Dictionary with support/resistance levels and metadata

    Returns:
        Dash Graph component with the confluence chart
    """
    # Extract necessary data
    volume_clusters = levels_data.get("volume_clusters", {})
    support_clusters = volume_clusters.get("support", [])
    resistance_clusters = volume_clusters.get("resistance", [])

    price_action = levels_data.get("price_action", {})
    price_action_support = price_action.get("support", [])
    price_action_resistance = price_action.get("resistance", [])

    vwap_levels = levels_data.get("vwap_levels", {})

    # Combine all levels
    all_levels = []

    # Add volume-based levels
    for level in support_clusters[:3]:
        all_levels.append(
            {"price": level, "source": "Volume", "type": "Support", "strength": 0.8}
        )

    for level in resistance_clusters[:3]:
        all_levels.append(
            {"price": level, "source": "Volume", "type": "Resistance", "strength": 0.8}
        )

    # Add price action levels
    for level in price_action_support[:3]:
        all_levels.append(
            {
                "price": level,
                "source": "Price Action",
                "type": "Support",
                "strength": 0.7,
            }
        )

    for level in price_action_resistance[:3]:
        all_levels.append(
            {
                "price": level,
                "source": "Price Action",
                "type": "Resistance",
                "strength": 0.7,
            }
        )

    # Add VWAP levels
    for key, level in vwap_levels.items():
        all_levels.append(
            {"price": level, "source": "VWAP", "type": key.upper(), "strength": 0.6}
        )

    # If no levels available, return placeholder
    if not all_levels:
        return html.Div("No confluence data available")

    # Create figure
    fig = go.Figure()

    # Add levels as scatter points
    for source in ["Volume", "Price Action", "VWAP"]:
        source_levels = [level for level in all_levels if level["source"] == source]

        if not source_levels:
            continue

        x_values = [level["price"] for level in source_levels]
        y_values = [level["source"] for level in source_levels]

        # Create colors based on type
        colors = []
        sizes = []
        hover_texts = []

        for level in source_levels:
            level_type = level["type"]
            strength = level["strength"]

            if "Support" in level_type:
                colors.append("rgba(0, 128, 0, 0.8)")
            elif "Resistance" in level_type:
                colors.append("rgba(220, 53, 69, 0.8)")
            else:
                colors.append("rgba(70, 130, 180, 0.8)")

            sizes.append(10 + (strength * 10))
            hover_texts.append(
                f"{level_type}: {level['price']:.4f}<br>Strength: {strength:.1f}"
            )

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="markers",
                marker=dict(
                    color=colors,
                    size=sizes,
                    line=dict(width=1, color="rgba(255, 255, 255, 0.8)"),
                ),
                name=source,
                text=hover_texts,
                hoverinfo="text",
            )
        )

    # Update layout
    fig.update_layout(
        height=150,
        margin=dict(l=10, r=10, t=10, b=30),
        xaxis=dict(
            title="Price", showgrid=True, gridwidth=1, gridcolor="rgba(0, 0, 0, 0.1)"
        ),
        yaxis=dict(title="Source", showgrid=False),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return dcc.Graph(figure=fig, config={"displayModeBar": False})


def get_vwap_color(vwap_key: str) -> str:
    """
    Get the color for a VWAP period.

    Args:
        vwap_key: VWAP period key (e.g., 'vwap_20')

    Returns:
        Color string
    """
    if "20" in vwap_key:
        return "#4682B4"  # SteelBlue
    elif "50" in vwap_key:
        return "#4B0082"  # Indigo
    else:
        return "#800080"  # Purple


def render_execution_recommendations(
    recommendations: Dict[str, Any], risk_tolerance: float = 0.5
) -> html.Div:
    """
    Render execution recommendations based on orderbook analysis.

    Args:
        recommendations: Dictionary containing execution recommendations
        risk_tolerance: Risk tolerance factor (0.0-1.0), default is 0.5 (moderate)

    Returns:
        Dash HTML component with execution recommendations
    """
    if not recommendations:
        return html.Div("No recommendations available")

    # Extract recommendation data
    is_buy = recommendations.get("is_buy", True)
    recommendation = recommendations.get("recommendation", {})
    market_conditions = recommendations.get("market_conditions", {})
    confidence_score = recommendations.get("confidence_score", 0.5)

    # Get order type and other parameters
    order_type = recommendation.get("order_type", "limit")
    limit_price = recommendation.get("limit_price")
    should_split = recommendation.get("should_split", False)
    num_parts = recommendation.get("num_parts", 1)
    expected_impact = recommendation.get("expected_impact")
    time_interval = recommendation.get("time_interval")

    # Confidence indicator color
    if confidence_score >= 0.7:
        confidence_color = "rgba(50, 171, 96, 0.8)"  # Green
        confidence_label = "High"
    elif confidence_score >= 0.4:
        confidence_color = "rgba(255, 193, 7, 0.8)"  # Yellow
        confidence_label = "Medium"
    else:
        confidence_color = "rgba(220, 53, 69, 0.8)"  # Red
        confidence_label = "Low"

    # Build the recommendation card
    return html.Div(
        [
            html.H5(
                f"{'Buy' if is_buy else 'Sell'} Recommendation",
                className="recommendation-header",
                style={
                    "color": (
                        "rgba(50, 171, 96, 1.0)" if is_buy else "rgba(220, 53, 69, 1.0)"
                    )
                },
            ),
            # Risk tolerance indicator
            html.Div(
                [
                    html.Label("Risk Tolerance:", className="recommendation-label"),
                    html.Div(
                        [
                            html.Span(
                                f"{risk_tolerance:.2f} - {'High' if risk_tolerance > 0.7 else 'Medium' if risk_tolerance > 0.3 else 'Low'}",
                                style={
                                    "marginRight": "10px",
                                    "fontWeight": "bold",
                                    "color": (
                                        "rgba(220, 53, 69, 0.8)"
                                        if risk_tolerance > 0.7
                                        else (
                                            "rgba(255, 193, 7, 0.8)"
                                            if risk_tolerance > 0.3
                                            else "rgba(50, 171, 96, 0.8)"
                                        )
                                    ),
                                },
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        style={
                                            "width": f"{risk_tolerance * 100}%",
                                            "backgroundColor": (
                                                "rgba(220, 53, 69, 0.8)"
                                                if risk_tolerance > 0.7
                                                else (
                                                    "rgba(255, 193, 7, 0.8)"
                                                    if risk_tolerance > 0.3
                                                    else "rgba(50, 171, 96, 0.8)"
                                                )
                                            ),
                                            "height": "8px",
                                            "borderRadius": "4px",
                                        }
                                    )
                                ],
                                style={
                                    "backgroundColor": "#f8f9fa",
                                    "borderRadius": "4px",
                                    "padding": "2px",
                                    "width": "100%",
                                },
                            ),
                        ]
                    ),
                ],
                className="risk-tolerance-indicator",
            ),
            # Confidence indicator
            html.Div(
                [
                    html.Label("Confidence Level:", className="recommendation-label"),
                    html.Div(
                        [
                            html.Span(
                                f"{confidence_score:.2f} - {confidence_label}",
                                style={
                                    "marginRight": "10px",
                                    "fontWeight": "bold",
                                    "color": confidence_color,
                                },
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        style={
                                            "width": f"{confidence_score * 100}%",
                                            "backgroundColor": confidence_color,
                                            "height": "8px",
                                            "borderRadius": "4px",
                                        }
                                    )
                                ],
                                style={
                                    "backgroundColor": "#f8f9fa",
                                    "borderRadius": "4px",
                                    "padding": "2px",
                                    "width": "100%",
                                },
                            ),
                        ]
                    ),
                ],
                className="confidence-indicator",
            ),
            html.Hr(style={"margin": "0.5rem 0"}),
            # Order type recommendation
            html.Div(
                [
                    html.Label(
                        "Recommended Order Type:", className="recommendation-label"
                    ),
                    html.Div(
                        order_type.upper(),
                        className=f"order-type-badge {'market-order' if order_type == 'market' else 'limit-order'}",
                    ),
                ],
                className="recommendation-item",
            ),
            # Limit price (if applicable)
            (
                html.Div(
                    [
                        html.Label(
                            "Recommended Limit Price:", className="recommendation-label"
                        ),
                        html.Div(
                            f"{limit_price:.8f}" if limit_price else "N/A",
                            className="limit-price-value",
                        ),
                    ],
                    className="recommendation-item",
                )
                if order_type == "limit"
                else html.Div()
            ),
            # Order splitting recommendation
            html.Div(
                [
                    html.Label(
                        "Order Execution Strategy:", className="recommendation-label"
                    ),
                    html.Div(
                        [
                            html.Div(
                                (
                                    f"Split into {num_parts} parts over {time_interval} seconds"
                                    if should_split
                                    else "Execute as single order"
                                ),
                                className="execution-strategy-value",
                            ),
                            (
                                html.Div(
                                    (
                                        "TWAP/VWAP execution recommended"
                                        if should_split
                                        else ""
                                    ),
                                    className="execution-substrategy-value",
                                )
                                if should_split
                                else html.Div()
                            ),
                        ]
                    ),
                ],
                className="recommendation-item",
            ),
            # Market impact (if available)
            (
                html.Div(
                    [
                        html.Label(
                            "Estimated Market Impact:", className="recommendation-label"
                        ),
                        html.Div(
                            f"{expected_impact:.2f}%" if expected_impact else "Unknown",
                            className="impact-value",
                            style={
                                "color": (
                                    "rgba(220, 53, 69, 0.8)"
                                    if expected_impact and expected_impact > 0.5
                                    else "inherit"
                                )
                            },
                        ),
                    ],
                    className="recommendation-item",
                )
                if expected_impact is not None
                else html.Div()
            ),
            html.Hr(style={"margin": "0.5rem 0"}),
            # Market conditions summary
            html.Div(
                [
                    html.Label("Market Conditions:", className="recommendation-label"),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span(
                                        "Imbalance: ", className="condition-label"
                                    ),
                                    html.Span(
                                        f"{market_conditions.get('imbalance', 0):.2f}",
                                        style={
                                            "color": (
                                                "rgba(50, 171, 96, 0.8)"
                                                if market_conditions.get("imbalance", 0)
                                                > 0
                                                else "rgba(220, 53, 69, 0.8)"
                                            )
                                        },
                                    ),
                                ],
                                className="condition-item",
                            ),
                            html.Div(
                                [
                                    html.Span("Spread: ", className="condition-label"),
                                    html.Span(
                                        f"{market_conditions.get('spread_pct', 0):.2f}%"
                                    ),
                                ],
                                className="condition-item",
                            ),
                            (
                                html.Div(
                                    [
                                        html.Span(
                                            "Position in Range: ",
                                            className="condition-label",
                                        ),
                                        html.Span(
                                            f"{market_conditions.get('range_position', 0.5):.2f}",
                                            style={
                                                "color": (
                                                    "rgba(50, 171, 96, 0.8)"
                                                    if market_conditions.get(
                                                        "range_position", 0.5
                                                    )
                                                    < 0.3
                                                    else (
                                                        "rgba(220, 53, 69, 0.8)"
                                                        if market_conditions.get(
                                                            "range_position", 0.5
                                                        )
                                                        > 0.7
                                                        else "rgba(70, 130, 180, 0.8)"
                                                    )
                                                )
                                            },
                                        ),
                                    ],
                                    className="condition-item",
                                )
                                if "range_position" in market_conditions
                                else html.Div()
                            ),
                        ]
                    ),
                ],
                className="market-conditions",
            ),
        ],
        className="execution-recommendations-container",
    )
