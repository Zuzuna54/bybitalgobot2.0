"""
Component Renderers Module

This module provides functions for rendering UI components and indicators
for the dashboard.
"""

from typing import Dict, Any, List, Optional
import dash_bootstrap_components as dbc
from dash import html, dcc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from src.dashboard.services.chart_service.base import (
    apply_chart_theme,
    create_empty_chart,
)


def render_imbalance_indicator(imbalance: float) -> html.Div:
    """
    Render an order book imbalance indicator.

    Args:
        imbalance: Order book imbalance value (-1.0 to 1.0)

    Returns:
        Dash HTML component
    """
    # Determine color based on imbalance
    if imbalance > 0.3:
        color = "success"  # Green for buy pressure
        icon = "bi bi-arrow-up-circle-fill"
        strength = "Strong Buy"
    elif imbalance > 0.1:
        color = "success"  # Light green for moderate buy
        icon = "bi bi-arrow-up-circle"
        strength = "Moderate Buy"
    elif imbalance < -0.3:
        color = "danger"  # Red for sell pressure
        icon = "bi bi-arrow-down-circle-fill"
        strength = "Strong Sell"
    elif imbalance < -0.1:
        color = "danger"  # Light red for moderate sell
        icon = "bi bi-arrow-down-circle"
        strength = "Moderate Sell"
    else:
        color = "secondary"  # Gray for neutral
        icon = "bi bi-dash-circle"
        strength = "Neutral"

    # Calculate percentage for display
    percentage = abs(round(imbalance * 100))

    return html.Div(
        [
            html.H6("Order Book Imbalance", className="card-subtitle mb-2 text-muted"),
            html.Div(
                [
                    html.I(className=f"{icon} me-2"),
                    html.Span(f"{strength} ({percentage}%)"),
                ],
                className=f"text-{color} d-flex align-items-center",
            ),
        ],
        className="p-3 border rounded",
    )


def render_liquidity_ratio(liquidity_data: Dict[str, Any]) -> html.Div:
    """
    Render a liquidity ratio indicator.

    Args:
        liquidity_data: Dictionary with liquidity metrics

    Returns:
        Dash HTML component
    """
    # Extract metrics
    buy_sell_ratio = liquidity_data.get("buy_sell_ratio", 1.0)
    buy_depth = liquidity_data.get("buy_depth", 0)
    sell_depth = liquidity_data.get("sell_depth", 0)

    # Determine color based on ratio
    if buy_sell_ratio > 1.5:
        color = "success"  # Green for high buy liquidity
        icon = "bi bi-water"
        strength = "High Buy Liquidity"
    elif buy_sell_ratio > 1.1:
        color = "info"  # Blue for moderate buy liquidity
        icon = "bi bi-droplet-half"
        strength = "Moderate Buy Liquidity"
    elif buy_sell_ratio < 0.67:  # 1/1.5
        color = "danger"  # Red for high sell liquidity
        icon = "bi bi-water"
        strength = "High Sell Liquidity"
    elif buy_sell_ratio < 0.91:  # 1/1.1
        color = "warning"  # Orange for moderate sell liquidity
        icon = "bi bi-droplet-half"
        strength = "Moderate Sell Liquidity"
    else:
        color = "secondary"  # Gray for balanced
        icon = "bi bi-droplet"
        strength = "Balanced Liquidity"

    # Format values for display
    ratio_display = f"{buy_sell_ratio:.2f}"
    buy_depth_display = f"{buy_depth:,.2f}"
    sell_depth_display = f"{sell_depth:,.2f}"

    return html.Div(
        [
            html.H6("Liquidity Ratio", className="card-subtitle mb-2 text-muted"),
            html.Div(
                [
                    html.I(className=f"{icon} me-2"),
                    html.Span(f"{strength} ({ratio_display})"),
                ],
                className=f"text-{color} d-flex align-items-center mb-2",
            ),
            html.Small(
                [
                    html.Span("Buy Depth: ", className="text-muted"),
                    html.Span(buy_depth_display, className="text-success me-3"),
                    html.Span("Sell Depth: ", className="text-muted"),
                    html.Span(sell_depth_display, className="text-danger"),
                ]
            ),
        ],
        className="p-3 border rounded",
    )


def render_support_resistance_levels(sr_levels: Dict[str, Any]) -> html.Div:
    """
    Render support and resistance levels.

    Args:
        sr_levels: Dictionary with support and resistance levels

    Returns:
        Dash HTML component
    """
    support_levels = sr_levels.get("support_levels", [])
    resistance_levels = sr_levels.get("resistance_levels", [])

    # Sort levels
    support_levels = sorted(support_levels, reverse=True)
    resistance_levels = sorted(resistance_levels)

    # Limit to top 3 of each
    support_levels = support_levels[:3]
    resistance_levels = resistance_levels[:3]

    # Create support level items
    support_items = []
    for i, level in enumerate(support_levels):
        strength = 3 - i  # Stronger levels first
        support_items.append(
            html.Div(
                [
                    html.I(
                        className="bi bi-arrow-up-circle-fill me-2",
                        style={"color": "#28a745", "opacity": 0.5 + (strength * 0.15)},
                    ),
                    html.Span(f"S{i+1}: {level:,.2f}"),
                ],
                className="d-flex align-items-center mb-1",
            )
        )

    # Create resistance level items
    resistance_items = []
    for i, level in enumerate(resistance_levels):
        strength = 3 - i  # Stronger levels first
        resistance_items.append(
            html.Div(
                [
                    html.I(
                        className="bi bi-arrow-down-circle-fill me-2",
                        style={"color": "#dc3545", "opacity": 0.5 + (strength * 0.15)},
                    ),
                    html.Span(f"R{i+1}: {level:,.2f}"),
                ],
                className="d-flex align-items-center mb-1",
            )
        )

    return html.Div(
        [
            html.H6("Support & Resistance", className="card-subtitle mb-3 text-muted"),
            html.Div(
                [
                    html.Div(
                        [html.H6("Resistance", className="mb-2")] + resistance_items,
                        className="col-6",
                    ),
                    html.Div(
                        [html.H6("Support", className="mb-2")] + support_items,
                        className="col-6",
                    ),
                ],
                className="row",
            ),
        ],
        className="p-3 border rounded",
    )


def render_level_strength_indicator(
    level_type: str, price: float, strength: float
) -> html.Div:
    """
    Render a support/resistance level strength indicator.

    Args:
        level_type: "support" or "resistance"
        price: Price level
        strength: Strength value (0.0 to 1.0)

    Returns:
        Dash HTML component
    """
    # Determine color and icon based on type and strength
    if level_type.lower() == "support":
        base_color = "success"
        icon = "bi bi-arrow-up-circle"
    else:
        base_color = "danger"
        icon = "bi bi-arrow-down-circle"

    # Add fill to icon if strong
    if strength > 0.7:
        icon += "-fill"

    # Create strength bars
    strength_pct = int(strength * 100)
    bars = []
    for i in range(5):
        threshold = i * 20
        bar_class = f"bg-{base_color}" if strength_pct >= threshold else "bg-secondary"
        bar_opacity = 0.4 + (i * 0.15) if strength_pct >= threshold else 0.2

        bars.append(
            html.Div(
                className=f"{bar_class}",
                style={
                    "width": "15px",
                    "height": f"{5 + (i * 2)}px",
                    "margin-right": "2px",
                    "opacity": bar_opacity,
                },
            )
        )

    return html.Div(
        [
            html.Div(
                [
                    html.I(className=f"{icon} me-2"),
                    html.Span(
                        f"{level_type.capitalize()}: {price:,.2f}",
                        className="me-2",
                    ),
                    html.Small(f"{strength_pct}%", className="text-muted"),
                ],
                className="d-flex align-items-center mb-1",
            ),
            html.Div(
                bars,
                className="d-flex align-items-end",
            ),
        ],
        className="mb-3",
    )


def create_level_confluence_chart(
    price_levels: List[Dict[str, Any]], current_price: float, height: int = 400
) -> go.Figure:
    """
    Create a chart showing confluence of support/resistance levels.

    Args:
        price_levels: List of dictionaries with level data
        current_price: Current market price
        height: Chart height in pixels

    Returns:
        Plotly figure object
    """
    if not price_levels:
        return create_empty_chart("No Price Levels Available")

    # Extract data
    prices = [level.get("price", 0) for level in price_levels]
    strengths = [level.get("strength", 0) for level in price_levels]
    types = [level.get("type", "unknown") for level in price_levels]
    sources = [level.get("source", "unknown") for level in price_levels]

    # Create colors based on type
    colors = [
        "rgba(40, 167, 69, 0.7)" if t.lower() == "support" else "rgba(220, 53, 69, 0.7)"
        for t in types
    ]

    # Create hover text
    hover_texts = [
        f"Price: {p:,.2f}<br>"
        f"Type: {t.capitalize()}<br>"
        f"Strength: {s:.2f}<br>"
        f"Source: {src.capitalize()}"
        for p, t, s, src in zip(prices, types, strengths, sources)
    ]

    # Create figure
    fig = go.Figure()

    # Add horizontal bars for levels
    fig.add_trace(
        go.Bar(
            x=strengths,
            y=[f"{p:,.2f}" for p in prices],
            orientation="h",
            marker_color=colors,
            text=types,
            hoverinfo="text",
            hovertext=hover_texts,
        )
    )

    # Add line for current price
    fig.add_shape(
        type="line",
        x0=0,
        x1=1,
        y0=current_price,
        y1=current_price,
        xref="paper",
        line=dict(color="rgba(0, 123, 255, 0.7)", width=2, dash="dash"),
    )

    # Add annotation for current price
    fig.add_annotation(
        x=0.95,
        y=current_price,
        xref="paper",
        text=f"Current: {current_price:,.2f}",
        showarrow=False,
        font=dict(color="rgba(0, 123, 255, 1)"),
        bgcolor="rgba(255, 255, 255, 0.7)",
    )

    # Apply standard theme
    fig = apply_chart_theme(fig, "Price Level Confluence")

    # Update layout
    fig.update_layout(
        height=height,
        xaxis_title="Strength",
        yaxis_title="Price",
        yaxis=dict(
            type="category",
            autorange="reversed",  # Higher prices at top
        ),
        margin=dict(l=10, r=10, t=40, b=10),
    )

    return fig


def render_execution_recommendations(recommendations: Dict[str, Any]) -> html.Div:
    """
    Render execution recommendations based on order book analysis.

    Args:
        recommendations: Dictionary with execution recommendations

    Returns:
        Dash HTML component
    """
    # Extract recommendation data
    recommendation_type = recommendations.get("recommendation", {}).get(
        "order_type", "unknown"
    )
    limit_price = recommendations.get("recommendation", {}).get("limit_price")
    should_split = recommendations.get("recommendation", {}).get("should_split", False)
    num_parts = recommendations.get("recommendation", {}).get("num_parts", 1)
    confidence = recommendations.get("confidence_score", 0.5)

    # Market conditions
    market_conditions = recommendations.get("market_conditions", {})
    imbalance = market_conditions.get("imbalance", 0)
    spread_pct = market_conditions.get("spread_pct", 0)

    # Determine recommendation color and icon
    if recommendation_type == "market":
        rec_color = "danger"
        rec_icon = "bi bi-lightning-fill"
        rec_text = "Market Order"
    elif recommendation_type == "limit":
        rec_color = "success"
        rec_icon = "bi bi-hourglass-split"
        rec_text = "Limit Order"
    else:
        rec_color = "secondary"
        rec_icon = "bi bi-question-circle"
        rec_text = "Unknown"

    # Create confidence indicator
    confidence_pct = int(confidence * 100)
    confidence_color = (
        "danger" if confidence < 0.3 else "warning" if confidence < 0.6 else "success"
    )

    # Create recommendation card
    return html.Div(
        [
            html.H6(
                "Execution Recommendations", className="card-subtitle mb-3 text-muted"
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.I(className=f"{rec_icon} me-2"),
                            html.Span(rec_text),
                        ],
                        className=f"text-{rec_color} d-flex align-items-center mb-2 fs-5",
                    ),
                    html.Div(
                        [
                            html.Span("Confidence: ", className="text-muted me-2"),
                            dbc.Progress(
                                value=confidence_pct,
                                color=confidence_color,
                                className="mb-2",
                                style={"height": "8px"},
                            ),
                            html.Small(f"{confidence_pct}%", className="text-muted"),
                        ],
                        className="mb-3",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span("Price: ", className="text-muted"),
                                    html.Span(
                                        (
                                            f"{limit_price:,.2f}"
                                            if limit_price
                                            else "Market"
                                        ),
                                        className=f"text-{rec_color}",
                                    ),
                                ],
                                className="me-3",
                            ),
                            html.Div(
                                [
                                    html.Span("Split: ", className="text-muted"),
                                    html.Span(
                                        (
                                            f"Yes ({num_parts} parts)"
                                            if should_split
                                            else "No"
                                        ),
                                    ),
                                ],
                            ),
                        ],
                        className="d-flex mb-3",
                    ),
                    html.Hr(className="my-2"),
                    html.Small(
                        [
                            html.Div(
                                [
                                    html.Span("Imbalance: ", className="text-muted"),
                                    html.Span(
                                        f"{imbalance:.2f}",
                                        className=(
                                            "text-success"
                                            if imbalance > 0
                                            else (
                                                "text-danger"
                                                if imbalance < 0
                                                else "text-secondary"
                                            )
                                        ),
                                    ),
                                ],
                                className="me-3",
                            ),
                            html.Div(
                                [
                                    html.Span("Spread: ", className="text-muted"),
                                    html.Span(f"{spread_pct:.2f}%"),
                                ],
                            ),
                        ],
                        className="d-flex",
                    ),
                ],
                className="p-3 border rounded",
            ),
        ],
    )
