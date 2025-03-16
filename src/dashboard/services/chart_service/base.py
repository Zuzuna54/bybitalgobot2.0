"""
Base Chart Service Module

This module provides common functionality and constants
for chart generation used throughout the dashboard.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from loguru import logger


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
    Apply standard chart theme to the given figure.

    Args:
        fig: Plotly figure to apply theme to
        title: Optional title for the chart

    Returns:
        Plotly figure with theme applied
    """
    # Update layout with theme settings
    fig.update_layout(
        template=CHART_THEME["template"],
        font=CHART_THEME["font"],
        legend=CHART_THEME["legend"],
        margin=CHART_THEME["margin"],
    )

    # Add title if provided
    if title:
        fig.update_layout(
            title={
                "text": title,
                "font": CHART_THEME["title_font"],
                "x": 0.5,
                "xanchor": "center",
            }
        )

    # Update axes
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

    logger.debug(
        f"Applied chart theme to figure{' with title: ' + title if title else ''}"
    )
    return fig


def create_empty_chart(title: str) -> go.Figure:
    """
    Create an empty chart with a message.

    Args:
        title: Title for the empty chart

    Returns:
        Empty Plotly figure with a message
    """
    fig = go.Figure()
    fig.add_annotation(
        text="No data available",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color="#666666"),
    )
    fig = apply_chart_theme(fig, title)
    return fig


def create_empty_sparkline() -> go.Figure:
    """
    Create an empty sparkline chart.

    Returns:
        Empty Plotly figure suitable for small sparkline display
    """
    fig = go.Figure()
    fig.update_layout(
        height=50,
        width=120,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    fig.update_xaxes(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        visible=False,
    )
    fig.update_yaxes(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        visible=False,
    )
    return fig


def filter_data_by_time_range(data: pd.DataFrame, time_range: str) -> pd.DataFrame:
    """
    Filter DataFrame by time range from present.

    Args:
        data: DataFrame with a DateTimeIndex
        time_range: Time range string (e.g., '1d', '1w', '1m', '3m', '6m', '1y', 'all')

    Returns:
        Filtered DataFrame
    """
    if data.empty or time_range == "all":
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
    elif time_range == "6m":
        start_date = now - timedelta(days=180)
    elif time_range == "1y":
        start_date = now - timedelta(days=365)
    else:
        logger.warning(f"Unknown time range: {time_range}, returning all data")
        return data

    logger.debug(
        f"Filtering data from {start_date} to {now} (time_range: {time_range})"
    )
    return data[data.index >= start_date]
