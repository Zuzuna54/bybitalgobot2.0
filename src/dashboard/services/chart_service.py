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


# Standard chart theme configuration
CHART_THEME = {
    "template": "plotly_white",
    "font": {
        "family": "Arial, sans-serif",
        "size": 12,
        "color": "#444"
    },
    "title_font": {
        "family": "Arial, sans-serif",
        "size": 16,
        "color": "#333"
    },
    "legend": {
        "orientation": "h",
        "yanchor": "bottom",
        "y": 1.02,
        "xanchor": "right",
        "x": 1
    },
    "colorway": [
        "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", 
        "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"
    ],
    "grid": {
        "showgrid": True,
        "gridcolor": "#f0f0f0",
        "zeroline": False
    },
    "margin": {
        "l": 40, 
        "r": 40, 
        "t": 40, 
        "b": 40
    }
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
        legend=CHART_THEME["legend"]
    )
    
    # Apply title if provided
    if title:
        fig.update_layout(
            title={
                "text": title,
                "font": CHART_THEME["title_font"],
                "x": 0.5,
                "xanchor": "center"
            }
        )
    
    # Apply grid settings
    fig.update_xaxes(
        showgrid=CHART_THEME["grid"]["showgrid"],
        gridcolor=CHART_THEME["grid"]["gridcolor"],
        zeroline=CHART_THEME["grid"]["zeroline"]
    )
    fig.update_yaxes(
        showgrid=CHART_THEME["grid"]["showgrid"],
        gridcolor=CHART_THEME["grid"]["gridcolor"],
        zeroline=CHART_THEME["grid"]["zeroline"]
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
        x=0.5, y=0.5,
        text="No data available",
        showarrow=False,
        font=dict(size=16, color="#888888")
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
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="lines",
        line=dict(color="green", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(0,255,0,0.1)"
    ))
    
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


def create_equity_curve_chart(equity_data: pd.DataFrame, time_range: str = "1m") -> go.Figure:
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
    fig.add_trace(go.Scatter(
        x=filtered_data.index,
        y=filtered_data['equity'],
        mode='lines',
        name='Equity',
        line=dict(color="#636EFA", width=2)
    ))
    
    # Add initial equity line
    initial_equity = filtered_data['equity'].iloc[0]
    fig.add_trace(go.Scatter(
        x=[filtered_data.index[0], filtered_data.index[-1]],
        y=[initial_equity, initial_equity],
        mode='lines',
        name='Initial Equity',
        line=dict(color="#EF553B", width=1, dash='dash')
    ))
    
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


def create_return_distribution_chart(returns_data: pd.DataFrame, time_range: str = "1m") -> go.Figure:
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
        labels={"return_pct": "Daily Return (%)"}
    )
    
    # Add a vertical line at zero
    fig.add_shape(
        type="line",
        x0=0, y0=0,
        x1=0, y1=1,
        yref="paper",
        line=dict(color="red", width=2, dash="dash")
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


def create_drawdown_chart(equity_data: pd.DataFrame, time_range: str = "1m") -> go.Figure:
    """
    Create a drawdown chart from equity data.
    
    Args:
        equity_data: DataFrame with equity and drawdown history
        time_range: Time range to display ("1d", "1w", "1m", "3m", "all")
        
    Returns:
        Plotly figure object
    """
    if equity_data is None or equity_data.empty or "drawdown_pct" not in equity_data.columns:
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
    fig.add_trace(go.Scatter(
        x=filtered_data.index,
        y=-filtered_data["drawdown_pct"],  # Negate to show as negative values
        mode="lines",
        line=dict(color="red", width=2),
        fill="tozeroy",
        fillcolor="rgba(255,0,0,0.1)",
        name="Drawdown"
    ))
    
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
            rangemode="nonpositive"  # Only show negative values
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
    fig.add_trace(go.Bar(
        x=names,
        y=returns,
        marker_color=colors,
        name="Return (%)"
    ))
    
    # Add trade count as a line on a secondary axis
    fig.add_trace(go.Scatter(
        x=names,
        y=trades,
        mode="markers+lines",
        marker=dict(size=10),
        line=dict(color="royalblue", width=2),
        name="Trade Count",
        yaxis="y2"
    ))
    
    # Update layout
    fig.update_layout(
        title="Strategy Performance Comparison",
        template="plotly_white",
        xaxis=dict(showgrid=False, zeroline=False, title="Strategy"),
        yaxis=dict(
            showgrid=True, 
            zeroline=True, 
            title="Return (%)",
            side="left"
        ),
        yaxis2=dict(
            showgrid=False,
            zeroline=False,
            title="Trade Count",
            side="right",
            overlaying="y",
            rangemode="nonnegative"
        ),
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        barmode="group"
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
    
    fig.add_trace(go.Pie(
        values=[wins, losses],
        labels=["Profitable", "Unprofitable"],
        hole=0.6,
        marker=dict(colors=["green", "red"]),
        textinfo="label+percent",
        insidetextorientation="radial"
    ))
    
    # Add text in the middle with win/loss count
    fig.add_annotation(
        text=f"{wins}/{losses}",
        font=dict(size=20, family="Arial", color="black"),
        showarrow=False,
        x=0.5,
        y=0.5
    )
    
    # Update layout
    fig.update_layout(
        title="Win/Loss Distribution",
        template="plotly_white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


def create_custom_indicator_chart(
    market_data: pd.DataFrame,
    indicator_name: str,
    symbol: str,
    timeframe: str
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
    fig.add_trace(go.Candlestick(
        x=market_data.index,
        open=market_data["open"],
        high=market_data["high"],
        low=market_data["low"],
        close=market_data["close"],
        name="Price",
        showlegend=True
    ))
    
    # Add indicator as a line
    fig.add_trace(go.Scatter(
        x=market_data.index,
        y=market_data[indicator_name],
        mode="lines",
        line=dict(color="purple", width=2),
        name=indicator_name,
        yaxis="y2"
    ))
    
    # Update layout
    fig.update_layout(
        title=f"{indicator_name} - {symbol} ({timeframe})",
        template="plotly_white",
        xaxis=dict(
            showgrid=False, 
            zeroline=False, 
            rangeslider=dict(visible=False),
            type="date"
        ),
        yaxis=dict(
            showgrid=True,
            zeroline=False,
            title="Price",
            side="left"
        ),
        yaxis2=dict(
            showgrid=False,
            zeroline=False,
            title=indicator_name,
            side="right",
            overlaying="y"
        ),
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig 