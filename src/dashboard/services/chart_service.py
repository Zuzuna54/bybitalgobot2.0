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


def create_candlestick_chart(
    candles: pd.DataFrame,
    symbol: str = "",
    title: Optional[str] = None,
    show_volume: bool = True,
    height: int = 500
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
    fig = go.Figure(data=[
        go.Candlestick(
            x=candles['timestamp'] if 'timestamp' in candles.columns else candles.index,
            open=candles['open'],
            high=candles['high'],
            low=candles['low'],
            close=candles['close'],
            name="Price"
        )
    ])
    
    # Add volume as bar chart on secondary y-axis
    if show_volume and 'volume' in candles.columns:
        fig.add_trace(
            go.Bar(
                x=candles['timestamp'] if 'timestamp' in candles.columns else candles.index,
                y=candles['volume'],
                name="Volume",
                marker_color='rgba(128, 128, 128, 0.5)',
                yaxis="y2"
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
        yaxis2=dict(
            title="Volume",
            overlaying="y",
            side="right",
            showgrid=False
        ),
        xaxis_rangeslider_visible=False,
        height=height
    )
    
    return fig


def create_orderbook_depth_chart(
    orderbook: Dict[str, Any],
    depth: int = 20,
    support_levels: Optional[List[float]] = None,
    resistance_levels: Optional[List[float]] = None,
    height: int = 500
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
    if not orderbook or 'bids' not in orderbook or 'asks' not in orderbook:
        return create_empty_chart("Orderbook Depth")
    
    bids = orderbook['bids']
    asks = orderbook['asks']
    
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
    fig.add_trace(go.Scatter(
        x=bid_prices[::-1],  # Reverse to show highest bids on left
        y=bid_cumulative[::-1],
        mode='lines',
        name='Bids',
        line=dict(color='rgba(50, 171, 96, 0.8)', width=2),
        fill='tozeroy',
        fillcolor='rgba(50, 171, 96, 0.3)'
    ))
    
    # Add ask depth trace
    fig.add_trace(go.Scatter(
        x=ask_prices,
        y=ask_cumulative,
        mode='lines',
        name='Asks',
        line=dict(color='rgba(220, 53, 69, 0.8)', width=2),
        fill='tozeroy',
        fillcolor='rgba(220, 53, 69, 0.3)'
    ))
    
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
                name=f"Support: {level:.2f}"
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
                name=f"Resistance: {level:.2f}"
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
        name="Current Price"
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
        font=dict(size=12, color="rgba(70, 130, 180, 1)")
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Price",
        yaxis_title="Cumulative Size",
        height=height,
        hovermode="x unified"
    )
    
    # Update hover templates
    fig.update_traces(
        hovertemplate="<b>Price</b>: $%{x:.2f}<br><b>Cumulative Size</b>: %{y:.4f}<extra></extra>"
    )
    
    return fig


def create_strategy_performance_graph(strategy_performance: List[Dict[str, Any]]) -> go.Figure:
    """
    Create the strategy performance comparison graph.
    
    Args:
        strategy_performance: List of strategy performance dictionaries
        
    Returns:
        Plotly figure object
    """
    if not strategy_performance:
        fig = go.Figure()
        fig.update_layout(
            title="No strategy performance data available",
            template="plotly_white"
        )
        return fig
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(strategy_performance)
    
    # Prepare metrics for visualization
    metrics = ['win_rate', 'profit_factor', 'avg_profit_pct', 'sharpe_ratio']
    display_names = {
        'win_rate': 'Win Rate',
        'profit_factor': 'Profit Factor',
        'avg_profit_pct': 'Avg. Profit %',
        'sharpe_ratio': 'Sharpe Ratio'
    }
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[display_names[m] for m in metrics],
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    # Add bars for each metric
    for i, metric in enumerate(metrics):
        row = i // 2 + 1
        col = i % 2 + 1
        
        # Sort by current metric
        sorted_df = df.sort_values(metric, ascending=False)
        
        # Add bars
        fig.add_trace(
            go.Bar(
                x=sorted_df['strategy_name'],
                y=sorted_df[metric],
                marker=dict(
                    color='rgba(50, 171, 96, 0.7)',
                    line=dict(color='rgba(50, 171, 96, 1.0)', width=1)
                ),
                name=display_names[metric]
            ),
            row=row, col=col
        )
    
    # Update layout
    fig.update_layout(
        height=600,
        showlegend=False,
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    # Apply standard theme
    apply_chart_theme(fig)
    
    return fig


def create_strategy_comparison_graph(
    strategy_performance: List[Dict[str, Any]],
    selected_strategies: List[str]
) -> go.Figure:
    """
    Create the strategy comparison graph.
    
    Args:
        strategy_performance: List of strategy performance dictionaries
        selected_strategies: List of strategy names to compare
        
    Returns:
        Plotly figure object
    """
    if not strategy_performance or not selected_strategies:
        fig = go.Figure()
        fig.update_layout(
            title="Select strategies to compare",
            template="plotly_white"
        )
        return fig
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(strategy_performance)
    
    # Filter selected strategies
    df = df[df['strategy_name'].isin(selected_strategies)]
    
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No data for selected strategies",
            template="plotly_white"
        )
        return fig
    
    # Select metrics for comparison
    metrics = [
        'win_rate', 'profit_factor', 'total_trades', 'total_pnl',
        'avg_profit_pct', 'max_drawdown', 'sharpe_ratio', 'avg_trade_duration_hours'
    ]
    
    # Format for display
    metric_display = {
        'win_rate': 'Win Rate (%)',
        'profit_factor': 'Profit Factor',
        'total_trades': 'Total Trades',
        'total_pnl': 'Total P&L ($)',
        'avg_profit_pct': 'Avg. Profit (%)',
        'max_drawdown': 'Max Drawdown (%)',
        'sharpe_ratio': 'Sharpe Ratio',
        'avg_trade_duration_hours': 'Avg. Trade Duration (hrs)'
    }
    
    # Create polar chart for comparison
    fig = go.Figure()
    
    # Normalize metrics to 0-1 range for radar chart
    df_norm = df.copy()
    for metric in metrics:
        if metric in df.columns:
            if df[metric].max() == df[metric].min():
                df_norm[metric] = 0.5
            else:
                # Normalize between 0 and 1
                if metric == 'max_drawdown':
                    # Invert drawdown (lower is better)
                    max_val = df[metric].max()
                    min_val = df[metric].min()
                    df_norm[metric] = 1 - ((df[metric] - min_val) / (max_val - min_val) if max_val > min_val else 0)
                else:
                    max_val = df[metric].max()
                    min_val = df[metric].min()
                    df_norm[metric] = (df[metric] - min_val) / (max_val - min_val) if max_val > min_val else 0
    
    # Add traces for each strategy
    for _, strategy in df.iterrows():
        strategy_name = strategy['strategy_name']
        strategy_norm = df_norm[df_norm['strategy_name'] == strategy_name].iloc[0]
        
        # Get values for available metrics
        available_metrics = [m for m in metrics if m in df.columns]
        values = [strategy_norm[m] for m in available_metrics]
        
        # Add radar trace
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=[metric_display.get(m, m) for m in available_metrics],
            fill='toself',
            name=strategy_name
        ))
    
    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        template="plotly_white",
        height=500,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    # Apply standard theme
    apply_chart_theme(fig)
    
    return fig


def create_detailed_performance_breakdown(
    strategy_performance: List[Dict[str, Any]],
    selected_strategy: str = None
) -> go.Figure:
    """
    Create a detailed performance breakdown by timeframe.
    
    Args:
        strategy_performance: List of strategy performance dictionaries
        selected_strategy: The specific strategy to analyze (if None, uses top performer)
        
    Returns:
        Plotly figure with detailed performance breakdown
    """
    if not strategy_performance:
        fig = go.Figure()
        fig.update_layout(
            title="No strategy performance data available",
            template="plotly_white"
        )
        return fig
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(strategy_performance)
    
    # If no strategy is selected, use the top performer by total PnL
    if not selected_strategy or selected_strategy not in df['strategy_name'].values:
        if 'total_pnl' in df.columns:
            df = df.sort_values('total_pnl', ascending=False)
            selected_strategy = df.iloc[0]['strategy_name']
        else:
            selected_strategy = df.iloc[0]['strategy_name']
    
    # Filter to selected strategy
    strategy_data = df[df['strategy_name'] == selected_strategy].iloc[0]
    
    # Check if we have timeframe data
    timeframe_data = strategy_data.get('timeframe_performance', {})
    if not timeframe_data:
        fig = go.Figure()
        fig.update_layout(
            title=f"No timeframe data available for {selected_strategy}",
            template="plotly_white"
        )
        return fig
    
    # Convert timeframe data to DataFrame
    timeframe_df = pd.DataFrame([
        {"timeframe": tf, **metrics}
        for tf, metrics in timeframe_data.items()
    ])
    
    # Create the figure with subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Win Rate by Timeframe", 
            "Profit Factor by Timeframe",
            "Average Profit % by Timeframe", 
            "Number of Trades by Timeframe"
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # Sort timeframes
    timeframe_order = {
        'hourly': 0, 
        'daily': 1, 
        'weekly': 2, 
        'monthly': 3
    }
    
    timeframe_df['sort_order'] = timeframe_df['timeframe'].map(
        lambda x: timeframe_order.get(x, 999)
    )
    timeframe_df = timeframe_df.sort_values('sort_order')
    
    # Add traces
    fig.add_trace(
        go.Bar(
            x=timeframe_df['timeframe'],
            y=timeframe_df['win_rate'],
            marker_color='rgba(50, 171, 96, 0.7)',
            name="Win Rate"
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=timeframe_df['timeframe'],
            y=timeframe_df['profit_factor'],
            marker_color='rgba(55, 83, 109, 0.7)',
            name="Profit Factor"
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(
            x=timeframe_df['timeframe'],
            y=timeframe_df['avg_profit_pct'],
            marker_color='rgba(70, 130, 180, 0.7)',
            name="Avg Profit %"
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=timeframe_df['timeframe'],
            y=timeframe_df['trade_count'],
            marker_color='rgba(214, 39, 40, 0.7)',
            name="Trade Count"
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        title_text=f"Performance Breakdown for {selected_strategy}",
        showlegend=False,
        template="plotly_white",
        margin=dict(l=40, r=40, t=80, b=40)
    )
    
    # Apply standard theme
    apply_chart_theme(fig)
    
    return fig


def create_market_condition_performance(
    strategy_performance: List[Dict[str, Any]],
    selected_strategy: str = None
) -> go.Figure:
    """
    Create a visualization of strategy performance under different market conditions.
    
    Args:
        strategy_performance: List of strategy performance dictionaries
        selected_strategy: The specific strategy to analyze (if None, uses top performer)
        
    Returns:
        Plotly figure with market condition performance data
    """
    if not strategy_performance:
        fig = go.Figure()
        fig.update_layout(
            title="No strategy performance data available",
            template="plotly_white"
        )
        return fig
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(strategy_performance)
    
    # If no strategy is selected, use the top performer by total PnL
    if not selected_strategy or selected_strategy not in df['strategy_name'].values:
        if 'total_pnl' in df.columns:
            df = df.sort_values('total_pnl', ascending=False)
            selected_strategy = df.iloc[0]['strategy_name']
        else:
            selected_strategy = df.iloc[0]['strategy_name']
    
    # Filter to selected strategy
    strategy_data = df[df['strategy_name'] == selected_strategy].iloc[0]
    
    # Check if we have market condition data
    market_condition_data = strategy_data.get('market_condition_performance', {})
    if not market_condition_data:
        fig = go.Figure()
        fig.update_layout(
            title=f"No market condition data available for {selected_strategy}",
            template="plotly_white"
        )
        return fig
    
    # Convert market condition data to DataFrame
    market_df = pd.DataFrame([
        {"condition": condition, **metrics}
        for condition, metrics in market_condition_data.items()
    ])
    
    # Create heatmap data
    conditions = market_df['condition'].tolist()
    metrics = ['win_rate', 'profit_factor', 'avg_profit_pct', 'sharpe_ratio']
    
    # Create normalized values for better heatmap visualization
    for metric in metrics:
        if metric in market_df.columns:
            max_val = market_df[metric].max()
            min_val = market_df[metric].min()
            if max_val > min_val:
                market_df[f'{metric}_normalized'] = (market_df[metric] - min_val) / (max_val - min_val)
            else:
                market_df[f'{metric}_normalized'] = 0.5
    
    # Create the figure with subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Win Rate by Market Condition", 
            "Profit Factor by Market Condition",
            "Avg Profit % by Market Condition", 
            "Sharpe Ratio by Market Condition"
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # Add bars for each metric
    for i, metric in enumerate(metrics):
        if metric in market_df.columns:
            row = i // 2 + 1
            col = i % 2 + 1
            
            fig.add_trace(
                go.Bar(
                    x=market_df['condition'],
                    y=market_df[metric],
                    marker=dict(
                        color=market_df[f'{metric}_normalized'],
                        colorscale='Viridis',
                        showscale=False
                    ),
                    text=market_df[metric].round(2),
                    textposition='auto',
                    name=metric.replace('_', ' ').title(),
                ),
                row=row, col=col
            )
    
    # Update layout
    fig.update_layout(
        height=600,
        title_text=f"Performance by Market Condition for {selected_strategy}",
        showlegend=False,
        template="plotly_white",
        margin=dict(l=40, r=40, t=80, b=40)
    )
    
    # Apply standard theme
    apply_chart_theme(fig)
    
    return fig


def create_strategy_correlation_matrix(strategy_performance: List[Dict[str, Any]]) -> go.Figure:
    """
    Create a correlation matrix visualization showing how strategies correlate with each other.
    
    Args:
        strategy_performance: List of strategy performance dictionaries
        
    Returns:
        Plotly figure with correlation matrix
    """
    if not strategy_performance or len(strategy_performance) < 2:
        fig = go.Figure()
        fig.update_layout(
            title="Insufficient data for correlation analysis",
            template="plotly_white"
        )
        return fig
    
    # Extract daily returns if available
    strategies_with_returns = []
    strategy_names = []
    
    for strategy in strategy_performance:
        daily_returns = strategy.get('daily_returns', {})
        if daily_returns:
            strategies_with_returns.append(daily_returns)
            strategy_names.append(strategy['strategy_name'])
    
    if not strategies_with_returns:
        fig = go.Figure()
        fig.update_layout(
            title="No daily returns data available for correlation analysis",
            template="plotly_white"
        )
        return fig
    
    # Create DataFrame with daily returns for each strategy
    returns_df = pd.DataFrame()
    
    for i, returns in enumerate(strategies_with_returns):
        series = pd.Series(returns)
        returns_df[strategy_names[i]] = series
    
    # Fill NaN values with 0 (for days where strategy didn't trade)
    returns_df = returns_df.fillna(0)
    
    # Calculate correlation matrix
    corr_matrix = returns_df.corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu_r',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Strategy Correlation Matrix",
        height=500,
        template="plotly_white",
        xaxis=dict(
            title="Strategy",
            tickangle=-45
        ),
        yaxis=dict(
            title="Strategy"
        ),
        margin=dict(l=60, r=40, t=80, b=60)
    )
    
    # Apply standard theme
    apply_chart_theme(fig)
    
    return fig


def create_pnl_by_symbol_graph(trade_history: List[Dict[str, Any]]) -> go.Figure:
    """
    Create the P&L by symbol graph.
    
    Args:
        trade_history: List of completed trade dictionaries
        
    Returns:
        Plotly figure object
    """
    if not trade_history:
        fig = go.Figure()
        fig.update_layout(
            title="No trade data available",
            template="plotly_white"
        )
        return fig
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(trade_history)
    
    # Filter only completed trades
    df = df[df['realized_pnl'].notna()]
    
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No completed trades available",
            template="plotly_white"
        )
        return fig
    
    # Group by symbol and calculate total P&L
    symbol_pnl = df.groupby('symbol')['realized_pnl'].sum().reset_index()
    symbol_pnl = symbol_pnl.sort_values('realized_pnl', ascending=False)
    
    # Create figure
    fig = go.Figure()
    
    # Add P&L bars
    fig.add_trace(go.Bar(
        x=symbol_pnl['symbol'],
        y=symbol_pnl['realized_pnl'],
        marker=dict(
            color=symbol_pnl['realized_pnl'].apply(
                lambda x: 'rgba(50, 171, 96, 0.7)' if x >= 0 else 'rgba(220, 53, 69, 0.7)'
            )
        )
    ))
    
    # Apply standard theme
    fig = apply_chart_theme(fig, "P&L by Symbol")
    
    # Update additional layout settings
    fig.update_layout(
        xaxis_title="Symbol",
        yaxis_title="P&L ($)",
        height=350
    )
    
    return fig


def create_win_loss_by_strategy_graph(trade_history: List[Dict[str, Any]]) -> go.Figure:
    """
    Create the win/loss by strategy graph.
    
    Args:
        trade_history: List of completed trade dictionaries
        
    Returns:
        Plotly figure object
    """
    if not trade_history:
        fig = go.Figure()
        fig.update_layout(
            title="No trade data available",
            template="plotly_white"
        )
        return fig
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(trade_history)
    
    # Filter only completed trades
    df = df[df['realized_pnl'].notna()]
    
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No completed trades available",
            template="plotly_white"
        )
        return fig
    
    # Group by strategy and count wins/losses
    df['is_win'] = df['realized_pnl'] > 0
    
    strategy_performance = df.groupby('strategy_name')['is_win'].agg(['sum', 'count']).reset_index()
    strategy_performance['win_rate'] = strategy_performance['sum'] / strategy_performance['count'] * 100
    strategy_performance['loss_count'] = strategy_performance['count'] - strategy_performance['sum']
    strategy_performance = strategy_performance.sort_values('win_rate', ascending=False)
    
    # Create figure
    fig = go.Figure()
    
    # Add win count bars
    fig.add_trace(go.Bar(
        x=strategy_performance['strategy_name'],
        y=strategy_performance['sum'],
        name='Wins',
        marker=dict(color='rgba(50, 171, 96, 0.7)')
    ))
    
    # Add loss count bars
    fig.add_trace(go.Bar(
        x=strategy_performance['strategy_name'],
        y=strategy_performance['loss_count'],
        name='Losses',
        marker=dict(color='rgba(220, 53, 69, 0.7)')
    ))
    
    # Apply standard theme
    fig = apply_chart_theme(fig, "Win/Loss by Strategy")
    
    # Update additional layout settings
    fig.update_layout(
        xaxis_title="Strategy",
        yaxis_title="Trade Count",
        barmode='stack',
        height=350
    )
    
    return fig 