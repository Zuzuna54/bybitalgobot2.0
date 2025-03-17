"""
Strategy Performance Views

This module provides visualization components for displaying strategy performance metrics,
including graphs and comparison views.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import html, dcc
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
from loguru import logger


def create_strategy_performance_graph(
    strategy_performance: List[Dict[str, Any]],
) -> go.Figure:
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
            title="No strategy performance data available", template="plotly_white"
        )
        return fig

    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(strategy_performance)

    # Prepare metrics for visualization
    metrics = ["win_rate", "profit_factor", "avg_profit_pct", "sharpe_ratio"]
    display_names = {
        "win_rate": "Win Rate",
        "profit_factor": "Profit Factor",
        "avg_profit_pct": "Avg. Profit %",
        "sharpe_ratio": "Sharpe Ratio",
    }

    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[display_names[m] for m in metrics],
        vertical_spacing=0.1,
        horizontal_spacing=0.1,
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
                x=sorted_df["strategy_name"],
                y=sorted_df[metric],
                marker=dict(
                    color="rgba(50, 171, 96, 0.7)",
                    line=dict(color="rgba(50, 171, 96, 1.0)", width=1),
                ),
                name=display_names[metric],
            ),
            row=row,
            col=col,
        )

    # Update layout
    fig.update_layout(
        height=600,
        showlegend=False,
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40),
    )

    return fig


def render_top_strategies_card(strategy_performance: List[Dict[str, Any]]) -> html.Div:
    """
    Render the top strategies card content.

    Args:
        strategy_performance: List of strategy performance dictionaries

    Returns:
        HTML Div with top strategies metrics
    """
    if not strategy_performance:
        return html.Div("No strategy data available")

    # Convert to DataFrame and sort by profit
    df = pd.DataFrame(strategy_performance)

    # Check for the correct profit field
    if "total_profit" in df.columns:
        profit_field = "total_profit"
    elif (
        "performance" in df.columns
        and df["performance"]
        .apply(lambda x: isinstance(x, dict) and "total_profit" in x)
        .any()
    ):
        # Extract total_profit from nested performance dict
        df["total_profit"] = df["performance"].apply(
            lambda x: x.get("total_profit", 0.0) if isinstance(x, dict) else 0.0
        )
        profit_field = "total_profit"
    else:
        # Fallback to a default field or 0
        logger.warning("Could not find profit field in strategy data")
        df["total_profit"] = 0.0
        profit_field = "total_profit"

    # Sort by the profit field
    df = df.sort_values(profit_field, ascending=False)

    # Take top 5 strategies
    top_strategies = df.head(5)

    # Create table rows
    rows = []
    for i, (_, strategy) in enumerate(top_strategies.iterrows()):
        # Get strategy name - could be in different fields based on the data structure
        if "name" in strategy:
            strategy_name = strategy["name"]
        elif "strategy_name" in strategy:
            strategy_name = strategy["strategy_name"]
        else:
            strategy_name = f"Strategy {i+1}"

        # Get win rate - could be nested in the performance dictionary
        if "win_rate" in strategy:
            win_rate = strategy["win_rate"]
        elif "performance" in strategy and isinstance(strategy["performance"], dict):
            win_rate = strategy["performance"].get("win_rate", 0.0)
        else:
            win_rate = 0.0

        # Get the profit value
        profit = strategy.get(profit_field, 0.0)

        rows.append(
            html.Tr(
                [
                    html.Td(i + 1),
                    html.Td(strategy_name),
                    html.Td(f"{win_rate:.1f}%"),
                    html.Td(f"${profit:.2f}"),
                ]
            )
        )

    # Add table
    return html.Div(
        [
            html.P("Top 5 Strategies by P&L", className="lead text-center"),
            html.Table(
                [
                    html.Thead(
                        html.Tr(
                            [
                                html.Th("#"),
                                html.Th("Strategy"),
                                html.Th("Win Rate"),
                                html.Th("Total P&L"),
                            ]
                        )
                    ),
                    html.Tbody(rows),
                ],
                className="table table-bordered table-hover table-sm",
            ),
        ]
    )


def create_strategy_comparison_graph(
    strategy_performance: List[Dict[str, Any]], selected_strategies: List[str]
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
        fig.update_layout(title="Select strategies to compare", template="plotly_white")
        return fig

    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(strategy_performance)

    # Filter selected strategies
    df = df[df["strategy_name"].isin(selected_strategies)]

    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No data for selected strategies", template="plotly_white"
        )
        return fig

    # Select metrics for comparison
    metrics = [
        "win_rate",
        "profit_factor",
        "total_trades",
        "total_pnl",
        "avg_profit_pct",
        "max_drawdown",
        "sharpe_ratio",
        "avg_trade_duration_hours",
    ]

    # Format for display
    metric_display = {
        "win_rate": "Win Rate (%)",
        "profit_factor": "Profit Factor",
        "total_trades": "Total Trades",
        "total_pnl": "Total P&L ($)",
        "avg_profit_pct": "Avg. Profit (%)",
        "max_drawdown": "Max Drawdown (%)",
        "sharpe_ratio": "Sharpe Ratio",
        "avg_trade_duration_hours": "Avg. Trade Duration (hrs)",
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
                if metric == "max_drawdown":
                    # Invert drawdown (lower is better)
                    max_val = df[metric].max()
                    min_val = df[metric].min()
                    df_norm[metric] = 1 - (
                        (df[metric] - min_val) / (max_val - min_val)
                        if max_val > min_val
                        else 0
                    )
                else:
                    max_val = df[metric].max()
                    min_val = df[metric].min()
                    df_norm[metric] = (
                        (df[metric] - min_val) / (max_val - min_val)
                        if max_val > min_val
                        else 0
                    )

    # Add traces for each strategy
    for _, strategy in df.iterrows():
        strategy_name = strategy["strategy_name"]
        strategy_norm = df_norm[df_norm["strategy_name"] == strategy_name].iloc[0]

        # Get values for available metrics
        available_metrics = [m for m in metrics if m in df.columns]
        values = [strategy_norm[m] for m in available_metrics]

        # Add radar trace
        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=[metric_display.get(m, m) for m in available_metrics],
                fill="toself",
                name=strategy_name,
            )
        )

    # Update layout
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        template="plotly_white",
        height=500,
        margin=dict(l=40, r=40, t=40, b=40),
    )

    return fig


def create_detailed_performance_breakdown(
    strategy_performance: List[Dict[str, Any]], selected_strategy: str = None
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
            title="No strategy performance data available", template="plotly_white"
        )
        return fig

    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(strategy_performance)

    # If no strategy is selected, use the top performer by total PnL
    if not selected_strategy or selected_strategy not in df["strategy_name"].values:
        if "total_pnl" in df.columns:
            df = df.sort_values("total_pnl", ascending=False)
            selected_strategy = df.iloc[0]["strategy_name"]
        else:
            selected_strategy = df.iloc[0]["strategy_name"]

    # Filter to selected strategy
    strategy_data = df[df["strategy_name"] == selected_strategy].iloc[0]

    # Check if we have timeframe data
    timeframe_data = strategy_data.get("timeframe_performance", {})
    if not timeframe_data:
        fig = go.Figure()
        fig.update_layout(
            title=f"No timeframe data available for {selected_strategy}",
            template="plotly_white",
        )
        return fig

    # Convert timeframe data to DataFrame
    timeframe_df = pd.DataFrame(
        [{"timeframe": tf, **metrics} for tf, metrics in timeframe_data.items()]
    )

    # Create the figure with subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Win Rate by Timeframe",
            "Profit Factor by Timeframe",
            "Average Profit % by Timeframe",
            "Number of Trades by Timeframe",
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    # Sort timeframes
    timeframe_order = {"hourly": 0, "daily": 1, "weekly": 2, "monthly": 3}

    timeframe_df["sort_order"] = timeframe_df["timeframe"].map(
        lambda x: timeframe_order.get(x, 999)
    )
    timeframe_df = timeframe_df.sort_values("sort_order")

    # Add traces
    fig.add_trace(
        go.Bar(
            x=timeframe_df["timeframe"],
            y=timeframe_df["win_rate"],
            marker_color="rgba(50, 171, 96, 0.7)",
            name="Win Rate",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=timeframe_df["timeframe"],
            y=timeframe_df["profit_factor"],
            marker_color="rgba(55, 83, 109, 0.7)",
            name="Profit Factor",
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Bar(
            x=timeframe_df["timeframe"],
            y=timeframe_df["avg_profit_pct"],
            marker_color="rgba(70, 130, 180, 0.7)",
            name="Avg Profit %",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=timeframe_df["timeframe"],
            y=timeframe_df["trade_count"],
            marker_color="rgba(214, 39, 40, 0.7)",
            name="Trade Count",
        ),
        row=2,
        col=2,
    )

    # Update layout
    fig.update_layout(
        height=600,
        title_text=f"Performance Breakdown for {selected_strategy}",
        showlegend=False,
        template="plotly_white",
        margin=dict(l=40, r=40, t=80, b=40),
    )

    return fig


def create_market_condition_performance(
    strategy_performance: List[Dict[str, Any]], selected_strategy: str = None
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
            title="No strategy performance data available", template="plotly_white"
        )
        return fig

    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(strategy_performance)

    # If no strategy is selected, use the top performer by total PnL
    if not selected_strategy or selected_strategy not in df["strategy_name"].values:
        if "total_pnl" in df.columns:
            df = df.sort_values("total_pnl", ascending=False)
            selected_strategy = df.iloc[0]["strategy_name"]
        else:
            selected_strategy = df.iloc[0]["strategy_name"]

    # Filter to selected strategy
    strategy_data = df[df["strategy_name"] == selected_strategy].iloc[0]

    # Check if we have market condition data
    market_condition_data = strategy_data.get("market_condition_performance", {})
    if not market_condition_data:
        fig = go.Figure()
        fig.update_layout(
            title=f"No market condition data available for {selected_strategy}",
            template="plotly_white",
        )
        return fig

    # Convert market condition data to DataFrame
    market_df = pd.DataFrame(
        [
            {"condition": condition, **metrics}
            for condition, metrics in market_condition_data.items()
        ]
    )

    # Create heatmap data
    conditions = market_df["condition"].tolist()
    metrics = ["win_rate", "profit_factor", "avg_profit_pct", "sharpe_ratio"]

    # Create normalized values for better heatmap visualization
    for metric in metrics:
        if metric in market_df.columns:
            max_val = market_df[metric].max()
            min_val = market_df[metric].min()
            if max_val > min_val:
                market_df[f"{metric}_normalized"] = (market_df[metric] - min_val) / (
                    max_val - min_val
                )
            else:
                market_df[f"{metric}_normalized"] = 0.5

    # Create the figure with subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Win Rate by Market Condition",
            "Profit Factor by Market Condition",
            "Avg Profit % by Market Condition",
            "Sharpe Ratio by Market Condition",
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    # Add bars for each metric
    for i, metric in enumerate(metrics):
        if metric in market_df.columns:
            row = i // 2 + 1
            col = i % 2 + 1

            fig.add_trace(
                go.Bar(
                    x=market_df["condition"],
                    y=market_df[metric],
                    marker=dict(
                        color=market_df[f"{metric}_normalized"],
                        colorscale="Viridis",
                        showscale=False,
                    ),
                    text=market_df[metric].round(2),
                    textposition="auto",
                    name=metric.replace("_", " ").title(),
                ),
                row=row,
                col=col,
            )

    # Update layout
    fig.update_layout(
        height=600,
        title_text=f"Performance by Market Condition for {selected_strategy}",
        showlegend=False,
        template="plotly_white",
        margin=dict(l=40, r=40, t=80, b=40),
    )

    return fig


def create_strategy_correlation_matrix(
    strategy_performance: List[Dict[str, Any]],
) -> go.Figure:
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
            title="Insufficient data for correlation analysis", template="plotly_white"
        )
        return fig

    # Extract daily returns if available
    strategies_with_returns = []
    strategy_names = []

    for strategy in strategy_performance:
        daily_returns = strategy.get("daily_returns", {})
        if daily_returns:
            strategies_with_returns.append(daily_returns)
            strategy_names.append(strategy["strategy_name"])

    if not strategies_with_returns:
        fig = go.Figure()
        fig.update_layout(
            title="No daily returns data available for correlation analysis",
            template="plotly_white",
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
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale="RdBu_r",
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text:.2f}",
            textfont={"size": 10},
            hoverongaps=False,
        )
    )

    fig.update_layout(
        title="Strategy Correlation Matrix",
        height=500,
        template="plotly_white",
        xaxis=dict(title="Strategy", tickangle=-45),
        yaxis=dict(title="Strategy"),
        margin=dict(l=60, r=40, t=80, b=60),
    )

    return fig
