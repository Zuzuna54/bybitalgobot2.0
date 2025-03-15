"""
Performance Panel Component for the Trading Dashboard

This module provides visualization components for displaying performance metrics,
equity curves, and other performance-related data.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, callback
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta


def create_performance_panel() -> html.Div:
    """
    Create the performance panel layout.
    
    Returns:
        Dash HTML Div containing the performance panel
    """
    return html.Div([
        html.H2("Performance Metrics", className="panel-header"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Key Metrics"),
                    dbc.CardBody(id="performance-metrics-card")
                ])
            ], width=5),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Equity Curve"),
                    dbc.CardBody([
                        dcc.Graph(id="equity-curve-graph")
                    ])
                ])
            ], width=7)
        ]),
        
        html.Br(),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Drawdown Analysis"),
                    dbc.CardBody([
                        dcc.Graph(id="drawdown-graph")
                    ])
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Return Distribution"),
                    dbc.CardBody([
                        dcc.Graph(id="return-distribution-graph")
                    ])
                ])
            ], width=6)
        ]),
        
        html.Br(),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Daily Performance"),
                    dbc.CardBody([
                        dcc.Graph(id="daily-performance-graph")
                    ])
                ])
            ], width=12)
        ]),
        
        dcc.Interval(
            id="performance-update-interval",
            interval=30 * 1000,  # 30 seconds in milliseconds
            n_intervals=0
        )
    ], id="performance-panel")


def render_metrics_card(metrics: Dict[str, Any]) -> html.Div:
    """
    Render the performance metrics card content.
    
    Args:
        metrics: Dictionary containing performance metrics
        
    Returns:
        HTML Div with formatted metrics
    """
    if not metrics:
        return html.Div("No performance data available")
    
    # Format metrics for display
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H4(f"{metrics.get('total_return', 0):.2f}%", className="metric-value"),
                html.P("Total Return", className="metric-label")
            ], width=4),
            dbc.Col([
                html.H4(f"{metrics.get('win_rate', 0) * 100:.1f}%", className="metric-value"),
                html.P("Win Rate", className="metric-label")
            ], width=4),
            dbc.Col([
                html.H4(f"{metrics.get('profit_factor', 0):.2f}", className="metric-value"),
                html.P("Profit Factor", className="metric-label")
            ], width=4)
        ]),
        
        html.Br(),
        
        dbc.Row([
            dbc.Col([
                html.H4(f"{metrics.get('max_drawdown', 0):.2f}%", className="metric-value-secondary"),
                html.P("Max Drawdown", className="metric-label")
            ], width=4),
            dbc.Col([
                html.H4(f"{metrics.get('sharpe_ratio', 0):.2f}", className="metric-value-secondary"),
                html.P("Sharpe Ratio", className="metric-label")
            ], width=4),
            dbc.Col([
                html.H4(f"{metrics.get('risk_reward_ratio', 0):.2f}", className="metric-value-secondary"),
                html.P("Risk/Reward", className="metric-label")
            ], width=4)
        ]),
        
        html.Br(),
        
        dbc.Row([
            dbc.Col([
                html.H4(f"{metrics.get('total_trades', 0)}", className="metric-value-tertiary"),
                html.P("Total Trades", className="metric-label")
            ], width=4),
            dbc.Col([
                html.H4(f"{metrics.get('avg_trade_duration', '0d')}", className="metric-value-tertiary"),
                html.P("Avg Duration", className="metric-label")
            ], width=4),
            dbc.Col([
                html.H4(f"${metrics.get('pnl_per_day', 0):.2f}", className="metric-value-tertiary"),
                html.P("Daily P&L", className="metric-label")
            ], width=4)
        ])
    ])


def create_equity_curve_graph(equity_data: pd.DataFrame) -> go.Figure:
    """
    Create the equity curve graph.
    
    Args:
        equity_data: DataFrame with equity curve data
        
    Returns:
        Plotly figure object
    """
    if equity_data is None or equity_data.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No equity data available",
            template="plotly_dark"
        )
        return fig
    
    fig = go.Figure()
    
    # Add equity curve line
    fig.add_trace(go.Scatter(
        x=equity_data.index,
        y=equity_data['equity'],
        mode='lines',
        name='Total Equity',
        line=dict(color='#17becf', width=2)
    ))
    
    # Add balance line (without unrealized PnL)
    if 'balance' in equity_data.columns:
        fig.add_trace(go.Scatter(
            x=equity_data.index,
            y=equity_data['balance'],
            mode='lines',
            name='Balance',
            line=dict(color='#7f7f7f', width=1, dash='dash')
        ))
    
    # Format the layout
    fig.update_layout(
        title="Equity Curve",
        xaxis_title="Date",
        yaxis_title="Value ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=400,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    return fig


def create_drawdown_graph(equity_data: pd.DataFrame) -> go.Figure:
    """
    Create the drawdown analysis graph.
    
    Args:
        equity_data: DataFrame with equity curve data
        
    Returns:
        Plotly figure object
    """
    if equity_data is None or equity_data.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No drawdown data available",
            template="plotly_dark"
        )
        return fig
    
    # Calculate drawdown
    equity_data = equity_data.copy()
    equity_data['peak'] = equity_data['equity'].cummax()
    equity_data['drawdown'] = -(equity_data['equity'] - equity_data['peak']) / equity_data['peak'] * 100
    
    fig = go.Figure()
    
    # Add drawdown area
    fig.add_trace(go.Scatter(
        x=equity_data.index,
        y=equity_data['drawdown'],
        fill='tozeroy',
        name='Drawdown',
        line=dict(color='rgba(220, 53, 69, 0.8)', width=1)
    ))
    
    # Format the layout
    fig.update_layout(
        title="Drawdown Analysis",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        yaxis=dict(autorange="reversed"),  # Invert y-axis to show drawdowns going down
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=300,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    return fig


def create_return_distribution_graph(daily_returns: pd.DataFrame) -> go.Figure:
    """
    Create the return distribution graph.
    
    Args:
        daily_returns: DataFrame with daily returns data
        
    Returns:
        Plotly figure object
    """
    if daily_returns is None or daily_returns.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No return data available",
            template="plotly_dark"
        )
        return fig
    
    returns = daily_returns['return_pct'].dropna()
    
    fig = go.Figure()
    
    # Add histogram for return distribution
    fig.add_trace(go.Histogram(
        x=returns,
        name='Daily Returns',
        marker=dict(color='rgba(50, 171, 96, 0.7)'),
        xbins=dict(size=0.5),
        autobinx=False
    ))
    
    # Format the layout
    fig.update_layout(
        title="Daily Return Distribution",
        xaxis_title="Return (%)",
        yaxis_title="Frequency",
        template="plotly_white",
        height=300,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    return fig


def create_daily_performance_graph(daily_summary: pd.DataFrame) -> go.Figure:
    """
    Create the daily performance graph.
    
    Args:
        daily_summary: DataFrame with daily performance data
        
    Returns:
        Plotly figure object
    """
    if daily_summary is None or daily_summary.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No daily performance data available",
            template="plotly_dark"
        )
        return fig
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Daily P&L", "Trade Count"),
        row_heights=[0.7, 0.3]
    )
    
    # Add daily P&L bars
    fig.add_trace(
        go.Bar(
            x=daily_summary.index,
            y=daily_summary['daily_pnl'],
            name='Daily P&L',
            marker=dict(
                color=daily_summary['daily_pnl'].apply(
                    lambda x: 'rgba(50, 171, 96, 0.7)' if x >= 0 else 'rgba(220, 53, 69, 0.7)'
                )
            )
        ),
        row=1, col=1
    )
    
    # Add trade count line
    fig.add_trace(
        go.Scatter(
            x=daily_summary.index,
            y=daily_summary['trade_count'],
            mode='lines',
            name='Trade Count',
            line=dict(color='#17becf', width=2)
        ),
        row=2, col=1
    )
    
    # Format the layout
    fig.update_layout(
        xaxis2_title="Date",
        yaxis_title="P&L ($)",
        yaxis2_title="Trade Count",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=400,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    return fig


# Define callbacks to update the performance panel components
def register_performance_callbacks(app: dash.Dash, get_performance_data_func) -> None:
    """
    Register the callbacks for the performance panel.
    
    Args:
        app: Dash application instance
        get_performance_data_func: Function to get performance data
    """
    @app.callback(
        [
            Output("performance-metrics-card", "children"),
            Output("equity-curve-graph", "figure"),
            Output("drawdown-graph", "figure"),
            Output("return-distribution-graph", "figure"),
            Output("daily-performance-graph", "figure")
        ],
        [Input("performance-update-interval", "n_intervals")]
    )
    def update_performance_panel(n_intervals):
        # Get performance data
        performance_data = get_performance_data_func()
        
        if not performance_data:
            # Return empty/default components if no data
            empty_fig = go.Figure()
            empty_fig.update_layout(template="plotly_white")
            return (
                html.Div("No performance data available"),
                empty_fig, empty_fig, empty_fig, empty_fig
            )
        
        # Get metrics and data
        metrics = performance_data.get("metrics", {})
        
        # Process equity curve data
        equity_curve_data = performance_data.get("equity_curve", None)
        if equity_curve_data:
            equity_df = pd.DataFrame(equity_curve_data)
            equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
            equity_df.set_index('timestamp', inplace=True)
        else:
            equity_df = pd.DataFrame()
        
        # Process daily summary data
        daily_summary_data = performance_data.get("daily_summary", None)
        if daily_summary_data:
            daily_df = pd.DataFrame(daily_summary_data)
            daily_df['date'] = pd.to_datetime(daily_df['date'])
            daily_df.set_index('date', inplace=True)
        else:
            daily_df = pd.DataFrame()
        
        # Create components
        metrics_card = render_metrics_card(metrics)
        equity_graph = create_equity_curve_graph(equity_df)
        drawdown_graph = create_drawdown_graph(equity_df)
        return_graph = create_return_distribution_graph(daily_df)
        daily_graph = create_daily_performance_graph(daily_df)
        
        return metrics_card, equity_graph, drawdown_graph, return_graph, daily_graph 