"""
Trading Panel Callbacks

This module provides dashboard callbacks and visualization functions for the trading panel.
"""

from typing import Dict, Any, List, Callable
import dash
import pandas as pd
import plotly.graph_objects as go
from dash.dependencies import Input, Output
from dash import html

from src.dashboard.components.trading.position_display import render_active_trades_table, render_trade_history_table
from src.dashboard.components.trading.order_manager import render_pending_orders_table


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
    
    # Format the layout
    fig.update_layout(
        title="P&L by Symbol",
        xaxis_title="Symbol",
        yaxis_title="P&L ($)",
        template="plotly_white",
        height=350,
        margin=dict(l=40, r=40, t=40, b=40)
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
    
    # Format the layout
    fig.update_layout(
        title="Win/Loss by Strategy",
        xaxis_title="Strategy",
        yaxis_title="Trade Count",
        barmode='stack',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=350,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    return fig


def register_trading_callbacks(app: dash.Dash, get_trade_data_func: Callable) -> None:
    """
    Register the callbacks for the trading panel.
    
    Args:
        app: Dash application instance
        get_trade_data_func: Function to get trade data
    """
    @app.callback(
        [
            Output("active-trades-content", "children"),
            Output("pending-orders-content", "children"),
            Output("trade-history-content", "children"),
            Output("pnl-by-symbol-graph", "figure"),
            Output("win-loss-by-strategy-graph", "figure")
        ],
        [Input("trading-update-interval", "n_intervals")]
    )
    def update_trading_panel(n_intervals):
        # Get trade data
        trade_data = get_trade_data_func()
        
        if not trade_data:
            # Return empty/default components if no data
            empty_fig = go.Figure()
            empty_fig.update_layout(template="plotly_white")
            return (
                html.Div("No active trades data available", className="no-data-message"),
                html.Div("No pending orders data available", className="no-data-message"),
                html.Div("No trade history data available", className="no-data-message"),
                empty_fig, empty_fig
            )
        
        # Get data
        active_trades = trade_data.get("active_trades", [])
        pending_orders = trade_data.get("pending_orders", [])
        completed_trades = trade_data.get("completed_trades", [])
        
        # Create components
        active_trades_table = render_active_trades_table(active_trades)
        pending_orders_table = render_pending_orders_table(pending_orders)
        trade_history_table = render_trade_history_table(completed_trades)
        pnl_graph = create_pnl_by_symbol_graph(completed_trades)
        win_loss_graph = create_win_loss_by_strategy_graph(completed_trades)
        
        return active_trades_table, pending_orders_table, trade_history_table, pnl_graph, win_loss_graph 