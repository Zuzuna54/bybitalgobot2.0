"""
Position Display Components

This module provides components for displaying active trades and trade history.
"""

from typing import Dict, Any, List
import pandas as pd
from dash import html, dash_table


def render_active_trades_table(active_trades: List[Dict[str, Any]]) -> html.Div:
    """
    Render the active trades table.
    
    Args:
        active_trades: List of active trade dictionaries
        
    Returns:
        HTML Div containing the active trades table
    """
    if not active_trades:
        return html.Div("No active trades", className="no-data-message")
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(active_trades)
    
    # Rename columns for display
    column_map = {
        "id": "ID",
        "symbol": "Symbol",
        "type": "Type",
        "entry_price": "Entry Price",
        "current_price": "Current Price",
        "size": "Size",
        "unrealized_pnl": "Unrealized P&L",
        "unrealized_pnl_pct": "P&L %",
        "entry_time": "Entry Time",
        "stop_loss": "Stop Loss",
        "take_profit": "Take Profit",
        "strategy_name": "Strategy"
    }
    
    # Format columns
    if 'entry_time' in df.columns:
        df['entry_time'] = pd.to_datetime(df['entry_time']).dt.strftime('%Y-%m-%d %H:%M')
    
    if 'unrealized_pnl' in df.columns:
        df['unrealized_pnl'] = df['unrealized_pnl'].apply(lambda x: f"${x:.2f}")
    
    if 'unrealized_pnl_pct' in df.columns:
        df['unrealized_pnl_pct'] = df['unrealized_pnl_pct'].apply(lambda x: f"{x:.2f}%")
    
    # Rename columns
    df = df.rename(columns=column_map)
    
    # Create table
    table = dash_table.DataTable(
        id='active-trades-table',
        columns=[{"name": column_map.get(i, i), "id": i} for i in df.columns],
        data=df.to_dict('records'),
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'left',
            'padding': '10px',
            'fontFamily': 'Arial',
            'fontSize': '14px'
        },
        style_header={
            'backgroundColor': '#f8f9fa',
            'fontWeight': 'bold',
            'borderBottom': '1px solid #dee2e6'
        },
        style_data_conditional=[
            {
                'if': {
                    'filter_query': '{Type} contains "long"'
                },
                'backgroundColor': 'rgba(188, 223, 204, 0.1)'
            },
            {
                'if': {
                    'filter_query': '{Type} contains "short"'
                },
                'backgroundColor': 'rgba(255, 200, 200, 0.1)'
            }
        ],
        sort_action="native",
        sort_mode="multi",
        filter_action="native",
        page_size=10
    )
    
    return html.Div([
        html.H4(f"Active Trades ({len(df)})", className="table-title"),
        table
    ])


def render_trade_history_table(trade_history: List[Dict[str, Any]]) -> html.Div:
    """
    Render the trade history table.
    
    Args:
        trade_history: List of completed trade dictionaries
        
    Returns:
        HTML Div containing the trade history table
    """
    if not trade_history:
        return html.Div("No trade history", className="no-data-message")
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(trade_history)
    
    # Rename columns for display
    column_map = {
        "id": "ID",
        "symbol": "Symbol",
        "side": "Side",
        "entry_price": "Entry Price",
        "exit_price": "Exit Price",
        "position_size": "Size",
        "realized_pnl": "Realized P&L",
        "realized_pnl_percent": "P&L %",
        "entry_time": "Entry Time",
        "exit_time": "Exit Time",
        "exit_reason": "Exit Reason",
        "strategy_name": "Strategy"
    }
    
    # Format columns
    if 'entry_time' in df.columns:
        df['entry_time'] = pd.to_datetime(df['entry_time']).dt.strftime('%Y-%m-%d %H:%M')
    
    if 'exit_time' in df.columns:
        df['exit_time'] = pd.to_datetime(df['exit_time']).dt.strftime('%Y-%m-%d %H:%M')
    
    if 'realized_pnl' in df.columns:
        df['realized_pnl'] = df['realized_pnl'].apply(lambda x: f"${x:.2f}" if x is not None else "")
    
    if 'realized_pnl_percent' in df.columns:
        df['realized_pnl_percent'] = df['realized_pnl_percent'].apply(lambda x: f"{x:.2f}%" if x is not None else "")
    
    # Rename columns
    df = df.rename(columns=column_map)
    
    # Create table
    table = dash_table.DataTable(
        id='trade-history-table',
        columns=[{"name": column_map.get(i, i), "id": i} for i in df.columns],
        data=df.to_dict('records'),
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'left',
            'padding': '10px',
            'fontFamily': 'Arial',
            'fontSize': '14px'
        },
        style_header={
            'backgroundColor': '#f8f9fa',
            'fontWeight': 'bold',
            'borderBottom': '1px solid #dee2e6'
        },
        style_data_conditional=[
            {
                'if': {
                    'filter_query': '{Realized P&L} contains "-"'
                },
                'color': 'red'
            },
            {
                'if': {
                    'filter_query': '{Realized P&L} contains "$" && !({Realized P&L} contains "-")'
                },
                'color': 'green'
            }
        ],
        sort_action="native",
        sort_mode="multi",
        filter_action="native",
        page_size=10
    )
    
    return html.Div([
        html.H4(f"Trade History ({len(df)})", className="table-title"),
        table
    ]) 