"""
Order Manager Components

This module provides components for displaying and managing pending orders.
"""

from typing import Dict, Any, List
import pandas as pd
from dash import html, dash_table


def render_pending_orders_table(pending_orders: List[Dict[str, Any]]) -> html.Div:
    """
    Render the pending orders table.
    
    Args:
        pending_orders: List of pending order dictionaries
        
    Returns:
        HTML Div containing the pending orders table
    """
    if not pending_orders:
        return html.Div("No pending orders", className="no-data-message")
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(pending_orders)
    
    # Rename columns for display
    column_map = {
        "id": "ID",
        "symbol": "Symbol",
        "type": "Type",
        "side": "Side",
        "quantity": "Quantity",
        "price": "Price",
        "status": "Status",
        "create_time": "Created At",
        "order_type": "Order Type",
        "time_in_force": "Time In Force"
    }
    
    # Format columns
    if 'create_time' in df.columns:
        df['create_time'] = pd.to_datetime(df['create_time']).dt.strftime('%Y-%m-%d %H:%M')
    
    # Rename columns
    df = df.rename(columns=column_map)
    
    # Create table
    table = dash_table.DataTable(
        id='pending-orders-table',
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
                    'filter_query': '{Side} contains "Buy"'
                },
                'backgroundColor': 'rgba(188, 223, 204, 0.1)'
            },
            {
                'if': {
                    'filter_query': '{Side} contains "Sell"'
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
        html.H4(f"Pending Orders ({len(df)})", className="table-title"),
        table
    ]) 