"""
Strategy Signals View

This module provides visualization components for displaying strategy signals and metrics.
"""

from typing import Dict, Any, List
import pandas as pd
from dash import html
from datetime import datetime


def render_recent_signals_table(signals_data: List[Dict[str, Any]]) -> html.Div:
    """
    Render the recent strategy signals table.
    
    Args:
        signals_data: List of strategy signal dictionaries
        
    Returns:
        HTML Div containing the signals table
    """
    if not signals_data:
        return html.Div("No recent strategy signals available")
    
    # Convert to DataFrame and sort by timestamp (most recent first)
    df = pd.DataFrame(signals_data)
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp', ascending=False)
    
    # Take most recent 10 signals
    recent_signals = df.head(10)
    
    # Create table rows
    rows = []
    for _, signal in recent_signals.iterrows():
        # Format timestamp
        timestamp = signal.get('timestamp', '')
        if isinstance(timestamp, (int, float)):
            # Convert unix timestamp to datetime
            formatted_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M')
        elif isinstance(timestamp, str):
            # Try to parse string timestamp
            try:
                formatted_time = datetime.fromisoformat(timestamp).strftime('%Y-%m-%d %H:%M')
            except (ValueError, TypeError):
                formatted_time = timestamp
        else:
            formatted_time = str(timestamp)
        
        # Format direction with color
        direction = signal.get('direction', '')
        if direction.lower() == 'buy':
            direction_cell = html.Td(direction, style={'color': 'green'})
        elif direction.lower() == 'sell':
            direction_cell = html.Td(direction, style={'color': 'red'})
        else:
            direction_cell = html.Td(direction)
        
        # Create the row
        rows.append(
            html.Tr([
                html.Td(signal.get('strategy_name', '')),
                html.Td(signal.get('symbol', '')),
                direction_cell,
                html.Td(f"{signal.get('price', ''):.2f}" if signal.get('price', '') else ''),
                html.Td(formatted_time)
            ])
        )
    
    # Add table
    return html.Div([
        html.P("Recent Strategy Signals", className="lead text-center"),
        html.Table(
            [
                html.Thead(
                    html.Tr([
                        html.Th("Strategy"),
                        html.Th("Symbol"),
                        html.Th("Direction"),
                        html.Th("Price"),
                        html.Th("Time")
                    ])
                ),
                html.Tbody(rows)
            ],
            className="table table-bordered table-hover table-sm"
        )
    ])


def format_signal_data(signals_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Format signal data for better display.
    
    Args:
        signals_data: List of strategy signal dictionaries
        
    Returns:
        Formatted signal data list
    """
    if not signals_data:
        return []
    
    formatted_data = []
    for signal in signals_data:
        # Format timestamp
        timestamp = signal.get('timestamp', '')
        if isinstance(timestamp, (int, float)):
            formatted_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M')
        elif isinstance(timestamp, str):
            try:
                formatted_time = datetime.fromisoformat(timestamp).strftime('%Y-%m-%d %H:%M')
            except (ValueError, TypeError):
                formatted_time = timestamp
        else:
            formatted_time = str(timestamp)
        
        # Format price
        price = signal.get('price', '')
        if isinstance(price, (int, float)):
            formatted_price = f"{price:.2f}"
        else:
            formatted_price = str(price)
        
        # Create formatted signal
        formatted_signal = {
            **signal,
            'formatted_time': formatted_time,
            'formatted_price': formatted_price
        }
        
        formatted_data.append(formatted_signal)
    
    return formatted_data 