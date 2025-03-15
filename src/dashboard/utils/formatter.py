"""
Dashboard Formatter Module

This module provides utilities for formatting data displayed in the dashboard.
"""

from typing import Dict, Any, List, Union, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.dashboard.utils.time_utils import format_timestamp, format_time_ago, format_duration


def format_currency(value: float, precision: int = 2, currency: str = "$") -> str:
    """
    Format a number as currency.
    
    Args:
        value: The numeric value to format
        precision: Number of decimal places
        currency: Currency symbol
        
    Returns:
        Formatted currency string
    """
    return f"{currency}{value:,.{precision}f}"


def format_percentage(value: float, precision: int = 2, include_sign: bool = False) -> str:
    """
    Format a number as a percentage.
    
    Args:
        value: The numeric value to format (as a decimal, e.g., 0.05 for 5%)
        precision: Number of decimal places
        include_sign: Whether to include a + sign for positive values
        
    Returns:
        Formatted percentage string
    """
    # Convert to percentage (multiply by 100)
    percentage = value * 100 if abs(value) < 10 else value
    
    # Add sign if requested
    if include_sign and percentage > 0:
        return f"+{percentage:,.{precision}f}%"
    
    return f"{percentage:,.{precision}f}%"


# format_timestamp, format_time_ago, and format_duration functions are now imported from time_utils


def format_trade_size(size: float, asset: str = "", precision: int = 8) -> str:
    """
    Format a trade size with appropriate precision based on the asset.
    
    Args:
        size: The trade size
        asset: The asset symbol
        precision: Default decimal precision
        
    Returns:
        Formatted trade size string
    """
    # Different assets may need different precision
    if asset.upper().endswith("BTC"):
        precision = 8
    elif asset.upper().endswith("ETH"):
        precision = 6
    elif asset.upper().endswith("USDT") or asset.upper().endswith("USD"):
        precision = 2
    
    # Handle very small values
    if 0 < abs(size) < 0.001:
        return f"{size:.8f} {asset}".rstrip("0").rstrip(".")
    
    return f"{size:,.{precision}f} {asset}".rstrip("0").rstrip(".")


def format_number_compact(value: float, precision: int = 2) -> str:
    """
    Format a number in a compact way (e.g., 1.5K, 2.3M).
    
    Args:
        value: The numeric value to format
        precision: Number of decimal places
        
    Returns:
        Formatted compact number string
    """
    abs_value = abs(value)
    
    if abs_value < 1000:
        return f"{value:,.{precision}f}"
    elif abs_value < 1000000:
        return f"{value/1000:,.{precision}f}K"
    elif abs_value < 1000000000:
        return f"{value/1000000:,.{precision}f}M"
    else:
        return f"{value/1000000000:,.{precision}f}B"


def format_symbol(symbol: str) -> str:
    """
    Format a trading symbol for display.
    
    Args:
        symbol: The trading symbol (e.g., BTCUSDT)
        
    Returns:
        Formatted symbol string
    """
    # Try to split into base and quote currencies
    if symbol.endswith("USDT"):
        base = symbol[:-4]
        quote = "USDT"
    elif symbol.endswith("USD"):
        base = symbol[:-3]
        quote = "USD"
    elif symbol.endswith("BUSD"):
        base = symbol[:-4]
        quote = "BUSD"
    else:
        # Default to 3-character quote currency
        base = symbol[:-3]
        quote = symbol[-3:]
    
    return f"{base}/{quote}"


def format_dataframe_to_dict(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Convert a pandas DataFrame to a list of dictionaries for JSON serialization.
    
    Args:
        df: The DataFrame to convert
        
    Returns:
        List of dictionaries representing the DataFrame rows
    """
    if df is None or df.empty:
        return []
    
    # Convert DataFrame to dict, handling special cases
    result = []
    
    for _, row in df.iterrows():
        row_dict = {}
        
        for column, value in row.items():
            # Handle NaN values
            if pd.isna(value):
                row_dict[column] = None
            # Handle datetime values
            elif isinstance(value, (pd.Timestamp, datetime)):
                row_dict[column] = value.isoformat()
            # Handle numpy data types
            elif isinstance(value, (np.integer, np.floating)):
                row_dict[column] = float(value) if isinstance(value, np.floating) else int(value)
            # Handle other values
            else:
                row_dict[column] = value
        
        result.append(row_dict)
    
    return result 