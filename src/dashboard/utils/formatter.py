"""
Dashboard Formatter Module

This module provides utilities for formatting data displayed in the dashboard.
"""

from typing import Dict, Any, List, Union, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


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


def format_timestamp(timestamp: Union[datetime, str, float, int], 
                     format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format a timestamp in a human-readable format.
    
    Args:
        timestamp: The timestamp to format (datetime, string, or epoch time)
        format_str: The format string to use
        
    Returns:
        Formatted timestamp string
    """
    if isinstance(timestamp, str):
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except ValueError:
            # Try other common formats
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
                try:
                    dt = datetime.strptime(timestamp, fmt)
                    break
                except ValueError:
                    continue
            else:
                return timestamp  # Return original if parsing fails
    elif isinstance(timestamp, (int, float)):
        # Assume epoch time in seconds if it's a reasonable value (year 1970-2100)
        if 0 <= timestamp <= 4102444800:  # Jan 1, 1970 to Jan 1, 2100
            dt = datetime.fromtimestamp(timestamp)
        # Assume milliseconds if it's a large value
        elif 946684800000 <= timestamp <= 4102444800000:  # Jan 1, 2000 to Jan 1, 2100
            dt = datetime.fromtimestamp(timestamp / 1000)
        else:
            return str(timestamp)  # Return original if value is unreasonable
    elif isinstance(timestamp, datetime):
        dt = timestamp
    else:
        return str(timestamp)  # Return original if type is not supported
    
    return dt.strftime(format_str)


def format_time_ago(timestamp: Union[datetime, str, float, int]) -> str:
    """
    Format a timestamp as a relative time (e.g., "5 minutes ago").
    
    Args:
        timestamp: The timestamp to format (datetime, string, or epoch time)
        
    Returns:
        Relative time string
    """
    # Convert to datetime if needed
    if isinstance(timestamp, str):
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except ValueError:
            try:
                dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                return timestamp  # Return original if parsing fails
    elif isinstance(timestamp, (int, float)):
        # Assume epoch time in seconds if it's a reasonable value (year 1970-2100)
        if 0 <= timestamp <= 4102444800:  # Jan 1, 1970 to Jan 1, 2100
            dt = datetime.fromtimestamp(timestamp)
        # Assume milliseconds if it's a large value
        elif 946684800000 <= timestamp <= 4102444800000:  # Jan 1, 2000 to Jan 1, 2100
            dt = datetime.fromtimestamp(timestamp / 1000)
        else:
            return str(timestamp)  # Return original if value is unreasonable
    elif isinstance(timestamp, datetime):
        dt = timestamp
    else:
        return str(timestamp)  # Return original if type is not supported
    
    # Calculate the time difference
    now = datetime.now()
    diff = now - dt
    
    # Format based on the time difference
    if diff.total_seconds() < 60:
        return "just now"
    elif diff.total_seconds() < 3600:
        minutes = int(diff.total_seconds() / 60)
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    elif diff.total_seconds() < 86400:
        hours = int(diff.total_seconds() / 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    elif diff.total_seconds() < 604800:
        days = int(diff.total_seconds() / 86400)
        return f"{days} day{'s' if days != 1 else ''} ago"
    else:
        return dt.strftime("%Y-%m-%d")


def format_duration(seconds: float) -> str:
    """
    Format a duration in seconds as a human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    # Calculate days, hours, minutes, seconds
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Format based on the duration
    if days > 0:
        return f"{int(days)}d {int(hours)}h {int(minutes)}m"
    elif hours > 0:
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    elif minutes > 0:
        return f"{int(minutes)}m {int(seconds)}s"
    else:
        return f"{int(seconds)}s"


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