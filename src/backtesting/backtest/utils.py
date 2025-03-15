"""
Utility functions for the backtesting engine.
"""

from typing import Dict, Any
import pandas as pd


def timeframe_to_minutes(timeframe: str) -> int:
    """
    Convert timeframe string to minutes.
    
    Args:
        timeframe: Timeframe string (1m, 5m, 15m, 1h, 4h, 1d)
        
    Returns:
        Timeframe in minutes
    """
    if timeframe.endswith('m'):
        return int(timeframe[:-1])
    elif timeframe.endswith('h'):
        return int(timeframe[:-1]) * 60
    elif timeframe.endswith('d'):
        return int(timeframe[:-1]) * 60 * 24
    return 60  # Default to 1 hour


def timeframe_to_pandas_freq(timeframe: str) -> str:
    """
    Convert timeframe string to pandas frequency string.
    
    Args:
        timeframe: Timeframe string (1m, 5m, 15m, 1h, 4h, 1d)
        
    Returns:
        Pandas frequency string
    """
    if timeframe.endswith('m'):
        return f"{timeframe[:-1]}T"
    elif timeframe.endswith('h'):
        return f"{timeframe[:-1]}H"
    elif timeframe.endswith('d'):
        return f"{timeframe[:-1]}D"
    return "1H"  # Default to 1 hour


def calculate_unrealized_pnl_pct(position: Dict[str, Any], current_price: float) -> float:
    """
    Calculate unrealized profit/loss percentage.
    
    Args:
        position: Position dictionary
        current_price: Current market price
        
    Returns:
        Unrealized PnL percentage
    """
    is_long = position["type"] == "long"
    price_diff = current_price - position["entry_price"] if is_long else position["entry_price"] - current_price
    return (price_diff / position["entry_price"]) * 100 