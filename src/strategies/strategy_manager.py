"""
Strategy Manager for the Algorithmic Trading System

This module manages multiple trading strategies, handles strategy selection,
signal aggregation, and strategy performance tracking.

It has been refactored into smaller components in the src/strategies/manager/ package
and now re-exports the StrategyManager class from there.
"""

# Re-export the StrategyManager class from the manager package
from src.strategies.manager.core import StrategyManager

# Re-export important functions that might be used elsewhere
from src.strategies.manager.loader import get_available_strategies
from src.strategies.manager.optimization import select_top_strategies

# Define the public API
__all__ = ['StrategyManager', 'get_available_strategies', 'select_top_strategies'] 