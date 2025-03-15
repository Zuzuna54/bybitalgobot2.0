"""
Trade Management Module for the Algorithmic Trading System

This module handles order execution, position management, and trade tracking.
It re-exports the refactored TradeManager class from the manager module.
"""

# Re-export TradeManager class from the refactored module
from src.trade_management.manager import TradeManager

# Re-export components that might be used directly
from src.trade_management.components import (
    OrderType, 
    OrderSide, 
    TradeStatus,
    Trade
)

__all__ = [
    'TradeManager',
    'OrderType',
    'OrderSide',
    'TradeStatus',
    'Trade'
] 