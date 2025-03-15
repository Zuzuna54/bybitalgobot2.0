"""
Paper Trading Simulator for the Algorithmic Trading System

This module provides functionality to simulate trading in real-time market conditions
without using real funds. It tracks simulated positions, portfolio value, and 
performance metrics.
"""

# Re-export the refactored PaperTradingSimulator
from src.paper_trading.simulator import PaperTradingSimulator

__all__ = ["PaperTradingSimulator"] 