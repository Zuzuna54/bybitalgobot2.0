"""
Backtesting Engine for the Algorithmic Trading System

This module provides functionality for backtesting trading strategies on historical data.
"""

# Re-export BacktestEngine from the refactored module
from src.backtesting.backtest import BacktestEngine

__all__ = ["BacktestEngine"]
