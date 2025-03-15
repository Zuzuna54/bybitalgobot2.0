"""
Execution reporting functionality for the Algorithmic Trading System.

This module provides functions for generating trade reports, summaries,
and exporting trade history.
"""

from typing import Dict, Any, List
import pandas as pd
from loguru import logger

from src.trade_management.components.position_tracker import Trade, TradeStatus


def get_trade_summary(
    trades: Dict[str, Trade],
    active_trades: Dict[str, Trade],
    completed_trades: Dict[str, Trade]
) -> Dict[str, Any]:
    """
    Generate a summary of all trades.
    
    Args:
        trades: Dictionary of all trades by ID
        active_trades: Dictionary of active trades by ID
        completed_trades: Dictionary of completed trades by ID
        
    Returns:
        Dictionary with trade summary statistics
    """
    total_trades = len(trades)
    active_trades_count = len(active_trades)
    completed_trades_count = len(completed_trades)
    
    total_pnl = 0.0
    winning_trades = 0
    losing_trades = 0
    
    for trade in completed_trades.values():
        if trade.realized_pnl is not None:
            total_pnl += trade.realized_pnl
            
            if trade.realized_pnl > 0:
                winning_trades += 1
            elif trade.realized_pnl < 0:
                losing_trades += 1
    
    win_rate = (winning_trades / completed_trades_count) * 100 if completed_trades_count > 0 else 0
    
    # Calculate average profit and loss
    avg_profit = 0
    avg_loss = 0
    
    total_profit = 0
    total_loss = 0
    
    for trade in completed_trades.values():
        if trade.realized_pnl is not None:
            if trade.realized_pnl > 0:
                total_profit += trade.realized_pnl
            elif trade.realized_pnl < 0:
                total_loss += abs(trade.realized_pnl)
    
    if winning_trades > 0:
        avg_profit = total_profit / winning_trades
    
    if losing_trades > 0:
        avg_loss = total_loss / losing_trades
    
    # Calculate profit factor
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
    
    return {
        "total_trades": total_trades,
        "active_trades": active_trades_count,
        "completed_trades": completed_trades_count,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "avg_profit": avg_profit,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor
    }


def get_trade_history_dataframe(trades: Dict[str, Trade]) -> pd.DataFrame:
    """
    Convert trades to a DataFrame.
    
    Args:
        trades: Dictionary of trades by ID
        
    Returns:
        DataFrame with trade history
    """
    trade_data = []
    
    for trade in trades.values():
        trade_data.append(trade.to_dict())
    
    df = pd.DataFrame(trade_data)
    
    # Convert datetime string columns to datetime objects if they exist
    datetime_columns = ["entry_time", "exit_time"]
    for col in datetime_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    
    return df


def save_trade_history(trades: Dict[str, Trade], file_path: str) -> None:
    """
    Save trade history to CSV file.
    
    Args:
        trades: Dictionary of trades by ID
        file_path: Path to save the CSV file
    """
    df = get_trade_history_dataframe(trades)
    df.to_csv(file_path, index=False)
    logger.info(f"Trade history saved to {file_path}")


def get_trades_by_symbol(trades: Dict[str, Trade], symbol: str) -> List[Trade]:
    """
    Get all trades for a specific symbol.
    
    Args:
        trades: Dictionary of trades by ID
        symbol: Trading pair symbol
        
    Returns:
        List of trades for the specified symbol
    """
    return [trade for trade in trades.values() if trade.symbol == symbol]


def get_trades_by_strategy(trades: Dict[str, Trade], strategy_name: str) -> List[Trade]:
    """
    Get all trades for a specific strategy.
    
    Args:
        trades: Dictionary of trades by ID
        strategy_name: Name of the strategy
        
    Returns:
        List of trades for the specified strategy
    """
    return [trade for trade in trades.values() if trade.strategy_name == strategy_name]


def get_performance_by_strategy(trades: Dict[str, Trade]) -> Dict[str, Dict[str, Any]]:
    """
    Calculate performance metrics grouped by strategy.
    
    Args:
        trades: Dictionary of trades by ID
        
    Returns:
        Dictionary with performance metrics by strategy
    """
    # Group trades by strategy name
    strategies = {}
    
    for trade in trades.values():
        if trade.status != TradeStatus.CLOSED:
            continue
            
        strategy_name = trade.strategy_name
        
        if strategy_name not in strategies:
            strategies[strategy_name] = []
            
        strategies[strategy_name].append(trade)
    
    # Calculate metrics for each strategy
    performance = {}
    
    for strategy_name, strategy_trades in strategies.items():
        total_trades = len(strategy_trades)
        total_pnl = sum(trade.realized_pnl for trade in strategy_trades if trade.realized_pnl is not None)
        winning_trades = sum(1 for trade in strategy_trades if trade.realized_pnl is not None and trade.realized_pnl > 0)
        
        if total_trades > 0:
            win_rate = (winning_trades / total_trades) * 100
        else:
            win_rate = 0
            
        performance[strategy_name] = {
            "total_trades": total_trades,
            "total_pnl": total_pnl,
            "winning_trades": winning_trades,
            "win_rate": win_rate
        }
    
    return performance 