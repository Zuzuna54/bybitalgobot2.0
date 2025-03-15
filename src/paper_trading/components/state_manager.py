"""
State management functionality for the paper trading simulator.

This module provides functions for saving and loading the state of the paper trading system.
"""

import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd
from loguru import logger


def save_state(
    data_dir: str,
    current_balance: float,
    active_positions: Dict[str, Dict[str, Any]],
    pending_orders: Dict[str, Dict[str, Any]],
    trade_history: List[Dict[str, Any]],
    equity_history: List[Dict[str, Any]]
) -> bool:
    """
    Save current state to disk.
    
    Args:
        data_dir: Directory to store data
        current_balance: Current account balance
        active_positions: Dictionary of active positions
        pending_orders: Dictionary of pending orders
        trade_history: List of all trades
        equity_history: List of equity history points
        
    Returns:
        True if successful, False otherwise
    """
    state = {
        "current_balance": current_balance,
        "active_positions": active_positions,
        "pending_orders": pending_orders,
        "trade_history": trade_history,
        "last_update": datetime.now().isoformat()
    }
    
    try:
        with open(os.path.join(data_dir, "simulator_state.json"), "w") as f:
            json.dump(state, f, default=json_serializer, indent=2)
            
        # Save equity history to CSV
        if equity_history:
            pd.DataFrame(equity_history).to_csv(
                os.path.join(data_dir, "equity_history.csv"),
                index=False
            )
            
        logger.debug("Paper trading state saved successfully")
        return True
    except Exception as e:
        logger.error(f"Error saving paper trading state: {e}")
        return False


def load_state(
    data_dir: str,
    initial_balance: float
) -> Dict[str, Any]:
    """
    Load saved state from disk if available.
    
    Args:
        data_dir: Directory where data is stored
        initial_balance: Initial account balance to use if no state is found
        
    Returns:
        Dictionary with loaded state or defaults
    """
    state_file = os.path.join(data_dir, "simulator_state.json")
    
    # Initialize default state
    state = {
        "current_balance": initial_balance,
        "active_positions": {},
        "pending_orders": {},
        "trade_history": [],
        "equity_history": []
    }
    
    if not os.path.exists(state_file):
        return state
    
    try:
        with open(state_file, "r") as f:
            loaded_state = json.load(f)
        
        state["current_balance"] = loaded_state.get("current_balance", initial_balance)
        state["active_positions"] = loaded_state.get("active_positions", {})
        state["pending_orders"] = loaded_state.get("pending_orders", {})
        state["trade_history"] = loaded_state.get("trade_history", [])
        
        # Convert string timestamps back to datetime objects
        for position in state["active_positions"].values():
            position["entry_time"] = datetime.fromisoformat(position["entry_time"])
        
        for trade in state["trade_history"]:
            trade["entry_time"] = datetime.fromisoformat(trade["entry_time"])
            if trade.get("exit_time"):
                trade["exit_time"] = datetime.fromisoformat(trade["exit_time"])
        
        # Load equity history from CSV if available
        equity_file = os.path.join(data_dir, "equity_history.csv")
        if os.path.exists(equity_file):
            df = pd.read_csv(equity_file)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            state["equity_history"] = df.to_dict("records")
        
        logger.info(f"Paper trading state loaded successfully with balance: ${state['current_balance']}")
    except Exception as e:
        logger.error(f"Error loading paper trading state: {e}")
    
    return state


def json_serializer(obj: Any) -> str:
    """
    Helper function to serialize datetime objects in JSON.
    
    Args:
        obj: Object to serialize
        
    Returns:
        Serialized string representation
        
    Raises:
        TypeError: If the object type is not serializable
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


def get_summary(
    initial_balance: float,
    current_balance: float,
    active_positions: Dict[str, Dict[str, Any]],
    pending_orders: Dict[str, Dict[str, Any]],
    trade_history: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Get a summary of current paper trading status.
    
    Args:
        initial_balance: Initial account balance
        current_balance: Current account balance
        active_positions: Dictionary of active positions
        pending_orders: Dictionary of pending orders
        trade_history: List of all trades
        
    Returns:
        Dictionary with current paper trading summary
    """
    unrealized_pnl = sum(
        position.get("unrealized_pnl", 0) for position in active_positions.values()
    )
    
    total_equity = current_balance + unrealized_pnl
    
    # Calculate performance metrics
    if initial_balance > 0:
        total_return_pct = (total_equity / initial_balance - 1) * 100
    else:
        total_return_pct = 0
    
    # Count trades
    closed_trades = [trade for trade in trade_history if trade.get("status") == "closed"]
    winning_trades = [trade for trade in closed_trades if trade.get("net_pnl", 0) > 0]
    
    return {
        "initial_balance": initial_balance,
        "current_balance": current_balance,
        "unrealized_pnl": unrealized_pnl,
        "total_equity": total_equity,
        "total_return_pct": total_return_pct,
        "active_positions": len(active_positions),
        "pending_orders": len(pending_orders),
        "completed_trades": len(closed_trades),
        "winning_trades": len(winning_trades),
        "win_rate": len(winning_trades) / len(closed_trades) if closed_trades else 0,
        "last_update": datetime.now().isoformat()
    }


def compare_to_backtest(
    paper_performance_report: Dict[str, Any],
    backtest_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compare paper trading results to backtest results.
    
    Args:
        paper_performance_report: Performance report from paper trading
        backtest_results: Results from backtesting
        
    Returns:
        Dictionary with comparison metrics
    """
    paper_metrics = paper_performance_report.get("metrics", {})
    backtest_metrics = backtest_results.get("performance_metrics", {})
    
    comparison = {}
    
    # Compare key metrics
    for metric in ["total_return", "win_rate", "profit_factor", "max_drawdown", "sharpe_ratio"]:
        paper_value = paper_metrics.get(metric, 0)
        backtest_value = backtest_metrics.get(metric, 0)
        
        if backtest_value != 0:
            difference_pct = (paper_value - backtest_value) / backtest_value * 100
        else:
            difference_pct = 0
            
        comparison[metric] = {
            "paper_trading": paper_value,
            "backtest": backtest_value,
            "difference": paper_value - backtest_value,
            "difference_pct": difference_pct
        }
    
    return comparison 