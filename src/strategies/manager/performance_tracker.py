"""
Performance Tracker for Strategy Management

This module provides functionality for tracking, updating, and persisting
performance metrics for trading strategies.
"""

from typing import Dict, Any, List, Optional
import os
import json
from loguru import logger


def update_strategy_performance(
    performance_data: Dict[str, Dict[str, Any]],
    strategy_name: str,
    trade_result: Dict[str, Any]
) -> None:
    """
    Update strategy performance based on completed trade.
    
    Args:
        performance_data: Dictionary of strategy performance metrics
        strategy_name: Name of the strategy
        trade_result: Trade result dictionary
    """
    # Get necessary trade information
    pnl = trade_result.get("realized_pnl", 0.0)
    is_successful = pnl > 0
    
    # Update performance metrics
    performance = performance_data[strategy_name]
    performance["signals_executed"] += 1
    
    if is_successful:
        performance["successful_signals"] += 1
    else:
        performance["failed_signals"] += 1
    
    performance["total_profit_loss"] += pnl
    performance["win_rate"] = (
        performance["successful_signals"] / performance["signals_executed"]
        if performance["signals_executed"] > 0 else 0.0
    )


def update_strategy_weight(
    performance_data: Dict[str, Dict[str, Any]],
    strategy_name: str,
    min_signals: int,
    max_weight_change: float
) -> None:
    """
    Update strategy weight based on performance metrics.
    
    Args:
        performance_data: Dictionary of strategy performance metrics
        strategy_name: Name of the strategy
        min_signals: Minimum number of signals required for weight adjustment
        max_weight_change: Maximum allowed weight change per update
    """
    performance = performance_data[strategy_name]
    
    # Need minimum number of signals to adjust weight
    if performance["signals_executed"] < min_signals:
        return
    
    # Calculate new weight based on win rate and profit
    win_rate = performance["win_rate"]
    profit_factor = 1.0
    
    if performance["total_profit_loss"] != 0:
        # Calculate a profit factor component
        total_profit = max(performance["total_profit_loss"], 0)
        total_loss = abs(min(performance["total_profit_loss"], 0))
        
        if total_loss > 0:
            profit_factor = total_profit / total_loss
        else:
            profit_factor = 2.0  # Only profits, no losses
    
    # Combine win rate and profit factor (50/50 weight)
    performance_score = (win_rate * 0.5) + (min(profit_factor, 2.0) / 4.0)
    
    # Calculate new weight (range 0.5 to 2.0)
    new_weight = 0.5 + (performance_score * 1.5)
    
    # Limit change rate
    current_weight = performance["weight"]
    limited_weight = max(min(new_weight, current_weight + max_weight_change), current_weight - max_weight_change)
    
    # Update weight
    performance["weight"] = limited_weight


def save_performance_data(
    performance_data: Dict[str, Dict[str, Any]],
    file_path: str
) -> bool:
    """
    Save strategy performance metrics to file.
    
    Args:
        performance_data: Dictionary of strategy performance metrics
        file_path: Path to save performance data
        
    Returns:
        True if saved successfully, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, "w") as f:
            json.dump(performance_data, f, indent=2)
        
        logger.info(f"Strategy performance saved to {file_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error saving strategy performance: {e}")
        return False


def load_performance_data(
    performance_data: Dict[str, Dict[str, Any]],
    file_path: str
) -> bool:
    """
    Load strategy performance metrics from file.
    
    Args:
        performance_data: Dictionary to update with loaded data
        file_path: Path to load performance data from
        
    Returns:
        True if loaded successfully, False otherwise
    """
    try:
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                saved_performance = json.load(f)
            
            # Merge with existing performance data
            for strategy_name, performance in saved_performance.items():
                if strategy_name in performance_data:
                    performance_data[strategy_name].update(performance)
            
            logger.info(f"Strategy performance loaded from {file_path}")
            return True
    
    except Exception as e:
        logger.error(f"Error loading strategy performance: {e}")
    
    return False


def get_strategy_performance_summary(
    performance_data: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Generate a summary of strategy performance.
    
    Args:
        performance_data: Dictionary of strategy performance metrics
        
    Returns:
        Dictionary with strategy performance summaries
    """
    summary = {}
    
    for strategy_name, performance in performance_data.items():
        summary[strategy_name] = {
            "signals_generated": performance.get("signals_generated", 0),
            "signals_executed": performance.get("signals_executed", 0),
            "win_rate": performance.get("win_rate", 0.0),
            "total_profit_loss": performance.get("total_profit_loss", 0.0),
            "weight": performance.get("weight", 1.0)
        }
    
    return summary 