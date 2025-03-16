"""
Position management functionality for the backtesting engine.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from loguru import logger

from src.risk_management.risk_manager import RiskManager
from src.performance.performance_tracker import PerformanceTracker
from src.testing.backtest.utils import calculate_unrealized_pnl_pct


def process_positions(
    symbol: str,
    current_time: pd.Timestamp,
    current_data: pd.DataFrame,
    risk_manager: RiskManager,
    performance_tracker: PerformanceTracker,
    current_positions: Dict[str, Dict[str, Any]],
    trades: List[Dict[str, Any]],
    slippage: float,
    commission_rate: float
) -> Optional[float]:
    """
    Process active positions for a symbol.
    
    Args:
        symbol: Symbol to process
        current_time: Current timestamp
        current_data: Current market data
        risk_manager: Risk manager instance
        performance_tracker: Performance tracker instance
        current_positions: Dictionary of current positions
        trades: List of trades
        slippage: Slippage amount
        commission_rate: Commission rate
        
    Returns:
        Updated balance if position was closed, None otherwise
    """
    # Skip if no position for this symbol
    if symbol not in current_positions:
        return None
    
    position = current_positions[symbol]
    is_long = position["type"] == "long"
    current_bar = current_data.iloc[-1]
    
    # Get prices
    current_price = current_bar["close"]
    high_price = current_bar["high"]
    low_price = current_bar["low"]
    
    # Update highest/lowest prices reached
    if is_long and high_price > position.get("highest_price", 0):
        position["highest_price"] = high_price
    elif not is_long and (position.get("lowest_price") is None or low_price < position["lowest_price"]):
        position["lowest_price"] = low_price
    
    # Check for stop loss hit
    if (is_long and low_price <= position["stop_loss"]) or (not is_long and high_price >= position["stop_loss"]):
        # Close at stop loss
        return close_position(
            symbol=symbol,
            current_time=current_time,
            current_data=current_data,
            exit_reason="stop_loss",
            risk_manager=risk_manager,
            performance_tracker=performance_tracker,
            current_positions=current_positions,
            trades=trades,
            slippage=slippage,
            commission_rate=commission_rate,
            strategy_manager=None  # Will be passed in actual implementation
        )
    
    # Check for take profit hit
    if (is_long and high_price >= position["take_profit"]) or (not is_long and low_price <= position["take_profit"]):
        # Close at take profit
        return close_position(
            symbol=symbol,
            current_time=current_time,
            current_data=current_data,
            exit_reason="take_profit",
            risk_manager=risk_manager,
            performance_tracker=performance_tracker,
            current_positions=current_positions,
            trades=trades,
            slippage=slippage,
            commission_rate=commission_rate,
            strategy_manager=None  # Will be passed in actual implementation
        )
    
    # Check if trailing stop should be used
    unrealized_pnl_pct = calculate_unrealized_pnl_pct(position, current_price)
    
    if risk_manager.should_use_trailing_stop(unrealized_pnl_pct):
        # Calculate new trailing stop
        new_stop = risk_manager.calculate_trailing_stop(
            current_price=current_price,
            entry_price=position["entry_price"],
            is_long=is_long,
            highest_price=position.get("highest_price"),
            lowest_price=position.get("lowest_price")
        )
        
        # Only move stop in the favorable direction
        if (is_long and new_stop > position["stop_loss"]) or (not is_long and new_stop < position["stop_loss"]):
            position["stop_loss"] = new_stop
            
            # Update the stop loss in trades list
            for trade in trades:
                if trade["id"] == position["id"]:
                    trade["stop_loss"] = new_stop
                    break
    
    return None


def close_position(
    symbol: str,
    current_time: pd.Timestamp,
    current_data: pd.DataFrame,
    exit_reason: str,
    risk_manager: RiskManager,
    performance_tracker: PerformanceTracker,
    current_positions: Dict[str, Dict[str, Any]],
    trades: List[Dict[str, Any]],
    slippage: float,
    commission_rate: float,
    strategy_manager: Any
) -> float:
    """
    Close a position and calculate results.
    
    Args:
        symbol: Symbol to close
        current_time: Current timestamp
        current_data: Current market data
        exit_reason: Reason for exiting the position
        risk_manager: Risk manager instance
        performance_tracker: Performance tracker instance
        current_positions: Dictionary of current positions
        trades: List of trades
        slippage: Slippage amount
        commission_rate: Commission rate
        strategy_manager: Strategy manager instance
        
    Returns:
        Updated account balance
    """
    # Skip if no position for this symbol
    if symbol not in current_positions:
        return None
    
    position = current_positions[symbol]
    is_long = position["type"] == "long"
    current_bar = current_data.iloc[-1]
    
    # Determine exit price with slippage
    if exit_reason == "stop_loss":
        exit_price = position["stop_loss"]
    elif exit_reason == "take_profit":
        exit_price = position["take_profit"]
    else:
        exit_price = current_bar["close"]
    
    # Apply slippage
    if is_long:
        exit_price *= (1 - slippage)
    else:
        exit_price *= (1 + slippage)
    
    # Calculate profit/loss
    price_diff = exit_price - position["entry_price"] if is_long else position["entry_price"] - exit_price
    pnl = price_diff * position["position_size"]
    pnl_pct = (price_diff / position["entry_price"]) * 100
    
    # Calculate commission
    commission = exit_price * position["position_size"] * commission_rate
    total_commission = position["commission_paid"] + commission
    
    # Net PnL after commission
    net_pnl = pnl - total_commission
    
    # Update balance
    new_balance = position["position_size"] * exit_price - commission
    
    # Update trade record
    for trade in trades:
        if trade["id"] == position["id"]:
            trade["exit_time"] = current_time
            trade["exit_price"] = exit_price
            trade["exit_reason"] = exit_reason
            trade["pnl"] = pnl
            trade["pnl_pct"] = pnl_pct
            trade["commission"] = total_commission
            trade["net_pnl"] = net_pnl
            trade["status"] = "closed"
            break
    
    # Log trade result
    logger.info(
        f"Closed {position['type']} position for {symbol} at {exit_price} "
        f"({exit_reason}): PnL = {net_pnl:.2f} ({pnl_pct:.2f}%)"
    )
    
    # Update risk manager
    risk_manager.update_trade_result({
        "realized_pnl": net_pnl,
        "status": "closed"
    })
    
    # Update performance tracker
    performance_tracker.add_trade({
        "id": position["id"],
        "symbol": symbol,
        "side": "Buy" if is_long else "Sell",
        "entry_price": position["entry_price"],
        "exit_price": exit_price,
        "position_size": position["position_size"],
        "entry_time": position["entry_time"],
        "exit_time": current_time,
        "realized_pnl": net_pnl,
        "realized_pnl_percent": pnl_pct,
        "strategy_name": position["strategy_name"],
        "exit_reason": exit_reason,
        "status": "closed"
    })
    
    # Update strategy performance
    if strategy_manager:
        strategy_manager.update_strategy_performance({
            "strategy_name": position["strategy_name"],
            "realized_pnl": net_pnl,
            "realized_pnl_percent": pnl_pct,
            "exit_reason": exit_reason
        })
    
    # Remove from current positions
    del current_positions[symbol]
    
    return new_balance


def calculate_equity(
    current_time: pd.Timestamp,
    historical_data: Dict[str, pd.DataFrame],
    current_balance: float,
    current_positions: Dict[str, Dict[str, Any]]
) -> float:
    """
    Calculate total equity (balance + unrealized PnL).
    
    Args:
        current_time: Current timestamp
        historical_data: Historical market data
        current_balance: Current account balance
        current_positions: Dictionary of current positions
        
    Returns:
        Total equity
    """
    equity = current_balance
    
    # Add unrealized PnL from open positions
    for symbol, position in current_positions.items():
        # Skip if no data for this symbol at current time
        if symbol not in historical_data:
            continue
        
        # Get latest price
        df = historical_data[symbol]
        if current_time not in df.index:
            continue
        
        current_price = df.loc[current_time, "close"]
        is_long = position["type"] == "long"
        
        # Calculate unrealized PnL
        price_diff = current_price - position["entry_price"] if is_long else position["entry_price"] - current_price
        unrealized_pnl = price_diff * position["position_size"]
        
        equity += unrealized_pnl
    
    return equity 