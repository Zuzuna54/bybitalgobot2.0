"""
Trade execution functionality for the backtesting engine.
"""

from typing import Dict, Any, Optional
import pandas as pd
from loguru import logger

from src.models.models import Signal, SignalType
from src.risk_management.risk_manager import RiskManager
from src.backtesting.backtest.utils import calculate_unrealized_pnl_pct


def execute_signal(
    signal: Signal,
    current_time: pd.Timestamp,
    current_data: pd.DataFrame,
    risk_manager: RiskManager,
    strategy_manager: Any,
    current_positions: Dict[str, Dict[str, Any]],
    current_balance: float,
    slippage: float,
    commission_rate: float,
    trades: list,
) -> Optional[str]:
    """
    Execute a trading signal in the backtest.

    Args:
        signal: Trading signal
        current_time: Current timestamp
        current_data: Current market data
        risk_manager: Risk manager instance
        strategy_manager: Strategy manager instance
        current_positions: Dictionary of current positions
        current_balance: Current account balance
        slippage: Slippage amount
        commission_rate: Commission rate
        trades: List of trades

    Returns:
        Trade ID if executed, None otherwise
    """
    # Skip if we already have a position for this symbol
    symbol = signal.symbol
    if symbol in current_positions:
        return None

    # Skip neutral/none signals
    if signal.signal_type == SignalType.NONE:
        return None

    # Check if we should take this trade based on risk management
    if not risk_manager.should_take_trade(
        symbol=symbol, signal_strength=signal.strength, account_balance=current_balance
    ):
        logger.debug(f"Risk manager rejected trade for {symbol}")
        return None

    # Get current price and apply slippage
    entry_price = signal.price
    is_long = signal.signal_type.name == "BUY"

    # Apply slippage
    if is_long:
        entry_price *= 1 + slippage
    else:
        entry_price *= 1 - slippage

    # Calculate stop loss
    stop_loss_price = None
    for strategy_name in signal.metadata.get(
        "contributing_strategies", [signal.metadata.get("strategy_name")]
    ):
        if strategy_name in strategy_manager.strategies:
            strategy = strategy_manager.strategies[strategy_name]
            stop_loss_price = strategy.get_stop_loss_price(
                entry_price, is_long, current_data
            )
            break

    # If no stop loss was calculated, use a default percentage
    if stop_loss_price is None:
        atr_value = current_data.iloc[-1].get("atr")
        stop_loss_price = risk_manager.calculate_stop_loss(
            entry_price, is_long, atr_value
        )

    # Calculate take profit based on risk-reward ratio
    take_profit_price = risk_manager.calculate_take_profit(
        entry_price, stop_loss_price, is_long
    )

    # Calculate position size
    position_size = risk_manager.calculate_position_size(
        symbol=symbol,
        account_balance=current_balance,
        entry_price=entry_price,
        stop_loss_price=stop_loss_price,
    )

    # Skip if position size is too small
    if position_size <= 0:
        return None

    # Calculate commission
    commission = entry_price * position_size * commission_rate

    # Update balance (subtract commission)
    new_balance = current_balance - commission

    # Generate trade ID
    trade_id = f"{symbol}-{current_time.strftime('%Y%m%d%H%M%S')}"

    # Create position
    current_positions[symbol] = {
        "id": trade_id,
        "symbol": symbol,
        "type": "long" if is_long else "short",
        "entry_time": current_time,
        "entry_price": entry_price,
        "stop_loss": stop_loss_price,
        "take_profit": take_profit_price,
        "position_size": position_size,
        "strategy_name": signal.metadata.get("strategy_name", "unknown"),
        "contributing_strategies": signal.metadata.get("contributing_strategies", []),
        "signal_strength": signal.strength,
        "commission_paid": commission,
        "highest_price": entry_price if is_long else None,
        "lowest_price": entry_price if not is_long else None,
    }

    # Add to trades list
    trades.append(
        {
            "id": trade_id,
            "symbol": symbol,
            "type": "long" if is_long else "short",
            "entry_time": current_time,
            "entry_price": entry_price,
            "stop_loss": stop_loss_price,
            "take_profit": take_profit_price,
            "position_size": position_size,
            "strategy_name": signal.metadata.get("strategy_name", "unknown"),
            "status": "open",
            "signal_strength": signal.strength,
        }
    )

    return trade_id, new_balance
