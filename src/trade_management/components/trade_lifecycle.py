"""
Trade lifecycle management for the Algorithmic Trading System.

This module provides functions for managing the lifecycle of trades,
including signal processing, trade execution, and trade closure.
"""

from typing import Dict, Any, Optional, List
import time
from datetime import datetime
from loguru import logger

from src.api.bybit_client import BybitClient
from src.risk_management.risk_manager import RiskManager
from src.models.models import Signal, SignalType

from src.trade_management.components.order_handler import (
    OrderSide,
    create_market_order,
    create_stop_order,
    create_take_profit_order,
    cancel_order,
    get_order_status,
)

from src.trade_management.components.position_tracker import Trade, TradeStatus


def process_trading_signal(
    signal: Signal,
    api_client: BybitClient,
    risk_manager: RiskManager,
    account_balance: float,
) -> Optional[Trade]:
    """
    Process a trading signal and create a Trade if valid.

    Args:
        signal: Trading signal
        api_client: Bybit API client
        risk_manager: Risk manager instance
        account_balance: Current account balance

    Returns:
        Trade object if signal is valid, None otherwise
    """
    # Skip neutral/none signals
    if signal.signal_type == SignalType.NONE:
        logger.info(
            f"No signal (neutral) received for {signal.symbol} - no trade executed"
        )
        return None

    # Check if we should take this trade based on risk rules
    if not risk_manager.should_take_trade(
        signal.symbol, signal.strength, account_balance
    ):
        logger.info(f"Risk manager rejected trade for {signal.symbol}")
        return None

    # Determine order side
    side = OrderSide.BUY if signal.signal_type == SignalType.BUY else OrderSide.SELL

    # Get current market price if not provided in signal
    entry_price = signal.price or float(api_client.get_symbol_price(signal.symbol))

    # Initialize indicators to calculate stop loss and take profit
    indicator_data = signal.metadata.get("indicators", {})
    atr_value = indicator_data.get("atr")

    # Calculate stop loss price
    stop_loss_price = risk_manager.calculate_stop_loss(
        entry_price=entry_price, is_long=(side == OrderSide.BUY), atr_value=atr_value
    )

    # Calculate take profit price
    take_profit_price = risk_manager.calculate_take_profit(
        entry_price=entry_price,
        stop_loss_price=stop_loss_price,
        is_long=(side == OrderSide.BUY),
    )

    # Get symbol information for lot size requirements
    instruments_info = api_client.market.get_instruments_info(symbol=signal.symbol)
    symbol_info = {}

    # Extract the symbol info from the result
    if instruments_info and "list" in instruments_info:
        for instrument in instruments_info.get("list", []):
            if instrument.get("symbol") == signal.symbol:
                symbol_info = instrument
                break

    min_order_qty = float(symbol_info.get("lotSizeFilter", {}).get("minOrderQty", 0))

    # Get recommended leverage
    volatility = None
    if "volatility" in indicator_data:
        volatility = indicator_data.get("volatility")
    leverage = risk_manager.get_recommended_leverage(signal.symbol, volatility)

    # Calculate position size
    position_size = risk_manager.calculate_position_size(
        symbol=signal.symbol,
        account_balance=account_balance,
        entry_price=entry_price,
        stop_loss_price=stop_loss_price,
        leverage=leverage,
        min_order_qty=min_order_qty,
    )

    # Skip if position size is too small
    if position_size <= 0:
        logger.warning(
            f"Position size calculation resulted in zero or negative value - skipping trade"
        )
        return None

    # Create the trade object
    trade = Trade(
        symbol=signal.symbol,
        side=side,
        entry_price=entry_price,
        stop_loss_price=stop_loss_price,
        take_profit_price=take_profit_price,
        position_size=position_size,
        strategy_name=signal.metadata.get("strategy_name", "unknown"),
        signal_strength=signal.strength,
        leverage=leverage,
    )

    return trade


def execute_trade_entry(
    trade: Trade,
    api_client: BybitClient,
    risk_manager: RiskManager,
    simulate: bool = False,
    simulation_delay_sec: float = 0.0,
) -> bool:
    """
    Execute a trade entry by placing the necessary orders.

    Args:
        trade: Trade to execute
        api_client: Bybit API client
        risk_manager: Risk manager instance
        simulate: Whether to simulate the trade instead of executing it
        simulation_delay_sec: Simulated delay for order execution

    Returns:
        True if trade was executed successfully, False otherwise
    """
    if simulate:
        # Simulate order execution
        if simulation_delay_sec > 0:
            time.sleep(simulation_delay_sec)

        # Simulate entry order
        entry_order_id = f"sim-{trade.id}-entry"
        trade.update_order(
            entry_order_id,
            {
                "orderId": entry_order_id,
                "symbol": trade.symbol,
                "side": trade.side.value,
                "price": trade.entry_price,
                "qty": trade.position_size,
                "orderStatus": "Filled",
                "timeInForce": "GTC",
            },
        )

        # Mark the trade as open
        trade.entry_time = datetime.now()
        trade.set_status(TradeStatus.OPEN)

        return True

    try:
        # Set leverage
        api_client.set_leverage(trade.symbol, trade.leverage)

        # Create market entry order
        entry_order_result = create_market_order(
            api_client=api_client,
            symbol=trade.symbol,
            side=trade.side,
            qty=trade.position_size,
            reduce_only=False,
            simulate=False,
        )

        logger.info(f"Entry order placed: {entry_order_result}")

        # Store order information
        if "orderId" in entry_order_result:
            entry_order_id = entry_order_result["orderId"]
            trade.update_order(entry_order_id, entry_order_result)

            # Check if the order has been filled
            order_status = get_order_status(api_client, trade.symbol, entry_order_id)

            if order_status.get("orderStatus") == "Filled":
                filled_price = float(order_status.get("avgPrice", trade.entry_price))

                # Update trade with actual entry price
                if filled_price != trade.entry_price:
                    # Recalculate stop loss and take profit with actual entry price
                    atr_value = None
                    # TODO: Get ATR value if needed

                    trade.entry_price = filled_price
                    trade.stop_loss_price = risk_manager.calculate_stop_loss(
                        entry_price=filled_price,
                        is_long=(trade.side == OrderSide.BUY),
                        atr_value=atr_value,
                    )

                    trade.take_profit_price = risk_manager.calculate_take_profit(
                        entry_price=filled_price,
                        stop_loss_price=trade.stop_loss_price,
                        is_long=(trade.side == OrderSide.BUY),
                    )

                # Mark the trade as open
                trade.entry_time = datetime.now()
                trade.set_status(TradeStatus.OPEN)

                # Place stop loss order
                sl_side = (
                    OrderSide.SELL if trade.side == OrderSide.BUY else OrderSide.BUY
                )
                stop_loss_order_result = create_stop_order(
                    api_client=api_client,
                    symbol=trade.symbol,
                    side=sl_side,
                    qty=trade.position_size,
                    stop_price=trade.stop_loss_price,
                    reduce_only=True,
                    simulate=False,
                )

                logger.info(f"Stop loss order placed: {stop_loss_order_result}")

                if "orderId" in stop_loss_order_result:
                    sl_order_id = stop_loss_order_result["orderId"]
                    trade.update_order(sl_order_id, stop_loss_order_result)

                # Place take profit order
                tp_side = (
                    OrderSide.SELL if trade.side == OrderSide.BUY else OrderSide.BUY
                )
                take_profit_order_result = create_take_profit_order(
                    api_client=api_client,
                    symbol=trade.symbol,
                    side=tp_side,
                    qty=trade.position_size,
                    take_profit_price=trade.take_profit_price,
                    reduce_only=True,
                    simulate=False,
                )

                logger.info(f"Take profit order placed: {take_profit_order_result}")

                if "orderId" in take_profit_order_result:
                    tp_order_id = take_profit_order_result["orderId"]
                    trade.update_order(tp_order_id, take_profit_order_result)

                return True
            else:
                logger.warning(f"Entry order not filled immediately: {order_status}")
                # Could implement a retry or check mechanism here
                return False
        else:
            logger.error(f"Failed to place entry order: {entry_order_result}")
            return False

    except Exception as e:
        logger.error(f"Error executing trade: {e}")
        trade.set_status(TradeStatus.REJECTED)
        return False


def close_trade_at_market(
    trade: Trade,
    exit_price: float,
    exit_time: datetime,
    exit_reason: str,
    api_client: BybitClient,
    risk_manager: RiskManager,
    simulate: bool = False,
) -> bool:
    """
    Close a trade at market price.

    Args:
        trade: Trade to close
        exit_price: Exit price
        exit_time: Exit timestamp
        exit_reason: Reason for exiting
        api_client: Bybit API client
        risk_manager: Risk manager instance
        simulate: Whether to simulate closing instead of executing it

    Returns:
        True if trade was closed successfully, False otherwise
    """
    if simulate:
        # Simulate trade closure
        trade.close_trade(exit_price, exit_time, exit_reason)

        # Update risk manager
        risk_manager.update_trade_result(trade.to_dict())
        return True

    try:
        # Cancel any existing orders
        for order_id, order_data in trade.orders.items():
            if order_data.get("orderStatus") not in ["Filled", "Canceled"]:
                cancel_order(api_client, trade.symbol, order_id)

        # Create market close order
        close_side = OrderSide.SELL if trade.side == OrderSide.BUY else OrderSide.BUY
        close_order_result = create_market_order(
            api_client=api_client,
            symbol=trade.symbol,
            side=close_side,
            qty=trade.position_size,
            reduce_only=True,
            simulate=False,
        )

        logger.info(f"Close order placed: {close_order_result}")

        # Check if the order was filled
        if "orderId" in close_order_result:
            close_order_id = close_order_result["orderId"]
            trade.update_order(close_order_id, close_order_result)

            # Get the actual fill price
            order_status = get_order_status(api_client, trade.symbol, close_order_id)
            if order_status.get("orderStatus") == "Filled":
                filled_exit_price = float(order_status.get("avgPrice", exit_price))

                # Close the trade with actual exit price
                trade.close_trade(filled_exit_price, exit_time, exit_reason)

                # Update risk manager
                risk_manager.update_trade_result(trade.to_dict())

                return True

        return False

    except Exception as e:
        logger.error(f"Error closing trade at market: {e}")
        return False
