"""
Trade Data Module

This module provides functions to retrieve and process trade-related data
for the dashboard.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loguru import logger


# Sample trade data for standalone mode
def _generate_sample_trades(count=50):
    """Generate sample trades for standalone mode."""
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT"]
    strategies = [
        "RSI Reversal",
        "Breakout",
        "Mean Reversion",
        "Grid Trading",
        "MACD Crossover",
    ]

    trades = []

    # Start from 30 days ago and move forward
    start_date = datetime.now() - timedelta(days=30)

    for i in range(count):
        # Create a trade with realistic properties
        entry_time = start_date + timedelta(hours=i * 12)
        exit_time = entry_time + timedelta(hours=np.random.randint(1, 24))

        # Exit time should not be in the future
        if exit_time > datetime.now():
            exit_time = datetime.now() - timedelta(minutes=np.random.randint(10, 60))

        symbol = np.random.choice(symbols)
        strategy = np.random.choice(strategies)

        # Base price on the symbol
        if symbol == "BTCUSDT":
            base_price = 50000
        elif symbol == "ETHUSDT":
            base_price = 3000
        elif symbol == "SOLUSDT":
            base_price = 100
        elif symbol == "BNBUSDT":
            base_price = 500
        else:  # ADAUSDT
            base_price = 1.5

        # Apply some price movement
        entry_price = base_price * (1 + np.random.normal(0, 0.05))
        exit_price = entry_price * (1 + np.random.normal(0, 0.07))

        # Determine if trade was profitable
        profitable = (
            exit_price > entry_price
            if np.random.random() > 0.4
            else exit_price < entry_price
        )

        # Calculate PnL
        size = np.random.uniform(0.1, 2.0)
        if symbol in ["BTCUSDT", "ETHUSDT"]:
            size = size / 10  # Smaller position sizes for high-priced assets

        pnl = size * (exit_price - entry_price)
        if not profitable:
            pnl = -abs(pnl)  # Ensure PnL is negative for losing trades

        trades.append(
            {
                "id": f"trade-{i+1}",
                "symbol": symbol,
                "strategy": strategy,
                "entry_time": entry_time.isoformat(),
                "exit_time": exit_time.isoformat(),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "size": size,
                "profitable": profitable,
                "pnl": pnl,
                "pnl_pct": (exit_price - entry_price) / entry_price * 100,
                "status": "closed",
            }
        )

    # Add a few open trades
    for i in range(3):
        entry_time = datetime.now() - timedelta(hours=np.random.randint(1, 12))
        symbol = np.random.choice(symbols)
        strategy = np.random.choice(strategies)

        # Base price on the symbol
        if symbol == "BTCUSDT":
            base_price = 50000
        elif symbol == "ETHUSDT":
            base_price = 3000
        elif symbol == "SOLUSDT":
            base_price = 100
        elif symbol == "BNBUSDT":
            base_price = 500
        else:  # ADAUSDT
            base_price = 1.5

        # Apply some price movement
        entry_price = base_price * (1 + np.random.normal(0, 0.05))
        current_price = entry_price * (1 + np.random.normal(0, 0.03))

        # Calculate unrealized PnL
        size = np.random.uniform(0.1, 2.0)
        if symbol in ["BTCUSDT", "ETHUSDT"]:
            size = size / 10  # Smaller position sizes for high-priced assets

        unrealized_pnl = size * (current_price - entry_price)

        trades.append(
            {
                "id": f"trade-open-{i+1}",
                "symbol": symbol,
                "strategy": strategy,
                "entry_time": entry_time.isoformat(),
                "exit_time": None,
                "entry_price": entry_price,
                "exit_price": None,
                "current_price": current_price,
                "size": size,
                "profitable": unrealized_pnl > 0,
                "unrealized_pnl": unrealized_pnl,
                "unrealized_pnl_pct": (current_price - entry_price) / entry_price * 100,
                "status": "open",
            }
        )

    return trades


def _initialize_trade_data(service):
    """
    Initialize trade data storage with sample data for standalone mode.

    Args:
        service: DashboardDataService instance
    """
    service._trade_data = {
        "completed_trades": _generate_sample_trades(40),
        "open_positions": _generate_sample_trades(3)[-3:],  # Last 3 are open trades
        "pending_orders": _generate_sample_pending_orders(),
        "trade_history": _generate_sample_trades(40)[
            :40
        ],  # First 40 are completed trades
    }

    # Set initial update timestamp
    service._data_updated_at["trades"] = datetime.now()

    logger.debug("Initialized sample trade data")


def _update_trade_data(service):
    """
    Update trade data from the trading system.

    Args:
        service: DashboardDataService instance
    """
    if service.standalone_mode:
        # In standalone mode, just update the timestamp
        service._data_updated_at["trades"] = datetime.now()
        return

    try:
        # Get data from trade manager if available
        if service.trade_manager:
            logger.debug("Updating trade data from trade manager")

            # Get completed trades
            completed_trades = service.trade_manager.get_completed_trades()
            if completed_trades is not None:
                service._trade_data["completed_trades"] = completed_trades
                service._trade_data["trade_history"] = completed_trades

            # Get open positions
            open_positions = service.trade_manager.get_open_positions()
            if open_positions is not None:
                service._trade_data["open_positions"] = open_positions

            # Get pending orders
            pending_orders = service.trade_manager.get_pending_orders()
            if pending_orders is not None:
                service._trade_data["pending_orders"] = pending_orders

        # Update timestamp
        service._data_updated_at["trades"] = datetime.now()

    except Exception as e:
        logger.error(f"Error updating trade data: {str(e)}")
        # Keep using existing data if update fails


def get_trade_data(self) -> Dict[str, Any]:
    """
    Get trade data for the dashboard.

    Returns:
        Dictionary with trade data
    """
    return self._trade_data


def _generate_sample_pending_orders():
    """Generate sample pending orders for standalone mode."""
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT"]
    order_types = ["LIMIT", "STOP", "TAKE_PROFIT"]

    orders = []

    for i in range(5):
        symbol = np.random.choice(symbols)
        order_type = np.random.choice(order_types)

        # Base price on the symbol
        if symbol == "BTCUSDT":
            base_price = 50000
        elif symbol == "ETHUSDT":
            base_price = 3000
        elif symbol == "SOLUSDT":
            base_price = 100
        elif symbol == "BNBUSDT":
            base_price = 500
        else:  # ADAUSDT
            base_price = 1.5

        # Modify price based on order type
        if order_type == "LIMIT":
            price = base_price * (
                1 - np.random.uniform(0.01, 0.05)
            )  # Buy limit below market
        elif order_type == "STOP":
            price = base_price * (
                1 + np.random.uniform(0.01, 0.05)
            )  # Stop above market
        else:  # TAKE_PROFIT
            price = base_price * (
                1 + np.random.uniform(0.05, 0.10)
            )  # Take profit higher

        size = np.random.uniform(0.1, 1.0)
        if symbol in ["BTCUSDT", "ETHUSDT"]:
            size = size / 10  # Smaller position sizes for high-priced assets

        orders.append(
            {
                "id": f"order-{i+1}",
                "symbol": symbol,
                "type": order_type,
                "side": "BUY" if order_type == "LIMIT" else "SELL",
                "price": price,
                "size": size,
                "status": "NEW",
                "created_at": (
                    datetime.now() - timedelta(hours=np.random.randint(1, 24))
                ).isoformat(),
            }
        )

    return orders
