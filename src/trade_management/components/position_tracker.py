"""
Position tracking functionality for the Algorithmic Trading System.

This module provides classes and functions for tracking trading positions,
including the Trade class that represents a single trade with all its attributes.
"""

from typing import Dict, Any, Optional, List
from enum import Enum
from datetime import datetime
from loguru import logger

from src.trade_management.components.order_handler import OrderSide


class TradeStatus(Enum):
    """Status of a trade."""
    PENDING = "pending"
    OPEN = "open"
    CLOSED = "closed"
    CANCELED = "canceled"
    REJECTED = "rejected"


class Trade:
    """Represents a single trade with all its attributes."""
    
    def __init__(
        self,
        symbol: str,
        side: OrderSide,
        entry_price: float,
        stop_loss_price: float,
        take_profit_price: float,
        position_size: float,
        strategy_name: str,
        signal_strength: float = 0.0,
        leverage: int = 1
    ):
        """
        Initialize a new trade.
        
        Args:
            symbol: Trading pair symbol
            side: Trade side (BUY or SELL)
            entry_price: Entry price
            stop_loss_price: Stop loss price
            take_profit_price: Take profit price
            position_size: Position size in base currency
            strategy_name: Name of the strategy that generated the signal
            signal_strength: Strength of the signal (0.0 to 1.0)
            leverage: Leverage to use for the trade
        """
        self.id = f"{symbol}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.symbol = symbol
        self.side = side
        self.entry_price = entry_price
        self.stop_loss_price = stop_loss_price
        self.take_profit_price = take_profit_price
        self.position_size = position_size
        self.strategy_name = strategy_name
        self.signal_strength = signal_strength
        self.leverage = leverage
        
        self.status = TradeStatus.PENDING
        self.orders: Dict[str, Dict[str, Any]] = {}
        
        self.entry_time: Optional[datetime] = None
        self.exit_time: Optional[datetime] = None
        self.exit_price: Optional[float] = None
        self.realized_pnl: Optional[float] = None
        self.realized_pnl_percent: Optional[float] = None
        self.exit_reason: Optional[str] = None
        
        # For tracking stop loss/take profit adjustments
        self.initial_stop_loss = stop_loss_price
        self.initial_take_profit = take_profit_price
        self.highest_price_reached: Optional[float] = None
        self.lowest_price_reached: Optional[float] = None
    
    @property
    def is_long(self) -> bool:
        """Whether this is a long trade."""
        return self.side == OrderSide.BUY
    
    def update_order(self, order_id: str, order_data: Dict[str, Any]) -> None:
        """
        Update order information.
        
        Args:
            order_id: Order ID
            order_data: Order data dictionary
        """
        self.orders[order_id] = order_data
    
    def set_status(self, status: TradeStatus) -> None:
        """
        Set trade status.
        
        Args:
            status: New status
        """
        self.status = status
        logger.debug(f"Trade {self.id} status set to {status.value}")
    
    def close_trade(
        self,
        exit_price: float,
        exit_time: datetime,
        exit_reason: str
    ) -> None:
        """
        Close the trade and calculate results.
        
        Args:
            exit_price: Exit price
            exit_time: Exit timestamp
            exit_reason: Reason for exiting
        """
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_reason = exit_reason
        
        # Calculate profit/loss
        if self.is_long:
            price_change = exit_price - self.entry_price
        else:
            price_change = self.entry_price - exit_price
        
        self.realized_pnl = price_change * self.position_size
        self.realized_pnl_percent = (price_change / self.entry_price) * 100 * self.leverage
        
        # Set status to closed
        self.set_status(TradeStatus.CLOSED)
        
        logger.info(
            f"Closed {self.side.value} trade for {self.symbol} at {exit_price} "
            f"({exit_reason}): PnL = {self.realized_pnl:.2f} ({self.realized_pnl_percent:.2f}%)"
        )
    
    def update_market_data(self, current_price: float, timestamp: datetime) -> None:
        """
        Update trade with current market data.
        
        Args:
            current_price: Current market price
            timestamp: Current timestamp
        """
        # Track highest/lowest prices for trailing stops
        if self.highest_price_reached is None or current_price > self.highest_price_reached:
            self.highest_price_reached = current_price
        
        if self.lowest_price_reached is None or current_price < self.lowest_price_reached:
            self.lowest_price_reached = current_price
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert trade to dictionary.
        
        Returns:
            Dictionary representation of the trade
        """
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side.value,
            "entry_price": self.entry_price,
            "stop_loss_price": self.stop_loss_price,
            "take_profit_price": self.take_profit_price,
            "position_size": self.position_size,
            "leverage": self.leverage,
            "strategy_name": self.strategy_name,
            "signal_strength": self.signal_strength,
            "status": self.status.value,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "exit_price": self.exit_price,
            "realized_pnl": self.realized_pnl,
            "realized_pnl_percent": self.realized_pnl_percent,
            "exit_reason": self.exit_reason,
            "initial_stop_loss": self.initial_stop_loss,
            "initial_take_profit": self.initial_take_profit
        }


def update_position_market_data(trade: Trade, current_price: float, timestamp: datetime) -> None:
    """
    Update a trade position with current market data.
    
    Args:
        trade: Trade to update
        current_price: Current market price
        timestamp: Current timestamp
    """
    trade.update_market_data(current_price, timestamp)


def calculate_unrealized_profit_pct(trade: Trade, current_price: float) -> float:
    """
    Calculate unrealized profit percentage for a trade.
    
    Args:
        trade: Trade
        current_price: Current market price
        
    Returns:
        Unrealized profit percentage
    """
    if trade.is_long:
        price_change = current_price - trade.entry_price
    else:
        price_change = trade.entry_price - current_price
    
    return (price_change / trade.entry_price) * 100 * trade.leverage


def get_active_positions_summary(active_trades: Dict[str, Trade]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get summary of active positions grouped by symbol.
    
    Args:
        active_trades: Dictionary of active trades by ID
        
    Returns:
        Dictionary of active positions by symbol
    """
    positions_by_symbol: Dict[str, List[Dict[str, Any]]] = {}
    
    for trade_id, trade in active_trades.items():
        symbol = trade.symbol
        
        if symbol not in positions_by_symbol:
            positions_by_symbol[symbol] = []
        
        positions_by_symbol[symbol].append({
            "id": trade.id,
            "side": trade.side.value,
            "entry_price": trade.entry_price,
            "current_sl": trade.stop_loss_price,
            "current_tp": trade.take_profit_price,
            "size": trade.position_size,
            "entry_time": trade.entry_time
        })
    
    return positions_by_symbol 