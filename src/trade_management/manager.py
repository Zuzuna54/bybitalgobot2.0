"""
Trade Management Module for the Algorithmic Trading System

This module provides the core TradeManager class which orchestrates the trade management
functionality, including order creation, position tracking, execution reporting,
and trade lifecycle management.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
import pandas as pd
from loguru import logger

from src.api.bybit_client import BybitClient
from src.risk_management.risk_manager import RiskManager
from src.models.models import Signal, SignalType

# Import components
from src.trade_management.components.order_handler import (
    create_market_order,
    create_limit_order,
    create_stop_order,
    create_take_profit_order,
    update_stop_loss_order,
    cancel_order,
    OrderType,
    OrderSide,
)

from src.trade_management.components.position_tracker import (
    Trade,
    TradeStatus,
    update_position_market_data,
    calculate_unrealized_profit_pct,
)

from src.trade_management.components.trade_lifecycle import (
    process_trading_signal,
    execute_trade_entry,
    close_trade_at_market,
)

from src.trade_management.components.execution_report import (
    get_trade_summary,
    get_trade_history_dataframe,
    save_trade_history,
)


class TradeManager:
    """Manages trade execution and tracking for the trading system."""

    def __init__(
        self,
        api_client: BybitClient,
        risk_manager: RiskManager,
        simulate: bool = False,
        simulation_delay_sec: float = 0.0,
    ):
        """
        Initialize the trade manager.

        Args:
            api_client: Bybit API client
            risk_manager: Risk manager instance
            simulate: Whether to simulate trades instead of executing them
            simulation_delay_sec: Simulated delay in seconds for order execution
        """
        self.api_client = api_client
        self.risk_manager = risk_manager
        self.simulate = simulate
        self.simulation_delay_sec = simulation_delay_sec

        # Trade tracking
        self.trades: Dict[str, Trade] = {}
        self.active_trades: Dict[str, Trade] = {}
        self.completed_trades: Dict[str, Trade] = {}

        logger.info(f"Trade manager initialized (simulation mode: {simulate})")

    def process_signal(self, signal: Signal) -> Optional[str]:
        """
        Process a trading signal and execute a trade if appropriate.

        Args:
            signal: Trading signal

        Returns:
            Trade ID if a trade was executed, None otherwise
        """
        # Get account information
        try:
            account_info = self.api_client.account.get_wallet_balance()

            # The wallet balance API returns a more complex structure, so we need to handle it properly
            account_balance = 0.0
            if account_info and "list" in account_info:
                # Find the first account with balance
                for account in account_info["list"]:
                    if "totalWalletBalance" in account:
                        account_balance = float(account["totalWalletBalance"])
                        break
        except Exception as e:
            logger.warning(f"Could not get account balance: {e}")
            # Use a default balance for simulation
            account_balance = 10000.0

        # Process the signal using the trade lifecycle component
        trade_info = process_trading_signal(
            signal=signal,
            account_balance=account_balance,
            risk_manager=self.risk_manager,
            existing_positions=self.active_trades,
        )

        if not trade_info:
            return None

        # Create a new trade
        trade = Trade(
            id=trade_info["id"],
            symbol=signal.symbol,
            direction=trade_info["direction"],
            entry_price=trade_info["entry_price"],
            target_price=trade_info["target_price"],
            stop_loss=trade_info["stop_loss"],
            size=trade_info["size"],
            entry_time=datetime.now(),
            status=TradeStatus.PENDING,
            metadata={
                "signal_id": signal.id,
                "signal_strength": signal.strength,
                "signal_type": signal.signal_type.name,
                "strategy": signal.metadata.get("strategy", "unknown"),
            },
        )

        # Execute the trade
        success = self.execute_trade(trade)

        if success:
            # Add to active trades
            self.trades[trade.id] = trade
            self.active_trades[trade.id] = trade
            logger.info(f"Trade executed: {trade.id}, {trade.direction} {trade.symbol}")
            return trade.id
        else:
            logger.warning(f"Failed to execute trade for signal: {signal.id}")
            return None

    def process_signals(self, signals: List[Signal]) -> List[Dict[str, Any]]:
        """
        Process multiple trading signals and execute trades if appropriate.

        Args:
            signals: List of trading signals

        Returns:
            List of executed trade information dictionaries
        """
        executed_trades = []

        # Log the signals
        logger.info(f"Processing {len(signals)} signals...")

        # Process each signal
        for signal in signals:
            try:
                # Process the signal
                trade_id = self.process_signal(signal)

                # If a trade was executed, add it to the list
                if trade_id:
                    trade = self.get_trade_by_id(trade_id)
                    if trade:
                        executed_trades.append(trade.to_dict())
                        logger.info(
                            f"Successfully executed trade {trade_id} for signal: {signal.signal_type.name} on {signal.symbol}"
                        )
                    else:
                        logger.warning(f"Trade {trade_id} not found after execution")
            except Exception as e:
                logger.error(f"Error processing signal {signal.id}: {str(e)}")

        # Log the results
        if executed_trades:
            logger.info(
                f"Executed {len(executed_trades)} trades out of {len(signals)} signals"
            )
        else:
            logger.info(f"No trades executed from {len(signals)} signals")

        return executed_trades

    def execute_trade(self, trade: Trade) -> bool:
        """
        Execute a trade by placing orders.

        Args:
            trade: Trade to execute

        Returns:
            True if trade was executed successfully, False otherwise
        """
        result = execute_trade_entry(
            trade=trade,
            api_client=self.api_client,
            risk_manager=self.risk_manager,
            simulate=self.simulate,
            simulation_delay_sec=self.simulation_delay_sec,
        )

        if result:
            logger.info(f"Trade executed successfully: {trade.id}")
            return True
        else:
            logger.error(f"Failed to execute trade: {trade.id}")
            return False

    def update_active_trades(self, market_data: Dict[str, Any]) -> None:
        """
        Update all active trades with current market data.

        Args:
            market_data: Current market data dictionary keyed by symbol
        """
        current_time = datetime.now()

        for trade_id, trade in list(self.active_trades.items()):
            symbol = trade.symbol

            # Skip if no market data for this symbol
            if symbol not in market_data:
                continue

            # Get current price
            current_price = float(market_data[symbol].get("price", 0))

            # Update trade market data
            update_position_market_data(trade, current_price, current_time)

            # Check for stop loss hit
            if (
                trade.side == OrderSide.BUY and current_price <= trade.stop_loss_price
            ) or (
                trade.side == OrderSide.SELL and current_price >= trade.stop_loss_price
            ):
                self._close_trade(trade, current_price, current_time, "stop_loss")

            # Check for take profit hit
            elif (
                trade.side == OrderSide.BUY and current_price >= trade.take_profit_price
            ) or (
                trade.side == OrderSide.SELL
                and current_price <= trade.take_profit_price
            ):
                self._close_trade(trade, current_price, current_time, "take_profit")

            # Update trailing stop if needed
            elif self.risk_manager.should_use_trailing_stop(
                calculate_unrealized_profit_pct(trade, current_price)
            ):
                # Calculate new trailing stop
                new_stop_price = self.risk_manager.calculate_trailing_stop(
                    current_price=current_price,
                    entry_price=trade.entry_price,
                    is_long=(trade.side == OrderSide.BUY),
                    highest_price=trade.highest_price_reached,
                    lowest_price=trade.lowest_price_reached,
                )

                # Only move stop loss in the favorable direction
                if (
                    trade.side == OrderSide.BUY
                    and new_stop_price > trade.stop_loss_price
                ) or (
                    trade.side == OrderSide.SELL
                    and new_stop_price < trade.stop_loss_price
                ):
                    logger.info(
                        f"Updating trailing stop for {trade.id} from {trade.stop_loss_price} to {new_stop_price}"
                    )
                    trade.stop_loss_price = new_stop_price
                    update_stop_loss_order(trade, self.api_client, self.simulate)

    def _close_trade(
        self, trade: Trade, exit_price: float, exit_time: datetime, exit_reason: str
    ) -> None:
        """
        Close a trade at the specified price.

        Args:
            trade: Trade to close
            exit_price: Exit price
            exit_time: Exit timestamp
            exit_reason: Reason for exiting
        """
        # Close the trade
        close_trade_at_market(
            trade=trade,
            exit_price=exit_price,
            exit_time=exit_time,
            exit_reason=exit_reason,
            api_client=self.api_client,
            risk_manager=self.risk_manager,
            simulate=self.simulate,
        )

        # Move from active to completed trades
        if trade.id in self.active_trades:
            del self.active_trades[trade.id]
            self.completed_trades[trade.id] = trade

    def get_trade_summary(self) -> Dict[str, Any]:
        """
        Get summary of all trades.

        Returns:
            Trade summary dictionary
        """
        return get_trade_summary(
            trades=self.trades,
            active_trades=self.active_trades,
            completed_trades=self.completed_trades,
        )

    def get_trade_history_df(self) -> pd.DataFrame:
        """
        Get trade history as a DataFrame.

        Returns:
            DataFrame with trade history
        """
        return get_trade_history_dataframe(self.trades)

    def save_trade_history(self, file_path: str) -> None:
        """
        Save trade history to CSV file.

        Args:
            file_path: Path to save the CSV file
        """
        save_trade_history(self.trades, file_path)

    def get_trade_by_id(self, trade_id: str) -> Optional[Trade]:
        """
        Get trade by ID.

        Args:
            trade_id: Trade ID

        Returns:
            Trade object or None if not found
        """
        return self.trades.get(trade_id)
