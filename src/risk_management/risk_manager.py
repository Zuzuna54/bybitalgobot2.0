"""
Risk Management Module for the Algorithmic Trading System

This module handles position sizing, risk control, and trade management to
ensure the trading system follows robust risk management practices.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import math

import pandas as pd
import numpy as np
from loguru import logger

from src.config.config_manager import RiskConfig


class RiskManager:
    """Manages risk parameters and position sizing for trades."""
    
    def __init__(self, risk_config: RiskConfig):
        """
        Initialize the risk manager.
        
        Args:
            risk_config: Risk management configuration
        """
        self.config = risk_config
        
        # Track performance metrics for circuit breaker
        self.consecutive_losses = 0
        self.max_daily_loss = 0.0
        self.position_history: List[Dict[str, Any]] = []
        self.daily_profit_loss = 0.0
        self.account_balance_history: List[float] = []
        
        logger.info("Risk manager initialized")
    
    def calculate_position_size(
        self,
        symbol: str,
        account_balance: float,
        entry_price: float,
        stop_loss_price: float,
        leverage: int = 1,
        min_order_qty: float = 0.0
    ) -> float:
        """
        Calculate position size based on risk parameters.
        
        Args:
            symbol: Trading pair symbol
            account_balance: Current account balance
            entry_price: Entry price for the trade
            stop_loss_price: Stop loss price for the trade
            leverage: Leverage to use (default 1)
            min_order_qty: Minimum order quantity for the symbol
            
        Returns:
            Position size in base currency
        """
        # Calculate the risk amount based on risk percentage
        risk_percentage = self.config.max_position_size_percent
        risk_amount = account_balance * (risk_percentage / 100.0)
        
        # Apply circuit breaker if needed
        if self.consecutive_losses >= self.config.circuit_breaker_consecutive_losses:
            # Reduce position size by 50% if circuit breaker is triggered
            risk_amount *= 0.5
            logger.warning(f"Circuit breaker applied - reducing position size by 50%")
        
        # Calculate price difference between entry and stop loss (as percentage)
        price_difference_pct = abs(entry_price - stop_loss_price) / entry_price
        
        # Calculate the position size in base currency
        if price_difference_pct > 0:
            # Apply leverage to increase position size (effectively reducing the equity at risk)
            position_size = (risk_amount * leverage) / (entry_price * price_difference_pct)
        else:
            # Fallback for invalid stop loss
            logger.warning("Invalid stop loss equals entry price. Using default risk of 1%")
            position_size = risk_amount / (entry_price * 0.01)
        
        # Make sure position size meets minimum order quantity
        if min_order_qty > 0 and position_size < min_order_qty:
            if min_order_qty * entry_price > account_balance * (risk_percentage / 100.0) * 2:
                # If min order size would risk more than 2x the intended risk, skip the trade
                logger.warning(f"Minimum order quantity too large for risk parameters - skipping trade")
                return 0.0
            
            logger.info(f"Adjusted position size from {position_size} to minimum {min_order_qty}")
            position_size = min_order_qty
        
        # Check that position doesn't exceed max open positions constraint
        if len(self.get_open_positions()) >= self.config.max_open_positions:
            logger.warning(f"Maximum open positions limit reached ({self.config.max_open_positions})")
            return 0.0
        
        # Round to appropriate precision for the symbol
        position_size = self._round_position_size(position_size, symbol)
        
        logger.info(f"Calculated position size: {position_size} (leverage: {leverage}x)")
        return position_size
    
    def _round_position_size(self, position_size: float, symbol: str) -> float:
        """
        Round position size to appropriate precision for the symbol.
        
        Args:
            position_size: Raw position size
            symbol: Trading pair symbol
            
        Returns:
            Rounded position size
        """
        # For BTC, round to 3 decimal places
        if symbol.startswith("BTC"):
            return math.floor(position_size * 1000) / 1000
        # For ETH, round to 2 decimal places
        elif symbol.startswith("ETH"):
            return math.floor(position_size * 100) / 100
        # For most other coins, round to 1 decimal place
        else:
            return math.floor(position_size * 10) / 10
    
    def update_trade_result(self, trade_result: Dict[str, Any]) -> None:
        """
        Update risk metrics based on a completed trade.
        
        Args:
            trade_result: Dictionary with trade information
        """
        # Extract trade information
        profit_loss = trade_result.get('realized_pnl', 0.0)
        
        # Update performance metrics
        self.position_history.append(trade_result)
        self.daily_profit_loss += profit_loss
        
        # Update consecutive losses counter
        if profit_loss < 0:
            self.consecutive_losses += 1
            logger.warning(f"Trade loss - consecutive losses: {self.consecutive_losses}")
        else:
            self.consecutive_losses = 0
            logger.info(f"Trade profit - consecutive losses reset to 0")
    
    def update_account_balance(self, balance: float) -> None:
        """
        Update the account balance history.
        
        Args:
            balance: Current account balance
        """
        self.account_balance_history.append(balance)
        
        # Check for maximum daily drawdown
        if len(self.account_balance_history) > 1:
            prev_balance = self.account_balance_history[-2]
            daily_change_pct = (balance - prev_balance) / prev_balance * 100
            
            if daily_change_pct < 0 and abs(daily_change_pct) > self.max_daily_loss:
                self.max_daily_loss = abs(daily_change_pct)
    
    def is_max_drawdown_breached(self) -> bool:
        """
        Check if maximum daily drawdown has been breached.
        
        Returns:
            True if maximum daily drawdown is breached
        """
        return self.max_daily_loss > self.config.max_daily_drawdown_percent
    
    def should_apply_circuit_breaker(self) -> bool:
        """
        Check if circuit breaker should be applied.
        
        Returns:
            True if circuit breaker should be applied
        """
        return self.consecutive_losses >= self.config.circuit_breaker_consecutive_losses
    
    def calculate_stop_loss(
        self,
        entry_price: float,
        is_long: bool,
        atr_value: Optional[float] = None
    ) -> float:
        """
        Calculate stop loss price based on ATR or percentage.
        
        Args:
            entry_price: Entry price for the trade
            is_long: Whether the trade is long or short
            atr_value: ATR value (if available)
            
        Returns:
            Stop loss price
        """
        if atr_value is not None:
            # ATR-based stop loss
            atr_multiplier = self.config.stop_loss_atr_multiplier
            
            if is_long:
                return entry_price - (atr_value * atr_multiplier)
            else:
                return entry_price + (atr_value * atr_multiplier)
        else:
            # Percentage-based stop loss (default 2%)
            stop_percentage = 0.02
            
            if is_long:
                return entry_price * (1 - stop_percentage)
            else:
                return entry_price * (1 + stop_percentage)
    
    def calculate_take_profit(
        self,
        entry_price: float,
        stop_loss_price: float,
        is_long: bool
    ) -> float:
        """
        Calculate take profit price based on risk-reward ratio.
        
        Args:
            entry_price: Entry price for the trade
            stop_loss_price: Stop loss price for the trade
            is_long: Whether the trade is long or short
            
        Returns:
            Take profit price
        """
        risk = abs(entry_price - stop_loss_price)
        risk_reward_ratio = self.config.take_profit_risk_reward_ratio
        
        if is_long:
            return entry_price + (risk * risk_reward_ratio)
        else:
            return entry_price - (risk * risk_reward_ratio)
    
    def should_use_trailing_stop(self, unrealized_profit_pct: float) -> bool:
        """
        Determine if trailing stop should be activated.
        
        Args:
            unrealized_profit_pct: Unrealized profit as a percentage
            
        Returns:
            True if trailing stop should be used
        """
        # Activate trailing stop if enabled and profit is at least 1%
        return self.config.use_trailing_stop and unrealized_profit_pct >= 1.0
    
    def calculate_trailing_stop(
        self,
        current_price: float,
        entry_price: float,
        is_long: bool,
        atr_value: Optional[float] = None,
        highest_price: Optional[float] = None,
        lowest_price: Optional[float] = None
    ) -> float:
        """
        Calculate trailing stop price.
        
        Args:
            current_price: Current price
            entry_price: Entry price for the trade
            is_long: Whether the trade is long or short
            atr_value: ATR value (if available)
            highest_price: Highest price since entry (for long positions)
            lowest_price: Lowest price since entry (for short positions)
            
        Returns:
            Trailing stop price
        """
        if atr_value is not None:
            # ATR-based trailing stop
            atr_multiplier = self.config.stop_loss_atr_multiplier
            
            if is_long:
                # For long positions, trail below price
                reference_price = highest_price if highest_price is not None else current_price
                return reference_price - (atr_value * atr_multiplier)
            else:
                # For short positions, trail above price
                reference_price = lowest_price if lowest_price is not None else current_price
                return reference_price + (atr_value * atr_multiplier)
        else:
            # Percentage-based trailing stop
            trail_percentage = 0.015  # 1.5%
            
            if is_long:
                reference_price = highest_price if highest_price is not None else current_price
                return reference_price * (1 - trail_percentage)
            else:
                reference_price = lowest_price if lowest_price is not None else current_price
                return reference_price * (1 + trail_percentage)
    
    def get_recommended_leverage(self, symbol: str, volatility: Optional[float] = None) -> int:
        """
        Get recommended leverage based on symbol and volatility.
        
        Args:
            symbol: Trading pair symbol
            volatility: Volatility estimate (if available)
            
        Returns:
            Recommended leverage value
        """
        default_leverage = self.config.default_leverage
        
        # If volatility data is available, adjust leverage
        if volatility is not None:
            # Reduce leverage for high volatility
            if volatility > 5.0:  # 5% daily volatility is high
                return max(1, default_leverage - 1)
            elif volatility < 1.0:  # 1% daily volatility is low
                return min(default_leverage + 1, 10)  # Don't exceed 10x
        
        # Symbol-specific adjustments
        if symbol.startswith("BTC"):
            # Bitcoin is generally less volatile than altcoins
            return min(default_leverage + 1, 10)
        elif symbol.startswith("ETH"):
            # Ethereum is somewhat less volatile than smaller altcoins
            return default_leverage
        else:
            # More conservative with altcoins
            return max(1, default_leverage - 1)
    
    def get_open_positions(self) -> List[Dict[str, Any]]:
        """
        Get list of currently open positions.
        
        Returns:
            List of open position dictionaries
        """
        # Filter position history to get only open positions
        open_positions = [
            position for position in self.position_history
            if position.get('status') == 'open'
        ]
        return open_positions
    
    def reset_daily_metrics(self) -> None:
        """Reset daily performance metrics."""
        self.daily_profit_loss = 0.0
        self.max_daily_loss = 0.0
        logger.info("Daily risk metrics reset")
    
    def should_take_trade(
        self,
        symbol: str,
        signal_strength: float,
        account_balance: float
    ) -> bool:
        """
        Determine if a trade should be taken based on risk parameters.
        
        Args:
            symbol: Trading pair symbol
            signal_strength: Signal strength (0.0 to 1.0)
            account_balance: Current account balance
            
        Returns:
            True if trade should be taken
        """
        # Check for max daily drawdown
        if self.is_max_drawdown_breached():
            logger.warning(f"Max daily drawdown breached ({self.max_daily_loss:.2f}% > {self.config.max_daily_drawdown_percent}%) - not taking new trades")
            return False
        
        # Check for maximum open positions
        if len(self.get_open_positions()) >= self.config.max_open_positions:
            logger.warning(f"Maximum open positions reached ({self.config.max_open_positions}) - not taking new trades")
            return False
        
        # Check for weak signals
        if signal_strength < 0.6:
            logger.info(f"Signal strength too low ({signal_strength:.2f}) - not taking trade")
            return False
        
        # Check for existing position in the same symbol
        for position in self.get_open_positions():
            if position.get('symbol') == symbol:
                logger.info(f"Already have position in {symbol} - not taking additional trade")
                return False
        
        # Check for circuit breaker
        if self.should_apply_circuit_breaker():
            # Still allow trades but with reduced size (handled in position sizing)
            logger.warning(f"Circuit breaker active after {self.consecutive_losses} consecutive losses")
        
        return True 