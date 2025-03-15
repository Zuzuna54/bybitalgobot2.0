"""
Base Strategy Class for the Algorithmic Trading System

This module defines the base class for all trading strategies used in the system.
Each specific strategy will inherit from this base class.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Union
from enum import Enum

import pandas as pd
import numpy as np
from loguru import logger

from src.indicators.indicator_manager import IndicatorManager


class SignalType(Enum):
    """Enum representing different types of trading signals."""
    BUY = 1
    SELL = -1
    NEUTRAL = 0


class Signal:
    """Class representing a trading signal."""
    
    def __init__(
        self,
        signal_type: SignalType,
        symbol: str,
        timestamp: pd.Timestamp,
        price: float,
        strength: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a trading signal.
        
        Args:
            signal_type: Type of signal (BUY, SELL, NEUTRAL)
            symbol: Trading pair symbol
            timestamp: Time when the signal was generated
            price: Price at signal generation
            strength: Signal strength (0.0 to 1.0)
            metadata: Additional signal information
        """
        self.signal_type = signal_type
        self.symbol = symbol
        self.timestamp = timestamp
        self.price = price
        self.strength = min(max(strength, 0.0), 1.0)  # Clamp between 0.0 and 1.0
        self.metadata = metadata or {}
    
    def __str__(self) -> str:
        """Get string representation of the signal."""
        return (f"Signal({self.signal_type.name}, {self.symbol}, {self.timestamp}, "
                f"price={self.price}, strength={self.strength:.2f})")


class BaseStrategy(ABC):
    """Base class for all trading strategies."""
    
    def __init__(
        self,
        name: str,
        config: Dict[str, Any],
        indicator_manager: IndicatorManager
    ):
        """
        Initialize the strategy.
        
        Args:
            name: Strategy name
            config: Strategy configuration
            indicator_manager: Indicator manager for technical indicators
        """
        self.name = name
        self.config = config
        self.indicator_manager = indicator_manager
        self.timeframe = config.get('timeframe', '1h')
        self.is_active = config.get('is_active', True)
        self.parameters = config.get('parameters', {})
        
        # Configuration validation
        self._validate_config()
        
        # Performance metrics
        self.signals: List[Signal] = []
        self.performance_metrics: Dict[str, float] = {
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'average_profit': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
        }
        
        # Strategy-specific fields
        self.strategy_indicators: List[str] = []
        self.required_columns: List[str] = []
        
        # Init strategy-specific indicators
        self._init_indicators()
        
        logger.info(f"Initialized strategy: {self.name}")
    
    def _validate_config(self) -> None:
        """
        Validate strategy configuration.
        
        Raises:
            ValueError: If configuration is invalid
        """
        if not self.name:
            raise ValueError("Strategy name cannot be empty")
            
        if 'parameters' not in self.config:
            raise ValueError("Strategy configuration must include parameters")
    
    @abstractmethod
    def _init_indicators(self) -> None:
        """
        Initialize strategy-specific indicators.
        
        This method should configure any indicators required by the strategy
        using the indicator_manager.
        """
        pass
    
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for strategy by applying required indicators.
        
        Args:
            data: Raw price data
            
        Returns:
            Prepared data with indicators applied
        """
        # Apply required indicators
        prepared_data = self.indicator_manager.apply_indicators(data, self.strategy_indicators)
        
        # Validate that all required columns are present
        missing_columns = [col for col in self.required_columns if col not in prepared_data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns for strategy {self.name}: {missing_columns}")
        
        return prepared_data
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        Generate trading signals based on the prepared data.
        
        Args:
            data: Prepared data with indicators
            
        Returns:
            List of trading signals
        """
        pass
    
    def analyze_performance(self, signals: List[Signal], price_data: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze strategy performance based on signals and actual price data.
        
        Args:
            signals: List of signals generated by the strategy
            price_data: Historical price data after signals
            
        Returns:
            Dictionary of performance metrics
        """
        # Placeholder for actual implementation
        return self.performance_metrics
    
    def get_stop_loss_price(self, data: pd.DataFrame, entry_price: float, is_long: bool) -> float:
        """
        Calculate the stop loss price for a trade.
        
        Args:
            data: Prepared data with indicators
            entry_price: Entry price of the trade
            is_long: Whether the trade is long or short
            
        Returns:
            Stop loss price
        """
        # Default implementation (can be overridden by subclasses)
        if 'atr' in data.columns:
            # Use ATR-based stop loss
            atr_value = data['atr'].iloc[-1]
            atr_multiplier = self.parameters.get('stop_loss_atr_multiplier', 2.0)
            
            if is_long:
                return entry_price - (atr_value * atr_multiplier)
            else:
                return entry_price + (atr_value * atr_multiplier)
        else:
            # Use percentage-based stop loss
            stop_percentage = self.parameters.get('stop_loss_percentage', 2.0) / 100.0
            
            if is_long:
                return entry_price * (1 - stop_percentage)
            else:
                return entry_price * (1 + stop_percentage)
    
    def get_take_profit_price(self, data: pd.DataFrame, entry_price: float, is_long: bool, stop_loss_price: float) -> float:
        """
        Calculate the take profit price for a trade.
        
        Args:
            data: Prepared data with indicators
            entry_price: Entry price of the trade
            is_long: Whether the trade is long or short
            stop_loss_price: Stop loss price
            
        Returns:
            Take profit price
        """
        # Default implementation (can be overridden by subclasses)
        risk = abs(entry_price - stop_loss_price)
        risk_reward_ratio = self.parameters.get('risk_reward_ratio', 2.0)
        
        if is_long:
            return entry_price + (risk * risk_reward_ratio)
        else:
            return entry_price - (risk * risk_reward_ratio)
    
    def should_adjust_position(self, data: pd.DataFrame, current_position: Dict[str, Any]) -> Tuple[bool, Optional[float]]:
        """
        Determine if an existing position should be adjusted.
        
        Args:
            data: Prepared data with indicators
            current_position: Current position information
            
        Returns:
            Tuple of (should_adjust, new_size_if_adjusting)
        """
        # Default implementation - no adjustment
        return False, None
    
    def should_use_trailing_stop(self, data: pd.DataFrame, current_position: Dict[str, Any]) -> bool:
        """
        Determine if trailing stop should be used for a position.
        
        Args:
            data: Prepared data with indicators
            current_position: Current position information
            
        Returns:
            True if trailing stop should be used
        """
        # Default implementation - use trailing stop if profit > 1 risk unit
        entry_price = current_position['entry_price']
        current_price = data['close'].iloc[-1]
        stop_price = current_position['stop_loss']
        initial_risk = abs(entry_price - stop_price)
        
        is_long = current_position['side'] == 'buy'
        current_profit = (current_price - entry_price) if is_long else (entry_price - current_price)
        
        # Enable trailing stop if profit > initial risk
        return current_profit > initial_risk
    
    def calculate_trailing_stop(self, data: pd.DataFrame, current_position: Dict[str, Any]) -> float:
        """
        Calculate the trailing stop price for a position.
        
        Args:
            data: Prepared data with indicators
            current_position: Current position information
            
        Returns:
            Trailing stop price
        """
        # Default implementation - ATR-based trailing stop if available
        is_long = current_position['side'] == 'buy'
        current_price = data['close'].iloc[-1]
        
        if 'atr' in data.columns:
            atr_value = data['atr'].iloc[-1]
            atr_multiplier = self.parameters.get('trailing_stop_atr_multiplier', 2.0)
            
            if is_long:
                return current_price - (atr_value * atr_multiplier)
            else:
                return current_price + (atr_value * atr_multiplier)
        else:
            # Use percentage-based trailing stop
            trail_percentage = self.parameters.get('trailing_stop_percentage', 1.5) / 100.0
            
            if is_long:
                return current_price * (1 - trail_percentage)
            else:
                return current_price * (1 + trail_percentage)
    
    def calculate_position_size(self, account_balance: float, risk_per_trade: float, entry_price: float, stop_loss_price: float) -> float:
        """
        Calculate the position size based on risk management parameters.
        
        Args:
            account_balance: Current account balance
            risk_per_trade: Percentage of account to risk per trade
            entry_price: Entry price of the trade
            stop_loss_price: Stop loss price
            
        Returns:
            Position size in base currency
        """
        # Calculate the risk amount in quote currency
        risk_amount = account_balance * (risk_per_trade / 100.0)
        
        # Calculate the price difference between entry and stop loss
        price_difference = abs(entry_price - stop_loss_price)
        
        # Calculate position size (risk amount / price difference)
        if price_difference > 0:
            position_size = risk_amount / price_difference
        else:
            # Fallback if stop loss is at entry price (shouldn't happen)
            logger.warning("Stop loss is equal to entry price. Using default position size.")
            position_size = risk_amount / (entry_price * 0.01)  # Assume 1% risk
        
        return position_size 