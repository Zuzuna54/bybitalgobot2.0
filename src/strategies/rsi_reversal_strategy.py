"""
RSI Reversal Strategy for the Algorithmic Trading System

This strategy generates buy signals when the RSI indicator crosses above the oversold level
and sell signals when it crosses below the overbought level, with additional confirmation filters.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

import pandas as pd
import numpy as np
from loguru import logger

from src.strategies.base_strategy import BaseStrategy, Signal, SignalType
from src.indicators.indicator_manager import IndicatorManager


class RSIReversalStrategy(BaseStrategy):
    """
    RSI Reversal Strategy.
    
    This strategy identifies potential trend reversals using the RSI indicator:
    - Buy Signal: RSI crosses above oversold level with price confirmation
    - Sell Signal: RSI crosses below overbought level with price confirmation
    
    Additional filters include volume confirmation and trend alignment.
    """
    
    def __init__(self, config: Dict[str, Any], indicator_manager: IndicatorManager):
        """
        Initialize the RSI Reversal Strategy.
        
        Args:
            config: Strategy configuration parameters
            indicator_manager: Indicator manager instance
        """
        super().__init__(config, indicator_manager)
        self.name = "rsi_reversal"
        self.description = "RSI Reversal with Price and Volume Confirmation"
        
        # Default configuration values
        self.rsi_length = config.get("rsi_length", 14)
        self.rsi_overbought = config.get("rsi_overbought", 70)
        self.rsi_oversold = config.get("rsi_oversold", 30)
        self.ema_fast_length = config.get("ema_fast_length", 8)
        self.ema_slow_length = config.get("ema_slow_length", 21)
        self.volume_ma_length = config.get("volume_ma_length", 20)
        self.volume_filter = config.get("volume_filter", True)
        self.volume_threshold = config.get("volume_threshold", 1.3)  # Volume must be 1.3x the average
        
        # For divergence detection
        self.detect_divergence = config.get("detect_divergence", True)
        self.divergence_lookback = config.get("divergence_lookback", 5)
        
        # For stop loss and take profit calculation
        self.atr_length = config.get("atr_length", 14)
        self.atr_stop_multiplier = config.get("atr_stop_multiplier", 2.0)
        self.risk_reward_ratio = config.get("risk_reward_ratio", 2.0)
        
        # Initialize indicators
        self._init_indicators()
    
    def _init_indicators(self) -> None:
        """Initialize the indicators required for this strategy."""
        # Add RSI
        self.indicators.add_indicator(
            "rsi",
            {
                "length": self.rsi_length
            }
        )
        
        # Add EMAs for trend confirmation
        self.indicators.add_indicator(
            "ema",
            {
                "length": self.ema_fast_length,
                "target_column": "close",
                "output_name": "ema_fast"
            }
        )
        
        self.indicators.add_indicator(
            "ema",
            {
                "length": self.ema_slow_length,
                "target_column": "close",
                "output_name": "ema_slow"
            }
        )
        
        # Add Volume MA
        self.indicators.add_indicator(
            "volume_sma",
            {
                "length": self.volume_ma_length,
                "target_column": "volume"
            }
        )
        
        # Add ATR for stop loss calculation
        self.indicators.add_indicator(
            "atr",
            {
                "length": self.atr_length
            }
        )
    
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        Generate trading signals based on RSI reversals.
        
        Args:
            data: Market data DataFrame with OHLCV data and indicators
            
        Returns:
            List of generated trading signals
        """
        signals = []
        
        # Ensure we have enough data
        if len(data) < self.rsi_length + 5:
            logger.warning(f"Not enough data for RSI Reversal strategy (need at least {self.rsi_length + 5} bars)")
            return signals
        
        # Get the last two data points
        current = data.iloc[-1]
        previous = data.iloc[-2]
        
        # Get indicator values
        current_rsi = current.get("rsi")
        previous_rsi = previous.get("rsi")
        current_close = current.get("close")
        previous_close = previous.get("close")
        current_volume = current.get("volume")
        volume_sma = current.get("volume_sma")
        ema_fast = current.get("ema_fast")
        ema_slow = current.get("ema_slow")
        
        # Skip if missing any required indicators
        if (current_rsi is None or previous_rsi is None or 
            current_close is None or ema_fast is None or ema_slow is None):
            logger.warning(f"Missing indicators for RSI Reversal strategy")
            return signals
        
        # Prepare signal metadata
        timestamp = pd.to_datetime(current.name)
        symbol = self.config.get("symbol", "UNKNOWN")
        
        # Check volume condition if enabled
        volume_condition = True
        if self.volume_filter and volume_sma is not None and current_volume is not None:
            volume_condition = current_volume > (volume_sma * self.volume_threshold)
        
        # Check for RSI oversold reversal (buy signal)
        if (previous_rsi <= self.rsi_oversold and 
            current_rsi > self.rsi_oversold and 
            current_close > previous_close and 
            volume_condition):
            
            # Calculate signal strength
            strength = self._calculate_buy_signal_strength(data)
            
            # Check for bullish RSI divergence if enabled
            divergence_found = False
            if self.detect_divergence:
                divergence_found = self._check_bullish_divergence(data)
                if divergence_found:
                    strength = min(strength + 0.15, 1.0)  # Increase strength if divergence found
            
            # Create signal
            signals.append(
                Signal(
                    signal_type=SignalType.BUY,
                    symbol=symbol,
                    timestamp=timestamp,
                    price=current_close,
                    strength=strength,
                    metadata={
                        "strategy_name": self.name,
                        "indicators": {
                            "rsi": current_rsi,
                            "ema_fast": ema_fast,
                            "ema_slow": ema_slow,
                            "volume_ratio": current_volume / volume_sma if volume_sma else 1.0,
                            "atr": current.get("atr", 0)
                        },
                        "divergence_found": divergence_found,
                        "reason": "RSI crossed above oversold level with price confirmation"
                    }
                )
            )
            
            logger.info(f"Generated BUY signal for {symbol} at {current_close} (RSI: {current_rsi:.2f}, strength: {strength:.2f})")
        
        # Check for RSI overbought reversal (sell signal)
        elif (previous_rsi >= self.rsi_overbought and 
              current_rsi < self.rsi_overbought and 
              current_close < previous_close and 
              volume_condition):
            
            # Calculate signal strength
            strength = self._calculate_sell_signal_strength(data)
            
            # Check for bearish RSI divergence if enabled
            divergence_found = False
            if self.detect_divergence:
                divergence_found = self._check_bearish_divergence(data)
                if divergence_found:
                    strength = min(strength + 0.15, 1.0)  # Increase strength if divergence found
            
            # Create signal
            signals.append(
                Signal(
                    signal_type=SignalType.SELL,
                    symbol=symbol,
                    timestamp=timestamp,
                    price=current_close,
                    strength=strength,
                    metadata={
                        "strategy_name": self.name,
                        "indicators": {
                            "rsi": current_rsi,
                            "ema_fast": ema_fast,
                            "ema_slow": ema_slow,
                            "volume_ratio": current_volume / volume_sma if volume_sma else 1.0,
                            "atr": current.get("atr", 0)
                        },
                        "divergence_found": divergence_found,
                        "reason": "RSI crossed below overbought level with price confirmation"
                    }
                )
            )
            
            logger.info(f"Generated SELL signal for {symbol} at {current_close} (RSI: {current_rsi:.2f}, strength: {strength:.2f})")
        
        return signals
    
    def _calculate_buy_signal_strength(self, data: pd.DataFrame) -> float:
        """
        Calculate buy signal strength based on RSI and trend alignment.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Signal strength (0.0 to 1.0)
        """
        current = data.iloc[-1]
        
        # Base strength starts at 0.7
        strength = 0.7
        
        # RSI component: stronger if RSI was deeper oversold
        rsi_values = data["rsi"].tail(3)
        min_rsi = min(rsi_values)
        
        # If RSI went extremely low, increase strength
        if min_rsi < 20:
            strength += 0.1
        
        # Trend component: check if price is above fast EMA
        if current["close"] > current["ema_fast"]:
            strength += 0.05
        
        # Volume component: stronger if volume is well above average
        if "volume" in data.columns and "volume_sma" in data.columns:
            volume_ratio = current["volume"] / current["volume_sma"] if current["volume_sma"] > 0 else 1.0
            if volume_ratio > 2.0:
                strength += 0.1
            elif volume_ratio > 1.5:
                strength += 0.05
        
        # Cap strength at 1.0
        return min(strength, 1.0)
    
    def _calculate_sell_signal_strength(self, data: pd.DataFrame) -> float:
        """
        Calculate sell signal strength based on RSI and trend alignment.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Signal strength (0.0 to 1.0)
        """
        current = data.iloc[-1]
        
        # Base strength starts at 0.7
        strength = 0.7
        
        # RSI component: stronger if RSI was extremely overbought
        rsi_values = data["rsi"].tail(3)
        max_rsi = max(rsi_values)
        
        # If RSI went extremely high, increase strength
        if max_rsi > 80:
            strength += 0.1
        
        # Trend component: check if price is below fast EMA
        if current["close"] < current["ema_fast"]:
            strength += 0.05
        
        # Volume component: stronger if volume is well above average
        if "volume" in data.columns and "volume_sma" in data.columns:
            volume_ratio = current["volume"] / current["volume_sma"] if current["volume_sma"] > 0 else 1.0
            if volume_ratio > 2.0:
                strength += 0.1
            elif volume_ratio > 1.5:
                strength += 0.05
        
        # Cap strength at 1.0
        return min(strength, 1.0)
    
    def _check_bullish_divergence(self, data: pd.DataFrame) -> bool:
        """
        Check for bullish RSI divergence.
        
        Bullish divergence occurs when price makes a lower low but RSI makes a higher low,
        indicating potential upward reversal.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            True if bullish divergence is detected
        """
        lookback = min(self.divergence_lookback, len(data) - 2)
        if lookback < 2:
            return False
        
        # Get the most recent data
        recent_data = data.tail(lookback + 1)
        
        # Find the lowest close prices and corresponding RSI values
        close_values = recent_data["close"].values
        rsi_values = recent_data["rsi"].values
        
        # Find the two lowest points in the close price
        lowest_indices = np.argsort(close_values)[:2]
        
        if len(lowest_indices) < 2:
            return False
        
        # Ensure the points are in chronological order
        lowest_indices = sorted(lowest_indices)
        
        # Check if price made a lower low but RSI made a higher low
        if (close_values[lowest_indices[1]] < close_values[lowest_indices[0]] and 
            rsi_values[lowest_indices[1]] > rsi_values[lowest_indices[0]]):
            return True
        
        return False
    
    def _check_bearish_divergence(self, data: pd.DataFrame) -> bool:
        """
        Check for bearish RSI divergence.
        
        Bearish divergence occurs when price makes a higher high but RSI makes a lower high,
        indicating potential downward reversal.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            True if bearish divergence is detected
        """
        lookback = min(self.divergence_lookback, len(data) - 2)
        if lookback < 2:
            return False
        
        # Get the most recent data
        recent_data = data.tail(lookback + 1)
        
        # Find the highest close prices and corresponding RSI values
        close_values = recent_data["close"].values
        rsi_values = recent_data["rsi"].values
        
        # Find the two highest points in the close price
        highest_indices = np.argsort(close_values)[-2:][::-1]
        
        if len(highest_indices) < 2:
            return False
        
        # Ensure the points are in chronological order
        highest_indices = sorted(highest_indices)
        
        # Check if price made a higher high but RSI made a lower high
        if (close_values[highest_indices[1]] > close_values[highest_indices[0]] and 
            rsi_values[highest_indices[1]] < rsi_values[highest_indices[0]]):
            return True
        
        return False
    
    def get_stop_loss_price(self, entry_price: float, is_long: bool, data: pd.DataFrame) -> float:
        """
        Calculate stop loss price based on ATR and market structure.
        
        Args:
            entry_price: Entry price for the trade
            is_long: Whether the position is long (True) or short (False)
            data: Market data DataFrame
            
        Returns:
            Stop loss price
        """
        # Get ATR value for volatility-based stop loss
        atr = data.iloc[-1].get("atr", 0)
        
        if is_long:
            # For long positions: use ATR-based stop or swing low, whichever is closer
            atr_stop = entry_price - (atr * self.atr_stop_multiplier)
            
            # Find the lowest low in the recent bars (swing low)
            lookback = min(10, len(data) - 1)
            swing_low = data["low"].iloc[-lookback:].min()
            
            # Use the higher of ATR stop or swing low
            return max(atr_stop, swing_low * 0.99)  # Add a 1% buffer below swing low
        else:
            # For short positions: use ATR-based stop or swing high, whichever is closer
            atr_stop = entry_price + (atr * self.atr_stop_multiplier)
            
            # Find the highest high in the recent bars (swing high)
            lookback = min(10, len(data) - 1)
            swing_high = data["high"].iloc[-lookback:].max()
            
            # Use the lower of ATR stop or swing high
            return min(atr_stop, swing_high * 1.01)  # Add a 1% buffer above swing high
    
    def calculate_take_profit(self, entry_price: float, stop_loss_price: float, is_long: bool) -> float:
        """
        Calculate take profit price based on risk-reward ratio.
        
        Args:
            entry_price: Entry price for the trade
            stop_loss_price: Stop loss price
            is_long: Whether the position is long (True) or short (False)
            
        Returns:
            Take profit price
        """
        risk = abs(entry_price - stop_loss_price)
        reward = risk * self.risk_reward_ratio
        
        if is_long:
            return entry_price + reward
        else:
            return entry_price - reward
    
    def should_adjust_position(self, position_data: Dict[str, Any], current_data: pd.DataFrame) -> Tuple[bool, float]:
        """
        Determine if position size should be adjusted based on RSI extremes.
        
        Args:
            position_data: Current position data
            current_data: Current market data
            
        Returns:
            Tuple of (should_adjust, adjustment_factor)
        """
        # If RSI is reaching extreme levels, consider adjusting the position
        current_rsi = current_data.iloc[-1].get("rsi", 50)
        
        is_long = position_data.get("side") == "Buy"
        
        if is_long:
            # For long positions, reduce size if RSI gets extremely overbought
            if current_rsi > 80:
                return True, 0.7  # Reduce position by 30%
        else:
            # For short positions, reduce size if RSI gets extremely oversold
            if current_rsi < 20:
                return True, 0.7  # Reduce position by 30%
        
        return False, 1.0  # No adjustment needed 