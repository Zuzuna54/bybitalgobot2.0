"""
MACD Trend Following Strategy for the Algorithmic Trading System

This strategy generates buy signals when the MACD line crosses above the signal line,
and sell signals when the MACD line crosses below the signal line.
It also filters signals based on the MACD histogram direction.
"""

from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np
from loguru import logger

from src.strategies.base_strategy import BaseStrategy, Signal, SignalType
from src.indicators.indicator_manager import IndicatorManager


class MACDTrendFollowingStrategy(BaseStrategy):
    """MACD Trend Following trading strategy."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        indicator_manager: IndicatorManager
    ):
        """
        Initialize the MACD Trend Following strategy.
        
        Args:
            config: Strategy configuration
            indicator_manager: Indicator manager for technical indicators
        """
        super().__init__('macd_trend_following', config, indicator_manager)
    
    def _init_indicators(self) -> None:
        """Initialize strategy-specific indicators."""
        # Get parameters
        fast_length = self.parameters.get('fast_length', 12)
        slow_length = self.parameters.get('slow_length', 26)
        signal_length = self.parameters.get('signal_length', 9)
        histogram_threshold = self.parameters.get('histogram_increasing_threshold', 0.0)
        
        # Define indicator names
        prefix = f"{self.name}_"
        macd_name = f"{prefix}macd"
        volume_name = f"{prefix}volume"
        
        # Check if indicators already exist, otherwise add them
        if macd_name not in self.indicator_manager.indicators:
            self.indicator_manager.add_indicator(
                macd_name, 'macd', {
                    'fast_period': fast_length,
                    'slow_period': slow_length,
                    'signal_period': signal_length
                }
            )
        
        if volume_name not in self.indicator_manager.indicators:
            self.indicator_manager.add_indicator(
                volume_name, 'obv', {'ma_period': 20}
            )
        
        # Store indicator names for this strategy
        self.strategy_indicators = [
            macd_name,
            volume_name
        ]
        
        # Define required columns for signal generation
        self.required_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'macd_line', 'macd_signal', 'macd_histogram',
            'obv', 'obv_ma'
        ]
        
        logger.info(f"Initialized indicators for MACD Trend Following strategy with fast={fast_length}, slow={slow_length}, signal={signal_length}")
    
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        Generate trading signals based on MACD crossovers.
        
        Args:
            data: Prepared data with indicators
            
        Returns:
            List of trading signals
        """
        signals = []
        
        # Make sure we have enough data
        if len(data) < 2:
            logger.warning("Not enough data to generate signals")
            return signals
        
        # Get the latest complete candle (second to last row)
        # Using second to last row to avoid acting on incomplete candles
        latest_complete_candle = data.iloc[-2]
        previous_candle = data.iloc[-3] if len(data) > 2 else None
        current_timestamp = latest_complete_candle.name
        current_price = latest_complete_candle['close']
        symbol = self.parameters.get('symbol', 'BTCUSDT')  # Default symbol
        
        # Skip if we don't have enough data to check previous values
        if previous_candle is None:
            return signals
        
        # Check for MACD line crossing above signal line (bullish)
        macd_cross_up = (
            latest_complete_candle['macd_line'] > latest_complete_candle['macd_signal'] and
            previous_candle['macd_line'] <= previous_candle['macd_signal']
        )
        
        # Check for MACD line crossing below signal line (bearish)
        macd_cross_down = (
            latest_complete_candle['macd_line'] < latest_complete_candle['macd_signal'] and
            previous_candle['macd_line'] >= previous_candle['macd_signal']
        )
        
        # Check if histogram is increasing or decreasing
        histogram_increasing = latest_complete_candle['macd_histogram'] > previous_candle['macd_histogram']
        histogram_decreasing = latest_complete_candle['macd_histogram'] < previous_candle['macd_histogram']
        
        # Volume confirmation
        volume_increasing = latest_complete_candle['volume'] > data['volume'].rolling(10).mean().iloc[-2]
        
        # Generate buy signal on bullish crossover with increasing histogram
        if macd_cross_up and histogram_increasing and volume_increasing:
            # Calculate signal strength based on histogram size and volume
            histogram_strength = min(abs(latest_complete_candle['macd_histogram']) / 0.01, 1.0)
            volume_ratio = latest_complete_candle['volume'] / data['volume'].rolling(20).mean().iloc[-2]
            volume_ratio = min(volume_ratio / 2.0, 1.0)  # Cap at 2x average
            
            signal_strength = 0.5 + (0.25 * histogram_strength) + (0.25 * volume_ratio)
            
            # Create signal metadata
            metadata = {
                'macd_line': latest_complete_candle['macd_line'],
                'macd_signal': latest_complete_candle['macd_signal'],
                'macd_histogram': latest_complete_candle['macd_histogram'],
                'volume_ratio': volume_ratio,
                'strategy': self.name,
                'strategy_name': self.name,
                'contributing_strategies': [self.name]
            }
            
            # Create and append signal
            signal = Signal(
                signal_type=SignalType.BUY,
                symbol=symbol,
                timestamp=current_timestamp,
                price=current_price,
                strength=signal_strength,
                metadata=metadata
            )
            signals.append(signal)
            logger.info(f"Generated BUY signal: {symbol} at {current_price} (strength: {signal_strength:.2f})")
            
        # Generate sell signal on bearish crossover with decreasing histogram
        elif macd_cross_down and histogram_decreasing and volume_increasing:
            # Calculate signal strength based on histogram size and volume
            histogram_strength = min(abs(latest_complete_candle['macd_histogram']) / 0.01, 1.0)
            volume_ratio = latest_complete_candle['volume'] / data['volume'].rolling(20).mean().iloc[-2]
            volume_ratio = min(volume_ratio / 2.0, 1.0)  # Cap at 2x average
            
            signal_strength = 0.5 + (0.25 * histogram_strength) + (0.25 * volume_ratio)
            
            # Create signal metadata
            metadata = {
                'macd_line': latest_complete_candle['macd_line'],
                'macd_signal': latest_complete_candle['macd_signal'],
                'macd_histogram': latest_complete_candle['macd_histogram'],
                'volume_ratio': volume_ratio,
                'strategy': self.name,
                'strategy_name': self.name,
                'contributing_strategies': [self.name]
            }
            
            # Create and append signal
            signal = Signal(
                signal_type=SignalType.SELL,
                symbol=symbol,
                timestamp=current_timestamp,
                price=current_price,
                strength=signal_strength,
                metadata=metadata
            )
            signals.append(signal)
            logger.info(f"Generated SELL signal: {symbol} at {current_price} (strength: {signal_strength:.2f})")
        
        return signals
    
    def get_stop_loss_price(self, data: pd.DataFrame, entry_price: float, is_long: bool) -> float:
        """
        Calculate the stop loss price for MACD trend following strategy.
        
        Args:
            data: Prepared data with indicators
            entry_price: Entry price of the trade
            is_long: Whether the trade is long or short
            
        Returns:
            Stop loss price
        """
        # For MACD trend following, use recent swing levels or ATR-based stop loss
        if 'atr' in data.columns:
            # ATR-based stop loss (multiply by different factors based on trend strength)
            atr_value = data['atr'].iloc[-1]
            
            # Adjust multiplier based on MACD strength
            macd_strength = abs(data['macd_line'].iloc[-1] - data['macd_signal'].iloc[-1])
            normalized_strength = min(macd_strength / 0.01, 3.0)  # Cap at 3.0
            atr_multiplier = 1.5 + (normalized_strength * 0.5)  # 1.5 to 3.0 range
            
            if is_long:
                return entry_price - (atr_value * atr_multiplier)
            else:
                return entry_price + (atr_value * atr_multiplier)
        else:
            # Percentage-based stop loss
            # Use tighter stop for stronger signals
            macd_strength = abs(data['macd_line'].iloc[-1] - data['macd_signal'].iloc[-1])
            normalized_strength = min(macd_strength / 0.01, 1.0)
            stop_percentage = 0.03 - (normalized_strength * 0.01)  # 2% to 3% range
            
            if is_long:
                return entry_price * (1 - stop_percentage)
            else:
                return entry_price * (1 + stop_percentage)
    
    def should_adjust_position(self, data: pd.DataFrame, current_position: Dict[str, Any]) -> tuple:
        """
        Determine if position size should be adjusted based on MACD trend strength.
        
        Args:
            data: Prepared data with indicators
            current_position: Current position information
            
        Returns:
            Tuple of (should_adjust, new_size_if_adjusting)
        """
        # Get current MACD values
        macd_line = data['macd_line'].iloc[-1]
        macd_signal = data['macd_signal'].iloc[-1]
        histogram = data['macd_histogram'].iloc[-1]
        
        # Determine trend strength from MACD values
        trend_strength = abs(macd_line - macd_signal)
        
        # Current position size
        current_size = current_position.get('size', 0)
        
        # Only adjust if significant change in trend strength
        if trend_strength > 0.02:  # Strong trend
            # Increase position size by 20% for strong trends
            new_size = current_size * 1.2
            return True, new_size
        elif trend_strength < 0.005:  # Weak trend
            # Decrease position size by 20% for weak trends
            new_size = current_size * 0.8
            return True, new_size
        
        # Check if histogram changes direction
        previous_histogram = data['macd_histogram'].iloc[-2]
        if (histogram > 0 and previous_histogram < 0) or (histogram < 0 and previous_histogram > 0):
            # Histogram changed direction, reduce position by 50%
            new_size = current_size * 0.5
            return True, new_size
        
        # No adjustment needed
        return False, None 