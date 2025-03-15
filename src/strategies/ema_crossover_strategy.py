"""
EMA Crossover Strategy for the Algorithmic Trading System

This strategy generates buy signals when a fast EMA crosses above a slow EMA,
and sell signals when the fast EMA crosses below the slow EMA.
Volume confirmation is used to filter signals.
"""

from typing import Dict, Any, List, Optional

import pandas as pd
import numpy as np
from loguru import logger

from src.strategies.base_strategy import BaseStrategy, Signal, SignalType
from src.indicators.indicator_manager import IndicatorManager


class EMACrossoverStrategy(BaseStrategy):
    """EMA Crossover trading strategy."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        indicator_manager: IndicatorManager
    ):
        """
        Initialize the EMA Crossover strategy.
        
        Args:
            config: Strategy configuration
            indicator_manager: Indicator manager for technical indicators
        """
        super().__init__('ema_crossover', config, indicator_manager)
    
    def _init_indicators(self) -> None:
        """Initialize strategy-specific indicators."""
        # Get parameters
        fast_period = self.parameters.get('fast_ema', 9)
        slow_period = self.parameters.get('slow_ema', 21)
        volume_threshold = self.parameters.get('volume_threshold', 1.5)
        
        # Define indicator names
        prefix = f"{self.name}_"
        fast_ema_name = f"{prefix}ema_fast"
        slow_ema_name = f"{prefix}ema_slow"
        crossover_name = f"{prefix}crossover"
        volume_name = f"{prefix}volume"
        
        # Check if indicators already exist, otherwise add them
        if fast_ema_name not in self.indicator_manager.indicators:
            self.indicator_manager.add_indicator(
                fast_ema_name, 'ma', {'period': fast_period, 'type': 'ema'}
            )
        
        if slow_ema_name not in self.indicator_manager.indicators:
            self.indicator_manager.add_indicator(
                slow_ema_name, 'ma', {'period': slow_period, 'type': 'ema'}
            )
        
        if crossover_name not in self.indicator_manager.indicators:
            self.indicator_manager.add_indicator(
                crossover_name, 'ma_crossover', {
                    'fast_period': fast_period,
                    'slow_period': slow_period,
                    'type': 'ema'
                }
            )
        
        if volume_name not in self.indicator_manager.indicators:
            self.indicator_manager.add_indicator(
                volume_name, 'obv', {'ma_period': 20}
            )
        
        # Store indicator names for this strategy
        self.strategy_indicators = [
            fast_ema_name,
            slow_ema_name,
            crossover_name,
            volume_name
        ]
        
        # Define required columns for signal generation
        self.required_columns = [
            'open', 'high', 'low', 'close', 'volume',
            f"ema_fast_{fast_period}", f"ema_slow_{slow_period}",
            'ma_crossover', 'obv', 'obv_ma'
        ]
        
        logger.info(f"Initialized indicators for EMA Crossover strategy with fast={fast_period}, slow={slow_period}")
    
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        Generate trading signals based on EMA crossovers.
        
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
        current_timestamp = latest_complete_candle.name
        current_price = latest_complete_candle['close']
        symbol = self.parameters.get('symbol', 'BTCUSDT')  # Default symbol
        
        # Check for crossover signal
        crossover_value = latest_complete_candle['ma_crossover']
        
        # Check volume confirmation (volume > recent average)
        volume_increasing = latest_complete_candle['obv'] > latest_complete_candle['obv_ma']
        
        # Generate buy signal on bullish crossover with volume confirmation
        if crossover_value == 1 and volume_increasing:
            # Calculate signal strength (0.5 to 1.0 based on volume)
            volume_ratio = latest_complete_candle['volume'] / data['volume'].rolling(20).mean().iloc[-2]
            volume_ratio = min(volume_ratio, 3.0)  # Cap at 3x average
            volume_strength = 0.5 + (0.5 * min(volume_ratio / 3.0, 1.0))
            
            # Create signal metadata
            metadata = {
                'fast_ema': latest_complete_candle[f"ema_fast_{self.parameters.get('fast_ema', 9)}"],
                'slow_ema': latest_complete_candle[f"ema_slow_{self.parameters.get('slow_ema', 21)}"],
                'volume_ratio': volume_ratio,
                'strategy': self.name
            }
            
            # Create and append signal
            signal = Signal(
                signal_type=SignalType.BUY,
                symbol=symbol,
                timestamp=current_timestamp,
                price=current_price,
                strength=volume_strength,
                metadata=metadata
            )
            signals.append(signal)
            logger.info(f"Generated BUY signal: {symbol} at {current_price} (strength: {volume_strength:.2f})")
            
        # Generate sell signal on bearish crossover with volume confirmation
        elif crossover_value == -1 and volume_increasing:
            # Calculate signal strength (0.5 to 1.0 based on volume)
            volume_ratio = latest_complete_candle['volume'] / data['volume'].rolling(20).mean().iloc[-2]
            volume_ratio = min(volume_ratio, 3.0)  # Cap at 3x average
            volume_strength = 0.5 + (0.5 * min(volume_ratio / 3.0, 1.0))
            
            # Create signal metadata
            metadata = {
                'fast_ema': latest_complete_candle[f"ema_fast_{self.parameters.get('fast_ema', 9)}"],
                'slow_ema': latest_complete_candle[f"ema_slow_{self.parameters.get('slow_ema', 21)}"],
                'volume_ratio': volume_ratio,
                'strategy': self.name
            }
            
            # Create and append signal
            signal = Signal(
                signal_type=SignalType.SELL,
                symbol=symbol,
                timestamp=current_timestamp,
                price=current_price,
                strength=volume_strength,
                metadata=metadata
            )
            signals.append(signal)
            logger.info(f"Generated SELL signal: {symbol} at {current_price} (strength: {volume_strength:.2f})")
        
        return signals
    
    def get_stop_loss_price(self, data: pd.DataFrame, entry_price: float, is_long: bool) -> float:
        """
        Calculate the stop loss price for EMA Crossover strategy.
        
        Args:
            data: Prepared data with indicators
            entry_price: Entry price of the trade
            is_long: Whether the trade is long or short
            
        Returns:
            Stop loss price
        """
        # For EMA Crossover, use the opposite EMA as a stop loss
        if is_long:
            # For long positions, stop loss below the slow EMA
            slow_ema = data[f"ema_slow_{self.parameters.get('slow_ema', 21)}"].iloc[-1]
            return min(slow_ema, entry_price * 0.97)  # Use the lower of slow EMA or 3% below entry
        else:
            # For short positions, stop loss above the slow EMA
            slow_ema = data[f"ema_slow_{self.parameters.get('slow_ema', 21)}"].iloc[-1]
            return max(slow_ema, entry_price * 1.03)  # Use the higher of slow EMA or 3% above entry
    
    def should_adjust_position(self, data: pd.DataFrame, current_position: Dict[str, Any]) -> tuple:
        """
        Determine if position size should be adjusted based on trend strength.
        
        Args:
            data: Prepared data with indicators
            current_position: Current position information
            
        Returns:
            Tuple of (should_adjust, new_size_if_adjusting)
        """
        # Get trend strength from the difference between EMAs
        fast_ema = data[f"ema_fast_{self.parameters.get('fast_ema', 9)}"].iloc[-1]
        slow_ema = data[f"ema_slow_{self.parameters.get('slow_ema', 21)}"].iloc[-1]
        
        # Calculate EMA difference as percentage
        ema_diff_pct = abs(fast_ema - slow_ema) / slow_ema * 100
        
        # Current position size
        current_size = current_position.get('size', 0)
        
        # Adjust position based on trend strength
        if ema_diff_pct > 2.0:  # Strong trend
            # Increase position size by 20% for strong trends
            new_size = current_size * 1.2
            return True, new_size
        elif ema_diff_pct < 0.5:  # Weak trend
            # Decrease position size by 20% for weak trends
            new_size = current_size * 0.8
            return True, new_size
        
        # No adjustment needed
        return False, None 