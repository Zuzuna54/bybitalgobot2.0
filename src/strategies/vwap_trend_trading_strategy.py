"""
VWAP Trend Trading Strategy for the Algorithmic Trading System

This strategy generates buy signals when price crosses above VWAP and sell signals
when price crosses below VWAP. It filters signals based on existing price trend direction.
"""

from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np
from loguru import logger

from src.strategies.base_strategy import BaseStrategy, Signal, SignalType
from src.indicators.indicator_manager import IndicatorManager


class VWAPTrendTradingStrategy(BaseStrategy):
    """VWAP Trend Trading Strategy."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        indicator_manager: IndicatorManager
    ):
        """
        Initialize the VWAP Trend Trading strategy.
        
        Args:
            config: Strategy configuration
            indicator_manager: Indicator manager for technical indicators
        """
        super().__init__('vwap_trend_trading', config, indicator_manager)
    
    def _init_indicators(self) -> None:
        """Initialize strategy-specific indicators."""
        # Get parameters
        vwap_period = self.parameters.get('vwap_period', 'daily')
        entry_threshold = self.parameters.get('entry_threshold', 0.1)
        exit_threshold = self.parameters.get('exit_threshold', 0.1)
        
        # Define indicator names
        prefix = f"{self.name}_"
        vwap_name = f"{prefix}vwap"
        ema_name = f"{prefix}ema"
        
        # Check if indicators already exist, otherwise add them
        if vwap_name not in self.indicator_manager.indicators:
            self.indicator_manager.add_indicator(
                vwap_name, 'vwap', {
                    'session_period': vwap_period
                }
            )
        
        if ema_name not in self.indicator_manager.indicators:
            self.indicator_manager.add_indicator(
                ema_name, 'ma', {
                    'period': 20,
                    'type': 'ema'
                }
            )
        
        # Store indicator names for this strategy
        self.strategy_indicators = [
            vwap_name,
            ema_name
        ]
        
        # Define required columns for signal generation
        self.required_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'vwap', 'ema_20'
        ]
        
        logger.info(f"Initialized indicators for VWAP Trend Trading strategy with period={vwap_period}")
    
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        Generate trading signals based on price and VWAP relationship.
        
        Args:
            data: Prepared data with indicators
            
        Returns:
            List of trading signals
        """
        signals = []
        
        # Make sure we have enough data
        if len(data) < 5:  # Need at least 5 candles to determine trend
            logger.warning("Not enough data to generate signals")
            return signals
        
        # Get the latest complete candle (second to last row)
        # Using second to last row to avoid acting on incomplete candles
        latest_complete_candle = data.iloc[-2]
        previous_candle = data.iloc[-3]
        current_timestamp = latest_complete_candle.name
        current_price = latest_complete_candle['close']
        symbol = self.parameters.get('symbol', 'BTCUSDT')  # Default symbol
        
        # Calculate entry threshold (percentage of current price)
        entry_threshold_pct = self.parameters.get('entry_threshold', 0.1) / 100
        entry_threshold_value = current_price * entry_threshold_pct
        
        # Get VWAP value
        vwap = latest_complete_candle['vwap']
        previous_vwap = previous_candle['vwap']
        
        # Check if price crossed above or below VWAP
        price_crossed_above_vwap = (
            latest_complete_candle['close'] > vwap and
            previous_candle['close'] <= previous_vwap
        )
        
        price_crossed_below_vwap = (
            latest_complete_candle['close'] < vwap and
            previous_candle['close'] >= previous_vwap
        )
        
        # Determine trend using EMA
        uptrend = latest_complete_candle['close'] > latest_complete_candle['ema_20']
        downtrend = latest_complete_candle['close'] < latest_complete_candle['ema_20']
        
        # Check significant price movement from VWAP
        price_distance_from_vwap = abs(latest_complete_candle['close'] - vwap) / vwap * 100
        significant_movement = price_distance_from_vwap > self.parameters.get('entry_threshold', 0.1)
        
        # Generate buy signal when price crosses above VWAP during uptrend
        if price_crossed_above_vwap and uptrend:
            # Calculate signal strength based on price movement and trend strength
            price_movement = (latest_complete_candle['close'] - vwap) / vwap
            trend_strength = (latest_complete_candle['close'] - latest_complete_candle['ema_20']) / latest_complete_candle['ema_20']
            
            # Normalize strengths to 0-1 range
            price_strength = min(price_movement * 100, 1.0)
            trend_strength = min(trend_strength * 100, 1.0)
            
            signal_strength = 0.5 + (0.25 * price_strength) + (0.25 * trend_strength)
            
            # Create signal metadata
            metadata = {
                'vwap': vwap,
                'ema': latest_complete_candle['ema_20'],
                'price_distance_pct': price_distance_from_vwap,
                'trend_strength': trend_strength,
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
            
        # Generate sell signal when price crosses below VWAP during downtrend
        elif price_crossed_below_vwap and downtrend:
            # Calculate signal strength based on price movement and trend strength
            price_movement = (vwap - latest_complete_candle['close']) / vwap
            trend_strength = (latest_complete_candle['ema_20'] - latest_complete_candle['close']) / latest_complete_candle['ema_20']
            
            # Normalize strengths to 0-1 range
            price_strength = min(price_movement * 100, 1.0)
            trend_strength = min(trend_strength * 100, 1.0)
            
            signal_strength = 0.5 + (0.25 * price_strength) + (0.25 * trend_strength)
            
            # Create signal metadata
            metadata = {
                'vwap': vwap,
                'ema': latest_complete_candle['ema_20'],
                'price_distance_pct': price_distance_from_vwap,
                'trend_strength': trend_strength,
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
        Calculate the stop loss price for VWAP trend trading strategy.
        
        Args:
            data: Prepared data with indicators
            entry_price: Entry price of the trade
            is_long: Whether the trade is long or short
            
        Returns:
            Stop loss price
        """
        # For VWAP trend trading, use VWAP as stop loss reference
        vwap = data['vwap'].iloc[-1]
        
        if 'atr' in data.columns:
            # ATR-based stop loss from VWAP level
            atr_value = data['atr'].iloc[-1]
            atr_multiplier = self.parameters.get('stop_loss_atr_multiplier', 1.5)
            
            if is_long:
                # For long positions, stop loss below VWAP
                return min(vwap - (atr_value * atr_multiplier), entry_price * 0.97)
            else:
                # For short positions, stop loss above VWAP
                return max(vwap + (atr_value * atr_multiplier), entry_price * 1.03)
        else:
            # Percentage-based stop loss relative to VWAP
            stop_percentage = self.parameters.get('stop_loss_percentage', 2.0) / 100
            
            if is_long:
                # For long trades, stop below VWAP
                return min(vwap * (1 - stop_percentage), entry_price * 0.97)
            else:
                # For short trades, stop above VWAP
                return max(vwap * (1 + stop_percentage), entry_price * 1.03)
    
    def should_adjust_position(self, data: pd.DataFrame, current_position: Dict[str, Any]) -> tuple:
        """
        Determine if position size should be adjusted based on VWAP and price relationship.
        
        Args:
            data: Prepared data with indicators
            current_position: Current position information
            
        Returns:
            Tuple of (should_adjust, new_size_if_adjusting)
        """
        # Get current values
        current_price = data['close'].iloc[-1]
        vwap = data['vwap'].iloc[-1]
        ema = data['ema_20'].iloc[-1]
        
        # Calculate distance from VWAP
        vwap_distance = abs(current_price - vwap) / vwap
        
        # Current position size
        current_size = current_position.get('size', 0)
        
        # Position side (long or short)
        is_long = current_position.get('type', 'long') == 'long'
        
        # Adjust position based on price movement relative to VWAP
        if is_long:
            if current_price < ema and current_price < vwap:
                # Price moved below both VWAP and EMA - reduce position by 50%
                new_size = current_size * 0.5
                return True, new_size
            elif current_price > vwap * 1.02:
                # Price moved significantly above VWAP - increase position by 20%
                new_size = current_size * 1.2
                return True, new_size
        else:  # Short position
            if current_price > ema and current_price > vwap:
                # Price moved above both VWAP and EMA - reduce position by 50%
                new_size = current_size * 0.5
                return True, new_size
            elif current_price < vwap * 0.98:
                # Price moved significantly below VWAP - increase position by 20%
                new_size = current_size * 1.2
                return True, new_size
        
        # No adjustment needed
        return False, None 