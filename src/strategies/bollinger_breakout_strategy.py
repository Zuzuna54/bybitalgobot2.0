"""
Bollinger Bands Breakout Strategy for the Algorithmic Trading System

This strategy generates buy signals when price breaks above the upper Bollinger Band
and sell signals when price breaks below the lower Bollinger Band, with confirmation
from volume and RSI indicators.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

import pandas as pd
import numpy as np
from loguru import logger

from src.strategies.base_strategy import BaseStrategy, Signal, SignalType
from src.indicators.indicator_manager import IndicatorManager


class BollingerBreakoutStrategy(BaseStrategy):
    """
    Bollinger Bands Breakout Strategy.

    This strategy identifies breakouts from Bollinger Bands:
    - Buy Signal: Price breaks above upper band with higher than average volume and RSI > 50
    - Sell Signal: Price breaks below lower band with higher than average volume and RSI < 50
    """

    def __init__(self, config: Dict[str, Any], indicator_manager: IndicatorManager):
        """
        Initialize the Bollinger Bands Breakout Strategy.

        Args:
            config: Strategy configuration parameters
            indicator_manager: Indicator manager instance
        """
        # Set name and initialize all parameters before calling parent constructor
        # to ensure they're available during _init_indicators
        self.name = "bollinger_breakout"
        self.description = "Bollinger Bands Breakout with Volume and RSI Confirmation"

        # Default configuration values
        self.bb_length = config.get("bb_length", 20)
        self.bb_std_dev = config.get("bb_std_dev", 2.0)
        self.volume_ma_length = config.get("volume_ma_length", 20)
        self.volume_threshold = config.get(
            "volume_threshold", 1.5
        )  # Volume must be 1.5x the average
        self.rsi_length = config.get("rsi_length", 14)
        self.rsi_overbought = config.get(
            "rsi_overbought", 70
        )  # Exit long position if RSI exceeds this
        self.rsi_oversold = config.get(
            "rsi_oversold", 30
        )  # Exit short position if RSI below this
        self.rsi_threshold = config.get("rsi_threshold", 50)
        self.atr_length = config.get("atr_length", 14)
        self.volatility_factor = config.get("volatility_factor", 1.0)
        self.atr_stop_multiplier = config.get("atr_stop_multiplier", 2.0)
        self.risk_reward_ratio = config.get("risk_reward_ratio", 2.0)

        # Now call the parent constructor
        super().__init__(config, indicator_manager)

        # For consecutive breakout detection
        self.consecutive_closes_required = config.get("consecutive_closes_required", 1)

        # Initialize indicators
        self._init_indicators()

    def _init_indicators(self) -> None:
        """Initialize the indicators required for this strategy."""
        # Define indicator names with prefix to ensure uniqueness
        prefix = f"{self.name}_"
        bollinger_name = f"{prefix}bollinger"
        rsi_name = f"{prefix}rsi"
        volume_sma_name = f"{prefix}volume_sma"
        atr_name = f"{prefix}atr"

        # Store indicator names for later reference
        self.strategy_indicators = [bollinger_name, rsi_name, volume_sma_name, atr_name]

        # Check if indicators already exist (to avoid duplicate initialization)
        indicators = self.indicator_manager.indicators.keys()

        # Add Bollinger Bands if they don't exist
        if bollinger_name not in indicators:
            self.indicator_manager.add_indicator(
                bollinger_name,
                "bollinger",
                {
                    "period": self.bb_length,
                    "std_dev": self.bb_std_dev,
                    "target_column": "close",
                },
            )

        # Add RSI for confirmation if it doesn't exist
        if rsi_name not in indicators:
            self.indicator_manager.add_indicator(
                rsi_name, "rsi", {"period": self.rsi_length}
            )

        # Add Volume SMA for volume filtering if it doesn't exist
        if volume_sma_name not in indicators:
            self.indicator_manager.add_indicator(
                volume_sma_name,
                "ma",
                {
                    "period": self.volume_ma_length,
                    "type": "sma",
                    "target_column": "volume",
                    "output_name": "volume_sma",
                },
            )

        # Add ATR for stop loss calculation if it doesn't exist
        if atr_name not in indicators:
            self.indicator_manager.add_indicator(
                atr_name, "atr", {"period": self.atr_length}
            )

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        Generate trading signals based on Bollinger Bands breakouts.

        Args:
            data: Market data DataFrame with OHLCV data and indicators

        Returns:
            List of generated trading signals
        """
        signals = []

        # Ensure we have enough data
        if len(data) < self.bb_length + 5:
            logger.warning(
                f"Not enough data for Bollinger Breakout strategy (need at least {self.bb_length + 5} bars)"
            )
            return signals

        # Get the current data (most recent bar)
        current = data.iloc[-1]
        previous = data.iloc[-2]

        # Debug output for troubleshooting
        logger.debug(f"Bollinger Breakout strategy columns: {list(data.columns)}")

        # Get indicator values - try both naming conventions
        # First try with strategy prefix
        bb_upper = current.get(f"{self.name}_bollinger_upper")
        if bb_upper is None:
            # Try with default column names
            bb_upper = current.get("bb_upper")

        bb_lower = current.get(f"{self.name}_bollinger_lower")
        if bb_lower is None:
            bb_lower = current.get("bb_lower")

        bb_middle = current.get(f"{self.name}_bollinger_middle")
        if bb_middle is None:
            bb_middle = current.get("bb_middle")

        rsi = current.get(f"{self.name}_rsi")
        if rsi is None:
            rsi = current.get("rsi")

        volume = current.get("volume")

        # Try additional formats for volume_sma
        volume_sma = current.get(f"{self.name}_volume_sma")
        if volume_sma is None:
            volume_sma = current.get("volume_sma")
            # If still not found, calculate it on the fly if we have enough data
            if volume_sma is None and len(data) > self.volume_ma_length:
                volume_sma = (
                    data["volume"].rolling(self.volume_ma_length).mean().iloc[-1]
                )

        atr = current.get(f"{self.name}_atr")
        if atr is None:
            atr = current.get("atr")

        # Skip if missing any required indicators
        if (
            bb_upper is None
            or bb_lower is None
            or rsi is None
            or volume is None
            or volume_sma is None
            or atr is None
        ):
            missing_indicators = []
            if bb_upper is None:
                missing_indicators.append("bollinger_upper")
            if bb_lower is None:
                missing_indicators.append("bollinger_lower")
            if rsi is None:
                missing_indicators.append("rsi")
            if volume is None:
                missing_indicators.append("volume")
            if volume_sma is None:
                missing_indicators.append("volume_sma")
            if atr is None:
                missing_indicators.append("atr")

            logger.warning(
                f"Missing indicators for Bollinger Breakout strategy: {', '.join(missing_indicators)}"
            )
            return signals

        # Previous values for confirmation
        prev_close = previous.get("close")
        prev_bb_upper = previous.get("bb_upper")
        if prev_bb_upper is None:
            prev_bb_upper = previous.get(f"{self.name}_bollinger_upper")

        prev_bb_lower = previous.get("bb_lower")
        if prev_bb_lower is None:
            prev_bb_lower = previous.get(f"{self.name}_bollinger_lower")

        # Check for breakouts
        close_price = current.get("close")
        timestamp = pd.to_datetime(current.name)
        symbol = self.config.get("symbol", "UNKNOWN")

        # Calculate band width for signal strength
        band_width = (bb_upper - bb_lower) / bb_middle

        # Check volume condition
        volume_condition = volume > (volume_sma * self.volume_threshold)

        # Calculate how far price is from the bands (normalized by ATR)
        distance_from_upper = (close_price - bb_upper) / atr if atr > 0 else 0
        distance_from_lower = (bb_lower - close_price) / atr if atr > 0 else 0

        # Track consecutive closes above/below bands
        consecutive_closes_above = 0
        consecutive_closes_below = 0

        # Count how many consecutive closes above/below bands
        for i in range(1, min(self.consecutive_closes_required + 1, len(data))):
            if data.iloc[-i]["close"] > data.iloc[-i]["bb_upper"]:
                consecutive_closes_above += 1
            else:
                break

        for i in range(1, min(self.consecutive_closes_required + 1, len(data))):
            if data.iloc[-i]["close"] < data.iloc[-i]["bb_lower"]:
                consecutive_closes_below += 1
            else:
                break

        # Upper band breakout (buy signal)
        if (
            close_price > bb_upper
            and prev_close <= prev_bb_upper
            and volume_condition
            and rsi > 50
            and consecutive_closes_above >= self.consecutive_closes_required
        ):

            # Calculate signal strength based on:
            # 1. How far price is above the upper band (normalized by ATR)
            # 2. RSI value (stronger if between 50-70, weaker if extremely overbought)
            # 3. Volume surge compared to average

            # Base strength from breakout magnitude
            strength = min(0.5 + (distance_from_upper * 0.2), 0.9)

            # Adjust strength based on RSI (ideal range 50-70)
            if rsi > self.rsi_overbought:
                # Reduce strength if overbought (potential reversal)
                strength *= 0.8
            else:
                # Normalize RSI component between 50-70 to 0-0.2 additional strength
                rsi_component = min((rsi - 50) / (self.rsi_overbought - 50), 1.0) * 0.2
                strength += rsi_component

            # Volume component (0-0.1 additional strength)
            volume_ratio = min(volume / volume_sma, 3.0)  # Cap at 3x average
            volume_component = ((volume_ratio - 1.0) / 2.0) * 0.1  # Normalize to 0-0.1
            strength += volume_component

            # Cap strength at 1.0
            strength = min(strength, 1.0)

            # Create signal
            signals.append(
                Signal(
                    signal_type=SignalType.BUY,
                    symbol=symbol,
                    timestamp=timestamp,
                    price=close_price,
                    strategy_name=self.name,
                    timeframe=self.timeframe,
                    strength=strength,
                    metadata={
                        "strategy_name": self.name,
                        "indicators": {
                            "bb_upper": bb_upper,
                            "bb_middle": bb_middle,
                            "bb_lower": bb_lower,
                            "band_width": band_width,
                            "rsi": rsi,
                            "volume_ratio": volume / volume_sma,
                            "atr": atr,
                        },
                        "reason": "Upper Bollinger Band breakout with volume confirmation",
                    },
                )
            )

            logger.info(
                f"Generated BUY signal for {symbol} at {close_price} (strength: {strength:.2f})"
            )

        # Lower band breakout (sell signal)
        elif (
            close_price < bb_lower
            and prev_close >= prev_bb_lower
            and volume_condition
            and rsi < 50
            and consecutive_closes_below >= self.consecutive_closes_required
        ):

            # Calculate signal strength similarly to buy signal
            strength = min(0.5 + (distance_from_lower * 0.2), 0.9)

            # Adjust strength based on RSI (ideal range 30-50)
            if rsi < self.rsi_oversold:
                # Reduce strength if oversold (potential reversal)
                strength *= 0.8
            else:
                # Normalize RSI component between 30-50 to 0-0.2 additional strength
                rsi_component = min((50 - rsi) / (50 - self.rsi_oversold), 1.0) * 0.2
                strength += rsi_component

            # Volume component (0-0.1 additional strength)
            volume_ratio = min(volume / volume_sma, 3.0)  # Cap at 3x average
            volume_component = ((volume_ratio - 1.0) / 2.0) * 0.1  # Normalize to 0-0.1
            strength += volume_component

            # Cap strength at 1.0
            strength = min(strength, 1.0)

            # Create signal
            signals.append(
                Signal(
                    signal_type=SignalType.SELL,
                    symbol=symbol,
                    timestamp=timestamp,
                    price=close_price,
                    strategy_name=self.name,
                    timeframe=self.timeframe,
                    strength=strength,
                    metadata={
                        "strategy_name": self.name,
                        "indicators": {
                            "bb_upper": bb_upper,
                            "bb_middle": bb_middle,
                            "bb_lower": bb_lower,
                            "band_width": band_width,
                            "rsi": rsi,
                            "volume_ratio": volume / volume_sma,
                            "atr": atr,
                        },
                        "reason": "Lower Bollinger Band breakout with volume confirmation",
                    },
                )
            )

            logger.info(
                f"Generated SELL signal for {symbol} at {close_price} (strength: {strength:.2f})"
            )

        return signals

    def get_stop_loss_price(
        self, entry_price: float, is_long: bool, data: pd.DataFrame
    ) -> float:
        """
        Calculate stop loss price based on ATR and Bollinger Bands.

        Args:
            entry_price: Entry price for the trade
            is_long: Whether the position is long (True) or short (False)
            data: Market data DataFrame

        Returns:
            Stop loss price
        """
        # Get ATR for volatility-based stop loss
        atr = data.iloc[-1].get("atr", 0)

        # Get Bollinger Bands
        bb_middle = data.iloc[-1].get("bb_middle", entry_price)

        if is_long:
            # For long positions: use middle band as reference, but at least 2 ATR below entry
            atr_stop = entry_price - (atr * 2)
            bb_stop = bb_middle

            # Use the higher of the two (closer to entry price)
            return max(atr_stop, bb_stop)
        else:
            # For short positions: use middle band as reference, but at least 2 ATR above entry
            atr_stop = entry_price + (atr * 2)
            bb_stop = bb_middle

            # Use the lower of the two (closer to entry price)
            return min(atr_stop, bb_stop)

    def should_adjust_position(
        self, position_data: Dict[str, Any], current_data: pd.DataFrame
    ) -> Tuple[bool, float]:
        """
        Determine if position size should be adjusted based on Bollinger Band width.

        Args:
            position_data: Current position data
            current_data: Current market data

        Returns:
            Tuple of (should_adjust, adjustment_factor)
        """
        # If Bollinger Band width is narrowing significantly, we might want to reduce position size
        # as this indicates reduced volatility and potentially lower profit potential

        # Get current and historical band width
        current_bb_upper = current_data.iloc[-1].get("bb_upper", 0)
        current_bb_lower = current_data.iloc[-1].get("bb_lower", 0)
        current_bb_middle = current_data.iloc[-1].get("bb_middle", 1)

        # Calculate band width as percentage of middle price
        current_band_width = (current_bb_upper - current_bb_lower) / current_bb_middle

        # Calculate historical average band width (last 5 periods)
        historical_width = []
        for i in range(2, min(7, len(current_data))):
            upper = current_data.iloc[-i].get("bb_upper", 0)
            lower = current_data.iloc[-i].get("bb_lower", 0)
            middle = current_data.iloc[-i].get("bb_middle", 1)
            if upper and lower and middle:
                historical_width.append((upper - lower) / middle)

        avg_historical_width = (
            np.mean(historical_width) if historical_width else current_band_width
        )

        # If current width is significantly less than historical, reduce position
        if current_band_width < avg_historical_width * 0.7:
            return True, 0.7  # Reduce position by 30%

        # If current width is significantly more than historical, increase position
        if current_band_width > avg_historical_width * 1.3:
            return True, 1.3  # Increase position by 30%

        return False, 1.0  # No adjustment needed
