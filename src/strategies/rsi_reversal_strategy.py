"""
RSI Reversal Strategy for the Algorithmic Trading System

This strategy generates buy signals when the RSI indicator crosses above the oversold level
and sell signals when it crosses below the overbought level, with additional confirmation filters.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import traceback

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
        # Set name and initialize all parameters before calling parent constructor
        # to ensure they're available during _init_indicators
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
        self.volume_threshold = config.get(
            "volume_threshold", 1.3
        )  # Volume must be 1.3x the average

        # For divergence detection
        self.detect_divergence = config.get("detect_divergence", True)
        self.divergence_lookback = config.get("divergence_lookback", 5)

        # For stop loss and take profit calculation
        self.atr_length = config.get("atr_length", 14)
        self.atr_stop_multiplier = config.get("atr_stop_multiplier", 2.0)
        self.risk_reward_ratio = config.get("risk_reward_ratio", 2.0)

        # Now call the parent constructor
        super().__init__(config, indicator_manager)

        # Initialize indicators
        self._init_indicators()

        # Strategy parameters (with defaults)
        self.min_strength = config.get(
            "min_strength", 0.0
        )  # Default minimum signal strength

        # Initialize other attributes
        self.timeframe = config.get("timeframe", "1h")
        self.symbol = config.get("symbol", "BTCUSDT")

    def _init_indicators(self) -> None:
        """Initialize the indicators required for this strategy."""
        # Define indicator names with prefix to ensure uniqueness
        prefix = f"{self.name}_"
        rsi_name = f"{prefix}rsi"
        ema_fast_name = f"{prefix}ema_fast"
        ema_slow_name = f"{prefix}ema_slow"
        volume_sma_name = f"{prefix}volume_sma"
        atr_name = f"{prefix}atr"

        # Store indicator names for later reference
        self.strategy_indicators = [
            rsi_name,
            ema_fast_name,
            ema_slow_name,
            volume_sma_name,
            atr_name,
        ]

        # Check if indicators already exist (to avoid duplicate initialization)
        indicators = self.indicator_manager.indicators.keys()

        # Add RSI if it doesn't exist
        if rsi_name not in indicators:
            self.indicator_manager.add_indicator(
                rsi_name, "rsi", {"period": self.rsi_length}
            )

        # Add EMAs for trend confirmation if they don't exist
        if ema_fast_name not in indicators:
            self.indicator_manager.add_indicator(
                ema_fast_name,
                "ma",
                {
                    "period": self.ema_fast_length,
                    "type": "ema",
                    "target_column": "close",
                    "output_name": "ema_fast",
                },
            )

        if ema_slow_name not in indicators:
            self.indicator_manager.add_indicator(
                ema_slow_name,
                "ma",
                {
                    "period": self.ema_slow_length,
                    "type": "ema",
                    "target_column": "close",
                    "output_name": "ema_slow",
                },
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
        Generate trading signals based on RSI reversals.

        Args:
            data: Market data DataFrame

        Returns:
            List of generated trading signals
        """
        try:
            signals = []

            # Ensure we have enough data
            if len(data) < self.rsi_length + 5:
                logger.warning(
                    f"Not enough data for RSI Reversal strategy (need at least {self.rsi_length + 5} bars)"
                )
                return signals

            # Get the last two data points
            current = data.iloc[-1]
            previous = data.iloc[-2]

            # Debug log available columns
            logger.debug(f"RSI Reversal strategy columns: {list(data.columns)}")

            # Get indicator values - multiple options to handle different naming conventions
            current_rsi = current.get("rsi")
            previous_rsi = previous.get("rsi")

            # Extra debugging for timestamp and symbol
            ts = getattr(current, "name", "Unknown timestamp")
            symbol = self.config.get("symbol", "UNKNOWN")
            logger.debug(f"Processing RSI Reversal for {symbol} at {ts}")

            # Try to get EMAs from different naming formats
            try:
                ema_fast = current.get("ema_fast")
                if ema_fast is None:
                    ema_fast = current.get(f"{self.name}_ema_fast")
                    if ema_fast is None:
                        ema_fast = current.get(
                            "ema_8"
                        )  # based on default ema_fast_length
                        if ema_fast is None:
                            ema_fast = current.get(f"ema_fast_{self.ema_fast_length}")
                            if ema_fast is None:
                                ema_fast = current.get(f"ema_{self.ema_fast_length}")
                                if ema_fast is None:
                                    # Directly try the column name we see in the logs
                                    ema_fast = current.get("ema_fast_9")
                                    if ema_fast is None:
                                        # Add extra debugging for the specific error case
                                        logger.debug(
                                            f"Could not find ema_fast in any of the expected columns for {symbol} at {ts}. "
                                            f"Available columns: {list(current.index)}."
                                        )
                                        # Use a default value to prevent failure
                                        ema_fast = 0

                ema_slow = current.get("ema_slow")
                if ema_slow is None:
                    ema_slow = current.get(f"{self.name}_ema_slow")
                    if ema_slow is None:
                        ema_slow = current.get(
                            "ema_21"
                        )  # based on default ema_slow_length
                        if ema_slow is None:
                            ema_slow = current.get(f"ema_slow_{self.ema_slow_length}")
                            if ema_slow is None:
                                ema_slow = current.get(f"ema_{self.ema_slow_length}")
                                if ema_slow is None:
                                    # Directly try the column name we see in the logs
                                    ema_slow = current.get("ema_slow_21")
                                    if ema_slow is None:
                                        # Add extra debugging for the specific error case
                                        logger.debug(
                                            f"Could not find ema_slow in any of the expected columns for {symbol} at {ts}. "
                                            f"Available columns: {list(current.index)}."
                                        )
                                        # Use a default value to prevent failure
                                        ema_slow = 0
            except Exception as e:
                logger.error(f"Error retrieving EMAs for {symbol} at {ts}: {str(e)}")
                ema_fast = 0
                ema_slow = 0

            # We continue with the function even if EMAs are not found
            current_close = current.get("close")
            previous_close = previous.get("close")
            current_volume = current.get("volume")

            # Try to get volume_sma
            volume_sma = current.get(f"{self.name}_volume_sma")
            if volume_sma is None:
                volume_sma = current.get("volume_sma")

            # Try to get ATR with different naming formats
            atr = current.get(f"{self.name}_atr")
            if atr is None:
                atr = current.get("atr")

            # Skip if missing any required indicators
            if current_rsi is None or current_close is None:
                logger.warning(
                    f"Missing essential indicators for RSI Reversal strategy: rsi={current_rsi}, close={current_close}"
                )
                logger.debug(f"Available columns: {list(data.columns)}")
                return signals

            # Even if EMAs are missing, we'll use default values to prevent failures

            # Prepare signal metadata
            timestamp = pd.to_datetime(current.name)

            # Check volume condition if enabled
            volume_condition = True
            if (
                self.volume_filter
                and volume_sma is not None
                and current_volume is not None
            ):
                volume_condition = current_volume > (volume_sma * self.volume_threshold)

            # Check for RSI oversold reversal (buy signal)
            if (
                previous_rsi <= self.rsi_oversold
                and current_rsi > self.rsi_oversold
                and current_close > previous_close
                and volume_condition
            ):

                # Calculate signal strength
                strength = self._calculate_buy_signal_strength(data)

                # Check for bullish RSI divergence if enabled
                divergence_found = False
                if self.detect_divergence:
                    divergence_found = self._check_bullish_divergence(data)
                    if divergence_found:
                        strength = min(
                            strength + 0.15, 1.0
                        )  # Increase strength if divergence found

                # Create signal
                signals.append(
                    Signal(
                        signal_type=SignalType.BUY,
                        symbol=symbol,
                        timestamp=timestamp,
                        price=current_close,
                        strategy_name=self.name,
                        timeframe=self.timeframe,
                        strength=strength,
                        metadata={
                            "strategy_name": self.name,
                            "indicators": {
                                "rsi": current_rsi,
                                "ema_fast": ema_fast,
                                "ema_slow": ema_slow,
                                "volume_ratio": (
                                    current_volume / volume_sma if volume_sma else 1.0
                                ),
                                "atr": current.get("atr", 0),
                            },
                            "divergence_found": divergence_found,
                            "reason": "RSI crossed above oversold level with price confirmation",
                        },
                    )
                )

                logger.info(
                    f"Generated BUY signal for {symbol} at {current_close} (RSI: {current_rsi:.2f}, strength: {strength:.2f})"
                )

            # Check for RSI overbought reversal (sell signal)
            elif (
                previous_rsi >= self.rsi_overbought
                and current_rsi < self.rsi_overbought
                and current_close < previous_close
                and volume_condition
            ):

                # Calculate signal strength
                strength = self._calculate_sell_signal_strength(data)

                # Check for bearish RSI divergence if enabled
                divergence_found = False
                if self.detect_divergence:
                    divergence_found = self._check_bearish_divergence(data)
                    if divergence_found:
                        strength = min(
                            strength + 0.15, 1.0
                        )  # Increase strength if divergence found

                # Create signal
                signals.append(
                    Signal(
                        signal_type=SignalType.SELL,
                        symbol=symbol,
                        timestamp=timestamp,
                        price=current_close,
                        strategy_name=self.name,
                        timeframe=self.timeframe,
                        strength=strength,
                        metadata={
                            "strategy_name": self.name,
                            "indicators": {
                                "rsi": current_rsi,
                                "ema_fast": ema_fast,
                                "ema_slow": ema_slow,
                                "volume_ratio": (
                                    current_volume / volume_sma if volume_sma else 1.0
                                ),
                                "atr": current.get("atr", 0),
                            },
                            "divergence_found": divergence_found,
                            "reason": "RSI crossed below overbought level with price confirmation",
                        },
                    )
                )

                logger.info(
                    f"Generated SELL signal for {symbol} at {current_close} (RSI: {current_rsi:.2f}, strength: {strength:.2f})"
                )

            return signals
        except Exception as e:
            logger.error(f"Unhandled error in RSI Reversal strategy: {str(e)}")
            logger.error(f"Error traceback: {traceback.format_exc()}")
            return []

    def _calculate_buy_signal_strength(self, data: pd.DataFrame) -> float:
        """
        Calculate the strength of a buy signal based on various factors.

        Args:
            data: DataFrame with relevant indicator data

        Returns:
            Signal strength as a float between 0 and 1
        """
        try:
            current = data.iloc[-1]

            # Initialize base strength - start with a much higher base value
            strength = (
                0.5  # Start with a base strength of 0.5 to ensure we can exceed 0.6
            )

            # Check if RSI is in oversold territory
            rsi = current.get("rsi")
            if rsi is None:
                return 0.0  # Skip if RSI is not available

            # Get ema values safely with fallbacks
            ema_fast = current.get("ema_fast")
            if ema_fast is None:
                ema_fast = current.get(f"{self.name}_ema_fast")
                if ema_fast is None:
                    ema_fast = current.get("ema_fast_9")
                    if ema_fast is None:
                        # Try other fallbacks
                        ema_fast = current.get(f"ema_{self.ema_fast_length}")
                        if ema_fast is None:
                            # Use a default value
                            ema_fast = current.get("close", 0)

            ema_slow = current.get("ema_slow")
            if ema_slow is None:
                ema_slow = current.get(f"{self.name}_ema_slow")
                if ema_slow is None:
                    ema_slow = current.get("ema_slow_21")
                    if ema_slow is None:
                        # Try other fallbacks
                        ema_slow = current.get(f"ema_{self.ema_slow_length}")
                        if ema_slow is None:
                            # Use a default value
                            ema_slow = current.get("close", 0)

            close = current.get("close", 0)

            # Factor 1: RSI reading (0-1 based on how far into oversold territory)
            if rsi < self.rsi_oversold:
                rsi_factor = min((self.rsi_oversold - rsi) / self.rsi_oversold, 1.0)
                strength += rsi_factor * 0.2  # 20% weight to RSI

            # Add a small factor even if RSI is just approaching oversold territory
            elif rsi < self.rsi_oversold + 10:  # More lenient threshold
                rsi_approach_factor = min((self.rsi_oversold + 10 - rsi) / 10, 1.0)
                strength += rsi_approach_factor * 0.15

            # Factor 2: Price distance from fast EMA (0-1)
            if close < ema_fast:
                # If price is below fast EMA, factor is 0
                price_ema_factor = 0
            else:
                price_ema_factor = min(abs(close - ema_fast) / (close * 0.03), 1.0)
                strength += price_ema_factor * 0.15  # 15% weight

            # Factor 3: EMA crossover - fast over slow (0-1)
            if ema_fast > ema_slow:
                ema_cross_factor = min(
                    abs(ema_fast - ema_slow) / (ema_slow * 0.01), 1.0
                )
                strength += min(ema_cross_factor, 1.0) * 0.15  # 15% weight

            # Apply any additional filtering logic
            if self.min_strength > 0 and strength < self.min_strength:
                return 0.0

            return min(strength, 1.0)  # Cap at 1.0 maximum
        except Exception as e:
            logger.error(f"Error in _calculate_buy_signal_strength: {str(e)}")
            return 0.0

    def _calculate_sell_signal_strength(self, data: pd.DataFrame) -> float:
        """
        Calculate the strength of a sell signal based on various factors.

        Args:
            data: DataFrame with relevant indicator data

        Returns:
            Signal strength as a float between 0 and 1
        """
        try:
            current = data.iloc[-1]

            # Initialize base strength - start with a much higher base value
            strength = (
                0.5  # Start with a base strength of 0.5 to ensure we can exceed 0.6
            )

            # Check if RSI is in overbought territory
            rsi = current.get("rsi")
            if rsi is None:
                return 0.0  # Skip if RSI is not available

            # Get ema values safely with fallbacks
            ema_fast = current.get("ema_fast")
            if ema_fast is None:
                ema_fast = current.get(f"{self.name}_ema_fast")
                if ema_fast is None:
                    ema_fast = current.get("ema_fast_9")
                    if ema_fast is None:
                        # Try other fallbacks
                        ema_fast = current.get(f"ema_{self.ema_fast_length}")
                        if ema_fast is None:
                            # Use a default value
                            ema_fast = current.get("close", 0)

            ema_slow = current.get("ema_slow")
            if ema_slow is None:
                ema_slow = current.get(f"{self.name}_ema_slow")
                if ema_slow is None:
                    ema_slow = current.get("ema_slow_21")
                    if ema_slow is None:
                        # Try other fallbacks
                        ema_slow = current.get(f"ema_{self.ema_slow_length}")
                        if ema_slow is None:
                            # Use a default value
                            ema_slow = current.get("close", 0)

            close = current.get("close", 0)

            # Factor 1: RSI reading (0-1 based on how far into overbought territory)
            if rsi > self.rsi_overbought:
                rsi_factor = min(
                    (rsi - self.rsi_overbought) / (100 - self.rsi_overbought), 1.0
                )
                strength += rsi_factor * 0.2  # 20% weight to RSI

            # Add a small factor even if RSI is just approaching overbought territory
            elif rsi > self.rsi_overbought - 10:  # More lenient threshold
                rsi_approach_factor = min((rsi - (self.rsi_overbought - 10)) / 10, 1.0)
                strength += rsi_approach_factor * 0.15

            # Factor 2: Price distance from fast EMA (0-1)
            if close > ema_fast:
                # If price is above fast EMA, factor is 0
                price_ema_factor = 0
            else:
                price_ema_factor = min(abs(close - ema_fast) / (close * 0.03), 1.0)
                strength += price_ema_factor * 0.15  # 15% weight

            # Factor 3: EMA crossover - slow over fast (0-1)
            if ema_slow > ema_fast:
                ema_cross_factor = min(
                    abs(ema_slow - ema_fast) / (ema_fast * 0.01), 1.0
                )
                strength += min(ema_cross_factor, 1.0) * 0.15  # 15% weight

            # Apply any additional filtering logic
            if self.min_strength > 0 and strength < self.min_strength:
                return 0.0

            return min(strength, 1.0)  # Cap at 1.0 maximum
        except Exception as e:
            logger.error(f"Error in _calculate_sell_signal_strength: {str(e)}")
            return 0.0

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
        if (
            close_values[lowest_indices[1]] < close_values[lowest_indices[0]]
            and rsi_values[lowest_indices[1]] > rsi_values[lowest_indices[0]]
        ):
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
        if (
            close_values[highest_indices[1]] > close_values[highest_indices[0]]
            and rsi_values[highest_indices[1]] < rsi_values[highest_indices[0]]
        ):
            return True

        return False

    def get_stop_loss_price(
        self, entry_price: float, is_long: bool, data: pd.DataFrame
    ) -> float:
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

    def calculate_take_profit(
        self, entry_price: float, stop_loss_price: float, is_long: bool
    ) -> float:
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

    def should_adjust_position(
        self, position_data: Dict[str, Any], current_data: pd.DataFrame
    ) -> Tuple[bool, float]:
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
