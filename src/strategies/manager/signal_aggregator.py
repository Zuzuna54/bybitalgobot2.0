"""
Signal Aggregator for the Algorithmic Trading System

This module provides functionality for aggregating and processing signals from
multiple trading strategies.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
from loguru import logger

from src.strategies.base_strategy import Signal, SignalType
from src.models.models import SignalStrength


def aggregate_signals(
    signals: List[Signal],
    data: pd.DataFrame,
    min_concurrent_strategies: int,
    weighted_aggregation: bool,
    signal_threshold: float,
    strategy_performance: Dict[str, Dict[str, Any]],
    default_weight: float,
) -> List[Signal]:
    """
    Aggregate signals from multiple strategies.

    Args:
        signals: List of signals from all strategies
        data: Market data DataFrame
        min_concurrent_strategies: Minimum number of strategies required for a valid signal
        weighted_aggregation: Whether to use weighted aggregation based on strategy performance
        signal_threshold: Minimum strength threshold for aggregated signals
        strategy_performance: Dictionary of strategy performance metrics
        default_weight: Default strategy weight

    Returns:
        List of aggregated signals
    """
    if not signals:
        return []

    # Group signals by symbol and type
    signal_groups: Dict[str, Dict[SignalType, List[Signal]]] = {}

    for signal in signals:
        symbol = signal.symbol
        signal_type = signal.signal_type

        if symbol not in signal_groups:
            signal_groups[symbol] = {
                SignalType.BUY: [],
                SignalType.SELL: [],
                SignalType.NONE: [],
            }

        signal_groups[symbol][signal_type].append(signal)

    # Process each group to create aggregated signals
    aggregated_signals: List[Signal] = []

    for symbol, type_signals in signal_groups.items():
        # Process buy signals
        buy_signals = type_signals[SignalType.BUY]
        if buy_signals:
            buy_signal = create_aggregated_signal(
                signals=buy_signals,
                signal_type=SignalType.BUY,
                data=data,
                min_concurrent_strategies=min_concurrent_strategies,
                weighted_aggregation=weighted_aggregation,
                signal_threshold=signal_threshold,
                strategy_performance=strategy_performance,
                default_weight=default_weight,
            )

            if buy_signal:
                aggregated_signals.append(buy_signal)

        # Process sell signals
        sell_signals = type_signals[SignalType.SELL]
        if sell_signals:
            sell_signal = create_aggregated_signal(
                signals=sell_signals,
                signal_type=SignalType.SELL,
                data=data,
                min_concurrent_strategies=min_concurrent_strategies,
                weighted_aggregation=weighted_aggregation,
                signal_threshold=signal_threshold,
                strategy_performance=strategy_performance,
                default_weight=default_weight,
            )

            if sell_signal:
                aggregated_signals.append(sell_signal)

    return aggregated_signals


def create_aggregated_signal(
    signals: List[Signal],
    signal_type: SignalType,
    data: pd.DataFrame,
    min_concurrent_strategies: int,
    weighted_aggregation: bool,
    signal_threshold: float,
    strategy_performance: Dict[str, Dict[str, Any]],
    default_weight: float,
) -> Optional[Signal]:
    """
    Create an aggregated signal from multiple signals of the same type.

    Args:
        signals: List of signals of the same type
        signal_type: Signal type (BUY, SELL, NONE)
        data: Market data DataFrame
        min_concurrent_strategies: Minimum number of strategies required for a valid signal
        weighted_aggregation: Whether to use weighted aggregation based on strategy performance
        signal_threshold: Minimum strength threshold for aggregated signals
        strategy_performance: Dictionary of strategy performance metrics
        default_weight: Default strategy weight

    Returns:
        Aggregated signal or None if signals don't meet criteria
    """
    if not signals:
        return None

    # Count concurrent strategies
    unique_strategies = set(
        signal.metadata.get("strategy_name", "unknown") for signal in signals
    )

    # Check if we have enough concurrent strategies
    if len(unique_strategies) < min_concurrent_strategies:
        logger.debug(
            f"Not enough concurrent strategies for {signal_type} signal: "
            f"{len(unique_strategies)} < {min_concurrent_strategies}"
        )
        return None

    # Symbol should be the same for all signals
    symbol = signals[0].symbol
    timestamp = signals[0].timestamp

    # Use the average price from all signals
    price = sum(signal.price for signal in signals) / len(signals)

    # Calculate weighted strength if enabled
    if weighted_aggregation:
        total_weight = 0.0
        weighted_strength = 0.0

        for signal in signals:
            strategy_name = signal.metadata.get("strategy_name", "unknown")
            strategy_weight = strategy_performance.get(strategy_name, {}).get(
                "weight", default_weight
            )

            weighted_strength += signal.strength * strategy_weight
            total_weight += strategy_weight

        strength = weighted_strength / total_weight if total_weight > 0 else 0.0
    else:
        # Simple average
        strength = sum(signal.strength for signal in signals) / len(signals)

    # Skip if aggregated strength is below threshold
    if strength < signal_threshold:
        logger.debug(
            f"Aggregated signal strength below threshold: {strength:.2f} < {signal_threshold}"
        )
        return None

    # Create metadata including all contributing strategies
    contributing_strategies = [
        signal.metadata.get("strategy_name", "unknown") for signal in signals
    ]
    contributing_signals = {
        signal.metadata.get("strategy_name", f"strategy_{i}"): {
            "strength": signal.strength,
            "reason": signal.metadata.get("reason", ""),
        }
        for i, signal in enumerate(signals)
    }

    # Get indicator values
    indicator_values = {}
    for signal in signals:
        if "indicators" in signal.metadata:
            for indicator, value in signal.metadata["indicators"].items():
                if indicator not in indicator_values:
                    indicator_values[indicator] = []
                indicator_values[indicator].append(value)

    # Average indicator values
    averaged_indicators = {
        indicator: sum(values) / len(values)
        for indicator, values in indicator_values.items()
    }

    # Assign qualitative strength
    strength_category = SignalStrength.WEAK
    if strength >= 0.9:
        strength_category = SignalStrength.VERY_STRONG
    elif strength >= 0.8:
        strength_category = SignalStrength.STRONG
    elif strength >= 0.7:
        strength_category = SignalStrength.MODERATE
    elif strength >= 0.6:
        strength_category = SignalStrength.WEAK

    # Create aggregated signal
    aggregated_signal = Signal(
        signal_type=signal_type,
        symbol=symbol,
        timestamp=timestamp,
        price=price,
        strategy_name="aggregated",
        timeframe=(
            signals[0].timeframe if signals else "1h"
        ),  # Use the timeframe from first signal or default to 1h
        strength=strength,
        metadata={
            "contributing_strategies": contributing_strategies,
            "contributing_signals": contributing_signals,
            "indicators": averaged_indicators,
            "strength_category": strength_category.name,
            "reason": f"Aggregated {signal_type.name} signal from {len(signals)} strategies with strength {strength:.2f}",
        },
    )

    logger.info(
        f"Created aggregated {signal_type.name} signal for {symbol} with strength {strength:.2f} "
        f"from {len(signals)} strategies: {', '.join(contributing_strategies)}"
    )

    return aggregated_signal
