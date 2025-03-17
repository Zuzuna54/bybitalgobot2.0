# Data Flow Implementation Details (Part 2)

This document continues the data flow implementation details, focusing on Strategy Signal Flow.

## 2. Strategy Signal Flow

The strategy signal flow establishes how trading signals are generated, aggregated, and processed into trading decisions.

### 2.1 Signal Generation Pipeline

#### Implementation Details

1. **Complete Signal Generation Process in Strategies**:

```python
# In src/strategies/base_strategy.py, enhance signal generation
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
from enum import Enum
from loguru import logger

class SignalType(Enum):
    """Types of trading signals."""
    BUY = "buy"
    SELL = "sell"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"
    NO_SIGNAL = "no_signal"

class Signal:
    """Trading signal with metadata."""

    def __init__(self, signal_type: SignalType, symbol: str, timestamp: int,
                 price: float, confidence: float = 1.0, strategy_name: str = "",
                 timeframe: str = "", metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a trading signal.

        Args:
            signal_type: Type of signal
            symbol: Trading pair symbol
            timestamp: Signal timestamp in milliseconds
            price: Current price when signal was generated
            confidence: Signal confidence (0.0 to 1.0)
            strategy_name: Name of strategy that generated the signal
            timeframe: Timeframe of the data that generated the signal
            metadata: Additional signal metadata
        """
        self.signal_type = signal_type
        self.symbol = symbol
        self.timestamp = timestamp
        self.price = price
        self.confidence = min(max(confidence, 0.0), 1.0)  # Ensure between 0 and 1
        self.strategy_name = strategy_name
        self.timeframe = timeframe
        self.metadata = metadata or {}

    def __str__(self) -> str:
        """String representation of the signal."""
        return (f"Signal({self.signal_type.value}, {self.symbol}, "
                f"price={self.price}, confidence={self.confidence:.2f}, "
                f"strategy={self.strategy_name})")

    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary."""
        return {
            "signal_type": self.signal_type.value,
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "price": self.price,
            "confidence": self.confidence,
            "strategy_name": self.strategy_name,
            "timeframe": self.timeframe,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Signal':
        """Create signal from dictionary."""
        return cls(
            signal_type=SignalType(data["signal_type"]),
            symbol=data["symbol"],
            timestamp=data["timestamp"],
            price=data["price"],
            confidence=data.get("confidence", 1.0),
            strategy_name=data.get("strategy_name", ""),
            timeframe=data.get("timeframe", ""),
            metadata=data.get("metadata", {})
        )

class BaseStrategy:
    """Base class for trading strategies."""

    def __init__(self, name: str, config: Dict[str, Any] = None, indicator_manager = None):
        """
        Initialize the strategy.

        Args:
            name: Strategy name
            config: Strategy configuration
            indicator_manager: Indicator manager for technical indicators
        """
        self.name = name
        self.config = config or {}
        self.indicator_manager = indicator_manager
        self.is_active = self.config.get("is_active", True)
        self.timeframe = self.config.get("timeframe", "1h")
        self.parameters = self.config.get("parameters", {})

        # Initialize technical indicators
        self.indicators = {}
        self._init_indicators()

        logger.info(f"Initialized strategy: {self.name}")

    def _init_indicators(self) -> None:
        """Initialize technical indicators (to be implemented by subclasses)."""
        pass

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        Generate trading signals from market data.

        Args:
            data: Market data as DataFrame

        Returns:
            List of generated signals
        """
        # Ensure data is not empty
        if data is None or len(data) == 0:
            logger.warning(f"Empty data provided to {self.name} strategy")
            return []

        try:
            # Calculate indicators if not already in data
            data = self._ensure_indicators(data)

            # Generate signals (to be implemented by subclasses)
            signals = self._generate_signals_impl(data)

            # Add metadata to signals
            for signal in signals:
                signal.strategy_name = self.name
                signal.timeframe = self.timeframe

                # Add strategy parameters as metadata
                if "parameters" not in signal.metadata:
                    signal.metadata["parameters"] = self.parameters.copy()

            return signals

        except Exception as e:
            logger.error(f"Error generating signals in {self.name} strategy: {str(e)}")
            return []

    def _ensure_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure all required indicators are calculated.

        Args:
            data: Market data as DataFrame

        Returns:
            DataFrame with indicators
        """
        if self.indicator_manager is None:
            logger.warning(f"No indicator manager available for {self.name} strategy")
            return data

        # Get list of required indicators
        required_indicators = self._get_required_indicators()

        # Check if indicators are already in the data
        missing_indicators = [ind for ind in required_indicators if ind not in data.columns]

        if not missing_indicators:
            return data

        # Calculate missing indicators
        return self.indicator_manager.add_indicators(data, missing_indicators)

    def _get_required_indicators(self) -> List[str]:
        """
        Get list of required indicators (to be implemented by subclasses).

        Returns:
            List of indicator names
        """
        return []

    def _generate_signals_impl(self, data: pd.DataFrame) -> List[Signal]:
        """
        Implementation of signal generation (to be implemented by subclasses).

        Args:
            data: Market data as DataFrame with indicators

        Returns:
            List of generated signals
        """
        # Default implementation (no signals)
        return []

    def calculate_signal_confidence(self, data: pd.DataFrame, index: int,
                                   signal_type: SignalType) -> float:
        """
        Calculate confidence score for a signal.

        Args:
            data: Market data as DataFrame
            index: Index of the bar generating the signal
            signal_type: Type of signal

        Returns:
            Confidence score (0.0 to 1.0)
        """
        # Default implementation (full confidence)
        return 1.0
```

2. **Implement Signal Validation and Enrichment**:

```python
# In src/strategies/manager/validator.py, implement signal validation
from typing import Dict, Any, List, Optional, Union
from src.strategies.base_strategy import Signal, SignalType
from loguru import logger

class SignalValidator:
    """Validates and enriches trading signals."""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the signal validator.

        Args:
            config: Validator configuration
        """
        self.config = config or {}

        # Configure validation settings
        self.enable_validation = self.config.get("enable_validation", True)
        self.min_confidence_threshold = self.config.get("min_confidence_threshold", 0.3)
        self.max_signal_age_ms = self.config.get("max_signal_age_ms", 60000)  # 1 minute

        logger.info("Signal validator initialized")

    def validate_signals(self, signals: List[Signal], current_time: int = None) -> List[Signal]:
        """
        Validate a list of signals.

        Args:
            signals: List of signals to validate
            current_time: Current timestamp in milliseconds (optional)

        Returns:
            List of validated signals
        """
        if not self.enable_validation:
            return signals

        if not signals:
            return []

        validated_signals = []

        for signal in signals:
            if self.validate_signal(signal, current_time):
                validated_signals.append(signal)

        return validated_signals

    def validate_signal(self, signal: Signal, current_time: int = None) -> bool:
        """
        Validate a single signal.

        Args:
            signal: Signal to validate
            current_time: Current timestamp in milliseconds (optional)

        Returns:
            Validation result
        """
        # Check signal type
        if signal.signal_type == SignalType.NO_SIGNAL:
            return False

        # Check confidence
        if signal.confidence < self.min_confidence_threshold:
            logger.debug(f"Signal rejected due to low confidence: {signal.confidence}")
            return False

        # Check age if current_time is provided
        if current_time is not None:
            signal_age = current_time - signal.timestamp

            if signal_age > self.max_signal_age_ms:
                logger.debug(f"Signal rejected due to age: {signal_age}ms")
                return False

        return True

    def enrich_signal(self, signal: Signal, market_data: Dict[str, Any] = None) -> Signal:
        """
        Enrich a signal with additional information.

        Args:
            signal: Signal to enrich
            market_data: Current market data (optional)

        Returns:
            Enriched signal
        """
        # Add market context if available
        if market_data:
            # Add current price if not set
            if signal.price <= 0 and "close" in market_data:
                signal.price = market_data["close"]

            # Add market data snapshot
            if "market_context" not in signal.metadata:
                signal.metadata["market_context"] = {}

            # Add key market data
            for key in ["open", "high", "low", "close", "volume"]:
                if key in market_data:
                    signal.metadata["market_context"][key] = market_data[key]

        return signal

    def normalize_confidence(self, signals: List[Signal]) -> List[Signal]:
        """
        Normalize confidence scores across signals.

        Args:
            signals: List of signals

        Returns:
            Signals with normalized confidence
        """
        if not signals:
            return []

        # Find max confidence
        max_confidence = max(signal.confidence for signal in signals)

        if max_confidence <= 0:
            return signals

        # Normalize all confidences
        for signal in signals:
            signal.confidence = signal.confidence / max_confidence

        return signals
```

3. **Add Signal Metadata for Tracking and Analysis**:

```python
# In src/strategies/manager/core.py, enhance strategy manager with signal metadata
from typing import Dict, Any, List, Optional, Union
from src.strategies.base_strategy import Signal, SignalType
from src.strategies.manager.validator import SignalValidator
import pandas as pd
import time
import json
import os
from datetime import datetime
from loguru import logger

class StrategyManager:
    # Add to existing class

    def __init__(self, config: Dict[str, Any] = None, indicator_manager = None):
        # Existing initialization code

        # Initialize signal validator
        self.signal_validator = SignalValidator(config.get("validator", {}))

        # Initialize signal tracking
        self.track_signals = config.get("track_signals", True)
        self.signal_history = []
        self.max_signal_history = config.get("max_signal_history", 1000)
        self.signal_history_file = config.get("signal_history_file", "data/signal_history.json")

        # Load signal history if tracking enabled
        if self.track_signals:
            self._load_signal_history()

    def generate_signals(self, market_data_dict: Dict[str, Dict[str, pd.DataFrame]]) -> List[Signal]:
        """
        Generate trading signals from all active strategies.

        Args:
            market_data_dict: Dictionary of market data by symbol and timeframe

        Returns:
            List of aggregated trading signals
        """
        all_signals = []
        strategy_signals = {}
        current_time = int(time.time() * 1000)

        # Generate signals from each strategy
        for strategy_name, strategy in self.strategies.items():
            try:
                # Check if strategy is active
                if not strategy.is_active:
                    continue

                # Generate signals for each symbol
                for symbol, symbol_data in market_data_dict.items():
                    # Get data for the strategy's timeframe
                    if strategy.timeframe not in symbol_data:
                        logger.warning(f"No data for timeframe {strategy.timeframe} needed by strategy {strategy_name}")
                        continue

                    # Get data for the strategy
                    data = symbol_data[strategy.timeframe]

                    # Generate signals
                    signals = strategy.generate_signals(data)

                    if signals:
                        # Validate signals
                        signals = self.signal_validator.validate_signals(signals, current_time)

                        if not signals:
                            continue

                        # Enrich signals with market data
                        last_bar = data.iloc[-1].to_dict() if len(data) > 0 else {}
                        for signal in signals:
                            self.signal_validator.enrich_signal(signal, last_bar)

                        # Track signals if enabled
                        if self.track_signals:
                            self._track_signals(signals)

                        # Store signals by strategy
                        if strategy_name not in strategy_signals:
                            strategy_signals[strategy_name] = []

                        strategy_signals[strategy_name].extend(signals)
                        all_signals.extend(signals)
            except Exception as e:
                logger.error(f"Error generating signals for strategy {strategy_name}: {str(e)}")

        # Aggregate signals if there are multiple
        if len(strategy_signals) > 1 and self.weighted_aggregation:
            aggregated_signals = self._aggregate_signals(strategy_signals)
            return aggregated_signals

        return all_signals

    def _track_signals(self, signals: List[Signal]) -> None:
        """
        Track signals for analysis.

        Args:
            signals: List of signals to track
        """
        # Add signals to history
        for signal in signals:
            self.signal_history.append({
                "timestamp": signal.timestamp,
                "signal": signal.to_dict(),
                "tracking_time": int(time.time() * 1000)
            })

        # Limit history size
        if len(self.signal_history) > self.max_signal_history:
            self.signal_history = self.signal_history[-self.max_signal_history:]

        # Save history periodically (every 100 signals)
        if len(self.signal_history) % 100 == 0:
            self._save_signal_history()

    def _load_signal_history(self) -> None:
        """Load signal history from file."""
        try:
            if os.path.exists(self.signal_history_file):
                with open(self.signal_history_file, 'r') as f:
                    self.signal_history = json.load(f)

                logger.info(f"Loaded {len(self.signal_history)} signals from history")
        except Exception as e:
            logger.error(f"Error loading signal history: {str(e)}")
            self.signal_history = []

    def _save_signal_history(self) -> None:
        """Save signal history to file."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.signal_history_file), exist_ok=True)

            # Save to file
            with open(self.signal_history_file, 'w') as f:
                json.dump(self.signal_history, f)

            logger.debug(f"Saved {len(self.signal_history)} signals to history")
        except Exception as e:
            logger.error(f"Error saving signal history: {str(e)}")

    def get_signal_history(self, symbol: Optional[str] = None,
                          strategy_name: Optional[str] = None,
                          signal_type: Optional[SignalType] = None,
                          start_time: Optional[int] = None,
                          end_time: Optional[int] = None,
                          limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get signal history with optional filters.

        Args:
            symbol: Filter by symbol (optional)
            strategy_name: Filter by strategy name (optional)
            signal_type: Filter by signal type (optional)
            start_time: Filter by start time (optional)
            end_time: Filter by end time (optional)
            limit: Maximum number of signals to return (optional)

        Returns:
            List of signal history entries
        """
        # Apply filters
        filtered_history = self.signal_history

        if symbol:
            filtered_history = [
                entry for entry in filtered_history
                if entry["signal"]["symbol"] == symbol
            ]

        if strategy_name:
            filtered_history = [
                entry for entry in filtered_history
                if entry["signal"]["strategy_name"] == strategy_name
            ]

        if signal_type:
            filtered_history = [
                entry for entry in filtered_history
                if entry["signal"]["signal_type"] == signal_type.value
            ]

        if start_time:
            filtered_history = [
                entry for entry in filtered_history
                if entry["timestamp"] >= start_time
            ]

        if end_time:
            filtered_history = [
                entry for entry in filtered_history
                if entry["timestamp"] <= end_time
            ]

        # Sort by timestamp (newest first)
        filtered_history = sorted(
            filtered_history,
            key=lambda entry: entry["timestamp"],
            reverse=True
        )

        # Apply limit
        if limit and limit > 0:
            filtered_history = filtered_history[:limit]

        return filtered_history

    def get_signal_stats(self, symbol: Optional[str] = None,
                        strategy_name: Optional[str] = None,
                        start_time: Optional[int] = None,
                        end_time: Optional[int] = None) -> Dict[str, Any]:
        """
        Get signal statistics.

        Args:
            symbol: Filter by symbol (optional)
            strategy_name: Filter by strategy name (optional)
            start_time: Filter by start time (optional)
            end_time: Filter by end time (optional)

        Returns:
            Dictionary of signal statistics
        """
        # Get filtered history
        history = self.get_signal_history(
            symbol=symbol,
            strategy_name=strategy_name,
            start_time=start_time,
            end_time=end_time
        )

        if not history:
            return {
                "total_signals": 0,
                "signal_counts": {},
                "strategy_counts": {},
                "symbol_counts": {}
            }

        # Count by signal type
        signal_counts = {}
        for entry in history:
            signal_type = entry["signal"]["signal_type"]
            signal_counts[signal_type] = signal_counts.get(signal_type, 0) + 1

        # Count by strategy
        strategy_counts = {}
        for entry in history:
            strategy = entry["signal"]["strategy_name"]
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        # Count by symbol
        symbol_counts = {}
        for entry in history:
            sym = entry["signal"]["symbol"]
            symbol_counts[sym] = symbol_counts.get(sym, 0) + 1

        return {
            "total_signals": len(history),
            "signal_counts": signal_counts,
            "strategy_counts": strategy_counts,
            "symbol_counts": symbol_counts
        }
```

The signal generation pipeline implementation includes three key components:

1. **Signal Generation** - A comprehensive signal generation process in the base strategy class
2. **Signal Validation** - Thorough validation and enrichment of signals to ensure quality
3. **Signal Tracking** - Detailed tracking and analysis of signal history for performance evaluation

These components form the foundation of the signal generation pipeline, converting market data into structured trading signals that can be processed by the rest of the system.

### 2.2 Signal Aggregation System

#### Implementation Details

1. **Implement Multi-Strategy Signal Aggregation**:

```python
# In src/strategies/manager/aggregator.py, implement signal aggregation
from typing import Dict, Any, List, Optional, Union
from src.strategies.base_strategy import Signal, SignalType
import pandas as pd
import time
from loguru import logger

class SignalAggregator:
    """Aggregates signals from multiple strategies."""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the signal aggregator.

        Args:
            config: Aggregator configuration
        """
        self.config = config or {}

        # Configure aggregation parameters
        self.conflict_resolution = self.config.get("conflict_resolution", "confidence_weighted")
        self.min_consensus_threshold = self.config.get("min_consensus_threshold", 0.5)
        self.aggregate_by_timeframe = self.config.get("aggregate_by_timeframe", True)

        logger.info(f"Signal aggregator initialized with method: {self.conflict_resolution}")

    def aggregate_signals(self, strategy_signals: Dict[str, List[Signal]],
                         strategy_weights: Dict[str, float] = None) -> List[Signal]:
        """
        Aggregate signals from multiple strategies.

        Args:
            strategy_signals: Dictionary of strategy names to signals
            strategy_weights: Dictionary of strategy names to weights

        Returns:
            List of aggregated signals
        """
        if not strategy_signals:
            return []

        # If only one strategy, return its signals
        if len(strategy_signals) == 1:
            strategy_name = list(strategy_signals.keys())[0]
            return strategy_signals[strategy_name]

        # Group signals by symbol
        signals_by_symbol = self._group_signals_by_symbol(strategy_signals)

        # Aggregate signals for each symbol
        aggregated_signals = []

        for symbol, symbol_signals in signals_by_symbol.items():
            # Group by timeframe if needed
            if self.aggregate_by_timeframe:
                signals_by_timeframe = self._group_signals_by_timeframe(symbol_signals)

                for timeframe, timeframe_signals in signals_by_timeframe.items():
                    # Aggregate signals for this symbol and timeframe
                    agg_signal = self._aggregate_symbol_signals(
                        symbol, timeframe_signals, strategy_weights
                    )

                    if agg_signal:
                        aggregated_signals.append(agg_signal)
            else:
                # Aggregate all signals for this symbol
                agg_signal = self._aggregate_symbol_signals(
                    symbol, symbol_signals, strategy_weights
                )

                if agg_signal:
                    aggregated_signals.append(agg_signal)

        return aggregated_signals

    def _group_signals_by_symbol(self, strategy_signals: Dict[str, List[Signal]]) -> Dict[str, List[Signal]]:
        """
        Group signals by symbol.

        Args:
            strategy_signals: Dictionary of strategy names to signals

        Returns:
            Dictionary of symbols to signals
        """
        signals_by_symbol = {}

        for strategy_name, signals in strategy_signals.items():
            for signal in signals:
                if signal.symbol not in signals_by_symbol:
                    signals_by_symbol[signal.symbol] = []

                signals_by_symbol[signal.symbol].append(signal)

        return signals_by_symbol

    def _group_signals_by_timeframe(self, signals: List[Signal]) -> Dict[str, List[Signal]]:
        """
        Group signals by timeframe.

        Args:
            signals: List of signals

        Returns:
            Dictionary of timeframes to signals
        """
        signals_by_timeframe = {}

        for signal in signals:
            # Use timeframe if available, otherwise use "unknown"
            timeframe = signal.timeframe or "unknown"

            if timeframe not in signals_by_timeframe:
                signals_by_timeframe[timeframe] = []

            signals_by_timeframe[timeframe].append(signal)

        return signals_by_timeframe

    def _aggregate_symbol_signals(self, symbol: str, signals: List[Signal],
                                 strategy_weights: Dict[str, float] = None) -> Optional[Signal]:
        """
        Aggregate signals for a symbol.

        Args:
            symbol: Symbol
            signals: List of signals for the symbol
            strategy_weights: Dictionary of strategy names to weights

        Returns:
            Aggregated signal or None
        """
        if not signals:
            return None

        # If only one signal, return it
        if len(signals) == 1:
            return signals[0]

        # Use appropriate aggregation method
        if self.conflict_resolution == "confidence_weighted":
            return self._aggregate_by_confidence(symbol, signals, strategy_weights)
        elif self.conflict_resolution == "majority_vote":
            return self._aggregate_by_majority_vote(symbol, signals, strategy_weights)
        elif self.conflict_resolution == "most_recent":
            return self._aggregate_by_most_recent(symbol, signals)
        elif self.conflict_resolution == "highest_confidence":
            return self._aggregate_by_highest_confidence(symbol, signals, strategy_weights)
        else:
            logger.warning(f"Unknown conflict resolution method: {self.conflict_resolution}")
            return signals[0]  # Default to first signal

    def _aggregate_by_confidence(self, symbol: str, signals: List[Signal],
                               strategy_weights: Dict[str, float] = None) -> Optional[Signal]:
        """
        Aggregate signals by weighted confidence.

        Args:
            symbol: Symbol
            signals: List of signals for the symbol
            strategy_weights: Dictionary of strategy names to weights

        Returns:
            Aggregated signal or None
        """
        # Initialize signals by type
        signal_types = {
            SignalType.BUY: [],
            SignalType.SELL: [],
            SignalType.CLOSE_LONG: [],
            SignalType.CLOSE_SHORT: []
        }

        # Group signals by type
        for signal in signals:
            if signal.signal_type in signal_types:
                signal_types[signal.signal_type].append(signal)

        # Calculate weighted confidence for each signal type
        weighted_confidence = {}

        for signal_type, type_signals in signal_types.items():
            if not type_signals:
                weighted_confidence[signal_type] = 0.0
                continue

            total_confidence = 0.0

            for signal in type_signals:
                # Get strategy weight (default to 1.0)
                strategy_weight = 1.0
                if strategy_weights and signal.strategy_name in strategy_weights:
                    strategy_weight = strategy_weights[signal.strategy_name]

                # Add weighted confidence
                total_confidence += signal.confidence * strategy_weight

            # Normalize by number of signals
            weighted_confidence[signal_type] = total_confidence / len(signals)

        # Find signal type with highest weighted confidence
        best_type = max(weighted_confidence, key=weighted_confidence.get)
        best_confidence = weighted_confidence[best_type]

        # Check if confidence is above threshold
        if best_confidence < self.min_consensus_threshold:
            return None

        # Get signals of the best type
        best_signals = signal_types[best_type]

        if not best_signals:
            return None

        # Create aggregated signal
        timestamp = int(time.time() * 1000)

        # Use average price from signals
        avg_price = sum(s.price for s in best_signals) / len(best_signals)

        # Get timeframe from majority of signals
        timeframe_count = {}
        for signal in best_signals:
            tf = signal.timeframe or "unknown"
            timeframe_count[tf] = timeframe_count.get(tf, 0) + 1

        timeframe = max(timeframe_count, key=timeframe_count.get)

        # Create metadata with contributing strategies
        metadata = {
            "aggregation_method": "confidence_weighted",
            "contributing_strategies": [signal.strategy_name for signal in best_signals],
            "weighted_confidence": best_confidence,
            "raw_confidence_values": [signal.confidence for signal in best_signals]
        }

        # Create aggregated signal
        agg_signal = Signal(
            signal_type=best_type,
            symbol=symbol,
            timestamp=timestamp,
            price=avg_price,
            confidence=best_confidence,
            strategy_name="aggregated",
            timeframe=timeframe,
            metadata=metadata
        )

        return agg_signal

    def _aggregate_by_majority_vote(self, symbol: str, signals: List[Signal],
                                  strategy_weights: Dict[str, float] = None) -> Optional[Signal]:
        """
        Aggregate signals by majority vote.

        Args:
            symbol: Symbol
            signals: List of signals for the symbol
            strategy_weights: Dictionary of strategy names to weights

        Returns:
            Aggregated signal or None
        """
        # Count votes for each signal type
        votes = {
            SignalType.BUY: 0,
            SignalType.SELL: 0,
            SignalType.CLOSE_LONG: 0,
            SignalType.CLOSE_SHORT: 0
        }

        # Count signals by type, weighted by strategy weight
        total_weight = 0

        for signal in signals:
            # Get strategy weight (default to 1.0)
            strategy_weight = 1.0
            if strategy_weights and signal.strategy_name in strategy_weights:
                strategy_weight = strategy_weights[signal.strategy_name]

            # Add vote
            if signal.signal_type in votes:
                votes[signal.signal_type] += strategy_weight

            total_weight += strategy_weight

        # Find signal type with most votes
        best_type = max(votes, key=votes.get)
        best_votes = votes[best_type]

        # Calculate consensus percentage
        consensus = best_votes / total_weight if total_weight > 0 else 0

        # Check if consensus is above threshold
        if consensus < self.min_consensus_threshold:
            return None

        # Get signals of the best type
        best_signals = [s for s in signals if s.signal_type == best_type]

        if not best_signals:
            return None

        # Create aggregated signal
        timestamp = int(time.time() * 1000)

        # Use average price from signals
        avg_price = sum(s.price for s in best_signals) / len(best_signals)

        # Use average confidence from signals
        avg_confidence = sum(s.confidence for s in best_signals) / len(best_signals)

        # Get timeframe from majority of signals
        timeframe_count = {}
        for signal in best_signals:
            tf = signal.timeframe or "unknown"
            timeframe_count[tf] = timeframe_count.get(tf, 0) + 1

        timeframe = max(timeframe_count, key=timeframe_count.get)

        # Create metadata with voting results
        metadata = {
            "aggregation_method": "majority_vote",
            "contributing_strategies": [signal.strategy_name for signal in best_signals],
            "vote_results": {k.value: v for k, v in votes.items()},
            "consensus": consensus,
            "total_votes": total_weight
        }

        # Create aggregated signal
        agg_signal = Signal(
            signal_type=best_type,
            symbol=symbol,
            timestamp=timestamp,
            price=avg_price,
            confidence=avg_confidence,
            strategy_name="aggregated",
            timeframe=timeframe,
            metadata=metadata
        )

        return agg_signal

    def _aggregate_by_most_recent(self, symbol: str, signals: List[Signal]) -> Optional[Signal]:
        """
        Aggregate signals by most recent.

        Args:
            symbol: Symbol
            signals: List of signals for the symbol

        Returns:
            Most recent signal
        """
        if not signals:
            return None

        # Find most recent signal
        most_recent = max(signals, key=lambda s: s.timestamp)

        # Add metadata
        most_recent.metadata["aggregation_method"] = "most_recent"
        most_recent.metadata["total_signals"] = len(signals)

        return most_recent

    def _aggregate_by_highest_confidence(self, symbol: str, signals: List[Signal],
                                       strategy_weights: Dict[str, float] = None) -> Optional[Signal]:
        """
        Aggregate signals by highest confidence.

        Args:
            symbol: Symbol
            signals: List of signals for the symbol
            strategy_weights: Dictionary of strategy names to weights

        Returns:
            Signal with highest confidence
        """
        if not signals:
            return None

        # Calculate weighted confidence for each signal
        weighted_signals = []

        for signal in signals:
            # Get strategy weight (default to 1.0)
            strategy_weight = 1.0
            if strategy_weights and signal.strategy_name in strategy_weights:
                strategy_weight = strategy_weights[signal.strategy_name]

            # Calculate weighted confidence
            weighted_confidence = signal.confidence * strategy_weight

            weighted_signals.append((signal, weighted_confidence))

        # Find signal with highest weighted confidence
        best_signal, best_confidence = max(weighted_signals, key=lambda x: x[1])

        # Add metadata
        best_signal.metadata["aggregation_method"] = "highest_confidence"
        best_signal.metadata["weighted_confidence"] = best_confidence
        best_signal.metadata["total_signals"] = len(signals)

        return best_signal
```

2. **Add Signal Conflict Resolution**:

```python
# In src/strategies/manager/core.py, enhance strategy manager with signal conflict resolution
from src.strategies.manager.aggregator import SignalAggregator

class StrategyManager:
    # Add to existing class

    def __init__(self, config: Dict[str, Any] = None, indicator_manager = None):
        # Existing initialization code

        # Initialize signal aggregator
        self.signal_aggregator = SignalAggregator(config.get("aggregator", {}))
        self.weighted_aggregation = config.get("weighted_aggregation", True)

        # Initialize strategy weights
        self.strategy_weights = config.get("strategy_weights", {})
        self.default_weight = config.get("default_strategy_weight", 1.0)

    def _aggregate_signals(self, strategy_signals: Dict[str, List[Signal]]) -> List[Signal]:
        """
        Aggregate signals from multiple strategies.

        Args:
            strategy_signals: Dictionary of strategy names to signals

        Returns:
            List of aggregated signals
        """
        # Use signal aggregator
        return self.signal_aggregator.aggregate_signals(
            strategy_signals=strategy_signals,
            strategy_weights=self.strategy_weights
        )

    def set_strategy_weight(self, strategy_name: str, weight: float) -> None:
        """
        Set weight for a strategy.

        Args:
            strategy_name: Strategy name
            weight: Weight value
        """
        self.strategy_weights[strategy_name] = weight
        logger.info(f"Set weight for strategy {strategy_name} to {weight}")

    def get_strategy_weights(self) -> Dict[str, float]:
        """
        Get all strategy weights.

        Returns:
            Dictionary of strategy names to weights
        """
        # Ensure all strategies have weights
        weights = self.strategy_weights.copy()

        for strategy_name in self.strategies.keys():
            if strategy_name not in weights:
                weights[strategy_name] = self.default_weight

        return weights

    def calculate_strategy_weights(self, performance_metrics: Dict[str, Dict[str, float]] = None) -> None:
        """
        Calculate strategy weights based on performance.

        Args:
            performance_metrics: Dictionary of strategy names to performance metrics
        """
        if not performance_metrics:
            return

        # Calculate weights based on performance
        new_weights = {}

        for strategy_name, metrics in performance_metrics.items():
            if strategy_name not in self.strategies:
                continue

            # Get performance metric (profit factor, win rate, etc.)
            # Use profit factor if available, otherwise use win rate
            if "profit_factor" in metrics and metrics["profit_factor"] > 0:
                performance = metrics["profit_factor"]
            elif "win_rate" in metrics and metrics["win_rate"] > 0:
                performance = metrics["win_rate"]
            else:
                performance = 1.0

            # Set weight based on performance
            new_weights[strategy_name] = performance

        # Normalize weights
        total_weight = sum(new_weights.values())

        if total_weight > 0:
            for strategy_name in new_weights:
                new_weights[strategy_name] = new_weights[strategy_name] / total_weight * len(new_weights)

        # Update weights
        self.strategy_weights.update(new_weights)

        logger.info(f"Updated strategy weights based on performance: {self.strategy_weights}")
```

3. **Create Confidence-Weighted Signal Processing**:

```python
# In src/strategies/processor.py, implement confidence-weighted signal processing
from typing import Dict, Any, List, Optional, Union
from src.strategies.base_strategy import Signal, SignalType
import time
from loguru import logger

class SignalProcessor:
    """Processes signals for trade decisions."""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the signal processor.

        Args:
            config: Processor configuration
        """
        self.config = config or {}

        # Configure processing parameters
        self.min_execution_confidence = self.config.get("min_execution_confidence", 0.7)
        self.confidence_scaling = self.config.get("confidence_scaling", True)
        self.position_sizing_method = self.config.get("position_sizing_method", "confidence")

        logger.info(f"Signal processor initialized with min confidence: {self.min_execution_confidence}")

    def process_signals(self, signals: List[Signal]) -> List[Dict[str, Any]]:
        """
        Process signals into trade instructions.

        Args:
            signals: List of signals to process

        Returns:
            List of trade instructions
        """
        if not signals:
            return []

        # Filter by minimum confidence
        filtered_signals = [
            s for s in signals
            if s.confidence >= self.min_execution_confidence
        ]

        if not filtered_signals:
            logger.debug(f"No signals meet minimum confidence threshold of {self.min_execution_confidence}")
            return []

        # Convert signals to trade instructions
        trades = []

        for signal in filtered_signals:
            try:
                trade = self._signal_to_trade(signal)
                trades.append(trade)
            except Exception as e:
                logger.error(f"Error converting signal to trade: {str(e)}")

        return trades

    def _signal_to_trade(self, signal: Signal) -> Dict[str, Any]:
        """
        Convert a signal to trade instruction.

        Args:
            signal: Signal to convert

        Returns:
            Trade instruction
        """
        # Base trade data
        trade = {
            "symbol": signal.symbol,
            "timestamp": signal.timestamp,
            "creation_time": int(time.time() * 1000),
            "signal_confidence": signal.confidence,
            "strategy": signal.strategy_name,
            "signal_type": signal.signal_type.value,
            "metadata": signal.metadata.copy()
        }

        # Add trade type and parameters based on signal type
        if signal.signal_type == SignalType.BUY:
            trade["trade_type"] = "buy"
            trade["side"] = "buy"
            trade["reduce_only"] = False

        elif signal.signal_type == SignalType.SELL:
            trade["trade_type"] = "sell"
            trade["side"] = "sell"
            trade["reduce_only"] = False

        elif signal.signal_type == SignalType.CLOSE_LONG:
            trade["trade_type"] = "close"
            trade["side"] = "sell"
            trade["reduce_only"] = True

        elif signal.signal_type == SignalType.CLOSE_SHORT:
            trade["trade_type"] = "close"
            trade["side"] = "buy"
            trade["reduce_only"] = True

        else:
            logger.warning(f"Unknown signal type: {signal.signal_type}")
            trade["trade_type"] = "unknown"

        # Add position sizing
        trade["position_size_factor"] = self._calculate_position_size_factor(signal)

        return trade

    def _calculate_position_size_factor(self, signal: Signal) -> float:
        """
        Calculate position size factor based on signal confidence.

        Args:
            signal: Signal

        Returns:
            Position size factor (0.0 to 1.0)
        """
        if not self.confidence_scaling:
            return 1.0

        if self.position_sizing_method == "confidence":
            # Use signal confidence directly
            return signal.confidence

        elif self.position_sizing_method == "square_root":
            # Use square root of confidence (less aggressive scaling)
            return min(1.0, max(0.0, signal.confidence ** 0.5))

        elif self.position_sizing_method == "step":
            # Step function (low, medium, high confidence)
            if signal.confidence >= 0.9:
                return 1.0
            elif signal.confidence >= 0.8:
                return 0.75
            elif signal.confidence >= 0.7:
                return 0.5
            else:
                return 0.25

        else:
            logger.warning(f"Unknown position sizing method: {self.position_sizing_method}")
            return 1.0
```

The signal aggregation system implementation includes three key components:

1. **Multi-Strategy Aggregation** - A flexible system for combining signals from multiple strategies
2. **Conflict Resolution** - Multiple methods for resolving conflicts between different strategies
3. **Confidence-Weighted Processing** - Processing of signals into trade instructions with confidence-based position sizing

These components work together to provide a robust signal aggregation system that can handle signals from multiple strategies with different characteristics and confidence levels.

### 2.3 Decision Making Pipeline

#### Implementation Details

1. **Implement Decision Making Based on Aggregated Signals**:

```python
# In src/trade_management/decision_maker.py, implement decision making pipeline
from typing import Dict, Any, List, Optional, Union
from src.strategies.base_strategy import Signal, SignalType
from src.strategies.processor import SignalProcessor
import time
import threading
import datetime
from loguru import logger

class DecisionMaker:
    """Makes trading decisions based on signals."""

    def __init__(self, config: Dict[str, Any] = None,
                trade_manager = None, risk_manager = None):
        """
        Initialize the decision maker.

        Args:
            config: Decision maker configuration
            trade_manager: Trade manager instance
            risk_manager: Risk manager instance
        """
        self.config = config or {}
        self.trade_manager = trade_manager
        self.risk_manager = risk_manager

        # Initialize signal processor
        self.signal_processor = SignalProcessor(config.get("signal_processor", {}))

        # Configure decision parameters
        self.enable_trading = self.config.get("enable_trading", True)
        self.trading_hours = self.config.get("trading_hours", {})
        self.cooldown_period_sec = self.config.get("cooldown_period_sec", 300)  # 5 minutes

        # Track last decision time for cooldown
        self.last_decision_time = {}  # Symbol -> last decision time
        self.lock = threading.RLock()

        logger.info("Decision maker initialized")

    def process_signals(self, signals: List[Signal]) -> List[Dict[str, Any]]:
        """
        Process signals and make trading decisions.

        Args:
            signals: List of signals to process

        Returns:
            List of trade decisions
        """
        if not signals:
            return []

        # Check if trading is enabled
        if not self.enable_trading:
            logger.info("Trading is disabled, not processing signals")
            return []

        # Filter signals during cooldown period
        filtered_signals = self._filter_cooldown_signals(signals)

        if not filtered_signals:
            return []

        # Check trading hours if configured
        if self.trading_hours:
            filtered_signals = self._filter_trading_hours(filtered_signals)

            if not filtered_signals:
                return []

        # Process signals into trade instructions
        trade_instructions = self.signal_processor.process_signals(filtered_signals)

        if not trade_instructions:
            return []

        # Apply risk management if available
        if self.risk_manager:
            trade_decisions = self._apply_risk_management(trade_instructions)
        else:
            trade_decisions = trade_instructions

        # Update last decision time for cooldown
        self._update_decision_times(trade_decisions)

        # Execute trades if trade manager is available
        if self.trade_manager and trade_decisions:
            self._execute_trades(trade_decisions)

        return trade_decisions

    def _filter_cooldown_signals(self, signals: List[Signal]) -> List[Signal]:
        """
        Filter signals that are in cooldown period.

        Args:
            signals: List of signals to filter

        Returns:
            Filtered list of signals
        """
        if self.cooldown_period_sec <= 0:
            return signals

        current_time = time.time()
        filtered_signals = []

        with self.lock:
            for signal in signals:
                # Check if symbol is in cooldown
                if signal.symbol in self.last_decision_time:
                    last_time = self.last_decision_time[signal.symbol]
                    elapsed = current_time - last_time

                    if elapsed < self.cooldown_period_sec:
                        logger.debug(f"Signal for {signal.symbol} in cooldown period ({elapsed:.1f}s < {self.cooldown_period_sec}s)")
                        continue

                filtered_signals.append(signal)

        return filtered_signals

    def _filter_trading_hours(self, signals: List[Signal]) -> List[Signal]:
        """
        Filter signals based on trading hours.

        Args:
            signals: List of signals to filter

        Returns:
            Filtered list of signals
        """
        if not self.trading_hours:
            return signals

        # Get current time
        now = datetime.datetime.now().time()

        # Check if current time is within trading hours
        is_trading_hours = False

        for start_str, end_str in self.trading_hours.get("periods", []):
            try:
                start_time = datetime.datetime.strptime(start_str, "%H:%M").time()
                end_time = datetime.datetime.strptime(end_str, "%H:%M").time()

                if start_time <= now <= end_time:
                    is_trading_hours = True
                    break
            except ValueError:
                logger.error(f"Invalid time format in trading hours: {start_str} - {end_str}")

        # Check excluded days
        today = datetime.datetime.now().strftime("%A").lower()
        excluded_days = [day.lower() for day in self.trading_hours.get("excluded_days", [])]

        if today in excluded_days:
            logger.info(f"Today ({today}) is excluded from trading hours")
            return []

        # Return all signals or none based on trading hours
        if is_trading_hours:
            return signals
        else:
            logger.info(f"Current time {now} is outside trading hours")
            return []

    def _apply_risk_management(self, trade_instructions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply risk management to trade instructions.

        Args:
            trade_instructions: List of trade instructions

        Returns:
            List of risk-adjusted trade decisions
        """
        risk_adjusted_trades = []

        for trade in trade_instructions:
            try:
                # Add risk assessment
                risk_assessment = self.risk_manager.assess_trade_risk(trade)

                # Skip trade if risk is too high
                if not risk_assessment.get("approved", False):
                    logger.info(f"Trade for {trade['symbol']} rejected by risk manager: {risk_assessment.get('reason', 'Unknown reason')}")
                    continue

                # Add risk assessment to trade
                trade["risk_assessment"] = risk_assessment

                # Apply position sizing
                trade = self._apply_position_sizing(trade, risk_assessment)

                risk_adjusted_trades.append(trade)

            except Exception as e:
                logger.error(f"Error applying risk management to trade: {str(e)}")

        return risk_adjusted_trades

    def _apply_position_sizing(self, trade: Dict[str, Any],
                             risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply position sizing to trade based on risk assessment.

        Args:
            trade: Trade instruction
            risk_assessment: Risk assessment

        Returns:
            Trade with position sizing
        """
        # Get base position size from risk assessment
        base_size = risk_assessment.get("position_size", 0.0)

        # Apply confidence-based scaling
        position_size_factor = trade.get("position_size_factor", 1.0)
        trade["position_size"] = base_size * position_size_factor

        # Apply maximum position limits
        max_size = risk_assessment.get("max_position_size", float("inf"))
        trade["position_size"] = min(trade["position_size"], max_size)

        # Round position size to appropriate precision
        symbol_precision = risk_assessment.get("symbol_precision", 8)
        trade["position_size"] = round(trade["position_size"], symbol_precision)

        # Add position sizing explanation
        trade["position_sizing"] = {
            "base_size": base_size,
            "size_factor": position_size_factor,
            "max_size": max_size,
            "final_size": trade["position_size"]
        }

        return trade

    def _execute_trades(self, trade_decisions: List[Dict[str, Any]]) -> None:
        """
        Execute trades using trade manager.

        Args:
            trade_decisions: List of trade decisions
        """
        if not self.trade_manager:
            logger.warning("No trade manager available, cannot execute trades")
            return

        for trade in trade_decisions:
            try:
                # Execute the trade
                result = self.trade_manager.execute_trade(trade)

                if result.get("success", False):
                    logger.info(f"Successfully executed trade for {trade['symbol']}")
                else:
                    logger.warning(f"Failed to execute trade for {trade['symbol']}: {result.get('error', 'Unknown error')}")

            except Exception as e:
                logger.error(f"Error executing trade: {str(e)}")

    def _update_decision_times(self, trade_decisions: List[Dict[str, Any]]) -> None:
        """
        Update last decision times for cooldown.

        Args:
            trade_decisions: List of trade decisions
        """
        current_time = time.time()

        with self.lock:
            for trade in trade_decisions:
                symbol = trade["symbol"]
                self.last_decision_time[symbol] = current_time
```

2. **Add Risk-Adjusted Position Sizing**:

```python
# In src/risk_management/position_sizer.py, implement risk-adjusted position sizing
from typing import Dict, Any, List, Optional, Union
import math
from loguru import logger

class PositionSizer:
    """Calculates position sizes based on risk parameters."""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the position sizer.

        Args:
            config: Position sizer configuration
        """
        self.config = config or {}

        # Risk parameters
        self.default_risk_percent = self.config.get("default_risk_percent", 1.0)
        self.max_position_size_percent = self.config.get("max_position_size_percent", 5.0)
        self.use_asymmetric_sizing = self.config.get("use_asymmetric_sizing", False)
        self.asymmetric_factor = self.config.get("asymmetric_factor", 1.5)

        logger.info(f"Position sizer initialized with risk percent: {self.default_risk_percent}%")

    def calculate_position_size(self,
                               account_balance: float,
                               signal_type: str,
                               symbol: str,
                               current_price: float,
                               stop_price: Optional[float] = None,
                               risk_percent: Optional[float] = None,
                               confidence: float = 1.0,
                               symbol_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Calculate position size based on risk parameters.

        Args:
            account_balance: Account balance
            signal_type: Signal type (buy, sell, close_long, close_short)
            symbol: Trading symbol
            current_price: Current price
            stop_price: Stop price (optional)
            risk_percent: Risk percent (optional)
            confidence: Signal confidence (0.0 to 1.0)
            symbol_info: Symbol information (optional)

        Returns:
            Dictionary with position sizing details
        """
        # Use provided risk percent or default
        risk_percent = risk_percent if risk_percent is not None else self.default_risk_percent

        # Adjust risk based on confidence
        adjusted_risk_percent = risk_percent * confidence

        # Calculate dollar risk amount
        risk_amount = account_balance * (adjusted_risk_percent / 100.0)

        # Get position size based on method
        if stop_price is not None and stop_price > 0:
            # Risk-based position sizing
            position_size = self._calculate_risk_based_size(
                risk_amount=risk_amount,
                current_price=current_price,
                stop_price=stop_price,
                symbol=symbol,
                symbol_info=symbol_info
            )
        else:
            # Percentage-based position sizing
            position_size = self._calculate_percentage_based_size(
                account_balance=account_balance,
                adjusted_risk_percent=adjusted_risk_percent,
                current_price=current_price,
                symbol=symbol,
                symbol_info=symbol_info
            )

        # Apply asymmetric sizing if enabled
        if self.use_asymmetric_sizing:
            position_size = self._apply_asymmetric_sizing(
                position_size=position_size,
                signal_type=signal_type
            )

        # Calculate position value
        position_value = position_size * current_price

        # Calculate percentage of account
        account_percentage = (position_value / account_balance) * 100

        # Ensure max position size is not exceeded
        if account_percentage > self.max_position_size_percent:
            # Scale down position size
            scale_factor = self.max_position_size_percent / account_percentage
            position_size *= scale_factor
            position_value = position_size * current_price
            account_percentage = self.max_position_size_percent

        # Round position size based on symbol precision
        position_size = self._round_position_size(position_size, symbol_info)

        # Return position sizing details
        return {
            "position_size": position_size,
            "position_value": position_size * current_price,
            "account_percentage": account_percentage,
            "risk_amount": risk_amount,
            "risk_percent": adjusted_risk_percent
        }

    def _calculate_risk_based_size(self, risk_amount: float, current_price: float,
                                 stop_price: float, symbol: str,
                                 symbol_info: Dict[str, Any] = None) -> float:
        """
        Calculate position size based on risk amount and stop price.

        Args:
            risk_amount: Risk amount in dollars
            current_price: Current price
            stop_price: Stop price
            symbol: Trading symbol
            symbol_info: Symbol information (optional)

        Returns:
            Position size
        """
        # Calculate price difference
        price_diff = abs(current_price - stop_price)

        if price_diff <= 0:
            logger.warning(f"Invalid price difference for {symbol}: {price_diff}")
            return 0.0

        # Calculate risk per unit
        risk_per_unit = price_diff

        # Calculate position size
        position_size = risk_amount / risk_per_unit

        return position_size

    def _calculate_percentage_based_size(self, account_balance: float,
                                       adjusted_risk_percent: float,
                                       current_price: float, symbol: str,
                                       symbol_info: Dict[str, Any] = None) -> float:
        """
        Calculate position size based on percentage of account.

        Args:
            account_balance: Account balance
            adjusted_risk_percent: Adjusted risk percent
            current_price: Current price
            symbol: Trading symbol
            symbol_info: Symbol information (optional)

        Returns:
            Position size
        """
        # Calculate position value
        position_value = account_balance * (adjusted_risk_percent / 100.0)

        # Calculate position size
        position_size = position_value / current_price

        return position_size

    def _apply_asymmetric_sizing(self, position_size: float, signal_type: str) -> float:
        """
        Apply asymmetric position sizing based on signal type.

        Args:
            position_size: Base position size
            signal_type: Signal type

        Returns:
            Adjusted position size
        """
        if not self.use_asymmetric_sizing:
            return position_size

        # Apply asymmetric factor for buy signals (long positions)
        if signal_type.lower() in ["buy"]:
            return position_size * self.asymmetric_factor
        # Apply inverse factor for sell signals (short positions)
        elif signal_type.lower() in ["sell"]:
            return position_size / self.asymmetric_factor
        else:
            return position_size

    def _round_position_size(self, position_size: float,
                           symbol_info: Dict[str, Any] = None) -> float:
        """
        Round position size based on symbol precision.

        Args:
            position_size: Position size
            symbol_info: Symbol information (optional)

        Returns:
            Rounded position size
        """
        # Get precision from symbol info or use default
        precision = 6
        min_qty = 0.0

        if symbol_info:
            precision = symbol_info.get("qty_precision", 6)
            min_qty = symbol_info.get("min_qty", 0.0)

        # Round position size
        rounded_size = round(position_size, precision)

        # Ensure minimum quantity
        if min_qty > 0 and rounded_size < min_qty:
            rounded_size = min_qty

        return rounded_size
```

3. **Create Trade Decision Documentation**:

```python
# In src/trade_management/trade_recorder.py, implement trade decision documentation
from typing import Dict, Any, List, Optional, Union
import json
import os
import time
import threading
from datetime import datetime
from loguru import logger

class TradeRecorder:
    """Records trade decisions for analysis and documentation."""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the trade recorder.

        Args:
            config: Recorder configuration
        """
        self.config = config or {}

        # Configure recorder
        self.enable_recording = self.config.get("enable_recording", True)
        self.record_directory = self.config.get("record_directory", "data/trade_records")
        self.max_records_in_memory = self.config.get("max_records_in_memory", 1000)

        # Initialize record storage
        self.trade_decisions = []
        self.trade_executions = []
        self.trade_results = []

        # Create record directory
        if self.enable_recording:
            os.makedirs(self.record_directory, exist_ok=True)

        # Initialize lock for thread safety
        self.lock = threading.RLock()

        logger.info(f"Trade recorder initialized at {self.record_directory}")

    def record_decision(self, trade_decision: Dict[str, Any]) -> None:
        """
        Record a trade decision.

        Args:
            trade_decision: Trade decision to record
        """
        if not self.enable_recording:
            return

        with self.lock:
            # Add timestamp if not present
            if "record_time" not in trade_decision:
                trade_decision["record_time"] = int(time.time() * 1000)

            # Add decision to records
            self.trade_decisions.append(trade_decision)

            # Save to file
            self._save_decision(trade_decision)

            # Limit records in memory
            if len(self.trade_decisions) > self.max_records_in_memory:
                self.trade_decisions = self.trade_decisions[-self.max_records_in_memory:]

    def record_execution(self, trade_execution: Dict[str, Any]) -> None:
        """
        Record a trade execution.

        Args:
            trade_execution: Trade execution to record
        """
        if not self.enable_recording:
            return

        with self.lock:
            # Add timestamp if not present
            if "record_time" not in trade_execution:
                trade_execution["record_time"] = int(time.time() * 1000)

            # Add execution to records
            self.trade_executions.append(trade_execution)

            # Save to file
            self._save_execution(trade_execution)

            # Limit records in memory
            if len(self.trade_executions) > self.max_records_in_memory:
                self.trade_executions = self.trade_executions[-self.max_records_in_memory:]

    def record_result(self, trade_result: Dict[str, Any]) -> None:
        """
        Record a trade result.

        Args:
            trade_result: Trade result to record
        """
        if not self.enable_recording:
            return

        with self.lock:
            # Add timestamp if not present
            if "record_time" not in trade_result:
                trade_result["record_time"] = int(time.time() * 1000)

            # Add result to records
            self.trade_results.append(trade_result)

            # Save to file
            self._save_result(trade_result)

            # Limit records in memory
            if len(self.trade_results) > self.max_records_in_memory:
                self.trade_results = self.trade_results[-self.max_records_in_memory:]

    def get_trade_history(self, symbol: Optional[str] = None,
                        strategy: Optional[str] = None,
                        start_time: Optional[int] = None,
                        end_time: Optional[int] = None,
                        limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get trade history with optional filters.

        Args:
            symbol: Filter by symbol (optional)
            strategy: Filter by strategy (optional)
            start_time: Filter by start time (optional)
            end_time: Filter by end time (optional)
            limit: Maximum number of trades to return (optional)

        Returns:
            List of trade records
        """
        with self.lock:
            # Start with all executions
            history = self.trade_executions.copy()

            # Apply filters
            if symbol:
                history = [t for t in history if t.get("symbol") == symbol]

            if strategy:
                history = [t for t in history if t.get("strategy") == strategy]

            if start_time:
                history = [t for t in history if t.get("record_time", 0) >= start_time]

            if end_time:
                history = [t for t in history if t.get("record_time", 0) <= end_time]

            # Sort by timestamp (newest first)
            history = sorted(history, key=lambda t: t.get("record_time", 0), reverse=True)

            # Apply limit
            if limit and limit > 0:
                history = history[:limit]

            return history

    def get_decision_history(self, symbol: Optional[str] = None,
                           strategy: Optional[str] = None,
                           start_time: Optional[int] = None,
                           end_time: Optional[int] = None,
                           limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get decision history with optional filters.

        Args:
            symbol: Filter by symbol (optional)
            strategy: Filter by strategy (optional)
            start_time: Filter by start time (optional)
            end_time: Filter by end time (optional)
            limit: Maximum number of decisions to return (optional)

        Returns:
            List of decision records
        """
        with self.lock:
            # Start with all decisions
            history = self.trade_decisions.copy()

            # Apply filters
            if symbol:
                history = [d for d in history if d.get("symbol") == symbol]

            if strategy:
                history = [d for d in history if d.get("strategy") == strategy]

            if start_time:
                history = [d for d in history if d.get("record_time", 0) >= start_time]

            if end_time:
                history = [d for d in history if d.get("record_time", 0) <= end_time]

            # Sort by timestamp (newest first)
            history = sorted(history, key=lambda d: d.get("record_time", 0), reverse=True)

            # Apply limit
            if limit and limit > 0:
                history = history[:limit]

            return history

    def _save_decision(self, decision: Dict[str, Any]) -> None:
        """
        Save a trade decision to file.

        Args:
            decision: Trade decision to save
        """
        try:
            # Create decision directory
            decision_dir = os.path.join(self.record_directory, "decisions")
            os.makedirs(decision_dir, exist_ok=True)

            # Create date directory
            date_str = datetime.fromtimestamp(decision.get("record_time", time.time() * 1000) / 1000).strftime("%Y-%m-%d")
            date_dir = os.path.join(decision_dir, date_str)
            os.makedirs(date_dir, exist_ok=True)

            # Create file name with timestamp and ID
            timestamp = decision.get("record_time", int(time.time() * 1000))
            decision_id = decision.get("id", str(timestamp))
            file_name = f"{timestamp}_{decision_id}.json"
            file_path = os.path.join(date_dir, file_name)

            # Save as JSON
            with open(file_path, "w") as f:
                json.dump(decision, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving trade decision: {str(e)}")

    def _save_execution(self, execution: Dict[str, Any]) -> None:
        """
        Save a trade execution to file.

        Args:
            execution: Trade execution to save
        """
        try:
            # Create execution directory
            execution_dir = os.path.join(self.record_directory, "executions")
            os.makedirs(execution_dir, exist_ok=True)

            # Create date directory
            date_str = datetime.fromtimestamp(execution.get("record_time", time.time() * 1000) / 1000).strftime("%Y-%m-%d")
            date_dir = os.path.join(execution_dir, date_str)
            os.makedirs(date_dir, exist_ok=True)

            # Create file name with timestamp and ID
            timestamp = execution.get("record_time", int(time.time() * 1000))
            execution_id = execution.get("order_id", str(timestamp))
            file_name = f"{timestamp}_{execution_id}.json"
            file_path = os.path.join(date_dir, file_name)

            # Save as JSON
            with open(file_path, "w") as f:
                json.dump(execution, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving trade execution: {str(e)}")

    def _save_result(self, result: Dict[str, Any]) -> None:
        """
        Save a trade result to file.

        Args:
            result: Trade result to save
        """
        try:
            # Create result directory
            result_dir = os.path.join(self.record_directory, "results")
            os.makedirs(result_dir, exist_ok=True)

            # Create date directory
            date_str = datetime.fromtimestamp(result.get("record_time", time.time() * 1000) / 1000).strftime("%Y-%m-%d")
            date_dir = os.path.join(result_dir, date_str)
            os.makedirs(date_dir, exist_ok=True)

            # Create file name with timestamp and ID
            timestamp = result.get("record_time", int(time.time() * 1000))
            result_id = result.get("trade_id", str(timestamp))
            file_name = f"{timestamp}_{result_id}.json"
            file_path = os.path.join(date_dir, file_name)

            # Save as JSON
            with open(file_path, "w") as f:
                json.dump(result, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving trade result: {str(e)}")
```

The decision making pipeline implementation includes three key components:

1. **Decision Making** - A comprehensive pipeline for converting signals into trade decisions
2. **Position Sizing** - Risk-adjusted position sizing based on account balance and confidence
3. **Trade Documentation** - Detailed recording of trade decisions, executions, and results

These components work together to provide a robust decision making pipeline that converts strategy signals into executable trades with appropriate risk management and documentation.
