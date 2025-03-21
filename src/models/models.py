"""
Core data models for the Algorithmic Trading System.

This file contains the core data models used throughout the system,
including signal types, trade signals, and other shared data structures.
"""

from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any, List


class SignalType(Enum):
    """Signal types for trading decisions."""

    BUY = "BUY"
    SELL = "SELL"
    EXIT_LONG = "EXIT_LONG"
    EXIT_SHORT = "EXIT_SHORT"
    NONE = "NONE"


class SignalStrength(Enum):
    """Signal strength indicators for trading decisions."""

    VERY_STRONG = "VERY_STRONG"
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"
    NEUTRAL = "NEUTRAL"
    UNKNOWN = "UNKNOWN"


@dataclass
class Signal:
    """Trading signal generated by strategies."""

    symbol: str
    signal_type: SignalType
    timestamp: datetime
    price: float
    strategy_name: str
    timeframe: str
    confidence: float = 0.0
    volume: Optional[float] = None
    strength: Any = SignalStrength.UNKNOWN
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        # Ensure metadata is a dictionary
        if self.metadata is None:
            self.metadata = {}

        # Convert string signal_type to enum if necessary
        if isinstance(self.signal_type, str):
            self.signal_type = SignalType[self.signal_type]

        # Convert strength to enum if necessary
        if isinstance(self.strength, str):
            self.strength = SignalStrength[self.strength]
        # Handle numeric strength values (like numpy.float64)
        elif isinstance(self.strength, (float, int)) or hasattr(self.strength, "item"):
            # If it's a numpy type that has item() method, convert to native Python float
            if hasattr(self.strength, "item"):
                numeric_strength = float(self.strength.item())
            else:
                numeric_strength = float(self.strength)

            # Map numeric strength to enum values
            if numeric_strength >= 0.8:
                self.strength = SignalStrength.VERY_STRONG
            elif numeric_strength >= 0.6:
                self.strength = SignalStrength.STRONG
            elif numeric_strength >= 0.4:
                self.strength = SignalStrength.MODERATE
            elif numeric_strength >= 0.2:
                self.strength = SignalStrength.WEAK
            else:
                self.strength = SignalStrength.NEUTRAL

    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary."""
        return {
            "symbol": self.symbol,
            "signal_type": self.signal_type.value,
            "timestamp": self.timestamp.isoformat(),
            "price": self.price,
            "strategy_name": self.strategy_name,
            "timeframe": self.timeframe,
            "confidence": self.confidence,
            "volume": self.volume,
            "strength": self.strength.value,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Signal":
        """Create a Signal instance from a dictionary."""
        # Convert string timestamp to datetime
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])

        # Convert string signal_type to enum
        if "signal_type" in data and isinstance(data["signal_type"], str):
            data["signal_type"] = SignalType(data["signal_type"])

        # Convert string strength to enum
        if "strength" in data and isinstance(data["strength"], str):
            data["strength"] = SignalStrength[data["strength"]]

        return cls(**data)
