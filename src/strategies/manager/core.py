"""
Strategy Manager Core for the Algorithmic Trading System

This module provides the main StrategyManager class that coordinates
strategy loading, signal generation, and performance tracking.
"""

from typing import Dict, Any, List, Optional, Union, Type
import pandas as pd
from loguru import logger

from src.strategies.base_strategy import BaseStrategy
from src.models.models import Signal, SignalType, SignalStrength
from src.indicators.indicator_manager import IndicatorManager

from src.strategies.manager.loader import load_strategy_class, get_available_strategies
from src.strategies.manager.signal_aggregator import (
    aggregate_signals,
    create_aggregated_signal,
)
from src.strategies.manager.performance_tracker import (
    update_strategy_performance,
    update_strategy_weight,
    save_performance_data,
    load_performance_data,
)
from src.strategies.manager.optimization import adjust_strategy_weight


class StrategyManager:
    """Manages multiple trading strategies and aggregates their signals."""

    def __init__(self, config: Dict[str, Any], indicator_manager: IndicatorManager):
        """
        Initialize the strategy manager.

        Args:
            config: System configuration dictionary
            indicator_manager: Indicator manager instance
        """
        self.config = config
        self.indicator_manager = indicator_manager

        # Strategy configuration - handle both dict and Pydantic models
        # Check if config is a dict or a Pydantic model
        if (
            hasattr(config, "dict")
            and callable(getattr(config, "dict"))
            or hasattr(config, "model_dump")
            and callable(getattr(config, "model_dump"))
        ):
            # It's a Pydantic model - convert to dict for compatibility
            config_dict = self._get_config_dict(config)
            self.strategy_configs = config_dict.get("strategies", {})
            self.strategy_weights = config_dict.get("strategy_weights", {})
            self.default_weight = config_dict.get("default_strategy_weight", 1.0)

            # Signal aggregation settings
            self.signal_threshold = config_dict.get("signal_threshold", 0.6)
            self.weighted_aggregation = config_dict.get(
                "use_weighted_aggregation", True
            )
            self.min_concurrent_strategies = config_dict.get(
                "min_concurrent_strategies", 1
            )
        else:
            # It's a dictionary
            self.strategy_configs = config.get("strategies", {})
            self.strategy_weights = config.get("strategy_weights", {})
            self.default_weight = config.get("default_strategy_weight", 1.0)

            # Signal aggregation settings
            self.signal_threshold = config.get("signal_threshold", 0.6)
            self.weighted_aggregation = config.get("use_weighted_aggregation", True)
            self.min_concurrent_strategies = config.get("min_concurrent_strategies", 1)

        # Dictionary to hold strategy instances
        self.strategies: Dict[str, BaseStrategy] = {}

        # Track strategy performance for dynamic weighting
        self.strategy_performance: Dict[str, Dict[str, Any]] = {}

        # Initialize strategies
        self._init_strategies()

        logger.info(
            f"Strategy manager initialized with {len(self.strategies)} strategies"
        )

    def _get_config_dict(self, config) -> Dict[str, Any]:
        """
        Convert Pydantic model config to a dictionary for compatibility.

        Args:
            config: Configuration object (Pydantic model or dict)

        Returns:
            Dictionary representation of the config
        """
        if hasattr(config, "model_dump") and callable(getattr(config, "model_dump")):
            # Pydantic v2
            return config.model_dump()
        elif hasattr(config, "dict") and callable(getattr(config, "dict")):
            # Pydantic v1
            return config.dict()
        return config  # Already a dict

    def _init_strategies(self) -> None:
        """Initialize all enabled strategies from configuration."""
        # Convert strategies to dictionary if it's a list
        if isinstance(self.strategy_configs, list):
            strategies_dict = {}
            for strategy in self.strategy_configs:
                if isinstance(strategy, dict) and "name" in strategy:
                    strategies_dict[strategy["name"]] = strategy
            self.strategy_configs = strategies_dict

        # Get available strategy names
        available_strategies = set(get_available_strategies())

        # Process each strategy
        for strategy_name, strategy_config in self.strategy_configs.items():
            # Skip disabled strategies
            if isinstance(strategy_config, dict) and not strategy_config.get(
                "is_active", True
            ):
                logger.info(f"Skipping disabled strategy: {strategy_name}")
                continue

            # Check if strategy exists
            if strategy_name not in available_strategies:
                logger.warning(f"Unknown strategy: {strategy_name}")
                continue

            try:
                # Load the strategy class
                strategy_class = load_strategy_class(strategy_name)
                if not strategy_class:
                    logger.warning(f"Failed to load strategy class: {strategy_name}")
                    continue

                # Initialize strategy
                strategy = strategy_class(
                    config=strategy_config, indicator_manager=self.indicator_manager
                )

                # Register strategy
                self.strategies[strategy_name] = strategy

                # Initialize strategy weight if not already set
                if strategy_name not in self.strategy_weights:
                    self.strategy_weights[strategy_name] = self.default_weight

                # Initialize performance tracking if not already done
                if strategy_name not in self.strategy_performance:
                    self.strategy_performance[strategy_name] = {
                        "signals_generated": 0,
                        "signals_executed": 0,
                        "successful_signals": 0,
                        "failed_signals": 0,
                        "total_profit_loss": 0.0,
                        "win_rate": 0.0,
                        "weight": self.strategy_weights.get(
                            strategy_name, self.default_weight
                        ),
                    }

                logger.info(f"Initialized strategy: {strategy_name}")
            except Exception as e:
                logger.error(f"Error initializing strategy {strategy_name}: {str(e)}")

        logger.info(f"Initialized {len(self.strategies)} strategies")

        # Check for no active strategies
        if not self.strategies:
            logger.warning("No active strategies initialized!")

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        Generate trading signals from all enabled strategies.

        Args:
            data: Market data DataFrame

        Returns:
            List of aggregated and filtered signals
        """
        all_signals: List[Signal] = []

        # Generate signals from each strategy
        for strategy_name, strategy in self.strategies.items():
            try:
                signals = strategy.generate_signals(data)

                # Track signals generated
                self.strategy_performance[strategy_name]["signals_generated"] += len(
                    signals
                )

                # Add signals to the list
                all_signals.extend(signals)

            except Exception as e:
                logger.error(
                    f"Error generating signals for strategy {strategy_name}: {e}"
                )

        # Aggregate and filter signals
        if all_signals:
            return aggregate_signals(
                all_signals,
                data,
                self.min_concurrent_strategies,
                self.weighted_aggregation,
                self.signal_threshold,
                self.strategy_performance,
                self.default_weight,
            )

        return []

    def update_performance(self, trade_result: Dict[str, Any]) -> None:
        """
        Update strategy performance based on completed trade.

        Args:
            trade_result: Trade result dictionary
        """
        strategy_name = trade_result.get("strategy_name", "unknown")

        # Skip if not a valid strategy
        if strategy_name not in self.strategy_performance:
            return

        # Update performance metrics
        update_strategy_performance(
            self.strategy_performance, strategy_name, trade_result
        )

        # Update strategy weight based on performance
        if self.config.get("dynamic_weighting", False):
            update_strategy_weight(
                self.strategy_performance,
                strategy_name,
                self.config.get("min_signals_for_weighting", 10),
                self.config.get("max_weight_change", 0.2),
            )

        logger.info(
            f"Updated performance for strategy {strategy_name}: "
            f"win rate {self.strategy_performance[strategy_name]['win_rate']:.2f}, "
            f"weight {self.strategy_performance[strategy_name]['weight']:.2f}"
        )

    def save_performance(self, file_path: str) -> None:
        """
        Save strategy performance metrics to file.

        Args:
            file_path: Path to save performance data
        """
        save_performance_data(self.strategy_performance, file_path)

    def load_performance(self, file_path: str) -> bool:
        """
        Load strategy performance metrics from file.

        Args:
            file_path: Path to load performance data from

        Returns:
            True if loaded successfully, False otherwise
        """
        return load_performance_data(self.strategy_performance, file_path)

    def get_enabled_strategies(self) -> List[str]:
        """
        Get list of enabled strategy names.

        Returns:
            List of strategy names
        """
        return list(self.strategies.keys())

    def get_strategy_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all strategies.

        Returns:
            Dictionary with strategy information
        """
        strategy_info = {}

        for name, strategy in self.strategies.items():
            strategy_info[name] = {
                "description": getattr(strategy, "description", ""),
                "performance": self.strategy_performance.get(name, {}),
                "enabled": True,
                "weight": self.strategy_performance.get(name, {}).get(
                    "weight", self.default_weight
                ),
            }

        # Add disabled strategies
        for name, config in self.strategy_configs.items():
            if not config.get("enabled", True) and name not in strategy_info:
                strategy_info[name] = {
                    "description": "",
                    "performance": {},
                    "enabled": False,
                    "weight": self.strategy_weights.get(name, self.default_weight),
                }

        return strategy_info

    def enable_strategy(self, strategy_name: str) -> bool:
        """
        Enable a disabled strategy.

        Args:
            strategy_name: Name of the strategy to enable

        Returns:
            True if strategy was enabled, False otherwise
        """
        if strategy_name in self.strategies:
            # Already enabled
            return True

        # Check if it's in the config
        if strategy_name in self.strategy_configs:
            self.strategy_configs[strategy_name]["enabled"] = True

            # Re-initialize the strategy
            self._init_strategies()

            return strategy_name in self.strategies

        return False

    def disable_strategy(self, strategy_name: str) -> bool:
        """
        Disable an enabled strategy.

        Args:
            strategy_name: Name of the strategy to disable

        Returns:
            True if strategy was disabled, False otherwise
        """
        if strategy_name in self.strategies:
            # Update config
            if strategy_name in self.strategy_configs:
                self.strategy_configs[strategy_name]["enabled"] = False

            # Remove from active strategies
            del self.strategies[strategy_name]

            return True

        return False

    def adjust_strategy_weight(self, strategy_name: str, weight: float) -> bool:
        """
        Manually adjust strategy weight.

        Args:
            strategy_name: Name of the strategy
            weight: New weight value (0.5 to 2.0)

        Returns:
            True if weight was adjusted, False otherwise
        """
        return adjust_strategy_weight(
            self.strategy_performance, self.strategy_weights, strategy_name, weight
        )
