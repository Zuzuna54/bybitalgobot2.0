"""
Configuration Manager for the Bybit Algorithmic Trading System

This module handles loading, validating, and providing access to
system configuration settings from JSON files and environment variables.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field

import pydantic
from loguru import logger
from pydantic import BaseModel, validator, Field

# Singleton instance
_config_manager_instance = None


def get_config_manager():
    """
    Get or create a singleton instance of the ConfigManager.

    Returns:
        ConfigManager instance
    """
    global _config_manager_instance

    if _config_manager_instance is None:
        # Look for config file in standard locations
        config_locations = [
            Path("config/config.json"),
            Path("config/default_config.json"),
            Path("src/config/config.json"),
            Path("src/config/default_config.json"),
            Path(os.environ.get("CONFIG_PATH", "")),
        ]

        # Use the first config file found
        config_path = None
        for path in config_locations:
            if path.exists():
                config_path = path
                break

        if config_path is None:
            logger.warning(
                "No configuration file found in standard locations. Using default_config.json"
            )
            # Default to the file in the same directory as this module
            config_path = Path(__file__).parent / "default_config.json"
            if not config_path.exists():
                logger.error(f"Default config file not found at {config_path}")
                raise FileNotFoundError(f"No configuration file found at {config_path}")

        logger.info(f"Loading configuration from {config_path}")
        _config_manager_instance = ConfigManager(config_path)

    return _config_manager_instance


class ExchangeConfig(BaseModel):
    """Exchange configuration settings."""

    name: str = "bybit"
    testnet: bool = True
    api_key: Optional[str] = None
    api_secret: Optional[str] = None

    @validator("api_key")
    def validate_api_key(cls, v, values):
        """Check if API key should be loaded from environment variables."""
        # Allow explicit None for testing
        if v is None and values.get("name") == "bybit":
            env_var = "BYBIT_API_KEY"
            env_value = os.environ.get(env_var)

            if env_value:
                return env_value

            # Only warn in non-testnet mode
            if not values.get("testnet", True):
                logger.warning(f"{env_var} environment variable not set")
        return v

    @validator("api_secret")
    def validate_api_secret(cls, v, values):
        """Check if API secret should be loaded from environment variables."""
        # Allow explicit None for testing
        if v is None and values.get("name") == "bybit":
            env_var = "BYBIT_API_SECRET"
            env_value = os.environ.get(env_var)

            if env_value:
                return env_value

            # Only warn in non-testnet mode
            if not values.get("testnet", True):
                logger.warning(f"{env_var} environment variable not set")
        return v


class PairConfig(BaseModel):
    """Trading pair configuration."""

    symbol: str
    quote_currency: str
    base_currency: str
    is_active: bool = True
    min_order_qty: float = 0.0
    max_leverage: int = 10
    tick_size: float = 0.0


class StrategyConfig(BaseModel):
    """Strategy configuration settings."""

    name: str
    is_active: bool = True
    timeframe: str = "1h"
    parameters: Dict[str, Any] = Field(default_factory=dict)


class RiskConfig(BaseModel):
    """Risk management configuration."""

    max_position_size_percent: float = 5.0
    max_daily_drawdown_percent: float = 3.0
    default_leverage: int = 2
    max_open_positions: int = 5
    use_trailing_stop: bool = True
    stop_loss_atr_multiplier: float = 2.0
    take_profit_risk_reward_ratio: float = 2.0
    circuit_breaker_consecutive_losses: int = 3


class BacktestConfig(BaseModel):
    """Backtesting configuration."""

    start_date: str
    end_date: str
    initial_balance: float = 10000.0
    fee_rate: float = 0.075  # In percentage (0.075% = 0.00075)
    slippage: float = 0.05  # In percentage


class SystemConfig(BaseModel):
    """Overall system configuration."""

    exchange: ExchangeConfig
    pairs: list[PairConfig]
    strategies: list[StrategyConfig]
    risk: RiskConfig
    backtest: Optional[BacktestConfig] = None
    max_workers: int = 5
    data_dir: str = "data"
    update_interval_seconds: int = 60
    system_id: str = "bybit_trading_system"


class ConfigManager:
    """Manages loading and access to system configuration."""

    def __init__(self, config_path: Path):
        """
        Initialize the configuration manager.

        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.config_data = self.load_configuration(config_path)
        self.validate_configuration()
        self.config = self._validate_config(self.config_data)

    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from JSON file.

        Returns:
            Dict containing the configuration data

        Raises:
            FileNotFoundError: If the configuration file does not exist
            json.JSONDecodeError: If the configuration file is not valid JSON
        """
        try:
            with open(self.config_path, "r") as config_file:
                return json.load(config_file)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file: {e}")
            raise

    def _validate_config(self, config_data: Dict[str, Any]) -> SystemConfig:
        """
        Validate the configuration data using Pydantic models.

        Args:
            config_data: Dictionary containing configuration data

        Returns:
            Validated SystemConfig object

        Raises:
            pydantic.ValidationError: If the configuration data is invalid
        """
        try:
            return SystemConfig(**config_data)
        except pydantic.ValidationError as e:
            logger.error(f"Configuration validation failed: {e}")
            raise

    def get_config(self) -> SystemConfig:
        """
        Get the validated system configuration.

        Returns:
            SystemConfig object
        """
        return self.config

    def save_config(self, config: SystemConfig, path: Optional[Path] = None) -> None:
        """
        Save the configuration to a JSON file.

        Args:
            config: SystemConfig object to save
            path: Optional path to save to, defaults to the original path
        """
        save_path = path or self.config_path

        try:
            with open(save_path, "w") as config_file:
                json.dump(config.dict(exclude_none=True), config_file, indent=2)
            logger.info(f"Configuration saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise

    def validate_configuration(self):
        """Validate the configuration for consistency and completeness."""
        # Check for required sections
        required_sections = ["exchange", "pairs", "strategies", "risk"]
        for section in required_sections:
            if section not in self.config_data:
                logger.warning(f"Missing required configuration section: {section}")
                self.config_data[section] = {}

        # Validate exchange configuration
        exchange_config = self.config_data.get("exchange", {})
        if "name" not in exchange_config:
            logger.warning("Exchange name not specified, defaulting to 'bybit'")
            exchange_config["name"] = "bybit"

        # Validate trading pairs
        pairs = self.config_data.get("pairs", [])
        if not pairs:
            logger.warning("No trading pairs specified")

        # Validate strategies
        strategies = self.config_data.get("strategies", [])
        if not strategies:
            logger.warning("No trading strategies specified")

        # Validate risk parameters
        risk_config = self.config_data.get("risk", {})
        required_risk_params = [
            "max_position_size_percent",
            "max_daily_drawdown_percent",
            "default_leverage",
        ]
        for param in required_risk_params:
            if param not in risk_config:
                logger.warning(f"Missing risk parameter: {param}")

    def load_configuration(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from file with inheritance.

        Args:
            config_path: Path to config file

        Returns:
            Configuration dictionary
        """
        # Load default configuration
        default_config = self._load_default_config()

        # Load user configuration if specified
        user_config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                user_config = json.load(f)

        # Merge configurations with user config taking precedence
        merged_config = self._deep_merge(default_config, user_config)

        # Apply environment variable overrides
        env_config = self._load_env_config()
        merged_config = self._deep_merge(merged_config, env_config)

        return merged_config

    def _load_default_config(self) -> Dict[str, Any]:
        """
        Load default configuration from the default_config.json file.

        Returns:
            Default configuration dictionary
        """
        default_config_path = Path(__file__).parent / "default_config.json"
        if not default_config_path.exists():
            logger.warning(
                "Default configuration file not found. Using empty configuration."
            )
            return {}

        try:
            with open(default_config_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load default configuration: {e}")
            return {}

    def _deep_merge(
        self, base: Dict[str, Any], override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Deep merge two dictionaries with override taking precedence.

        Args:
            base: Base dictionary
            override: Dictionary with overrides

        Returns:
            Merged dictionary
        """
        result = base.copy()

        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                # Recursively merge dictionaries
                result[key] = self._deep_merge(result[key], value)
            else:
                # Override or add value
                result[key] = value

        return result

    def _load_env_config(self) -> Dict[str, Any]:
        """
        Load configuration from environment variables.

        Returns:
            Configuration dictionary from environment variables
        """
        env_config = {}

        # Exchange config from environment
        if os.environ.get("BYBIT_API_KEY"):
            if "exchange" not in env_config:
                env_config["exchange"] = {}
            env_config["exchange"]["api_key"] = os.environ.get("BYBIT_API_KEY")

        if os.environ.get("BYBIT_API_SECRET"):
            if "exchange" not in env_config:
                env_config["exchange"] = {}
            env_config["exchange"]["api_secret"] = os.environ.get("BYBIT_API_SECRET")

        if os.environ.get("BYBIT_USE_TESTNET"):
            if "exchange" not in env_config:
                env_config["exchange"] = {}
            env_config["exchange"]["testnet"] = (
                os.environ.get("BYBIT_USE_TESTNET").lower() == "true"
            )

        return env_config

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key with nested key support.

        Args:
            key: Dot-separated path to the configuration value (e.g. 'exchange.api_key')
            default: Default value to return if the key doesn't exist

        Returns:
            The configuration value or the default if not found
        """
        parts = key.split(".")
        value = self.config_data

        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default

        return value

    def get_int(self, key: str, default: int = 0) -> int:
        """
        Get an integer configuration value.

        Args:
            key: Dot-separated path to the configuration value
            default: Default value to return if the key doesn't exist

        Returns:
            The integer value or the default if not found
        """
        value = self.get(key, default)
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    def get_float(self, key: str, default: float = 0.0) -> float:
        """
        Get a float configuration value.

        Args:
            key: Dot-separated path to the configuration value
            default: Default value to return if the key doesn't exist

        Returns:
            The float value or the default if not found
        """
        value = self.get(key, default)
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def get_bool(self, key: str, default: bool = False) -> bool:
        """
        Get a boolean configuration value.

        Args:
            key: Dot-separated path to the configuration value
            default: Default value to return if the key doesn't exist

        Returns:
            The boolean value or the default if not found
        """
        value = self.get(key, default)
        if isinstance(value, bool):
            return value

        # Handle string values
        if isinstance(value, str):
            return value.lower() in ("true", "yes", "1", "on")

        # Try to convert to boolean
        try:
            return bool(value)
        except (ValueError, TypeError):
            return default

    def get_list(self, key: str, default: Optional[list] = None) -> list:
        """
        Get a list configuration value.

        Args:
            key: Dot-separated path to the configuration value
            default: Default value to return if the key doesn't exist

        Returns:
            The list value or the default if not found
        """
        default = default or []
        value = self.get(key, default)

        if isinstance(value, list):
            return value

        # Try to convert to list
        try:
            return list(value)
        except (ValueError, TypeError):
            return default

    def get_dict(self, key: str, default: Optional[dict] = None) -> dict:
        """
        Get a dictionary configuration value.

        Args:
            key: Dot-separated path to the configuration value
            default: Default value to return if the key doesn't exist

        Returns:
            The dictionary value or the default if not found
        """
        default = default or {}
        value = self.get(key, default)

        if isinstance(value, dict):
            return value

        return default

    def get_path(self, key: str, default: Optional[str] = None) -> str:
        """
        Get a file or directory path from the configuration.

        Args:
            key: Dot-separated path to the configuration value
            default: Default value to return if the key doesn't exist

        Returns:
            The path string or the default if not found
        """
        path_str = self.get(key, default)
        if path_str is None:
            return default

        # Return path as string
        return str(path_str)
