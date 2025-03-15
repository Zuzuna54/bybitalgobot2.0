"""
Configuration Manager for the Bybit Algorithmic Trading System

This module handles loading, validating, and providing access to
system configuration settings from JSON files and environment variables.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

import pydantic
from loguru import logger
from pydantic import BaseModel, validator, Field


class ExchangeConfig(BaseModel):
    """Exchange configuration settings."""
    name: str = "bybit"
    testnet: bool = True
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    
    @validator('api_key', 'api_secret', pre=True, always=True)
    def validate_credentials(cls, v, values, **kwargs):
        """Check if credentials should be loaded from environment variables."""
        field_name = kwargs['field'].name
        
        # Allow explicit None for testing
        if v is None and values.get('name') == 'bybit':
            env_var = f"BYBIT_{field_name.upper()}"
            env_value = os.environ.get(env_var)
            
            if env_value:
                return env_value
            
            # Only warn in non-testnet mode
            if not values.get('testnet', True):
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
        self.config_data = self._load_config()
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
            with open(self.config_path, 'r') as config_file:
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
            with open(save_path, 'w') as config_file:
                json.dump(config.dict(exclude_none=True), config_file, indent=2)
            logger.info(f"Configuration saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise 