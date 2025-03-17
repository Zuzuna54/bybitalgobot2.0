# Configuration System

## Overview

The Configuration System is a robust framework for managing all settings and parameters used throughout the trading application. It provides a structured and validated approach to configuration management, enabling consistent access to settings while ensuring their correctness.

The system is designed with flexibility and ease of use in mind, supporting various configuration sources, environment-based overrides, nested configuration access, and strong typing through Pydantic validation.

## Features

- **JSON Configuration Files**: Load settings from structured JSON configuration files
- **Configuration Validation**: Validate configuration using Pydantic models
- **Nested Configuration Access**: Access deeply nested configuration using dot notation
- **Type Conversion**: Automatic type conversion for configuration values
- **Default Values**: Support for default values when configurations are missing
- **Environment Variable Integration**: Override settings with environment variables
- **Config Path Discovery**: Automatic discovery of configuration files in standard locations
- **Singleton Pattern**: Global access to configuration through a singleton instance
- **Serialization**: Save modified configurations back to files
- **Type Safety**: Strong typing for configuration values with validation

## Architecture

The Configuration System follows a clean, modular architecture:

### Core Components

1. **ConfigManager**: The main class for configuration loading, validation, and access
2. **Pydantic Models**: Data models that provide validation and typing for configuration
3. **Configuration Discovery**: Logic to find configuration files in standard locations
4. **Default Configuration**: Fallback settings when no custom configuration is provided

### Main Modules

1. **config_manager.py**: Contains the ConfigManager class and Pydantic models
2. \***\*init**.py\*\*: Exports the public interface of the configuration system
3. **default_config.json**: Default configuration values

## Detailed Functionality

### ConfigManager

The `ConfigManager` class is the main entry point for configuration management:

- **Initialization**: Load configuration from file and validate it
- **Configuration Access**: Provide type-safe access to configuration values
- **Nested Configuration**: Access nested configuration using dot notation
- **Type Conversion**: Convert configuration values to specific types
- **Default Values**: Return default values when configuration is missing

```python
# Initialize the config manager
config_manager = ConfigManager(Path("config.json"))

# Get configuration values
api_key = config_manager.get("exchange.api_key")
max_workers = config_manager.get_int("max_workers", 5)
is_active = config_manager.get_bool("pairs.0.is_active", True)
```

### Configuration Models

The system uses Pydantic models for validation and typing:

- **SystemConfig**: Top-level configuration model
- **ExchangeConfig**: Exchange connection settings
- **PairConfig**: Trading pair configuration
- **StrategyConfig**: Strategy settings and parameters
- **RiskConfig**: Risk management parameters
- **BacktestConfig**: Backtesting settings

```python
# Access validated configuration
config = config_manager.get_config()
exchange_name = config.exchange.name
strategies = config.strategies
```

### Singleton Access

For convenient global access, a singleton pattern is implemented:

```python
# Get the singleton instance
from src.config import get_config_manager

config_manager = get_config_manager()
api_key = config_manager.get("exchange.api_key")
```

## Configuration Structure

The configuration is structured hierarchically:

### Exchange Configuration

```json
"exchange": {
  "name": "bybit",
  "testnet": true,
  "api_key": "YOUR_API_KEY",
  "api_secret": "YOUR_API_SECRET"
}
```

### Trading Pairs

```json
"pairs": [
  {
    "symbol": "BTCUSDT",
    "quote_currency": "USDT",
    "base_currency": "BTC",
    "is_active": true,
    "min_order_qty": 0.001,
    "max_leverage": 10,
    "tick_size": 0.5
  }
]
```

### Strategies

```json
"strategies": [
  {
    "name": "ema_crossover",
    "is_active": true,
    "timeframe": "1h",
    "parameters": {
      "fast_ema": 9,
      "slow_ema": 21,
      "volume_threshold": 1.5
    }
  }
]
```

### Risk Management

```json
"risk": {
  "max_position_size_percent": 5.0,
  "max_daily_drawdown_percent": 3.0,
  "default_leverage": 2,
  "max_open_positions": 5,
  "use_trailing_stop": true,
  "stop_loss_atr_multiplier": 2.0,
  "take_profit_risk_reward_ratio": 2.0,
  "circuit_breaker_consecutive_losses": 3
}
```

### Backtesting

```json
"backtest": {
  "start_date": "2023-01-01T00:00:00Z",
  "end_date": "2023-12-31T23:59:59Z",
  "initial_balance": 10000.0,
  "fee_rate": 0.075,
  "slippage": 0.05
}
```

## Developer Guide: Extending the Configuration System

### Adding New Configuration Models

To add new configuration sections:

1. Define a new Pydantic model class in `config_manager.py`
2. Add the new model as a field in the `SystemConfig` class
3. Update the default configuration in `default_config.json`

Example:

```python
class NotificationConfig(BaseModel):
    """Notification settings."""

    enable_email: bool = False
    enable_telegram: bool = False
    email_recipients: List[str] = Field(default_factory=list)
    telegram_chat_id: Optional[str] = None

# Update SystemConfig to include the new model
class SystemConfig(BaseModel):
    # Existing fields...
    notifications: Optional[NotificationConfig] = None
```

### Adding Environment Variable Support

To override configuration with environment variables:

1. Modify the `_load_config` method in `ConfigManager`
2. Add environment variable processing logic

Example:

```python
def _load_config(self) -> Dict[str, Any]:
    # Load from file first
    config_data = self._load_from_file()

    # Override with environment variables
    if os.environ.get("EXCHANGE_API_KEY"):
        config_data.setdefault("exchange", {})["api_key"] = os.environ["EXCHANGE_API_KEY"]

    if os.environ.get("EXCHANGE_API_SECRET"):
        config_data.setdefault("exchange", {})["api_secret"] = os.environ["EXCHANGE_API_SECRET"]

    return config_data
```

### Creating Custom Type Converters

To add support for custom type conversions:

1. Add a new getter method to the `ConfigManager` class
2. Implement the conversion logic

Example:

```python
def get_datetime(self, key: str, default: Optional[datetime] = None) -> Optional[datetime]:
    """
    Get a datetime configuration value.

    Args:
        key: Dot-separated path to the configuration value
        default: Default value to return if the key doesn't exist

    Returns:
        The datetime value or the default if not found
    """
    value = self.get(key)
    if value is None:
        return default

    if isinstance(value, datetime):
        return value

    try:
        return datetime.fromisoformat(value.replace('Z', '+00:00'))
    except (ValueError, AttributeError):
        return default
```

### Adding Configuration Migrations

To support migrating configurations between versions:

1. Create a migration function in `config_manager.py`
2. Add version tracking to the configuration

Example:

```python
def migrate_config(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Migrate configuration from older versions.

    Args:
        config_data: The configuration data to migrate

    Returns:
        The migrated configuration data
    """
    version = config_data.get("version", "1.0")

    # Migrate from version 1.0 to 2.0
    if version == "1.0":
        # Update structure
        if "risk_management" in config_data:
            config_data["risk"] = config_data.pop("risk_management")

        # Update version
        config_data["version"] = "2.0"

    return config_data
```

## Best Practices

### Configuration Management

- Store sensitive data like API keys in environment variables, not in committed configuration files
- Use separate configuration files for different environments (development, testing, production)
- Implement proper error handling for missing or invalid configuration
- Document all configuration options and their default values
- Validate configuration values to catch errors early

### Configuration Access

- Always provide sensible default values when accessing configuration
- Use the typed getter methods (`get_int`, `get_bool`, etc.) for type safety
- Access the validated Pydantic models directly for complex configuration objects
- Use the singleton pattern for global access to configuration

### Configuration Extensibility

- Design configuration structures to be backward compatible
- Implement migration logic for configuration changes
- Use optional fields for new configuration options
- Document the configuration structure and validation rules

## Implementation Details

### Configuration Loading Process

The configuration loading process follows these steps:

1. Look for configuration files in standard locations
2. Load the first configuration file found, or use the default configuration
3. Parse the JSON configuration data
4. Validate the configuration using Pydantic models
5. Create the `ConfigManager` instance with the validated configuration

### Validation Rules

The Pydantic models apply the following validation rules:

- **API Key & Secret**: Can be None, but will log warnings if not provided
- **Exchange Name**: Must be a non-empty string
- **Trading Pairs**: Must have valid symbol, quote currency, and base currency
- **Strategies**: Must have a name and valid parameters
- **Risk Parameters**: Must be within reasonable ranges
- **Timeframes**: Must be valid timeframe strings

### Type Conversion

The ConfigManager provides these type conversion methods:

- **get_int**: Convert to integer
- **get_float**: Convert to float
- **get_bool**: Convert to boolean (handles "true"/"false" strings)
- **get_list**: Convert to list
- **get_dict**: Convert to dictionary
- **get_path**: Convert to path string

## Dependencies

- **pydantic**: For data validation and settings management
- **loguru**: For enhanced logging
- **pathlib**: For path manipulation
- **json**: For JSON parsing and serialization

## Conclusion

The Configuration System provides a robust and flexible foundation for managing application settings. With its validation, type safety, and convenient access patterns, it ensures configuration values are correct and easily accessible throughout the application.

The modular design and extension points allow for easy customization and evolution of the configuration structure, while maintaining backward compatibility and supporting multiple environments.
