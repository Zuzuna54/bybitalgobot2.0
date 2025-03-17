# Core System Implementation Details

This document provides detailed implementation instructions for the core trading system components, focusing on system initialization, configuration management, and basic data flow.

## Table of Contents

1. [System Initialization](#1-system-initialization)
2. [API Client Integration](#2-api-client-integration)
3. [Core Data Flow Establishment](#3-core-data-flow-establishment)
4. [Risk Management Integration](#4-risk-management-integration)
5. [Performance Tracking Setup](#5-performance-tracking-setup)

## 1. System Initialization

### 1.1 Enhance Main Module Entry Point

#### Implementation Details

1. **Refine Command-Line Argument Parsing**:

```python
# In src/main.py, update the argument parsing section
def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Bybit Algorithmic Trading System")

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--backtest", action="store_true", help="Run in backtest mode")
    mode_group.add_argument("--paper", action="store_true", help="Run in paper trading mode")
    mode_group.add_argument("--live", action="store_true", help="Run in live trading mode")

    # Configuration
    parser.add_argument("--config", type=str, default="config/default_config.json",
                        help="Path to configuration file")
    parser.add_argument("--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        default="INFO", help="Logging level")

    # Dashboard options
    parser.add_argument("--with-dashboard", action="store_true",
                        help="Start the dashboard along with the trading system")
    parser.add_argument("--dashboard-port", type=int, default=8050,
                        help="Port for the dashboard (default: 8050)")

    # Testing options
    parser.add_argument("--dry-run", action="store_true",
                        help="Initialize system but don't execute trades")

    return parser.parse_args()
```

2. **Implement Proper Mode Selection**:

```python
# In src/main.py, enhance the system initialization based on mode
def __init__(self, config_path: str, log_level: str = "INFO", mode: str = "paper"):
    """
    Initialize the trading system.

    Args:
        config_path: Path to configuration file
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        mode: Trading mode (backtest, paper, live)
    """
    # Set up logging
    self._setup_logging(log_level)

    logger.info(f"Initializing trading system with config: {config_path}, mode: {mode}")

    # Set operating mode
    self.mode = mode
    self.is_backtest = mode == "backtest"
    self.is_paper = mode == "paper"
    self.is_live = mode == "live"

    # Load configuration
    self.config_manager = ConfigManager(config_path)
    self.config = self.config_manager.get_config()

    # Validate configuration for the selected mode
    self._validate_config_for_mode()

    # Initialize components
    self._initialize_components()

    # For graceful shutdown
    self.is_running = False
    self.shutdown_requested = False

    logger.info("Trading system initialized")
```

3. **Add Initialization Validation**:

```python
# In src/main.py, add validation checks
def _validate_config_for_mode(self):
    """Validate configuration for the selected operating mode."""
    if self.is_live:
        # Check for API credentials in live mode
        if not self._check_api_credentials():
            raise ValueError("API credentials required for live trading")

        # Check for risk parameters in live mode
        if not self._check_risk_parameters():
            raise ValueError("Risk parameters must be configured for live trading")

    if self.is_backtest:
        # Check for backtest configuration
        if "backtest" not in self.config:
            raise ValueError("Backtest configuration required for backtest mode")

        # Check for required backtest parameters
        backtest_config = self.config.get("backtest", {})
        required_params = ["start_date", "end_date", "initial_balance"]
        for param in required_params:
            if param not in backtest_config:
                raise ValueError(f"Missing required backtest parameter: {param}")
```

### 1.2 Configuration Management Enhancements

#### Implementation Details

1. **Implement Consistent Configuration Validation**:

```python
# In src/config/config_manager.py, enhance configuration validation
def validate_configuration(self):
    """Validate the configuration for consistency and completeness."""
    # Check for required sections
    required_sections = ["exchange", "pairs", "strategies", "risk"]
    for section in required_sections:
        if section not in self.config:
            logger.warning(f"Missing required configuration section: {section}")
            self.config[section] = {}

    # Validate exchange configuration
    exchange_config = self.config.get("exchange", {})
    if "name" not in exchange_config:
        logger.warning("Exchange name not specified, defaulting to 'bybit'")
        exchange_config["name"] = "bybit"

    # Validate trading pairs
    pairs = self.config.get("pairs", [])
    if not pairs:
        logger.warning("No trading pairs specified")

    # Validate strategies
    strategies = self.config.get("strategies", [])
    if not strategies:
        logger.warning("No trading strategies specified")

    # Validate risk parameters
    risk_config = self.config.get("risk", {})
    required_risk_params = [
        "max_position_size_percent",
        "max_daily_drawdown_percent",
        "default_leverage"
    ]
    for param in required_risk_params:
        if param not in risk_config:
            logger.warning(f"Missing risk parameter: {param}")
```

2. **Create Structured Configuration Objects**:

```python
# In src/config/config_manager.py, add structured configuration classes
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union

@dataclass
class ExchangeConfig:
    """Exchange configuration."""
    name: str = "bybit"
    testnet: bool = True
    api_key: str = ""
    api_secret: str = ""

@dataclass
class PairConfig:
    """Trading pair configuration."""
    symbol: str
    quote_currency: str
    base_currency: str
    is_active: bool = True
    min_order_qty: float = 0.0
    max_leverage: int = 1
    tick_size: float = 0.0

@dataclass
class StrategyConfig:
    """Strategy configuration."""
    name: str
    is_active: bool = True
    timeframe: str = "1h"
    parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RiskConfig:
    """Risk management configuration."""
    max_position_size_percent: float = 1.0
    max_daily_drawdown_percent: float = 1.0
    default_leverage: int = 1
    max_open_positions: int = 5
    use_trailing_stop: bool = False
    stop_loss_atr_multiplier: float = 2.0
    take_profit_risk_reward_ratio: float = 2.0
    circuit_breaker_consecutive_losses: int = 3

@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    start_date: str
    end_date: str
    initial_balance: float = 10000.0
    fee_rate: float = 0.075
    slippage: float = 0.05

@dataclass
class SystemConfig:
    """Complete system configuration."""
    exchange: ExchangeConfig
    pairs: List[PairConfig]
    strategies: List[StrategyConfig]
    risk: RiskConfig
    backtest: Optional[BacktestConfig] = None
    max_workers: int = 5
    data_dir: str = "data"
    update_interval_seconds: int = 60
    system_id: str = "bybit_trading_system"
```

3. **Ensure Proper Config Inheritance**:

```python
# In src/config/config_manager.py, implement configuration inheritance
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
        with open(config_path, 'r') as f:
            user_config = json.load(f)

    # Merge configurations with user config taking precedence
    merged_config = self._deep_merge(default_config, user_config)

    # Apply environment variable overrides
    env_config = self._load_env_config()
    merged_config = self._deep_merge(merged_config, env_config)

    return merged_config

def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
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
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
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
        env_config["exchange"]["testnet"] = os.environ.get("BYBIT_USE_TESTNET").lower() == "true"

    return env_config
```

### 1.3 Component Lifecycle Management

#### Implementation Details

1. **Implement Proper Initialization Order**:

```python
# In src/main.py, implement proper component initialization order
def _initialize_components(self) -> None:
    """Initialize all trading system components in proper order."""
    # Step 1: Load environment variables and configuration
    load_dotenv()
    logger.info("Loading environment variables from .env file")

    # Step 2: Initialize API client (needed by most other components)
    self._initialize_api_client()

    # Step 3: Initialize market data service
    self._initialize_market_data()

    # Step 4: Initialize technical indicators
    self._initialize_indicators()

    # Step 5: Initialize strategy manager (depends on indicators)
    self._initialize_strategy_manager()

    # Step 6: Initialize risk manager
    self._initialize_risk_manager()

    # Step 7: Initialize trade manager (depends on API, risk manager)
    self._initialize_trade_manager()

    # Step 8: Initialize performance tracker
    self._initialize_performance_tracker()

    # Step 9: Initialize backtesting engine if needed
    if self.is_backtest:
        self._initialize_backtest_engine()

    # Step 10: Initialize paper trading if needed
    if self.is_paper:
        self._initialize_paper_trading()

    # Step 11: Initialize websockets for real-time data
    if not self.is_backtest:
        self._initialize_websockets()

    # Step 12: Initialize dashboard if requested
    if self.with_dashboard:
        self._initialize_dashboard()

    logger.info("All components initialized successfully")
```

2. **Add Dependency Validation**:

```python
# In src/main.py, add dependency validation
def _validate_dependencies(self):
    """Validate component dependencies."""
    # Check that required components are initialized
    required_components = ["api_client", "indicator_manager", "strategy_manager",
                          "risk_manager", "trade_manager", "performance_tracker"]

    for component_name in required_components:
        if not hasattr(self, component_name) or getattr(self, component_name) is None:
            logger.error(f"Required component not initialized: {component_name}")
            return False

    # Check mode-specific components
    if self.is_backtest and (not hasattr(self, "backtest_engine") or
                             self.backtest_engine is None):
        logger.error("Backtest engine not initialized for backtest mode")
        return False

    if self.is_paper and (not hasattr(self, "paper_trading_engine") or
                          self.paper_trading_engine is None):
        logger.error("Paper trading engine not initialized for paper trading mode")
        return False

    return True
```

3. **Create Graceful Shutdown Procedure**:

```python
# In src/main.py, implement graceful shutdown
def _handle_shutdown(self, signum, frame):
    """Handle shutdown signal."""
    logger.info(f"Received shutdown signal: {signum}")
    self.shutdown_requested = True

def _shutdown(self):
    """Perform graceful system shutdown."""
    logger.info("Initiating graceful shutdown sequence")

    # Set flags to stop processing loops
    self.is_running = False

    # Save performance data
    if hasattr(self, "performance_tracker") and self.performance_tracker:
        logger.info("Saving performance data")
        # Add code to save performance data

    # Close WebSocket connections
    if hasattr(self, "websocket_manager") and self.websocket_manager:
        logger.info("Closing WebSocket connections")
        self.websocket_manager.close_all()

    # Close open positions in paper trading
    if self.is_paper and hasattr(self, "paper_trading_engine") and self.paper_trading_engine:
        logger.info("Closing paper trading positions")
        self.paper_trading_engine.close_all_positions()

    # Shut down dashboard if running
    if hasattr(self, "dashboard_thread") and self.dashboard_thread:
        logger.info("Shutting down dashboard")
        # Add code to properly terminate dashboard

    logger.info("Shutdown complete")
```

## 2. API Client Integration

### 2.1 API Authentication Enhancement

#### Implementation Details

1. **Improve API Key and Secret Management**:

```python
# In src/api/bybit/core/connection.py, enhance authentication
def set_auth_credentials(self, api_key: str, api_secret: str) -> None:
    """
    Set or update authentication credentials.

    Args:
        api_key: Bybit API key
        api_secret: Bybit API secret
    """
    if not api_key or not api_secret:
        logger.warning("Empty API credentials provided")
        return

    # Validate API key format (basic check)
    if not isinstance(api_key, str) or len(api_key) < 5:
        logger.error("Invalid API key format")
        return

    # Store credentials
    self.api_key = api_key
    self.api_secret = api_secret
    self.is_authenticated = True

    logger.info("API credentials updated")
```

2. **Add Proper Error Handling for Authentication Failures**:

```python
# In src/api/bybit/core/api_client.py, enhance error handling
def handle_response(self, response) -> Dict[str, Any]:
    """
    Handle API response and process errors.

    Args:
        response: Response from API request

    Returns:
        Processed response data

    Raises:
        APIError: If the response contains an error
    """
    try:
        data = response.json()

        # Check for API errors
        if 'ret_code' in data and data['ret_code'] != 0:
            error_code = data.get('ret_code')
            error_msg = data.get('ret_msg', 'Unknown error')

            # Handle authentication errors specifically
            if error_code in [10003, 10004, 10005]:
                logger.error(f"Authentication error: {error_msg}")
                # Invalidate credentials to prevent further failed attempts
                self.connection_manager.invalidate_credentials()
                raise AuthenticationError(f"Authentication failed: {error_msg}")

            # Handle rate limiting errors
            elif error_code in [10006, 10007]:
                logger.warning(f"Rate limit error: {error_msg}")
                raise RateLimitError(f"Rate limit exceeded: {error_msg}")

            # Handle other API errors
            else:
                logger.error(f"API error: {error_code} - {error_msg}")
                raise APIError(f"API error {error_code}: {error_msg}")

        return data

    except ValueError:
        logger.error(f"Failed to parse JSON response: {response.text}")
        raise APIError("Invalid JSON response")
```

3. **Implement Reconnection Logic**:

```python
# In src/api/bybit/core/connection.py, add reconnection logic
def make_request(self, method: str, endpoint: str, params: Optional[Dict] = None,
                retry_count: int = 3, retry_delay: float = 1.0) -> Dict[str, Any]:
    """
    Make an API request with retry logic.

    Args:
        method: HTTP method (GET, POST, etc.)
        endpoint: API endpoint
        params: Request parameters
        retry_count: Number of retries on failure
        retry_delay: Delay between retries in seconds

    Returns:
        Response data

    Raises:
        APIError: If the request fails after all retries
    """
    url = self.get_url(endpoint)
    headers = self.get_headers()

    # Add authentication if available
    if self.is_authenticated:
        params = self.sign_request(params or {})

    for attempt in range(retry_count + 1):
        try:
            # Make the request
            if method.upper() == 'GET':
                response = requests.get(url, params=params, headers=headers, timeout=10)
            elif method.upper() == 'POST':
                response = requests.post(url, json=params, headers=headers, timeout=10)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            # Check for HTTP errors
            response.raise_for_status()

            # Process the response
            return self.handle_response(response)

        except (requests.exceptions.RequestException, ValueError) as e:
            # Log the error
            logger.warning(f"Request failed (attempt {attempt+1}/{retry_count+1}): {str(e)}")

            # Retry if not the last attempt
            if attempt < retry_count:
                # Exponential backoff
                sleep_time = retry_delay * (2 ** attempt)
                logger.info(f"Retrying in {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
            else:
                # All retries failed
                logger.error(f"Request failed after {retry_count+1} attempts")
                raise APIError(f"Request failed: {str(e)}")
```

## Additional Sections

For brevity, only key implementation details have been shown above. The full implementation plan includes detailed instructions for all sections including:

- Market Data Service Improvements
- WebSocket Integration
- Core Data Flow Establishment
- Risk Management Integration
- Performance Tracking Setup

Each section includes code snippets, implementation guidance, and validation checks to ensure proper functionality.
