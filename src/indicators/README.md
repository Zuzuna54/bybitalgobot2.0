# Technical Indicators System

## Overview

The Technical Indicators System provides a comprehensive framework for calculating, managing, and applying technical indicators to market data. It features a modular design with a base indicator class, specialized indicator categories, and a centralized indicator manager to coordinate indicator application.

This system allows for easy extension with new indicators and provides a consistent interface for applying indicators to market data within the Algorithmic Trading System.

## Architecture

The system follows a class-based architecture with inheritance from a common base class:

```
BaseIndicator (Abstract Base Class)
├── Momentum Indicators
│   ├── RSI
│   ├── MACD
│   ├── StochasticOscillator
│   └── MomentumIndex
├── Trend Indicators
│   ├── MovingAverage
│   ├── MovingAverageCrossover
│   ├── ADX
│   ├── IchimokuCloud
│   └── Supertrend
├── Volatility Indicators
│   ├── BollingerBands
│   ├── ATR
│   ├── KeltnerChannels
│   ├── HistoricalVolatility
│   └── ChaikinVolatility
└── Volume Indicators
    ├── VWAP
    ├── OBV
    ├── MFI
    ├── ChaikinMoneyFlow
    └── VolumeProfile
```

The `IndicatorManager` provides a central point for registering, configuring, and applying indicators to market data.

## Components

### Base Indicator (`base_indicator.py`)

The foundation of the indicator system is the `BaseIndicator` abstract base class, which:

- Defines the common interface for all indicators
- Provides utility methods for data validation and manipulation
- Includes helper functions for moving averages and normalization

Key methods:

- `calculate(data)`: Abstract method that all indicators must implement
- `validate_data(data)`: Validates input data has required columns
- `get_column_names()`: Returns the column names added by the indicator
- `normalize_value(value, min_val, max_val)`: Utility for normalizing values
- `moving_average(data, window, ma_type)`: Utility for calculating various moving averages

### Indicator Manager (`indicator_manager.py`)

The `IndicatorManager` serves as the central coordinator for all indicators:

- Maintains a registry of available indicator types
- Manages instances of indicators with unique names
- Provides methods for applying indicators to market data
- Offers utility functions for configuring indicators for specific strategies

Key methods:

- `add_indicator(name, indicator_type, params)`: Registers a new indicator instance
- `remove_indicator(name)`: Removes an indicator
- `get_indicator(name)`: Retrieves an indicator by name
- `apply_indicators(data, indicator_names)`: Applies specified indicators to market data
- `configure_default_indicators()`: Sets up a standard set of indicators
- `configure_strategy_indicators(strategy_config)`: Configures indicators for a specific strategy

### Indicator Categories

#### Momentum Indicators (`momentum_indicators.py`)

Indicators that measure the rate of price change:

1. **RSI (Relative Strength Index)**

   - Measures the speed and change of price movements
   - Parameters: `period`
   - Columns: `rsi`

2. **MACD (Moving Average Convergence Divergence)**

   - Shows the relationship between two moving averages
   - Parameters: `fast_period`, `slow_period`, `signal_period`
   - Columns: `macd_line`, `macd_signal`, `macd_histogram`

3. **StochasticOscillator**

   - Compares closing price to price range over a period
   - Parameters: `k_period`, `k_slowing`, `d_period`
   - Columns: `stoch_k`, `stoch_d`

4. **MomentumIndex**
   - Basic price momentum measurement
   - Parameters: `period`, `normalize`
   - Columns: `momentum`

#### Trend Indicators (`trend_indicators.py`)

Indicators that help identify and follow market trends:

1. **MovingAverage**

   - Smooths price data over specified periods
   - Parameters: `period`, `type` (sma, ema, wma), `source`
   - Columns: `[type]_[period]` (e.g., `sma_20`)

2. **MovingAverageCrossover**

   - Identifies crossovers between fast and slow moving averages
   - Parameters: `fast_period`, `slow_period`, `type`, `source`
   - Columns: `[type]_fast_[period]`, `[type]_slow_[period]`, `ma_crossover`

3. **ADX (Average Directional Index)**

   - Measures trend strength
   - Parameters: `period`
   - Columns: `adx`, `+di`, `-di`

4. **IchimokuCloud**

   - Complex indicator showing support/resistance, momentum, and trend
   - Parameters: `conversion_period`, `base_period`, `lagging_span_period`, `displacement`
   - Columns: `ichimoku_conversion`, `ichimoku_base`, `ichimoku_spanA`, `ichimoku_spanB`

5. **Supertrend**
   - Trend-following indicator with stop and reverse strategy
   - Parameters: `period`, `multiplier`
   - Columns: `supertrend`, `supertrend_direction`

#### Volatility Indicators (`volatility_indicators.py`)

Indicators that measure market volatility:

1. **BollingerBands**

   - Shows price volatility with bands around a moving average
   - Parameters: `period`, `std_dev`, `source`
   - Columns: `bb_middle`, `bb_upper`, `bb_lower`, `bb_bandwidth`, `bb_pct_b`

2. **ATR (Average True Range)**

   - Measures market volatility
   - Parameters: `period`, `ema_smoothing`
   - Columns: `atr`, `atr_pct`

3. **KeltnerChannels**

   - Similar to Bollinger Bands using ATR for band width
   - Parameters: `ema_period`, `atr_period`, `atr_multiplier`, `use_sma`
   - Columns: `kc_middle`, `kc_upper`, `kc_lower`, `atr`, `kc_bandwidth`

4. **HistoricalVolatility**

   - Measures price volatility over a specified period
   - Parameters: `period`, `trading_periods`, `use_log_returns`
   - Columns: `hist_volatility`, `annualized_volatility`

5. **ChaikinVolatility**
   - Volatility indicator developed by Marc Chaikin
   - Parameters: `ema_period`, `change_period`, `roc_period`
   - Columns: `chaikin_volatility`

#### Volume Indicators (`volume_indicators.py`)

Indicators that incorporate trading volume:

1. **VWAP (Volume Weighted Average Price)**

   - Price weighted by volume over a period
   - Parameters: `reset_period` ('daily', 'weekly', None)
   - Columns: `vwap`

2. **OBV (On-Balance Volume)**

   - Cumulative indicator that relates volume to price change
   - Parameters: `use_close`, `ma_period`, `ma_type`
   - Columns: `obv`, `obv_ma`

3. **MFI (Money Flow Index)**

   - Volume-weighted RSI
   - Parameters: `period`, `overbought`, `oversold`
   - Columns: `mfi`, `mfi_signal`

4. **ChaikinMoneyFlow**

   - Measures money flow volume over a period
   - Parameters: `period`
   - Columns: `cmf`

5. **VolumeProfile**
   - Analyzes volume distribution across price levels
   - Parameters: `price_bins`, `volume_threshold`, `window`
   - Columns: `value_area_high`, `value_area_low`, `point_of_control`

## Usage Examples

### Basic Usage

```python
from src.indicators.indicator_manager import IndicatorManager
import pandas as pd

# Initialize the indicator manager
indicator_mgr = IndicatorManager()

# Add indicators with custom parameters
indicator_mgr.add_indicator('my_rsi', 'rsi', {'period': 14})
indicator_mgr.add_indicator('my_macd', 'macd', {
    'fast_period': 12,
    'slow_period': 26,
    'signal_period': 9
})

# Load market data
data = pd.read_csv('market_data.csv')

# Apply indicators to the data
result = indicator_mgr.apply_indicators(data)

# Use the results
print(f"Latest RSI: {result['rsi'].iloc[-1]}")
print(f"MACD Histogram: {result['macd_histogram'].iloc[-1]}")
```

### Configuring Default Indicators

```python
# Initialize the indicator manager with default indicators
indicator_mgr = IndicatorManager()
indicator_mgr.configure_default_indicators()

# List configured indicators
current_indicators = indicator_mgr.get_current_indicators()
print(f"Configured indicators: {current_indicators}")
```

### Strategy-Specific Indicators

```python
# Configure indicators for a specific strategy
strategy_config = {
    'name': 'ema_crossover',
    'parameters': {
        'fast_ema': 9,
        'slow_ema': 21
    }
}

indicator_mgr = IndicatorManager()
configured_indicators = indicator_mgr.configure_strategy_indicators(strategy_config)

# Apply these indicators to market data
data = pd.read_csv('market_data.csv')
result = indicator_mgr.apply_indicators(data, configured_indicators)
```

## Developer Guide

### Adding a New Indicator

To add a new indicator:

1. **Choose the appropriate category** based on indicator type (momentum, trend, volatility, volume)
2. **Create a new class** inheriting from `BaseIndicator`
3. **Implement required methods**:
   - `__init__` for parameter initialization
   - `calculate` to compute indicator values
   - `get_column_names` to report added columns

Example:

```python
from typing import Dict, Any, List
import pandas as pd
from src.indicators.base_indicator import BaseIndicator

class MyNewIndicator(BaseIndicator):
    """My custom indicator."""

    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize the indicator.

        Args:
            params: Dictionary with parameters:
                - period: Look-back period (default 14)
        """
        default_params = {'period': 14}
        if params:
            default_params.update(params)
        super().__init__(default_params)

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate indicator values.

        Args:
            data: DataFrame with price data

        Returns:
            DataFrame with added indicator column
        """
        if not self.validate_data(data):
            raise ValueError("Input data missing required columns")

        period = self.params['period']
        df = data.copy()

        # Calculate the indicator values
        df['my_indicator'] = df['close'].rolling(window=period).mean() / df['close']

        return df

    def get_column_names(self) -> List[str]:
        """Get column names added by this indicator."""
        return ['my_indicator']
```

4. **Register the indicator** in the `IndicatorManager._build_indicator_registry` method:

```python
def _build_indicator_registry(self) -> Dict[str, Type[BaseIndicator]]:
    """Build the registry of available indicators."""
    registry = {
        # ... existing indicators ...

        # Add your new indicator
        'my_indicator': MyNewIndicator,
    }
    return registry
```

5. **Import the new indicator** in the appropriate module imports.

### Modifying an Existing Indicator

To modify an existing indicator:

1. **Identify the indicator class** to modify
2. **Update its parameters** in the `__init__` method if needed
3. **Modify the calculation logic** in the `calculate` method
4. **Update the column names** in the `get_column_names` method if you've changed outputs

### Creating a Custom Indicator Manager

For specialized use cases, you might want to create a custom indicator manager:

```python
from src.indicators.indicator_manager import IndicatorManager

class MyCustomIndicatorManager(IndicatorManager):
    """Custom indicator manager with specialized functionality."""

    def __init__(self):
        """Initialize with custom configuration."""
        super().__init__()

    def configure_custom_indicators(self):
        """Set up a custom set of indicators."""
        self.add_indicator('specialized_rsi', 'rsi', {'period': 7})
        self.add_indicator('specialized_ma', 'ma', {'period': 5, 'type': 'ema'})

    def apply_with_custom_logic(self, data):
        """Apply indicators with custom pre/post processing."""
        # Pre-processing
        df = data.copy()

        # Apply indicators
        df = self.apply_indicators(df)

        # Post-processing
        df['custom_signal'] = df['specialized_rsi'] > 50

        return df
```

## Best Practices

1. **Descriptive naming**: Use clear, descriptive names for indicator instances
2. **Parameter documentation**: Document all parameters in the class docstring
3. **Error handling**: Validate inputs and handle edge cases appropriately
4. **Performance optimization**: Consider performance for large datasets
5. **Memory management**: Drop intermediate calculation columns to save memory
6. **Testing**: Create unit tests for each indicator to ensure correct calculations
7. **Consistent interfaces**: Follow the base class interface for all new indicators

## API Reference

### BaseIndicator

```python
class BaseIndicator(ABC):
    def __init__(self, params: Optional[Dict[str, Any]] = None)
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame
    def validate_data(self, data: pd.DataFrame) -> bool
    def get_column_names(self) -> List[str]
    @staticmethod
    def normalize_value(value: float, min_val: float, max_val: float) -> float
    @staticmethod
    def moving_average(data: pd.Series, window: int, ma_type: str = 'sma') -> pd.Series
```

### IndicatorManager

```python
class IndicatorManager:
    def __init__(self)
    def add_indicator(self, name: str, indicator_type: str, params: Optional[Dict[str, Any]] = None) -> None
    def remove_indicator(self, name: str) -> None
    def get_indicator(self, name: str) -> BaseIndicator
    def apply_indicators(self, data: pd.DataFrame, indicator_names: Optional[List[str]] = None) -> pd.DataFrame
    def get_available_indicators(self) -> List[str]
    def get_current_indicators(self) -> Dict[str, str]
    def configure_default_indicators(self) -> None
    def configure_strategy_indicators(self, strategy_config: Dict[str, Any]) -> List[str]
```

## Troubleshooting

Common issues and solutions:

1. **NaN values in results**:

   - Check if you have enough data to cover the look-back period
   - Handle edge cases in the calculation method

2. **Performance issues**:

   - Use vectorized operations instead of loops when possible
   - Consider removing intermediate calculation columns

3. **Inconsistent results**:
   - Ensure data is properly sorted by timestamp
   - Verify your input data has the required columns

## Future Enhancements

Planned improvements for the indicators system:

1. **Parallelized calculations** for better performance with large datasets
2. **Caching mechanisms** to avoid redundant calculations
3. **Interactive visualization** tools for indicator analysis
4. **Machine learning integration** for optimizing indicator parameters
5. **Real-time indicator updates** for live trading
