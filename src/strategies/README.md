# Trading Strategies System

The Trading Strategies System is a powerful, modular framework for implementing, managing, and optimizing algorithmic trading strategies. It provides a comprehensive API for strategy development, signal generation, performance tracking, and dynamic optimization.

## Overview

The strategies system is built on a foundation of:

- A **base strategy class** that defines the core strategy interface and provides common functionality
- A collection of **individual strategy implementations** for different trading approaches
- A modular **strategy manager** that controls loading, execution, and performance tracking
- A **signal aggregation system** that intelligently combines signals from multiple strategies
- A **performance tracking system** for measuring and optimizing strategy performance
- A **strategy optimization framework** for parameter tuning and weight adjustment

## Architecture

![Trading Strategies Architecture](../docs/images/strategy_architecture.png)

The trading strategies system is organized into the following components:

- `base_strategy.py`: Defines the abstract base class for all strategies
- Strategy implementations (e.g., `ema_crossover_strategy.py`, `rsi_reversal_strategy.py`)
- `manager/` directory:
  - `core.py`: The main `StrategyManager` class
  - `loader.py`: Dynamic strategy loading functionality
  - `signal_aggregator.py`: Signal aggregation and processing
  - `performance_tracker.py`: Strategy performance management
  - `optimization.py`: Strategy parameter optimization

## System Integrations

The strategies system integrates with several other components of the trading platform:

### Indicator System

All strategies use the `IndicatorManager` from the `src.indicators` module to:

- Request and calculate technical indicators
- Access pre-computed indicator values
- Manage indicator dependencies efficiently

```python
# Example of indicator integration
def _init_indicators(self):
    self.indicators.add_indicator(
        "rsi",
        {"length": 14}
    )

    self.indicators.add_indicator(
        "ema",
        {"length": 21, "target_column": "close"}
    )
```

### Data Models

The strategies system uses shared data models from `src.models`:

- `SignalType` enum for categorizing signals (BUY/SELL/NEUTRAL)
- `SignalStrength` enum for qualitative signal strength
- Common data structures for interoperability with other system components

### Trading Execution Components

The signals generated by strategies are consumed by:

- **Order Management**: Converts signals to executable orders
- **Position Management**: Tracks and manages open positions
- **Risk Management**: Applies risk controls to signals
- **Paper Trading**: Simulates execution in a risk-free environment

## Trading Strategies

The system includes the following built-in trading strategies:

1. **EMA Crossover** (`ema_crossover_strategy.py`):

   - Generates signals based on fast and slow EMA crossovers
   - Uses volume confirmation to filter signals

2. **RSI Reversal** (`rsi_reversal_strategy.py`):

   - Identifies potential reversals using the RSI indicator
   - Detects bullish and bearish divergences
   - Includes volume and trend confirmations

3. **Bollinger Breakout** (`bollinger_breakout_strategy.py`):

   - Identifies breakouts from Bollinger Bands
   - Includes volatility-adjusted position sizing

4. **MACD Trend Following** (`macd_trend_following_strategy.py`):

   - Generates signals based on MACD crossovers and histogram patterns
   - Identifies potential trend reversals

5. **VWAP Trend Trading** (`vwap_trend_trading_strategy.py`):
   - Uses VWAP as a trend indicator
   - Combines with additional momentum indicators

## Signal Generation

Each strategy implements the `generate_signals()` method, which analyzes market data and produces trading signals. A Signal includes:

- Signal type (BUY/SELL/NEUTRAL)
- Symbol
- Timestamp
- Price
- Strength (0.0-1.0)
- Metadata (strategy-specific information)

## Strategy Manager

The `StrategyManager` class provides a unified interface for working with multiple strategies:

- Dynamically loads and initializes strategies from configuration
- Aggregates signals from different strategies
- Tracks performance metrics for each strategy
- Dynamically adjusts strategy weights based on performance
- Persists performance data between sessions

## Usage

### Basic Usage

```python
from src.strategies.strategy_manager import StrategyManager
from src.indicators.indicator_manager import IndicatorManager

# Initialize managers
indicator_manager = IndicatorManager(config)
strategy_manager = StrategyManager(config, indicator_manager)

# Fetch market data
market_data = fetch_market_data(symbol, timeframe)

# Generate signals
signals = strategy_manager.generate_signals(market_data)

# Process signals
for signal in signals:
    print(f"Signal: {signal.signal_type} for {signal.symbol} at {signal.price}")
    print(f"Strength: {signal.strength}, Strategy: {signal.metadata.get('strategy_name')}")
```

### Creating Custom Strategies

To create a custom strategy:

1. Create a new Python file in the `src/strategies` directory named `<your_strategy_name>_strategy.py`
2. Define a class that inherits from `BaseStrategy`
3. Implement the required methods (`_init_indicators`, `generate_signals`, etc.)
4. Add your strategy to the dynamic loader in `manager/loader.py`

Example:

```python
from src.strategies.base_strategy import BaseStrategy, Signal, SignalType

class MyCustomStrategy(BaseStrategy):
    def __init__(self, config, indicator_manager):
        super().__init__(config, indicator_manager)
        self.name = "my_custom"
        self.description = "My Custom Trading Strategy"
        self._init_indicators()

    def _init_indicators(self):
        # Initialize required indicators
        self.indicators.add_indicator("rsi", {"length": 14})

    def generate_signals(self, data):
        signals = []
        # Generate signals based on your logic
        # ...
        return signals

    def get_stop_loss_price(self, entry_price, is_long, data):
        # Custom stop loss logic
        return entry_price * 0.95 if is_long else entry_price * 1.05
```

## Signal Aggregation

The system aggregates signals from multiple strategies to produce higher-confidence trading decisions. This includes:

- Filtering for minimum strategy agreement
- Weighted aggregation based on strategy performance
- Signal strength thresholds
- Metadata combination for transparency

## Performance Tracking

The performance tracking system maintains metrics for each strategy:

- Win rate
- Total profit/loss
- Number of signals generated/executed
- Strategy weight (for signal aggregation)

These metrics are used to dynamically adjust strategy weights and optimize performance over time.

## Strategy Optimization

The strategy optimization framework provides several capabilities for fine-tuning strategy performance:

### Weight Optimization

The system can dynamically adjust strategy weights based on performance metrics:

```python
# Manually adjust a strategy's weight
strategy_manager.adjust_strategy_weight("rsi_reversal", 1.5)

# Automatically optimize all strategy weights
optimized_weights = optimize_strategy_weights(
    strategy_performance,
    current_weights,
    min_trades=20
)

# Apply the optimized weights
for strategy_name, weight in optimized_weights.items():
    strategy_manager.adjust_strategy_weight(strategy_name, weight)
```

### Strategy Selection

You can select the top-performing strategies for live trading:

```python
# Get the top 3 strategies based on performance
top_strategies = select_top_strategies(
    strategy_performance,
    max_strategies=3,
    min_trades=10
)

# Enable only the top strategies
for strategy in strategy_manager.get_enabled_strategies():
    if strategy not in top_strategies:
        strategy_manager.disable_strategy(strategy)
```

### Performance-Based Recommendations

The system can provide recommendations for strategy adjustments:

```python
# Get recommended weight adjustments
recommendations = get_recommended_adjustments(
    strategy_performance,
    strategy_weights
)

for rec in recommendations:
    print(f"Recommended adjustment for {rec['strategy_name']}: "
          f"{rec['current_weight']} -> {rec['recommended_weight']}")
```

## Configuration

Strategies are configured through the central configuration system:

```json
{
  "strategies": {
    "ema_crossover": {
      "enabled": true,
      "fast_ema": 9,
      "slow_ema": 21,
      "volume_threshold": 1.5
    },
    "rsi_reversal": {
      "enabled": true,
      "rsi_length": 14,
      "rsi_overbought": 70,
      "rsi_oversold": 30
    }
  },
  "strategy_weights": {
    "ema_crossover": 1.0,
    "rsi_reversal": 1.2
  },
  "default_strategy_weight": 1.0,
  "signal_threshold": 0.6,
  "use_weighted_aggregation": true,
  "min_concurrent_strategies": 1,
  "dynamic_weighting": true
}
```

## API Reference

### BaseStrategy

The abstract base class for all trading strategies.

**Key Methods:**

- `__init__(config, indicator_manager)`: Initialize strategy
- `prepare_data(data)`: Prepare market data for signal generation
- `generate_signals(data)`: Generate trading signals
- `get_stop_loss_price(entry_price, is_long, data)`: Calculate stop loss price
- `calculate_take_profit(entry_price, stop_loss_price, is_long)`: Calculate take profit price
- `should_adjust_position(position_data, current_data)`: Determine if position size should be adjusted

### StrategyManager

Manages multiple trading strategies and aggregates their signals.

**Key Methods:**

- `__init__(config, indicator_manager)`: Initialize strategy manager
- `generate_signals(data)`: Generate signals from all enabled strategies
- `update_performance(trade_result)`: Update strategy performance based on trade result
- `save_performance(file_path)`: Save performance metrics to file
- `load_performance(file_path)`: Load performance metrics from file
- `get_enabled_strategies()`: Get list of enabled strategy names
- `get_strategy_info()`: Get information about all strategies
- `enable_strategy(strategy_name)`: Enable a disabled strategy
- `disable_strategy(strategy_name)`: Disable an enabled strategy
- `adjust_strategy_weight(strategy_name, weight)`: Manually adjust strategy weight

### Optimization Functions

The optimization module provides functions for strategy optimization:

**Key Functions:**

- `adjust_strategy_weight(strategy_performance, strategy_weights, strategy_name, weight)`: Manually adjust strategy weight
- `optimize_strategy_weights(strategy_performance, strategy_weights, min_trades)`: Optimize weights based on performance
- `select_top_strategies(strategy_performance, max_strategies, min_trades)`: Select best-performing strategies
- `get_recommended_adjustments(strategy_performance, strategy_weights)`: Get weight adjustment recommendations

## Developer Guide

### Best Practices

1. **Signal Generation**:

   - Filter out noise and low-confidence signals
   - Use multiple confirmation factors
   - Scale signal strength based on conviction
   - Provide detailed metadata for debugging

2. **Risk Management**:

   - Always implement `get_stop_loss_price()` and `calculate_take_profit()`
   - Consider market volatility in risk calculations
   - Account for slippage and execution risk

3. **Performance Optimization**:

   - Keep indicator calculations efficient
   - Cache calculations where appropriate
   - Handle edge cases (missing data, market closes, etc.)

4. **Strategy Optimization**:
   - Use performance metrics to tune strategy parameters
   - Test strategies across different market conditions
   - Implement proper validation to avoid overfitting
   - Regularly review and adjust strategy weights

### Testing Strategies

Each strategy should be tested using:

1. **Backtesting**: Test historical performance
2. **Paper Trading**: Verify real-time signal generation
3. **Unit Tests**: Verify specific strategy logic
4. **Integration Tests**: Validate interaction with other system components

## Future Enhancements

- Machine learning integration for signal optimization
- Advanced portfolio management strategies
- Risk-adjusted performance metrics
- Strategy auto-generation and genetic optimization
- Market regime detection and strategy rotation
