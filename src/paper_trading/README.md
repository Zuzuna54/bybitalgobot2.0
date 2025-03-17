# Paper Trading System

## Overview

The Paper Trading System provides a realistic simulation environment for testing trading strategies in real-time market conditions without risking real funds. It mimics the behavior of an actual trading system by processing strategy signals, managing positions, handling order execution, and tracking performance metrics.

This system allows developers and traders to validate their strategies in a production-like environment before deploying them with real capital, providing a critical bridge between backtesting and live trading.

## Architecture

The Paper Trading System follows a modular design with the following components:

```
PaperTradingSimulator (Main Class)
├── Order Processor
│   ├── Process pending orders
│   ├── Process limit orders
│   ├── Process stop orders
│   ├── Process take profit orders
│   └── Calculate execution prices
├── Position Manager
│   ├── Update positions
│   ├── Close positions
│   ├── Update stop losses
│   └── Calculate equity
├── Execution Engine
│   ├── Execute paper trades
│   ├── Process strategy signals
│   └── Retrieve market data
└── State Manager
    ├── Save state
    ├── Load state
    ├── Generate summaries
    └── Compare to backtest results
```

## Components

### Paper Trading Simulator (`simulator.py`)

The `PaperTradingSimulator` class serves as the main interface to the paper trading system:

- Initializes and coordinates all components
- Manages real-time simulation loop
- Tracks account balance, positions, orders, and trade history
- Handles saving and loading state
- Provides performance reporting and summary statistics

Key methods:

- `setup()`: Configure the simulator with dependencies
- `start()`: Begin the simulation for specified symbols
- `stop()`: Stop the simulation and save state
- `get_summary()`: Get a summary of the paper trading status
- `get_performance_report()`: Get detailed performance metrics

### Order Processor (`components/order_processor.py`)

The order processor handles all aspects of order management:

- Processes pending orders based on current market prices
- Determines when limit, stop, and take profit orders should be triggered
- Calculates execution prices with slippage
- Manages order lifecycle from creation to execution

Key functions:

- `process_pending_orders()`: Check and execute all pending orders
- `process_limit_order()`: Process limit order execution conditions
- `process_stop_order()`: Process stop loss order execution conditions
- `process_take_profit_order()`: Process take profit order execution conditions
- `calculate_execution_price()`: Calculate execution price with slippage

### Position Manager (`components/position_manager.py`)

The position manager tracks and updates open positions:

- Updates position valuations with current market prices
- Monitors stop loss and take profit levels
- Implements trailing stop mechanisms
- Calculates unrealized P&L and total equity
- Handles position closing and performance tracking

Key functions:

- `update_positions()`: Update valuations and check exit conditions for all positions
- `close_position()`: Close a position and calculate results
- `update_stop_loss_in_trade_history()`: Update trailing stops in trade records
- `calculate_total_equity()`: Calculate total account value including open positions

### Execution Engine (`components/execution_engine.py`)

The execution engine processes trading signals and executes paper trades:

- Executes trades based on strategy signals
- Calculates position sizes based on risk parameters
- Simulates execution latency and slippage
- Fetches and prepares market data for strategy evaluation
- Applies risk management rules to trade execution

Key functions:

- `execute_paper_trade()`: Create a paper trade based on a signal
- `get_market_data()`: Retrieve and prepare market data for a symbol
- `process_strategy_signals()`: Process signals from strategies and execute trades

### State Manager (`components/state_manager.py`)

The state manager handles persistence and reporting:

- Saves and loads simulator state to/from disk
- Generates performance summaries and reports
- Compares paper trading results with backtest results
- Provides JSON serialization for complex data types

Key functions:

- `save_state()`: Save current state to disk
- `load_state()`: Load saved state from disk
- `get_summary()`: Generate a summary of paper trading status
- `compare_to_backtest()`: Compare paper trading results to backtest results

## Usage

### Basic Usage

```python
from src.paper_trading import PaperTradingSimulator
from src.api.bybit_client import BybitClient
from src.data.market_data import MarketData
from src.strategies.strategy_manager import StrategyManager
from src.risk_management.risk_manager import RiskManager

# Initialize the paper trading simulator
config = {
    "default_timeframe": "1h",
    "max_positions": 5
}
paper_sim = PaperTradingSimulator(
    config=config,
    initial_balance=10000.0,
    slippage=0.05,  # 0.05% slippage
    commission=0.075  # 0.075% commission
)

# Initialize dependencies
api_client = BybitClient(testnet=True)
market_data = MarketData()
risk_manager = RiskManager()
strategy_manager = StrategyManager()

# Set up simulator
paper_sim.setup(
    api_client=api_client,
    market_data=market_data,
    risk_manager=risk_manager,
    strategy_manager=strategy_manager
)

# Start paper trading
symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
paper_sim.start(symbols, update_interval_sec=5)

# Run for some time...

# Get a summary
summary = paper_sim.get_summary()
print(f"Current equity: ${summary['total_equity']}")
print(f"Return: {summary['total_return_pct']}%")

# Stop paper trading
paper_sim.stop()
```

### Loading Saved State

```python
# Initialize with the same data directory to load previous state
paper_sim = PaperTradingSimulator(
    config=config,
    initial_balance=10000.0,
    data_dir="data/paper_trading"
)

# Setup, then state will be loaded automatically on start
paper_sim.setup(api_client, market_data, risk_manager, strategy_manager)
paper_sim.start(symbols)
```

### Comparing to Backtest

```python
# Get backtest results from the backtesting module
from src.backtesting import BacktestEngine
backtest_engine = BacktestEngine()
backtest_results = backtest_engine.run(strategy, "BTCUSDT", "2023-01-01", "2023-01-31")

# Compare paper trading performance to backtest
paper_performance = paper_sim.get_performance_report()
comparison = paper_sim.compare_to_backtest(paper_performance, backtest_results)

print(f"Return difference: {comparison['total_return']['difference_pct']}%")
print(f"Win rate difference: {comparison['win_rate']['difference_pct']}%")
```

## Developer Guide

### Extending the Paper Trading System

#### Customizing Risk Management

To implement custom risk management logic:

1. Extend the `RiskManager` class with your custom logic:

```python
from src.risk_management.risk_manager import RiskManager

class CustomRiskManager(RiskManager):
    def calculate_position_size(self, symbol, account_balance, entry_price, stop_loss_price):
        # Custom position sizing logic
        risk_per_trade = account_balance * 0.01  # 1% risk per trade
        price_distance = abs(entry_price - stop_loss_price)
        return risk_per_trade / price_distance

    def should_take_trade(self, symbol, signal_strength, account_balance):
        # Custom trade filtering logic
        if signal_strength < 0.7:
            return False
        return True
```

2. Use your custom risk manager with the paper trading simulator:

```python
risk_manager = CustomRiskManager()
paper_sim.setup(api_client, market_data, risk_manager, strategy_manager)
```

#### Adding Custom Execution Logic

To add custom execution behavior:

1. Create a subclass of `PaperTradingSimulator`:

```python
from src.paper_trading import PaperTradingSimulator

class CustomPaperTradingSimulator(PaperTradingSimulator):
    def _execute_order(self, order, price):
        # Custom execution logic
        # Example: Add random execution delays
        import random
        import time
        delay = random.uniform(0.1, 2.0)  # Random delay between 100ms and 2s
        time.sleep(delay)

        # Call parent method to complete execution
        super()._execute_order(order, price)
```

2. Use your custom simulator class:

```python
paper_sim = CustomPaperTradingSimulator(config, initial_balance=10000.0)
```

### Implementing Custom Position Management

To implement custom position management:

1. Create a module with your custom position management functions:

```python
def custom_update_positions(active_positions, get_current_price_func, close_position_func, risk_manager):
    # Custom position update logic
    for symbol, position in active_positions.items():
        # Implement your custom position management logic here
        pass
```

2. Integrate it with the paper trading simulator:

```python
from src.paper_trading import PaperTradingSimulator

class CustomPositionManager(PaperTradingSimulator):
    def _simulation_loop(self, symbols, update_interval_sec):
        while self.is_running and not self.stop_event.is_set():
            try:
                # Use custom position management
                custom_update_positions(
                    active_positions=self.active_positions,
                    get_current_price_func=self.market_data.get_current_price,
                    close_position_func=self._close_position,
                    risk_manager=self.risk_manager
                )

                # Rest of the loop logic...
                time.sleep(update_interval_sec)

            except Exception as e:
                logger.error(f"Error in paper trading simulation loop: {e}")
                time.sleep(update_interval_sec)
```

## API Reference

### PaperTradingSimulator

```python
class PaperTradingSimulator:
    def __init__(self, config, initial_balance=10000.0, data_dir="data/paper_trading",
                 slippage=0.05, commission=0.075, latency_ms=100)

    def setup(self, api_client, market_data, risk_manager, strategy_manager)

    def start(self, symbols, update_interval_sec=5)

    def stop()

    def get_summary() -> Dict[str, Any]

    def get_performance_report() -> Dict[str, Any]

    def compare_to_backtest(backtest_results) -> Dict[str, Any]
```

### Order Processor

```python
def process_pending_orders(pending_orders, get_current_price_func, close_position_func, execute_order_func) -> None

def process_limit_order(order, current_price, pending_orders, execute_order_func) -> None

def process_stop_order(order, current_price, pending_orders, close_position_func) -> None

def process_take_profit_order(order, current_price, pending_orders, close_position_func) -> None

def calculate_execution_price(symbol, is_long, slippage, get_current_price_func) -> float
```

### Position Manager

```python
def update_positions(active_positions, get_current_price_func, close_position_func, risk_manager) -> None

def close_position(symbol, exit_price, exit_reason, active_positions, trade_history,
                 commission_rate, risk_manager, performance_tracker, strategy_manager) -> Optional[float]

def update_stop_loss_in_trade_history(trade_id, new_stop_loss, trade_history) -> None

def calculate_total_equity(current_balance, active_positions) -> float
```

### Execution Engine

```python
def execute_paper_trade(signal, market_data, strategy_manager, risk_manager, performance_tracker,
                      active_positions, trade_history, current_balance, slippage,
                      commission_rate, latency_ms) -> Optional[Tuple[str, float]]

def get_market_data(symbol, market_data, strategy_manager) -> Optional[Dict[str, Any]]

def process_strategy_signals(symbols, market_data, strategy_manager, risk_manager, performance_tracker,
                           active_positions, trade_history, current_balance, slippage,
                           commission_rate, latency_ms) -> Optional[float]
```

### State Manager

```python
def save_state(data_dir, current_balance, active_positions, pending_orders,
             trade_history, equity_history) -> bool

def load_state(data_dir, initial_balance) -> Dict[str, Any]

def get_summary(initial_balance, current_balance, active_positions,
              pending_orders, trade_history) -> Dict[str, Any]

def compare_to_backtest(paper_performance_report, backtest_results) -> Dict[str, Any]
```

## Best Practices

1. **Realistic Simulation**: Configure slippage, commission, and latency parameters to match real trading conditions.

2. **Risk Management**: Always implement proper risk management rules to avoid unrealistic results.

3. **State Persistence**: Regularly save simulator state to avoid data loss during long simulations.

4. **Symbol Selection**: Start with a small set of symbols to ensure the simulator runs efficiently.

5. **Validation**: Compare paper trading results with backtest results to validate strategy performance.

6. **Resource Management**: Monitor memory and CPU usage during long-running simulations.

7. **Data Quality**: Ensure high-quality market data feeds for accurate simulation results.

8. **Testing**: Thoroughly test custom implementations before using them for strategy evaluation.

## Troubleshooting

### Common Issues

1. **High CPU Usage**:

   - Reduce the number of monitored symbols
   - Increase the update interval
   - Optimize strategy calculations

2. **Memory Leaks**:

   - Limit the size of trade history
   - Periodically clear old data
   - Use efficient data structures

3. **Unrealistic Results**:

   - Verify slippage and commission settings
   - Ensure proper risk management
   - Check for data quality issues

4. **Slow Execution**:
   - Reduce the complexity of strategies
   - Optimize market data retrieval
   - Use faster data structures for lookups

### Debugging

To enable detailed logging for debugging:

```python
import logging
from loguru import logger

# Configure logger
logger.remove()
logger.add(sys.stderr, level="DEBUG")

# Initialize simulator with debug settings
paper_sim = PaperTradingSimulator(
    config=config,
    initial_balance=10000.0,
    slippage=0.05,
    commission=0.075
)
```

## Future Enhancements

Planned improvements for the paper trading system:

1. **Multi-exchange support** to simulate trading across different exchanges
2. **Advanced order types** such as OCO (One-Cancels-the-Other) orders
3. **Portfolio optimization** tools to improve capital allocation
4. **Performance visualization** with interactive charts
5. **Scenario analysis** to test strategies under different market conditions
6. **Event simulation** to test strategy responses to market events
7. **Real-time strategy optimization** based on paper trading results
