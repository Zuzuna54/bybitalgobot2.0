# Trade Management System

The Trade Management System handles the execution, tracking, and lifecycle management of trades in the Algorithmic Trading System. It provides a robust framework for converting trading signals into executable orders, managing positions, and generating performance reports.

## Overview

The Trade Management system is designed to:

- Process trading signals and convert them into executable orders
- Manage position tracking and trade lifecycle states
- Execute market, limit, stop, and take profit orders
- Calculate and apply stop loss and take profit levels
- Track trade performance and generate execution reports
- Provide trade history and performance analytics

## Architecture

The Trade Management system is built around a central `TradeManager` class that orchestrates the following components:

- **Order Handler**: Creates and manages different types of orders
- **Position Tracker**: Tracks the status and details of trades
- **Trade Lifecycle**: Manages the process from signal to execution to closure
- **Execution Reporter**: Generates performance metrics and trade history data

### System Integration

The Trade Management module integrates with:

- **API Client** (`src/api/bybit`): For executing orders on the exchange
- **Risk Management** (`src/risk_management`): For applying risk rules and position sizing
- **Strategy System** (`src/strategies`): For receiving trade signals
- **Dashboard** (`src/dashboard`): For displaying trade data and performance

## Components

### 1. Trade Manager

The `TradeManager` class is the central component that orchestrates the trade management process:

```python
from src.trade_management import TradeManager
from src.api.bybit import BybitClient
from src.risk_management import RiskManager

# Initialize managers
api_client = BybitClient(api_key, api_secret)
risk_manager = RiskManager(config)

# Create trade manager
trade_manager = TradeManager(
    api_client=api_client,
    risk_manager=risk_manager,
    simulate=False
)

# Process a trading signal
trade_id = trade_manager.process_signal(signal)

# Update active trades with current market data
trade_manager.update_active_trades(market_data)

# Get trade summary
summary = trade_manager.get_trade_summary()
```

### 2. Order Handler

The order handler component provides functions for creating and managing different types of orders:

- `create_market_order()`: Create a market order
- `create_limit_order()`: Create a limit order
- `create_stop_order()`: Create a stop loss order
- `create_take_profit_order()`: Create a take profit order
- `update_stop_loss_order()`: Update an existing stop loss order
- `cancel_order()`: Cancel an existing order
- `get_order_status()`: Get the status of an order

### 3. Position Tracker

The position tracker component manages trade objects and their states:

- `Trade` class: Represents a trade with all its attributes
- `TradeStatus` enum: Defines possible trade states (PENDING, OPEN, CLOSED, etc.)
- `update_position_market_data()`: Update a trade with current market data
- `calculate_unrealized_profit_pct()`: Calculate unrealized profit percentage
- `get_active_positions_summary()`: Get summary of active positions

### 4. Trade Lifecycle

The trade lifecycle component manages the process from signal to execution to closure:

- `process_trading_signal()`: Process a signal and create a trade if valid
- `execute_trade_entry()`: Execute a trade entry by placing necessary orders
- `close_trade_at_market()`: Close a trade at market price

### 5. Execution Reporter

The execution reporter component generates performance metrics and trade history:

- `get_trade_summary()`: Generate summary of all trades
- `get_trade_history_dataframe()`: Convert trades to a DataFrame
- `save_trade_history()`: Save trade history to a CSV file
- `get_trades_by_symbol()`: Get trades for a specific symbol
- `get_trades_by_strategy()`: Get trades for a specific strategy
- `get_performance_by_strategy()`: Calculate performance metrics by strategy

## Usage Examples

### Processing a Trading Signal

```python
from src.models.models import Signal, SignalType

# Create a trading signal
signal = Signal(
    signal_type=SignalType.BUY,
    symbol="BTCUSDT",
    price=50000.0,
    timestamp=datetime.now(),
    strength=0.8,
    metadata={
        "strategy_name": "ema_crossover",
        "indicators": {
            "atr": 1200.0,
            "volatility": 0.03
        }
    }
)

# Process the signal
trade_id = trade_manager.process_signal(signal)

# Check if a trade was created
if trade_id:
    print(f"Trade created with ID: {trade_id}")
else:
    print("Signal did not result in a trade")
```

### Managing Active Trades

```python
# Get current market data (example)
market_data = {
    "BTCUSDT": {"price": 51000.0},
    "ETHUSDT": {"price": 3000.0}
}

# Update all active trades with current prices
trade_manager.update_active_trades(market_data)

# Get a specific trade by ID
trade = trade_manager.get_trade_by_id("BTCUSDT-20230101123045")

# Get trade summary
summary = trade_manager.get_trade_summary()
print(f"Total trades: {summary['total_trades']}")
print(f"Win rate: {summary['win_rate']:.1f}%")
print(f"Total PnL: {summary['total_pnl']:.2f}")
```

### Generating Trade Reports

```python
# Get trade history as DataFrame
trade_history_df = trade_manager.get_trade_history_df()

# Save trade history to CSV
trade_manager.save_trade_history("trade_history.csv")

# Get performance by strategy
strategy_performance = get_performance_by_strategy(trade_manager.trades)
for strategy_name, metrics in strategy_performance.items():
    print(f"Strategy: {strategy_name}")
    print(f"  Win rate: {metrics['win_rate']:.1f}%")
    print(f"  Total PnL: {metrics['total_pnl']:.2f}")
```

## Developer Guide

### Extending the System

#### Adding New Order Types

To add a new order type:

1. Add a new value to the `OrderType` enum in `order_handler.py`
2. Create a new function to handle the order creation
3. Update the components package exports in `__init__.py`

Example:

```python
# In order_handler.py
class OrderType(Enum):
    """Order types for trade execution."""
    MARKET = "Market"
    LIMIT = "Limit"
    STOP_MARKET = "StopMarket"
    STOP_LIMIT = "StopLimit"
    TAKE_PROFIT_MARKET = "TakeProfitMarket"
    TAKE_PROFIT_LIMIT = "TakeProfitLimit"
    TRAILING_STOP = "TrailingStop"  # New order type

def create_trailing_stop_order(
    api_client: BybitClient,
    symbol: str,
    side: OrderSide,
    qty: float,
    activation_price: float,
    callback_rate: float,
    reduce_only: bool = True,
    simulate: bool = False
) -> Dict[str, Any]:
    """
    Create a trailing stop order.

    Args:
        api_client: Bybit API client
        symbol: Trading pair symbol
        side: Order side (BUY or SELL)
        qty: Order quantity
        activation_price: Activation price for the trailing stop
        callback_rate: Callback rate in percentage
        reduce_only: Whether this order should only reduce position
        simulate: Whether to simulate the order instead of placing it

    Returns:
        Order result dictionary
    """
    # Implementation here
    ...

# In __init__.py
from src.trade_management.components.order_handler import (
    # ... existing imports
    create_trailing_stop_order
)

__all__ = [
    # ... existing exports
    'create_trailing_stop_order'
]
```

#### Customizing Trade Lifecycle

To customize the trade lifecycle:

1. Extend or modify the `process_trading_signal` function to add custom validation
2. Create a specialized trade execution function for specific order types
3. Add custom exit conditions in the `update_active_trades` method

Example for custom exit conditions:

```python
# In TradeManager class
def update_active_trades(self, market_data: Dict[str, Any]) -> None:
    """Update all active trades with current market data."""
    current_time = datetime.now()

    for trade_id, trade in list(self.active_trades.items()):
        symbol = trade.symbol

        # Skip if no market data for this symbol
        if symbol not in market_data:
            continue

        # Get current price
        current_price = float(market_data[symbol].get("price", 0))

        # Update trade market data
        update_position_market_data(trade, current_price, current_time)

        # Custom exit condition: Time-based exit
        if trade.entry_time and (current_time - trade.entry_time).days >= 7:
            self._close_trade(trade, current_price, current_time, "time_exit")
            continue

        # Standard exit conditions
        if (trade.side == OrderSide.BUY and current_price <= trade.stop_loss_price) or \
           (trade.side == OrderSide.SELL and current_price >= trade.stop_loss_price):
            self._close_trade(trade, current_price, current_time, "stop_loss")

        # ... other conditions
```

### Best Practices

1. **Error Handling**:

   - Always wrap exchange API calls in try/except blocks
   - Include proper logging for trade execution events
   - Handle edge cases like partial fills, rejection, and timeout

2. **Performance Optimization**:

   - Cache trade data to minimize API calls
   - Use batch operations when possible
   - Implement rate limiting to avoid API restrictions

3. **Testing**:

   - Use the simulation mode for testing strategies before live execution
   - Create unit tests for each component
   - Implement integration tests for the full trade lifecycle

4. **Security**:
   - Always validate trade parameters before execution
   - Implement execution safeguards like maximum order size limits
   - Keep API credentials secure and use environment variables

## API Reference

### TradeManager

The main class for trade management.

**Methods:**

- `__init__(api_client, risk_manager, simulate, simulation_delay_sec)`: Initialize manager
- `process_signal(signal)`: Process a trading signal
- `execute_trade(trade)`: Execute a trade by placing orders
- `update_active_trades(market_data)`: Update active trades with current market data
- `_close_trade(trade, exit_price, exit_time, exit_reason)`: Close a trade
- `get_trade_summary()`: Get summary of all trades
- `get_trade_history_df()`: Get trade history as DataFrame
- `save_trade_history(file_path)`: Save trade history to CSV
- `get_trade_by_id(trade_id)`: Get a specific trade by ID

### Trade

Class representing a single trade.

**Properties:**

- `id`: Unique trade identifier
- `symbol`: Trading pair symbol
- `side`: Order side (BUY/SELL)
- `entry_price`: Entry price
- `stop_loss_price`: Stop loss price
- `take_profit_price`: Take profit price
- `position_size`: Position size
- `status`: Trade status (PENDING/OPEN/CLOSED/CANCELED/REJECTED)
- `orders`: Dictionary of orders associated with the trade
- `entry_time`: Entry timestamp
- `exit_time`: Exit timestamp
- `exit_price`: Exit price
- `realized_pnl`: Realized profit/loss
- `realized_pnl_percent`: Realized profit/loss percentage
- `exit_reason`: Reason for exiting the trade

**Methods:**

- `update_order(order_id, order_data)`: Update order information
- `set_status(status)`: Set trade status
- `close_trade(exit_price, exit_time, exit_reason)`: Close the trade
- `update_market_data(current_price, timestamp)`: Update with current market data
- `to_dict()`: Convert trade to dictionary

## Future Enhancements

- Advanced order types (trailing stops, OCO orders)
- Dynamic take profit and stop loss adjustments
- Multi-exchange support
- Trade scaling (pyramiding) functionality
- Enhanced backtesting integration
- Real-time P&L tracking and visualization
- Machine learning-based trade execution optimization
