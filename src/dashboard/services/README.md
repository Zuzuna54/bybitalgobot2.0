# Dashboard Services

This directory contains service classes that handle complex functionality for the dashboard. Services act as an abstraction layer between the data sources and the UI components, providing centralized and reusable functionality.

## Available Services

### Chart Service (`chart_service/`)

The chart service has been refactored into a modular package structure for better maintainability. It provides centralized visualization functionality for all dashboard components. This service handles the creation of charts and graphs using Plotly, ensuring consistent styling and behavior across the application.

#### Chart Service Package Structure:

- `chart_service/__init__.py` - Exports all chart functions for backward compatibility
- `chart_service/base.py` - Base chart utilities and theme
- `chart_service/performance_charts.py` - Performance visualization charts
- `chart_service/market_charts.py` - Market data visualization charts
- `chart_service/orderbook_charts.py` - Order book visualization charts
- `chart_service/strategy_charts.py` - Strategy performance visualization charts
- `chart_service/component_renderers.py` - UI component renderers

The original `chart_service.py` file is kept for backward compatibility but is marked as deprecated and will be removed in a future version.

#### Available Chart Functions:

**Base Utilities:**

- `apply_chart_theme(fig, title)` - Applies the dashboard's theme to any Plotly figure
- `create_empty_chart(message)` - Creates an empty chart with a message displayed
- `create_empty_sparkline()` - Creates an empty sparkline chart
- `filter_data_by_time_range(data, time_range)` - Filters DataFrame by time range from present

**Performance Charts:**

- `create_return_sparkline(returns_data)` - Creates a small sparkline for returns
- `create_equity_curve_chart(equity_data, time_range)` - Creates an equity curve visualization
- `create_return_distribution_chart(returns_data, time_range)` - Creates a return distribution histogram
- `create_drawdown_chart(equity_data, time_range)` - Creates a drawdown visualization
- `create_strategy_performance_chart(strategy_data)` - Creates a strategy performance visualization
- `create_trade_win_loss_chart(trade_data)` - Creates a win/loss donut chart
- `create_daily_performance_graph(daily_summary)` - Creates a daily performance graph

**Market Charts:**

- `create_custom_indicator_chart(market_data, indicator_name, symbol, timeframe)` - Creates a custom indicator chart
- `create_candlestick_chart(candles, symbol, title, show_volume, height)` - Creates a candlestick chart with optional volume bars

**Orderbook Charts:**

- `create_orderbook_depth_chart(orderbook, depth, support_levels, resistance_levels, height)` - Creates an orderbook depth chart
- `create_orderbook_heatmap(orderbook, height, title)` - Creates a heatmap visualization of the orderbook
- `create_orderbook_imbalance_chart(orderbook, depth_levels, height, title)` - Creates a bid/ask imbalance visualization
- `create_liquidity_profile_chart(orderbook, price_range_pct, height, title)` - Creates a liquidity profile visualization
- `create_orderbook_depth_graph` - Alias for `create_orderbook_depth_chart` (deprecated)

**Strategy Charts:**

- `create_strategy_performance_graph(strategies)` - Creates a strategy performance comparison graph
- `create_strategy_comparison_graph(strategy_performance, selected_strategies)` - Creates a bar chart comparing key metrics across strategies
- `create_detailed_performance_breakdown(strategy_performance, selected_strategy)` - Creates a detailed performance breakdown for a selected strategy
- `create_market_condition_performance(strategy_performance, selected_strategy)` - Creates a visualization of strategy performance across different market conditions
- `create_strategy_correlation_matrix(strategy_performance)` - Creates a correlation matrix heatmap for strategies

**Component Renderers:**

- `render_imbalance_indicator(imbalance)` - Renders an order book imbalance indicator
- `render_liquidity_ratio(liquidity_data)` - Renders a liquidity ratio indicator
- `render_support_resistance_levels(sr_levels)` - Renders support and resistance levels
- `render_level_strength_indicator(level_type, price, strength)` - Renders a support/resistance level strength indicator
- `create_level_confluence_chart(price_levels, current_price, height)` - Creates a chart showing confluence of support/resistance levels
- `render_execution_recommendations(recommendations)` - Renders execution recommendations based on order book analysis

#### Usage Example:

```python
from src.dashboard.services.chart_service import create_equity_curve_chart, apply_chart_theme

# Create a chart
fig = create_equity_curve_chart(performance_data, time_range="1m")

# Customize the chart (if needed)
fig.update_layout(height=600)

# Apply the dashboard theme
fig = apply_chart_theme(fig, "Custom Equity Curve")
```

### Data Service (`data_service/`)

The data service has been refactored into a modular package structure for better maintainability. It handles data retrieval and processing for dashboard components, providing a centralized interface for accessing trading system data.

#### Data Service Package Structure:

- `data_service/__init__.py` - Exports the DashboardDataService class and data retrieval methods
- `data_service/base.py` - Core data service class and utility functions
- `data_service/performance_data.py` - Performance data retrieval and processing
- `data_service/trade_data.py` - Trade data retrieval and processing
- `data_service/system_data.py` - System status and configuration data
- `data_service/market_data.py` - Market and orderbook data retrieval
- `data_service/strategy_data.py` - Strategy data retrieval and processing

The original `data_service.py` file is kept for backward compatibility but will be marked as deprecated and removed in a future version.

#### Available Data Methods:

**Core Methods:**

- `DashboardDataService.__init__(api_client, trade_manager, performance_tracker, strategy_manager, market_data_service, orderbook_analyzer, config)` - Initializes the data service with required components
- `update_all_data()` - Updates all dashboard data at once

**Performance Data:**

- `get_performance_data()` - Returns performance data including equity history, drawdown, metrics, and monthly returns

**Trade Data:**

- `get_trade_data()` - Returns trade data including completed trades, open positions, and pending orders

**System Data:**

- `get_system_status()` - Returns system status information including uptime and resource usage
- `get_data_freshness()` - Returns information about when each data category was last updated

**Market Data:**

- `get_orderbook_data(symbol, depth)` - Returns orderbook data for a specific symbol
- `get_market_data(symbol)` - Returns market data including price, candles, and indicators for a symbol

**Strategy Data:**

- `get_strategy_data(strategy_id)` - Returns strategy data including performance metrics and signals

#### Usage Example:

```python
from src.dashboard.services.data_service import DashboardDataService

# Initialize the data service
data_service = DashboardDataService(
    api_client=api_client,
    trade_manager=trade_manager,
    performance_tracker=performance_tracker,
    strategy_manager=strategy_manager
)

# Get performance data
performance_data = data_service.get_performance_data()

# Get data for a specific strategy
strategy_data = data_service.get_strategy_data(strategy_id="macd_crossover")
```

### Notification Service (`notification_service/`)

The notification service has been refactored into a modular package structure for better maintainability. It manages in-app notifications and alerts for the dashboard, providing a centralized system for displaying toast notifications and handling error messages.

#### Notification Service Package Structure:

- `notification_service/__init__.py` - Exports notification functions and components
- `notification_service/components.py` - UI components for notifications
- `notification_service/toast.py` - Toast notification creation and management
- `notification_service/callbacks.py` - Dash callbacks for notification handling
- `notification_service/error_handler.py` - Error handling utilities
- `notification_service/constants.py` - Constants and configuration

The original `notification_service.py` file is kept for backward compatibility but will be marked as deprecated and removed in a future version.

#### Available Notification Functions:

**UI Components:**

- `create_notification_components()` - Creates the notification store and container for the dashboard layout

**Toast Management:**

- `create_toast(message, notification_type, title, duration, dismissable, icon, color, id)` - Creates a toast notification
- `update_notifications(notifications)` - Updates the notification container with toast components
- `manage_notifications(n_clicks, current_notifications)` - Manages notifications when they are clicked (dismissed)
- `add_notification(notification_store, message, notification_type, title, duration, dismissable)` - Adds a notification to the store

**Error Handling:**

- `with_error_handling(callback_func)` - Decorator to add error handling to dashboard callbacks
- `log_error(error, context)` - Logs an error with context information

#### Usage Example:

```python
from src.dashboard.services.notification_service import create_notification_components, create_toast

# Create notification components for the layout
notification_store, notification_container = create_notification_components()

# Add these components to your dashboard layout
app.layout = html.Div([
    # Other components...
    notification_store,
    notification_container,
    # More components...
])

# Create a notification
notification = create_toast(
    message="Operation completed successfully",
    notification_type="success",
    title="Success",
    duration=5000
)

# Add error handling to callbacks
@with_error_handling
def my_callback(input_value):
    # Callback logic that might raise exceptions
    return processed_value
```

### Update Service (`update_service/`)

The update service has been refactored into a modular package structure for better maintainability. It manages real-time updates for dashboard components, handling the scheduling and delivery of data updates.

#### Update Service Package Structure:

- `update_service/__init__.py` - Exports the UpdateService class and update utility functions
- `update_service/service.py` - Core UpdateService class for managing updates
- `update_service/handlers.py` - Functions for registering and managing update handlers
- `update_service/utils.py` - Utility functions for the update service
- `update_service/config.py` - Configuration defaults and settings

The original `update_service.py` file is kept for backward compatibility but will be marked as deprecated and removed in a future version.

#### Available Update Service Functions:

**Core Service:**

- `UpdateService.__init__(data_service, update_intervals)` - Initializes the update service with a data service and update intervals
- `start()` - Starts the update service
- `stop()` - Stops the update service
- `get_status()` - Returns the current status of the update service

**Update Handlers:**

- `register_update_handler(data_type, callback)` - Registers a callback function to be called when data is updated
- `unregister_update_handler(data_type, callback)` - Unregisters a callback function
- `trigger_update(data_type, update_service)` - Triggers an update for a specific data type

**Utility Functions:**

- `get_last_update_time(data_type)` - Returns the time of the last update for a data type
- `get_next_update_time(data_type)` - Returns the time of the next scheduled update
- `is_update_due(data_type)` - Checks if an update is due for a data type

#### Usage Example:

```python
from src.dashboard.services.update_service import UpdateService, register_update_handler

# Initialize the update service
update_service = UpdateService(
    data_service=data_service,
    update_intervals={
        "performance": 30,  # 30 seconds
        "trades": 5,        # 5 seconds
        "orderbook": 1      # 1 second
    }
)

# Register a callback for trade updates
def on_trade_update():
    # Handle trade data update
    print("Trade data updated")

register_update_handler("trades", on_trade_update)

# Start the update service
update_service.start()
```

## Best Practices

1. **Always use services for shared functionality**: Instead of duplicating code in components, use or extend the appropriate service.

2. **Centralize visualization in chart_service package**: All chart creation functions should be defined in the chart service modules to ensure consistent styling and behavior.

3. **Apply the dashboard theme**: Always apply the dashboard theme to charts using `apply_chart_theme()` to maintain visual consistency.

4. **Handle errors gracefully**: All service methods should include appropriate error handling and provide sensible defaults when data is unavailable.

5. **Document new functionality**: When adding new service methods, include docstrings explaining parameters and return values.

6. **Follow the modular structure**: When adding new chart types or data retrieval methods, place them in the appropriate module based on their functionality.
