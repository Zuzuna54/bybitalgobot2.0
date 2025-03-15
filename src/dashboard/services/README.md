# Dashboard Services

This directory contains service classes that handle complex functionality for the dashboard. Services act as an abstraction layer between the data sources and the UI components, providing centralized and reusable functionality.

## Available Services

### Chart Service (`chart_service.py`)

Provides centralized visualization functionality for all dashboard components. This service handles the creation of charts and graphs using Plotly, ensuring consistent styling and behavior across the application.

#### Available Chart Functions:

**General Charts:**

- `apply_chart_theme(fig, title)` - Applies the dashboard's theme to any Plotly figure
- `create_empty_chart(message)` - Creates an empty chart with a message displayed
- `create_empty_sparkline()` - Creates an empty sparkline chart

**Performance Charts:**

- `create_equity_curve(performance_data)` - Creates an equity curve visualization
- `create_return_sparkline(performance_data)` - Creates a small sparkline for returns
- `create_drawdown_chart(performance_data)` - Creates a drawdown visualization
- `create_monthly_returns_heatmap(returns_data)` - Creates a heatmap of monthly returns
- `create_daily_pnl_chart(daily_pnl)` - Creates a daily profit/loss bar chart
- `create_profit_distribution_chart(completed_trades)` - Creates a histogram of trade profit distribution
- `create_performance_dashboard(equity_history, completed_trades, daily_pnl, returns_data)` - Creates a complete set of performance dashboard charts

**Trading Charts:**

- `create_pnl_by_symbol_graph(trade_data)` - Creates a P&L chart broken down by symbol
- `create_win_loss_by_strategy_graph(trade_data)` - Creates a win/loss chart by strategy
- `create_trade_distribution_chart(trade_data)` - Creates a chart showing trade distribution

**Market Charts:**

- `create_candlestick_chart(ohlcv_data)` - Creates a candlestick chart for market data
- `create_volume_profile(ohlcv_data)` - Creates a volume profile visualization

**Orderbook Charts:**

- `create_orderbook_depth_chart(orderbook, sr_levels)` - Creates an orderbook depth chart
- `create_orderbook_heatmap(orderbook, levels)` - Creates a heatmap visualization of the orderbook
- `create_orderbook_imbalance_chart(orderbook, depth_levels)` - Creates a bid/ask imbalance visualization
- `create_liquidity_profile_chart(orderbook, price_range_pct)` - Creates a liquidity profile visualization

**Strategy Charts:**

- `create_strategy_performance_graph(strategy_data)` - Creates a strategy performance visualization
- `create_strategy_comparison_chart(strategy_data)` - Creates a chart comparing multiple strategies

#### Usage Example:

```python
from src.dashboard.services.chart_service import create_equity_curve, apply_chart_theme

# Create a chart
fig = create_equity_curve(performance_data)

# Customize the chart (if needed)
fig.update_layout(height=600)

# Apply the dashboard theme
fig = apply_chart_theme(fig, "Custom Equity Curve")
```

#### Performance Dashboard Example:

```python
from src.dashboard.services.chart_service import create_performance_dashboard

# Generate all performance charts at once
charts = create_performance_dashboard(
    equity_history=equity_data,
    completed_trades=trades_data,
    daily_pnl=pnl_data,
    returns_data=returns_df,
    time_range="3m"
)

# Access individual charts
equity_fig = charts["equity_curve"]
drawdown_fig = charts["drawdown"]
profit_dist_fig = charts["profit_distribution"]
```

### Data Service (`data_service.py`)

Handles data retrieval and processing for dashboard components, providing a centralized interface for accessing trading system data.

### Notification Service (`notification_service.py`)

Manages in-app notifications and alerts for the dashboard.

### Update Service (`update_service.py`)

Manages real-time updates for dashboard components, handling the scheduling and delivery of data updates.

## Best Practices

1. **Always use services for shared functionality**: Instead of duplicating code in components, use or extend the appropriate service.

2. **Centralize visualization in chart_service.py**: All chart creation functions should be defined in the chart service to ensure consistent styling and behavior.

3. **Apply the dashboard theme**: Always apply the dashboard theme to charts using `apply_chart_theme()` to maintain visual consistency.

4. **Handle errors gracefully**: All service methods should include appropriate error handling and provide sensible defaults when data is unavailable.

5. **Document new functionality**: When adding new service methods, include docstrings explaining parameters and return values.
