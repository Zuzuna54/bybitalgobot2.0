# Dashboard Components Guide

This guide provides information about the dashboard components, their purposes, and how to use them effectively in layouts.

## Component Categories

The dashboard contains several categories of components:

1. **Performance Components**: Display trading performance metrics and visualizations
2. **Trading Components**: Show active trades and order history
3. **Orderbook Components**: Visualize market depth and orderbook data
4. **Market Components**: Display market data and price movements
5. **Strategy Components**: Display strategy configuration and performance

## Performance Components

Performance components provide visualizations and metrics for trading performance analysis.

### Metrics Card

**Purpose**: Display key performance metrics with sparklines.

**Usage**:

```python
from src.dashboard.components.performance.metrics_card import create_metrics_card

layout = html.Div([
    create_metrics_card(
        id_prefix="performance",
        title="Performance Metrics",
        metrics=[
            {"name": "Win Rate", "value": "65%", "trend": "up"},
            {"name": "Sharpe", "value": "1.85", "trend": "up"},
            {"name": "Max DD", "value": "-12.5%", "trend": "neutral"}
        ]
    )
])
```

### Equity Curve

**Purpose**: Display account equity over time with drawdown visualization.

**Usage**:

```python
from src.dashboard.components.performance.charts import create_equity_chart

layout = html.Div([
    create_equity_chart(
        id_prefix="performance",
        height=400
    )
])
```

### Monthly Returns

**Purpose**: Display returns by month and year in a heatmap format.

**Usage**:

```python
from src.dashboard.components.performance.charts import create_monthly_returns_chart

layout = html.Div([
    create_monthly_returns_chart(
        id_prefix="performance",
        height=350
    )
])
```

## Trading Components

Trading components provide interfaces for monitoring and managing trades.

### Position Table

**Purpose**: Display active trading positions with key metrics.

**Usage**:

```python
from src.dashboard.components.trading.position_table import create_position_table

layout = html.Div([
    create_position_table(id_prefix="trading")
])
```

### Order History

**Purpose**: Display historical orders with filtering capabilities.

**Usage**:

```python
from src.dashboard.components.trading.order_history import create_order_history

layout = html.Div([
    create_order_history(
        id_prefix="trading",
        page_size=10,
        include_filters=True
    )
])
```

### Trade Controls

**Purpose**: Provide interface for manual trade execution.

**Usage**:

```python
from src.dashboard.components.trading.controls import create_trade_controls

layout = html.Div([
    create_trade_controls(
        id_prefix="trading",
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    )
])
```

## Orderbook Components

Orderbook components visualize order book data and market liquidity.

### Depth Chart

**Purpose**: Visualize order book depth for a trading pair.

**Usage**:

```python
from src.dashboard.components.orderbook.depth_chart import create_depth_chart

layout = html.Div([
    create_depth_chart(
        id_prefix="orderbook",
        height=400,
        default_symbol="BTCUSDT"
    )
])
```

### Market Trades

**Purpose**: Display recent market trades with volume visualization.

**Usage**:

```python
from src.dashboard.components.orderbook.market_trades import create_market_trades

layout = html.Div([
    create_market_trades(
        id_prefix="orderbook",
        max_trades=20,
        default_symbol="BTCUSDT"
    )
])
```

### Orderbook Imbalance

**Purpose**: Show order book imbalance metrics for market analysis.

**Usage**:

```python
from src.dashboard.components.orderbook.imbalance import create_imbalance_indicator

layout = html.Div([
    create_imbalance_indicator(
        id_prefix="orderbook",
        default_symbol="BTCUSDT"
    )
])
```

## Market Components

Market components display trading pair market data and price information.

### Price Chart

**Purpose**: Interactive price chart with technical indicators.

**Usage**:

```python
from src.dashboard.components.market.price_chart import create_price_chart

layout = html.Div([
    create_price_chart(
        id_prefix="market",
        height=500,
        default_symbol="BTCUSDT",
        default_timeframe="1h"
    )
])
```

### Market Overview

**Purpose**: Display market overview with multiple pairs.

**Usage**:

```python
from src.dashboard.components.market.market_overview import create_market_overview

layout = html.Div([
    create_market_overview(
        id_prefix="market",
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT"]
    )
])
```

### Volume Profile

**Purpose**: Display volume profile for price analysis.

**Usage**:

```python
from src.dashboard.components.market.volume_profile import create_volume_profile

layout = html.Div([
    create_volume_profile(
        id_prefix="market",
        default_symbol="BTCUSDT",
        height=350
    )
])
```

## Strategy Components

Strategy components provide interfaces for strategy configuration and monitoring.

### Strategy Card

**Purpose**: Display strategy information and key metrics.

**Usage**:

```python
from src.dashboard.components.strategy.strategy_card import create_strategy_card

layout = html.Div([
    create_strategy_card(
        id_prefix="strategy",
        strategy_id="momentum_strategy"
    )
])
```

### Strategy Panel

**Purpose**: Comprehensive strategy monitoring panel.

**Usage**:

```python
from src.dashboard.components.strategy.panel import create_strategy_panel

layout = html.Div([
    create_strategy_panel(id_prefix="strategy")
])
```

### Strategy Configuration

**Purpose**: Interface for configuring trading strategies.

**Usage**:

```python
from src.dashboard.components.strategy.strategy_config import create_strategy_config

layout = html.Div([
    create_strategy_config(
        id_prefix="strategy",
        strategy_id="momentum_strategy"
    )
])
```

## Utility Components

The dashboard includes several utility components for common UI patterns.

### Loading Indicator

**Purpose**: Provide visual feedback during data loading.

**Usage**:

```python
from src.dashboard.components.loading import create_loading_indicator

layout = html.Div([
    create_loading_indicator(
        id_prefix="my-component",
        children=[
            # Component that requires loading
            html.Div(id="my-content")
        ]
    )
])
```

### Error Display

**Purpose**: Display error information with consistent styling.

**Usage**:

```python
from src.dashboard.components.error_display import create_error_display

# In a callback
@app.callback(...)
def update_component(input_value):
    try:
        # Process data
        return processed_data
    except Exception as e:
        return create_error_display(
            error=str(e),
            title="Data Processing Error",
            suggestion="Refresh the page or try again later."
        )
```

### Notification System

**Purpose**: Display system notifications and alerts.

**Usage**:

```python
from src.dashboard.services.notification_service import create_notification_components, show_notification

# In layout
layout = html.Div([
    create_notification_components(id_prefix="app"),
    # Other components
])

# In a callback
@app.callback(...)
def process_action(n_clicks):
    # Process action
    show_notification(
        notification_id="app",
        message="Operation completed successfully",
        type="success"
    )
    return "Done"
```

## Component Composition

Components can be composed to create complex layouts:

### Tab-Based Layout

```python
import dash_bootstrap_components as dbc
from dash import html, dcc

from src.dashboard.components.performance.metrics_card import create_metrics_card
from src.dashboard.components.performance.charts import create_equity_chart
from src.dashboard.components.trading.position_table import create_position_table

def create_dashboard_tab():
    return html.Div([
        dbc.Row([
            dbc.Col([
                create_metrics_card(id_prefix="dashboard")
            ], width=12)
        ]),
        dbc.Row([
            dbc.Col([
                create_equity_chart(id_prefix="dashboard")
            ], width=8),
            dbc.Col([
                create_position_table(id_prefix="dashboard")
            ], width=4)
        ])
    ])
```

### Card-Based Layout

```python
import dash_bootstrap_components as dbc
from dash import html

from src.dashboard.components.performance.charts import create_equity_chart
from src.dashboard.components.market.price_chart import create_price_chart

def create_card_component(title, component):
    return dbc.Card([
        dbc.CardHeader(title),
        dbc.CardBody([
            component
        ])
    ], className="mb-4")

def create_card_layout():
    return html.Div([
        dbc.Row([
            dbc.Col([
                create_card_component(
                    "Equity Curve",
                    create_equity_chart(id_prefix="card-layout")
                )
            ], width=6),
            dbc.Col([
                create_card_component(
                    "Price Chart",
                    create_price_chart(id_prefix="card-layout")
                )
            ], width=6)
        ])
    ])
```

## Best Practices

When working with dashboard components:

1. **Use Consistent ID Prefixes**: Always use the id_prefix parameter to ensure unique component IDs.

2. **Error Handling**: Wrap components with proper error handling to prevent dashboard crashes.

3. **Responsive Design**: Consider different screen sizes when composing layouts.

4. **Loading States**: Add loading indicators for components that fetch data asynchronously.

5. **State Management**: Use dcc.Store for component-specific state management.

6. **Documentation**: Document component parameters and return values.

7. **Styling**: Use consistent styling through Bootstrap classes and the dashboard theme.
