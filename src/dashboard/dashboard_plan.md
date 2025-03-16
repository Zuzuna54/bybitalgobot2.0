# Dashboard Implementation Plan

## Overview

This document outlines a comprehensive plan for improving the algorithmic trading dashboard's architecture, efficiency, and maintainability. Based on a thorough analysis of the codebase, we've identified several areas for optimization and restructuring, with callback registration consolidation and code duplication elimination as primary concerns.

## Implementation Progress

### Completed Items:

- ✅ Consolidated callback registrations in `router/callbacks.py`
- ✅ Created a single entry point for callback registration with `initialize_callbacks`
- ✅ Removed redundant callback registration code from `app.py`
- ✅ Fixed import path inconsistencies in `router/callbacks.py`
- ✅ Implemented standard chart theme configuration in `chart_service.py`
- ✅ Created an `apply_chart_theme` function for consistent styling
- ✅ Enhanced caching with timestamp-based invalidation through `CacheManager`
- ✅ Implemented partial data updates through utility functions
- ✅ Added proper error handling in system callbacks
- ✅ Created a standardized `DataTransformer` utility for consistent data transformations
- ✅ Implemented reusable transformation methods for equity, trade, orderbook, and strategy data
- ✅ Applied caching to expensive data transformations
- ✅ Improved error handling and fallbacks in data transformations
- ✅ Standardized data transformation logic in the data service
- ✅ Added comprehensive market data transformation and retrieval
- ✅ Implemented the `standalone_mode` property in `DashboardDataService` to simplify mode checking
- ✅ Moved chart creation functions from `performance_layout.py` to `chart_service.py`
- ✅ Added error handling to chart-related callbacks
- ✅ Added `create_candlestick_chart` function to `chart_service.py`
- ✅ Added `create_orderbook_depth_chart` function to `chart_service.py`
- ✅ Updated market and orderbook callbacks to use the centralized chart service
- ✅ Moved strategy chart functions (`create_strategy_performance_graph`, etc.) to `chart_service.py`
- ✅ Updated strategy callbacks to use the centralized chart service
- ✅ Moved trading chart functions (`create_pnl_by_symbol_graph`, `create_win_loss_by_strategy_graph`) to `chart_service.py`
- ✅ Updated trading callbacks to use the centralized chart service
- ✅ Created centralized time utility module (`time_utils.py`) with comprehensive date/time functions
- ✅ Removed duplicate time formatting functions from `formatter.py`
- ✅ Updated `update_service.py` to use centralized time utility functions
- ✅ Updated `helper.py` to use centralized time utility functions
- ✅ Updated `logger.py` to use centralized time utility functions
- ✅ Updated `transformers.py` to use centralized time utility functions
- ✅ Created a comprehensive utility catalog in `utils/README.md`
- ✅ Created Plotly-based orderbook visualization functions in `chart_service.py`:
  - ✅ Added `create_orderbook_heatmap` for visualizing order book as a heatmap
  - ✅ Added `create_orderbook_imbalance_chart` for visualizing bid/ask imbalance
  - ✅ Added `create_liquidity_profile_chart` for visualizing liquidity distribution
- ✅ Updated orderbook panel and callbacks to use new centralized visualization functions:
  - ✅ Updated `components/orderbook/callbacks.py` to include new chart outputs
  - ✅ Updated `components/orderbook/panel.py` to display the new visualizations
  - ✅ Added error handling for new visualization components
- ✅ Created performance visualization functions in `chart_service.py`:
  - ✅ Added `create_daily_pnl_chart` for visualizing daily profit/loss data
  - ✅ Added `create_profit_distribution_chart` for visualizing trade profit distribution
  - ✅ Added `create_monthly_returns_heatmap` for visualizing monthly performance
  - ✅ Added `create_performance_dashboard` for generating a complete set of performance dashboard charts
- ✅ Added `allow_duplicate=True` for notification callbacks
- ✅ Implemented proper locking for shared data
- ✅ Implemented Plotly-based orderbook visualization functions to replace matplotlib-based ones in `src/trade_execution/orderbook/visualization.py`:
  - ✅ Added `create_orderbook_depth_chart` to replace `plot_order_book_depth`
  - ✅ Added `create_orderbook_heatmap` to replace `create_order_book_heatmap`
  - ✅ Added `create_orderbook_imbalance_chart` to replace `visualize_imbalance`
  - ✅ Added `create_liquidity_profile_chart` to replace `plot_liquidity_profile`
- ✅ Implemented Plotly-based backtesting visualization functions to replace matplotlib-based ones in `src/backtesting/backtest/results_processing.py`:
  - ✅ Added `create_backtest_equity_curve` to replace `generate_equity_curve_chart`
  - ✅ Added `create_backtest_trade_distribution` to replace `generate_trade_distribution`
  - ✅ Added `create_backtest_strategy_comparison` to replace `generate_strategy_comparison`
  - ✅ Added `create_backtest_monthly_returns` to replace `generate_monthly_returns`

### In Progress:

- 🔄 Optimizing callback dependencies to reduce unnecessary updates
- 🔄 Implementing a data service layer for consistent data access across components
- 🔄 Creating a comprehensive testing framework for dashboard components

### Not Started

- ⏱️ Convert UI updates to client-side callbacks where appropriate
- ⏱️ Implement a performance monitoring system for dashboard components
- ⏱️ Create a configuration management system for dashboard settings
- ⏱️ Develop a user preferences system for customizing dashboard layout and appearance
- ⏱️ Implement a notification system for important events and alerts

## Architecture

### Component Structure

The dashboard is organized into the following main components:

1. **Layouts**: High-level page layouts that combine multiple components

   - Main Layout (`main_layout.py`)
   - Market Layout (`market_layout.py`)
   - Trading Layout (`trading_layout.py`)
   - Strategy Layout (`strategy_layout.py`)
   - Settings Layout (`settings_layout.py`)

2. **Components**: Reusable UI elements

   - Orderbook Panel (`components/orderbook/`)
   - Trading Panel (`components/trading/`)
   - Strategy Panel (`components/strategy/`)
   - Market Panel (`components/market/`)
   - Performance Panel (`components/performance/`)

3. **Services**: Data processing and business logic

   - Chart Service (`services/chart_service.py`)
   - Data Service (`services/data_service.py`)
   - Notification Service (`services/notification_service.py`)

4. **Router**: Callback management and routing

   - Callback Registry (`router/callback_registry.py`)
   - Callbacks (`router/callbacks.py`)
   - URL Routing (`router/url_routing.py`)

5. **Utils**: Utility functions and helpers
   - Time Utilities (`utils/time_utils.py`)
   - Data Transformers (`utils/transformers.py`)
   - Cache Manager (`utils/cache_manager.py`)

### Data Flow

1. **Data Sources** → **Data Service** → **Components** → **UI**

   - External data sources provide raw data
   - Data service processes and transforms data
   - Components consume data and create visualizations
   - UI displays visualizations and accepts user input

2. **User Input** → **Callbacks** → **Data Service** → **UI Update**
   - User interacts with UI elements
   - Callbacks process user input
   - Data service updates data based on user input
   - UI is updated to reflect changes

### Callback Registration System

The new callback registration system uses a decorator-based approach:

```python
@callback_registrar(name="component_name")
def register_component_callbacks(app: dash.Dash, data_service: Optional[Any] = None, **kwargs) -> None:
    # Register callbacks for the component
    pass
```

The `CallbackRegistry` class manages all callback registrars and provides a centralized interface for registering and executing callbacks. This approach:

1. Standardizes callback registration parameters
2. Provides a consistent interface for all components
3. Supports dependency injection for data services
4. Improves error handling and logging
5. Makes it easier to manage callback dependencies

## Next Steps

1. **Data Service Layer**

   - Create a comprehensive data service interface
   - Implement data providers for different data sources
   - Standardize data access patterns across components

2. **Client-Side Callbacks**

   - Identify UI updates that can be moved to client-side
   - Implement client-side callback patterns
   - Reduce server load for purely presentational changes

3. **Performance Monitoring**

   - Add timing metrics for callback execution
   - Create a debug panel for monitoring performance
   - Implement performance optimization strategies

4. **Testing Framework**

   - Create unit tests for services and utilities
   - Implement component testing
   - Set up integration testing for the dashboard

5. **User Preferences**
   - Implement user settings storage
   - Create UI for customizing dashboard appearance
   - Support layout customization
