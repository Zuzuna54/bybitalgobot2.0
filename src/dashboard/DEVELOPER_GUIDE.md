# Dashboard Developer Guide

This guide provides detailed information for developers working with the Algorithmic Trading Dashboard. It covers key extension points, customization options, and best practices for development.

## Table of Contents

- [Callback Registration](#callback-registration)
- [Component Development](#component-development)
- [Data Service Integration](#data-service-integration)
- [Chart Creation](#chart-creation)
- [Layout Management](#layout-management)
- [Error Handling](#error-handling)
- [Performance Optimization](#performance-optimization)
- [Testing](#testing)

## Callback Registration

The dashboard uses a centralized callback registration system to manage interactive behavior. This system improves maintainability, error handling, and organization of callbacks.

### Using the Callback Registry

1. **Import the Decorator**:

   ```python
   from src.dashboard.router.callback_registry import callback_registrar
   ```

2. **Create a Registrar Function**:

   ```python
   @callback_registrar(name="my_component_callbacks")
   def register_my_component_callbacks(app, data_service=None, **kwargs):
       # Register component callbacks here
       @app.callback(
           output=Output("my-output-id", "children"),
           inputs=Input("my-input-id", "value")
       )
       def update_component(input_value):
           # Callback logic
           return f"Processed: {input_value}"
   ```

3. **Access Data Services**:
   ```python
   @callback_registrar(name="data_dependent_callbacks")
   def register_data_dependent_callbacks(app, data_service=None, **kwargs):
       @app.callback(
           output=Output("performance-display", "children"),
           inputs=Input("refresh-button", "n_clicks")
       )
       def update_performance(n_clicks):
           if data_service:
               performance_data = data_service.get_performance_data()
               return json.dumps(performance_data)
           return "Data service not available"
   ```

### Best Practices for Callbacks

1. **Name Registrars Clearly**: Use descriptive names for registrar functions that indicate what component or feature they control.

2. **Group Related Callbacks**: Keep related callbacks together in the same registrar function.

3. **Handle Data Service Availability**: Always check if the data service is available before using it.

4. **Use Typed Parameters**: Add type hints to callback parameters for better code understanding.

5. **Register in Central File**: Add your registrar to the central callbacks.py file:

   ```python
   from src.dashboard.components.my_component.callbacks import register_my_component_callbacks

   def initialize_callbacks(app, data_service=None):
       registry = CallbackRegistry(app, data_service)
       # ... other registrations
       registry.register("my_component", register_my_component_callbacks)
       # ... register and execute callbacks
   ```

## Component Development

The dashboard uses reusable components to build its interface. Follow these guidelines when creating new components:

### Component Structure

1. **Component Factory Pattern**:

   ```python
   def create_my_component(id_prefix="my-component"):
       """
       Create a new instance of my component.

       Args:
           id_prefix: Prefix for component IDs to ensure uniqueness

       Returns:
           A Dash HTML or Bootstrap component
       """
       return html.Div(
           id=f"{id_prefix}-container",
           children=[
               html.H3("My Component", id=f"{id_prefix}-title"),
               dcc.Graph(id=f"{id_prefix}-graph"),
               html.Button("Refresh", id=f"{id_prefix}-refresh-btn")
           ]
       )
   ```

2. **Component Directory Organization**:

   - Place related components in the same subdirectory
   - Use `__init__.py` to expose component factories
   - Separate presentation from callbacks

3. **ID Naming Conventions**:
   - Always use a prefix parameter to ensure uniqueness
   - Use consistent naming patterns: `{prefix}-{element-type}-{purpose}`
   - Include component type in ID: `-btn`, `-graph`, `-input`, etc.

### Composing Components

1. **Nested Components**:

   ```python
   def create_complex_component(id_prefix="complex"):
       return html.Div(
           id=f"{id_prefix}-container",
           children=[
               create_my_component(id_prefix=f"{id_prefix}-first"),
               create_my_component(id_prefix=f"{id_prefix}-second")
           ]
       )
   ```

2. **Component State Management**:
   - Use `dcc.Store` for component-specific state
   - Consider where state should be stored (component-local vs. global)

## Data Service Integration

The dashboard's data service is organized into domain-specific modules. When extending the data service:

### Adding New Data Functions

1. **Choose the Appropriate Module**:

   - Performance metrics → `performance_data.py`
   - Trade information → `trade_data.py`
   - Market data → `market_data.py`
   - Strategy configuration → `strategy_data.py`
   - System status → `system_data.py`
   - If none fit, consider creating a new domain-specific module

2. **Function Implementation Pattern**:

   ```python
   def _initialize_my_data(service):
       """Initialize with sample data for standalone mode."""
       service._my_data = {
           "key_metric": sample_value,
           "another_metric": another_sample
       }
       service._data_updated_at["my_data"] = datetime.now()

   def _update_my_data(service):
       """Update data from the trading system."""
       if service.standalone_mode:
           service._data_updated_at["my_data"] = datetime.now()
           return

       try:
           # Get data from appropriate system component
           if service.some_component:
               data = service.some_component.get_relevant_data()
               service._my_data = transform_data(data)
               service._data_updated_at["my_data"] = datetime.now()
       except Exception as e:
           logger.error(f"Error updating my data: {str(e)}")

   def get_my_data(self):
       """Get my data for the dashboard."""
       return self._my_data
   ```

3. **Register in base.py**:
   Add to DashboardDataService initialization:

   ```python
   # In __init__ method
   self._my_data = {}

   # In _data_updated_at initialization
   self._data_updated_at = {
       # ... existing keys
       "my_data": None,
   }

   # In _initialize_standalone_mode
   _initialize_my_data(self)

   # In update_all_data
   _update_my_data(self)

   # Export at the module level
   get_my_data = get_my_data
   ```

### Using Data Transformations

1. **Create Helper Functions**: Place data transformation logic in the appropriate utility modules:

   ```python
   # In utils/transformers.py
   def transform_my_data(raw_data):
       """Transform raw data into dashboard-friendly format."""
       # Transformation logic
       return transformed_data
   ```

2. **Apply Caching**: For expensive operations, use caching mechanisms:

   ```python
   from src.dashboard.utils.cache import cached_computation

   @cached_computation(ttl_seconds=300)  # 5 minute cache
   def get_expensive_calculation(self, param1, param2):
       # Complex calculation
       return result
   ```

## Chart Creation

The chart service is organized into domain-specific modules. When creating new visualizations:

### Adding New Charts

1. **Choose the Appropriate Module**:

   - Performance charts → `performance_charts.py`
   - Market data charts → `market_charts.py`
   - Orderbook visualizations → `orderbook_charts.py`
   - Strategy charts → `strategy_charts.py`
   - Reusable components → `component_renderers.py`

2. **Chart Function Pattern**:

   ```python
   def create_my_visualization(data, height=400, width=None, title="My Chart"):
       """
       Create a new visualization.

       Args:
           data: The data to visualize
           height: Chart height in pixels
           width: Chart width in pixels (None for responsive)
           title: Chart title

       Returns:
           Plotly figure object
       """
       # Import base styling
       from src.dashboard.services.chart_service.base import (
           get_chart_theme,
           apply_chart_style
       )

       # Initialize figure with theme
       theme = get_chart_theme()
       fig = go.Figure()

       # Add traces
       fig.add_trace(go.Scatter(
           x=data["x"],
           y=data["y"],
           mode="lines",
           name="My Data",
           line=dict(color=theme["colors"][0])
       ))

       # Apply consistent styling
       fig = apply_chart_style(
           fig,
           title=title,
           height=height,
           width=width
       )

       return fig
   ```

3. **Reuse Theme Settings**:

   - Import and use styling functions from `chart_service/base.py`
   - Follow the color scheme and layout patterns of existing charts

4. **Support Interactivity**:
   - Add hover templates for rich tooltips
   - Include appropriate zoom and selection tools
   - Consider mobile responsiveness

## Layout Management

The dashboard uses a tab-based layout system. When creating or modifying layouts:

### Creating Tab Layouts

1. **Layout Function Pattern**:

   ```python
   def create_my_tab_layout():
       """
       Create the layout for the My Feature tab.

       Returns:
           A Dash HTML structure for the tab content
       """
       return html.Div(
           id="my-tab-content",
           children=[
               html.H2("My Feature"),
               html.Div(
                   className="row",
                   children=[
                       dbc.Col(
                           create_my_component(id_prefix="my-tab"),
                           width=6
                       ),
                       dbc.Col(
                           create_another_component(id_prefix="my-tab"),
                           width=6
                       )
                   ]
               ),
               # Store for tab state
               dcc.Store(id="my-tab-state")
           ]
       )
   ```

2. **Add to Main Layout**:

   - Add a new tab in `layouts/main_layout.py`
   - Register in the tab navigation and content switching callbacks

3. **Register Tab Callbacks**:
   - Use the callback registration system for tab-specific callbacks
   - Handle tab visibility changes to optimize performance

### Responsive Design

1. **Use Bootstrap Grid System**:

   - Leverage dbc.Row and dbc.Col for responsive layouts
   - Consider different screen sizes in your design

2. **Conditional Display**:
   - Use CSS and callbacks to show/hide elements based on screen size
   - Consider simplified views for mobile devices

## Error Handling

Proper error handling is essential for a robust dashboard. Follow these guidelines:

### Component Error Handling

1. **Wrap Component Rendering**:

   ```python
   def create_safe_component(id_prefix="safe", fallback_content=None):
       try:
           return create_my_component(id_prefix)
       except Exception as e:
           logger.exception(f"Error creating component: {str(e)}")
           return html.Div(
               id=f"{id_prefix}-error",
               className="error-container",
               children=fallback_content or [
                   html.H4("Component Error"),
                   html.P(f"Error: {str(e)}")
               ]
           )
   ```

2. **Callback Error Handling**:

   ```python
   @app.callback(...)
   def update_with_error_handling(input_value):
       try:
           # Normal callback logic
           return processed_result
       except Exception as e:
           logger.exception(f"Callback error: {str(e)}")
           return html.Div(className="error-message", children=[
               html.P(f"Error processing data: {str(e)}")
           ])
   ```

3. **Data Service Error Handling**:
   - Always use try/except in data service functions
   - Log detailed error information
   - Return gracefully with fallback data when possible

## Performance Optimization

The dashboard includes a comprehensive performance optimization system to ensure responsiveness and efficiency, even with large datasets and complex visualizations. This section outlines the key optimization components and how to use them effectively.

### Enhanced Caching System

The enhanced caching system provides sophisticated caching mechanisms with memory awareness and multiple eviction policies.

1. **Using Enhanced Cache Decorators**:

   ```python
   from src.dashboard.utils.enhanced_cache import cache

   @cache(ttl=300, category="market_data", priority=8)
   def fetch_market_data(symbol):
       # Expensive operation to fetch market data
       return data
   ```

2. **Direct Cache Access**:

   ```python
   from src.dashboard.utils.enhanced_cache import get_cache

   # Get a cache instance
   cache = get_cache(
       name="strategy_cache",
       max_size_mb=100,
       cleanup_interval=300,
       default_ttl=600,
       eviction_policy="lru"  # Options: "lru", "lfu", "fifo", "priority"
   )

   # Store data
   cache.set(
       key="strategy_performance:MACD",
       value=performance_data,
       ttl=300,  # 5 minutes
       category="strategy",
       priority=7  # Higher = more important (1-10)
   )

   # Retrieve data
   data = cache.get("strategy_performance:MACD")
   ```

3. **Cache Categories and Invalidation**:

   ```python
   from src.dashboard.utils.enhanced_cache import invalidate_category

   # Invalidate all entries in a category
   invalidate_category("market_data")

   # Invalidate a specific function's cached results
   from src.dashboard.utils.enhanced_cache import invalidate_cache
   invalidate_cache(fetch_market_data, symbol="BTCUSD")
   ```

4. **Monitoring Cache Performance**:

   ```python
   # Get cache statistics
   stats = cache.get_stats()
   print(f"Cache hit ratio: {stats['hit_ratio'] * 100:.1f}%")
   print(f"Cache size: {stats['size_mb']:.2f} MB")
   print(f"Entries by category: {stats['categories']}")
   ```

### Memory Monitoring

The memory monitoring system tracks memory usage and provides alerts when thresholds are exceeded.

1. **Starting the Memory Monitor**:

   ```python
   from src.dashboard.utils.memory_monitor import start_memory_monitoring

   # Start with custom thresholds
   start_memory_monitoring(
       warning_threshold_mb=800,
       critical_threshold_mb=1500
   )
   ```

2. **Registering Alert Handlers**:

   ```python
   from src.dashboard.utils.memory_monitor import register_memory_alert_callback

   def handle_memory_alert(level, memory_info, message):
       if level == "critical":
           # Take emergency action like clearing caches
           from src.dashboard.utils.enhanced_cache import clear_all_caches
           clear_all_caches()
           logger.critical(f"Memory alert: {message}")
       elif level == "warning":
           # Take less drastic action
           logger.warning(f"Memory alert: {message}")

   register_memory_alert_callback(handle_memory_alert)
   ```

3. **Checking Memory Usage**:

   ```python
   from src.dashboard.utils.memory_monitor import get_current_memory_usage, get_memory_monitor

   # Get current usage
   usage = get_current_memory_usage()
   print(f"Current memory: {usage['rss'] / (1024*1024):.1f} MB")

   # Get memory trend
   monitor = get_memory_monitor()
   trend = monitor.get_memory_trend()
   print(f"Memory trend: {trend['trend']}")
   print(f"Growth rate: {trend['rate_mb_per_minute']:.2f} MB/min")
   ```

### Callback Optimization

The callback optimization system analyzes callback dependencies and provides tools for improving callback performance.

1. **Using Optimized Callbacks**:

   ```python
   from src.dashboard.router.callback_registry import CallbackRegistry

   # Get the registry (typically from your app instance)
   registry = app.callback_registry  # Or however you access it

   # Register an optimized callback
   @registry.register_optimized_callback(
       outputs=[Output('graph', 'figure')],
       inputs=[
           Input('dropdown', 'value'),
           Input('interval', 'n_intervals')
       ],
       throttle_ms=500,  # Limit to execute at most once every 500ms
       debounce_ms=200,  # Wait for inputs to settle
       callback_id="update_graph",  # Optional identifier
       priority=7  # Higher priority callbacks execute first
   )
   def update_graph(dropdown_value, n_intervals):
       # Callback logic
       return figure
   ```

2. **Analyzing Callback Performance**:

   ```python
   # Get performance statistics
   stats = registry.get_performance_stats()

   # Check execution times
   for name, time in stats['execution_times'].items():
       print(f"Callback {name}: {time:.3f}s")

   # Get dependency optimizer statistics
   optimizer_stats = stats['optimizer']
   print(f"Total callbacks: {optimizer_stats['total_callbacks']}")
   print(f"Avg execution time: {optimizer_stats['avg_execution_time_ms']} ms")
   ```

3. **Finding Optimization Opportunities**:

   ```python
   # From your app instance
   report = app.callback_registry.optimizer.generate_optimization_report()

   # Check recommendations
   for rec in report['recommendations']:
       print(f"- {rec['description']}")

   # Find expensive callbacks
   expensive = report['expensive_callbacks']
   for cb in expensive:
       print(f"Callback {cb['id']}: {cb['avg_time_ms']} ms, called {cb['call_count']} times")
   ```

### Clientside Callbacks

Clientside callbacks move callback execution to the browser for UI-only updates, reducing server load and improving responsiveness.

1. **Basic Clientside Callback Registration**:

   ```python
   from src.dashboard.utils.clientside_callbacks import register_clientside_callback

   # Register a custom clientside callback
   register_clientside_callback(
       app,
       outputs=Output("output-div", "children"),
       inputs=Input("input-field", "value"),
       clientside_function="""
           function(value) {
               return value ? "You entered: " + value : "Enter something...";
           }
       """
   )
   ```

2. **Common UI Patterns**:

   ```python
   from src.dashboard.utils.clientside_callbacks import (
       register_visibility_toggle,
       register_tab_content_visibility,
       register_dropdown_options_update
   )

   # Toggle component visibility
   register_visibility_toggle(
       app,
       container_id="collapsible-content",
       trigger_id="toggle-button",
       trigger_property="n_clicks",
       initial_state=False
   )

   # Handle tab content visibility
   register_tab_content_visibility(
       app,
       tab_content_prefix="tab-content-",
       tabs_id="main-tabs",
       tab_count=3
   )

   # Update dropdown options from store
   register_dropdown_options_update(
       app,
       dropdown_id="symbol-selector",
       data_store_id="available-symbols-store"
   )
   ```

3. **Status Indicators**:

   ```python
   from src.dashboard.utils.clientside_callbacks import register_status_indicator_update

   # Update status indicator based on status store
   register_status_indicator_update(
       app,
       indicator_id="connection-status",
       status_store_id="connection-status-store",
       class_mapping={
           "connected": "status-connected",
           "connecting": "status-connecting",
           "disconnected": "status-disconnected"
       }
   )
   ```

### Pattern Matching Callbacks

Pattern matching utilities help optimize multiple similar callbacks into a single callback with shared logic.

1. **Basic Pattern Matching**:

   ```python
   from src.dashboard.utils.pattern_matching import (
       register_pattern_callback,
       create_pattern_id
   )

   # Create component IDs with pattern
   symbol_display = html.Div([
       html.Div(id=create_pattern_id("price", "BTC"), children="Loading..."),
       html.Div(id=create_pattern_id("price", "ETH"), children="Loading..."),
       html.Div(id=create_pattern_id("price", "SOL"), children="Loading...")
   ])

   # Register a single callback that updates all price displays
   register_pattern_callback(
       app,
       outputs=Output({"type": "price", "index": MATCH}, "children"),
       inputs=Input("price-data-store", "data"),
       pattern_type="price",
       callback_func=lambda index, data: f"{index}: ${data.get(index, 0):.2f}"
   )
   ```

2. **Finding Matching Components**:

   ```python
   from src.dashboard.utils.pattern_matching import find_matching_components

   # Find all price display components
   component_ids = ["price-BTC", "price-ETH", "volume-BTC", "marketcap-BTC"]
   price_components = find_matching_components(r"^price-", component_ids)
   # Result: ["price-BTC", "price-ETH"]
   ```

3. **Multi-output Callbacks**:

   ```python
   from src.dashboard.utils.pattern_matching import create_multi_output_callback

   # Update multiple components with a single callback
   create_multi_output_callback(
       app,
       component_ids=["btc-price", "eth-price", "sol-price"],
       property_name="children",
       input_id="price-update-interval",
       input_property="n_intervals",
       callback_func=lambda n_intervals, *args: [
           f"BTC: ${fetch_price('BTC'):.2f}",
           f"ETH: ${fetch_price('ETH'):.2f}",
           f"SOL: ${fetch_price('SOL'):.2f}"
       ]
   )
   ```

4. **Using the DynamicCallbackManager**:

   ```python
   from src.dashboard.utils.pattern_matching import DynamicCallbackManager

   # Create a manager instance
   manager = DynamicCallbackManager(app)

   # Register dynamic callbacks
   manager.register_dynamic_callback(
       output_pattern="indicator",
       output_property="style",
       input_pattern="threshold",
       input_property="value",
       callback_func=lambda index, value, *args:
           {"color": "green" if value > 0 else "red"}
   )
   ```

### Best Practices for Performance

1. **Choose the Right Optimization**:

   - **Use Enhanced Caching** for expensive data operations
   - **Use Clientside Callbacks** for UI-only updates
   - **Use Pattern Matching** for similar components
   - **Use Throttling/Debouncing** for high-frequency updates

2. **Cache Optimization**:

   - Set appropriate TTLs based on data volatility
   - Use categories to organize cache entries
   - Set priorities based on importance
   - Monitor cache hit ratio (aim for >80%)

3. **Memory Management**:

   - Monitor memory usage trends
   - Register handlers for memory alerts
   - Implement cleanup procedures for large datasets
   - Use the memory monitor to detect leaks

4. **Callback Performance**:

   - Minimize callback dependencies
   - Use throttling for interval-based updates
   - Review optimization reports regularly
   - Combine related callbacks when possible

5. **Data Loading**:
   - Load data only when needed
   - Use progressive enhancement
   - Implement pagination for large datasets
   - Consider downsample/aggregate techniques for visualization

For more detailed information about performance optimizations, refer to the comprehensive guide in `docs/PERFORMANCE_OPTIMIZATION.md`.

## Testing

Ensure dashboard reliability with comprehensive testing:

### Component Testing

1. **Unit Testing Components**:

   - Test component factory functions
   - Verify correct ID generation
   - Check conditional rendering logic

2. **Callback Testing**:

   - Test callback logic with mock inputs
   - Verify output formatting
   - Test error handling paths

3. **Integration Testing**:
   - Test interactions between components
   - Verify data flow through the system
   - Test tab switching and state persistence

### Visual Testing

1. **Screenshot Testing**:

   - Capture screenshots of components for visual regression testing
   - Compare across different browser sizes

2. **Accessibility Testing**:
   - Verify color contrast
   - Test keyboard navigation
   - Check screen reader compatibility

---

## Advanced Topics

### Custom Theming

The dashboard supports customization of its visual appearance:

1. **Theme Configuration**:

   - Modify theme settings in `chart_service/base.py`
   - Create alternate themes for different use cases

2. **CSS Customization**:
   - Add custom CSS to `assets/style.css`
   - Use consistent class naming conventions

### API Integration

To integrate with additional external systems:

1. **Create an API Client Module**:

   - Implement in `services/api_client/`
   - Follow consistent error handling patterns

2. **Register with Data Service**:
   - Add as a dependency to DashboardDataService
   - Create appropriate data access functions

### State Management

For complex dashboard state:

1. **Global State Stores**:

   - Use dcc.Store with storage_type="session" for persistent state
   - Define clear state schema and update patterns

2. **URL-based State**:
   - Consider encoding important state in URL parameters
   - Implement state restoration on page load

---

This guide covers the main aspects of dashboard development. For further assistance or questions, refer to the codebase documentation or contact the dashboard development team.
