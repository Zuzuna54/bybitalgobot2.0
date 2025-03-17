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

Keep the dashboard responsive and efficient with these practices:

### Callback Optimization

1. **Minimize Trigger Frequency**:

   - Use appropriate interval times for auto-refreshing components
   - Consider using debounce patterns for input-driven updates

2. **Use Pattern-Matching Callbacks**:

   - For similar components, use pattern-matching callbacks instead of individual ones
   - Helps when dynamically generating components

3. **Implement Clientside Callbacks**:
   ```python
   app.clientside_callback(
       """
       function(value) {
           return value.toUpperCase();
       }
       """,
       Output("client-output", "children"),
       Input("client-input", "value")
   )
   ```

### Data Loading Strategies

1. **Implement Lazy Loading**:

   - Load data only when tabs become active
   - Use dcc.Loading for visual feedback

2. **Progressive Enhancement**:
   - Show simple views first, then enhance with detailed data
   - Consider skeleton loading patterns

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
