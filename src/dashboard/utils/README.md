# Dashboard Utilities

This directory contains utility modules that provide common functionality used throughout the dashboard. These utilities are designed to be reusable, efficient, and consistent.

## Available Utility Modules

### `time_utils.py`

Centralized date and time utility functions for consistent time handling across the dashboard.

- `timestamp_ms()`: Get current timestamp in milliseconds
- `timestamp_ns()`: Get current timestamp in nanoseconds
- `get_current_time()`: Get current time as datetime object
- `get_current_time_as_string(fmt)`: Get current time as formatted string
- `parse_timestamp(timestamp)`: Parse a timestamp into a datetime object
- `format_timestamp(timestamp, format_str)`: Format a timestamp as string
- `format_time_ago(timestamp)`: Format a timestamp as a human-readable time ago string
- `get_date_range(start_date, end_date, as_string, fmt)`: Get a list of dates between start and end dates
- `filter_data_by_time_range(data, time_range, date_column)`: Filter DataFrame by time range
- `format_duration(seconds)`: Format a duration in seconds as a human-readable string
- `is_update_due(last_update, interval_seconds)`: Check if an update is due based on interval
- `get_next_update_time(last_update, interval_seconds)`: Calculate the next update time

### `formatter.py`

Data formatting utilities for consistent display of values across the dashboard.

- `format_currency(value, precision, currency)`: Format a number as currency
- `format_percentage(value, precision, include_sign)`: Format a number as a percentage
- `format_trade_size(size, asset, precision)`: Format a trade size with appropriate precision
- `format_number_compact(value, precision)`: Format a number in a compact way (e.g., 1.5K, 2.3M)
- `format_symbol(symbol)`: Format a trading symbol for display
- `format_dataframe_to_dict(df)`: Convert a pandas DataFrame to a list of dictionaries for JSON serialization

### `helper.py`

General utility functions that don't fit into more specific categories.

- `generate_id(prefix, length)`: Generate a random ID with an optional prefix
- `generate_uuid()`: Generate a UUID string
- `get_system_info()`: Get system information
- `safe_divide(numerator, denominator, default)`: Safely divide two numbers
- `moving_average(data, window)`: Calculate the moving average of a list of numbers
- `calculate_percent_change(old_value, new_value, precision)`: Calculate the percentage change
- `truncate_string(text, max_length, suffix)`: Truncate a string to a maximum length
- `get_nested_dict_value(data, path, default, separator)`: Get a value from a nested dictionary
- `set_nested_dict_value(data, path, value, separator)`: Set a value in a nested dictionary
- `deep_merge(dict1, dict2)`: Deep merge two dictionaries
- `load_json_file(file_path)`: Load and parse a JSON file
- `save_json_file(data, file_path, indent)`: Save data to a JSON file
- `encode_base64(data)`: Encode data as base64
- `decode_base64(data)`: Decode base64 data
- `generate_hash(data, algorithm)`: Generate a hash of the input data
- `list_to_chunks(items, chunk_size)`: Split a list into chunks of a specified size
- `get_duplicate_items(items)`: Find duplicate items in a list
- `url_encode(text)`: URL-encode a string
- `url_decode(text)`: URL-decode a string
- `format_bytes(size_bytes)`: Format a byte size as a human-readable string

### `cache.py`

Caching utilities for improving performance by storing and reusing computed results.

- `cached(ttl_seconds, key_prefix, key_func)`: Decorator for caching function results
- `CacheManager`: Class for managing cache operations with methods:
  - `get(key)`: Get a value from the cache
  - `set(key, value, ttl_seconds)`: Set a value in the cache
  - `delete(key)`: Delete a value from the cache
  - `clear()`: Clear all values from the cache
  - `get_or_compute(key, compute_func, ttl_seconds)`: Get a cached value or compute it

### `transformers.py`

Data transformation utilities for standardizing data formats across the dashboard.

- `DataTransformer`: Class with methods for transforming various data types:
  - `transform_equity_data(equity_data)`: Transform equity data for visualization
  - `transform_trade_data(trades_data)`: Transform trade data for display
  - `transform_orderbook_data(orderbook_data)`: Transform orderbook data
  - `transform_strategy_data(strategy_data)`: Transform strategy data
  - `transform_market_data(market_data)`: Transform market data

### `converters.py`

Type conversion utilities for safely converting between data types.

- `to_float(value, default)`: Convert a value to float
- `to_int(value, default)`: Convert a value to integer
- `to_bool(value, default)`: Convert a value to boolean
- `to_datetime(value, default)`: Convert a value to datetime
- `to_list(value, default)`: Convert a value to list
- `to_dict(value, default)`: Convert a value to dictionary

### `logger.py`

Logging utilities for consistent logging across the dashboard.

- `get_logger(name)`: Get a configured logger instance
- `log_exception(e, context)`: Log an exception with additional context
- `measure_execution_time(func)`: Decorator to measure and log function execution time

## Best Practices

1. **Use Centralized Utilities**: Always use these centralized utility functions instead of implementing similar functionality in component files.

2. **Avoid Duplication**: If you need functionality that's similar to an existing utility but with slight differences, consider extending the existing utility rather than creating a new one.

3. **Maintain Consistency**: Follow the established patterns when adding new utility functions.

4. **Document New Utilities**: Add proper docstrings and update this README when adding new utility functions.

5. **Error Handling**: Ensure all utility functions have appropriate error handling and fallbacks.

6. **Performance Considerations**: Be mindful of performance implications, especially for functions that may be called frequently.

## Adding New Utilities

When adding new utility functions, follow these guidelines:

1. Determine the appropriate module based on the function's purpose.
2. Add comprehensive docstrings with parameter descriptions and return values.
3. Implement proper error handling and fallbacks.
4. Add type hints for better IDE support and code clarity.
5. Update this README with the new function's description.
6. Consider adding unit tests for the new functionality.
