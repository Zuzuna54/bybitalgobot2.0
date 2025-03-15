"""
Dashboard Utilities Package

This package provides various utility functions and classes for the dashboard.
"""

# Import functions from converters module
from .converters import (
    to_float, to_int, to_bool, to_datetime, to_list, to_dict, to_dataframe,
    parse_timeframe, convert_df_types, json_serialize
)

# Import functions from config_manager module
from .config_manager import get_config_manager

# Import functions from logger module
from .logger import get_logger, log_exception, measure_execution_time

# Import functions from cache module
from .cache import get_cache, cached, invalidate_cache, precache

# Import functions from helper module
from .helper import (
    generate_id, generate_uuid, timestamp_ms, timestamp_ns,
    get_date_range, get_current_time_as_string, get_system_info,
    safe_divide, moving_average, calculate_percent_change,
    truncate_string, get_nested_dict_value, set_nested_dict_value,
    deep_merge, load_json_file, save_json_file,
    encode_base64, decode_base64, generate_hash,
    list_to_chunks, get_duplicate_items,
    url_encode, url_decode, build_query_string, parse_query_string,
    to_snake_case, to_camel_case, to_title_case,
    find_files, safe_filename, trim_dataframe,
    convert_size_to_bytes, format_bytes, wait_for_condition
)

# Version information
__version__ = '1.0.0' 