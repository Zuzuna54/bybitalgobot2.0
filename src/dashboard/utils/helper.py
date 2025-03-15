"""
Dashboard Helper Module

This module provides various utility functions for the dashboard that don't fit
into more specific utility categories.
"""

import os
import sys
import platform
import json
import re
import random
import string
from typing import Dict, Any, Optional, List, Union, Tuple, Callable
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import time
import uuid
import base64
import hashlib
import urllib.parse
import socket
import inspect
import importlib.util
from pathlib import Path

from .logger import get_logger
from .converters import to_float, to_int, to_bool, to_datetime, to_list, to_dict

logger = get_logger("helper")


def generate_id(prefix: str = "", length: int = 8) -> str:
    """
    Generate a random ID with an optional prefix.
    
    Args:
        prefix: Optional prefix for the ID
        length: Length of the random part of the ID
        
    Returns:
        Generated ID
    """
    random_part = ''.join(random.choices(
        string.ascii_lowercase + string.digits, k=length))
    
    if prefix:
        return f"{prefix}-{random_part}"
    
    return random_part


def generate_uuid() -> str:
    """
    Generate a UUID string.
    
    Returns:
        UUID string
    """
    return str(uuid.uuid4())


def timestamp_ms() -> int:
    """
    Get the current timestamp in milliseconds.
    
    Returns:
        Current timestamp in milliseconds
    """
    return int(time.time() * 1000)


def timestamp_ns() -> int:
    """
    Get the current timestamp in nanoseconds.
    
    Returns:
        Current timestamp in nanoseconds
    """
    return time.time_ns()


def get_date_range(start_date: Union[str, datetime, date],
                  end_date: Union[str, datetime, date],
                  as_string: bool = False,
                  fmt: str = "%Y-%m-%d") -> List[Union[datetime, str]]:
    """
    Get a list of dates between start_date and end_date (inclusive).
    
    Args:
        start_date: Start date
        end_date: End date
        as_string: Whether to return dates as strings
        fmt: Date format string if as_string is True
        
    Returns:
        List of dates
    """
    # Convert to datetime if needed
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, fmt)
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, fmt)
    
    # Convert date to datetime if needed
    if isinstance(start_date, date) and not isinstance(start_date, datetime):
        start_date = datetime.combine(start_date, datetime.min.time())
    if isinstance(end_date, date) and not isinstance(end_date, datetime):
        end_date = datetime.combine(end_date, datetime.min.time())
    
    # Calculate date range
    delta = end_date - start_date
    dates = [start_date + timedelta(days=i) for i in range(delta.days + 1)]
    
    # Convert to strings if requested
    if as_string:
        return [date.strftime(fmt) for date in dates]
    
    return dates


def get_current_time_as_string(fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Get the current time as a formatted string.
    
    Args:
        fmt: Date format string
        
    Returns:
        Formatted current time
    """
    return datetime.now().strftime(fmt)


def get_system_info() -> Dict[str, Any]:
    """
    Get system information.
    
    Returns:
        Dictionary of system information
    """
    try:
        info = {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "hostname": socket.gethostname(),
            "python_version": platform.python_version(),
            "timezone": time.tzname,
            "time": get_current_time_as_string(),
        }
        
        return info
    
    except Exception as e:
        logger.error(f"Error getting system info: {str(e)}")
        return {"error": str(e)}


def safe_divide(numerator: Union[int, float], 
                denominator: Union[int, float], 
                default: Union[int, float] = 0) -> float:
    """
    Safely divide two numbers, returning a default value if the denominator is zero.
    
    Args:
        numerator: The numerator
        denominator: The denominator
        default: Default value to return if denominator is zero
        
    Returns:
        Division result or default value
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ValueError, ZeroDivisionError):
        return default


def moving_average(data: List[Union[int, float]], window: int = 3) -> List[float]:
    """
    Calculate the moving average of a list of numbers.
    
    Args:
        data: List of numbers
        window: Window size for the moving average
        
    Returns:
        List of moving averages
    """
    if not data or window <= 0 or window > len(data):
        return []
    
    result = []
    for i in range(len(data)):
        if i < window - 1:
            # Not enough data points yet, use available data
            window_data = data[:i+1]
        else:
            # Full window
            window_data = data[i-window+1:i+1]
        
        # Calculate average for this window
        avg = sum(window_data) / len(window_data)
        result.append(avg)
    
    return result


def calculate_percent_change(old_value: Union[int, float], 
                            new_value: Union[int, float], 
                            precision: int = 2) -> float:
    """
    Calculate the percentage change between two values.
    
    Args:
        old_value: Old value
        new_value: New value
        precision: Number of decimal places
        
    Returns:
        Percentage change
    """
    if old_value == 0:
        return 0.0
    
    try:
        percent_change = ((new_value - old_value) / abs(old_value)) * 100
        return round(percent_change, precision)
    except (TypeError, ValueError):
        return 0.0


def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate a string to a maximum length, adding a suffix if truncated.
    
    Args:
        text: String to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated string
    """
    if not text or len(text) <= max_length:
        return text
    
    return text[:max_length-len(suffix)] + suffix


def get_nested_dict_value(data: Dict[str, Any], 
                         path: str, 
                         default: Any = None, 
                         separator: str = ".") -> Any:
    """
    Get a value from a nested dictionary using a dot-notation path.
    
    Args:
        data: Nested dictionary
        path: Dot-notation path to the value
        default: Default value if the path doesn't exist
        separator: Path separator
        
    Returns:
        Value from the nested dictionary or default
    """
    if not data or not path:
        return default
    
    parts = path.split(separator)
    current = data
    
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default
    
    return current


def set_nested_dict_value(data: Dict[str, Any], 
                         path: str, 
                         value: Any, 
                         separator: str = ".") -> Dict[str, Any]:
    """
    Set a value in a nested dictionary using a dot-notation path.
    
    Args:
        data: Nested dictionary
        path: Dot-notation path to the value
        value: Value to set
        separator: Path separator
        
    Returns:
        Updated dictionary
    """
    if not data or not path:
        return data
    
    parts = path.split(separator)
    current = data
    
    for i, part in enumerate(parts[:-1]):
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    
    current[parts[-1]] = value
    return data


def deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries, with dict2 values taking precedence.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary (values take precedence)
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def load_json_file(file_path: str) -> Dict[str, Any]:
    """
    Load and parse a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Parsed JSON data or empty dict on error
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {str(e)}")
        return {}


def save_json_file(data: Dict[str, Any], file_path: str, indent: int = 2) -> bool:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save the file
        indent: JSON indentation
        
    Returns:
        True if saved successfully, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=indent)
        
        return True
    except Exception as e:
        logger.error(f"Error saving JSON file {file_path}: {str(e)}")
        return False


def encode_base64(data: Union[str, bytes]) -> str:
    """
    Encode data as base64.
    
    Args:
        data: String or bytes to encode
        
    Returns:
        Base64-encoded string
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    
    return base64.b64encode(data).decode('utf-8')


def decode_base64(data: str) -> bytes:
    """
    Decode base64 data.
    
    Args:
        data: Base64-encoded string
        
    Returns:
        Decoded bytes
    """
    return base64.b64decode(data)


def generate_hash(data: Union[str, bytes], algorithm: str = 'sha256') -> str:
    """
    Generate a hash of the input data.
    
    Args:
        data: Data to hash
        algorithm: Hash algorithm (md5, sha1, sha256, etc.)
        
    Returns:
        Hexadecimal hash string
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    
    hash_obj = hashlib.new(algorithm)
    hash_obj.update(data)
    return hash_obj.hexdigest()


def list_to_chunks(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split a list into chunks of a specified size.
    
    Args:
        items: List of items
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    if not items or chunk_size <= 0:
        return []
    
    return [items[i:i+chunk_size] for i in range(0, len(items), chunk_size)]


def get_duplicate_items(items: List[Any]) -> List[Any]:
    """
    Find duplicate items in a list.
    
    Args:
        items: List of items
        
    Returns:
        List of duplicate items
    """
    seen = set()
    duplicates = set()
    
    for item in items:
        if item in seen:
            duplicates.add(item)
        seen.add(item)
    
    return list(duplicates)


def url_encode(text: str) -> str:
    """
    URL-encode a string.
    
    Args:
        text: String to encode
        
    Returns:
        URL-encoded string
    """
    return urllib.parse.quote(text)


def url_decode(text: str) -> str:
    """
    URL-decode a string.
    
    Args:
        text: String to decode
        
    Returns:
        URL-decoded string
    """
    return urllib.parse.unquote(text)


def build_query_string(params: Dict[str, Any]) -> str:
    """
    Build a URL query string from a dictionary of parameters.
    
    Args:
        params: Dictionary of parameters
        
    Returns:
        Query string
    """
    return urllib.parse.urlencode(params)


def parse_query_string(query_string: str) -> Dict[str, str]:
    """
    Parse a URL query string into a dictionary.
    
    Args:
        query_string: Query string to parse
        
    Returns:
        Dictionary of parameters
    """
    if query_string.startswith('?'):
        query_string = query_string[1:]
    
    return dict(urllib.parse.parse_qsl(query_string))


def to_snake_case(text: str) -> str:
    """
    Convert a string to snake_case.
    
    Args:
        text: String to convert
        
    Returns:
        snake_case string
    """
    # Replace non-alphanumeric characters with spaces
    s1 = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    
    # Add spaces before capital letters
    s2 = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', s1)
    
    # Convert to lowercase and replace spaces with underscores
    return re.sub(r'\s+', '_', s2.lower()).strip('_')


def to_camel_case(text: str) -> str:
    """
    Convert a string to camelCase.
    
    Args:
        text: String to convert
        
    Returns:
        camelCase string
    """
    # Replace non-alphanumeric characters with spaces
    s1 = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    
    # Split by whitespace
    words = s1.split()
    
    if not words:
        return ""
    
    # First word lowercase, rest with capital first letter
    return words[0].lower() + ''.join(word.capitalize() for word in words[1:])


def to_title_case(text: str) -> str:
    """
    Convert a string to Title Case.
    
    Args:
        text: String to convert
        
    Returns:
        Title Case string
    """
    # Replace non-alphanumeric characters with spaces
    s1 = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    
    # Split by whitespace
    words = s1.split()
    
    # Capitalize each word
    return ' '.join(word.capitalize() for word in words)


def find_files(directory: str, pattern: str = "*", recursive: bool = True) -> List[str]:
    """
    Find files in a directory matching a pattern.
    
    Args:
        directory: Directory to search
        pattern: Glob pattern for file matching
        recursive: Whether to search recursively
        
    Returns:
        List of matching file paths
    """
    try:
        path = Path(directory)
        if recursive:
            return [str(f) for f in path.glob(f"**/{pattern}") if f.is_file()]
        else:
            return [str(f) for f in path.glob(pattern) if f.is_file()]
    except Exception as e:
        logger.error(f"Error finding files in {directory}: {str(e)}")
        return []


def safe_filename(filename: str) -> str:
    """
    Convert a string to a safe filename by removing or replacing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Safe filename
    """
    # Replace invalid filename characters with underscores
    s1 = re.sub(r'[\\/*?:"<>|]', '_', filename)
    
    # Replace multiple consecutive underscores with a single one
    s2 = re.sub(r'_+', '_', s1)
    
    # Remove leading/trailing underscores and spaces
    return s2.strip('_ ')


def get_module_functions(module_name: str) -> List[str]:
    """
    Get a list of function names defined in a module.
    
    Args:
        module_name: Module name
        
    Returns:
        List of function names
    """
    try:
        module = sys.modules.get(module_name)
        
        if module is None:
            # Try to import the module
            try:
                module = importlib.import_module(module_name)
            except Exception as e:
                logger.error(f"Error importing module {module_name}: {str(e)}")
                return []
        
        # Get functions from the module
        functions = []
        for name, obj in inspect.getmembers(module):
            if inspect.isfunction(obj) and obj.__module__ == module_name:
                functions.append(name)
        
        return functions
    
    except Exception as e:
        logger.error(f"Error getting functions from module {module_name}: {str(e)}")
        return []


def get_object_attributes(obj: Any) -> Dict[str, Any]:
    """
    Get a dictionary of object attributes that don't start with underscore.
    
    Args:
        obj: Object to inspect
        
    Returns:
        Dictionary of attributes
    """
    return {key: value for key, value in vars(obj).items() if not key.startswith('_')}


def trim_dataframe(df: pd.DataFrame, max_rows: int = 1000) -> pd.DataFrame:
    """
    Trim a DataFrame to a maximum number of rows.
    
    Args:
        df: DataFrame to trim
        max_rows: Maximum number of rows
        
    Returns:
        Trimmed DataFrame
    """
    if df is None or df.empty or len(df) <= max_rows:
        return df
    
    return df.iloc[:max_rows]


def convert_size_to_bytes(size_str: str) -> int:
    """
    Convert a size string (e.g., '1KB', '2MB') to bytes.
    
    Args:
        size_str: Size string
        
    Returns:
        Size in bytes
    """
    size_str = size_str.upper()
    
    units = {
        'B': 1,
        'KB': 1024,
        'MB': 1024 ** 2,
        'GB': 1024 ** 3,
        'TB': 1024 ** 4,
    }
    
    pattern = r'^(\d+(?:\.\d+)?)\s*([A-Za-z]+)$'
    match = re.match(pattern, size_str)
    
    if match:
        value, unit = match.groups()
        if unit in units:
            return int(float(value) * units[unit])
    
    # If the format is not recognized, try to convert to integer
    try:
        return int(size_str)
    except ValueError:
        return 0


def format_bytes(size_bytes: int) -> str:
    """
    Format bytes into a human-readable string.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0B"
    
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    i = 0
    while size_bytes >= 1024 and i < len(units) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.2f}{units[i]}"


def wait_for_condition(condition_func: Callable[[], bool], 
                       timeout: float = 10.0, 
                       interval: float = 0.1) -> bool:
    """
    Wait for a condition to be met.
    
    Args:
        condition_func: Function that returns True when the condition is met
        timeout: Maximum time to wait in seconds
        interval: Interval between checks in seconds
        
    Returns:
        True if the condition was met, False if the timeout was reached
    """
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        if condition_func():
            return True
        time.sleep(interval)
    
    return False 