"""
Dashboard Converters Module

This module provides utilities for converting between different data types and formats in the dashboard.
"""

from typing import Dict, Any, List, Union, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import json


def to_float(value: Any, default: float = 0.0) -> float:
    """
    Convert a value to float safely.
    
    Args:
        value: The value to convert
        default: Default value if conversion fails
        
    Returns:
        Converted float value or default
    """
    if value is None:
        return default
    
    try:
        return float(value)
    except (ValueError, TypeError):
        if isinstance(value, str):
            # Try to handle percentage strings
            if value.endswith('%'):
                try:
                    return float(value.rstrip('%')) / 100.0
                except (ValueError, TypeError):
                    pass
            
            # Try to handle currency strings
            for currency in ['$', '€', '£', '¥']:
                if value.startswith(currency):
                    try:
                        return float(value.lstrip(currency).replace(',', ''))
                    except (ValueError, TypeError):
                        pass
        
        return default


def to_int(value: Any, default: int = 0) -> int:
    """
    Convert a value to integer safely.
    
    Args:
        value: The value to convert
        default: Default value if conversion fails
        
    Returns:
        Converted integer value or default
    """
    if value is None:
        return default
    
    try:
        # First try direct conversion
        return int(value)
    except (ValueError, TypeError):
        # Try float conversion first then to int
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return default


def to_bool(value: Any, default: bool = False) -> bool:
    """
    Convert a value to boolean safely.
    
    Args:
        value: The value to convert
        default: Default value if conversion fails
        
    Returns:
        Converted boolean value or default
    """
    if value is None:
        return default
    
    if isinstance(value, bool):
        return value
    
    if isinstance(value, (int, float)):
        return bool(value)
    
    if isinstance(value, str):
        # Check common string representations of boolean values
        value = value.lower().strip()
        if value in ('true', 'yes', 'y', '1', 'on'):
            return True
        if value in ('false', 'no', 'n', '0', 'off'):
            return False
    
    # If we couldn't convert, return the default
    return default


def to_datetime(value: Any, default: Optional[datetime] = None) -> Optional[datetime]:
    """
    Convert a value to datetime safely.
    
    Args:
        value: The value to convert
        default: Default value if conversion fails
        
    Returns:
        Converted datetime value or default
    """
    if value is None:
        return default
    
    if isinstance(value, datetime):
        return value
    
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    
    if isinstance(value, (int, float)):
        # Assume it's a timestamp
        try:
            # If it's in seconds (Unix timestamp)
            if value < 2147483648:  # Max 32-bit int
                return datetime.fromtimestamp(value)
            # If it's in milliseconds
            return datetime.fromtimestamp(value / 1000)
        except (ValueError, OSError, OverflowError):
            return default
    
    if isinstance(value, str):
        # Try common date formats
        for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y'):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
        
        # Try ISO format
        try:
            return datetime.fromisoformat(value.replace('Z', '+00:00'))
        except ValueError:
            pass
    
    return default


def to_list(value: Any, default: Optional[List] = None) -> List:
    """
    Convert a value to a list safely.
    
    Args:
        value: The value to convert
        default: Default value if conversion fails
        
    Returns:
        Converted list or default
    """
    if default is None:
        default = []
    
    if value is None:
        return default
    
    if isinstance(value, list):
        return value
    
    if isinstance(value, tuple):
        return list(value)
    
    if isinstance(value, set):
        return list(value)
    
    if isinstance(value, dict):
        return list(value.items())
    
    if isinstance(value, str):
        # Try to parse as JSON
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            # If it's not JSON, split by comma
            return [item.strip() for item in value.split(',') if item.strip()]
    
    # If all else fails, wrap in a list
    return [value]


def to_dict(value: Any, default: Optional[Dict] = None) -> Dict:
    """
    Convert a value to a dictionary safely.
    
    Args:
        value: The value to convert
        default: Default value if conversion fails
        
    Returns:
        Converted dictionary or default
    """
    if default is None:
        default = {}
    
    if value is None:
        return default
    
    if isinstance(value, dict):
        return value
    
    if isinstance(value, str):
        # Try to parse as JSON
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return default
    
    if isinstance(value, (list, tuple)) and all(isinstance(item, (list, tuple)) and len(item) == 2 for item in value):
        # Convert list of tuples to dict
        return dict(value)
    
    if hasattr(value, '__dict__'):
        # Convert object to dict
        return vars(value)
    
    return default


def to_dataframe(data: Any, default: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Convert various data formats to a pandas DataFrame.
    
    Args:
        data: The data to convert
        default: Default DataFrame if conversion fails
        
    Returns:
        Converted DataFrame or default
    """
    if default is None:
        default = pd.DataFrame()
    
    if data is None:
        return default
    
    if isinstance(data, pd.DataFrame):
        return data
    
    try:
        if isinstance(data, dict):
            # Try to convert a dict to DataFrame
            return pd.DataFrame(data)
        elif isinstance(data, (list, tuple)):
            # Try to convert a list to DataFrame
            return pd.DataFrame(data)
        elif isinstance(data, str):
            # Try to parse as JSON
            try:
                parsed = json.loads(data)
                return pd.DataFrame(parsed)
            except json.JSONDecodeError:
                # Try to parse as CSV
                try:
                    return pd.read_csv(data)
                except Exception:
                    return default
        else:
            return default
    except Exception:
        return default


def parse_timeframe(timeframe: str) -> Optional[timedelta]:
    """
    Parse a timeframe string into a timedelta object.
    
    Args:
        timeframe: String representation of a timeframe (e.g., '1m', '1h', '1d')
        
    Returns:
        Timedelta object or None if parsing fails
    """
    if not timeframe:
        return None
    
    try:
        # Extract the number and unit
        value = int(''.join(filter(str.isdigit, timeframe)))
        unit = ''.join(filter(str.isalpha, timeframe)).lower()
        
        # Convert to timedelta
        if unit == 'm':
            return timedelta(minutes=value)
        elif unit == 'h':
            return timedelta(hours=value)
        elif unit == 'd':
            return timedelta(days=value)
        elif unit == 'w':
            return timedelta(weeks=value)
        elif unit == 'M':
            # Approximate a month as 30 days
            return timedelta(days=30 * value)
        else:
            return None
    except (ValueError, AttributeError):
        return None


def convert_df_types(df: pd.DataFrame, type_map: Dict[str, str]) -> pd.DataFrame:
    """
    Convert DataFrame column types based on a type mapping.
    
    Args:
        df: The DataFrame to convert
        type_map: Dictionary mapping column names to type strings
        
    Returns:
        DataFrame with converted types
    """
    if df is None or df.empty or not type_map:
        return df
    
    df_copy = df.copy()
    
    for column, dtype in type_map.items():
        if column not in df_copy.columns:
            continue
        
        try:
            if dtype == 'float':
                df_copy[column] = df_copy[column].apply(lambda x: to_float(x))
            elif dtype == 'int':
                df_copy[column] = df_copy[column].apply(lambda x: to_int(x))
            elif dtype == 'bool':
                df_copy[column] = df_copy[column].apply(lambda x: to_bool(x))
            elif dtype == 'datetime':
                df_copy[column] = df_copy[column].apply(lambda x: to_datetime(x))
            elif dtype == 'str':
                df_copy[column] = df_copy[column].astype(str)
        except Exception:
            # If conversion fails, leave as is
            continue
    
    return df_copy


def json_serialize(obj: Any) -> Any:
    """
    Convert Python objects to JSON-serializable objects.
    
    Args:
        obj: The object to serialize
        
    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    else:
        return obj 