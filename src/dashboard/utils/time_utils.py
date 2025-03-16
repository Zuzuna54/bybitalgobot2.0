"""
Time Utility Module

This module provides centralized date and time utility functions for the dashboard.
It consolidates time-related operations to reduce code duplication and ensure consistency.
"""

from typing import Union, List, Optional
from datetime import datetime, timedelta, date
import pandas as pd
import time


def timestamp_ms() -> int:
    """
    Get current timestamp in milliseconds.

    Returns:
        Current timestamp in milliseconds
    """
    return int(time.time() * 1000)


def timestamp_ns() -> int:
    """
    Get current timestamp in nanoseconds.

    Returns:
        Current timestamp in nanoseconds
    """
    return int(time.time() * 1_000_000_000)


def get_current_time() -> datetime:
    """
    Get current time as datetime object.

    Returns:
        Current datetime
    """
    return datetime.now()


def get_current_time_as_string(fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Get current time as formatted string.

    Args:
        fmt: Format string (default: "%Y-%m-%d %H:%M:%S")

    Returns:
        Formatted current time string
    """
    return datetime.now().strftime(fmt)


def parse_timestamp(timestamp: Union[int, float, str]) -> datetime:
    """
    Parse a timestamp into a datetime object.

    Args:
        timestamp: Unix timestamp (in seconds or milliseconds) or datetime string

    Returns:
        Datetime object
    """
    if isinstance(timestamp, (int, float)):
        # Check if timestamp is in milliseconds (13 digits)
        if timestamp > 1_000_000_000_000:
            return datetime.fromtimestamp(timestamp / 1000)
        return datetime.fromtimestamp(timestamp)

    if isinstance(timestamp, str):
        try:
            # Try ISO format
            return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError:
            try:
                # Try common formats
                for fmt in [
                    "%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%d",
                    "%m/%d/%Y %H:%M:%S",
                    "%m/%d/%Y",
                ]:
                    try:
                        return datetime.strptime(timestamp, fmt)
                    except ValueError:
                        continue
                # If all else fails, parse with pandas
                return pd.to_datetime(timestamp)
            except Exception:
                raise ValueError(f"Could not parse timestamp: {timestamp}")

    raise TypeError(f"Unsupported timestamp type: {type(timestamp)}")


def format_timestamp(
    timestamp: Union[datetime, str, float, int], format_str: str = "%Y-%m-%d %H:%M:%S"
) -> str:
    """
    Format a timestamp as string.

    Args:
        timestamp: The timestamp to format (datetime, unix timestamp, or string)
        format_str: Format string for the output

    Returns:
        Formatted timestamp string
    """
    # Handle None case
    if timestamp is None:
        return ""

    # Convert to datetime if needed
    if not isinstance(timestamp, datetime):
        try:
            timestamp = parse_timestamp(timestamp)
        except (ValueError, TypeError):
            return str(timestamp)

    # Format the datetime
    return timestamp.strftime(format_str)


def format_time_ago(timestamp: Union[datetime, str, float, int]) -> str:
    """
    Format a timestamp as a human-readable time ago string.

    Args:
        timestamp: The timestamp to format (datetime, unix timestamp, or string)

    Returns:
        Human-readable time difference (e.g., "2 hours ago")
    """
    # Convert to datetime if needed
    if not isinstance(timestamp, datetime):
        try:
            timestamp = parse_timestamp(timestamp)
        except (ValueError, TypeError):
            return "unknown time ago"

    # Calculate time difference
    now = datetime.now()
    delta = now - timestamp

    # Convert to appropriate units
    seconds = delta.total_seconds()

    if seconds < 60:
        return f"{int(seconds)} seconds ago"

    minutes = seconds / 60
    if minutes < 60:
        return f"{int(minutes)} minutes ago"

    hours = minutes / 60
    if hours < 24:
        return f"{int(hours)} hours ago"

    days = hours / 24
    if days < 7:
        return f"{int(days)} days ago"

    weeks = days / 7
    if weeks < 4:
        return f"{int(weeks)} weeks ago"

    months = days / 30.44  # Average month length
    if months < 12:
        return f"{int(months)} months ago"

    years = days / 365.25  # Account for leap years
    return f"{int(years)} years ago"


def get_date_range(
    start_date: Union[str, datetime, date],
    end_date: Union[str, datetime, date],
    as_string: bool = False,
    fmt: str = "%Y-%m-%d",
) -> List[Union[datetime, str]]:
    """
    Get a list of dates between start and end dates, inclusive.

    Args:
        start_date: Start date
        end_date: End date
        as_string: Whether to return dates as strings
        fmt: Date format for strings (if as_string is True)

    Returns:
        List of dates (datetime objects or strings)
    """
    # Convert to datetime if needed
    if not isinstance(start_date, datetime):
        start_date = parse_timestamp(start_date)

    if not isinstance(end_date, datetime):
        end_date = parse_timestamp(end_date)

    # Ensure start date is before end date
    if start_date > end_date:
        start_date, end_date = end_date, start_date

    # Create date range
    dates = []
    current_date = start_date

    while current_date <= end_date:
        if as_string:
            dates.append(current_date.strftime(fmt))
        else:
            dates.append(current_date)

        current_date += timedelta(days=1)

    return dates


def filter_data_by_time_range(
    data: pd.DataFrame, time_range: str, date_column: str = None
) -> pd.DataFrame:
    """
    Filter DataFrame by time range.

    Args:
        data: DataFrame to filter
        time_range: Time range to filter by (e.g., "1d", "1w", "1m", "3m", "all")
        date_column: Name of the timestamp column (if not using DataFrame index)

    Returns:
        Filtered DataFrame
    """
    if data is None or data.empty:
        return pd.DataFrame()

    if time_range == "all":
        return data

    now = datetime.now()

    if time_range == "1d":
        start_date = now - timedelta(days=1)
    elif time_range == "1w":
        start_date = now - timedelta(weeks=1)
    elif time_range == "1m":
        start_date = now - timedelta(days=30)
    elif time_range == "3m":
        start_date = now - timedelta(days=90)
    elif time_range == "6m":
        start_date = now - timedelta(days=180)
    elif time_range == "1y":
        start_date = now - timedelta(days=365)
    else:
        # Default to 1 month if invalid range
        start_date = now - timedelta(days=30)

    try:
        if date_column is not None:
            # Filter by specified date column
            return data[data[date_column] >= start_date]
        else:
            # Filter by index (assumes DatetimeIndex)
            return data[data.index >= start_date]
    except Exception as e:
        print(f"Error filtering data by time range: {str(e)}")
        return data


def format_duration(seconds: float) -> str:
    """
    Format a duration in seconds as a human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string (e.g., "2h 30m 15s" or "45m 30s")
    """
    if seconds < 0:
        return "0s"

    # Handle different time units
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)

    # Build the formatted string
    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0 or (days > 0 and (minutes > 0 or seconds > 0)):
        parts.append(f"{hours}h")
    if minutes > 0 or (hours > 0 and seconds > 0):
        parts.append(f"{minutes}m")
    if seconds > 0 or not parts:
        parts.append(f"{seconds}s")

    return " ".join(parts)


def is_update_due(last_update: Optional[datetime], interval_seconds: float) -> bool:
    """
    Check if an update is due based on the last update time and interval.

    Args:
        last_update: Last update time (or None if no previous update)
        interval_seconds: Update interval in seconds

    Returns:
        True if update is due, False otherwise
    """
    if last_update is None:
        return True

    now = datetime.now()
    elapsed = (now - last_update).total_seconds()
    return elapsed >= interval_seconds


def get_next_update_time(
    last_update: Optional[datetime], interval_seconds: float
) -> Optional[datetime]:
    """
    Calculate the next update time based on last update and interval.

    Args:
        last_update: Last update time
        interval_seconds: Update interval in seconds

    Returns:
        Next scheduled update time or None if no previous update
    """
    if last_update is None:
        return None

    return last_update + timedelta(seconds=interval_seconds)


def is_recent(
    timestamp: Union[datetime, str, float, int], threshold_seconds: int = 300
) -> bool:
    """
    Check if a timestamp is within a recent time window.

    Args:
        timestamp: The timestamp to check (datetime, unix timestamp, or string)
        threshold_seconds: Time threshold in seconds (default: 300 seconds = 5 minutes)

    Returns:
        True if the timestamp is within the threshold, False otherwise
    """
    # Convert to datetime if needed
    if not isinstance(timestamp, datetime):
        try:
            timestamp = parse_timestamp(timestamp)
        except (ValueError, TypeError):
            return False

    # Calculate time difference
    now = datetime.now()
    delta = now - timestamp

    # Return whether the timestamp is within the threshold
    return delta.total_seconds() <= threshold_seconds


# Alias for backward compatibility
timestamp_to_datetime = parse_timestamp
