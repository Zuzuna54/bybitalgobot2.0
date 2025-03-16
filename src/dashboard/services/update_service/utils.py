"""
Update Service Utilities

This module provides utility functions for the update service.
"""

from typing import Optional
from datetime import datetime, timedelta


def is_update_due(last_update: Optional[datetime], interval_seconds: int) -> bool:
    """
    Check if an update is due based on the last update time and interval.

    Args:
        last_update: Time of the last update, or None if never updated
        interval_seconds: Update interval in seconds

    Returns:
        True if an update is due, False otherwise
    """
    # If never updated, an update is due
    if last_update is None:
        return True

    # If interval is 0 or negative, updates are disabled
    if interval_seconds <= 0:
        return False

    # Calculate the next update time
    next_update = last_update + timedelta(seconds=interval_seconds)

    # Check if the next update time is in the past
    return datetime.now() >= next_update


def get_next_update_time(
    last_update: Optional[datetime], interval_seconds: int
) -> Optional[datetime]:
    """
    Calculate the next update time based on the last update and interval.

    Args:
        last_update: Time of the last update, or None if never updated
        interval_seconds: Update interval in seconds

    Returns:
        Datetime of the next update, or None if not applicable
    """
    # If never updated, return current time
    if last_update is None:
        return datetime.now()

    # If interval is 0 or negative, updates are disabled
    if interval_seconds <= 0:
        return None

    # Calculate the next update time
    return last_update + timedelta(seconds=interval_seconds)


def format_time_until_update(next_update: Optional[datetime]) -> str:
    """
    Format the time until the next update as a human-readable string.

    Args:
        next_update: Datetime of the next update, or None if not applicable

    Returns:
        String like "2m 30s" or "Now" or "Disabled"
    """
    if next_update is None:
        return "Disabled"

    # Calculate time difference
    now = datetime.now()

    # If next update is in the past, return "Now"
    if next_update <= now:
        return "Now"

    # Calculate time difference in seconds
    time_diff = (next_update - now).total_seconds()

    # Format time difference
    minutes, seconds = divmod(int(time_diff), 60)
    hours, minutes = divmod(minutes, 60)

    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"
