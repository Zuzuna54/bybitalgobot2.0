"""
System Data Module

This module provides functions to retrieve and process system-related data
for the dashboard, including system status and configuration.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime, timedelta
import platform
import psutil
import os
from loguru import logger


def _initialize_system_data(service):
    """
    Initialize system data storage with sample data for standalone mode.

    Args:
        service: DashboardDataService instance
    """
    # Set system status for standalone mode
    service._system_running = True
    service._system_start_time = datetime.now() - timedelta(
        hours=2
    )  # Simulate 2 hours of uptime
    service._system_mode = "Standalone"

    # Set initial update timestamp
    service._data_updated_at["system"] = datetime.now()

    logger.debug("Initialized system data")


def _update_system_status(service):
    """
    Update system status information.

    Args:
        service: DashboardDataService instance
    """
    # In standalone mode, just update the running state
    if service.standalone_mode:
        service._data_updated_at["system"] = datetime.now()
        return

    try:
        # Update system running state and mode
        if service.trade_manager:
            service._system_running = service.trade_manager.is_running()
            service._system_mode = service.trade_manager.get_trading_mode()

        # Update system start time if it's None and system is running
        if service._system_start_time is None and service._system_running:
            service._system_start_time = datetime.now()

        # Update timestamp
        service._data_updated_at["system"] = datetime.now()

    except Exception as e:
        logger.error(f"Error updating system status: {str(e)}")
        # Keep using existing data if update fails


def get_system_status(self) -> Dict[str, Any]:
    """
    Get system status information.

    Returns:
        Dictionary with system status information
    """
    current_time = datetime.now()

    # Calculate uptime if system is running
    uptime_seconds = 0
    if self._system_running and self._system_start_time:
        uptime_seconds = (current_time - self._system_start_time).total_seconds()

    # Format uptime as string
    uptime_str = _format_uptime(uptime_seconds)

    # Get system resources
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory_percent = psutil.virtual_memory().percent

    # Get system and Python version info
    system_info = {
        "system": platform.system(),
        "release": platform.release(),
        "python_version": platform.python_version(),
    }

    return {
        "running": self._system_running,
        "mode": self._system_mode,
        "uptime": uptime_str,
        "uptime_seconds": uptime_seconds,
        "started_at": (
            self._system_start_time.isoformat() if self._system_start_time else None
        ),
        "cpu_percent": cpu_percent,
        "memory_percent": memory_percent,
        "system_info": system_info,
    }


def get_data_freshness(self) -> Dict[str, Dict[str, Any]]:
    """
    Get information about data freshness for each data category.

    Returns:
        Dictionary with data freshness information
    """
    current_time = datetime.now()
    freshness = {}

    for category, updated_at in self._data_updated_at.items():
        if updated_at:
            # Calculate how many seconds ago the data was updated
            age_seconds = (current_time - updated_at).total_seconds()
            age_str = _format_elapsed_time(age_seconds)

            freshness[category] = {
                "updated_at": updated_at.isoformat(),
                "age_seconds": age_seconds,
                "age": age_str,
                "is_fresh": age_seconds < 60,  # Fresh if less than 60 seconds old
            }
        else:
            freshness[category] = {
                "updated_at": None,
                "age_seconds": None,
                "age": "Never",
                "is_fresh": False,
            }

    return freshness


def _format_uptime(seconds: float) -> str:
    """
    Format uptime seconds as a human-readable string.

    Args:
        seconds: Number of seconds

    Returns:
        Formatted string like "2d 5h 3m 12s"
    """
    if seconds < 0:
        return "0s"

    days, remainder = divmod(int(seconds), 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)

    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0 or days > 0:
        parts.append(f"{hours}h")
    if minutes > 0 or hours > 0 or days > 0:
        parts.append(f"{minutes}m")

    parts.append(f"{seconds}s")

    return " ".join(parts)


def _format_elapsed_time(seconds: float) -> str:
    """
    Format elapsed time as a human-readable string.

    Args:
        seconds: Number of seconds

    Returns:
        Formatted string like "5 minutes ago" or "just now"
    """
    if seconds < 10:
        return "just now"
    elif seconds < 60:
        return f"{int(seconds)} seconds ago"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    elif seconds < 86400:
        hours = int(seconds / 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    else:
        days = int(seconds / 86400)
        return f"{days} day{'s' if days != 1 else ''} ago"
