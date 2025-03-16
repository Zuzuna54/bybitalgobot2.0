"""
Update Service Configuration

This module provides configuration defaults and settings for the update service.
"""

# Default update intervals in seconds for different data types
DEFAULT_UPDATE_INTERVALS = {
    "performance": 30,  # 30 seconds
    "trades": 5,  # 5 seconds
    "orderbook": 1,  # 1 second
    "strategy": 10,  # 10 seconds
    "market": 5,  # 5 seconds
    "system": 2,  # 2 seconds
    "all": 60,  # 60 seconds (full refresh)
}

# Minimum allowed update intervals (to prevent excessive updates)
MIN_UPDATE_INTERVALS = {
    "performance": 10,  # 10 seconds
    "trades": 2,  # 2 seconds
    "orderbook": 0.5,  # 0.5 seconds
    "strategy": 5,  # 5 seconds
    "market": 2,  # 2 seconds
    "system": 1,  # 1 second
    "all": 30,  # 30 seconds
}

# Update priority (lower number = higher priority)
UPDATE_PRIORITY = {
    "system": 1,
    "orderbook": 2,
    "market": 3,
    "trades": 4,
    "strategy": 5,
    "performance": 6,
    "all": 10,
}
