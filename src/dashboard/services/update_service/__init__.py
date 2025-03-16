"""
Dashboard Update Service Package

This package manages real-time updates for the dashboard components.
It coordinates the timing and frequency of data refreshes.
"""

from src.dashboard.services.update_service.service import UpdateService
from src.dashboard.services.update_service.handlers import (
    register_update_handler,
    unregister_update_handler,
    trigger_update,
)
from src.dashboard.services.update_service.utils import (
    get_last_update_time,
    get_next_update_time,
    is_update_due,
)

# Define exports
__all__ = [
    "UpdateService",
    "register_update_handler",
    "unregister_update_handler",
    "trigger_update",
    "get_last_update_time",
    "get_next_update_time",
    "is_update_due",
]
