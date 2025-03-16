"""
Dashboard Notification Service Package

This package provides a centralized notification system for displaying toast notifications
and handling error messages throughout the dashboard.
"""

from src.dashboard.services.notification_service.components import (
    create_notification_components,
)
from src.dashboard.services.notification_service.toast import create_toast
from src.dashboard.services.notification_service.callbacks import (
    register_notification_callbacks,
    update_notifications,
    manage_notifications,
)
from src.dashboard.services.notification_service.error_handler import (
    with_error_handling,
)

# Define exports
__all__ = [
    "create_notification_components",
    "create_toast",
    "register_notification_callbacks",
    "update_notifications",
    "manage_notifications",
    "with_error_handling",
]
