"""
Dashboard Notification Service Module (DEPRECATED)

This module is being refactored into smaller, more maintainable modules.
Please use the new modules in the src/dashboard/services/notification_service/ package:

- components.py: UI components for notifications
- toast.py: Toast notification creation and management
- callbacks.py: Dash callbacks for notification handling
- error_handler.py: Error handling utilities
- constants.py: Constants and configuration

This file is kept for backward compatibility but will be removed in a future version.
"""

import warnings

# Add deprecation warning
warnings.warn(
    "The notification_service.py module is deprecated and will be removed in a future version. "
    "Please use the notification_service package instead. "
    "Import from src.dashboard.services.notification_service directly, which now refers to the package.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export all notification service functions from the new package
from src.dashboard.services.notification_service import *
