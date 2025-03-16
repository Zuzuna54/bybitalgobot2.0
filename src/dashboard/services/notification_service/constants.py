"""
Constants for the notification service
"""

# Notification types with corresponding icons and colors
NOTIFICATION_TYPES = {
    "info": {"icon": "info-circle", "color": "info"},
    "success": {"icon": "check-circle", "color": "success"},
    "warning": {"icon": "exclamation-triangle", "color": "warning"},
    "error": {"icon": "exclamation-circle", "color": "danger"},
    "trade": {"icon": "exchange-alt", "color": "primary"},
    "system": {"icon": "server", "color": "secondary"},
}

# Default notification settings
DEFAULT_DURATION = 5000  # milliseconds
MAX_NOTIFICATIONS = 10
