"""
Toast notification creation and management
"""

import uuid
from typing import Dict, Any, Optional
import dash_bootstrap_components as dbc
from dash import html

from src.dashboard.services.notification_service.constants import (
    NOTIFICATION_TYPES,
    DEFAULT_DURATION,
)


def create_toast(
    message: str,
    notification_type: str = "info",
    title: Optional[str] = None,
    duration: int = DEFAULT_DURATION,
    dismissable: bool = True,
    icon: Optional[str] = None,
    color: Optional[str] = None,
    id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a toast notification.

    Args:
        message: The notification message
        notification_type: Type of notification (info, success, warning, error, trade, system)
        title: Optional title for the notification
        duration: Duration in milliseconds to show the notification (0 for no auto-dismiss)
        dismissable: Whether the notification can be dismissed by the user
        icon: Optional custom icon (overrides the default for the notification type)
        color: Optional custom color (overrides the default for the notification type)
        id: Optional custom ID for the notification

    Returns:
        Dictionary with notification data
    """
    # Generate a unique ID if not provided
    if id is None:
        id = f"notification-{str(uuid.uuid4())}"

    # Get notification type settings
    notification_settings = NOTIFICATION_TYPES.get(
        notification_type, NOTIFICATION_TYPES["info"]
    )

    # Use provided icon/color or default from notification type
    icon = icon or notification_settings["icon"]
    color = color or notification_settings["color"]

    # Use notification type as title if not provided
    if title is None:
        title = notification_type.capitalize()

    # Create notification data
    notification = {
        "id": id,
        "type": notification_type,
        "title": title,
        "message": message,
        "icon": icon,
        "color": color,
        "duration": duration,
        "dismissable": dismissable,
        "timestamp": None,  # Will be set when added to the store
    }

    return notification


def create_toast_component(notification: Dict[str, Any]) -> dbc.Toast:
    """
    Create a Dash Bootstrap Toast component from notification data.

    Args:
        notification: Notification data dictionary

    Returns:
        dbc.Toast component
    """
    return dbc.Toast(
        html.Div(notification["message"]),
        id={"type": "notification-toast", "index": notification["id"]},
        header=notification["title"],
        icon=notification["icon"],
        dismissable=notification["dismissable"],
        duration=notification["duration"] if notification["duration"] > 0 else None,
        is_open=True,
        color=notification["color"],
        style={"marginBottom": "10px"},
    )
