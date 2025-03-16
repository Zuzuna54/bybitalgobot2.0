"""
Callbacks for the notification service
"""

import time
from typing import Dict, Any, List, Optional
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State, ALL
import dash
from loguru import logger

from src.dashboard.router.callback_registry import callback_registrar
from src.dashboard.services.notification_service.toast import (
    create_toast,
    create_toast_component,
)
from src.dashboard.services.notification_service.constants import MAX_NOTIFICATIONS


def register_notification_callbacks(
    app: dash.Dash, data_service: Optional[Any] = None, **kwargs
) -> None:
    """
    Register notification callbacks with the Dash app.

    Args:
        app: Dash application instance
        data_service: Optional data service instance (not used but required for callback registry)
        **kwargs: Additional keyword arguments
    """

    # Register callback to update notifications
    @app.callback(
        Output("notification-container", "children"),
        Input("notification-store", "data"),
        prevent_initial_call=True,
    )
    def _update_notifications(notifications):
        return update_notifications(notifications)

    # Register callback to manage notifications
    @app.callback(
        Output("notification-store", "data"),
        Input({"type": "notification-toast", "index": ALL}, "n_clicks"),
        State("notification-store", "data"),
        prevent_initial_call=True,
    )
    def _manage_notifications(n_clicks, current_notifications):
        return manage_notifications(n_clicks, current_notifications)


def update_notifications(notifications: List[Dict[str, Any]]) -> List[dbc.Toast]:
    """
    Update the notification container with toast components.

    Args:
        notifications: List of notification data dictionaries

    Returns:
        List of dbc.Toast components
    """
    if not notifications:
        return []

    # Sort notifications by timestamp (newest first)
    sorted_notifications = sorted(
        notifications, key=lambda x: x.get("timestamp", 0), reverse=True
    )

    # Limit the number of notifications shown
    limited_notifications = sorted_notifications[:MAX_NOTIFICATIONS]

    # Create toast components
    toast_components = [
        create_toast_component(notification) for notification in limited_notifications
    ]

    return toast_components


def manage_notifications(
    n_clicks: List[int], current_notifications: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Manage notifications when they are clicked (dismissed).

    Args:
        n_clicks: List of click counts for each notification
        current_notifications: Current list of notifications

    Returns:
        Updated list of notifications
    """
    # If no notifications or no clicks, return current notifications
    if not n_clicks or not any(n is not None for n in n_clicks):
        return current_notifications

    # Get the context to determine which notification was clicked
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_notifications

    # Get the ID of the clicked notification
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    try:
        # Parse the JSON ID
        notification_id = dash.callback_context.triggered_id["index"]

        # Remove the clicked notification
        updated_notifications = [
            n for n in current_notifications if n["id"] != notification_id
        ]

        logger.debug(f"Removed notification: {notification_id}")
        return updated_notifications
    except Exception as e:
        logger.error(f"Error managing notifications: {str(e)}")
        return current_notifications


def add_notification(
    notification_store: List[Dict[str, Any]],
    message: str,
    notification_type: str = "info",
    title: str = None,
    duration: int = 5000,
    dismissable: bool = True,
) -> List[Dict[str, Any]]:
    """
    Add a notification to the notification store.

    Args:
        notification_store: Current notification store data
        message: Notification message
        notification_type: Type of notification
        title: Notification title
        duration: Duration in milliseconds
        dismissable: Whether the notification can be dismissed

    Returns:
        Updated notification store data
    """
    # Create the notification
    notification = create_toast(
        message=message,
        notification_type=notification_type,
        title=title,
        duration=duration,
        dismissable=dismissable,
    )

    # Add timestamp
    notification["timestamp"] = time.time()

    # Add to store
    updated_store = notification_store.copy()
    updated_store.append(notification)

    return updated_store
