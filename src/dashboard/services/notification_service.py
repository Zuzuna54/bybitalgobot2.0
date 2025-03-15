"""
Dashboard Notification Service Module

This module provides a centralized notification system for displaying toast notifications
and handling error messages throughout the dashboard.
"""

import uuid
import time
from typing import Dict, Any, List, Tuple, Optional
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State, ALL
import dash
from loguru import logger

# Notification types with corresponding icons and colors
NOTIFICATION_TYPES = {
    "info": {"icon": "info-circle", "color": "info"},
    "success": {"icon": "check-circle", "color": "success"},
    "warning": {"icon": "exclamation-triangle", "color": "warning"},
    "error": {"icon": "exclamation-circle", "color": "danger"},
    "trade": {"icon": "exchange-alt", "color": "primary"},
    "system": {"icon": "server", "color": "secondary"}
}


def create_notification_components() -> Tuple[dcc.Store, html.Div]:
    """
    Create the notification store and container for the dashboard.
    
    Returns:
        tuple: (notification_store, notification_container) for the dashboard layout
    """
    # Store to keep track of active notifications
    notification_store = dcc.Store(id="notification-store", data=[])
    
    # Container where notifications will be displayed
    notification_container = html.Div(
        id="notification-container",
        style={
            "position": "fixed", 
            "top": "20px", 
            "right": "20px", 
            "width": "350px",
            "zIndex": 1000
        }
    )
    
    return notification_store, notification_container


def create_toast(
    message: str, 
    notification_type: str = "info", 
    header: Optional[str] = None, 
    id: Optional[str] = None, 
    duration: int = 4000
) -> dbc.Toast:
    """
    Create a toast notification component.
    
    Args:
        message: The notification message
        notification_type: Type of notification (info, success, warning, error, trade, system)
        header: Optional header text
        id: Optional specific ID for the toast
        duration: Duration in milliseconds before auto-dismissal
        
    Returns:
        Dash Bootstrap Toast component
    """
    # Use a default if notification type is not recognized
    if notification_type not in NOTIFICATION_TYPES:
        notification_type = "info"
    
    # Get icon and color for this notification type
    icon = NOTIFICATION_TYPES[notification_type]["icon"]
    color = NOTIFICATION_TYPES[notification_type]["color"]
    
    # Generate a unique ID if none provided
    if id is None:
        id = str(uuid.uuid4())
    
    # Use a default header if none provided
    if header is None:
        header = notification_type.capitalize()
    
    # Create the toast
    toast = dbc.Toast(
        [html.P(message, className="mb-0")],
        id={"type": "notification-toast", "index": id},
        header=[
            html.I(className=f"fas fa-{icon} me-2"),
            html.Strong(header)
        ],
        dismissable=True,
        is_open=True,
        icon=color,
        duration=duration,
        style={"maxWidth": "100%"}
    )
    
    return toast


def register_notification_callbacks(app: dash.Dash) -> None:
    """
    Register notification system callbacks.
    
    Args:
        app: The Dash application instance
    """
    logger.debug("Registering notification callbacks")
    
    @app.callback(
        Output("notification-container", "children"),
        Input("notification-store", "data"),
        prevent_initial_call=False,
        allow_duplicate=True  # Allow duplicate callbacks for notification outputs
    )
    def update_notifications(notifications: List[Dict[str, Any]]) -> List[dbc.Toast]:
        """
        Update the notification container with current notifications.
        
        Args:
            notifications: List of notification data dictionaries
            
        Returns:
            List of Toast components to display
        """
        if not notifications:
            return []
        
        # Create a toast for each notification
        toasts = []
        for notification in notifications:
            toast = create_toast(
                message=notification.get("message", ""),
                notification_type=notification.get("type", "info"),
                header=notification.get("header"),
                id=notification.get("id"),
                duration=notification.get("duration", 4000)
            )
            toasts.append(toast)
        
        return toasts
    
    @app.callback(
        Output("notification-store", "data"),
        [
            Input({"type": "notification-toast", "index": ALL}, "n_dismiss"),
            Input("add-notification-trigger", "data")
        ],
        State("notification-store", "data"),
        prevent_initial_call=True,
        allow_duplicate=True  # Allow duplicate callbacks for notification store
    )
    def manage_notifications(
        dismissed: List[int], 
        new_notification: Dict[str, Any], 
        current_notifications: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Manage notifications - add new ones and remove dismissed ones.
        
        Args:
            dismissed: List of dismiss counts for each notification
            new_notification: Data for a new notification to add
            current_notifications: Current list of notifications
            
        Returns:
            Updated list of notifications
        """
        ctx = dash.callback_context
        triggered = ctx.triggered[0]["prop_id"] if ctx.triggered else ""
        
        # Initialize if None
        if current_notifications is None:
            current_notifications = []
        
        # Handle dismiss events
        if "n_dismiss" in triggered:
            # Get the indices of dismissed toasts
            indices = [
                trigger["index"]
                for trigger in ctx.triggered_id_indices
                if trigger.get("type") == "notification-toast"
            ]
            
            # Remove dismissed notifications
            current_notifications = [
                n for n in current_notifications
                if n.get("id") not in indices
            ]
        
        # Handle new notification
        elif "add-notification-trigger" in triggered and new_notification:
            # Add ID if not present
            if "id" not in new_notification:
                new_notification["id"] = str(uuid.uuid4())
            
            # Add timestamp
            new_notification["timestamp"] = time.time()
            
            # Add to notifications list
            current_notifications.append(new_notification)
        
        # Limit the number of notifications to avoid clutter
        if len(current_notifications) > 5:
            # Sort by timestamp and keep only the most recent
            current_notifications.sort(key=lambda n: n.get("timestamp", 0), reverse=True)
            current_notifications = current_notifications[:5]
        
        return current_notifications


def with_error_handling(callback_func):
    """
    Decorator for wrapping dashboard callbacks with error handling.
    
    Args:
        callback_func: The callback function to wrap
        
    Returns:
        Wrapped function with error handling
    """
    def wrapper(*args, **kwargs):
        try:
            return callback_func(*args, **kwargs)
        except Exception as e:
            logger.exception(f"Error in callback: {e}")
            
            # Return a notification of the error
            error_notification = {
                "type": "error",
                "message": f"Error: {str(e)}",
                "header": "Callback Error",
                "duration": 8000
            }
            
            # The wrapped function should return a tuple where the last element
            # is the notification trigger output
            result = list(callback_func(*args, **kwargs)) if callback_func(*args, **kwargs) else []
            
            # Replace the last element (notification trigger) with our error notification
            if result and len(result) > 0:
                result[-1] = error_notification
            else:
                result = [None] * (len(dash.callback_context.outputs) - 1) + [error_notification]
            
            return tuple(result)
    
    # Update wrapper function metadata
    wrapper.__name__ = callback_func.__name__
    return wrapper 