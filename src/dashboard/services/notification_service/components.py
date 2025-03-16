"""
UI components for the notification service
"""

from typing import Tuple
from dash import html, dcc
import dash_bootstrap_components as dbc


def create_notification_components() -> Tuple[dcc.Store, html.Div]:
    """
    Create the notification components for the dashboard.

    Returns:
        Tuple containing:
            - dcc.Store: Store for notification data
            - html.Div: Container for displaying notifications
    """
    # Create a store for notifications
    notification_store = dcc.Store(id="notification-store", data=[])

    # Create a container for notifications
    notification_container = html.Div(
        id="notification-container",
        style={
            "position": "fixed",
            "top": "20px",
            "right": "20px",
            "width": "350px",
            "maxWidth": "100%",
            "zIndex": "1000",
        },
    )

    return notification_store, notification_container
