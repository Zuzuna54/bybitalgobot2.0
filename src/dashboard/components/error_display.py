"""
Standardized Error Display Components

This module provides standardized components for displaying errors in the dashboard.
"""

from dash import html
import dash_bootstrap_components as dbc


def create_error_message(message, title=None, error_details=None, show_details=False):
    """
    Create a standardized error message component.
    
    Args:
        message (str): The error message to display
        title (str, optional): Optional title for the error
        error_details (str, optional): Optional technical details about the error
        show_details (bool, optional): Whether to initially show error details
        
    Returns:
        dbc.Card: A card component displaying the error message
    """
    if title is None:
        title = "An Error Occurred"
        
    components = [
        html.Div([
            html.I(className="fas fa-exclamation-circle text-danger me-2"),
            html.Span(title, className="h5 text-danger")
        ], className="d-flex align-items-center"),
        html.Hr(),
        html.P(message, className="my-2")
    ]
    
    if error_details is not None:
        details_id = "error-details-collapse"
        components.extend([
            html.Hr(),
            dbc.Button(
                "Show Technical Details" if not show_details else "Hide Technical Details",
                id="toggle-error-details",
                color="link",
                size="sm",
                className="p-0 text-decoration-none"
            ),
            dbc.Collapse(
                dbc.Card(
                    dbc.CardBody(
                        html.Pre(error_details, className="text-muted small mb-0")
                    ),
                    className="mt-2 bg-light"
                ),
                id=details_id,
                is_open=show_details
            )
        ])
    
    return dbc.Card(
        dbc.CardBody(components),
        className="border-danger mb-3"
    )


def create_validation_feedback(message, state="invalid"):
    """
    Create validation feedback for form inputs.
    
    Args:
        message (str): The validation message
        state (str): Either "valid" or "invalid"
        
    Returns:
        html.Div: A validation feedback component
    """
    class_name = "valid-feedback" if state == "valid" else "invalid-feedback"
    
    return html.Div(
        message,
        className=class_name
    )


def create_empty_state(message, icon_name="folder-open", action_button=None):
    """
    Create an empty state display when no data is available.
    
    Args:
        message (str): Message to display
        icon_name (str): Name of the FontAwesome icon to display
        action_button (component, optional): Optional button for actions
        
    Returns:
        dbc.Card: A card component displaying the empty state
    """
    components = [
        html.Div([
            html.I(className=f"fas fa-{icon_name} fa-3x text-muted")
        ], className="text-center mb-3"),
        html.P(message, className="text-center text-muted")
    ]
    
    if action_button is not None:
        components.append(
            html.Div([
                action_button
            ], className="text-center mt-3")
        )
    
    return dbc.Card(
        dbc.CardBody(components),
        className="border-light bg-light-subtle py-5"
    )


def create_status_indicator(status, message=None):
    """
    Create a status indicator component that shows the state of an operation.
    
    Args:
        status (str): Status type (success, warning, danger, info)
        message (str, optional): Optional message to display
        
    Returns:
        html.Div: A status indicator component
    """
    icon_map = {
        "success": "check-circle",
        "warning": "exclamation-triangle",
        "danger": "times-circle",
        "info": "info-circle"
    }
    
    icon = icon_map.get(status, "question-circle")
    
    return html.Div([
        html.I(className=f"fas fa-{icon} text-{status} me-2"),
        html.Span(message if message else status.capitalize())
    ], className="d-flex align-items-center") 