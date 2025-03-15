"""
Loading Indicators for Dashboard Components

This module provides loading indicator components to be used for long-running operations.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc


def create_loading_container(component, id=None, type="circle", color="#119DFF"):
    """
    Wrap a component with a loading indicator.
    
    Args:
        component: The component to wrap
        id (str, optional): Optional ID for the loading component
        type (str): Type of loading indicator (circle, dot, default, grow)
        color (str): Color of the loading indicator
        
    Returns:
        dcc.Loading: A loading component wrapping the provided component
    """
    return dcc.Loading(
        id=id,
        children=component,
        type=type,
        color=color,
    )


def create_fullscreen_loading():
    """
    Create a fullscreen loading overlay.
    
    Returns:
        html.Div: A fullscreen loading overlay
    """
    return html.Div(
        id="fullscreen-loading",
        className="fullscreen-loading",
        style={
            "display": "none",
            "position": "fixed",
            "top": 0,
            "left": 0,
            "width": "100%",
            "height": "100%",
            "backgroundColor": "rgba(0, 0, 0, 0.5)",
            "zIndex": 9999,
            "display": "flex",
            "justifyContent": "center",
            "alignItems": "center",
        },
        children=[
            dbc.Spinner(
                color="primary",
                size="lg",
            ),
            html.Div(
                "Loading...",
                style={
                    "color": "white",
                    "marginTop": "20px",
                    "fontSize": "20px",
                }
            )
        ]
    )


def create_overlay_loading(message="Processing..."):
    """
    Create a semi-transparent loading overlay with a message.
    
    Args:
        message (str): Message to display during loading
        
    Returns:
        html.Div: A loading overlay with a message
    """
    return html.Div(
        className="overlay-loading",
        style={
            "position": "absolute",
            "top": 0,
            "left": 0,
            "width": "100%",
            "height": "100%",
            "backgroundColor": "rgba(255, 255, 255, 0.8)",
            "zIndex": 100,
            "display": "flex",
            "flexDirection": "column",
            "justifyContent": "center",
            "alignItems": "center",
        },
        children=[
            dbc.Spinner(
                color="primary",
                size="lg",
            ),
            html.Div(
                message,
                style={
                    "marginTop": "15px",
                    "fontWeight": "bold",
                }
            )
        ]
    )


def create_button_loading_wrapper(button, loading_state_id):
    """
    Create a wrapper for a button with a loading state.
    
    Args:
        button: The button component to wrap
        loading_state_id (str): ID to track the loading state
        
    Returns:
        html.Div: A div containing the button and a loading indicator
    """
    return html.Div(
        style={"position": "relative"},
        children=[
            button,
            dbc.Spinner(
                id=f"{loading_state_id}-spinner",
                color="primary",
                size="sm",
                style={
                    "position": "absolute",
                    "top": "50%",
                    "right": "10px",
                    "transform": "translateY(-50%)",
                    "display": "none",
                }
            )
        ]
    ) 