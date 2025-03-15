"""
Dashboard Routes Module

This module defines URL routes and general navigation for the dashboard application.
"""

from typing import Dict, Any, Optional, Callable
import dash
from dash import html
from loguru import logger


def configure_routes(app: dash.Dash) -> None:
    """
    Configure URL routes for the dashboard application.
    
    Args:
        app: The Dash application instance
    """
    @app.callback(
        dash.Output("page-content", "children"),
        [dash.Input("url", "pathname")]
    )
    def display_page(pathname: str) -> html.Div:
        """
        Route to different pages based on URL pathname.
        
        Args:
            pathname: URL pathname
            
        Returns:
            Dashboard component to render
        """
        # Default to main dashboard
        if pathname == "/" or pathname == "/dashboard":
            from src.dashboard.layouts.main_layout import create_dashboard_layout
            return create_dashboard_layout()
        # Other routes can be added here
        # For example:
        # elif pathname == "/settings":
        #     return create_settings_page()
        else:
            # 404 page
            return html.Div([
                html.H1("404 - Page Not Found"),
                html.P(f"The path {pathname} was not found on this server."),
                html.A("Return to Dashboard", href="/")
            ])


def register_routes(app: dash.Dash) -> None:
    """
    Register URL routes for the dashboard application.
    
    Args:
        app: The Dash application instance
    """
    logger.debug("Registering dashboard routes")
    configure_routes(app) 