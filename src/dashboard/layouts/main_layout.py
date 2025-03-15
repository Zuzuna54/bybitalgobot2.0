"""
Main Dashboard Layout Module

This module provides the main layout for the dashboard application.
"""

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from datetime import datetime
from loguru import logger

from src.dashboard.layouts.trading_layout import create_trading_panel
from src.dashboard.layouts.performance_layout import create_performance_panel
from src.dashboard.layouts.settings_layout import create_settings_panel
from src.dashboard.services.notification_service import create_notification_components


def create_header() -> html.Div:
    """
    Create the dashboard header.
    
    Returns:
        Dash HTML Div containing the header layout
    """
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H1("Bybit Algorithmic Trading Dashboard", className="dashboard-title")
            ], width=8),
            dbc.Col([
                html.Div([
                    html.Span("System Status: ", className="status-label"),
                    html.Span(id="system-status-badge", className="status-badge"),
                    html.Div(id="current-time", className="current-time")
                ], className="system-status-container")
            ], width=4)
        ]),
        html.Hr()
    ], className="dashboard-header")


def create_navigation() -> dbc.Tabs:
    """
    Create the main navigation tabs.
    
    Returns:
        Dash Bootstrap Tabs component
    """
    logger.debug("Creating navigation tabs")
    tabs = dbc.Tabs([
        dbc.Tab(label="Performance", tab_id="performance-tab"),
        dbc.Tab(label="Trading", tab_id="trading-tab"),
        dbc.Tab(label="Orderbook", tab_id="orderbook-tab"),
        dbc.Tab(label="Strategies", tab_id="strategy-tab"),
        dbc.Tab(label="Settings", tab_id="settings-tab"),
    ], id="tabs", active_tab="performance-tab")
    logger.debug("Navigation tabs created with performance-tab active by default")
    return tabs


def create_footer() -> html.Div:
    """
    Create the dashboard footer.
    
    Returns:
        Dash HTML Div containing the footer layout
    """
    return html.Div([
        html.Hr(),
        html.P([
            "Bybit Algorithmic Trading System © ", 
            str(datetime.now().year),
            " - ",
            html.Span(id="system-version", children="v1.0.0")
        ], className="text-center"),
        html.Div(id="system-status-info", className="system-status-info")
    ], className="dashboard-footer")


def create_tab_content() -> html.Div:
    """
    Create the container for tab content.
    
    Returns:
        Dash HTML Div containing the tab content container
    """
    logger.debug("Creating initial tab content containers")
    
    # Only create empty containers that will be populated by callbacks
    return html.Div([
        # Each tab content is initially hidden except performance tab
        html.Div(id="performance-content", style={"display": "block"}),
        html.Div(id="trading-content", style={"display": "none"}),
        html.Div(id="orderbook-content", style={"display": "none"}),
        html.Div(id="strategy-content", style={"display": "none"}),
        html.Div(id="settings-content", style={"display": "none"}),
        # Store the currently active tab
        dcc.Store(id="active-tab-store", data="performance-tab"),
    ], className="tab-content")


def create_dashboard_layout() -> html.Div:
    """
    Create the complete dashboard layout.
    
    Returns:
        Dash HTML Div containing the complete dashboard layout
    """
    # Create notification components
    notification_store, notification_container = create_notification_components()
    
    # Create main layout
    layout = html.Div([
        # URL location for routing
        dcc.Location(id="url", refresh=False),
        
        # Loading indicators, notifications, and hidden data stores
        html.Div([
            # Notification system
            notification_store,
            notification_container,
            
            # Store for app state
            dcc.Store(id="app-state", data={}),
            
            # System status
            dcc.Store(id="system-status-store", data={}),
            
            # Action result store
            dcc.Store(id="system-action-result", data=None),
            
            # Confirmation action store
            dcc.Store(id="confirmation-action", data=None),
            
            # Risk parameters result store
            dcc.Store(id="risk-parameters-result", data=None),
            
            # Error display store
            dcc.Store(id="error-data", data=None),
            
            # Notification trigger
            dcc.Store(id="add-notification-trigger", data=None),
            
            # Intervals
            dcc.Interval(id="status-update-interval", interval=5000),  # 5 seconds
            dcc.Interval(id="data-update-interval", interval=10000),   # 10 seconds
            dcc.Interval(id="clock-interval", interval=1000),          # 1 second
            
            # Modals
            create_confirmation_modal(),
            
            # Alerts
            dbc.Alert(
                "Action performed successfully",
                id="action-result-alert",
                color="success",
                dismissable=True,
                is_open=False,
                duration=4000,
                className="action-alert"
            ),
            dbc.Alert(
                "Risk parameters saved successfully",
                id="risk-parameters-save-alert",
                color="success",
                dismissable=True,
                is_open=False,
                duration=4000,
                className="action-alert"
            ),
        ], className="system-containers"),
        
        # Header
        create_header(),
        
        # Navigation
        create_navigation(),
        
        # Main content container
        html.Div([
            # Tab content
            create_tab_content(),
        ], id="page-content", className="main-content"),
        
        # Footer
        create_footer()
    ], className="dashboard-container")
    
    return layout


def create_confirmation_modal() -> dbc.Modal:
    """
    Create a confirmation modal for system actions.
    
    Returns:
        Dash Bootstrap Modal component
    """
    return dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Confirm Action", id="confirmation-modal-title")),
        dbc.ModalBody("Are you sure you want to perform this action?", id="confirmation-modal-body"),
        dbc.ModalFooter([
            dbc.Button(
                "Cancel", 
                id="cancel-action-button", 
                className="ms-auto", 
                color="secondary"
            ),
            dbc.Button(
                "Confirm", 
                id="confirm-action-button", 
                color="primary"
            ),
        ]),
    ], id="confirmation-modal", is_open=False)


def create_orderbook_panel():
    """Placeholder for orderbook panel until we implement it properly."""
    return html.Div([
        html.H3("Orderbook Analysis"),
        html.P("Orderbook visualization will be displayed here.")
    ])


def create_strategy_panel():
    """Placeholder for strategy panel until we implement it properly."""
    return html.Div([
        html.H3("Strategy Management"),
        html.P("Strategy configuration and monitoring will be displayed here.")
    ])


def register_layout_callbacks(app: dash.Dash, get_system_status_func) -> None:
    """
    Register callbacks for the main dashboard layout.
    
    Args:
        app: The Dash application instance
        get_system_status_func: Function to get system status
    """
    # Check for duplicate callback registrations
    logger.debug("Starting layout callback registrations")
    
    # Get the number of existing callbacks for detecting duplicates
    initial_callback_count = len(app.callback_map) if hasattr(app, 'callback_map') else 0
    logger.debug(f"Current callback count before layout callbacks: {initial_callback_count}")
    
    @app.callback(
        dash.Output("current-time", "children"),
        [dash.Input("clock-interval", "n_intervals")]
    )
    def update_time(n_intervals):
        """Update the current time display."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    @app.callback(
        [
            dash.Output("system-status-store", "data"),
            dash.Output("system-status-badge", "children"),
            dash.Output("system-status-badge", "className"),
            dash.Output("system-status-info", "children"),
            dash.Output("system-status-display", "children")
        ],
        [dash.Input("status-update-interval", "n_intervals")]
    )
    def update_system_status(n_intervals):
        """Update the system status display with the latest status."""
        if get_system_status_func is None:
            status_data = {
                "status": "Unknown",
                "details": "Status function not available",
                "is_running": False,
                "uptime": "0s",
                "mode": "unknown"
            }
        else:
            status_data = get_system_status_func()
        
        status = status_data.get("status", "Unknown")
        
        # Set badge class based on status
        if status == "Running":
            badge_class = "status-badge status-running"
        elif status == "Stopped":
            badge_class = "status-badge status-stopped"
        elif status == "Paused":
            badge_class = "status-badge status-paused"
        else:
            badge_class = "status-badge status-unknown"
        
        # Create status badge
        status_badge = html.Span(status)
        
        # Create system info
        details = status_data.get("details", "No details available")
        uptime = status_data.get("uptime", "0s")
        mode = status_data.get("mode", "Unknown")
        
        system_info = html.Div([
            html.Span(f"Mode: {mode} | "),
            html.Span(f"Uptime: {uptime} | "),
            html.Span(f"Status: {status} | "),
            html.Span(details)
        ])
        
        # Create system status display for settings panel
        system_display = html.Div([
            html.H5("System Status"),
            html.P([
                html.Strong("Status: "), status,
                html.Br(),
                html.Strong("Mode: "), mode,
                html.Br(),
                html.Strong("Uptime: "), uptime,
                html.Br(),
                html.Strong("Details: "), details
            ]),
            html.Div(id="system-status-display-extra")
        ])
        
        return status_data, status_badge, badge_class, system_info, system_display
    
    # Note: Tab switching callbacks have been moved to register_tab_switching_callbacks
    # to avoid duplicate callback registration
        
    # Log the final callback count to detect if all callbacks were registered
    final_callback_count = len(app.callback_map) if hasattr(app, 'callback_map') else 0
    new_callbacks = final_callback_count - initial_callback_count
    logger.debug(f"Added {new_callbacks} layout callbacks. Total callbacks: {final_callback_count}")


def register_tab_switching_callbacks(app: dash.Dash) -> None:
    """
    Register only the tab switching callbacks.
    
    Args:
        app: The Dash application instance
    """
    logger.debug("Starting tab switching callback registration")
    
    # Tab switching callback - only updates the active-tab-store
    @app.callback(
        dash.Output("active-tab-store", "data"),
        [dash.Input("tabs", "active_tab")],
        prevent_initial_call=True
    )
    def update_active_tab(active_tab):
        """Simply update the active tab store without side effects."""
        logger.debug(f"Updating active tab store: {active_tab}")
        return active_tab
    
    # Tab content visibility callback - separate from active tab store update
    @app.callback(
        [
            dash.Output("performance-content", "style"),
            dash.Output("trading-content", "style"),
            dash.Output("orderbook-content", "style"),
            dash.Output("strategy-content", "style"),
            dash.Output("settings-content", "style"),
        ],
        [dash.Input("active-tab-store", "data")],
        prevent_initial_call=True
    )
    def switch_tab_visibility(active_tab):
        """
        Switch between content tabs based on active tab.
        
        Args:
            active_tab: The ID of the active tab
            
        Returns:
            List of style dictionaries for each tab content
        """
        logger.debug(f"Tab visibility callback triggered with active_tab={active_tab}")
        
        tab_styles = {
            "performance-content": {"display": "none"},
            "trading-content": {"display": "none"},
            "orderbook-content": {"display": "none"},
            "strategy-content": {"display": "none"},
            "settings-content": {"display": "none"}
        }
        
        # Set the active tab to display
        tab_map = {
            "performance-tab": "performance-content",
            "trading-tab": "trading-content",
            "orderbook-tab": "orderbook-content",
            "strategy-tab": "strategy-content",
            "settings-tab": "settings-content"
        }
        
        active_content = tab_map.get(active_tab)
        if active_content:
            tab_styles[active_content] = {"display": "block"}
            logger.debug(f"Setting {active_content} to visible, others to hidden")
        else:
            logger.warning(f"Unknown tab ID: {active_tab}")
        
        return [
            tab_styles["performance-content"],
            tab_styles["trading-content"],
            tab_styles["orderbook-content"],
            tab_styles["strategy-content"],
            tab_styles["settings-content"]
        ]
    
    # Lazy-loading callbacks for tab content
    @app.callback(
        dash.Output("performance-content", "children"),
        [dash.Input("active-tab-store", "data")],
        prevent_initial_call=True
    )
    def load_performance_content(active_tab):
        """Load performance panel content when the performance tab is active."""
        logger.debug(f"Performance content loading callback triggered with active_tab={active_tab}")
        if active_tab == "performance-tab":
            logger.debug("Loading performance panel content")
            from src.dashboard.layouts.performance_layout import create_performance_panel
            return create_performance_panel()
        return dash.no_update
    
    @app.callback(
        dash.Output("trading-content", "children"),
        [dash.Input("active-tab-store", "data")],
        prevent_initial_call=True
    )
    def load_trading_content(active_tab):
        """Load trading panel content when the trading tab is active."""
        logger.debug(f"Trading content loading callback triggered with active_tab={active_tab}")
        if active_tab == "trading-tab":
            logger.debug("Loading trading panel content")
            from src.dashboard.components.trading_panel import create_trading_panel
            return create_trading_panel()
        logger.debug("Skipping trading content loading (not active tab)")
        return dash.no_update
    
    @app.callback(
        dash.Output("orderbook-content", "children"),
        [dash.Input("active-tab-store", "data")],
        prevent_initial_call=True
    )
    def load_orderbook_content(active_tab):
        """Load orderbook panel content when the orderbook tab is active."""
        logger.debug(f"Orderbook content loading callback triggered with active_tab={active_tab}")
        if active_tab == "orderbook-tab":
            logger.debug("Loading orderbook panel content")
            from src.dashboard.components.orderbook_panel import create_orderbook_panel
            return create_orderbook_panel()
        logger.debug("Skipping orderbook content loading (not active tab)")
        return dash.no_update
    
    @app.callback(
        dash.Output("strategy-content", "children"),
        [dash.Input("active-tab-store", "data")],
        prevent_initial_call=True
    )
    def load_strategy_content(active_tab):
        """Load strategy panel content when the strategy tab is active."""
        logger.debug(f"Strategy content loading callback triggered with active_tab={active_tab}")
        if active_tab == "strategy-tab":
            logger.debug("Loading strategy panel content")
            from src.dashboard.components.strategy_panel import create_strategy_panel
            return create_strategy_panel()
        logger.debug("Skipping strategy content loading (not active tab)")
        return dash.no_update
    
    @app.callback(
        dash.Output("settings-content", "children"),
        [dash.Input("active-tab-store", "data")],
        prevent_initial_call=True
    )
    def load_settings_content(active_tab):
        """Load settings panel content when the settings tab is active."""
        logger.debug(f"Settings content loading callback triggered with active_tab={active_tab}")
        if active_tab == "settings-tab":
            logger.debug("Loading settings panel content")
            from src.dashboard.layouts.settings_layout import create_settings_panel
            return create_settings_panel()
        logger.debug("Skipping settings content loading (not active tab)")
        return dash.no_update
    
    logger.debug("Tab switching callbacks registered successfully") 