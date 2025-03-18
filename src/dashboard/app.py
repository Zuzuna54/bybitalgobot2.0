"""
Main Dashboard Application for the Algorithmic Trading System

This module initializes and runs the dashboard application.
"""

import os
import json
import traceback
import threading
from typing import Dict, Any, Optional
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from flask import Flask

from loguru import logger

# Set logger level to DEBUG for detailed logging
logger.remove()
logger.add(lambda msg: print(msg, flush=True), level="DEBUG")

from src.dashboard.layouts.main_layout import create_dashboard_layout
from src.dashboard.services.data_service import DashboardDataService
from src.dashboard.services.component_registry import ComponentRegistry

# Import the centralized callback registration function
from src.dashboard.router.callbacks import initialize_callbacks


# Initialize data provider and components
data_service = None
component_registry = ComponentRegistry()
initialization_lock = threading.RLock()  # Re-entrant lock for thread safety


def initialize_dashboard(
    api_client=None,
    trade_manager=None,
    performance_tracker=None,
    risk_manager=None,
    strategy_manager=None,
    market_data=None,
    paper_trading=None,
    orderbook_analyzer=None,
    **components,  # Accept additional components through kwargs
) -> dash.Dash:
    """
    Initialize the dashboard application with thread-safety.

    Args:
        api_client: Bybit API client
        trade_manager: Trade manager instance
        performance_tracker: Performance tracker instance
        risk_manager: Risk manager instance
        strategy_manager: Strategy manager instance
        market_data: Market data instance
        paper_trading: Paper trading simulator instance
        orderbook_analyzer: Order book analyzer instance
        **components: Additional components to register

    Returns:
        Initialized Dash application
    """
    global data_service, component_registry

    with initialization_lock:
        try:
            # Register all provided components
            component_dict = {
                "api_client": api_client,
                "trade_manager": trade_manager,
                "performance_tracker": performance_tracker,
                "risk_manager": risk_manager,
                "strategy_manager": strategy_manager,
                "market_data": market_data,
                "paper_trading": paper_trading,
                "orderbook_analyzer": orderbook_analyzer,
            }

            # Add any additional components from kwargs
            component_dict.update(components)

            # Filter out None values
            component_dict = {k: v for k, v in component_dict.items() if v is not None}

            # Log registered components
            logger.info(
                f"Initializing dashboard with {len(component_dict)} components: {', '.join(component_dict.keys())}"
            )

            # Register components with the registry
            component_registry.register_many(component_dict)

            # Initialize data provider with component registry
            try:
                data_service = DashboardDataService(
                    component_registry=component_registry
                )
                logger.info("Dashboard data service initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing dashboard data service: {str(e)}")
                logger.debug(traceback.format_exc())
                # Continue with limited functionality
                data_service = DashboardDataService(
                    component_registry=component_registry
                )

            # Initialize Flask server with thread-safe config
            server = Flask(__name__)
            server.config["PROPAGATE_EXCEPTIONS"] = (
                True  # Ensure errors are properly propagated
            )

            # Create Dash app with thread-safe configuration
            app = dash.Dash(
                __name__,
                server=server,
                external_stylesheets=[
                    dbc.themes.BOOTSTRAP,
                    # Add FontAwesome for icons in notifications
                    "https://use.fontawesome.com/releases/v5.15.4/css/all.css",
                ],
                suppress_callback_exceptions=True,
                assets_folder="assets",
                meta_tags=[
                    {
                        "name": "viewport",
                        "content": "width=device-width, initial-scale=1",
                    }
                ],
            )

            # Set app title
            app.title = "Bybit Algorithmic Trading Dashboard"

            # Set app layout
            app.layout = create_dashboard_layout()

            # Register all callbacks through the centralized registration function
            try:
                initialize_callbacks(app, data_service)
                logger.info("Dashboard callbacks initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing dashboard callbacks: {str(e)}")
                logger.debug(traceback.format_exc())
                # Continue with limited callback functionality

            return app
        except Exception as e:
            logger.error(f"Critical error during dashboard initialization: {str(e)}")
            logger.debug(traceback.format_exc())

            # Create a minimal fallback app in case of severe errors
            server = Flask(__name__)
            app = dash.Dash(__name__, server=server)
            app.layout = html.Div(
                [
                    html.H1("Dashboard Error"),
                    html.P(f"Error initializing dashboard: {str(e)}"),
                    html.P("Please check the logs for more information."),
                ]
            )
            return app


# All callback registrations have been moved to router/callbacks.py


def get_orderbook_data(symbol=None, depth=None):
    """
    Helper function to get orderbook data for component callbacks.

    Args:
        symbol: Trading symbol
        depth: Orderbook depth

    Returns:
        Orderbook data dictionary
    """
    global data_service
    if data_service:
        return data_service.get_orderbook_data(symbol, depth)
    return {"error": "Data service not initialized"}


def get_strategy_data(strategy_id=None):
    """
    Helper function to get strategy data for component callbacks.

    Args:
        strategy_id: Strategy identifier

    Returns:
        Strategy data dictionary
    """
    global data_service
    if data_service:
        return data_service.get_strategy_data(strategy_id)
    return {"error": "Data service not initialized"}


def run_dashboard(debug=False, host="0.0.0.0", port=8050):
    """
    Run the dashboard application in standalone mode.

    Args:
        debug: Enable debug mode
        host: Host to bind server to
        port: Port to bind server to
    """
    logger.info(f"Starting dashboard in standalone mode on {host}:{port}")

    # Initialize the dashboard in standalone mode
    app = initialize_dashboard()

    # Run the server
    app.run_server(debug=debug, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the trading dashboard")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind server to")
    parser.add_argument("--port", type=int, default=8050, help="Port to bind server to")

    args = parser.parse_args()

    run_dashboard(debug=args.debug, host=args.host, port=args.port)
