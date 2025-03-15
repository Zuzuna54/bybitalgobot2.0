"""
Main Dashboard Application for the Algorithmic Trading System

This module initializes and runs the dashboard application.
"""

import os
import json
import traceback
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
# Import the centralized callback registration function
from src.dashboard.router.callbacks import initialize_callbacks


# Initialize data provider and components
data_service = None


def initialize_dashboard(
    api_client=None,
    trade_manager=None,
    performance_tracker=None,
    risk_manager=None,
    strategy_manager=None,
    market_data=None,
    paper_trading=None,
    orderbook_analyzer=None
) -> dash.Dash:
    """
    Initialize the dashboard application.
    
    Args:
        api_client: Bybit API client
        trade_manager: Trade manager instance
        performance_tracker: Performance tracker instance
        risk_manager: Risk manager instance
        strategy_manager: Strategy manager instance
        market_data: Market data instance
        paper_trading: Paper trading simulator instance
        orderbook_analyzer: Order book analyzer instance
        
    Returns:
        Initialized Dash application
    """
    global data_service
    
    # Initialize data provider
    data_service = DashboardDataService(
        api_client=api_client,
        trade_manager=trade_manager,
        performance_tracker=performance_tracker,
        risk_manager=risk_manager,
        strategy_manager=strategy_manager,
        market_data=market_data,
        paper_trading=paper_trading,
        orderbook_analyzer=orderbook_analyzer
    )
    
    # Initialize Flask server
    server = Flask(__name__)
    
    # Create Dash app with debugging enabled
    app = dash.Dash(
        __name__,
        server=server,
        external_stylesheets=[
            dbc.themes.BOOTSTRAP,
            # Add FontAwesome for icons in notifications
            "https://use.fontawesome.com/releases/v5.15.4/css/all.css"
        ],
        suppress_callback_exceptions=True,
        assets_folder='assets'
    )
    
    # Set app title
    app.title = "Bybit Algorithmic Trading Dashboard"
    
    # Set app layout
    app.layout = create_dashboard_layout()
    
    # Register all callbacks through the centralized registration function
    initialize_callbacks(app, data_service)
    
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