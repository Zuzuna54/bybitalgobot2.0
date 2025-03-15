"""
Main Dashboard Application for the Algorithmic Trading System

This module initializes and runs the dashboard application.
"""

import os
import json
import traceback
from typing import Dict, Any, Optional, Callable
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from flask import Flask
import numpy as np
from datetime import datetime

from loguru import logger

# Set logger level to DEBUG for detailed logging
logger.remove()
logger.add(lambda msg: print(msg, flush=True), level="DEBUG")

# Import the router modules
try:
    from src.dashboard.router.routes import register_routes
    from src.dashboard.router.callbacks import register_all_callbacks
except ImportError as e:
    logger.warning(f"Router module import failed: {str(e)}")

# Import specific components and layouts
from src.dashboard.layouts.main_layout import create_dashboard_layout, register_tab_switching_callbacks
from src.dashboard.services.data_service import DashboardDataService

# Import only the callbacks that aren't covered by the router
from src.dashboard.components.performance_panel import register_performance_callbacks
from src.dashboard.components.trading.callbacks import register_trading_callbacks
from src.dashboard.components.orderbook.callbacks import register_orderbook_callbacks
from src.dashboard.components.strategy.callbacks import register_strategy_callbacks
from src.dashboard.layouts.settings_layout import register_settings_callbacks

from src.dashboard.services.notification_service import register_notification_callbacks


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
    
    # Store data accessor functions in app config for router to access
    app.server.config['get_performance_data'] = get_performance_data
    app.server.config['get_trade_data'] = get_trade_data
    app.server.config['get_orderbook_data'] = get_orderbook_data
    app.server.config['get_strategy_data'] = get_strategy_data
    app.server.config['get_system_status'] = get_system_status
    
    # Set app layout
    app.layout = create_dashboard_layout()
    
    # Logger function to help debug callbacks
    def log_callback_registration(callback_name):
        logger.info(f"Registering callback: {callback_name}")
    
    logger.info("Beginning callback registrations for dashboard")
    
    # Use the router's callback registration system exclusively
    try:
        # Only proceed if the imports were successful
        if 'register_routes' in globals() and 'register_all_callbacks' in globals():
            logger.info("Using router module for callback registration")
            # Register routes
            register_routes(app)
            
            # Register all callbacks through the router
            register_all_callbacks(app, get_system_status)
            
            logger.info("Successfully registered callbacks using router module")
        else:
            logger.error("Router modules not properly imported - dashboard may not function correctly")
    except Exception as e:
        # Log the error but don't fall back to direct registration to avoid duplicates
        logger.error(f"Router callback registration failed: {str(e)}")
        logger.error(traceback.format_exc())
        logger.error("Dashboard callbacks failed to register - functionality will be limited")
    
    logger.info("Dashboard initialization completed")
    
    return app


def register_system_callbacks(app):
    """Register callbacks for system control and settings."""
    
    # Handle confirmed system actions
    @app.callback(
        [dash.Output("action-result-alert", "is_open"),
         dash.Output("add-notification-trigger", "data")],
        [dash.Input("system-action-result", "data")],
        prevent_initial_call=True
    )
    def process_system_action(action):
        if not action:
            return False, None
        
        global data_service
        success = False
        notification = None
        
        try:
            if action == "start":
                success = data_service.start_trading()
                message = "Trading system started" if success else "Failed to start trading system"
                notification_type = "success" if success else "error"
                notification = {
                    "message": message,
                    "type": notification_type,
                    "header": "System Control"
                }
            elif action == "stop":
                success = data_service.stop_trading()
                message = "Trading system stopped" if success else "Failed to stop trading system"
                notification_type = "success" if success else "error"
                notification = {
                    "message": message,
                    "type": notification_type,
                    "header": "System Control"
                }
            elif action == "pause":
                success = data_service.pause_trading()
                message = "Trading system paused" if success else "Failed to pause trading system"
                notification_type = "success" if success else "error"
                notification = {
                    "message": message,
                    "type": notification_type,
                    "header": "System Control"
                }
            elif action == "reset":
                success = data_service.reset_paper_trading()
                message = "Paper trading reset successfully" if success else "Failed to reset paper trading"
                notification_type = "success" if success else "error"
                notification = {
                    "message": message,
                    "type": notification_type,
                    "header": "System Control"
                }
        except Exception as e:
            logger.error(f"Error during system action {action}: {str(e)}")
            logger.error(traceback.format_exc())
            notification = {
                "message": f"Error during {action}: {str(e)}",
                "type": "error",
                "header": "System Error"
            }
            
        return True, notification
    
    # Handle debug notification test
    @app.callback(
        dash.Output("add-notification-trigger", "data", allow_duplicate=True),
        [dash.Input("debug-show-notification", "n_clicks")],
        [
            dash.State("debug-notification-type", "value"),
            dash.State("debug-notification-message", "value"),
            dash.State("debug-notification-duration", "value")
        ],
        prevent_initial_call=True
    )
    def show_test_notification(n_clicks, notification_type, message, duration):
        """Show a test notification from the debug panel."""
        if not n_clicks:
            return dash.no_update
        
        notification = {
            "message": message,
            "type": notification_type,
            "header": f"Test {notification_type.capitalize()} Notification",
            "duration": duration
        }
        
        return notification
    
    # Handle debug error simulation
    @app.callback(
        dash.Output("add-notification-trigger", "data", allow_duplicate=True),
        [dash.Input("debug-simulate-error", "n_clicks")],
        prevent_initial_call=True
    )
    def simulate_error(n_clicks):
        """Simulate an error for testing error handling."""
        if not n_clicks:
            return dash.no_update
        
        try:
            # Deliberately cause an error
            result = 1 / 0
        except Exception as e:
            logger.error(f"Simulated error: {str(e)}")
            logger.error(traceback.format_exc())
            
            notification = {
                "message": f"This is a simulated error: {str(e)}",
                "type": "error",
                "header": "Simulated Error",
                "duration": 6000
            }
            
            return notification
        
        return dash.no_update
    
    # Simulate long operation with loading indicator
    @app.callback(
        [
            dash.Output("debug-loading-test", "children"),
            dash.Output("add-notification-trigger", "data", allow_duplicate=True)
        ],
        [dash.Input("debug-loading-test", "n_clicks")],
        prevent_initial_call=True
    )
    def simulate_long_operation(n_clicks):
        """Simulate a long-running operation with loading state."""
        if not n_clicks:
            return dash.no_update, dash.no_update
        
        # Simulate a long operation
        import time
        time.sleep(3)
        
        notification = {
            "message": "Long operation completed successfully after 3 seconds",
            "type": "success",
            "header": "Operation Complete"
        }
        
        return "Simulate Long Operation", notification
    
    # Risk parameters callback has been moved to settings_layout.py to avoid duplication


def run_dashboard(
    api_client=None,
    trade_manager=None,
    performance_tracker=None,
    risk_manager=None,
    strategy_manager=None,
    market_data=None,
    paper_trading=None,
    orderbook_analyzer=None,
    host: str = "0.0.0.0",
    port: int = 8050,
    debug: bool = False
) -> None:
    """
    Initialize and run the dashboard.
    
    Args:
        api_client: Bybit API client
        trade_manager: Trade manager instance
        performance_tracker: Performance tracker instance
        risk_manager: Risk manager instance
        strategy_manager: Strategy manager instance
        market_data: Market data instance
        paper_trading: Paper trading simulator instance
        orderbook_analyzer: Order book analyzer instance
        host: Host to run the server on
        port: Port to run the server on
        debug: Whether to run in debug mode
    """
    app = initialize_dashboard(
        api_client=api_client,
        trade_manager=trade_manager,
        performance_tracker=performance_tracker,
        risk_manager=risk_manager,
        strategy_manager=strategy_manager,
        market_data=market_data,
        paper_trading=paper_trading,
        orderbook_analyzer=orderbook_analyzer
    )
    
    logger.info(f"Starting dashboard on http://{host}:{port}")
    
    # Run app
    app.run_server(host=host, port=port, debug=debug)


def shutdown_dashboard() -> None:
    """Shutdown the dashboard gracefully."""
    global data_service
    
    if data_service:
        # Stop data provider
        data_service.stop()
    
    logger.info("Dashboard shut down")


# Data provider getter functions used by callback registrations
def get_performance_data() -> Dict[str, Any]:
    """Get performance data for the dashboard."""
    global data_service
    if data_service:
        return data_service.get_performance_data()
    return {}


def get_trade_data() -> Dict[str, Any]:
    """Get trade data for the dashboard."""
    global data_service
    if data_service:
        return data_service.get_trade_data()
    return {}


def get_orderbook_data() -> Dict[str, Any]:
    """Get order book data for the dashboard."""
    global data_service
    if data_service:
        return data_service.get_orderbook_data()
    return {}


def get_strategy_data() -> Dict[str, Any]:
    """Get strategy data for the dashboard."""
    global data_service
    if data_service:
        return data_service.get_strategy_data()
    return {}


def get_system_status() -> Dict[str, Any]:
    """Get system status for the dashboard."""
    global data_service
    if data_service:
        return data_service.get_system_status()
    return {
        "is_running": False,
        "is_paused": False,
        "components": {},
        "data_freshness": {}
    }


# Entry point for running the dashboard standalone
if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the trading dashboard")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8050, help="Port to run the server on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    args = parser.parse_args()
    
    # Load example data for standalone mode (will be replaced with real data in integrated mode)
    from src.trade_execution.orderbook_analyzer import OrderBookAnalyzer
    
    # Create a sample orderbook analyzer
    orderbook_analyzer = OrderBookAnalyzer()
    
    # Add sample data for common symbols to ensure visualization works
    sample_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "BNBUSDT"]
    
    for symbol in sample_symbols:
        # Create sample orderbook data
        if symbol == "BTCUSDT":
            mid_price = 27500.0
        elif symbol == "ETHUSDT":
            mid_price = 1800.0
        elif symbol == "SOLUSDT":
            mid_price = 20.0
        elif symbol == "DOGEUSDT":
            mid_price = 0.07
        elif symbol == "BNBUSDT":
            mid_price = 300.0
        else:
            mid_price = 100.0
        
        # Create bid and ask prices around the mid price
        spread = mid_price * 0.0005  # 0.05% spread
        best_bid = mid_price - spread/2
        best_ask = mid_price + spread/2
        
        # Create sample bids and asks
        bids = []
        asks = []
        
        # Generate sample bids (sorted by price descending)
        for i in range(10):
            price = best_bid * (1 - 0.0005 * i)
            size = np.random.uniform(0.1, 2.0) if symbol == "BTCUSDT" else np.random.uniform(1, 20)
            bids.append([str(price), str(size)])
        
        # Generate sample asks (sorted by price ascending)
        for i in range(10):
            price = best_ask * (1 + 0.0005 * i)
            size = np.random.uniform(0.1, 2.0) if symbol == "BTCUSDT" else np.random.uniform(1, 20)
            asks.append([str(price), str(size)])
        
        # Add to the analyzer
        orderbook_data = {
            "bids": bids,
            "asks": asks,
            "timestamp": datetime.now().timestamp()
        }
        orderbook_analyzer.update_orderbook(symbol, orderbook_data)
    
    logger.info("Starting dashboard in standalone mode (with sample data)")
    
    # Run dashboard with minimal setup for standalone mode
    run_dashboard(
        orderbook_analyzer=orderbook_analyzer,
        host=args.host,
        port=args.port,
        debug=args.debug
    ) 