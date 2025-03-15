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
from src.dashboard.layouts.main_layout import register_tab_switching_callbacks
from src.dashboard.layouts.performance_layout import register_performance_callbacks
from src.dashboard.layouts.trading_layout import register_trading_callbacks
from src.dashboard.components.orderbook import register_orderbook_callbacks
from src.dashboard.components.strategy import register_strategy_callbacks
from src.dashboard.services.data_service import DashboardDataService
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
    
    # Set app layout
    app.layout = create_dashboard_layout()
    
    # Simplified logger function to help debug callbacks
    def log_callback_registration(callback_name):
        logger.info(f"Registering callback: {callback_name}")
    
    logger.info("Beginning callback registrations for dashboard")
    
    # Register callbacks with logging
    log_callback_registration("register_tab_switching_callbacks")
    register_tab_switching_callbacks(app)
    
    log_callback_registration("register_performance_callbacks")
    register_performance_callbacks(app)
    
    log_callback_registration("register_trading_callbacks")
    register_trading_callbacks(app)
    
    log_callback_registration("register_orderbook_callbacks")
    register_orderbook_callbacks(app, get_orderbook_data)
    
    log_callback_registration("register_strategy_callbacks")
    register_strategy_callbacks(app, get_strategy_data, strategy_manager)
    
    # Register system control callbacks
    log_callback_registration("register_system_callbacks")
    register_system_callbacks(app)
    
    # Register notification callbacks
    log_callback_registration("register_notification_callbacks")
    register_notification_callbacks(app)
    
    logger.info("All dashboard callbacks registered successfully")
    
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
    
    # Handle risk parameters save
    @app.callback(
        [dash.Output("risk-parameters-result", "data"),
         dash.Output("risk-parameters-save-alert", "is_open"),
         dash.Output("add-notification-trigger", "data", allow_duplicate=True)],
        [dash.Input("save-risk-parameters-button", "n_clicks")],
        [
            dash.State("position-size-slider", "value"),
            dash.State("max-drawdown-slider", "value"),
            dash.State("default-leverage-input", "value"),
            dash.State("max-positions-input", "value"),
            dash.State("stop-loss-atr-input", "value"),
            dash.State("risk-reward-input", "value"),
            dash.State("trailing-stop-toggle", "value"),
            dash.State("circuit-breaker-input", "value")
        ],
        prevent_initial_call=True
    )
    def save_risk_parameters(
        n_clicks, position_size, max_drawdown, default_leverage, 
        max_positions, stop_loss_atr, risk_reward, use_trailing_stop, circuit_breaker
    ):
        if not n_clicks:
            return "", False, None
        
        notification = None
        
        try:
            # Prepare risk parameters dictionary
            risk_params = {
                "position_size": position_size,
                "max_drawdown": max_drawdown,
                "default_leverage": default_leverage,
                "max_positions": max_positions,
                "stop_loss_atr": stop_loss_atr,
                "risk_reward": risk_reward,
                "use_trailing_stop": True if use_trailing_stop and use_trailing_stop[0] else False,
                "circuit_breaker": circuit_breaker
            }
            
            global data_service
            success = data_service.set_risk_parameters(risk_params)
            
            if success:
                notification = {
                    "message": "Risk parameters saved successfully",
                    "type": "success",
                    "header": "Risk Management"
                }
            else:
                notification = {
                    "message": "Failed to save risk parameters",
                    "type": "error",
                    "header": "Risk Management"
                }
            
            return "success" if success else "error", True, notification
        
        except Exception as e:
            logger.error(f"Error saving risk parameters: {str(e)}")
            logger.error(traceback.format_exc())
            notification = {
                "message": f"Error saving risk parameters: {str(e)}",
                "type": "error",
                "header": "Risk Management Error"
            }
            return "error", True, notification


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
    
    logger.info("Starting dashboard in standalone mode (with sample data)")
    
    # Run dashboard with minimal setup for standalone mode
    run_dashboard(
        orderbook_analyzer=orderbook_analyzer,
        host=args.host,
        port=args.port,
        debug=args.debug
    ) 