"""
Dashboard Callbacks Module

This module registers and manages callbacks for the dashboard components.
It serves as a central registry for coordinating callback registration.
"""

from typing import Dict, Any, Optional, Callable, List
import traceback
import dash
from loguru import logger

# Global data service reference
data_service = None


def initialize_callbacks(app: dash.Dash, dashboard_data_service=None) -> None:
    """
    Initialize all dashboard callbacks. This is the single entry point for callback registration.
    
    Args:
        app: The Dash application instance
        dashboard_data_service: The dashboard data service instance
    """
    global data_service
    data_service = dashboard_data_service
    
    # Register all callbacks
    register_all_callbacks(app)
    
    logger.info("All dashboard callbacks registered successfully")


def register_all_callbacks(app: dash.Dash) -> None:
    """
    Register all callbacks for the dashboard application.
    This function consolidates all callback registrations in one place.
    
    Args:
        app: The Dash application instance
    """
    logger.info("Beginning callback registrations for dashboard")
    
    # Helper function to log callback registration
    def log_callback_registration(callback_name):
        logger.info(f"Registering callback: {callback_name}")
    
    # Import layout callbacks
    from src.dashboard.layouts.main_layout import register_tab_switching_callbacks
    from src.dashboard.layouts.performance_layout import register_performance_callbacks
    from src.dashboard.layouts.trading_layout import register_trading_callbacks
    from src.dashboard.layouts.settings_layout import register_settings_callbacks
    
    # Import component callbacks
    from src.dashboard.components.orderbook import register_orderbook_callbacks
    from src.dashboard.components.strategy import register_strategy_callbacks
    from src.dashboard.components.market import register_market_callbacks
    
    # Import service callbacks
    from src.dashboard.services.notification_service import register_notification_callbacks
    
    # Register layout callbacks
    log_callback_registration("register_tab_switching_callbacks")
    register_tab_switching_callbacks(app)
    
    log_callback_registration("register_performance_callbacks")
    register_performance_callbacks(app)
    
    log_callback_registration("register_trading_callbacks")
    register_trading_callbacks(app)
    
    log_callback_registration("register_settings_callbacks")
    register_settings_callbacks(app)
    
    # Register component callbacks with data access functions
    log_callback_registration("register_orderbook_callbacks")
    register_orderbook_callbacks(app, get_orderbook_data)
    
    log_callback_registration("register_strategy_callbacks")
    register_strategy_callbacks(app, get_strategy_data, get_strategy_manager())
    
    log_callback_registration("register_market_callbacks")
    register_market_callbacks(app, get_market_data)
    
    # Register system and service callbacks
    log_callback_registration("register_system_callbacks")
    register_system_callbacks(app)
    
    log_callback_registration("register_notification_callbacks")
    register_notification_callbacks(app)
    
    logger.info("All dashboard callbacks registered successfully")


def register_system_callbacks(app: dash.Dash) -> None:
    """
    Register system-level callbacks for the dashboard.
    
    Args:
        app: The Dash application instance
    """
    logger.debug("Registering system callbacks")
    
    @app.callback(
        [dash.Output("action-result-alert", "is_open"),
         dash.Output("add-notification-trigger", "data")],
        [dash.Input("system-action-result", "data")],
        prevent_initial_call=True,
        allow_duplicate=True
    )
    def process_system_action(action):
        """
        Process system action results.
        
        Args:
            action: System action data
            
        Returns:
            List of outputs for callbacks
        """
        if not action:
            return False, None
        
        global data_service
        success = False
        notification = None
        
        try:
            if not data_service:
                raise ValueError("Data service is not initialized")
                
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
            success = False
            
        return True, notification
    
    # Register debug notification test callback
    @app.callback(
        dash.Output("test-notification-trigger", "data"),
        dash.Input("test-notification-button", "n_clicks"),
        prevent_initial_call=True,
        allow_duplicate=True
    )
    def trigger_test_notification(n_clicks):
        """
        Trigger a test notification for debugging purposes.
        
        Args:
            n_clicks: Button click count
            
        Returns:
            Notification data
        """
        if not n_clicks:
            return None
            
        return {
            "type": "info",
            "message": "This is a test notification",
            "header": "Test Notification",
            "duration": 3000
        }


# Data access wrapper functions for components
def get_orderbook_data(symbol=None, depth=None):
    """
    Get orderbook data from the data service.
    
    Args:
        symbol: Symbol to get orderbook data for
        depth: Depth of orderbook to retrieve
        
    Returns:
        Dictionary with orderbook data
    """
    try:
        global data_service
        if data_service:
            return data_service.get_orderbook_data(symbol, depth)
        else:
            logger.warning("Data service not initialized in get_orderbook_data")
            return {}
    except Exception as e:
        logger.error(f"Error in get_orderbook_data: {str(e)}")
        return {}


def get_market_data(symbol=None):
    """
    Get market data from the data service.
    
    Args:
        symbol: Symbol to get market data for
        
    Returns:
        Dictionary with market data
    """
    try:
        global data_service
        if data_service:
            return data_service.get_market_data(symbol)
        else:
            logger.warning("Data service not initialized in get_market_data")
            return {}
    except Exception as e:
        logger.error(f"Error in get_market_data: {str(e)}")
        return {}


def get_strategy_data(strategy_id=None):
    """
    Get strategy data from the data service.
    
    Args:
        strategy_id: Strategy ID to get data for
        
    Returns:
        Dictionary with strategy data
    """
    global data_service
    try:
        if data_service:
            return data_service.get_strategy_data(strategy_id)
        else:
            logger.warning("Data service not initialized in get_strategy_data")
            return {"error": "Data service not initialized"}
    except Exception as e:
        logger.error(f"Error in get_strategy_data: {str(e)}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}


def get_strategy_manager():
    """
    Get the strategy manager instance.
    
    Returns:
        Strategy manager instance or None
    """
    global data_service
    if data_service:
        return data_service.strategy_manager
    else:
        logger.warning("Data service not initialized in get_strategy_manager")
        return None 