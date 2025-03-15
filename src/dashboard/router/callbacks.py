"""
Dashboard Router Callback Registration

This module provides functions for registering dashboard callbacks.
"""

import dash
from loguru import logger
from typing import Dict, Any, List, Callable, Optional


def register_all_callbacks(app: dash.Dash, get_system_status_func=None) -> None:
    """
    Register all dashboard callbacks.
    
    Args:
        app: The Dash application instance
        get_system_status_func: Function to get the system status
    """
    logger.debug("Registering all dashboard callbacks")
    
    try:
        # Import tab-specific callbacks
        from src.dashboard.components.performance_panel import register_performance_callbacks
        from src.dashboard.layouts.settings_layout import register_settings_callbacks
        from src.dashboard.components.trading.callbacks import register_trading_callbacks
        from src.dashboard.components.orderbook.callbacks import register_orderbook_callbacks
        from src.dashboard.components.strategy.callbacks import register_strategy_callbacks
        
        # Import layout and service callbacks
        from src.dashboard.layouts.main_layout import register_tab_switching_callbacks
        from src.dashboard.services.notification_service import register_notification_callbacks
        
        # Get data accessor functions from the application
        get_performance_data = app.server.config.get('get_performance_data', lambda: {})
        get_trade_data = app.server.config.get('get_trade_data', lambda: {})
        get_orderbook_data = app.server.config.get('get_orderbook_data', lambda: {})
        get_strategy_data = app.server.config.get('get_strategy_data', lambda: {})
        
        # Register all callbacks with proper parameters and error handling
        logger.info("Registering tab switching callbacks")
        register_tab_switching_callbacks(app)
        
        logger.info("Registering performance callbacks")
        register_performance_callbacks(app, get_performance_data)
        
        logger.info("Registering trading callbacks")
        register_trading_callbacks(app, get_trade_data)
        
        logger.info("Registering orderbook callbacks")
        register_orderbook_callbacks(app, get_orderbook_data)
        
        logger.info("Registering strategy callbacks")
        register_strategy_callbacks(app, get_strategy_data)
        
        logger.info("Registering settings callbacks")
        register_settings_callbacks(app, get_system_status_func)
        
        logger.info("Registering notification callbacks")
        register_notification_callbacks(app)
        
        # Register system callbacks
        logger.info("Registering system callbacks")
        register_system_callbacks(app)
        
        logger.info("All dashboard callbacks registered successfully")
    
    except Exception as e:
        logger.error(f"Error registering callbacks: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise


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
        prevent_initial_call=True
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
        
        success = action.get("success", False)
        message = action.get("message", "")
        action_type = action.get("action", "")
        
        # Create notification
        notification = {
            "type": "success" if success else "error",
            "message": message,
            "header": f"System Action: {action_type}",
            "duration": 5000
        }
        
        return success, notification 