"""
Dashboard Callbacks Module

This module registers and manages callbacks for the dashboard components.
It serves as a central registry for coordinating callback registration.
"""

from typing import Dict, Any, Optional, Callable, List
import dash
from loguru import logger


def register_all_callbacks(
    app: dash.Dash,
    get_system_status_func: Optional[Callable] = None,
) -> None:
    """
    Register all callbacks for the dashboard application.
    
    Args:
        app: The Dash application instance
        get_system_status_func: Function to get system status
    """
    logger.debug("Registering all dashboard callbacks")
    
    # Import and register component-specific callbacks
    from src.dashboard.components.main_layout import register_layout_callbacks
    from src.dashboard.components.trading_layout import register_trading_callbacks
    from src.dashboard.components.performance_layout import register_performance_callbacks
    from src.dashboard.components.settings_layout import register_settings_callbacks
    from src.dashboard.services.notification_service import register_notification_callbacks
    
    # Register component callbacks
    register_layout_callbacks(app, get_system_status_func)
    register_trading_callbacks(app)
    register_performance_callbacks(app)
    register_settings_callbacks(app)
    register_notification_callbacks(app)
    
    # Register system-level callbacks
    register_system_callbacks(app)
    
    logger.debug("All dashboard callbacks registered")


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
    def process_system_action(action: Dict[str, Any]) -> List:
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