"""
Dashboard Configuration Manager

This module provides access to the system configuration for dashboard components.
It re-exports the get_config_manager function from the main config package.
"""

from src.config.config_manager import get_config_manager, ConfigManager

# Re-export for backward compatibility
__all__ = ['get_config_manager', 'ConfigManager'] 