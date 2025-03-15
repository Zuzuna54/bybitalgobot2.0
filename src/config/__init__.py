"""
Configuration Package

This package provides utilities for loading and accessing system configuration.
"""

from .config_manager import get_config_manager, ConfigManager

__all__ = ['get_config_manager', 'ConfigManager'] 