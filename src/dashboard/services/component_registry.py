"""
Component Registry for Dashboard Integration

This module provides a registry for trading system components used by the dashboard.
"""

from typing import Dict, Any, Optional
import threading
from loguru import logger


class ComponentRegistry:
    """Registry for trading system components used by the dashboard."""

    def __init__(self):
        """Initialize the component registry."""
        self._components = {}
        self._lock = threading.RLock()  # Re-entrant lock for thread safety

    def register(self, name: str, component: Any) -> None:
        """
        Register a component with the registry.

        Args:
            name: Component name
            component: Component instance
        """
        with self._lock:
            self._components[name] = component

    def get(self, name: str) -> Optional[Any]:
        """
        Get a component from the registry.

        Args:
            name: Component name

        Returns:
            Component instance or None if not found
        """
        with self._lock:
            return self._components.get(name)

    def register_many(self, components: Dict[str, Any]) -> None:
        """
        Register multiple components at once.

        Args:
            components: Dictionary of component names and instances
        """
        with self._lock:
            for name, component in components.items():
                self._components[name] = component

    def list_components(self) -> Dict[str, str]:
        """
        Get a list of registered components with their types.

        Returns:
            Dictionary of component names and their types
        """
        with self._lock:
            return {
                name: type(component).__name__
                for name, component in self._components.items()
            }

    def is_registered(self, name: str) -> bool:
        """
        Check if a component is registered.

        Args:
            name: Component name

        Returns:
            True if the component is registered, False otherwise
        """
        with self._lock:
            return name in self._components
