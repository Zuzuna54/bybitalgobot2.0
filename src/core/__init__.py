"""
Core system module containing common utilities and base functionality.
"""

# Export component lifecycle management functionality
from src.core.component_lifecycle import (
    initialize_component_system,
    component_manager,
    register_component,
    register_shutdown_handler,
    ComponentStatus,
)

__all__ = [
    "initialize_component_system",
    "component_manager",
    "register_component",
    "register_shutdown_handler",
    "ComponentStatus",
]
