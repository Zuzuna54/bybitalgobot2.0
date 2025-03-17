#!/usr/bin/env python3
"""
Test script for component lifecycle management
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.component_lifecycle import (
    ComponentManager,
    ComponentStatus,
    initialize_component_system,
    component_manager,
)


def test_component_initialization():
    """Test basic component initialization"""
    print("Testing component lifecycle management...")

    # Create a new component manager for testing
    manager = ComponentManager()
    print(f"Test ComponentManager created: {manager is not None}")

    # Check component statuses
    statuses = [status for status in dir(ComponentStatus) if not status.startswith("_")]
    print(f"Component statuses: {statuses}")

    # Check the singleton instance
    print(f"Global component_manager exists: {component_manager is not None}")

    # Initialize component system
    initialize_component_system(register_signal_handlers=False)
    print("Component system initialized")

    # Register some test components
    def init_component_a():
        print("Component A initialized")
        return "Component A"

    def init_component_b():
        print("Component B initialized")
        return "Component B"

    # Register the components
    manager.register_component(
        name="component_a", init_method=init_component_a, dependencies=[]
    )

    manager.register_component(
        name="component_b", init_method=init_component_b, dependencies=["component_a"]
    )

    # Initialize all components
    print("\nInitializing components...")
    manager.initialize_all()

    # Get component instances
    component_a = manager.get_component("component_a")
    component_b = manager.get_component("component_b")

    print(f"Component A instance: {component_a}")
    print(f"Component B instance: {component_b}")

    # Get status report
    status_report = manager.get_status_report()
    print("\nComponent Status Report:")
    print(f"Timestamp: {status_report['timestamp']}")
    print(f"Initialization time: {status_report.get('initialization_time')} seconds")

    for name, comp_info in status_report["components"].items():
        print(f"\n  {name}:")
        print(f"    Status: {comp_info['status']}")
        print(f"    Optional: {comp_info['optional']}")
        print(f"    Dependencies: {comp_info['dependencies']}")
        print(f"    Init time: {comp_info['initialization_time']}")
        print(f"    Initialized at: {comp_info['initialized_at']}")

    # Shutdown all components
    print("\nShutting down components...")
    manager.shutdown_all()

    print("\nComponent lifecycle test completed successfully!")


if __name__ == "__main__":
    test_component_initialization()
