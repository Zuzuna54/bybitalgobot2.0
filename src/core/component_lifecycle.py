"""
Component Lifecycle Management

This module provides functionality for managing the lifecycle of system components,
including initialization order, dependency validation, and graceful shutdown procedures.
"""

import os
import signal
import importlib
from typing import Dict, Any, List, Optional, Callable, Set, Tuple
from datetime import datetime
import time
from functools import wraps
from pathlib import Path

from loguru import logger


class ComponentStatus:
    """Component status enumeration."""

    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    FAILED = "failed"
    SHUTTING_DOWN = "shutting_down"
    SHUTDOWN = "shutdown"


class Component:
    """Represents a system component with lifecycle information."""

    def __init__(
        self,
        name: str,
        init_method: Callable,
        dependencies: List[str] = None,
        optional: bool = False,
    ):
        """
        Initialize a component.

        Args:
            name: Component name
            init_method: Initialization method for the component
            dependencies: List of component names this component depends on
            optional: Whether this component is optional
        """
        self.name = name
        self.init_method = init_method
        self.dependencies = dependencies or []
        self.optional = optional
        self.status = ComponentStatus.UNINITIALIZED
        self.instance = None
        self.error = None
        self.initialization_time = None
        self.initialized_at = None  # Timestamp when the component was initialized

    def initialize(self) -> bool:
        """
        Initialize the component.

        Returns:
            True if initialization was successful, False otherwise
        """
        if self.status == ComponentStatus.INITIALIZED:
            return True

        self.status = ComponentStatus.INITIALIZING
        start_time = time.time()

        try:
            self.instance = self.init_method()
            self.status = ComponentStatus.INITIALIZED
            self.initialization_time = time.time() - start_time
            self.initialized_at = datetime.now()
            logger.info(
                f"Component '{self.name}' initialized in {self.initialization_time:.3f}s"
            )
            return True
        except Exception as e:
            self.error = str(e)
            self.status = ComponentStatus.FAILED
            if self.optional:
                logger.warning(
                    f"Optional component '{self.name}' failed to initialize: {e}"
                )
                return False
            else:
                logger.error(
                    f"Required component '{self.name}' failed to initialize: {e}"
                )
                raise RuntimeError(
                    f"Failed to initialize required component '{self.name}': {e}"
                )


class ComponentManager:
    """
    Manages the lifecycle of system components.

    The ComponentManager ensures that components are initialized in the correct order
    based on their dependencies, validates that all required components are available,
    and provides a graceful shutdown mechanism.
    """

    def __init__(self):
        """Initialize the component manager."""
        self.components: Dict[str, Component] = {}
        self.initialized: Set[str] = set()
        self.shutdown_handlers: Dict[str, Callable] = {}
        self.is_shutting_down = False
        self.initialization_start_time = None
        self.initialization_end_time = None

    def register_component(
        self,
        name: str,
        init_method: Callable,
        dependencies: List[str] = None,
        optional: bool = False,
    ) -> None:
        """
        Register a component with the manager.

        Args:
            name: Component name
            init_method: Initialization method for the component
            dependencies: List of component names this component depends on
            optional: Whether this component is optional
        """
        # Validate dependencies exist
        if dependencies:
            for dep in dependencies:
                if dep not in self.components and dep != name:
                    logger.warning(
                        f"Component '{name}' depends on unregistered component '{dep}'"
                    )

        self.components[name] = Component(
            name=name,
            init_method=init_method,
            dependencies=dependencies or [],
            optional=optional,
        )

    def register_shutdown_handler(self, name: str, handler: Callable) -> None:
        """
        Register a shutdown handler for a component.

        Args:
            name: Component name
            handler: Shutdown handler method
        """
        if name not in self.components:
            logger.warning(
                f"Registering shutdown handler for unregistered component '{name}'"
            )

        self.shutdown_handlers[name] = handler

    def initialize_all(self) -> bool:
        """
        Initialize all registered components in dependency order.

        Returns:
            True if all required components were initialized, False otherwise
        """
        self.initialization_start_time = datetime.now()
        logger.info("Starting component initialization sequence")

        # Reset initialized set
        self.initialized = set()

        # Find the correct initialization order based on dependencies
        initialization_order = self._get_initialization_order()
        logger.info(
            f"Component initialization order: {', '.join(initialization_order)}"
        )

        # Initialize components in order
        for name in initialization_order:
            component = self.components[name]

            # Check if dependencies are initialized
            missing_deps = [
                dep for dep in component.dependencies if dep not in self.initialized
            ]
            if missing_deps:
                if component.optional:
                    logger.warning(
                        f"Skipping optional component '{name}' due to missing dependencies: {missing_deps}"
                    )
                    continue
                else:
                    error_msg = f"Cannot initialize component '{name}': missing dependencies {missing_deps}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)

            # Initialize the component
            try:
                logger.info(f"Initializing component '{name}'...")
                if component.initialize():
                    self.initialized.add(name)
                    logger.info(f"Component '{name}' successfully initialized")
            except Exception as e:
                if component.optional:
                    logger.warning(
                        f"Optional component '{name}' failed to initialize: {e}"
                    )
                else:
                    logger.error(f"Failed to initialize required component '{name}'")
                    self.initialization_end_time = datetime.now()
                    raise

        # Validate that all required components are initialized
        self._validate_required_components()

        self.initialization_end_time = datetime.now()
        total_time = (
            self.initialization_end_time - self.initialization_start_time
        ).total_seconds()
        logger.info(f"Component initialization completed in {total_time:.3f}s")

        return True

    def _validate_required_components(self) -> bool:
        """
        Validate that all required components are initialized.

        Returns:
            True if all required components are initialized, False otherwise

        Raises:
            RuntimeError: If any required component is not initialized
        """
        missing_required = [
            name
            for name, component in self.components.items()
            if not component.optional and name not in self.initialized
        ]

        if missing_required:
            error_msg = f"Failed to initialize required components: {missing_required}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        return True

    def _get_initialization_order(self) -> List[str]:
        """
        Determine the correct order to initialize components based on dependencies.

        Returns:
            List of component names in initialization order

        Raises:
            RuntimeError: If circular dependencies are detected
        """
        # Create a copy of the dependency graph
        dependency_graph = {
            name: component.dependencies.copy()
            for name, component in self.components.items()
        }

        # Find initialization order (topological sort)
        initialization_order = []
        no_dependencies = [name for name, deps in dependency_graph.items() if not deps]

        while no_dependencies:
            # Add a node with no dependencies to the order
            name = no_dependencies.pop(0)
            initialization_order.append(name)

            # Remove this node from the dependency lists of other nodes
            for other_name, deps in list(dependency_graph.items()):
                if name in deps:
                    deps.remove(name)
                    # If this node now has no dependencies, add it to the list
                    if (
                        not deps
                        and other_name not in initialization_order
                        and other_name not in no_dependencies
                    ):
                        no_dependencies.append(other_name)

        # Check for circular dependencies
        remaining_deps = {name: deps for name, deps in dependency_graph.items() if deps}
        if remaining_deps:
            error_msg = f"Circular dependencies detected: {remaining_deps}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        return initialization_order

    def get_component(self, name: str) -> Any:
        """
        Get a component instance by name.

        Args:
            name: Component name

        Returns:
            Component instance or None if not found or not initialized

        Raises:
            ValueError: If component is not registered
            RuntimeError: If component is required but not initialized
        """
        if name not in self.components:
            raise ValueError(f"Component '{name}' not registered")

        component = self.components[name]

        if name not in self.initialized:
            if component.optional:
                logger.warning(f"Optional component '{name}' not initialized")
                return None
            else:
                raise RuntimeError(f"Required component '{name}' not initialized")

        return component.instance

    def validate_dependencies(self) -> bool:
        """
        Validate dependencies between components to ensure all are satisfied.

        Returns:
            True if all dependencies are satisfied, False otherwise

        Raises:
            RuntimeError: If any required dependency is not satisfied
        """
        for name, component in self.components.items():
            # Skip checking dependencies for optional uninitialized components
            if component.optional and name not in self.initialized:
                continue

            # For initialized components, check their dependencies
            if name in self.initialized:
                missing_deps = [
                    dep for dep in component.dependencies if dep not in self.initialized
                ]
                if missing_deps:
                    error_msg = f"Component '{name}' has unsatisfied dependencies: {missing_deps}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)

        return True

    def shutdown_all(self) -> None:
        """
        Perform graceful shutdown of all initialized components.

        Components are shut down in reverse initialization order to ensure dependencies
        are respected.
        """
        if self.is_shutting_down:
            logger.warning("Shutdown already in progress")
            return

        self.is_shutting_down = True
        logger.info("Initiating graceful shutdown sequence")

        # Get components in reverse initialization order
        shutdown_order = list(self.initialized)
        shutdown_order.reverse()

        # Shutdown each component
        for name in shutdown_order:
            if name in self.shutdown_handlers:
                try:
                    logger.info(f"Shutting down component '{name}'")
                    self.shutdown_handlers[name]()
                    logger.info(f"Component '{name}' shutdown complete")
                except Exception as e:
                    logger.error(f"Error shutting down component '{name}': {e}")
            else:
                logger.warning(f"No shutdown handler registered for component '{name}'")

        self.is_shutting_down = False
        logger.info("Shutdown sequence complete")

    def get_status_report(self) -> Dict[str, Any]:
        """
        Generate a status report for all components.

        Returns:
            Dictionary with component status information
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "components": {},
            "initialization_time": None,
        }

        if self.initialization_start_time and self.initialization_end_time:
            report["initialization_time"] = (
                self.initialization_end_time - self.initialization_start_time
            ).total_seconds()

        for name, component in self.components.items():
            report["components"][name] = {
                "status": component.status,
                "optional": component.optional,
                "dependencies": component.dependencies,
                "initialization_time": component.initialization_time,
                "error": component.error,
                "initialized_at": (
                    component.initialized_at.isoformat()
                    if component.initialized_at
                    else None
                ),
            }

        return report


# Create a global component manager instance
component_manager = ComponentManager()


def initialize_component_system(register_signal_handlers: bool = True) -> None:
    """
    Initialize the component lifecycle management system.

    Args:
        register_signal_handlers: Whether to register signal handlers for shutdown
    """
    if register_signal_handlers:
        # Register signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown")
            component_manager.shutdown_all()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    logger.info("Component lifecycle management system initialized")


def register_component(
    name: str, dependencies: List[str] = None, optional: bool = False
) -> Callable:
    """
    Decorator to register a component initialization method.

    Args:
        name: Component name
        dependencies: List of component names this component depends on
        optional: Whether this component is optional

    Returns:
        Decorated function
    """

    def decorator(func):
        component_manager.register_component(
            name=name, init_method=func, dependencies=dependencies, optional=optional
        )

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


def register_shutdown_handler(name: str) -> Callable:
    """
    Decorator to register a component shutdown handler.

    Args:
        name: Component name

    Returns:
        Decorated function
    """

    def decorator(func):
        component_manager.register_shutdown_handler(name, func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator
