"""
Dashboard Callback Registry

This module defines a standard interface for callback registration functions
and provides utilities for managing callback dependencies.
"""

from typing import Dict, Any, Optional, Callable, List, Protocol, TypeVar, Union
import dash
from dash import Input, Output, State
from loguru import logger
import time
import inspect
import uuid

from .dependency_optimizer import optimize_callback, create_callback_optimizer

# Type definitions
AppInstance = TypeVar("AppInstance", bound=dash.Dash)
DataServiceType = TypeVar("DataServiceType")
DataAccessFunc = Callable[..., Any]


class CallbackRegistrar(Protocol):
    """Protocol defining the standard interface for callback registration functions."""

    def __call__(
        self,
        app: AppInstance,
        data_service: Optional[DataServiceType] = None,
        **kwargs: Any,
    ) -> None:
        """
        Register callbacks for a component or layout.

        Args:
            app: The Dash application instance
            data_service: Optional data service instance for data access
            **kwargs: Additional keyword arguments specific to the registrar
        """
        ...


class CallbackRegistry:
    """
    Central registry for managing callback registration functions.

    This class provides a standardized way to register callbacks across
    the dashboard, ensuring consistent parameter patterns and dependency
    injection.
    """

    def __init__(
        self, app: AppInstance, data_service: Optional[DataServiceType] = None
    ):
        """
        Initialize the callback registry.

        Args:
            app: The Dash application instance
            data_service: Optional data service instance for data access
        """
        self.app = app
        self.data_service = data_service
        self.registrars: Dict[str, CallbackRegistrar] = {}
        self.registered: List[str] = []
        self.execution_times: Dict[str, float] = {}
        self.execution_counts: Dict[str, int] = {}

        # Initialize callback optimizer
        self.optimizer = create_callback_optimizer(app)

        # Set flag for using optimized registration
        self.use_optimization = True

        # Time tracking
        self.start_time = time.time()

    def register(self, name: str, registrar: CallbackRegistrar) -> None:
        """
        Add a callback registrar to the registry.

        Args:
            name: Name of the registrar (for logging and tracking)
            registrar: The callback registration function
        """
        self.registrars[name] = registrar
        logger.debug(f"Added callback registrar: {name}")

    def execute(self, name: str, **kwargs: Any) -> None:
        """
        Execute a specific callback registrar.

        Args:
            name: Name of the registrar to execute
            **kwargs: Additional keyword arguments to pass to the registrar
        """
        if name not in self.registrars:
            logger.warning(f"Callback registrar not found: {name}")
            return

        if name in self.registered:
            logger.debug(f"Callback registrar already executed: {name}")
            return

        logger.info(f"Executing callback registrar: {name}")
        start_time = time.time()

        try:
            # Call the registrar with standard parameters plus any additional kwargs
            self.registrars[name](
                app=self.app, data_service=self.data_service, **kwargs
            )
            self.registered.append(name)

            # Track execution time
            elapsed = time.time() - start_time
            self.execution_times[name] = elapsed
            self.execution_counts[name] = self.execution_counts.get(name, 0) + 1

            logger.info(
                f"Successfully executed callback registrar: {name} in {elapsed:.2f}s"
            )
        except Exception as e:
            logger.error(f"Error executing callback registrar {name}: {str(e)}")
            logger.exception(e)

    def execute_all(self, **kwargs: Any) -> None:
        """
        Execute all registered callback registrars.

        Args:
            **kwargs: Additional keyword arguments to pass to all registrars
        """
        logger.info("Executing all callback registrars")
        overall_start = time.time()

        # Get optimized execution order if available
        if hasattr(self, "optimizer") and self.use_optimization:
            # For initial setup, we can't optimize yet
            # We'll rely on the dependency graph built during execution
            registrars_to_execute = list(self.registrars.keys())
        else:
            registrars_to_execute = list(self.registrars.keys())

        for name in registrars_to_execute:
            self.execute(name, **kwargs)

        elapsed = time.time() - overall_start
        logger.info(
            f"Completed execution of {len(self.registered)} callback registrars in {elapsed:.2f}s"
        )

        # Generate optimization report if appropriate
        if hasattr(self, "optimizer") and self.use_optimization:
            try:
                report = self.optimizer.generate_optimization_report()
                if report["recommendations"]:
                    logger.info(
                        f"Optimization recommendations available: {len(report['recommendations'])}"
                    )
                    for rec in report["recommendations"]:
                        logger.info(f"- {rec['description']}")
            except Exception as e:
                logger.error(f"Error generating optimization report: {str(e)}")

    def get_data_access_function(self, func_name: str) -> Optional[DataAccessFunc]:
        """
        Get a data access function from the data service.

        This is a helper method to standardize how data access functions
        are retrieved and passed to callback registrars.

        Args:
            func_name: Name of the data access function

        Returns:
            The data access function or None if not available
        """
        if not self.data_service:
            logger.warning(f"Data service not available for function: {func_name}")
            return None

        func = getattr(self.data_service, func_name, None)
        if not func or not callable(func):
            logger.warning(f"Data access function not found: {func_name}")
            return None

        return func

    def register_optimized_callback(
        self,
        outputs: List[Output],
        inputs: List[Union[Input, State]],
        callback_function: Callable,
        callback_id: Optional[str] = None,
        throttle_ms: Optional[int] = None,
        debounce_ms: Optional[int] = None,
        batch_updates: bool = False,
        priority: int = 5,
    ) -> Callable:
        """
        Register an optimized callback with the dependency optimizer.

        Args:
            outputs: List of outputs
            inputs: List of inputs
            callback_function: Callback function
            callback_id: Optional callback ID (will be auto-generated if None)
            throttle_ms: Optional throttle timeout in milliseconds
            debounce_ms: Optional debounce timeout in milliseconds
            batch_updates: Whether to batch updates
            priority: Priority level (1-10)

        Returns:
            The wrapped callback function
        """
        if not self.use_optimization:
            # Fall back to regular Dash callback
            return self.app.callback(*outputs, inputs=inputs)(callback_function)

        # Generate a unique ID if not provided
        if callback_id is None:
            # Try to use the function name
            callback_id = (
                f"{callback_function.__module__}.{callback_function.__qualname__}"
            )
            # Append a unique suffix to avoid collisions
            callback_id = f"{callback_id}_{str(uuid.uuid4())[:8]}"

        # Use the optimizer to register the callback
        return optimize_callback(
            app=self.app,
            callback_id=callback_id,
            outputs=outputs,
            inputs=inputs,
            throttle_ms=throttle_ms,
            debounce_ms=debounce_ms,
            batch_updates=batch_updates,
            priority=priority,
        )(callback_function)

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for callback registrars.

        Returns:
            Dictionary with performance statistics
        """
        total_elapsed = time.time() - self.start_time

        stats = {
            "total_registrars": len(self.registrars),
            "executed_registrars": len(self.registered),
            "total_elapsed_seconds": total_elapsed,
            "execution_times": self.execution_times,
            "execution_counts": self.execution_counts,
        }

        # Add optimizer stats if available
        if hasattr(self, "optimizer") and self.use_optimization:
            try:
                optimizer_stats = self.optimizer.get_optimization_stats()
                stats["optimizer"] = optimizer_stats
            except Exception as e:
                logger.error(f"Error getting optimizer stats: {str(e)}")

        return stats


# Decorator for standardizing callback registrar functions
def callback_registrar(name: str):
    """
    Decorator for standardizing callback registrar functions.

    This decorator ensures that all callback registrar functions follow
    the standard interface defined by CallbackRegistrar.

    Args:
        name: Name of the registrar (for logging and tracking)

    Returns:
        Decorated function
    """

    def decorator(func: Callable) -> CallbackRegistrar:
        def wrapper(
            app: AppInstance,
            data_service: Optional[DataServiceType] = None,
            **kwargs: Any,
        ) -> None:
            logger.debug(f"Executing callback registrar: {name}")
            return func(app=app, data_service=data_service, **kwargs)

        # Add metadata to the wrapper function
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.__module__ = func.__module__
        wrapper.__qualname__ = func.__qualname__
        wrapper.__annotations__ = func.__annotations__

        return wrapper

    return decorator
