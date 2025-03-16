"""
Dashboard Callback Registry

This module defines a standard interface for callback registration functions
and provides utilities for managing callback dependencies.
"""

from typing import Dict, Any, Optional, Callable, List, Protocol, TypeVar, Union
import dash
from loguru import logger

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

        try:
            # Call the registrar with standard parameters plus any additional kwargs
            self.registrars[name](
                app=self.app, data_service=self.data_service, **kwargs
            )
            self.registered.append(name)
            logger.info(f"Successfully executed callback registrar: {name}")
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

        for name in self.registrars:
            self.execute(name, **kwargs)

        logger.info(
            f"Completed execution of {len(self.registered)} callback registrars"
        )

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
