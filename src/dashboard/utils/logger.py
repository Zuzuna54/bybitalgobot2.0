"""
Dashboard Logging Module

This module provides a centralized logging system for the dashboard, with
configurable log levels, formatting, and outputs.
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Optional, Dict, Any, Union, List
import datetime
from pathlib import Path

from src.config.config_manager import get_config_manager
from src.dashboard.utils.time_utils import (
    get_current_time,
    get_current_time_as_string,
    format_duration,
)

# Configure logging defaults
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_DIRECTORY = "logs"
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


class DashboardLogger:
    """
    Centralized logging system for the Bybit Algorithmic Trading Dashboard.
    """

    def __init__(
        self,
        name: str = "dashboard",
        log_level: str = None,
        log_format: str = None,
        log_date_format: str = None,
        log_directory: str = None,
        log_to_console: bool = True,
        log_to_file: bool = True,
        max_file_size: int = 10 * 1024 * 1024,  # 10 MB
        backup_count: int = 5,
        rotating_when: str = "midnight",
        debug_mode: bool = False,
    ):
        """
        Initialize the dashboard logger.

        Args:
            name: Logger name
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_format: Format string for log messages
            log_date_format: Format string for log timestamps
            log_directory: Directory to store log files
            log_to_console: Whether to log to console
            log_to_file: Whether to log to file
            max_file_size: Maximum file size in bytes for rotating file handler
            backup_count: Number of backup files to keep
            rotating_when: When to rotate logs ('midnight', 'h', 'd', 'w0'-'w6')
            debug_mode: Whether to enable debug mode
        """
        # Get configuration
        config = get_config_manager()

        # Set instance variables from parameters or config
        self.name = name
        self.log_level = log_level or config.get(
            "dashboard.log_level", DEFAULT_LOG_LEVEL
        )
        self.log_format = log_format or config.get(
            "dashboard.log_format", DEFAULT_LOG_FORMAT
        )
        self.log_date_format = log_date_format or config.get(
            "dashboard.log_date_format", DEFAULT_DATE_FORMAT
        )
        self.log_directory = (
            log_directory or config.get_path("logs_dir") or DEFAULT_LOG_DIRECTORY
        )
        self.log_to_console = (
            log_to_console
            if log_to_console is not None
            else config.get_bool("dashboard.log_to_console", True)
        )
        self.log_to_file = (
            log_to_file
            if log_to_file is not None
            else config.get_bool("dashboard.log_to_file", True)
        )
        self.max_file_size = max_file_size or config.get_int(
            "dashboard.max_log_file_size", 10 * 1024 * 1024
        )
        self.backup_count = backup_count or config.get_int(
            "dashboard.log_backup_count", 5
        )
        self.rotating_when = rotating_when or config.get(
            "dashboard.log_rotating_when", "midnight"
        )
        self.debug_mode = (
            debug_mode
            if debug_mode is not None
            else config.get_bool("dashboard.debug_mode", False)
        )

        # Convert log level string to logging level constant
        self.log_level_value = LOG_LEVELS.get(self.log_level.upper(), logging.INFO)
        if self.debug_mode:
            self.log_level_value = logging.DEBUG

        # Create logger
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """
        Set up and configure the logger.

        Returns:
            Configured logger instance
        """
        # Create logger
        logger = logging.getLogger(self.name)
        logger.setLevel(self.log_level_value)
        logger.propagate = False  # Prevent propagation to root logger

        # Clear existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Create formatter
        formatter = logging.Formatter(self.log_format, self.log_date_format)

        # Add console handler if enabled
        if self.log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            console_handler.setLevel(self.log_level_value)
            logger.addHandler(console_handler)

        # Add file handler if enabled
        if self.log_to_file:
            # Create log directory if it doesn't exist
            os.makedirs(self.log_directory, exist_ok=True)

            # Create log file path
            log_file_path = os.path.join(self.log_directory, f"{self.name}.log")

            # Create rotating file handler
            if self.rotating_when:
                # Time-based rotation
                file_handler = TimedRotatingFileHandler(
                    log_file_path,
                    when=self.rotating_when,
                    backupCount=self.backup_count,
                )
            else:
                # Size-based rotation
                file_handler = RotatingFileHandler(
                    log_file_path,
                    maxBytes=self.max_file_size,
                    backupCount=self.backup_count,
                )

            file_handler.setFormatter(formatter)
            file_handler.setLevel(self.log_level_value)
            logger.addHandler(file_handler)

        return logger

    def get_logger(self) -> logging.Logger:
        """
        Get the configured logger instance.

        Returns:
            The logger instance
        """
        return self.logger

    def set_level(self, level: Union[str, int]) -> None:
        """
        Set the logging level.

        Args:
            level: Logging level (either string name or logging level constant)
        """
        if isinstance(level, str):
            level = LOG_LEVELS.get(level.upper(), logging.INFO)

        self.log_level_value = level
        self.logger.setLevel(level)

        for handler in self.logger.handlers:
            handler.setLevel(level)

    def enable_debug(self) -> None:
        """Enable debug logging."""
        self.set_level(logging.DEBUG)
        self.debug_mode = True

    def disable_debug(self) -> None:
        """Disable debug logging and revert to previous log level."""
        self.set_level(LOG_LEVELS.get(self.log_level.upper(), logging.INFO))
        self.debug_mode = False


# Singleton logger instance
_dashboard_logger = None


def get_logger(name: str = "dashboard") -> logging.Logger:
    """
    Get a configured logger instance for the specified name.

    Args:
        name: Logger name

    Returns:
        The logger instance
    """
    global _dashboard_logger

    if _dashboard_logger is None:
        _dashboard_logger = DashboardLogger(name)

    # If requested logger has a different name than the one we created,
    # create a child logger with that name
    if name != _dashboard_logger.name:
        return logging.getLogger(f"{_dashboard_logger.name}.{name}")

    return _dashboard_logger.get_logger()


def log_exception(e: Exception, context: Optional[Dict[str, Any]] = None) -> str:
    """
    Log an exception with additional context information.

    Args:
        e: The exception to log
        context: Additional context information

    Returns:
        The error message that was logged
    """
    logger = get_logger()

    # Format exception message with context
    error_type = type(e).__name__
    error_msg = str(e)
    timestamp = get_current_time_as_string(DEFAULT_DATE_FORMAT)

    context_str = ""
    if context:
        context_str = " | Context: " + ", ".join(
            [f"{k}={v}" for k, v in context.items()]
        )

    message = f"Exception: {error_type}: {error_msg}{context_str}"

    # Log exception with traceback
    logger.exception(message)

    return message


def measure_execution_time(func):
    """
    Decorator to measure and log the execution time of a function.

    Args:
        func: The function to decorate

    Returns:
        The decorated function
    """

    def wrapper(*args, **kwargs):
        logger = get_logger()
        start_time = get_current_time()

        try:
            result = func(*args, **kwargs)

            # Calculate execution time
            end_time = get_current_time()
            execution_time = (end_time - start_time).total_seconds()

            # Log execution time
            logger.debug(
                f"Function {func.__name__} executed in {execution_time:.4f} seconds"
            )

            return result

        except Exception as e:
            # Log exception
            end_time = get_current_time()
            execution_time = (end_time - start_time).total_seconds()
            logger.error(
                f"Function {func.__name__} raised {type(e).__name__} after {execution_time:.4f} seconds: {str(e)}"
            )
            raise

    return wrapper
