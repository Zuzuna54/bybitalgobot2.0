"""
Strategy Loader for the Algorithmic Trading System

This module provides functionality for dynamically loading strategy classes
from their modules based on strategy names.
"""

from typing import Dict, Any, List, Type, Optional, Union
import importlib
import inspect
import os
import sys
from pathlib import Path
from loguru import logger

from src.strategies.base_strategy import BaseStrategy


def load_strategy_class(strategy_name: str) -> Optional[Type[BaseStrategy]]:
    """
    Dynamically load a strategy class.

    Args:
        strategy_name: Name of the strategy module (e.g., 'ema_crossover')

    Returns:
        Strategy class or None if not found
    """
    try:
        # Convert strategy name to proper module name
        # e.g., convert "ema_crossover" to "src.strategies.ema_crossover_strategy"
        module_name = f"src.strategies.{strategy_name}_strategy"

        # Import the module
        module = importlib.import_module(module_name)

        # Find the strategy class in the module
        for name, obj in inspect.getmembers(module):
            if (
                inspect.isclass(obj)
                and issubclass(obj, BaseStrategy)
                and obj != BaseStrategy
            ):
                logger.info(
                    f"Successfully loaded strategy class: {name} from module {module_name}"
                )
                return obj

        logger.error(f"No strategy class found in module {module_name}")
        return None
    except ImportError as e:
        logger.error(f"Failed to import strategy module {strategy_name}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error loading strategy {strategy_name}: {str(e)}")
        return None


def get_available_strategies() -> List[str]:
    """
    Get a list of available strategy names.

    Returns:
        List of strategy names
    """
    strategies = []

    # Get all Python files in the strategies directory
    strategies_dir = Path(__file__).parent.parent
    for file in strategies_dir.glob("*_strategy.py"):
        if file.is_file() and not file.name.startswith("__"):
            strategy_name = file.name.replace("_strategy.py", "")
            strategies.append(strategy_name)

    logger.info(
        f"Found {len(strategies)} available strategies: {', '.join(strategies)}"
    )
    return strategies
