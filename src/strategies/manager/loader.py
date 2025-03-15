"""
Strategy Loader for the Algorithmic Trading System

This module provides functionality for dynamically loading strategy classes
from their modules based on strategy names.
"""

from typing import Optional, Type, Dict
import importlib
from loguru import logger

from src.strategies.base_strategy import BaseStrategy


def load_strategy_class(strategy_name: str) -> Optional[Type[BaseStrategy]]:
    """
    Dynamically load strategy class from module.
    
    Args:
        strategy_name: Name of the strategy
        
    Returns:
        Strategy class or None if not found
    """
    try:
        # Mapping from strategy name to class name and module
        strategy_map = {
            "ema_crossover": {
                "module": "src.strategies.ema_crossover_strategy",
                "class": "EMACrossoverStrategy"
            },
            "bollinger_breakout": {
                "module": "src.strategies.bollinger_breakout_strategy",
                "class": "BollingerBreakoutStrategy"
            },
            "rsi_reversal": {
                "module": "src.strategies.rsi_reversal_strategy",
                "class": "RSIReversalStrategy"
            },
            "macd_divergence": {
                "module": "src.strategies.macd_divergence_strategy",
                "class": "MACDDivergenceStrategy"
            },
            "ichimoku_cloud": {
                "module": "src.strategies.ichimoku_cloud_strategy",
                "class": "IchimokuCloudStrategy"
            },
            "support_resistance": {
                "module": "src.strategies.support_resistance_strategy",
                "class": "SupportResistanceStrategy"
            },
            "volume_profile": {
                "module": "src.strategies.volume_profile_strategy",
                "class": "VolumeProfileStrategy"
            },
            "fibonacci_retracement": {
                "module": "src.strategies.fibonacci_retracement_strategy",
                "class": "FibonacciRetracementStrategy"
            },
            "triple_supertrend": {
                "module": "src.strategies.triple_supertrend_strategy",
                "class": "TripleSupertrendStrategy"
            },
            "harmonic_pattern": {
                "module": "src.strategies.harmonic_pattern_strategy",
                "class": "HarmonicPatternStrategy"
            }
        }
        
        if strategy_name in strategy_map:
            module_name = strategy_map[strategy_name]["module"]
            class_name = strategy_map[strategy_name]["class"]
            
            module = importlib.import_module(module_name)
            strategy_class = getattr(module, class_name)
            
            return strategy_class
        else:
            # Try using a standardized naming convention for custom strategies
            module_name = f"src.strategies.{strategy_name}_strategy"
            
            try:
                module = importlib.import_module(module_name)
                
                # Try to find the strategy class by constructing its likely name
                class_name = "".join(word.capitalize() for word in strategy_name.split("_")) + "Strategy"
                strategy_class = getattr(module, class_name)
                
                return strategy_class
            except (ImportError, AttributeError):
                logger.warning(f"Could not find strategy class for {strategy_name}")
                return None
    
    except Exception as e:
        logger.error(f"Error loading strategy class {strategy_name}: {e}")
        return None


def get_available_strategies() -> Dict[str, str]:
    """
    Get a dictionary of all available strategies.
    
    Returns:
        Dictionary mapping strategy names to their descriptions
    """
    # List of known strategies
    known_strategies = [
        "ema_crossover",
        "bollinger_breakout",
        "rsi_reversal",
        "macd_divergence",
        "ichimoku_cloud",
        "support_resistance",
        "volume_profile",
        "fibonacci_retracement",
        "triple_supertrend",
        "harmonic_pattern"
    ]
    
    available_strategies = {}
    
    for strategy_name in known_strategies:
        strategy_class = load_strategy_class(strategy_name)
        if strategy_class:
            description = getattr(strategy_class, "description", "No description available")
            available_strategies[strategy_name] = description
    
    return available_strategies 