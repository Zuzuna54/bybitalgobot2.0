"""
Strategy Data Module

This module provides functions to retrieve and process strategy-related data
for the dashboard.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loguru import logger


def _initialize_strategy_data(service):
    """
    Initialize strategy data storage with sample data for standalone mode.

    Args:
        service: DashboardDataService instance
    """
    # Generate sample strategies
    sample_strategies = _generate_sample_strategies()

    # Store in service
    service._strategy_data = {
        "strategies": sample_strategies,
        "signals": _generate_sample_signals(),
        "market_conditions": _generate_sample_market_conditions(),
    }

    # Set initial update timestamp
    service._data_updated_at["strategy"] = datetime.now()

    logger.debug("Initialized sample strategy data")


def _update_strategy_data(service):
    """
    Update strategy data from the strategy manager.

    Args:
        service: DashboardDataService instance
    """
    if service.standalone_mode:
        # In standalone mode, just update the timestamp
        service._data_updated_at["strategy"] = datetime.now()
        return

    try:
        # Get data from strategy manager if available
        if service.strategy_manager:
            logger.debug("Updating strategy data from strategy manager")

            # Get active strategies
            strategies = service.strategy_manager.get_strategies()
            if strategies:
                # Convert to list if it's a dictionary
                if isinstance(strategies, dict):
                    strategy_list = []
                    for strategy_id, strategy in strategies.items():
                        if hasattr(strategy, "to_dict"):
                            strategy_dict = strategy.to_dict()
                            strategy_dict["id"] = strategy_id
                            strategy_list.append(strategy_dict)
                        else:
                            # Basic conversion
                            strategy_list.append(
                                {
                                    "id": strategy_id,
                                    "name": getattr(strategy, "name", strategy_id),
                                    "enabled": getattr(strategy, "enabled", True),
                                    "description": getattr(strategy, "description", ""),
                                }
                            )
                    service._strategy_data["strategies"] = strategy_list
                else:
                    service._strategy_data["strategies"] = strategies

            # Get recent signals
            signals = service.strategy_manager.get_recent_signals()
            if signals:
                service._strategy_data["signals"] = signals

            # Get market condition performance
            market_conditions = (
                service.strategy_manager.get_market_condition_performance()
            )
            if market_conditions:
                service._strategy_data["market_conditions"] = market_conditions

        # Update timestamp
        service._data_updated_at["strategy"] = datetime.now()

    except Exception as e:
        logger.error(f"Error updating strategy data: {str(e)}")
        # Keep using existing data if update fails


def get_strategy_data(self, strategy_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Get strategy data for the dashboard.

    Args:
        strategy_id: Optional ID of a specific strategy to get data for

    Returns:
        Dictionary with strategy data
    """
    if strategy_id:
        # Find the requested strategy
        for strategy in self._strategy_data.get("strategies", []):
            if strategy.get("id") == strategy_id:
                return strategy

        # Strategy not found
        return {}

    # Return all strategy data
    return self._strategy_data


def _generate_sample_strategies():
    """Generate sample strategy data for standalone mode."""
    strategy_names = [
        "RSI Reversal",
        "Bollinger Band Breakout",
        "MACD Crossover",
        "Moving Average Trend",
        "Volume Breakout",
    ]

    strategies = []

    for i, name in enumerate(strategy_names):
        # Generate performance history (last 30 days)
        performance_history = []
        start_value = 10000
        current_value = start_value

        for day in range(30):
            timestamp = datetime.now() - timedelta(days=30 - day)

            # Add some trend and randomness
            if i % 2 == 0:  # Even strategies perform well
                daily_return = np.random.normal(0.003, 0.01)
            else:  # Odd strategies perform less well
                daily_return = np.random.normal(0.001, 0.015)

            # Apply return
            current_value *= 1 + daily_return

            performance_history.append(
                {"timestamp": timestamp.isoformat(), "value": current_value}
            )

        # Calculate metrics
        total_return = (current_value - start_value) / start_value
        win_rate = 0.5 + np.random.uniform(-0.1, 0.3)
        sharpe_ratio = 1.0 + np.random.uniform(-0.5, 1.5)
        max_drawdown = -np.random.uniform(0.05, 0.2)

        # Generate some daily returns
        daily_returns = []
        for day in range(60):
            date = datetime.now() - timedelta(days=60 - day)
            ret = np.random.normal(0.002, 0.015)
            daily_returns.append({"date": date.date().isoformat(), "return": ret})

        # Create strategy dictionary
        strategy = {
            "id": f"strategy-{i+1}",
            "strategy_name": name,
            "description": f"Sample {name} strategy for testing",
            "enabled": np.random.random() > 0.3,  # 70% chance of being enabled
            "performance_history": performance_history,
            "total_return": total_return,
            "win_rate": win_rate,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "daily_returns": daily_returns,
            "metrics": {
                "win_rate": win_rate,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "total_return": total_return,
                "profit_factor": 1.2 + np.random.uniform(-0.2, 0.8),
                "average_trade": 0.5 + np.random.uniform(-0.3, 0.5),
                "average_win": 1.2 + np.random.uniform(-0.2, 0.5),
                "average_loss": -0.8 + np.random.uniform(-0.3, 0.1),
                "win_loss_ratio": 1.5 + np.random.uniform(-0.2, 0.5),
                "expectancy": 0.3 + np.random.uniform(-0.2, 0.3),
                "trade_count": int(50 + np.random.uniform(-20, 30)),
            },
            "market_condition_performance": _generate_market_condition_performance(),
        }

        strategies.append(strategy)

    return strategies


def _generate_sample_signals():
    """Generate sample strategy signals for standalone mode."""
    strategy_names = ["RSI Reversal", "Bollinger Band Breakout", "MACD Crossover"]
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT"]
    signal_types = ["BUY", "SELL", "HOLD"]

    signals = []

    # Generate 20 signals over the last 48 hours
    for i in range(20):
        timestamp = datetime.now() - timedelta(hours=np.random.uniform(0, 48))
        strategy = np.random.choice(strategy_names)
        symbol = np.random.choice(symbols)
        signal_type = np.random.choice(signal_types)

        # Generate price based on symbol
        if symbol == "BTCUSDT":
            price = 50000 * (1 + np.random.normal(0, 0.05))
        elif symbol == "ETHUSDT":
            price = 3000 * (1 + np.random.normal(0, 0.05))
        elif symbol == "SOLUSDT":
            price = 100 * (1 + np.random.normal(0, 0.05))
        elif symbol == "BNBUSDT":
            price = 500 * (1 + np.random.normal(0, 0.05))
        else:  # ADAUSDT
            price = 1.5 * (1 + np.random.normal(0, 0.05))

        # Create signal
        signal = {
            "id": f"signal-{i+1}",
            "timestamp": timestamp.isoformat(),
            "strategy": strategy,
            "symbol": symbol,
            "type": signal_type,
            "price": price,
            "confidence": np.random.uniform(0.6, 0.95),
            "executed": np.random.random() > 0.3,  # 70% chance of being executed
        }

        signals.append(signal)

    # Sort by timestamp (most recent first)
    signals.sort(key=lambda x: x["timestamp"], reverse=True)

    return signals


def _generate_sample_market_conditions():
    """Generate sample market condition data for standalone mode."""
    condition_types = [
        "uptrend",
        "downtrend",
        "ranging",
        "volatile",
        "low_volatility",
        "high_volume",
        "low_volume",
    ]

    # Current distribution of market conditions
    current_conditions = {}

    # Assign random probabilities that sum to 1
    total = 0
    for i, condition in enumerate(condition_types):
        # Last condition gets the remainder to ensure sum is 1
        if i == len(condition_types) - 1:
            prob = 1.0 - total
        else:
            prob = np.random.uniform(0.05, 0.3)
            # Make sure we don't exceed 1
            if total + prob > 0.95:
                prob = 0.95 - total
            total += prob

        current_conditions[condition] = prob

    # Create historical condition data
    historical_conditions = []

    for day in range(30):
        date = (datetime.now() - timedelta(days=30 - day)).date()

        # Create a slightly different distribution for each day
        day_conditions = {}
        day_total = 0

        for i, condition in enumerate(condition_types):
            base_prob = current_conditions[condition]
            # Add some random variation
            variation = np.random.uniform(-0.1, 0.1) * base_prob
            prob = max(0.01, min(0.5, base_prob + variation))

            if i == len(condition_types) - 1:
                prob = 1.0 - day_total
            else:
                if day_total + prob > 0.95:
                    prob = 0.95 - day_total
                day_total += prob

            day_conditions[condition] = prob

        historical_conditions.append(
            {"date": date.isoformat(), "conditions": day_conditions}
        )

    return {
        "current": current_conditions,
        "historical": historical_conditions,
    }


def _generate_market_condition_performance():
    """Generate market condition performance data for a strategy."""
    conditions = [
        "uptrend",
        "downtrend",
        "ranging",
        "volatile",
        "low_volatility",
    ]

    performance = {}

    for condition in conditions:
        # Generate performance metrics for each condition
        if condition == "uptrend":
            win_rate = 0.7 + np.random.uniform(-0.1, 0.1)
            return_val = 0.15 + np.random.uniform(-0.05, 0.1)
            trade_count = int(25 + np.random.uniform(-5, 10))
        elif condition == "downtrend":
            win_rate = 0.4 + np.random.uniform(-0.1, 0.1)
            return_val = -0.05 + np.random.uniform(-0.1, 0.15)
            trade_count = int(15 + np.random.uniform(-5, 5))
        elif condition == "ranging":
            win_rate = 0.55 + np.random.uniform(-0.1, 0.1)
            return_val = 0.05 + np.random.uniform(-0.05, 0.05)
            trade_count = int(30 + np.random.uniform(-5, 10))
        elif condition == "volatile":
            win_rate = 0.5 + np.random.uniform(-0.15, 0.15)
            return_val = 0.1 + np.random.uniform(-0.2, 0.2)
            trade_count = int(20 + np.random.uniform(-5, 5))
        else:  # low_volatility
            win_rate = 0.6 + np.random.uniform(-0.1, 0.1)
            return_val = 0.03 + np.random.uniform(-0.02, 0.02)
            trade_count = int(10 + np.random.uniform(-3, 3))

        performance[condition] = {
            "win_rate": win_rate,
            "return": return_val,
            "trade_count": trade_count,
        }

    return performance
