"""
Performance Data Module

This module provides functions to retrieve and process performance-related data
for the dashboard.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loguru import logger

from src.dashboard.utils.transformers import data_transformer

# Sample data for standalone mode
SAMPLE_EQUITY_DATA = [
    {
        "timestamp": (datetime.now() - timedelta(days=i)).isoformat(),
        "value": 10000 * (1 + 0.01 * (30 - i)),
    }
    for i in range(30)
]

SAMPLE_DRAWDOWN_DATA = [
    {
        "timestamp": (datetime.now() - timedelta(days=i)).isoformat(),
        "value": -5.0 * abs(np.sin(i / 10)),
    }
    for i in range(30)
]

SAMPLE_WIN_RATE = 0.62
SAMPLE_SHARPE_RATIO = 1.85
SAMPLE_MAX_DRAWDOWN = -12.5


def _initialize_performance_data(service):
    """
    Initialize performance data storage with sample data for standalone mode.

    Args:
        service: DashboardDataService instance
    """
    service._performance_data = {
        "equity_history": SAMPLE_EQUITY_DATA,
        "drawdown_history": SAMPLE_DRAWDOWN_DATA,
        "win_rate": SAMPLE_WIN_RATE,
        "sharpe_ratio": SAMPLE_SHARPE_RATIO,
        "max_drawdown": SAMPLE_MAX_DRAWDOWN,
        "monthly_returns": _generate_sample_monthly_returns(),
        "metrics": _generate_sample_performance_metrics(),
    }

    # Set initial update timestamp
    service._data_updated_at["performance"] = datetime.now()

    logger.debug("Initialized sample performance data")


def _update_performance_data(service):
    """
    Update performance data from the trading system.

    Args:
        service: DashboardDataService instance
    """
    if service.standalone_mode:
        # In standalone mode, just update the timestamp
        service._data_updated_at["performance"] = datetime.now()
        return

    try:
        # Get data from performance tracker if available
        if service.performance_tracker:
            logger.debug("Updating performance data from performance tracker")

            # Get equity history
            equity_history = service.performance_tracker.get_equity_history()
            if isinstance(equity_history, pd.DataFrame):
                equity_history_list = []
                for timestamp, row in equity_history.iterrows():
                    equity_history_list.append(
                        {"timestamp": timestamp.isoformat(), "value": row["equity"]}
                    )
                service._performance_data["equity_history"] = equity_history_list

            # Get drawdown history
            drawdown_history = service.performance_tracker.get_drawdown_history()
            if isinstance(drawdown_history, pd.DataFrame):
                drawdown_history_list = []
                for timestamp, row in drawdown_history.iterrows():
                    drawdown_history_list.append(
                        {
                            "timestamp": timestamp.isoformat(),
                            "value": row["drawdown"] * 100,  # Convert to percentage
                        }
                    )
                service._performance_data["drawdown_history"] = drawdown_history_list

            # Get key metrics
            metrics = service.performance_tracker.get_performance_metrics()
            if metrics:
                service._performance_data["win_rate"] = metrics.get("win_rate", 0)
                service._performance_data["sharpe_ratio"] = metrics.get(
                    "sharpe_ratio", 0
                )
                service._performance_data["max_drawdown"] = (
                    metrics.get("max_drawdown", 0) * 100
                )  # Convert to percentage
                service._performance_data["metrics"] = metrics

            # Get monthly returns
            monthly_returns = service.performance_tracker.get_monthly_returns()
            if isinstance(monthly_returns, pd.DataFrame):
                monthly_returns_dict = {}
                for year in monthly_returns.index.get_level_values(0).unique():
                    monthly_returns_dict[str(year)] = {}
                    for month in range(1, 13):
                        if (year, month) in monthly_returns.index:
                            monthly_returns_dict[str(year)][str(month)] = float(
                                monthly_returns.loc[(year, month)]
                            )
                        else:
                            monthly_returns_dict[str(year)][str(month)] = None
                service._performance_data["monthly_returns"] = monthly_returns_dict

        # Update timestamp
        service._data_updated_at["performance"] = datetime.now()

    except Exception as e:
        logger.error(f"Error updating performance data: {str(e)}")
        # Keep using existing data if update fails


def get_performance_data(self, timeframe="all"):
    """
    Get performance metrics data.

    Args:
        timeframe: Time period for performance data ('day', 'week', 'month', 'all')

    Returns:
        Dictionary with performance metrics
    """
    # If we have a real performance tracker, use it
    if not self.is_standalone and self.performance_tracker:
        try:
            # Get raw performance data
            if hasattr(self.performance_tracker, "get_performance_metrics"):
                raw_metrics = self.performance_tracker.get_performance_metrics(
                    timeframe
                )

                # Process and transform the data for dashboard display
                processed_metrics = data_transformer.transform_performance_metrics(
                    raw_metrics
                )

                # Update the cache
                self._performance_data = processed_metrics
                self._data_updated_at["performance"] = datetime.now()
                self._increment_data_version("performance")

                return processed_metrics
        except Exception as e:
            logger.error(f"Error fetching performance data: {str(e)}")
            # Fall back to cached data
            logger.debug("Falling back to cached performance data")

    # Check if we need to refresh the cache for standalone mode
    last_update = self._data_updated_at.get("performance")
    if self.is_standalone and (
        last_update is None or (datetime.now() - last_update).seconds > 60
    ):
        _update_performance_data(self)

    # If no performance tracker or error occurred, return cached data
    return self._performance_data


def _generate_sample_monthly_returns():
    """Generate sample monthly returns data for standalone mode."""
    current_year = datetime.now().year
    monthly_returns = {
        str(current_year): {},
        str(current_year - 1): {},
    }

    # Generate some realistic monthly returns
    for year in monthly_returns.keys():
        for month in range(1, 13):
            # Skip future months in current year
            if year == str(current_year) and month > datetime.now().month:
                monthly_returns[year][str(month)] = None
            else:
                # Generate a return between -10% and +15%
                monthly_returns[year][str(month)] = (np.random.random() * 25 - 10) / 100

    return monthly_returns


def _generate_sample_performance_metrics():
    """Generate sample performance metrics for standalone mode."""
    return {
        "win_rate": SAMPLE_WIN_RATE,
        "sharpe_ratio": SAMPLE_SHARPE_RATIO,
        "max_drawdown": SAMPLE_MAX_DRAWDOWN / 100,  # Store as decimal
        "total_return": 0.325,  # 32.5%
        "annualized_return": 0.218,  # 21.8%
        "volatility": 0.145,  # 14.5%
        "sortino_ratio": 2.12,
        "calmar_ratio": 1.75,
        "profit_factor": 1.62,
        "average_win": 0.0218,  # 2.18%
        "average_loss": -0.0124,  # -1.24%
        "win_loss_ratio": 1.76,
        "expectancy": 0.0089,  # 0.89%
        "recovery_factor": 2.6,
        "risk_of_ruin": 0.0012,  # 0.12%
    }
