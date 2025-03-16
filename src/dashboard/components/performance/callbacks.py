"""
Performance Panel Callbacks for the Trading Dashboard

This module provides callbacks for updating the performance panel components.
"""

from typing import Any, Callable, Optional
import dash
from dash.dependencies import Input, Output
import pandas as pd
from loguru import logger

from src.dashboard.components.performance.metrics import render_metrics_card
from src.dashboard.services.chart_service import (
    create_equity_curve_chart,
    create_drawdown_chart,
    create_return_distribution_chart,
    create_daily_performance_graph,
    create_empty_chart,
)
from src.dashboard.components.error_display import callback_error_handler
from src.dashboard.router.callback_registry import callback_registrar


@callback_registrar(name="performance")
def register_performance_callbacks(
    app: dash.Dash, get_performance_data_func: Optional[Callable] = None, **kwargs
) -> None:
    """
    Register the callbacks for the performance panel.

    Args:
        app: Dash application instance
        get_performance_data_func: Function that returns performance data
        **kwargs: Additional keyword arguments
    """
    logger.debug("Registering performance callbacks")

    # Get the performance data function from kwargs if not directly provided
    if not get_performance_data_func and kwargs.get("get_performance_data_func"):
        get_performance_data_func = kwargs.get("get_performance_data_func")

    # Fallback to an empty function if none is provided
    if not get_performance_data_func:
        logger.warning("No performance data function provided, using empty function")
        get_performance_data_func = lambda: {
            "metrics": {},
            "equity_curve": pd.DataFrame(),
            "daily_returns": pd.DataFrame(),
            "daily_summary": pd.DataFrame(),
        }

    @app.callback(
        [
            Output("performance-metrics-card", "children"),
            Output("equity-curve-graph", "figure"),
            Output("drawdown-graph", "figure"),
            Output("return-distribution-graph", "figure"),
            Output("daily-performance-graph", "figure"),
        ],
        [Input("performance-update-interval", "n_intervals")],
    )
    @callback_error_handler
    def update_performance_panel(n_intervals):
        """
        Update the performance panel components with the latest data.

        Args:
            n_intervals: Number of interval updates

        Returns:
            Tuple of (metrics_card, equity_curve, drawdown, return_distribution, daily_performance)
        """
        # Set the number of outputs for the error handler
        update_performance_panel._dash_output_count = 5

        # Get performance data
        try:
            performance_data = get_performance_data_func()

            if not performance_data:
                # Return empty components if no data is available
                return (
                    render_metrics_card({}),
                    create_empty_chart("Equity Curve"),
                    create_empty_chart("Drawdown Analysis"),
                    create_empty_chart("Return Distribution"),
                    create_empty_chart("Daily Performance"),
                )

            # Extract data
            metrics = performance_data.get("metrics", {})
            equity_curve = performance_data.get("equity_curve", pd.DataFrame())
            daily_returns = performance_data.get("daily_returns", pd.DataFrame())
            daily_summary = performance_data.get("daily_summary", pd.DataFrame())

            # Create components
            metrics_card = render_metrics_card(metrics)
            equity_fig = create_equity_curve_chart(equity_curve)
            drawdown_fig = create_drawdown_chart(equity_curve)
            returns_fig = create_return_distribution_chart(daily_returns)
            daily_fig = create_daily_performance_graph(daily_summary)

            return metrics_card, equity_fig, drawdown_fig, returns_fig, daily_fig

        except Exception as e:
            logger.exception(f"Error updating performance panel: {str(e)}")

            # Return error components
            error_message = f"Error loading performance data: {str(e)}"

            return (
                render_metrics_card({}),
                create_empty_chart(f"Equity Curve - {error_message}"),
                create_empty_chart(f"Drawdown Analysis - {error_message}"),
                create_empty_chart(f"Return Distribution - {error_message}"),
                create_empty_chart(f"Daily Performance - {error_message}"),
            )
