"""
Market Panel Callbacks for the Trading Dashboard

This module provides callbacks for updating the market panel components.
"""

from typing import Dict, Any, List, Callable, Optional
import dash
from dash import html, dcc, callback, Input, Output, State
import plotly.graph_objects as go
import pandas as pd
from loguru import logger

from src.dashboard.router.callback_registry import callback_registrar
from src.dashboard.components.market.panel import create_market_panel
from src.dashboard.components.error_display import create_error_message
from src.dashboard.services.chart_service import create_candlestick_chart


@callback_registrar(name="market")
def register_market_callbacks(
    app: dash.Dash, data_service: Optional[Any] = None, **kwargs
) -> None:
    """
    Register callbacks for market data components.

    Args:
        app: The Dash application instance
        data_service: Data service instance
        **kwargs: Additional keyword arguments
    """
    logger.debug("Registering market callbacks")

    # Get the market data function from kwargs or data_service
    get_market_data_func = kwargs.get("get_market_data_func")
    if not get_market_data_func and data_service:
        get_market_data_func = getattr(data_service, "get_market_data", None)

    if not get_market_data_func:
        logger.warning("No market data function provided, using empty function")
        get_market_data_func = lambda symbol=None, timeframe=None: {}

    @app.callback(
        [
            Output("market-price-card", "children"),
            Output("market-stats-card", "children"),
            Output("market-chart-container", "children"),
        ],
        [
            Input("market-interval-component", "n_intervals"),
            Input("market-symbol-dropdown", "value"),
            Input("market-timeframe-dropdown", "value"),
        ],
        prevent_initial_call=False,
    )
    def update_market_data(n_intervals, symbol, timeframe):
        """
        Update market data components.

        Args:
            n_intervals: Interval trigger count
            symbol: Selected symbol
            timeframe: Selected timeframe

        Returns:
            Updated market components
        """
        try:
            # Default components for error states
            default_price_card = html.Div(
                "No price data available", className="no-data-message"
            )
            default_stats_card = html.Div(
                "No market statistics available", className="no-data-message"
            )
            default_chart = html.Div(
                "No chart data available", className="no-data-message"
            )

            # Validate inputs
            if not symbol or not timeframe:
                return default_price_card, default_stats_card, default_chart

            # Get market data - only pass symbol parameter since the data service doesn't accept timeframe
            market_data = get_market_data_func(symbol=symbol)

            if not market_data:
                return default_price_card, default_stats_card, default_chart

            # Extract data components
            price_data = market_data.get("price_data", {})
            market_stats = market_data.get("market_stats", {})
            candle_data = market_data.get("candle_data", pd.DataFrame())

            # Create price card
            price_card = (
                create_price_card(price_data, symbol)
                if price_data
                else default_price_card
            )

            # Create stats card
            stats_card = (
                create_stats_card(market_stats) if market_stats else default_stats_card
            )

            # Create chart
            if isinstance(candle_data, pd.DataFrame) and not candle_data.empty:
                # Create a title that includes both symbol and timeframe
                chart_title = f"{symbol} - {timeframe}" if timeframe else symbol
                chart = create_candlestick_chart(
                    candle_data, symbol=symbol, title=chart_title
                )
            else:
                chart = default_chart

            return price_card, stats_card, chart
        except Exception as e:
            logger.error(f"Error updating market data: {str(e)}")
            error_message = create_error_message(
                f"Failed to update market data: {str(e)}"
            )
            return error_message, error_message, error_message

    # Rest of the callbacks remain the same...
    # ... existing code ...


# Helper functions for creating UI components
def create_price_card(price_data: Dict[str, Any], symbol: str) -> html.Div:
    """
    Create a price information card.

    Args:
        price_data: Dictionary containing price information
        symbol: Trading symbol

    Returns:
        A Dash HTML Div containing the price card
    """
    # Extract price information
    current_price = price_data.get("current_price", "N/A")
    price_change = price_data.get("price_change", 0)
    price_change_pct = price_data.get("price_change_pct", 0)

    # Determine color based on price change
    color_class = "text-success" if price_change >= 0 else "text-danger"
    change_symbol = "+" if price_change >= 0 else ""

    return html.Div(
        [
            html.H4(f"{symbol} Price", className="card-title"),
            html.H2(f"{current_price}", className="price-value"),
            html.Div(
                [
                    html.Span(
                        f"{change_symbol}{price_change:.2f} ", className=color_class
                    ),
                    html.Span(
                        f"({change_symbol}{price_change_pct:.2f}%)",
                        className=color_class,
                    ),
                ],
                className="price-change",
            ),
        ],
        className="price-card",
    )


def create_stats_card(market_stats: Dict[str, Any]) -> html.Div:
    """
    Create a market statistics card.

    Args:
        market_stats: Dictionary containing market statistics

    Returns:
        A Dash HTML Div containing the stats card
    """
    # Create stat rows
    stat_rows = []
    for label, value in market_stats.items():
        formatted_label = label.replace("_", " ").title()
        stat_rows.append(
            html.Div(
                [
                    html.Span(formatted_label, className="stat-label"),
                    html.Span(str(value), className="stat-value"),
                ],
                className="stat-row",
            )
        )

    return html.Div(
        [
            html.H4("Market Statistics", className="card-title"),
            html.Div(stat_rows, className="stats-container"),
        ],
        className="stats-card",
    )
