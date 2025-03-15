"""
Market Component Callbacks

This module registers callbacks for market data components.
"""

from typing import Dict, Any, Callable, Optional
import dash
from dash import html, dcc, callback, Input, Output, State
import plotly.graph_objs as go
import pandas as pd
from loguru import logger

from src.dashboard.components.market.market_panel import create_market_panel
from src.dashboard.components.error_display import create_error_message
from src.dashboard.services.chart_service import create_candlestick_chart


def register_market_callbacks(app: dash.Dash, get_market_data_func: Callable) -> None:
    """
    Register callbacks for market data components.
    
    Args:
        app: The Dash application instance
        get_market_data_func: Function to get market data
    """
    logger.debug("Registering market callbacks")
    
    @app.callback(
        [Output("market-price-card", "children"),
         Output("market-stats-card", "children"),
         Output("market-chart-container", "children")],
        [Input("market-interval-component", "n_intervals"),
         Input("market-symbol-dropdown", "value"),
         Input("market-timeframe-dropdown", "value")],
        prevent_initial_call=False
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
            # Get market data using the provided function
            market_data = get_market_data_func(symbol)
            
            if not market_data or "error" in market_data:
                return (
                    create_error_message("No market data available"),
                    create_error_message("No market statistics available"),
                    create_error_message("No chart data available")
                )
            
            # Create price card
            price_card = html.Div([
                html.H4(f"{market_data['symbol']} Price"),
                html.H2(f"${market_data['last_price']:.2f}"),
                html.Div([
                    html.Span(
                        f"{market_data['change_pct_24h']:.2f}%",
                        style={
                            "color": "green" if market_data['change_pct_24h'] >= 0 else "red",
                            "font-weight": "bold"
                        }
                    ),
                    html.Span(" (24h)", style={"color": "gray"})
                ])
            ])
            
            # Create stats card
            stats_card = html.Div([
                html.H4("Market Statistics"),
                html.Div([
                    html.Div([
                        html.Span("Bid: ", style={"font-weight": "bold"}),
                        html.Span(f"${market_data['bid']:.2f}")
                    ]),
                    html.Div([
                        html.Span("Ask: ", style={"font-weight": "bold"}),
                        html.Span(f"${market_data['ask']:.2f}")
                    ]),
                    html.Div([
                        html.Span("Spread: ", style={"font-weight": "bold"}),
                        html.Span(f"${market_data['spread']:.2f} ({market_data.get('spread_pct', 0):.3f}%)")
                    ]),
                    html.Div([
                        html.Span("24h High: ", style={"font-weight": "bold"}),
                        html.Span(f"${market_data['high_24h']:.2f}")
                    ]),
                    html.Div([
                        html.Span("24h Low: ", style={"font-weight": "bold"}),
                        html.Span(f"${market_data['low_24h']:.2f}")
                    ]),
                    html.Div([
                        html.Span("24h Volume: ", style={"font-weight": "bold"}),
                        html.Span(f"${market_data['volume_24h']:,.2f}")
                    ])
                ])
            ])
            
            # Create price chart
            candles = market_data.get('candles', pd.DataFrame())
            
            if candles.empty:
                chart = create_error_message("No candle data available")
            else:
                # Filter by timeframe if specified
                if timeframe and timeframe != "all":
                    from src.dashboard.utils.transformers import DataTransformer
                    candles = DataTransformer.filter_data_by_time_range(
                        candles, 
                        time_range=timeframe,
                        date_column="timestamp"
                    )
                
                # Use chart service to create candlestick chart
                fig = create_candlestick_chart(
                    candles=candles,
                    symbol=market_data['symbol'],
                    title=f"{market_data['symbol']} Price Chart",
                    show_volume=True,
                    height=500
                )
                
                chart = dcc.Graph(figure=fig)
            
            return price_card, stats_card, chart
            
        except Exception as e:
            logger.error(f"Error updating market data: {str(e)}")
            return (
                create_error_message(f"Error: {str(e)}"),
                create_error_message(f"Error: {str(e)}"),
                create_error_message(f"Error: {str(e)}")
            ) 