"""
Performance Panel Layout Module

This module provides the layout for the performance panel in the dashboard.
It displays trading performance metrics, charts, and historical data.
"""

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from loguru import logger


def create_performance_panel() -> html.Div:
    """
    Create the performance panel layout.
    
    Returns:
        Dash HTML Div containing the performance panel layout
    """
    # Create time range selector
    time_range_selector = dbc.Card([
        dbc.CardHeader("Time Range"),
        dbc.CardBody([
            dbc.RadioItems(
                id="performance-time-range",
                options=[
                    {"label": "1 Day", "value": "1d"},
                    {"label": "1 Week", "value": "1w"},
                    {"label": "1 Month", "value": "1m"},
                    {"label": "3 Months", "value": "3m"},
                    {"label": "All", "value": "all"},
                ],
                value="1w",
                inline=True,
            ),
        ]),
    ])
    
    # Create performance metrics cards
    performance_metrics = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Total Return"),
                dbc.CardBody([
                    html.H3(id="total-return-value", children="0.00%"),
                    dcc.Graph(
                        id="total-return-sparkline",
                        figure=create_empty_sparkline(),
                        config={"displayModeBar": False},
                        style={"height": "50px"}
                    ),
                ]),
            ]),
        ], md=3),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Win Rate"),
                dbc.CardBody([
                    html.H3(id="win-rate-value", children="0.00%"),
                    html.Div(id="win-rate-ratio", children="0 / 0 trades"),
                ]),
            ]),
        ], md=3),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Profit Factor"),
                dbc.CardBody([
                    html.H3(id="profit-factor-value", children="0.00"),
                    html.Div(id="average-trade-value", children="$0.00 avg trade"),
                ]),
            ]),
        ], md=3),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Max Drawdown"),
                dbc.CardBody([
                    html.H3(id="max-drawdown-value", children="0.00%"),
                    html.Div(id="drawdown-duration", children="0 days duration"),
                ]),
            ]),
        ], md=3),
    ])
    
    # Create main performance charts
    performance_charts = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Equity Curve"),
                dbc.CardBody([
                    dcc.Graph(
                        id="equity-curve-chart",
                        figure=create_empty_chart("Equity Curve"),
                        style={"height": "300px"}
                    ),
                ]),
            ]),
        ], md=12),
    ])
    
    # Create trade analytics section
    trade_analytics = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Trade Distribution"),
                dbc.CardBody([
                    dcc.Graph(
                        id="trade-distribution-chart",
                        figure=create_empty_chart("Trade Distribution"),
                        style={"height": "250px"}
                    ),
                ]),
            ]),
        ], md=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Strategy Performance"),
                dbc.CardBody([
                    dcc.Graph(
                        id="strategy-performance-chart",
                        figure=create_empty_chart("Strategy Performance"),
                        style={"height": "250px"}
                    ),
                ]),
            ]),
        ], md=6),
    ])
    
    # Create recent trades table
    recent_trades = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Recent Trades"),
                dbc.CardBody([
                    html.Div(id="recent-trades-table", className="data-table"),
                ]),
            ]),
        ], md=12),
    ])
    
    # Assemble the complete panel
    return html.Div([
        time_range_selector,
        html.Br(),
        performance_metrics,
        html.Br(),
        performance_charts,
        html.Br(),
        trade_analytics,
        html.Br(),
        recent_trades,
    ])


def create_empty_chart(title: str) -> go.Figure:
    """
    Create an empty chart figure with a title.
    
    Args:
        title: The chart title
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    fig.update_layout(
        title=title,
        template="plotly_white",
        showlegend=False,
        xaxis=dict(showgrid=True, zeroline=False, showticklabels=True),
        yaxis=dict(showgrid=True, zeroline=False, showticklabels=True),
        margin=dict(l=40, r=40, t=40, b=40),
    )
    fig.add_annotation(
        x=0.5, y=0.5,
        text="No data available",
        showarrow=False,
        font=dict(size=16)
    )
    return fig


def create_empty_sparkline() -> go.Figure:
    """
    Create an empty sparkline chart.
    
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    # Add a flat line
    x = list(range(10))
    y = [0] * 10
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color="gray", width=1)))
    return fig


def register_performance_callbacks(app: dash.Dash) -> None:
    """
    Register callbacks for the performance panel.
    
    Args:
        app: The Dash application instance
    """
    @app.callback(
        [
            dash.Output("total-return-value", "children"),
            dash.Output("win-rate-value", "children"),
            dash.Output("win-rate-ratio", "children"),
            dash.Output("profit-factor-value", "children"),
            dash.Output("average-trade-value", "children"),
            dash.Output("max-drawdown-value", "children"),
            dash.Output("drawdown-duration", "children"),
        ],
        [dash.Input("data-update-interval", "n_intervals"),
         dash.Input("performance-time-range", "value")]
    )
    def update_performance_metrics(n_intervals, time_range):
        """
        Update performance metric displays.
        
        Args:
            n_intervals: Update interval counter
            time_range: Selected time range
            
        Returns:
            Updated performance metrics
        """
        # Placeholder values - in a real implementation, these would come from the data provider
        total_return = "5.23%"
        win_rate = "62.5%"
        win_rate_ratio = "15 / 24 trades"
        profit_factor = "2.34"
        average_trade = "$32.45 avg trade"
        max_drawdown = "7.82%"
        drawdown_duration = "3 days duration"
        
        return total_return, win_rate, win_rate_ratio, profit_factor, average_trade, max_drawdown, drawdown_duration
    
    @app.callback(
        dash.Output("equity-curve-chart", "figure"),
        [dash.Input("data-update-interval", "n_intervals"),
         dash.Input("performance-time-range", "value")]
    )
    def update_equity_curve(n_intervals, time_range):
        """
        Update the equity curve chart.
        
        Args:
            n_intervals: Update interval counter
            time_range: Selected time range
            
        Returns:
            Updated equity curve chart
        """
        # Generate sample data for demonstration
        days = 30  # Default to a month of data
        if time_range == "1d":
            days = 1
        elif time_range == "1w":
            days = 7
        elif time_range == "1m":
            days = 30
        elif time_range == "3m":
            days = 90
        
        # Create sample data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq="D")
        
        # Create a sample equity curve with some randomness
        np.random.seed(42)  # For reproducibility
        equity = 10000.0  # Starting equity
        equities = [equity]
        
        for i in range(1, len(dates)):
            daily_return = np.random.normal(0.001, 0.01)  # Mean 0.1%, std 1%
            equity *= (1 + daily_return)
            equities.append(equity)
        
        # Create the figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=equities,
            mode="lines",
            line=dict(color="green", width=2),
            name="Equity"
        ))
        
        # Add drawdown shading
        drawdowns = []
        peak = equities[0]
        for e in equities:
            peak = max(peak, e)
            drawdowns.append((peak - e) / peak * 100)  # Drawdown as percentage
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=[equities[i] - equities[i] * drawdowns[i] / 100 for i in range(len(equities))],
            mode="none",
            fill="tonexty",
            fillcolor="rgba(255,0,0,0.1)",
            name="Drawdown"
        ))
        
        # Update layout
        fig.update_layout(
            title="Equity Curve",
            template="plotly_white",
            showlegend=True,
            xaxis=dict(showgrid=True, zeroline=False, title="Date"),
            yaxis=dict(showgrid=True, zeroline=False, title="Equity ($)"),
            margin=dict(l=40, r=40, t=40, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
    
    @app.callback(
        dash.Output("total-return-sparkline", "figure"),
        [dash.Input("data-update-interval", "n_intervals")]
    )
    def update_return_sparkline(n_intervals):
        """
        Update the total return sparkline.
        
        Args:
            n_intervals: Update interval counter
            
        Returns:
            Updated sparkline chart
        """
        # Generate sample data
        x = list(range(30))
        y = [10000 * (1 + 0.0023) ** i for i in range(30)]
        
        # Create the figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(color="green", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(0,255,0,0.1)"
        ))
        
        # Update layout
        fig.update_layout(
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=0, r=0, t=0, b=0),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        
        return fig
    
    @app.callback(
        dash.Output("recent-trades-table", "children"),
        [dash.Input("data-update-interval", "n_intervals")]
    )
    def update_recent_trades(n_intervals):
        """
        Update the recent trades table.
        
        Args:
            n_intervals: Update interval counter
            
        Returns:
            Updated trades table
        """
        # Sample trade data for demonstration
        trades = [
            {"date": "2023-03-15 14:30", "symbol": "BTCUSDT", "side": "Buy", "strategy": "EMA Cross", "price": "25432.50", "quantity": "0.05", "pnl": "+$35.20"},
            {"date": "2023-03-14 09:45", "symbol": "ETHUSDT", "side": "Sell", "strategy": "RSI Oversold", "price": "1645.75", "quantity": "0.3", "pnl": "-$12.80"},
            {"date": "2023-03-13 22:15", "symbol": "SOLUSDT", "side": "Buy", "strategy": "Bollinger Band", "price": "21.34", "quantity": "10", "pnl": "+$28.60"},
            {"date": "2023-03-12 11:05", "symbol": "BNBUSDT", "side": "Sell", "strategy": "MACD Signal", "price": "305.20", "quantity": "0.5", "pnl": "+$42.15"},
            {"date": "2023-03-11 16:50", "symbol": "ADAUSDT", "side": "Buy", "strategy": "EMA Cross", "price": "0.3245", "quantity": "500", "pnl": "-$8.75"},
        ]
        
        # Create the table
        return dbc.Table([
            html.Thead(html.Tr([
                html.Th("Date/Time"),
                html.Th("Symbol"),
                html.Th("Side"),
                html.Th("Strategy"),
                html.Th("Price"),
                html.Th("Quantity"),
                html.Th("P&L"),
            ])),
            html.Tbody([
                html.Tr([
                    html.Td(trade["date"]),
                    html.Td(trade["symbol"]),
                    html.Td(html.Span(trade["side"], className=f"trade-{trade['side'].lower()}")),
                    html.Td(trade["strategy"]),
                    html.Td(trade["price"]),
                    html.Td(trade["quantity"]),
                    html.Td(html.Span(
                        trade["pnl"],
                        className="trade-profit" if trade["pnl"].startswith("+") else "trade-loss"
                    )),
                ]) for trade in trades
            ])
        ], striped=True, bordered=True, hover=True, size="sm") 