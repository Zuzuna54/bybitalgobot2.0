"""
Event-based Data Updates Example

This example demonstrates how to use the EventManager and EventDataService
to implement event-based real-time data updates in the dashboard.
"""

import time
import json
import threading
from datetime import datetime
import dash
from dash import dcc, html, callback
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

from src.dashboard.services.data_service.base import DashboardDataService
from src.dashboard.services.event_manager import EventManager, EventType, event_manager
from src.dashboard.services.event_data_service import EventDataService


# Initialize data service and event data service
data_service = DashboardDataService()
event_data_service = EventDataService(data_service)

# Initialize Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)
app.title = "Event-based Updates Example"

# Create layout
app.layout = html.Div(
    [
        html.H1("Event-based Data Updates Example"),
        # Market data section
        html.Div(
            [
                html.H2("Market Data"),
                html.Div(id="market-data-display"),
                # Use interval component for polling
                dcc.Interval(
                    id="market-data-interval",
                    interval=1000,  # 1 second in milliseconds
                    n_intervals=0,
                ),
            ]
        ),
        # Performance data section
        html.Div(
            [
                html.H2("Performance Data"),
                html.Div(id="performance-data-display"),
                # Use interval component for polling
                dcc.Interval(
                    id="performance-data-interval",
                    interval=5000,  # 5 seconds in milliseconds
                    n_intervals=0,
                ),
            ]
        ),
        # System status section
        html.Div(
            [
                html.H2("System Status"),
                html.Div(id="system-status-display"),
                # Use interval component for polling
                dcc.Interval(
                    id="system-status-interval",
                    interval=10000,  # 10 seconds in milliseconds
                    n_intervals=0,
                ),
            ]
        ),
        # Data version info
        html.Div(
            [
                html.H3("Data Versions"),
                html.Pre(id="data-versions-display"),
                dcc.Interval(
                    id="data-versions-interval",
                    interval=1000,  # 1 second
                    n_intervals=0,
                ),
            ]
        ),
        # Simulation controls
        html.Div(
            [
                html.H3("Simulation Controls"),
                html.Button(
                    "Simulate Market Update", id="simulate-market-btn", className="mr-2"
                ),
                html.Button(
                    "Simulate Performance Update",
                    id="simulate-performance-btn",
                    className="mr-2",
                ),
                html.Button("Simulate System Update", id="simulate-system-btn"),
            ],
            style={"margin-top": "20px"},
        ),
    ]
)


# Callback to display market data
@app.callback(
    Output("market-data-display", "children"),
    [Input("market-data-interval", "n_intervals")],
)
def update_market_display(n_intervals):
    # Get market data from data service
    market_data = data_service.get_market_data("BTCUSDT")

    if not market_data:
        return html.Div("No market data available")

    # Format and display data
    return html.Div(
        [
            html.P(f"Last Price: {market_data.get('last_price', 'N/A')}"),
            html.P(f"24h Change: {market_data.get('price_change_24h', 'N/A')}%"),
            html.P(f"Volume: {market_data.get('volume_24h', 'N/A')}"),
            html.P(f"Updated at: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}"),
        ]
    )


# Callback to display performance data
@app.callback(
    Output("performance-data-display", "children"),
    [Input("performance-data-interval", "n_intervals")],
)
def update_performance_display(n_intervals):
    # Get performance data from data service
    performance_data = data_service.get_performance_data()

    if not performance_data or not performance_data.get("summary"):
        return html.Div("No performance data available")

    # Get summary metrics
    summary = performance_data.get("summary", {})

    # Format and display data
    return html.Div(
        [
            html.P(f"Total Return: {summary.get('total_return', 'N/A')}"),
            html.P(f"Win Rate: {summary.get('win_rate', 'N/A')}"),
            html.P(f"Profit Factor: {summary.get('profit_factor', 'N/A')}"),
            html.P(f"Updated at: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}"),
        ]
    )


# Callback to display system status
@app.callback(
    Output("system-status-display", "children"),
    [Input("system-status-interval", "n_intervals")],
)
def update_system_status_display(n_intervals):
    # Get system status from data service
    system_status = data_service.get_system_status()

    if not system_status:
        return html.Div("No system status available")

    # Format and display data
    return html.Div(
        [
            html.P(f"Status: {system_status.get('status', 'Unknown')}"),
            html.P(f"Uptime: {system_status.get('uptime', 'N/A')}"),
            html.P(f"CPU Usage: {system_status.get('cpu_usage', 'N/A')}%"),
            html.P(f"Memory Usage: {system_status.get('memory_usage', 'N/A')}%"),
            html.P(f"Updated at: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}"),
        ]
    )


# Callback to display data versions
@app.callback(
    Output("data-versions-display", "children"),
    [Input("data-versions-interval", "n_intervals")],
)
def update_data_versions_display(n_intervals):
    # Get data versions
    versions = {
        data_type: data_service.get_data_version(data_type)
        for data_type in [
            "market",
            "performance",
            "system",
            "trades",
            "orderbook",
            "strategy",
        ]
    }

    # Format as JSON
    return json.dumps(versions, indent=2)


# Callbacks for simulation buttons
@app.callback(
    Output("market-data-display", "style"), [Input("simulate-market-btn", "n_clicks")]
)
def simulate_market_update(n_clicks):
    if n_clicks:
        # Simulate a market data update event
        event_data_service.publish_data_update(
            "market",
            {
                "symbol": "BTCUSDT",
                "market_data": {
                    "last_price": round(50000 + (n_clicks * 100), 2),
                    "price_change_24h": round((n_clicks * 0.5) % 10, 2),
                    "volume_24h": round(1000000 + (n_clicks * 10000), 2),
                },
            },
        )

    # No style change, just trigger the update
    return {}


@app.callback(
    Output("performance-data-display", "style"),
    [Input("simulate-performance-btn", "n_clicks")],
)
def simulate_performance_update(n_clicks):
    if n_clicks:
        # Simulate a performance data update event
        event_data_service.publish_data_update(
            "performance",
            {
                "metrics": {
                    "summary": {
                        "total_return": f"{round((n_clicks * 0.5) % 30, 2)}%",
                        "win_rate": f"{round(50 + (n_clicks * 0.5) % 30, 2)}%",
                        "profit_factor": round(1 + (n_clicks * 0.05) % 2, 2),
                    }
                }
            },
        )

    # No style change, just trigger the update
    return {}


@app.callback(
    Output("system-status-display", "style"), [Input("simulate-system-btn", "n_clicks")]
)
def simulate_system_update(n_clicks):
    if n_clicks:
        # Simulate a system status update event
        event_data_service.publish_data_update(
            "system",
            {
                "status": "Running" if n_clicks % 2 == 0 else "Warning",
                "uptime": f"{n_clicks * 10} minutes",
                "cpu_usage": round((n_clicks * 5) % 80, 1),
                "memory_usage": round((n_clicks * 3) % 90, 1),
            },
        )

    # No style change, just trigger the update
    return {}


# Simulate data generation thread
def generate_sample_data():
    """Generate sample data updates in a background thread."""
    while True:
        try:
            # Simulate market data updates every 2 seconds
            event_data_service.publish_data_update(
                "market",
                {
                    "symbol": "BTCUSDT",
                    "market_data": {
                        "last_price": round(50000 + (time.time() % 1000), 2),
                        "price_change_24h": round((-5 + (time.time() / 100) % 10), 2),
                        "volume_24h": round(1000000 + (time.time() % 500000), 2),
                    },
                },
            )

            # Sleep between updates
            time.sleep(2)

            # Simulate performance data updates every 5 seconds
            if int(time.time()) % 5 == 0:
                event_data_service.publish_data_update(
                    "performance",
                    {
                        "metrics": {
                            "summary": {
                                "total_return": f"{round(10 + (time.time() / 1000) % 20, 2)}%",
                                "win_rate": f"{round(50 + (time.time() / 500) % 30, 2)}%",
                                "profit_factor": round(
                                    1 + (time.time() / 10000) % 2, 2
                                ),
                            }
                        }
                    },
                )

            # Simulate system status updates every 10 seconds
            if int(time.time()) % 10 == 0:
                event_data_service.publish_data_update(
                    "system",
                    {
                        "status": "Running",
                        "uptime": f"{int(time.time() % 3600)} seconds",
                        "cpu_usage": round((time.time() / 100) % 80, 1),
                        "memory_usage": round((time.time() / 120) % 90, 1),
                    },
                )

        except Exception as e:
            print(f"Error in data generation thread: {str(e)}")

        # Short sleep to avoid high CPU usage
        time.sleep(0.1)


if __name__ == "__main__":
    # Initialize the event data service
    event_data_service.initialize()

    # Start the data generation thread
    data_thread = threading.Thread(target=generate_sample_data, daemon=True)
    data_thread.start()

    # Run the Dash app
    app.run_server(debug=True, port=8050)
