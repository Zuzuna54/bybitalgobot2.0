"""
Orderbook Panel Callbacks for the Trading Dashboard

This module provides callbacks for updating the orderbook panel components.
"""

from typing import Dict, Any, List, Callable, Tuple
import dash
import plotly.graph_objects as go
from dash import html, dcc
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

from src.dashboard.components.orderbook.visualization import (
    render_imbalance_indicator,
    render_liquidity_ratio,
    render_support_resistance_levels,
    render_execution_recommendations
)
from src.dashboard.components.orderbook.data_processing import (
    calculate_orderbook_imbalance,
    calculate_liquidity_ratio,
    identify_support_resistance_levels,
    generate_execution_recommendations
)
from src.dashboard.services.chart_service import create_orderbook_depth_chart


def register_orderbook_callbacks(app: dash.Dash, get_orderbook_data_func: Callable) -> None:
    """
    Register the callbacks for the order book panel.
    
    Args:
        app: Dash application instance
        get_orderbook_data_func: Function to get order book data
    """
    @app.callback(
        [
            Output("available-symbols-store", "data"),
            Output("websocket-status-store", "data"),
            Output("data-freshness-store", "data")
        ],
        [Input("orderbook-update-interval", "n_intervals")]
    )
    def update_available_symbols_and_status(n_intervals):
        """Update the available symbols store and WebSocket status."""
        # Get order book data for all symbols
        orderbook_data = get_orderbook_data_func()
        
        # Default values
        available_symbols = []
        websocket_status = {"status": "disconnected", "active_connections": 0, "total_connections": 0}
        freshness_data = {"status": "stale", "last_update": "Never updated"}
        
        if not orderbook_data:
            return available_symbols, websocket_status, freshness_data
        
        # Extract available symbols
        if 'symbols' in orderbook_data:
            available_symbols = list(orderbook_data['symbols'].keys())
        
        # Extract WebSocket status
        if 'websocket_status' in orderbook_data:
            websocket_status = {
                "connections": orderbook_data['websocket_status'],
                "active_connections": sum(1 for conn in orderbook_data['websocket_status'].values() 
                                        if conn.get('status') == 'connected'),
                "total_connections": len(orderbook_data['websocket_status'])
            }
        
        # Extract freshness data
        if 'freshness' in orderbook_data:
            freshness_data = orderbook_data['freshness']
        
        return available_symbols, websocket_status, freshness_data
    
    @app.callback(
        Output("orderbook-symbol-dropdown", "options"),
        [Input("available-symbols-store", "data")]
    )
    def update_symbol_dropdown(available_symbols):
        """Update the symbol dropdown options."""
        if not available_symbols:
            return []
        
        # Create dropdown options
        return [{'label': symbol, 'value': symbol} for symbol in available_symbols]
    
    @app.callback(
        [
            Output("websocket-status-badge", "children"),
            Output("websocket-status-badge", "className"),
            Output("orderbook-last-update-time", "children"),
            Output("ws-connection-details", "children")
        ],
        [
            Input("websocket-status-store", "data"),
            Input("data-freshness-store", "data")
        ]
    )
    def update_websocket_status_indicators(websocket_status, freshness_data):
        """Update the WebSocket status indicators."""
        # Default values
        badge_text = "Disconnected"
        badge_class = "connection-status-badge status-disconnected"
        last_update = "Never"
        connection_details = html.Div("No connection details available")
        
        # Update from freshness data
        if freshness_data:
            last_update = freshness_data.get("last_update", "Never")
        
        # Update from WebSocket status
        if websocket_status:
            active_connections = websocket_status.get("active_connections", 0)
            total_connections = websocket_status.get("total_connections", 0)
            
            if active_connections > 0:
                badge_text = f"Connected ({active_connections}/{total_connections})"
                badge_class = "connection-status-badge status-connected"
            else:
                if total_connections > 0:
                    badge_text = f"Partially Connected (0/{total_connections})"
                    badge_class = "connection-status-badge status-warning"
            
            # Generate connection details
            connections = websocket_status.get("connections", {})
            if connections:
                # Create a table to display connection details
                connection_rows = []
                for conn_id, conn_info in connections.items():
                    status_class = {
                        "connected": "text-success",
                        "connecting": "text-warning",
                        "reconnecting": "text-warning",
                        "error": "text-danger"
                    }.get(conn_info.get("status", ""), "text-secondary")
                    
                    row = html.Tr([
                        html.Td(conn_info.get("display_name", conn_id)),
                        html.Td(html.Span(conn_info.get("status", "unknown"), className=status_class)),
                        html.Td(f"{conn_info.get('seconds_since_last_message', 0):.1f}s ago"),
                        html.Td(conn_info.get("reconnection_attempts", 0))
                    ])
                    connection_rows.append(row)
                
                if connection_rows:
                    connection_details = html.Div([
                        html.H6("Active Connections"),
                        html.Table([
                            html.Thead(html.Tr([
                                html.Th("Connection"),
                                html.Th("Status"),
                                html.Th("Last Message"),
                                html.Th("Reconnections")
                            ])),
                            html.Tbody(connection_rows)
                        ], className="table table-sm")
                    ])
                else:
                    connection_details = html.Div("No active connections")
        
        return badge_text, badge_class, last_update, connection_details
    
    @app.callback(
        [
            Output("orderbook-data-freshness-indicator", "children"),
            Output("orderbook-data-freshness-indicator", "className"),
            Output("liquidity-data-freshness-indicator", "children"),
            Output("liquidity-data-freshness-indicator", "className"),
            Output("support-resistance-freshness-indicator", "children"),
            Output("support-resistance-freshness-indicator", "className")
        ],
        [Input("data-freshness-store", "data")]
    )
    def update_data_freshness_indicators(freshness_data):
        """Update the data freshness indicators."""
        # Default values
        freshness_text = "●"
        orderbook_class = "data-freshness-indicator status-stale"
        liquidity_class = "data-freshness-indicator status-stale"
        support_resistance_class = "data-freshness-indicator status-stale"
        
        if freshness_data:
            status = freshness_data.get("status", "stale")
            
            if status == "fresh":
                orderbook_class = "data-freshness-indicator status-fresh"
                liquidity_class = "data-freshness-indicator status-fresh"
                support_resistance_class = "data-freshness-indicator status-fresh"
            elif status == "warning":
                orderbook_class = "data-freshness-indicator status-warning"
                liquidity_class = "data-freshness-indicator status-warning"
                support_resistance_class = "data-freshness-indicator status-warning"
        
        return freshness_text, orderbook_class, freshness_text, liquidity_class, freshness_text, support_resistance_class
    
    @app.callback(
        [
            Output("orderbook-imbalance-indicator", "children"),
            Output("liquidity-ratio-indicator", "children"),
            Output("orderbook-depth-graph", "figure"),
            Output("support-resistance-content", "children"),
            Output("execution-recommendations-content", "children")
        ],
        [
            Input("orderbook-update-interval", "n_intervals"),
            Input("orderbook-symbol-dropdown", "value")
        ],
        [State("risk-tolerance-slider", "value")]
    )
    def update_orderbook_panel(n_intervals, selected_symbol, risk_tolerance=50):
        """Update all orderbook panel components based on selected symbol."""
        # Get order book data
        orderbook_data = get_orderbook_data_func()
        
        # Normalize risk tolerance from slider (0-100) to 0.0-1.0 scale
        risk_tolerance_normalized = risk_tolerance / 100.0 if risk_tolerance is not None else 0.5
        
        if not orderbook_data or 'symbols' not in orderbook_data or not selected_symbol:
            # Return empty/default components if no data
            empty_fig = go.Figure()
            empty_fig.update_layout(template="plotly_white")
            return (
                html.Div("Select a symbol", className="text-center text-muted"),
                html.Div("Select a symbol", className="text-center text-muted"),
                empty_fig,
                html.Div("Select a symbol", className="text-center text-muted"),
                html.Div([
                    html.P("Set your risk tolerance:"),
                    dcc.Slider(
                        id="risk-tolerance-slider",
                        min=0,
                        max=100,
                        step=5,
                        value=50,
                        marks={
                            0: "Low Risk",
                            50: "Balanced",
                            100: "High Risk"
                        }
                    ),
                    html.P("Select a symbol to view recommendations", className="mt-3 text-center text-muted")
                ])
            )
        
        # Get data for selected symbol
        symbol_data = orderbook_data['symbols'].get(selected_symbol, {})
        
        if not symbol_data or 'orderbook' not in symbol_data:
            # Return empty/default components if no data for selected symbol
            empty_fig = go.Figure()
            empty_fig.update_layout(template="plotly_white")
            return (
                html.Div(f"No orderbook data for {selected_symbol}", className="text-center text-danger"),
                html.Div(f"No orderbook data for {selected_symbol}", className="text-center text-danger"),
                empty_fig,
                html.Div(f"No orderbook data for {selected_symbol}", className="text-center text-danger"),
                html.Div([
                    html.P("Set your risk tolerance:"),
                    dcc.Slider(
                        id="risk-tolerance-slider",
                        min=0,
                        max=100,
                        step=5,
                        value=50,
                        marks={
                            0: "Low Risk",
                            50: "Balanced",
                            100: "High Risk"
                        }
                    ),
                    html.P(f"No data available for {selected_symbol}", className="mt-3 text-center text-danger")
                ])
            )
        
        # Extract data
        orderbook = symbol_data['orderbook']
        liquidity = symbol_data.get('liquidity', {})
        imbalance = symbol_data.get('imbalance', {})
        support_levels = symbol_data.get('support_levels', [])
        resistance_levels = symbol_data.get('resistance_levels', [])
        
        # Calculate metrics if not already available
        if not imbalance:
            imbalance = calculate_orderbook_imbalance(orderbook)
        
        if not liquidity:
            liquidity = calculate_liquidity_ratio(orderbook)
        
        if not support_levels or not resistance_levels:
            levels = identify_support_resistance_levels(orderbook)
            support_levels = levels['support_levels']
            resistance_levels = levels['resistance_levels']
        
        # Generate execution recommendations based on orderbook analysis and risk tolerance
        recommendations = generate_execution_recommendations(
            orderbook, 
            imbalance, 
            liquidity, 
            support_levels, 
            resistance_levels,
            risk_tolerance=risk_tolerance_normalized
        )
        
        # Render components
        imbalance_indicator = render_imbalance_indicator(imbalance)
        liquidity_ratio_indicator = render_liquidity_ratio(liquidity)
        depth_graph = create_orderbook_depth_chart(
            orderbook, 
            depth=20,
            support_levels=support_levels, 
            resistance_levels=resistance_levels
        )
        support_resistance_content = render_support_resistance_levels(support_levels, resistance_levels)
        execution_recommendations_content = render_execution_recommendations(recommendations, risk_tolerance_normalized)
        
        return (
            imbalance_indicator,
            liquidity_ratio_indicator,
            depth_graph,
            support_resistance_content,
            execution_recommendations_content
        )
    
    @app.callback(
        Output("order-size-input", "value"),
        [Input("auto-size-btn", "n_clicks")],
        [State("orderbook-symbol-dropdown", "value")]
    )
    def calculate_optimal_order_size(n_clicks, selected_symbol):
        """Calculate optimal order size based on current market conditions."""
        if n_clicks is None or not selected_symbol:
            return ""
            
        # Get order book data
        orderbook_data = get_orderbook_data_func()
        
        if not orderbook_data or 'symbols' not in orderbook_data:
            return ""
            
        symbol_data = orderbook_data['symbols'].get(selected_symbol, {})
        
        if not symbol_data or 'orderbook' not in symbol_data:
            return ""
            
        orderbook = symbol_data['orderbook']
        
        # Use depth analysis to get optimal trade size
        from src.trade_execution.orderbook.depth_analysis import get_optimal_trade_size
        
        try:
            # Get optimal size with conservative impact (0.3%)
            optimal_size = get_optimal_trade_size(orderbook, max_impact_pct=0.3)
            # Round to appropriate precision based on symbol
            return round(optimal_size, 5)
        except Exception:
            return "" 