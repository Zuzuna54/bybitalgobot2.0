"""
Strategy Panel Callbacks

This module provides callback functions for the strategy panel components
to handle interactive updates and data refresh.
"""

from typing import Dict, Any, List, Tuple, Callable
import dash
from dash import Output, Input, State, callback, html, dcc
import dash_bootstrap_components as dbc
import json
import plotly.graph_objects as go
import pandas as pd

from src.dashboard.services.chart_service import (
    create_strategy_performance_graph,
    create_strategy_comparison_graph,
    create_detailed_performance_breakdown,
    create_market_condition_performance,
    create_strategy_correlation_matrix
)
from .performance_view import render_top_strategies_card
from .signals_view import render_recent_signals_table


def register_strategy_callbacks(app: dash.Dash, get_strategy_data_func: Callable[[], Dict[str, Any]], strategy_manager=None) -> None:
    """
    Register all callback functions for strategy panel components.
    
    Args:
        app: Dash application instance
        get_strategy_data_func: Function to get strategy data
        strategy_manager: Optional strategy manager instance for activation/deactivation
    
    This function sets up the callbacks for:
    - Updating strategy performance graph based on new data
    - Updating top strategies card based on new data
    - Updating recent signals table based on new data
    - Updating strategy comparison graph based on selected strategies
    - Storing available strategy names for dropdown
    - Activating/deactivating strategies
    - Displaying detailed performance breakdowns
    """
    @app.callback(
        Output("strategy-names-store", "data"),
        [Input("strategy-update-interval", "n_intervals")]
    )
    def update_strategy_names(n_intervals):
        # Get strategy data
        strategy_data = get_strategy_data_func()
        
        if not strategy_data or 'strategy_performance' not in strategy_data:
            return []
        
        # Extract strategy names
        strategy_performance = strategy_data.get('strategy_performance', [])
        return [s['strategy_name'] for s in strategy_performance]
    
    @app.callback(
        Output("strategy-comparison-dropdown", "options"),
        [Input("strategy-names-store", "data")]
    )
    def update_strategy_dropdown(strategy_names):
        if not strategy_names:
            return []
        
        # Create dropdown options
        return [{'label': name, 'value': name} for name in strategy_names]
    
    # Separate callback for strategy detail dropdown
    @app.callback(
        Output("strategy-detail-dropdown", "options"),
        [Input("strategy-names-store", "data")]
    )
    def update_strategy_detail_dropdown(strategy_names):
        if not strategy_names:
            return []
        
        # Create dropdown options
        return [{'label': name, 'value': name} for name in strategy_names]
    
    # Main performance metrics callback
    @app.callback(
        [
            Output("strategy-performance-graph", "figure"),
            Output("top-strategies-content", "children"),
            Output("recent-signals-content", "children")
        ],
        [Input("strategy-update-interval", "n_intervals")]
    )
    def update_main_strategy_metrics(n_intervals):
        # Get strategy data
        strategy_data = get_strategy_data_func()
        
        if not strategy_data:
            # Return empty/default components if no data
            empty_fig = go.Figure()
            empty_fig.update_layout(template="plotly_white")
            return (
                empty_fig,
                html.Div("No strategy data available", className="no-data-message"),
                html.Div("No signals data available", className="no-data-message")
            )
        
        # Get data
        strategy_performance = strategy_data.get('strategy_performance', [])
        recent_signals = strategy_data.get('recent_signals', [])
        
        # Create components
        performance_graph = create_strategy_performance_graph(strategy_performance)
        top_strategies = render_top_strategies_card(strategy_performance)
        signals_table = render_recent_signals_table(recent_signals)
        
        return performance_graph, top_strategies, signals_table
    
    # Comparison graph callback
    @app.callback(
        Output("strategy-comparison-graph", "figure"),
        [
            Input("strategy-update-interval", "n_intervals"),
            Input("strategy-comparison-dropdown", "value")
        ]
    )
    def update_comparison_graph(n_intervals, selected_strategies):
        # Get strategy data
        strategy_data = get_strategy_data_func()
        
        if not strategy_data:
            empty_fig = go.Figure()
            empty_fig.update_layout(
                title="No strategy data available for comparison",
                template="plotly_white"
            )
            return empty_fig
        
        # Get data
        strategy_performance = strategy_data.get('strategy_performance', [])
        
        # Create comparison graph
        return create_strategy_comparison_graph(
            strategy_performance, 
            selected_strategies if selected_strategies else []
        )
        
    # Correlation matrix callback
    @app.callback(
        Output("strategy-correlation-matrix", "figure"),
        [Input("strategy-update-interval", "n_intervals")]
    )
    def update_correlation_matrix(n_intervals):
        # Get strategy data
        strategy_data = get_strategy_data_func()
        
        if not strategy_data or 'strategy_performance' not in strategy_data:
            empty_fig = go.Figure()
            empty_fig.update_layout(
                title="No strategy data available for correlation analysis",
                template="plotly_white"
            )
            return empty_fig
        
        # Get performance data
        strategy_performance = strategy_data.get('strategy_performance', [])
        
        # Create correlation matrix
        return create_strategy_correlation_matrix(strategy_performance)
    
    # Detailed performance callback
    @app.callback(
        Output("detailed-performance-breakdown", "figure"),
        [
            Input("strategy-detail-dropdown", "value"),
            Input("strategy-update-interval", "n_intervals")
        ]
    )
    def update_detailed_performance(selected_strategy, n_intervals):
        # Get strategy data
        strategy_data = get_strategy_data_func()
        
        if not strategy_data or 'strategy_performance' not in strategy_data:
            empty_fig = go.Figure()
            empty_fig.update_layout(
                title="No strategy data available",
                template="plotly_white"
            )
            return empty_fig
        
        # Get performance data
        strategy_performance = strategy_data.get('strategy_performance', [])
        
        # Create detailed breakdown
        return create_detailed_performance_breakdown(strategy_performance, selected_strategy)
    
    # Market condition performance callback
    @app.callback(
        Output("market-condition-performance", "figure"),
        [
            Input("strategy-detail-dropdown", "value"),
            Input("strategy-update-interval", "n_intervals")
        ]
    )
    def update_market_condition_performance(selected_strategy, n_intervals):
        # Get strategy data
        strategy_data = get_strategy_data_func()
        
        if not strategy_data or 'strategy_performance' not in strategy_data:
            empty_fig = go.Figure()
            empty_fig.update_layout(
                title="No strategy data available",
                template="plotly_white"
            )
            return empty_fig
        
        # Get performance data
        strategy_performance = strategy_data.get('strategy_performance', [])
        
        # Create market condition performance visualization
        return create_market_condition_performance(strategy_performance, selected_strategy)
    
    # Strategy activation controls callback
    @app.callback(
        Output("strategy-activation-controls", "children"),
        [Input("strategy-update-interval", "n_intervals")]
    )
    def update_activation_controls(n_intervals):
        # Get strategy data
        strategy_data = get_strategy_data_func()
        
        if not strategy_data or 'strategy_performance' not in strategy_data:
            return html.Div("No strategies available", className="no-data-message")
        
        # Get performance data
        strategy_performance = strategy_data.get('strategy_performance', [])
        
        # Create strategy activation controls
        return create_strategy_activation_controls(strategy_performance, strategy_manager)
    
    # Only register strategy activation toggle callback if strategy manager is provided
    if strategy_manager:
        @app.callback(
            Output("strategy-activation-status", "children"),
            [Input({"type": "strategy-toggle", "index": dash.ALL}, "value")],
            [State({"type": "strategy-toggle", "index": dash.ALL}, "id")]
        )
        def toggle_strategy_activation(toggle_values, toggle_ids):
            # Check if we have any toggles
            if not toggle_values or not toggle_ids:
                return html.Div("No changes made")
            
            # Process each toggle
            results = []
            for i, (toggle_value, toggle_id) in enumerate(zip(toggle_values, toggle_ids)):
                strategy_name = toggle_id["index"]
                
                # Determine if we're activating or deactivating
                is_active = bool(toggle_value)
                
                # Call the appropriate method on the strategy manager
                try:
                    if is_active:
                        success = strategy_manager.enable_strategy(strategy_name)
                        status = "activated" if success else "activation failed"
                    else:
                        # Check if this would deactivate all strategies
                        active_strategies = [
                            tid["index"] for tid, val in zip(toggle_ids, toggle_values) 
                            if val and tid["index"] != strategy_name
                        ]
                        
                        if not active_strategies:
                            # Don't allow deactivating all strategies
                            results.append(html.Div(
                                f"Cannot deactivate all strategies - {strategy_name} remains active",
                                className="text-warning"
                            ))
                            continue
                        
                        success = strategy_manager.disable_strategy(strategy_name)
                        status = "deactivated" if success else "deactivation failed"
                    
                    results.append(html.Div(
                        f"Strategy {strategy_name} {status}",
                        className="text-success" if success else "text-danger"
                    ))
                except Exception as e:
                    results.append(html.Div(
                        f"Error toggling {strategy_name}: {str(e)}",
                        className="text-danger"
                    ))
            
            # Return status messages
            return html.Div(results)


def create_strategy_activation_controls(strategy_performance: List[Dict[str, Any]], strategy_manager=None) -> html.Div:
    """
    Create strategy activation toggle controls.
    
    Args:
        strategy_performance: List of strategy performance dictionaries
        strategy_manager: Optional strategy manager instance
        
    Returns:
        HTML Div with strategy activation controls
    """
    if not strategy_performance:
        return html.Div("No strategies available")
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(strategy_performance)
    
    # Create toggle switches for each strategy
    toggle_switches = []
    for _, strategy in df.iterrows():
        strategy_name = strategy['strategy_name']
        
        # Determine if strategy is enabled
        is_enabled = True  # Default to enabled
        if strategy_manager:
            strategy_info = strategy_manager.get_strategy_info()
            if strategy_name in strategy_info:
                is_enabled = strategy_info[strategy_name].get('enabled', True)
        
        # Create toggle switch
        toggle_switches.append(
            html.Div([
                html.Label(
                    [
                        strategy_name,
                        dbc.Switch(
                            id={"type": "strategy-toggle", "index": strategy_name},
                            value=is_enabled,
                            className="ms-2"
                        )
                    ],
                    className="d-flex align-items-center"
                )
            ], className="mb-2")
        )
    
    # Create the controls container
    return html.Div([
        html.H5("Strategy Activation Controls", className="mb-3"),
        html.Div(toggle_switches),
        html.Div(id="strategy-activation-status", className="mt-3")
    ], className="strategy-activation-controls mt-4 p-3 border rounded") 