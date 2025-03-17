"""
Callback Dependency Optimizer

This module provides utilities for optimizing callback dependencies,
reducing redundant callback executions, and improving dashboard performance.
"""

import time
import re
import inspect
from typing import Dict, List, Set, Callable, Any, Optional, Union, Tuple
from collections import defaultdict, deque

from dash import Dash, Input, Output, State, ALL, MATCH
from dash.dependencies import DashDependency
import networkx as nx
from loguru import logger

from ..utils.memory_monitor import get_current_memory_usage


class DependencyGraph:
    """
    A directed graph representation of callback dependencies.

    This class builds and manages a graph of callback dependencies,
    enabling optimization through dependency analysis.
    """

    def __init__(self):
        """Initialize the dependency graph."""
        self.graph = nx.DiGraph()
        self.components = {}  # Map component IDs to nodes
        self.callbacks = {}  # Map callback IDs to metadata
        self.callback_counts = defaultdict(int)  # Track execution counts
        self.execution_times = defaultdict(list)  # Track execution times

    def add_callback(
        self,
        callback_id: str,
        inputs: List[Union[Input, State]],
        outputs: List[Output],
        function: Callable,
        priority: int = 5,
    ) -> None:
        """
        Add a callback to the dependency graph.

        Args:
            callback_id: Unique identifier for the callback
            inputs: List of Input and State dependencies
            outputs: List of Output dependencies
            function: The callback function
            priority: Priority level (1-10, higher = more important)
        """
        # Store callback metadata
        self.callbacks[callback_id] = {
            "inputs": inputs,
            "outputs": outputs,
            "function": function,
            "priority": priority,
            "calls": 0,
            "avg_time": 0.0,
            "last_execution": 0.0,
        }

        # Add nodes for each input and output component
        for input_dep in inputs:
            component_id = self._get_component_id(input_dep)
            if component_id not in self.components:
                self.components[component_id] = f"component:{component_id}"
                self.graph.add_node(self.components[component_id], type="component")

        for output_dep in outputs:
            component_id = self._get_component_id(output_dep)
            if component_id not in self.components:
                self.components[component_id] = f"component:{component_id}"
                self.graph.add_node(self.components[component_id], type="component")

        # Add callback node
        callback_node = f"callback:{callback_id}"
        self.graph.add_node(callback_node, type="callback", priority=priority)

        # Add edges from inputs to callback
        for input_dep in inputs:
            component_id = self._get_component_id(input_dep)
            component_node = self.components[component_id]
            self.graph.add_edge(component_node, callback_node, type="input")

        # Add edges from callback to outputs
        for output_dep in outputs:
            component_id = self._get_component_id(output_dep)
            component_node = self.components[component_id]
            self.graph.add_edge(callback_node, component_node, type="output")

        logger.debug(f"Added callback {callback_id} to dependency graph")

    def _get_component_id(self, dependency: DashDependency) -> str:
        """
        Get a string representation of a component ID from a dependency.

        Args:
            dependency: Dash dependency object

        Returns:
            String representation of the component ID
        """
        component_id = dependency.component_id
        property_name = dependency.component_property

        if isinstance(component_id, dict):
            # Handle pattern matching callbacks
            return f"{str(component_id)}:{property_name}"

        return f"{component_id}:{property_name}"

    def track_execution(self, callback_id: str, execution_time: float) -> None:
        """
        Track the execution of a callback.

        Args:
            callback_id: Callback identifier
            execution_time: Execution time in seconds
        """
        if callback_id not in self.callbacks:
            return

        # Update execution metrics
        self.callback_counts[callback_id] += 1
        self.callbacks[callback_id]["calls"] += 1
        self.callbacks[callback_id]["last_execution"] = time.time()

        # Update average execution time
        times = self.execution_times[callback_id]
        times.append(execution_time)
        # Keep only the last 10 execution times
        if len(times) > 10:
            times.pop(0)

        # Calculate new average
        avg_time = sum(times) / len(times)
        self.callbacks[callback_id]["avg_time"] = avg_time

    def find_redundant_callbacks(self) -> List[str]:
        """
        Find potentially redundant callbacks in the graph.

        Returns:
            List of callback IDs that might be redundant
        """
        redundant = []

        # Check for callbacks that update the same outputs
        output_map = defaultdict(list)

        for callback_id, metadata in self.callbacks.items():
            for output in metadata["outputs"]:
                component_id = self._get_component_id(output)
                output_map[component_id].append(callback_id)

        # Find components with multiple callbacks updating them
        for component_id, callback_ids in output_map.items():
            if len(callback_ids) > 1:
                # Sort by priority (lowest first)
                sorted_callbacks = sorted(
                    callback_ids, key=lambda x: self.callbacks[x]["priority"]
                )
                # The lowest priority callbacks might be redundant
                redundant.extend(sorted_callbacks[:-1])

        return redundant

    def find_expensive_callbacks(
        self, threshold_ms: float = 100.0
    ) -> List[Dict[str, Any]]:
        """
        Find expensive callbacks based on average execution time.

        Args:
            threshold_ms: Threshold in milliseconds

        Returns:
            List of expensive callback details
        """
        expensive = []

        for callback_id, metadata in self.callbacks.items():
            avg_time = metadata["avg_time"] * 1000  # Convert to ms

            if avg_time > threshold_ms:
                expensive.append(
                    {
                        "callback_id": callback_id,
                        "avg_time_ms": avg_time,
                        "calls": metadata["calls"],
                        "inputs": len(metadata["inputs"]),
                        "outputs": len(metadata["outputs"]),
                    }
                )

        # Sort by average time (descending)
        return sorted(expensive, key=lambda x: x["avg_time_ms"], reverse=True)

    def find_cascading_callbacks(self) -> List[List[str]]:
        """
        Find cascading callback chains.

        Returns:
            List of callback chains
        """
        # Get all callback nodes
        callback_nodes = [
            node
            for node in self.graph.nodes
            if self.graph.nodes[node].get("type") == "callback"
        ]

        chains = []

        for callback_node in callback_nodes:
            # Find all paths from this callback to other callbacks
            for successor in nx.descendants(self.graph, callback_node):
                if self.graph.nodes[successor].get("type") == "callback":
                    # Find the path(s) between these callbacks
                    for path in nx.all_simple_paths(
                        self.graph, callback_node, successor
                    ):
                        # Extract just the callback IDs from the path
                        callback_chain = [
                            node.split(":")[1]
                            for node in path
                            if self.graph.nodes[node].get("type") == "callback"
                        ]
                        if len(callback_chain) > 1:
                            chains.append(callback_chain)

        return chains

    def optimize_execution_order(self) -> List[str]:
        """
        Determine optimal callback execution order.

        Returns:
            List of callback IDs in optimal execution order
        """
        # Use topological sort for basic ordering
        try:
            topo_order = list(nx.topological_sort(self.graph))

            # Extract callback IDs from the node names
            callback_order = [
                node.split(":")[1]
                for node in topo_order
                if node.startswith("callback:")
            ]

            return callback_order
        except nx.NetworkXUnfeasible:
            # Graph has cycles, fall back to prioritized list
            logger.warning(
                "Dependency graph contains cycles, using priority-based ordering"
            )

            # Sort callbacks by priority (highest first)
            return sorted(
                self.callbacks.keys(), key=lambda x: -self.callbacks[x]["priority"]
            )

    def get_callback_stats(self) -> Dict[str, Any]:
        """
        Get statistics about callbacks and their execution.

        Returns:
            Dictionary of callback statistics
        """
        total_callbacks = len(self.callbacks)
        total_components = len(self.components)

        # Calculate average metrics
        avg_execution_time = 0.0
        total_calls = 0

        if self.callback_counts:
            total_time = sum(
                self.callbacks[cid]["avg_time"] * self.callbacks[cid]["calls"]
                for cid in self.callbacks
            )
            total_calls = sum(self.callback_counts.values())
            if total_calls > 0:
                avg_execution_time = total_time / total_calls

        # Get memory info
        memory_info = get_current_memory_usage()

        # Find top and bottom callbacks by calls
        most_called = sorted(
            self.callbacks.items(), key=lambda x: x[1]["calls"], reverse=True
        )[:5]

        least_called = sorted(self.callbacks.items(), key=lambda x: x[1]["calls"])[:5]

        # Find slowest callbacks
        slowest = sorted(
            self.callbacks.items(), key=lambda x: x[1]["avg_time"], reverse=True
        )[:5]

        return {
            "total_callbacks": total_callbacks,
            "total_components": total_components,
            "total_calls": total_calls,
            "avg_execution_time_ms": avg_execution_time * 1000,
            "memory_usage_mb": memory_info.get("rss", 0) / (1024 * 1024),
            "most_called": [
                {"id": cid, "calls": data["calls"]} for cid, data in most_called
            ],
            "least_called": [
                {"id": cid, "calls": data["calls"]} for cid, data in least_called
            ],
            "slowest_callbacks": [
                {"id": cid, "avg_time_ms": data["avg_time"] * 1000}
                for cid, data in slowest
            ],
        }

    def visualize_graph(self, output_file: str = "callback_graph.html") -> None:
        """
        Create a visualization of the dependency graph.

        Args:
            output_file: File to save the visualization to
        """
        try:
            import plotly.graph_objects as go
            import networkx as nx

            # Create a spring layout for the graph
            pos = nx.spring_layout(self.graph)

            # Create edge traces
            edge_trace = go.Scatter(
                x=[],
                y=[],
                line=dict(width=1, color="#888"),
                hoverinfo="none",
                mode="lines",
            )

            for edge in self.graph.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_trace["x"] += (x0, x1, None)
                edge_trace["y"] += (y0, y1, None)

            # Create node traces for components and callbacks
            component_trace = go.Scatter(
                x=[],
                y=[],
                text=[],
                mode="markers",
                hoverinfo="text",
                marker=dict(
                    showscale=False,
                    color="blue",
                    size=10,
                ),
            )

            callback_trace = go.Scatter(
                x=[],
                y=[],
                text=[],
                mode="markers",
                hoverinfo="text",
                marker=dict(
                    showscale=False,
                    color="red",
                    size=15,
                ),
            )

            # Add component nodes
            for node in self.graph.nodes():
                x, y = pos[node]
                node_type = self.graph.nodes[node].get("type", "unknown")

                if node_type == "component":
                    component_trace["x"] += (x,)
                    component_trace["y"] += (y,)
                    component_trace["text"] += (node,)
                elif node_type == "callback":
                    callback_id = node.split(":")[1]
                    callback_trace["x"] += (x,)
                    callback_trace["y"] += (y,)

                    # Add callback details to hover text
                    if callback_id in self.callbacks:
                        metadata = self.callbacks[callback_id]
                        hover_text = (
                            f"Callback: {callback_id}<br>"
                            f"Calls: {metadata['calls']}<br>"
                            f"Avg time: {metadata['avg_time']*1000:.2f} ms<br>"
                            f"Priority: {metadata['priority']}"
                        )
                    else:
                        hover_text = f"Callback: {callback_id}"

                    callback_trace["text"] += (hover_text,)

            # Create figure
            fig = go.Figure(
                data=[edge_trace, component_trace, callback_trace],
                layout=go.Layout(
                    title="Callback Dependency Graph",
                    showlegend=False,
                    hovermode="closest",
                    margin=dict(b=20, l=5, r=5, t=40),
                    annotations=[
                        dict(
                            text="Callback Dependency Visualization",
                            showarrow=False,
                            xref="paper",
                            yref="paper",
                            x=0.005,
                            y=-0.002,
                        )
                    ],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                ),
            )

            # Write to HTML file
            fig.write_html(output_file)
            logger.info(f"Dependency graph visualization saved to {output_file}")

        except ImportError:
            logger.error("Visualization requires plotly and networkx")
        except Exception as e:
            logger.error(f"Error generating visualization: {str(e)}")


class CallbackOptimizer:
    """
    Optimizes callback execution through various strategies.

    This class provides utilities for throttling, debouncing,
    and batching callbacks to improve dashboard performance.
    """

    def __init__(self, app: Dash):
        """
        Initialize the callback optimizer.

        Args:
            app: Dash application instance
        """
        self.app = app
        self.dependency_graph = DependencyGraph()
        self.optimized_callbacks = {}
        self.last_execution = {}
        self.pending_executions = defaultdict(list)
        self.throttle_timeouts = {}
        self.debounce_timeouts = {}

    def wrap_callback(
        self,
        callback_id: str,
        callback_func: Callable,
        inputs: List[Union[Input, State]],
        outputs: List[Output],
        throttle_ms: Optional[int] = None,
        debounce_ms: Optional[int] = None,
        batch_updates: bool = False,
        priority: int = 5,
    ) -> Callable:
        """
        Wrap a callback function with optimization strategies.

        Args:
            callback_id: Unique identifier for the callback
            callback_func: Original callback function
            inputs: List of inputs and states
            outputs: List of outputs
            throttle_ms: Throttle timeout in milliseconds
            debounce_ms: Debounce timeout in milliseconds
            batch_updates: Whether to batch updates
            priority: Priority level (1-10)

        Returns:
            Optimized callback function
        """
        # Add to dependency graph
        self.dependency_graph.add_callback(
            callback_id=callback_id,
            inputs=inputs,
            outputs=outputs,
            function=callback_func,
            priority=priority,
        )

        # Store optimization settings
        self.optimized_callbacks[callback_id] = {
            "throttle_ms": throttle_ms,
            "debounce_ms": debounce_ms,
            "batch_updates": batch_updates,
            "priority": priority,
        }

        # Create the optimized wrapper
        @self.app.callback(*outputs, inputs=inputs)
        def optimized_callback(*args, **kwargs):
            start_time = time.time()

            # Apply throttling if configured
            if throttle_ms is not None:
                now = time.time()
                last_time = self.last_execution.get(callback_id, 0)
                elapsed = (now - last_time) * 1000  # Convert to ms

                if elapsed < throttle_ms:
                    # If we're throttling, use the last known result
                    if callback_id in self.pending_executions:
                        logger.debug(f"Throttling callback {callback_id}")
                        return self.pending_executions[callback_id][-1]

            # Apply debouncing if configured
            if debounce_ms is not None:
                # Store args for later execution
                self.pending_executions[callback_id].append((args, kwargs))

                # If we already have a timeout, clear it
                if callback_id in self.debounce_timeouts:
                    self.debounce_timeouts.pop(callback_id)

                # Schedule execution after the debounce period
                def execute_debounced():
                    # Get the most recent args
                    if not self.pending_executions[callback_id]:
                        return None

                    latest_args, latest_kwargs = self.pending_executions[callback_id][
                        -1
                    ]
                    self.pending_executions[callback_id] = []

                    # Execute the callback
                    result = callback_func(*latest_args, **latest_kwargs)
                    self.last_execution[callback_id] = time.time()

                    # Track execution time
                    end_time = time.time()
                    execution_time = end_time - start_time
                    self.dependency_graph.track_execution(callback_id, execution_time)

                    return result

                # In Dash, we can't actually implement true debouncing with timeouts
                # since we're in a request/response cycle, but we can track the concept
                logger.debug(f"Debouncing callback {callback_id}")
                # We'll just use the latest
                return callback_func(*args, **kwargs)

            # Normal execution (with batch handling if enabled)
            if batch_updates:
                # Store the pending execution
                self.pending_executions[callback_id].append((args, kwargs))

                # We would normally use timeouts for batching, but in Dash
                # we'll just simulate it by returning the result directly
                logger.debug(f"Batched callback {callback_id}")

            # Execute the callback
            result = callback_func(*args, **kwargs)
            self.last_execution[callback_id] = time.time()

            # Track execution time
            end_time = time.time()
            execution_time = end_time - start_time
            self.dependency_graph.track_execution(callback_id, execution_time)

            return result

        return optimized_callback

    def get_optimization_stats(self) -> Dict[str, Any]:
        """
        Get statistics about optimized callbacks.

        Returns:
            Dictionary of optimization statistics
        """
        # Get base stats from dependency graph
        stats = self.dependency_graph.get_callback_stats()

        # Add optimization-specific stats
        optimized_count = len(self.optimized_callbacks)
        throttled_count = sum(
            1
            for settings in self.optimized_callbacks.values()
            if settings.get("throttle_ms") is not None
        )
        debounced_count = sum(
            1
            for settings in self.optimized_callbacks.values()
            if settings.get("debounce_ms") is not None
        )
        batched_count = sum(
            1
            for settings in self.optimized_callbacks.values()
            if settings.get("batch_updates", False)
        )

        stats.update(
            {
                "optimized_callbacks": optimized_count,
                "throttled_callbacks": throttled_count,
                "debounced_callbacks": debounced_count,
                "batched_callbacks": batched_count,
            }
        )

        return stats

    def generate_optimization_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive optimization report.

        Returns:
            Dictionary with optimization insights and recommendations
        """
        # Get basic stats
        stats = self.get_optimization_stats()

        # Find potential optimizations
        redundant_callbacks = self.dependency_graph.find_redundant_callbacks()
        expensive_callbacks = self.dependency_graph.find_expensive_callbacks()
        cascading_chains = self.dependency_graph.find_cascading_callbacks()

        # Generate recommendations
        recommendations = []

        if redundant_callbacks:
            recommendations.append(
                {
                    "type": "redundant_callbacks",
                    "description": "Consider combining these callbacks that update the same outputs",
                    "items": redundant_callbacks,
                }
            )

        if expensive_callbacks:
            for cb in expensive_callbacks:
                if cb["avg_time_ms"] > 500:  # Very slow callbacks
                    recommendations.append(
                        {
                            "type": "expensive_callback",
                            "description": f"Optimize this slow callback ({cb['callback_id']}) - {cb['avg_time_ms']:.2f}ms",
                            "callback_id": cb["callback_id"],
                            "suggestion": "Consider using clientside callbacks or memoization",
                        }
                    )
                elif cb["avg_time_ms"] > 100:  # Moderately slow
                    recommendations.append(
                        {
                            "type": "expensive_callback",
                            "description": f"Consider throttling callback {cb['callback_id']} - {cb['avg_time_ms']:.2f}ms",
                            "callback_id": cb["callback_id"],
                            "suggestion": "Add throttling or debouncing to limit execution frequency",
                        }
                    )

        if cascading_chains and len(cascading_chains) > 0:
            longest_chains = sorted(cascading_chains, key=len, reverse=True)[:3]
            for chain in longest_chains:
                if len(chain) > 3:  # Significant chains
                    recommendations.append(
                        {
                            "type": "callback_chain",
                            "description": f"Consider combining this callback chain (length {len(chain)})",
                            "chain": chain,
                            "suggestion": "Merge these callbacks to reduce update cycles",
                        }
                    )

        # Check memory usage
        memory_mb = stats.get("memory_usage_mb", 0)
        if memory_mb > 500:  # High memory usage
            recommendations.append(
                {
                    "type": "memory_usage",
                    "description": f"High memory usage detected ({memory_mb:.1f} MB)",
                    "suggestion": "Review the caching strategy and large data objects",
                }
            )

        # Return the complete report
        return {
            "stats": stats,
            "redundant_callbacks": redundant_callbacks,
            "expensive_callbacks": expensive_callbacks,
            "cascading_chains": cascading_chains,
            "recommendations": recommendations,
        }


def create_callback_optimizer(app: Dash) -> CallbackOptimizer:
    """
    Create and return a callback optimizer instance.

    Args:
        app: Dash application instance

    Returns:
        Callback optimizer instance
    """
    return CallbackOptimizer(app)


def optimize_callback(
    app: Dash,
    callback_id: str,
    outputs: List[Output],
    inputs: List[Union[Input, State]],
    throttle_ms: Optional[int] = None,
    debounce_ms: Optional[int] = None,
    batch_updates: bool = False,
    priority: int = 5,
) -> Callable:
    """
    Decorator for optimizing a callback function.

    Args:
        app: Dash application instance
        callback_id: Unique identifier for the callback
        outputs: List of outputs
        inputs: List of inputs and states
        throttle_ms: Throttle timeout in milliseconds
        debounce_ms: Debounce timeout in milliseconds
        batch_updates: Whether to batch updates
        priority: Priority level (1-10)

    Returns:
        Decorator function
    """
    # Get or create the optimizer
    optimizer = getattr(app, "_callback_optimizer", None)
    if optimizer is None:
        optimizer = create_callback_optimizer(app)
        setattr(app, "_callback_optimizer", optimizer)

    def decorator(func):
        return optimizer.wrap_callback(
            callback_id=callback_id,
            callback_func=func,
            inputs=inputs,
            outputs=outputs,
            throttle_ms=throttle_ms,
            debounce_ms=debounce_ms,
            batch_updates=batch_updates,
            priority=priority,
        )

    return decorator
