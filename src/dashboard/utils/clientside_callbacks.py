"""
Clientside Callbacks Utility

This module provides utilities for registering clientside callbacks
to optimize dashboard performance for UI-only interactions.
"""

from typing import Dict, Any, List, Union, Optional
import json
import dash
from dash import clientside_callback
from dash.dependencies import Input, Output, State, ClientsideFunction
from loguru import logger


def register_clientside_callback(
    app: dash.Dash,
    outputs: Union[Output, List[Output]],
    inputs: Union[Input, List[Input]],
    clientside_function: str,
    states: Optional[Union[State, List[State]]] = None,
    prevent_initial_call: bool = False,
) -> None:
    """
    Register a clientside callback with error handling.

    Args:
        app: The Dash application instance
        outputs: Output or list of Output objects
        inputs: Input or list of Input objects
        clientside_function: JavaScript function as a string
        states: Optional State or list of State objects
        prevent_initial_call: Whether to prevent initial callback execution
    """
    try:
        clientside_callback(
            clientside_function,
            outputs,
            inputs,
            states if states else [],
            prevent_initial_call=prevent_initial_call,
        )
        logger.debug(f"Registered clientside callback for {outputs}")
    except Exception as e:
        logger.error(f"Error registering clientside callback: {str(e)}")


# Common clientside callbacks for frequently used patterns


def register_visibility_toggle(
    app: dash.Dash,
    container_id: str,
    trigger_id: str,
    trigger_property: str = "n_clicks",
    initial_state: bool = False,
) -> None:
    """
    Register a clientside callback to toggle the visibility of a container.

    Args:
        app: The Dash application instance
        container_id: ID of the container to toggle
        trigger_id: ID of the element triggering the toggle
        trigger_property: Property of the trigger element to watch
        initial_state: Initial visibility state
    """
    clientside_function = (
        """
    function(n_clicks) {
        if (n_clicks === undefined || n_clicks === null) {
            return """
        + str(initial_state).lower()
        + """;
        }
        var elem = document.getElementById('"""
        + container_id
        + """');
        if (elem) {
            return !elem.style.display || elem.style.display === 'none';
        }
        return """
        + str(initial_state).lower()
        + """;
    }
    """
    )

    register_clientside_callback(
        app,
        Output(container_id, "style"),
        Input(trigger_id, trigger_property),
        clientside_function,
    )


def register_tab_content_visibility(
    app: dash.Dash,
    tab_content_prefix: str,
    tabs_id: str,
    tab_count: int,
) -> None:
    """
    Register clientside callbacks to show/hide tab content based on active tab.

    Args:
        app: The Dash application instance
        tab_content_prefix: Prefix for tab content IDs (e.g., "tab-content-")
        tabs_id: ID of the tabs component
        tab_count: Number of tabs
    """
    outputs = [Output(f"{tab_content_prefix}{i}", "style") for i in range(tab_count)]

    clientside_function = (
        """
    function(active_tab) {
        var styles = [];
        for (var i = 0; i < """
        + str(tab_count)
        + """; i++) {
            if (active_tab === 'tab-' + i.toString()) {
                styles.push({display: 'block'});
            } else {
                styles.push({display: 'none'});
            }
        }
        return styles;
    }
    """
    )

    register_clientside_callback(
        app, outputs, Input(tabs_id, "active_tab"), clientside_function
    )


def register_dropdown_options_update(
    app: dash.Dash,
    dropdown_id: str,
    data_store_id: str,
    data_property: str = "data",
    value_field: str = "value",
    label_field: str = "label",
) -> None:
    """
    Register a clientside callback to update dropdown options from a data store.

    Args:
        app: The Dash application instance
        dropdown_id: ID of the dropdown to update
        data_store_id: ID of the data store containing the options
        data_property: Property of the data store containing the data
        value_field: Field in the data to use for option values
        label_field: Field in the data to use for option labels
    """
    clientside_function = (
        """
    function(data) {
        if (!data) {
            return [];
        }
        
        try {
            return data.map(function(item) {
                return {
                    value: item."""
        + value_field
        + """,
                    label: item."""
        + label_field
        + """
                };
            });
        } catch (e) {
            console.error("Error updating dropdown options:", e);
            return [];
        }
    }
    """
    )

    register_clientside_callback(
        app,
        Output(dropdown_id, "options"),
        Input(data_store_id, data_property),
        clientside_function,
    )


def register_status_indicator_update(
    app: dash.Dash,
    indicator_id: str,
    status_store_id: str,
    status_property: str = "data",
    status_field: str = "status",
    class_mapping: Dict[str, str] = None,
) -> None:
    """
    Register a clientside callback to update a status indicator.

    Args:
        app: The Dash application instance
        indicator_id: ID of the indicator element
        status_store_id: ID of the data store containing the status
        status_property: Property of the data store containing the data
        status_field: Field in the data indicating the status
        class_mapping: Mapping of status values to CSS classes
    """
    if class_mapping is None:
        class_mapping = {
            "success": "success-indicator",
            "warning": "warning-indicator",
            "error": "error-indicator",
            "info": "info-indicator",
        }

    # Convert mapping to JSON string for JavaScript
    class_mapping_json = json.dumps(class_mapping)

    clientside_function = (
        """
    function(status_data) {
        if (!status_data) {
            return "unknown-status";
        }
        
        try {
            var status = status_data."""
        + status_field
        + """;
            var mapping = """
        + class_mapping_json
        + """;
            
            return mapping[status] || "unknown-status";
        } catch (e) {
            console.error("Error updating status indicator:", e);
            return "unknown-status";
        }
    }
    """
    )

    register_clientside_callback(
        app,
        Output(indicator_id, "className"),
        Input(status_store_id, status_property),
        clientside_function,
    )
