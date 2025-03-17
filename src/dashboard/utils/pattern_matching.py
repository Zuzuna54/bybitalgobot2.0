"""
Pattern Matching Callbacks Utility

This module provides utilities to work with pattern-matching callbacks
to optimize multiple similar callbacks into a single callback with shared logic.
"""

from typing import Dict, Any, List, Union, Callable, Optional, Pattern
import re
import dash
from dash import callback, ALL, MATCH, ctx
from dash.dependencies import Input, Output, State
from loguru import logger

# Type definition for callback function
CallbackFunc = Callable[..., Any]


def create_pattern_id(pattern: str, id_value: Any) -> Dict[str, Any]:
    """
    Create a pattern-matching ID dictionary.

    Args:
        pattern: The pattern name
        id_value: The ID value

    Returns:
        Pattern-matching ID dictionary
    """
    return {"type": pattern, "index": id_value}


def register_pattern_callback(
    app: dash.Dash,
    outputs: Union[Output, List[Output]],
    inputs: Union[Input, List[Input]],
    pattern_type: str,
    callback_func: CallbackFunc,
    states: Optional[Union[State, List[State]]] = None,
    prevent_initial_call: bool = False,
) -> None:
    """
    Register a pattern-matching callback.

    Args:
        app: The Dash application instance
        outputs: Output or list of Output objects
        inputs: Input or list of Input objects
        pattern_type: The pattern type to match
        callback_func: The callback function
        states: Optional State or list of State objects
        prevent_initial_call: Whether to prevent initial callback execution
    """
    try:

        @app.callback(
            outputs,
            inputs,
            states if states else [],
            prevent_initial_call=prevent_initial_call,
        )
        def pattern_callback(*args):
            # Extract the triggered input's pattern index
            triggered = ctx.triggered_id
            if (
                triggered
                and isinstance(triggered, dict)
                and triggered.get("type") == pattern_type
            ):
                index = triggered.get("index")
                # Call the callback function with the pattern index and other args
                return callback_func(index, *args)

            # Return appropriate default values if not triggered by pattern
            # (Handle multiple outputs if needed)
            if isinstance(outputs, list):
                return [None] * len(outputs)
            return None

        logger.debug(f"Registered pattern callback for type: {pattern_type}")
    except Exception as e:
        logger.error(f"Error registering pattern callback: {str(e)}")


def create_multi_output_callback(
    app: dash.Dash,
    component_ids: List[str],
    property_name: str,
    input_id: str,
    input_property: str,
    callback_func: CallbackFunc,
    states: Optional[Union[State, List[State]]] = None,
    prevent_initial_call: bool = False,
) -> None:
    """
    Create a callback that updates multiple similar components with a single function.

    Args:
        app: The Dash application instance
        component_ids: List of component IDs to update
        property_name: Property to update for all components
        input_id: ID of the input component
        input_property: Property of the input component to watch
        callback_func: Callback function that returns values for all outputs
        states: Optional State or list of State objects
        prevent_initial_call: Whether to prevent initial callback execution
    """
    outputs = [Output(component_id, property_name) for component_id in component_ids]

    @app.callback(
        outputs,
        Input(input_id, input_property),
        states if states else [],
        prevent_initial_call=prevent_initial_call,
    )
    def update_multiple_components(input_value, *state_values):
        try:
            # Call the callback function which should return values for all outputs
            return callback_func(input_value, *state_values)
        except Exception as e:
            logger.error(f"Error in multi-output callback: {str(e)}")
            # Return None for all outputs in case of error
            return [None] * len(outputs)


def find_matching_components(
    pattern: Union[str, Pattern], component_ids: List[str]
) -> List[str]:
    """
    Find component IDs that match a pattern.

    Args:
        pattern: Regex pattern string or compiled pattern
        component_ids: List of component IDs to search

    Returns:
        List of matching component IDs
    """
    if isinstance(pattern, str):
        pattern = re.compile(pattern)

    return [
        component_id for component_id in component_ids if pattern.match(component_id)
    ]


class DynamicCallbackManager:
    """Manager for dynamically generated pattern-matching callbacks."""

    def __init__(self, app: dash.Dash):
        """
        Initialize the dynamic callback manager.

        Args:
            app: The Dash application instance
        """
        self.app = app
        self.registered_patterns = set()

    def register_dynamic_callback(
        self,
        output_pattern: str,
        output_property: str,
        input_pattern: str,
        input_property: str,
        callback_func: CallbackFunc,
        states: Optional[List[State]] = None,
        prevent_initial_call: bool = False,
    ) -> None:
        """
        Register a dynamic callback using pattern matching.

        Args:
            output_pattern: Pattern type for outputs
            output_property: Property to update for outputs
            input_pattern: Pattern type for inputs
            input_property: Property to watch for inputs
            callback_func: Callback function
            states: Optional State or list of State objects
            prevent_initial_call: Whether to prevent initial callback execution
        """
        pattern_key = f"{output_pattern}:{input_pattern}"

        if pattern_key in self.registered_patterns:
            logger.debug(f"Pattern {pattern_key} already registered")
            return

        try:

            @self.app.callback(
                Output({"type": output_pattern, "index": MATCH}, output_property),
                Input({"type": input_pattern, "index": MATCH}, input_property),
                states if states else [],
                prevent_initial_call=prevent_initial_call,
            )
            def dynamic_callback(input_value, *state_values):
                # Get the pattern index from the triggered context
                pattern_index = (
                    ctx.triggered_id.get("index") if ctx.triggered_id else None
                )

                if pattern_index is not None:
                    try:
                        # Call the callback function with pattern index
                        return callback_func(pattern_index, input_value, *state_values)
                    except Exception as e:
                        logger.error(
                            f"Error in dynamic callback for index {pattern_index}: {str(e)}"
                        )
                        return None
                return None

            self.registered_patterns.add(pattern_key)
            logger.debug(f"Registered dynamic callback for pattern: {pattern_key}")
        except Exception as e:
            logger.error(f"Error registering dynamic callback: {str(e)}")

    def register_multi_pattern_callback(
        self,
        output_patterns: List[Dict[str, Any]],
        output_property: str,
        input_patterns: List[Dict[str, Any]],
        input_property: str,
        callback_func: CallbackFunc,
        states: Optional[List[State]] = None,
        prevent_initial_call: bool = False,
    ) -> None:
        """
        Register a callback that works with multiple pattern types simultaneously.

        Args:
            output_patterns: List of output patterns
            output_property: Property to update for outputs
            input_patterns: List of input patterns
            input_property: Property to watch for inputs
            callback_func: Callback function
            states: Optional State or list of State objects
            prevent_initial_call: Whether to prevent initial callback execution
        """
        pattern_key = str(output_patterns) + str(input_patterns)

        if pattern_key in self.registered_patterns:
            logger.debug(f"Multi-pattern already registered")
            return

        try:
            outputs = [Output(pattern, output_property) for pattern in output_patterns]
            inputs = [Input(pattern, input_property) for pattern in input_patterns]

            @self.app.callback(
                outputs,
                inputs,
                states if states else [],
                prevent_initial_call=prevent_initial_call,
            )
            def multi_pattern_callback(*args):
                input_values = args[: len(inputs)]
                state_values = args[len(inputs) :]

                try:
                    # Call the callback function with all input values
                    return callback_func(input_values, state_values)
                except Exception as e:
                    logger.error(f"Error in multi-pattern callback: {str(e)}")
                    return [None] * len(outputs)

            self.registered_patterns.add(pattern_key)
            logger.debug(f"Registered multi-pattern callback")
        except Exception as e:
            logger.error(f"Error registering multi-pattern callback: {str(e)}")
