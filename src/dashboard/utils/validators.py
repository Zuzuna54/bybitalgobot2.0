"""
Dashboard Validators Module

This module provides utilities for validating user inputs in the dashboard.
"""

from typing import Dict, Any, Union, Optional, List, Tuple
import re


def validate_symbol(symbol: str) -> Tuple[bool, str]:
    """
    Validate that a trading symbol follows the expected format.
    
    Args:
        symbol: The trading symbol to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not symbol:
        return False, "Symbol cannot be empty"
    
    # Check for common valid formats (e.g., BTCUSDT, BTC-USDT, BTC/USDT)
    valid_patterns = [
        r'^[A-Z0-9]{2,10}(USDT|USD|BTC|ETH|BNB)$',  # BTCUSDT format
        r'^[A-Z0-9]{2,10}-[A-Z0-9]{2,5}$',  # BTC-USDT format
        r'^[A-Z0-9]{2,10}/[A-Z0-9]{2,5}$'   # BTC/USDT format
    ]
    
    for pattern in valid_patterns:
        if re.match(pattern, symbol.upper()):
            return True, ""
    
    return False, "Invalid symbol format"


def validate_number(value: Union[str, int, float], 
                    min_value: Optional[float] = None,
                    max_value: Optional[float] = None) -> Tuple[bool, str]:
    """
    Validate that a value is a number within the specified range.
    
    Args:
        value: The value to validate
        min_value: Minimum allowed value (optional)
        max_value: Maximum allowed value (optional)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Convert to float if it's a string
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return False, "Value must be a number"
    
    # Check range if specified
    if min_value is not None and value < min_value:
        return False, f"Value must be at least {min_value}"
    
    if max_value is not None and value > max_value:
        return False, f"Value must be at most {max_value}"
    
    return True, ""


def validate_percentage(value: Union[str, int, float]) -> Tuple[bool, str]:
    """
    Validate that a value is a valid percentage (0-100).
    
    Args:
        value: The value to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    return validate_number(value, min_value=0, max_value=100)


def validate_timeframe(timeframe: str) -> Tuple[bool, str]:
    """
    Validate that a timeframe is in a valid format.
    
    Args:
        timeframe: The timeframe to validate (e.g., 1m, 5m, 1h, 1d)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    valid_timeframes = [
        '1m', '3m', '5m', '15m', '30m',  # Minutes
        '1h', '2h', '4h', '6h', '8h', '12h',  # Hours
        '1d', '3d', '1w', '1M'  # Days, Weeks, Months
    ]
    
    if timeframe not in valid_timeframes:
        return False, f"Invalid timeframe. Must be one of: {', '.join(valid_timeframes)}"
    
    return True, ""


def validate_date(date_str: str, format_str: str = '%Y-%m-%d') -> Tuple[bool, str]:
    """
    Validate that a string is a valid date in the specified format.
    
    Args:
        date_str: The date string to validate
        format_str: The expected date format
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    import datetime
    
    try:
        datetime.datetime.strptime(date_str, format_str)
        return True, ""
    except ValueError:
        return False, f"Invalid date format. Expected format: {format_str}"


def validate_api_key(api_key: str) -> Tuple[bool, str]:
    """
    Validate that a string looks like a valid API key.
    
    Args:
        api_key: The API key to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not api_key:
        return False, "API key cannot be empty"
    
    # Check for common API key patterns
    # Most keys are alphanumeric and have certain lengths
    if len(api_key) < 10:
        return False, "API key is too short"
    
    if not re.match(r'^[A-Za-z0-9\-_]+$', api_key):
        return False, "API key contains invalid characters"
    
    return True, ""


def validate_leverage(leverage: Union[str, int, float]) -> Tuple[bool, str]:
    """
    Validate that a leverage value is within acceptable range.
    
    Args:
        leverage: The leverage value to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    return validate_number(leverage, min_value=1, max_value=100)


def validate_risk_params(risk_params: Dict[str, Any]) -> Dict[str, str]:
    """
    Validate a set of risk parameters and return any validation errors.
    
    Args:
        risk_params: Dictionary of risk parameters to validate
        
    Returns:
        Dictionary of field_name -> error_message for any invalid fields
    """
    errors = {}
    
    # Validate position size
    if "max_position_size_percent" in risk_params:
        valid, message = validate_percentage(risk_params["max_position_size_percent"])
        if not valid:
            errors["max_position_size_percent"] = message
    
    # Validate max drawdown
    if "max_daily_drawdown_percent" in risk_params:
        valid, message = validate_percentage(risk_params["max_daily_drawdown_percent"])
        if not valid:
            errors["max_daily_drawdown_percent"] = message
    
    # Validate leverage
    if "default_leverage" in risk_params:
        valid, message = validate_leverage(risk_params["default_leverage"])
        if not valid:
            errors["default_leverage"] = message
    
    # Validate max positions
    if "max_open_positions" in risk_params:
        valid, message = validate_number(risk_params["max_open_positions"], min_value=1, max_value=100)
        if not valid:
            errors["max_open_positions"] = message
    
    # Validate stop loss multiplier
    if "stop_loss_atr_multiplier" in risk_params:
        valid, message = validate_number(risk_params["stop_loss_atr_multiplier"], min_value=0.1, max_value=10)
        if not valid:
            errors["stop_loss_atr_multiplier"] = message
    
    # Validate risk reward ratio
    if "take_profit_risk_reward_ratio" in risk_params:
        valid, message = validate_number(risk_params["take_profit_risk_reward_ratio"], min_value=0.1, max_value=10)
        if not valid:
            errors["take_profit_risk_reward_ratio"] = message
    
    # Validate circuit breaker
    if "circuit_breaker_consecutive_losses" in risk_params:
        valid, message = validate_number(risk_params["circuit_breaker_consecutive_losses"], min_value=1, max_value=20)
        if not valid:
            errors["circuit_breaker_consecutive_losses"] = message
    
    return errors


def validate_order_params(order_params: Dict[str, Any]) -> Dict[str, str]:
    """
    Validate order parameters and return any validation errors.
    
    Args:
        order_params: Dictionary of order parameters to validate
        
    Returns:
        Dictionary of field_name -> error_message for any invalid fields
    """
    errors = {}
    
    # Validate symbol
    if "symbol" in order_params:
        valid, message = validate_symbol(order_params["symbol"])
        if not valid:
            errors["symbol"] = message
    else:
        errors["symbol"] = "Symbol is required"
    
    # Validate side
    if "side" in order_params:
        if order_params["side"] not in ["buy", "sell"]:
            errors["side"] = "Side must be 'buy' or 'sell'"
    else:
        errors["side"] = "Side is required"
    
    # Validate order type
    if "order_type" in order_params:
        valid_types = ["market", "limit", "stop", "take_profit"]
        if order_params["order_type"] not in valid_types:
            errors["order_type"] = f"Order type must be one of: {', '.join(valid_types)}"
    else:
        errors["order_type"] = "Order type is required"
    
    # Validate quantity
    if "quantity" in order_params:
        valid, message = validate_number(order_params["quantity"], min_value=0)
        if not valid:
            errors["quantity"] = message
    else:
        errors["quantity"] = "Quantity is required"
    
    # Validate price for non-market orders
    if order_params.get("order_type") != "market" and "price" in order_params:
        valid, message = validate_number(order_params["price"], min_value=0)
        if not valid:
            errors["price"] = message
    elif order_params.get("order_type") != "market" and "price" not in order_params:
        errors["price"] = "Price is required for non-market orders"
    
    return errors 