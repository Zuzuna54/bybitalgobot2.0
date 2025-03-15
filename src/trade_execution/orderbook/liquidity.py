"""
Order Book Liquidity Analysis

This module provides functions for analyzing liquidity in the order book,
including depth analysis, liquidity metrics, and execution quality assessment.
"""

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from loguru import logger


def analyze_liquidity(orderbook: Dict[str, Any], depth_percentage: float = 0.02) -> Dict[str, float]:
    """
    Analyze liquidity based on order book data.
    
    Args:
        orderbook: Order book data from the exchange
        depth_percentage: Depth to consider (as percentage of mid price)
        
    Returns:
        Dictionary with buy and sell liquidity metrics
    """
    # Extract bids and asks
    bids = orderbook.get("bids", [])
    asks = orderbook.get("asks", [])
    
    if not bids or not asks:
        return {"buy_liquidity": 0.0, "sell_liquidity": 0.0, "liquidity_ratio": 1.0}
    
    # Calculate mid price
    best_bid = float(bids[0][0])
    best_ask = float(asks[0][0])
    mid_price = (best_bid + best_ask) / 2
    
    # Define depth range
    lower_bound = mid_price * (1 - depth_percentage)
    upper_bound = mid_price * (1 + depth_percentage)
    
    # Calculate liquidity in range
    buy_liquidity = sum(
        float(bid[1]) for bid in bids 
        if float(bid[0]) >= lower_bound
    )
    
    sell_liquidity = sum(
        float(ask[1]) for ask in asks 
        if float(ask[0]) <= upper_bound
    )
    
    # Calculate liquidity ratio (buy/sell)
    liquidity_ratio = buy_liquidity / sell_liquidity if sell_liquidity > 0 else 1.0
    
    return {
        "buy_liquidity": buy_liquidity,
        "sell_liquidity": sell_liquidity,
        "liquidity_ratio": liquidity_ratio
    }


def should_split_order(orderbook: Dict[str, Any], order_size: float, is_buy: bool) -> Tuple[bool, int]:
    """
    Determine if an order should be split to reduce market impact.
    
    Args:
        orderbook: Order book data from the exchange
        order_size: Size of the order
        is_buy: Whether it's a buy or sell order
        
    Returns:
        Tuple of (should_split, recommended_parts)
    """
    # Get the relevant side of the book
    liquidity_side = orderbook.get("asks" if is_buy else "bids", [])
    
    if not liquidity_side:
        return False, 1
    
    # Calculate average size per level
    total_levels = len(liquidity_side)
    total_size = sum(float(level[1]) for level in liquidity_side)
    avg_size_per_level = total_size / total_levels if total_levels > 0 else 0
    
    # Determine if splitting is necessary
    if order_size <= avg_size_per_level:
        return False, 1
    
    # Calculate recommended number of parts
    recommended_parts = int(np.ceil(order_size / avg_size_per_level))
    
    # Cap at a reasonable number
    recommended_parts = min(recommended_parts, 10)
    
    return True, recommended_parts


def calculate_execution_quality(orderbook: Dict[str, Any], execution_price: float, 
                               order_size: float, is_buy: bool) -> Dict[str, float]:
    """
    Calculate execution quality metrics compared to the current order book.
    
    Args:
        orderbook: Order book data from the exchange
        execution_price: Actual execution price
        order_size: Size of the executed order
        is_buy: Whether it was a buy or sell order
        
    Returns:
        Dictionary with execution quality metrics
    """
    # Extract bids and asks
    bids = orderbook.get("bids", [])
    asks = orderbook.get("asks", [])
    
    if not bids or not asks:
        return {
            "price_improvement": 0.0,
            "slippage": 0.0,
            "execution_speed": 0.0
        }
    
    # Calculate benchmarks
    best_bid = float(bids[0][0])
    best_ask = float(asks[0][0])
    mid_price = (best_bid + best_ask) / 2
    
    # Calculate theoretical execution price
    theoretical_price = 0.0
    remaining_size = order_size
    value_executed = 0.0
    
    liquidity_side = asks if is_buy else bids
    
    for level in liquidity_side:
        price = float(level[0])
        size = float(level[1])
        
        if remaining_size <= size:
            value_executed += remaining_size * price
            remaining_size = 0
            break
        else:
            value_executed += size * price
            remaining_size -= size
    
    # If we couldn't fill the entire order with available liquidity
    if remaining_size > 0:
        logger.warning(f"Insufficient liquidity to calculate theoretical price for order of size {order_size}")
        # Use the last available price
        value_executed += remaining_size * float(liquidity_side[-1][0])
    
    theoretical_price = value_executed / order_size
    
    # Calculate price improvement (positive is better)
    if is_buy:
        # For buy orders, lower is better
        price_improvement = ((theoretical_price - execution_price) / theoretical_price) * 100
    else:
        # For sell orders, higher is better
        price_improvement = ((execution_price - theoretical_price) / theoretical_price) * 100
    
    # Calculate slippage from mid price (lower is better)
    if is_buy:
        slippage = ((execution_price - mid_price) / mid_price) * 100
    else:
        slippage = ((mid_price - execution_price) / mid_price) * 100
    
    # Since we don't have execution speed data, we'll estimate it based on liquidity
    # Higher liquidity generally means faster execution
    liquidity_metrics = analyze_liquidity(orderbook)
    relevant_liquidity = liquidity_metrics["buy_liquidity"] if is_buy else liquidity_metrics["sell_liquidity"]
    
    # Normalize to a 0-100 scale (higher is better)
    # This is a simplified model and would need to be calibrated in practice
    execution_speed = min(100, relevant_liquidity / order_size * 100) if order_size > 0 else 100
    
    return {
        "price_improvement": price_improvement,
        "slippage": slippage,
        "execution_speed": execution_speed
    } 