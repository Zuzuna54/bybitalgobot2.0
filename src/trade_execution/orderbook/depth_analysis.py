"""
Order Book Depth Analysis

This module provides functions for analyzing order book depth, including
identification of significant price levels and calculation of market impact.
"""

from typing import Dict, Any, List, Tuple
import pandas as pd
from loguru import logger


def get_significant_levels(orderbook: Dict[str, Any], num_levels: int = 5) -> Dict[str, List[float]]:
    """
    Identify significant price levels in the order book.
    
    Args:
        orderbook: Order book data from the exchange
        num_levels: Number of significant levels to return
        
    Returns:
        Dictionary with support and resistance levels
    """
    # Extract bids and asks
    bids = orderbook.get("bids", [])
    asks = orderbook.get("asks", [])
    
    if not bids or not asks:
        return {"support_levels": [], "resistance_levels": []}
    
    # Convert to DataFrame
    bids_df = pd.DataFrame(bids, columns=["price", "size"])
    asks_df = pd.DataFrame(asks, columns=["price", "size"])
    
    bids_df["price"] = bids_df["price"].astype(float)
    bids_df["size"] = bids_df["size"].astype(float)
    asks_df["price"] = asks_df["price"].astype(float)
    asks_df["size"] = asks_df["size"].astype(float)
    
    # Identify support levels (large bid clusters)
    # Group similar prices and find cumulative sizes
    bids_df["price_bin"] = (bids_df["price"] / (bids_df["price"].std() * 0.1)).astype(int)
    bid_clusters = bids_df.groupby("price_bin").agg({
        "price": "mean",
        "size": "sum"
    }).sort_values("size", ascending=False)
    
    # Identify resistance levels (large ask clusters)
    asks_df["price_bin"] = (asks_df["price"] / (asks_df["price"].std() * 0.1)).astype(int)
    ask_clusters = asks_df.groupby("price_bin").agg({
        "price": "mean",
        "size": "sum"
    }).sort_values("size", ascending=False)
    
    # Get top support and resistance levels
    support_levels = bid_clusters.head(num_levels)["price"].tolist()
    resistance_levels = ask_clusters.head(num_levels)["price"].tolist()
    
    return {
        "support_levels": support_levels,
        "resistance_levels": resistance_levels
    }


def calculate_market_impact(orderbook: Dict[str, Any], order_size: float, is_buy: bool) -> float:
    """
    Calculate the estimated market impact of an order.
    
    Args:
        orderbook: Order book data from the exchange
        order_size: Order size in base currency
        is_buy: Whether it's a buy or sell order
        
    Returns:
        Estimated price impact percentage
    """
    # Extract bids and asks
    liquidity_side = orderbook.get("asks" if is_buy else "bids", [])
    
    if not liquidity_side:
        return 0.0
    
    # Initial price
    initial_price = float(liquidity_side[0][0])
    
    # Simulate market order execution
    remaining_size = order_size
    executed_value = 0
    
    for level in liquidity_side:
        price = float(level[0])
        size = float(level[1])
        
        if remaining_size <= size:
            executed_value += remaining_size * price
            remaining_size = 0
            break
        else:
            executed_value += size * price
            remaining_size -= size
    
    # If we couldn't fill the entire order
    if remaining_size > 0:
        logger.warning(f"Insufficient liquidity to fill order of size {order_size}")
        # Estimate impact using the last available price
        executed_value += remaining_size * float(liquidity_side[-1][0])
    
    # Calculate average execution price
    avg_price = executed_value / order_size
    
    # Calculate price impact
    if is_buy:
        impact_percent = (avg_price - initial_price) / initial_price * 100
    else:
        impact_percent = (initial_price - avg_price) / initial_price * 100
    
    return impact_percent


def get_optimal_trade_size(orderbook: Dict[str, Any], max_impact_pct: float = 0.5, is_buy: bool = True) -> float:
    """
    Calculate the optimal trade size to minimize market impact.
    
    Args:
        orderbook: Order book data from the exchange
        max_impact_pct: Maximum acceptable price impact percentage
        is_buy: Whether it's a buy or sell order
        
    Returns:
        Optimal order size
    """
    # Extract bids and asks
    liquidity_side = orderbook.get("asks" if is_buy else "bids", [])
    
    if not liquidity_side:
        return 0.0
    
    # Binary search to find optimal size
    min_size = 0.0
    max_size = sum(float(level[1]) for level in liquidity_side)
    
    optimal_size = 0.0
    
    # 10 iterations should be enough for a good approximation
    for _ in range(10):
        test_size = (min_size + max_size) / 2
        impact = calculate_market_impact(orderbook, test_size, is_buy)
        
        if impact <= max_impact_pct:
            # We can handle larger size
            optimal_size = test_size
            min_size = test_size
        else:
            # Too much impact, reduce size
            max_size = test_size
    
    return optimal_size


def get_optimal_limit_price(orderbook: Dict[str, Any], is_buy: bool, urgency: float = 0.5) -> float:
    """
    Get the optimal limit price based on order book and urgency.
    
    Args:
        orderbook: Order book data from the exchange
        is_buy: Whether it's a buy or sell order
        urgency: Urgency factor (0.0 to 1.0, higher means more aggressive)
        
    Returns:
        Optimal limit price
    """
    # Extract bids and asks
    bids = orderbook.get("bids", [])
    asks = orderbook.get("asks", [])
    
    if not bids or not asks:
        return 0.0
    
    # Get top of book prices
    best_bid = float(bids[0][0])
    best_ask = float(asks[0][0])
    
    # Calculate spread
    spread = best_ask - best_bid
    
    if is_buy:
        # Buy order, start from best ask and move towards best bid
        optimal_price = best_ask - (spread * (1 - urgency))
        # Ensure we don't go below best bid
        optimal_price = max(optimal_price, best_bid)
    else:
        # Sell order, start from best bid and move towards best ask
        optimal_price = best_bid + (spread * (1 - urgency))
        # Ensure we don't go above best ask
        optimal_price = min(optimal_price, best_ask)
    
    return optimal_price 