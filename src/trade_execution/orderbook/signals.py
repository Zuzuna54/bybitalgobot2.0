"""
Order Book Signal Generation

This module provides functions for generating trading signals based on order book analysis,
including imbalance detection, entry and exit signals.
"""

from typing import Dict, Any, Tuple
import numpy as np
from loguru import logger


def detect_order_book_imbalance(orderbook: Dict[str, Any], depth_levels: int = 10) -> float:
    """
    Detect order book imbalance to predict short-term price movement.
    
    Args:
        orderbook: Order book data from the exchange
        depth_levels: Number of price levels to consider
        
    Returns:
        Imbalance score (-1.0 to 1.0, positive means bullish)
    """
    # Extract bids and asks
    bids = orderbook.get("bids", [])
    asks = orderbook.get("asks", [])
    
    if not bids or not asks:
        return 0.0
    
    # Limit to specified depth
    bids = bids[:depth_levels] if depth_levels < len(bids) else bids
    asks = asks[:depth_levels] if depth_levels < len(asks) else asks
    
    # Calculate total volume on each side
    bid_volume = sum(float(bid[1]) for bid in bids)
    ask_volume = sum(float(ask[1]) for ask in asks)
    
    # Calculate imbalance score
    total_volume = bid_volume + ask_volume
    
    if total_volume == 0:
        return 0.0
    
    # Imbalance ranges from -1.0 (all asks) to 1.0 (all bids)
    imbalance = (bid_volume - ask_volume) / total_volume
    
    return imbalance


def detect_price_walls(orderbook: Dict[str, Any], threshold_multiplier: float = 3.0) -> Dict[str, Any]:
    """
    Detect significant price walls in the order book.
    
    Args:
        orderbook: Order book data from the exchange
        threshold_multiplier: Multiplier for average size to detect walls
        
    Returns:
        Dictionary with support and resistance walls
    """
    # Extract bids and asks
    bids = orderbook.get("bids", [])
    asks = orderbook.get("asks", [])
    
    if not bids or not asks:
        return {"support_walls": [], "resistance_walls": []}
    
    # Calculate average size for bids and asks
    bid_sizes = [float(bid[1]) for bid in bids]
    ask_sizes = [float(ask[1]) for ask in asks]
    
    avg_bid_size = np.mean(bid_sizes) if bid_sizes else 0
    avg_ask_size = np.mean(ask_sizes) if ask_sizes else 0
    
    # Find support walls (unusually large bids)
    support_walls = [
        {"price": float(bid[0]), "size": float(bid[1])}
        for bid in bids
        if float(bid[1]) > avg_bid_size * threshold_multiplier
    ]
    
    # Find resistance walls (unusually large asks)
    resistance_walls = [
        {"price": float(ask[0]), "size": float(ask[1])}
        for ask in asks
        if float(ask[1]) > avg_ask_size * threshold_multiplier
    ]
    
    return {
        "support_walls": support_walls,
        "resistance_walls": resistance_walls
    }


def analyze_spread_changes(current_orderbook: Dict[str, Any], 
                          previous_orderbook: Dict[str, Any]) -> Dict[str, float]:
    """
    Analyze changes in the bid-ask spread over time.
    
    Args:
        current_orderbook: Current order book data
        previous_orderbook: Previous order book data
        
    Returns:
        Dictionary with spread analysis metrics
    """
    # Extract current bids and asks
    current_bids = current_orderbook.get("bids", [])
    current_asks = current_orderbook.get("asks", [])
    
    # Extract previous bids and asks
    previous_bids = previous_orderbook.get("bids", [])
    previous_asks = previous_orderbook.get("asks", [])
    
    if not current_bids or not current_asks or not previous_bids or not previous_asks:
        return {
            "spread_change_pct": 0.0,
            "bid_change_pct": 0.0,
            "ask_change_pct": 0.0
        }
    
    # Calculate current spread
    current_best_bid = float(current_bids[0][0])
    current_best_ask = float(current_asks[0][0])
    current_spread = current_best_ask - current_best_bid
    current_spread_pct = current_spread / current_best_bid * 100
    
    # Calculate previous spread
    previous_best_bid = float(previous_bids[0][0])
    previous_best_ask = float(previous_asks[0][0])
    previous_spread = previous_best_ask - previous_best_bid
    previous_spread_pct = previous_spread / previous_best_bid * 100
    
    # Calculate changes
    spread_change_pct = current_spread_pct - previous_spread_pct
    bid_change_pct = (current_best_bid - previous_best_bid) / previous_best_bid * 100
    ask_change_pct = (current_best_ask - previous_best_ask) / previous_best_ask * 100
    
    return {
        "spread_change_pct": spread_change_pct,
        "bid_change_pct": bid_change_pct,
        "ask_change_pct": ask_change_pct
    }


def generate_entry_signal(orderbook: Dict[str, Any], 
                         market_trend: str = "neutral") -> Dict[str, Any]:
    """
    Generate an entry signal based on order book analysis.
    
    Args:
        orderbook: Order book data from the exchange
        market_trend: Current market trend ("bullish", "bearish", or "neutral")
        
    Returns:
        Dictionary with entry signal details
    """
    # Get order book imbalance
    imbalance = detect_order_book_imbalance(orderbook)
    
    # Detect price walls
    walls = detect_price_walls(orderbook)
    
    # Define threshold for strong signal
    imbalance_threshold = 0.3  # 30% imbalance for strong signal
    
    # Determine signal direction
    if imbalance > imbalance_threshold:
        direction = "buy"
        strength = abs(imbalance)
    elif imbalance < -imbalance_threshold:
        direction = "sell"
        strength = abs(imbalance)
    else:
        # No strong signal from order book alone
        # Use market trend as a reference
        if market_trend == "bullish":
            direction = "buy"
            strength = 0.5  # Moderate strength
        elif market_trend == "bearish":
            direction = "sell"
            strength = 0.5  # Moderate strength
        else:
            direction = "neutral"
            strength = 0.0
    
    # Analyze price walls for potential targets
    if direction == "buy" and walls["resistance_walls"]:
        # First resistance wall is the target for a buy
        target_price = min(wall["price"] for wall in walls["resistance_walls"])
    elif direction == "sell" and walls["support_walls"]:
        # First support wall is the target for a sell
        target_price = max(wall["price"] for wall in walls["support_walls"])
    else:
        # No clear walls, set target as None
        target_price = None
    
    # Extract best bid and ask
    bids = orderbook.get("bids", [])
    asks = orderbook.get("asks", [])
    
    if bids and asks:
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        mid_price = (best_bid + best_ask) / 2
    else:
        mid_price = None
    
    return {
        "signal": direction,
        "strength": strength,
        "target_price": target_price,
        "current_mid_price": mid_price,
        "imbalance_score": imbalance
    }


def generate_exit_signal(orderbook: Dict[str, Any], 
                        position_side: str,
                        entry_price: float) -> Dict[str, Any]:
    """
    Generate an exit signal based on order book analysis.
    
    Args:
        orderbook: Order book data from the exchange
        position_side: Current position side ("buy" or "sell")
        entry_price: Entry price of the position
        
    Returns:
        Dictionary with exit signal details
    """
    # Get order book imbalance
    imbalance = detect_order_book_imbalance(orderbook)
    
    # Extract bids and asks
    bids = orderbook.get("bids", [])
    asks = orderbook.get("asks", [])
    
    if not bids or not asks:
        return {
            "should_exit": False,
            "reason": "insufficient_data",
            "urgency": 0.0
        }
    
    best_bid = float(bids[0][0])
    best_ask = float(asks[0][0])
    mid_price = (best_bid + best_ask) / 2
    
    # Calculate profit/loss
    if position_side == "buy":
        pnl_pct = (mid_price - entry_price) / entry_price * 100
        # For long positions, opposing imbalance is bearish
        opposing_imbalance = imbalance < -0.3
    else:  # position_side == "sell"
        pnl_pct = (entry_price - mid_price) / entry_price * 100
        # For short positions, opposing imbalance is bullish
        opposing_imbalance = imbalance > 0.3
    
    # Default values
    should_exit = False
    reason = "hold_position"
    urgency = 0.0
    
    # Check for exit signals
    if pnl_pct >= 2.0:
        # Take profit at 2% or more
        should_exit = True
        reason = "take_profit"
        urgency = min(1.0, pnl_pct / 5.0)  # Higher profit, higher urgency
    elif pnl_pct <= -1.5:
        # Cut losses at -1.5% or worse
        should_exit = True
        reason = "stop_loss"
        urgency = min(1.0, abs(pnl_pct) / 3.0)  # Bigger loss, higher urgency
    elif opposing_imbalance:
        # Opposing order book imbalance
        should_exit = True
        reason = "order_book_reversal"
        urgency = 0.7  # Moderately urgent
    
    return {
        "should_exit": should_exit,
        "reason": reason,
        "urgency": urgency,
        "current_pnl_pct": pnl_pct,
        "imbalance_score": imbalance
    } 