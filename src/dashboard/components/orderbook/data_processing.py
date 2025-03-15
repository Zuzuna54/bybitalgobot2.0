"""
Orderbook Data Processing for the Trading Dashboard

This module provides functions for processing and transforming orderbook data
for visualization and analysis.
"""

from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def calculate_orderbook_imbalance(orderbook: Dict[str, Any]) -> float:
    """
    Calculate the order book imbalance score.
    
    Args:
        orderbook: Order book data dictionary with bids and asks
        
    Returns:
        Imbalance score (-1.0 to 1.0)
    """
    if not orderbook or 'bids' not in orderbook or 'asks' not in orderbook:
        return 0.0
    
    # Extract bids and asks
    bids = orderbook.get('bids', [])
    asks = orderbook.get('asks', [])
    
    if not bids or not asks:
        return 0.0
    
    # Calculate total volume at top price levels (e.g., top 10)
    bid_volume = sum(float(bid[1]) for bid in bids[:10])
    ask_volume = sum(float(ask[1]) for ask in asks[:10])
    
    # Calculate imbalance
    total_volume = bid_volume + ask_volume
    if total_volume == 0:
        return 0.0
    
    # Normalize to -1.0 to 1.0 range
    # Positive values indicate buy pressure, negative values indicate sell pressure
    imbalance = (bid_volume - ask_volume) / total_volume
    
    return imbalance


def calculate_liquidity_ratio(orderbook: Dict[str, Any], price_range_pct: float = 0.02) -> Dict[str, float]:
    """
    Calculate the liquidity ratio and related metrics.
    
    Args:
        orderbook: Order book data dictionary with bids and asks
        price_range_pct: Percentage range around mid price to consider (default 2%)
        
    Returns:
        Dictionary with liquidity metrics
    """
    if not orderbook or 'bids' not in orderbook or 'asks' not in orderbook:
        return {
            'buy_sell_ratio': 1.0,
            'buy_liquidity': 0.0,
            'sell_liquidity': 0.0
        }
    
    # Extract bids and asks
    bids = orderbook.get('bids', [])
    asks = orderbook.get('asks', [])
    
    if not bids or not asks:
        return {
            'buy_sell_ratio': 1.0,
            'buy_liquidity': 0.0,
            'sell_liquidity': 0.0
        }
    
    # Calculate mid price
    best_bid = float(bids[0][0])
    best_ask = float(asks[0][0])
    mid_price = (best_bid + best_ask) / 2
    
    # Define price range
    lower_price = mid_price * (1 - price_range_pct)
    upper_price = mid_price * (1 + price_range_pct)
    
    # Calculate buy and sell liquidity within range
    buy_liquidity = sum(float(bid[1]) * float(bid[0]) for bid in bids if float(bid[0]) >= lower_price)
    sell_liquidity = sum(float(ask[1]) * float(ask[0]) for ask in asks if float(ask[0]) <= upper_price)
    
    # Calculate ratio
    if sell_liquidity == 0:
        buy_sell_ratio = 2.0  # Cap at 2.0 when no sell liquidity
    else:
        buy_sell_ratio = min(buy_liquidity / sell_liquidity, 2.0)  # Cap at 2.0
    
    return {
        'buy_sell_ratio': buy_sell_ratio,
        'buy_liquidity': buy_liquidity,
        'sell_liquidity': sell_liquidity
    }


def identify_support_resistance_levels(
    orderbook: Dict[str, Any], 
    historical_prices: Optional[pd.DataFrame] = None,
    vwap_periods: List[int] = [20, 50, 200],
    depth: int = 10,
    volume_cluster_threshold: float = 2.0,
    price_action_weight: float = 0.4
) -> Dict[str, Any]:
    """
    Identify support and resistance levels from orderbook data using volume clustering
    and historical price action.
    
    Args:
        orderbook: Order book data dictionary with bids and asks
        historical_prices: DataFrame of historical prices (optional)
        vwap_periods: List of periods for VWAP calculations
        depth: Number of price levels to consider
        volume_cluster_threshold: Threshold multiplier for identifying volume clusters
        price_action_weight: Weight given to historical price action (0.0-1.0)
        
    Returns:
        Dictionary with support and resistance levels and additional information
    """
    if not orderbook or 'bids' not in orderbook or 'asks' not in orderbook:
        return {
            'support_levels': [],
            'resistance_levels': [],
            'vwap_levels': {},
            'volume_clusters': {
                'support': [],
                'resistance': []
            }
        }
    
    # Extract bids and asks
    bids = orderbook.get('bids', [])
    asks = orderbook.get('asks', [])
    
    # Convert to DataFrames
    bids_df = pd.DataFrame(bids[:depth], columns=['price', 'size'])
    asks_df = pd.DataFrame(asks[:depth], columns=['price', 'size'])
    
    # Convert to numeric
    bids_df['price'] = pd.to_numeric(bids_df['price'])
    bids_df['size'] = pd.to_numeric(bids_df['size'])
    asks_df['price'] = pd.to_numeric(asks_df['price'])
    asks_df['size'] = pd.to_numeric(asks_df['size'])
    
    # Calculate volume clusters using binning technique
    # Group prices into bins and find where volume accumulates
    price_std = bids_df['price'].std() * 0.02  # Dynamic bin size based on price volatility
    
    # Create price bins for bids
    bids_df['price_bin'] = (bids_df['price'] / price_std).astype(int)
    bid_clusters = bids_df.groupby('price_bin').agg({
        'price': 'mean',
        'size': 'sum'
    }).reset_index()
    
    # Calculate average volume
    avg_bid_volume = bid_clusters['size'].mean()
    
    # Find significant volume clusters
    significant_bids = bid_clusters[bid_clusters['size'] > avg_bid_volume * volume_cluster_threshold]
    
    # Same for asks
    asks_df['price_bin'] = (asks_df['price'] / price_std).astype(int)
    ask_clusters = asks_df.groupby('price_bin').agg({
        'price': 'mean',
        'size': 'sum'
    }).reset_index()
    
    avg_ask_volume = ask_clusters['size'].mean()
    significant_asks = ask_clusters[ask_clusters['size'] > avg_ask_volume * volume_cluster_threshold]
    
    # Extract support and resistance from volume clusters
    volume_support = significant_bids.sort_values('size', ascending=False)['price'].tolist()
    volume_resistance = significant_asks.sort_values('size', ascending=False)['price'].tolist()
    
    # Calculate VWAP levels if historical data is available
    vwap_levels = {}
    if historical_prices is not None and not historical_prices.empty:
        for period in vwap_periods:
            if len(historical_prices) >= period:
                # Use only the most recent periods
                recent_data = historical_prices.tail(period).copy()
                
                # Calculate VWAP
                recent_data['vwap'] = (recent_data['price'] * recent_data['volume']).cumsum() / recent_data['volume'].cumsum()
    
                # Get the latest VWAP
                vwap_levels[f'vwap_{period}'] = recent_data['vwap'].iloc[-1]
    
    # Initialize levels from volume clusters
    support_levels = volume_support
    resistance_levels = volume_resistance
    
    # Incorporate price action history if available
    price_action_levels = {
        'support': [],
        'resistance': []
    }
    
    if historical_prices is not None and not historical_prices.empty and price_action_weight > 0:
        # Find recent swing lows for support
        historical_prices['rolling_min'] = historical_prices['low'].rolling(window=5, center=True).min()
        swing_lows = historical_prices[historical_prices['low'] == historical_prices['rolling_min']]['low'].tolist()
        
        # Find recent swing highs for resistance
        historical_prices['rolling_max'] = historical_prices['high'].rolling(window=5, center=True).max()
        swing_highs = historical_prices[historical_prices['high'] == historical_prices['rolling_max']]['high'].tolist()
        
        # Add to price action levels
        price_action_levels['support'] = swing_lows[-3:] if len(swing_lows) > 0 else []  # Last 3 swing lows
        price_action_levels['resistance'] = swing_highs[-3:] if len(swing_highs) > 0 else []  # Last 3 swing highs
        
        # Combine levels with weighting
        if price_action_levels['support'] and support_levels:
            # Create weighted combination
            support_levels = [
                s * (1 - price_action_weight) + p * price_action_weight
                for s in support_levels[:2]  # Top 2 volume-based supports
                for p in price_action_levels['support'][:2]  # Top 2 price action supports
            ] + support_levels[2:]  # Keep remaining volume-based supports
        
        if price_action_levels['resistance'] and resistance_levels:
            # Create weighted combination
            resistance_levels = [
                r * (1 - price_action_weight) + p * price_action_weight
                for r in resistance_levels[:2]  # Top 2 volume-based resistances
                for p in price_action_levels['resistance'][:2]  # Top 2 price action resistances
            ] + resistance_levels[2:]  # Keep remaining volume-based resistances
    
    # Sort levels
    support_levels.sort(reverse=True)
    resistance_levels.sort()
    
    return {
        'support_levels': support_levels,
        'resistance_levels': resistance_levels,
        'vwap_levels': vwap_levels,
        'volume_clusters': {
            'support': volume_support,
            'resistance': volume_resistance
        },
        'price_action': price_action_levels
    }


def generate_execution_recommendations(
    orderbook: Dict[str, Any],
    is_buy: bool = True,
    order_size: Optional[float] = None,
    sr_levels: Optional[Dict[str, Any]] = None,
    risk_tolerance: float = 0.5  # 0.0 (very conservative) to 1.0 (aggressive)
) -> Dict[str, Any]:
    """
    Generate enhanced execution recommendations based on orderbook analysis and support/resistance levels.
    
    Args:
        orderbook: Order book data dictionary with bids and asks
        is_buy: Whether this is a buy recommendation (True) or sell (False)
        order_size: Optional order size for impact estimation
        sr_levels: Support and resistance levels from identify_support_resistance_levels
        risk_tolerance: Risk tolerance level (0.0-1.0)
        
    Returns:
        Dictionary with enhanced execution recommendations
    """
    # Calculate orderbook metrics
    imbalance = calculate_orderbook_imbalance(orderbook)
    liquidity = calculate_liquidity_ratio(orderbook)
    
    # Extract necessary data
    buy_sell_ratio = liquidity.get('buy_sell_ratio', 1.0)
    
    # Get SR levels if not provided
    if sr_levels is None:
        sr_levels = identify_support_resistance_levels(orderbook)
    
    # Extract best bid/ask
    bids = orderbook.get('bids', [])
    asks = orderbook.get('asks', [])
    
    if not bids or not asks:
        return {
            'recommendation': 'insufficient_data',
            'reason': 'No orderbook data available',
            'market_conditions': liquidity
        }
    
    best_bid = float(bids[0][0])
    best_ask = float(asks[0][0])
    mid_price = (best_bid + best_ask) / 2
    current_spread = best_ask - best_bid
    spread_pct = current_spread / mid_price * 100
    
    # Analyze current price position relative to support/resistance
    support_levels = sr_levels.get('support_levels', [])
    resistance_levels = sr_levels.get('resistance_levels', [])
    
    # Find nearest levels
    nearest_support = next((s for s in support_levels if s < mid_price), None)
    nearest_resistance = next((r for r in resistance_levels if r > mid_price), None)
    
    # Calculate distance to nearest levels
    support_distance = (mid_price - nearest_support) / mid_price * 100 if nearest_support else None
    resistance_distance = (nearest_resistance - mid_price) / mid_price * 100 if nearest_resistance else None
    
    # Assess market position
    if nearest_support and nearest_resistance:
        # Calculate position between support and resistance (0.0 = at support, 1.0 = at resistance)
        range_position = (mid_price - nearest_support) / (nearest_resistance - nearest_support)
    else:
        range_position = 0.5  # Default to middle if levels not available
    
    # Calculate expected market impact if order size provided
    expected_impact = None
    if order_size:
        if is_buy:
            consumed_levels = 0
            remaining_size = order_size
            last_price = best_ask
            
            for level in asks:
                price = float(level[0])
                size = float(level[1])
                
                if remaining_size <= size:
                    last_price = price
                    break
                
                consumed_levels += 1
                remaining_size -= size
                last_price = price
            
            expected_impact = (last_price - best_ask) / best_ask * 100
        else:
            consumed_levels = 0
            remaining_size = order_size
            last_price = best_bid
            
            for level in bids:
                price = float(level[0])
                size = float(level[1])
                
                if remaining_size <= size:
                    last_price = price
                    break
                
                consumed_levels += 1
                remaining_size -= size
                last_price = price
            
            expected_impact = (best_bid - last_price) / best_bid * 100
    
    # Determine optimal order type
    # For buys:
    # - Use limit at/below resistance when near resistance or during sell pressure
    # - Use market when strong buy pressure or near support
    # For sells:
    # - Use limit at/above support when near support or during buy pressure
    # - Use market when strong sell pressure or near resistance
    
    order_type = 'limit'  # Default to limit orders
    
    if is_buy:
        # Buy orders
        if (imbalance < -0.3 or (range_position > 0.8 and resistance_distance and resistance_distance < 0.5)):
            # Strong sell pressure or near resistance - use limit
            order_type = 'limit'
            # Adjust limit price based on risk tolerance
            limit_price_modifier = 0.2 * (1 - risk_tolerance)  # Higher risk tolerance = smaller modifier
            limit_price = best_ask * (1 - limit_price_modifier * abs(imbalance))
        elif (imbalance > 0.3 or (range_position < 0.2 and support_distance and support_distance < 0.5)):
            # Strong buy pressure or near support - consider market order
            order_type = 'market' if risk_tolerance > 0.5 else 'limit'
            # If still using limit, use aggressive limit near ask
            limit_price = best_ask * (1 + 0.0001) if order_type == 'limit' else None
        else:
            # Moderate conditions - use standard limit
            limit_price = best_ask * (1 - 0.05 * (1 - risk_tolerance))
    else:
        # Sell orders
        if (imbalance > 0.3 or (range_position < 0.2 and support_distance and support_distance < 0.5)):
            # Strong buy pressure or near support - use limit
            order_type = 'limit'
            # Adjust limit price based on risk tolerance
            limit_price_modifier = 0.2 * (1 - risk_tolerance)  # Higher risk tolerance = smaller modifier
            limit_price = best_bid * (1 + limit_price_modifier * abs(imbalance))
        elif (imbalance < -0.3 or (range_position > 0.8 and resistance_distance and resistance_distance < 0.5)):
            # Strong sell pressure or near resistance - consider market order
            order_type = 'market' if risk_tolerance > 0.5 else 'limit'
            # If still using limit, use aggressive limit near bid
            limit_price = best_bid * (1 - 0.0001) if order_type == 'limit' else None
        else:
            # Moderate conditions - use standard limit
            limit_price = best_bid * (1 + 0.05 * (1 - risk_tolerance))
    
    # Determine if order should be split (TWAP/VWAP)
    should_split = False
    num_parts = 1
    
    # Split if large order or high volatility conditions
    if expected_impact and expected_impact > 0.3:
        should_split = True
        # Higher impact = more parts (up to 5)
        num_parts = min(int(expected_impact * 10) + 1, 5)
    elif abs(imbalance) > 0.7:
        # High imbalance suggests unstable conditions
        should_split = True
        num_parts = 3
    
    # Create enhanced recommendation
    recommendation = {
        'symbol': orderbook.get('symbol', ''),
        'timestamp': datetime.now().isoformat(),
        'is_buy': is_buy,
        'recommendation': {
            'order_type': order_type,
            'limit_price': limit_price if order_type == 'limit' else None,
            'should_split': should_split,
            'num_parts': num_parts,
            'expected_impact': expected_impact,
            'time_interval': 60 if should_split else None,  # seconds between parts
        },
        'market_conditions': {
            **liquidity,
            'imbalance': imbalance,
            'spread_pct': spread_pct,
            'range_position': range_position,
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance,
            'support_distance_pct': support_distance,
            'resistance_distance_pct': resistance_distance
        },
        'confidence_score': calculate_confidence_score(
            imbalance=imbalance, 
            range_position=range_position, 
            is_buy=is_buy
        )
    }
    
    return recommendation 


def calculate_confidence_score(
    imbalance: float, 
    range_position: float, 
    is_buy: bool
) -> float:
    """
    Calculate a confidence score for a recommendation.
    
    Args:
        imbalance: Order book imbalance (-1.0 to 1.0)
        range_position: Position between support and resistance (0.0 = at support, 1.0 = at resistance)
        is_buy: Whether this is a buy recommendation
        
    Returns:
        Confidence score (0.0 to 1.0)
    """
    # For buys, higher confidence when:
    # - Strong positive imbalance (buy pressure)
    # - Price near support (low range_position)
    #
    # For sells, higher confidence when:
    # - Strong negative imbalance (sell pressure)
    # - Price near resistance (high range_position)
    
    if is_buy:
        # Convert imbalance to 0-1 scale for buys (higher = better)
        imbalance_score = (imbalance + 1) / 2
        # Lower range_position is better for buys
        position_score = 1 - range_position
    else:
        # Convert imbalance to 0-1 scale for sells (lower imbalance = higher score)
        imbalance_score = (-imbalance + 1) / 2
        # Higher range_position is better for sells
        position_score = range_position
    
    # Weight the factors (can be adjusted)
    confidence = (imbalance_score * 0.6) + (position_score * 0.4)
    
    # Ensure within 0-1 range
    return max(0.0, min(1.0, confidence)) 