"""
Order Book Analyzer

This module provides the main OrderBookAnalyzer class which integrates
various order book analysis functions from other modules.
"""

from typing import Dict, Any, Tuple, Optional, List
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from loguru import logger

# Import functionality from other modules
from src.trade_execution.orderbook.depth_analysis import (
    get_significant_levels,
    calculate_market_impact,
    get_optimal_trade_size,
    get_optimal_limit_price
)
from src.trade_execution.orderbook.liquidity import (
    analyze_liquidity,
    should_split_order,
    calculate_execution_quality
)
from src.trade_execution.orderbook.signals import (
    detect_order_book_imbalance,
    generate_entry_signal,
    generate_exit_signal
)


class OrderBookAnalyzer:
    """
    Analyzes order book data to optimize trade executions.
    
    This class provides a high-level interface to various order book analysis
    functions and maintains an internal cache of order book data.
    """
    
    def __init__(self, depth_threshold: int = 5, historical_data_length: int = 100):
        """
        Initialize the order book analyzer.
        
        Args:
            depth_threshold: Threshold for significant levels determination
            historical_data_length: Number of historical orderbook snapshots to keep
        """
        self.depth_threshold = depth_threshold
        self.orderbook_cache: Dict[str, Dict[str, Any]] = {}
        
        # Historical price data for better support/resistance detection
        self.historical_prices: Dict[str, List[Dict[str, Any]]] = {}
        self.max_history_length = historical_data_length
        
        # Cache for calculated levels
        self.sr_levels_cache: Dict[str, Dict[str, Any]] = {}
        self.sr_cache_expiry: Dict[str, float] = {}
        self.sr_cache_lifetime = 300  # 5 minutes in seconds
        
        logger.info("Order book analyzer initialized")
    
    def update_orderbook(self, symbol: str, orderbook_data: Dict[str, Any]) -> None:
        """
        Update the internal orderbook cache with new data.
        
        Args:
            symbol: Trading pair symbol
            orderbook_data: Order book data from the exchange
        """
        self.orderbook_cache[symbol] = orderbook_data
        
        # Extract mid price for historical tracking
        if 'bids' in orderbook_data and 'asks' in orderbook_data and orderbook_data['bids'] and orderbook_data['asks']:
            best_bid = float(orderbook_data['bids'][0][0])
            best_ask = float(orderbook_data['asks'][0][0])
            mid_price = (best_bid + best_ask) / 2
            timestamp = datetime.now()
            
            # Update historical price data
            if symbol not in self.historical_prices:
                self.historical_prices[symbol] = []
            
            # Add new price point
            self.historical_prices[symbol].append({
                'timestamp': timestamp,
                'price': mid_price,
                'high': best_ask,
                'low': best_bid,
                'volume': sum(float(bid[1]) for bid in orderbook_data['bids'][:5]) + 
                          sum(float(ask[1]) for ask in orderbook_data['asks'][:5])
            })
            
            # Trim history if too long
            if len(self.historical_prices[symbol]) > self.max_history_length:
                self.historical_prices[symbol] = self.historical_prices[symbol][-self.max_history_length:]
    
    def get_orderbook(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get the current orderbook for a symbol from the cache.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Order book data if available, None otherwise
        """
        orderbook = self.orderbook_cache.get(symbol)
        
        if not orderbook:
            logger.warning(f"No order book data available for {symbol}")
            return None
            
        return orderbook
    
    def get_historical_prices_df(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Get historical price data as a DataFrame.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            DataFrame with historical prices or None if not available
        """
        history = self.historical_prices.get(symbol)
        
        if not history or len(history) < 10:  # Need at least 10 data points
            return None
        
        return pd.DataFrame(history)
    
    def identify_support_resistance_levels(self, symbol: str, force_recalculate: bool = False) -> Dict[str, Any]:
        """
        Identify support and resistance levels using orderbook data and price history.
        
        Args:
            symbol: Trading pair symbol
            force_recalculate: Whether to force recalculation of levels
            
        Returns:
            Dictionary with support and resistance levels
        """
        # Check if we have a fresh cached result
        current_time = time.time()
        cache_expiry = self.sr_cache_expiry.get(symbol, 0)
        
        if not force_recalculate and symbol in self.sr_levels_cache and current_time < cache_expiry:
            return self.sr_levels_cache[symbol]
        
        # Get orderbook data
        orderbook = self.get_orderbook(symbol)
        if not orderbook:
            return {
                'support_levels': [],
                'resistance_levels': [],
                'vwap_levels': {},
                'volume_clusters': {
                    'support': [],
                    'resistance': []
                }
            }
        
        # Get historical price data if available
        historical_df = self.get_historical_prices_df(symbol)
        
        # Call the enhanced support/resistance detection function from data_processing
        from src.dashboard.components.orderbook.data_processing import identify_support_resistance_levels
        
        sr_levels = identify_support_resistance_levels(
            orderbook=orderbook,
            historical_prices=historical_df,
            vwap_periods=[20, 50, 200],
            depth=20,  # Analyze more depth
            volume_cluster_threshold=2.0,
            price_action_weight=0.4
        )
        
        # Cache the result
        self.sr_levels_cache[symbol] = sr_levels
        self.sr_cache_expiry[symbol] = current_time + self.sr_cache_lifetime
        
        return sr_levels
    
    def recommend_entry_strategy(
        self, 
        symbol: str, 
        order_size: float, 
        is_buy: bool,
        risk_tolerance: float = 0.5  # 0.0 (conservative) to 1.0 (aggressive)
    ) -> Dict[str, Any]:
        """
        Recommend an optimal entry strategy based on enhanced order book analysis.
        
        Args:
            symbol: Trading pair symbol
            order_size: Order size in base currency
            is_buy: Whether it's a buy or sell order
            risk_tolerance: Risk tolerance level (0-1)
            
        Returns:
            Dictionary with enhanced entry strategy recommendations
        """
        orderbook = self.get_orderbook(symbol)
        if not orderbook:
            return {
                "symbol": symbol,
                "error": "No order book data available"
            }
        
        # Get support/resistance levels
        sr_levels = self.identify_support_resistance_levels(symbol)
        
        # Use the enhanced recommendation function from data_processing
        from src.dashboard.components.orderbook.data_processing import generate_execution_recommendations
        
        recommendations = generate_execution_recommendations(
            orderbook=orderbook,
            is_buy=is_buy,
            order_size=order_size,
            sr_levels=sr_levels,
            risk_tolerance=risk_tolerance
        )
        
        # Add additional metadata
        recommendations["timestamp"] = datetime.now().isoformat()
        recommendations["symbol"] = symbol
        recommendations["analysis_source"] = "enhanced_orderbook_analyzer"
        
        return recommendations
    
    def recommend_exit_strategy(
        self, 
        symbol: str, 
        position_size: float, 
        is_long: bool, 
        entry_price: Optional[float] = None,
        risk_tolerance: float = 0.5
    ) -> Dict[str, Any]:
        """
        Recommend an optimal exit strategy based on enhanced order book analysis.
        
        Args:
            symbol: Trading pair symbol
            position_size: Position size in base currency
            is_long: Whether it's a long position (True) or short position (False)
            entry_price: Original entry price (optional)
            risk_tolerance: Risk tolerance level (0-1)
            
        Returns:
            Dictionary with exit strategy recommendations
        """
        orderbook = self.get_orderbook(symbol)
        if not orderbook:
            return {
                "symbol": symbol,
                "error": "No order book data available"
            }
        
        # Get support/resistance levels
        sr_levels = self.identify_support_resistance_levels(symbol)
        
        # Extract necessary data
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        
        if not bids or not asks:
            return {
                "symbol": symbol,
                "error": "Order book is empty"
            }
        
        # Current market price
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        mid_price = (best_bid + best_ask) / 2
        
        # Calculate imbalance
        imbalance = detect_order_book_imbalance(orderbook)
        
        # Determine if we're near support or resistance
        support_levels = sr_levels.get('support_levels', [])
        resistance_levels = sr_levels.get('resistance_levels', [])
        
        nearest_support = next((s for s in support_levels if s < mid_price), None)
        nearest_resistance = next((r for r in resistance_levels if r > mid_price), None)
        
        support_distance = (mid_price - nearest_support) / mid_price * 100 if nearest_support else None
        resistance_distance = (nearest_resistance - mid_price) / mid_price * 100 if nearest_resistance else None
        
        # Calculate position between support and resistance
        if nearest_support and nearest_resistance:
            range_position = (mid_price - nearest_support) / (nearest_resistance - nearest_support)
        else:
            range_position = 0.5
            
        # Generate exit recommendations based on position type and market conditions
        if is_long:
            # For long positions (selling)
            if (range_position > 0.8 and resistance_distance and resistance_distance < 0.5) or imbalance < -0.3:
                # Near resistance or bearish imbalance - consider taking profit
                action = "take_profit"
                urgency = 0.8
                order_type = "limit" if risk_tolerance < 0.7 else "market"
                limit_price = best_bid if order_type == "limit" else None
            elif (range_position < 0.2 and support_distance and support_distance < 0.3) and imbalance > 0.3:
                # Near support with bullish imbalance - consider holding
                action = "hold"
                urgency = 0.3
                order_type = "limit"
                limit_price = best_bid * 1.01  # Set limit slightly above bid
            else:
                # Neutral conditions - moderate urgency
                action = "adjust_based_on_pnl"
                urgency = 0.5
                order_type = "limit"
                limit_price = best_bid
                
            # Adjust based on entry price if available
            if entry_price:
                pnl_pct = (mid_price - entry_price) / entry_price * 100
                
                if pnl_pct > 3.0:
                    # Significant profit - consider securing gains
                    action = "take_profit"
                    urgency = min(0.5 + (pnl_pct / 10), 0.9)  # Higher urgency for higher profits
                elif pnl_pct < -2.0:
                    # In loss - consider cutting losses if not near support
                    if not (range_position < 0.2 and imbalance > 0.3):
                        action = "cut_loss"
                        urgency = min(0.5 + (abs(pnl_pct) / 10), 0.9)
        else:
            # For short positions (buying back)
            if (range_position < 0.2 and support_distance and support_distance < 0.5) or imbalance > 0.3:
                # Near support or bullish imbalance - consider taking profit
                action = "take_profit"
                urgency = 0.8
                order_type = "limit" if risk_tolerance < 0.7 else "market"
                limit_price = best_ask if order_type == "limit" else None
            elif (range_position > 0.8 and resistance_distance and resistance_distance < 0.3) and imbalance < -0.3:
                # Near resistance with bearish imbalance - consider holding
                action = "hold"
                urgency = 0.3
                order_type = "limit"
                limit_price = best_ask * 0.99  # Set limit slightly below ask
            else:
                # Neutral conditions - moderate urgency
                action = "adjust_based_on_pnl"
                urgency = 0.5
                order_type = "limit"
                limit_price = best_ask
                
            # Adjust based on entry price if available
            if entry_price:
                pnl_pct = (entry_price - mid_price) / entry_price * 100
                
                if pnl_pct > 3.0:
                    # Significant profit - consider securing gains
                    action = "take_profit"
                    urgency = min(0.5 + (pnl_pct / 10), 0.9)
                elif pnl_pct < -2.0:
                    # In loss - consider cutting losses if not near resistance
                    if not (range_position > 0.8 and imbalance < -0.3):
                        action = "cut_loss"
                        urgency = min(0.5 + (abs(pnl_pct) / 10), 0.9)
        
        # Determine if order should be split
        if position_size > get_optimal_trade_size(orderbook, max_impact_pct=0.5, is_buy=not is_long):
            should_split = True
            num_parts = 3
        else:
            should_split = False
            num_parts = 1
            
        # Calculate confidence score based on market conditions and recommendation
        if action == "take_profit":
            confidence = urgency * 0.8 + 0.1  # Higher urgency = higher confidence
        elif action == "cut_loss":
            confidence = urgency * 0.7 + 0.1  # Slightly lower confidence for cutting losses
        elif action == "hold":
            confidence = (1 - urgency) * 0.8 + 0.1  # Lower urgency = higher confidence for holding
        else:
            confidence = 0.5  # Neutral confidence for other actions
        
        # Create recommendation
        recommendation = {
            "symbol": symbol,
            "is_long": is_long,
            "timestamp": datetime.now().isoformat(),
            "recommendation": {
                "action": action,
                "urgency": urgency,
            "order_type": order_type,
                "limit_price": limit_price,
            "should_split": should_split,
            "num_parts": num_parts,
                "confidence": confidence
            },
            "market_conditions": {
                "imbalance": imbalance,
                "range_position": range_position,
                "nearest_support": nearest_support,
                "nearest_resistance": nearest_resistance,
                "support_distance_pct": support_distance,
                "resistance_distance_pct": resistance_distance
            }
        }
        
        if entry_price:
            recommendation["trade_metrics"] = {
                "entry_price": entry_price,
                "current_price": mid_price,
                "pnl_pct": (mid_price - entry_price) / entry_price * 100 if is_long else
                           (entry_price - mid_price) / entry_price * 100
            }
        
        return recommendation 