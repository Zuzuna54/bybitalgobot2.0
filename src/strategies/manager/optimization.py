"""
Strategy Optimization for the Algorithmic Trading System

This module provides functionality for optimizing trading strategies,
adjusting weights, and performance-based strategy selection.
"""

from typing import Dict, Any, List, Optional, Tuple
from loguru import logger


def adjust_strategy_weight(
    strategy_performance: Dict[str, Dict[str, Any]],
    strategy_weights: Dict[str, float],
    strategy_name: str,
    weight: float
) -> bool:
    """
    Manually adjust strategy weight.
    
    Args:
        strategy_performance: Dictionary of strategy performance metrics
        strategy_weights: Dictionary of strategy weights
        strategy_name: Name of the strategy
        weight: New weight value (0.5 to 2.0)
        
    Returns:
        True if weight was adjusted, False otherwise
    """
    if strategy_name in strategy_performance:
        # Limit weight to valid range
        limited_weight = max(min(weight, 2.0), 0.5)
        
        # Update weight
        strategy_performance[strategy_name]["weight"] = limited_weight
        strategy_weights[strategy_name] = limited_weight
        
        logger.info(f"Adjusted weight for strategy {strategy_name} to {limited_weight:.2f}")
        return True
    
    logger.warning(f"Could not adjust weight for unknown strategy: {strategy_name}")
    return False


def optimize_strategy_weights(
    strategy_performance: Dict[str, Dict[str, Any]],
    strategy_weights: Dict[str, float],
    min_trades: int = 20
) -> Dict[str, float]:
    """
    Optimize strategy weights based on performance metrics.
    
    Args:
        strategy_performance: Dictionary of strategy performance metrics
        strategy_weights: Current dictionary of strategy weights
        min_trades: Minimum number of trades required for optimization
        
    Returns:
        Dictionary with optimized strategy weights
    """
    optimized_weights = strategy_weights.copy()
    
    for strategy_name, performance in strategy_performance.items():
        # Skip strategies with insufficient data
        if performance.get("signals_executed", 0) < min_trades:
            continue
        
        # Calculate performance score
        win_rate = performance.get("win_rate", 0.0)
        total_pnl = performance.get("total_profit_loss", 0.0)
        
        # Simple weighted formula: 60% win rate, 40% profit
        # Normalize profit to -1 to +1 range using sigmoid-like function
        normalized_pnl = min(max(total_pnl / (abs(total_pnl) + 100.0), -1.0), 1.0)
        
        # Calculate score (0 to 1 range)
        score = (win_rate * 0.6) + ((normalized_pnl + 1.0) / 2.0 * 0.4)
        
        # Map score to weight range (0.5 to 2.0)
        optimized_weight = 0.5 + (score * 1.5)
        
        # Update weight
        optimized_weights[strategy_name] = optimized_weight
        
        logger.debug(f"Optimized weight for {strategy_name}: {optimized_weight:.2f} (score: {score:.2f})")
    
    return optimized_weights


def select_top_strategies(
    strategy_performance: Dict[str, Dict[str, Any]],
    max_strategies: int = 5,
    min_trades: int = 10
) -> List[str]:
    """
    Select top performing strategies based on performance metrics.
    
    Args:
        strategy_performance: Dictionary of strategy performance metrics
        max_strategies: Maximum number of strategies to select
        min_trades: Minimum number of trades required for consideration
        
    Returns:
        List of strategy names sorted by performance
    """
    # Filter strategies with enough trades
    eligible_strategies = [
        (name, perf) for name, perf in strategy_performance.items()
        if perf.get("signals_executed", 0) >= min_trades
    ]
    
    # Calculate composite score for each strategy
    strategy_scores = []
    for name, perf in eligible_strategies:
        win_rate = perf.get("win_rate", 0.0)
        profit = perf.get("total_profit_loss", 0.0)
        
        # Calculate profit factor
        if profit > 0:
            wins = perf.get("successful_signals", 0)
            losses = perf.get("failed_signals", 0)
            avg_win = profit / wins if wins > 0 else 0
            avg_loss = abs(profit) / losses if losses > 0 else 0
            profit_factor = avg_win / avg_loss if avg_loss > 0 else 2.0
        else:
            profit_factor = 0.0
        
        # Composite score (win rate * profit factor)
        score = win_rate * max(min(profit_factor, 3.0), 0.0) / 3.0
        
        strategy_scores.append((name, score))
    
    # Sort by score and limit to max_strategies
    sorted_strategies = sorted(strategy_scores, key=lambda x: x[1], reverse=True)
    top_strategies = [name for name, _ in sorted_strategies[:max_strategies]]
    
    return top_strategies


def get_recommended_adjustments(
    strategy_performance: Dict[str, Dict[str, Any]],
    strategy_weights: Dict[str, float]
) -> List[Dict[str, Any]]:
    """
    Get recommended weight adjustments based on strategy performance.
    
    Args:
        strategy_performance: Dictionary of strategy performance metrics
        strategy_weights: Current dictionary of strategy weights
        
    Returns:
        List of recommended adjustments
    """
    recommendations = []
    
    for strategy_name, performance in strategy_performance.items():
        signals_executed = performance.get("signals_executed", 0)
        
        # Need at least 10 trades for a recommendation
        if signals_executed < 10:
            continue
        
        win_rate = performance.get("win_rate", 0.0)
        current_weight = performance.get("weight", 1.0)
        
        # Calculate ideal weight based on win rate
        ideal_weight = 0.5 + (win_rate * 1.5)
        
        # If there's a significant difference, recommend an adjustment
        if abs(ideal_weight - current_weight) > 0.2:
            recommended_weight = (ideal_weight + current_weight) / 2.0  # Gradual adjustment
            
            recommendations.append({
                "strategy_name": strategy_name,
                "current_weight": current_weight,
                "recommended_weight": round(recommended_weight, 2),
                "win_rate": win_rate,
                "trades": signals_executed
            })
    
    return recommendations 