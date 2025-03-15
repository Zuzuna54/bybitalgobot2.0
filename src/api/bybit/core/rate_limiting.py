"""
Rate Limiting for Bybit API

This module provides functionality for managing API rate limits,
including throttling, request tracking, and backoff strategies.
"""

import time
import threading
from typing import Dict, Any, Optional, Callable, List, Tuple
from datetime import datetime, timedelta
from loguru import logger


class RateLimitManager:
    """
    Manages rate limiting for API requests to prevent hitting limits.
    """
    
    def __init__(self):
        """
        Initialize the rate limit manager.
        """
        # Default rate limits for Bybit API
        self.rate_limits = {
            # Endpoint category: (requests_per_second, max_burst)
            'market': (10, 20),            # Market data endpoints
            'account': (5, 10),            # Account endpoints
            'order': (5, 10),              # Order endpoints
            'position': (5, 10),           # Position endpoints
            'wallet': (5, 10),             # Wallet endpoints
            'default': (3, 5)              # Default for uncategorized endpoints
        }
        
        # Request tracking: category -> list of timestamps
        self.request_history: Dict[str, List[float]] = {}
        self.lock = threading.RLock()
    
    def update_rate_limits(self, limits: Dict[str, Tuple[int, int]]) -> None:
        """
        Update rate limits for specific categories.
        
        Args:
            limits: Dictionary mapping categories to (requests_per_second, max_burst) tuples
        """
        with self.lock:
            for category, limit in limits.items():
                self.rate_limits[category] = limit
            logger.debug(f"Rate limits updated: {limits}")
    
    def _clean_history(self, category: str, current_time: float) -> None:
        """
        Clean up old request history.
        
        Args:
            category: Request category
            current_time: Current timestamp
        """
        if category in self.request_history:
            # Keep only requests in the last minute
            cutoff = current_time - 60
            self.request_history[category] = [
                t for t in self.request_history[category] if t > cutoff
            ]
    
    def acquire(self, category: str = 'default') -> float:
        """
        Acquire permission to make a request, waiting if necessary.
        
        Args:
            category: Request category
            
        Returns:
            Delay time in seconds (0 if no delay was needed)
        """
        with self.lock:
            # Get the appropriate rate limit
            requests_per_second, max_burst = self.rate_limits.get(
                category, self.rate_limits['default']
            )
            
            current_time = time.time()
            
            # Initialize history for this category if not exists
            if category not in self.request_history:
                self.request_history[category] = []
            
            # Clean up old history
            self._clean_history(category, current_time)
            
            # Count recent requests (last second)
            recent_count = sum(
                1 for t in self.request_history[category] 
                if t > current_time - 1
            )
            
            # Calculate delay if needed
            delay = 0.0
            if recent_count >= requests_per_second:
                # Calculate the earliest time we can make the next request
                next_slot = self.request_history[category][-requests_per_second] + 1
                delay = max(0, next_slot - current_time)
                
                if delay > 0:
                    logger.debug(f"Rate limit throttling for {category}: waiting {delay:.3f}s")
                    time.sleep(delay)
                    current_time = time.time()
            
            # Record this request
            self.request_history[category].append(current_time)
            
            return delay
    
    def reset(self) -> None:
        """
        Reset all rate limiting history.
        """
        with self.lock:
            self.request_history.clear()
            logger.debug("Rate limit history reset")


# Global rate limit manager instance
_rate_limit_manager = None


def initialize_rate_limiters() -> RateLimitManager:
    """
    Initialize the global rate limit manager.
    
    Returns:
        The initialized rate limit manager
    """
    global _rate_limit_manager
    if _rate_limit_manager is None:
        _rate_limit_manager = RateLimitManager()
        logger.debug("Rate limit manager initialized")
    return _rate_limit_manager


def rate_limited(category: str = 'default'):
    """
    Decorator for rate-limiting API calls.
    
    Args:
        category: Request category for rate limiting
        
    Returns:
        Decorated function with rate limiting
    """
    # Ensure rate limit manager is initialized
    global _rate_limit_manager
    if _rate_limit_manager is None:
        _rate_limit_manager = initialize_rate_limiters()
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Acquire permission to make the request
            _rate_limit_manager.acquire(category)
            return func(*args, **kwargs)
        return wrapper
    
    return decorator