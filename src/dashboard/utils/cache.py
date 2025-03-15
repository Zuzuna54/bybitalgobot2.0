"""
Dashboard Caching Module

This module provides utilities for caching data to improve performance in the dashboard.
It includes both in-memory and persistent caching mechanisms.
"""

import os
import json
import pickle
import time
import hashlib
from typing import Dict, Any, Optional, List, Union, Callable, Tuple
from functools import wraps
from datetime import datetime, timedelta
import threading
import shutil

from .logger import get_logger, measure_execution_time
from .config_manager import get_config_manager

logger = get_logger("cache")


class CacheEntry:
    """Represents a single entry in the cache."""
    
    def __init__(self, key: str, value: Any, expire_seconds: Optional[int] = None):
        """
        Initialize a cache entry.
        
        Args:
            key: Cache key
            value: Value to cache
            expire_seconds: Seconds until the cache entry expires
        """
        self.key = key
        self.value = value
        self.timestamp = time.time()
        self.expire_time = None if expire_seconds is None else self.timestamp + expire_seconds
        self.access_count = 0
    
    def is_expired(self) -> bool:
        """
        Check if the cache entry has expired.
        
        Returns:
            True if the cache entry has expired, False otherwise
        """
        if self.expire_time is None:
            return False
        return time.time() > self.expire_time
    
    def get_age_seconds(self) -> float:
        """
        Get the age of the cache entry in seconds.
        
        Returns:
            Age in seconds
        """
        return time.time() - self.timestamp
    
    def refresh(self, expire_seconds: Optional[int] = None) -> None:
        """
        Refresh the cache entry's expiration time.
        
        Args:
            expire_seconds: New expiration time in seconds
        """
        self.timestamp = time.time()
        
        if expire_seconds is not None:
            self.expire_time = self.timestamp + expire_seconds
        elif self.expire_time is not None:
            # Extend by the original expiration period
            original_expiry = self.expire_time - (self.timestamp - self.get_age_seconds())
            self.expire_time = self.timestamp + original_expiry
    
    def increment_access(self) -> int:
        """
        Increment the access count for this cache entry.
        
        Returns:
            The new access count
        """
        self.access_count += 1
        return self.access_count


class DashboardCache:
    """
    Caching system for the dashboard with in-memory and persistent storage options.
    """
    
    def __init__(self, 
                 name: str = "dashboard_cache",
                 max_size: int = 1000,
                 default_ttl: int = 3600,
                 cleanup_interval: int = 300,
                 persistent: bool = False,
                 cache_dir: Optional[str] = None):
        """
        Initialize the dashboard cache.
        
        Args:
            name: Cache name
            max_size: Maximum number of items in memory cache
            default_ttl: Default time-to-live in seconds
            cleanup_interval: Interval in seconds for cache cleanup
            persistent: Whether to enable persistent storage
            cache_dir: Directory for persistent cache storage
        """
        self.name = name
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cleanup_interval = cleanup_interval
        self.persistent = persistent
        
        # Get configuration
        config = get_config_manager()
        
        # Set instance variables from parameters or config
        self.max_size = max_size or config.get_int("dashboard.cache_max_size", 1000)
        self.default_ttl = default_ttl or config.get_int("dashboard.cache_expiry", 3600)
        self.cleanup_interval = cleanup_interval or config.get_int("dashboard.cache_cleanup_interval", 300)
        self.persistent = persistent if persistent is not None else config.get_bool("dashboard.cache_persistent", False)
        
        # Set up cache directory
        if cache_dir:
            self.cache_dir = cache_dir
        else:
            base_dir = config.get_path("data_dir") or "data"
            self.cache_dir = os.path.join(base_dir, "cache")
        
        # Create cache directory if it doesn't exist and persistent caching is enabled
        if self.persistent and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize the in-memory cache
        self.cache: Dict[str, CacheEntry] = {}
        
        # Set up cache lock for thread safety
        self.cache_lock = threading.RLock()
        
        # Start the cleanup thread
        if self.cleanup_interval > 0:
            self.start_cleanup_thread()
    
    def start_cleanup_thread(self) -> None:
        """Start the background thread for cache cleanup."""
        self.cleanup_thread = threading.Thread(target=self._cleanup_thread, daemon=True)
        self.cleanup_thread.start()
    
    def _cleanup_thread(self) -> None:
        """Background thread function for periodic cache cleanup."""
        while True:
            time.sleep(self.cleanup_interval)
            try:
                self.cleanup()
            except Exception as e:
                logger.error(f"Error during cache cleanup: {str(e)}")
    
    @measure_execution_time
    def cleanup(self) -> int:
        """
        Clean up expired cache entries.
        
        Returns:
            Number of entries removed
        """
        removed_count = 0
        
        with self.cache_lock:
            # Find expired keys
            expired_keys = [key for key, entry in self.cache.items() if entry.is_expired()]
            
            # Remove expired entries
            for key in expired_keys:
                self.cache.pop(key, None)
                removed_count += 1
            
            # If still over max size, remove least accessed entries
            if len(self.cache) > self.max_size:
                # Sort entries by access count (ascending)
                sorted_entries = sorted(
                    self.cache.items(),
                    key=lambda x: (x[1].access_count, -x[1].timestamp)
                )
                
                # Remove oldest, least accessed entries
                to_remove = len(self.cache) - self.max_size
                for key, _ in sorted_entries[:to_remove]:
                    self.cache.pop(key, None)
                    removed_count += 1
        
        logger.debug(f"Cache cleanup removed {removed_count} entries, {len(self.cache)} remaining")
        return removed_count
    
    def _generate_key(self, base_key: str, args: tuple, kwargs: dict) -> str:
        """
        Generate a cache key from a base key and function arguments.
        
        Args:
            base_key: Base key string
            args: Function positional arguments
            kwargs: Function keyword arguments
            
        Returns:
            Generated cache key
        """
        # Create a string representation of the arguments
        arg_str = str(args) + str(sorted(kwargs.items()))
        
        # Generate a hash of the arguments
        arg_hash = hashlib.md5(arg_str.encode()).hexdigest()
        
        # Combine base key with argument hash
        return f"{base_key}:{arg_hash}"
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            default: Default value if key not found or expired
            
        Returns:
            Cached value or default
        """
        with self.cache_lock:
            # Check in-memory cache
            entry = self.cache.get(key)
            
            if entry is not None:
                if entry.is_expired():
                    # Remove expired entry
                    self.cache.pop(key, None)
                else:
                    # Update access count and return value
                    entry.increment_access()
                    logger.debug(f"Cache hit for key: {key}")
                    return entry.value
            
            # If not in memory and persistent is enabled, try to load from disk
            if self.persistent:
                value = self._load_from_disk(key)
                if value is not None:
                    # Add to in-memory cache
                    self.set(key, value)
                    logger.debug(f"Loaded from persistent cache: {key}")
                    return value
        
        logger.debug(f"Cache miss for key: {key}")
        return default
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
        """
        with self.cache_lock:
            # Use default TTL if not specified
            if ttl is None:
                ttl = self.default_ttl
            
            # Create new cache entry
            entry = CacheEntry(key, value, ttl)
            
            # Store in memory cache
            self.cache[key] = entry
            
            # Store in persistent cache if enabled
            if self.persistent:
                self._save_to_disk(key, value)
        
        logger.debug(f"Set cache key: {key} (TTL: {ttl}s)")
        
        # If cache is getting too big, trigger cleanup
        if len(self.cache) > self.max_size:
            self.cleanup()
    
    def delete(self, key: str) -> bool:
        """
        Delete a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if the key was deleted, False if it wasn't found
        """
        with self.cache_lock:
            # Remove from in-memory cache
            removed = self.cache.pop(key, None) is not None
            
            # Remove from persistent cache if enabled
            if self.persistent:
                self._delete_from_disk(key)
        
        if removed:
            logger.debug(f"Deleted cache key: {key}")
        
        return removed
    
    def clear(self) -> int:
        """
        Clear all items from the cache.
        
        Returns:
            Number of items cleared
        """
        with self.cache_lock:
            count = len(self.cache)
            self.cache.clear()
            
            # Clear persistent cache if enabled
            if self.persistent:
                self._clear_disk_cache()
        
        logger.debug(f"Cleared cache ({count} items)")
        return count
    
    def refresh(self, key: str, ttl: Optional[int] = None) -> bool:
        """
        Refresh the expiration time of a cache entry.
        
        Args:
            key: Cache key
            ttl: New time-to-live in seconds
            
        Returns:
            True if the key was found and refreshed, False otherwise
        """
        with self.cache_lock:
            entry = self.cache.get(key)
            
            if entry is not None:
                entry.refresh(ttl)
                logger.debug(f"Refreshed cache key: {key}")
                return True
            
            return False
    
    def stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary of cache statistics
        """
        with self.cache_lock:
            # Count expired entries
            expired_count = sum(1 for entry in self.cache.values() if entry.is_expired())
            
            # Calculate average age
            avg_age = 0
            if self.cache:
                avg_age = sum(entry.get_age_seconds() for entry in self.cache.values()) / len(self.cache)
            
            stats = {
                "name": self.name,
                "size": len(self.cache),
                "max_size": self.max_size,
                "expired": expired_count,
                "avg_age_seconds": avg_age,
                "persistent": self.persistent,
                "cache_dir": self.cache_dir if self.persistent else None,
            }
        
        return stats
    
    def _get_disk_path(self, key: str) -> str:
        """
        Get the disk path for a cache key.
        
        Args:
            key: Cache key
            
        Returns:
            Disk path for the cache key
        """
        # Create a safe filename from the key
        safe_key = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{safe_key}.cache")
    
    def _save_to_disk(self, key: str, value: Any) -> bool:
        """
        Save a value to the persistent cache.
        
        Args:
            key: Cache key
            value: Value to cache
            
        Returns:
            True if the value was saved successfully, False otherwise
        """
        if not self.persistent:
            return False
        
        try:
            # Get disk path
            disk_path = self._get_disk_path(key)
            
            # Create cache directory if it doesn't exist
            os.makedirs(os.path.dirname(disk_path), exist_ok=True)
            
            # Save the value to disk
            with open(disk_path, 'wb') as f:
                pickle.dump({
                    'key': key,
                    'value': value,
                    'timestamp': time.time(),
                    'expire_time': time.time() + self.default_ttl
                }, f)
            
            return True
        
        except Exception as e:
            logger.error(f"Error saving to persistent cache: {str(e)}")
            return False
    
    def _load_from_disk(self, key: str) -> Optional[Any]:
        """
        Load a value from the persistent cache.
        
        Args:
            key: Cache key
            
        Returns:
            The cached value or None if not found or expired
        """
        if not self.persistent:
            return None
        
        try:
            # Get disk path
            disk_path = self._get_disk_path(key)
            
            # Check if the file exists
            if not os.path.exists(disk_path):
                return None
            
            # Load the value from disk
            with open(disk_path, 'rb') as f:
                data = pickle.load(f)
            
            # Check if the value has expired
            if data.get('expire_time', 0) < time.time():
                # Remove expired file
                os.remove(disk_path)
                return None
            
            return data.get('value')
        
        except Exception as e:
            logger.error(f"Error loading from persistent cache: {str(e)}")
            return None
    
    def _delete_from_disk(self, key: str) -> bool:
        """
        Delete a value from the persistent cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if the value was deleted, False otherwise
        """
        if not self.persistent:
            return False
        
        try:
            # Get disk path
            disk_path = self._get_disk_path(key)
            
            # Check if the file exists
            if os.path.exists(disk_path):
                os.remove(disk_path)
                return True
            
            return False
        
        except Exception as e:
            logger.error(f"Error deleting from persistent cache: {str(e)}")
            return False
    
    def _clear_disk_cache(self) -> bool:
        """
        Clear all values from the persistent cache.
        
        Returns:
            True if the cache was cleared, False otherwise
        """
        if not self.persistent:
            return False
        
        try:
            # Check if the cache directory exists
            if os.path.exists(self.cache_dir):
                # Remove all cache files
                for filename in os.listdir(self.cache_dir):
                    if filename.endswith('.cache'):
                        os.remove(os.path.join(self.cache_dir, filename))
                
                return True
            
            return False
        
        except Exception as e:
            logger.error(f"Error clearing persistent cache: {str(e)}")
            return False


# Global cache instance
_cache_instance = None


def get_cache() -> DashboardCache:
    """
    Get the singleton cache instance.
    
    Returns:
        The dashboard cache instance
    """
    global _cache_instance
    
    if _cache_instance is None:
        _cache_instance = DashboardCache()
    
    return _cache_instance


def cached(ttl: Optional[int] = None, key_prefix: Optional[str] = None):
    """
    Decorator for caching function results.
    
    Args:
        ttl: Time-to-live in seconds
        key_prefix: Optional prefix for the cache key
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_cache()
            
            # Generate cache key
            func_name = key_prefix or func.__name__
            cache_key = cache._generate_key(func_name, args, kwargs)
            
            # Try to get from cache
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                return cached_value
            
            # Call the function
            value = func(*args, **kwargs)
            
            # Store in cache
            cache.set(cache_key, value, ttl)
            
            return value
        
        return wrapper
    
    return decorator


def invalidate_cache(key_prefix: str) -> int:
    """
    Invalidate cache entries with the given prefix.
    
    Args:
        key_prefix: Prefix for cache keys to invalidate
        
    Returns:
        Number of entries invalidated
    """
    cache = get_cache()
    count = 0
    
    with cache.cache_lock:
        # Find keys with the prefix
        keys_to_delete = [key for key in cache.cache.keys() if key.startswith(key_prefix)]
        
        # Delete matching keys
        for key in keys_to_delete:
            cache.delete(key)
            count += 1
    
    logger.debug(f"Invalidated {count} cache entries with prefix '{key_prefix}'")
    return count


def precache(func: Callable, *args, **kwargs) -> Any:
    """
    Pre-cache a function call.
    
    Args:
        func: Function to call and cache
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Function result
    """
    # Get cache
    cache = get_cache()
    
    # Generate cache key
    cache_key = cache._generate_key(func.__name__, args, kwargs)
    
    # Call the function
    value = func(*args, **kwargs)
    
    # Store in cache
    cache.set(cache_key, value)
    
    logger.debug(f"Pre-cached function call: {func.__name__}")
    return value


class CacheManager:
    """
    Centralized cache manager for the dashboard.
    
    This class provides methods for storing and retrieving cached data,
    with support for automatic invalidation based on timestamps.
    """
    
    def __init__(self):
        """Initialize the cache manager."""
        self._cache = {}
        self._timestamps = {}
        self._lock = threading.RLock()
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the cache.
        
        Args:
            key: The cache key
            default: Default value to return if key doesn't exist
            
        Returns:
            Cached value or default
        """
        with self._lock:
            return self._cache.get(key, default)
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """
        Set a value in the cache with optional time-to-live.
        
        Args:
            key: The cache key
            value: Value to cache
            ttl_seconds: Time-to-live in seconds (None means no expiration)
        """
        with self._lock:
            self._cache[key] = value
            if ttl_seconds is not None:
                expiry_time = datetime.now() + timedelta(seconds=ttl_seconds)
                self._timestamps[key] = expiry_time
            else:
                self._timestamps[key] = None
    
    def invalidate(self, key: str) -> None:
        """
        Invalidate a cached value.
        
        Args:
            key: The cache key to invalidate
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._timestamps.pop(key, None)
    
    def invalidate_pattern(self, pattern: str) -> None:
        """
        Invalidate all cache keys matching a pattern.
        
        Args:
            pattern: String pattern to match (simple contains check)
        """
        with self._lock:
            keys_to_remove = [k for k in self._cache.keys() if pattern in k]
            for key in keys_to_remove:
                del self._cache[key]
                self._timestamps.pop(key, None)
    
    def invalidate_all(self) -> None:
        """Invalidate all cached values."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
    
    def is_valid(self, key: str) -> bool:
        """
        Check if a cache key is valid (exists and not expired).
        
        Args:
            key: The cache key to check
            
        Returns:
            True if valid, False otherwise
        """
        with self._lock:
            if key not in self._cache:
                return False
            
            timestamp = self._timestamps.get(key)
            if timestamp is None:
                return True  # No expiration
            
            return datetime.now() < timestamp
    
    def clean_expired(self) -> int:
        """
        Remove all expired cache entries.
        
        Returns:
            Number of entries removed
        """
        with self._lock:
            now = datetime.now()
            keys_to_remove = [
                k for k, ts in self._timestamps.items()
                if ts is not None and now > ts
            ]
            
            for key in keys_to_remove:
                del self._cache[key]
                del self._timestamps[key]
            
            return len(keys_to_remove)


# Create a global instance of the cache manager
cache_manager = CacheManager()


def cached(ttl_seconds: Optional[int] = 60, key_prefix: str = ""):
    """
    Decorator for caching function results.
    
    Args:
        ttl_seconds: Time-to-live in seconds (None means no expiration)
        key_prefix: Prefix for cache keys
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate a cache key based on function name, args, and kwargs
            cache_key = f"{key_prefix}:{func.__name__}:{hash(str(args))}-{hash(str(kwargs))}"
            
            # Check if we have a valid cached result
            if cache_manager.is_valid(cache_key):
                return cache_manager.get(cache_key)
            
            # Call the original function
            result = func(*args, **kwargs)
            
            # Cache the result
            cache_manager.set(cache_key, result, ttl_seconds)
            
            return result
        
        # Add a method to invalidate this function's cache
        def invalidate_cache(*args, **kwargs):
            if args or kwargs:
                # Invalidate specific cache entry
                cache_key = f"{key_prefix}:{func.__name__}:{hash(str(args))}-{hash(str(kwargs))}"
                cache_manager.invalidate(cache_key)
            else:
                # Invalidate all cache entries for this function
                cache_manager.invalidate_pattern(f"{key_prefix}:{func.__name__}:")
        
        wrapper.invalidate_cache = invalidate_cache
        
        return wrapper
    
    return decorator


def timed_cache(func=None, *, seconds: int = 60):
    """
    Simple time-based cache decorator.
    Can be used as @timed_cache or @timed_cache(seconds=30)
    
    Args:
        func: Function to decorate
        seconds: Cache TTL in seconds
        
    Returns:
        Decorated function
    """
    def decorator_timed_cache(func):
        # Dictionary to store cached results and timestamps
        cache_data = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create a key from the function arguments
            key = str(args) + str(kwargs)
            
            # Get the current timestamp
            now = time.time()
            
            # Check if the result is in the cache and not expired
            if key in cache_data:
                timestamp, result = cache_data[key]
                if now - timestamp < seconds:
                    return result
            
            # Call the function and cache the result
            result = func(*args, **kwargs)
            cache_data[key] = (now, result)
            
            return result
        
        # Add a method to clear the cache
        wrapper.clear_cache = lambda: cache_data.clear()
        
        return wrapper
    
    # Handle both @timed_cache and @timed_cache(seconds=30) syntax
    if func is None:
        return decorator_timed_cache
    return decorator_timed_cache(func)


def partial_update(original_data: Dict[str, Any], update_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update only changed parts of the data.
    
    Args:
        original_data: Original data dictionary
        update_data: New data dictionary
        
    Returns:
        Merged data dictionary with only updated values
    """
    result = original_data.copy()
    
    # Only update keys that have changed
    for key, value in update_data.items():
        if key not in original_data or original_data[key] != value:
            result[key] = value
    
    return result 