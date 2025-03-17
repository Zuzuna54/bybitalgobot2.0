"""
Enhanced Caching Utility

This module provides an enhanced caching system for the dashboard that includes:
- Automatic memory monitoring
- Cache entry expiration policies
- Prioritized cache eviction
- Statistics tracking
- Cache access patterns analysis
"""

import time
import threading
import functools
import inspect
import hashlib
import json
from typing import Dict, Any, List, Callable, Optional, Union, Tuple, Set
from datetime import datetime, timedelta
from loguru import logger
from collections import defaultdict

from src.dashboard.utils.memory_monitor import (
    get_memory_monitor,
    register_memory_alert_callback,
)


class CacheEntry:
    """
    Represents a single cached entry with metadata.
    """

    def __init__(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        category: str = "default",
        priority: int = 5,
        max_size: Optional[int] = None,
    ):
        """
        Initialize a cache entry.

        Args:
            key: Cache key
            value: Cached value
            ttl: Time to live in seconds (None = no expiration)
            category: Category for grouping cache entries
            priority: Priority level (1-10, higher = more important)
            max_size: Maximum size in bytes (None = no limit)
        """
        self.key = key
        self.value = value
        self.ttl = ttl
        self.category = category
        self.priority = min(max(priority, 1), 10)  # Clamp between 1-10
        self.created_at = time.time()
        self.last_accessed = self.created_at
        self.access_count = 0
        self.max_size = max_size

        # Calculate approximate size
        try:
            # Use sys.getsizeof if available
            import sys

            self._size = sys.getsizeof(value)
        except (ImportError, TypeError):
            # Fallback to string representation size
            try:
                self._size = len(str(value)) * 2  # Rough approximation
            except:
                self._size = 1024  # Default if can't determine

    @property
    def size(self) -> int:
        """Get the approximate size of the cached value in bytes."""
        return self._size

    @property
    def age(self) -> float:
        """Get the age of the entry in seconds."""
        return time.time() - self.created_at

    @property
    def time_since_access(self) -> float:
        """Get the time since last access in seconds."""
        return time.time() - self.last_accessed

    @property
    def is_expired(self) -> bool:
        """Check if the entry has expired."""
        if self.ttl is None:
            return False
        return time.time() > (self.created_at + self.ttl)

    @property
    def exceeds_max_size(self) -> bool:
        """Check if the entry exceeds its maximum size."""
        if self.max_size is None:
            return False
        return self.size > self.max_size

    def access(self) -> None:
        """Record an access to this cache entry."""
        self.last_accessed = time.time()
        self.access_count += 1


class EnhancedCache:
    """
    Enhanced caching system with memory monitoring and smart eviction policies.
    """

    def __init__(
        self,
        name: str = "default",
        max_size_mb: Optional[float] = 100,
        cleanup_interval: int = 300,
        enable_memory_monitoring: bool = True,
        default_ttl: Optional[int] = None,
        eviction_policy: str = "lru",
    ):
        """
        Initialize the enhanced cache.

        Args:
            name: Name of this cache instance
            max_size_mb: Maximum cache size in MB (None = no limit)
            cleanup_interval: Interval in seconds for cleanup tasks
            enable_memory_monitoring: Whether to enable memory monitoring
            default_ttl: Default time to live for cache entries
            eviction_policy: Policy for evicting entries when cache is full
                Options: "lru", "lfu", "fifo", "priority"
        """
        self.name = name
        self.max_size = max_size_mb * 1024 * 1024 if max_size_mb is not None else None
        self.cleanup_interval = cleanup_interval
        self.default_ttl = default_ttl
        self.eviction_policy = eviction_policy
        self._cache: Dict[str, CacheEntry] = {}
        self._categories: Dict[str, Set[str]] = defaultdict(set)
        self._current_size = 0
        self._lock = threading.RLock()
        self._last_cleanup = time.time()
        self._cleanup_thread = None
        self._should_stop = threading.Event()
        self._hit_count = 0
        self._miss_count = 0
        self._eviction_count = 0

        # Register for memory alerts if enabled
        if enable_memory_monitoring:
            register_memory_alert_callback(self._handle_memory_alert)
            logger.info(f"Cache '{name}' registered for memory alerts")

        # Start background cleanup thread
        self._start_cleanup_thread()

    def _start_cleanup_thread(self) -> None:
        """Start the background cleanup thread."""
        if self._cleanup_thread is not None and self._cleanup_thread.is_alive():
            return

        self._should_stop.clear()
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop, daemon=True, name=f"cache-cleanup-{self.name}"
        )
        self._cleanup_thread.start()
        logger.debug(f"Started cleanup thread for cache '{self.name}'")

    def _cleanup_loop(self) -> None:
        """Background thread that periodically cleans up expired entries."""
        while not self._should_stop.wait(self.cleanup_interval):
            try:
                self.cleanup()
            except Exception as e:
                logger.error(f"Error in cache cleanup: {str(e)}")

    def _handle_memory_alert(
        self, level: str, memory_info: Dict[str, Any], message: str
    ) -> None:
        """
        Handle memory alert by clearing cache entries if necessary.

        Args:
            level: Alert level ("warning" or "critical")
            memory_info: Memory information
            message: Alert message
        """
        if level == "critical":
            # On critical memory alert, aggressively clear cache
            self.clear()
            logger.warning(f"Cache '{self.name}' cleared due to critical memory alert")
        elif level == "warning":
            # On warning, clear a percentage of the cache
            self.trim(percent=50)
            logger.info(f"Cache '{self.name}' trimmed by 50% due to memory warning")

    def calculate_key(self, *args, **kwargs) -> str:
        """
        Calculate a cache key from function arguments.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            String cache key
        """
        # Convert args and kwargs to a stable string representation
        key_parts = []

        # Add positional args
        for arg in args:
            try:
                key_parts.append(str(arg))
            except:
                # If argument can't be converted to string, use its type
                key_parts.append(f"<{type(arg).__name__}>")

        # Add keyword args (sorted for consistency)
        for k in sorted(kwargs.keys()):
            try:
                key_parts.append(f"{k}={kwargs[k]}")
            except:
                # If argument can't be converted to string, use its type
                key_parts.append(f"{k}=<{type(kwargs[k]).__name__}>")

        # Join key parts and create hash
        key_string = ",".join(key_parts)
        key_hash = hashlib.md5(key_string.encode()).hexdigest()

        return key_hash

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the cache.

        Args:
            key: Cache key
            default: Default value if key not found

        Returns:
            Cached value or default
        """
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]

                # Check if entry is expired
                if entry.is_expired:
                    self._remove_entry(key)
                    self._miss_count += 1
                    return default

                # Update access metadata
                entry.access()
                self._hit_count += 1
                return entry.value

            self._miss_count += 1
            return default

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        category: str = "default",
        priority: int = 5,
        max_size: Optional[int] = None,
    ) -> bool:
        """
        Set a value in the cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (None = default TTL)
            category: Category for grouping cache entries
            priority: Priority level (1-10, higher = more important)
            max_size: Maximum size in bytes for this entry

        Returns:
            True if successful, False otherwise
        """
        # Use default TTL if not specified
        if ttl is None:
            ttl = self.default_ttl

        with self._lock:
            # Create new cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                ttl=ttl,
                category=category,
                priority=priority,
                max_size=max_size,
            )

            # Check if entry exceeds its own max size
            if entry.exceeds_max_size:
                logger.warning(f"Cache entry '{key}' exceeds its maximum size")
                return False

            # If we already have this key, remove the old entry first
            if key in self._cache:
                old_entry = self._cache[key]
                old_category = old_entry.category
                self._current_size -= old_entry.size

                # Remove from old category if changed
                if old_category != category:
                    self._categories[old_category].discard(key)

            # Check if we need to make room
            if self.max_size is not None:
                new_size = self._current_size + entry.size
                if new_size > self.max_size:
                    # Need to evict entries
                    self._evict_entries(entry.size)

            # Store the entry
            self._cache[key] = entry
            self._current_size += entry.size
            self._categories[category].add(key)

            return True

    def _evict_entries(self, required_space: int) -> None:
        """
        Evict entries to make room for new entries.

        Args:
            required_space: Amount of space needed in bytes
        """
        if not self._cache:
            return

        # If we need more space than the max size, we can't cache this item
        if self.max_size is not None and required_space > self.max_size:
            logger.warning(
                f"Required space ({required_space} bytes) exceeds maximum cache size"
            )
            return

        # Keep evicting until we have enough space
        space_to_free = required_space - (self.max_size - self._current_size)
        if space_to_free <= 0:
            return

        entries_to_evict = []

        # Order entries based on eviction policy
        if self.eviction_policy == "lru":
            # Least Recently Used
            entries = sorted(self._cache.items(), key=lambda x: x[1].last_accessed)
        elif self.eviction_policy == "lfu":
            # Least Frequently Used
            entries = sorted(self._cache.items(), key=lambda x: x[1].access_count)
        elif self.eviction_policy == "fifo":
            # First In, First Out
            entries = sorted(self._cache.items(), key=lambda x: x[1].created_at)
        elif self.eviction_policy == "priority":
            # By Priority (lowest first)
            entries = sorted(self._cache.items(), key=lambda x: x[1].priority)
        else:
            # Default to LRU
            entries = sorted(self._cache.items(), key=lambda x: x[1].last_accessed)

        # Determine which entries to evict
        freed_space = 0
        for key, entry in entries:
            if freed_space >= space_to_free:
                break

            entries_to_evict.append(key)
            freed_space += entry.size

        # Evict the entries
        for key in entries_to_evict:
            self._remove_entry(key)
            self._eviction_count += 1

        logger.debug(
            f"Evicted {len(entries_to_evict)} entries to free {freed_space} bytes"
        )

    def _remove_entry(self, key: str) -> None:
        """
        Remove a cache entry.

        Args:
            key: Cache key to remove
        """
        if key in self._cache:
            entry = self._cache[key]
            self._current_size -= entry.size
            self._categories[entry.category].discard(key)
            del self._cache[key]

    def delete(self, key: str) -> bool:
        """
        Delete a key from the cache.

        Args:
            key: Cache key to delete

        Returns:
            True if key was present, False otherwise
        """
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                return True
            return False

    def clear(self) -> None:
        """Clear the entire cache."""
        with self._lock:
            self._cache.clear()
            self._categories.clear()
            self._current_size = 0
            logger.info(f"Cache '{self.name}' cleared")

    def cleanup(self) -> int:
        """
        Remove expired entries from the cache.

        Returns:
            Number of entries removed
        """
        with self._lock:
            # Find expired entries
            expired_keys = [
                key for key, entry in self._cache.items() if entry.is_expired
            ]

            # Remove expired entries
            for key in expired_keys:
                self._remove_entry(key)

            self._last_cleanup = time.time()

            if expired_keys:
                logger.debug(
                    f"Removed {len(expired_keys)} expired entries from cache '{self.name}'"
                )

            return len(expired_keys)

    def trim(self, percent: float) -> int:
        """
        Trim the cache by removing a percentage of entries.

        Args:
            percent: Percentage to remove (0-100)

        Returns:
            Number of entries removed
        """
        with self._lock:
            if not self._cache:
                return 0

            # Calculate number of entries to remove
            entries_to_remove = int(len(self._cache) * (percent / 100.0))
            if entries_to_remove <= 0:
                return 0

            # Order entries based on policy
            if self.eviction_policy == "lru":
                entries = sorted(self._cache.items(), key=lambda x: x[1].last_accessed)
            elif self.eviction_policy == "lfu":
                entries = sorted(self._cache.items(), key=lambda x: x[1].access_count)
            else:
                entries = sorted(self._cache.items(), key=lambda x: x[1].created_at)

            # Take the first n entries to remove
            keys_to_remove = [key for key, _ in entries[:entries_to_remove]]

            # Remove the entries
            for key in keys_to_remove:
                self._remove_entry(key)

            logger.info(
                f"Trimmed {len(keys_to_remove)} entries ({percent}%) from cache '{self.name}'"
            )

            return len(keys_to_remove)

    def clear_category(self, category: str) -> int:
        """
        Clear all entries in a category.

        Args:
            category: Category name

        Returns:
            Number of entries removed
        """
        with self._lock:
            if category not in self._categories:
                return 0

            # Get keys in this category
            keys = list(self._categories[category])

            # Remove all keys in this category
            for key in keys:
                self._remove_entry(key)

            # The category set should be empty now, but clean it up anyway
            self._categories[category].clear()

            logger.info(
                f"Cleared {len(keys)} entries from category '{category}' in cache '{self.name}'"
            )

            return len(keys)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache.

        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            stats = {
                "name": self.name,
                "entries": len(self._cache),
                "size_bytes": self._current_size,
                "size_mb": self._current_size / (1024 * 1024),
                "max_size_mb": (
                    self.max_size / (1024 * 1024) if self.max_size is not None else None
                ),
                "percent_full": (
                    (self._current_size / self.max_size * 100) if self.max_size else 0
                ),
                "categories": {
                    category: len(keys) for category, keys in self._categories.items()
                },
                "hits": self._hit_count,
                "misses": self._miss_count,
                "hit_ratio": (
                    (self._hit_count / (self._hit_count + self._miss_count))
                    if (self._hit_count + self._miss_count) > 0
                    else 0
                ),
                "evictions": self._eviction_count,
                "last_cleanup": self._last_cleanup,
                "time_since_cleanup": time.time() - self._last_cleanup,
            }

            # Get some stats on entry ages
            if self._cache:
                ages = [entry.age for entry in self._cache.values()]
                stats["entry_age_avg"] = sum(ages) / len(ages)
                stats["entry_age_min"] = min(ages)
                stats["entry_age_max"] = max(ages)

                # Get access count stats
                access_counts = [entry.access_count for entry in self._cache.values()]
                stats["access_count_avg"] = sum(access_counts) / len(access_counts)
                stats["access_count_min"] = min(access_counts)
                stats["access_count_max"] = max(access_counts)

            return stats

    def close(self) -> None:
        """Clean up resources and stop background threads."""
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._should_stop.set()
            self._cleanup_thread.join(timeout=1.0)
            if self._cleanup_thread.is_alive():
                logger.warning(
                    f"Cache cleanup thread for '{self.name}' did not terminate cleanly"
                )


# Global cache registry
_cache_instances: Dict[str, EnhancedCache] = {}


def get_cache(
    name: str = "default",
    max_size_mb: Optional[float] = 100,
    cleanup_interval: int = 300,
    default_ttl: Optional[int] = None,
    eviction_policy: str = "lru",
) -> EnhancedCache:
    """
    Get or create a cache instance.

    Args:
        name: Cache name
        max_size_mb: Maximum cache size in MB
        cleanup_interval: Cleanup interval in seconds
        default_ttl: Default TTL for cache entries
        eviction_policy: Cache eviction policy

    Returns:
        Cache instance
    """
    global _cache_instances

    if name not in _cache_instances:
        _cache_instances[name] = EnhancedCache(
            name=name,
            max_size_mb=max_size_mb,
            cleanup_interval=cleanup_interval,
            default_ttl=default_ttl,
            eviction_policy=eviction_policy,
        )

    return _cache_instances[name]


def clear_all_caches() -> None:
    """Clear all cache instances."""
    global _cache_instances

    for cache in _cache_instances.values():
        cache.clear()

    logger.info(f"Cleared all caches ({len(_cache_instances)} instances)")


def cache(
    ttl: Optional[int] = None,
    category: str = "default",
    priority: int = 5,
    cache_name: str = "default",
) -> Callable:
    """
    Cache decorator for functions.

    Args:
        ttl: Time to live in seconds
        category: Cache category
        priority: Cache priority (1-10)
        cache_name: Name of cache to use

    Returns:
        Decorated function
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Get the cache instance
            cache_instance = get_cache(cache_name)

            # Create a key based on the function name and arguments
            prefix = f"{func.__module__}.{func.__qualname__}"
            key = f"{prefix}:{cache_instance.calculate_key(*args, **kwargs)}"

            # Try to get from cache
            result = cache_instance.get(key)
            if result is not None:
                return result

            # Call the function
            result = func(*args, **kwargs)

            # Cache the result
            cache_instance.set(
                key=key, value=result, ttl=ttl, category=category, priority=priority
            )

            return result

        return wrapper

    return decorator


def invalidate_cache(
    func: Callable, *args, cache_name: str = "default", **kwargs
) -> bool:
    """
    Invalidate cached results for a function.

    Args:
        func: Function whose cache to invalidate
        *args: Function arguments
        cache_name: Name of cache to use
        **kwargs: Function keyword arguments

    Returns:
        True if cache was invalidated, False otherwise
    """
    # Get the cache instance
    cache_instance = get_cache(cache_name)

    # Create a key based on the function name and arguments
    prefix = f"{func.__module__}.{func.__qualname__}"

    if args or kwargs:
        # Invalidate specific key
        key = f"{prefix}:{cache_instance.calculate_key(*args, **kwargs)}"
        return cache_instance.delete(key)
    else:
        # Invalidate all keys for this function
        # This is implemented by clearing all keys with the function prefix
        keys_to_remove = [
            key for key in cache_instance._cache.keys() if key.startswith(prefix)
        ]

        for key in keys_to_remove:
            cache_instance.delete(key)

        return len(keys_to_remove) > 0


def invalidate_category(category: str, cache_name: str = "default") -> int:
    """
    Invalidate all cached items in a category.

    Args:
        category: Category to invalidate
        cache_name: Name of cache to use

    Returns:
        Number of items invalidated
    """
    cache_instance = get_cache(cache_name)
    return cache_instance.clear_category(category)
