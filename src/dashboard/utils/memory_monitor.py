"""
Memory Monitoring Utility

This module provides utilities for monitoring memory usage
to optimize caching and prevent memory leaks in the dashboard.
"""

import os
import sys
import time
import threading
import gc
import psutil
import tracemalloc
from datetime import datetime
from typing import Dict, Any, List, Callable, Optional, Union, Tuple
from loguru import logger


class MemoryMonitor:
    """
    Monitor memory usage of the dashboard application.

    This class provides utilities to track memory usage over time,
    detect memory spikes, and provide alerts when memory usage
    exceeds configured thresholds.
    """

    def __init__(
        self,
        warning_threshold_mb: float = 500,
        critical_threshold_mb: float = 1000,
        monitor_interval_seconds: float = 30.0,
        enable_tracemalloc: bool = False,
        snapshot_count: int = 5,
    ):
        """
        Initialize the memory monitor.

        Args:
            warning_threshold_mb: Memory threshold for warnings (MB)
            critical_threshold_mb: Memory threshold for critical alerts (MB)
            monitor_interval_seconds: Interval between memory checks
            enable_tracemalloc: Whether to enable tracemalloc for detailed memory analysis
            snapshot_count: Number of memory snapshots to keep
        """
        self.warning_threshold = warning_threshold_mb * 1024 * 1024  # Convert to bytes
        self.critical_threshold = (
            critical_threshold_mb * 1024 * 1024
        )  # Convert to bytes
        self.monitor_interval = monitor_interval_seconds
        self.process = psutil.Process(os.getpid())
        self.enable_tracemalloc = enable_tracemalloc
        self.snapshot_count = snapshot_count
        self.snapshots = []
        self.memory_history = []
        self.monitor_thread = None
        self.shutdown_flag = threading.Event()
        self.alert_callbacks = []

        # Initialize tracemalloc if enabled
        if self.enable_tracemalloc:
            tracemalloc.start()
            logger.info("Tracemalloc enabled for memory tracking")

    def start_monitoring(self) -> None:
        """Start the memory monitoring thread."""
        if self.monitor_thread is not None and self.monitor_thread.is_alive():
            logger.warning("Memory monitor is already running")
            return

        self.shutdown_flag.clear()
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True, name="memory-monitor"
        )
        self.monitor_thread.start()
        logger.info("Memory monitoring started")

    def stop_monitoring(self) -> None:
        """Stop the memory monitoring thread."""
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            logger.warning("Memory monitor is not running")
            return

        self.shutdown_flag.set()
        self.monitor_thread.join(timeout=5.0)
        if self.monitor_thread.is_alive():
            logger.warning("Memory monitor thread did not terminate cleanly")
        else:
            logger.info("Memory monitoring stopped")

        # Stop tracemalloc if it was enabled
        if self.enable_tracemalloc and tracemalloc.is_tracing():
            tracemalloc.stop()

    def _monitoring_loop(self) -> None:
        """Internal monitoring loop that runs in a separate thread."""
        while not self.shutdown_flag.is_set():
            try:
                # Measure current memory usage
                memory_info = self.get_memory_info()

                # Store in history
                if len(self.memory_history) >= 100:
                    self.memory_history.pop(0)  # Keep history bounded
                self.memory_history.append(memory_info)

                # Check thresholds and trigger alerts if necessary
                self._check_thresholds(memory_info)

                # Take tracemalloc snapshot if enabled
                if self.enable_tracemalloc and tracemalloc.is_tracing():
                    self._take_snapshot()

                # Wait for the next check interval
                self.shutdown_flag.wait(self.monitor_interval)
            except Exception as e:
                logger.error(f"Error in memory monitoring loop: {str(e)}")
                # Reduce check frequency on error
                self.shutdown_flag.wait(self.monitor_interval * 2)

    def get_memory_info(self) -> Dict[str, Any]:
        """
        Get current memory usage information.

        Returns:
            Dictionary with memory usage details
        """
        # Get process memory info
        mem_info = self.process.memory_info()

        # Get system memory info
        system_mem = psutil.virtual_memory()

        # Calculate percentage of system memory used by this process
        process_percent = (mem_info.rss / system_mem.total) * 100

        return {
            "timestamp": datetime.now().isoformat(),
            "rss": mem_info.rss,  # Resident Set Size
            "vms": mem_info.vms,  # Virtual Memory Size
            "percent": process_percent,
            "system_percent": system_mem.percent,
            "system_available": system_mem.available,
            "system_total": system_mem.total,
        }

    def _check_thresholds(self, memory_info: Dict[str, Any]) -> None:
        """
        Check if memory usage exceeds thresholds and trigger alerts.

        Args:
            memory_info: Memory information dictionary
        """
        rss = memory_info["rss"]

        if rss > self.critical_threshold:
            message = f"CRITICAL: Memory usage ({rss / (1024*1024):.1f} MB) exceeds critical threshold"
            logger.critical(message)
            self._trigger_alerts("critical", memory_info, message)
            # Force garbage collection on critical threshold
            gc.collect()
        elif rss > self.warning_threshold:
            message = f"WARNING: Memory usage ({rss / (1024*1024):.1f} MB) exceeds warning threshold"
            logger.warning(message)
            self._trigger_alerts("warning", memory_info, message)

    def _trigger_alerts(
        self, level: str, memory_info: Dict[str, Any], message: str
    ) -> None:
        """
        Trigger alert callbacks.

        Args:
            level: Alert level ("warning" or "critical")
            memory_info: Memory information dictionary
            message: Alert message
        """
        for callback in self.alert_callbacks:
            try:
                callback(level, memory_info, message)
            except Exception as e:
                logger.error(f"Error in memory alert callback: {str(e)}")

    def register_alert_callback(
        self, callback: Callable[[str, Dict[str, Any], str], None]
    ) -> None:
        """
        Register a callback function to be called on memory alerts.

        Args:
            callback: Function taking (level, memory_info, message) as arguments
        """
        if callback not in self.alert_callbacks:
            self.alert_callbacks.append(callback)
            logger.debug(f"Registered memory alert callback: {callback.__name__}")

    def _take_snapshot(self) -> None:
        """Take a tracemalloc snapshot and store it."""
        snapshot = tracemalloc.take_snapshot()
        timestamp = datetime.now().isoformat()

        # Keep only the specified number of snapshots
        if len(self.snapshots) >= self.snapshot_count:
            self.snapshots.pop(0)

        self.snapshots.append((timestamp, snapshot))

    def get_memory_growth(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Analyze memory growth between the first and last snapshots.

        Args:
            top_n: Number of top memory-consuming items to return

        Returns:
            List of dictionaries with memory growth information
        """
        if not self.enable_tracemalloc or len(self.snapshots) < 2:
            return []

        # Get the first and last snapshots
        first_timestamp, first_snapshot = self.snapshots[0]
        last_timestamp, last_snapshot = self.snapshots[-1]

        # Compare snapshots to see memory growth
        stats = last_snapshot.compare_to(first_snapshot, "lineno")

        # Format the results
        results = []
        for stat in stats[:top_n]:
            results.append(
                {
                    "file": stat.traceback[0].filename,
                    "line": stat.traceback[0].lineno,
                    "size_diff": stat.size_diff,
                    "count_diff": stat.count_diff,
                    "size": stat.size,
                    "count": stat.count,
                }
            )

        return results

    def get_detailed_object_stats(self) -> Dict[str, Any]:
        """
        Get detailed statistics about Python objects in memory.

        Returns:
            Dictionary with object counts by type
        """
        # Force garbage collection before counting
        gc.collect()

        # Count objects by type
        objects = {}
        for obj in gc.get_objects():
            obj_type = type(obj).__name__
            if obj_type in objects:
                objects[obj_type] += 1
            else:
                objects[obj_type] = 1

        # Sort by count (descending)
        sorted_objects = dict(sorted(objects.items(), key=lambda x: x[1], reverse=True))

        return sorted_objects

    def get_memory_trend(self) -> Dict[str, Any]:
        """
        Analyze the memory usage trend.

        Returns:
            Dictionary with trend information
        """
        if len(self.memory_history) < 2:
            return {
                "trend": "unknown",
                "rate": 0.0,
                "history_points": len(self.memory_history),
            }

        # Calculate trend over the last 10 points (or all if less)
        points = min(10, len(self.memory_history))
        recent_history = self.memory_history[-points:]

        start_memory = recent_history[0]["rss"]
        end_memory = recent_history[-1]["rss"]

        # Calculate rate of change in MB per minute
        time_diff = datetime.fromisoformat(
            recent_history[-1]["timestamp"]
        ) - datetime.fromisoformat(recent_history[0]["timestamp"])
        minutes = time_diff.total_seconds() / 60.0

        if minutes > 0:
            rate = ((end_memory - start_memory) / (1024 * 1024)) / minutes
        else:
            rate = 0.0

        # Determine trend
        if rate > 2.0:  # More than 2MB per minute
            trend = "increasing_rapidly"
        elif rate > 0.5:
            trend = "increasing"
        elif rate < -2.0:
            trend = "decreasing_rapidly"
        elif rate < -0.5:
            trend = "decreasing"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "rate": rate,
            "rate_mb_per_minute": rate,
            "history_points": len(self.memory_history),
            "current_mb": end_memory / (1024 * 1024),
            "start_mb": start_memory / (1024 * 1024),
        }


# Singleton instance
_memory_monitor = None


def get_memory_monitor() -> MemoryMonitor:
    """
    Get the singleton memory monitor instance.

    Returns:
        Memory monitor instance
    """
    global _memory_monitor
    if _memory_monitor is None:
        _memory_monitor = MemoryMonitor()
    return _memory_monitor


def start_memory_monitoring(
    warning_threshold_mb: float = 500, critical_threshold_mb: float = 1000
) -> None:
    """
    Start monitoring memory usage.

    Args:
        warning_threshold_mb: Memory threshold for warnings (MB)
        critical_threshold_mb: Memory threshold for critical alerts (MB)
    """
    monitor = get_memory_monitor()
    monitor.warning_threshold = warning_threshold_mb * 1024 * 1024
    monitor.critical_threshold = critical_threshold_mb * 1024 * 1024
    monitor.start_monitoring()


def stop_memory_monitoring() -> None:
    """Stop memory monitoring."""
    monitor = get_memory_monitor()
    monitor.stop_monitoring()


def get_current_memory_usage() -> Dict[str, Any]:
    """
    Get current memory usage information.

    Returns:
        Dictionary with memory usage details
    """
    monitor = get_memory_monitor()
    return monitor.get_memory_info()


def register_memory_alert_callback(
    callback: Callable[[str, Dict[str, Any], str], None],
) -> None:
    """
    Register a callback for memory alerts.

    Args:
        callback: Function to call on memory alerts
    """
    monitor = get_memory_monitor()
    monitor.register_alert_callback(callback)
