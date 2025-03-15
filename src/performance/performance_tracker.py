"""
Performance Tracking for the Algorithmic Trading System

This module provides functionality for tracking, analyzing, and reporting 
trading performance metrics. It has been refactored to improve maintainability
and now re-exports components from the performance components package.
"""

# Re-export the main components
from src.performance.components.tracker import PerformanceTracker
from src.performance.components.metrics_calculator import PerformanceMetrics

# Define public API
__all__ = ['PerformanceTracker', 'PerformanceMetrics'] 