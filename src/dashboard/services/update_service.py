"""
Dashboard Update Service Module

This module manages real-time updates for the dashboard components.
It coordinates the timing and frequency of data refreshes.
"""

from typing import Dict, Any, Callable, Optional, List
import threading
import time
from datetime import datetime, timedelta
from loguru import logger

from src.dashboard.utils.time_utils import is_update_due, get_next_update_time


class UpdateService:
    """Service for managing real-time updates to the dashboard."""
    
    def __init__(self, data_service=None):
        """
        Initialize the update service.
        
        Args:
            data_service: The data service providing data for updates
        """
        self.data_service = data_service
        
        # Update intervals in seconds
        self.update_intervals = {
            "performance": 30,  # 30 seconds
            "trades": 5,        # 5 seconds
            "orderbook": 1,     # 1 second
            "strategy": 10,     # 10 seconds
            "market": 5,        # 5 seconds
            "system": 2         # 2 seconds
        }
        
        # Last update times
        self.last_updates = {
            "performance": None,
            "trades": None,
            "orderbook": None,
            "strategy": None,
            "market": None,
            "system": None
        }
        
        # Update callbacks
        self.update_callbacks: Dict[str, List[Callable]] = {
            "performance": [],
            "trades": [],
            "orderbook": [],
            "strategy": [],
            "market": [],
            "system": [],
            "all": []  # Callbacks to execute on any update
        }
        
        # Update thread
        self.update_thread = None
        self.stop_flag = threading.Event()
        self.is_running = False
    
    def start(self) -> None:
        """Start the update service."""
        if self.is_running:
            logger.warning("Update service is already running")
            return
        
        logger.info("Starting dashboard update service")
        self.stop_flag.clear()
        self.is_running = True
        
        # Start update thread
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
    
    def stop(self) -> None:
        """Stop the update service."""
        if not self.is_running:
            logger.warning("Update service is not running")
            return
        
        logger.info("Stopping dashboard update service")
        self.stop_flag.set()
        
        # Wait for thread to stop
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=2.0)
            
        self.is_running = False
    
    def register_callback(self, data_type: str, callback: Callable) -> None:
        """
        Register a callback for data updates.
        
        Args:
            data_type: Type of data to monitor ("performance", "trades", etc.)
            callback: Function to call when data is updated
        """
        if data_type not in self.update_callbacks:
            logger.warning(f"Unknown data type for callback: {data_type}")
            return
        
        self.update_callbacks[data_type].append(callback)
        logger.debug(f"Registered callback for {data_type} updates")
    
    def unregister_callback(self, data_type: str, callback: Callable) -> None:
        """
        Unregister a callback for data updates.
        
        Args:
            data_type: Type of data monitored
            callback: Function previously registered
        """
        if data_type not in self.update_callbacks:
            logger.warning(f"Unknown data type for callback: {data_type}")
            return
        
        if callback in self.update_callbacks[data_type]:
            self.update_callbacks[data_type].remove(callback)
            logger.debug(f"Unregistered callback for {data_type} updates")
    
    def trigger_update(self, data_type: str) -> None:
        """
        Manually trigger an update for a specific data type.
        
        Args:
            data_type: Type of data to update
        """
        if data_type not in self.update_intervals:
            logger.warning(f"Unknown data type for update: {data_type}")
            return
        
        self._process_update(data_type)
    
    def _update_loop(self) -> None:
        """Main update loop that checks for and processes updates."""
        while not self.stop_flag.is_set():
            try:
                # Check each data type for update need
                for data_type in self.update_intervals.keys():
                    self._check_and_update(data_type)
                
                # Sleep briefly
                time.sleep(0.1)
            
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                time.sleep(1.0)  # Longer sleep on error
    
    def _check_and_update(self, data_type: str) -> None:
        """
        Check if an update is needed for a data type and process it.
        
        Args:
            data_type: Type of data to check
        """
        # Skip if data service is not available
        if self.data_service is None:
            return
        
        # Get last update time
        last_update = self.last_updates[data_type]
        if last_update is None:
            # First update
            self._process_update(data_type)
            return
        
        # Check if update interval has elapsed
        interval = self.update_intervals[data_type]
        if is_update_due(last_update, interval):
            self._process_update(data_type)
    
    def _process_update(self, data_type: str) -> None:
        """
        Process an update for a specific data type.
        
        Args:
            data_type: Type of data to update
        """
        # Record update time
        self.last_updates[data_type] = datetime.now()
        
        # Call specific update methods based on data type
        try:
            # Notify callbacks
            self._notify_callbacks(data_type)
            
            # Notify "all" callbacks
            self._notify_callbacks("all")
            
        except Exception as e:
            logger.error(f"Error processing {data_type} update: {e}")
    
    def _notify_callbacks(self, data_type: str) -> None:
        """
        Notify callbacks about a data update.
        
        Args:
            data_type: Type of data that was updated
        """
        callbacks = self.update_callbacks.get(data_type, [])
        
        for callback in callbacks:
            try:
                callback(data_type)
            except Exception as e:
                logger.error(f"Error in {data_type} update callback: {e}")
    
    def get_last_update_time(self, data_type: str) -> Optional[datetime]:
        """
        Get the time of the last update for a data type.
        
        Args:
            data_type: Type of data
            
        Returns:
            Datetime of the last update, or None if no update has occurred
        """
        return self.last_updates.get(data_type)
    
    def get_next_update_time(self, data_type: str) -> Optional[datetime]:
        """
        Get the time of the next scheduled update for a data type.
        
        Args:
            data_type: Type of data
            
        Returns:
            Datetime of the next scheduled update, or None if no update has occurred
        """
        last_update = self.last_updates.get(data_type)
        if last_update is None:
            return datetime.now()  # Immediate update needed
        
        interval = self.update_intervals.get(data_type, 60)
        return get_next_update_time(last_update, interval)
    
    def is_update_due(self, data_type: str) -> bool:
        """
        Check if an update is due for a data type.
        
        Args:
            data_type: Type of data
            
        Returns:
            True if an update is due, False otherwise
        """
        last_update = self.last_updates.get(data_type)
        interval = self.update_intervals.get(data_type, 60)
        return is_update_due(last_update, interval) 