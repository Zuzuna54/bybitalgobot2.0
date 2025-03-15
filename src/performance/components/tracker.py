"""
Performance Tracker for the Algorithmic Trading System

This module contains the main PerformanceTracker class that handles tracking,
analyzing, and reporting trading performance metrics.
"""

from typing import Dict, Any, List, Optional, Union
import os
import json
import pandas as pd
from datetime import datetime

from src.performance.components.metrics_calculator import (
    PerformanceMetrics, 
    calculate_metrics,
    calculate_daily_summary,
    calculate_strategy_comparison
)
from src.performance.components.report_generator import (
    generate_performance_summary,
    generate_full_performance_report,
    save_performance_report
)
from src.performance.components.visualization import generate_performance_dashboard
from src.performance.components.storage import (
    save_metrics_to_json,
    load_metrics_from_json,
    save_trades_to_csv,
    load_trades_from_csv,
    save_daily_summary_to_csv,
    save_equity_curve_to_csv,
    create_timestamped_filename
)


class PerformanceTracker:
    """
    Tracks, analyzes, and reports trading performance metrics.
    """
    
    def __init__(self, initial_balance: float = 10000.0):
        """
        Initialize the performance tracker.
        
        Args:
            initial_balance: Initial account balance
        """
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.metrics = PerformanceMetrics()
        
        self.completed_trades = []
        self.active_trades = []
        self.daily_pnl = {}
        self.equity_curve = [initial_balance]
        self.equity_timestamps = [datetime.now().isoformat()]
        self.equity_history = [
            {
                "timestamp": datetime.now().isoformat(),
                "balance": initial_balance,
                "open_positions_value": 0.0,
                "total_equity": initial_balance
            }
        ]
    
    def update_balance(self, new_balance: float, 
                       open_positions_value: float = 0.0) -> None:
        """
        Update the current account balance and equity history.
        
        Args:
            new_balance: Current account balance
            open_positions_value: Current value of open positions
        """
        self.current_balance = new_balance
        timestamp = datetime.now().isoformat()
        total_equity = new_balance + open_positions_value
        
        # Update equity history
        self.equity_curve.append(total_equity)
        self.equity_timestamps.append(timestamp)
        
        # Add equity history point
        self.equity_history.append({
            "timestamp": timestamp,
            "balance": new_balance,
            "open_positions_value": open_positions_value,
            "total_equity": total_equity
        })
        
        # Recalculate metrics
        self._calculate_metrics()
    
    def add_trade(self, trade: Dict[str, Any]) -> None:
        """
        Add a trade to the performance tracker.
        
        Args:
            trade: Trade data dictionary
        """
        trade_status = trade.get("status", "active")
        
        # Handle completed trades
        if trade_status == "completed":
            # Check if this trade is already in completed_trades
            existing_ids = [t.get("trade_id") for t in self.completed_trades]
            if trade.get("trade_id") not in existing_ids:
                self.completed_trades.append(trade)
            
            # Remove from active trades if present
            active_ids = [t.get("trade_id") for t in self.active_trades]
            if trade.get("trade_id") in active_ids:
                self.active_trades = [
                    t for t in self.active_trades 
                    if t.get("trade_id") != trade.get("trade_id")
                ]
            
            # Update daily PnL
            exit_date = trade.get("exit_time", "")
            if isinstance(exit_date, str) and len(exit_date) >= 10:
                date_str = exit_date[:10]  # Get YYYY-MM-DD part
                pnl = trade.get("realized_pnl", 0.0)
                
                if date_str in self.daily_pnl:
                    self.daily_pnl[date_str] += pnl
                else:
                    self.daily_pnl[date_str] = pnl
        
        # Handle active trades
        elif trade_status == "active":
            # Check if this trade is already in active_trades
            existing_ids = [t.get("trade_id") for t in self.active_trades]
            if trade.get("trade_id") not in existing_ids:
                self.active_trades.append(trade)
        
        # Recalculate metrics
        self._calculate_metrics()
    
    def _calculate_metrics(self) -> None:
        """
        Calculate performance metrics from trades.
        """
        self.metrics = calculate_metrics(
            completed_trades=self.completed_trades,
            active_trades=self.active_trades,
            daily_pnl=self.daily_pnl,
            initial_balance=self.initial_balance,
            current_balance=self.current_balance,
            equity_curve=self.equity_curve,
            equity_timestamps=self.equity_timestamps
        )
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """
        Get the current performance metrics.
        
        Returns:
            Current performance metrics
        """
        self._calculate_metrics()
        return self.metrics
    
    def get_daily_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a summary of daily performance.
        
        Returns:
            Dictionary with daily performance data
        """
        return calculate_daily_summary(self.completed_trades)
    
    def get_equity_curve_data(self) -> List[Dict[str, Any]]:
        """
        Get equity curve data.
        
        Returns:
            List of equity history points
        """
        return self.equity_history
    
    def get_strategy_comparison(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a comparison of strategy performance.
        
        Returns:
            Dictionary with strategy performance comparison
        """
        return calculate_strategy_comparison(self.completed_trades)
    
    def generate_summary(self) -> str:
        """
        Generate a human-readable summary of trading performance.
        
        Returns:
            Performance summary string
        """
        return generate_performance_summary(
            self.metrics,
            self.completed_trades,
            self.daily_pnl
        )
    
    def generate_report(self, include_trades: bool = True) -> Dict[str, Any]:
        """
        Generate a full performance report.
        
        Args:
            include_trades: Whether to include trade lists in the report
            
        Returns:
            Dictionary containing the performance report
        """
        return generate_full_performance_report(
            self.metrics,
            self.daily_pnl,
            len(self.active_trades),
            len(self.completed_trades),
            include_trades=include_trades,
            completed_trades=self.completed_trades if include_trades else None,
            active_trades=self.active_trades if include_trades else None
        )
    
    def save_report(self, 
                    output_directory: str,
                    generate_charts: bool = True,
                    base_filename: str = "performance") -> Dict[str, str]:
        """
        Save the performance report to files.
        
        Args:
            output_directory: Directory to save the report
            generate_charts: Whether to generate performance charts
            base_filename: Base filename for report files
            
        Returns:
            Dictionary with paths to generated files
        """
        # Create timestamped filename
        ts_filename = create_timestamped_filename(base_filename)
        
        # Ensure output directory exists
        os.makedirs(output_directory, exist_ok=True)
        
        # Dictionary to store file paths
        saved_files = {}
        
        # Save performance report
        saved_files.update(
            save_performance_report(
                self.metrics,
                self.completed_trades,
                self.get_daily_summary(),
                self.equity_history,
                output_directory,
                ts_filename
            )
        )
        
        # Generate charts if requested
        if generate_charts:
            chart_files = generate_performance_dashboard(
                self.metrics,
                self.completed_trades,
                self.equity_history,
                self.daily_pnl,
                output_directory,
                ts_filename
            )
            saved_files.update(chart_files)
        
        return saved_files
    
    def reset(self) -> None:
        """
        Reset the performance tracker to its initial state.
        """
        self.__init__(self.initial_balance)
    
    def load_from_files(self,
                       metrics_file: str,
                       trades_file: str,
                       equity_file: Optional[str] = None) -> bool:
        """
        Load performance data from files.
        
        Args:
            metrics_file: Path to metrics JSON file
            trades_file: Path to trades CSV file
            equity_file: Path to equity curve CSV file (optional)
            
        Returns:
            True if data was loaded successfully, False otherwise
        """
        try:
            # Load metrics
            self.metrics = load_metrics_from_json(metrics_file)
            
            # Load trades
            trades = load_trades_from_csv(trades_file)
            
            # Separate completed and active trades
            self.completed_trades = [t for t in trades if t.get("status") == "completed"]
            self.active_trades = [t for t in trades if t.get("status") == "active"]
            
            # Rebuild daily PnL
            self.daily_pnl = {}
            for trade in self.completed_trades:
                exit_date = trade.get("exit_time", "")
                if isinstance(exit_date, str) and len(exit_date) >= 10:
                    date_str = exit_date[:10]  # Get YYYY-MM-DD part
                    pnl = trade.get("realized_pnl", 0.0)
                    
                    if date_str in self.daily_pnl:
                        self.daily_pnl[date_str] += pnl
                    else:
                        self.daily_pnl[date_str] = pnl
            
            # Set balance from metrics
            self.initial_balance = self.metrics.initial_balance
            self.current_balance = self.metrics.current_balance
            
            # Load equity curve if provided
            if equity_file and os.path.exists(equity_file):
                from src.performance.components.storage import load_equity_curve_from_csv
                self.equity_history = load_equity_curve_from_csv(equity_file)
                
                # Extract equity curve and timestamps from history
                self.equity_curve = [point.get("total_equity") for point in self.equity_history]
                self.equity_timestamps = [point.get("timestamp") for point in self.equity_history]
            
            return True
            
        except Exception as e:
            print(f"Error loading performance data: {str(e)}")
            return False 