"""
Performance Data Storage for the Algorithmic Trading System

This module provides functionality for saving and loading performance data,
including metrics, trade history, and equity curves.
"""

from typing import Dict, Any, List, Optional
import os
import json
import csv
import pandas as pd
from datetime import datetime

from src.performance.components.metrics_calculator import PerformanceMetrics


def save_metrics_to_json(
    metrics: PerformanceMetrics,
    file_path: str
) -> None:
    """
    Save performance metrics to a JSON file.
    
    Args:
        metrics: Performance metrics object
        file_path: Path to save the metrics file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Convert metrics to a dictionary
    metrics_dict = metrics.to_dict()
    
    # Save metrics to JSON file
    with open(file_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)


def load_metrics_from_json(file_path: str) -> PerformanceMetrics:
    """
    Load performance metrics from a JSON file.
    
    Args:
        file_path: Path to the metrics JSON file
        
    Returns:
        PerformanceMetrics object
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Metrics file not found: {file_path}")
    
    with open(file_path, "r") as f:
        metrics_dict = json.load(f)
    
    # Convert dictionary to metrics object
    return PerformanceMetrics.from_dict(metrics_dict)


def save_trades_to_csv(
    trades: List[Dict[str, Any]],
    file_path: str
) -> None:
    """
    Save trade history to a CSV file.
    
    Args:
        trades: List of trade dictionaries
        file_path: Path to save the trades file
    """
    if not trades:
        return
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Convert trades to DataFrame
    df = pd.DataFrame(trades)
    
    # Ensure datetime columns are properly formatted
    for col in ["entry_time", "exit_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col]).dt.strftime("%Y-%m-%d %H:%M:%S")
    
    # Save DataFrame to CSV
    df.to_csv(file_path, index=False)


def load_trades_from_csv(file_path: str) -> List[Dict[str, Any]]:
    """
    Load trade history from a CSV file.
    
    Args:
        file_path: Path to the trades CSV file
        
    Returns:
        List of trade dictionaries
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Trades file not found: {file_path}")
    
    # Load DataFrame from CSV
    df = pd.read_csv(file_path)
    
    # Convert datetime columns
    for col in ["entry_time", "exit_time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    
    # Convert DataFrame to list of dictionaries
    return df.to_dict(orient="records")


def save_daily_summary_to_csv(
    daily_summary: Dict[str, Dict[str, Any]],
    file_path: str
) -> None:
    """
    Save daily performance summary to a CSV file.
    
    Args:
        daily_summary: Dictionary of daily summaries
        file_path: Path to save the daily summary file
    """
    if not daily_summary:
        return
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Convert daily summary to DataFrame
    records = []
    for date, summary in daily_summary.items():
        record = {"date": date}
        record.update(summary)
        records.append(record)
    
    df = pd.DataFrame(records)
    
    # Sort by date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
    
    # Save DataFrame to CSV
    df.to_csv(file_path, index=False)


def load_daily_summary_from_csv(file_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load daily performance summary from a CSV file.
    
    Args:
        file_path: Path to the daily summary CSV file
        
    Returns:
        Dictionary of daily summaries
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Daily summary file not found: {file_path}")
    
    # Load DataFrame from CSV
    df = pd.read_csv(file_path)
    
    # Convert DataFrame to dictionary
    daily_summary = {}
    for _, row in df.iterrows():
        date = row["date"]
        summary = row.drop("date").to_dict()
        daily_summary[date] = summary
    
    return daily_summary


def save_equity_curve_to_csv(
    equity_history: List[Dict[str, Any]],
    file_path: str
) -> None:
    """
    Save equity curve data to a CSV file.
    
    Args:
        equity_history: List of equity history points
        file_path: Path to save the equity curve file
    """
    if not equity_history:
        return
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Convert equity history to DataFrame
    df = pd.DataFrame(equity_history)
    
    # Ensure timestamp is properly formatted
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    
    # Save DataFrame to CSV
    df.to_csv(file_path, index=False)


def load_equity_curve_from_csv(file_path: str) -> List[Dict[str, Any]]:
    """
    Load equity curve data from a CSV file.
    
    Args:
        file_path: Path to the equity curve CSV file
        
    Returns:
        List of equity history points
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Equity curve file not found: {file_path}")
    
    # Load DataFrame from CSV
    df = pd.read_csv(file_path)
    
    # Convert timestamp column
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Convert DataFrame to list of dictionaries
    return df.to_dict(orient="records")


def create_timestamped_filename(base_filename: str) -> str:
    """
    Create a timestamped filename for performance reports.
    
    Args:
        base_filename: Base filename
        
    Returns:
        Timestamped filename
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_filename}_{timestamp}" 