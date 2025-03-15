"""
Performance Metrics Calculator for the Algorithmic Trading System

This module provides functionality for calculating various trading performance metrics
including profit/loss statistics, risk metrics, and strategy comparisons.
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd
import numpy as np


@dataclass
class PerformanceMetrics:
    """Performance metrics for trading system."""
    
    # General metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    break_even_trades: int = 0
    win_rate: float = 0.0
    
    # Profit and loss metrics
    total_profit_loss: float = 0.0
    total_profit: float = 0.0
    total_loss: float = 0.0
    profit_factor: float = 0.0
    average_profit: float = 0.0
    average_loss: float = 0.0
    largest_profit: float = 0.0
    largest_loss: float = 0.0
    
    # Return metrics
    return_percent: float = 0.0
    annualized_return: float = 0.0
    daily_return: float = 0.0
    
    # Risk metrics
    max_drawdown: float = 0.0
    max_drawdown_percent: float = 0.0
    risk_reward_ratio: float = 0.0
    sharpe_ratio: float = 0.0
    
    # Duration metrics
    average_trade_duration_hours: float = 0.0
    max_trade_duration_hours: float = 0.0
    min_trade_duration_hours: float = 0.0
    
    # Strategy specific
    strategy_performance: Dict[str, Dict[str, float]] = None
    
    # Time metrics
    best_day: Optional[str] = None
    worst_day: Optional[str] = None
    equity_curve: List[float] = None
    equity_timestamps: List[str] = None
    
    def __post_init__(self):
        """Initialize default values for complex types."""
        if self.strategy_performance is None:
            self.strategy_performance = {}
        if self.equity_curve is None:
            self.equity_curve = []
        if self.equity_timestamps is None:
            self.equity_timestamps = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceMetrics':
        """Create metrics from dictionary."""
        return cls(**data)


def calculate_metrics(
    completed_trades: List[Dict[str, Any]],
    active_trades: Dict[str, Dict[str, Any]],
    daily_pnl: Dict[str, float],
    initial_balance: float,
    current_balance: float,
    equity_curve: List[float],
    equity_timestamps: List[str]
) -> PerformanceMetrics:
    """
    Calculate performance metrics from trade data.
    
    Args:
        completed_trades: List of completed trades
        active_trades: Dictionary of active trades by ID
        daily_pnl: Dictionary of daily PnL values by date
        initial_balance: Initial account balance
        current_balance: Current account balance
        equity_curve: List of equity values
        equity_timestamps: List of timestamps for equity values
        
    Returns:
        PerformanceMetrics object with calculated metrics
    """
    metrics = PerformanceMetrics()
    
    # Copy equity data
    metrics.equity_curve = equity_curve.copy()
    metrics.equity_timestamps = equity_timestamps.copy()
    
    if not completed_trades:
        return metrics
    
    # Convert trades to DataFrame for analysis
    df = pd.DataFrame(completed_trades)
    
    # General metrics
    total_trades = len(df)
    winning_trades = len(df[df["realized_pnl"] > 0])
    losing_trades = len(df[df["realized_pnl"] < 0])
    break_even_trades = total_trades - winning_trades - losing_trades
    win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
    
    # Profit and loss metrics
    total_profit_loss = df["realized_pnl"].sum()
    total_profit = df[df["realized_pnl"] > 0]["realized_pnl"].sum()
    total_loss = abs(df[df["realized_pnl"] < 0]["realized_pnl"].sum())
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
    
    average_profit = df[df["realized_pnl"] > 0]["realized_pnl"].mean() if winning_trades > 0 else 0.0
    average_loss = df[df["realized_pnl"] < 0]["realized_pnl"].mean() if losing_trades > 0 else 0.0
    largest_profit = df["realized_pnl"].max()
    largest_loss = df["realized_pnl"].min()
    
    # Return metrics
    return_percent = (total_profit_loss / initial_balance) * 100
    
    # Duration metrics
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    df["duration"] = (df["exit_time"] - df["entry_time"])
    
    average_duration = df["duration"].mean()
    average_trade_duration_hours = average_duration.total_seconds() / 3600 if not pd.isna(average_duration) else 0.0
    
    max_duration = df["duration"].max()
    max_trade_duration_hours = max_duration.total_seconds() / 3600 if not pd.isna(max_duration) else 0.0
    
    min_duration = df["duration"].min()
    min_trade_duration_hours = min_duration.total_seconds() / 3600 if not pd.isna(min_duration) else 0.0
    
    # Calculate daily returns for risk metrics
    df["exit_date"] = df["exit_time"].dt.date
    daily_returns = df.groupby("exit_date")["realized_pnl"].sum()
    
    # Calculate drawdown
    if len(equity_curve) > 0:
        equity_series = pd.Series(equity_curve)
        rolling_max = equity_series.cummax()
        drawdown_series = (rolling_max - equity_series) / rolling_max * 100
        max_drawdown_pct = drawdown_series.max()
        max_drawdown = (rolling_max - equity_series).max()
    else:
        max_drawdown_pct = 0.0
        max_drawdown = 0.0
    
    # Calculate Sharpe ratio if we have enough data
    if len(daily_returns) > 1:
        daily_return_values = daily_returns.values
        mean_return = np.mean(daily_return_values)
        std_return = np.std(daily_return_values, ddof=1)
        sharpe_ratio = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0.0
        
        daily_return_avg = mean_return
        annualized_return = ((1 + (mean_return / current_balance)) ** 252 - 1) * 100
    else:
        sharpe_ratio = 0.0
        daily_return_avg = 0.0
        annualized_return = 0.0
    
    # Strategy performance
    strategy_performance = {}
    
    if "strategy_name" in df.columns:
        strategy_stats = df.groupby("strategy_name").agg({
            "realized_pnl": ["count", "sum", "mean"],
            "realized_pnl_percent": ["mean", "sum"]
        })
        
        for strategy, stats in strategy_stats.iterrows():
            strategy_count = stats[("realized_pnl", "count")]
            strategy_wins = len(df[(df["strategy_name"] == strategy) & (df["realized_pnl"] > 0)])
            
            strategy_performance[strategy] = {
                "count": int(strategy_count),
                "wins": int(strategy_wins),
                "win_rate": float(strategy_wins / strategy_count if strategy_count > 0 else 0.0),
                "total_pnl": float(stats[("realized_pnl", "sum")]),
                "avg_pnl": float(stats[("realized_pnl", "mean")]),
                "avg_pnl_percent": float(stats[("realized_pnl_percent", "mean")]),
            }
    
    # Time-based metrics
    if daily_pnl:
        best_day = max(daily_pnl.items(), key=lambda x: x[1])[0]
        worst_day = min(daily_pnl.items(), key=lambda x: x[1])[0]
    else:
        best_day = None
        worst_day = None
    
    # Risk-reward ratio
    risk_reward_ratio = abs(average_profit / average_loss) if average_loss != 0 else float('inf')
    
    # Update metrics
    metrics.total_trades = total_trades
    metrics.winning_trades = winning_trades
    metrics.losing_trades = losing_trades
    metrics.break_even_trades = break_even_trades
    metrics.win_rate = win_rate
    
    metrics.total_profit_loss = total_profit_loss
    metrics.total_profit = total_profit
    metrics.total_loss = total_loss
    metrics.profit_factor = profit_factor
    metrics.average_profit = average_profit
    metrics.average_loss = average_loss
    metrics.largest_profit = largest_profit
    metrics.largest_loss = largest_loss
    
    metrics.return_percent = return_percent
    metrics.annualized_return = annualized_return
    metrics.daily_return = daily_return_avg
    
    metrics.max_drawdown = max_drawdown
    metrics.max_drawdown_percent = max_drawdown_pct
    metrics.risk_reward_ratio = risk_reward_ratio
    metrics.sharpe_ratio = sharpe_ratio
    
    metrics.average_trade_duration_hours = average_trade_duration_hours
    metrics.max_trade_duration_hours = max_trade_duration_hours
    metrics.min_trade_duration_hours = min_trade_duration_hours
    
    metrics.strategy_performance = strategy_performance
    
    metrics.best_day = best_day
    metrics.worst_day = worst_day
    
    return metrics


def calculate_daily_summary(completed_trades: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Calculate daily performance summary.
    
    Args:
        completed_trades: List of completed trades
        
    Returns:
        DataFrame with daily performance metrics
    """
    if not completed_trades:
        return pd.DataFrame()
    
    df = pd.DataFrame(completed_trades)
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    daily_summary = df.groupby(df["exit_time"].dt.date).agg({
        "id": "count",
        "realized_pnl": ["sum", "mean", "min", "max"],
        "realized_pnl_percent": ["mean", "min", "max"]
    })
    
    daily_summary.columns = [f"{col[0]}_{col[1]}" for col in daily_summary.columns]
    daily_summary = daily_summary.rename(columns={"id_count": "trade_count"})
    
    # Calculate win rate per day
    def win_rate(x):
        wins = (x > 0).sum()
        return wins / len(x) if len(x) > 0 else 0
    
    daily_win_rate = df.groupby(df["exit_time"].dt.date)["realized_pnl"].apply(win_rate)
    daily_summary["win_rate"] = daily_win_rate
    
    # Calculate cumulative equity
    daily_summary["cumulative_pnl"] = daily_summary["realized_pnl_sum"].cumsum()
    
    return daily_summary


def calculate_strategy_comparison(trades_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Calculate comparative performance between strategies.
    
    Args:
        trades_df: DataFrame with trade data
        
    Returns:
        Dictionary with strategy comparisons
    """
    if "strategy_name" not in trades_df.columns or trades_df.empty:
        return {}
    
    # Group by strategy
    strategy_stats = trades_df.groupby("strategy_name").agg({
        "realized_pnl": ["count", "sum", "mean", "std"],
        "realized_pnl_percent": ["mean", "std"]
    })
    
    # Calculate win rate per strategy
    strategy_wins = {}
    for strategy in trades_df["strategy_name"].unique():
        strategy_trades = trades_df[trades_df["strategy_name"] == strategy]
        wins = len(strategy_trades[strategy_trades["realized_pnl"] > 0])
        total = len(strategy_trades)
        win_rate = wins / total if total > 0 else 0
        strategy_wins[strategy] = {
            "wins": wins,
            "total": total,
            "win_rate": win_rate
        }
    
    # Build comparison dictionary
    comparison = {}
    for strategy, row in strategy_stats.iterrows():
        count = int(row[("realized_pnl", "count")])
        total_pnl = float(row[("realized_pnl", "sum")])
        avg_pnl = float(row[("realized_pnl", "mean")])
        std_pnl = float(row[("realized_pnl", "std")])
        avg_pnl_pct = float(row[("realized_pnl_percent", "mean")])
        
        # Calculate Sharpe ratio (if we have enough trades and standard deviation > 0)
        sharpe = 0.0
        if count > 1 and std_pnl > 0:
            sharpe = avg_pnl / std_pnl * np.sqrt(252)  # Annualized
        
        comparison[strategy] = {
            "count": count,
            "win_rate": strategy_wins[strategy]["win_rate"],
            "total_pnl": total_pnl,
            "avg_pnl": avg_pnl,
            "avg_pnl_percent": avg_pnl_pct,
            "sharpe": sharpe
        }
    
    return comparison 