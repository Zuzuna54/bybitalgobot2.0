"""
Results processing functionality for the backtesting engine.
"""

import os
import json
from typing import Dict, Any, List
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

from src.performance.performance_tracker import PerformanceTracker


def generate_results(
    trades: List[Dict[str, Any]],
    equity_curve: List[Dict[str, Any]],
    initial_balance: float,
    current_balance: float,
    performance_tracker: PerformanceTracker,
    strategy_performance: Dict[str, Dict[str, Any]],
    commission_rate: float,
    slippage: float
) -> Dict[str, Any]:
    """
    Generate backtest results summary.
    
    Args:
        trades: List of all trades
        equity_curve: Equity curve data
        initial_balance: Initial account balance
        current_balance: Final account balance
        performance_tracker: Performance tracker instance
        strategy_performance: Strategy performance data
        commission_rate: Commission rate used
        slippage: Slippage amount used
        
    Returns:
        Dictionary with backtest results
    """
    # Get performance metrics
    metrics = performance_tracker.get_metrics()
    
    # Calculate additional metrics
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df = trades_df[trades_df["status"] == "closed"]
        
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df["net_pnl"] > 0])
        losing_trades = len(trades_df[trades_df["net_pnl"] < 0])
        
        if total_trades > 0:
            win_rate = winning_trades / total_trades
            avg_profit = trades_df[trades_df["net_pnl"] > 0]["net_pnl"].mean() if winning_trades > 0 else 0
            avg_loss = trades_df[trades_df["net_pnl"] < 0]["net_pnl"].mean() if losing_trades > 0 else 0
            profit_factor = (
                trades_df[trades_df["net_pnl"] > 0]["net_pnl"].sum() / 
                abs(trades_df[trades_df["net_pnl"] < 0]["net_pnl"].sum())
            ) if losing_trades > 0 and abs(trades_df[trades_df["net_pnl"] < 0]["net_pnl"].sum()) > 0 else float('inf')
        else:
            win_rate = 0
            avg_profit = 0
            avg_loss = 0
            profit_factor = 0
    else:
        total_trades = 0
        winning_trades = 0
        losing_trades = 0
        win_rate = 0
        avg_profit = 0
        avg_loss = 0
        profit_factor = 0
    
    # Calculate drawdown
    if equity_curve:
        equity_df = pd.DataFrame(equity_curve)
        equity_df["equity"] = pd.to_numeric(equity_df["equity"])
        rolling_max = equity_df["equity"].cummax()
        drawdown_series = (rolling_max - equity_df["equity"]) / rolling_max * 100
        max_drawdown = drawdown_series.max()
    else:
        max_drawdown = 0
    
    # Strategy metrics
    filtered_strategy_performance = {
        name: performance for name, performance in strategy_performance.items()
        if performance.get("signals_executed", 0) > 0
    }
    
    # Compile results
    results = {
        "summary": {
            "initial_balance": initial_balance,
            "final_balance": current_balance,
            "total_return": ((current_balance / initial_balance) - 1) * 100,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate * 100,
            "profit_factor": profit_factor,
            "average_profit": avg_profit,
            "average_loss": avg_loss,
            "max_drawdown": max_drawdown,
            "commission_rate": commission_rate,
            "slippage": slippage
        },
        "performance_metrics": metrics,
        "strategy_metrics": filtered_strategy_performance,
        "trades": trades,
        "equity_curve": equity_curve
    }
    
    return results


def save_results(results: Dict[str, Any], output_dir: str) -> str:
    """
    Save backtest results to files.
    
    Args:
        results: Backtest results dictionary
        output_dir: Directory to save results
        
    Returns:
        Path to saved results
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_path = os.path.join(output_dir, f"backtest_{timestamp}")
    
    # Create directory
    os.makedirs(base_path, exist_ok=True)
    
    # Save summary
    with open(os.path.join(base_path, "summary.json"), "w") as f:
        json.dump(results["summary"], f, indent=2, default=str)
    
    # Save performance metrics
    with open(os.path.join(base_path, "performance_metrics.json"), "w") as f:
        json.dump(results["performance_metrics"], f, indent=2, default=str)
    
    # Save strategy metrics
    with open(os.path.join(base_path, "strategy_metrics.json"), "w") as f:
        json.dump(results["strategy_metrics"], f, indent=2, default=str)
    
    # Save trades
    pd.DataFrame(results["trades"]).to_csv(os.path.join(base_path, "trades.csv"), index=False)
    
    # Save equity curve
    pd.DataFrame(results["equity_curve"]).to_csv(os.path.join(base_path, "equity_curve.csv"), index=False)
    
    # Generate charts
    generate_charts(base_path, results)
    
    logger.info(f"Backtest results saved to {base_path}")
    return base_path


def generate_charts(base_path: str, results: Dict[str, Any]) -> None:
    """
    Generate charts from backtest results.
    
    Args:
        base_path: Base path to save charts
        results: Backtest results dictionary
    """
    charts_dir = os.path.join(base_path, "charts")
    os.makedirs(charts_dir, exist_ok=True)
    
    # Set up plotting style
    plt.style.use('seaborn-darkgrid')
    
    # Equity curve
    if results["equity_curve"]:
        plt.figure(figsize=(12, 6))
        
        equity_df = pd.DataFrame(results["equity_curve"])
        equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"])
        
        plt.plot(equity_df["timestamp"], equity_df["equity"], label="Equity", linewidth=2)
        plt.plot(equity_df["timestamp"], equity_df["balance"], label="Balance", linewidth=1, alpha=0.7)
        
        # Add drawdown as a secondary axis
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax2.fill_between(
            equity_df["timestamp"], 
            0, 
            equity_df["drawdown_pct"], 
            alpha=0.3, 
            color='red', 
            label="Drawdown %"
        )
        ax2.set_ylim(bottom=0, top=max(equity_df["drawdown_pct"]) * 1.5 if max(equity_df["drawdown_pct"]) > 0 else 10)
        ax2.invert_yaxis()
        
        plt.title("Equity Curve")
        plt.xlabel("Date")
        ax1.set_ylabel("Equity")
        ax2.set_ylabel("Drawdown %")
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, "equity_curve.png"), dpi=300)
        plt.close()
    
    # Trade outcomes
    if results["trades"]:
        trades_df = pd.DataFrame(results["trades"])
        trades_df = trades_df[trades_df["status"] == "closed"]
        
        if not trades_df.empty:
            plt.figure(figsize=(10, 6))
            
            plt.hist(trades_df["net_pnl"], bins=20, alpha=0.7, color="skyblue", edgecolor="black")
            plt.axvline(x=0, color="red", linestyle="--")
            
            plt.title("Trade Profit/Loss Distribution")
            plt.xlabel("Profit/Loss")
            plt.ylabel("Frequency")
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(charts_dir, "trade_distribution.png"), dpi=300)
            plt.close()
    
    # Strategy performance comparison
    if results["strategy_metrics"]:
        strategies = list(results["strategy_metrics"].keys())
        win_rates = [results["strategy_metrics"][s]["win_rate"] * 100 for s in strategies]
        total_pnls = [results["strategy_metrics"][s]["total_profit_loss"] for s in strategies]
        
        plt.figure(figsize=(12, 6))
        
        ax1 = plt.subplot(1, 2, 1)
        bars = ax1.bar(strategies, win_rates, alpha=0.7)
        
        # Color bars
        for i, bar in enumerate(bars):
            if win_rates[i] >= 50:
                bar.set_color("green")
            else:
                bar.set_color("red")
        
        ax1.set_title("Strategy Win Rates")
        ax1.set_xlabel("Strategy")
        ax1.set_ylabel("Win Rate (%)")
        ax1.set_ylim(0, max(win_rates) * 1.2 if max(win_rates) > 0 else 100)
        plt.xticks(rotation=45, ha="right")
        
        ax2 = plt.subplot(1, 2, 2)
        bars = ax2.bar(strategies, total_pnls, alpha=0.7)
        
        # Color bars
        for i, bar in enumerate(bars):
            if total_pnls[i] >= 0:
                bar.set_color("green")
            else:
                bar.set_color("red")
        
        ax2.set_title("Strategy Total P&L")
        ax2.set_xlabel("Strategy")
        ax2.set_ylabel("Profit/Loss")
        plt.xticks(rotation=45, ha="right")
        
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, "strategy_comparison.png"), dpi=300)
        plt.close()
    
    # Monthly returns
    if results["equity_curve"]:
        equity_df = pd.DataFrame(results["equity_curve"])
        equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"])
        
        # Resample to monthly returns
        equity_df = equity_df.set_index("timestamp")
        monthly_equity = equity_df["equity"].resample("M").last()
        monthly_returns = monthly_equity.pct_change().dropna() * 100
        
        if len(monthly_returns) > 0:
            plt.figure(figsize=(12, 6))
            
            bars = plt.bar(monthly_returns.index, monthly_returns.values, alpha=0.7)
            
            # Color bars
            for i, bar in enumerate(bars):
                if monthly_returns.iloc[i] >= 0:
                    bar.set_color("green")
                else:
                    bar.set_color("red")
            
            plt.title("Monthly Returns")
            plt.xlabel("Month")
            plt.ylabel("Return (%)")
            plt.grid(True, alpha=0.3, axis="y")
            
            plt.tight_layout()
            plt.savefig(os.path.join(charts_dir, "monthly_returns.png"), dpi=300)
            plt.close() 