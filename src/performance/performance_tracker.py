"""
Performance Tracker Module

This module provides functionality for tracking, analyzing, and reporting
trading system performance metrics.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import defaultdict


class PerformanceTracker:
    """
    Tracks and analyzes trading performance metrics.

    The PerformanceTracker handles:
    - Recording and analyzing trade history
    - Tracking account balance and equity
    - Calculating key performance metrics
    - Generating performance reports
    - Persisting performance data
    """

    def __init__(
        self,
        initial_balance: float = 10000.0,
        data_directory: str = "data/performance",
        reporting_currency: str = "USD",
    ):
        """
        Initialize performance tracker.

        Args:
            initial_balance: Initial account balance
            data_directory: Directory for saving performance data
            reporting_currency: Currency symbol for reports
        """
        # Create data directory if it doesn't exist
        os.makedirs(data_directory, exist_ok=True)

        # Configuration
        self.initial_balance = initial_balance
        self.data_directory = data_directory
        self.reporting_currency = reporting_currency

        # Performance data
        self.current_balance = initial_balance
        self.peak_balance = initial_balance
        self.trades = []
        self.monthly_returns = defaultdict(dict)

        # Time series data
        self.equity_history = []
        self.drawdown_history = []
        self.unrealized_pnl = 0.0

        # Add initial equity point
        self._add_equity_point(
            datetime.now(), self.current_balance, self.unrealized_pnl
        )

    def add_trade(self, trade: Dict[str, Any]) -> None:
        """
        Add a completed trade to the performance history.

        Args:
            trade: Trade dictionary with details
        """
        try:
            # Ensure the trade has all required fields
            required_fields = [
                "symbol",
                "entry_time",
                "exit_time",
                "entry_price",
                "exit_price",
                "quantity",
                "realized_pnl",
            ]

            for field in required_fields:
                if field not in trade:
                    raise ValueError(f"Trade missing required field: {field}")

            # Store the trade
            self.trades.append(trade)

            # Update the balance with realized PnL
            if "realized_pnl" in trade:
                self.current_balance += trade["realized_pnl"]

                # Update peak balance if needed
                if self.current_balance > self.peak_balance:
                    self.peak_balance = self.current_balance

            # Record equity point
            exit_time = trade.get("exit_time")
            if exit_time:
                # Convert string to datetime if needed
                if isinstance(exit_time, str):
                    exit_time = datetime.fromisoformat(exit_time.replace("Z", "+00:00"))

                self._add_equity_point(
                    exit_time, self.current_balance, self.unrealized_pnl
                )

            # Update monthly returns
            self._update_monthly_returns(trade)

            logger.debug(
                f"Added trade: {trade['symbol']} with PnL: {trade.get('realized_pnl', 0)}"
            )

        except Exception as e:
            logger.error(f"Error adding trade: {e}")

    def update_balance(self, balance: float, unrealized_pnl: float = 0.0) -> None:
        """
        Update the current account balance and unrealized PnL.

        Args:
            balance: Current account balance
            unrealized_pnl: Current unrealized profit/loss
        """
        self.current_balance = balance
        self.unrealized_pnl = unrealized_pnl

        # Update peak balance if needed
        total_equity = balance + unrealized_pnl
        if total_equity > self.peak_balance:
            self.peak_balance = total_equity

        # Record equity point
        self._add_equity_point(datetime.now(), balance, unrealized_pnl)

    def _add_equity_point(
        self, timestamp: datetime, balance: float, unrealized_pnl: float
    ) -> None:
        """
        Add a point to the equity curve and calculate drawdown.

        Args:
            timestamp: Time of the equity point
            balance: Current account balance
            unrealized_pnl: Current unrealized profit/loss
        """
        # Calculate total equity
        total_equity = balance + unrealized_pnl

        # Calculate drawdown
        drawdown = 0.0
        if self.peak_balance > 0:
            drawdown = (total_equity - self.peak_balance) / self.peak_balance

        # Add to histories
        self.equity_history.append(
            {
                "timestamp": timestamp,
                "balance": balance,
                "unrealized_pnl": unrealized_pnl,
                "equity": total_equity,
            }
        )

        self.drawdown_history.append({"timestamp": timestamp, "drawdown": drawdown})

    def _update_monthly_returns(self, trade: Dict[str, Any]) -> None:
        """
        Update monthly returns based on a completed trade.

        Args:
            trade: Completed trade
        """
        if "exit_time" not in trade or "realized_pnl" not in trade:
            return

        # Get the trade exit month
        exit_time = trade["exit_time"]
        if isinstance(exit_time, str):
            exit_time = datetime.fromisoformat(exit_time.replace("Z", "+00:00"))

        year = exit_time.year
        month = exit_time.month

        # Initialize if needed
        if month not in self.monthly_returns[year]:
            self.monthly_returns[year][month] = 0.0

        # Add the PnL to the monthly return
        self.monthly_returns[year][month] += trade["realized_pnl"]

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate performance metrics.

        Returns:
            Dictionary of performance metrics
        """
        # Only calculate metrics if we have trades
        if not self.trades:
            return self._empty_metrics()

        # Convert trades to DataFrame for analysis
        df_trades = pd.DataFrame(self.trades)

        # Calculate basic metrics
        total_trades = len(df_trades)
        winning_trades = len(df_trades[df_trades["realized_pnl"] > 0])
        losing_trades = len(df_trades[df_trades["realized_pnl"] < 0])

        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        # Calculate returns
        total_pnl = df_trades["realized_pnl"].sum()
        total_return = (
            total_pnl / self.initial_balance if self.initial_balance > 0 else 0.0
        )

        # Calculate average trade metrics
        avg_win = (
            df_trades[df_trades["realized_pnl"] > 0]["realized_pnl"].mean()
            if winning_trades > 0
            else 0.0
        )
        avg_loss = (
            df_trades[df_trades["realized_pnl"] < 0]["realized_pnl"].mean()
            if losing_trades > 0
            else 0.0
        )
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

        # Calculate drawdown
        max_drawdown = (
            min([x["drawdown"] for x in self.drawdown_history])
            if self.drawdown_history
            else 0.0
        )

        # Calculate more advanced metrics if we have equity history
        if len(self.equity_history) > 1:
            # Convert equity history to DataFrame
            df_equity = pd.DataFrame(self.equity_history)
            df_equity["timestamp"] = pd.to_datetime(df_equity["timestamp"])
            df_equity.set_index("timestamp", inplace=True)

            # Resample to daily and calculate daily returns
            daily_equity = (
                df_equity["equity"].resample("D").last().fillna(method="ffill")
            )
            daily_returns = daily_equity.pct_change().dropna()

            # Calculate volatility and risk-adjusted returns
            volatility = daily_returns.std() * (252**0.5)  # Annualized
            sharpe_ratio = (
                (daily_returns.mean() * 252) / volatility if volatility > 0 else 0.0
            )

            # Calculate downside deviation for Sortino ratio
            downside_returns = daily_returns[daily_returns < 0]
            downside_deviation = (
                downside_returns.std() * (252**0.5)
                if len(downside_returns) > 0
                else 0.0
            )
            sortino_ratio = (
                (daily_returns.mean() * 252) / downside_deviation
                if downside_deviation > 0
                else 0.0
            )

            # Calculate Calmar ratio
            calmar_ratio = (
                (total_return / abs(max_drawdown)) if max_drawdown != 0 else 0.0
            )

            # Trading days calculation
            trading_days = (df_equity.index[-1] - df_equity.index[0]).days
            annualized_return = (
                ((1 + total_return) ** (365 / max(1, trading_days))) - 1
                if trading_days > 0
                else 0.0
            )
        else:
            volatility = 0.0
            sharpe_ratio = 0.0
            sortino_ratio = 0.0
            calmar_ratio = 0.0
            annualized_return = 0.0

        # Profit factor
        gross_profits = df_trades[df_trades["realized_pnl"] > 0]["realized_pnl"].sum()
        gross_losses = abs(
            df_trades[df_trades["realized_pnl"] < 0]["realized_pnl"].sum()
        )
        profit_factor = (
            gross_profits / gross_losses if gross_losses > 0 else float("inf")
        )

        # Expectancy
        expectancy = (
            (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
            if total_trades > 0
            else 0.0
        )

        # Recovery factor
        recovery_factor = (
            total_pnl / abs(max_drawdown * self.initial_balance)
            if max_drawdown != 0
            else 0.0
        )

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar_ratio,
            "profit_factor": profit_factor,
            "average_win": avg_win,
            "average_loss": avg_loss,
            "win_loss_ratio": win_loss_ratio,
            "expectancy": expectancy,
            "recovery_factor": recovery_factor,
        }

    def _empty_metrics(self) -> Dict[str, Any]:
        """
        Return empty metrics when no trades are available.

        Returns:
            Dictionary with zero values for all metrics
        """
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "total_return": 0.0,
            "annualized_return": 0.0,
            "volatility": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
            "profit_factor": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "win_loss_ratio": 0.0,
            "expectancy": 0.0,
            "recovery_factor": 0.0,
        }

    def get_equity_history(self) -> pd.DataFrame:
        """
        Get equity history as a DataFrame.

        Returns:
            DataFrame with equity history
        """
        if not self.equity_history:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(
                columns=["timestamp", "balance", "unrealized_pnl", "equity"]
            )

        df = pd.DataFrame(self.equity_history)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        return df

    def get_drawdown_history(self) -> pd.DataFrame:
        """
        Get drawdown history as a DataFrame.

        Returns:
            DataFrame with drawdown history
        """
        if not self.drawdown_history:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=["timestamp", "drawdown"])

        df = pd.DataFrame(self.drawdown_history)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        return df

    def get_monthly_returns(self) -> pd.DataFrame:
        """
        Get monthly returns as a DataFrame.

        Returns:
            DataFrame with monthly returns
        """
        # Convert monthly returns to DataFrame
        data = []
        for year, months in self.monthly_returns.items():
            for month, value in months.items():
                # Calculate percentage return relative to initial balance
                pct_return = (
                    value / self.initial_balance if self.initial_balance > 0 else 0.0
                )
                data.append((year, month, pct_return))

        if not data:
            return pd.DataFrame(columns=["year", "month", "return"])

        df = pd.DataFrame(data, columns=["year", "month", "return"])
        df.set_index(["year", "month"], inplace=True)
        return df

    def save_performance_report(self, filename: str) -> str:
        """
        Save performance report to file.

        Args:
            filename: Base filename (without extension)

        Returns:
            Path to saved report file
        """
        # Create data directory if it doesn't exist
        os.makedirs(self.data_directory, exist_ok=True)

        # Generate report data
        report = self.generate_performance_summary()

        # Save to JSON file
        report_path = os.path.join(self.data_directory, f"{filename}.json")
        try:
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Performance report saved to {report_path}")

            # Generate and save charts
            self._generate_performance_charts(
                os.path.join(self.data_directory, filename)
            )

            return report_path
        except Exception as e:
            logger.error(f"Error saving performance report: {e}")
            return ""

    def generate_performance_summary(self) -> Dict[str, Any]:
        """
        Generate a performance summary report.

        Returns:
            Dictionary with performance summary
        """
        # Get performance metrics
        metrics = self.get_performance_metrics()

        # Basic trade statistics
        trade_stats = {
            "total_trades": metrics["total_trades"],
            "winning_trades": metrics["winning_trades"],
            "losing_trades": metrics["losing_trades"],
            "win_rate": metrics["win_rate"],
        }

        # Return metrics
        return_metrics = {
            "initial_balance": self.initial_balance,
            "current_balance": self.current_balance,
            "total_pnl": metrics["total_pnl"],
            "total_return": metrics["total_return"],
            "annualized_return": metrics["annualized_return"],
        }

        # Risk metrics
        risk_metrics = {
            "max_drawdown": metrics["max_drawdown"],
            "volatility": metrics["volatility"],
            "sharpe_ratio": metrics["sharpe_ratio"],
            "sortino_ratio": metrics["sortino_ratio"],
            "calmar_ratio": metrics["calmar_ratio"],
        }

        # Trade quality metrics
        trade_quality = {
            "average_win": metrics["average_win"],
            "average_loss": metrics["average_loss"],
            "win_loss_ratio": metrics["win_loss_ratio"],
            "profit_factor": metrics["profit_factor"],
            "expectancy": metrics["expectancy"],
        }

        # Generate timestamp
        generated_at = datetime.now().isoformat()

        # Return full report
        return {
            "generated_at": generated_at,
            "reporting_currency": self.reporting_currency,
            "trade_statistics": trade_stats,
            "return_metrics": return_metrics,
            "risk_metrics": risk_metrics,
            "trade_quality": trade_quality,
        }

    def generate_full_performance_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive performance report with all data.

        Returns:
            Dictionary with full performance report
        """
        # Get performance summary
        summary = self.generate_performance_summary()

        # Add equity history
        equity_df = self.get_equity_history()
        equity_history = []
        for timestamp, row in equity_df.iterrows():
            equity_history.append(
                {
                    "timestamp": timestamp.isoformat(),
                    "balance": float(row["balance"]),
                    "unrealized_pnl": float(row["unrealized_pnl"]),
                    "equity": float(row["equity"]),
                }
            )

        # Add drawdown history
        drawdown_df = self.get_drawdown_history()
        drawdown_history = []
        for timestamp, row in drawdown_df.iterrows():
            drawdown_history.append(
                {
                    "timestamp": timestamp.isoformat(),
                    "drawdown": float(row["drawdown"]),
                }
            )

        # Add monthly returns
        monthly_returns_df = self.get_monthly_returns()
        monthly_returns = {}
        for (year, month), value in monthly_returns_df.iterrows():
            if str(year) not in monthly_returns:
                monthly_returns[str(year)] = {}
            monthly_returns[str(year)][str(month)] = float(value["return"])

        # Add trade history (with cleanup for JSON serialization)
        trades = []
        for trade in self.trades:
            clean_trade = {}
            for key, value in trade.items():
                # Convert datetime objects to strings
                if isinstance(value, datetime):
                    clean_trade[key] = value.isoformat()
                else:
                    clean_trade[key] = value
            trades.append(clean_trade)

        # Combine everything into full report
        full_report = {
            "summary": summary,
            "equity_history": equity_history,
            "drawdown_history": drawdown_history,
            "monthly_returns": monthly_returns,
            "trades": trades,
        }

        return full_report

    def _generate_performance_charts(self, base_path: str) -> None:
        """
        Generate performance charts and save as image files.

        Args:
            base_path: Base path for saving charts (without extension)
        """
        try:
            # Get data
            equity_df = self.get_equity_history()
            drawdown_df = self.get_drawdown_history()

            if len(equity_df) < 2:
                logger.warning("Not enough data points to generate charts")
                return

            # Set plot style
            plt.style.use("seaborn-v0_8-darkgrid")

            # Figure 1: Equity Curve
            plt.figure(figsize=(12, 6))
            plt.plot(
                equity_df.index,
                equity_df["equity"],
                label="Equity",
                color="#1f77b4",
                linewidth=2,
            )
            plt.title("Equity Curve", fontsize=16)
            plt.xlabel("Date", fontsize=12)
            plt.ylabel(f"Equity ({self.reporting_currency})", fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{base_path}_equity.png", dpi=100)
            plt.close()

            # Figure 2: Drawdown Chart
            plt.figure(figsize=(12, 6))
            plt.fill_between(
                drawdown_df.index,
                0,
                drawdown_df["drawdown"] * 100,  # Convert to percentage
                color="#d62728",
                alpha=0.4,
            )
            plt.plot(
                drawdown_df.index,
                drawdown_df["drawdown"] * 100,  # Convert to percentage
                color="#d62728",
                linewidth=1,
            )
            plt.title("Drawdown Chart", fontsize=16)
            plt.xlabel("Date", fontsize=12)
            plt.ylabel("Drawdown (%)", fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{base_path}_drawdown.png", dpi=100)
            plt.close()

            # Figure 3: Monthly Returns Heatmap
            monthly_returns_df = self.get_monthly_returns()
            if not monthly_returns_df.empty:
                # Pivot the data to create a year vs month matrix
                monthly_pivot = pd.pivot_table(
                    monthly_returns_df.reset_index(),
                    values="return",
                    index="year",
                    columns="month",
                )

                # Plot heatmap
                plt.figure(figsize=(12, 8))
                cmap = (
                    plt.cm.RdYlGn
                )  # Red for negative, yellow for neutral, green for positive
                ax = plt.pcolormesh(
                    monthly_pivot.columns,
                    monthly_pivot.index,
                    monthly_pivot.values,
                    cmap=cmap,
                )

                # Add colorbar
                cbar = plt.colorbar(ax)
                cbar.set_label("Monthly Return (%)", rotation=270, labelpad=20)

                # Add text annotations
                for i in range(len(monthly_pivot.index)):
                    for j in range(len(monthly_pivot.columns)):
                        if not np.isnan(monthly_pivot.values[i, j]):
                            value = (
                                monthly_pivot.values[i, j] * 100
                            )  # Convert to percentage
                            text_color = "white" if abs(value) > 5 else "black"
                            plt.text(
                                j + 0.5,
                                i + 0.5,
                                f"{value:.1f}%",
                                ha="center",
                                va="center",
                                color=text_color,
                            )

                # Customize the plot
                plt.title("Monthly Returns Heatmap", fontsize=16)
                plt.xlabel("Month", fontsize=12)
                plt.ylabel("Year", fontsize=12)
                plt.xticks(
                    np.arange(1, 13) + 0.5,
                    [
                        "Jan",
                        "Feb",
                        "Mar",
                        "Apr",
                        "May",
                        "Jun",
                        "Jul",
                        "Aug",
                        "Sep",
                        "Oct",
                        "Nov",
                        "Dec",
                    ],
                )
                plt.yticks(
                    np.arange(len(monthly_pivot.index)) + 0.5, monthly_pivot.index
                )
                plt.tight_layout()
                plt.savefig(f"{base_path}_monthly_returns.png", dpi=100)
                plt.close()

        except Exception as e:
            logger.error(f"Error generating performance charts: {e}")

    # Aliases for backtesting compatibility
    def get_metrics(self) -> Dict[str, Any]:
        """Alias for get_performance_metrics() for compatibility with backtesting."""
        return self.get_performance_metrics()
