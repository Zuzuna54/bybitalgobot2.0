"""
Data Transformation Utilities

This module provides utilities for standardizing data transformations
across the dashboard. It centralizes common data transformations to
reduce code duplication and ensure consistency.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from loguru import logger

from src.dashboard.utils.cache import cached


class DataTransformer:
    """
    Centralized data transformation utilities for the dashboard.
    
    This class provides standardized methods for transforming data
    formats, calculating metrics, and preparing data for visualization.
    """
    
    @staticmethod
    @cached(ttl_seconds=300, key_prefix="transform_equity_data")
    def transform_equity_data(equity_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Transform equity data for dashboard visualizations.
        
        Args:
            equity_data: DataFrame with equity history
            
        Returns:
            Dictionary with transformed equity metrics
        """
        if equity_data is None or equity_data.empty:
            return {
                "return_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "drawdown_duration_days": 0,
                "volatility": 0.0,
                "sharpe_ratio": 0.0,
                "equity_curve": pd.DataFrame(),
                "drawdown_curve": pd.DataFrame(),
                "rolling_returns": pd.DataFrame()
            }
        
        try:
            # Calculate return percentage
            first_equity = equity_data["equity"].iloc[0]
            last_equity = equity_data["equity"].iloc[-1]
            return_pct = (last_equity - first_equity) / first_equity * 100
            
            # Calculate drawdowns if not already in the data
            if "drawdown_pct" not in equity_data.columns:
                running_max = equity_data["equity"].cummax()
                drawdown_pct = (running_max - equity_data["equity"]) / running_max * 100
                equity_data["drawdown_pct"] = drawdown_pct
            
            # Calculate max drawdown and duration
            max_drawdown = equity_data["drawdown_pct"].max()
            
            # Calculate drawdown duration (rough approximation)
            is_drawdown = equity_data["drawdown_pct"] > 0
            if is_drawdown.any():
                # Group consecutive True values
                drawdown_groups = (is_drawdown != is_drawdown.shift()).cumsum()
                drawdown_periods = is_drawdown.groupby(drawdown_groups).sum()
                drawdown_duration_days = drawdown_periods.max() if len(drawdown_periods) > 0 else 0
            else:
                drawdown_duration_days = 0
            
            # Calculate daily returns if not a datetime index
            if not isinstance(equity_data.index, pd.DatetimeIndex):
                equity_data.index = pd.to_datetime(equity_data.index)
            
            # Resample to daily data if higher frequency
            daily_data = equity_data.resample('D').last().fillna(method='ffill')
            
            # Calculate daily returns
            daily_returns = daily_data["equity"].pct_change().dropna()
            
            # Calculate volatility (annualized)
            volatility = daily_returns.std() * np.sqrt(252) * 100 if len(daily_returns) > 0 else 0
            
            # Calculate Sharpe ratio (annualized, assuming 0% risk-free rate)
            mean_daily_return = daily_returns.mean() if len(daily_returns) > 0 else 0
            sharpe_ratio = (mean_daily_return * 252) / (daily_returns.std() * np.sqrt(252)) if len(daily_returns) > 0 and daily_returns.std() > 0 else 0
            
            # Calculate 7-day rolling returns
            rolling_returns = daily_data["equity"].pct_change(7).dropna() * 100
            rolling_returns_df = pd.DataFrame({
                "date": rolling_returns.index,
                "return_pct": rolling_returns.values
            })
            
            # Prepare drawdown curve for visualization
            drawdown_curve = pd.DataFrame({
                "date": equity_data.index,
                "drawdown_pct": equity_data["drawdown_pct"]
            })
            
            # Prepare equity curve for visualization
            equity_curve = pd.DataFrame({
                "date": equity_data.index,
                "equity": equity_data["equity"]
            })
            
            return {
                "return_pct": return_pct,
                "max_drawdown_pct": max_drawdown,
                "drawdown_duration_days": drawdown_duration_days,
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "equity_curve": equity_curve,
                "drawdown_curve": drawdown_curve,
                "rolling_returns": rolling_returns_df
            }
        
        except Exception as e:
            logger.error(f"Error transforming equity data: {str(e)}")
            return {
                "return_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "drawdown_duration_days": 0,
                "volatility": 0.0,
                "sharpe_ratio": 0.0,
                "equity_curve": pd.DataFrame(),
                "drawdown_curve": pd.DataFrame(),
                "rolling_returns": pd.DataFrame()
            }
    
    @staticmethod
    @cached(ttl_seconds=300, key_prefix="transform_trade_data")
    def transform_trade_data(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Transform trade data for dashboard visualizations.
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            Dictionary with transformed trade metrics
        """
        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_profit": 0.0,
                "avg_loss": 0.0,
                "largest_profit": 0.0,
                "largest_loss": 0.0,
                "profit_trades": 0,
                "loss_trades": 0,
                "by_symbol": {},
                "by_strategy": {},
                "by_direction": {}
            }
        
        try:
            # Calculate basic metrics
            total_trades = len(trades)
            profitable_trades = [t for t in trades if t.get("profitable", False)]
            losing_trades = [t for t in trades if not t.get("profitable", False)]
            
            profit_trades = len(profitable_trades)
            loss_trades = len(losing_trades)
            
            win_rate = profit_trades / total_trades if total_trades > 0 else 0.0
            
            # Calculate profit metrics
            total_profit = sum(t.get("profit_amount", 0) for t in profitable_trades)
            total_loss = abs(sum(t.get("profit_amount", 0) for t in losing_trades))
            profit_factor = total_profit / total_loss if total_loss > 0 else 0.0
            
            avg_profit = total_profit / profit_trades if profit_trades > 0 else 0.0
            avg_loss = total_loss / loss_trades if loss_trades > 0 else 0.0
            
            largest_profit = max([t.get("profit_amount", 0) for t in profitable_trades]) if profitable_trades else 0.0
            largest_loss = min([t.get("profit_amount", 0) for t in losing_trades]) if losing_trades else 0.0
            
            # Group by symbol
            symbols = set(t.get("symbol", "unknown") for t in trades)
            by_symbol = {}
            
            for symbol in symbols:
                symbol_trades = [t for t in trades if t.get("symbol") == symbol]
                symbol_profitable = [t for t in symbol_trades if t.get("profitable", False)]
                
                by_symbol[symbol] = {
                    "total": len(symbol_trades),
                    "profitable": len(symbol_profitable),
                    "win_rate": len(symbol_profitable) / len(symbol_trades) if len(symbol_trades) > 0 else 0.0,
                    "profit_amount": sum(t.get("profit_amount", 0) for t in symbol_trades)
                }
            
            # Group by strategy
            strategies = set(t.get("strategy", "unknown") for t in trades)
            by_strategy = {}
            
            for strategy in strategies:
                strategy_trades = [t for t in trades if t.get("strategy") == strategy]
                strategy_profitable = [t for t in strategy_trades if t.get("profitable", False)]
                
                by_strategy[strategy] = {
                    "total": len(strategy_trades),
                    "profitable": len(strategy_profitable),
                    "win_rate": len(strategy_profitable) / len(strategy_trades) if len(strategy_trades) > 0 else 0.0,
                    "profit_amount": sum(t.get("profit_amount", 0) for t in strategy_trades)
                }
            
            # Group by direction
            by_direction = {
                "long": {"total": 0, "profitable": 0, "win_rate": 0.0, "profit_amount": 0.0},
                "short": {"total": 0, "profitable": 0, "win_rate": 0.0, "profit_amount": 0.0}
            }
            
            long_trades = [t for t in trades if t.get("direction", "").lower() == "long"]
            short_trades = [t for t in trades if t.get("direction", "").lower() == "short"]
            
            by_direction["long"]["total"] = len(long_trades)
            by_direction["long"]["profitable"] = len([t for t in long_trades if t.get("profitable", False)])
            by_direction["long"]["win_rate"] = by_direction["long"]["profitable"] / by_direction["long"]["total"] if by_direction["long"]["total"] > 0 else 0.0
            by_direction["long"]["profit_amount"] = sum(t.get("profit_amount", 0) for t in long_trades)
            
            by_direction["short"]["total"] = len(short_trades)
            by_direction["short"]["profitable"] = len([t for t in short_trades if t.get("profitable", False)])
            by_direction["short"]["win_rate"] = by_direction["short"]["profitable"] / by_direction["short"]["total"] if by_direction["short"]["total"] > 0 else 0.0
            by_direction["short"]["profit_amount"] = sum(t.get("profit_amount", 0) for t in short_trades)
            
            return {
                "total_trades": total_trades,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "avg_profit": avg_profit,
                "avg_loss": avg_loss,
                "largest_profit": largest_profit,
                "largest_loss": largest_loss,
                "profit_trades": profit_trades,
                "loss_trades": loss_trades,
                "by_symbol": by_symbol,
                "by_strategy": by_strategy,
                "by_direction": by_direction
            }
        
        except Exception as e:
            logger.error(f"Error transforming trade data: {str(e)}")
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_profit": 0.0,
                "avg_loss": 0.0,
                "largest_profit": 0.0,
                "largest_loss": 0.0,
                "profit_trades": 0,
                "loss_trades": 0,
                "by_symbol": {},
                "by_strategy": {},
                "by_direction": {}
            }
    
    @staticmethod
    @cached(ttl_seconds=60, key_prefix="transform_orderbook_data")
    def transform_orderbook_data(orderbook: Dict[str, Any], depth: int = 10) -> Dict[str, Any]:
        """
        Transform orderbook data for dashboard visualizations.
        
        Args:
            orderbook: Raw orderbook data
            depth: Depth of the orderbook to include (number of levels)
            
        Returns:
            Dictionary with transformed orderbook metrics
        """
        if not orderbook or "bids" not in orderbook or "asks" not in orderbook:
            return {
                "bids": [],
                "asks": [],
                "spread": 0.0,
                "midprice": 0.0,
                "imbalance": 0.0,
                "bid_volume": 0.0,
                "ask_volume": 0.0,
                "total_volume": 0.0
            }
        
        try:
            bids = orderbook.get("bids", [])[:depth]
            asks = orderbook.get("asks", [])[:depth]
            
            # Calculate basic metrics
            if not bids or not asks:
                return {
                    "bids": bids,
                    "asks": asks,
                    "spread": 0.0,
                    "midprice": 0.0,
                    "imbalance": 0.0,
                    "bid_volume": 0.0,
                    "ask_volume": 0.0,
                    "total_volume": 0.0
                }
            
            # Get best bid/ask
            best_bid = bids[0][0] if len(bids) > 0 else 0
            best_ask = asks[0][0] if len(asks) > 0 else 0
            
            # Calculate metrics
            spread = best_ask - best_bid if best_bid > 0 and best_ask > 0 else 0
            midprice = (best_bid + best_ask) / 2 if best_bid > 0 and best_ask > 0 else 0
            
            # Calculate volumes
            bid_volume = sum(level[1] for level in bids)
            ask_volume = sum(level[1] for level in asks)
            total_volume = bid_volume + ask_volume
            
            # Calculate imbalance
            imbalance = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0
            
            # Prepare formatted levels
            formatted_bids = [{"price": level[0], "size": level[1]} for level in bids]
            formatted_asks = [{"price": level[0], "size": level[1]} for level in asks]
            
            return {
                "bids": formatted_bids,
                "asks": formatted_asks,
                "spread": spread,
                "midprice": midprice,
                "imbalance": imbalance,
                "bid_volume": bid_volume,
                "ask_volume": ask_volume,
                "total_volume": total_volume
            }
        
        except Exception as e:
            logger.error(f"Error transforming orderbook data: {str(e)}")
            return {
                "bids": [],
                "asks": [],
                "spread": 0.0,
                "midprice": 0.0,
                "imbalance": 0.0,
                "bid_volume": 0.0,
                "ask_volume": 0.0,
                "total_volume": 0.0
            }
    
    @staticmethod
    def format_time_ago(timestamp: Optional[datetime]) -> str:
        """
        Format a timestamp as a human-readable "time ago" string.
        
        Args:
            timestamp: Datetime object
            
        Returns:
            Human-readable string of time elapsed
        """
        if timestamp is None:
            return "never"
        
        now = datetime.now()
        delta = now - timestamp
        seconds = delta.total_seconds()
        
        if seconds < 60:
            return f"{int(seconds)} seconds ago"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        elif seconds < 86400:
            hours = int(seconds / 3600)
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        elif seconds < 604800:
            days = int(seconds / 86400)
            return f"{days} day{'s' if days != 1 else ''} ago"
        else:
            weeks = int(seconds / 604800)
            return f"{weeks} week{'s' if weeks != 1 else ''} ago"
    
    @staticmethod
    def format_duration(seconds: Union[int, float]) -> str:
        """
        Format a duration in seconds as a human-readable string.
        
        Args:
            seconds: Duration in seconds
            
        Returns:
            Human-readable duration string
        """
        if seconds < 60:
            return f"{int(seconds)}s"
        
        minutes, seconds = divmod(seconds, 60)
        if minutes < 60:
            return f"{int(minutes)}m {int(seconds)}s"
        
        hours, minutes = divmod(minutes, 60)
        if hours < 24:
            return f"{int(hours)}h {int(minutes)}m"
        
        days, hours = divmod(hours, 24)
        return f"{int(days)}d {int(hours)}h"
    
    @staticmethod
    def filter_data_by_time_range(
        data: pd.DataFrame,
        time_range: str = "1m",
        date_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Filter dataframe by a time range.
        
        Args:
            data: DataFrame to filter
            time_range: Time range code ("1d", "1w", "1m", "3m", "6m", "1y", "all")
            date_column: Column name containing dates (if None, uses index)
            
        Returns:
            Filtered DataFrame
        """
        if data is None or data.empty or time_range == "all":
            return data
        
        # Set the date series to filter on
        if date_column is not None and date_column in data.columns:
            dates = data[date_column]
        else:
            if not isinstance(data.index, pd.DatetimeIndex):
                try:
                    data.index = pd.to_datetime(data.index)
                except:
                    logger.warning("Could not convert index to datetime for time range filtering")
                    return data
            dates = data.index
        
        now = datetime.now()
        
        # Calculate the start date based on the time range
        if time_range == "1d":
            start_date = now - timedelta(days=1)
        elif time_range == "1w":
            start_date = now - timedelta(weeks=1)
        elif time_range == "1m":
            start_date = now - timedelta(days=30)
        elif time_range == "3m":
            start_date = now - timedelta(days=90)
        elif time_range == "6m":
            start_date = now - timedelta(days=180)
        elif time_range == "1y":
            start_date = now - timedelta(days=365)
        else:
            # Default to all data
            return data
        
        # Filter the data
        if date_column is not None:
            return data[dates >= start_date]
        else:
            return data[data.index >= start_date]

    @staticmethod
    @cached(ttl_seconds=60, key_prefix="transform_strategy_data")
    def transform_strategy_data(strategy_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform strategy data for dashboard visualizations.
        
        Args:
            strategy_data: Raw strategy data
            
        Returns:
            Dictionary with transformed strategy metrics
        """
        if not strategy_data:
            return {
                "id": None,
                "name": "Unknown",
                "description": "",
                "status": "inactive",
                "parameters": {},
                "performance": {
                    "win_rate": 0.0,
                    "profit_factor": 0.0,
                    "total_trades": 0,
                    "profitable_trades": 0,
                    "losing_trades": 0
                },
                "indicators": [],
                "signals": [],
                "positions": []
            }
        
        try:
            # Extract basic info
            strategy_id = strategy_data.get("id", "unknown")
            name = strategy_data.get("name", f"Strategy {strategy_id}")
            description = strategy_data.get("description", "")
            status = strategy_data.get("status", "inactive").lower()
            parameters = strategy_data.get("parameters", {})
            
            # Extract performance data
            performance_data = strategy_data.get("performance", {})
            performance = {
                "win_rate": performance_data.get("win_rate", 0.0),
                "profit_factor": performance_data.get("profit_factor", 0.0),
                "total_trades": performance_data.get("total_trades", 0),
                "profitable_trades": performance_data.get("profitable_trades", 0),
                "losing_trades": performance_data.get("losing_trades", 0),
                "average_profit": performance_data.get("average_profit", 0.0),
                "average_loss": performance_data.get("average_loss", 0.0),
                "total_profit": performance_data.get("total_profit", 0.0)
            }
            
            # Extract indicators
            raw_indicators = strategy_data.get("indicators", [])
            indicators = []
            
            for ind in raw_indicators:
                indicator = {
                    "name": ind.get("name", "Unknown"),
                    "value": ind.get("value", 0.0),
                    "type": ind.get("type", "other"),
                    "timestamp": ind.get("timestamp", datetime.now().timestamp())
                }
                indicators.append(indicator)
            
            # Extract signals
            raw_signals = strategy_data.get("signals", [])
            signals = []
            
            for sig in raw_signals:
                signal = {
                    "type": sig.get("type", "unknown"),
                    "direction": sig.get("direction", "neutral"),
                    "strength": sig.get("strength", 0.0),
                    "timestamp": sig.get("timestamp", datetime.now().timestamp()),
                    "symbol": sig.get("symbol", "unknown")
                }
                signals.append(signal)
            
            # Extract positions
            raw_positions = strategy_data.get("positions", [])
            positions = []
            
            for pos in raw_positions:
                position = {
                    "symbol": pos.get("symbol", "unknown"),
                    "direction": pos.get("direction", "none"),
                    "size": pos.get("size", 0.0),
                    "entry_price": pos.get("entry_price", 0.0),
                    "current_price": pos.get("current_price", 0.0),
                    "profit_loss": pos.get("profit_loss", 0.0),
                    "profit_loss_pct": pos.get("profit_loss_pct", 0.0),
                    "open_time": pos.get("open_time", datetime.now().timestamp())
                }
                positions.append(position)
            
            return {
                "id": strategy_id,
                "name": name,
                "description": description,
                "status": status,
                "parameters": parameters,
                "performance": performance,
                "indicators": indicators,
                "signals": signals,
                "positions": positions
            }
        
        except Exception as e:
            logger.error(f"Error transforming strategy data: {str(e)}")
            return {
                "id": None,
                "name": "Unknown",
                "description": "",
                "status": "inactive",
                "parameters": {},
                "performance": {
                    "win_rate": 0.0,
                    "profit_factor": 0.0,
                    "total_trades": 0,
                    "profitable_trades": 0,
                    "losing_trades": 0
                },
                "indicators": [],
                "signals": [],
                "positions": []
            }


# Create a singleton instance for easy access
data_transformer = DataTransformer() 