"""
Dashboard Data Service Module

This module handles data retrieval and processing for the dashboard components.
It provides a centralized interface for accessing trading system data.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from loguru import logger

from src.dashboard.utils.transformers import data_transformer


class DashboardDataService:
    """Service for retrieving and processing data for the dashboard."""
    
    def __init__(
        self,
        api_client=None,
        trade_manager=None,
        performance_tracker=None,
        risk_manager=None,
        strategy_manager=None,
        market_data=None,
        paper_trading=None,
        orderbook_analyzer=None
    ):
        """
        Initialize the data service.
        
        Args:
            api_client: Bybit API client
            trade_manager: Trade manager instance
            performance_tracker: Performance tracker instance
            risk_manager: Risk manager instance
            strategy_manager: Strategy manager instance
            market_data: Market data instance
            paper_trading: Paper trading simulator instance
            orderbook_analyzer: Orderbook analyzer instance
        """
        self.api_client = api_client
        self.trade_manager = trade_manager
        self.performance_tracker = performance_tracker
        self.risk_manager = risk_manager
        self.strategy_manager = strategy_manager
        self.market_data = market_data
        self.paper_trading = paper_trading
        self.orderbook_analyzer = orderbook_analyzer
        
        # Data storage
        self._performance_data = {}
        self._trade_data = {}
        self._orderbook_data = {}
        self._strategy_data = {}
        self._market_data_cache = {}
        
        # Data freshness tracking
        self._data_updated_at = {
            "performance": None,
            "trades": None,
            "orderbook": None,
            "strategy": None,
            "market": None,
            "system": None
        }
        
        # System status
        self._system_running = False
        self._system_start_time = None
        self._system_mode = "Stopped"
        
        # Initialize in standalone mode if components are missing
        if (self.api_client is None or self.trade_manager is None or 
                self.performance_tracker is None):
            self._initialize_standalone_mode()
    
    @property
    def standalone_mode(self) -> bool:
        """
        Check if the dashboard is running in standalone mode.
        
        Returns:
            True if running in standalone mode, False otherwise
        """
        return (self.api_client is None or self.trade_manager is None or 
                self.performance_tracker is None)
    
    def _initialize_standalone_mode(self):
        """Initialize with sample data for standalone dashboard mode."""
        logger.info("Initializing dashboard in standalone mode with sample data")
        
        # Create sample performance data
        self._performance_data = {
            "total_return_pct": 8.45,
            "win_rate": 0.625,
            "total_trades": 24,
            "profitable_trades": 15,
            "profit_factor": 2.34,
            "average_trade_return": 32.45,
            "max_drawdown_pct": 7.82,
            "drawdown_duration_days": 3,
            "sharpe_ratio": 1.87,
            "equity_history": self._generate_sample_equity_history(days=30),
            "daily_returns": self._generate_sample_daily_returns(days=30),
            "trade_history": self._generate_sample_trade_history(count=20)
        }
    
    def _generate_sample_equity_history(self, days: int = 30) -> pd.DataFrame:
        """
        Generate sample equity history for standalone mode.
        
        Args:
            days: Number of days of history to generate
            
        Returns:
            DataFrame with equity history
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq="D")
        
        # Generate random equity curve (starting at 10000, with small daily changes)
        np.random.seed(42)  # For reproducibility
        equity = 10000.0
        equities = [equity]
        
        for i in range(1, len(dates)):
            daily_return = np.random.normal(0.001, 0.01)  # Mean 0.1%, std 1%
            equity *= (1 + daily_return)
            equities.append(equity)
        
        # Calculate drawdowns
        running_max = np.maximum.accumulate(equities)
        drawdowns = (running_max - equities) / running_max * 100
        
        # Create DataFrame
        df = pd.DataFrame({
            "date": dates,
            "equity": equities,
            "drawdown_pct": drawdowns
        })
        df.set_index("date", inplace=True)
        
        return df
    
    def _generate_sample_daily_returns(self, days: int = 30) -> pd.DataFrame:
        """
        Generate sample daily returns for standalone mode.
        
        Args:
            days: Number of days of history to generate
            
        Returns:
            DataFrame with daily returns
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq="D")
        
        # Generate random daily returns
        np.random.seed(42)  # For reproducibility
        daily_returns = np.random.normal(0.001, 0.015, size=len(dates))  # Mean 0.1%, std 1.5%
        
        # Create DataFrame
        df = pd.DataFrame({
            "date": dates,
            "return_pct": daily_returns * 100
        })
        df.set_index("date", inplace=True)
        
        return df
    
    def _generate_sample_trade_history(self, count: int = 20) -> List[Dict[str, Any]]:
        """
        Generate sample trade history for standalone mode.
        
        Args:
            count: Number of trades to generate
            
        Returns:
            List of trade dictionaries
        """
        trades = []
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT"]
        strategies = ["EMA Cross", "RSI Oversold", "Bollinger Band", "MACD Signal"]
        
        end_date = datetime.now()
        
        for i in range(count):
            # Generate timestamps going backward from now
            trade_time = end_date - timedelta(hours=i*8)
            
            # Randomize trade details
            symbol = symbols[i % len(symbols)]
            strategy = strategies[i % len(strategies)]
            
            # Determine if trade was profitable (slightly biased towards profitable)
            profitable = np.random.random() < 0.6
            
            # Generate trade details
            if symbol == "BTCUSDT":
                price = round(np.random.uniform(25000, 28000), 2)
                quantity = round(np.random.uniform(0.01, 0.1), 3)
            elif symbol == "ETHUSDT":
                price = round(np.random.uniform(1500, 1800), 2)
                quantity = round(np.random.uniform(0.1, 0.5), 2)
            elif symbol == "SOLUSDT":
                price = round(np.random.uniform(15, 25), 2)
                quantity = round(np.random.uniform(5, 20), 1)
            elif symbol == "BNBUSDT":
                price = round(np.random.uniform(280, 350), 2)
                quantity = round(np.random.uniform(0.2, 1.0), 2)
            else:  # ADAUSDT
                price = round(np.random.uniform(0.30, 0.40), 4)
                quantity = round(np.random.uniform(100, 500), 0)
            
            # Calculate PnL
            pnl = round(price * quantity * (0.02 if profitable else -0.015), 2)
            pnl_str = f"+${pnl}" if pnl >= 0 else f"-${abs(pnl)}"
            
            # Create trade
            trade = {
                "id": f"trade-{i+1}",
                "symbol": symbol,
                "strategy": strategy,
                "side": "Buy" if i % 2 == 0 else "Sell",
                "entry_time": trade_time.strftime("%Y-%m-%d %H:%M"),
                "exit_time": (trade_time + timedelta(hours=4)).strftime("%Y-%m-%d %H:%M"),
                "entry_price": price,
                "exit_price": round(price * (1.02 if profitable else 0.985), 2),
                "quantity": quantity,
                "pnl": pnl,
                "pnl_str": pnl_str,
                "profitable": profitable
            }
            
            trades.append(trade)
        
        return trades
    
    def get_performance_data(self) -> Dict[str, Any]:
        """
        Get performance data for the dashboard.
        
        Returns:
            Dictionary containing performance metrics
        """
        # Update timestamp for data freshness tracking
        self._data_updated_at["performance"] = datetime.now()
        
        try:
            if self.standalone_mode:
                # Use sample data in standalone mode
                equity_data = pd.DataFrame({
                    "date": pd.date_range(end=datetime.now(), periods=90, freq='D'),
                    "equity": self._generated_equity
                }).set_index("date")
                
                # Use the DataTransformer to transform equity data 
                return data_transformer.transform_equity_data(equity_data)
            else:
                # In connected mode, get data from the performance tracker
                if self.performance_tracker:
                    equity_data = None
                    if hasattr(self.performance_tracker, 'get_equity_curve'):
                        equity_data = self.performance_tracker.get_equity_curve()
                    
                    if equity_data is not None and not equity_data.empty:
                        # Use the DataTransformer to transform equity data
                        return data_transformer.transform_equity_data(equity_data)
                    
                    # Fallback for older API or missing data
                    daily_returns = None
                    if hasattr(self.performance_tracker, 'get_daily_returns'):
                        daily_returns = self.performance_tracker.get_daily_returns()
                    
                    if daily_returns is not None and not daily_returns.empty:
                        # Construct equity curve from daily returns
                        initial_equity = 10000  # Assumed starting equity
                        cumulative_returns = (daily_returns + 1).cumprod() * initial_equity
                        equity_data = pd.DataFrame({
                            "equity": cumulative_returns
                        })
                        
                        # Use the DataTransformer to transform constructed equity data
                        return data_transformer.transform_equity_data(equity_data)
                    
                    # Fallback when only metrics are available
                    try:
                        total_return_pct = self.performance_tracker.get_total_return() * 100 \
                            if hasattr(self.performance_tracker, 'get_total_return') else 0
                        
                        max_drawdown_pct = self.performance_tracker.get_max_drawdown() * 100 \
                            if hasattr(self.performance_tracker, 'get_max_drawdown') else 0
                        
                        sharpe_ratio = self.performance_tracker.get_sharpe_ratio() \
                            if hasattr(self.performance_tracker, 'get_sharpe_ratio') else 0
                        
                        # Return basic metrics when no time series data is available
                        return {
                            "return_pct": total_return_pct,
                            "max_drawdown_pct": max_drawdown_pct,
                            "drawdown_duration_days": 0,
                            "volatility": 0.0,
                            "sharpe_ratio": sharpe_ratio,
                            "equity_curve": pd.DataFrame(),
                            "drawdown_curve": pd.DataFrame(),
                            "rolling_returns": pd.DataFrame()
                        }
                    except Exception as e:
                        logger.warning(f"Error fetching performance metrics: {str(e)}")
                        # Fall back to sample data
                        logger.info("Falling back to sample performance data")
                        return self._get_sample_performance_data()
                else:
                    # Fallback when no performance tracker is available
                    logger.info("No performance tracker available, using sample data")
                    return self._get_sample_performance_data()
        except Exception as e:
            logger.error(f"Error retrieving performance data: {str(e)}")
            return self._get_sample_performance_data()
    
    def _get_sample_performance_data(self) -> Dict[str, Any]:
        """
        Generate sample performance data for standalone mode or fallback.
        
        Returns:
            Dictionary with sample performance metrics
        """
        equity_data = pd.DataFrame({
            "date": pd.date_range(end=datetime.now(), periods=90, freq='D'),
            "equity": self._generated_equity
        }).set_index("date")
        
        # Use the DataTransformer to standardize the transformation
        return data_transformer.transform_equity_data(equity_data)
    
    def get_trade_data(self) -> Dict[str, Any]:
        """
        Get trade and order history data.
        
        Returns:
            Dictionary containing active trades and order history
        """
        # Update timestamp for data freshness tracking
        self._data_updated_at["trades"] = datetime.now()
        
        try:
            if self.standalone_mode:
                # Use sample data in standalone mode
                sample_trades = self._generate_sample_trade_history(30)
                # Use the DataTransformer to transform trade data
                return data_transformer.transform_trade_data(sample_trades)
            else:
                # In connected mode, get data from the trade manager
                if self.trade_manager:
                    trades = []
                    
            # Get active trades
                if hasattr(self.trade_manager, 'get_active_trades'):
                    active_trades = self.trade_manager.get_active_trades()
                    if active_trades:
                            trades.extend(active_trades)
                    
                    # Get trade history
                    if hasattr(self.trade_manager, 'get_closed_trades'):
                        trade_history = self.trade_manager.get_closed_trades(limit=100)
                        if trade_history:
                            trades.extend(trade_history)
                    
                    # If we have trades, transform them
                    if trades:
                        # Use the DataTransformer to standardize the transformation
                        return data_transformer.transform_trade_data(trades)
                    
                    # Fall back to sample data if no trades found
                    logger.info("No trades found, using sample data")
                    return self._get_sample_trade_data()
                else:
                    # Fallback when no trade manager is available
                    logger.info("No trade manager available, using sample data")
                    return self._get_sample_trade_data()
        except Exception as e:
            logger.error(f"Error retrieving trade data: {str(e)}")
            return self._get_sample_trade_data()
    
    def _get_sample_trade_data(self) -> Dict[str, Any]:
        """
        Generate sample trade data for standalone mode or fallback.
        
        Returns:
            Dictionary with sample trade metrics
        """
        sample_trades = self._generate_sample_trade_history(30)
        # Use the DataTransformer to standardize the transformation
        return data_transformer.transform_trade_data(sample_trades)
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get system status information.
        
        Returns:
            Dictionary with system status
        """
        # If no components available, return the tracking status
        if self.trade_manager is None:
            status = "Stopped"
            if self._system_running:
                status = "Running"
                
            # Calculate uptime
            uptime = "0s"
            if self._system_start_time:
                uptime_seconds = (datetime.now() - self._system_start_time).total_seconds()
                # Format as days, hours, minutes, seconds
                days, remainder = divmod(uptime_seconds, 86400)
                hours, remainder = divmod(remainder, 3600)
                minutes, seconds = divmod(remainder, 60)
                
                uptime_parts = []
                if days > 0:
                    uptime_parts.append(f"{int(days)}d")
                if hours > 0 or days > 0:
                    uptime_parts.append(f"{int(hours)}h")
                if minutes > 0 or hours > 0 or days > 0:
                    uptime_parts.append(f"{int(minutes)}m")
                uptime_parts.append(f"{int(seconds)}s")
                
                uptime = "".join(uptime_parts)
            
            return {
                "status": status,
                "is_running": self._system_running,
                "mode": self._system_mode,
                "uptime": uptime,
                "details": "System is in standalone dashboard mode"
            }
        
        # In connected mode, fetch real status
        try:
            # Get status from trade manager
            is_running = self.trade_manager.is_running()
            start_time = self.trade_manager.get_start_time()
            
            # Determine status text
            if is_running:
                status = "Running"
                details = "System is actively trading"
            else:
                status = "Stopped"
                details = "System is not currently trading"
            
            # Calculate uptime
            uptime = "0s"
            if start_time:
                uptime_seconds = (datetime.now() - start_time).total_seconds()
                # Format as days, hours, minutes, seconds
                days, remainder = divmod(uptime_seconds, 86400)
                hours, remainder = divmod(remainder, 3600)
                minutes, seconds = divmod(remainder, 60)
                
                uptime_parts = []
                if days > 0:
                    uptime_parts.append(f"{int(days)}d")
                if hours > 0 or days > 0:
                    uptime_parts.append(f"{int(hours)}h")
                if minutes > 0 or hours > 0 or days > 0:
                    uptime_parts.append(f"{int(minutes)}m")
                uptime_parts.append(f"{int(seconds)}s")
                
                uptime = "".join(uptime_parts)
            
            # Determine mode
            mode = "Live Trading"
            if self.paper_trading:
                mode = "Paper Trading"
            
            status_data = {
                "status": status,
                "is_running": is_running,
                "mode": mode,
                "uptime": uptime,
                "details": details
            }
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            # Fall back to unknown status
            status_data = {
                "status": "Unknown",
                "is_running": False,
                "mode": "Unknown",
                "uptime": "0s",
                "details": f"Error: {str(e)}"
            }
        
        # Update freshness timestamp
        self._data_updated_at["system"] = datetime.now()
        
        return status_data
    
    def is_data_fresh(self, data_type: str) -> bool:
        """
        Check if the specified data type is considered fresh.
        
        Args:
            data_type: Type of data to check freshness for
            
        Returns:
            True if data is fresh, False otherwise
        """
        if data_type not in self._data_updated_at:
            return False
        
        last_update = self._data_updated_at[data_type]
        if last_update is None:
            return False
        
        # Define freshness thresholds for different data types
        thresholds = {
            "performance": timedelta(minutes=5),
            "trades": timedelta(minutes=1),
            "orderbook": timedelta(seconds=10),
            "strategy": timedelta(minutes=1),
            "market": timedelta(seconds=30),
            "system": timedelta(seconds=10)
        }
        
        threshold = thresholds.get(data_type, timedelta(minutes=5))
        return datetime.now() - last_update < threshold
    
    def get_data_freshness(self) -> Dict[str, Dict[str, Any]]:
        """
        Get freshness information for all data types.
        
        Returns:
            Dictionary with freshness status for each data type
        """
        freshness = {}
        
        for data_type, last_update in self._data_updated_at.items():
            if last_update is None:
                freshness[data_type] = {
                    "fresh": False,
                    "updated": "Never",
                    "age_seconds": None
                }
            else:
                age = datetime.now() - last_update
                age_seconds = age.total_seconds()
                
                # Format age as human-readable string
                if age_seconds < 60:
                    updated = f"{int(age_seconds)} seconds ago"
                elif age_seconds < 3600:
                    updated = f"{int(age_seconds / 60)} minutes ago"
                else:
                    updated = f"{int(age_seconds / 3600)} hours ago"
                
                freshness[data_type] = {
                    "fresh": self.is_data_fresh(data_type),
                    "updated": updated,
                    "age_seconds": age_seconds
                }
        
    def get_orderbook_data(self, symbol: Optional[str] = None, depth: int = 10) -> Dict[str, Any]:
        """
        Get orderbook data for the specified symbol.
        
        Args:
            symbol: The symbol to get orderbook data for (optional)
            depth: The depth of the orderbook to return (default: 10)
            
        Returns:
            Dictionary containing orderbook data with bids, asks, and derived metrics
        """
        # Update timestamp for data freshness tracking
        self._data_updated_at["orderbook"] = datetime.now()
        
        try:
            if self.standalone_mode:
                # Generate sample orderbook data in standalone mode
                orderbook = self._generate_sample_orderbook(symbol)
                # Use the DataTransformer to transform orderbook data
                return data_transformer.transform_orderbook_data(orderbook, depth)
            else:
                # In connected mode, get data from the market data source
                if self.market_data:
                    # Get the current symbol if not specified
                    if symbol is None and hasattr(self.market_data, 'get_current_symbol'):
                        symbol = self.market_data.get_current_symbol()
                    elif symbol is None:
                        # Default symbol
                        symbol = "BTC/USD"
                    
                    # Get orderbook data
                    if hasattr(self.market_data, 'get_orderbook'):
                        orderbook = self.market_data.get_orderbook(symbol)
                        if orderbook:
                            # Use the DataTransformer to standardize the transformation
                            return data_transformer.transform_orderbook_data(orderbook, depth)
                    
                    # Fall back to sample data if no orderbook found
                    logger.info(f"No orderbook data found for {symbol}, using sample data")
                    return self._get_sample_orderbook_data(symbol, depth)
                else:
                    # Fallback when no market data source is available
                    logger.info("No market data source available, using sample data")
                    return self._get_sample_orderbook_data(symbol, depth)
        except Exception as e:
            logger.error(f"Error retrieving orderbook data: {str(e)}")
            return self._get_sample_orderbook_data(symbol, depth)
    
    def _get_sample_orderbook_data(self, symbol: Optional[str] = None, depth: int = 10) -> Dict[str, Any]:
        """
        Generate sample orderbook data for standalone mode or fallback.
        
        Args:
            symbol: The symbol to generate orderbook for
            depth: The depth of the orderbook to return
            
        Returns:
            Dictionary with sample orderbook data
        """
        orderbook = self._generate_sample_orderbook(symbol)
        # Use the DataTransformer to standardize the transformation
        return data_transformer.transform_orderbook_data(orderbook, depth)
    
    def _generate_sample_orderbook(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate sample orderbook data.
        
        Args:
            symbol: The trading symbol to generate orderbook for
            
        Returns:
            Dictionary with bids and asks
        """
        if symbol is None:
            symbol = "BTC/USD"
        
        # Generate a realistic price based on the symbol
        if "BTC" in symbol:
            base_price = 35000 + np.random.normal(0, 100)
        elif "ETH" in symbol:
            base_price = 2000 + np.random.normal(0, 20)
        elif "SOL" in symbol:
            base_price = 75 + np.random.normal(0, 1)
        else:
            base_price = 100 + np.random.normal(0, 1)
        
        # Generate bids (buy orders) below the base price
        bids = []
        for i in range(20):
            price = base_price * (1 - (i * 0.001) - np.random.uniform(0, 0.0005))
            size = np.random.uniform(0.1, 5) if "BTC" in symbol else np.random.uniform(1, 50)
            bids.append([price, size])
        
        # Generate asks (sell orders) above the base price
        asks = []
        for i in range(20):
            price = base_price * (1 + (i * 0.001) + np.random.uniform(0, 0.0005))
            size = np.random.uniform(0.1, 5) if "BTC" in symbol else np.random.uniform(1, 50)
            asks.append([price, size])
        
        return {
            "symbol": symbol,
            "bids": sorted(bids, key=lambda x: x[0], reverse=True),  # Sort bids in descending order
            "asks": sorted(asks, key=lambda x: x[0]),  # Sort asks in ascending order
            "timestamp": datetime.now().timestamp()
        }
    
    def get_strategy_data(self, strategy_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get strategy data and statistics.
        
        Args:
            strategy_id: Optional strategy ID to filter by
            
        Returns:
            Dictionary containing strategy data and performance metrics
        """
        # Update timestamp for data freshness tracking
        self._data_updated_at["strategy"] = datetime.now()
        
        try:
            if self.standalone_mode:
                # Use sample data in standalone mode
                strategy_data = self._generate_sample_strategy_data(strategy_id)
                # Use the DataTransformer to transform strategy data
                return data_transformer.transform_strategy_data(strategy_data)
            else:
                # In connected mode, get data from the strategy manager
                if self.strategy_manager:
                    strategy_data = {}
                    
                    # Get strategy data
                    if strategy_id is not None and hasattr(self.strategy_manager, 'get_strategy'):
                        strategy_data = self.strategy_manager.get_strategy(strategy_id)
                    elif hasattr(self.strategy_manager, 'get_active_strategy'):
                        strategy_data = self.strategy_manager.get_active_strategy()
                    elif hasattr(self.strategy_manager, 'get_strategies'):
                        strategies = self.strategy_manager.get_strategies()
                        if strategies and len(strategies) > 0:
                            # Get the first strategy
                            strategy_data = strategies[0]
                    
                    # If we have strategy data, transform it
                    if strategy_data:
                        # Use the DataTransformer to standardize the transformation
                        return data_transformer.transform_strategy_data(strategy_data)
                    
                    # Fall back to sample data if no strategy found
                    logger.info("No strategy data found, using sample data")
                    return self._get_sample_strategy_data(strategy_id)
                else:
                    # Fallback when no strategy manager is available
                    logger.info("No strategy manager available, using sample data")
                    return self._get_sample_strategy_data(strategy_id)
        except Exception as e:
            logger.error(f"Error retrieving strategy data: {str(e)}")
            return self._get_sample_strategy_data(strategy_id)
    
    def _get_sample_strategy_data(self, strategy_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate sample strategy data for standalone mode or fallback.
        
        Args:
            strategy_id: Optional strategy ID
            
        Returns:
            Dictionary with sample strategy data
        """
        strategy_data = self._generate_sample_strategy_data(strategy_id)
        # Use the DataTransformer to standardize the transformation
        return data_transformer.transform_strategy_data(strategy_data)
    
    def _generate_sample_strategy_data(self, strategy_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate sample strategy data.
        
        Args:
            strategy_id: Optional strategy ID to use
            
        Returns:
            Dictionary with sample strategy information
        """
        if strategy_id is None:
            # Generate a random strategy ID if not provided
            strategies = ["momentum", "mean_reversion", "trend_following", "breakout"]
            strategy_id = np.random.choice(strategies)
        
        # Generate strategy name and description
        if strategy_id == "momentum":
            name = "Momentum Strategy"
            description = "Capitalizes on continuation of existing market trends"
        elif strategy_id == "mean_reversion":
            name = "Mean Reversion Strategy"
            description = "Assumes prices will revert to their historical average"
        elif strategy_id == "trend_following":
            name = "Trend Following Strategy"
            description = "Uses indicators to identify and follow market trends"
        elif strategy_id == "breakout":
            name = "Breakout Strategy"
            description = "Enters trades when price breaks support or resistance levels"
        else:
            name = f"Strategy {strategy_id}"
            description = "Custom trading strategy"
        
        # Generate random performance metrics
        win_rate = np.random.uniform(0.4, 0.7)
        total_trades = np.random.randint(50, 200)
        profitable_trades = int(total_trades * win_rate)
        losing_trades = total_trades - profitable_trades
        
        avg_profit = np.random.uniform(1.5, 3.0)
        avg_loss = np.random.uniform(0.8, 1.5)
        total_profit = (profitable_trades * avg_profit) - (losing_trades * avg_loss)
        profit_factor = (profitable_trades * avg_profit) / (losing_trades * avg_loss) if losing_trades > 0 else 0
        
        # Generate random parameters based on strategy type
        parameters = {}
        if strategy_id == "momentum":
            parameters = {
                "lookback_period": np.random.randint(10, 30),
                "threshold": np.random.uniform(0.5, 2.0),
                "exit_after_bars": np.random.randint(5, 15)
            }
        elif strategy_id == "mean_reversion":
            parameters = {
                "ma_period": np.random.randint(20, 50),
                "std_dev_threshold": np.random.uniform(1.5, 3.0),
                "profit_target_pct": np.random.uniform(1.0, 3.0)
            }
        elif strategy_id == "trend_following":
            parameters = {
                "fast_ma": np.random.randint(5, 20),
                "slow_ma": np.random.randint(20, 50),
                "trailing_stop_pct": np.random.uniform(1.0, 3.0)
            }
        elif strategy_id == "breakout":
            parameters = {
                "breakout_period": np.random.randint(20, 40),
                "volume_filter": np.random.uniform(1.2, 2.0),
                "max_risk_pct": np.random.uniform(0.5, 2.0)
            }
        
        # Generate sample indicators
        indicators = []
        if strategy_id == "momentum":
            indicators.append({
                "name": "RSI",
                "value": np.random.uniform(30, 70),
                "type": "oscillator",
                "timestamp": (datetime.now() - timedelta(minutes=np.random.randint(1, 10))).timestamp()
            })
            indicators.append({
                "name": "Momentum",
                "value": np.random.uniform(-10, 10),
                "type": "momentum",
                "timestamp": (datetime.now() - timedelta(minutes=np.random.randint(1, 10))).timestamp()
            })
        elif strategy_id == "mean_reversion":
            indicators.append({
                "name": "Z-Score",
                "value": np.random.uniform(-2.5, 2.5),
                "type": "mean_reversion",
                "timestamp": (datetime.now() - timedelta(minutes=np.random.randint(1, 10))).timestamp()
            })
            indicators.append({
                "name": "Bollinger %B",
                "value": np.random.uniform(0, 1),
                "type": "oscillator",
                "timestamp": (datetime.now() - timedelta(minutes=np.random.randint(1, 10))).timestamp()
            })
        elif strategy_id == "trend_following":
            indicators.append({
                "name": "MACD",
                "value": np.random.uniform(-5, 5),
                "type": "trend",
                "timestamp": (datetime.now() - timedelta(minutes=np.random.randint(1, 10))).timestamp()
            })
            indicators.append({
                "name": "ADX",
                "value": np.random.uniform(10, 50),
                "type": "trend_strength",
                "timestamp": (datetime.now() - timedelta(minutes=np.random.randint(1, 10))).timestamp()
            })
        elif strategy_id == "breakout":
            indicators.append({
                "name": "ATR",
                "value": np.random.uniform(1, 10),
                "type": "volatility",
                "timestamp": (datetime.now() - timedelta(minutes=np.random.randint(1, 10))).timestamp()
            })
            indicators.append({
                "name": "Volume Ratio",
                "value": np.random.uniform(0.5, 3),
                "type": "volume",
                "timestamp": (datetime.now() - timedelta(minutes=np.random.randint(1, 10))).timestamp()
            })
        
        # Generate sample signals
        signals = []
        directions = ["buy", "sell", "neutral"]
        signal_types = ["entry", "exit", "warning", "filter"]
        
        # Add 2-3 sample signals
        for _ in range(np.random.randint(2, 4)):
            direction = np.random.choice(directions, p=[0.4, 0.4, 0.2])
            signal_type = np.random.choice(signal_types, p=[0.4, 0.3, 0.2, 0.1])
            signals.append({
                "type": signal_type,
                "direction": direction,
                "strength": np.random.uniform(0.1, 1.0),
                "timestamp": (datetime.now() - timedelta(minutes=np.random.randint(1, 60))).timestamp(),
                "symbol": np.random.choice(["BTC/USD", "ETH/USD", "SOL/USD"])
            })
        
        # Generate sample positions (0-2 positions)
        positions = []
        symbols = ["BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD"]
        directions = ["long", "short"]
        
        # Randomly decide if we have any positions
        if np.random.random() > 0.3:
            # Add 1-2 sample positions
            for _ in range(np.random.randint(1, 3)):
                symbol = np.random.choice(symbols)
                direction = np.random.choice(directions)
                
                if symbol == "BTC/USD":
                    entry_price = np.random.uniform(30000, 40000)
                    current_price = entry_price * (1 + np.random.uniform(-0.05, 0.05))
                    size = np.random.uniform(0.1, 1.0)
                elif symbol == "ETH/USD":
                    entry_price = np.random.uniform(1800, 2200)
                    current_price = entry_price * (1 + np.random.uniform(-0.05, 0.05))
                    size = np.random.uniform(1, 5)
                elif symbol == "SOL/USD":
                    entry_price = np.random.uniform(60, 90)
                    current_price = entry_price * (1 + np.random.uniform(-0.05, 0.05))
                    size = np.random.uniform(5, 20)
                else:
                    entry_price = np.random.uniform(0.05, 0.15)
                    current_price = entry_price * (1 + np.random.uniform(-0.05, 0.05))
                    size = np.random.uniform(1000, 5000)
                
                # Calculate profit/loss
                if direction == "long":
                    profit_loss = (current_price - entry_price) * size
                    profit_loss_pct = (current_price - entry_price) / entry_price * 100
                else:
                    profit_loss = (entry_price - current_price) * size
                    profit_loss_pct = (entry_price - current_price) / entry_price * 100
                
                positions.append({
                    "symbol": symbol,
                    "direction": direction,
                    "size": size,
                    "entry_price": entry_price,
                    "current_price": current_price,
                    "profit_loss": profit_loss,
                    "profit_loss_pct": profit_loss_pct,
                    "open_time": (datetime.now() - timedelta(hours=np.random.randint(1, 48))).timestamp()
                })
        
        return {
            "id": strategy_id,
            "name": name,
            "description": description,
            "status": np.random.choice(["active", "paused", "stopped"], p=[0.7, 0.2, 0.1]),
            "parameters": parameters,
            "performance": {
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "total_trades": total_trades,
                "profitable_trades": profitable_trades,
                "losing_trades": losing_trades,
                "average_profit": avg_profit,
                "average_loss": avg_loss,
                "total_profit": total_profit
            },
            "indicators": indicators,
            "signals": signals,
            "positions": positions
        }
    
    def get_market_data(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get market data for the specified symbol.
        
        Args:
            symbol: The symbol to get market data for (optional)
            
        Returns:
            Dictionary containing market data with price information, indicators, and candles
        """
        # Update timestamp for data freshness tracking
        self._data_updated_at["market"] = datetime.now()
        
        try:
            if self.standalone_mode:
                # Generate sample market data in standalone mode
                market_data = self._generate_sample_market_data(symbol)
                # Use the DataTransformer to transform market data
                return data_transformer.transform_market_data(market_data)
            else:
                # In connected mode, get data from the market data source
                if self.market_data:
                    # Get the current symbol if not specified
                    if symbol is None and hasattr(self.market_data, 'get_current_symbol'):
                        symbol = self.market_data.get_current_symbol()
                    elif symbol is None:
                        # Default symbol
                        symbol = "BTC/USD"
                    
                    # Check cache first
                    cache_key = f"market_data_{symbol}"
                    if cache_key in self._market_data_cache:
                        cached_data, timestamp = self._market_data_cache[cache_key]
                        # Check if cache is fresh (less than 5 seconds old)
                        if (datetime.now() - timestamp).total_seconds() < 5:
                            return cached_data
                    
                    # Get market data
                    market_data = {}
                    
                    # Attempt to get ticker data
                    if hasattr(self.market_data, 'get_ticker'):
                        ticker = self.market_data.get_ticker(symbol)
                        if ticker:
                            market_data.update(ticker)
                    
                    # Attempt to get candle data
                    if hasattr(self.market_data, 'get_candles'):
                        candles = self.market_data.get_candles(symbol, timeframe='1m', limit=100)
                        if candles:
                            market_data['candles'] = candles
                    
                    # Attempt to get indicators
                    if hasattr(self.market_data, 'get_indicators'):
                        indicators = self.market_data.get_indicators(symbol)
                        if indicators:
                            market_data['indicators'] = indicators
                    
                    # If we have market data, transform it
                    if market_data:
                        # Transform the data
                        transformed_data = data_transformer.transform_market_data(market_data)
                        
                        # Cache the result
                        self._market_data_cache[cache_key] = (transformed_data, datetime.now())
                        
                        return transformed_data
                    
                    # Fall back to sample data if no market data found
                    logger.info(f"No market data found for {symbol}, using sample data")
                    return self._get_sample_market_data(symbol)
                else:
                    # Fallback when no market data source is available
                    logger.info("No market data source available, using sample data")
                    return self._get_sample_market_data(symbol)
        except Exception as e:
            logger.error(f"Error retrieving market data: {str(e)}")
            return self._get_sample_market_data(symbol)
    
    def _get_sample_market_data(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate sample market data for standalone mode or fallback.
        
        Args:
            symbol: The symbol to generate market data for
            
        Returns:
            Dictionary with sample market data
        """
        market_data = self._generate_sample_market_data(symbol)
        # Use the DataTransformer to standardize the transformation
        return data_transformer.transform_market_data(market_data)
    
    def _generate_sample_market_data(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate sample market data.
        
        Args:
            symbol: The symbol to generate data for
            
        Returns:
            Dictionary with sample market data
        """
        # Default symbol if none provided
        if symbol is None:
            symbol = "BTC/USD"
        
        # Generate a realistic price based on the symbol
        base_price = 0.0
        if "BTC" in symbol:
            base_price = 40000 + np.random.normal(0, 200)
        elif "ETH" in symbol:
            base_price = 2800 + np.random.normal(0, 50)
        elif "BNB" in symbol:
            base_price = 320 + np.random.normal(0, 5)
        elif "ADA" in symbol:
            base_price = 0.35 + np.random.normal(0, 0.01)
        elif "SOL" in symbol:
            base_price = 75 + np.random.normal(0, 2)
        elif "XRP" in symbol:
            base_price = 0.55 + np.random.normal(0, 0.01)
        else:
            base_price = 100 + np.random.normal(0, 3)
        
        # Ensure price is positive
        price = max(0.01, base_price)
        
        # Create bid and ask with a small spread
        spread_pct = 0.05  # 0.05% spread
        spread = price * (spread_pct / 100)
        bid = price - (spread / 2)
        ask = price + (spread / 2)
        
        # Generate 24h statistics
        open_24h = price * (1 + np.random.normal(0, 0.02))  # Within ±2% of current price
        high_24h = max(price, open_24h) * (1 + abs(np.random.normal(0, 0.01)))
        low_24h = min(price, open_24h) * (1 - abs(np.random.normal(0, 0.01)))
        volume_24h = np.random.uniform(100, 1000) * price
        
        # Calculate change
        change_24h = price - open_24h
        change_pct_24h = (change_24h / open_24h) * 100
        
        # Generate sample indicators
        indicators = {
            "rsi": {
                "value": np.random.uniform(30, 70),
                "type": "oscillator"
            },
            "ma_50": {
                "value": price * (1 + np.random.normal(0, 0.01)),
                "type": "price"
            },
            "ma_200": {
                "value": price * (1 + np.random.normal(0, 0.02)),
                "type": "price"
            },
            "volume_avg": {
                "value": volume_24h * 0.9,  # Slightly less than today's volume
                "type": "volume"
            }
        }
        
        # Generate sample candles
        candles = []
        start_time = datetime.now() - timedelta(hours=24)
        
        # Start with a price around open_24h
        candle_price = open_24h
        
        for i in range(24 * 60):  # 24 hours of 1-minute candles
            candle_time = start_time + timedelta(minutes=i)
            
            # Random price movement with some trend persistence
            price_change = np.random.normal(0, 0.0005) + (0.0001 * (price - candle_price) / price)
            candle_price = candle_price * (1 + price_change)
            
            # Generate OHLC
            candle_open = candle_price
            candle_high = candle_open * (1 + abs(np.random.normal(0, 0.0003)))
            candle_low = candle_open * (1 - abs(np.random.normal(0, 0.0003)))
            candle_close = candle_price * (1 + np.random.normal(0, 0.0002))
            
            # Ensure proper ordering of prices
            candle_high = max(candle_open, candle_close, candle_high)
            candle_low = min(candle_open, candle_close, candle_low)
            
            # Random volume
            candle_volume = np.random.uniform(0.5, 1.5) * (volume_24h / (24 * 60))
            
            candles.append({
                "timestamp": candle_time,
                "open": candle_open,
                "high": candle_high,
                "low": candle_low,
                "close": candle_close,
                "volume": candle_volume
            })
        
        return {
            "symbol": symbol,
            "last_price": price,
            "bid": bid,
            "ask": ask,
            "open_24h": open_24h,
            "high_24h": high_24h,
            "low_24h": low_24h,
            "volume_24h": volume_24h,
            "change_24h": change_24h,
            "change_pct_24h": change_pct_24h,
            "timestamp": datetime.now(),
            "indicators": indicators,
            "candles": candles
        } 