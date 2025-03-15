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
            Dictionary with performance metrics
        """
        # In standalone mode, return the sample data
        if self.performance_tracker is None:
            return self._performance_data
        
        # In connected mode, fetch from the performance tracker
        data = {}
        
        try:
            # Get summary from performance tracker
            summary = self.performance_tracker.get_performance_summary()
            
            # Get detailed metrics
            data = {
                "total_return_pct": summary.get("total_return", 0.0),
                "win_rate": summary.get("win_rate", 0.0),
                "total_trades": summary.get("total_trades", 0),
                "profitable_trades": summary.get("profitable_trades", 0),
                "profit_factor": summary.get("profit_factor", 0.0),
                "average_trade_return": summary.get("average_trade_profit", 0.0),
                "max_drawdown_pct": summary.get("max_drawdown", 0.0),
                "drawdown_duration_days": summary.get("drawdown_duration", 0),
                "sharpe_ratio": summary.get("sharpe_ratio", 0.0),
                
                # Get historical data
                "equity_history": self.performance_tracker.get_equity_curve(),
                "daily_returns": self.performance_tracker.get_daily_returns(),
                "trade_history": self.performance_tracker.get_trade_history()
            }
        except Exception as e:
            logger.error(f"Error getting performance data: {e}")
            # Fall back to sample data if there's an error
            data = self._performance_data
        
        # Update freshness timestamp
        self._data_updated_at["performance"] = datetime.now()
        
        return data
    
    def get_trade_data(self) -> Dict[str, Any]:
        """
        Get trade data for the dashboard.
        
        Returns:
            Dictionary with active trades and order history
        """
        # In standalone mode, return sample data
        if self.trade_manager is None:
            active_trades = []
            order_history = self._performance_data.get("trade_history", [])[:5]
            return {"active_trades": active_trades, "order_history": order_history}
        
        # In connected mode, fetch from the trade manager
        data = {}
        
        try:
            # Get active trades
            active_trades = self.trade_manager.get_active_trades()
            
            # Get order history
            order_history = self.trade_manager.get_order_history(limit=20)
            
            data = {
                "active_trades": active_trades,
                "order_history": order_history
            }
        except Exception as e:
            logger.error(f"Error getting trade data: {e}")
            # Fall back to empty data if there's an error
            data = {"active_trades": [], "order_history": []}
        
        # Update freshness timestamp
        self._data_updated_at["trades"] = datetime.now()
        
        return data
    
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
        
        return freshness
        
    def get_orderbook_data(self) -> Dict[str, Any]:
        """
        Get order book data for the dashboard.
        
        Returns:
            Dictionary with order book data for all symbols
        """
        # In standalone mode, return sample data
        if self.orderbook_analyzer is None:
            # Generate sample orderbook data for common symbols
            symbols = {
                "BTCUSDT": self._generate_sample_orderbook("BTCUSDT"),
                "ETHUSDT": self._generate_sample_orderbook("ETHUSDT"),
                "SOLUSDT": self._generate_sample_orderbook("SOLUSDT"),
                "DOGEUSDT": self._generate_sample_orderbook("DOGEUSDT"),
                "BNBUSDT": self._generate_sample_orderbook("BNBUSDT")
            }
            
            # Generate sample WebSocket status
            websocket_status = {
                "BTCUSDT": True,
                "ETHUSDT": True,
                "SOLUSDT": True,
                "DOGEUSDT": False,
                "BNBUSDT": True
            }
            
            # Generate sample freshness data
            now = datetime.now()
            freshness = {
                "last_update": now.strftime("%Y-%m-%d %H:%M:%S"),
                "status": "fresh",
                "age_seconds": 0
            }
            
            orderbook_data = {
                "symbols": symbols,
                "websocket_status": websocket_status,
                "freshness": freshness
            }
            
            # Store in cache
            self._orderbook_data = orderbook_data
            
            # Update freshness timestamp
            self._data_updated_at["orderbook"] = now
            
            return orderbook_data
        
        # In connected mode, fetch from the orderbook analyzer
        try:
            # Get orderbook data from analyzer
            symbols_data = {}
            websocket_status = {}
            
            if self.orderbook_analyzer:
                # Get all symbols with orderbook data
                for symbol in self.orderbook_analyzer.get_tracked_symbols():
                    # Get orderbook for this symbol
                    orderbook = self.orderbook_analyzer.get_orderbook(symbol)
                    if orderbook:
                        symbols_data[symbol] = orderbook
                    
                    # Get connection status
                    is_connected = self.orderbook_analyzer.is_connected(symbol)
                    websocket_status[symbol] = is_connected
            
            # Generate freshness data
            now = datetime.now()
            freshness = {
                "last_update": now.strftime("%Y-%m-%d %H:%M:%S"),
                "status": "fresh",
                "age_seconds": 0
            }
            
            orderbook_data = {
                "symbols": symbols_data,
                "websocket_status": websocket_status,
                "freshness": freshness
            }
            
            # Store in cache
            self._orderbook_data = orderbook_data
            
            # Update freshness timestamp
            self._data_updated_at["orderbook"] = now
            
            return orderbook_data
            
        except Exception as e:
            logger.error(f"Error getting orderbook data: {e}")
            # Fall back to cached data or empty data
            if self._orderbook_data:
                return self._orderbook_data
            return {"symbols": {}, "websocket_status": {}, "freshness": {"status": "error"}}
    
    def _generate_sample_orderbook(self, symbol: str) -> Dict[str, Any]:
        """
        Generate sample orderbook data for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with sample orderbook data
        """
        # Set base price based on symbol
        if symbol == "BTCUSDT":
            base_price = np.random.uniform(60000, 70000)
        elif symbol == "ETHUSDT":
            base_price = np.random.uniform(3000, 3500)
        elif symbol == "SOLUSDT":
            base_price = np.random.uniform(120, 150)
        elif symbol == "DOGEUSDT":
            base_price = np.random.uniform(0.10, 0.15)
        elif symbol == "BNBUSDT":
            base_price = np.random.uniform(500, 600)
        else:
            base_price = 100.0
        
        # Generate bids (buy orders) - slightly below base price
        bids = []
        for i in range(10):
            price = base_price * (1 - 0.001 * (i + 1))
            size = np.random.uniform(0.1, 2.0)
            bids.append([round(price, 2), round(size, 4)])
        
        # Generate asks (sell orders) - slightly above base price
        asks = []
        for i in range(10):
            price = base_price * (1 + 0.001 * (i + 1))
            size = np.random.uniform(0.1, 2.0)
            asks.append([round(price, 2), round(size, 4)])
        
        # Calculate market metrics
        spread = asks[0][0] - bids[0][0]
        spread_pct = (spread / base_price) * 100
        
        # Calculate volume
        bid_volume = sum(bid[1] for bid in bids)
        ask_volume = sum(ask[1] for ask in asks)
        
        # Calculate imbalance
        imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
        
        return {
            "symbol": symbol,
            "timestamp": datetime.now().timestamp() * 1000,
            "bids": bids,
            "asks": asks,
            "best_bid": bids[0][0],
            "best_ask": asks[0][0],
            "spread": round(spread, 2),
            "spread_pct": round(spread_pct, 4),
            "mid_price": round((bids[0][0] + asks[0][0]) / 2, 2),
            "bid_volume": round(bid_volume, 4),
            "ask_volume": round(ask_volume, 4),
            "total_volume": round(bid_volume + ask_volume, 4),
            "imbalance": round(imbalance, 4)
        } 