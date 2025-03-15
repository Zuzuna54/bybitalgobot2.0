"""
Paper Trading Simulator for the Algorithmic Trading System

This module provides the main PaperTradingSimulator class that manages paper trading
functionality using components for order processing, position management, trade execution,
and state management.
"""

from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import time
import os
import threading
from pathlib import Path

from loguru import logger

from src.api.bybit_client import BybitClient
from src.trade_management.trade_manager import TradeManager
from src.performance.performance_tracker import PerformanceTracker
from src.risk_management.risk_manager import RiskManager
from src.data.market_data import MarketData
from src.strategies.strategy_manager import StrategyManager
from src.paper_trading.components import (
    # Order processor
    process_pending_orders,
    calculate_execution_price,
    
    # Position manager
    update_positions,
    close_position,
    calculate_total_equity,
    
    # Execution engine
    execute_paper_trade,
    get_market_data,
    process_strategy_signals,
    
    # State manager
    save_state,
    load_state,
    get_summary,
    compare_to_backtest
)


class PaperTradingSimulator:
    """Simulates trading in a paper trading environment."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        initial_balance: float = 10000.0,
        data_dir: str = "data/paper_trading",
        slippage: float = 0.05,  # 0.05% slippage
        commission: float = 0.075,  # 0.075% fee
        latency_ms: int = 100  # Simulated latency in milliseconds
    ):
        """
        Initialize the paper trading simulator.
        
        Args:
            config: Configuration dictionary
            initial_balance: Initial account balance in USDT
            data_dir: Directory to store paper trading data
            slippage: Simulated slippage in percentage
            commission: Simulated commission in percentage
            latency_ms: Simulated latency in milliseconds
        """
        self.config = config
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.data_dir = data_dir
        self.slippage = slippage / 100.0  # Convert to decimal
        self.commission = commission / 100.0  # Convert to decimal
        self.latency_ms = latency_ms
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize components
        self.api_client = None
        self.market_data = None
        self.trade_manager = None
        self.risk_manager = None
        self.performance_tracker = None
        self.strategy_manager = None
        
        # Trading state
        self.active_positions: Dict[str, Dict[str, Any]] = {}
        self.pending_orders: Dict[str, Dict[str, Any]] = {}
        self.trade_history: List[Dict[str, Any]] = []
        self.equity_history: List[Dict[str, float]] = []
        
        # Runtime state
        self.is_running = False
        self.stop_event = threading.Event()
        self.last_update_time = datetime.now()
        
        logger.info(f"Paper trading simulator initialized with balance: ${initial_balance}")
    
    def setup(
        self,
        api_client: BybitClient,
        market_data: MarketData,
        risk_manager: RiskManager,
        strategy_manager: StrategyManager
    ) -> None:
        """
        Set up the simulator components.
        
        Args:
            api_client: Bybit API client
            market_data: Market data manager
            risk_manager: Risk manager
            strategy_manager: Strategy manager
        """
        self.api_client = api_client
        self.market_data = market_data
        self.risk_manager = risk_manager
        self.strategy_manager = strategy_manager
        
        # Initialize specialized instances for paper trading
        self.trade_manager = TradeManager(
            api_client=api_client,
            risk_manager=risk_manager,
            simulate=True  # Enable simulation mode in trade manager
        )
        
        self.performance_tracker = PerformanceTracker(
            initial_balance=self.initial_balance,
            data_directory=self.data_dir
        )
        
        logger.info("Paper trading simulator components set up successfully")
    
    def start(self, symbols: List[str], update_interval_sec: int = 5) -> None:
        """
        Start paper trading simulation.
        
        Args:
            symbols: List of trading pair symbols
            update_interval_sec: Simulator update interval in seconds
        """
        if self.is_running:
            logger.warning("Paper trading simulator is already running")
            return
        
        if not self.api_client or not self.market_data or not self.trade_manager:
            logger.error("Paper trading simulator not fully set up. Call setup() first.")
            return
        
        self.is_running = True
        self.stop_event.clear()
        
        # Initialize data streams for all symbols
        for symbol in symbols:
            self.market_data.start_ticker_stream(symbol)
            self.market_data.start_klines_stream(symbol, self.config.get('default_timeframe', '1h'))
            self.market_data.start_orderbook_stream(symbol)
        
        # Load existing state if available
        self._load_state()
        
        # Start simulator thread
        simulator_thread = threading.Thread(
            target=self._simulation_loop,
            args=(symbols, update_interval_sec),
            daemon=True
        )
        simulator_thread.start()
        
        logger.info(f"Paper trading simulator started for symbols: {symbols}")
    
    def stop(self) -> None:
        """Stop the paper trading simulation."""
        if not self.is_running:
            logger.warning("Paper trading simulator is not running")
            return
        
        self.is_running = False
        self.stop_event.set()
        
        # Save current state
        self._save_state()
        
        logger.info("Paper trading simulator stopped")
    
    def _simulation_loop(self, symbols: List[str], update_interval_sec: int) -> None:
        """
        Main simulation loop for paper trading.
        
        Args:
            symbols: List of trading pair symbols
            update_interval_sec: Simulator update interval in seconds
        """
        while self.is_running and not self.stop_event.is_set():
            try:
                current_time = datetime.now()
                
                # Process signals from strategies
                new_balance = process_strategy_signals(
                    symbols=symbols,
                    market_data=self.market_data,
                    strategy_manager=self.strategy_manager,
                    risk_manager=self.risk_manager,
                    performance_tracker=self.performance_tracker,
                    active_positions=self.active_positions,
                    trade_history=self.trade_history,
                    current_balance=self.current_balance,
                    slippage=self.slippage,
                    commission_rate=self.commission,
                    latency_ms=self.latency_ms
                )
                
                if new_balance is not None:
                    self.current_balance = new_balance
                
                # Process pending orders
                process_pending_orders(
                    pending_orders=self.pending_orders,
                    get_current_price_func=self.market_data.get_current_price,
                    close_position_func=self._close_position,
                    execute_order_func=self._execute_order
                )
                
                # Update position valuations
                update_positions(
                    active_positions=self.active_positions,
                    get_current_price_func=self.market_data.get_current_price,
                    close_position_func=self._close_position,
                    risk_manager=self.risk_manager
                )
                
                # Update performance tracker
                self._update_performance()
                
                # Periodically save state
                if (current_time - self.last_update_time).total_seconds() > 60:
                    self._save_state()
                    self.last_update_time = current_time
                
                # Wait for next update interval
                time.sleep(update_interval_sec)
                
            except Exception as e:
                logger.error(f"Error in paper trading simulation loop: {e}")
                time.sleep(update_interval_sec)
    
    def _close_position(self, symbol: str, exit_price: float, exit_reason: str) -> None:
        """
        Close a position and calculate results.
        
        Args:
            symbol: Symbol to close
            exit_price: Exit price
            exit_reason: Reason for exiting the position
        """
        new_balance = close_position(
            symbol=symbol,
            exit_price=exit_price,
            exit_reason=exit_reason,
            active_positions=self.active_positions,
            trade_history=self.trade_history,
            commission_rate=self.commission,
            risk_manager=self.risk_manager,
            performance_tracker=self.performance_tracker,
            strategy_manager=self.strategy_manager
        )
        
        if new_balance is not None:
            self.current_balance += new_balance
    
    def _execute_order(self, order: Dict[str, Any], price: float) -> None:
        """
        Execute a pending order.
        
        Args:
            order: Order details
            price: Execution price
        """
        # Implementation depends on order type (market, limit, etc.)
        # This is a placeholder for now
        pass
    
    def _update_performance(self) -> None:
        """Update performance metrics with current valuation."""
        # Calculate total equity
        total_equity = calculate_total_equity(
            current_balance=self.current_balance,
            active_positions=self.active_positions
        )
        
        # Calculate unrealized PnL
        unrealized_pnl = total_equity - self.current_balance
        
        # Add to equity history
        self.equity_history.append({
            "timestamp": datetime.now(),
            "balance": self.current_balance,
            "equity": total_equity,
            "unrealized_pnl": unrealized_pnl
        })
        
        # Update performance tracker
        self.performance_tracker.update_balance(self.current_balance, unrealized_pnl)
    
    def _save_state(self) -> None:
        """Save current state to disk."""
        save_state(
            data_dir=self.data_dir,
            current_balance=self.current_balance,
            active_positions=self.active_positions,
            pending_orders=self.pending_orders,
            trade_history=self.trade_history,
            equity_history=self.equity_history
        )
    
    def _load_state(self) -> None:
        """Load saved state from disk if available."""
        state = load_state(
            data_dir=self.data_dir,
            initial_balance=self.initial_balance
        )
        
        self.current_balance = state["current_balance"]
        self.active_positions = state["active_positions"]
        self.pending_orders = state["pending_orders"]
        self.trade_history = state["trade_history"]
        self.equity_history = state.get("equity_history", [])
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current paper trading status.
        
        Returns:
            Dictionary with current paper trading summary
        """
        return get_summary(
            initial_balance=self.initial_balance,
            current_balance=self.current_balance,
            active_positions=self.active_positions,
            pending_orders=self.pending_orders,
            trade_history=self.trade_history
        )
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate a detailed performance report.
        
        Returns:
            Dictionary with detailed performance metrics
        """
        if self.performance_tracker:
            return self.performance_tracker.generate_full_performance_report()
        return {}
    
    def compare_to_backtest(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare paper trading results to backtest results.
        
        Args:
            backtest_results: Results from backtesting
            
        Returns:
            Dictionary with comparison metrics
        """
        paper_performance_report = self.get_performance_report()
        return compare_to_backtest(paper_performance_report, backtest_results) 