"""
Backtesting Engine for the Algorithmic Trading System

This module provides the core BacktestEngine class for backtesting trading strategies on historical data.
"""

import os
import time
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm
from loguru import logger

from src.config.config_manager import ConfigManager
from src.api.bybit.client import BybitClient
from src.indicators.indicator_manager import IndicatorManager
from src.strategies.strategy_manager import StrategyManager
from src.trade_management.trade_manager import TradeManager
from src.risk_management.risk_manager import RiskManager
from src.performance.performance_tracker import PerformanceTracker
from src.models.models import Signal

# Import utility modules and components
from src.backtesting.backtest.utils import (
    timeframe_to_minutes,
    timeframe_to_pandas_freq,
    calculate_unrealized_pnl_pct,
)
from src.backtesting.backtest.trade_execution import execute_signal
from src.backtesting.backtest.position_management import (
    process_positions,
    close_position,
    calculate_equity,
)
from src.backtesting.backtest.results_processing import (
    generate_results,
    save_results,
    generate_charts,
)


class BacktestEngine:
    """Backtesting engine for evaluating trading strategies."""

    def __init__(
        self,
        config_manager: Union[Dict[str, Any], ConfigManager],
        market_data: Optional[BybitClient] = None,
        indicator_manager: Optional[IndicatorManager] = None,
        strategy_manager: Optional[StrategyManager] = None,
        output_dir: str = "data/backtest_results",
    ):
        """
        Initialize the backtesting engine.

        Args:
            config_manager: Configuration manager instance or dictionary
            market_data: Bybit client instance (optional)
            indicator_manager: Indicator manager instance (optional)
            strategy_manager: Strategy manager instance (optional)
            output_dir: Directory to save backtest results
        """
        # Set up configuration
        if isinstance(config_manager, ConfigManager):
            self.config_manager = config_manager
            self.config = config_manager.get_config()
        else:
            # Backward compatibility for dictionary input
            self.config = config_manager
            try:
                # Try to create a config manager from the dictionary
                # This is not ideal but maintains backward compatibility
                self.config_manager = ConfigManager(Path("config/default_config.json"))
                logger.warning(
                    "Creating a ConfigManager from default config because dictionary was provided"
                )
            except Exception as e:
                logger.error(f"Error creating ConfigManager: {e}")
                self.config_manager = None

        # Check if config is a Pydantic model and convert to dict as needed
        if (
            hasattr(self.config, "model_dump")
            and callable(getattr(self.config, "model_dump"))
            or hasattr(self.config, "dict")
            and callable(getattr(self.config, "dict"))
        ):
            self.config_dict = self._get_config_dict(self.config)
        else:
            self.config_dict = self.config  # Already a dict

        # Check if config is a Pydantic model
        self.is_pydantic = hasattr(self.config, "dict") and callable(
            getattr(self.config, "dict")
        )

        # Initialize components if not provided
        if self.is_pydantic:
            # Handle Pydantic model
            exchange_config = getattr(self.config, "exchange", {})
            self.market_data = market_data or BybitClient(
                testnet=getattr(exchange_config, "testnet", True),
                api_key=getattr(exchange_config, "api_key", ""),
                api_secret=getattr(exchange_config, "api_secret", ""),
                data_dir=getattr(self.config, "data_dir", "data"),
            )
        else:
            # Handle dictionary
            self.market_data = market_data or BybitClient(
                testnet=self.config.get("testnet", True),
                api_key=self.config.get("api_key", ""),
                api_secret=self.config.get("api_secret", ""),
                data_dir=self.config.get("data_dir", "data"),
            )

        self.indicator_manager = indicator_manager or IndicatorManager()

        # Initialize strategy manager if not provided
        if strategy_manager is None:
            self.strategy_manager = StrategyManager(self.config, self.indicator_manager)
        else:
            self.strategy_manager = strategy_manager

        # Set up output directory
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Backtest configuration
        if self.is_pydantic:
            # Handle Pydantic model
            backtest_config = getattr(self.config, "backtest", {})
            self.initial_balance = getattr(backtest_config, "initial_balance", 10000.0)
            self.commission_rate = getattr(backtest_config, "commission_rate", 0.001)
            self.slippage = getattr(backtest_config, "slippage", 0.0005)
        else:
            # Handle dictionary
            self.backtest_config = self.config.get("backtest", {})
            self.initial_balance = self.backtest_config.get("initial_balance", 10000.0)
            self.commission_rate = self.backtest_config.get("commission_rate", 0.001)
            self.slippage = self.backtest_config.get("slippage", 0.0005)

        # Results storage
        self.trades: List[Dict[str, Any]] = []
        self.equity_curve: List[Dict[str, Any]] = []
        self.signals: List[Dict[str, Any]] = []

        # For tracking current position and balance
        self.current_positions: Dict[str, Dict[str, Any]] = {}
        self.current_balance = self.initial_balance

        logger.info(
            f"Backtest engine initialized with {len(self.strategy_manager.get_enabled_strategies())} strategies"
        )

    def _get_config_dict(self, config) -> Dict[str, Any]:
        """
        Convert Pydantic model config to a dictionary for compatibility.

        Args:
            config: Configuration object (Pydantic model or dict)

        Returns:
            Dictionary representation of the config
        """
        if hasattr(config, "model_dump") and callable(getattr(config, "model_dump")):
            # Pydantic v2
            return config.model_dump()
        elif hasattr(config, "dict") and callable(getattr(config, "dict")):
            # Pydantic v1
            return config.dict()
        return config  # Already a dict

    def run_backtest(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        timeframe: str = "1h",
        warmup_bars: int = 100,
        enable_progress_bar: bool = True,
    ) -> Dict[str, Any]:
        """
        Run a backtest for the given symbols and time period.

        Args:
            symbols: List of symbols to test
            start_date: Start date for the backtest (YYYY-MM-DD)
            end_date: End date for the backtest (YYYY-MM-DD)
            timeframe: Timeframe to use (1m, 5m, 15m, 1h, 4h, 1d)
            warmup_bars: Number of warmup bars to include before start date
            enable_progress_bar: Whether to display progress bar

        Returns:
            Dictionary with backtest results
        """
        logger.info(
            f"Starting backtest for {len(symbols)} symbols from {start_date} to {end_date}"
        )

        # Initialize risk manager and performance tracker
        if self.is_pydantic:
            # Handle Pydantic model
            risk_config = None
            if hasattr(self.config, "risk_management"):
                risk_config = self.config.risk_management
            elif hasattr(self.config, "risk"):
                risk_config = self.config.risk
            else:
                risk_config = {}
        else:
            # Handle dictionary
            risk_config = self.config.get(
                "risk_management", self.config.get("risk", {})
            )

        risk_manager = RiskManager(risk_config)

        performance_tracker = PerformanceTracker(
            initial_balance=self.initial_balance,
            data_directory=os.path.join(self.output_dir, "performance"),
        )

        # Initialize trade manager
        trade_manager = TradeManager(
            api_client=None,  # Not used in backtest
            risk_manager=risk_manager,
            simulate=True,
        )

        # Reset state
        self.trades = []
        self.equity_curve = []
        self.signals = []
        self.current_positions = {}
        self.current_balance = self.initial_balance

        # Initialize equity curve with starting balance
        self.equity_curve.append(
            {
                "timestamp": pd.to_datetime(start_date),
                "balance": self.initial_balance,
                "equity": self.initial_balance,
                "drawdown_pct": 0.0,
            }
        )

        # Load historical data for all symbols
        historical_data = {}

        for symbol in symbols:
            # Load data with warmup period
            start_date_with_warmup = (
                pd.to_datetime(start_date)
                - pd.Timedelta(
                    days=int(warmup_bars * timeframe_to_minutes(timeframe) / (60 * 24))
                )
            ).strftime("%Y-%m-%d")

            try:
                df = self.market_data.data.fetch_historical_klines(
                    symbol=symbol,
                    interval=timeframe,
                    start_time=start_date_with_warmup,
                    end_time=end_date,
                    use_cache=True,
                )

                if df is not None and not df.empty:
                    # Apply indicators to data
                    df = self.indicator_manager.apply_indicators(df)
                    historical_data[symbol] = df
                    logger.info(f"Loaded {len(df)} bars for {symbol}")
                else:
                    logger.warning(f"No data available for {symbol} - skipping")

            except Exception as e:
                logger.error(f"Error loading data for {symbol}: {e}")

        if not historical_data:
            logger.error("No valid data available for any symbols")
            return {"success": False, "error": "No valid data available"}

        # Find common date range across all symbols
        earliest_start = min(df.index[warmup_bars] for df in historical_data.values())
        latest_end = max(df.index[-1] for df in historical_data.values())

        # Create a common date range
        date_range = pd.date_range(
            start=max(pd.to_datetime(start_date), earliest_start),
            end=min(pd.to_datetime(end_date), latest_end),
            freq=timeframe_to_pandas_freq(timeframe),
        )

        # Set up progress bar if enabled
        if enable_progress_bar:
            pbar = tqdm(total=len(date_range), desc="Backtesting")

        # Iterate through each date in the range
        for current_time in date_range:
            # Update progress bar
            if enable_progress_bar:
                pbar.update(1)

            # Process each symbol
            for symbol, df in historical_data.items():
                # Find the row for the current time
                if current_time not in df.index:
                    continue

                current_idx = df.index.get_loc(current_time)

                # Skip if we don't have enough data
                if current_idx < warmup_bars:
                    continue

                # Get data up to the current bar
                current_data = df.iloc[: current_idx + 1]
                current_bar = current_data.iloc[-1]

                # Process active positions
                new_balance = process_positions(
                    symbol=symbol,
                    current_time=current_time,
                    current_data=current_data,
                    risk_manager=risk_manager,
                    performance_tracker=performance_tracker,
                    current_positions=self.current_positions,
                    trades=self.trades,
                    slippage=self.slippage,
                    commission_rate=self.commission_rate,
                )

                if new_balance is not None:
                    self.current_balance = new_balance

                # Generate signals from strategies
                signals = self.strategy_manager.generate_signals(current_data)

                # Record signals
                for signal in signals:
                    self.signals.append(signal.to_dict())

                # Execute signals if there's no current position
                if symbol not in self.current_positions and signals:
                    for signal in signals:
                        # Skip signals for other symbols
                        if signal.symbol != symbol:
                            continue

                        # Create a trade from the signal
                        result = execute_signal(
                            signal=signal,
                            current_time=current_time,
                            current_data=current_data,
                            risk_manager=risk_manager,
                            strategy_manager=self.strategy_manager,
                            current_positions=self.current_positions,
                            current_balance=self.current_balance,
                            slippage=self.slippage,
                            commission_rate=self.commission_rate,
                            trades=self.trades,
                        )

                        if result:
                            trade_id, new_balance = result
                            self.current_balance = new_balance
                            logger.info(
                                f"Executed {signal.signal_type.name} trade for {symbol} at {signal.price}"
                            )

            # Calculate equity at current time
            current_equity = calculate_equity(
                current_time=current_time,
                historical_data=historical_data,
                current_balance=self.current_balance,
                current_positions=self.current_positions,
            )

            # Update equity curve
            max_equity = max([point["equity"] for point in self.equity_curve])
            drawdown_pct = (
                ((max_equity - current_equity) / max_equity) * 100
                if max_equity > 0
                else 0
            )

            self.equity_curve.append(
                {
                    "timestamp": current_time,
                    "balance": self.current_balance,
                    "equity": current_equity,
                    "drawdown_pct": drawdown_pct,
                }
            )

            # Update performance tracker
            performance_tracker.update_balance(
                self.current_balance, current_equity - self.current_balance
            )

        # Close the progress bar if enabled
        if enable_progress_bar:
            pbar.close()

        # Close any remaining positions at the end of the backtest
        for symbol, position in list(self.current_positions.items()):
            new_balance = close_position(
                symbol=symbol,
                current_time=date_range[-1],
                current_data=historical_data[symbol],
                exit_reason="end_of_backtest",
                risk_manager=risk_manager,
                performance_tracker=performance_tracker,
                current_positions=self.current_positions,
                trades=self.trades,
                slippage=self.slippage,
                commission_rate=self.commission_rate,
                strategy_manager=self.strategy_manager,
            )

            if new_balance is not None:
                self.current_balance = new_balance

        # Generate results
        results = generate_results(
            trades=self.trades,
            equity_curve=self.equity_curve,
            initial_balance=self.initial_balance,
            current_balance=self.current_balance,
            performance_tracker=performance_tracker,
            strategy_performance=self.strategy_manager.strategy_performance,
            commission_rate=self.commission_rate,
            slippage=self.slippage,
        )

        # Add signals to results
        results["signals"] = self.signals

        # Save results
        save_results(results, self.output_dir)

        logger.info(f"Backtest completed with {len(self.trades)} trades")
        return results
