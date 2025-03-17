#!/usr/bin/env python3
"""
Main Application Module for the Algorithmic Trading System

This module orchestrates all components of the trading system and provides the main entry point.
"""

import argparse
import json
import os
import sys
from pathlib import Path
import time
import signal
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from dotenv import load_dotenv
import pandas as pd

# Import system components
from src.config.config_manager import ConfigManager
from src.api.bybit_client import BybitClient
from src.indicators.indicator_manager import IndicatorManager
from src.strategies.strategy_manager import StrategyManager
from src.trade_management.trade_manager import TradeManager
from src.risk_management.risk_manager import RiskManager
from src.performance.performance_tracker import PerformanceTracker
from src.backtesting.backtest_engine import BacktestEngine


class TradingSystem:
    """Main trading system class that orchestrates all components."""

    def __init__(self, config_path: str, log_level: str = "INFO"):
        """
        Initialize the trading system.

        Args:
            config_path: Path to configuration file
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        # Set up logging
        self._setup_logging(log_level)

        logger.info(f"Initializing trading system with config: {config_path}")

        # Load configuration
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()

        # Initialize components
        self._initialize_components()

        # For graceful shutdown
        self.is_running = False
        self.shutdown_requested = False

        logger.info("Trading system initialized")

    def _setup_logging(self, log_level: str) -> None:
        """
        Set up logging configuration.

        Args:
            log_level: Logging level
        """
        # Remove default logger
        logger.remove()

        # Add console logger
        logger.add(
            sys.stdout,
            level=log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        )

        # Add file logger
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)

        logger.add(
            os.path.join(log_dir, f"trading_{datetime.now().strftime('%Y%m%d')}.log"),
            level=log_level,
            rotation="00:00",  # New file at midnight
            retention="30 days",  # Keep logs for 30 days
            compression="zip",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        )

    def _initialize_components(self) -> None:
        """Initialize all trading system components."""
        # Create API client
        api_config = self.config.get("exchange", {})
        self.api_client = BybitClient(
            api_key=api_config.get("api_key", ""),
            api_secret=api_config.get("api_secret", ""),
            testnet=api_config.get("testnet", True),
        )

        # Initialize market data module
        self.market_data = BybitClient(
            api_key=api_config.get("api_key", ""),
            api_secret=api_config.get("api_secret", ""),
            testnet=api_config.get("testnet", True),
        )

        # Initialize indicator manager
        self.indicator_manager = IndicatorManager()

        # Initialize strategy manager
        strategy_config = self.config.get("strategies", {})
        self.strategy_manager = StrategyManager(self.config, self.indicator_manager)

        # Initialize risk manager
        risk_config = self.config.get("risk_management", {})
        self.risk_manager = RiskManager(risk_config)

        # Initialize trade manager
        self.trade_manager = TradeManager(
            api_client=self.api_client,
            risk_manager=self.risk_manager,
            simulate=self.config.get("simulation_mode", True),
        )

        # Initialize performance tracker
        performance_config = self.config.get("performance", {})
        self.performance_tracker = PerformanceTracker(
            initial_balance=performance_config.get("initial_balance", 10000.0),
            data_directory=performance_config.get("data_directory", "data/performance"),
        )

        # Initialize backtest engine (if needed)
        self.backtest_engine = None
        if self.config.get("backtest", {}).get("enabled", False):
            self.backtest_engine = BacktestEngine(
                config=self.config,
                market_data=self.market_data,
                indicator_manager=self.indicator_manager,
                strategy_manager=self.strategy_manager,
            )

    def start(self) -> None:
        """Start the trading system."""
        logger.info("Starting trading system")

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Determine operating mode
        if self.config.get("backtest", {}).get("enabled", False):
            self._run_backtest()
        elif self.config.get("paper_trading", False):
            self._run_paper_trading()
        else:
            self._run_live_trading()

    def _signal_handler(self, sig, frame) -> None:
        """
        Handle system signals for graceful shutdown.

        Args:
            sig: Signal number
            frame: Current stack frame
        """
        if not self.shutdown_requested:
            logger.info("Shutdown signal received, gracefully shutting down...")
            self.shutdown_requested = True
        else:
            logger.warning("Force shutdown requested, exiting immediately!")
            sys.exit(1)

    def _run_backtest(self) -> None:
        """Run the system in backtest mode."""
        logger.info("Starting backtest")

        backtest_config = self.config.get("backtest", {})

        # Get backtest parameters
        symbols = backtest_config.get("symbols", [])
        start_date = backtest_config.get("start_date", "2022-01-01")
        end_date = backtest_config.get("end_date", datetime.now().strftime("%Y-%m-%d"))
        timeframe = backtest_config.get("timeframe", "1h")

        # Run backtest
        results = self.backtest_engine.run_backtest(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe,
            enable_progress_bar=True,
        )

        # Print summary
        self._print_backtest_summary(results)

        logger.info("Backtest completed")

    def _print_backtest_summary(self, results: Dict[str, Any]) -> None:
        """
        Print backtest summary.

        Args:
            results: Backtest results
        """
        summary = results.get("summary", {})

        print("\n" + "=" * 60)
        print(" " * 20 + "BACKTEST SUMMARY")
        print("=" * 60)
        print(f"Initial Balance: ${summary.get('initial_balance', 0):.2f}")
        print(f"Final Balance: ${summary.get('final_balance', 0):.2f}")
        print(f"Total Return: {summary.get('total_return', 0):.2f}%")
        print(f"Total Trades: {summary.get('total_trades', 0)}")
        print(f"Win Rate: {summary.get('win_rate', 0):.2f}%")
        print(f"Profit Factor: {summary.get('profit_factor', 0):.2f}")
        print(f"Max Drawdown: {summary.get('max_drawdown', 0):.2f}%")
        print("=" * 60)

        # Print strategy performance
        strategy_metrics = results.get("strategy_metrics", {})
        if strategy_metrics:
            print("\nStrategy Performance:")
            print("-" * 60)
            for strategy, metrics in strategy_metrics.items():
                print(f"Strategy: {strategy}")
                print(f"  Win Rate: {metrics.get('win_rate', 0) * 100:.2f}%")
                print(f"  Total P&L: ${metrics.get('total_profit_loss', 0):.2f}")
                print(f"  Signals Executed: {metrics.get('signals_executed', 0)}")
                print("-" * 40)

        print("\nDetailed results saved to:", self.backtest_engine.output_dir)

    def _run_paper_trading(self) -> None:
        """Run the system in paper trading mode."""
        logger.info("Starting paper trading mode")

        # Set simulation flag
        self.trade_manager.simulate = True

        # Run trading loop
        self._run_trading_loop(is_paper_trading=True)

    def _run_live_trading(self) -> None:
        """Run the system in live trading mode."""
        logger.info("Starting live trading mode")

        # Set simulation flag
        self.trade_manager.simulate = False

        # Confirm before starting live trading
        if not self._confirm_live_trading():
            logger.info("Live trading canceled by user")
            return

        # Run trading loop
        self._run_trading_loop(is_paper_trading=False)

    def _confirm_live_trading(self) -> bool:
        """
        Ask for confirmation before starting live trading.

        Returns:
            True if confirmed, False otherwise
        """
        print("\n" + "!" * 80)
        print("! WARNING: You are about to start LIVE TRADING with REAL MONEY !")
        print("!" * 80)

        # Get account balance
        try:
            account_info = self.api_client.get_account_info()
            balance = account_info.get("walletBalance", 0)
            print(f"\nCurrent account balance: ${float(balance):.2f}")
        except Exception as e:
            logger.error(f"Failed to get account balance: {e}")
            print("\nCould not retrieve account balance. Please check API connection.")

        # Ask for confirmation
        response = input("\nAre you sure you want to start live trading? (yes/no): ")
        return response.lower() in ("yes", "y")

    def _run_trading_loop(self, is_paper_trading: bool) -> None:
        """
        Run the main trading loop.

        Args:
            is_paper_trading: Whether running in paper trading mode
        """
        # Get trading parameters
        trading_config = self.config.get("trading", {})
        symbols = trading_config.get("symbols", [])
        timeframe = trading_config.get("timeframe", "1h")
        update_interval = trading_config.get("update_interval_seconds", 60)

        if not symbols:
            logger.error("No symbols specified in configuration")
            return

        logger.info(f"Trading {len(symbols)} symbols: {', '.join(symbols)}")
        logger.info(
            f"Timeframe: {timeframe}, Update interval: {update_interval} seconds"
        )

        # Initialize market data for all symbols
        for symbol in symbols:
            self.market_data.initialize_symbol(symbol, timeframe)

        # Set running flag
        self.is_running = True
        self.shutdown_requested = False

        # Main trading loop
        while self.is_running and not self.shutdown_requested:
            try:
                logger.debug("Starting trading iteration")

                # Get account information
                account_info = self.api_client.get_account_info()
                account_balance = float(account_info.get("walletBalance", 0))

                # Current market data and unrealized PnL
                unrealized_pnl = 0.0
                current_market_data = {}

                # Process each symbol
                for symbol in symbols:
                    try:
                        # Get current market data
                        market_data = self.market_data.get_latest_data(
                            symbol, timeframe, bars=100
                        )

                        if market_data is None or market_data.empty:
                            logger.warning(f"No data available for {symbol}, skipping")
                            continue

                        # Apply indicators
                        market_data = self.indicator_manager.apply_indicators(
                            market_data
                        )

                        # Store current data
                        current_market_data[symbol] = {
                            "price": market_data.iloc[-1]["close"],
                            "time": market_data.index[-1],
                        }

                        # Generate signals
                        signals = self.strategy_manager.generate_signals(market_data)

                        if signals:
                            for signal in signals:
                                logger.info(
                                    f"Signal generated for {symbol}: {signal.signal_type.name} - strength: {signal.strength:.2f}"
                                )

                                # Process signal
                                trade_id = self.trade_manager.process_signal(signal)

                                if trade_id:
                                    trade = self.trade_manager.get_trade_by_id(trade_id)
                                    self.performance_tracker.add_trade(trade)

                        # Update active trades
                        self.trade_manager.update_active_trades(current_market_data)

                    except Exception as e:
                        logger.error(f"Error processing symbol {symbol}: {e}")

                # Update performance tracker
                self.performance_tracker.update_balance(account_balance, unrealized_pnl)

                # Save performance metrics periodically
                if (
                    datetime.now().minute % 15 == 0
                    and datetime.now().second < update_interval
                ):
                    self._save_performance_metrics()

                # Sleep until next update
                logger.debug(
                    f"Sleeping for {update_interval} seconds until next update"
                )

                # Check for shutdown every second
                for _ in range(update_interval):
                    if self.shutdown_requested:
                        break
                    time.sleep(1)

            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                # Sleep before retrying
                time.sleep(10)

        # Shutdown procedure
        self._shutdown()

    def _save_performance_metrics(self) -> None:
        """Save performance metrics to file."""
        try:
            # Generate report timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")

            # Save performance report
            self.performance_tracker.save_performance_report(f"report_{timestamp}")

            # Save strategy performance
            strategy_file = os.path.join(
                self.performance_tracker.data_directory,
                f"strategy_performance_{timestamp}.json",
            )
            self.strategy_manager.save_performance(strategy_file)

            logger.info(f"Performance metrics saved: {timestamp}")

        except Exception as e:
            logger.error(f"Error saving performance metrics: {e}")

    def _shutdown(self) -> None:
        """Perform cleanup and shutdown operations."""
        logger.info("Shutting down trading system")

        # Save final performance metrics
        self._save_performance_metrics()

        # Display summary
        summary = self.performance_tracker.generate_performance_summary()
        print("\n" + "=" * 60)
        print(summary)
        print("=" * 60)

        logger.info("Trading system shutdown complete")
        self.is_running = False


def main():
    """Main entry point for the trading system."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Cryptocurrency Algorithmic Trading System"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--log-level",
        "-l",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--backtest", "-b", action="store_true", help="Run in backtest mode"
    )

    args = parser.parse_args()

    # Create and start the trading system
    trading_system = TradingSystem(args.config, args.log_level)

    # Override backtest flag if specified in command line
    if args.backtest:
        trading_system.config["backtest"]["enabled"] = True

    # Start the system
    trading_system.start()


if __name__ == "__main__":
    main()
