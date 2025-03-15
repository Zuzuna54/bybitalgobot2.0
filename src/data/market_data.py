"""
Market Data Management for the Algorithmic Trading System

This module handles retrieving, processing, and storing market data
from the Bybit exchange, including real-time and historical data.
"""

import os
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Dict, List, Any, Optional, Callable, Union

import numpy as np
import pandas as pd
from loguru import logger

from src.api.bybit_client import BybitClient
from src.config.config_manager import SystemConfig, PairConfig


class MarketData:
    """Class for managing market data retrieval and processing."""
    
    def __init__(self, config: SystemConfig, bybit_client: BybitClient):
        """
        Initialize the market data manager.
        
        Args:
            config: System configuration
            bybit_client: Initialized Bybit API client
        """
        self.config = config
        self.client = bybit_client
        
        # Create data directory if it doesn't exist
        self.data_dir = Path(config.data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize data caches
        self.klines_cache: Dict[str, pd.DataFrame] = {}
        self.tickers_cache: Dict[str, Dict[str, Any]] = {}
        self.orderbooks_cache: Dict[str, Dict[str, Any]] = {}
        
        # Locks for thread safety
        self.klines_lock = Lock()
        self.tickers_lock = Lock()
        self.orderbooks_lock = Lock()
        
        # Callbacks for data updates
        self.klines_callbacks: Dict[str, List[Callable]] = {}
        self.tickers_callbacks: Dict[str, List[Callable]] = {}
        
        logger.info("Market data manager initialized")
    
    # Historical Data Methods
    
    def fetch_historical_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[Union[int, datetime, str]] = None,
        end_time: Optional[Union[int, datetime, str]] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch historical klines/candlestick data for a symbol.
        
        Args:
            symbol: Trading pair symbol
            interval: Kline interval
            start_time: Start time (can be timestamp, datetime, or ISO string)
            end_time: End time (can be timestamp, datetime, or ISO string)
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with historical klines data
        """
        # Convert datetime objects to timestamps if needed
        if start_time and isinstance(start_time, datetime):
            start_time = int(start_time.timestamp() * 1000)
        elif start_time and isinstance(start_time, str):
            start_time = int(datetime.fromisoformat(start_time.replace('Z', '+00:00')).timestamp() * 1000)
            
        if end_time and isinstance(end_time, datetime):
            end_time = int(end_time.timestamp() * 1000)
        elif end_time and isinstance(end_time, str):
            end_time = int(datetime.fromisoformat(end_time.replace('Z', '+00:00')).timestamp() * 1000)
        
        # Set default times if not provided
        if not end_time:
            end_time = int(time.time() * 1000)
            
        if not start_time:
            # Default to 1000 candles before end_time
            interval_ms = self._interval_to_milliseconds(interval)
            start_time = end_time - (interval_ms * 1000)
        
        # Check for cached data
        cache_key = f"{symbol}_{interval}"
        if use_cache and cache_key in self.klines_cache:
            cached_df = self.klines_cache[cache_key]
            cached_start = cached_df.index.min().timestamp() * 1000
            cached_end = cached_df.index.max().timestamp() * 1000
            
            # If cached data fully covers the requested range, return it
            if cached_start <= start_time and cached_end >= end_time:
                mask = (cached_df.index >= datetime.fromtimestamp(start_time / 1000)) & \
                       (cached_df.index <= datetime.fromtimestamp(end_time / 1000))
                return cached_df[mask].copy()
        
        # Calculate how many candles we need to fetch
        interval_ms = self._interval_to_milliseconds(interval)
        total_candles = (end_time - start_time) // interval_ms + 1
        
        # Bybit API limits to 200 candles per request, so we may need multiple requests
        max_candles_per_request = 200
        all_klines = []
        
        current_start = start_time
        remaining_candles = total_candles
        
        while remaining_candles > 0:
            # Calculate how many candles to request in this batch
            candles_to_fetch = min(remaining_candles, max_candles_per_request)
            
            # Fetch klines
            try:
                klines_data = self.client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=candles_to_fetch,
                    start_time=current_start
                )
                
                if not klines_data:
                    logger.warning(f"No klines data returned for {symbol} {interval}")
                    break
                
                all_klines.extend(klines_data)
                
                # Update for next iteration
                last_kline_time = int(klines_data[-1]['timestamp'])
                current_start = last_kline_time + interval_ms
                remaining_candles -= len(klines_data)
                
                # If we got fewer candles than requested, we've reached the end
                if len(klines_data) < candles_to_fetch:
                    break
                
                # Avoid rate limits
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error fetching klines for {symbol} {interval}: {e}")
                break
        
        # Convert to DataFrame
        if not all_klines:
            logger.warning(f"No historical klines data retrieved for {symbol} {interval}")
            return pd.DataFrame()
        
        # Parse klines data into DataFrame
        df = self._parse_klines_to_dataframe(all_klines)
        
        # Update cache
        with self.klines_lock:
            self.klines_cache[cache_key] = df
        
        # Save to disk if we fetched significant data
        if len(df) > 100:
            self._save_klines_to_disk(symbol, interval, df)
        
        return df
    
    def _parse_klines_to_dataframe(self, klines_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Parse klines data from Bybit API into a pandas DataFrame.
        
        Args:
            klines_data: List of klines data from Bybit API
            
        Returns:
            Formatted DataFrame with klines data
        """
        df = pd.DataFrame(klines_data)
        
        # Convert timestamp to datetime and use as index
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Convert numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'turnover']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col])
        
        return df
    
    def _save_klines_to_disk(self, symbol: str, interval: str, df: pd.DataFrame) -> None:
        """
        Save klines data to disk for later use.
        
        Args:
            symbol: Trading pair symbol
            interval: Kline interval
            df: DataFrame containing klines data
        """
        # Create directory structure
        symbol_dir = self.data_dir / 'klines' / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        file_path = symbol_dir / f"{interval}.csv"
        df.to_csv(file_path)
        
        logger.info(f"Saved {len(df)} klines for {symbol} {interval} to {file_path}")
    
    def load_klines_from_disk(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        """
        Load klines data from disk if available.
        
        Args:
            symbol: Trading pair symbol
            interval: Kline interval
            
        Returns:
            DataFrame with klines data or None if not available
        """
        file_path = self.data_dir / 'klines' / symbol / f"{interval}.csv"
        
        if not file_path.exists():
            return None
        
        try:
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            
            # Update cache
            cache_key = f"{symbol}_{interval}"
            with self.klines_lock:
                self.klines_cache[cache_key] = df
                
            logger.info(f"Loaded {len(df)} klines for {symbol} {interval} from disk")
            return df
            
        except Exception as e:
            logger.error(f"Error loading klines from disk: {e}")
            return None
    
    # Real-time Data Methods
    
    def start_ticker_stream(self, symbol: str, callback: Optional[Callable] = None) -> None:
        """
        Start streaming ticker data for a symbol.
        
        Args:
            symbol: Trading pair symbol
            callback: Optional callback function to be called on ticker updates
        """
        if callback:
            if symbol not in self.tickers_callbacks:
                self.tickers_callbacks[symbol] = []
            self.tickers_callbacks[symbol].append(callback)
        
        def ticker_handler(ticker_data):
            with self.tickers_lock:
                self.tickers_cache[symbol] = ticker_data
            
            # Call registered callbacks
            if symbol in self.tickers_callbacks:
                for cb in self.tickers_callbacks[symbol]:
                    try:
                        cb(ticker_data)
                    except Exception as e:
                        logger.error(f"Error in ticker callback: {e}")
        
        self.client.subscribe_to_ticker(symbol, ticker_handler)
        logger.info(f"Started ticker stream for {symbol}")
    
    def start_klines_stream(self, symbol: str, interval: str, callback: Optional[Callable] = None) -> None:
        """
        Start streaming klines data for a symbol.
        
        Args:
            symbol: Trading pair symbol
            interval: Kline interval
            callback: Optional callback function to be called on kline updates
        """
        cache_key = f"{symbol}_{interval}"
        
        if callback:
            if cache_key not in self.klines_callbacks:
                self.klines_callbacks[cache_key] = []
            self.klines_callbacks[cache_key].append(callback)
        
        def klines_handler(kline_data):
            # Parse new kline data
            new_kline_df = self._parse_klines_to_dataframe([kline_data])
            
            # Update cache
            with self.klines_lock:
                if cache_key in self.klines_cache:
                    # Append or update existing data
                    existing_df = self.klines_cache[cache_key]
                    timestamp = new_kline_df.index[0]
                    
                    if timestamp in existing_df.index:
                        # Update existing candle
                        existing_df.loc[timestamp] = new_kline_df.loc[timestamp]
                    else:
                        # Append new candle
                        self.klines_cache[cache_key] = pd.concat([existing_df, new_kline_df])
                else:
                    # If no cached data exists, fetch historical first
                    self.fetch_historical_klines(symbol, interval)
            
            # Call registered callbacks
            if cache_key in self.klines_callbacks:
                for cb in self.klines_callbacks[cache_key]:
                    try:
                        cb(new_kline_df)
                    except Exception as e:
                        logger.error(f"Error in klines callback: {e}")
        
        self.client.subscribe_to_klines(symbol, interval, klines_handler)
        logger.info(f"Started klines stream for {symbol} {interval}")
    
    def start_orderbook_stream(self, symbol: str, depth: str = "50", callback: Optional[Callable] = None) -> None:
        """
        Start streaming orderbook data for a symbol.
        
        Args:
            symbol: Trading pair symbol
            depth: Orderbook depth
            callback: Optional callback function to be called on orderbook updates
        """
        def orderbook_handler(orderbook_data):
            with self.orderbooks_lock:
                self.orderbooks_cache[symbol] = orderbook_data
            
            if callback:
                try:
                    callback(orderbook_data)
                except Exception as e:
                    logger.error(f"Error in orderbook callback: {e}")
        
        self.client.subscribe_to_orderbook(symbol, depth, orderbook_handler)
        logger.info(f"Started orderbook stream for {symbol}")
    
    def get_current_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get the current ticker data for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Ticker data or None if not available
        """
        with self.tickers_lock:
            return self.tickers_cache.get(symbol)
    
    def get_current_orderbook(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get the current orderbook data for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Orderbook data or None if not available
        """
        with self.orderbooks_lock:
            return self.orderbooks_cache.get(symbol)
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get the current price for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Current price or None if not available
        """
        ticker = self.get_current_ticker(symbol)
        
        if ticker and 'lastPrice' in ticker:
            return float(ticker['lastPrice'])
        
        # Fallback to API call if not in cache
        try:
            ticker_data = self.client.get_ticker(symbol)
            if ticker_data and 'lastPrice' in ticker_data:
                return float(ticker_data['lastPrice'])
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
        
        return None
    
    # Helper Methods
    
    def _interval_to_milliseconds(self, interval: str) -> int:
        """
        Convert a kline interval string to milliseconds.
        
        Args:
            interval: Kline interval string (e.g., '1m', '1h', '1d')
            
        Returns:
            Interval in milliseconds
            
        Raises:
            ValueError: If the interval format is invalid
        """
        unit = interval[-1]
        value = int(interval[:-1])
        
        if unit == 'm':
            return value * 60 * 1000
        elif unit == 'h':
            return value * 60 * 60 * 1000
        elif unit == 'd':
            return value * 24 * 60 * 60 * 1000
        elif unit == 'w':
            return value * 7 * 24 * 60 * 60 * 1000
        elif unit == 'M':
            # Approximate a month as 30 days
            return value * 30 * 24 * 60 * 60 * 1000
        else:
            raise ValueError(f"Invalid interval format: {interval}")
    
    def prepare_data_for_all_pairs(self) -> None:
        """
        Initialize data streams and caches for all active trading pairs.
        """
        for pair_config in self.config.pairs:
            if not pair_config.is_active:
                continue
                
            symbol = pair_config.symbol
            
            # Start ticker stream
            self.start_ticker_stream(symbol)
            
            # Load or fetch historical data for each timeframe used by strategies
            timeframes = set()
            for strategy in self.config.strategies:
                if strategy.is_active:
                    timeframes.add(strategy.timeframe)
            
            for timeframe in timeframes:
                # Try to load from disk first
                df = self.load_klines_from_disk(symbol, timeframe)
                
                # If not available on disk, fetch from API
                if df is None or len(df) < 1000:
                    self.fetch_historical_klines(symbol, timeframe)
                
                # Start streaming for this timeframe
                self.start_klines_stream(symbol, timeframe)
            
            # Start orderbook stream for order execution optimization
            self.start_orderbook_stream(symbol)
            
            logger.info(f"Prepared data streams for {symbol}")
    
    def cleanup(self) -> None:
        """Clean up resources when shutting down."""
        self.client.close_all_websockets()
        logger.info("Market data manager cleaned up") 