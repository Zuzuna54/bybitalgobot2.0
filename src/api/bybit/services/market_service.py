"""
Market Data Service for Bybit API

This module provides functionality for retrieving market data from Bybit,
including tickers, orderbooks, klines, and instrument information.
"""

from typing import Dict, Any, Optional, List
from loguru import logger

from src.api.bybit.core.connection import ConnectionManager
from src.api.bybit.core.error_handling import with_error_handling
from src.api.bybit.core.rate_limiting import rate_limited
from src.api.bybit.core.api_client import make_request


class MarketDataService:
    """
    Service for retrieving market data from Bybit.
    """
    
    def __init__(self, connection_manager: ConnectionManager):
        """
        Initialize the market data service.
        
        Args:
            connection_manager: Connection manager instance
        """
        self.connection_manager = connection_manager
    
    @rate_limited('market')
    @with_error_handling
    def get_tickers(
        self, 
        category: str = "linear", 
        symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get tickers for all symbols or a specific symbol.
        
        Args:
            category: Product type (linear, inverse, spot)
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            
        Returns:
            Ticker data
        """
        endpoint = "/v5/market/tickers"
        params = {'category': category}
        
        if symbol:
            params['symbol'] = symbol
        
        response = make_request(
            connection_manager=self.connection_manager,
            method="GET",
            endpoint=endpoint,
            params=params
        )
        
        return response.get('result', {})
    
    @rate_limited('market')
    @with_error_handling
    def get_orderbook(
        self, 
        symbol: str, 
        limit: int = 50, 
        category: str = "linear"
    ) -> Dict[str, Any]:
        """
        Get orderbook for a specific symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            limit: Depth of the orderbook (1-200)
            category: Product type (linear, inverse, spot)
            
        Returns:
            Orderbook data
        """
        endpoint = "/v5/market/orderbook"
        params = {
            'symbol': symbol,
            'limit': limit,
            'category': category
        }
        
        response = make_request(
            connection_manager=self.connection_manager,
            method="GET",
            endpoint=endpoint,
            params=params
        )
        
        return response.get('result', {})
    
    @rate_limited('market')
    @with_error_handling
    def get_klines(
        self, 
        symbol: str, 
        interval: str, 
        start_time: Optional[int] = None, 
        end_time: Optional[int] = None, 
        limit: int = 200, 
        category: str = "linear"
    ) -> Dict[str, Any]:
        """
        Get kline/candlestick data for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Kline interval (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M)
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds
            limit: Number of candles to return (max 200)
            category: Product type (linear, inverse, spot)
            
        Returns:
            Kline data
        """
        endpoint = "/v5/market/kline"
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit,
            'category': category
        }
        
        if start_time:
            params['start'] = start_time
            
        if end_time:
            params['end'] = end_time
        
        response = make_request(
            connection_manager=self.connection_manager,
            method="GET",
            endpoint=endpoint,
            params=params
        )
        
        return response.get('result', {})
    
    @rate_limited('market')
    @with_error_handling
    def get_trades(
        self, 
        symbol: str, 
        limit: int = 50, 
        category: str = "linear"
    ) -> Dict[str, Any]:
        """
        Get recent trades for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            limit: Number of trades to return (max 1000)
            category: Product type (linear, inverse, spot)
            
        Returns:
            Recent trades data
        """
        endpoint = "/v5/market/trades"
        params = {
            'symbol': symbol,
            'limit': limit,
            'category': category
        }
        
        response = make_request(
            connection_manager=self.connection_manager,
            method="GET",
            endpoint=endpoint,
            params=params
        )
        
        return response.get('result', {})
    
    @rate_limited('market')
    @with_error_handling
    def get_instruments_info(
        self, 
        category: str = "linear", 
        symbol: Optional[str] = None, 
        status: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get instrument information.
        
        Args:
            category: Product type (linear, inverse, spot)
            symbol: Trading pair symbol
            status: Instrument status
            
        Returns:
            Instrument information
        """
        endpoint = "/v5/market/instruments-info"
        params = {'category': category}
        
        if symbol:
            params['symbol'] = symbol
            
        if status:
            params['status'] = status
            
        response = make_request(
            connection_manager=self.connection_manager,
            method="GET",
            endpoint=endpoint,
            params=params
        )
        
        return response.get('result', {})
    
    @rate_limited('market')
    @with_error_handling
    def get_mark_price_klines(
        self, 
        symbol: str, 
        interval: str, 
        start_time: Optional[int] = None, 
        end_time: Optional[int] = None, 
        limit: int = 200
    ) -> Dict[str, Any]:
        """
        Get mark price klines/candlestick data.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Kline interval (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M)
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds
            limit: Number of candles to return (max 200)
            
        Returns:
            Mark price kline data
        """
        endpoint = "/v5/market/mark-price-kline"
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        if start_time:
            params['start'] = start_time
            
        if end_time:
            params['end'] = end_time
            
        response = make_request(
            connection_manager=self.connection_manager,
            method="GET",
            endpoint=endpoint,
            params=params
        )
        
        return response.get('result', {})
    
    @rate_limited('market')
    @with_error_handling
    def get_open_interest(
        self, 
        symbol: str, 
        interval: str, 
        start_time: Optional[int] = None, 
        end_time: Optional[int] = None, 
        limit: int = 50
    ) -> Dict[str, Any]:
        """
        Get open interest data.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Data interval (5min, 15min, 30min, 1h, 4h, 1d)
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds
            limit: Number of records to return (max 200)
            
        Returns:
            Open interest data
        """
        endpoint = "/v5/market/open-interest"
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        if start_time:
            params['startTime'] = start_time
            
        if end_time:
            params['endTime'] = end_time
            
        response = make_request(
            connection_manager=self.connection_manager,
            method="GET",
            endpoint=endpoint,
            params=params
        )
        
        return response.get('result', {})