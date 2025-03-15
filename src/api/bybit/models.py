"""
Bybit API Data Models for the Algorithmic Trading System

This module provides data models and type definitions for the Bybit API.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Union
from datetime import datetime


@dataclass
class Ticker:
    """Ticker data model."""
    symbol: str
    last_price: float
    high_price_24h: float
    low_price_24h: float
    price_24h_percent_change: float
    volume_24h: float
    turnover_24h: float
    timestamp: datetime
    
    @classmethod
    def from_api_response(cls, response: Dict[str, Any]) -> 'Ticker':
        """Create a Ticker instance from API response data."""
        return cls(
            symbol=response.get('symbol', ''),
            last_price=float(response.get('lastPrice', 0)),
            high_price_24h=float(response.get('highPrice24h', 0)),
            low_price_24h=float(response.get('lowPrice24h', 0)),
            price_24h_percent_change=float(response.get('price24hPcnt', 0)),
            volume_24h=float(response.get('volume24h', 0)),
            turnover_24h=float(response.get('turnover24h', 0)),
            timestamp=datetime.fromtimestamp(int(response.get('timestamp', 0)) / 1000)
        )


@dataclass
class OrderBookEntry:
    """Order book price level entry."""
    price: float
    size: float


@dataclass
class OrderBook:
    """Order book data model."""
    symbol: str
    bids: List[OrderBookEntry]
    asks: List[OrderBookEntry]
    timestamp: datetime
    
    @classmethod
    def from_api_response(cls, response: Dict[str, Any]) -> 'OrderBook':
        """Create an OrderBook instance from API response data."""
        bids = [
            OrderBookEntry(float(bid[0]), float(bid[1]))
            for bid in response.get('b', [])
        ]
        
        asks = [
            OrderBookEntry(float(ask[0]), float(ask[1]))
            for ask in response.get('a', [])
        ]
        
        return cls(
            symbol=response.get('s', ''),
            bids=bids,
            asks=asks,
            timestamp=datetime.fromtimestamp(int(response.get('ts', 0)) / 1000)
        )


@dataclass
class Kline:
    """Candlestick/kline data model."""
    symbol: str
    interval: str
    open_time: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    turnover: float
    
    @classmethod
    def from_api_response(cls, response: Dict[str, Any], interval: str) -> 'Kline':
        """Create a Kline instance from API response data."""
        return cls(
            symbol=response.get('symbol', ''),
            interval=interval,
            open_time=datetime.fromtimestamp(int(response.get('start', 0)) / 1000),
            open_price=float(response.get('open', 0)),
            high_price=float(response.get('high', 0)),
            low_price=float(response.get('low', 0)),
            close_price=float(response.get('close', 0)),
            volume=float(response.get('volume', 0)),
            turnover=float(response.get('turnover', 0))
        )


@dataclass
class Position:
    """Trading position data model."""
    symbol: str
    position_side: str  # 'Buy' or 'Sell'
    size: float
    entry_price: float
    leverage: float
    mark_price: float
    unrealized_pnl: float
    take_profit: Optional[float]
    stop_loss: Optional[float]
    created_time: datetime
    
    @classmethod
    def from_api_response(cls, response: Dict[str, Any]) -> 'Position':
        """Create a Position instance from API response data."""
        return cls(
            symbol=response.get('symbol', ''),
            position_side=response.get('side', ''),
            size=float(response.get('size', 0)),
            entry_price=float(response.get('entryPrice', 0)),
            leverage=float(response.get('leverage', 1)),
            mark_price=float(response.get('markPrice', 0)),
            unrealized_pnl=float(response.get('unrealisedPnl', 0)),
            take_profit=float(response.get('takeProfit', 0)) if response.get('takeProfit') else None,
            stop_loss=float(response.get('stopLoss', 0)) if response.get('stopLoss') else None,
            created_time=datetime.fromtimestamp(int(response.get('createdTime', 0)) / 1000)
        )


@dataclass
class Order:
    """Order data model."""
    order_id: str
    symbol: str
    side: str  # 'Buy' or 'Sell'
    order_type: str  # 'Limit', 'Market'
    price: Optional[float]
    qty: float
    time_in_force: str
    order_status: str
    reduce_only: bool
    created_time: datetime
    updated_time: datetime
    
    @classmethod
    def from_api_response(cls, response: Dict[str, Any]) -> 'Order':
        """Create an Order instance from API response data."""
        return cls(
            order_id=response.get('orderId', ''),
            symbol=response.get('symbol', ''),
            side=response.get('side', ''),
            order_type=response.get('orderType', ''),
            price=float(response.get('price', 0)) if response.get('price') else None,
            qty=float(response.get('qty', 0)),
            time_in_force=response.get('timeInForce', ''),
            order_status=response.get('orderStatus', ''),
            reduce_only=response.get('reduceOnly', False),
            created_time=datetime.fromtimestamp(int(response.get('createdTime', 0)) / 1000),
            updated_time=datetime.fromtimestamp(int(response.get('updatedTime', 0)) / 1000)
        )


@dataclass
class Wallet:
    """Wallet balance data model."""
    coin: str
    equity: float
    available_balance: float
    used_margin: float
    order_margin: float
    position_margin: float
    unrealized_pnl: float
    
    @classmethod
    def from_api_response(cls, response: Dict[str, Any]) -> 'Wallet':
        """Create a Wallet instance from API response data."""
        return cls(
            coin=response.get('coin', ''),
            equity=float(response.get('equity', 0)),
            available_balance=float(response.get('availableBalance', 0)),
            used_margin=float(response.get('usedMargin', 0)),
            order_margin=float(response.get('orderMargin', 0)),
            position_margin=float(response.get('positionMargin', 0)),
            unrealized_pnl=float(response.get('unrealisedPnl', 0))
        ) 