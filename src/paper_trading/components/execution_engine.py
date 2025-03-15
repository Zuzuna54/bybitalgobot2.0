"""
Trade execution functionality for the paper trading simulator.

This module provides functions for executing trades in the paper trading system.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import time
from loguru import logger

from src.paper_trading.components.order_processor import calculate_execution_price


def execute_paper_trade(
    signal: Any,
    market_data,
    strategy_manager,
    risk_manager,
    performance_tracker,
    active_positions: Dict[str, Dict[str, Any]],
    trade_history: List[Dict[str, Any]],
    current_balance: float,
    slippage: float,
    commission_rate: float,
    latency_ms: int
) -> Optional[Tuple[str, float]]:
    """
    Execute a paper trade based on a signal.
    
    Args:
        signal: Trading signal
        market_data: Market data manager
        strategy_manager: Strategy manager
        risk_manager: Risk manager
        performance_tracker: Performance tracker
        active_positions: Dictionary of active positions
        trade_history: List of all trades
        current_balance: Current account balance
        slippage: Slippage amount as a decimal
        commission_rate: Commission rate as a decimal
        latency_ms: Simulated latency in milliseconds
        
    Returns:
        Tuple of (trade_id, new_balance) if trade executed, None otherwise
    """
    symbol = signal.symbol
    is_long = signal.signal_type.name == "BUY"
    
    # Get execution price with slippage
    entry_price = calculate_execution_price(
        symbol, 
        is_long, 
        slippage, 
        market_data.get_current_price
    )
    
    # Add simulated latency
    if latency_ms > 0:
        time.sleep(latency_ms / 1000.0)
    
    # Get prepared data for stop loss calculation
    market_data_dict = get_market_data(symbol, market_data, strategy_manager)
    if not market_data_dict:
        logger.error(f"Cannot execute paper trade for {symbol} - market data unavailable")
        return None
    
    # Calculate stop loss price using the strategy that generated the signal
    strategy_name = signal.metadata.get('strategy_name', 'unknown')
    strategy = strategy_manager.get_strategy(strategy_name)
    stop_loss_price = strategy.get_stop_loss_price(market_data_dict, entry_price, is_long)
    
    # Calculate take profit price
    take_profit_price = strategy.get_take_profit_price(
        market_data_dict, entry_price, is_long, stop_loss_price
    )
    
    # Calculate position size
    position_size = risk_manager.calculate_position_size(
        symbol=symbol,
        account_balance=current_balance,
        entry_price=entry_price,
        stop_loss_price=stop_loss_price
    )
    
    # Skip if position size is too small
    if position_size <= 0:
        logger.info(f"Skipping {symbol} trade - position size too small")
        return None
    
    # Calculate commission
    commission = entry_price * position_size * commission_rate
    
    # Update balance (subtract commission)
    new_balance = current_balance - commission
    
    # Generate trade ID
    trade_id = f"{symbol}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # Create paper trade
    paper_trade = {
        "id": trade_id,
        "symbol": symbol,
        "type": "long" if is_long else "short",
        "entry_time": datetime.now(),
        "entry_price": entry_price,
        "stop_loss": stop_loss_price,
        "take_profit": take_profit_price,
        "position_size": position_size,
        "strategy_name": strategy_name,
        "signal_strength": signal.strength,
        "commission_paid": commission,
        "status": "open",
        "unrealized_pnl": 0.0,
        "highest_price": entry_price if is_long else None,
        "lowest_price": entry_price if not is_long else None
    }
    
    # Add to active positions
    active_positions[symbol] = paper_trade
    
    # Add to trade history
    trade_history.append(paper_trade)
    
    logger.info(f"Executed paper trade: {symbol} {'BUY' if is_long else 'SELL'} at {entry_price} with size {position_size}")
    
    # Update risk manager
    risk_manager.update_account_balance(new_balance)
    
    # Notify performance tracker
    performance_tracker.add_trade({
        "id": trade_id,
        "symbol": symbol,
        "side": "Buy" if is_long else "Sell",
        "entry_price": entry_price,
        "exit_price": None,
        "position_size": position_size,
        "entry_time": paper_trade["entry_time"],
        "exit_time": None,
        "realized_pnl": None,
        "realized_pnl_percent": None,
        "strategy_name": strategy_name,
        "exit_reason": None,
        "status": "open"
    })
    
    return trade_id, new_balance


def get_market_data(symbol: str, market_data, strategy_manager) -> Optional[Dict[str, Any]]:
    """
    Get prepared market data for a symbol.
    
    Args:
        symbol: Trading pair symbol
        market_data: Market data manager
        strategy_manager: Strategy manager
        
    Returns:
        Dictionary with market data or None if not available
    """
    # Get ticker data
    ticker = market_data.get_current_ticker(symbol)
    if not ticker:
        return None
    
    # Get the latest candles
    try:
        timeframe = market_data.config.get('default_timeframe', '1h')
        df = market_data.fetch_historical_klines(
            symbol=symbol,
            interval=timeframe,
            limit=100  # Get last 100 candles
        )
        
        # Apply indicators
        prepared_data = strategy_manager.prepare_data_for_strategy(df, symbol)
        return prepared_data
    except Exception as e:
        logger.error(f"Error getting market data for {symbol}: {e}")
        return None


def process_strategy_signals(
    symbols: List[str],
    market_data,
    strategy_manager,
    risk_manager,
    performance_tracker,
    active_positions: Dict[str, Dict[str, Any]],
    trade_history: List[Dict[str, Any]],
    current_balance: float,
    slippage: float,
    commission_rate: float,
    latency_ms: int
) -> Optional[float]:
    """
    Process signals from strategies and execute paper trades.
    
    Args:
        symbols: List of trading pair symbols
        market_data: Market data manager
        strategy_manager: Strategy manager
        risk_manager: Risk manager
        performance_tracker: Performance tracker
        active_positions: Dictionary of active positions
        trade_history: List of all trades
        current_balance: Current account balance
        slippage: Slippage amount as a decimal
        commission_rate: Commission rate as a decimal
        latency_ms: Simulated latency in milliseconds
        
    Returns:
        Updated balance if trades were executed, None otherwise
    """
    updated_balance = current_balance
    
    for symbol in symbols:
        try:
            # Get market data for this symbol
            market_data_dict = get_market_data(symbol, market_data, strategy_manager)
            if not market_data_dict:
                continue
            
            # Generate signals
            signals = strategy_manager.generate_signals(market_data_dict)
            
            for signal in signals:
                # Skip if we already have a position for this symbol
                if symbol in active_positions:
                    logger.debug(f"Skipping signal for {symbol} - position already exists")
                    continue
                
                # Check if we should take this trade based on risk management
                if not risk_manager.should_take_trade(
                    symbol=symbol,
                    signal_strength=signal.strength,
                    account_balance=updated_balance
                ):
                    logger.debug(f"Risk manager rejected trade for {symbol}")
                    continue
                
                # Process the signal and create a simulated trade
                result = execute_paper_trade(
                    signal=signal,
                    market_data=market_data,
                    strategy_manager=strategy_manager,
                    risk_manager=risk_manager,
                    performance_tracker=performance_tracker,
                    active_positions=active_positions,
                    trade_history=trade_history,
                    current_balance=updated_balance,
                    slippage=slippage,
                    commission_rate=commission_rate,
                    latency_ms=latency_ms
                )
                
                # Update balance if trade was executed
                if result:
                    _, updated_balance = result
        
        except Exception as e:
            logger.error(f"Error processing signals for {symbol}: {e}")
    
    # Only return balance if it changed
    if updated_balance != current_balance:
        return updated_balance
        
    return None 