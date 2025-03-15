#!/usr/bin/env python
"""
Bybit API Client Example

This script demonstrates the capabilities of the refactored Bybit API client,
including market data retrieval, account operations, and WebSocket functionality.
"""

import os
import sys
import json
import time
import asyncio
from typing import Dict, Any, List, Optional
from queue import Queue
from dotenv import load_dotenv
from loguru import logger
from pathlib import Path

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the Bybit client
from src.api.bybit import BybitClient


def setup_logger():
    """Configure the logger for the example."""
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="DEBUG"
    )


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


async def main():
    """Main example function demonstrating the Bybit API client."""
    # Load environment variables manually
    try:
        # Manual parsing of .env file
        env_path = os.path.join(os.getcwd(), '.env')
        env_vars = {}
        
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                    
                # Look for KEY=VALUE pattern
                import re
                match = re.match(r'^([A-Za-z0-9_]+)=(.*)$', line)
                if match:
                    key, value = match.groups()
                    env_vars[key] = value
                    # Also set in environment
                    os.environ[key] = value
        
        # Get API credentials directly from parsed variables
        api_key = env_vars.get('BYBIT_API_KEY', '')
        api_secret = env_vars.get('BYBIT_API_SECRET', '')
        use_testnet_str = env_vars.get('BYBIT_USE_TESTNET', 'True')
        use_testnet = use_testnet_str.lower() in ('true', '1', 't')
    except Exception as e:
        print(f"Error manually loading .env file: {e}")
        # Fallback to environment variables
        api_key = os.getenv("BYBIT_API_KEY", "")
        api_secret = os.getenv("BYBIT_API_SECRET", "")
        use_testnet = os.getenv("BYBIT_USE_TESTNET", "True").lower() in ("true", "1", "t")
    
    # Print the values being loaded (for debugging)
    print(f"API credentials loaded from .env file:")
    print(f"  API Key: {'Present ('+api_key[:4]+'...)' if api_key else 'Missing'}")
    print(f"  API Secret: {'Present ('+api_secret[:4]+'...)' if api_secret else 'Missing'}")
    print(f"  Use Testnet: {use_testnet}")
    
    # Check if we have API credentials
    has_credentials = bool(api_key and api_secret)
    
    # Initialize client
    print_section("Initializing Bybit Client")
    
    # Prepare client arguments
    client_args = {"testnet": use_testnet}
    if has_credentials:
        client_args["api_key"] = api_key
        client_args["api_secret"] = api_secret
    
    # Create client
    client = BybitClient(**client_args)
    
    print(f"Using testnet: {use_testnet}")
    print(f"API key provided: {'Yes' if has_credentials else 'No'}")
    print()
    
    # Verify API credentials if provided
    if has_credentials:
        print("Verifying API credentials...")
        try:
            # Add DEBUG level logs
            logger.debug(f"API key: {api_key}")
            logger.debug(f"API secret: {api_secret[:4]}***")
            
            # First check direct connection manager verification
            print("Testing direct connection verification method:")
            is_valid_direct = client.connection_manager.verify_credentials()
            print(f"Direct verification result: {is_valid_direct}")
            
            # Then check account service verification
            print("Testing account service verification method:")
            is_valid = client.account.verify_credentials()
            print(f"Credentials valid: {is_valid}")
        except Exception as e:
            print(f"Error verifying credentials: {e}")
            print("Continuing with example using public endpoints only...")
            has_credentials = False
    else:
        print("No API credentials provided. Only public endpoints will be available.")
    
    # Basic information
    print_section("Basic Information")
    
    # Get server time
    try:
        server_time = client.connection_manager.get_server_time()
        server_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(server_time / 1000))
        print(f"Server time: {server_time} ({server_time_str})")
    except Exception as e:
        print(f"Error getting server time: {e}")
        # Fallback to local time
        server_time = int(time.time() * 1000)
        server_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(server_time / 1000))
        print(f"Using local time instead: {server_time} ({server_time_str})")
    
    # Market data examples
    print_section("Market Data Examples")
    
    # Example 1: Get ticker data
    print("\n1. Fetching ticker data for BTCUSDT...")
    try:
        tickers = client.market.get_tickers(category="linear", symbol="BTCUSDT")
        if tickers and 'list' in tickers and tickers['list']:
            ticker = tickers['list'][0]
            print(f"  Symbol: {ticker.get('symbol')}")
            print(f"  Last Price: {ticker.get('lastPrice')}")
            print(f"  24h Change: {ticker.get('price24hPcnt')}%")
            print(f"  24h High: {ticker.get('highPrice24h')}")
            print(f"  24h Low: {ticker.get('lowPrice24h')}")
            print(f"  24h Volume: {ticker.get('volume24h')}")
        else:
            print("  No ticker data available")
    except Exception as e:
        print(f"  Error fetching ticker data: {e}")
    
    # Example 2: Get orderbook
    print("\n2. Fetching orderbook for BTCUSDT (depth 5)...")
    try:
        orderbook = client.market.get_orderbook(symbol="BTCUSDT", limit=5)
        if orderbook and 'a' in orderbook and 'b' in orderbook:
            print("  Asks (Sell Orders):")
            for i, ask in enumerate(orderbook['a'][:5], 1):
                print(f"    {i}. Price: {ask[0]}, Size: {ask[1]}")
            
            print("  Bids (Buy Orders):")
            for i, bid in enumerate(orderbook['b'][:5], 1):
                print(f"    {i}. Price: {bid[0]}, Size: {bid[1]}")
        else:
            print("  No orderbook data available")
    except Exception as e:
        print(f"  Error fetching orderbook: {e}")
    
    # Example 3: Get klines (candlestick data)
    print("\n3. Fetching recent klines for BTCUSDT (1h timeframe, last 5)...")
    try:
        klines = client.market.get_klines(
            symbol="BTCUSDT",
            interval="60",
            limit=5
        )
        if klines and 'list' in klines:
            print("  Recent candles (Time, Open, High, Low, Close, Volume):")
            for i, kline in enumerate(klines['list'][:5], 1):
                timestamp = time.strftime('%Y-%m-%d %H:%M', time.localtime(int(kline[0])/1000))
                print(f"    {i}. {timestamp}: O:{kline[1]} H:{kline[2]} L:{kline[3]} C:{kline[4]} V:{kline[5]}")
        else:
            print("  No kline data available")
    except Exception as e:
        print(f"  Error fetching klines: {e}")
    
    # Account data examples (requires authentication)
    print_section("Account Data Examples")
    
    if not has_credentials:
        print("\nSkipping account data examples - no valid API credentials provided")
    else:
        # Example 1: Get wallet balance
        print("\n1. Fetching wallet balance...")
        try:
            wallet = client.account.get_wallet_balance()
            print("  Wallet balances:")
            if 'list' in wallet:
                for account in wallet['list']:
                    account_type = account.get('accountType', 'Unknown')
                    total_equity = account.get('totalEquity', '0')
                    total_wallet_balance = account.get('totalWalletBalance', '0')
                    print(f"    Account Type: {account_type}")
                    print(f"    Total Equity: {total_equity}")
                    print(f"    Total Wallet Balance: {total_wallet_balance}")
                    
                    if 'coin' in account:
                        print("    Coins:")
                        for coin_data in account['coin']:
                            coin = coin_data.get('coin', 'Unknown')
                            available = coin_data.get('availableToWithdraw', '0')
                            wallet_balance = coin_data.get('walletBalance', '0')
                            print(f"      {coin}: Available: {available}, Balance: {wallet_balance}")
            else:
                print("  No wallet data available")
        except Exception as e:
            print(f"  Error fetching wallet data: {e}")
        
        # Example 2: Get positions
        print("\n2. Fetching positions...")
        try:
            # Need to provide either symbol or settleCoin
            positions = client.account.get_positions(category="linear", symbol="BTCUSDT")
            print("  Current positions:")
            if 'list' in positions:
                if positions['list']:
                    for position in positions['list']:
                        symbol = position.get('symbol', 'Unknown')
                        size = position.get('size', '0')
                        side = position.get('side', 'None')
                        position_value = position.get('positionValue', '0')
                        leverage = position.get('leverage', '0')
                        print(f"    Symbol: {symbol}")
                        print(f"    Size: {size}")
                        print(f"    Side: {side}")
                        print(f"    Position Value: {position_value}")
                        print(f"    Leverage: {leverage}")
                else:
                    print("    No open positions")
            else:
                print("    No position data available")
        except Exception as e:
            print(f"  Error fetching position data: {e}")
    
    # WebSocket example (if we have API credentials)
    print("\n=== WebSocket Example ===")

    try:
        ws_service = client.websocket
        
        # Define callback function to process ticker updates
        def process_ticker_update(data):
            if 'data' in data:
                ticker_data = data['data']
                print(f"\nReceived ticker update for {ticker_data.get('symbol', 'Unknown')}:")
                print(f"  Last Price: {ticker_data.get('lastPrice', 'N/A')}")
                print(f"  Bid Price: {ticker_data.get('bid1Price', 'N/A')} (Size: {ticker_data.get('bid1Size', 'N/A')})")
                print(f"  Ask Price: {ticker_data.get('ask1Price', 'N/A')} (Size: {ticker_data.get('ask1Size', 'N/A')})")
                print(f"  24h Volume: {ticker_data.get('volume24h', 'N/A')}")
                print(f"  24h Turnover: {ticker_data.get('turnover24h', 'N/A')}")
        
        # Start WebSocket service
        ws_service.start()
        print("WebSocket service started. Waiting for connection to establish...")
        
        # Wait for WebSocket connection to establish before subscribing
        max_retries = 10
        retry_interval = 1  # seconds
        for attempt in range(max_retries):
            # Check if WebSocket is connected
            if ws_service.public_ws and ws_service.public_ws.sock and getattr(ws_service.public_ws.sock, 'connected', False):
                print(f"WebSocket connected after {attempt+1} attempts")
                
                # Subscribe to ticker updates for BTCUSDT
                print("Subscribing to BTCUSDT ticker updates...")
                ws_service.subscribe_to_ticker("BTCUSDT", process_ticker_update)
                break
            else:
                print(f"Waiting for WebSocket connection, attempt {attempt+1}/{max_retries}...")
                time.sleep(retry_interval)
        else:
            print("Failed to establish WebSocket connection after maximum retries")
        
        # Keep the main thread running to receive WebSocket messages
        print("Waiting for ticker updates (press Ctrl+C to stop)...")
        try:
            # Wait for 30 seconds to receive updates
            time.sleep(30)
        except KeyboardInterrupt:
            print("Keyboard interrupt received, stopping...")
        
        # Stop WebSocket service
        ws_service.stop()
        print("WebSocket service stopped.")
        
    except Exception as e:
        print(f"Error in WebSocket example: {e}")
    
    print_section("Example Completed")
    print("Thanks for using the Bybit API Client!")
    print("Note: Authentication errors are expected if you haven't provided valid API credentials.")
    print("The example demonstrates both authenticated and unauthenticated endpoints.")


if __name__ == "__main__":
    # Set up logging
    setup_logger()
    
    # Run the example
    asyncio.run(main()) 