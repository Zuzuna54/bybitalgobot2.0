"""
Example Usage of the Bybit Client

This script demonstrates how to use the refactored Bybit client
with the new modular structure.
"""

import os
import time
import asyncio
from loguru import logger
from dotenv import load_dotenv

from src.api.bybit.client import BybitClient


def configure_logger():
    """Configure the logger."""
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg),
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )


async def main():
    """Main execution function."""
    # Load environment variables
    load_dotenv()
    
    # Configure logger
    configure_logger()
    
    # Get API credentials from environment variables
    api_key = os.getenv("BYBIT_API_KEY", "")
    api_secret = os.getenv("BYBIT_API_SECRET", "")
    use_testnet = os.getenv("BYBIT_USE_TESTNET", "True").lower() == "true"
    
    logger.info(f"Initializing Bybit client (testnet: {use_testnet})")
    
    # Initialize client
    client = BybitClient(
        testnet=use_testnet,
        api_key=api_key,
        api_secret=api_secret
    )
    
    try:
        # Verify API credentials
        if api_key and api_secret:
            is_valid = client.verify_credentials()
            logger.info(f"API credentials valid: {is_valid}")
        
        # Get server time
        server_time = client.get_server_time()
        logger.info(f"Server time: {server_time}")
        
        # Get market data
        logger.info("Getting market data...")
        tickers = client.market.get_tickers(symbol="BTCUSDT")
        logger.info(f"BTCUSDT ticker: {tickers}")
        
        # Get orderbook
        orderbook = client.market.get_orderbook(symbol="BTCUSDT", limit=5)
        logger.info(f"BTCUSDT orderbook: {orderbook}")
        
        # Get wallet balance (requires authentication)
        if api_key and api_secret:
            logger.info("Getting wallet balance...")
            wallet = client.account.get_wallet_balance()
            logger.info(f"Wallet balance: {wallet}")
            
            # Get positions
            logger.info("Getting positions...")
            positions = client.account.get_positions(symbol="BTCUSDT")
            logger.info(f"Positions: {positions}")
        
        # Demonstrate WebSocket functionality
        logger.info("Setting up WebSocket...")
        
        # Define callback function
        def handle_ticker_data(data):
            logger.info(f"Ticker WebSocket data: {data}")
        
        # Start WebSocket
        client.websocket.start()
        
        # Subscribe to ticker updates
        client.websocket.subscribe_public(
            topics=["tickers.BTCUSDT"],
            callback=handle_ticker_data
        )
        
        # Wait for some WebSocket data
        logger.info("Waiting for WebSocket data (5 seconds)...")
        await asyncio.sleep(5)
        
        # Unsubscribe and stop
        client.websocket.unsubscribe(topics=["tickers.BTCUSDT"])
        client.websocket.stop()
        logger.info("WebSocket stopped")
        
    except Exception as e:
        logger.error(f"Error in example: {e}")
    
    logger.info("Example completed")


if __name__ == "__main__":
    asyncio.run(main()) 