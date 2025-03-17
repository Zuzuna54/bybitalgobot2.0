# API Integration System

## Overview

The API Integration System provides a modular, maintainable interface for interacting with cryptocurrency exchanges. Currently, the system implements a comprehensive client for the Bybit API, supporting market data retrieval, account management, order execution, and real-time WebSocket streams.

The API architecture follows a service-oriented design with clear separation of concerns, making it easy to maintain, extend, and test.

## Directory Structure

```
src/api/
├── bybit/                     # Bybit API implementation
│   ├── client.py              # Main client class
│   ├── models.py              # Data models
│   ├── __init__.py            # Package initialization
│   ├── README.md              # Detailed documentation
│   ├── core/                  # Core functionality
│   │   ├── connection.py      # Connection management
│   │   ├── api_client.py      # HTTP request handling
│   │   ├── error_handling.py  # Error processing
│   │   └── rate_limiting.py   # Request rate limiting
│   └── services/              # Service modules
│       ├── market_service.py  # Market data retrieval
│       ├── account_service.py # Account operations
│       ├── order_service.py   # Order management
│       ├── websocket_service.py # WebSocket streams
│       ├── data_service.py    # Enhanced data management
│       └── __init__.py        # Services initialization
└── bybit_client.py            # Backward compatibility layer
```

## Key Features

- **Complete API Coverage**: Support for all major Bybit V5 API endpoints
- **Robust Error Handling**: Comprehensive error handling with detailed messages
- **Rate Limiting**: Built-in rate limiting to prevent API request throttling
- **WebSocket Support**: Real-time data streams for market data and account updates
- **Flexible Authentication**: Support for API key and secret-based authentication
- **Testnet Support**: Ability to switch between mainnet and testnet environments
- **Type Annotations**: Complete type annotations for improved IDE support
- **Comprehensive Logging**: Detailed logging for debugging and monitoring
- **Data Models**: Strongly-typed data models for all API responses
- **Data Caching**: In-memory and disk-based caching for market data
- **Data Persistence**: Save and load historical data from disk

## Main Components

### Bybit Client

The `BybitClient` class is the main entry point for interacting with the Bybit API:

```python
from src.api.bybit_client import BybitClient

# Initialize client
client = BybitClient(
    testnet=True,  # Use Bybit testnet
    api_key="YOUR_API_KEY",  # Optional
    api_secret="YOUR_API_SECRET",  # Optional
    data_dir="data"  # Directory for data caching
)

# Access services
ticker = client.market.get_tickers(symbol="BTCUSDT")
klines = client.data.fetch_historical_klines(
    symbol="BTCUSDT",
    interval="1h",
    start_time="2023-01-01T00:00:00Z"
)
```

### Service Modules

The client provides access to several service modules:

1. **Market Service**: Retrieves market data

   ```python
   # Get ticker data
   tickers = client.market.get_tickers(category="linear", symbol="BTCUSDT")

   # Get orderbook
   orderbook = client.market.get_orderbook(symbol="BTCUSDT", limit=50)

   # Get historical klines (candles)
   klines = client.market.get_klines(symbol="BTCUSDT", interval="60", limit=200)
   ```

2. **Account Service**: Manages account-related operations

   ```python
   # Get wallet balance
   balance = client.account.get_wallet_balance(account_type="UNIFIED")

   # Get positions
   positions = client.account.get_positions(category="linear", symbol="BTCUSDT")

   # Set leverage
   client.account.set_leverage(category="linear", symbol="BTCUSDT", buy_leverage="2")
   ```

3. **Order Service**: Handles order operations

   ```python
   # Place a limit order
   order = client.order.place_order(
       category="linear",
       symbol="BTCUSDT",
       side="Buy",
       order_type="Limit",
       qty="0.001",
       price="30000"
   )

   # Get active orders
   active_orders = client.order.get_active_orders(category="linear", symbol="BTCUSDT")
   ```

4. **WebSocket Service**: Provides real-time data streams

   ```python
   # Define a callback for ticker updates
   def on_ticker_update(data):
       print(f"Ticker update: {data}")

   # Subscribe to ticker channel
   client.websocket.subscribe_public_channel(
       symbol="BTCUSDT",
       channel="tickers",
       callback=on_ticker_update
   )

   # Start WebSocket connection
   client.websocket.start()
   ```

5. **Data Service**: Enhanced market data management

   ```python
   # Fetch historical data with caching
   historical_data = client.data.fetch_historical_klines(
       symbol="BTCUSDT",
       interval="1h",
       start_time="2023-01-01T00:00:00Z",
       end_time="2023-01-07T00:00:00Z",
       use_cache=True
   )

   # Get current price
   current_price = client.data.get_current_price("BTCUSDT")
   ```

## Usage Examples

### Basic Usage

```python
from src.api.bybit_client import BybitClient

# Initialize client
client = BybitClient(testnet=True)

# Get market data
btc_ticker = client.market.get_tickers(symbol="BTCUSDT")
print(f"BTC Price: {btc_ticker['result']['list'][0]['lastPrice']}")

# Fetch historical data
btc_klines = client.data.fetch_historical_klines(
    symbol="BTCUSDT",
    interval="1h",
    start_time="2023-03-01T00:00:00Z",
    end_time="2023-03-02T00:00:00Z"
)
print(f"Loaded {len(btc_klines)} hourly candles")
```

### Authenticated Operations

```python
from src.api.bybit_client import BybitClient
import os
from dotenv import load_dotenv

# Load API keys from environment
load_dotenv()
api_key = os.getenv("BYBIT_API_KEY")
api_secret = os.getenv("BYBIT_API_SECRET")

# Initialize client with authentication
client = BybitClient(
    testnet=True,
    api_key=api_key,
    api_secret=api_secret
)

# Get wallet balance
balance = client.account.get_wallet_balance(account_type="UNIFIED")
print(f"USDT Balance: {balance['result']['list'][0]['coin'][0]['walletBalance']}")

# Place an order
order = client.order.place_order(
    category="linear",
    symbol="BTCUSDT",
    side="Buy",
    order_type="Limit",
    qty="0.001",
    price="30000",
    time_in_force="GTC"
)
print(f"Order placed: {order['result']['orderId']}")
```

### WebSocket Streaming

```python
from src.api.bybit_client import BybitClient
import time

# Initialize client
client = BybitClient(testnet=True)

# Define callback functions
def handle_ticker(message):
    print(f"Ticker: {message['data']['symbol']} - {message['data']['lastPrice']}")

def handle_orderbook(message):
    print(f"Orderbook update for {message['data']['s']} - Bids: {len(message['data']['b'])}")

# Subscribe to channels
client.websocket.subscribe_public_channel(
    symbol="BTCUSDT",
    channel="tickers",
    callback=handle_ticker
)

client.websocket.subscribe_public_channel(
    symbol="BTCUSDT",
    channel="orderbook.50",
    callback=handle_orderbook
)

# Start WebSocket connection
client.websocket.start()

# Keep the script running to receive updates
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    # Stop WebSocket connection
    client.websocket.stop()
```

## Error Handling

The client includes comprehensive error handling:

```python
from src.api.bybit_client import BybitClient
from src.api.bybit.core.error_handling import BybitAPIException, ConnectionException

try:
    client = BybitClient(testnet=True)
    result = client.market.get_tickers(symbol="INVALID_SYMBOL")
except BybitAPIException as e:
    print(f"API Error: {e.message}, Code: {e.code}")
except ConnectionException as e:
    print(f"Connection Error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Extending the API Client

To add new functionality:

1. Identify the appropriate service module
2. Add your method to the service class
3. Update the client interface if necessary

Example - Adding a new method to MarketDataService:

```python
# In src/api/bybit/services/market_service.py
def get_funding_rate(self, symbol: str) -> Dict[str, Any]:
    """
    Get the current funding rate for a perpetual contract.

    Args:
        symbol: The symbol to get funding rate for

    Returns:
        Dictionary with funding rate data
    """
    endpoint = "/v5/market/funding/history"
    params = {
        "category": "linear",
        "symbol": symbol,
        "limit": 1
    }

    response = make_request(
        connection_manager=self.connection_manager,
        method="GET",
        endpoint=endpoint,
        params=params
    )

    return response.get("result", {})
```

## Developer Guide

For more detailed information about the Bybit API client, refer to the [Bybit API Client README](bybit/README.md).

## Future Enhancements

1. **Additional Exchange Support**: Implement clients for other cryptocurrency exchanges
2. **Advanced WebSocket Management**: Improve connection reliability and reconnection strategies
3. **Enhanced Caching Strategies**: Implement more sophisticated caching for improved performance
4. **Asynchronous API**: Add async/await support for non-blocking operations
5. **Advanced Error Recovery**: Implement retry mechanisms and fallback strategies
