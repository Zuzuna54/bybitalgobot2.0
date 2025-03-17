# Bybit API Client

## Overview

The Bybit API Client is a comprehensive Python library for interacting with the Bybit cryptocurrency exchange API. It provides a structured and easy-to-use interface for accessing market data, managing trading accounts, executing orders, and subscribing to real-time WebSocket streams.

This client is designed with a focus on reliability, performance, and developer experience, featuring built-in rate limiting, error handling, and both synchronous and asynchronous interfaces.

## Features

- **Complete API Coverage**: Supports all major Bybit V5 API endpoints
- **Robust Error Handling**: Comprehensive error handling with detailed error messages
- **Rate Limiting**: Built-in rate limiting to prevent API request throttling
- **WebSocket Support**: Real-time data streams for market data and account updates
- **Flexible Authentication**: Support for API key and secret-based authentication
- **Testnet Support**: Ability to switch between mainnet and testnet environments
- **Type Annotations**: Complete type annotations for improved IDE support
- **Comprehensive Logging**: Detailed logging for debugging and monitoring
- **Data Models**: Strongly-typed data models for all API responses
- **Data Caching**: In-memory and disk-based caching for market data
- **Data Persistence**: Save and load historical data from disk
- **Simplified Data Access**: Convenient methods for accessing market data

## Architecture

The Bybit API Client follows a modular architecture organized by functional domains:

### Core Components

1. **Client (`client.py`)**: The main entry point providing access to all services and modules.
2. **Connection Manager (`core/connection.py`)**: Manages API connectivity, URLs, and authentication.
3. **API Client (`core/api_client.py`)**: Handles low-level HTTP requests and response processing.
4. **Error Handling (`core/error_handling.py`)**: Provides error detection, classification, and handling.
5. **Rate Limiting (`core/rate_limiting.py`)**: Implements request rate limiting to comply with API restrictions.

### Services

1. **Market Data Service (`services/market_service.py`)**: Retrieves market data such as tickers, orderbooks, and candles.
2. **Account Service (`services/account_service.py`)**: Manages account-related operations like balance and positions.
3. **Order Service (`services/order_service.py`)**: Handles order operations including placement, cancellation, and retrieval.
4. **WebSocket Service (`services/websocket_service.py`)**: Provides real-time data streams through WebSocket connections.
5. **Data Service (`services/data_service.py`)**: Enhanced market data management with caching and persistence.

### Data Models

1. **Models (`models.py`)**: Defines strongly-typed data models for all API response types, including:
   - `Ticker`: Market ticker data with price and volume information
   - `OrderBook`: Order book data with bids and asks
   - `Kline`: Candlestick/kline historical price data
   - `Position`: Trading position information
   - `Order`: Order data for placed/active orders
   - `Wallet`: Wallet balance information

### Supporting Components

1. **Package Initialization (`__init__.py`)**: Exports public interfaces and backward compatibility layers.
2. **Bybit Client (`bybit_client.py`)**: Re-exports the BybitClient class for backward compatibility.

## Detailed Functionality

### Connection Management

The `ConnectionManager` class manages API connections and provides the following features:

- Endpoint URL generation for both mainnet and testnet environments
- Authentication credential management
- Request signing for authenticated endpoints
- Server time synchronization
- WebSocket URL generation

```python
# Initialize with testnet and API credentials
connection_manager = ConnectionManager(
    testnet=True,
    api_key="YOUR_API_KEY",
    api_secret="YOUR_API_SECRET",
    recv_window=5000
)

# Get API base URL
base_url = connection_manager.base_url

# Generate authentication signature
signature = connection_manager.get_signature(params)
```

### Market Data Service

The `MarketDataService` provides access to public market data:

- **Tickers**: Latest price, volume, and other statistics for symbols
- **Orderbook**: Market depth for a specific symbol
- **Klines (Candles)**: Historical price data with various time intervals
- **Trades**: Recent trades for a specific symbol
- **Instruments Info**: Exchange information about trading instruments

```python
# Get ticker data for BTCUSDT
tickers = client.market.get_tickers(category="linear", symbol="BTCUSDT")

# Get orderbook for BTCUSDT with depth of 50
orderbook = client.market.get_orderbook(symbol="BTCUSDT", limit=50)

# Get 1-hour candles for BTCUSDT
klines = client.market.get_klines(
    symbol="BTCUSDT",
    interval="60",
    limit=200
)

# Get recent trades for BTCUSDT
trades = client.market.get_trades(symbol="BTCUSDT", limit=50)

# Get instruments information
instruments = client.market.get_instruments_info(category="linear")
```

### Enhanced Data Service

The `DataService` provides advanced market data management with caching and persistence:

- **Historical Data Caching**: Cache market data in memory for faster access
- **On-Disk Persistence**: Save and load historical data from disk
- **Simplified Data Access**: Convenient methods for accessing current market data
- **Automatic Data Management**: Automatically manage data loading, caching, and persistence
- **Thread-Safe**: All operations are thread-safe for concurrent access

```python
# Fetch historical klines with caching
historical_data = client.data.fetch_historical_klines(
    symbol="BTCUSDT",
    interval="1h",
    start_time="2023-01-01T00:00:00Z",
    end_time="2023-01-07T00:00:00Z",
    use_cache=True  # Use cached data if available
)

# Start a ticker stream with callback
def on_ticker_update(ticker_data):
    print(f"Received new ticker: {ticker_data}")

client.data.start_ticker_stream("BTCUSDT", callback=on_ticker_update)

# Start a klines stream with callback
def on_kline_update(kline_data):
    print(f"Received new candle: {kline_data}")

client.data.start_klines_stream("BTCUSDT", "1h", callback=on_kline_update)

# Get current price (from cache or API)
current_price = client.data.get_current_price("BTCUSDT")

# Get current orderbook (from cache)
orderbook = client.data.get_current_orderbook("BTCUSDT")
```

### Account Service

The `AccountService` manages account-related operations:

- **Wallet Balance**: Retrieve account balance information
- **Positions**: Get current open positions
- **Position Risk Limits**: Get and set risk limits for positions
- **Account Settings**: Configure account parameters
- **Transaction Logs**: Retrieve transaction history

```python
# Get wallet balance
wallet_balance = client.account.get_wallet_balance(account_type="UNIFIED")

# Get open positions for BTCUSDT
positions = client.account.get_positions(category="linear", symbol="BTCUSDT")

# Get account transaction logs
transaction_logs = client.account.get_transaction_log(category="linear")

# Set leverage for BTCUSDT
leverage = client.account.set_leverage(category="linear", symbol="BTCUSDT", buy_leverage="2", sell_leverage="2")

# Verify API credentials
is_valid = client.account.verify_credentials()
```

### Order Service

The `OrderService` handles trading orders:

- **Place Order**: Create new orders with various parameters
- **Cancel Order**: Cancel existing orders
- **Cancel All Orders**: Cancel all open orders
- **Amend Order**: Modify existing orders
- **Get Active Orders**: Retrieve open orders
- **Get Order History**: Retrieve historical orders
- **Get Execution History**: Retrieve trade execution history

```python
# Place a limit order
order = client.order.place_order(
    category="linear",
    symbol="BTCUSDT",
    side="Buy",
    order_type="Limit",
    qty="0.001",
    price="30000",
    time_in_force="GTC"
)

# Cancel an order
cancel_result = client.order.cancel_order(
    category="linear",
    symbol="BTCUSDT",
    order_id="1234567890"
)

# Get active orders
active_orders = client.order.get_active_orders(
    category="linear",
    symbol="BTCUSDT"
)

# Get order history
order_history = client.order.get_order_history(
    category="linear",
    symbol="BTCUSDT",
    limit=50
)

# Get execution history
executions = client.order.get_execution_list(
    category="linear",
    symbol="BTCUSDT"
)
```

### WebSocket Service

The `WebSocketService` provides real-time data streams:

- **Public Channels**: Market data streams (tickers, trades, orderbook, etc.)
- **Private Channels**: Account update streams (positions, orders, executions, etc.)
- **Custom Callbacks**: Register callback functions for processing incoming messages
- **Connection Management**: Automatic reconnection and ping/pong for connection health

```python
# Define a callback function for ticker updates
def process_ticker_update(data):
    print(f"Received ticker update: {data}")

# Start WebSocket connections
client.websocket.start()

# Subscribe to ticker updates for BTCUSDT
client.websocket.subscribe_to_ticker("BTCUSDT", callback=process_ticker_update)

# Wait for some time to receive updates
import time
time.sleep(60)  # Wait for 60 seconds

# Stop WebSocket connections
client.websocket.stop()
```

## Error Handling and Rate Limiting

The client includes comprehensive error handling and rate limiting:

- **Error Handling Decorator**: The `@with_error_handling` decorator processes API responses
- **Error Classification**: Errors are classified by type (client error, server error, etc.)
- **Detailed Error Messages**: Human-readable error messages with contextual information
- **Rate Limiting Decorator**: The `@rate_limited` decorator enforces API rate limits
- **Configurable Rate Limits**: Different rate limits for different API endpoints

```python
# Error handling is built into all API methods
try:
    result = client.market.get_tickers(category="linear", symbol="BTCUSDT")
except Exception as e:
    print(f"An error occurred: {e}")
```

## Usage Examples

### Basic Usage

```python
from src.api.bybit import BybitClient

# Initialize the client
client = BybitClient(
    testnet=True,  # Use testnet environment
    api_key="YOUR_API_KEY",
    api_secret="YOUR_API_SECRET"
)

# Get server time
server_time = client.get_server_time()
print(f"Server time: {server_time}")

# Get ticker data for BTCUSDT
tickers = client.market.get_tickers(category="linear", symbol="BTCUSDT")
print(f"BTCUSDT last price: {tickers['list'][0]['lastPrice']}")

# Get wallet balance
if client.verify_credentials():
    wallet_balance = client.account.get_wallet_balance()
    print(f"Wallet balance: {wallet_balance}")
else:
    print("Invalid API credentials")
```

### Trading Example

```python
from src.api.bybit import BybitClient

# Initialize the client
client = BybitClient(api_key="YOUR_API_KEY", api_secret="YOUR_API_SECRET")

# Get current price
ticker = client.market.get_tickers(category="linear", symbol="BTCUSDT")
current_price = float(ticker['list'][0]['lastPrice'])

# Calculate order parameters
buy_price = current_price * 0.99  # 1% below current price
qty = "0.001"  # BTC amount to buy

# Place a limit buy order
order_result = client.order.place_order(
    category="linear",
    symbol="BTCUSDT",
    side="Buy",
    order_type="Limit",
    qty=qty,
    price=str(buy_price),
    time_in_force="GTC",
    reduce_only=False,
    close_on_trigger=False
)

print(f"Order placed: {order_result}")

# Monitor the order status
order_id = order_result['orderId']
active_order = client.order.get_active_orders(
    category="linear",
    symbol="BTCUSDT",
    order_id=order_id
)

print(f"Order status: {active_order}")
```

### WebSocket Example

```python
from src.api.bybit import BybitClient
import time

# Initialize the client
client = BybitClient(api_key="YOUR_API_KEY", api_secret="YOUR_API_SECRET")

# Define callback functions
def process_ticker_update(data):
    if 'data' in data:
        ticker_data = data['data']
        print(f"Symbol: {ticker_data.get('symbol')}")
        print(f"Last Price: {ticker_data.get('lastPrice')}")
        print(f"24h Change: {ticker_data.get('price24hPcnt')}%")
        print(f"24h Volume: {ticker_data.get('volume24h')}")
        print("-" * 50)

def process_order_update(data):
    print(f"Order update received: {data}")

# Start WebSocket service
client.websocket.start()

# Subscribe to public and private channels
client.websocket.subscribe_to_ticker("BTCUSDT", process_ticker_update)

if client.verify_credentials():
    # Subscribe to private order updates (requires authentication)
    client.websocket.subscribe_private(["order"], process_order_update)

# Keep the main thread running
try:
    print("Listening for WebSocket updates (press Ctrl+C to stop)...")
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopping...")

# Stop WebSocket connections
client.websocket.stop()
```

### Enhanced Data Service Example

```python
from src.api.bybit import BybitClient
import pandas as pd
import matplotlib.pyplot as plt

# Initialize the client
client = BybitClient(data_dir="market_data")

# Fetch historical data with caching and persistence
btc_data = client.data.fetch_historical_klines(
    symbol="BTCUSDT",
    interval="1d",
    start_time="2023-01-01",
    end_time="2023-12-31"
)

# Print data summary
print(f"Retrieved {len(btc_data)} days of data for BTCUSDT")
print(f"Date range: {btc_data.index.min()} to {btc_data.index.max()}")

# Calculate some basic indicators
btc_data['sma_20'] = btc_data['close'].rolling(window=20).mean()
btc_data['sma_50'] = btc_data['close'].rolling(window=50).mean()

# Plot the data
plt.figure(figsize=(12, 6))
plt.plot(btc_data.index, btc_data['close'], label='BTC Price')
plt.plot(btc_data.index, btc_data['sma_20'], label='20-day SMA')
plt.plot(btc_data.index, btc_data['sma_50'], label='50-day SMA')
plt.title('BTC/USDT Price with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()

# Set up real-time data streaming
def on_kline_update(kline_df):
    latest_candle = kline_df.iloc[0]
    print(f"New candle: Open={latest_candle['open']}, Close={latest_candle['close']}")
    # You could update your charts or run trading algorithms here

# Start streaming klines data
client.data.start_klines_stream("BTCUSDT", "1m", callback=on_kline_update)

# Keep the program running to receive updates
try:
    print("Waiting for real-time updates (press Ctrl+C to stop)...")
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopping...")
```

## Developer Guide: Extending the API Client

### Adding New Endpoints

To add a new endpoint to an existing service:

1. Locate the appropriate service class in the `services` directory
2. Add a new method with appropriate type annotations and parameters
3. Use the `api_client.request` method to make the API call
4. Apply the `@with_error_handling` and `@rate_limited` decorators as needed

Example:

```python
@with_error_handling
@rate_limited("market")
def get_new_endpoint_data(self, symbol: str, param: str) -> Dict[str, Any]:
    """
    Get data from a new endpoint.

    Args:
        symbol: The trading symbol
        param: Additional parameter

    Returns:
        Response data from the API
    """
    path = "/v5/market/new-endpoint"
    params = {
        "symbol": symbol,
        "param": param
    }

    return self.api_client.request("GET", path, params=params)
```

### Creating a New Service

To add a completely new service:

1. Create a new file in the `services` directory (e.g., `new_service.py`)
2. Define a service class that takes a `ConnectionManager` instance
3. Implement methods for the new service functionality
4. Update the `BybitClient` class to initialize and expose the new service

### Adding New Data Models

To add new data models:

1. Edit the `models.py` file
2. Define a new dataclass with appropriate fields and types
3. Implement a `from_api_response` class method to convert API responses

Example:

```python
@dataclass
class NewModel:
    """New data model."""
    field1: str
    field2: float
    timestamp: datetime

    @classmethod
    def from_api_response(cls, response: Dict[str, Any]) -> 'NewModel':
        """Create a NewModel instance from API response data."""
        return cls(
            field1=response.get('field1', ''),
            field2=float(response.get('field2', 0)),
            timestamp=datetime.fromtimestamp(int(response.get('timestamp', 0)) / 1000)
        )
```

### Extending the Data Service

To extend the data service with new functionality:

1. Add new methods to the `DataService` class in `services/data_service.py`
2. Implement caching and persistence logic as needed
3. Add thread-safe access with locks

Example:

```python
def get_aggregated_orderbook(self, symbol: str, depth: int = 10) -> pd.DataFrame:
    """
    Get an aggregated orderbook for a symbol.

    Args:
        symbol: Trading pair symbol
        depth: Number of price levels to include

    Returns:
        DataFrame with aggregated orderbook data
    """
    with self.orderbooks_lock:
        orderbook = self.orderbooks_cache.get(symbol)

    if not orderbook:
        # Fetch fresh orderbook if not in cache
        orderbook_response = self.market_service.get_orderbook(symbol=symbol, limit=depth)
        orderbook = orderbook_response.get('result', {})

    # Process the orderbook data
    bids = pd.DataFrame(orderbook.get('bids', []), columns=['price', 'size'])
    asks = pd.DataFrame(orderbook.get('asks', []), columns=['price', 'size'])

    # Convert to numeric
    bids = bids.apply(pd.to_numeric)
    asks = asks.apply(pd.to_numeric)

    # Calculate cumulative sizes
    bids['cumulative_size'] = bids['size'].cumsum()
    asks['cumulative_size'] = asks['size'].cumsum()

    # Combine bids and asks
    result = pd.DataFrame({
        'bid_price': bids['price'].values[:depth],
        'bid_size': bids['size'].values[:depth],
        'bid_cumulative': bids['cumulative_size'].values[:depth],
        'ask_price': asks['price'].values[:depth],
        'ask_size': asks['size'].values[:depth],
        'ask_cumulative': asks['cumulative_size'].values[:depth],
    })

    return result
```

## Best Practices

### Authentication Security

- Store your API key and secret securely (e.g., using environment variables)
- Use IP restrictions on your Bybit API keys when possible
- Only grant the minimum necessary permissions to your API keys

### Error Handling

- Always implement proper error handling around API calls
- Consider implementing retries for temporary errors
- Log detailed error information for debugging

### Rate Limiting

- Respect the rate limits imposed by Bybit
- The client's built-in rate limiting helps, but additional care may be needed for high-frequency applications
- Monitor your API usage through Bybit's developer dashboard

### WebSocket Connections

- Implement proper reconnection logic for WebSocket disconnections
- Process WebSocket messages efficiently to avoid blocking the message queue
- Close WebSocket connections when they're no longer needed

### Data Management

- Use the enhanced data service for efficient market data handling
- Enable data caching for frequently accessed data
- Configure the data directory for persistent storage
- Use appropriate timeframes to minimize data volume while meeting your application's needs

## Implementation Details

### Connection Manager

The `ConnectionManager` class:

- Manages different base URLs for mainnet (`https://api.bybit.com`) and testnet (`https://api-testnet.bybit.com`)
- Creates authentication signatures using HMAC-SHA256
- Automatically synchronizes local time with server time
- Provides methods for verifying API credentials

### Rate Limiting

The `rate_limiting.py` module:

- Implements token bucket algorithm for rate limiting
- Configures different rate limits for different API endpoint categories
- Provides a `@rate_limited` decorator to enforce limits
- Handles waiting or raising exceptions when limits are exceeded

### API Client

The `ApiClient` class:

- Implements low-level HTTP request handling
- Supports both synchronous and asynchronous requests
- Handles request signing for authenticated endpoints
- Processes API responses and extracts relevant data

### Data Service

The `DataService` class:

- Implements in-memory caching for market data
- Provides disk-based persistence for historical data
- Offers thread-safe access through locks
- Manages WebSocket streams with automatic data updates
- Provides convenient methods for current market data

## Dependencies

- **pydantic**: For data validation and settings management
- **loguru**: For enhanced logging
- **pathlib**: For path manipulation
- **json**: For JSON parsing and serialization
- **websocket-client**: For WebSocket connections
- **requests**: For HTTP API requests
- **pandas**: For data manipulation and analysis
- **numpy**: For numerical computations

## Conclusion

The Bybit API Client provides a powerful and developer-friendly interface to the Bybit cryptocurrency exchange. With its comprehensive feature set, robust error handling, and efficient design, it serves as an ideal foundation for building trading applications, data collection systems, or any project that requires interaction with the Bybit exchange.

The modular design allows for easy extension and customization, ensuring that the system can adapt to a wide range of trading approaches and requirements.

For further assistance or to report issues, please contact the development team or submit an issue on the project repository.
