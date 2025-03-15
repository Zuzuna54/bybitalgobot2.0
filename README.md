# Bybit Algorithmic Trading System

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)

A comprehensive, modular cryptocurrency trading platform designed for automated trading on the Bybit exchange.

## Features

- 🤖 **Multiple Trading Strategies**: Includes 10+ configurable strategies from simple moving average crossovers to complex multi-indicator systems
- 📊 **Real-time Dashboard**: Monitor trades, performance, and market data through an interactive web dashboard
- 📈 **Backtesting Engine**: Test strategies on historical data to validate performance before live deployment
- 📝 **Paper Trading**: Practice trading with simulated funds in a real-market environment
- 🛡️ **Risk Management**: Built-in risk controls including position sizing, stop-losses, and maximum drawdown limits
- 📱 **Performance Tracking**: Detailed analytics and performance metrics for your trading strategies
- 🔄 **API Integration**: Seamless integration with Bybit exchange API with a modular, maintainable design
- ⚙️ **Configurable**: Extensive configuration options for all system components

## Screenshot

_[Dashboard screenshot would go here]_

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Bybit account (optional for backtesting)
- API keys (for paper/live trading)

### Installation

1. Clone the repository:

```bash
   git clone https://github.com/yourusername/bybit-trading-system.git
   cd bybit-trading-system
```

2. Create a virtual environment and install dependencies:

```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
```

3. Set up your environment variables:

Create a `.env` file in the root directory:

```
# Bybit API Configuration
BYBIT_USE_TESTNET=True
   BYBIT_API_KEY=your_api_key
   BYBIT_API_SECRET=your_api_secret
```

### Running the Example

To test the Bybit API client:

```bash
python src/examples/bybit_example.py
```

This example demonstrates:

- Connecting to the Bybit API
- Fetching market data (ticker, orderbook, klines)
- Account operations (requires API credentials)
- WebSocket streaming for real-time data

### Running the System

#### Backtesting

Test strategies on historical data:

```bash
python src/main.py --config config/backtest_config.yaml --backtest
```

#### Paper Trading

Practice with simulated funds:

```bash
python src/main.py --config config/paper_trading_config.yaml
```

#### Live Trading

Trade with real funds (use with caution):

```bash
python src/main.py --config config/live_trading_config.yaml
```

#### Running the Dashboard

The dashboard can be run separately:

```bash
python run_dashboard.py --mode paper --config config/paper_trading_config.yaml
```

## Configuration

The system uses a hierarchical configuration system:

1. Default configuration in `src/config/default_config.json`
2. User-specific configuration overrides
3. Command-line parameter overrides

Example configuration:

```json
{
  "exchange": {
    "name": "bybit",
    "testnet": true
  },
  "pairs": [
    {
      "symbol": "BTCUSDT",
      "quote_currency": "USDT",
      "base_currency": "BTC",
      "is_active": true
    }
  ],
  "strategies": [
    {
      "name": "ema_crossover",
      "is_active": true,
      "timeframe": "1h",
      "parameters": {
        "fast_ema": 9,
        "slow_ema": 21
      }
    }
  ],
  "risk": {
    "max_position_size_percent": 5.0,
    "max_daily_drawdown_percent": 3.0,
    "default_leverage": 2
  }
}
```

## Bybit API Client Architecture

The Bybit API client has been refactored to follow a modular, maintainable design:

### Core Components

- **Connection Management** (`src/api/bybit/core/connection.py`): Handles API endpoints, authentication, and connection settings
- **Error Handling** (`src/api/bybit/core/error_handling.py`): Centralized error handling and custom exceptions
- **Rate Limiting** (`src/api/bybit/core/rate_limiting.py`): Implements rate limiting to comply with Bybit API restrictions
- **API Client** (`src/api/bybit/core/api_client.py`): Core request handling and response processing

### Service Modules

- **Account Service** (`src/api/bybit/services/account_service.py`): Manages account operations, balances, and positions
- **Market Service** (`src/api/bybit/services/market_service.py`): Handles market data retrieval (tickers, orderbooks, etc.)
- **Order Service** (`src/api/bybit/services/order_service.py`): Manages order placement, cancellation, and status
- **WebSocket Service** (`src/api/bybit/services/websocket_service.py`): Provides real-time data streaming

### Client Interface

- **Client** (`src/api/bybit/client.py`): Main entry point that integrates all services
- **Models** (`src/api/bybit/models.py`): Data models and type definitions

## Available Strategies

The system includes multiple trading strategies:

1. **EMA Crossover**: Uses exponential moving average crossovers to identify trend changes
2. **RSI Reversal**: Identifies potential reversals using the Relative Strength Index
3. **Bollinger Mean Reversion**: Trades mean reversion with Bollinger Bands
4. **MACD Trend Following**: Uses MACD indicator for trend following
5. **Breakout Trading**: Identifies and trades price breakouts
6. **VWAP Trend Trading**: Uses Volume Weighted Average Price for trend identification
7. **ATR Volatility Scalping**: Scalps markets based on volatility measured by Average True Range
8. **ADX Strength Confirmation**: Uses ADX to confirm trend strength
9. **Golden Cross**: Identifies long-term trend changes with MA crossovers
10. **Keltner Channel Breakout**: Trades breakouts from Keltner Channels

## System Architecture

The system follows a modular architecture with clear separation of concerns:

- **Main Application** (`src/main.py`): Core orchestration
- **Dashboard** (`run_dashboard.py`, `src/dashboard/`): Visualization and control
- **API Integration** (`src/api/`): Exchange communication with modular design
- **Market Data** (`src/data/`): Data handling
- **Trading Strategies** (`src/strategies/`): Strategy implementation
- **Risk Management** (`src/risk_management/`): Risk control
- **Trade Management** (`src/trade_management/`): Order execution
- **Performance** (`src/performance/`): Performance tracking
- **Backtesting** (`src/testing/`): Historical testing

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational purposes only. Do not risk money which you are afraid to lose. USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHORS AND ALL AFFILIATES ASSUME NO RESPONSIBILITY FOR YOUR TRADING RESULTS.

---

Built with ❤️ by [Your Name]
