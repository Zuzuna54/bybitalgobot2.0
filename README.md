# Bybit Algorithmic Trading System

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)

A comprehensive, modular cryptocurrency trading platform designed for automated trading on the Bybit exchange.

## Features

- 🤖 **Multiple Trading Strategies**: Includes several configurable strategies from EMA crossovers to advanced multi-indicator systems
- 📊 **Real-time Dashboard**: Monitor trades, performance, and market data through an interactive web dashboard
- 📈 **Backtesting Engine**: Test strategies on historical data to validate performance before live deployment
- 📝 **Paper Trading**: Practice trading with simulated funds in a real-market environment
- 🛡️ **Risk Management**: Built-in risk controls including position sizing, stop-losses, and maximum drawdown limits
- 📱 **Performance Tracking**: Detailed analytics and performance metrics for your trading strategies
- 🔄 **API Integration**: Seamless integration with Bybit V5 API with a modular, maintainable design
- ⚙️ **Configurable**: Extensive configuration options for all system components
- 💾 **Data Caching**: In-memory and on-disk caching for market data to improve performance
- 🧩 **Modular Architecture**: Clearly separated components for easy maintenance and extension

## System Architecture Overview

The system follows a modular architecture with clear separation of concerns:

```
src/
├── api/                  # API integration with Bybit exchange
├── backtesting/          # Backtesting engine for strategy testing
├── config/               # Configuration management
├── dashboard/            # Web-based monitoring dashboard
├── indicators/           # Technical indicators for strategy development
├── models/               # Data models and structures
├── paper_trading/        # Paper trading simulation engine
├── performance/          # Performance tracking and reporting
├── risk_management/      # Risk controls and management
├── strategies/           # Trading strategy implementations
├── trade_execution/      # Order execution handling
├── trade_management/     # Trade and position management
└── main.py               # Main application entry point
```

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
bash run_dashboard.sh --mode paper --config config/paper_trading_config.yaml
```

## Core Components

### Bybit API Client

The Bybit API client follows a modern, modular architecture for improved maintainability and performance:

#### Key Features:

- **Complete API Coverage**: Supports all major Bybit V5 API endpoints
- **Robust Error Handling**: Comprehensive error handling with detailed error messages
- **Rate Limiting**: Built-in rate limiting to prevent API request throttling
- **WebSocket Support**: Real-time data streams for market data and account updates
- **Data Caching**: In-memory and disk-based caching for market data
- **Data Persistence**: Save and load historical data from disk

#### Architecture:

- **Client (`src/api/bybit/client.py`)**: Main entry point providing access to all services
- **Connection Manager (`src/api/bybit/core/connection.py`)**: Manages API connectivity and authentication
- **Service Modules**:
  - **Market Service**: Retrieves market data (tickers, orderbooks, candles)
  - **Account Service**: Manages account operations (balance, positions)
  - **Order Service**: Handles order operations (placement, cancellation)
  - **WebSocket Service**: Provides real-time data streaming
  - **Data Service**: Enhanced market data with caching and persistence

### Trading Strategies

The system includes multiple implemented trading strategies:

1. **EMA Crossover**: Uses exponential moving average crossovers to identify trend changes
2. **RSI Reversal**: Identifies potential reversals using the Relative Strength Index
3. **Bollinger Breakout**: Trades breakouts from Bollinger Bands
4. **MACD Trend Following**: Uses MACD indicator for trend following
5. **VWAP Trend Trading**: Uses Volume Weighted Average Price for trend identification

- To BE IMPLEMENTED

6. **Bollinger Mean Reversion**: Trades mean reversion with Bollinger Bands
7. **Breakout Trading**: Identifies and trades price breakouts
8. **ATR Volatility Scalping**: Scalps markets based on volatility measured by Average True Range
9. **ADX Strength Confirmation**: Uses ADX to confirm trend strength
10. **Golden Cross**: Identifies long-term trend changes with MA crossovers
11. **Keltner Channel Breakout**: Trades breakouts from Keltner Channels

- TO BE IMPLEMETED

All strategies inherit from a common `BaseStrategy` class that provides core functionality and a consistent interface.

#### Strategy Manager:

The `StrategyManager` orchestrates strategy operations:

- Dynamically loads and initializes strategies
- Aggregates signals from different strategies
- Tracks performance metrics
- Dynamically adjusts strategy weights based on performance

### Performance Tracking

The Performance Tracking System provides comprehensive monitoring and analysis of trading results:

#### Key Features:

- **Trade Recording**: Captures detailed information about each trade
- **Performance Metrics**: Calculates KPIs (win rate, Sharpe ratio, drawdown, etc.)
- **Equity Tracking**: Monitors account balance and equity over time
- **Visualization**: Generates performance charts (equity curve, drawdown, etc.)
- **Reporting**: Creates structured performance reports for analysis

#### Core Components:

- **PerformanceTracker**: Central class managing all tracking functionality
- **Metrics Calculator**: Computes a wide range of performance statistics
- **Reporting Engine**: Generates detailed performance reports and visualizations

### Backtesting Engine

The Backtesting Engine allows testing trading strategies on historical data:

#### Key Features:

- **Historical Data Processing**: Loads and prepares market data for backtesting
- **Strategy Evaluation**: Tests trading strategies with realistic simulations
- **Performance Analysis**: Calculates detailed performance metrics
- **Trade Simulation**: Simulates order execution with slippage and fees
- **Visualization**: Generates performance charts and trade visualizations

### Dashboard

The Dashboard provides a web-based interface for monitoring and controlling the trading system:

#### Key Features:

- **Performance Monitoring**: Real-time tracking of trading performance
- **Market Data Visualization**: Charts and displays for market data
- **Trade Management**: View and manage active trades
- **Strategy Management**: Monitor and control trading strategies
- **System Configuration**: Configure system parameters through the UI

#### Architecture:

- **Application Layer**: The `app.py` serves as the entry point for the Dash application
- **Layout System**: Reusable UI components organized as a tab-based interface
- **Data Services**: Centralized data providers that fetch and process data
- **Callback System**: Manages interactive updates and user interactions

### Risk Management

The Risk Management System provides protection against excessive losses:

#### Key Features:

- **Position Sizing**: Calculates appropriate position sizes based on risk tolerance
- **Stop Loss Management**: Implements and tracks stop losses
- **Risk Limits**: Enforces maximum risk exposure limits
- **Drawdown Control**: Monitors and manages drawdown
- **Circuit Breakers**: Halts trading when risk metrics exceed thresholds

## Configuration System

The system uses a hierarchical configuration approach:

1. **Default Configuration**: Base settings in `src/config/default_config.json`
2. **User Configuration**: User-specific settings that override defaults
3. **Command-line Parameters**: Runtime parameters that take highest precedence

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

## Developer Guide

### Creating Custom Strategies

To create a custom trading strategy:

1. Create a new Python file in the `src/strategies` directory:

```python
from src.strategies.base_strategy import BaseStrategy, Signal, SignalType

class MyCustomStrategy(BaseStrategy):
    def __init__(self, config, indicator_manager):
        super().__init__(config, indicator_manager)
        self.name = "my_custom"
        self.description = "My Custom Trading Strategy"
        self._init_indicators()

    def _init_indicators(self):
        # Initialize required indicators
        self.indicators.add_indicator("rsi", {"length": 14})
        self.indicators.add_indicator("ema", {"length": 50})

    def generate_signals(self, data):
        signals = []
        # Generate signals based on your logic
        # Example: Generate a buy signal when RSI crosses below 30
        rsi = data["rsi_14"].iloc[-1]
        prev_rsi = data["rsi_14"].iloc[-2]

        if prev_rsi >= 30 and rsi < 30:
            signals.append(Signal(
                signal_type=SignalType.BUY,
                symbol=data.symbol,
                timestamp=data.index[-1],
                price=data["close"].iloc[-1],
                strength=0.8,
                metadata={"strategy": self.name}
            ))

        return signals
```

2. Register your strategy in the strategy manager:
   - Add your strategy to `src/strategies/manager/loader.py`
   - Make sure to import and register the strategy class

### Extending the API Client

To add new functionality to the Bybit API client:

1. Identify the appropriate service module for your addition
2. Add your new method to the service class
3. Update the main client class to expose the new functionality if needed

Example - Adding a new method to `MarketDataService`:

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

### Working with the Performance Tracker

To use the Performance Tracker in your code:

```python
from src.performance.performance_tracker import PerformanceTracker

# Initialize the performance tracker
performance_tracker = PerformanceTracker(
    initial_balance=10000.0,
    data_directory="data/performance"
)

# Record a completed trade
performance_tracker.add_trade({
    "symbol": "BTCUSDT",
    "entry_time": "2023-01-01T10:00:00",
    "exit_time": "2023-01-02T14:30:00",
    "entry_price": 50000.0,
    "exit_price": 52000.0,
    "quantity": 0.1,
    "realized_pnl": 200.0,
    "direction": "long",
    "strategy": "ema_crossover"
})

# Update account balance and unrealized P&L
performance_tracker.update_balance(10200.0, 150.0)

# Get performance metrics
metrics = performance_tracker.get_performance_metrics()

# Generate and save a performance report
report_path = performance_tracker.save_performance_report("daily_report")
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Project Roadmap

Planned future enhancements:

1. **Machine Learning Integration**: Add ML-based strategy optimization and prediction
2. **Multi-Exchange Support**: Expand beyond Bybit to other cryptocurrency exchanges
3. **Advanced Risk Management**: Implement portfolio-level risk controls and hedging strategies
4. **Real-time Alerts**: Add configurable alerts for important events
5. **Mobile Interface**: Develop a mobile companion app for on-the-go monitoring
6. **Strategy Marketplace**: Create a marketplace for sharing and discovering strategies

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational purposes only. Do not risk money which you are afraid to lose. USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHORS AND ALL AFFILIATES ASSUME NO RESPONSIBILITY FOR YOUR TRADING RESULTS.

---

Built with ❤️ by Your Name
