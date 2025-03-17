# Performance Tracking System

The Performance Tracking System is a comprehensive component of the trading platform that monitors, analyzes, and reports on trading performance. It provides detailed metrics, visualizations, and data persistence for both real-time and post-trade analysis.

## Overview

The performance tracking system offers:

- **Trade Recording**: Captures and stores detailed information about each trade
- **Performance Metrics**: Calculates key performance indicators like win rate, Sharpe ratio, and drawdown
- **Equity Tracking**: Monitors account balance and equity changes over time
- **Visualization**: Generates charts and visual representations of performance
- **Reporting**: Creates structured performance reports for analysis
- **Data Persistence**: Saves performance data for historical analysis

## Architecture

The Performance Tracking System follows a modular design centered around the `PerformanceTracker` class:

```
src/performance/
├── README.md
└── performance_tracker.py  # Core implementation of the PerformanceTracker
```

### Core Components

- **PerformanceTracker**: Central class that manages all performance tracking functionality
- **Trade Recording System**: Records trade details and outcomes
- **Metrics Calculator**: Computes performance metrics and statistics
- **Equity Tracker**: Monitors account balance and equity curves
- **Reporting Engine**: Generates performance reports and visualizations

## Key Features

### 1. Trade Recording

The system records detailed information about each trade:

- Entry and exit prices
- Trade duration
- Position size
- Realized profit/loss
- Trade direction (long/short)
- Associated strategy

### 2. Performance Metrics

The system calculates a comprehensive set of performance metrics:

- **Basic Metrics**:
  - Win rate
  - Total P&L
  - Number of trades
- **Return Metrics**:
  - Total return
  - Annualized return
- **Risk Metrics**:
  - Maximum drawdown
  - Volatility
  - Sharpe ratio
  - Sortino ratio
  - Calmar ratio
- **Trade Quality Metrics**:
  - Average win/loss
  - Win/loss ratio
  - Profit factor
  - Expectancy
  - Recovery factor

### 3. Equity Curve Tracking

The system maintains detailed records of equity changes:

- Account balance history
- Equity curve (balance + unrealized P&L)
- Drawdown history
- Monthly returns

### 4. Visualization

The system generates various performance charts:

- Equity curve charts
- Drawdown charts
- Monthly returns heatmap
- Performance distribution charts

### 5. Reporting

The system creates structured reports for analysis:

- Summary performance reports
- Detailed trade analysis
- Period-based reports (daily, monthly, yearly)
- JSON-formatted data for integration with other systems

## Usage

### Basic Usage

```python
from src.performance.performance_tracker import PerformanceTracker

# Initialize the performance tracker
performance_tracker = PerformanceTracker(
    initial_balance=10000.0,
    data_directory="data/performance",
    reporting_currency="USD"
)

# Add a completed trade
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
performance_tracker.update_balance(
    balance=10200.0,
    unrealized_pnl=150.0
)

# Get performance metrics
metrics = performance_tracker.get_performance_metrics()
print(f"Win Rate: {metrics['win_rate']:.2%}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")

# Save performance report
report_path = performance_tracker.save_performance_report("daily_report_20230105")
```

### Accessing Historical Data

```python
# Get equity history as DataFrame
equity_df = performance_tracker.get_equity_history()

# Get drawdown history as DataFrame
drawdown_df = performance_tracker.get_drawdown_history()

# Get monthly returns as DataFrame
monthly_returns = performance_tracker.get_monthly_returns()
```

### Generating Reports

```python
# Generate a summary report
summary = performance_tracker.generate_performance_summary()

# Generate a full report with all data
full_report = performance_tracker.generate_full_performance_report()
```

## Integration with Other Systems

The Performance Tracking System integrates with several components of the trading platform:

### Trading System

The main trading system uses the performance tracker to:

- Record completed trades
- Monitor account balance
- Save periodic performance reports

```python
# In main.py
self.performance_tracker.add_trade(trade)
self.performance_tracker.update_balance(account_balance, unrealized_pnl)
self.performance_tracker.save_performance_report(f"report_{timestamp}")
```

### Dashboard

The dashboard uses performance data to display:

- Real-time equity curves
- Performance metrics
- Drawdown charts
- Monthly returns heatmaps

```python
# In dashboard/services/data_service/performance_data.py
equity_history = service.performance_tracker.get_equity_history()
metrics = service.performance_tracker.get_performance_metrics()
```

### Backtesting Engine

The backtesting engine uses the performance tracker to:

- Evaluate strategy performance
- Compare strategies
- Generate backtest reports

```python
# In backtesting/backtest/engine.py
performance_tracker.update_balance(current_balance, unrealized_pnl)
metrics = performance_tracker.get_metrics()
```

## Developer Guide

### Adding Custom Metrics

To add custom performance metrics:

1. Modify the `get_performance_metrics()` method in `performance_tracker.py`
2. Add your custom metric calculation
3. Add the metric to the returned dictionary

```python
def get_performance_metrics(self) -> Dict[str, Any]:
    # Existing metrics calculation...

    # Add your custom metric
    custom_metric = calculate_custom_metric(self.trades)

    return {
        # Existing metrics...
        "custom_metric": custom_metric
    }
```

### Creating Custom Reports

To create custom performance reports:

1. Create a new method in `PerformanceTracker` class
2. Use existing data and metrics to generate your report
3. Format and return the report data

```python
def generate_custom_report(self) -> Dict[str, Any]:
    metrics = self.get_performance_metrics()
    equity_df = self.get_equity_history()

    # Generate custom report data
    report_data = {
        "custom_section": calculate_custom_data(metrics, equity_df)
    }

    return report_data
```

## Future Enhancements

Planned improvements for the Performance Tracking System:

1. **Advanced Risk Metrics**: Add Value at Risk (VaR), Expected Shortfall, and Omega Ratio
2. **Portfolio Analysis**: Add correlation analysis for multi-asset portfolios
3. **Machine Learning Integration**: Add anomaly detection and pattern recognition
4. **Real-time Alerts**: Implement alert system for performance thresholds
5. **Interactive Reporting**: Add interactive report generation with user-defined parameters
6. **Benchmark Comparison**: Add capability to compare performance against benchmarks
7. **Attribution Analysis**: Add performance attribution by strategy, asset class, or time period
