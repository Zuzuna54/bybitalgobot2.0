# Risk Management System

## Overview

The Risk Management System provides comprehensive risk control mechanisms for the algorithmic trading platform. It is designed to protect capital, manage exposure, and implement best practices in trading risk management through position sizing, stop loss management, circuit breakers, and other risk mitigation techniques.

This module serves as a critical safeguard to prevent excessive losses and ensure the trading system follows predefined risk parameters, even when executing automated strategies.

## Architecture

The Risk Management System follows a modular design centered around the `RiskManager` class, which applies different risk control mechanisms:

```
RiskManager (Main Class)
├── Position Sizing
│   ├── Calculate position size based on risk percentage
│   ├── Apply circuit breaker adjustments
│   └── Respect minimum order quantities
├── Trade Filtering
│   ├── Maximum open positions limit
│   ├── Signal strength threshold
│   ├── Maximum drawdown protection
│   └── Circuit breaker activation
├── Stop Loss Management
│   ├── Fixed percentage stop loss
│   ├── ATR-based stop loss
│   ├── Trailing stop activation
│   └── Trailing stop calculation
└── Performance Tracking
    ├── Trade result monitoring
    ├── Account balance history
    ├── Consecutive loss tracking
    └── Maximum drawdown monitoring
```

## Components

### Risk Manager (`risk_manager.py`)

The `RiskManager` class serves as the main component and entry point for all risk management functionality:

- **Position Sizing**: Calculates appropriate position sizes based on risk parameters
- **Stop Loss Management**: Determines stop loss and take profit levels
- **Trade Filtering**: Decides whether to take trades based on risk metrics
- **Performance Tracking**: Monitors trading performance for risk adjustment

Key methods:

- `calculate_position_size()`: Determines position size based on account balance and risk parameters
- `should_take_trade()`: Decides if a trade should be executed based on risk factors
- `calculate_stop_loss()`: Computes stop loss levels using fixed percentage or ATR
- `calculate_take_profit()`: Computes take profit levels based on risk-reward ratio
- `calculate_trailing_stop()`: Manages stop loss adjustments for profitable trades
- `update_trade_result()`: Updates internal metrics based on completed trades

### Risk Configuration

The risk management system is configured through the `RiskConfig` class, which defines the following parameters:

- `max_position_size_percent`: Maximum percentage of account balance to risk per trade (default: 5.0%)
- `max_daily_drawdown_percent`: Maximum allowed daily drawdown before halting trading (default: 3.0%)
- `default_leverage`: Default leverage to apply when not specified otherwise (default: 2x)
- `max_open_positions`: Maximum number of positions allowed at one time (default: 5)
- `use_trailing_stop`: Whether to enable trailing stops (default: true)
- `stop_loss_atr_multiplier`: Multiplier for ATR-based stop loss (default: 2.0)
- `take_profit_risk_reward_ratio`: Risk-reward ratio for take profit levels (default: 2.0)
- `circuit_breaker_consecutive_losses`: Number of consecutive losses before activating circuit breaker (default: 3)

## Features

### Position Sizing

The risk management system calculates position sizes based on:

1. **Account Risk Percentage**: Limits the risk per trade to a percentage of account balance
2. **Stop Loss Distance**: Factors in the distance between entry and stop loss
3. **Leverage**: Adjusts position size based on selected leverage
4. **Circuit Breaker**: Reduces position size after consecutive losses
5. **Minimum Order Quantity**: Ensures positions meet exchange requirements

Example position sizing calculation:

```python
# Risk 2% of a $10,000 account on a trade with a 2% stop loss
account_balance = 10000.0
risk_percentage = 2.0  # 2% risk
risk_amount = account_balance * (risk_percentage / 100.0)  # $200

entry_price = 50000.0  # BTC at $50,000
stop_loss_price = 49000.0  # Stop loss at $49,000 (2% below entry)
price_difference_pct = abs(entry_price - stop_loss_price) / entry_price  # 0.02 (2%)

position_size = risk_amount / (entry_price * price_difference_pct)  # 0.2 BTC
```

### Stop Loss Management

Multiple approaches to stop loss calculation:

1. **Percentage-Based**: Fixed percentage below/above entry price
2. **ATR-Based**: Dynamic stop loss based on Average True Range for volatility-adjusted protection
3. **Trailing Stop**: Moves stop loss to lock in profits as price moves favorably

The system handles both:

- **Initial Stop Loss**: Set when the position is opened
- **Trailing Stop**: Adjusted as the position moves in favor

### Circuit Breaker

The circuit breaker mechanism helps protect capital during adverse market conditions:

1. **Consecutive Loss Monitoring**: Tracks consecutive losing trades
2. **Position Size Reduction**: Reduces position size after hitting threshold
3. **Risk Limitation**: Helps prevent large drawdowns during losing streaks

### Trade Filtering

The risk manager filters trades based on:

1. **Maximum Open Positions**: Limits the number of concurrent positions
2. **Signal Strength**: Requires minimum signal quality threshold
3. **Drawdown Protection**: Prevents new trades if daily drawdown limit is exceeded
4. **Symbol Exposure**: Prevents multiple positions in the same symbol

## Usage

### Basic Usage

```python
from src.risk_management.risk_manager import RiskManager
from src.config.config_manager import RiskConfig

# Initialize with default risk parameters
risk_config = RiskConfig(
    max_position_size_percent=2.0,
    max_daily_drawdown_percent=3.0,
    default_leverage=2,
    max_open_positions=5
)
risk_manager = RiskManager(risk_config)

# Calculate position size for a trade
position_size = risk_manager.calculate_position_size(
    symbol="BTCUSDT",
    account_balance=10000.0,
    entry_price=50000.0,
    stop_loss_price=49000.0
)

# Determine if a trade should be taken
should_trade = risk_manager.should_take_trade(
    symbol="BTCUSDT",
    signal_strength=0.75,
    account_balance=10000.0
)

# Calculate stop loss price
stop_loss = risk_manager.calculate_stop_loss(
    entry_price=50000.0,
    is_long=True,
    atr_value=1500.0  # Optional ATR value
)

# Calculate take profit price
take_profit = risk_manager.calculate_take_profit(
    entry_price=50000.0,
    stop_loss_price=stop_loss,
    is_long=True
)

# Update risk manager with trade result
risk_manager.update_trade_result({
    "realized_pnl": 150.0,
    "status": "closed"
})
```

### Integration with Trading Systems

```python
# Inside a trading system or strategy
def execute_trade(symbol, signal, account_balance):
    # Check if trade meets risk criteria
    if not risk_manager.should_take_trade(symbol, signal.strength, account_balance):
        logger.info(f"Trade rejected by risk manager for {symbol}")
        return None

    # Calculate entry price (simplified)
    entry_price = get_current_price(symbol)

    # Calculate stop loss price
    atr_value = calculate_atr(symbol)
    stop_loss = risk_manager.calculate_stop_loss(
        entry_price=entry_price,
        is_long=(signal.direction == "BUY"),
        atr_value=atr_value
    )

    # Calculate position size
    position_size = risk_manager.calculate_position_size(
        symbol=symbol,
        account_balance=account_balance,
        entry_price=entry_price,
        stop_loss_price=stop_loss
    )

    # Execute the trade with calculated parameters
    trade_result = place_order(
        symbol=symbol,
        side=signal.direction,
        quantity=position_size,
        price=entry_price,
        stop_loss=stop_loss
    )

    return trade_result
```

## Developer Guide

### Extending Risk Management

#### Adding Custom Position Sizing Logic

To implement custom position sizing:

1. Subclass the `RiskManager` class:

```python
from src.risk_management.risk_manager import RiskManager

class EnhancedRiskManager(RiskManager):
    def calculate_position_size(
        self,
        symbol: str,
        account_balance: float,
        entry_price: float,
        stop_loss_price: float,
        leverage: int = 1,
        min_order_qty: float = 0.0
    ) -> float:
        # Custom position sizing logic

        # For example, position size based on Kelly criterion
        win_rate = self.calculate_win_rate_for_symbol(symbol)
        expected_return = self.calculate_expected_return(symbol)

        kelly_fraction = win_rate - ((1 - win_rate) / expected_return)
        kelly_fraction = min(kelly_fraction, 0.2)  # Cap at 20%

        position_size = (account_balance * kelly_fraction) / entry_price

        # Apply additional constraints and rounding
        return self._round_position_size(position_size, symbol)

    def calculate_win_rate_for_symbol(self, symbol: str) -> float:
        # Implement win rate calculation based on historical data
        # ...
        return win_rate

    def calculate_expected_return(self, symbol: str) -> float:
        # Implement expected return calculation
        # ...
        return expected_return
```

2. Use your enhanced risk manager:

```python
risk_manager = EnhancedRiskManager(risk_config)
```

#### Creating Custom Stop Loss Strategies

To implement custom stop loss strategies:

```python
from src.risk_management.risk_manager import RiskManager

class AdvancedStopLossManager(RiskManager):
    def calculate_stop_loss(
        self,
        entry_price: float,
        is_long: bool,
        atr_value: Optional[float] = None,
        recent_swing_points: Optional[List[float]] = None
    ) -> float:
        # Use swing points if available
        if recent_swing_points:
            if is_long:
                # For long positions, find nearest swing low below entry
                valid_swing_lows = [p for p in recent_swing_points if p < entry_price]
                if valid_swing_lows:
                    return max(valid_swing_lows)
            else:
                # For short positions, find nearest swing high above entry
                valid_swing_highs = [p for p in recent_swing_points if p > entry_price]
                if valid_swing_highs:
                    return min(valid_swing_highs)

        # Fall back to default ATR-based or percentage-based stop loss
        return super().calculate_stop_loss(entry_price, is_long, atr_value)
```

#### Implementing Custom Risk Filters

To add custom risk filters:

```python
from src.risk_management.risk_manager import RiskManager

class EnhancedRiskFilters(RiskManager):
    def should_take_trade(
        self,
        symbol: str,
        signal_strength: float,
        account_balance: float,
        market_volatility: Optional[float] = None,
        trade_correlation: Optional[float] = None
    ) -> bool:
        # First apply standard risk filters
        if not super().should_take_trade(symbol, signal_strength, account_balance):
            return False

        # Additional filter: Market volatility
        if market_volatility and market_volatility > 4.0:  # 4% volatility threshold
            logger.info(f"Rejecting trade due to high market volatility: {market_volatility}%")
            return False

        # Additional filter: Correlation with existing positions
        if trade_correlation and trade_correlation > 0.7:  # 70% correlation threshold
            logger.info(f"Rejecting trade due to high correlation with existing positions")
            return False

        return True
```

### Risk Management Best Practices

1. **Always Validate Risk Parameters**

   - Ensure risk percentages are reasonable (1-3% per trade is common)
   - Verify stop loss prices are valid and not too close to entry
   - Test position sizing calculations with multiple scenarios

2. **Handle Edge Cases**

   - Account for zero or negative values in calculations
   - Implement sensible defaults when optional parameters are missing
   - Handle situations where minimum order size exceeds risk-based size

3. **Logging and Monitoring**

   - Log all risk decisions for later analysis
   - Track key metrics like win rate, average position size, and drawdown
   - Create alerts for when risk limits are approached or exceeded

4. **Testing Risk Management**
   - Create unit tests with multiple scenarios
   - Test with extreme market conditions
   - Verify behavior with consecutive losses

## API Reference

### RiskManager

```python
class RiskManager:
    def __init__(self, risk_config: RiskConfig)

    def calculate_position_size(
        self,
        symbol: str,
        account_balance: float,
        entry_price: float,
        stop_loss_price: float,
        leverage: int = 1,
        min_order_qty: float = 0.0
    ) -> float

    def _round_position_size(self, position_size: float, symbol: str) -> float

    def update_trade_result(self, trade_result: Dict[str, Any]) -> None

    def update_account_balance(self, balance: float) -> None

    def is_max_drawdown_breached(self) -> bool

    def should_apply_circuit_breaker(self) -> bool

    def calculate_stop_loss(
        self,
        entry_price: float,
        is_long: bool,
        atr_value: Optional[float] = None
    ) -> float

    def calculate_take_profit(
        self,
        entry_price: float,
        stop_loss_price: float,
        is_long: bool
    ) -> float

    def should_use_trailing_stop(self, unrealized_profit_pct: float) -> bool

    def calculate_trailing_stop(
        self,
        current_price: float,
        entry_price: float,
        is_long: bool,
        atr_value: Optional[float] = None,
        highest_price: Optional[float] = None,
        lowest_price: Optional[float] = None
    ) -> float

    def get_recommended_leverage(self, symbol: str, volatility: Optional[float] = None) -> int

    def get_open_positions(self) -> List[Dict[str, Any]]

    def reset_daily_metrics(self) -> None

    def should_take_trade(
        self,
        symbol: str,
        signal_strength: float,
        account_balance: float
    ) -> bool
```

### RiskConfig

```python
class RiskConfig:
    max_position_size_percent: float = 5.0
    max_daily_drawdown_percent: float = 3.0
    default_leverage: int = 2
    max_open_positions: int = 5
    use_trailing_stop: bool = True
    stop_loss_atr_multiplier: float = 2.0
    take_profit_risk_reward_ratio: float = 2.0
    circuit_breaker_consecutive_losses: int = 3
```

## Integration with Other Modules

The Risk Management module interacts with several other parts of the trading system:

1. **Trading Strategies**: Provides position sizing and filtering for strategy signals
2. **Order Management**: Sets stop loss and take profit levels for orders
3. **Portfolio Management**: Maintains portfolio exposure within risk limits
4. **Performance Tracking**: Monitors and reports on risk-adjusted performance
5. **Backtesting**: Applies consistent risk rules during strategy testing

## Future Enhancements

Planned improvements for the risk management system:

1. **Portfolio-level risk management** to consider correlations between positions
2. **Dynamic risk adjustment** based on recent performance and market conditions
3. **Machine learning integration** for predictive risk assessment
4. **Multi-timeframe volatility analysis** for more precise risk estimation
5. **Custom risk profiles** for different market regimes (trending, ranging, high volatility)
6. **Risk-adjusted performance metrics** for strategy evaluation
7. **Visualizations** of risk metrics and position sizing
