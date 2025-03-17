# Risk Manager Module Documentation

## Overview
The Risk Manager module provides comprehensive risk management and position sizing functionality for trading operations. It implements multiple risk control mechanisms, dynamic position sizing, and drawdown protection.

## Class: RiskManager

### Constructor
```python
def __init__(self, mt5_handler: Optional[MT5Handler] = None)
```
Initializes the RiskManager with configurable risk parameters:

#### Core Risk Parameters
- `max_risk_per_trade`: 1% max risk per trade
- `max_daily_loss`: 2% max daily loss
- `max_daily_risk`: 3% max daily risk exposure
- `max_weekly_loss`: 5% max weekly loss
- `max_monthly_loss`: 10% max monthly loss
- `max_drawdown_pause`: 5% drawdown pause threshold

#### Position Management
- `max_concurrent_trades`: 1 position at a time
- `max_daily_trades`: 2 trades per day
- `max_weekly_trades`: 8 trades per week
- `min_trades_spacing`: 4 hours between trades

### Risk Management Functions

#### calculate_position_size
```python
def calculate_position_size(self, account_balance: float, risk_per_trade: float, entry_price: float, stop_loss_price: float, symbol: str, market_condition: str = 'normal', volatility_state: str = 'normal', session: str = 'normal', correlation: float = 0.0, confidence_score: float = 0.5) -> float
```
Calculates optimal position size based on multiple factors including risk, market conditions, and volatility.

#### validate_trade
```python
def validate_trade(self, account_balance: float, risk_amount: float, entry_price: float, stop_loss: float, take_profit: float, signal_type: str, confidence: float, current_daily_risk: float = 0.0, current_weekly_risk: float = 0.0, daily_trades: int = 0, weekly_trades: int = 0, current_drawdown: float = 0.0, consecutive_losses: int = 0, last_trade_time: Optional[datetime] = None, correlations: Optional[Dict[str, float]] = None) -> Tuple[bool, str]
```
Validates trade parameters against risk limits and trading rules.

### Position Sizing Functions

#### calculate_dynamic_position_size
```python
def calculate_dynamic_position_size(self, account_balance: float, risk_amount: float, entry_price: float, stop_loss: float, symbol: str, market_condition: str, volatility_state: str, session: str, correlation: float, confidence_score: float) -> float
```
Calculates position size with dynamic adjustments based on market conditions.

#### apply_partial_profits
```python
def apply_partial_profits(self, position_size: float, entry_price: float, stop_loss: float, min_lot: float = 0.01) -> List[Dict]
```
Calculates partial profit targets while respecting minimum lot size constraints.

### Risk Validation Functions

#### validate_trade_risk
```python
def validate_trade_risk(self, account_balance: float, risk_amount: float, current_daily_risk: float, current_weekly_risk: float, daily_trades: int, weekly_trades: int, current_drawdown: float, consecutive_losses: int, last_trade_time: Optional[datetime] = None, correlations: Optional[Dict[str, float]] = None) -> Tuple[bool, str]
```
Validates trade risk against multiple risk parameters.

#### check_daily_limits
```python
def check_daily_limits(self, account_balance: float, new_trade_risk: float) -> tuple[bool, str]
```
Checks if adding a new trade's risk will exceed daily risk limits.

### Stop Loss Management

#### calculate_stop_loss
```python
def calculate_stop_loss(self, df: pd.DataFrame, signal_type: str, entry_price: float, atr_value: Optional[float] = None, swing_low: Optional[float] = None, swing_high: Optional[float] = None, volatility_state: str = 'normal', market_condition: str = 'normal') -> Tuple[float, List[Dict[str, float]]]
```
Calculates optimal stop loss and take profit levels based on market conditions.

#### calculate_trailing_stop
```python
def calculate_trailing_stop(self, trade: Dict[str, Any], current_price: float, current_atr: Optional[float] = None, market_condition: str = 'normal') -> Tuple[bool, float]
```
Calculates trailing stop levels based on price action and market conditions.

### Performance Monitoring

#### calculate_drawdown
```python
def calculate_drawdown(self) -> float
```
Calculates current drawdown based on peak balance versus current balance.

#### reset_daily_stats
```python
def reset_daily_stats(self)
```
Resets daily trading statistics.

## Usage Example
```python
risk_manager = RiskManager(mt5_handler)

# Calculate position size
position_size = risk_manager.calculate_position_size(
    account_balance=10000,
    risk_per_trade=0.01,
    entry_price=1.2000,
    stop_loss_price=1.1950,
    symbol="EURUSD",
    market_condition="trending",
    volatility_state="normal"
)

# Validate trade
is_valid, reason = risk_manager.validate_trade(
    account_balance=10000,
    risk_amount=100,
    entry_price=1.2000,
    stop_loss=1.1950,
    take_profit=1.2100,
    signal_type="BUY",
    confidence=0.8
)
```

## Notes
- Implements comprehensive risk management rules
- Provides dynamic position sizing based on multiple factors
- Includes drawdown protection mechanisms
- Supports partial profit targets
- Implements trade spacing and frequency limits
- Provides detailed validation feedback
- Includes performance tracking and statistics 