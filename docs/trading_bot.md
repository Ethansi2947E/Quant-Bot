# Trading Bot Documentation

## Overview
The `TradingBot` class is the core component of the trading system, responsible for managing signal generators, executing trades, and handling all trading operations. It integrates with MT5, manages multiple signal generators, and handles risk management.

## Class: TradingBot

### Initialization
```python
def __init__(self, config=None, signal_generator_class: Type[SignalGenerator] = SignalGenerator)
```
Initializes the trading bot with optional configuration and signal generator class.
- **Parameters**:
  - `config`: Optional configuration override (defaults to TRADING_CONFIG)
  - `signal_generator_class`: Signal generator class to use (defaults to SignalGenerator)
- **Functionality**:
  - Initializes MT5 connection
  - Sets up risk management
  - Configures signal generators
  - Initializes market analysis components

### Core Functions

#### Signal Generator Management
```python
def _init_signal_generators(self, default_generator_class: Type[SignalGenerator] = None)
```
Initializes signal generators from configuration.
- **Usage**: Called during initialization to set up signal generators
- **Parameters**: 
  - `default_generator_class`: Default signal generator if none specified in config
- **Returns**: None

```python
def change_signal_generator(self, signal_generator_class: Type[SignalGenerator])
```
Changes the active signal generator.
- **Usage**: Used to switch between different signal generation strategies
- **Parameters**: 
  - `signal_generator_class`: New signal generator class to use
- **Returns**: None

#### Trade Management

```python
async def execute_trade_from_signal(self, signal: Dict, is_addition: bool = False)
```
Executes a trade based on a generated signal.
- **Usage**: Called when a valid signal is generated
- **Parameters**:
  - `signal`: Trading signal dictionary
  - `is_addition`: Whether this is adding to an existing position
- **Returns**: None

```python
async def handle_signal_with_existing_positions(self, signal: Dict, existing_positions: List[Dict])
```
Handles new signals when there are existing positions.
- **Usage**: Manages position conflicts and additions
- **Parameters**:
  - `signal`: New trading signal
  - `existing_positions`: List of existing positions
- **Returns**: None

```python
async def manage_open_trades(self)
```
Monitors and manages open positions.
- **Usage**: Called periodically to update stop losses and take profits
- **Parameters**: None
- **Returns**: None

#### Market Analysis

```python
def is_market_open(self) -> bool
```
Checks if the market is currently open.
- **Usage**: Called before attempting trades
- **Returns**: Boolean indicating market status

```python
def analyze_session(self) -> str
```
Determines the current trading session.
- **Usage**: Used for session-specific trading rules
- **Returns**: String indicating current session (asian, london, new_york)

#### Signal Processing

```python
async def process_signals(self, signals: List[Dict])
```
Processes and validates trading signals.
- **Usage**: Called when new signals are generated
- **Parameters**:
  - `signals`: List of trading signals to process
- **Returns**: None

#### Risk Management

```python
def _calculate_stop_loss(self, direction: str, entry_price: float, order_block: Optional[Dict] = None)
```
Calculates stop loss levels.
- **Usage**: Called during signal generation
- **Parameters**:
  - `direction`: Trade direction (BUY/SELL)
  - `entry_price`: Entry price
  - `order_block`: Optional order block for stop loss calculation
- **Returns**: Stop loss price

#### System Management

```python
async def start(self)
```
Starts the trading bot.
- **Usage**: Called to initialize and start the trading system
- **Returns**: None

```python
async def stop(self, cleanup_only=False)
```
Stops the trading bot.
- **Usage**: Called for graceful shutdown
- **Parameters**:
  - `cleanup_only`: Whether to only clean up resources
- **Returns**: None

```python
async def main_loop(self)
```
Main trading loop.
- **Usage**: Core loop that manages trading operations
- **Returns**: None

### Configuration Options

#### Trading Configuration
- `symbols`: List of symbols to trade
- `timeframes`: List of timeframes to analyze
- `risk_per_trade`: Maximum risk per trade (default: 0.01)
- `max_position_size`: Maximum position size allowed
- `allow_position_additions`: Whether to allow adding to positions
- `close_on_reverse_signal`: Whether to close positions on reverse signals

#### Risk Management Configuration
- `max_daily_risk`: Maximum daily risk allowed
- `max_concurrent_trades`: Maximum number of concurrent trades
- `trailing_stop_enabled`: Whether to use trailing stops
- `min_rr_ratio`: Minimum risk-reward ratio required

### Usage Example
```python
# Initialize trading bot
bot = TradingBot(config=custom_config)

# Start trading
await bot.start()

# Change signal generator
await bot.change_signal_generator(SignalGenerator3)

# Stop trading
await bot.stop()
```

### Error Handling
The trading bot includes comprehensive error handling:
- MT5 connection errors
- Signal generation errors
- Trade execution errors
- Position management errors

All errors are logged and appropriate actions are taken to maintain system stability.

### Logging
The system uses structured logging with different levels:
- INFO: Normal operation information
- WARNING: Potential issues that don't stop operation
- ERROR: Serious issues that need attention
- DEBUG: Detailed information for troubleshooting

### Dependencies
- MetaTrader5 (MT5)
- Signal Generators
- Risk Manager
- Market Analysis components
- Database for trade tracking

### Best Practices
1. Always use proper risk management settings
2. Monitor system logs regularly
3. Test new signal generators in simulation first
4. Keep configuration up to date
5. Regularly backup trading data 