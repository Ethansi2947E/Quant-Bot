# MetaTrader 5 Handler Module Documentation

## Overview
The MT5 Handler module provides a comprehensive interface for interacting with the MetaTrader 5 trading platform. It handles all trading operations, market data retrieval, and account management functions through the MT5 API.

## Class: MT5Handler

### Constructor
```python
def __init__(self)
```
Initializes the MT5Handler class and establishes connection to the MT5 terminal.

### Connection Management

#### initialize
```python
def initialize(self) -> bool
```
Initializes connection to MT5 terminal using credentials from configuration. Returns True if successful.

#### shutdown
```python
def shutdown(self)
```
Safely closes the MT5 connection and performs cleanup.

### Account Operations

#### get_account_info
```python
def get_account_info(self) -> Dict[str, Any]
```
Retrieves current account information including:
- Balance
- Equity
- Margin
- Free margin
- Leverage
- Account currency

### Market Data Functions

#### get_market_data
```python
def get_market_data(self, symbol: str, timeframe: str, num_candles: int = 1000) -> Optional[pd.DataFrame]
```
Fetches market data (OHLCV) for specified symbol and timeframe.

#### get_historical_data
```python
def get_historical_data(self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]
```
Retrieves historical price data between specified dates.

#### get_rates
```python
async def get_rates(self, symbol: str, timeframe: str, num_candles: int = 1000) -> Optional[pd.DataFrame]
```
Async wrapper around get_market_data for compatibility with async code.

### Trading Operations

#### place_market_order
```python
def place_market_order(self, symbol: str, order_type: str, volume: float, stop_loss: float, take_profit: float, comment: str = "") -> Optional[Dict[str, Any]]
```
Places a market order with specified parameters. Includes retry logic and validation.

#### execute_trade
```python
def execute_trade(self, trade_params: Dict[str, Any]) -> Optional[List[int]]
```
Executes complex trades with multiple take-profit levels. Returns list of ticket numbers.

#### modify_position
```python
def modify_position(self, ticket: int, new_sl: float, new_tp: float) -> bool
```
Modifies stop loss and take profit levels of an existing position.

#### close_position
```python
def close_position(self, ticket: int) -> bool
```
Closes a specific position by ticket number.

### Position Management

#### get_open_positions
```python
def get_open_positions(self) -> List[Dict[str, Any]]
```
Retrieves all currently open positions with detailed information.

### Market Information

#### get_spread
```python
def get_spread(self, symbol) -> float
```
Gets current spread for a symbol in pips.

#### get_min_stop_distance
```python
def get_min_stop_distance(self, symbol: str) -> Optional[float]
```
Calculates minimum allowed stop distance for a symbol.

#### get_symbol_info
```python
def get_symbol_info(self, symbol: str) -> Optional[Any]
```
Retrieves detailed symbol information from MT5.

## Usage Example
```python
mt5_handler = MT5Handler()
if mt5_handler.initialize():
    # Get market data
    data = mt5_handler.get_market_data("EURUSD", "M15")
    
    # Place a trade
    result = mt5_handler.place_market_order(
        symbol="EURUSD",
        order_type="BUY",
        volume=0.1,
        stop_loss=1.1000,
        take_profit=1.1100
    )
    
    # Close all operations
    mt5_handler.shutdown()
```

## Notes
- Requires MetaTrader 5 terminal to be installed and configured
- All operations include retry logic and error handling
- Supports both synchronous and asynchronous operations
- Implements best practices for order execution and management
- Includes comprehensive validation for all trading operations
- Uses configuration from MT5_CONFIG for connection settings 