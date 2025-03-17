# Telegram Bot Module Documentation

## Overview
The Telegram Bot module provides a comprehensive interface for trading notifications, alerts, and command handling through Telegram. It implements secure user authentication, error handling, and real-time trading updates.

## Class: TelegramBot

### Constructor
```python
def __init__(self)
```
Initializes the TelegramBot with:
- Trading state management
- Performance metrics tracking
- User authentication
- Command handlers

### Core Functions

#### initialize
```python
async def initialize(self, config) -> bool
```
Initializes the Telegram bot with configuration settings and sets up command handlers. Returns success status.

#### stop
```python
async def stop(self)
```
Safely stops the Telegram bot and cleans up resources.

### Command Handlers

#### start_command
```python
async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE)
```
Handles the /start command for bot initialization.

#### handle_enable_command
```python
async def handle_enable_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE)
```
Enables trading functionality.

#### handle_disable_command
```python
async def handle_disable_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE)
```
Disables trading functionality.

#### status_command
```python
async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE)
```
Shows current trading status.

### Alert Functions

#### send_setup_alert
```python
async def send_setup_alert(self, symbol: str, timeframe: str, setup_type: str, confidence: float)
```
Sends trading setup alerts to users.

#### send_trade_alert
```python
async def send_trade_alert(self, chat_id: int, symbol: str, direction: str, entry: float, sl: float, tp: float, confidence: float, reason: str)
```
Sends detailed trade alerts with entry, stop loss, and take profit levels.

#### send_trade_error_alert
```python
async def send_trade_error_alert(self, symbol: str, error_type: str, details: str, retry_count: int = 0, additional_info: dict = None)
```
Sends trade execution error alerts with detailed information.

### Performance Updates

#### send_performance_update
```python
async def send_performance_update(self, chat_id: int, total_trades: int, winning_trades: int, total_profit: float)
```
Sends trading performance updates.

#### update_metrics
```python
def update_metrics(self, trade_result: Dict)
```
Updates performance metrics with new trade results.

### Trade Management

#### send_trade_update
```python
async def send_trade_update(self, trade_id: int, symbol: str, action: str, price: float, pnl: Optional[float] = None, r_multiple: Optional[float] = None)
```
Sends trade status updates including profit/loss information.

#### send_management_alert
```python
async def send_management_alert(self, message: str, alert_type: str = "info") -> None
```
Sends trade management alerts with different severity levels.

### Security Functions

#### check_auth
```python
async def check_auth(self, chat_id: int) -> bool
```
Verifies user authorization for bot commands.

### Error Handling

#### send_error_alert
```python
async def send_error_alert(self, message: str) -> bool
```
Sends error alerts with retry logic and timeout handling.

## Usage Example
```python
bot = TelegramBot()
await bot.initialize(config)

# Send trade alert
await bot.send_trade_alert(
    chat_id=12345,
    symbol="EURUSD",
    direction="BUY",
    entry=1.2000,
    sl=1.1950,
    tp=1.2100,
    confidence=0.8,
    reason="Strong trend continuation"
)

# Send performance update
await bot.send_performance_update(
    chat_id=12345,
    total_trades=100,
    winning_trades=65,
    total_profit=1500.0
)
```

## Notes
- Implements secure user authentication
- Includes comprehensive error handling
- Supports multiple alert types
- Provides performance tracking
- Implements retry logic for message delivery
- Includes timeout handling
- Supports multiple users
- Provides detailed trade information
- Implements command rate limiting 