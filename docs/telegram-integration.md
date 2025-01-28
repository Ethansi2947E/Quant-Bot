# Telegram Integration

## Overview
The Telegram integration provides real-time notifications and command handling for the trading bot. It allows users to monitor trades, receive alerts, and control the bot's operation through Telegram messages.

## Features

### Authentication
- User authentication based on allowed user IDs
- Secure command handling for authorized users only
- Automatic unauthorized access logging

### Commands
- `/start` - Initialize bot and display welcome message
- `/enable` - Enable automated trading
- `/disable` - Disable automated trading
- `/status` - Check bot status and performance
- `/metrics` - View detailed performance metrics
- `/history` - View recent trade history
- `/help` - Display available commands
- `/stop` - Stop bot and close all trades

### Notifications

#### Trade Alerts
```python
await bot.send_trade_alert(
    chat_id=user_id,
    symbol="EURUSD",
    direction="BUY",
    entry=1.1000,
    sl=1.0950,
    tp=1.1100,
    confidence=0.85,
    reason="Strong uptrend on H4"
)
```

#### Performance Updates
```python
await bot.send_performance_update(
    chat_id=user_id,
    total_trades=10,
    winning_trades=7,
    total_profit=500.50
)
```

#### Error Notifications
```python
await bot.notify_error(
    chat_id=user_id,
    error="Connection to MT5 lost"
)
```

### Additional Features
- Setup formation alerts
- Trade management notifications
- News impact alerts
- Real-time performance updates

## Implementation

### Initialization
```python
async def initialize(self):
    """Initialize the Telegram bot with retry mechanism."""
    # Validate configuration
    # Set up command handlers
    # Start bot with retries
    # Send startup message to authorized users
```

### Error Handling
- Comprehensive error handling for all operations
- Automatic reconnection attempts
- Error logging and notification

### Message Formatting
- HTML formatting for better readability
- Emoji usage for visual feedback
- Consistent message structure

## Configuration
Required environment variables:
- `TELEGRAM_BOT_TOKEN` - Bot token from BotFather
- `TELEGRAM_ALLOWED_USERS` - Comma-separated list of authorized user IDs

## Testing
- Unit tests for all core functionality
- Integration tests for end-to-end workflows
- Mock testing for Telegram API calls

## Best Practices
1. Always validate user authorization before processing commands
2. Use HTML parsing mode for formatted messages
3. Include timestamps in notifications
4. Implement rate limiting for message sending
5. Handle connection errors gracefully
6. Keep message queue for failed deliveries