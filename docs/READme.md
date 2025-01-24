# Trading Bot

A sophisticated automated trading bot with Telegram integration for real-time monitoring and control.

## Prerequisites

Before getting started, ensure you have the following installed:
- Python 3.7 or higher
- pip (Python package installer)
- MetaTrader 5 (MT5) platform
- Telegram account
- Git

## Features

- **Automated Trading**: Fully automated trading system with customizable strategies
- **Telegram Integration**: Control and monitor your trades through a Telegram bot
- **Real-time Alerts**: Get instant notifications for:
  - Trade setups and entries
  - Position management updates
  - Performance metrics
  - Error alerts
  - News impacts
- **Risk Management**: Built-in risk management features including:
  - Stop-loss and take-profit management
  - Position sizing
  - Maximum drawdown protection
  - Risk per trade limits
  - Daily risk limits
- **Performance Tracking**: Comprehensive metrics including:
  - Win rate
  - Total profit/loss
  - Maximum drawdown
  - Trade history
  - Risk-adjusted returns
  - Daily/Monthly performance

## Commands

The bot responds to the following Telegram commands:

- `/start` - Initialize the bot
- `/enable` - Enable automated trading
- `/disable` - Disable trading
- `/status` - Check current bot status
- `/metrics` - View performance metrics
- `/history` - View recent trade history
- `/help` - Display available commands
- `/stop` - Stop the bot and close all trades

## Setup

1. Clone the repository:
```bash
git clone https://github.com/Ethansi2947E/Trading-bot2.git
cd Trading-bot2
```

2. Create and activate a virtual environment (recommended):
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure the bot:
- Copy `config/config.example.py` to `config/config.py`
- Add your Telegram bot token and allowed user IDs
- Configure trading parameters
- Set up MT5 credentials and server details
- Configure risk management parameters

5. Start the bot:
```bash
python main.py
```

## Detailed Configuration

### MT5 Configuration
```python
MT5_CONFIG = {
    'server': 'YOUR_BROKER_SERVER',
    'login': 'YOUR_ACCOUNT_NUMBER',
    'password': 'YOUR_PASSWORD',
    'timeout': 60000,
    'symbols': ['EURUSD', 'GBPUSD', 'USDJPY'],  # Add your trading symbols
}
```

### Trading Configuration
```python
TRADING_CONFIG = {
    'risk_per_trade': 1.0,  # Risk percentage per trade (1.0 = 1%)
    'max_daily_risk': 5.0,  # Maximum daily risk percentage
    'default_lot_size': 0.01,
    'max_spread': 5.0,  # Maximum allowed spread in pips
    'trading_hours': {
        'start': '00:00',
        'end': '23:59',
    },
    'excluded_hours': [],  # Hours to exclude from trading
}
```

### Telegram Configuration
```python
TELEGRAM_CONFIG = {
    'bot_token': 'YOUR_BOT_TOKEN',
    'allowed_user_ids': ['YOUR_USER_ID'],
    'notification_settings': {
        'trade_alerts': True,
        'performance_updates': True,
        'error_alerts': True,
        'news_alerts': True,
    }
}
```

## Directory Structure

```
Trading-bot2/
├── config/         # Configuration files
│   ├── config.py          # Main configuration
│   └── config.example.py  # Example configuration
├── docs/           # Documentation
├── logs/          # Log files
│   ├── trading_bot.log    # Main log file
│   └── error.log         # Error log file
├── src/           # Source code
│   ├── telegram_bot.py    # Telegram bot implementation
│   ├── trading_bot.py     # Trading logic implementation
│   ├── risk_manager.py    # Risk management module
│   ├── strategy.py        # Trading strategies
│   └── utils.py          # Utility functions
├── tests/         # Test files
├── main.py        # Entry point
└── requirements.txt # Python dependencies
```

## Error Handling

The bot includes comprehensive error handling:
- Connection retry mechanism with exponential backoff
- Automatic reconnection to MT5 platform
- Error logging with different severity levels
- Real-time error alerts via Telegram
- Graceful shutdown procedures
- Trade verification and validation

### Common Error Solutions
1. MT5 Connection Issues:
   - Verify MT5 platform is running
   - Check credentials in config
   - Ensure stable internet connection

2. Telegram Connection Issues:
   - Verify bot token
   - Check allowed user IDs
   - Test bot in Telegram

3. Trading Errors:
   - Check symbol availability
   - Verify trading hours
   - Monitor spread conditions
   - Check account balance

## Security

- Telegram commands are restricted to authorized users only
- API keys and sensitive data are stored in configuration files
- All actions are logged for audit purposes
- Secure storage of credentials
- IP-based access restrictions
- Regular security audits

## Monitoring and Maintenance

### Log Files
- `trading_bot.log`: Main activity log
- `error.log`: Error tracking
- Log rotation enabled by default
- Configurable log levels and formats

### Performance Monitoring
- Real-time performance metrics
- Daily summary reports
- Risk exposure monitoring
- Strategy performance analysis

### Backup and Recovery
- Regular state backups
- Configuration backups
- Recovery procedures
- Data persistence

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Add unit tests for new features
- Update documentation
- Use type hints
- Comment complex logic

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This trading bot is for educational purposes only. Use it at your own risk. The developers are not responsible for any financial losses incurred through the use of this software. 

### Risk Warning
- Trading involves substantial risk
- Past performance is not indicative of future results
- Test thoroughly in a demo account first
- Never risk more than you can afford to lose
- Understand all features before live trading 