# Configuration

The trading bot system is highly configurable, allowing users to customize various aspects of the bot's behavior and functionality. The main configuration file, `config.py`, contains several sections that define different settings and parameters.

## Configuration Sections

### 1. MT5 Configuration

The `MT5_CONFIG` section defines the connection details for the MetaTrader 5 (MT5) platform:

- `server`: The name of the MT5 server to connect to.
- `login`: The login ID for the MT5 account.
- `password`: The password for the MT5 account.
- `timeout`: The timeout value in milliseconds for MT5 requests.

Example:
```python
MT5_CONFIG = {
    "server": "MetaQuotes-Demo",
    "login": 12345678,
    "password": "password",
    "timeout": 60000,
}
```

### 2. Trading Configuration

The `TRADING_CONFIG` section defines the trading-related settings:

- `symbols`: A list of trading symbols (currency pairs) to monitor and trade.
- `timeframes`: A list of timeframes to analyze for each symbol.
- `risk_per_trade`: The maximum risk per trade as a percentage of the account balance.
- `max_daily_risk`: The maximum daily risk as a percentage of the account balance.

Example:
```python
TRADING_CONFIG = {
    "symbols": ["EURUSD", "GBPUSD", "USDJPY"],
    "timeframes": ["M5", "M15", "H1"],
    "risk_per_trade": 0.02,
    "max_daily_risk": 0.06,
}
```

### 3. Telegram Configuration

The `TELEGRAM_CONFIG` section defines the settings for Telegram integration:

- `bot_token`: The token for the Telegram bot.
- `allowed_user_ids`: A list of user IDs allowed to interact with the bot.

Example:
```python
TELEGRAM_CONFIG = {
    "bot_token": "YOUR_TELEGRAM_BOT_TOKEN",
    "allowed_user_ids": [123456789, 987654321],
}
```

### 4. AI Configuration

The `AI_CONFIG` section defines the settings for AI integration:

- `openai_api_key`: The API key for OpenAI's GPT.
- `news_api_key`: The API key for the news API.
- `sentiment_threshold`: The minimum sentiment score to consider for news analysis.

Example:
```python
AI_CONFIG = {
    "openai_api_key": "YOUR_OPENAI_API_KEY",
    "news_api_key": "YOUR_NEWS_API_KEY",
    "sentiment_threshold": 0.5,
}
```

### 5. Database Configuration

The `DB_CONFIG` section defines the settings for the database:

- `url`: The URL for the SQLite database file.
- `echo`: A boolean value indicating whether to echo SQL queries for debugging purposes.

Example:
```python
DB_CONFIG = {
    "url": f"sqlite:///data/trading_bot.db",
    "echo": False,
}
```

### 6. Logging Configuration

The `LOG_CONFIG` section defines the settings for logging:

- `format`: The format string for log messages.
- `level`: The minimum log level to capture.
- `rotation`: The log file rotation interval.
- `retention`: The log file retention period.
- `compression`: The compression method for rotated log files.

Example:
```python
LOG_CONFIG = {
    "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    "level": "DEBUG",
    "rotation": "1 day",
    "retention": "1 month",
    "compression": "zip",
}
```

## Customizing the Configuration

To customize the trading bot system, users can modify the values in the `config.py` file according to their specific requirements. It's important to ensure that the modified values are valid and compatible with the bot's functionality.

Users should be cautious when modifying the configuration and thoroughly test any changes before deploying the bot in a live trading environment.

## Conclusion

The configuration file, `config.py`, provides a centralized location for users to customize various aspects of the trading bot system. By understanding the different configuration sections and their purposes, users can tailor the bot's behavior to suit their trading strategies and preferences. 