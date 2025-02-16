# Configuration Guide

## Overview
This document outlines the configuration settings for the trading bot. All settings can be found in the `config/config.py` file.

## Recent Updates (March 2024)

### Timeframe-Specific Settings

The system now uses optimized settings for each timeframe:

```python
TIMEFRAME_THRESHOLDS = {
    'M5': {
        'base_score': 0.65,
        'ranging_market': 0.45,
        'trending_market': 0.75,
        'volatility_filter': 1.2,
        'min_trend_strength': 0.60,
        'rr_ratio': 2.0,
        'min_confirmations': 2
    },
    'M15': {
        'base_score': 0.70,
        'ranging_market': 0.45,
        'trending_market': 0.80,
        'volatility_filter': 1.2,
        'min_trend_strength': 0.65,
        'rr_ratio': 2.0,
        'min_confirmations': 3
    },
    'H1': {
        'base_score': 0.65,
        'ranging_market': 0.40,
        'trending_market': 0.75,
        'volatility_filter': 1.2,
        'min_trend_strength': 0.60,
        'rr_ratio': 2.4,
        'min_confirmations': 2
    },
    'H4': {
        'base_score': 0.65,
        'ranging_market': 0.40,
        'trending_market': 0.70,
        'volatility_filter': 1.1,
        'min_trend_strength': 0.60,
        'rr_ratio': 2.8,
        'min_confirmations': 2
    }
}
```

### Component Weights

Each timeframe has specific component weights:

```python
TIMEFRAME_WEIGHTS = {
    'M5': {
        'structure': 0.35,
        'volume': 0.25,
        'smc': 0.25,
        'mtf': 0.15
    },
    'M15': {
        'structure': 0.35,
        'volume': 0.25,
        'smc': 0.25,
        'mtf': 0.15
    },
    'H1': {
        'structure': 0.30,
        'volume': 0.30,
        'smc': 0.25,
        'mtf': 0.15
    },
    'H4': {
        'structure': 0.35,
        'volume': 0.25,
        'smc': 0.25,
        'mtf': 0.15
    }
}
```

### Currency Pair Settings

Optimized settings for each currency pair:

```python
SYMBOL_MULTIPLIERS = {
    'EURUSD': {
        'multiplier': 1.00,
        'volatility_threshold': 1.5,
        'rsi_overbought': 70,
        'rsi_oversold': 30
    },
    'GBPUSD': {
        'multiplier': 0.90,
        'volatility_threshold': 1.8,
        'rsi_overbought': 78,
        'rsi_oversold': 22
    },
    'USDJPY': {
        'multiplier': 0.85,
        'volatility_threshold': 1.4,
        'rsi_overbought': 75,
        'rsi_oversold': 25
    },
    'AUDUSD': {
        'multiplier': 1.15,
        'volatility_threshold': 1.6,
        'rsi_overbought': 72,
        'rsi_oversold': 28
    }
}
```

### Backtesting Configuration

```python
BACKTEST_CONFIG = {
    "initial_balance": 10000,
    "risk_per_trade": 0.01,
    "commission": 0.00007,
    "enable_visualization": True,
    "save_results": True,
    "results_dir": "backtest_results",
    "data_cache_dir": "data_cache"
}
```

### Risk Management Settings

```python
RISK_CONFIG = {
    "max_daily_trades": 4,
    "max_concurrent_trades": 2,
    "min_trades_spacing": 1,
    "max_daily_loss": 0.015,
    "max_drawdown_pause": 0.05,
    "max_weekly_trades": 16,
    "min_win_rate_continue": 0.30,
    "max_risk_per_trade": 0.01,
    "consecutive_loss_limit": 4,
    "volatility_scaling": True,
    "partial_tp_enabled": True,
    "recovery_mode": {
        "enabled": True,
        "drawdown_trigger": 0.05,
        "position_size_reduction": 0.5,
        "min_wins_to_exit": 2
    }
}
```

### Market Structure Configuration

```python
MARKET_STRUCTURE_CONFIG = {
    "swing_detection": {
        "H4": {
            "lookback_periods": 10,
            "threshold_pips": 3,
            "min_swing_pips": 10
        },
        "H1": {
            "lookback_periods": 8,
            "threshold_pips": 2.5,
            "min_swing_pips": 8
        },
        "M15": {
            "lookback_periods": 6,
            "threshold_pips": 1.5,
            "min_swing_pips": 5
        },
        "M5": {
            "lookback_periods": 5,
            "threshold_pips": 1.0,
            "min_swing_pips": 3
        }
    },
    "structure_levels": {
        "H4": {
            "ob_size": 5.0,
            "fvg_threshold": 2.5,
            "bos_threshold": 2.5
        },
        "H1": {
            "ob_size": 4.0,
            "fvg_threshold": 2.0,
            "bos_threshold": 2.0
        },
        "M15": {
            "ob_size": 2.5,
            "fvg_threshold": 1.5,
            "bos_threshold": 1.5
        },
        "M5": {
            "ob_size": 1.5,
            "fvg_threshold": 1.0,
            "bos_threshold": 1.0
        }
    },
    "timeframe_weights": {
        "H4": 1.0,
        "H1": 0.8,
        "M15": 0.7,
        "M5": 0.5
    }
}
```

### Session Configuration

```python
SESSION_CONFIG = {
    "london": {
        "start": "08:00",
        "end": "16:00",
        "pairs": ["EURUSD", "GBPUSD", "USDJPY"],
        "min_range_pips": 6,
        "max_range_pips": 173,
        "volatility_factor": 1.2
    },
    "new_york": {
        "start": "13:00",
        "end": "21:00",
        "pairs": ["EURUSD", "GBPUSD", "AUDUSD"],
        "min_range_pips": 6,
        "max_range_pips": 173,
        "volatility_factor": 1.2
    },
    "asia": {
        "start": "00:00",
        "end": "08:00",
        "pairs": ["USDJPY", "AUDUSD"],
        "min_range_pips": 6,
        "max_range_pips": 115,
        "volatility_factor": 1.0
    }
}
```

### Signal Thresholds

```python
SIGNAL_THRESHOLDS = {
    "strong": 0.65,
    "moderate": 0.55,
    "weak": 0.45,
    "minimum": 0.35
}
```

### Confirmation Requirements

```python
CONFIRMATION_CONFIG = {
    "min_required": 2,
    "weights": {
        "smt_divergence": 0.3,
        "liquidity_sweep": 0.3,
        "momentum": 0.2,
        "pattern": 0.2
    }
}
```

### Market Condition Filters

```python
MARKET_CONDITION_FILTERS = {
    "min_daily_range": 0.0012,
    "max_daily_range": 0.0140,
    "min_volume_threshold": 600,
    "max_spread_threshold": 0.0004,
    "correlation_threshold": 0.80,
    "trend_strength_min": 0.45,
    "volatility_percentile": 0.15,
    "momentum_threshold": 0.012,
    "M15": {
        "min_daily_range": 0.0010,
        "max_daily_range": 0.0120,
        "min_volume_threshold": 400,
        "max_spread_threshold": 0.0004,
        "min_confirmations": 2
    }
}
```

### Trade Exit Configuration

```python
TRADE_EXITS = {
    "partial_tp_ratio": 0.5,
    "tp_levels": [
        {"ratio": 1.0, "size": 0.5},
        {"ratio": 2.0, "size": 0.5}
    ],
    "trailing_stop": {
        "enabled": True,
        "activation_ratio": 1.0,
        "trail_points": 0.5
    }
}
```

## Environment Variables

Create a `.env` file in the root directory with the following settings:

```env
MT5_LOGIN=your_login
MT5_PASSWORD=your_password
MT5_SERVER=your_server
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

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