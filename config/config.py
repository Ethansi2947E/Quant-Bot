from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# MT5 Configuration
MT5_CONFIG = {
    "server": os.getenv("MT5_SERVER", "MetaQuotes-Demo"),
    "login": int(os.getenv("MT5_LOGIN", "0")),
    "password": os.getenv("MT5_PASSWORD", ""),
    "timeout": 60000,
}

# Trading Configuration
TRADING_CONFIG = {
    "symbols": ["EURUSD", "GBPUSD", "USDJPY"],  # Trading pairs
    "timeframes": ["M5", "M15", "H1"],  # Timeframes to analyze
    "risk_per_trade": 0.02,  # 2% risk per trade
    "max_daily_risk": 0.06,  # 6% max daily risk
}

# Telegram Configuration
TELEGRAM_CONFIG = {
    "bot_token": os.getenv("TELEGRAM_BOT_TOKEN", ""),
    "allowed_user_ids": [
        "6018798296"  # Add the user ID here
    ],
}

# AI Configuration
AI_CONFIG = {
    "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
    "news_api_key": os.getenv("NEWS_API_KEY", ""),
    "sentiment_threshold": 0.5,  # Minimum sentiment score to consider
}

# Database Configuration
DB_CONFIG = {
    "url": f"sqlite:///{DATA_DIR}/trading_bot.db",
    "echo": False,
}

# Logging Configuration
LOG_CONFIG = {
    "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    "level": "DEBUG",
    "rotation": "1 day",
    "retention": "1 month",
    "compression": "zip",
}

# Session Configuration
SESSION_CONFIG = {
    "asian_session": {
        "start": "00:00",  # UTC
        "end": "08:00",    # UTC
        "volatility_factor": 0.7,  # Lower expected volatility
        "pairs": ["USDJPY", "AUDJPY", "EURJPY"],
        "min_range_pips": 10,
        "max_range_pips": 100
    },
    "london_session": {
        "start": "08:00",  # UTC
        "end": "16:00",    # UTC
        "volatility_factor": 1.0,  # Normal volatility
        "pairs": ["EURUSD", "GBPUSD", "EURGBP", "USDJPY"],
        "min_range_pips": 10,
        "max_range_pips": 150
    },
    "new_york_session": {
        "start": "13:00",  # UTC
        "end": "21:00",    # UTC
        "volatility_factor": 1.0,  # Normal volatility
        "pairs": ["EURUSD", "GBPUSD", "USDCAD", "USDJPY"],
        "min_range_pips": 10,
        "max_range_pips": 150
    }
}

# Market Structure Configuration
MARKET_STRUCTURE_CONFIG = {
    "swing_detection": {
        "lookback_periods": 10,
        "threshold_pips": 5,
        "min_swing_pips": 15
    },
    "structure_levels": {
        "ob_size": 10,      # Order block size in pips
        "fvg_threshold": 5,  # Fair value gap threshold in pips
        "bos_threshold": 5   # Break of structure threshold in pips
    },
    "timeframe_weights": {
        "TIMEFRAME_H4": 1.0,
        "TIMEFRAME_H1": 0.8,
        "TIMEFRAME_M15": 0.6,
        "TIMEFRAME_M5": 0.4
    }
} 