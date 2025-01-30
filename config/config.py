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
    "symbols": ["EURUSD", "GBPUSD", "USDJPY", "AUDCAD"],  # Trading pairs
    "timeframes": ["M5", "M15", "H1", "H4"],  # Timeframes to analyze
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
        "volatility_factor": 0.7,
        "pairs": ["USDJPY", "AUDJPY", "EURJPY"],
        "min_range_pips": 4,    # Decreased from 6
        "max_range_pips": 130   # Increased from 115
    },
    "london_session": {
        "start": "08:00",  # UTC
        "end": "16:00",    # UTC
        "volatility_factor": 1.0,
        "pairs": ["EURUSD", "GBPUSD", "EURGBP", "USDJPY"],
        "min_range_pips": 5,    # Decreased from 6
        "max_range_pips": 200   # Increased from 173
    },
    "new_york_session": {
        "start": "13:00",  # UTC
        "end": "21:00",    # UTC
        "volatility_factor": 1.0,
        "pairs": ["EURUSD", "GBPUSD", "USDCAD", "USDJPY"],
        "min_range_pips": 5,    # Decreased from 6
        "max_range_pips": 200   # Increased from 173
    }
}

# Market Structure Configuration
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

# Signal Classification Thresholds
SIGNAL_THRESHOLDS = {
    "strong": 0.6,    # Decreased from 0.7
    "moderate": 0.4,  # Decreased from 0.5
    "weak": 0.25      # Decreased from 0.3
}

# Confirmation Requirements
CONFIRMATION_CONFIG = {
    "min_required": 2,  # Reduced from 3
    "weights": {
        "smt_divergence": 0.3,
        "liquidity_sweep": 0.3,
        "momentum": 0.2,
        "pattern": 0.2
    }
}

# Backtesting Configuration
BACKTEST_CONFIG = {
    "start_date": "2024-01-01",  # Start date for backtesting
    "end_date": "2024-03-14",    # End date for backtesting
    "initial_balance": 10000,     # Initial account balance for backtesting
    "commission": 0.0001,         # Commission per trade (0.01%)
    "spread": 2,                  # Spread in points
    "symbols": ["EURUSD", "GBPUSD", "USDJPY", "AUDCAD"],  # Symbols to backtest
    "timeframes": ["M15", "H1", "H4"],          # Timeframes to analyze
    "risk_per_trade": 0.01,      # Risk per trade (1% of balance)
    "enable_visualization": True, # Enable trade visualization
    "save_results": True,        # Save backtest results to file
    "results_dir": "backtest_results"  # Directory to save results
} 