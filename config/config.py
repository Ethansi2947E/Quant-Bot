from pathlib import Path
from dotenv import load_dotenv
import os
import logging
from typing import Dict, Any

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
    "symbols": ['AUDCAD', 'CADJPY', 'XAUUSD', 'EURCHF', 'EURJPY', 'AUDUSD'],
    "timeframes": ['M5', 'M15'],
    "risk_per_trade": 0.008,
    "max_daily_risk": 0.06,
    "min_volatility": 5.0,
    "max_spread": 20.0,
    "min_atr": 4.0,
    "max_atr": 100.0,
    "volatility_factor": 1.2,
    "spread_factor": 1.5,
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
SESSION_CONFIG: Dict[str, Dict[str, Any]] = {
    "asia_session": {
        "enabled": True,
        "start": "00:00",
        "end": "08:00",
        "pairs": [],
        "min_range_pips": 4,
        "max_range_pips": 115,
        "volatility_factor": 1
    },
    "london_session": {
        "enabled": True,
        "start": "08:00",
        "end": "16:00",
        "pairs": [],
        "min_range_pips": 5,
        "max_range_pips": 173,
        "volatility_factor": 1.2
    },
    "new_york_session": {
        "enabled": True,
        "start": "13:00",
        "end": "21:00",
        "pairs": [],
        "min_range_pips": 5,
        "max_range_pips": 173,
        "volatility_factor": 1.2
    }
}

# Market Schedule Configuration
MARKET_SCHEDULE_CONFIG = {
    "trading_hours": {
        "forex": {
            "sunday_open": "17:00",    # Sunday NY time
            "friday_close": "17:00",   # Friday NY time
            "daily_break": None        # Forex trades 24h except weekends
        },
    },
    "holidays": {
        "2024": {
            "new_years": "2024-01-01",
            "good_friday": "2024-03-29",
            "easter_monday": "2024-04-01",
            "christmas": "2024-12-25",
            "boxing_day": "2024-12-26"
        }
    },
    "partial_trading_days": {
        "2024": {
            "christmas_eve": {"date": "2024-12-24", "close_time": "13:00"},
            "new_years_eve": {"date": "2024-12-31", "close_time": "13:00"}
        }
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
    "strong": 0.8,
    "moderate": 0.6,
    "weak": 0.4,
    "minimum": 0.2,
}

# Log the signal thresholds
logging.info(f"Loaded SIGNAL_THRESHOLDS: {SIGNAL_THRESHOLDS}")

# Confirmation Requirements
CONFIRMATION_CONFIG = {
    "min_required": 2,  # Reduced from 3
    "weights": {
        "smt_divergence": 0.4,
        "liquidity_sweep": 0.4,
        "momentum": 0.2
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
    "timeframes": ["M15"],        # Only using 15-minute timeframe
    "risk_per_trade": 0.01,      # Risk per trade (1% of balance)
    "enable_visualization": True, # Enable trade visualization
    "save_results": True,        # Save backtest results to file
    "results_dir": "backtest_results"  # Directory to save results
}

# Signal Generator Configuration
SIGNAL_CONFIG = {
    'timeframe_thresholds': {
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
            'base_score': 0.15,
            'ranging_market': 0.15,
            'trending_market': 0.25,
            'volatility_filter': 1.2,
            'min_trend_strength': 0.15,
            'rr_ratio': 1.0,
            'min_confirmations': 1
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
    },
    'timeframe_weights': {
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
    },
    'signal_thresholds': {
        'strong': 0.75,
        'moderate': 0.65,
        'weak': 0.55,
        'minimum': 0.45
    },
    'timeframe_multipliers': {
        'H4': 1.25,
        'H1': 1.0,
        'M15': 0.85,
        'M5': 0.75
    },
    'base_thresholds': {
        'score': 0.15,
        'ranging_market': 0.15,
        'trending_market': 0.12
    },
    'component_weights': {
        'structure': 0.40,
        'volume': 0.20,
        'smc': 0.30,
        'mtf': 0.10
    }
}

# Risk Management Configuration
RISK_CONFIG = {
    'max_daily_trades': 4,
    'max_concurrent_trades': 2,
    'min_trades_spacing': 1,
    'max_daily_loss': 0.015,
    'max_drawdown_pause': 0.05,
    'max_weekly_trades': 16,
    'min_win_rate_continue': 0.30,
    'max_risk_per_trade': 0.01,
    'consecutive_loss_limit': 4,
    'volatility_scaling': True,
    'partial_tp_enabled': True,
    'recovery_mode': {
        'enabled': True,
        'drawdown_trigger': 0.05,
        'position_size_reduction': 0.5,
        'min_wins_to_exit': 2
    },
    'M15': {
        'max_daily_trades': 5,
        'max_concurrent_trades': 2,
        'min_trades_spacing': 1,
        'max_daily_loss': 0.015,
        'max_drawdown_pause': 0.05,
        'max_weekly_trades': 20,
        'consecutive_loss_limit': 4
    }
}

# Position Sizing Configuration
POSITION_CONFIG = {
    'volatility_scaling': {
        'high_volatility': 0.5,
        'normal_volatility': 1.0,
        'low_volatility': 0.75,
        'atr_multipliers': {
            'high': 1.5,
            'low': 0.5
        }
    }
}

# Market Condition Filters
MARKET_FILTERS = {
    'min_daily_range': 0.0008,
    'max_daily_range': 0.0150,
    'min_volume_threshold': 400,
    'max_spread_threshold': 0.0004,
    'correlation_threshold': 0.50,
    'trend_strength_min': 0.40,
    'volatility_percentile': 0.10,
    'momentum_threshold': 0.008,
    'structure_quality_min': 0.60,
    'M15': {
        'min_daily_range': 0.0006,
        'max_daily_range': 0.0140,
        'min_volume_threshold': 300,
        'max_spread_threshold': 0.0004,
        'min_confirmations': 2
    }
}

# Trade Exit Configuration
TRADE_EXIT_CONFIG = {
    'partial_tp_ratio': 0.5,
    'tp_levels': [
        {'ratio': 1.0, 'size': 0.5},
        {'ratio': 2.0, 'size': 0.5}
    ],
    'trailing_stop': {
        'enabled': True,
        'activation_ratio': 1.0,
        'trail_points': 0.5
    }
}

# Volatility Thresholds
VOLATILITY_CONFIG = {
    'default': {
        'H4': 1.5,
        'H1': 1.4,
        'M15': 1.25,
        'M5': 1.2
    }
}

# RSI Configuration
RSI_CONFIG = {
    'default': {'overbought': 70, 'oversold': 30}
} 