# config.py -- Centralized configuration for Trading Bot
"""
This file contains all configuration for the trading bot system, including:
- MT5 connection
- Trading and risk management
- Telegram integration
- Logging
- Trade exit logic

Unused/legacy configs have been removed for clarity.
"""

from pathlib import Path
from dotenv import load_dotenv
import os

# --- Load environment variables ---
load_dotenv()

# --- Base paths ---
BASE_DIR = Path(__file__).parent.parent

# ================= MT5 Configuration =================
MT5_CONFIG = {
    "server": os.getenv("MT5_SERVER", "MetaQuotes-Demo"),
    "login": int(os.getenv("MT5_LOGIN", "0")),
    "password": os.getenv("MT5_PASSWORD", ""),
    "timeout": 10,
}

# ================= Trading Configuration =================
TRADING_CONFIG = {
    "symbols": [
        "Volatility 10 Index",
        "Crash 500 Index",
        "Crash 1000 Index",
        "Boom 300 Index",
        "XAUUSD",
        "BTCUSD",
        "Boom 1000 Index",
        "Jump 50 Index",
        "Step Index",
        "Range Break 200 Index",
    ],
    "fixed_lot_size": 1.0,
    "use_fixed_lot_size": True,
    "max_lot_size": 1.0,
    "max_daily_risk": 0.06,
    "spread_factor": 1.5,
    "allow_position_additions": False,
    "max_position_size": 2.0,
    "position_addition_threshold": 0.5,

    # --- Enhanced Data Management ---

    "data_management": {
        "use_direct_fetch": True,
        "real_time_bars_count": 10,
        "price_tolerance": 0.0003,
        "validate_trades": True,
        "tick_delay_tolerance": 2.0,
    },

    "close_positions_on_shutdown": False,
    "signal_generators": [
        "confluence_price_action",
        "breakout_reversal",
        "price_action_sr"
    ],
}

# ================= Telegram Configuration =================
TELEGRAM_CONFIG = {
    "token": os.getenv("TELEGRAM_BOT_TOKEN"),
    "allowed_users": [int(id) for id in os.getenv("TELEGRAM_ALLOWED_USERS", "").split(",") if id.strip()],
    "enabled": True
}

# ================= Logging Configuration =================
LOG_CONFIG = {
    "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    "level": "INFO",
    "use_file_logging": False
}

# ================= Risk Management Configuration =================
def get_risk_config():
    return {
        'max_daily_trades': 15,
        'max_concurrent_trades': 1000,
        'min_trades_spacing': 1,
        'max_daily_loss': 0.015,
        'max_weekly_loss': 0.04,
        'max_monthly_loss': 0.08,
        'max_drawdown_pause': 0.05,
        'max_weekly_trades': 40,
        'min_win_rate_continue': 0.30,
        'max_risk_per_trade': 0.008,
        'consecutive_loss_limit': 4,
        'volatility_scaling': True,
        'partial_tp_enabled': True,
        'recovery_mode': {
            'enabled': True,
            'drawdown_trigger': 0.05,
            'position_size_reduction': 0.5,
            'min_wins_to_exit': 2
        }
    }

# ================= Trade Exit Configuration =================
TRADE_EXIT_CONFIG = {
    'partial_tp_ratio': 0.5,
    'tp_levels': [
        {'ratio': 0.5, 'size': 0.4},
        {'ratio': 1.0, 'size': 0.3},
        {'ratio': 1.5, 'size': 0.3}
    ],
    'trailing_stop': {
        'enabled': True,
        'activation_ratio': 0.5,
        'trail_points': 10.0,
        'trailing_activation_factor': 0.5,
        'min_profit_activation': 0.2,
        'buffer_pips': 2,
        'auto_sl_setup': True,
        'auto_sl_percent': 0.02,
        'break_even_enabled': True,
        'break_even_pips': 5,
        'break_even_buffer_pips': 0.5,
    }
}

# ================= Risk Manager Configuration =================
def get_risk_manager_config():
    return {
        'max_risk_per_trade': 0.008,
        'max_daily_loss': 0.015,
        'max_daily_risk': 0.03,
        'max_weekly_loss': 10,
        'max_monthly_loss': 10,
        'max_drawdown_pause': 0.05,
        'min_risk_reward': 1.0,
        'max_concurrent_trades': 1000,
        'max_daily_trades': 8,
        'max_weekly_trades': 8,
        'min_trades_spacing': 1,
        'use_fixed_lot_size': TRADING_CONFIG['use_fixed_lot_size'],
        'fixed_lot_size': TRADING_CONFIG['fixed_lot_size'],
        'max_lot_size': TRADING_CONFIG['max_lot_size'],
        'consecutive_loss_limit': 2,
        'drawdown_position_scale': {
            0.02: 0.75,
            0.03: 0.50,
            0.04: 0.25,
            0.05: 0.0
        },
        'partial_tp_levels': [
            {'size': 0.4, 'ratio': 0.5},
            {'size': 0.3, 'ratio': 1.0},
            {'size': 0.3, 'ratio': 1.5}
        ],
        'volatility_position_scale': {
            'extreme': 0.25,
            'high': 0.50,
            'normal': 1.0,
            'low': 0.75
        },
        'recovery_mode': {
            'enabled': True,
            'threshold': 0.05,
            'position_scale': 0.5,
            'win_streak_required': 3,
            'max_trades_per_day': 2,
            'min_win_rate': 0.40
        },
        'correlation_limits': {
            'max_correlation': 0.7,
            'lookback_period': 20,
            'min_trades_for_calc': 50,
            'high_correlation_scale': 0.5
        },
        'session_risk_multipliers': {
            'london_open': 1.0,
            'london_ny_overlap': 1.0,
            'ny_open': 1.0,
            'asian': 0.5,
            'pre_news': 0.0,
            'post_news': 0.5
        }
    }

# Create a default RISK_MANAGER_CONFIG for imports
RISK_MANAGER_CONFIG = get_risk_manager_config()