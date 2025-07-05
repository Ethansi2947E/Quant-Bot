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
        # "XAUUSD",
        "BTCUSD",
        "ETHUSD",
        "XRPUSD",
        "SOLUSD",
        "DOGUSD",
        "ADAUSD",
        "DOTUSD",
        "Boom 1000 Index",
        "Jump 50 Index",
        "Jump 75 Index",
        "Step Index",
        "Range Break 200 Index",
        # "GBPUSD",
        # "EURUSD",
        # "USDJPY",
        # "USDCAD",
        # "AUDUSD",
        # "NZDUSD",
    ],
    "fixed_lot_size": 1.0,
    "use_fixed_lot_size": False,
    "max_lot_size": 1.0,
    "max_daily_risk": 0.06,
    "spread_factor": 1.5,
    "allow_position_additions": False,
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
       # "MintAlgoStrategy",
        "SuperT",
        #"LuxAlgoPremiumStrategy"
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
def get_risk_manager_config():
    return {
        'max_risk_per_trade': 0.008,
        'max_daily_loss': 0.015,
        'min_risk_reward': 0.5,
        'max_concurrent_trades': 1000,
        'use_fixed_lot_size': TRADING_CONFIG['use_fixed_lot_size'],
        'fixed_lot_size': TRADING_CONFIG['fixed_lot_size'],
        'max_lot_size': TRADING_CONFIG['max_lot_size'],
    }

# Create a default RISK_MANAGER_CONFIG for imports
RISK_MANAGER_CONFIG = get_risk_manager_config()

# ================= Trade Exit Configuration =================
TRADE_EXIT_CONFIG = {
    'partial_tp_ratio': 0.5,
    'tp_levels': [
        {'ratio': 0.5, 'size': 0.4},
        {'ratio': 1.0, 'size': 0.3},
        {'ratio': 1.5, 'size': 0.3}
    ],
    'trailing_stop': {
        'enabled': False, # General enable/disable for trailing stops
        # --- Instrument Category Rules (processed in order, first match wins) ---
        'instrument_category_rules': [
            # Specific Symbols (highest priority)
            {'symbol_is': 'XAUUSD', 'category': 'metals_gold'},
            {'symbol_is': 'BTCUSD', 'category': 'crypto_btc'},

            # MT5 Path Based (examples, adjust to your broker's paths)
            {'path_contains': 'Forex\\\\Major', 'category': 'forex_major'},
            {'path_contains': 'Forex\\\\Minor', 'category': 'forex_minor'},
            {'path_contains': 'Forex\\\\Exotic', 'category': 'forex_exotic'},
            {'path_contains': 'Indices\\\\Volatility', 'category': 'volatility_indices'},
            {'path_contains': 'Indices\\\\CrashBoom', 'category': 'crash_boom_indices'}, # Assuming a path like "Indices\CrashBoom\Crash 500 Index"
            {'path_contains': 'Indices\\\\Jump', 'category': 'jump_indices'},
            {'path_contains': 'Indices\\\\StepRange', 'category': 'step_range_indices'}, # Assuming "Indices\StepRange\Step Index"
            {'path_contains': 'Metals', 'category': 'metals_other'}, # For other metals if XAUUSD is special
            {'path_contains': 'Crypto', 'category': 'crypto_other'}, # For other cryptos if BTCUSD is special

            # Fallback Regex/Symbol Name Contains (lower priority)
            {'symbol_contains': 'EUR', 'category': 'forex_eur_pairs'}, # Example for EUR specific
            {'symbol_contains': 'VOLATILITY', 'category': 'volatility_indices_fallback'}, # If path fails
            {'symbol_contains': 'CRASH', 'category': 'crash_boom_indices_fallback'},
            {'symbol_contains': 'BOOM', 'category': 'crash_boom_indices_fallback'},
            {'symbol_contains': 'JUMP', 'category': 'jump_indices_fallback'},
            {'symbol_contains': 'STEP', 'category': 'step_range_indices_fallback'},
            {'symbol_contains': 'RANGE BREAK', 'category': 'step_range_indices_fallback'},
        ],

        # --- Instrument Category Settings ---
        # These are the parameter sets. The 'default' is crucial.
        'instrument_category_settings': {
            'default': { # General fallback settings
                'mode': 'atr', # More robust default
                'trail_points': 20.0, # Pips, only if mode is 'pips'
                'atr_multiplier': 2.0,
                'atr_period': 14,
                'percent': 0.01, # Percentage, only if mode is 'percent'
                'break_even_enabled': True,
                'break_even_pips': 10,
                'break_even_buffer_pips': 1,
                'activation_ratio': 0.5, # When to start trailing (e.g., 0.5 = 50% of initial risk gained)
                'min_profit_activation': 0.2, # Alternative: min profit in R before activation
                'auto_sl_setup': True, # If position opened with no SL, set one automatically
                'auto_sl_percent': 0.02, # e.g. 2% of entry price
            },
            'forex_major': {
                'mode': 'pips',
                'trail_points': 15.0,
                'atr_multiplier': 1.8, # Keep for potential mode switch
                'percent': 0.005,    # Keep for potential mode switch
                'break_even_pips': 8,
                'activation_ratio': 0.6,
            },
            'forex_minor': {
                'mode': 'pips',
                'trail_points': 20.0,
                'atr_multiplier': 2.0,
                'percent': 0.007,
                'break_even_pips': 10,
            },
            'forex_exotic': {
                'mode': 'atr',
                'trail_points': 30.0,
                'atr_multiplier': 2.5,
                'percent': 0.012,
                'break_even_pips': 15,
            },
             'forex_eur_pairs': { # Example of a more specific regex/contains based category
                'mode': 'pips',
                'trail_points': 12.0, # Tighter for EUR pairs example
                'atr_multiplier': 1.5,
                'break_even_pips': 7,
            },
            'metals_gold': { # Specific for XAUUSD
                'mode': 'atr',
                'trail_points': 50.0, # Value in price points for XAUUSD
                'atr_multiplier': 2.0, # ATR multiplier
                'percent': 0.008,
                'break_even_pips': 20, # Value in price points
                'activation_ratio': 0.4,
            },
            'metals_other': {
                'mode': 'atr',
                'trail_points': 60.0,
                'atr_multiplier': 2.2,
                'percent': 0.01,
                'break_even_pips': 25,
            },
            'crypto_btc': { # Specific for BTCUSD
                'mode': 'percent',
                'trail_points': 100.0, # Basis points if mode was different, here it's just a placeholder
                'atr_multiplier': 2.5, # ATR for crypto can be large
                'percent': 0.015, # 1.5% trailing
                'break_even_pips': 50, # Price points
                'activation_ratio': 0.3,
            },
            'crypto_other': {
                'mode': 'percent',
                'trail_points': 150.0,
                'atr_multiplier': 3.0,
                'percent': 0.02, # 2% trailing
                'break_even_pips': 75,
            },
            'volatility_indices': {
                'mode': 'atr',
                'trail_points': 80.0, # Price points
                'atr_multiplier': 2.8,
                'percent': 0.012,
                'break_even_pips': 25, # Price points
            },
            'volatility_indices_fallback': { # If path fails, use symbol_contains
                'mode': 'atr',
                'trail_points': 85.0,
                'atr_multiplier': 3.0,
                'break_even_pips': 30,
            },
            'crash_boom_indices': {
                'mode': 'percent', # Often these move fast, percent might be better
                'trail_points': 150.0,
                'atr_multiplier': 3.5,
                'percent': 0.020, # 2%
                'break_even_pips': 40, # Price points
            },
            'crash_boom_indices_fallback': {
                'mode': 'percent',
                'trail_points': 160.0,
                'atr_multiplier': 3.7,
                'percent': 0.022,
                'break_even_pips': 45,
            },
            'jump_indices': {
                'mode': 'atr',
                'trail_points': 60.0,
                'atr_multiplier': 2.5,
                'percent': 0.010,
                'break_even_pips': 20,
            },
            'jump_indices_fallback': {
                'mode': 'atr',
                'trail_points': 65.0,
                'atr_multiplier': 2.6,
                'break_even_pips': 22,
            },
            'step_range_indices': {
                'mode': 'pips', # Or ATR depending on typical movement
                'trail_points': 40.0,
                'atr_multiplier': 2.0,
                'percent': 0.008,
                'break_even_pips': 15,
            },
            'step_range_indices_fallback': {
                'mode': 'pips',
                'trail_points': 45.0,
                'atr_multiplier': 2.2,
                'break_even_pips': 18,
            }
            # Add other categories as needed
        }
    }
}