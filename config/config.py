from pathlib import Path
from dotenv import load_dotenv
import os
from typing import Dict, Any

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent

# MT5 Configuration
MT5_CONFIG = {
    "server": os.getenv("MT5_SERVER", "MetaQuotes-Demo"),
    "login": int(os.getenv("MT5_LOGIN", "0")),
    "password": os.getenv("MT5_PASSWORD", ""),
    "timeout": 10,
}

# Trading Configuration
TRADING_CONFIG = {
    "symbols": [
        {"symbol": "AUDCADm", "timeframe": "M1"},
        {"symbol": "BTCUSDm", "timeframe": "M1"},
        {"symbol": "AUDJPYm", "timeframe": "M1"},
        {"symbol": "CADJPYm", "timeframe": "M1"},
        {"symbol": "EURCADm", "timeframe": "M1"},
        {"symbol": "XAUUSDm", "timeframe": "M1"},
        {"symbol": "EURCHFm", "timeframe": "M1"},
        {"symbol": "EURGBPm", "timeframe": "M1"},
        {"symbol": "EURJPYm", "timeframe": "M1"},
        {"symbol": "AUDUSDm", "timeframe": "M1"},
        {"symbol": "GBPUSDm", "timeframe": "M1"},
        {"symbol": "NZDUSDm", "timeframe": "M1"},
        {"symbol": "USDCADm", "timeframe": "M1"},
        {"symbol": "USDJPYm", "timeframe": "M1"},
        {"symbol": "USDCHFm", "timeframe": "M1"},
        {"symbol": "XAGUSDm", "timeframe": "M1"},
    ],
    "timeframes": ['M1', 'H1'],  # Updated timeframes for the breakout_reversal strategy
    "fixed_lot_size": 0.01,  # Fixed lot size to use if use_fixed_lot_size is true
    "use_fixed_lot_size": True,  # When true, use fixed lot size instead of risk-based calculation
    "max_lot_size": 0.3,  # Maximum lot size even when using risk-based calculation
    "max_daily_risk": 0.06,
    "spread_factor": 1.5,

    # Position addition settings
    "allow_position_additions": True,  # Disable adding to existing positions
    "max_position_size": 2.0,         # Maximum total position size after additions
    "position_addition_threshold": 0.5,  # Minimum distance in ATR for adding positions

    # Enhanced Data Management Configuration
    "data_management": {
        "use_direct_fetch": True,      # Use direct fetching for all timeframes
        "real_time_bars_count": 10,    # Number of recent bars to fetch for validation
        "price_tolerance": 0.0003,     # 0.03% price tolerance for validation
        "validate_trades": True,       # Validate trades against real-time data before execution
        "tick_delay_tolerance": 2.0,   # Maximum allowed tick delay in seconds
    },
    
    # Shutdown behavior
    "close_positions_on_shutdown": False,  # Whether to close all open positions when shutting down
    
    "signal_generators": [ 
        "breakout_reversal"  # Your price action strategy is the primary and only strategy
    ],
}

# Telegram Configuration
TELEGRAM_CONFIG = {
    "bot_token": os.getenv("TELEGRAM_BOT_TOKEN", ""),
    "allowed_user_ids": [
        "6018798296",
        "5421178210"
    ],
}

# Logging Configuration
LOG_CONFIG = {
    "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    "level": "DEBUG",
    # No file logging settings
    "use_file_logging": False
}

# Session Configuration - ALL TIMES IN LOCAL MACHINE TIME
SESSION_CONFIG: Dict[str, Dict[str, Any]] = {
    "asia_session": {
        "enabled": True,
        "start": "00:00",  # Local machine time
        "end": "08:00",    # Local machine time
        "pairs": [],
        "min_range_pips": 4,
        "max_range_pips": 115,
        "volatility_factor": 1
    },
    "london_session": {
        "enabled": True,
        "start": "08:00",  # Local machine time
        "end": "16:00",    # Local machine time
        "pairs": [],
        "min_range_pips": 5,
        "max_range_pips": 173,
        "volatility_factor": 1.2
    },
    "new_york_session": {
        "enabled": True,
        "start": "13:00",  # Local machine time
        "end": "21:00",    # Local machine time
        "pairs": [],
        "min_range_pips": 5,
        "max_range_pips": 173,
        "volatility_factor": 1.2
    }
}
   
# Risk Management Configuration
def get_risk_config(timeframe="M15"):
    # Default values based on timeframe
    if timeframe == "M1":
        return {
            'max_daily_trades': 15,
            'max_concurrent_trades': 3,
            'min_trades_spacing': 1,
            'max_daily_loss': 0.02,
            'max_weekly_loss': 0.05,
            'max_monthly_loss': 0.10,
            'max_drawdown_pause': 0.05,
            'max_weekly_trades': 60,
            'min_win_rate_continue': 0.30,
            'max_risk_per_trade': 0.004,
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
    elif timeframe == "M5":
        return {
            'max_daily_trades': 10,
            'max_concurrent_trades': 2,
            'min_trades_spacing': 1,
            'max_daily_loss': 0.015,
            'max_weekly_loss': 0.04,
            'max_monthly_loss': 0.08,
            'max_drawdown_pause': 0.05,
            'max_weekly_trades': 40,
            'min_win_rate_continue': 0.30,
            'max_risk_per_trade': 0.006,
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
    else:  # Default M15 and higher
        return {
            'max_daily_trades': 8,
            'max_concurrent_trades': 2,
            'min_trades_spacing': 1,
            'max_daily_loss': 0.015,
            'max_weekly_loss': 0.03,
            'max_monthly_loss': 0.06,
            'max_drawdown_pause': 0.05,
            'max_weekly_trades': 32,
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
def get_market_filters(timeframe="M15"):
    # Default values based on timeframe
    if timeframe == "M1":
        max_spread = 10.0
        correlation_threshold = 0.70
        min_confirmations = 3
    elif timeframe == "M5":
        max_spread = 15.0
        correlation_threshold = 0.65
        min_confirmations = 2
    elif timeframe == "H1":
        max_spread = 40.0
        correlation_threshold = 0.45
        min_confirmations = 2
    else:  # Default M15 and other timeframes
        max_spread = 30.0
        correlation_threshold = 0.50
        min_confirmations = 2
    
    return {
        'min_daily_range': 0.0008,
        'max_daily_range': 0.0150,
        'min_volume_threshold': 400,
        'max_spread_threshold': max_spread,
        'correlation_threshold': correlation_threshold,
        'trend_strength_min': 0.40,
        'volatility_percentile': 0.10,
        'momentum_threshold': 0.008,
        'structure_quality_min': 0.60,
        'min_confirmations': min_confirmations
    }

# Trade Exit Configuration
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
        'trail_points': 0.5
    }
}

# Volatility Thresholds
def get_volatility_config(timeframe="M15"):
    # Default values based on timeframe
    if timeframe == "M1":
        return 0.9
    elif timeframe == "M5":
        return 1.0
    elif timeframe == "M15":
        return 1.2
    elif timeframe == "H1":
        return 1.5
    elif timeframe == "H4":
        return 1.8
    else:  # D1 and others
        return 2.0

# Pattern Detector Configuration
def get_pattern_detector_config(timeframe="M15"):
    # Adjust parameters based on timeframe
    min_sweep_factor = 1.0
    if timeframe == "M1":
        min_sweep_factor = 0.6
    elif timeframe == "M5":
        min_sweep_factor = 0.8
    elif timeframe == "H1":
        min_sweep_factor = 1.2
    elif timeframe == "H4":
        min_sweep_factor = 1.5
    elif timeframe == "D1":
        min_sweep_factor = 2.0
        
    min_sweep_pips = 3.0 * min_sweep_factor
    max_sweep_pips = 30.0 * min_sweep_factor
    
    return {
        # Core parameters
        'min_sweep_pips': min_sweep_pips,
        'max_sweep_pips': max_sweep_pips,
        'minimum_volume_ratio': 1.2,
        'equal_level_threshold': 0.0001,
        'ob_threshold': 0.0015,
        'fvg_threshold': 0.00015,
        'liquidity_threshold': 0.0020,
        'manipulation_threshold': 0.0015,
        
        # Pattern-specific thresholds
        'amd_pattern': {
            'min_volume_ratio': 1.2,
            'lookback_period': 20
        },
        
        'turtle_soup': {
            'min_sweep_pips': min_sweep_pips,
            'max_sweep_pips': max_sweep_pips,
            'atr_multiplier': 0.5
        },
        
        'sh_bms_rto': {
            'equal_level_threshold': 0.0001,
            'liquidity_threshold': 0.0015,
            'max_rto_bars': 12
        }
    }

# Risk Manager Configuration
def get_risk_manager_config(timeframe="M15"):
    # Default values based on timeframe
    if timeframe == "M1":
        max_risk_per_trade = 0.004
        max_daily_loss = 0.02
        max_concurrent_trades = 3
        max_daily_trades = 15
        min_trades_spacing = 1
    elif timeframe == "M5":
        max_risk_per_trade = 0.006
        max_daily_loss = 0.015
        max_concurrent_trades = 2
        max_daily_trades = 10
        min_trades_spacing = 1
    elif timeframe == "H1":
        max_risk_per_trade = 0.01
        max_daily_loss = 0.02
        max_concurrent_trades = 2
        max_daily_trades = 4
        min_trades_spacing = 2
    elif timeframe == "H4":
        max_risk_per_trade = 0.015
        max_daily_loss = 0.025
        max_concurrent_trades = 1
        max_daily_trades = 2
        min_trades_spacing = 4
    else:  # M15 is default
        max_risk_per_trade = 0.008
        max_daily_loss = 0.015
        max_concurrent_trades = 2
        max_daily_trades = 8
        min_trades_spacing = 1
    
    return {
        # Core risk parameters
        'max_risk_per_trade': max_risk_per_trade,
        'max_daily_loss': max_daily_loss,
        'max_daily_risk': 0.03,
        'max_weekly_loss': 10,
        'max_monthly_loss': 10,
        'max_drawdown_pause': 0.05,
        
        # Position management
        'max_concurrent_trades': max_concurrent_trades,
        'max_daily_trades': max_daily_trades,
        'max_weekly_trades': 8,
        'min_trades_spacing': min_trades_spacing,
        'use_fixed_lot_size': TRADING_CONFIG['use_fixed_lot_size'],
        'fixed_lot_size': TRADING_CONFIG['fixed_lot_size'],
        'max_lot_size': TRADING_CONFIG['max_lot_size'],
        
        # Drawdown controls
        'consecutive_loss_limit': 2,
        'drawdown_position_scale': {
            0.02: 0.75,   # 75% size at 2% drawdown
            0.03: 0.50,   # 50% size at 3% drawdown
            0.04: 0.25,   # 25% size at 4% drawdown
            0.05: 0.0     # Stop trading at 5% drawdown
        },
        
        # Partial profit targets
        'partial_tp_levels': [
            {'size': 0.4, 'ratio': 0.5},  # 40% at 0.5R
            {'size': 0.3, 'ratio': 1.0},  # 30% at 1R
            {'size': 0.3, 'ratio': 1.5}   # 30% at 1.5R
        ],
        
        # Volatility-based sizing
        'volatility_position_scale': {
            'extreme': 0.25,  # 25% size in extreme volatility
            'high': 0.50,     # 50% size in high volatility
            'normal': 1.0,    # Normal size
            'low': 0.75       # 75% size in low volatility
        },
        
        # Recovery mode
        'recovery_mode': {
            'enabled': True,
            'threshold': 0.05,        # 5% drawdown activates recovery
            'position_scale': 0.5,    # 50% position size
            'win_streak_required': 3,  # Need 3 winners to exit
            'max_trades_per_day': 2,   # Limited trades in recovery
            'min_win_rate': 0.40      # Min win rate to exit recovery
        },
        
        # Correlation controls
        'correlation_limits': {
            'max_correlation': 0.7,
            'lookback_period': 20,
            'min_trades_for_calc': 50,
            'high_correlation_scale': 0.5
        },
        
        # Session-based adjustments
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
RISK_MANAGER_CONFIG = get_risk_manager_config("M15")

# Timeframe Configuration
TIMEFRAME_CONFIG = {
    "supported_timeframes": ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1"],
    "update_frequencies": {
        "M1": 60,     # Update every 60 seconds
        "M5": 300,    # Update every 5 minutes
        "M15": 900,   # Update every 15 minutes
        "M30": 1800,  # Update every 30 minutes
        "H1": 3600,   # Update every hour
        "H4": 14400,  # Update every 4 hours
        "D1": 86400,  # Update every day
        "W1": 604800  # Update every week
    },
    "disabled_timeframes": [],  # Timeframes to disable globally
    "max_lookback_bars": {
        "M1": 10000,
        "M5": 5000,
        "M15": 3000,
        "M30": 2000,
        "H1": 1500,
        "H4": 1000,
        "D1": 500,
        "W1": 200
    }
}


