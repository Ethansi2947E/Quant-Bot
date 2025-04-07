"""
Market Schedule Configuration

Contains market hours, holidays, and instrument-specific trading schedules.
This configuration is used by the MarketHours utility to determine market open/close status.
"""

# Market schedule configuration
MARKET_SCHEDULE_CONFIG = {
    # Forex standard trading week (24/5)
    'forex_week_start': {'day': 6, 'hour': 22, 'minute': 0},  # Sunday 22:00 UTC
    'forex_week_end': {'day': 4, 'hour': 22, 'minute': 0},    # Friday 22:00 UTC
    
    # Default timezone (can be overridden)
    'timezone': 'UTC',
    
    # Commonly traded forex symbols
    'forex_symbols': [
        # Major pairs
        'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'NZDUSD', 'USDCAD',
        # Minor pairs
        'EURJPY', 'GBPJPY', 'AUDJPY', 'CADJPY', 'NZDJPY',
        'EURGBP', 'EURCHF', 'EURAUD', 'EURCAD',
        # Add broker suffixes for MT5 symbols
        'EURUSDm', 'GBPUSDm', 'USDJPYm', 'USDCHFm', 'AUDUSDm', 'NZDUSDm', 'USDCADm',
        'EURJPYm', 'GBPJPYm', 'AUDJPYm', 'CADJPYm', 'NZDJPYm',
        'EURGBPm', 'EURCHFm', 'EURAUDm', 'EURCADm'
    ],
    
    # Metals symbols
    'metals_symbols': [
        'XAUUSD', 'XAGUSD', 'XPTUSD', 'XPDUSD',
        'XAUUSDm', 'XAGUSDm', 'XPTUSDm', 'XPDUSDm'
    ],
    
    # Indices symbols
    'indices_symbols': [
        'US30', 'SPX500', 'NAS100', 'UK100', 'GER30', 'JPN225',
        'US30m', 'SPX500m', 'NAS100m', 'UK100m', 'GER30m', 'JPN225m'
    ],
    
    # Cryptocurrency symbols (24/7 trading)
    'crypto_symbols': [
        # Major cryptocurrencies
        'BTCUSD', 'ETHUSD', 'LTCUSD', 'XRPUSD', 'BCHUSD', 'ADAUSD',
        'DOTUSD', 'SOLUSD', 'DOGEUSD', 'MATICUSD',
        # With broker suffixes
        'BTCUSDm', 'ETHUSDm', 'LTCUSDm', 'XRPUSDm', 'BCHUSDm', 'ADAUSDm',
        'DOTUSDm', 'SOLUSDm', 'DOGEUSDm', 'MATICUSDm',
        # Variations
        'BTCEUR', 'ETHEUR', 'BTCGBP', 'ETHGBP',
        'BTCEURm', 'ETHEURm', 'BTCGBPm', 'ETHGBPm'
    ],
    
    # Specific symbol schedules (for exceptions)
    'symbol_schedules': {
        # Example: US30 has specific trading hours
        'us30': {
            'trading_days': ['0', '1', '2', '3', '4'],  # Monday-Friday
            'hours': [
                {'start': 0, 'end': 22}  # 00:00-22:00 UTC
            ]
        },
        'us30m': {
            'trading_days': ['0', '1', '2', '3', '4'],  # Monday-Friday
            'hours': [
                {'start': 0, 'end': 22}  # 00:00-22:00 UTC
            ]
        }
    },
    
    # Holiday calendar
    'holidays': [
        # Fixed annual holidays
        {'month': 1, 'day': 1},       # New Year's Day
        {'month': 12, 'day': 25},     # Christmas
        
        # Specific holidays for 2025
        {'month': 4, 'day': 18, 'year': 2025},  # Good Friday 2025
        {'month': 5, 'day': 26, 'year': 2025},  # Memorial Day 2025
        {'month': 7, 'day': 4, 'year': 2025},   # Independence Day 2025
        {'month': 9, 'day': 1, 'year': 2025},   # Labor Day 2025
        {'month': 11, 'day': 27, 'year': 2025}, # Thanksgiving 2025
        {'month': 12, 'day': 26, 'year': 2025}  # Boxing Day 2025
    ]
} 