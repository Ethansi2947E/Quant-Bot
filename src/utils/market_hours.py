"""
Market Hours Utility

A utility for checking if markets are open based on:
1. Forex market hours (24/5)
2. Symbol-specific trading hours
3. Special holidays and market closures
4. Cryptocurrency markets (24/7)
"""

import datetime
import pytz
from datetime import time, timedelta
from typing import Dict, List, Optional, Set, Tuple
from loguru import logger

class MarketHours:
    """
    Utility class to check if markets are open for various symbols.
    
    Key features:
    - Handles standard forex trading hours (24/5)
    - Supports symbol-specific trading hours
    - Takes into account holidays and special closures
    - Uses local machine time with timezone conversion
    - Supports cryptocurrency 24/7 trading hours
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize market hours checker with configuration.
        
        Args:
            config: Optional configuration dictionary with market hours settings
        """
        self.config = config or {}
        
        # Forex standard trading week (24/5)
        # Sunday 22:00 GMT to Friday 22:00 GMT
        self.forex_week_start = self.config.get('forex_week_start', {'day': 6, 'hour': 22, 'minute': 0})  # Sunday
        self.forex_week_end = self.config.get('forex_week_end', {'day': 4, 'hour': 22, 'minute': 0})      # Friday
        
        # Default timezone
        self.default_timezone = pytz.timezone(self.config.get('timezone', 'UTC'))
        
        # Symbol-specific overrides (for instruments with specific trading hours)
        self.symbol_schedules = self.config.get('symbol_schedules', {})
        
        # Holidays and special closures
        self.holidays = self.config.get('holidays', [])
        
        # Common symbol types
        self.forex_symbols = set([s.lower() for s in self.config.get('forex_symbols', [
            'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'NZDUSD', 'USDCAD',
            'EURJPY', 'GBPJPY', 'AUDJPY', 'CADJPY', 'NZDJPY',
            'EURGBP', 'EURCHF', 'EURAUD', 'EURCAD'
        ])])
        
        self.metals_symbols = set([s.lower() for s in self.config.get('metals_symbols', [
            'XAUUSD', 'XAGUSD', 'XPTUSD', 'XPDUSD'
        ])])
        
        self.indices_symbols = set([s.lower() for s in self.config.get('indices_symbols', [
            'US30', 'SPX500', 'NAS100', 'UK100', 'GER30', 'JPN225'
        ])])
        
        # Add cryptocurrency symbols
        self.crypto_symbols = set([s.lower() for s in self.config.get('crypto_symbols', [
            'BTCUSD', 'ETHUSD', 'LTCUSD', 'XRPUSD', 'BCHUSD', 'ADAUSD',
            'DOTUSD', 'SOLUSD', 'DOGEUSD', 'MATICUSD'
        ])])
        
        logger.info("Market hours checker initialized")
    
    def is_market_open(self, symbol: str, current_time: Optional[datetime.datetime] = None) -> bool:
        """
        Check if the market is open for a specific symbol.
        
        Args:
            symbol: Symbol to check
            current_time: Optional specific time to check (defaults to current time)
            
        Returns:
            True if market is open, False otherwise
        """
        symbol = symbol.lower().replace('m', '')  # Normalize symbol name (remove 'm' suffix)
        
        # Use current time if not provided
        if current_time is None:
            current_time = datetime.datetime.now(self.default_timezone)
        elif current_time.tzinfo is None:
            # Add timezone to naive datetime
            current_time = self.default_timezone.localize(current_time)
        
        # Check if today is a holiday
        if self._is_holiday(current_time):
            logger.debug(f"Market is closed for {symbol} - holiday")
            return False
            
        # Check symbol-specific schedule
        if symbol in self.symbol_schedules:
            return self._check_symbol_schedule(symbol, current_time)
        
        # Use symbol type to determine trading hours
        if self._is_crypto_symbol(symbol):
            # Cryptocurrencies trade 24/7
            return self._is_crypto_open(current_time)
        elif self._is_forex_symbol(symbol):
            return self._is_forex_open(current_time)
        elif self._is_metals_symbol(symbol):
            return self._is_metals_open(current_time)
        elif self._is_indices_symbol(symbol):
            return self._is_indices_open(symbol, current_time)
        else:
            # Check if it could be a crypto symbol before defaulting to forex
            if 'btc' in symbol or 'eth' in symbol or 'usd' in symbol:
                logger.debug(f"Symbol {symbol} appears to be crypto, using 24/7 trading hours")
                return self._is_crypto_open(current_time)
            
            # Default to forex hours for other unknown symbols
            logger.debug(f"Using default forex hours for unknown symbol: {symbol}")
            return self._is_forex_open(current_time)
    
    def _is_forex_symbol(self, symbol: str) -> bool:
        """Check if symbol is a forex pair."""
        return symbol.lower() in self.forex_symbols
    
    def _is_metals_symbol(self, symbol: str) -> bool:
        """Check if symbol is a metal."""
        return symbol.lower() in self.metals_symbols
    
    def _is_indices_symbol(self, symbol: str) -> bool:
        """Check if symbol is an index."""
        return symbol.lower() in self.indices_symbols
    
    def _is_crypto_symbol(self, symbol: str) -> bool:
        """Check if symbol is a cryptocurrency."""
        symbol = symbol.lower()
        return symbol in self.crypto_symbols or 'btc' in symbol or 'eth' in symbol
    
    def _is_crypto_open(self, current_time: datetime.datetime) -> bool:
        """
        Check if cryptocurrency markets are open - they operate 24/7.
        
        Args:
            current_time: Current datetime
            
        Returns:
            Always returns True as crypto markets are always open
        """
        # Crypto markets are always open (24/7)
        return True
    
    def _is_forex_open(self, current_time: datetime.datetime) -> bool:
        """
        Check if forex markets are open.
        
        Forex markets typically operate 24/5:
        - Open: Sunday 22:00 GMT
        - Close: Friday 22:00 GMT
        """
        # Convert to UTC for simplified forex market hour checks
        utc_time = current_time.astimezone(pytz.UTC)
        
        # Get weekday (0=Monday, 6=Sunday)
        weekday = utc_time.weekday()
        
        # Check if within forex trading week
        start_day = self.forex_week_start['day']
        start_hour = self.forex_week_start['hour']
        start_minute = self.forex_week_start['minute']
        
        end_day = self.forex_week_end['day']
        end_hour = self.forex_week_end['hour']
        end_minute = self.forex_week_end['minute']
        
        # Check if before end time on Friday
        if weekday < end_day or (weekday == end_day and 
                                (utc_time.hour < end_hour or 
                                 (utc_time.hour == end_hour and utc_time.minute < end_minute))):
            # Check if after start time on Sunday
            if weekday > start_day or (weekday == start_day and 
                                      (utc_time.hour > start_hour or 
                                       (utc_time.hour == start_hour and utc_time.minute >= start_minute))):
                return True
        
        logger.debug(f"Forex market closed at {utc_time} UTC (weekday: {weekday})")
        return False
    
    def _is_metals_open(self, current_time: datetime.datetime) -> bool:
        """
        Check if metals markets are open.
        
        Metals markets typically follow forex hours with slight variations:
        - Open: Sunday 22:00 GMT
        - Close: Friday 22:00 GMT
        """
        # Metals follow forex hours
        return self._is_forex_open(current_time)
    
    def _is_indices_open(self, symbol: str, current_time: datetime.datetime) -> bool:
        """
        Check if indices markets are open.
        
        Indices markets have specific trading hours based on their local exchanges.
        """
        # Simplified check - most indices follow their local exchange hours
        # For a production system, you would implement specific hours for each index
        
        # Default to forex hours for now
        return self._is_forex_open(current_time)
    
    def _check_symbol_schedule(self, symbol: str, current_time: datetime.datetime) -> bool:
        """
        Check if a symbol is open based on its specific schedule.
        
        Args:
            symbol: Symbol to check
            current_time: Time to check
            
        Returns:
            True if market is open, False otherwise
        """
        schedule = self.symbol_schedules.get(symbol, {})
        if not schedule:
            return self._is_forex_open(current_time)
        
        # Convert to UTC for simplified checks
        utc_time = current_time.astimezone(pytz.UTC)
        weekday = utc_time.weekday()
        
        # Check if today is a trading day
        if str(weekday) not in schedule.get('trading_days', ['0', '1', '2', '3', '4']):
            return False
        
        # Check if current time is within trading hours
        hours = schedule.get('hours', [])
        for hour_range in hours:
            start_hour = hour_range.get('start', 0)
            end_hour = hour_range.get('end', 24)
            
            if start_hour <= utc_time.hour < end_hour:
                return True
                
        return False
    
    def _is_holiday(self, current_time: datetime.datetime) -> bool:
        """
        Check if the current time is during a holiday.
        
        Args:
            current_time: Time to check
            
        Returns:
            True if it's a holiday, False otherwise
        """
        # Extract date components
        month = current_time.month
        day = current_time.day
        year = current_time.year
        
        # Check fixed holidays
        for holiday in self.holidays:
            if isinstance(holiday, dict):
                # Format: {'month': 12, 'day': 25}  # Christmas
                if holiday.get('month') == month and holiday.get('day') == day:
                    return True
                
                # Format: {'month': 12, 'day': 25, 'year': 2025}  # Christmas 2025
                if (holiday.get('month') == month and 
                    holiday.get('day') == day and 
                    holiday.get('year', year) == year):
                    return True
        
        return False
    
    def get_next_market_open(self, symbol: str, from_time: Optional[datetime.datetime] = None) -> datetime.datetime:
        """
        Get the next time the market will open for a symbol.
        
        Args:
            symbol: Symbol to check
            from_time: Optional time to start checking from (defaults to current time)
            
        Returns:
            Datetime of next market open
        """
        # Start from current time if not provided
        if from_time is None:
            from_time = datetime.datetime.now(self.default_timezone)
        elif from_time.tzinfo is None:
            # Add timezone to naive datetime
            from_time = self.default_timezone.localize(from_time)
        
        # If market is already open, return the current time
        if self.is_market_open(symbol, from_time):
            return from_time
        
        # Check in 1-hour increments for the next 7 days
        check_time = from_time
        for _ in range(7 * 24):  # Check for a week
            check_time += timedelta(hours=1)
            if self.is_market_open(symbol, check_time):
                return check_time
        
        # If no open found, return next Sunday at 22:00 UTC
        next_sunday = from_time + timedelta(days=(6 - from_time.weekday() + 7) % 7)
        return next_sunday.replace(hour=22, minute=0, second=0, microsecond=0)
    
    def get_next_market_close(self, symbol: str, from_time: Optional[datetime.datetime] = None) -> datetime.datetime:
        """
        Get the next time the market will close for a symbol.
        
        Args:
            symbol: Symbol to check
            from_time: Optional time to start checking from (defaults to current time)
            
        Returns:
            Datetime of next market close
        """
        # Start from current time if not provided
        if from_time is None:
            from_time = datetime.datetime.now(self.default_timezone)
        elif from_time.tzinfo is None:
            # Add timezone to naive datetime
            from_time = self.default_timezone.localize(from_time)
        
        # If market is already closed, return the next open time
        if not self.is_market_open(symbol, from_time):
            return from_time
        
        # For forex, default close is Friday 22:00 UTC
        if self._is_forex_symbol(symbol) or self._is_metals_symbol(symbol):
            utc_time = from_time.astimezone(pytz.UTC)
            weekday = utc_time.weekday()
            
            # If before Friday 22:00
            if weekday < 4 or (weekday == 4 and utc_time.hour < 22):
                next_close = utc_time.replace(hour=22, minute=0, second=0, microsecond=0)
                if weekday < 4:
                    days_to_add = 4 - weekday
                    next_close += timedelta(days=days_to_add)
                return next_close
        
        # Check in 1-hour increments for the next 2 days
        check_time = from_time
        for _ in range(2 * 24):  # Check for 2 days
            check_time += timedelta(hours=1)
            if not self.is_market_open(symbol, check_time):
                return check_time
        
        # If no close found, return next Friday at 22:00 UTC
        next_friday = from_time + timedelta(days=(4 - from_time.weekday() + 7) % 7)
        return next_friday.replace(hour=22, minute=0, second=0, microsecond=0) 