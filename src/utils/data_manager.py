"""
Data Manager Component for efficient market data handling.

This module provides a centralized data management system that:
1. Caches market data for different symbols and timeframes
2. Updates data based on configurable frequencies
3. Provides preprocessed data to signal generators
4. Supports resampling from M1 data to higher timeframes
5. Implements a warmup phase for proper initialization
"""

# pyright: reportAttributeAccessIssue=false
# pyright: reportReturnType=false

import time
from typing import Dict, Set, Optional, List, Tuple, Union, Any
import pandas as pd
import numpy as np
from loguru import logger
import asyncio
from datetime import datetime
from src.mt5_handler import MT5Handler
import copy

class DataManager:
    """
    Manages market data for the trading system.
    
    This class is responsible for:
    - Efficiently caching market data
    - Updating data based on configurable timeframes
    - Coordinating data distribution to signal generators
    - Preprocessing raw data from MT5
    - Managing direct multi-timeframe data fetching
    - Validating cached data against real-time MT5 ticks
    """
    
    def __init__(self, mt5_handler: MT5Handler, config: Dict):
        """
        Initialize the DataManager.
        
        Args:
            mt5_handler: MT5Handler instance for fetching data
            config: Configuration dictionary containing timeframe settings
        """
        self.mt5_handler = mt5_handler
        self.config = config
        
        # Data cache structure: {symbol: {timeframe: pandas_dataframe}}
        self.data_cache = {}
        
        # Track required timeframes
        self.required_timeframes = set()
        
        # Track last update time for each symbol-timeframe combination
        # Format: {symbol_timeframe: timestamp}
        self.last_update = {}
        
        # Initialize update frequencies from config
        timeframe_config = self.config.get("timeframe_config", {})
        self.update_frequencies = timeframe_config.get("update_frequencies", {
            "M1": 60,     # Update every 60 seconds
            "M5": 300,    # Update every 5 minutes
            "M15": 900,   # Update every 15 minutes
            "M30": 1800,  # Update every 30 minutes
            "H1": 3600,   # Update every hour
            "H4": 14400,  # Update every 4 hours
            "D1": 86400,  # Update every day
            "W1": 604800  # Update every week
        })
        
        # Initialize max lookback periods from config
        self.max_lookback = timeframe_config.get("max_lookback_bars", {
            "M1": 10000,
            "M5": 5000,
            "M15": 3000,
            "M30": 2000,
            "H1": 1500,
            "H4": 1000,
            "D1": 500,
            "W1": 200
        })
        
        # Warmup configuration
        self.warmup_complete = False
        self.warmup_bars_required = {
            "M1": 500,
            "M5": 200,
            "M15": 100,
            "H1": 50,
            "H4": 30
        }
        
        # Always use direct fetch, NEVER use resampling
        self.use_direct_fetch = True
        self.real_time_bars_count = 10  # Number of bars to fetch directly from MT5 for real-time validation
        self.validation_frequency = {
            "M1": 60,    # Validate M1 data every 60 seconds
            "M5": 300,   # Validate M5 data every 5 minutes
            "M15": 900,  # Validate M15 data every 15 minutes
            "M30": 1800, # Validate M30 data every 30 minutes
            "H1": 3600,  # Validate H1 data every hour
            "H4": 3600,  # Validate H4 data every hour
        }
        self.last_validation = {}  # Track when each symbol-timeframe combo was last validated
        
        # --- NEW: requirements registry ---
        # Format: { (symbol, timeframe): lookback }
        self.requirements = {}
        
        logger.info("DataManager initialized with direct MT5 fetching")
    
    def register_timeframes(self, timeframes: List[str]) -> None:
        """
        Register timeframes required by signal generators.
        
        Args:
            timeframes: List of timeframes to register
        """
        for tf in timeframes:
            self.required_timeframes.add(tf)
        
        # Always include M1 as it's our base timeframe for resampling
        self.required_timeframes.add("M1")
        
        logger.debug(f"Registered timeframes: {self.required_timeframes}")
    
    def get_registered_timeframes(self) -> Set[str]:
        """
        Get all registered timeframes.
        
        Returns:
            Set of registered timeframes
        """
        return self.required_timeframes
    
    def perform_warmup(self, symbols: List[str]) -> bool:
        """
        Perform initial data loading warmup phase.
        
        Args:
            symbols: List of symbols to load data for
            
        Returns:
            True if warmup completed successfully
        """
        logger.info("Starting warmup phase to collect sufficient historical data...")
        warmup_success = True
        
        for symbol in symbols:
            logger.info(f"Loading warmup data for {symbol}")
            
            for tf in self.required_timeframes:
                # Only direct fetch data for each timeframe - no resampling
                logger.info(f"Directly fetching {tf} data for {symbol}")
                tf_data = self.mt5_handler.get_market_data(
                    symbol, tf, self.warmup_bars_required.get(tf, 100)
                )
                
                if tf_data is None or len(tf_data) < self.warmup_bars_required.get(tf, 100):
                    logger.warning(f"Failed to get enough {tf} data for {symbol}. "
                                  f"Got {0 if tf_data is None else len(tf_data)} bars, "
                                  f"needed {self.warmup_bars_required.get(tf, 100)}")
                    continue
                
                logger.info(f"Loaded {len(tf_data)} {tf} bars for {symbol}")
                
                # Store the data
                if symbol not in self.data_cache:
                    self.data_cache[symbol] = {}
                
                # Preprocess the data
                tf_data = self._preprocess_data(tf_data, symbol, tf)
                self.data_cache[symbol][tf] = tf_data
                self.last_update[f"{symbol}_{tf}"] = time.time()
                
                # NO RESAMPLING - removed resampling code completely
        
        # Set warmup status
        self.warmup_complete = True
        logger.info("✅ Warmup phase completed successfully")
            
        return warmup_success
    
    def update_data(self, symbol: str, timeframe: str, force: bool = False) -> Optional[pd.DataFrame]:
        """
        Update data for a specific symbol and timeframe, using direct MT5 fetching only.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe to update
            force: Force update regardless of last update time
            
        Returns:
            Updated DataFrame or None if update failed
        """
        logger.debug(f"[DataManager] update_data: symbol={symbol}, timeframe={timeframe}, force={force}")
        now = time.time()
        key = f"{symbol}_{timeframe}"
        
        # Check if update is needed
        if not force and key in self.last_update:
            update_frequency = self.update_frequencies.get(timeframe, 60)
            if now - self.last_update[key] < update_frequency:
                return self._get_cached_data(symbol, timeframe)
        
        try:
            # ONLY use direct fetch from MT5 - never resample
            lookback = self.max_lookback.get(timeframe, 1000)
            data = self.mt5_handler.get_market_data(symbol, timeframe, lookback)
            
            if data is not None:
                # Apply preprocessing
                data = self._preprocess_data(data, symbol, timeframe)
                
                # Update cache
                if symbol not in self.data_cache:
                    self.data_cache[symbol] = {}
                
                self.data_cache[symbol][timeframe] = data
                self.last_update[key] = now
                
                logger.debug(f"Updated data for {symbol} {timeframe} via direct fetch")
                return data
            else:
                logger.error(f"Failed to fetch {timeframe} data for {symbol}")
                return None
        
        except Exception as e:
            logger.error(f"Error updating data for {symbol} {timeframe}: {str(e)}")
            return None
    
    def update_resampled_timeframes(self, symbol: str) -> Dict[str, bool]:
        """
        This function is disabled - we only use direct fetching now.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Empty dictionary (function disabled)
        """
        logger.debug(f"Resampling disabled, using direct fetch only for all timeframes")
        return {}
    
    def _resample_timeframe(self, symbol: str, timeframe: str) -> bool:
        """
        This function is disabled - we only use direct fetching now.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe to resample to
            
        Returns:
            Always False (function disabled)
        """
        logger.debug(f"Resampling disabled for {symbol}/{timeframe}, using direct fetch only")
        return False
    
    def _resample_from_m1(self, m1_data: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
        """
        This function is disabled - we only use direct fetching now.
        
        Args:
            m1_data: DataFrame containing M1 data
            target_timeframe: Target timeframe (e.g., 'M5', 'H1')
            
        Returns:
            Empty DataFrame (function disabled)
        """
        logger.debug(f"Resampling disabled for {target_timeframe}, using direct fetch only")
        return pd.DataFrame()
    
    def get_data(self, symbol: str, timeframe: str, force_update: bool = False) -> Any:  # pyright: ignore
        """
        Get data for a specific symbol and timeframe.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe to get data for
            force_update: Force update regardless of last update time
            
        Returns:
            DataFrame containing market data or None if not available
        """
        logger.debug(f"[DataManager] get_data: symbol={symbol}, timeframe={timeframe}, force_update={force_update}")
        if force_update:
            return self.update_data(symbol, timeframe, force=True)
        
        data = self._get_cached_data(symbol, timeframe)
        
        if data is None or data.empty:
            return self.update_data(symbol, timeframe)
        
        return data
    
    def get_data_batch(self, symbols: List[str], timeframes: List[str]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Get data for multiple symbols and timeframes efficiently.
        
        Args:
            symbols: List of trading symbols
            timeframes: List of timeframes
            
        Returns:
            Nested dictionary with format {symbol: {timeframe: dataframe}}
        """
        result = {}
        
        for symbol in symbols:
            result[symbol] = {}
            for timeframe in timeframes:
                data = self.get_data(symbol, timeframe)
                if data is not None:
                    result[symbol][timeframe] = data
        
        return result
    
    def get_batch_for_registered_timeframes(self, symbols: List[str]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Get data for all registered timeframes for the provided symbols.
        
        Args:
            symbols: List of trading symbols
            
        Returns:
            Nested dictionary with format {symbol: {timeframe: dataframe}}
        """
        return self.get_data_batch(symbols, list(self.required_timeframes))
    
    def update_all(self, symbols: List[str], force: bool = False) -> Dict[str, Dict[str, bool]]:
        """
        Update all registered timeframes for the provided symbols.
        Using direct fetch only, no resampling.
        
        Args:
            symbols: List of trading symbols
            force: Force update regardless of last update time
            
        Returns:
            Status dictionary {symbol: {timeframe: success}}
        """
        result = {}
        
        for symbol in symbols:
            result[symbol] = {}
            
            # Update all timeframes via direct fetch
            for timeframe in self.required_timeframes:
                    data = self.update_data(symbol, timeframe, force=force)
                    result[symbol][timeframe] = data is not None
                
                # Log validation results if any
            logger.debug(f"Updated data for {symbol}: {result[symbol]}")
        
        return result
    
    def update_required_timeframes(self, symbols: List[str]) -> None:
        """
        Update only timeframes that need updating based on frequency settings.
        Uses direct fetching for all timeframes with a hybrid validation approach.
        
        Args:
            symbols: List of trading symbols
        """
        now = time.time()
        
        for symbol in symbols:
            for timeframe in self.required_timeframes:
                key = f"{symbol}_{timeframe}"
                last_update = self.last_update.get(key, 0)
                update_frequency = self.update_frequencies.get(timeframe, 60)
                
                if now - last_update >= update_frequency:
                    if self.use_direct_fetch:
                        # Direct fetch for each timeframe that needs updating
                        self.update_data(symbol, timeframe)
                    else:
                        # Original approach: Update M1 first, then resample
                        if timeframe == "M1":
                            self.update_data(symbol, "M1")
                            # Use resampling for higher timeframes
                            self.update_resampled_timeframes(symbol)
                            break  # M1 update will handle resampling all higher timeframes
                        else:
                            # Only update higher timeframes directly if M1 resampling wasn't just done
                            m1_key = f"{symbol}_M1"
                            m1_last_update = self.last_update.get(m1_key, 0)
                            if now - m1_last_update >= update_frequency:
                                self.update_data(symbol, timeframe, force=True)
    
    def _preprocess_data(self, data: Any, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Preprocess raw data from MT5.
        
        Args:
            data: Raw DataFrame or ndarray from MT5
            symbol: Symbol associated with the data
            timeframe: Timeframe associated with the data
            
        Returns:
            Preprocessed DataFrame
        """
        if data is None or (hasattr(data, '__len__') and len(data) == 0):
            logger.warning(f"No data to preprocess for {symbol} {timeframe}")
            return pd.DataFrame()
        
        # Convert data to DataFrame if it's not already
        if not isinstance(data, pd.DataFrame):
            try:
                # Convert numpy array or other data structure to DataFrame
                df = pd.DataFrame(data)  # pyright: ignore
            except Exception as e:
                logger.error(f"Error converting data to DataFrame for {symbol} {timeframe}: {e}")
                return pd.DataFrame()
        else:
            # If already a DataFrame, make a copy
            df = data.copy()
        
        # Sort the DataFrame if it has a sortable index
        if hasattr(df, 'sort_index'):  # pyright: ignore
            df = df.sort_index()  # pyright: ignore
        
        # Ensure essential columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                logger.warning(f"Column {col} missing in {symbol} {timeframe} data, adding zeros")
                df[col] = 0.0
        
        # Fix any data quality issues
        for col in ['open', 'high', 'low', 'close']:
            # Replace NaN values with previous value
            df[col] = df[col].ffill()
            
            # If still NaN (at the beginning), replace with next value
            df[col] = df[col].bfill()
            
            # If still NaN after both ffill and bfill, replace with zeros
            df[col] = df[col].fillna(0.0)
        
        # Fill volume with zeros for NaN values
        df['volume'] = df['volume'].fillna(0.0)
        
        # Ensure high is the highest price
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        
        # Ensure low is the lowest price
        df['low'] = df[['open', 'low', 'close']].min(axis=1)
        
        return df
    
    def _get_cached_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Get data from cache.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            
        Returns:
            DataFrame with market data or None if unavailable
        """
        if symbol in self.data_cache and timeframe in self.data_cache[symbol]:
            return self.data_cache[symbol][timeframe]
        
        return None
    
    def is_new_candle(self, symbol: str, timeframe: str, current_time: datetime) -> bool:
        """
        Check if a new candle has formed for the given symbol and timeframe.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            current_time: Current time to check against
            
        Returns:
            True if a new candle has formed
        """
        key = f"{symbol}_{timeframe}"
        last_time = self.last_candle_time.get(key)
        
        if last_time is None:
            # Initialize with current time
            self.last_candle_time[key] = current_time
            return False
        
        if timeframe.startswith('M'):
            minutes = int(timeframe[1:])
            new_candle_time = current_time.replace(
                second=0, microsecond=0,
                minute=(current_time.minute // minutes) * minutes
            )
            
            # Check if we crossed into a new candle period
            if new_candle_time != last_time:
                self.last_candle_time[key] = new_candle_time
                return True
            
        elif timeframe.startswith('H'):
            hours = int(timeframe[1:])
            new_candle_time = current_time.replace(
                second=0, microsecond=0, minute=0,
                hour=(current_time.hour // hours) * hours
            )
            
            # Check if we crossed into a new candle period
            if new_candle_time != last_time:
                self.last_candle_time[key] = new_candle_time
                return True
        
        return False
    
    def timeframe_to_seconds(self, timeframe: str) -> int:
        """
        Convert timeframe string to seconds.
        
        Args:
            timeframe: Timeframe string (e.g., 'M1', 'H1')
            
        Returns:
            Number of seconds in the timeframe
        """
        if timeframe.startswith('M'):
            return int(timeframe[1:]) * 60
        elif timeframe.startswith('H'):
            return int(timeframe[1:]) * 3600
        elif timeframe.startswith('D'):
            return int(timeframe[1:]) * 86400
        elif timeframe.startswith('W'):
            return int(timeframe[1:]) * 604800
        return 60  # Default to 1 minute

    def register_timeframe(self, symbol: str, timeframe: str, lookback: int):
        """
        Register a (symbol, timeframe, lookback) requirement from a strategy.
        Ensures the cache is preloaded with the required lookback.
        """
        key = (symbol, timeframe)
        prev_lookback = self.requirements.get(key, 0)
        if lookback > prev_lookback:
            self.requirements[key] = lookback
        else:
            # Already registered with sufficient lookback
            lookback = prev_lookback
        # Preload cache if needed
        if symbol not in self.data_cache:
            self.data_cache[symbol] = {}
        df = self.data_cache[symbol].get(timeframe)
        needs_fetch = df is None or len(df) < lookback
        if needs_fetch:
            logger.info(f"Preloading {lookback} bars for {symbol}/{timeframe} from MT5 for strategy registration")
            df = self.mt5_handler.get_market_data(symbol, timeframe, lookback)
            if df is not None:
                df = self._preprocess_data(df, symbol, timeframe)
                self.data_cache[symbol][timeframe] = df
                self.last_update[f"{symbol}_{timeframe}"] = time.time()
            else:
                logger.warning(f"Failed to preload data for {symbol}/{timeframe}")

    # --- NEW: Append-only cache update and last candle tracking ---
    async def update_on_tick(self, symbol: str):
        """
        For each registered timeframe for the symbol, check if a new candle has formed.
        If so, fetch and append the new bar to the cache.
        """
        if not hasattr(self, 'last_candle_time'):
            self.last_candle_time = {}  # {(symbol, timeframe): last_datetime}
        if not hasattr(self, '_locks'):
            self._locks = {}  # {symbol: asyncio.Lock}
        if symbol not in self._locks:
            self._locks[symbol] = asyncio.Lock()
        async with self._locks[symbol]:
            for (sym, timeframe), lookback in self.requirements.items():
                if sym != symbol:
                    continue
                # Get the last candle time in cache
                df = self.data_cache.get(symbol, {}).get(timeframe)
                last_time = None
                if df is not None and not df.empty:
                    last_time = df.index[-1]
                self.last_candle_time[(symbol, timeframe)] = last_time
                # Fetch the latest bar from MT5 (just 2 bars for safety)
                latest_df = self.mt5_handler.get_market_data(symbol, timeframe, 2)
                if latest_df is None or latest_df.empty:
                    logger.warning(f"No data returned for {symbol}/{timeframe} on tick update")
                    continue
                latest_df = self._preprocess_data(latest_df, symbol, timeframe)
                latest_bar_time = latest_df.index[-1]
                # If new bar, append it
                if last_time is None or latest_bar_time > last_time:
                    logger.info(f"New candle detected for {symbol}/{timeframe} at {latest_bar_time}")
                    # Append only the new bar(s)
                    new_bars = latest_df[latest_df.index > last_time] if last_time is not None else latest_df
                    if df is not None and not df.empty:
                        updated_df = pd.concat([df, new_bars])
                        # Drop duplicates just in case
                        updated_df = updated_df[~updated_df.index.duplicated(keep='last')]
                    else:
                        updated_df = new_bars
                    # Trim to max lookback
                    max_lookback = self.requirements[(symbol, timeframe)]
                    if len(updated_df) > max_lookback:
                        updated_df = updated_df.iloc[-max_lookback:]
                    self.data_cache[symbol][timeframe] = updated_df
                    self.last_candle_time[(symbol, timeframe)] = updated_df.index[-1]
                    self.last_update[f"{symbol}_{timeframe}"] = time.time()
                else:
                    logger.debug(f"No new candle for {symbol}/{timeframe} (last: {last_time}, latest: {latest_bar_time})")

    # --- NEW: Data retrieval API (deepcopy) ---
    def get_data_window(self, symbol: str, timeframe: str, lookback: int) -> pd.DataFrame:
        """
        Return a deepcopy of the latest N bars from the cache for (symbol, timeframe).
        If not enough data, return as much as available.
        """
        df = self.data_cache.get(symbol, {}).get(timeframe)
        if df is None or df.empty:
            logger.warning(f"No cached data for {symbol}/{timeframe} in get_data_window")
            return pd.DataFrame()
        # Get the latest N bars
        window = df.iloc[-lookback:] if lookback > 0 else df.copy()
        return copy.deepcopy(window) 