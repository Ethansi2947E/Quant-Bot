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
            "M1": 500,  # For M1 base timeframe
            "M5": 200,  # For resampled M5
            "M15": 100, # For resampled M15
            "H1": 50,   # For resampled H1
            "H4": 30    # For resampled H4
        }
        
        # Define timeframe hierarchy
        self.timeframe_hierarchy = {
            "M1": {"parent": None, "children": ["M5"]},
            "M5": {"parent": "M1", "children": ["M15"]},
            "M15": {"parent": "M5", "children": ["H1"]},
            "H1": {"parent": "M15", "children": ["H4"]},
            "H4": {"parent": "H1", "children": ["D1"]},
            "D1": {"parent": "H4", "children": ["W1"]},
            "W1": {"parent": "D1", "children": []}
        }
        
        # Track last candle completion time for each timeframe
        self.last_candle_time = {}
        
        # New: Real-time validation configuration
        self.use_direct_fetch = True  # Use direct fetching instead of just resampling
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
        
        logger.info("DataManager initialized with hybrid direct fetching and caching capabilities")
    
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
    
    async def perform_warmup(self, symbols: List[str]) -> bool:
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
                # Directly fetch data for each timeframe instead of just M1
                logger.info(f"Directly fetching {tf} data for {symbol}")
                tf_data = await self.mt5_handler.get_rates(
                    symbol, tf, self.warmup_bars_required.get(tf, 100)
                )
                
                if tf_data is None or len(tf_data) < self.warmup_bars_required.get(tf, 100):
                    logger.warning(f"Failed to get enough {tf} data for {symbol}. "
                                  f"Got {0 if tf_data is None else len(tf_data)} bars, "
                                  f"needed {self.warmup_bars_required.get(tf, 100)}")
                    
                    # If M1 failed, mark warmup as unsuccessful
                    if tf == "M1":
                        warmup_success = False
                    continue
                
                logger.info(f"Loaded {len(tf_data)} {tf} bars for {symbol}")
                
                # Store the data
                if symbol not in self.data_cache:
                    self.data_cache[symbol] = {}
                
                # Preprocess the data
                tf_data = self._preprocess_data(tf_data, symbol, tf)
                self.data_cache[symbol][tf] = tf_data
                self.last_update[f"{symbol}_{tf}"] = time.time()
                
                # If it's M1 data, also create resampled versions as a backup
                if tf == "M1" and self.use_direct_fetch:
                    for higher_tf in [t for t in self.required_timeframes if t != "M1"]:
                        logger.info(f"Also resampling M1 data to {higher_tf} for redundancy")
                        resampled_data = self._resample_from_m1(tf_data, higher_tf)
                        
                        # Store resampled data if we don't have direct data yet
                        if higher_tf not in self.data_cache[symbol]:
                            self.data_cache[symbol][higher_tf] = resampled_data
                            self.last_update[f"{symbol}_{higher_tf}"] = time.time()
        
        # Set warmup status
        self.warmup_complete = warmup_success
        
        if warmup_success:
            logger.info("✅ Warmup phase completed successfully")
        else:
            logger.warning("⚠️ Warmup phase completed with some issues")
            
        return warmup_success
    
    def update_data(self, symbol: str, timeframe: str, force: bool = False) -> Optional[pd.DataFrame]:
        """
        Update data for a specific symbol and timeframe, using hybrid approach.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe to update
            force: Force update regardless of last update time
            
        Returns:
            Updated DataFrame or None if update failed
        """
        now = time.time()
        key = f"{symbol}_{timeframe}"
        
        # Check if update is needed
        if not force and key in self.last_update:
            update_frequency = self.update_frequencies.get(timeframe, 60)
            if now - self.last_update[key] < update_frequency:
                return self._get_cached_data(symbol, timeframe)
        
        # Direct fetch from MT5 for this timeframe
        if self.use_direct_fetch:
            try:
                # Fetch the most recent bars directly from MT5
                recent_lookback = self.real_time_bars_count
                recent_data = self.mt5_handler.get_market_data(symbol, timeframe, recent_lookback)
                
                # Fetch historical data for the timeframe
                lookback = self.max_lookback.get(timeframe, 1000)
                historical_data = self.mt5_handler.get_market_data(symbol, timeframe, lookback)
                
                if historical_data is not None:
                    # Apply preprocessing
                    historical_data = self._preprocess_data(historical_data, symbol, timeframe)
                    
                    # Update cache
                    if symbol not in self.data_cache:
                        self.data_cache[symbol] = {}
                    
                    self.data_cache[symbol][timeframe] = historical_data
                    self.last_update[key] = now
                    
                    # Validate recent data if available
                    if recent_data is not None and len(recent_data) > 0:
                        self._validate_recent_data(symbol, timeframe, recent_data)
                        
                    logger.debug(f"Updated data for {symbol} {timeframe} using direct fetch")
                    return historical_data
                else:
                    # If direct fetch for the timeframe failed, try to resample from M1
                    logger.warning(f"Direct fetch failed for {symbol} {timeframe}, trying to resample from M1")
                    return self._try_resample_from_m1(symbol, timeframe, now)
                
            except Exception as e:
                logger.error(f"Error updating data for {symbol} {timeframe}: {str(e)}")
                return self._try_resample_from_m1(symbol, timeframe, now)
        else:
            # Original approach: M1 fetch and resampling
            if timeframe == "M1":
                # Get lookback bars for M1
                lookback = self.max_lookback.get(timeframe, 1000)
                data = self.mt5_handler.get_market_data(symbol, timeframe, lookback)
                
                if data is not None:
                    data = self._preprocess_data(data, symbol, timeframe)
                    
                    # Update cache
                    if symbol not in self.data_cache:
                        self.data_cache[symbol] = {}
                    
                    self.data_cache[symbol][timeframe] = data
                    self.last_update[key] = now
                    
                    logger.debug(f"Updated M1 data for {symbol}")
                    return data
                else:
                    logger.error(f"Failed to fetch M1 data for {symbol}")
                    return None
            else:
                # For higher timeframes, try resampling from M1
                return self._try_resample_from_m1(symbol, timeframe, now)
        
        return None
    
    def _try_resample_from_m1(self, symbol: str, timeframe: str, now: float) -> Optional[pd.DataFrame]:
        """Helper method to attempt resampling from M1 data as a fallback."""
        key = f"{symbol}_{timeframe}"
        if symbol in self.data_cache and "M1" in self.data_cache[symbol]:
            m1_data = self.data_cache[symbol]["M1"]
            # Resample and update
            resampled_data = self._resample_from_m1(m1_data, timeframe)
            
            if len(resampled_data) > 0:
                # Update cache
                if symbol not in self.data_cache:
                    self.data_cache[symbol] = {}
                
                self.data_cache[symbol][timeframe] = resampled_data
                self.last_update[key] = now
                
                logger.debug(f"Updated {timeframe} data for {symbol} using resampling")
                return resampled_data
        
        # Fallback to direct fetch if resampling fails or M1 data not available
        try:
            lookback = self.max_lookback.get(timeframe, 1000)
            data = self.mt5_handler.get_market_data(symbol, timeframe, lookback)
            
            if data is not None:
                data = self._preprocess_data(data, symbol, timeframe)
                
                # Update cache
                if symbol not in self.data_cache:
                    self.data_cache[symbol] = {}
                
                self.data_cache[symbol][timeframe] = data
                self.last_update[key] = now
                
                logger.debug(f"Updated data for {symbol} {timeframe} using direct fetch")
                return data
            
        except Exception as e:
            logger.error(f"Error in fallback update for {symbol} {timeframe}: {str(e)}")
        
        return None
    
    def _validate_recent_data(self, symbol: str, timeframe: str, recent_data: pd.DataFrame) -> None:
        """
        Validate recent data against cached data and update if needed.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe to validate
            recent_data: Recently fetched data to validate against
        """
        try:
            key = f"{symbol}_{timeframe}"
            if symbol not in self.data_cache or timeframe not in self.data_cache[symbol]:
                return
            
            cached_data = self.data_cache[symbol][timeframe]
            if cached_data is None or cached_data.empty or recent_data is None or recent_data.empty:
                return
            
            # Get the most recent timestamp in cached data
            if isinstance(cached_data.index, pd.DatetimeIndex):
                latest_cached_time = cached_data.index[-1]
            else:
                return
            
            # Preprocess recent data for comparison
            recent_data = self._preprocess_data(recent_data, symbol, timeframe)
            
            # Find any new bars in recent data
            new_bars = recent_data[recent_data.index > latest_cached_time]
            if len(new_bars) > 0:
                # Append new bars to cached data
                updated_data = pd.concat([cached_data, new_bars])
                updated_data = updated_data[~updated_data.index.duplicated(keep='last')]
                updated_data = updated_data.sort_index()
                
                # Update the cache
                self.data_cache[symbol][timeframe] = updated_data
                self.last_update[key] = time.time()
                logger.debug(f"Added {len(new_bars)} new bars to {symbol} {timeframe} cache")
            
            # Check if the last bar has been updated (e.g., price changed)
            if len(recent_data) > 0 and len(cached_data) > 0:
                recent_last_bar = recent_data.iloc[-1]
                cached_last_bar = None
                
                # Find the corresponding bar in cached data
                if recent_data.index[-1] in cached_data.index:
                    cached_last_bar = cached_data.loc[recent_data.index[-1]]
                
                if cached_last_bar is not None:
                    # Check if the bar data has changed
                    if (recent_last_bar['close'] != cached_last_bar['close'] or
                        recent_last_bar['high'] != cached_last_bar['high'] or
                        recent_last_bar['low'] != cached_last_bar['low'] or
                        recent_last_bar['tick_volume'] != cached_last_bar['tick_volume']):
                        
                        # Update the most recent bar
                        self.data_cache[symbol][timeframe].loc[recent_data.index[-1]] = recent_last_bar
                        self.last_update[key] = time.time()
                        logger.debug(f"Updated last bar data for {symbol} {timeframe}")
        
        except Exception as e:
            logger.error(f"Error validating recent data for {symbol} {timeframe}: {str(e)}")
    
    def update_resampled_timeframes(self, symbol: str) -> Dict[str, bool]:
        """
        Update all higher timeframes by direct fetching and resampling.
        
        Args:
            symbol: Symbol to update
            
        Returns:
            Dictionary of success status for each timeframe
        """
        results = {}
        
        # Update all required timeframes
        for tf in self.required_timeframes:
            if tf == "M1":
                continue  # Skip M1 as it's handled separately
            
            try:
                if self.use_direct_fetch:
                    # Direct fetch from MT5 for real-time accuracy
                    recent_lookback = self.real_time_bars_count
                    recent_data = self.mt5_handler.get_market_data(symbol, tf, recent_lookback)
                    
                    if recent_data is not None and len(recent_data) > 0:
                        # Use the direct fetched data to update/validate cache
                        self._validate_recent_data(symbol, tf, recent_data)
                        results[tf] = True
                    else:
                        # Fall back to resampling if direct fetch fails
                        results[tf] = self._resample_timeframe(symbol, tf)
                else:
                    # Original resampling approach
                    results[tf] = self._resample_timeframe(symbol, tf)
            
            except Exception as e:
                logger.error(f"Error updating {tf} for {symbol}: {str(e)}")
                results[tf] = False
        
        return results
    
    def _resample_timeframe(self, symbol: str, timeframe: str) -> bool:
        """Helper method to resample a specific timeframe from M1 data."""
        if symbol not in self.data_cache or "M1" not in self.data_cache[symbol]:
            logger.debug(f"No M1 data available for {symbol}, cannot resample")
            return False
        
        try:
            base_df = self.data_cache[symbol]["M1"]
            resampled_df = self._resample_from_m1(base_df, timeframe)
            
            # Check if we have new data
            if symbol in self.data_cache and timeframe in self.data_cache[symbol]:
                existing_df = self.data_cache[symbol][timeframe]
                if len(existing_df) > 0 and len(resampled_df) > 0:
                    if existing_df.index[-1] != resampled_df.index[-1]:
                        # New candle formed
                        key = f"{symbol}_{timeframe}"
                        self.last_candle_time[key] = resampled_df.index[-1]
                        logger.debug(f"New {timeframe} candle formed for {symbol}")
            
            # Store the resampled data
            if symbol not in self.data_cache:
                self.data_cache[symbol] = {}
            
            self.data_cache[symbol][timeframe] = resampled_df
            self.last_update[f"{symbol}_{timeframe}"] = time.time()
            return True
            
        except Exception as e:
            logger.error(f"Error resampling {timeframe} data for {symbol}: {str(e)}")
            return False
    
    def _resample_from_m1(self, m1_data: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
        """
        Resample M1 data to higher timeframe.
        
        Args:
            m1_data: M1 dataframe
            target_timeframe: Target timeframe (M5, M15, H1)
            
        Returns:
            Resampled dataframe
        """
        # Convert timeframe to pandas resample rule
        resample_map = {
            "M1": "1T",
            "M5": "5T",
            "M15": "15T",
            "M30": "30T",
            "H1": "1h",
            "H4": "4h",
            "D1": "1D",
            "W1": "1W"
        }
        rule = resample_map.get(target_timeframe, "1h")
        
        # Ensure we have a copy and the index is datetime
        df = m1_data.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            # Create a datetime index if needed
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                logger.error(f"Error converting index to datetime: {str(e)}")
                # Try to create a datetime index from the time column
                if 'time' in df.columns:
                    df = df.set_index(pd.to_datetime(df['time']))
        
        # Resample
        try:
            # Convert 'H' to 'h' in the rule to avoid deprecation warning
            if rule.startswith('H'):
                rule = 'h' + rule[1:]
                
            resampled = df.resample(rule).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'tick_volume': 'sum'
            })
            
            # Drop NaN values
            resampled = resampled.dropna()
            
            # Ensure all necessary columns exist
            required_columns = ['open', 'high', 'low', 'close', 'tick_volume']
            for col in required_columns:
                if col not in resampled.columns:
                    if col == 'tick_volume' and 'volume' in resampled.columns:
                        resampled['tick_volume'] = resampled['volume']
                    else:
                        logger.warning(f"Column {col} missing in resampled data, adding empty")
                        resampled[col] = 0
            
            return resampled
            
        except Exception as e:
            logger.error(f"Error in resampling: {str(e)}")
            return pd.DataFrame()  # Return empty DataFrame on error
    
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
        
        Args:
            symbols: List of trading symbols
            force: Force update regardless of last update time
            
        Returns:
            Status dictionary {symbol: {timeframe: success}}
        """
        result = {}
        
        for symbol in symbols:
            result[symbol] = {}
            
            if self.use_direct_fetch:
                # Update all timeframes directly
                for timeframe in self.required_timeframes:
                    data = self.update_data(symbol, timeframe, force=force)
                    result[symbol][timeframe] = data is not None
                    
                # Cross-validate with real-time data
                validation_results = {}
                for timeframe in self.required_timeframes:
                    validation_key = f"{symbol}_{timeframe}"
                    validation_time = self.last_validation.get(validation_key, 0)
                    validation_freq = self.validation_frequency.get(timeframe, 300)
                    
                    # Validate if enough time has passed or if forced
                    if force or (time.time() - validation_time >= validation_freq):
                        # Fetch the most recent bars directly from MT5
                        recent_data = self.mt5_handler.get_market_data(symbol, timeframe, self.real_time_bars_count)
                        if recent_data is not None:
                            self._validate_recent_data(symbol, timeframe, recent_data)
                            self.last_validation[validation_key] = time.time()
                            validation_results[timeframe] = True
                        else:
                            validation_results[timeframe] = False
                
                # Log validation results if any
                if validation_results:
                    logger.debug(f"Validated real-time data for {symbol}: {validation_results}")
            else:
                # Original approach: Update M1 first, then resample
                m1_result = self.update_data(symbol, "M1", force=force) is not None
                result[symbol]["M1"] = m1_result
                
                if m1_result:
                    # Use resampling for higher timeframes
                    resampling_results = self.update_resampled_timeframes(symbol)
                    for tf, success in resampling_results.items():
                        result[symbol][tf] = success
                else:
                    # Fall back to direct fetching for each timeframe
                    for timeframe in self.required_timeframes:
                        if timeframe == "M1":
                            continue  # Already tried
                        data = self.update_data(symbol, timeframe, force=force)
                        result[symbol][timeframe] = data is not None
        
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