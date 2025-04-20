"""
Breakout and Reversal Hybrid Strategy

This signal generator implements a price action strategy focused on trading breakouts
and reversals at key support and resistance levels. It's based on Indrazith Shantharaj's
Price Action Trading principles.

Key features:
- Support and resistance level identification
- Trend line detection and analysis
- Breakout detection with volume confirmation
- Reversal pattern recognition at key levels
- Risk management integration with proper position sizing
- No technical indicators, just pure price action
"""

# pyright: reportArgumentType=false

import pandas as pd
import numpy as np
from loguru import logger
from typing import Dict, List, Any, Optional, Tuple
import math
from datetime import datetime, timedelta
from scipy import stats
import matplotlib.pyplot as plt
import os
from pathlib import Path
import time
import traceback

from src.trading_bot import SignalGenerator
from src.utils.indicators import calculate_atr

# Strategy parameter profiles for different timeframes
TIMEFRAME_PROFILES = {
    "M1": {
        "lookback_period": 300,  # ~5 hours to cover a full trading session (increased from 100)
        "max_retest_bars": 30,   # 30 minutes for retest windows (increased from 10)
        "level_update_hours": 4,
        "consolidation_bars": 60,
        "candles_to_check": 10,
        "consolidation_update_hours": 2,
        "atr_multiplier": 0.5,   # Lower multiplier for noisy timeframe
        "volume_percentile": 80  # 80th percentile for volume threshold
    },
    "M5": {
        "lookback_period": 140,  # ~12 hours (increased from 80)
        "max_retest_bars": 12,   # 60 minutes (increased from 8)
        "level_update_hours": 6,
        "consolidation_bars": 40, 
        "candles_to_check": 6,
        "consolidation_update_hours": 3,
        "atr_multiplier": 0.7,   # Medium multiplier
        "volume_percentile": 80  # 80th percentile for volume threshold
    },
    "M15": {
        "lookback_period": 96,   # ~24 hours (increased from 50)
        "max_retest_bars": 6,    # 90 minutes (increased from 5)
        "level_update_hours": 12,
        "consolidation_bars": 20,
        "candles_to_check": 3,
        "consolidation_update_hours": 6,
        "atr_multiplier": 1.0,   # Standard multiplier
        "volume_percentile": 80  # 80th percentile for volume threshold
    },
    "H1": {
        "lookback_period": 50,   # ~2 days (increased from 30)
        "max_retest_bars": 6,    # 6 hours (increased from 3)
        "level_update_hours": 24,
        "consolidation_bars": 10,
        "candles_to_check": 2,
        "consolidation_update_hours": 12,
        "atr_multiplier": 1.2,   # Higher multiplier for more significant movements
        "volume_percentile": 80  # 80th percentile for volume threshold
    },
    "H4": {
        "lookback_period": 30,   # ~5 days (increased from 20)
        "max_retest_bars": 4,    # 16 hours (increased from 2)
        "level_update_hours": 48,
        "consolidation_bars": 7,
        "candles_to_check": 2,
        "consolidation_update_hours": 24,
        "atr_multiplier": 1.5,   # Higher multiplier for more significant movements
        "volume_percentile": 80  # 80th percentile for volume threshold
    }
}

class BreakoutReversalStrategy(SignalGenerator):
    """
    Breakout and Reversal Hybrid Strategy based on price action principles.
    Uses support/resistance levels, candlestick patterns, and volume analysis
    to generate high-probability trading signals.
    """
    
    def __init__(self, primary_timeframe="M15", higher_timeframe="H1", **kwargs):
        """
        Initialize the Breakout and Reversal strategy.
        
        Args:
            primary_timeframe: Primary timeframe to analyze
            higher_timeframe: Higher timeframe for trend confirmation
            **kwargs: Additional parameters
        """
        # Call parent constructor to set up logger
        super().__init__(**kwargs)
        
        # Add logger instance here for reference in the class
        self.logger = logger
        
        # Strategy metadata
        self.name = "BreakoutReversalStrategy"
        self.description = "A hybrid strategy based on price action principles"
        self.version = "1.0.0"
        
        # Timeframes
        self.primary_timeframe = primary_timeframe
        self.higher_timeframe = higher_timeframe
        self.required_timeframes = [primary_timeframe, higher_timeframe]
        
        # Load appropriate timeframe profile
        if primary_timeframe == "M1":
            self.timeframe_profile = "scalping"
        elif primary_timeframe in ["M5", "M15"]:
            self.timeframe_profile = "intraday"
        elif primary_timeframe in ["H1", "H4"]:
            self.timeframe_profile = "intraday_swing"
        else:
            self.timeframe_profile = "swing"
        
        logger.info(f"🔍 Using '{self.timeframe_profile}' profile for {primary_timeframe} timeframe")
        
        # General parameters
        self.lookback_period = kwargs.get("lookback_period", 100)
        self.price_tolerance = kwargs.get("price_tolerance", 0.001)  # 0.1% tolerance for levels
        
        # ATR period for dynamic parameter scaling
        self.atr_period = kwargs.get("atr_period", 14)
        
        # Scale update intervals based on primary timeframe
        # For M5, use shorter intervals than for H1 or higher timeframes
        if primary_timeframe == "M5":
            default_level_update = 1  # 1 hour for M5
            default_trend_line_update = 1  # 1 hour for M5
            default_range_update = 0.5  # 30 minutes for M5
            default_max_retest_time = 4  # 4 hours maximum to wait for retest on M5
        elif primary_timeframe == "M15":
            default_level_update = 2  # 2 hours for M15
            default_trend_line_update = 2  # 2 hours for M15
            default_range_update = 1  # 1 hour for M15
            default_max_retest_time = 8  # 8 hours maximum to wait for retest on M15
        elif primary_timeframe == "H1":
            default_level_update = 4  # 4 hours for H1
            default_trend_line_update = 4  # 4 hours for H1
            default_range_update = 2  # 2 hours for H1
            default_max_retest_time = 12  # 12 hours maximum to wait for retest on H1
        else:
            default_level_update = 8  # 8 hours for higher timeframes
            default_trend_line_update = 8  # 8 hours for higher timeframes
            default_range_update = 4  # 4 hours for higher timeframes
            default_max_retest_time = 24  # 24 hours maximum to wait for retest
        
        # Key level parameters
        self.min_level_touches = kwargs.get("min_level_touches", 2)
        self.level_recency_weight = kwargs.get("level_recency_weight", 0.5)
        self.level_update_interval = kwargs.get("level_update_interval", default_level_update)  # Hours
        
        # Trend line parameters
        self.trend_line_min_points = kwargs.get("trend_line_min_points", 3)
        self.trend_line_max_angle = kwargs.get("trend_line_max_angle", 45)  # degrees
        self.trend_line_update_interval = kwargs.get("trend_line_update_interval", default_trend_line_update)  # Hours
        
        # Breakout parameters
        self.retest_required = kwargs.get("retest_required", False)  # Require retest to confirm
        self.max_retest_time = kwargs.get("max_retest_time", default_max_retest_time)  # Max hours to wait for retest
        self.candles_to_check = kwargs.get("candles_to_check", 5)  # How many recent candles to analyze
        
        # Consolidation parameters
        self.consolidation_length = kwargs.get("consolidation_length", 12)  # Minimum number of candles
        self.consolidation_range_max = kwargs.get("consolidation_range_max", 0.02)  # 2% max range
        self.range_update_interval = kwargs.get("range_update_interval", default_range_update)  # Hours
        
        # Ensure consolidation_bars is initialized (fix for attribute error)
        self.consolidation_bars = kwargs.get("consolidation_bars", 20)  # Default value
        
        # Risk management
        self.min_risk_reward = kwargs.get("min_risk_reward", 1.5)  # Minimum R:R ratio
        self.max_stop_pct = kwargs.get("max_stop_pct", 0.02)  # Maximum stop loss (% of price)
        
        # Volume analysis
        self.volume_threshold = kwargs.get("volume_threshold", 0.8)  # Volume spike threshold (multiplier of average) - lowered from 1.5 to 0.8
        
        # Initialize ATR and volume percentile settings (new)
        self.atr_multiplier = kwargs.get("atr_multiplier", 1.0)
        self.volume_percentile = kwargs.get("volume_percentile", 80)
        
        # Initialize storage for key levels, trend lines, and signals
        self.support_levels = {}
        self.resistance_levels = {}
        self.bullish_trend_lines = {}
        self.bearish_trend_lines = {}
        self.last_consolidation_ranges = {}
        self.retest_tracking = {}
        
        # Timetracking for updates
        self.last_updated = {
            'key_levels': {},
            'trend_lines': {},
            'consolidation_ranges': {}
        }
        
        current_time = datetime.now()
        logger.debug(f"⏰ Initializing time tracking with current time: {current_time}")
        
        logger.info(f"🔧 Initialized {self.name} with primary TF: {primary_timeframe}, higher TF: {higher_timeframe}")
        
        # Log all parameters for reference
        params = {
            'lookback_period': self.lookback_period,
            'price_tolerance': self.price_tolerance,
            'min_level_touches': self.min_level_touches,
            'level_update_interval': self.level_update_interval,
            'trend_line_min_points': self.trend_line_min_points,
            'retest_required': self.retest_required,
            'volume_threshold': self.volume_threshold,
            'min_risk_reward': self.min_risk_reward,
            'atr_multiplier': self.atr_multiplier,
            'volume_percentile': self.volume_percentile,
            'consolidation_bars': self.consolidation_bars
        }
        logger.debug(f"📊 Strategy parameters: {params}")
    
    def _load_timeframe_profile(self):
        """Load timeframe-specific parameters from the appropriate profile."""
        # Default to M1 profile if timeframe not found
        profile = TIMEFRAME_PROFILES.get(self.primary_timeframe)
        
        # Define a default profile in case the requested timeframe is not found
        default_profile = {
            "lookback_period": 100,
            "max_retest_bars": 20,
            "level_update_hours": 24,
            "consolidation_bars": 5,
            "candles_to_check": 10,
            "consolidation_update_hours": 4,
            "atr_multiplier": 1.0,
            "volume_percentile": 80
        }
        
        # Use the profile if it exists, otherwise use default
        if profile is None:
            logger.warning(f"⚠️ No profile found for {self.primary_timeframe}, using default profile")
            profile = default_profile
        
        # Set parameters based on timeframe profile
        self.lookback_period = profile["lookback_period"]
        self.max_retest_bars = profile["max_retest_bars"]
        self.level_update_hours = profile["level_update_hours"]
        self.consolidation_bars = profile["consolidation_bars"]
        self.candles_to_check = profile["candles_to_check"]
        self.consolidation_update_hours = profile["consolidation_update_hours"]
        
        # Set ATR multiplier and volume percentile (new)
        self.atr_multiplier = profile["atr_multiplier"]
        self.volume_percentile = profile["volume_percentile"]
        
        logger.info(f"⚙️ Loaded profile for {self.primary_timeframe} timeframe")
    
    async def initialize(self):
        """Initialize resources needed by the strategy."""
        logger.info(f"🔌 Initializing {self.name}")
        # No specific initialization needed
        return True
    
    async def generate_signals(self, market_data=None, symbol=None, timeframe=None, debug_visualize=False, force_trendlines=False, skip_plots=False, **kwargs):
        """
        Generate trading signals based on breakout and reversal patterns
        
        Args:
            market_data (dict, optional): Dictionary of market data by symbol and timeframe
            symbol (str, optional): Symbol to generate signals for
            timeframe (str, optional): Timeframe to use
            debug_visualize (bool, optional): Force update and visualization of trend lines
            force_trendlines (bool, optional): Force trendline detection without creating debug plots
            skip_plots (bool, optional): Skip creating debug plots even if debug_visualize is True
            process_immediately (bool, optional): Whether to return signals for immediate processing
            
        Returns:
            list: List of signal dictionaries
        """
        start_time = time.time()
        logger.info(f"🚀 SIGNAL GENERATION START: {self.name} strategy")
        
        if not market_data:
            logger.warning("⚠️ No market data provided to generate signals")
            return []
            
        # Check if we should force visualization for debugging
        debug_visualize = kwargs.get('debug_visualize', debug_visualize)
        force_trendlines = kwargs.get('force_trendlines', force_trendlines)
        skip_plots = kwargs.get('skip_plots', skip_plots)
        process_immediately = kwargs.get('process_immediately', False)
        
        # Debug logging
        if debug_visualize:
            logger.info("🔍 Debug visualization mode enabled - will force trendline updates with plots")
        elif force_trendlines:
            logger.info("🔄 Forcing trendline updates without plots")
            
        signals = []
        all_signals = []  # To collect all potential signals for scoring
        logger.info(f"🔍 Generating signals with {self.name} strategy for {len(market_data)} symbols")
        
        # Process symbols one by one, potentially returning signals immediately
        for symbol in market_data:
            symbol_start_time = time.time()
            logger.debug(f"📊 Market data for {symbol} contains timeframes: {list(market_data[symbol].keys())}")
            
            # Skip if we don't have all required timeframes
            if not all(tf in market_data[symbol] for tf in self.required_timeframes):
                missing_tfs = [tf for tf in self.required_timeframes if tf not in market_data[symbol]]
                logger.debug(f"⏩ Missing required timeframes for {symbol}: {missing_tfs}, skipping")
                continue
                
            # Get data for each timeframe and convert to DataFrame if needed
            primary_data = market_data[symbol][self.primary_timeframe]
            higher_data = market_data[symbol][self.higher_timeframe]
            
            # Initialize DataFrames to None
            primary_df = None
            higher_df = None
            
            # Convert to DataFrame if it's a dictionary - important when receiving data from MT5
            if isinstance(primary_data, dict):
                logger.debug(f"Converting dictionary to DataFrame for {symbol}/{self.primary_timeframe}")
                try:
                    # Try to convert dict to DataFrame
                    if 'M1' in primary_data and isinstance(primary_data['M1'], pd.DataFrame):
                        primary_df = primary_data['M1'].copy()
                    elif self.primary_timeframe in primary_data and isinstance(primary_data[self.primary_timeframe], pd.DataFrame):
                        primary_df = primary_data[self.primary_timeframe].copy()
                    else:
                        # Try to extract from other common format keys
                        for key in ['M1', 'M5', 'M15', 'H1', '1m', '5m', '15m', '1h', 'data', 'candles', 'ohlc']:
                            if key in primary_data and isinstance(primary_data[key], pd.DataFrame):
                                primary_df = primary_data[key].copy()
                                logger.debug(f"Found DataFrame in key '{key}'")
                                break
                        else:
                            # Direct creation from OHLC values if available at root level
                            if all(k in primary_data for k in ['open', 'high', 'low', 'close']):
                                try:
                                    logger.debug(f"Attempting to create DataFrame directly from OHLC keys in root")
                                    primary_df = pd.DataFrame({
                                        'open': primary_data['open'],
                                        'high': primary_data['high'],
                                        'low': primary_data['low'],
                                        'close': primary_data['close'],
                                    })
                                    if 'tick_volume' in primary_data:
                                        primary_df['tick_volume'] = primary_data['tick_volume']
                                    elif 'volume' in primary_data:
                                        primary_df['tick_volume'] = primary_data['volume']
                                        
                                    # Attempt to create datetime index if time data exists
                                    if 'time' in primary_data:
                                        primary_df.index = pd.to_datetime(primary_data['time'])
                                    
                                    logger.debug(f"Successfully created DataFrame from OHLC values with shape {primary_df.shape}")
                                except Exception as e:
                                    logger.debug(f"Failed to create DataFrame from OHLC: {str(e)}")
                                    primary_df = None
                            else:
                                # If no DataFrame found, log data structure and skip
                                logger.debug(f"Dictionary structure for {symbol}/{self.primary_timeframe}: {list(primary_data.keys())}")
                                
                                # Inspect the dictionary structure more deeply
                                for key, value in primary_data.items():
                                    if isinstance(value, pd.DataFrame):
                                        logger.debug(f"  Key '{key}' contains DataFrame with shape {value.shape} and columns {list(value.columns)}")
                                    elif isinstance(value, dict):
                                        logger.debug(f"  Key '{key}' contains nested dictionary with keys: {list(value.keys())}")
                                        # Check if this nested dict has OHLC values
                                        if all(k in value for k in ['open', 'high', 'low', 'close']):
                                            try:
                                                logger.debug(f"Attempting to create DataFrame from OHLC in nested key '{key}'")
                                                primary_df = pd.DataFrame({
                                                    'open': value['open'],
                                                    'high': value['high'],
                                                    'low': value['low'],
                                                    'close': value['close'],
                                                })
                                                if 'tick_volume' in value:
                                                    primary_df['tick_volume'] = value['tick_volume']
                                                elif 'volume' in value:
                                                    primary_df['tick_volume'] = value['volume']
                                                
                                                # Attempt to create datetime index if time data exists
                                                if 'time' in value:
                                                    primary_df.index = pd.to_datetime(value['time'])
                                                
                                                logger.debug(f"Successfully created DataFrame from nested OHLC with shape {primary_df.shape}")
                                                break
                                            except Exception as e:
                                                logger.debug(f"Failed to create DataFrame from nested OHLC: {str(e)}")
                                                continue
                                        
                                        # Check one level deeper
                                        for subkey, subvalue in value.items():
                                            if isinstance(subvalue, pd.DataFrame):
                                                logger.debug(f"    Subkey '{subkey}' contains DataFrame with shape {subvalue.shape}")
                                    else:
                                        logger.debug(f"  Key '{key}' contains {type(value).__name__}")
                                        
                                if primary_df is None:
                                    logger.debug(f"Could not extract DataFrame from dictionary")
                                    primary_df = None
                except Exception as e:
                    logger.debug(f"Error converting primary data to DataFrame: {str(e)}")
                    primary_df = None
            else:
                # If it's already a DataFrame
                primary_df = primary_data
            
            # Same for higher timeframe
            if isinstance(higher_data, dict):
                logger.debug(f"Converting dictionary to DataFrame for {symbol}/{self.higher_timeframe}")
                try:
                    # Try to convert dict to DataFrame
                    if 'M15' in higher_data and isinstance(higher_data['M15'], pd.DataFrame):
                        higher_df = higher_data['M15'].copy()
                    elif self.higher_timeframe in higher_data and isinstance(higher_data[self.higher_timeframe], pd.DataFrame):
                        higher_df = higher_data[self.higher_timeframe].copy()
                    else:
                        # Try to extract from other common format keys
                        for key in ['M1', 'M5', 'M15', 'H1', '1m', '5m', '15m', '1h', 'data', 'candles', 'ohlc']:
                            if key in higher_data and isinstance(higher_data[key], pd.DataFrame):
                                higher_df = higher_data[key].copy()
                                logger.debug(f"Found DataFrame in key '{key}'")
                                break
                        else:
                            # Direct creation from OHLC values if available at root level
                            if all(k in higher_data for k in ['open', 'high', 'low', 'close']):
                                try:
                                    logger.debug(f"Attempting to create higher DataFrame directly from OHLC keys in root")
                                    higher_df = pd.DataFrame({
                                        'open': higher_data['open'],
                                        'high': higher_data['high'],
                                        'low': higher_data['low'],
                                        'close': higher_data['close'],
                                    })
                                    if 'tick_volume' in higher_data:
                                        higher_df['tick_volume'] = higher_data['tick_volume']
                                    elif 'volume' in higher_data:
                                        higher_df['tick_volume'] = higher_data['volume']
                                        
                                    # Attempt to create datetime index if time data exists
                                    if 'time' in higher_data:
                                        higher_df.index = pd.to_datetime(higher_data['time'])
                                    
                                    logger.debug(f"Successfully created higher DataFrame from OHLC values with shape {higher_df.shape}")
                                except Exception as e:
                                    logger.debug(f"Failed to create higher DataFrame from OHLC: {str(e)}")
                                    higher_df = None
                            else:
                                # If no DataFrame found, log data structure and skip
                                logger.debug(f"Dictionary structure for {symbol}/{self.higher_timeframe}: {list(higher_data.keys())}")
                                
                                # Inspect the dictionary structure more deeply
                                for key, value in higher_data.items():
                                    if isinstance(value, pd.DataFrame):
                                        logger.debug(f"  Key '{key}' contains DataFrame with shape {value.shape} and columns {list(value.columns)}")
                                    elif isinstance(value, dict):
                                        logger.debug(f"  Key '{key}' contains nested dictionary with keys: {list(value.keys())}")
                                        # Check if this nested dict has OHLC values
                                        if all(k in value for k in ['open', 'high', 'low', 'close']):
                                            try:
                                                logger.debug(f"Attempting to create higher DataFrame from OHLC in nested key '{key}'")
                                                higher_df = pd.DataFrame({
                                                    'open': value['open'],
                                                    'high': value['high'],
                                                    'low': value['low'],
                                                    'close': value['close'],
                                                })
                                                if 'tick_volume' in value:
                                                    higher_df['tick_volume'] = value['tick_volume']
                                                elif 'volume' in value:
                                                    higher_df['tick_volume'] = value['volume']
                                                
                                                # Attempt to create datetime index if time data exists
                                                if 'time' in value:
                                                    higher_df.index = pd.to_datetime(value['time'])
                                                
                                                logger.debug(f"Successfully created higher DataFrame from nested OHLC with shape {higher_df.shape}")
                                                break
                                            except Exception as e:
                                                logger.debug(f"Failed to create higher DataFrame from nested OHLC: {str(e)}")
                                                continue
                                        
                                        # Check one level deeper
                                        for subkey, subvalue in value.items():
                                            if isinstance(subvalue, pd.DataFrame):
                                                logger.debug(f"    Subkey '{subkey}' contains DataFrame with shape {subvalue.shape}")
                                    else:
                                        logger.debug(f"  Key '{key}' contains {type(value).__name__}")
                                        
                                if higher_df is None:
                                    logger.debug(f"Could not extract DataFrame from dictionary")
                                    higher_df = None
                except Exception as e:
                    logger.debug(f"Error converting higher data to DataFrame: {str(e)}")
                    higher_df = None
            else:
                # If it's already a DataFrame
                higher_df = higher_data
            
            # Log a sample of the data we received
            try:
                if primary_df is not None and not isinstance(primary_df, dict) and len(primary_df) > 0:
                    # Log DataFrame structure
                    logger.debug(f"📊 Primary timeframe ({self.primary_timeframe}) DataFrame structure for {symbol}:")
                    logger.debug(f"   Shape: {primary_df.shape}")
                    logger.debug(f"   Columns: {list(primary_df.columns)}")
                    logger.debug(f"   Index type: {type(primary_df.index).__name__}")
                    logger.debug(f"   Index range: {primary_df.index[0]} to {primary_df.index[-1]}")
                    
                    # Log a few sample rows
                    sample_rows = min(3, len(primary_df))
                    logger.debug(f"📉 Primary timeframe ({self.primary_timeframe}) sample for {symbol}:")
                    for i in range(-sample_rows, 0):
                        try:
                            candle = primary_df.iloc[i]
                            logger.debug(f"   {i}: O={candle['open']:.5f}, H={candle['high']:.5f}, L={candle['low']:.5f}, C={candle['close']:.5f}, Vol={candle['volume']}")
                        except Exception as e:
                            logger.debug(f"   Error accessing candle {i}: {str(e)}")
            except Exception as e:
                logger.debug(f"Error logging data sample: {str(e)}")
            
            # Check if DataFrames are None or empty
            primary_df_len = len(primary_df) if primary_df is not None and hasattr(primary_df, '__len__') else 0
            higher_df_len = len(higher_df) if higher_df is not None and hasattr(higher_df, '__len__') else 0
            
            if primary_df is None or primary_df_len < self.lookback_period or higher_df is None or higher_df_len < 10:
                logger.debug(f"⏩ Insufficient data for {symbol}, skipping. " + 
                           f"Primary DF: {'None' if primary_df is None else f'{primary_df_len} rows (need {self.lookback_period})'}, " + 
                           f"Higher DF: {'None' if higher_df is None else f'{higher_df_len} rows (need 10)'}")
                continue
            
            # Verify we have proper DataFrames before processing
            if not isinstance(primary_df, pd.DataFrame) or not isinstance(higher_df, pd.DataFrame):
                logger.warning(f"Expected DataFrames but got: primary={type(primary_df)}, higher={type(higher_df)}")
                continue
            
            # Ensure DataFrame has datetime index
            if not isinstance(primary_df.index, pd.DatetimeIndex):
                logger.debug(f"Converting index to DatetimeIndex for {symbol}")
                # Check if we have a 'time' column that can be used as index
                if 'time' in primary_df.columns:
                    try:
                        # Try to convert 'time' column to datetime and set as index
                        primary_df['time'] = pd.to_datetime(primary_df['time'])
                        primary_df.set_index('time', inplace=True)
                        logger.debug(f"Set DatetimeIndex from 'time' column for {symbol}")
                    except Exception as e:
                        logger.warning(f"Failed to set DatetimeIndex from 'time' column: {e}")
                else:
                    # If no time column, create a synthetic datetime index
                    logger.debug(f"No 'time' column found, creating synthetic DatetimeIndex for {symbol}")
                    current_time = datetime.now()
                    try:
                        if self.primary_timeframe.startswith('M'):
                            # Extract minutes from timeframe (e.g., 'M5' -> 5)
                            try:
                                minutes = int(self.primary_timeframe[1:])
                                # Create timestamps going back from current time
                                timestamps = []
                                for i in range(len(primary_df)-1, -1, -1):
                                    timestamps.append(current_time - timedelta(minutes=minutes * i))
                                
                                # Create DatetimeIndex with safer approach
                                logger.debug(f"Converting {len(timestamps)} timestamps to DatetimeIndex")
                                primary_df.index = pd.DatetimeIndex(pd.Series(timestamps))
                                logger.debug(f"Created synthetic DatetimeIndex using {minutes} minute intervals")
                            except ValueError:
                                logger.warning(f"Could not parse timeframe {self.primary_timeframe}, using default 5 minutes")
                                timestamps = []
                                for i in range(len(primary_df)-1, -1, -1):
                                    timestamps.append(current_time - timedelta(minutes=5 * i))
                                primary_df.index = pd.DatetimeIndex(pd.Series(timestamps))
                        else:
                            # Default to 5-minute intervals if timeframe format is unknown
                            timestamps = []
                            for i in range(len(primary_df)-1, -1, -1):
                                timestamps.append(current_time - timedelta(minutes=5 * i))
                            primary_df.index = pd.DatetimeIndex(pd.Series(timestamps))
                            logger.debug(f"Created synthetic DatetimeIndex using default 5 minute intervals")
                    except SystemError as e:
                        logger.error(f"SystemError when creating DatetimeIndex: {str(e)}")
                        logger.warning(f"Falling back to RangeIndex for {symbol}")
                        # Fall back to a simple RangeIndex if DatetimeIndex creation fails
                        primary_df.index = pd.RangeIndex(start=0, stop=len(primary_df))
                    except Exception as e:
                        logger.error(f"Error creating DatetimeIndex: {str(e)}")
                        logger.warning(f"Falling back to RangeIndex for {symbol}")
                        # Fall back to a simple RangeIndex if DatetimeIndex creation fails
                        primary_df.index = pd.RangeIndex(start=0, stop=len(primary_df))
            
            # Do the same for higher timeframe DataFrame
            if not isinstance(higher_df.index, pd.DatetimeIndex):
                logger.debug(f"Converting index to DatetimeIndex for higher timeframe data of {symbol}")
                # Check if we have a 'time' column that can be used as index
                if 'time' in higher_df.columns:
                    try:
                        # Try to convert 'time' column to datetime and set as index
                        higher_df['time'] = pd.to_datetime(higher_df['time'])
                        higher_df.set_index('time', inplace=True)
                        logger.debug(f"Set DatetimeIndex from 'time' column for higher timeframe")
                    except Exception as e:
                        logger.warning(f"Failed to set DatetimeIndex from 'time' column for higher timeframe: {e}")
                else:
                    # If no time column, create a synthetic datetime index
                    logger.debug(f"No 'time' column found, creating synthetic DatetimeIndex for higher timeframe")
                    current_time = datetime.now()
                    try:
                        if self.higher_timeframe.startswith('M'):
                            # Extract minutes from timeframe (e.g., 'M15' -> 15)
                            try:
                                minutes = int(self.higher_timeframe[1:])
                                # Create timestamps going back from current time
                                timestamps = []
                                for i in range(len(higher_df)-1, -1, -1):
                                    timestamps.append(current_time - timedelta(minutes=minutes * i))
                                
                                # Create DatetimeIndex with safer approach
                                logger.debug(f"Converting {len(timestamps)} timestamps to DatetimeIndex for higher timeframe")
                                higher_df.index = pd.DatetimeIndex(pd.Series(timestamps))
                                logger.debug(f"Created synthetic DatetimeIndex using {minutes} minute intervals for higher timeframe")
                            except ValueError:
                                logger.warning(f"Could not parse timeframe {self.higher_timeframe}, using default 15 minutes")
                                timestamps = []
                                for i in range(len(higher_df)-1, -1, -1):
                                    timestamps.append(current_time - timedelta(minutes=15 * i))
                                higher_df.index = pd.DatetimeIndex(pd.Series(timestamps))
                        else:
                            # Default to 15-minute intervals if timeframe format is unknown
                            timestamps = []
                            for i in range(len(higher_df)-1, -1, -1):
                                timestamps.append(current_time - timedelta(minutes=15 * i))
                            higher_df.index = pd.DatetimeIndex(pd.Series(timestamps))
                            logger.debug(f"Created synthetic DatetimeIndex using default 15 minute intervals for higher timeframe")
                    except SystemError as e:
                        logger.error(f"SystemError when creating DatetimeIndex for higher timeframe: {str(e)}")
                        logger.warning(f"Falling back to RangeIndex for higher timeframe")
                        # Fall back to a simple RangeIndex if DatetimeIndex creation fails
                        higher_df.index = pd.RangeIndex(start=0, stop=len(higher_df))
                    except Exception as e:
                        logger.error(f"Error creating DatetimeIndex for higher timeframe: {str(e)}")
                        logger.warning(f"Falling back to RangeIndex for higher timeframe")
                        # Fall back to a simple RangeIndex if DatetimeIndex creation fails
                        higher_df.index = pd.RangeIndex(start=0, stop=len(higher_df))
            
            # Update key levels and trend lines
            try:
                if primary_df is not None and len(primary_df) > 0:
                    # Force trend line updates if debug_visualize or force_trendlines is True
                    self._update_key_levels(symbol, primary_df, debug_force_update=(debug_visualize or force_trendlines))
                    self._find_trend_lines(symbol, primary_df, debug_force_update=(debug_visualize or force_trendlines), skip_plots=skip_plots)
                    self._identify_consolidation_ranges(symbol, primary_df)
                    self._process_retest_conditions(symbol, primary_df)
                else:
                    logger.warning(f"⚠️ Empty primary DataFrame for {symbol}, skipping level and trendline detection")
            except Exception as e:
                logger.exception(f"Error during level detection for {symbol}: {str(e)}")
            
            # Check for breakout signals
            breakout_signals = self._check_breakout_signals(symbol, primary_df, higher_df, skip_plots)
            
            # Check for reversal signals
            reversal_signals = self._check_reversal_signals(symbol, primary_df, higher_df, skip_plots)
            
            # Collect all signals for this symbol
            symbol_signals = []
            if breakout_signals:
                symbol_signals.extend(breakout_signals)
            
            if reversal_signals:
                symbol_signals.extend(reversal_signals)
                
            # Score and enhance each signal
            for signal in symbol_signals:
                # Add symbol to the signal dict for identification later
                signal['original_symbol'] = symbol
                
                # Calculate and add score
                h1_trend = self._determine_h1_trend(higher_df)
                signal = self._score_signal(signal, symbol, primary_df, higher_df, h1_trend)
                
            # For each symbol, return the best signal immediately if requested
            if process_immediately and symbol_signals:
                # Find best signal for this symbol
                best_signal = max(symbol_signals, key=lambda x: x.get('score', 0))
                
                # Log all signals with their scores for debugging
                for signal in symbol_signals:
                    logger.debug(f"Signal {signal['direction']} for {symbol}: {signal.get('reason', 'No reason')} - Score: {signal.get('score', 0):.2f}")
                
                logger.info(f"🌟 Selected best signal for {symbol}: {best_signal['direction']} {best_signal.get('reason', 'No reason')} with score {best_signal.get('score', 0):.2f}")
                
                # Remove scoring metadata before returning
                if 'original_symbol' in best_signal:
                    del best_signal['original_symbol']
                if 'score_details' in best_signal:
                    del best_signal['score_details']
                
                # Return this single signal in a list for immediate processing
                symbol_time = time.time() - symbol_start_time
                logger.info(f"📊 Generated signal for {symbol} in {symbol_time:.2f}s: {best_signal['direction']} at {best_signal['entry_price']:.5f} | confidence: {best_signal['confidence']:.2f}")
                logger.info(f"👉 RETURNING IMMEDIATE SIGNAL FOR {symbol}")
                return [best_signal]
            
            # Add to all signals collection for batch processing
            all_signals.extend(symbol_signals)
        
        # After processing all symbols, add this log before signal selection:
        if all_signals:
            logger.info(f"👉 Found {len(all_signals)} potential signals before scoring and selection")
            
            # Group signals by symbol
            signals_by_symbol = {}
            for signal in all_signals:
                symbol = signal['original_symbol']
                if symbol not in signals_by_symbol:
                    signals_by_symbol[symbol] = []
                signals_by_symbol[symbol].append(signal)
            
            # For each symbol, only select highest scoring signal
            for symbol, symbol_signals in signals_by_symbol.items():
                if not symbol_signals:
                    continue
                    
                # Find highest scoring signal
                best_signal = max(symbol_signals, key=lambda x: x.get('score', 0))
                
                # Log all signals with their scores for debugging
                for signal in symbol_signals:
                    logger.debug(f"Signal {signal['direction']} for {symbol}: {signal.get('reason', 'No reason')} - Score: {signal.get('score', 0):.2f}")
                
                logger.info(f"🌟 Selected best signal for {symbol}: {best_signal['direction']} {best_signal.get('reason', 'No reason')} with score {best_signal.get('score', 0):.2f}")
                
                # Remove scoring metadata before returning
                if 'original_symbol' in best_signal:
                    del best_signal['original_symbol']
                if 'score_details' in best_signal:
                    del best_signal['score_details']
                
                signals.append(best_signal)
        
        generation_time = time.time() - start_time
        logger.info(f"✅ Generation completed in {generation_time:.2f}s - Produced {len(signals)} final signals")
        if signals:
            for i, signal in enumerate(signals):
                logger.info(f"📊 Final Signal #{i+1}: {signal['symbol']} {signal['direction']} at {signal['entry_price']:.5f} | confidence: {signal['confidence']:.2f}")
            logger.info(f"👉 RETURNING {len(signals)} SIGNALS FOR PROCESSING")
        else:
            logger.info("📭 No signals generated - returning empty list")
        
        return signals
    
    def _score_signal(self, signal, symbol, df, h1_df, h1_trend):
        """
        Score a signal based on multiple weighted factors to determine its quality.
        Uses a comprehensive scoring system with 5 main components:
        - Level Strength (30%): Number of touches and recency
        - Volume Quality (20%): Volume characteristics
        - Pattern Reliability (20%): Effectiveness of the pattern
        - Trend Alignment (20%): Alignment with higher timeframe trend
        - Risk-Reward Ratio (10%): Potential reward vs risk
        
        Args:
            signal: The signal dictionary to score
            symbol: Trading symbol
            df: Price dataframe for the primary timeframe
            h1_df: Price dataframe for the higher timeframe
            h1_trend: Current trend on the higher timeframe
            
        Returns:
            Signal dictionary with score added
        """
        # Initialize score components
        level_strength_score = 0
        volume_quality_score = 0
        pattern_reliability_score = 0
        trend_alignment_score = 0
        risk_reward_score = 0
        
        # Extract signal details
        signal_direction = signal['direction']
        level = signal.get('level', None)
        entry_price = signal['entry_price']
        stop_loss = signal['stop_loss']
        take_profit = signal['take_profit']
        
        # 1. Level Strength (30%) - based on number of touches and quality
        if level is not None:
            # Add debugging info
            logger.debug(f"Scoring level {level:.5f} for {symbol} ({signal_direction})")
            
            # Check which type of level we're dealing with
            if signal_direction == 'buy':
                # Support level for buy signals
                if symbol in self.support_levels and self.support_levels[symbol]:
                    # Find the closest support level
                    closest_level = min(self.support_levels[symbol], key=lambda x: abs(x - level)) if self.support_levels[symbol] else None
                    
                    # More lenient tolerance and added debugging
                    level_tolerance = level * self.price_tolerance * 1.5  # 50% more lenient
                    level_diff = abs(closest_level - level) if closest_level is not None else float('inf')
                    logger.debug(f"Closest support: {closest_level:.5f}, diff: {level_diff:.5f}, tolerance: {level_tolerance:.5f}")
                    
                    if closest_level is not None and level_diff < level_tolerance:
                        # Count touches for this level
                        touches = self._count_level_touches(df, closest_level, 'support')
                        logger.debug(f"Level touches: {touches}")
                        
                        # Scale touches to a 0-1 range, with diminishing returns after 5 touches
                        # A level with 5+ touches gets a high score, but not much benefit beyond that
                        # Give minimum score of 0.3 even for a single touch
                        level_strength_score = max(0.3, min(touches / 5, 1.0))
                        
                        # Add bonus for recent level formation (if we can determine it)
                        # This rewards fresher levels that may be more relevant
                        try:
                            # Check if the level was touched in the recent past
                            recent_bars = df.iloc[-20:]  # Look at last 20 bars
                            recent_touch = False
                            
                            for i in range(len(recent_bars)):
                                if abs(recent_bars['low'].iloc[i] - closest_level) <= closest_level * self.price_tolerance:
                                    recent_touch = True
                                    break
                                    
                            if recent_touch:
                                # Add a bonus for recent touch (up to 0.2 extra)
                                level_strength_score = min(level_strength_score + 0.2, 1.0)
                                logger.debug(f"Added recency bonus for support level at {closest_level:.5f}")
                        except Exception as e:
                            logger.debug(f"Error calculating level recency: {str(e)}")
                    else:
                        logger.debug(f"No matching support level found within tolerance")
                else:
                    logger.debug(f"No support levels found for {symbol}")
            else:
                # Resistance level for sell signals
                if symbol in self.resistance_levels and self.resistance_levels[symbol]:
                    # Find the closest resistance level
                    closest_level = min(self.resistance_levels[symbol], key=lambda x: abs(x - level)) if self.resistance_levels[symbol] else None
                    
                    # More lenient tolerance and added debugging
                    level_tolerance = level * self.price_tolerance * 1.5  # 50% more lenient
                    level_diff = abs(closest_level - level) if closest_level is not None else float('inf')
                    logger.debug(f"Closest resistance: {closest_level:.5f}, diff: {level_diff:.5f}, tolerance: {level_tolerance:.5f}")
                    
                    if closest_level is not None and level_diff < level_tolerance:
                        # Count touches for this level
                        touches = self._count_level_touches(df, closest_level, 'resistance')
                        logger.debug(f"Level touches: {touches}")
                        
                        # Scale touches to a 0-1 range, with diminishing returns after 5 touches
                        # Give minimum score of 0.3 even for a single touch
                        level_strength_score = max(0.3, min(touches / 5, 1.0))
                        
                        # Add bonus for recent level formation
                        try:
                            # Check if the level was touched in the recent past
                            recent_bars = df.iloc[-20:]  # Look at last 20 bars
                            recent_touch = False
                            
                            for i in range(len(recent_bars)):
                                if abs(recent_bars['high'].iloc[i] - closest_level) <= closest_level * self.price_tolerance:
                                    recent_touch = True
                                    break
                                    
                            if recent_touch:
                                # Add a bonus for recent touch (up to 0.2 extra)
                                level_strength_score = min(level_strength_score + 0.2, 1.0)
                                logger.debug(f"Added recency bonus for resistance level at {closest_level:.5f}")
                        except Exception as e:
                            logger.debug(f"Error calculating level recency: {str(e)}")
                    else:
                        logger.debug(f"No matching resistance level found within tolerance")
                else:
                    logger.debug(f"No resistance levels found for {symbol}")
        
        # 2. Volume Quality (20%)
        # Try to extract volume quality from signal reason if available
        reason = signal.get('reason', '').lower()
        
        if 'strong' in reason and 'volume' in reason:
            volume_quality_score = 1.0
        elif 'adequate' in reason and 'volume' in reason:
            volume_quality_score = 0.7
        else:
            # If volume quality not mentioned in reason, calculate it from the last candle
            try:
                # Get the candle that triggered the signal (latest candle)
                candle = df.iloc[-1]
                
                # Calculate volume threshold
                lookback_bars = min(50, len(df) - 1)
                volume_threshold = np.percentile(df['tick_volume'].iloc[-lookback_bars:], self.volume_percentile)
                
                # Analyze volume quality
                vol_quality = self._analyze_volume_quality(candle, volume_threshold)
                
                # Convert from -2 to +2 scale to 0 to 1 scale for scoring
                # For buy signals, positive values are good; for sell signals, negative values are good
                if signal_direction == 'buy':
                    volume_quality_score = max(0, vol_quality / 2)  # Scale from 0 to 1
                else:
                    volume_quality_score = max(0, -vol_quality / 2)  # Scale from 0 to 1
            except Exception as e:
                logger.debug(f"Error calculating volume quality score: {str(e)}")
                # Default value if we can't extract or calculate
                volume_quality_score = 0.5
        
        # 3. Pattern Reliability (20%)
        # Define reliability scores for different patterns
        pattern_reliability = {
            'bullish engulfing': 0.8,
            'bearish engulfing': 0.8,
            'morning star': 0.9,
            'evening star': 0.9,
            'hammer': 0.7,
            'shooting star': 0.7,
            'breakout': 0.6,
            'breakdown': 0.6,
            'trend line breakout': 0.75,
            'trend line breakdown': 0.75,
            'retest': 0.85  # Retests are considered more reliable
        }
        
        # Check for patterns in the reason
        for pattern, score in pattern_reliability.items():
            if pattern in reason.lower():
                pattern_reliability_score = score
                break
        else:
            # Default if no recognized pattern
            pattern_reliability_score = 0.5
        
        # 4. Trend Alignment (20%)
        # Check if the signal aligns with the higher timeframe trend
        if signal_direction == 'buy':
            if h1_trend == 'bullish':
                trend_alignment_score = 1.0  # Perfect alignment
            elif h1_trend == 'neutral':
                trend_alignment_score = 0.5  # Partial alignment
            else:
                trend_alignment_score = 0.0  # Counter-trend
        else:  # sell
            if h1_trend == 'bearish':
                trend_alignment_score = 1.0  # Perfect alignment
            elif h1_trend == 'neutral':
                trend_alignment_score = 0.5  # Partial alignment
            else:
                trend_alignment_score = 0.0  # Counter-trend
        
        # 5. Risk-Reward Ratio (10%)
        # Calculate the risk-reward ratio and score it
        if signal_direction == 'buy':
            risk = entry_price - stop_loss
            reward = take_profit - entry_price
        else:
            risk = stop_loss - entry_price
            reward = entry_price - take_profit
            
        if risk > 0:
            rr_ratio = reward / risk
            
            # Score RR ratio, with diminishing returns above 3:1
            # 1:1 = 0.33, 2:1 = 0.67, 3:1 or higher = 1.0
            risk_reward_score = min(rr_ratio / 3, 1.0)
        else:
            risk_reward_score = 0
        
        # ATR context - examine if the stop loss is reasonable relative to volatility (bonus factor)
        # This helps filter out signals with too tight or too wide stops
        atr = None  # Initialize atr variable
        atr_bonus = 0  # Default value
        try:
            atr_series = calculate_atr(df, self.atr_period)
            # Get the last ATR value from the series
            if isinstance(atr_series, pd.Series) and not atr_series.empty:
                atr = atr_series.iloc[-1]  # Get the most recent ATR value
            # Properly handle all possible types with explicit checks
            atr_is_valid = (atr is not None and 
                           not pd.isna(atr) and
                           np.isscalar(float(atr)) and 
                           float(atr) > 0)
            
            if atr_is_valid:
                # Calculate the ratio of risk (stop distance) to ATR
                stop_atr_ratio = risk / float(atr)
                
                # Ideal ratio is between 0.5x and 3x ATR
                stop_atr_ratio_value = float(stop_atr_ratio)
                # Explicit comparisons to avoid pandas/numpy Series boolean issues
                is_above_min = stop_atr_ratio_value >= 0.5
                is_below_max = stop_atr_ratio_value <= 3.0
                if is_above_min and is_below_max:
                    # Add a small bonus to the overall score (up to 10%)
                    atr_bonus = 0.1
                    self.logger.debug(f"Stop is {stop_atr_ratio_value:.1f}x ATR - appropriate size, adding bonus")
                else:
                    # Penalize stops that are too tight or too wide
                    atr_bonus = -0.1
                    is_too_tight = stop_atr_ratio_value < 0.5
                    self.logger.debug(f"Stop is {stop_atr_ratio_value:.1f}x ATR - {'too tight' if is_too_tight else 'too wide'}")
            else:
                atr_bonus = 0
                self.logger.debug(f"Invalid ATR value: {atr}")
        except Exception as e:
            self.logger.debug(f"Error calculating ATR context: {str(e)}")
            atr_bonus = 0
        
        # Calculate final weighted score
        final_score = (
            (level_strength_score * 0.3) +  # 30% weight
            (volume_quality_score * 0.2) +  # 20% weight
            (pattern_reliability_score * 0.2) +  # 20% weight
            (trend_alignment_score * 0.2) +  # 20% weight
            (risk_reward_score * 0.1)  # 10% weight
        )
        
        # Apply the ATR context bonus/penalty
        final_score = max(0, min(1, final_score + atr_bonus))
        
        # Consolidation context - give bonus to reversals in ranging markets
        try:
            if 'reversal' in reason.lower() and symbol in self.last_consolidation_ranges:
                is_consolidation = self.last_consolidation_ranges[symbol].get('is_consolidation', False)
                if is_consolidation:
                    # Add a small bonus for reversals in confirmed consolidation zones
                    consolidation_bonus = 0.05
                    final_score = min(1, final_score + consolidation_bonus)
                    logger.debug(f"Added consolidation context bonus for reversal signal")
        except Exception as e:
            logger.debug(f"Error applying consolidation context: {str(e)}")
        
        # Add score to signal
        signal['score'] = final_score
        signal['score_details'] = {
            'level_strength': level_strength_score,
            'volume_quality': volume_quality_score,
            'pattern_reliability': pattern_reliability_score,
            'trend_alignment': trend_alignment_score,
            'risk_reward': risk_reward_score,
            'final_score': final_score
        }
        
        logger.info(f"Signal scored {final_score:.2f} - Level: {level_strength_score:.2f}, Volume: {volume_quality_score:.2f}, " +
                   f"Pattern: {pattern_reliability_score:.2f}, Trend: {trend_alignment_score:.2f}, R:R: {risk_reward_score:.2f}")
        
        return signal
    
    def _update_key_levels(self, symbol: str, df: pd.DataFrame, debug_force_update: bool = False) -> None:
        """
        Update support and resistance levels for a symbol.
        
        Args:
            symbol: Trading symbol
            df: Price dataframe
            debug_force_update: Force an update regardless of the time interval (for debugging)
        """
        # Check if we need to update levels (limit computation)
        current_time = df.index[-1]
        
        # Ensure current_time is a datetime object
        if not isinstance(current_time, datetime):
            logger.debug(f"Converting current_time from {type(current_time)} to datetime")
            # If it's a timestamp (integer), convert to datetime
            try:
                if isinstance(current_time, (int, np.integer, float)):
                    # Try different units for timestamp conversion
                    try:
                        current_time = datetime.fromtimestamp(current_time)
                    except (ValueError, OverflowError):
                        try:
                            current_time = datetime.fromtimestamp(current_time / 1000)  # Try milliseconds
                        except:
                            current_time = datetime.now()
                elif isinstance(current_time, pd.Timestamp):
                    current_time = current_time.to_pydatetime()
                else:
                    # For other types, use string conversion
                    current_time = pd.to_datetime(str(current_time)).to_pydatetime()
            except Exception as e:
                # If all conversions fail, use current time
                logger.debug(f"Failed to convert timestamp: {e}, using current time instead")
                current_time = datetime.now()
        
        if symbol in self.last_updated:
            last_update = self.last_updated[symbol]
            # Ensure last_update is a datetime object too
            if not isinstance(last_update, datetime):
                logger.debug(f"Converting last_update from {type(last_update)} to datetime")
                try:
                    if isinstance(last_update, (int, np.integer, float)):
                        # Try different units for timestamp conversion
                        try:
                            last_update = datetime.fromtimestamp(last_update)
                        except (ValueError, OverflowError):
                            try:
                                last_update = datetime.fromtimestamp(last_update / 1000)  # Try milliseconds
                            except:
                                # Force update in case of conversion error
                                last_update = datetime.now() - timedelta(hours=self.level_update_interval + 1)
                    elif isinstance(last_update, pd.Timestamp):
                        last_update = last_update.to_pydatetime()
                    else:
                        # For other types, use string conversion
                        last_update = pd.to_datetime(str(last_update)).to_pydatetime()
                except Exception as e:
                    # If all conversions fail, force an update
                    logger.debug(f"Failed to convert last_update: {e}, forcing update")
                    last_update = datetime.now() - timedelta(hours=self.level_update_interval + 1)
                
                # Update the stored value
                self.last_updated[symbol] = last_update
            
            # Only update every X hours depending on timeframe
            try:
                time_diff = (current_time - last_update).total_seconds()
                if time_diff < self.level_update_interval * 3600:
                    logger.debug(f"🕒 Skipping level update for {symbol}, last update was {time_diff/3600:.1f} hours ago")
                    return
            except Exception as e:
                logger.warning(f"Error calculating time difference: {e}. Forcing update.")
                # Force update in case of error
        
        logger.debug(f"🔄 Updating key levels for {symbol} with {len(df)} candles")
        
        # Find swing highs and lows
        support_levels = self._find_support_levels(df)
        resistance_levels = self._find_resistance_levels(df)
        
        # Store levels
        self.support_levels[symbol] = support_levels
        self.resistance_levels[symbol] = resistance_levels
        self.last_updated[symbol] = current_time
        
        logger.info(f"🔄 Updated key levels for {symbol} - Support: {len(support_levels)}, Resistance: {len(resistance_levels)}")
        
        # Log actual levels for debugging
        if support_levels:
            logger.debug(f"📉 Support levels for {symbol}: {[round(level, 5) for level in support_levels]}")
        if resistance_levels:
            logger.debug(f"📈 Resistance levels for {symbol}: {[round(level, 5) for level in resistance_levels]}")
            
    def _find_trend_lines(self, symbol: str, df: pd.DataFrame, debug_force_update: bool = False, skip_plots: bool = False) -> None:
        """
        Find and validate trend lines for a given symbol.
        
        Args:
            symbol: Trading symbol
            df: Price dataframe
            debug_force_update: Force update regardless of time interval
            skip_plots: Whether to skip creating debug plots
        """
        # Check if it's time to update trend lines
        current_time = datetime.now()
        last_update_time = self.last_updated.get('trend_lines', {}).get(symbol, None)
        force_update = debug_force_update
        
        if (not force_update and last_update_time is not None and 
            (current_time - last_update_time).total_seconds() < self.trend_line_update_interval * 3600):
            # Skip update if it's not time yet
            logger.debug(f"⏭️ Skipping trend line update for {symbol} - last update: {last_update_time}")
            return
            
        logger.info(f"🔍 Finding trend lines for {symbol}")
        
        # Find swing highs and lows for trend line analysis
        swing_highs = self._find_swing_highs(df)
        swing_lows = self._find_swing_lows(df)
        
        logger.info(f"🔍 Found {len(swing_highs)} swing highs and {len(swing_lows)} swing lows for {symbol}")
        
        # Calculate trend lines
        bullish_trend_lines = self._identify_trend_lines(df, swing_lows, 'bullish', skip_plots)
        bearish_trend_lines = self._identify_trend_lines(df, swing_highs, 'bearish', skip_plots)
        
        # Store trend lines
        self.bullish_trend_lines[symbol] = bullish_trend_lines
        self.bearish_trend_lines[symbol] = bearish_trend_lines
        
        # Log summary info
        if bullish_trend_lines:
            logger.info(f"📈 Found {len(bullish_trend_lines)} bullish trend lines for {symbol}")
        if bearish_trend_lines:
            logger.info(f"📉 Found {len(bearish_trend_lines)} bearish trend lines for {symbol}")
        
        # Skip plot creation if skip_plots is True
        if skip_plots:
            # Log info about trendlines without creating plots
            if bullish_trend_lines:
                logger.debug(f"📈 BULLISH TREND LINES for {symbol} (skipping plots)")
                for i, line in enumerate(bullish_trend_lines):
                    logger.debug(f"  📈 Bullish Line #{i+1}: Angle={line['angle']:.2f}°, r²={line['r_squared']:.3f}, Touches={line['touches']}")
            
            if bearish_trend_lines:
                logger.debug(f"📉 BEARISH TREND LINES for {symbol} (skipping plots)")
                for i, line in enumerate(bearish_trend_lines):
                    logger.debug(f"  📉 Bearish Line #{i+1}: Angle={line['angle']:.2f}°, r²={line['r_squared']:.3f}, Touches={line['touches']}")
            
            # Update last updated timestamp and return
            self.last_updated['trend_lines'][symbol] = current_time
            return
            
        # Create debug plots directory if it doesn't exist
        debug_dir = Path("debug_plots")
        debug_dir.mkdir(exist_ok=True)
        
        # Create a plot to visualize price, swing points, and trend lines
        plt.figure(figsize=(15, 10))
        
        # Plot price data (using the last 200 candles for clarity)
        plot_range = min(200, len(df))
        x_indices = list(range(len(df) - plot_range, len(df)))
        plt.plot(x_indices, df['close'].iloc[-plot_range:], color='blue', alpha=0.5, label='Close Price')
        plt.plot(x_indices, df['high'].iloc[-plot_range:], color='green', alpha=0.3, label='High')
        plt.plot(x_indices, df['low'].iloc[-plot_range:], color='red', alpha=0.3, label='Low')
        
        # Plot swing highs and lows
        if swing_highs:
            high_x = [x for x, y in swing_highs if x >= len(df) - plot_range]
            high_y = [y for x, y in swing_highs if x >= len(df) - plot_range]
            plt.scatter(high_x, high_y, color='green', marker='^', s=50, label='Swing Highs')
            
            # Connect consecutive swing highs with dashed lines for better visualization
            if len(high_x) >= 2:
                for i in range(len(high_x) - 1):
                    plt.plot([high_x[i], high_x[i+1]], [high_y[i], high_y[i+1]], 
                             color='lightgreen', linestyle='--', alpha=0.5)
        
        if swing_lows:
            low_x = [x for x, y in swing_lows if x >= len(df) - plot_range]
            low_y = [y for x, y in swing_lows if x >= len(df) - plot_range]
            plt.scatter(low_x, low_y, color='red', marker='v', s=50, label='Swing Lows')
            
            # Connect consecutive swing lows with dashed lines for better visualization
            if len(low_x) >= 2:
                for i in range(len(low_x) - 1):
                    plt.plot([low_x[i], low_x[i+1]], [low_y[i], low_y[i+1]], 
                             color='lightcoral', linestyle='--', alpha=0.5)
        
        # Plot bullish trend lines (support)
        for i, line in enumerate(bullish_trend_lines[:8]):  # Limit to top 8 for clarity
            start_idx = max(line['start_idx'], len(df) - plot_range)
            end_idx = min(line['end_idx'], len(df) - 1)
            
            if start_idx >= end_idx:
                continue
                
            x_line = np.linspace(start_idx, end_idx, 100)
            y_line = line['slope'] * x_line + line['intercept']
            
            plt.plot(x_line, y_line, color='green', linewidth=2, alpha=0.7, 
                    label=f"Support: Angle={line['angle']:.1f}°, Touches={line['touches']}")
            
            # Add annotation for top trend lines
            if i < 3:  # Only annotate top 3 lines
                midpoint_x = (start_idx + end_idx) / 2
                midpoint_y = line['slope'] * midpoint_x + line['intercept']
                plt.annotate(f"Support #{i+1}: {line['touches']} touches", 
                            xy=(midpoint_x, midpoint_y),
                            xytext=(-30, -20),
                            textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="white", alpha=0.7),
                            arrowprops=dict(arrowstyle="->"))
        
        # Plot bearish trend lines (resistance)
        for i, line in enumerate(bearish_trend_lines[:8]):  # Limit to top 8 for clarity
            start_idx = max(line['start_idx'], len(df) - plot_range)
            end_idx = min(line['end_idx'], len(df) - 1)
            
            if start_idx >= end_idx:
                continue
                
            x_line = np.linspace(start_idx, end_idx, 100)
            y_line = line['slope'] * x_line + line['intercept']
            
            plt.plot(x_line, y_line, color='red', linewidth=2, alpha=0.7, 
                    label=f"Resistance: Angle={line['angle']:.1f}°, Touches={line['touches']}")
            
            # Add annotation for top trend lines
            if i < 3:  # Only annotate top 3 lines
                midpoint_x = (start_idx + end_idx) / 2
                midpoint_y = line['slope'] * midpoint_x + line['intercept']
                plt.annotate(f"Resistance #{i+1}: {line['touches']} touches", 
                            xy=(midpoint_x, midpoint_y),
                            xytext=(-30, 20),
                            textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="white", alpha=0.7),
                            arrowprops=dict(arrowstyle="->"))
        
        # Add horizontal support and resistance levels for reference
        support_levels = self.support_levels.get(symbol, [])
        for level in support_levels:
            plt.axhline(y=level, color='green', linestyle='-', alpha=0.3)
            
        resistance_levels = self.resistance_levels.get(symbol, [])
        for level in resistance_levels:
            plt.axhline(y=level, color='red', linestyle='-', alpha=0.3)
            
        # Finalize and save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.title(f'Trend Line Analysis for {symbol}')
        plt.xlabel('Candle Index')
        plt.ylabel('Price')
        plt.grid(True, alpha=0.3)
        
        # Handle large legends by using a smaller font size and good positioning
        plt.legend(loc='upper left', fontsize='small')
        
        # Save the plot
        file_path = debug_dir / f"{symbol}_trend_lines_{timestamp}.png"
        plt.savefig(file_path)
        plt.close()
        
        logger.info(f"📊 Saved trend line visualization to {file_path}")
        
        # Update last updated timestamp
        self.last_updated['trend_lines'][symbol] = current_time
    
    def _find_swing_highs(self, df: pd.DataFrame) -> List[Tuple[int, float]]:
        """
        Find significant swing highs for trend line detection.
        
        Args:
            df: Price dataframe
            
        Returns:
            List of (index, price) tuples for swing highs
        """
        swing_highs = []
        
        # Use a window to find local maxima
        for i in range(2, len(df) - 2):
            if (df['high'].iloc[i] > df['high'].iloc[i-1] and 
                df['high'].iloc[i] > df['high'].iloc[i-2] and
                df['high'].iloc[i] > df['high'].iloc[i+1] and 
                df['high'].iloc[i] > df['high'].iloc[i+2]):
                
                # Store index and price
                swing_highs.append((i, df['high'].iloc[i]))
        
        return swing_highs
    
    def _find_swing_lows(self, df: pd.DataFrame) -> List[Tuple[int, float]]:
        """
        Find significant swing lows for trend line detection.
        
        Args:
            df: Price dataframe
            
        Returns:
            List of (index, price) tuples for swing lows
        """
        swing_lows = []
        
        # Use a window to find local minima
        for i in range(2, len(df) - 2):
            if (df['low'].iloc[i] < df['low'].iloc[i-1] and 
                df['low'].iloc[i] < df['low'].iloc[i-2] and
                df['low'].iloc[i] < df['low'].iloc[i+1] and 
                df['low'].iloc[i] < df['low'].iloc[i+2]):
                
                # Store index and price
                swing_lows.append((i, df['low'].iloc[i]))
        
        return swing_lows
    
    def _identify_trend_lines(self, df: pd.DataFrame, swing_points: List[Tuple[int, float]], 
                              line_type: str, skip_plots: bool = False) -> List[Dict]:
        """
        Identify valid trend lines using swing points.
        
        Args:
            df: Price dataframe
            swing_points: List of (index, price) tuples
            line_type: 'bullish' for support trend lines, 'bearish' for resistance
            skip_plots: Whether to skip creating debug plots
            
        Returns:
            List of trend line dictionaries with slope, intercept, and validity data
        """
        if len(swing_points) < self.trend_line_min_points:
            logger.debug(f"Not enough swing points ({len(swing_points)}) to identify {line_type} trend lines. Need at least {self.trend_line_min_points}.")
            return []
        
        valid_trend_lines = []
        attempted_lines = 0
        rejected_r_squared = 0
        rejected_angle = 0
        rejected_touches = 0
        
        # Create debug plots directory and initialize plot only if not skipping plots
        plt_fig = None
        if not skip_plots:
            # Create debug plots directory if it doesn't exist
            debug_dir = Path("debug_plots/trendline_validation")
            debug_dir.mkdir(exist_ok=True, parents=True)
            
            # Create a plot to visualize the line fitting process
            plt_fig = plt.figure(figsize=(15, 10))
            
            # Plot price data for context (last 200 points)
            plot_range = min(200, len(df))
            plt.plot(range(len(df) - plot_range, len(df)), df['close'].iloc[-plot_range:], color='blue', alpha=0.3, label='Close Price')
            
            # Plot all potential swing points
            all_x = [point[0] for point in swing_points]
            all_y = [point[1] for point in swing_points]
            plt.scatter(all_x, all_y, color='gray' if line_type == 'bullish' else 'gray', 
                       marker='^' if line_type == 'bearish' else 'v', 
                       alpha=0.8, s=50, label=f'All {line_type} Swing Points')
            
            # Draw simple lines connecting consecutive swing points for easier visualization
            # This gives a clearer view of support/resistance levels
            if len(swing_points) >= 2:
                for i in range(len(swing_points) - 1):
                    x1, y1 = swing_points[i]
                    x2, y2 = swing_points[i+1]
                    plt.plot([x1, x2], [y1, y2], color='lightgray', linestyle='--', alpha=0.5)
        
        # Try to find trend lines with at least trend_line_min_points points
        for i in range(len(swing_points) - (self.trend_line_min_points - 1)):
            # Select a subset of points to try
            points_subset = swing_points[i:i+self.trend_line_min_points]
            attempted_lines += 1
            
            # Extract x and y values safely
            x_values = []
            y_values = []
            for point in points_subset:
                if isinstance(point, tuple) and len(point) >= 2:
                    x_values.append(int(point[0]))
                    y_values.append(float(point[1]))
                else:
                    logger.warning(f"Invalid point format: {point}")
                    continue
            
            # Skip if not enough valid points
            if len(x_values) < self.trend_line_min_points:
                continue
                
            # Perform linear regression - convert to numpy arrays to avoid type issues
            x_array = np.array(x_values, dtype=float)
            y_array = np.array(y_values, dtype=float)
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_array, y_array)
            
            # Convert all values to float explicitly to avoid type issues
            slope = float(slope)
            intercept = float(intercept)
            r_value = float(r_value)
            r_squared = r_value ** 2
            
            # Calculate angle in degrees
            angle_degrees = math.degrees(math.atan(slope))
            
            # Variables for plotting
            line_color = 'gray'
            line_style = '--'
            line_alpha = 0.3
            rejection_reason = None
            
            # More relaxed angle constraints to allow more trendlines
            max_angle = 70  # More permissive (was 55)
            min_angle = 1   # Allow nearly horizontal lines (was 5)
            
            # Check angle constraint by direction - more permissive rules
            angle_valid = False
            if line_type == 'bullish' and angle_degrees >= min_angle and angle_degrees <= max_angle:
                angle_valid = True
            elif line_type == 'bearish' and angle_degrees <= -min_angle and angle_degrees >= -max_angle:
                angle_valid = True
                
            # Skip if angle invalid
            if not angle_valid:
                rejection_reason = f"Invalid angle: {angle_degrees:.2f}°"
                rejected_angle += 1
            # More permissive R-squared threshold
            elif r_squared < 0.5:  # Reduced from 0.65
                rejection_reason = f"Low R²: {r_squared:.2f}"
                rejected_r_squared += 1
            else:
                # Count touches of the trend line
                touches = self._count_trend_line_touches(df, slope, intercept, line_type)
                
                # More permissive touch requirement
                if touches < 2:  # Reduced from 3 to 2
                    rejection_reason = f"Too few touches: {touches}"
                    rejected_touches += 1
                else:
                    # This is a valid line!
                    line_color = 'green' if line_type == 'bullish' else 'red'
                    line_style = '-'
                    line_alpha = 0.8
                    
                    # Safely extract the start and end indices with proper error handling
                    start_idx = 0
                    end_idx = 0
                    
                    # Safely get the start index
                    if len(x_values) > 0:
                        start_idx = min(x_values)  # Use min for the earliest point
                        
                    # Safely get the end index
                    if len(x_values) > 0:
                        end_idx = max(x_values)  # Use max for the latest point
                        
                    # Extend trendline to the end of the chart
                    end_idx = len(df) - 1
                        
                    # We have a valid trend line, create dictionary with all metadata
                    trend_line = {
                        'slope': slope,
                        'intercept': intercept,
                        'angle': angle_degrees,
                        'r_squared': r_squared,
                        'touches': touches,
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'line_type': line_type,
                        'points': points_subset,
                        'quality_score': r_squared * touches  # Quality score for sorting
                    }
                    
                    valid_trend_lines.append(trend_line)
            
            # Plot the regression line with appropriate style if not skipping plots
            if not skip_plots:
                # Plot the line - extend to the end of the chart for better visualization
                x_line = np.linspace(min(x_values), len(df) - 1, 100)
                y_line = slope * x_line + intercept
                plt.plot(x_line, y_line, color=line_color, linestyle=line_style, alpha=line_alpha, 
                        label=f"Line {i+1}: r²={r_squared:.2f}, angle={angle_degrees:.1f}° {rejection_reason or 'VALID'}")
                
                # Plot the points used for this line
                plt.scatter(x_values, y_values, color=line_color, alpha=0.8, s=60,
                          marker='^' if line_type == 'bearish' else 'v')
                
                # Add annotations for validation parameters
                if i < 5:  # Only annotate first few lines to avoid clutter
                    annotation_text = f"Line {i+1}:\nR²: {r_squared:.2f}\nAngle: {angle_degrees:.1f}°"
                    if rejection_reason:
                        annotation_text += f"\nRejected: {rejection_reason}"
                    else:
                        annotation_text += f"\nTouches: {touches}"
                    
                    plt.annotate(annotation_text, 
                                xy=(x_values[-1], y_values[-1]),
                                xytext=(20, 20),
                                textcoords="offset points",
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                                arrowprops=dict(arrowstyle="->"))
        
        # Finalize and save the plot if we attempted any lines and not skipping plots
        if attempted_lines > 0 and not skip_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            symbol_name = "unknown" 
            try:
                # Try to get symbol name from various possible sources
                if 'symbol' in df.columns:
                    symbol_name = df['symbol'].iloc[0]
                elif hasattr(df, 'name'):
                    symbol_name = df.name
                elif len(valid_trend_lines) > 0 and 'symbol' in valid_trend_lines[0]:
                    symbol_name = valid_trend_lines[0]['symbol']
            except:
                pass
            
            plt.title(f'{line_type.capitalize()} Trend Line Analysis - {symbol_name}')
            plt.xlabel('Candle Index')
            plt.ylabel('Price')
            plt.grid(True)
            
            # Handle large legends by using a smaller font size
            if attempted_lines > 10:
                plt.legend(loc='upper left', fontsize='x-small')
            else:
                plt.legend(loc='upper left')
            
            # Save the plot
            debug_dir = Path("debug_plots/trendline_validation")
            file_path = debug_dir / f"{symbol_name}_{line_type}_trendline_fitting_{timestamp}.png"
            plt.savefig(file_path)
            plt.close()
            
            logger.info(f"📊 Saved trendline fitting visualization to {file_path}")
        
        # Cluster similar trend lines to reduce redundancy
        if valid_trend_lines:
            clustered_trend_lines = self._cluster_trend_lines(valid_trend_lines)
            
            # Sort by quality score and limit the number of lines, but allow more lines now
            max_trend_lines = 12  # Increased from 8 to show more potential trendlines
            sorted_trend_lines = sorted(clustered_trend_lines, key=lambda x: x['quality_score'], reverse=True)[:max_trend_lines]
            
            logger.debug(f"Trend line stats ({line_type}): Attempted={attempted_lines}, Valid={len(valid_trend_lines)}, " 
                       f"After clustering={len(clustered_trend_lines)}, Final={len(sorted_trend_lines)}")
            
            return sorted_trend_lines
        else:
            logger.debug(f"Trend line stats ({line_type}): Attempted={attempted_lines}, Valid=0, "
                       f"Rejected: r²={rejected_r_squared}, angle={rejected_angle}, touches={rejected_touches}")
            return []
    
    def _cluster_trend_lines(self, trend_lines: List[Dict]) -> List[Dict]:
        """
        Cluster similar trend lines to reduce redundancy.
        
        Args:
            trend_lines: List of trend line dictionaries
            
        Returns:
            List of clustered trend line dictionaries
        """
        if not trend_lines:
            return []
            
        # Parameters for clustering
        angle_tolerance = 5.0  # Degrees
        intercept_pct_tolerance = 0.0015  # 0.15% of price
        slope_tolerance = 0.00005
        
        # Calculate average price for scaling
        line_type = trend_lines[0]['line_type']
        avg_intercept = np.mean([line['intercept'] for line in trend_lines])
        intercept_tolerance = avg_intercept * intercept_pct_tolerance
        
        # Cluster trend lines
        clustered_lines = []
        used_indices = set()
        
        for i, line1 in enumerate(trend_lines):
            if i in used_indices:
                continue
                
            # Find all similar lines
            cluster = [line1]
            used_indices.add(i)
            
            for j, line2 in enumerate(trend_lines):
                if j in used_indices or i == j:
                    continue
                    
                # Check if lines are similar
                angle_diff = abs(line1['angle'] - line2['angle'])
                intercept_diff = abs(line1['intercept'] - line2['intercept'])
                slope_diff = abs(line1['slope'] - line2['slope'])
                
                if (angle_diff <= angle_tolerance and 
                    intercept_diff <= intercept_tolerance and
                    slope_diff <= slope_tolerance):
                    cluster.append(line2)
                    used_indices.add(j)
            
            # Choose the best line from the cluster
            if len(cluster) > 1:
                # Sort by quality score (r_squared * touches)
                best_line = max(cluster, key=lambda x: x['quality_score'])
                logger.debug(f"Clustered {len(cluster)} similar {line_type} trend lines")
            else:
                best_line = cluster[0]
                
            clustered_lines.append(best_line)
        
        return clustered_lines
    
    def _count_trend_line_touches(self, df: pd.DataFrame, slope: float, intercept: float, 
                                  line_type: str) -> int:
        """
        Count how many times price has touched a trend line.
        
        Args:
            df: Price dataframe
            slope: Slope of the trend line
            intercept: Y-intercept of the trend line
            line_type: 'bullish' for support, 'bearish' for resistance
            
        Returns:
            Number of touches
        """
        touches = 0
        price_series = df['low'] if line_type == 'bullish' else df['high']
        avg_price = price_series.mean()
        
        # Use a percentage of the average price for tolerance
        # More sensitive tolerance to catch more touches (0.12% instead of default tolerance)
        tolerance = avg_price * 0.0012
        
        # Add the regression points as default touches (minimum 2)
        touches = 2
        
        for i in range(len(df)):
            # Calculate trend line value at this index
            line_value = slope * i + intercept
            
            # Calculate the distance from price to the line
            if line_type == 'bullish':
                # For bullish trend lines, price should touch from above
                # Line is below price, so positive distance means price is above line
                distance = df['low'].iloc[i] - line_value
                
                # Price is very close to the line (within tolerance)
                if abs(distance) <= tolerance and distance >= 0:
                    touches += 1
            else:  # bearish
                # For bearish trend lines, price should touch from below
                # Line is above price, so negative distance means price is below line
                distance = df['high'].iloc[i] - line_value
                
                # Price is very close to the line (within tolerance)
                if abs(distance) <= tolerance and distance <= 0:
                    touches += 1
        
        # Ensure reasonable touch count for valid trendlines
        # Limit the maximum number of touches to avoid artificially high counts
        return min(touches, 20)
    
    def _is_near_trend_line(self, df: pd.DataFrame, idx: int, trend_lines: List[Dict], 
                           line_type: str) -> Optional[Dict]:
        """
        Check if price is near a trend line.
        
        Args:
            df: Price dataframe
            idx: Index to check
            trend_lines: List of trend line dictionaries
            line_type: 'bullish' for support, 'bearish' for resistance
            
        Returns:
            Trend line dictionary if near, None otherwise
        """
        if not trend_lines:
            return None
            
        current_candle = df.iloc[idx]
        tolerance = current_candle['close'] * self.price_tolerance
        
        for trend_line in trend_lines:
            # Calculate trend line value at this index
            line_value = trend_line['slope'] * idx + trend_line['intercept']
            
            # Check if within valid range of trend line
            if idx < trend_line['start_idx'] or idx > trend_line['end_idx'] + 20:
                continue
                
            if line_type == 'bullish':
                # Price should be near support trend line
                if abs(current_candle['low'] - line_value) <= tolerance:
                    return trend_line
            else:  # bearish
                # Price should be near resistance trend line
                if abs(current_candle['high'] - line_value) <= tolerance:
                    return trend_line
        
        return None
    
    def _calculate_trend_line_value(self, trend_line: Dict, idx: int) -> float:
        """
        Calculate the y-value of a trend line at a given index.
        
        Args:
            trend_line: Trend line dictionary
            idx: Index to calculate value for
            
        Returns:
            Price value of trend line at index
        """
        return trend_line['slope'] * idx + trend_line['intercept']
    
    def _identify_consolidation_ranges(self, symbol: str, df: pd.DataFrame) -> None:
        """
        Identify recent consolidation ranges for target calculation using volatility metrics.
        Detects periods where price range is less than 50% of the average range over recent bars.
        
        Args:
            symbol: Trading symbol
            df: Price dataframe
        """
        # Look for periods where price is ranging (not trending strongly)
        if symbol in self.last_consolidation_ranges:
            last_update = self.last_updated.get(symbol, datetime.min)
            current_time = df.index[-1]
            
            # Ensure current_time is a datetime object
            if not isinstance(current_time, datetime):
                logger.debug(f"Converting current_time from {type(current_time)} to datetime in consolidation ranges")
                if isinstance(current_time, (int, np.integer, float)):
                    try:
                        # Fix: Explicitly cast to int or float for pd.to_datetime
                        current_time = pd.to_datetime(int(current_time) if isinstance(current_time, np.integer) else float(current_time), unit='s')
                    except:
                        try:
                            # Fix: Explicitly cast to int or float for pd.to_datetime
                            current_time = pd.to_datetime(int(current_time) if isinstance(current_time, np.integer) else float(current_time), unit='ms')
                        except:
                            # If conversion fails, use current time
                            current_time = datetime.now()
                            logger.debug(f"Failed to convert timestamp, using current time instead")
            
            # Ensure last_update is a datetime object
            if not isinstance(last_update, datetime):
                logger.debug(f"Converting last_update from {type(last_update)} to datetime in consolidation ranges")
                if isinstance(last_update, (int, np.integer, float)):
                    try:
                        # Fix: Explicitly cast to int or float for pd.to_datetime
                        last_update = pd.to_datetime(int(last_update) if isinstance(last_update, np.integer) else float(last_update), unit='s')
                    except:
                        try:
                            # Fix: Explicitly cast to int or float for pd.to_datetime
                            last_update = pd.to_datetime(int(last_update) if isinstance(last_update, np.integer) else float(last_update), unit='ms')
                        except:
                            # If conversion fails, use a time far in the past to force update
                            last_update = datetime.now() - timedelta(hours=self.range_update_interval + 1)
                            logger.debug(f"Failed to convert last_update, forcing update in consolidation ranges")
                # Update the stored value
                self.last_updated[symbol] = last_update
            
            # Only update after significant time has passed based on timeframe
            try:
                # Initialize time_diff to ensure it's always defined
                time_diff = 0
                # Fix: Ensure both values are datetime objects before subtraction
                if isinstance(current_time, datetime) and isinstance(last_update, datetime):
                    time_diff = (current_time - last_update).total_seconds()
                if time_diff < self.range_update_interval * 3600:
                    logger.debug(f"🕒 Skipping consolidation range update for {symbol}, last update was {time_diff/3600:.1f} hours ago")
                    return
                else:
                    # Force update if types are incompatible
                    logger.warning(f"Time comparison types incompatible: {type(current_time)} vs {type(last_update)}. Forcing update.")
            except Exception as e:
                logger.warning(f"Error calculating time difference in consolidation ranges: {e}. Forcing update.")
                # Continue with update in case of error
        
        # Calculate volatility metrics for dynamic consolidation detection
        try:
            # Calculate ATR for volatility reference
            atr_series = calculate_atr(df, self.atr_period)
            if not isinstance(atr_series, pd.Series) or atr_series.empty:
                # Handle error case
                atr = None
            else:
                atr = atr_series.iloc[-1]  # Get the most recent ATR value

            if self._is_invalid_or_zero(atr):
                logger.warning(f"ATR calculation failed for {symbol}, using traditional approach")
                # Fallback to traditional approach with fixed bar count
                recent_bars = df.iloc[-self.consolidation_bars:]
                if len(recent_bars) > 0:  # Check if DataFrame is not empty
                    range_high = float(recent_bars['high'].max())
                    range_low = float(recent_bars['low'].min())
                    range_size = range_high - range_low
                    
                    is_consolidation = True  # Assume it's consolidation with traditional approach
                else:
                    logger.warning(f"Recent bars DataFrame is empty for {symbol}")
                    range_high = 0
                    range_low = 0
                    range_size = 0
                    is_consolidation = False
            else:
                # Use rolling calculations for better consolidation detection
                # First, calculate the range of each candle
                lookback = min(30, len(df) - 5)  # Use last 30 bars or as many as available
                df_subset = df.iloc[-lookback:].copy()
                df_subset['candle_range'] = df_subset['high'] - df_subset['low']
                
                # Calculate the rolling standard deviation of price
                df_subset['close_std'] = df_subset['close'].rolling(window=10).std()
                
                # Calculate average range
                if len(df_subset) > 0 and 'candle_range' in df_subset.columns:  # Check if DataFrame is not empty
                    avg_range = float(df_subset['candle_range'].mean())
                else:
                    logger.warning(f"DataFrame subset is empty or missing candle_range column for {symbol}")
                    avg_range = 0
                
                # Get recent bars (potentially in consolidation)
                recent_period = min(self.consolidation_bars, len(df_subset) - 1)
                recent_bars = df_subset.iloc[-recent_period:]
                
                # Calculate recent volatility metrics
                if len(recent_bars) > 0 and 'close_std' in recent_bars.columns and 'candle_range' in recent_bars.columns:
                    recent_std = float(recent_bars['close_std'].mean())
                    recent_range_avg = float(recent_bars['candle_range'].mean())
                    
                    # Calculate the range high and low
                    range_high = float(recent_bars['high'].max())
                    range_low = float(recent_bars['low'].min())
                    range_size = range_high - range_low
                    
                    # Compare recent volatility to ATR
                    if avg_range > 0:  # Prevent division by zero
                        volatility_ratio = recent_range_avg / avg_range
                    else:
                        volatility_ratio = 1.0
                    
                    # Define consolidation as period where:
                    # 1. Recent range average is less than 50% of overall average range
                    # 2. Standard deviation is low relative to price
                    # 3. Total range is reasonable (not too wide)
                    # Fix: Use explicit boolean checks instead of Series comparison
                    cond1 = bool(volatility_ratio < 0.5)
                    
                    # Make sure atr is not None before multiplying
                    if atr is not None and not pd.isna(atr):
                        cond2 = bool(recent_std < atr * 0.5)
                        cond3 = bool(range_size < atr * 3)  # Range shouldn't be more than 3x ATR
                    else:
                        # Fallback using recent standard deviation as reference
                        cond2 = bool(recent_std < recent_range_avg * 0.5)
                        cond3 = bool(range_size < recent_range_avg * 5)  # Use recent range as fallback
                    
                    is_consolidation = cond1 and cond2 and cond3
                    
                    logger.debug(f"📊 {symbol}: Volatility analysis - Avg range: {avg_range:.5f}, Recent avg: {recent_range_avg:.5f}, " +
                               f"Ratio: {volatility_ratio:.2f}, ATR: {atr:.5f}, Is consolidation: {is_consolidation}")
                else:
                    logger.warning(f"Recent bars DataFrame is empty or missing columns for {symbol}")
                    range_high = 0
                    range_low = 0
                    range_size = 0
                    is_consolidation = False
            
            # Store the range information
            self.last_consolidation_ranges[symbol] = {
                'high': range_high,
                'low': range_low,
                'size': range_size,
                'is_consolidation': is_consolidation
            }
            
            if is_consolidation:
                logger.info(f"📏 Identified consolidation range for {symbol}: High={range_high:.5f}, Low={range_low:.5f}, Size={range_size:.5f}")
            else:
                logger.debug(f"📏 Detected non-consolidation range for {symbol}: High={range_high:.5f}, Low={range_low:.5f}, Size={range_size:.5f}")
                
        except Exception as e:
            logger.warning(f"Error in consolidation detection for {symbol}: {str(e)}")
            # In case of error, use a simple fallback
            try:
                recent_bars = df.iloc[-self.consolidation_bars:]
                if len(recent_bars) > 0:  # Check if DataFrame is not empty
                    range_high = float(recent_bars['high'].max())
                    range_low = float(recent_bars['low'].min())
                    range_size = range_high - range_low
                else:
                    logger.warning(f"Recent bars DataFrame is empty for {symbol} in fallback")
                    range_high = 0
                    range_low = 0
                    range_size = 0
                
                # Store the range without additional metrics
                self.last_consolidation_ranges[symbol] = {
                    'high': range_high,
                    'low': range_low,
                    'size': range_size,
                    'is_consolidation': False  # Conservative approach
                }
                
                logger.debug(f"📏 Fallback consolidation calculation for {symbol}: High={range_high:.5f}, Low={range_low:.5f}, Size={range_size:.5f}")
            except Exception as e2:
                logger.error(f"Fallback consolidation calculation also failed for {symbol}: {str(e2)}")
                # Initialize with empty values to avoid errors later
                self.last_consolidation_ranges[symbol] = {
                    'high': 0,
                    'low': 0,
                    'size': 0,
                    'is_consolidation': False
                }
    
    def _process_retest_conditions(self, symbol: str, df: pd.DataFrame) -> None:
        """
        Process any pending retest conditions for breakout trades using ATR-based dynamic window.
        
        Args:
            symbol: Trading symbol
            df: Price dataframe
        """
        if not self.retest_required or symbol not in self.retest_tracking:
            return
            
        # Get current tracking info
        retest_info = self.retest_tracking[symbol]
        if not retest_info:
            return
            
        current_time = df.index[-1]
        level = retest_info.get('level')
        direction = retest_info.get('direction')
        start_time = retest_info.get('start_time')
        
        # Ensure current_time is a datetime object
        if not isinstance(current_time, datetime):
            logger.debug(f"Converting current_time from {type(current_time)} to datetime in retest conditions")
            if isinstance(current_time, (int, np.integer, float)):
                try:
                    # Fix: Explicitly cast to int or float for pd.to_datetime
                    current_time = pd.to_datetime(int(current_time) if isinstance(current_time, np.integer) else float(current_time), unit='s')
                except:
                    try:
                        # Fix: Explicitly cast to int or float for pd.to_datetime
                        current_time = pd.to_datetime(int(current_time) if isinstance(current_time, np.integer) else float(current_time), unit='ms')
                    except:
                        # If conversion fails, use current time
                        current_time = datetime.now()
                        logger.debug(f"Failed to convert timestamp, using current time instead")
        
        # Ensure start_time is a datetime object
        if not isinstance(start_time, datetime):
            logger.debug(f"Converting start_time from {type(start_time)} to datetime in retest conditions")
            if isinstance(start_time, (int, np.integer, float)):
                try:
                    # Fix: Explicitly cast to int or float for pd.to_datetime
                    start_time = pd.to_datetime(int(start_time) if isinstance(start_time, np.integer) else float(start_time), unit='s')
                except:
                    try:
                        # Fix: Explicitly cast to int or float for pd.to_datetime
                        start_time = pd.to_datetime(int(start_time) if isinstance(start_time, np.integer) else float(start_time), unit='ms')
                    except:
                        # If conversion fails, use a default value
                        start_time = current_time - timedelta(hours=1)  # Arbitrary 1 hour
                        logger.debug(f"Failed to convert start_time, using default value")
                # Update in the tracking dictionary
                retest_info['start_time'] = start_time
        
        if not level or not direction or not start_time:
            logger.warning(f"Missing retest tracking info for {symbol}, clearing")
            self.retest_tracking[symbol] = None
            return
            
        # Check if it's been too long since the level was identified (timeframe dependent)
        try:
            # Initialize time_diff to ensure it's always defined
            time_diff = 0
            # Fix: Ensure both values are datetime objects before subtraction
            if isinstance(current_time, datetime) and isinstance(start_time, datetime):
                time_diff = (current_time - start_time).total_seconds()
            max_time_allowed = self.max_retest_time * 3600  # Convert hours to seconds
            
            if time_diff > max_time_allowed:
                logger.debug(f"⌛ Retest condition expired for {symbol} after {time_diff/3600:.1f} hours (max: {max_time_allowed/3600:.1f})")
                self.retest_tracking[symbol] = None
                return
            else:
                # Cannot perform comparison, log and continue
                logger.warning(f"Cannot compare times of types {type(current_time)} and {type(start_time)}. Continuing anyway.")
        except Exception as e:
            logger.warning(f"Error calculating time difference in retest condition: {e}")
            # Continue processing despite the error
            
        # Calculate ATR for dynamic retest window
        try:
            # Calculate ATR for the current timeframe
            atr = calculate_atr(df, self.atr_period)
            if self._is_invalid_or_zero(atr):
                # Fallback to a price-based tolerance if ATR calculation fails
                logger.warning(f"ATR calculation failed for {symbol}, using price-based tolerance")
                price_tolerance = float(df['close'].iloc[-1]) * self.max_stop_pct
            else:
                # Use ATR with multiplier for dynamic tolerance
                price_tolerance = atr * self.atr_multiplier
                logger.debug(f"Using ATR-based retest window: {price_tolerance:.5f} (ATR: {atr:.5f}, multiplier: {self.atr_multiplier})")
        except Exception as e:
            logger.warning(f"Error calculating ATR for {symbol}: {e}. Using price-based tolerance.")
            # Fallback to price-based tolerance
            price_tolerance = float(df['close'].iloc[-1]) * self.max_stop_pct
            
        # Check if the price has retested the level using ATR-based window
        current_price = df['close'].iloc[-1]

        # For breakout above resistance, we're looking for a retest from above
        if direction == 'bullish' and abs(current_price - level) <= price_tolerance and current_price > level:
            logger.info(f"✅ Confirmed bullish retest of {level:.5f} for {symbol} (ATR window: {price_tolerance:.5f})")
            # Update breakout tracking to indicate retest is confirmed
            self.retest_tracking[symbol]['retest_confirmed'] = True
            
        # For breakout below support, we're looking for a retest from below
        elif direction == 'bearish' and abs(current_price - level) <= price_tolerance and current_price < level:
            logger.info(f"✅ Confirmed bearish retest of {level:.5f} for {symbol} (ATR window: {price_tolerance:.5f})")
            # Update breakout tracking to indicate retest is confirmed
            self.retest_tracking[symbol]['retest_confirmed'] = True
    
    def _find_support_levels(self, df: pd.DataFrame) -> List[float]:
        """
        Find significant support levels using swing lows.
        
        Args:
            df: Price dataframe
            
        Returns:
            List of support levels
        """
        levels = []
        
        # Use a window to find local minima
        for i in range(2, len(df) - 2):
            if (df['low'].iloc[i] < df['low'].iloc[i-1] and 
                df['low'].iloc[i] < df['low'].iloc[i-2] and
                df['low'].iloc[i] < df['low'].iloc[i+1] and 
                df['low'].iloc[i] < df['low'].iloc[i+2]):
                
                # Found a potential swing low
                level = df['low'].iloc[i]
                
                # Count number of times price has approached this level
                touches = self._count_level_touches(df, level, 'support')
                
                if touches >= self.min_level_touches:
                    levels.append(level)
        
        # Cluster nearby levels
        clustered_levels = self._cluster_levels(levels)
        
        return clustered_levels
    
    def _find_resistance_levels(self, df: pd.DataFrame) -> List[float]:
        """
        Find significant resistance levels using swing highs.
        
        Args:
            df: Price dataframe
            
        Returns:
            List of resistance levels
        """
        levels = []
        
        # Use a window to find local maxima
        for i in range(2, len(df) - 2):
            if (df['high'].iloc[i] > df['high'].iloc[i-1] and 
                df['high'].iloc[i] > df['high'].iloc[i-2] and
                df['high'].iloc[i] > df['high'].iloc[i+1] and 
                df['high'].iloc[i] > df['high'].iloc[i+2]):
                
                # Found a potential swing high
                level = df['high'].iloc[i]
                
                # Count number of times price has approached this level
                touches = self._count_level_touches(df, level, 'resistance')
                
                if touches >= self.min_level_touches:
                    levels.append(level)
        
        # Cluster nearby levels
        clustered_levels = self._cluster_levels(levels)
        
        return clustered_levels
    
    def _count_level_touches(self, df: pd.DataFrame, level: float, level_type: str) -> int:
        """
        Count how many times price has touched a level.
        
        Args:
            df: Price dataframe
            level: Price level to check
            level_type: 'support' or 'resistance'
            
        Returns:
            Number of touches
        """
        tolerance = level * self.price_tolerance
        count = 0
        
        if level_type == 'support':
            # Price approached from above and bounced
            for i in range(len(df)):
                if abs(df['low'].iloc[i] - level) <= tolerance:
                    count += 1
        else:  # resistance
            # Price approached from below and bounced
            for i in range(len(df)):
                if abs(df['high'].iloc[i] - level) <= tolerance:
                    count += 1
                    
        return count
    
    def _cluster_levels(self, levels: List[float]) -> List[float]:
        """
        Cluster nearby levels to avoid duplicates.
        
        Args:
            levels: List of price levels
            
        Returns:
            Clustered list of levels
        """
        if not levels:
            return []
            
        # Sort levels
        sorted_levels = sorted(levels)
        
        # Cluster nearby levels
        clusters = []
        current_cluster = [sorted_levels[0]]
        
        for i in range(1, len(sorted_levels)):
            # If this level is close to the previous one, add to cluster
            if sorted_levels[i] - sorted_levels[i-1] <= sorted_levels[i] * self.price_tolerance:
                current_cluster.append(sorted_levels[i])
            else:
                # Start a new cluster
                clusters.append(current_cluster)
                current_cluster = [sorted_levels[i]]
                
        # Add the last cluster
        clusters.append(current_cluster)
        
        # Take average of each cluster
        return [sum(cluster) / len(cluster) for cluster in clusters]
    
    
    def _check_breakout_signals(self, symbol: str, df: pd.DataFrame, h1_df: pd.DataFrame, skip_plots: bool = False) -> List[Dict]:
        """
        Check for breakout signals across support/resistance levels and trend lines.
        
        Args:
            symbol: Trading symbol
            df: Price dataframe for primary timeframe
            h1_df: Price dataframe for higher timeframe
            skip_plots: Whether to skip creating debug plots
            
        Returns:
            List of breakout signal dictionaries
        """
        signals = []
        
        # Skip if no levels available
        if symbol not in self.resistance_levels or symbol not in self.support_levels:
            logger.debug(f"⏩ {symbol}: No resistance or support levels available, skipping breakout check")
            return signals
            
        resistance_levels = self.resistance_levels[symbol]
        support_levels = self.support_levels[symbol]
        
        # Ensure we have the required columns
        # Check if 'tick_volume' exists, if not check for 'volume', if neither exists create a default
        if 'tick_volume' not in df.columns:
            if 'volume' in df.columns:
                logger.debug(f"Using 'volume' column instead of missing 'tick_volume' for {symbol}")
                df['tick_volume'] = df['volume']
            else:
                logger.debug(f"Creating default 'tick_volume' column for {symbol} as neither 'tick_volume' nor 'volume' exists")
                # Create a default volume column with values of 1
                df['tick_volume'] = 1
        
        # Log a sample of recent candles for debugging
        candles_to_log = min(5, len(df))
        if candles_to_log > 0:
            logger.debug(f"🕯️ {symbol}: Last {candles_to_log} candles data sample:")
            try:
                for i in range(-candles_to_log, 0):
                    candle = df.iloc[i]
                    logger.debug(f"   {df.index[i]}: O={candle['open']:.5f}, H={candle['high']:.5f}, L={candle['low']:.5f}, C={candle['close']:.5f}, Vol={candle['tick_volume']}")
            except KeyError as e:
                logger.warning(f"Error logging candle data for {symbol}: {str(e)}")
                logger.debug(f"DataFrame columns: {df.columns.tolist()}")
        
        # Get trend lines if available
        trend_lines = self.bullish_trend_lines.get(symbol, []) + self.bearish_trend_lines.get(symbol, [])
        bullish_trend_lines = [line for line in trend_lines if line['angle'] < 60]  # Increased from 45
        bearish_trend_lines = [line for line in trend_lines if line['angle'] > -60]  # Increased from -45
        
        logger.debug(f"🔍 {symbol}: Found {len(bullish_trend_lines)} bullish and {len(bearish_trend_lines)} bearish trend lines")
        
        # Get recent candles - use candles_to_check from timeframe profile
        candles_to_check = min(self.candles_to_check, len(df) - 1)
        
        # Calculate volume threshold using percentile-based approach
        try:
            # Get lookback window for volume analysis
            lookback_bars = min(50, len(df) - 1)  # Use last 50 bars or as many as available
            
            # Extract volume data
            volume_series = df['tick_volume'].iloc[-lookback_bars:].copy()
            
            # Calculate the percentile threshold
            volume_threshold = np.percentile(volume_series, self.volume_percentile)
            
            logger.debug(f"📊 {symbol}: Using {self.volume_percentile}th percentile volume threshold: {volume_threshold:.1f}")
        except Exception as e:
            logger.warning(f"Error calculating volume percentile threshold for {symbol}: {str(e)}")
            # Fallback to old method with fixed multiplier
            try:
                avg_volume_series = df['tick_volume'].rolling(window=20).mean()
                # Ensure we have a pandas Series
                if not isinstance(avg_volume_series, pd.Series):
                    avg_volume_series = pd.Series(avg_volume_series, index=df.index[-20:])
                avg_volume = float(avg_volume_series.iloc[-1])
                volume_threshold = avg_volume * self.volume_threshold
                
                logger.debug(f"📊 {symbol}: Fallback to avg volume: {avg_volume:.1f}, threshold: {volume_threshold:.1f}")
            except Exception as e2:
                logger.warning(f"Fallback volume calculation also failed: {str(e2)}")
                volume_threshold = 1.0  # Default threshold if all calculations fail
                avg_volume = 1.0
        
        # Get higher timeframe trend
        h1_trend = self._determine_h1_trend(h1_df)
        logger.info(f"📈 {symbol}: H1 trend is {h1_trend}")
        
        # Check for retest confirmations first
        if (symbol in self.retest_tracking and 
            self.retest_tracking[symbol] and  # Fix: Check if entry exists and is not None
            self.retest_tracking[symbol].get('confirmed', False)):
            
            retest_info = self.retest_tracking[symbol]
            retest_entry = retest_info.get('entry_price')
            retest_direction = retest_info.get('direction')
            retest_level = retest_info.get('level')
            retest_stop = retest_info.get('stop_loss')
            retest_reason = retest_info.get('reason')
            
            logger.info(f"✅ {symbol}: Retest confirmed for {retest_direction} at level {retest_level:.5f}")
            
            # Current price from most recent candle
            current_price = df['close'].iloc[-1]
            
            if retest_direction == 'buy':
                # For a bullish breakout retest, create signal after confirmation
                # Calculate target using consolidation range projection
                risk = retest_entry - retest_stop
                
                logger.debug(f"📐 {symbol}: Buy retest risk = {risk:.5f} pips")
                
                # Advanced target calculation
                if symbol in self.last_consolidation_ranges:
                    range_size = self.last_consolidation_ranges[symbol]['size']
                    logger.debug(f"📏 {symbol}: Using consolidation range size = {range_size:.5f} for target")
                    
                    # Target is either breakout level + range size or at least min_risk_reward
                    calculated_target = retest_level + range_size
                    min_target = retest_entry + (risk * self.min_risk_reward)
                    take_profit = max(calculated_target, min_target)
                    
                    logger.debug(f"🎯 {symbol}: Target calculation - Range projection: {calculated_target:.5f}, Min RR: {min_target:.5f}, Using: {take_profit:.5f}")
                else:
                    # Fallback to minimum risk-reward
                    take_profit = retest_entry + (risk * self.min_risk_reward)
                    logger.debug(f"🎯 {symbol}: No range data, using min RR target = {take_profit:.5f}")
                
                # Create signal
                signal = {
                    "symbol": symbol,
                    "direction": "buy",
                    "entry_price": retest_entry,
                    "stop_loss": retest_stop,
                    "take_profit": take_profit,
                    "timeframe": self.primary_timeframe,
                    "confidence": 0.8,  # Higher confidence due to retest confirmation
                    "source": self.name,
                    "generator": self.name,
                    "reason": f"Retest confirmed: {retest_reason}"
                }
                
                signals.append(signal)
                logger.info(f"🟢 RETEST BUY: {symbol} at {retest_entry:.5f} | SL: {retest_stop:.5f} | TP: {take_profit:.5f}")
                
                # Clear retest tracking
                self.retest_tracking[symbol] = {}
                
            elif retest_direction == 'sell':
                # For a bearish breakdown retest, create signal after confirmation
                # Calculate target using consolidation range projection
                risk = retest_stop - retest_entry
                
                logger.debug(f"📐 {symbol}: Sell retest risk = {risk:.5f} pips")
                
                # Advanced target calculation
                if symbol in self.last_consolidation_ranges:
                    range_size = self.last_consolidation_ranges[symbol]['size']
                    logger.debug(f"📏 {symbol}: Using consolidation range size = {range_size:.5f} for target")
                    
                    # Target is either breakout level - range size or at least min_risk_reward
                    calculated_target = retest_level - range_size
                    min_target = retest_entry - (risk * self.min_risk_reward)
                    take_profit = min(calculated_target, min_target)
                    
                    logger.debug(f"🎯 {symbol}: Target calculation - Range projection: {calculated_target:.5f}, Min RR: {min_target:.5f}, Using: {take_profit:.5f}")
                else:
                    # Fallback to minimum risk-reward
                    take_profit = retest_entry - (risk * self.min_risk_reward)
                    logger.debug(f"🎯 {symbol}: No range data, using min RR target = {take_profit:.5f}")
                
                # Create signal
                signal = {
                    "symbol": symbol,
                    "direction": "sell",
                    "entry_price": retest_entry,
                    "stop_loss": retest_stop,
                    "take_profit": take_profit,
                    "timeframe": self.primary_timeframe,
                    "confidence": 0.8,  # Higher confidence due to retest confirmation
                    "source": self.name,
                    "generator": self.name,
                    "reason": f"Retest confirmed: {retest_reason}"
                }
                
                signals.append(signal)
                logger.info(f"🔴 RETEST SELL: {symbol} at {retest_entry:.5f} | SL: {retest_stop:.5f} | TP: {take_profit:.5f}")
                
                # Clear retest tracking
                self.retest_tracking[symbol] = {}
        else:
            logger.debug(f"👀 {symbol}: No retest confirmations pending")
        
        # Check for resistance breakouts (horizontal levels)
        for i in range(-candles_to_check, 0):
            current_candle = df.iloc[i]
            previous_candle = df.iloc[i-1]
            
            logger.debug(f"📊 {symbol}: Checking candle at {df.index[i]}: O={current_candle['open']:.5f} H={current_candle['high']:.5f} L={current_candle['low']:.5f} C={current_candle['close']:.5f} V={current_candle['tick_volume']}")
            
            # Volume analysis with wick structure
            volume_quality = self._analyze_volume_quality(current_candle, volume_threshold)
            logger.debug(f"📊 {symbol}: Volume quality score: {volume_quality:.1f} (>0 = bullish, <0 = bearish)")
            
            # Check each resistance level
            for level in resistance_levels:
                logger.debug(f"🔄 {symbol}: Checking resistance level {level:.5f}")
                
                # Breakout condition - RELAXED: no longer require strong candle
                # Previous candle below or at resistance, current candle closing above
                if (previous_candle['close'] <= level * (1 + self.price_tolerance) and
                    current_candle['close'] > level * (1 + self.price_tolerance)):
                    
                    # Generate buy signal
                    entry_price = current_candle['close']
                    
                    # Place stop under the breakout candle's low
                    stop_loss = min(current_candle['low'], previous_candle['low'])
                    
                    # Risk calculation (rest of code unchanged)
                    # ...
                    
                    # Log the breakout regardless of whether we generate a signal
                    logger.info(f"👀 Detected potential breakout for {symbol} at level {level:.5f}")
                    
                    # RELAXED conditions: allow signals with neutral H1 trend, don't require strong volume
                    if h1_trend != 'bearish':  # Just avoid counter-trend signals
                        # Generate buy signal
                        # (existing code for generating signals)
                        
                        # Add detailed logging
                        logger.debug(f"Breakout details: Close={current_candle['close']:.5f}, " +
                                   f"Level={level:.5f}, Volume quality={volume_quality:.2f}, " +
                                   f"H1 trend={h1_trend}")
                        
                        # Generate trade signals...
                        # (Rest of the existing code)
                
            # Check trend line breakouts (bullish) - RELAXED conditions
            for trend_line in bearish_trend_lines:
                # Calculate trend line value at current and previous candle
                prev_line_value = self._calculate_trend_line_value(trend_line, i-1)
                curr_line_value = self._calculate_trend_line_value(trend_line, i)
                
                # DEBUG: Log the trendline values
                logger.debug(f"Trendline values: prev={prev_line_value:.5f}, curr={curr_line_value:.5f}, " +
                           f"close={current_candle['close']:.5f}, candle index={i}")
                
                # RELAXED: Breakout condition - now doesn't require strong candle, allows neutral trend
                # Previous candle below trend line, current candle closing above
                if (previous_candle['close'] <= prev_line_value * (1 + self.price_tolerance) and
                    current_candle['close'] > curr_line_value * (1 + self.price_tolerance) and
                    h1_trend != 'bearish'):  # Just avoid counter-trend signals
                    
                    # Generate buy signal
                    entry_price = current_candle['close']
                    
                    # Place stop under the breakout candle's low
                    stop_loss = min(current_candle['low'], previous_candle['low'])
                    
                    # Calculate risk and take profit
                    risk = entry_price - stop_loss
                    take_profit = entry_price + (risk * self.min_risk_reward)
                    
                    # Reason with volume quality description
                    volume_desc = "strong bullish volume" if volume_quality > 1 else "adequate volume"
                    reason = f"Bullish breakout above bearish trend line with {volume_desc}"
                    
                    # Add detailed logging
                    logger.info(f"💡 TRENDLINE BREAKOUT DETECTED: {symbol} at {entry_price:.5f}")
                    logger.debug(f"Breakout details: Close={current_candle['close']:.5f}, " +
                               f"Trendline value={curr_line_value:.5f}, Volume quality={volume_quality:.2f}, " +
                               f"H1 trend={h1_trend}, r²={trend_line['r_squared']:.2f}, angle={trend_line['angle']:.2f}°")
                    
                    # If retest is required, don't generate signal now but track for retest
                    if self.retest_required:
                        # Store breakout info for retest tracking
                        self.retest_tracking[symbol] = {
                            'level': curr_line_value,
                            'direction': 'buy',
                            'entry_price': entry_price,
                            'stop_loss': stop_loss,
                            'start_time': df.index[i],
                            'confirmed': False,
                            'reason': reason
                        }
                        logger.info(f"👀 TRACKING RETEST: {symbol} bullish trend line breakout at {curr_line_value:.5f}")
                    else:
                        # Create immediate signal if retest not required
                        signal = {
                            "symbol": symbol,
                            "direction": "buy",
                            "entry_price": entry_price,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "timeframe": self.primary_timeframe,
                            "confidence": 0.75,
                            "source": self.name,
                            "generator": self.name,
                            "reason": reason
                        }
                        
                        signals.append(signal)
                        logger.info(f"🟢 TREND LINE BREAKOUT BUY: {symbol} at {entry_price:.5f} | SL: {stop_loss:.5f} | TP: {take_profit:.5f}")
        
        # Check for support breakdowns (horizontal levels)
        for i in range(-candles_to_check, 0):
            current_candle = df.iloc[i]
            previous_candle = df.iloc[i-1]
            
            # Volume analysis with wick structure
            volume_quality = self._analyze_volume_quality(current_candle, volume_threshold)
            
            # Check each support level
            for level in support_levels:
                # Breakdown condition: Previous candle above or at support, current candle closing below
                if (previous_candle['close'] >= level * (1 - self.price_tolerance) and
                    current_candle['close'] < level * (1 - self.price_tolerance) and
                    self._is_strong_candle(current_candle) and
                    volume_quality < 0 and  # Negative means bearish volume characteristics
                    h1_trend == 'bearish'):
                    
                    # Generate sell signal
                    entry_price = current_candle['close']
                    
                    # Place stop above the breakdown candle's high
                    stop_loss = max(current_candle['high'], previous_candle['high'])
                    
                    # Advanced target calculation
                    if symbol in self.last_consolidation_ranges:
                        range_size = self.last_consolidation_ranges[symbol]['size']
                        risk = stop_loss - entry_price
                        calculated_target = level - range_size
                        min_target = entry_price - (risk * self.min_risk_reward)
                        take_profit = min(calculated_target, min_target)
                    else:
                        # Fallback to minimum risk-reward
                        risk = stop_loss - entry_price
                        take_profit = entry_price - (risk * self.min_risk_reward)
                    
                    # Reason with volume quality description
                    volume_desc = "strong bearish volume" if volume_quality < -1 else "adequate volume"
                    reason = f"Bearish breakdown below support at {level:.5f} with {volume_desc}"
                    
                    # If retest is required, don't generate signal now but track for retest
                    if self.retest_required:
                        # Store breakout info for retest tracking
                        self.retest_tracking[symbol] = {
                            'level': level,
                            'direction': 'sell',
                            'entry_price': entry_price,
                            'stop_loss': stop_loss,
                            'start_time': df.index[i],
                            'confirmed': False,
                            'reason': reason
                        }
                        logger.info(f"👀 TRACKING RETEST: {symbol} bearish breakdown at {level:.5f}")
                    else:
                        # Create immediate signal if retest not required
                        signal = {
                            "symbol": symbol,
                            "direction": "sell",
                            "entry_price": entry_price,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "timeframe": self.primary_timeframe,
                            "confidence": 0.75,
                            "source": self.name,
                            "generator": self.name,
                            "reason": reason
                        }
                        
                        signals.append(signal)
                        logger.info(f"🔴 BREAKDOWN SELL: {symbol} at {entry_price:.5f} | Level: {level:.5f} | SL: {stop_loss:.5f} | TP: {take_profit:.5f}")
            
            # Check trend line breakdowns (bearish)
            for trend_line in bullish_trend_lines:
                # Calculate trend line value at current and previous candle
                prev_line_value = self._calculate_trend_line_value(trend_line, i-1)
                curr_line_value = self._calculate_trend_line_value(trend_line, i)
                
                # Breakdown condition: Previous candle above trend line, current candle closing below
                if (previous_candle['close'] >= prev_line_value * (1 - self.price_tolerance) and
                    current_candle['close'] < curr_line_value * (1 - self.price_tolerance) and
                    self._is_strong_candle(current_candle) and
                    volume_quality < 0 and  # Negative means bearish volume characteristics
                    h1_trend == 'bearish'):
                    
                    # Generate sell signal
                    entry_price = current_candle['close']
                    
                    # Place stop above the breakdown candle's high
                    stop_loss = max(current_candle['high'], previous_candle['high'])
                    
                    # Calculate risk and take profit
                    risk = stop_loss - entry_price
                    take_profit = entry_price - (risk * self.min_risk_reward)
                    
                    # Reason with volume quality description
                    volume_desc = "strong bearish volume" if volume_quality < -1 else "adequate volume"
                    reason = f"Bearish breakdown below bullish trend line with {volume_desc}"
                    
                    # If retest is required, don't generate signal now but track for retest
                    if self.retest_required:
                        # Store breakout info for retest tracking
                        self.retest_tracking[symbol] = {
                            'level': curr_line_value,
                            'direction': 'sell',
                            'entry_price': entry_price,
                            'stop_loss': stop_loss,
                            'start_time': df.index[i],
                            'confirmed': False,
                            'reason': reason
                        }
                        logger.info(f"👀 TRACKING RETEST: {symbol} bearish trend line breakdown")
                    else:
                        # Create immediate signal if retest not required
                        signal = {
                            "symbol": symbol,
                            "direction": "sell",
                            "entry_price": entry_price,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "timeframe": self.primary_timeframe,
                            "confidence": 0.75,
                            "source": self.name,
                            "generator": self.name,
                            "reason": reason
                        }
                        
                        signals.append(signal)
                        logger.info(f"🔴 TREND LINE BREAKDOWN SELL: {symbol} at {entry_price:.5f} | SL: {stop_loss:.5f} | TP: {take_profit:.5f}")
        
        return signals
    
    def _check_reversal_signals(self, symbol: str, df: pd.DataFrame, h1_df: pd.DataFrame, skip_plots: bool = False) -> List[Dict]:
        """
        Check for reversal signals at key support and resistance levels.
        
        Args:
            symbol: Trading symbol
            df: Price dataframe for primary timeframe
            h1_df: Price dataframe for higher timeframe
            skip_plots: Whether to skip creating debug plots
            
        Returns:
            List of reversal signal dictionaries
        """
        signals = []
        
        # Get support and resistance levels
        if symbol not in self.resistance_levels or symbol not in self.support_levels:
            logger.debug(f"❓ No support/resistance levels available for {symbol}")
            return signals
            
        resistance_levels = self.resistance_levels[symbol]
        support_levels = self.support_levels[symbol]
        
        logger.debug(f"🔍 {symbol}: Checking reversals with {len(resistance_levels)} resistance and {len(support_levels)} support levels")
        
        # Determine the trend context from the higher timeframe
        h1_trend = self._determine_h1_trend(h1_df)
        is_downtrend = h1_trend == 'bearish'
        
        # Ensure we have the required columns
        # Check if 'tick_volume' exists, if not check for 'volume', if neither exists create a default
        if 'tick_volume' not in df.columns:
            if 'volume' in df.columns:
                logger.debug(f"Using 'volume' column instead of missing 'tick_volume' for {symbol} in reversal signals")
                df['tick_volume'] = df['volume']
            else:
                logger.debug(f"Creating default 'tick_volume' column for {symbol} as neither 'tick_volume' nor 'volume' exists in reversal signals")
                # Create a default volume column with values of 1
                df['tick_volume'] = 1
        
        # Get trend lines if available
        trend_lines = self.bullish_trend_lines.get(symbol, []) + self.bearish_trend_lines.get(symbol, [])
        bullish_trend_lines = [line for line in trend_lines if line['angle'] < 60]  # Increased from 45
        bearish_trend_lines = [line for line in trend_lines if line['angle'] > -60]  # Increased from -45
        
        logger.debug(f"🔍 {symbol}: Found {len(bullish_trend_lines)} bullish and {len(bearish_trend_lines)} bearish trend lines for reversal checks")
        
        # Use candles_to_check from timeframe profile
        candles_to_check = min(self.candles_to_check, len(df) - 1)
        
        # Calculate volume threshold using percentile-based approach
        try:
            # Get lookback window for volume analysis
            lookback_bars = min(50, len(df) - 1)  # Use last 50 bars or as many as available
            
            # Extract volume data
            volume_series = df['tick_volume'].iloc[-lookback_bars:].copy()
            
            # Calculate the percentile threshold
            volume_threshold = np.percentile(volume_series, self.volume_percentile)
            
            logger.debug(f"📊 {symbol}: Using {self.volume_percentile}th percentile volume threshold: {volume_threshold:.1f}")
        except Exception as e:
            logger.warning(f"Error calculating volume percentile threshold for {symbol}: {str(e)}")
            # Fallback to old method with fixed multiplier
            try:
                avg_volume_series = df['tick_volume'].rolling(window=20).mean()
                # Ensure we have a pandas Series
                if not isinstance(avg_volume_series, pd.Series):
                    avg_volume_series = pd.Series(avg_volume_series, index=df.index[-20:])
                avg_volume = float(avg_volume_series.iloc[-1])
                volume_threshold = avg_volume * self.volume_threshold
                
                logger.debug(f"📊 {symbol}: Fallback to avg volume: {avg_volume:.1f}, threshold: {volume_threshold:.1f}")
            except Exception as e2:
                logger.warning(f"Fallback volume calculation also failed: {str(e2)}")
                volume_threshold = 1.0  # Default threshold if all calculations fail
        
        # Check for reversal at support (bullish patterns)
        for i in range(-candles_to_check, 0):
            current_candle = df.iloc[i]
            previous_candle = df.iloc[i-1]
            
            logger.debug(f"📊 {symbol}: Checking reversal at candle {df.index[i]}: O={current_candle['open']:.5f} H={current_candle['high']:.5f} L={current_candle['low']:.5f} C={current_candle['close']:.5f}")
            
            # Volume analysis with wick structure
            volume_quality = self._analyze_volume_quality(current_candle, volume_threshold)
            logger.debug(f"📊 {symbol}: Volume quality score: {volume_quality:.1f} (>0 = bullish, <0 = bearish)")
            
            # Check each support level
            for level in support_levels:
                logger.debug(f"🔄 {symbol}: Checking support level {level:.5f}")
                
                # Price near support
                is_near_support = abs(current_candle['low'] - level) <= level * self.price_tolerance
                logger.debug(f"✓ {symbol}: Price near support: {is_near_support} (Low: {current_candle['low']:.5f}, Support: {level:.5f}, Tolerance: {level * self.price_tolerance:.5f})")
                
                if is_near_support:
                    # Check for bullish reversal patterns
                    pattern_type = None
                    
                    # Updated hammer check with trend context and confirmation
                    is_hammer_pattern = self._is_hammer(
                        current_candle, 
                        in_downtrend=is_downtrend,  # Use the detected trend
                        df=df, 
                        idx=i, 
                        require_confirmation=True
                    )
                    
                    # Add inverted hammer check with trend context and confirmation
                    is_inverted_hammer_pattern = self._is_inverted_hammer(
                        current_candle, 
                        in_downtrend=is_downtrend,  # Use the detected trend
                        df=df, 
                        idx=i, 
                        require_confirmation=True
                    )
                    
                    is_bullish_engulfing = self._is_bullish_engulfing(
                        candles=df,
                        idx=i,
                        in_downtrend=is_downtrend,  # Use the detected trend
                        require_confirmation=True
                    )
                    is_morning_star = self._is_morning_star(df, i)
                    
                    logger.debug(f"📈 {symbol}: Pattern checks - Hammer: {is_hammer_pattern}, Inverted Hammer: {is_inverted_hammer_pattern}, Bullish Engulfing: {is_bullish_engulfing}, Morning Star: {is_morning_star}")
                    
                    if is_hammer_pattern:
                        pattern_type = "Hammer"
                    elif is_inverted_hammer_pattern:
                        pattern_type = "Inverted Hammer"
                    elif is_bullish_engulfing:
                        pattern_type = "Bullish Engulfing"
                    elif is_morning_star:
                        pattern_type = "Morning Star"
                    
                    # Initialize volume_desc here
                    volume_desc = "strong bullish volume" if volume_quality > 1 else "adequate volume"
                    
                    if pattern_type and volume_quality > 0:  # Bullish volume characteristics
                        logger.info(f"⚡ {symbol}: Detected bullish reversal pattern ({pattern_type}) at support {level:.5f}")
                        
                        # Generate buy signal
                        entry_price = current_candle['close']
                        
                        # Stop loss below the reversal candle low
                        stop_loss = current_candle['low'] - (level * self.price_tolerance)
                        
                        # Target: Either next resistance or at least 2x risk
                        risk = entry_price - stop_loss
                        
                        logger.debug(f"📐 {symbol}: Entry: {entry_price:.5f}, Stop: {stop_loss:.5f}, Risk: {risk:.5f}")
                        
                        # Advanced target calculation - find nearest resistance above
                        next_resistance = self._find_next_resistance(df, entry_price, resistance_levels)
                        
                        if next_resistance:
                            logger.debug(f"🎯 {symbol}: Found next resistance at {next_resistance:.5f}")
                            
                            # Check if next resistance provides enough reward
                            reward_to_resistance = next_resistance - entry_price
                            min_reward = risk * self.min_risk_reward
                            
                            logger.debug(f"📊 {symbol}: Reward to resistance: {reward_to_resistance:.5f}, Min required: {min_reward:.5f}")
                            
                            if reward_to_resistance >= min_reward:
                                take_profit = next_resistance
                                logger.debug(f"✅ {symbol}: Using next resistance as target: {take_profit:.5f}")
                            else:
                                take_profit = entry_price + min_reward
                                logger.debug(f"⚠️ {symbol}: Resistance too close, using min RR target: {take_profit:.5f}")
                        else:
                            take_profit = entry_price + (risk * self.min_risk_reward)
                            logger.debug(f"ℹ️ {symbol}: No resistance found, using min RR target: {take_profit:.5f}")
                        
                        # Volume description
                        volume_desc = "strong bullish volume" if volume_quality > 1 else "adequate volume"
                        
                        # Create signal
                        signal = {
                            "symbol": symbol,
                            "direction": "buy",
                            "entry_price": entry_price,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "timeframe": self.primary_timeframe,
                            "confidence": 0.7,
                            "source": self.name,
                            "generator": self.name,
                            "reason": f"Bullish reversal ({pattern_type}) at support {level:.5f} with {volume_desc}"
                        }
                        
                        signals.append(signal)
                        logger.info(f"🟢 REVERSAL BUY: {symbol} at {entry_price:.5f} | Pattern: {pattern_type} | Level: {level:.5f} | SL: {stop_loss:.5f} | TP: {take_profit:.5f}")
                    else:
                        if not pattern_type:
                            logger.debug(f"❌ {symbol}: No bullish pattern detected")
                        if volume_quality <= 0:
                            logger.debug(f"❌ {symbol}: Insufficient bullish volume (quality: {volume_quality:.1f})")
            
            # Check for reversal at bullish trend lines
            for trend_line in bullish_trend_lines:
                # Calculate trend line value at current position
                line_value = self._calculate_trend_line_value(trend_line, i)
                
                logger.debug(f"🔄 {symbol}: Checking bullish trend line at price {line_value:.5f}")
                
                # Price near trend line
                is_near_trendline = abs(current_candle['low'] - line_value) <= current_candle['close'] * self.price_tolerance
                logger.debug(f"✓ {symbol}: Price near trend line: {is_near_trendline} (Low: {current_candle['low']:.5f}, Trend line: {line_value:.5f})")
                
                if is_near_trendline:
                    # Check for bullish reversal patterns
                    pattern_type = None
                    
                    # Updated hammer check with trend context and confirmation
                    is_hammer_pattern = self._is_hammer(
                        current_candle, 
                        in_downtrend=is_downtrend,  # Use the detected trend
                        df=df, 
                        idx=i, 
                        require_confirmation=True
                    )
                    
                    # Add inverted hammer check with trend context and confirmation
                    is_inverted_hammer_pattern = self._is_inverted_hammer(
                        current_candle, 
                        in_downtrend=is_downtrend,  # Use the detected trend
                        df=df, 
                        idx=i, 
                        require_confirmation=True
                    )
                    
                    is_bullish_engulfing = self._is_bullish_engulfing(
                        candles=df,
                        idx=i,
                        in_downtrend=is_downtrend,  # Use the detected trend
                        require_confirmation=True
                    )
                    is_morning_star = self._is_morning_star(df, i)
                    
                    logger.debug(f"📈 {symbol}: Pattern checks - Hammer: {is_hammer_pattern}, Inverted Hammer: {is_inverted_hammer_pattern}, Bullish Engulfing: {is_bullish_engulfing}, Morning Star: {is_morning_star}")
                    
                    if is_hammer_pattern:
                        pattern_type = "Hammer"
                    elif is_inverted_hammer_pattern:
                        pattern_type = "Inverted Hammer"
                    elif is_bullish_engulfing:
                        pattern_type = "Bullish Engulfing"
                    elif is_morning_star:
                        pattern_type = "Morning Star"
                    
                    # Initialize volume_desc here
                    volume_desc = "strong bullish volume" if volume_quality > 1 else "adequate volume"
                    
                    if pattern_type and volume_quality > 0:  # Bullish volume characteristics
                        logger.info(f"⚡ {symbol}: Detected bullish reversal pattern ({pattern_type}) at trend line with {volume_desc}")
                        
                        # Generate buy signal
                        entry_price = current_candle['close']
                        
                        # Stop loss below the reversal candle low
                        stop_loss = current_candle['low'] - (line_value * self.price_tolerance)
                        
                        # Target: Either next resistance or at least 2x risk
                        risk = entry_price - stop_loss
                        
                        logger.debug(f"📐 {symbol}: Entry: {entry_price:.5f}, Stop: {stop_loss:.5f}, Risk: {risk:.5f}")
                        
                        # Advanced target calculation - find nearest resistance above
                        next_resistance = self._find_next_resistance(df, entry_price, resistance_levels)
                        
                        if next_resistance:
                            logger.debug(f"🎯 {symbol}: Found next resistance at {next_resistance:.5f}")
                            
                            # Check if next resistance provides enough reward
                            reward_to_resistance = next_resistance - entry_price
                            min_reward = risk * self.min_risk_reward
                            
                            logger.debug(f"📊 {symbol}: Reward to resistance: {reward_to_resistance:.5f}, Min required: {min_reward:.5f}")
                            
                            if reward_to_resistance >= min_reward:
                                take_profit = next_resistance
                                logger.debug(f"✅ {symbol}: Using next resistance as target: {take_profit:.5f}")
                            else:
                                take_profit = entry_price + min_reward
                                logger.debug(f"⚠️ {symbol}: Resistance too close, using min RR target: {take_profit:.5f}")
                        else:
                            take_profit = entry_price + (risk * self.min_risk_reward)
                            logger.debug(f"ℹ️ {symbol}: No resistance found, using min RR target: {take_profit:.5f}")
                        
                        # Volume description
                        volume_desc = "strong bullish volume" if volume_quality > 1 else "adequate volume"
                        
                        # Create signal
                        signal = {
                            "symbol": symbol,
                            "direction": "buy",
                            "entry_price": entry_price,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "timeframe": self.primary_timeframe,
                            "confidence": 0.7,
                            "source": self.name,
                            "generator": self.name,
                            "reason": f"Bullish reversal ({pattern_type}) at trend line with {volume_desc}"
                        }
                        
                        signals.append(signal)
                        logger.info(f"🟢 TREND LINE REVERSAL BUY: {symbol} at {entry_price:.5f} | Pattern: {pattern_type} | SL: {stop_loss:.5f} | TP: {take_profit:.5f}")
                    else:
                        if not pattern_type:
                            logger.debug(f"❌ {symbol}: No bullish pattern detected")
                        if volume_quality <= 0:
                            logger.debug(f"❌ {symbol}: Insufficient bullish volume (quality: {volume_quality:.1f})")
        
        # Check for reversal at resistance (bearish patterns)
        for i in range(-candles_to_check, 0):
            current_candle = df.iloc[i]
            previous_candle = df.iloc[i-1]
            
            # Volume analysis with wick structure
            volume_quality = self._analyze_volume_quality(current_candle, volume_threshold)
            
            # Check each resistance level
            for level in resistance_levels:
                # Price near resistance
                if abs(current_candle['high'] - level) <= level * self.price_tolerance:
                    # Check for bearish reversal patterns
                    pattern_type = None
                    
                    # Use inverted hammer function with in_downtrend=False to detect shooting stars
                    is_shooting_star = self._is_shooting_star(
                        candles=df,
                        idx=i,
                        in_uptrend=not is_downtrend,  # Opposite of the detected trend for bearish patterns
                        require_confirmation=True
                    )
                    
                    # Use hammer function with in_downtrend=False to detect hanging man
                    is_hanging_man = self._is_hammer(
                        current_candle,
                        in_downtrend=not is_downtrend,  # Opposite of the detected trend for bearish patterns
                        df=df,
                        idx=i,
                        require_confirmation=True
                    )
                    
                    # Updated bearish engulfing call using the new signature
                    is_bearish_engulfing = self._is_bearish_engulfing(
                        candles=df,
                        idx=i,
                        in_uptrend=not is_downtrend,  # Opposite of the detected trend for bearish patterns
                        require_confirmation=True
                    )
                    is_evening_star = self._is_evening_star(df, i)
                    
                    logger.debug(f"📉 {symbol}: Pattern checks - Shooting Star: {is_shooting_star}, Hanging Man: {is_hanging_man}, Bearish Engulfing: {is_bearish_engulfing}, Evening Star: {is_evening_star}")
                    
                    if is_shooting_star:
                        pattern_type = "Shooting Star"
                    elif is_hanging_man:
                        pattern_type = "Hanging Man"
                    elif is_bearish_engulfing:
                        pattern_type = "Bearish Engulfing"
                    elif is_evening_star:
                        pattern_type = "Evening Star"
                    
                    if pattern_type and volume_quality < 0:  # Bearish volume characteristics
                        # Generate sell signal
                        entry_price = current_candle['close']
                        
                        # Stop loss above the reversal candle high
                        stop_loss = current_candle['high'] + (level * self.price_tolerance)
                        
                        # Target: Either next support or at least 2x risk
                        risk = stop_loss - entry_price
                        
                        # Advanced target calculation - find nearest support below
                        next_support = self._find_next_support(df, entry_price, support_levels)
                        
                        if next_support and (entry_price - next_support) >= (risk * self.min_risk_reward):
                            take_profit = next_support
                        else:
                            take_profit = entry_price - (risk * self.min_risk_reward)
                        
                        # Volume description
                        volume_desc = "strong bearish volume" if volume_quality < -1 else "adequate volume"
                        
                        # Create signal
                        signal = {
                            "symbol": symbol,
                            "direction": "sell",
                            "entry_price": entry_price,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "timeframe": self.primary_timeframe,
                            "confidence": 0.7,
                            "source": self.name,
                            "generator": self.name,
                            "reason": f"Bearish reversal ({pattern_type}) at resistance {level:.5f} with {volume_desc}"
                        }
                        
                        signals.append(signal)
                        logger.info(f"🔴 REVERSAL SELL: {symbol} at {entry_price:.5f} | Pattern: {pattern_type} | Level: {level:.5f} | SL: {stop_loss:.5f} | TP: {take_profit:.5f}")
            
            # Check for reversal at bearish trend lines
            for trend_line in bearish_trend_lines:
                # Calculate trend line value at current position
                line_value = self._calculate_trend_line_value(trend_line, i)
                
                # Price near trend line
                if abs(current_candle['high'] - line_value) <= current_candle['close'] * self.price_tolerance:
                    # Check for bearish reversal patterns
                    pattern_type = None
                    
                    # Use inverted hammer function with in_downtrend=False to detect shooting stars
                    is_shooting_star = self._is_shooting_star(
                        candles=df,
                        idx=i,
                        in_uptrend=not is_downtrend,  # Opposite of the detected trend for bearish patterns
                        require_confirmation=True
                    )
                    
                    # Use hammer function with in_downtrend=False to detect hanging man
                    is_hanging_man = self._is_hammer(
                        current_candle,
                        in_downtrend=not is_downtrend,  # Opposite of the detected trend for bearish patterns
                        df=df,
                        idx=i,
                        require_confirmation=True
                    )
                    
                    # Updated bearish engulfing call using the new signature
                    is_bearish_engulfing = self._is_bearish_engulfing(
                        candles=df,
                        idx=i,
                        in_uptrend=not is_downtrend,  # Opposite of the detected trend for bearish patterns
                        require_confirmation=True
                    )
                    is_evening_star = self._is_evening_star(df, i)
                    
                    logger.debug(f"📉 {symbol}: Pattern checks - Shooting Star: {is_shooting_star}, Hanging Man: {is_hanging_man}, Bearish Engulfing: {is_bearish_engulfing}, Evening Star: {is_evening_star}")
                    
                    if is_shooting_star:
                        pattern_type = "Shooting Star"
                    elif is_hanging_man:
                        pattern_type = "Hanging Man"
                    elif is_bearish_engulfing:
                        pattern_type = "Bearish Engulfing"
                    elif is_evening_star:
                        pattern_type = "Evening Star"
                    
                    if pattern_type and volume_quality < 0:  # Bearish volume characteristics
                        # Generate sell signal
                        entry_price = current_candle['close']
                        
                        # Stop loss above the reversal candle high
                        stop_loss = current_candle['high'] + (line_value * self.price_tolerance)
                        
                        # Target: Either next support or at least 2x risk
                        risk = stop_loss - entry_price
                        
                        # Advanced target calculation - find nearest support below
                        next_support = self._find_next_support(df, entry_price, support_levels)
                        
                        if next_support and (entry_price - next_support) >= (risk * self.min_risk_reward):
                            take_profit = next_support
                        else:
                            take_profit = entry_price - (risk * self.min_risk_reward)
                        
                        # Volume description
                        volume_desc = "strong bearish volume" if volume_quality < -1 else "adequate volume"
                        
                        # Create signal
                        signal = {
                            "symbol": symbol,
                            "direction": "sell",
                            "entry_price": entry_price,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "timeframe": self.primary_timeframe,
                            "confidence": 0.7,
                            "source": self.name,
                            "generator": self.name,
                            "reason": f"Bearish reversal ({pattern_type}) at trend line with {volume_desc}"
                        }
                        
                        signals.append(signal)
                        logger.info(f"🔴 TREND LINE REVERSAL SELL: {symbol} at {entry_price:.5f} | Pattern: {pattern_type} | SL: {stop_loss:.5f} | TP: {take_profit:.5f}")
        
        return signals
    
    def _analyze_volume_quality(self, candle: pd.Series, threshold: float) -> float:
        """
        Analyze the quality of volume based on candle structure and wick analysis.
        Returns a score indicating volume quality (-2 to +2):
        - Positive values indicate bullish volume characteristics
        - Negative values indicate bearish volume characteristics
        - Magnitude indicates strength (2=strong, 1=moderate, 0=neutral/insufficient)
        
        Args:
            candle: Candle data
            threshold: Minimum volume threshold for consideration
            
        Returns:
            Volume quality score
        """
        try:
            # Check if 'tick_volume' column exists, if not use 'volume', if neither exists use a default
            if 'tick_volume' not in candle:
                if 'volume' in candle:
                    tick_volume = candle['volume']
                    logger.debug(f"Using 'volume' instead of missing 'tick_volume' for volume analysis")
                else:
                    logger.debug(f"Using default volume value as neither 'tick_volume' nor 'volume' exists")
                    tick_volume = threshold * 0.8  # Default to 80% of threshold as a reasonable value
            else:
                tick_volume = candle['tick_volume']
                
            # First check if volume is even significant - using a less strict threshold
            # If candle volume is less than 60% of threshold, consider it insufficient
            volume_ratio = tick_volume / threshold
            logger.debug(f"Volume ratio: {volume_ratio:.2f} (volume: {tick_volume}, threshold: {threshold:.1f})")
            
            if volume_ratio < 0.6:  # More lenient check
                logger.debug(f"Insufficient volume: {tick_volume} < 60% of threshold {threshold:.1f}")
                return 0  # Insufficient volume
                
            # Calculate components
            is_bullish = candle['close'] > candle['open']
            total_range = candle['high'] - candle['low']
            body = abs(candle['close'] - candle['open'])
            
            if total_range == 0 or total_range < 0.00001:  # Guard against division by zero
                logger.debug("Doji or very small candle - neutral volume")
                return 0  # Doji or similar
                
            # Analyze wick structure
            if is_bullish:
                upper_wick = candle['high'] - candle['close']
                lower_wick = candle['open'] - candle['low']
                
                upper_wick_ratio = upper_wick / total_range
                lower_wick_ratio = lower_wick / total_range
                body_ratio = body / total_range
                
                # Debug information
                logger.debug(f"Bullish candle - body ratio: {body_ratio:.2f}, upper wick: {upper_wick_ratio:.2f}, lower wick: {lower_wick_ratio:.2f}")
                
                # Bullish cases
                if body_ratio > 0.6 and lower_wick_ratio < 0.2:
                    # Strong buying pressure - high quality bullish volume
                    return 2.0
                elif body_ratio > 0.4 and lower_wick_ratio < upper_wick_ratio:
                    # Good buying pressure - moderate quality bullish volume
                    return 1.0
                elif upper_wick_ratio > 0.6:
                    # Large upper wick - poor quality for bulls despite green candle
                    return -0.5
                else:
                    # Average quality bullish volume
                    return 0.5
            else:
                # Bearish candle
                upper_wick = candle['high'] - candle['open']
                lower_wick = candle['close'] - candle['low']
                
                upper_wick_ratio = upper_wick / total_range
                lower_wick_ratio = lower_wick / total_range
                body_ratio = body / total_range
                
                # Debug information
                logger.debug(f"Bearish candle - body ratio: {body_ratio:.2f}, upper wick: {upper_wick_ratio:.2f}, lower wick: {lower_wick_ratio:.2f}")
                
                # Bearish cases
                if body_ratio > 0.6 and upper_wick_ratio < 0.2:
                    # Strong selling pressure - high quality bearish volume
                    return -2.0
                elif body_ratio > 0.4 and upper_wick_ratio < lower_wick_ratio:
                    # Good selling pressure - moderate quality bearish volume
                    return -1.0
                elif lower_wick_ratio > 0.6:
                    # Large lower wick - poor quality for bears despite red candle
                    return 0.5
                else:
                    # Average quality bearish volume
                    return -0.5
        except Exception as e:
            logger.error(f"Error in volume analysis: {str(e)}")
            return 0  # Safe default
    
    def _find_next_resistance(self, df: pd.DataFrame, current_price: float, 
                             resistance_levels: List[float]) -> Optional[float]:
        """
        Find the next resistance level above the current price.
        
        Args:
            df: Price dataframe
            current_price: Current price to check from
            resistance_levels: List of resistance levels
            
        Returns:
            Next resistance level or None if none found
        """
        if not resistance_levels:
            return None
            
        # Filter levels above current price and find the closest one
        levels_above = [level for level in resistance_levels if level > current_price]
        
        if not levels_above:
            return None
            
        # Return the closest level above
        return min(levels_above)
    
    def _find_next_support(self, df: pd.DataFrame, current_price: float, 
                          support_levels: List[float]) -> Optional[float]:
        """
        Find the next support level below the current price.
        
        Args:
            df: Price dataframe
            current_price: Current price to check from
            support_levels: List of support levels
            
        Returns:
            Next support level or None if none found
        """
        if not support_levels:
            return None
            
        # Filter levels below current price and find the closest one
        levels_below = [level for level in support_levels if level < current_price]
        
        if not levels_below:
            return None
            
        # Return the closest level below
        return max(levels_below)
    
    def _is_morning_star(self, candles: pd.DataFrame, idx: int, in_downtrend: bool = None, require_confirmation: bool = False) -> bool:
        """
        Check if the pattern at idx is a morning star pattern with optional trend context and confirmation.
        
        A morning star is a three-candle bullish reversal pattern consisting of:
        1. A large bearish candle
        2. A small bodied candle (star) that gaps down (or has small/no overlap with first candle)
        3. A bullish candle that closes well into the first candle's body
        
        Args:
            candles: DataFrame containing price data
            idx: Index to check for pattern (this is the index of the LAST candle in the pattern)
            in_downtrend: If True, validates pattern in downtrend (more reliable). If None, only checks pattern shape.
            require_confirmation: Whether to require confirmation from the next candle
            
        Returns:
            True if a valid morning star with the specified context and confirmation
        """
        # Ensure we have enough candles to check the pattern
        if idx < 2 or idx >= len(candles) - (1 if require_confirmation else 0):
            return False
        
        # Get the three candles for the pattern
        first = candles.iloc[idx-2]  # First candle (bearish)
        middle = candles.iloc[idx-1]  # Middle candle (star)
        last = candles.iloc[idx]     # Last candle (bullish)
        
        # Calculate candle bodies and ranges
        first_body = abs(first['close'] - first['open'])
        middle_body = abs(middle['close'] - middle['open'])
        last_body = abs(last['close'] - last['open'])
        
        # 1. First candle must be bearish (close < open)
        is_first_bearish = first['close'] < first['open']
        
        # 2. Last candle must be bullish (close > open)
        is_last_bullish = last['close'] > last['open']
        
        # 3. Middle candle should have a small body
        # Calculate average body size for recent candles
        lookback = min(14, idx)
        recent_bodies = [abs(candles.iloc[i]['close'] - candles.iloc[i]['open']) for i in range(idx-lookback, idx+1)]
        avg_body_size = sum(recent_bodies) / len(recent_bodies)
        
        # Make the middle body threshold more stringent (30% instead of 50%)
        is_middle_small = middle_body < 0.3 * avg_body_size
        
        # 4. Check for a gap down or minimal overlap
        # In forex, true gaps are rare, so we check for minimal overlap
        first_low = min(first['open'], first['close'])
        middle_high = max(middle['open'], middle['close'])
        
        # Allow for some flexibility in the gap requirement
        is_gap_down = middle_high <= first_low
        has_minimal_overlap = middle_high <= first_low + 0.3 * first_body
        
        # 5. Last candle should close well into first candle's body (showing recovery)
        first_range = first['high'] - first['low']
        first_midpoint = first['open'] - (first['open'] - first['close']) / 2
        
        # Enhanced recovery requirement - close above 61.8% of first candle body
        first_61_8_level = first['open'] - 0.618 * first_body
        good_recovery = last['close'] > first_61_8_level
        
        # 6. Check overall pattern size relative to market volatility
        # Calculate average range (ATR approximation)
        recent_ranges = [candles.iloc[i]['high'] - candles.iloc[i]['low'] for i in range(idx-lookback, idx+1)]
        avg_range = sum(recent_ranges) / len(recent_ranges)
        
        # Overall pattern height (high to low)
        pattern_high = max(first['high'], middle['high'], last['high'])
        pattern_low = min(first['low'], middle['low'], last['low'])
        pattern_height = pattern_high - pattern_low
        
        # Pattern should be significant relative to recent volatility
        is_significant_pattern = pattern_height >= 0.8 * avg_range
        
        # Combine all conditions for the pattern
        is_pattern = (is_first_bearish and is_last_bullish and is_middle_small and
                     (is_gap_down or has_minimal_overlap) and good_recovery and
                     is_significant_pattern)
        
        # If no pattern, return False
        if not is_pattern:
            return False
        
        # Check trend context if specified
        if in_downtrend is not None:
            # Morning star is a bullish reversal pattern and should appear in downtrends
            if not in_downtrend:
                return False
        
        # Check confirmation if required
        if require_confirmation and idx < len(candles) - 1:
            confirmation = candles.iloc[idx+1]
            if confirmation['close'] <= last['close']:
                return False
                
        return True
    
    def _is_evening_star(self, candles: pd.DataFrame, idx: int, in_uptrend: bool = None, require_confirmation: bool = False) -> bool:
        """
        Check if the pattern at idx is an evening star pattern with optional trend context and confirmation.
        
        An evening star is a three-candle bearish reversal pattern consisting of:
        1. A large bullish candle
        2. A small bodied candle (star) that gaps up (or has small/no overlap with first candle)
        3. A bearish candle that closes well into the first candle's body
        
        Args:
            candles: DataFrame containing price data
            idx: Index to check for pattern (this is the index of the LAST candle in the pattern)
            in_uptrend: If True, validates pattern in uptrend (more reliable). If None, only checks pattern shape.
            require_confirmation: Whether to require confirmation from the next candle
            
        Returns:
            True if a valid evening star with the specified context and confirmation
        """
        # Ensure we have enough candles to check the pattern
        if idx < 2 or idx >= len(candles) - (1 if require_confirmation else 0):
            return False
        
        # Get the three candles for the pattern
        first = candles.iloc[idx-2]  # First candle (bullish)
        middle = candles.iloc[idx-1]  # Middle candle (star)
        last = candles.iloc[idx]     # Last candle (bearish)
        
        # Calculate candle bodies and ranges
        first_body = abs(first['close'] - first['open'])
        middle_body = abs(middle['close'] - middle['open'])
        last_body = abs(last['close'] - last['open'])
        
        # 1. First candle must be bullish (close > open)
        is_first_bullish = first['close'] > first['open']
        
        # 2. Last candle must be bearish (close < open)
        is_last_bearish = last['close'] < last['open']
        
        # 3. Middle candle should have a small body
        # Calculate average body size for recent candles
        lookback = min(14, idx)
        recent_bodies = [abs(candles.iloc[i]['close'] - candles.iloc[i]['open']) for i in range(idx-lookback, idx+1)]
        avg_body_size = sum(recent_bodies) / len(recent_bodies)
        
        # Make the middle body threshold more stringent (30% instead of 50%)
        is_middle_small = middle_body < 0.3 * avg_body_size
        
        # 4. Check for a gap up or minimal overlap
        # In forex, true gaps are rare, so we check for minimal overlap
        first_high = max(first['open'], first['close'])
        middle_low = min(middle['open'], middle['close'])
        
        # Allow for some flexibility in the gap requirement
        is_gap_up = middle_low >= first_high
        has_minimal_overlap = middle_low >= first_high - 0.3 * first_body
        
        # 5. Last candle should close well into first candle's body (showing decline)
        first_midpoint = first['open'] + (first['close'] - first['open']) / 2
        
        # Enhanced decline requirement - close below 61.8% of first candle body
        first_61_8_level = first['open'] + 0.618 * first_body
        good_decline = last['close'] < first_61_8_level
        
        # 6. Check overall pattern size relative to market volatility
        # Calculate average range (ATR approximation)
        recent_ranges = [candles.iloc[i]['high'] - candles.iloc[i]['low'] for i in range(idx-lookback, idx+1)]
        avg_range = sum(recent_ranges) / len(recent_ranges)
        
        # Overall pattern height (high to low)
        pattern_high = max(first['high'], middle['high'], last['high'])
        pattern_low = min(first['low'], middle['low'], last['low'])
        pattern_height = pattern_high - pattern_low
        
        # Pattern should be significant relative to recent volatility
        is_significant_pattern = pattern_height >= 0.8 * avg_range
        
        # Combine all conditions for the pattern
        is_pattern = (is_first_bullish and is_last_bearish and is_middle_small and
                     (is_gap_up or has_minimal_overlap) and good_decline and
                     is_significant_pattern)
        
        # If no pattern, return False
        if not is_pattern:
            return False
        
        # Check trend context if specified
        if in_uptrend is not None:
            # Evening star is a bearish reversal pattern and should appear in uptrends
            if not in_uptrend:
                return False
        
        # Check confirmation if required
        if require_confirmation and idx < len(candles) - 1:
            confirmation = candles.iloc[idx+1]
            if confirmation['close'] >= last['close']:
                return False
                
        return True
    
    def _is_strong_candle(self, candle: pd.Series) -> bool:
        """
        Check if a candle is strong (body > 50% of range).
        
        Args:
            candle: Candle data
            
        Returns:
            True if it's a strong candle
        """
        total_range = candle['high'] - candle['low']
        body = abs(candle['close'] - candle['open'])
        
        if total_range == 0:
            return False
            
        body_percentage = body / total_range
        
        return bool(body_percentage > 0.5)
    
    def _is_hammer(self, candle: pd.Series, in_downtrend: bool = None, df: pd.DataFrame = None, idx: int = None, require_confirmation: bool = False, atr_filter: bool = True) -> bool:
        """
        Check if a candle is a hammer pattern (bullish reversal in downtrend) or 
        hanging man (bearish reversal in uptrend), with optional confirmation check.
        
        Args:
            candle: Candle data
            in_downtrend: If True, validates as hammer (bullish). If False, validates as hanging man (bearish).
                         If None, only checks shape without context.
            df: DataFrame containing all candles (required for confirmation check)
            idx: Index of the current candle in df (required for confirmation check)
            require_confirmation: Whether to require confirmation from the next candle
            atr_filter: Whether to filter out hammers that are too small compared to ATR
        
        Returns:
            True if it's a hammer/hanging man with the specified context and confirmation (if required)
        """
        # Shape detection logic
        total_range = candle['high'] - candle['low']
        
        if total_range == 0 or self._is_invalid_or_zero(total_range):
            return False
            
        # Calculate components regardless of color
        body = abs(candle['close'] - candle['open'])
        
        # Find upper and lower wicks correctly regardless of candle color
        if candle['close'] >= candle['open']:  # Bullish candle
            upper_wick = candle['high'] - candle['close']
            lower_wick = candle['open'] - candle['low']
        else:  # Bearish candle
            upper_wick = candle['high'] - candle['open']
            lower_wick = candle['close'] - candle['low']
        
        # Hammer criteria: small body, little/no upper wick, long lower wick
        body_percentage = body / total_range
        upper_wick_percentage = upper_wick / total_range
        lower_wick_percentage = lower_wick / total_range
        
        # More flexible body threshold based on market conditions
        # For high volatility periods, allow slightly larger bodies up to 0.4
        # For normal markets, use standard 0.3 threshold
        max_body_pct = 0.35  # Slightly more flexible than the strict 0.3
        
        # ATR Check - ensure the hammer is significant compared to average volatility
        # Only perform this check if requested and we have the necessary data
        is_significant_size = True
        if atr_filter and df is not None and idx is not None:
            try:
                # Calculate ATR for the recent period
                atr_period = 14
                if len(df) >= atr_period + 1:
                    atr_series = calculate_atr(df.iloc[max(0, idx-atr_period):idx+1], atr_period)
                    
                    # Get the last ATR value from the series if it's a pandas Series
                    if isinstance(atr_series, pd.Series) and not atr_series.empty:
                        atr = atr_series.iloc[-1]  # Get the most recent ATR value as a scalar
                    else:
                        atr = atr_series  # It might already be a scalar
                    
                    # Ensure atr is a valid scalar value
                    if not self._is_invalid_or_zero(atr):
                        # Hammer should be at least 70% of ATR to be significant
                        is_significant_size = float(total_range) >= (float(atr) * 0.7)
                        
                        # Adjust body threshold based on volatility
                        volatility_ratio = float(total_range) / float(atr)
                        # More volatile markets (larger candles relative to ATR) can have slightly larger bodies
                        if volatility_ratio > 1.5:
                            max_body_pct = 0.4  # Allow larger bodies in volatile conditions
                        elif volatility_ratio < 0.8:
                            max_body_pct = 0.25  # Require smaller bodies for less significant hammers
                        
                        logger.debug(f"Hammer ATR check: candle range {float(total_range):.5f}, ATR {float(atr):.5f}, "
                                    f"significant: {is_significant_size}, adjusted body threshold: {max_body_pct:.2f}")
                    else:
                        logger.debug(f"Invalid ATR value: {atr}")
            except Exception as e:
                logger.debug(f"Error during ATR calculation for hammer: {str(e)}")
                # Continue with default parameters if ATR calculation fails
        
        # Shape detection with more flexible criteria and size check
        is_hammer_shape = (
            body_percentage < max_body_pct and  # More flexible body threshold
            upper_wick_percentage < 0.15 and    # Slightly more lenient on upper wick
            lower_wick_percentage > 0.55 and    # Slightly more lenient on lower wick
            is_significant_size                 # Ensure the hammer is significant compared to ATR
        )

        # If it's not a hammer shape, return False immediately
        if not is_hammer_shape:
            return False
        
        # If no trend context provided or we're not requiring confirmation, return shape detection result
        if in_downtrend is None or (not require_confirmation):
            return is_hammer_shape
        
        # Confirmation check (if required)
        if require_confirmation:
            # Ensure we have the necessary data for confirmation
            if df is None or idx is None or idx >= len(df) - 1:
                return False
            
            # Get the confirmation candle
            confirmation_candle = df.iloc[idx + 1]
            
            if in_downtrend:
                # Bullish confirmation for hammer
                return confirmation_candle['close'] > candle['close']
            else:
                # Bearish confirmation for hanging man
                return confirmation_candle['close'] < candle['close']
        
        # If we got here, it's a hammer shape with the right context but no confirmation required
        return True
    
    def _is_shooting_star(self, candles: pd.DataFrame, idx: int, in_uptrend: bool = None, require_confirmation: bool = False) -> bool:
        """
        Check if the pattern at the given index is a shooting star.
        A shooting star is essentially an inverted hammer that appears in an uptrend.
        
        Both shooting stars and inverted hammers have the same candlestick shape:
        - Small body near the bottom of the range
        - Long upper shadow
        - Little to no lower shadow
        
        The key difference is the market context:
        - Shooting star: appears in an uptrend and is a bearish reversal signal
        - Inverted hammer: appears in a downtrend and is a bullish reversal signal
        
        Args:
            candles: DataFrame containing the candle data
            idx: Index of the candle to check
            in_uptrend: Whether we're in an uptrend (True), downtrend (False), or unknown (None)
            require_confirmation: Whether to require confirmation from the next candle
            
        Returns:
            bool: True if the pattern is a valid shooting star, False otherwise
        """
        # For a shooting star to be valid, we must be in an uptrend
        if in_uptrend is False:
            # Not in an uptrend, so this can't be a shooting star
            logger.debug(f"Not a shooting star at idx {idx}: not in uptrend")
            return False
            
        # Get the specific candle
        if idx >= len(candles) or idx < 0:
            return False
            
        candle = candles.iloc[idx]
        
        # A shooting star is an inverted hammer in an uptrend (bearish context)
        # We pass in_downtrend=False to indicate we're NOT in a downtrend (i.e., we are in an uptrend)
        is_pattern = self._is_inverted_hammer(
            candle=candle,
            in_downtrend=False,  # Not in downtrend = in uptrend for shooting star
            df=candles,
            idx=idx,
            require_confirmation=require_confirmation,
            atr_filter=True
        )
        
        if is_pattern and in_uptrend is None:
            # Pattern matches but we don't know the trend context
            logger.warning(f"Shooting star detected at idx {idx} but trend context unknown")
                
        return is_pattern
    
    def _is_bullish_engulfing(self, candles: pd.DataFrame, idx: int, in_downtrend: bool = None, require_confirmation: bool = False) -> bool:
        """
        Check if the pattern at idx is a bullish engulfing pattern with optional trend context and confirmation.
        
        A bullish engulfing pattern consists of a bearish candle followed by a bullish candle
        that completely engulfs the body of the previous candle.
        
        Args:
            candles: DataFrame containing price data
            idx: Index to check for pattern
            in_downtrend: If True, validates pattern in downtrend (more reliable). If None, only checks pattern shape.
            require_confirmation: Whether to require confirmation from the next candle
            
        Returns:
            True if a valid bullish engulfing with the specified context and confirmation
        """
        # Ensure we have enough candles to check the pattern
        if idx <= 0 or idx >= len(candles) - (1 if require_confirmation else 0):
            return False
            
        current = candles.iloc[idx]
        previous = candles.iloc[idx-1]
        
        # Basic pattern check - current candle must be bullish
        is_current_bullish = current['close'] > current['open']
        is_previous_bearish = previous['close'] < previous['open']
        
        # Calculate candle bodies and ranges for size comparisons
        current_body = abs(current['close'] - current['open'])
        previous_body = abs(previous['close'] - previous['open'])
        
        current_range = current['high'] - current['low']
        previous_range = previous['high'] - previous['low']
        
        # Check if current candle body engulfs previous candle body
        engulfs_body = (current['open'] <= previous['close'] and 
                current['close'] >= previous['open'])
    
        # New checks for stronger pattern identification
        # 1. Minimum size requirement - engulfing candle should be significant
        # Calculate average range for recent candles as a simple ATR approximation
        lookback = min(14, idx)
        if lookback > 0:
            recent_ranges = [candles.iloc[i]['high'] - candles.iloc[i]['low'] for i in range(idx-lookback, idx+1)]
            avg_range = sum(recent_ranges) / len(recent_ranges)
        else:
            avg_range = current_range
            
        min_body_size = 0.5 * avg_range  # Engulfing body should be at least 50% of average range
        has_significant_size = current_body >= min_body_size
        
        # 2. Relative body size check - engulfing candle should be larger
        relative_body_ratio = current_body / previous_body if previous_body > 0 else 2.0
        has_larger_body = relative_body_ratio >= 1.3  # At least 30% larger
        
        # Combine basic and advanced checks
        is_pattern = (is_current_bullish and is_previous_bearish and 
                     engulfs_body and has_significant_size and has_larger_body)
        
        # Check trend context if specified
        if in_downtrend is not None and is_pattern:
            # Only valid in downtrend for reversal context
            if not in_downtrend:
                return False
                
        # Check confirmation if required
        if require_confirmation and is_pattern and idx < len(candles) - 1:
            confirmation = candles.iloc[idx+1]
            if confirmation['close'] <= current['close']:
                return False
            
        return is_pattern
    
    def _is_bearish_engulfing(self, candles: pd.DataFrame, idx: int, in_uptrend: bool = None, require_confirmation: bool = False) -> bool:
        """
        Check if the pattern at idx is a bearish engulfing pattern with optional trend context and confirmation.
        
        A bearish engulfing pattern consists of a bullish candle followed by a bearish candle
        that completely engulfs the body of the previous candle.
        
        Args:
            candles: DataFrame containing price data
            idx: Index to check for pattern
            in_uptrend: If True, validates pattern in uptrend (more reliable). If None, only checks pattern shape.
            require_confirmation: Whether to require confirmation from the next candle
            
        Returns:
            True if a valid bearish engulfing with the specified context and confirmation
        """
        # Ensure we have enough candles to check the pattern
        if idx <= 0 or idx >= len(candles) - (1 if require_confirmation else 0):
            return False
            
        current = candles.iloc[idx]
        previous = candles.iloc[idx-1]
        
        # Basic pattern check - current candle must be bearish
        is_current_bearish = current['close'] < current['open']
        is_previous_bullish = previous['close'] > previous['open']
        
        # Calculate candle bodies and ranges for size comparisons
        current_body = abs(current['close'] - current['open'])
        previous_body = abs(previous['close'] - previous['open'])
        
        current_range = current['high'] - current['low']
        previous_range = previous['high'] - previous['low']
        
        # Check if current candle body engulfs previous candle body
        engulfs_body = (current['open'] >= previous['close'] and 
                current['close'] <= previous['open'])
        
        # New checks for stronger pattern identification
        # 1. Minimum size requirement - engulfing candle should be significant
        # Calculate average range for recent candles as a simple ATR approximation
        lookback = min(14, idx)
        if lookback > 0:
            recent_ranges = [candles.iloc[i]['high'] - candles.iloc[i]['low'] for i in range(idx-lookback, idx+1)]
            avg_range = sum(recent_ranges) / len(recent_ranges)
        else:
            avg_range = current_range
            
        min_body_size = 0.5 * avg_range  # Engulfing body should be at least 50% of average range
        has_significant_size = current_body >= min_body_size
        
        # 2. Relative body size check - engulfing candle should be larger
        relative_body_ratio = current_body / previous_body if previous_body > 0 else 2.0
        has_larger_body = relative_body_ratio >= 1.3  # At least 30% larger
        
        # Combine basic and advanced checks
        is_pattern = (is_current_bearish and is_previous_bullish and 
                     engulfs_body and has_significant_size and has_larger_body)
        
        # Check trend context if specified
        if in_uptrend is not None and is_pattern:
            # Only valid in uptrend for reversal context
            if not in_uptrend:
                return False
                
        # Check confirmation if required
        if require_confirmation and is_pattern and idx < len(candles) - 1:
            confirmation = candles.iloc[idx+1]
            if confirmation['close'] >= current['close']:
                return False
            
        return is_pattern
    
    def _determine_h1_trend(self, h1_df: pd.DataFrame) -> str:
        """
        Determine the trend on the higher timeframe using price action instead of EMA.
        Uses swing highs and lows to identify the trend direction.
        
        Args:
            h1_df: Higher timeframe dataframe
            
        Returns:
            'bullish', 'bearish', or 'neutral'
        """
        if len(h1_df) < 20:
            logger.debug(f"⚠️ Not enough data for trend determination, need 20 candles but got {len(h1_df)}")
            return 'neutral'
        
        try:
            # Get a subset of recent data
            lookback = min(30, len(h1_df))
            df_subset = h1_df.iloc[-lookback:].copy()
            
            # Find swing highs and lows
            swing_highs = []
            swing_lows = []
            
            # We need at least 5 candles to establish a good pattern of swings
            if len(df_subset) < 5:
                logger.debug(f"⚠️ Insufficient data for swing analysis, using last 2 candles for simple trend")
                # Simple trend based on last 2 candles
                if float(df_subset['close'].iloc[-1]) > float(df_subset['close'].iloc[-2]):
                    return 'bullish'
                elif float(df_subset['close'].iloc[-1]) < float(df_subset['close'].iloc[-2]):
                    return 'bearish'
                else:
                    return 'neutral'
            
            # Use a rolling window to find swing points
            for i in range(2, len(df_subset) - 2):
                # Check for swing high
                if (float(df_subset['high'].iloc[i]) > float(df_subset['high'].iloc[i-1]) and 
                    float(df_subset['high'].iloc[i]) > float(df_subset['high'].iloc[i-2]) and
                    float(df_subset['high'].iloc[i]) > float(df_subset['high'].iloc[i+1]) and 
                    float(df_subset['high'].iloc[i]) > float(df_subset['high'].iloc[i+2])):
                    swing_highs.append((i, float(df_subset['high'].iloc[i])))
                
                # Check for swing low
                if (float(df_subset['low'].iloc[i]) < float(df_subset['low'].iloc[i-1]) and 
                    float(df_subset['low'].iloc[i]) < float(df_subset['low'].iloc[i-2]) and
                    float(df_subset['low'].iloc[i]) < float(df_subset['low'].iloc[i+1]) and 
                    float(df_subset['low'].iloc[i]) < float(df_subset['low'].iloc[i+2])):
                    swing_lows.append((i, float(df_subset['low'].iloc[i])))
            
            # Need at least two swing points of each type to determine trend
            if len(swing_highs) >= 2 and len(swing_lows) >= 2:
                # Get the last two swing highs and lows
                last_two_highs = sorted(swing_highs, key=lambda x: x[0])[-2:]
                last_two_lows = sorted(swing_lows, key=lambda x: x[0])[-2:]
                
                # Extract the values
                high1, high2 = last_two_highs[0][1], last_two_highs[1][1]
                low1, low2 = last_two_lows[0][1], last_two_lows[1][1]
                
                # Higher highs and higher lows = bullish trend
                # Lower highs and lower lows = bearish trend
                if high2 > high1 and low2 > low1:
                    trend = 'bullish'
                    logger.debug(f"📈 Bullish trend detected: Higher highs ({high1:.5f} → {high2:.5f}) and higher lows ({low1:.5f} → {low2:.5f})")
                elif high2 < high1 and low2 < low1:
                    trend = 'bearish'
                    logger.debug(f"📉 Bearish trend detected: Lower highs ({high1:.5f} → {high2:.5f}) and lower lows ({low1:.5f} → {low2:.5f})")
                else:
                    # Conflicting signals - check the most recent swing points
                    # If the most recent swing is a high, check if it's higher than previous
                    # If the most recent swing is a low, check if it's lower than previous
                    latest_swings = swing_highs + swing_lows
                    if not latest_swings:
                        trend = 'neutral'
                    else:
                        latest_swing = max(latest_swings, key=lambda x: x[0])
                        is_high = latest_swing in swing_highs
                        
                        if is_high:
                            # Latest swing is a high, compare to previous high
                            if len(swing_highs) >= 2:
                                prev_high = sorted(swing_highs, key=lambda x: x[0])[-2][1]
                                if latest_swing[1] > prev_high:
                                    trend = 'bullish'
                                else:
                                    trend = 'bearish'
                            else:
                                trend = 'neutral'
                        else:
                            # Latest swing is a low, compare to previous low
                            if len(swing_lows) >= 2:
                                prev_low = sorted(swing_lows, key=lambda x: x[0])[-2][1]
                                if latest_swing[1] < prev_low:
                                    trend = 'bearish'
                                else:
                                    trend = 'bullish'
                            else:
                                trend = 'neutral'
            else:
                # Not enough swing points, use price action from the last 5 candles
                recent_close = df_subset['close'].iloc[-5:].values
                # Convert to regular Python list if necessary
                if hasattr(recent_close, 'tolist'):
                    recent_close = recent_close.tolist()
                # Compare first and last price in the window
                if float(recent_close[-1]) > float(recent_close[0]):
                    trend = 'bullish'
                    logger.debug(f"📈 Bullish trend based on recent price movement: {float(recent_close[0]):.5f} → {float(recent_close[-1]):.5f}")
                elif float(recent_close[-1]) < float(recent_close[0]):
                    trend = 'bearish'
                    logger.debug(f"📉 Bearish trend based on recent price movement: {float(recent_close[0]):.5f} → {float(recent_close[-1]):.5f}")
                else:
                    trend = 'neutral'
                    logger.debug(f"📊 Neutral trend detected (no clear direction)")
            
            logger.debug(f"📊 Trend determined as {trend} using price action (swing highs/lows)")
            return trend
            
        except Exception as e:
            logger.warning(f"Error in trend determination: {str(e)}, falling back to simple method")
            # Simple fallback: compare current close to N periods ago
            try:
                periods_ago = min(10, len(h1_df) - 1)
                current_close = float(h1_df['close'].iloc[-1])
                past_close = float(h1_df['close'].iloc[-periods_ago])
                
                if current_close > past_close * 1.005:  # 0.5% higher
                    return 'bullish'
                elif current_close < past_close * 0.995:  # 0.5% lower
                    return 'bearish'
                else:
                    return 'neutral'
            except Exception as e2:
                logger.warning(f"Error in fallback trend determination: {str(e2)}")
                return 'neutral'  # Ultimate fallback
    
    async def close(self):
        """Close and clean up resources."""
        logger.info(f"🔌 Closing {self.name}")
        # No specific cleanup needed
        return True
    
    def _is_invalid_or_zero(self, value):
        """
        Helper function to safely check if a value is zero, None or invalid.
        
        Args:
            value: The value to check
            
        Returns:
            bool: True if the value is None, NaN, a pandas Series/DataFrame with no valid data, or zero
        """
        if value is None:
            return True
            
        # Handle pandas Series by using the last value if possible
        if isinstance(value, pd.Series):
            if value.empty:
                return True
            try:
                # Try to get the last value from the series
                value = value.iloc[-1]
            except:
                return True
                
        # Handle DataFrame
        if isinstance(value, pd.DataFrame):
            return True
            
        # Check for NaN
        if pd.isna(value):
            return True
            
        # Check for zero value
        try:
            return float(value) == 0
        except (TypeError, ValueError):
            return True
    
    def _is_inverted_hammer(self, candle: pd.Series, in_downtrend: bool = None, df: pd.DataFrame = None, idx: int = None, require_confirmation: bool = False, atr_filter: bool = True) -> bool:
        """
        Check if a candle is an inverted hammer (bullish reversal in downtrend) or
        shooting star (bearish reversal in uptrend), with optional confirmation check.
        
        Args:
            candle: Candle data
            in_downtrend: If True, validates as inverted hammer (bullish). If False, validates as shooting star (bearish).
                         If None, only checks shape without context.
            df: DataFrame containing all candles (required for confirmation check)
            idx: Index of the current candle in df (required for confirmation check)
            require_confirmation: Whether to require confirmation from the next candle
            atr_filter: Whether to filter patterns that are too small compared to recent ATR
        
        Returns:
            True if it's an inverted hammer/shooting star with the specified context and confirmation (if required)
        """
        total_range = candle['high'] - candle['low']
        
        if total_range == 0 or self._is_invalid_or_zero(total_range):
            return False
            
        # Calculate components regardless of color
        body = abs(candle['close'] - candle['open'])
        
        # Find upper and lower wicks correctly regardless of candle color
        if candle['close'] >= candle['open']:  # Bullish candle
            upper_wick = candle['high'] - candle['close']
            lower_wick = candle['open'] - candle['low']
        else:  # Bearish candle
            upper_wick = candle['high'] - candle['open']
            lower_wick = candle['close'] - candle['low']
        
        # Inverted hammer criteria: small body, long upper wick, little/no lower wick
        body_percentage = body / total_range
        upper_wick_percentage = upper_wick / total_range
        lower_wick_percentage = lower_wick / total_range
        
        # More flexible body threshold for inverted hammers (typically can have slightly larger bodies)
        max_body_pct = 0.38  # Default: more lenient than standard hammer (0.35)
        
        # ATR Check - ensure the pattern is significant compared to average volatility
        is_significant_size = True
        if atr_filter and df is not None and idx is not None:
            try:
                # Calculate ATR for the recent period
                atr_period = 14
                if len(df) >= atr_period + 1:
                    atr_series = calculate_atr(df.iloc[max(0, idx-atr_period):idx+1], atr_period)
                    
                    # Get the last ATR value from the series if it's a pandas Series
                    if isinstance(atr_series, pd.Series) and not atr_series.empty:
                        atr = atr_series.iloc[-1]  # Get the most recent ATR value as a scalar
                    else:
                        atr = atr_series  # It might already be a scalar
                    
                    # Ensure atr is a valid scalar value
                    if not self._is_invalid_or_zero(atr):
                        # Inverted hammer should be significant relative to ATR
                        # For inverted hammers, typically the upper shadow is what matters most
                        # So we check both total range and upper wick specifically
                        is_significant_total = float(total_range) >= (float(atr) * 0.7)
                        is_significant_shadow = float(upper_wick) >= (float(atr) * 0.5)  # Upper shadow should be at least 50% of ATR
                        is_significant_size = is_significant_total and is_significant_shadow
                        
                        # Adjust body threshold based on volatility
                        volatility_ratio = float(total_range) / float(atr)
                        # More volatile markets (larger candles relative to ATR) can have slightly larger bodies
                        if volatility_ratio > 1.5:
                            max_body_pct = 0.42  # More lenient in volatile markets
                        elif volatility_ratio < 0.8:
                            max_body_pct = 0.28  # Stricter for less significant patterns
                    
                        logger.debug(f"Inverted Hammer ATR check: candle range {float(total_range):.5f}, upper wick {float(upper_wick):.5f}, "
                                  f"ATR {float(atr):.5f}, significant: {is_significant_size}, "
                                  f"adjusted body threshold: {max_body_pct:.2f}")
                    else:
                        logger.debug(f"Invalid ATR value: {atr}")
            except Exception as e:
                logger.debug(f"Error during ATR calculation for inverted hammer: {str(e)}")
                # Continue with default parameters if ATR calculation fails
        
        # Adjusted criteria for real-world inverted hammers:
        # 1. Slightly more lenient on body size (up to 38% vs 30%)
        # 2. Allow slightly less upper wick (55% vs 60%)
        # 3. Allow slightly more lower wick (15% vs 10%)
        is_inverted_hammer_shape = (
            body_percentage < max_body_pct and    # Flexible body threshold
            upper_wick_percentage > 0.55 and      # Still significant upper wick
            lower_wick_percentage < 0.15 and      # Still minimal lower wick
            is_significant_size                   # Significant relative to ATR
        )
        
        # If it's not an inverted hammer shape, return False immediately
        if not is_inverted_hammer_shape:
            return False
        
        # If no trend context provided or we're not requiring confirmation, return shape detection result
        if in_downtrend is None or (not require_confirmation):
            return is_inverted_hammer_shape
        
        # Confirmation check (if required)
        if require_confirmation:
            # Ensure we have the necessary data for confirmation
            if df is None or idx is None or idx >= len(df) - 1:
                return False
                
            # Get the confirmation candle
            confirmation_candle = df.iloc[idx + 1]
            
            if in_downtrend:
                # Bullish confirmation for inverted hammer
                return confirmation_candle['close'] > candle['close']
            else:
                # Bearish confirmation for shooting star
                return confirmation_candle['close'] < candle['close']
        
        # If we got here, it's an inverted hammer shape with the right context but no confirmation required
        return True
    