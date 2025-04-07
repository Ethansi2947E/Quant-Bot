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

from src.trading_bot import SignalGenerator
from src.utils.indicators import calculate_atr

# Strategy parameter profiles for different timeframes
TIMEFRAME_PROFILES = {
    "M1": {
        "lookback_period": 100,
        "max_retest_bars": 10,
        "level_update_hours": 4,
        "consolidation_bars": 60,
        "candles_to_check": 10,
        "consolidation_update_hours": 2
    },
    "M5": {
        "lookback_period": 80,
        "max_retest_bars": 8,
        "level_update_hours": 6,
        "consolidation_bars": 40, 
        "candles_to_check": 6,
        "consolidation_update_hours": 3
    },
    "M15": {
        "lookback_period": 50,
        "max_retest_bars": 5,
        "level_update_hours": 12,
        "consolidation_bars": 20,
        "candles_to_check": 3,
        "consolidation_update_hours": 6
    },
    "H1": {
        "lookback_period": 30,
        "max_retest_bars": 3,
        "level_update_hours": 24,
        "consolidation_bars": 10,
        "candles_to_check": 2,
        "consolidation_update_hours": 12
    },
    "H4": {
        "lookback_period": 20,
        "max_retest_bars": 2,
        "level_update_hours": 48,
        "consolidation_bars": 7,
        "candles_to_check": 2,
        "consolidation_update_hours": 24
    }
}

class BreakoutReversalStrategy(SignalGenerator):
    """
    Breakout and Reversal Hybrid Strategy based on price action principles.
    Uses support/resistance levels, candlestick patterns, and volume analysis
    to generate high-probability trading signals.
    """
    
    def __init__(self, primary_timeframe="M5", higher_timeframe="M15", **kwargs):
        """
        Initialize the Breakout and Reversal strategy.
        
        Args:
            primary_timeframe: Primary timeframe to analyze
            higher_timeframe: Higher timeframe for trend confirmation
            **kwargs: Additional parameters
        """
        # Call parent constructor to set up logger
        super().__init__()
        
        # Strategy metadata
        self.name = "BreakoutReversalStrategy"
        self.description = "A hybrid strategy based on price action principles"
        self.version = "1.0.0"
        
        # Timeframes
        self.primary_timeframe = primary_timeframe
        self.higher_timeframe = higher_timeframe
        self.required_timeframes = [primary_timeframe, higher_timeframe]
        
        # Load appropriate timeframe profile
        if primary_timeframe in ["M5", "M15"]:
            self.timeframe_profile = "intraday"
        else:
            self.timeframe_profile = "swing"
        
        # General parameters
        self.lookback_period = kwargs.get("lookback_period", 100)
        self.price_tolerance = kwargs.get("price_tolerance", 0.001)  # 0.1% tolerance for levels
        
        # Key level parameters
        self.min_level_touches = kwargs.get("min_level_touches", 2)
        self.level_recency_weight = kwargs.get("level_recency_weight", 0.5)
        self.level_update_interval = kwargs.get("level_update_interval", 8)  # Hours
        
        # Trend line parameters
        self.trend_line_min_points = kwargs.get("trend_line_min_points", 3)
        self.trend_line_max_angle = kwargs.get("trend_line_max_angle", 45)  # degrees
        self.trend_line_update_interval = kwargs.get("trend_line_update_interval", 8)  # Hours
        
        # Breakout parameters
        self.retest_required = kwargs.get("retest_required", False)  # Require retest to confirm
        self.max_retest_time = kwargs.get("max_retest_time", 24)  # Max hours to wait for retest
        self.candles_to_check = kwargs.get("candles_to_check", 5)  # How many recent candles to analyze
        
        # Consolidation parameters
        self.consolidation_length = kwargs.get("consolidation_length", 12)  # Minimum number of candles
        self.consolidation_range_max = kwargs.get("consolidation_range_max", 0.02)  # 2% max range
        self.range_update_interval = kwargs.get("range_update_interval", 4)  # Hours
        
        # Risk management
        self.min_risk_reward = kwargs.get("min_risk_reward", 1.5)  # Minimum R:R ratio
        self.max_stop_pct = kwargs.get("max_stop_pct", 0.02)  # Maximum stop loss (% of price)
        
        # Volume analysis
        self.volume_threshold = kwargs.get("volume_threshold", 0.8)  # Volume spike threshold (multiplier of average) - lowered from 1.5 to 0.8
        
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
            'min_risk_reward': self.min_risk_reward
        }
        logger.debug(f"📊 Strategy parameters: {params}")
    
    def _load_timeframe_profile(self):
        """Load timeframe-specific parameters from the appropriate profile."""
        # Default to M1 profile if timeframe not found
        profile = TIMEFRAME_PROFILES.get(self.primary_timeframe, TIMEFRAME_PROFILES["M5"])
        
        # Set parameters based on timeframe profile
        self.lookback_period = profile["lookback_period"]
        self.max_retest_bars = profile["max_retest_bars"]
        self.level_update_hours = profile["level_update_hours"]
        self.consolidation_bars = profile["consolidation_bars"]
        self.candles_to_check = profile["candles_to_check"]
        self.consolidation_update_hours = profile["consolidation_update_hours"]
        
        logger.info(f"⚙️ Loaded profile for {self.primary_timeframe} timeframe")
    
    async def initialize(self):
        """Initialize resources needed by the strategy."""
        logger.info(f"🔌 Initializing {self.name}")
        # No specific initialization needed
        return True
    
    async def generate_signals(self, market_data=None, symbol=None, timeframe=None, **kwargs):
        """
        Generate trading signals based on price action strategies.
        
        Args:
            market_data: Dictionary with format {symbol: {timeframe: dataframe}}
            symbol: Trading symbol
            timeframe: Timeframe of the data
            
        Returns:
            List of signal dictionaries
        """
        if not market_data:
            logger.warning("⚠️ No market data provided to generate signals")
            return []
            
        signals = []
        logger.info(f"🔍 Generating signals with {self.name} strategy")
        
        for symbol in market_data:
            # Log market data format for this symbol
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
                            logger.debug(f"   {i}: O={candle['open']:.5f}, H={candle['high']:.5f}, L={candle['low']:.5f}, C={candle['close']:.5f}, Vol={candle['tick_volume']}")
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
                    if self.primary_timeframe.startswith('M'):
                        # Extract minutes from timeframe (e.g., 'M5' -> 5)
                        try:
                            minutes = int(self.primary_timeframe[1:])
                            # Create timestamps going back from current time
                            timestamps = [current_time - timedelta(minutes=minutes * i) for i in range(len(primary_df)-1, -1, -1)]
                            primary_df.index = pd.DatetimeIndex(timestamps)
                            logger.debug(f"Created synthetic DatetimeIndex using {minutes} minute intervals")
                        except ValueError:
                            logger.warning(f"Could not parse timeframe {self.primary_timeframe}, using default 5 minutes")
                            timestamps = [current_time - timedelta(minutes=5 * i) for i in range(len(primary_df)-1, -1, -1)]
                            primary_df.index = pd.DatetimeIndex(timestamps)
                    else:
                        # Default to 5-minute intervals if timeframe format is unknown
                        timestamps = [current_time - timedelta(minutes=5 * i) for i in range(len(primary_df)-1, -1, -1)]
                        primary_df.index = pd.DatetimeIndex(timestamps)
                        logger.debug(f"Created synthetic DatetimeIndex using default 5 minute intervals")
            
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
                    if self.higher_timeframe.startswith('M'):
                        # Extract minutes from timeframe (e.g., 'M15' -> 15)
                        try:
                            minutes = int(self.higher_timeframe[1:])
                            # Create timestamps going back from current time
                            timestamps = [current_time - timedelta(minutes=minutes * i) for i in range(len(higher_df)-1, -1, -1)]
                            higher_df.index = pd.DatetimeIndex(timestamps)
                            logger.debug(f"Created synthetic DatetimeIndex using {minutes} minute intervals for higher timeframe")
                        except ValueError:
                            logger.warning(f"Could not parse timeframe {self.higher_timeframe}, using default 15 minutes")
                            timestamps = [current_time - timedelta(minutes=15 * i) for i in range(len(higher_df)-1, -1, -1)]
                            higher_df.index = pd.DatetimeIndex(timestamps)
                    else:
                        # Default to 15-minute intervals if timeframe format is unknown
                        timestamps = [current_time - timedelta(minutes=15 * i) for i in range(len(higher_df)-1, -1, -1)]
                        higher_df.index = pd.DatetimeIndex(timestamps)
                        logger.debug(f"Created synthetic DatetimeIndex using default 15 minute intervals for higher timeframe")
                
            # Update key levels
            self._update_key_levels(symbol, primary_df)
            
            # Find trend lines
            self._find_trend_lines(symbol, primary_df)
            
            # Identify consolidation ranges for advanced target calculation
            self._identify_consolidation_ranges(symbol, primary_df)
            
            # Process any pending retest conditions
            self._process_retest_conditions(symbol, primary_df)
            
            # Check for breakout signals
            breakout_signals = self._check_breakout_signals(symbol, primary_df, higher_df)
            
            # Check for reversal signals
            reversal_signals = self._check_reversal_signals(symbol, primary_df, higher_df)
            
            # Update signals
            if breakout_signals:
                signals.extend(breakout_signals)
            
            if reversal_signals:
                signals.extend(reversal_signals)
        
        logger.info(f"✅ Generated {len(signals)} signals with {self.name}")
        return signals
    
    def _update_key_levels(self, symbol: str, df: pd.DataFrame) -> None:
        """
        Update support and resistance levels for a symbol.
        
        Args:
            symbol: Trading symbol
            df: Price dataframe
        """
        # Check if we need to update levels (limit computation)
        current_time = df.index[-1]
        
        # Ensure current_time is a datetime object
        if not isinstance(current_time, datetime):
            logger.debug(f"Converting current_time from {type(current_time)} to datetime")
            # If it's a timestamp (integer), convert to datetime
            if isinstance(current_time, (int, np.integer, float)):
                try:
                    current_time = pd.to_datetime(current_time, unit='s')
                except:
                    try:
                        current_time = pd.to_datetime(current_time, unit='ms')
                    except:
                        # If conversion fails, use current time
                        current_time = datetime.now()
                        logger.debug(f"Failed to convert timestamp, using current time instead")
        
        if symbol in self.last_updated:
            last_update = self.last_updated[symbol]
            # Ensure last_update is a datetime object too
            if not isinstance(last_update, datetime):
                logger.debug(f"Converting last_update from {type(last_update)} to datetime")
                if isinstance(last_update, (int, np.integer, float)):
                    try:
                        last_update = pd.to_datetime(last_update, unit='s')
                    except:
                        try:
                            last_update = pd.to_datetime(last_update, unit='ms')
                        except:
                            # If conversion fails, use a time far in the past to force update
                            last_update = datetime.now() - timedelta(hours=self.level_update_interval + 1)
                            logger.debug(f"Failed to convert last_update, forcing update")
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
            
    def _find_trend_lines(self, symbol: str, df: pd.DataFrame) -> None:
        """
        Find bullish and bearish trend lines using linear regression on swing points.
        
        Args:
            symbol: Trading symbol
            df: Price dataframe
        """
        # Skip if we've recently updated
        current_time = df.index[-1]
        
        # Ensure current_time is a datetime object
        if not isinstance(current_time, datetime):
            logger.debug(f"Converting current_time from {type(current_time)} to datetime")
            # If it's a timestamp (integer), convert to datetime
            if isinstance(current_time, (int, np.integer, float)):
                try:
                    current_time = pd.to_datetime(current_time, unit='s')
                except:
                    try:
                        current_time = pd.to_datetime(current_time, unit='ms')
                    except:
                        # If conversion fails, use current time
                        current_time = datetime.now()
                        logger.debug(f"Failed to convert timestamp, using current time instead")
        
        if symbol in self.last_updated:
            last_update = self.last_updated[symbol]
            # Ensure last_update is a datetime object too
            if not isinstance(last_update, datetime):
                logger.debug(f"Converting last_update from {type(last_update)} to datetime")
                if isinstance(last_update, (int, np.integer, float)):
                    try:
                        last_update = pd.to_datetime(last_update, unit='s')
                    except:
                        try:
                            last_update = pd.to_datetime(last_update, unit='ms')
                        except:
                            # If conversion fails, use a time far in the past to force update
                            last_update = datetime.now() - timedelta(hours=self.trend_line_update_interval + 1)
                            logger.debug(f"Failed to convert last_update, forcing update")
                # Update the stored value
                self.last_updated[symbol] = last_update
            
            # Only update every X hours along with horizontal levels
            try:
                time_diff = (current_time - last_update).total_seconds()
                if time_diff < self.trend_line_update_interval * 3600:
                    logger.debug(f"🕒 Skipping trend line update for {symbol}, last update was {time_diff/3600:.1f} hours ago")
                    return
            except Exception as e:
                logger.warning(f"Error calculating time difference: {e}. Forcing update.")
                # Force update in case of error
        
        logger.debug(f"📊 Finding trend lines for {symbol} with {len(df)} candles")
        
        # Find swing highs and lows for trend line analysis
        swing_highs = self._find_swing_highs(df)
        swing_lows = self._find_swing_lows(df)
        
        logger.debug(f"🔍 Found {len(swing_highs)} swing highs and {len(swing_lows)} swing lows for {symbol}")
        
        # Get trend lines
        bullish_trend_lines = self._identify_trend_lines(df, swing_lows, 'bullish')
        bearish_trend_lines = self._identify_trend_lines(df, swing_highs, 'bearish')
        
        # Store trend lines
        self.bullish_trend_lines[symbol] = bullish_trend_lines
        self.bearish_trend_lines[symbol] = bearish_trend_lines
        self.last_updated[symbol] = current_time
        
        logger.info(f"📈 Identified trend lines for {symbol} - Bullish: {len(bullish_trend_lines)}, Bearish: {len(bearish_trend_lines)}")
    
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
                              line_type: str) -> List[Dict]:
        """
        Identify valid trend lines using swing points.
        
        Args:
            df: Price dataframe
            swing_points: List of (index, price) tuples
            line_type: 'bullish' for support trend lines, 'bearish' for resistance
            
        Returns:
            List of trend line dictionaries with slope, intercept, and validity data
        """
        if len(swing_points) < self.trend_line_min_points:
            return []
        
        valid_trend_lines = []
        
        # Try to find trend lines with at least trend_line_min_points points
        for i in range(len(swing_points) - (self.trend_line_min_points - 1)):
            # Select a subset of points to try
            points_subset = swing_points[i:i+self.trend_line_min_points]
            
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
            
            # Calculate angle in degrees
            angle_degrees = math.degrees(math.atan(slope))
            
            # Valid trend line criteria
            # For bullish (support) lines: upward slope but < max_slope
            # For bearish (resistance) lines: downward slope but > -max_slope
            if ((line_type == 'bullish' and 0 < angle_degrees < self.trend_line_max_angle) or
                (line_type == 'bearish' and 0 > angle_degrees > -self.trend_line_max_angle)):
                
                # Calculate fit quality
                r_squared = r_value ** 2
                
                # Only accept trend lines with good fit
                if r_squared > 0.7:
                    # Count price touches
                    touches = self._count_trend_line_touches(df, slope, intercept, line_type)
                    
                    if touches >= self.min_level_touches:
                        # Safely extract the start and end indices
                        start_idx = 0
                        end_idx = 0
                        if points_subset and len(points_subset) > 0:
                            if isinstance(points_subset[0], tuple) and len(points_subset[0]) > 0:
                                # Explicitly convert to int to avoid type issues
                                try:
                                    start_idx = int(float(points_subset[0][0]))
                                except (ValueError, TypeError, IndexError):
                                    logger.warning(f"Could not convert start index to int: {points_subset[0]}")
                                    start_idx = 0
                                    
                        if points_subset and len(points_subset) > 0:
                            if isinstance(points_subset[-1], tuple) and len(points_subset[-1]) > 0:
                                # Explicitly convert to int to avoid type issues
                                try:
                                    end_idx = int(float(points_subset[-1][0]))
                                except (ValueError, TypeError, IndexError):
                                    logger.warning(f"Could not convert end index to int: {points_subset[-1]}")
                                    end_idx = 0
                        
                        trend_line = {
                            'slope': slope,
                            'intercept': intercept,
                            'r_squared': r_squared,
                            'angle': angle_degrees,
                            'points': points_subset,
                            'touches': touches,
                            'start_idx': start_idx,
                            'end_idx': end_idx
                        }
                        valid_trend_lines.append(trend_line)
        
        return valid_trend_lines
    
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
        tolerance = df['close'].mean() * self.price_tolerance  # Use avg price for tolerance
        
        for i in range(len(df)):
            # Calculate trend line value at this index
            line_value = slope * i + intercept
            
            if line_type == 'bullish':
                # For bullish trend lines, price should touch from above
                if abs(df['low'].iloc[i] - line_value) <= tolerance:
                    touches += 1
            else:  # bearish
                # For bearish trend lines, price should touch from below
                if abs(df['high'].iloc[i] - line_value) <= tolerance:
                    touches += 1
        
        return touches
    
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
        Identify recent consolidation ranges for target calculation.
        
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
                        current_time = pd.to_datetime(current_time, unit='s')
                    except:
                        try:
                            current_time = pd.to_datetime(current_time, unit='ms')
                        except:
                            # If conversion fails, use current time
                            current_time = datetime.now()
                            logger.debug(f"Failed to convert timestamp, using current time instead")
            
            # Ensure last_update is a datetime object
            if not isinstance(last_update, datetime):
                logger.debug(f"Converting last_update from {type(last_update)} to datetime in consolidation ranges")
                if isinstance(last_update, (int, np.integer, float)):
                    try:
                        last_update = pd.to_datetime(last_update, unit='s')
                    except:
                        try:
                            last_update = pd.to_datetime(last_update, unit='ms')
                        except:
                            # If conversion fails, use a time far in the past to force update
                            last_update = datetime.now() - timedelta(hours=self.range_update_interval + 1)
                            logger.debug(f"Failed to convert last_update, forcing update in consolidation ranges")
                # Update the stored value
                self.last_updated[symbol] = last_update
            
            # Only update after significant time has passed based on timeframe
            try:
                time_diff = (current_time - last_update).total_seconds()
                if time_diff < self.range_update_interval * 3600:
                    logger.debug(f"🕒 Skipping consolidation range update for {symbol}, last update was {time_diff/3600:.1f} hours ago")
                    return
            except Exception as e:
                logger.warning(f"Error calculating time difference in consolidation ranges: {e}. Forcing update.")
                # Continue with update in case of error
        
        # Get last X bars based on timeframe profile
        recent_bars = df.iloc[-self.consolidation_length:]
        
        # Calculate range
        range_high = recent_bars['high'].max()
        range_low = recent_bars['low'].min()
        range_size = range_high - range_low
        
        # Store the range
        self.last_consolidation_ranges[symbol] = {
            'high': range_high,
            'low': range_low,
            'size': range_size
        }
        
        logger.debug(f"📏 Identified consolidation range for {symbol}: High={range_high:.5f}, Low={range_low:.5f}, Size={range_size:.5f}")
    
    def _process_retest_conditions(self, symbol: str, df: pd.DataFrame) -> None:
        """
        Process any pending retest conditions for breakout trades.
        
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
                    current_time = pd.to_datetime(current_time, unit='s')
                except:
                    try:
                        current_time = pd.to_datetime(current_time, unit='ms')
                    except:
                        # If conversion fails, use current time
                        current_time = datetime.now()
                        logger.debug(f"Failed to convert timestamp, using current time instead")
        
        # Ensure start_time is a datetime object
        if not isinstance(start_time, datetime):
            logger.debug(f"Converting start_time from {type(start_time)} to datetime in retest conditions")
            if isinstance(start_time, (int, np.integer, float)):
                try:
                    start_time = pd.to_datetime(start_time, unit='s')
                except:
                    try:
                        start_time = pd.to_datetime(start_time, unit='ms')
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
        # Get end candle that's max_retest_bars after the start
        try:
            time_diff = (current_time - start_time).total_seconds()
            max_time_allowed = self.max_retest_time * 3600  # Convert hours to seconds
            
            if time_diff > max_time_allowed:
                logger.debug(f"⌛ Retest condition expired for {symbol} after {time_diff/3600:.1f} hours (max: {max_time_allowed/3600:.1f})")
                self.retest_tracking[symbol] = None
                return
        except Exception as e:
            logger.warning(f"Error calculating time difference in retest condition: {e}")
            # Continue processing despite the error
            
        # Check if the price has retested the level
        current_price = df['close'].iloc[-1]
        price_tolerance = current_price * self.max_stop_pct

        # For breakout above resistance, we're looking for a retest from above
        if direction == 'bullish' and abs(current_price - level) <= price_tolerance and current_price > level:
            logger.info(f"✅ Confirmed bullish retest of {level:.5f} for {symbol}")
            # Update breakout tracking to indicate retest is confirmed
            self.retest_tracking[symbol]['retest_confirmed'] = True
            
        # For breakout below support, we're looking for a retest from below
        elif direction == 'bearish' and abs(current_price - level) <= price_tolerance and current_price < level:
            logger.info(f"✅ Confirmed bearish retest of {level:.5f} for {symbol}")
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
    
    
    def _check_breakout_signals(self, symbol: str, df: pd.DataFrame, h1_df: pd.DataFrame) -> List[Dict]:
        """
        Check for breakout signals at resistance and support levels, and trend lines.
        
        Args:
            symbol: Trading symbol
            df: Price dataframe (primary timeframe)
            h1_df: Higher timeframe dataframe
            
        Returns:
            List of breakout signals
        """
        signals = []
        
        # Skip if no levels available
        if symbol not in self.resistance_levels or symbol not in self.support_levels:
            logger.debug(f"⏩ {symbol}: No resistance or support levels available, skipping breakout check")
            return signals
            
        resistance_levels = self.resistance_levels[symbol]
        support_levels = self.support_levels[symbol]
        
        # Log a sample of recent candles for debugging
        candles_to_log = min(5, len(df))
        if candles_to_log > 0:
            logger.debug(f"🕯️ {symbol}: Last {candles_to_log} candles data sample:")
            for i in range(-candles_to_log, 0):
                candle = df.iloc[i]
                logger.debug(f"   {df.index[i]}: O={candle['open']:.5f}, H={candle['high']:.5f}, L={candle['low']:.5f}, C={candle['close']:.5f}, Vol={candle['tick_volume']}")
        
        # Get trend lines if available
        trend_lines = self.bullish_trend_lines.get(symbol, []) + self.bearish_trend_lines.get(symbol, [])
        bullish_trend_lines = [line for line in trend_lines if line['angle'] < self.trend_line_max_angle]
        bearish_trend_lines = [line for line in trend_lines if line['angle'] > -self.trend_line_max_angle]
        
        logger.debug(f"🔍 {symbol}: Found {len(bullish_trend_lines)} bullish and {len(bearish_trend_lines)} bearish trend lines")
        
        # Get recent candles - use candles_to_check from timeframe profile
        candles_to_check = min(self.candles_to_check, len(df) - 1)
        
        # Calculate volume threshold and analyze volume characteristics
        avg_volume_series = df['tick_volume'].rolling(window=20).mean()
        # Ensure we have a pandas Series
        if not isinstance(avg_volume_series, pd.Series):
            avg_volume_series = pd.Series(avg_volume_series, index=df.index[-20:])
        avg_volume = float(avg_volume_series.iloc[-1])
        volume_threshold = avg_volume * self.volume_threshold
        
        logger.debug(f"📊 {symbol}: Avg volume: {avg_volume:.1f}, threshold: {volume_threshold:.1f}")
        
        # Get higher timeframe trend
        h1_trend = self._determine_h1_trend(h1_df)
        logger.info(f"📈 {symbol}: H1 trend is {h1_trend}")
        
        # Check for retest confirmations first
        if (symbol in self.retest_tracking and 
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
                
                # Breakout condition: Previous candle below or at resistance, current candle closing above
                prev_below = previous_candle['close'] <= level * (1 + self.price_tolerance)
                curr_above = current_candle['close'] > level * (1 + self.price_tolerance)
                is_strong = self._is_strong_candle(current_candle)
                bull_volume = volume_quality > 0
                
                logger.debug(f"✓ {symbol}: Prev below: {prev_below}, Curr above: {curr_above}, Strong candle: {is_strong}, Bullish volume: {bull_volume}, H1 trend: {h1_trend}")
                
                if (prev_below and curr_above and is_strong and bull_volume and h1_trend == 'bullish'):
                    logger.info(f"⚡ {symbol}: Detected potential breakout above {level:.5f}")
                    
                    # Generate buy signal
                    entry_price = current_candle['close']
                    
                    # Place stop under the breakout candle's low
                    stop_loss = min(current_candle['low'], previous_candle['low'])
                    
                    logger.debug(f"📐 {symbol}: Entry: {entry_price:.5f}, Stop: {stop_loss:.5f}, Risk: {entry_price - stop_loss:.5f}")
                    
                    # Advanced target calculation
                    if symbol in self.last_consolidation_ranges:
                        range_size = self.last_consolidation_ranges[symbol]['size']
                        risk = entry_price - stop_loss
                        
                        logger.debug(f"📏 {symbol}: Using consolidation range size = {range_size:.5f} for target")
                        
                        calculated_target = level + range_size
                        min_target = entry_price + (risk * self.min_risk_reward)
                        take_profit = max(calculated_target, min_target)
                        
                        logger.debug(f"🎯 {symbol}: Target calculation - Range projection: {calculated_target:.5f}, Min RR: {min_target:.5f}, Using: {take_profit:.5f}")
                    else:
                        # Fallback to minimum risk-reward
                        risk = entry_price - stop_loss
                        take_profit = entry_price + (risk * self.min_risk_reward)
                        logger.debug(f"🎯 {symbol}: No range data, using min RR target = {take_profit:.5f}")
                    
                    # Reason with volume quality description
                    volume_desc = "strong bullish volume" if volume_quality > 1 else "adequate volume"
                    reason = f"Bullish breakout above resistance at {level:.5f} with {volume_desc}"
                    
                    # If retest is required, don't generate signal now but track for retest
                    if self.retest_required:
                        # Store breakout info for retest tracking
                        self.retest_tracking[symbol] = {
                            'level': level,
                            'direction': 'buy',
                            'entry_price': entry_price,
                            'stop_loss': stop_loss,
                            'start_time': df.index[i],
                            'confirmed': False,
                            'reason': reason
                        }
                        logger.info(f"👀 TRACKING RETEST: {symbol} bullish breakout at {level:.5f}")
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
                        logger.info(f"🟢 BREAKOUT BUY: {symbol} at {entry_price:.5f} | Level: {level:.5f} | SL: {stop_loss:.5f} | TP: {take_profit:.5f}")
                else:
                    if not prev_below:
                        logger.debug(f"❌ {symbol}: Previous candle not below resistance")
                    if not curr_above:
                        logger.debug(f"❌ {symbol}: Current candle not above resistance")
                    if not is_strong:
                        logger.debug(f"❌ {symbol}: Not a strong candle")
                    if not bull_volume:
                        logger.debug(f"❌ {symbol}: Insufficient bullish volume")
                    if h1_trend != 'bullish':
                        logger.debug(f"❌ {symbol}: H1 trend not bullish")
        
            # Check trend line breakouts (bullish)
            for trend_line in bearish_trend_lines:
                # Calculate trend line value at current and previous candle
                prev_line_value = self._calculate_trend_line_value(trend_line, i-1)
                curr_line_value = self._calculate_trend_line_value(trend_line, i)
                
                # Breakout condition: Previous candle below trend line, current candle closing above
                if (previous_candle['close'] <= prev_line_value * (1 + self.price_tolerance) and
                    current_candle['close'] > curr_line_value * (1 + self.price_tolerance) and
                    self._is_strong_candle(current_candle) and
                    volume_quality > 0 and  # Positive means bullish volume characteristics
                    h1_trend == 'bullish'):
                    
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
                        logger.info(f"👀 TRACKING RETEST: {symbol} bullish trend line breakout")
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
    
    def _check_reversal_signals(self, symbol: str, df: pd.DataFrame, h1_df: pd.DataFrame) -> List[Dict]:
        """
        Check for reversal signals at support and resistance levels.
        
        Args:
            symbol: Trading symbol
            df: Price dataframe (primary timeframe)
            h1_df: Higher timeframe dataframe
            
        Returns:
            List of reversal signals
        """
        signals = []
        
        # Get support and resistance levels
        if symbol not in self.resistance_levels or symbol not in self.support_levels:
            logger.debug(f"❓ No support/resistance levels available for {symbol}")
            return signals
            
        resistance_levels = self.resistance_levels[symbol]
        support_levels = self.support_levels[symbol]
        
        logger.debug(f"🔍 {symbol}: Checking reversals with {len(resistance_levels)} resistance and {len(support_levels)} support levels")
        
        # Get trend lines if available
        trend_lines = self.bullish_trend_lines.get(symbol, []) + self.bearish_trend_lines.get(symbol, [])
        bullish_trend_lines = [line for line in trend_lines if line['angle'] < self.trend_line_max_angle]
        bearish_trend_lines = [line for line in trend_lines if line['angle'] > -self.trend_line_max_angle]
        
        logger.debug(f"🔍 {symbol}: Found {len(bullish_trend_lines)} bullish and {len(bearish_trend_lines)} bearish trend lines for reversal checks")
        
        # Use candles_to_check from timeframe profile
        candles_to_check = min(self.candles_to_check, len(df) - 1)
        
        # Calculate volume threshold
        avg_volume_series = df['tick_volume'].rolling(window=20).mean()
        # Ensure we have a pandas Series
        if not isinstance(avg_volume_series, pd.Series):
            avg_volume_series = pd.Series(avg_volume_series, index=df.index[-20:])
        avg_volume = float(avg_volume_series.iloc[-1])
        volume_threshold = avg_volume * self.volume_threshold
        
        logger.debug(f"📊 {symbol}: Avg volume: {avg_volume:.1f}, threshold: {volume_threshold:.1f}")
        
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
                    
                    is_hammer = self._is_hammer(current_candle)
                    is_bullish_engulfing = self._is_bullish_engulfing(current_candle, previous_candle)
                    is_morning_star = self._is_morning_star(df, i)
                    
                    logger.debug(f"📈 {symbol}: Pattern checks - Hammer: {is_hammer}, Bullish Engulfing: {is_bullish_engulfing}, Morning Star: {is_morning_star}")
                    
                    if is_hammer:
                        pattern_type = "Hammer"
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
                    
                    is_hammer = self._is_hammer(current_candle)
                    is_bullish_engulfing = self._is_bullish_engulfing(current_candle, previous_candle)
                    is_morning_star = self._is_morning_star(df, i)
                    
                    logger.debug(f"📈 {symbol}: Pattern checks - Hammer: {is_hammer}, Bullish Engulfing: {is_bullish_engulfing}, Morning Star: {is_morning_star}")
                    
                    if is_hammer:
                        pattern_type = "Hammer"
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
                    
                    if self._is_shooting_star(current_candle):
                        pattern_type = "Shooting Star"
                    elif self._is_bearish_engulfing(current_candle, previous_candle):
                        pattern_type = "Bearish Engulfing"
                    elif self._is_evening_star(df, i):
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
                    
                    if self._is_shooting_star(current_candle):
                        pattern_type = "Shooting Star"
                    elif self._is_bearish_engulfing(current_candle, previous_candle):
                        pattern_type = "Bearish Engulfing"
                    elif self._is_evening_star(df, i):
                        pattern_type = "Evening Star"
                    
                    if pattern_type and volume_quality < 0:  # Bearish volume characteristics
                        # Generate sell signal
                        entry_price = current_candle['close']
                        
                        # Stop loss above the reversal candle high
                        stop_loss = current_candle['high'] + (line_value * self.price_tolerance)
                        
                        # Target: Either next support or at least 2x risk
                        risk = stop_loss - entry_price
                        
                        # Find nearest support below
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
            # First check if volume is even significant - using a less strict threshold
            # If candle volume is less than 60% of threshold, consider it insufficient
            volume_ratio = candle['tick_volume'] / threshold
            logger.debug(f"Volume ratio: {volume_ratio:.2f} (tick_volume: {candle['tick_volume']}, threshold: {threshold:.1f})")
            
            if volume_ratio < 0.6:  # More lenient check
                logger.debug(f"Insufficient volume: {candle['tick_volume']} < 60% of threshold {threshold:.1f}")
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
    
    def _is_morning_star(self, df: pd.DataFrame, idx: int) -> bool:
        """
        Check if candles form a morning star pattern (bullish reversal).
        Requires 3 candles.
        
        Args:
            df: Price dataframe
            idx: Index of the last candle in the pattern
            
        Returns:
            True if it's a morning star pattern
        """
        if idx < 2 or idx >= len(df):
            return False
            
        first = df.iloc[idx-2]  # First candle (bearish)
        second = df.iloc[idx-1]  # Middle candle (small)
        third = df.iloc[idx]    # Final candle (bullish)
        
        # First candle should be bearish (close < open)
        if first['close'] >= first['open']:
            return False
            
        # Third candle should be bullish (close > open)
        if third['close'] <= third['open']:
            return False
            
        # Second candle should be small
        second_body = abs(second['close'] - second['open'])
        first_body = abs(first['close'] - first['open'])
        third_body = abs(third['close'] - third['open'])
        
        # Second candle body should be smaller than both first and third
        if second_body >= first_body * 0.5 or second_body >= third_body * 0.5:
            return False
            
        # Third candle should close at least into the first candle's body
        if third['close'] < (first['open'] + first['close']) / 2:
            return False
            
        # Gap down between first and second
        if second['high'] <= first['low']:
            return True
            
        # If no gap, middle candle should have small body
        if second_body < first_body * 0.3 and second_body < third_body * 0.3:
            return True
            
        return False
    
    def _is_evening_star(self, df: pd.DataFrame, idx: int) -> bool:
        """
        Check if candles form an evening star pattern (bearish reversal).
        Requires 3 candles.
        
        Args:
            df: Price dataframe
            idx: Index of the last candle in the pattern
            
        Returns:
            True if it's an evening star pattern
        """
        if idx < 2 or idx >= len(df):
            return False
            
        first = df.iloc[idx-2]  # First candle (bullish)
        second = df.iloc[idx-1]  # Middle candle (small)
        third = df.iloc[idx]    # Final candle (bearish)
        
        # First candle should be bullish (close > open)
        if first['close'] <= first['open']:
            return False
            
        # Third candle should be bearish (close < open)
        if third['close'] >= third['open']:
            return False
            
        # Second candle should be small
        second_body = abs(second['close'] - second['open'])
        first_body = abs(first['close'] - first['open'])
        third_body = abs(third['close'] - third['open'])
        
        # Second candle body should be smaller than both first and third
        if second_body >= first_body * 0.5 or second_body >= third_body * 0.5:
            return False
            
        # Third candle should close at least into the first candle's body
        if third['close'] > (first['open'] + first['close']) / 2:
            return False
            
        # Gap up between first and second
        if second['low'] >= first['high']:
            return True
            
        # If no gap, middle candle should have small body
        if second_body < first_body * 0.3 and second_body < third_body * 0.3:
            return True
            
        return False
    
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
    
    def _is_hammer(self, candle: pd.Series) -> bool:
        """
        Check if a candle is a hammer pattern (bullish).
        
        Args:
            candle: Candle data
            
        Returns:
            True if it's a hammer
        """
        if candle['close'] <= candle['open']:
            return False  # Not bullish
            
        total_range = candle['high'] - candle['low']
        
        if total_range == 0:
            return False
            
        body = candle['close'] - candle['open']
        upper_wick = candle['high'] - candle['close']
        lower_wick = candle['open'] - candle['low']
        
        # Hammer criteria: small body, little/no upper wick, long lower wick
        body_percentage = body / total_range
        upper_wick_percentage = upper_wick / total_range
        lower_wick_percentage = lower_wick / total_range
        
        return bool(body_percentage < 0.3 and 
                upper_wick_percentage < 0.1 and 
                lower_wick_percentage > 0.6)
    
    def _is_shooting_star(self, candle: pd.Series) -> bool:
        """
        Check if a candle is a shooting star pattern (bearish).
        
        Args:
            candle: Candle data
            
        Returns:
            True if it's a shooting star
        """
        if candle['close'] >= candle['open']:
            return False  # Not bearish
            
        total_range = candle['high'] - candle['low']
        
        if total_range == 0:
            return False
            
        body = abs(candle['close'] - candle['open'])
        upper_wick = candle['high'] - candle['open']
        lower_wick = candle['close'] - candle['low']
        
        # Shooting star criteria: small body, long upper wick, little/no lower wick
        body_percentage = body / total_range
        upper_wick_percentage = upper_wick / total_range
        lower_wick_percentage = lower_wick / total_range
        
        return bool(body_percentage < 0.3 and 
                upper_wick_percentage > 0.6 and 
                lower_wick_percentage < 0.1)
    
    def _is_bullish_engulfing(self, current: pd.Series, previous: pd.Series) -> bool:
        """
        Check if current and previous candles form a bullish engulfing pattern.
        
        Args:
            current: Current candle data
            previous: Previous candle data
            
        Returns:
            True if it's a bullish engulfing pattern
        """
        # Previous candle must be bearish (close < open)
        if previous['close'] >= previous['open']:
            return False
            
        # Current candle must be bullish (close > open)
        if current['close'] <= current['open']:
            return False
            
        # Current candle body must engulf previous candle body
        return bool(current['open'] <= previous['close'] and 
                current['close'] >= previous['open'])
    
    def _is_bearish_engulfing(self, current: pd.Series, previous: pd.Series) -> bool:
        """
        Check if current and previous candles form a bearish engulfing pattern.
        
        Args:
            current: Current candle data
            previous: Previous candle data
            
        Returns:
            True if it's a bearish engulfing pattern
        """
        # Previous candle must be bullish (close > open)
        if previous['close'] <= previous['open']:
            return False
            
        # Current candle must be bearish (close < open)
        if current['close'] >= current['open']:
            return False
            
        # Current candle body must engulf previous candle body
        return bool(current['open'] >= previous['close'] and 
                current['close'] <= previous['open'])
    
    def _determine_h1_trend(self, h1_df: pd.DataFrame) -> str:
        """
        Determine the trend on the higher timeframe (H1).
        
        Args:
            h1_df: Higher timeframe dataframe
            
        Returns:
            'bullish', 'bearish', or 'neutral'
        """
        if len(h1_df) < 20:
            logger.debug(f"⚠️ Not enough data for trend determination, need 20 candles but got {len(h1_df)}")
            return 'neutral'
            
        # Use slope of EMA(20) to determine trend
        h1_df['ema20'] = h1_df['close'].ewm(span=20, adjust=False).mean()
        
        recent_ema = h1_df['ema20'].iloc[-5:]
        ema_slope = recent_ema.iloc[-1] - recent_ema.iloc[0]
        
        trend = 'neutral'
        if ema_slope > 0:
            trend = 'bullish'
        elif ema_slope < 0:
            trend = 'bearish'
            
        logger.debug(f"📊 Trend determined as {trend}, EMA slope: {ema_slope:.5f}")
        return trend
    
    async def close(self):
        """Close and clean up resources."""
        logger.info(f"🔌 Closing {self.name}")
        # No specific cleanup needed
        return True
