"""
Technical indicators module for trading strategies.

This module provides a comprehensive set of technical indicators for use in 
trading strategies, with optimized implementations for performance.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Union, Any, List

def calculate_moving_average(
    data: pd.DataFrame, 
    column: str = 'close', 
    period: int = 20, 
    ma_type: str = 'simple'
) -> Union[pd.Series, pd.DataFrame, np.ndarray, Any]:
    """
    Calculate moving average.
    
    Args:
        data: Price data with OHLCV columns
        column: Column name to use for calculation
        period: Period for moving average
        ma_type: 'simple', 'exponential', 'weighted', or 'hull'
        
    Returns:
        Series containing moving average values
    """
    if data is None or data.empty or len(data) < period:
        return pd.Series(dtype=float)
        
    if ma_type.lower() == 'simple':
        return data[column].rolling(window=period).mean()
    elif ma_type.lower() == 'exponential':
        return data[column].ewm(span=period, adjust=False).mean()
    elif ma_type.lower() == 'weighted':
        weights = np.arange(1, period + 1)
        return data[column].rolling(window=period).apply(
            lambda x: np.sum(weights * x) / np.sum(weights), raw=True
        )
    elif ma_type.lower() == 'hull':
        # Hull Moving Average: HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
        half_length = int(period / 2)
        sqrt_length = int(np.sqrt(period))
        
        # First WMA with period/2
        wma_half = calculate_moving_average(data, column, half_length, 'weighted')
        
        # Second WMA with full period
        wma_full = calculate_moving_average(data, column, period, 'weighted')
        
        # 2 * WMA(n/2) - WMA(n)
        raw_hma = 2 * wma_half - wma_full
        
        # Create a DataFrame for the final WMA calculation
        raw_hma_df = pd.DataFrame({column: raw_hma})
        
        # Final WMA with sqrt(period)
        return calculate_moving_average(raw_hma_df, column, sqrt_length, 'weighted')
    else:
        raise ValueError(f"Unknown MA type: {ma_type}")

def calculate_rsi(
    data: pd.DataFrame, 
    column: str = 'close', 
    period: int = 14
) -> Union[pd.Series, pd.DataFrame, np.ndarray, Any]:
    """
    Calculate Relative Strength Index.
    
    Args:
        data: Price data with OHLCV columns
        column: Column name to use for calculation
        period: RSI period
        
    Returns:
        Series containing RSI values
    """
    if data is None or data.empty or len(data) < period + 1:
        return pd.Series(dtype=float)
        
    delta = data[column].diff()
    
    # Make two series: one for gains, one for losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # First average gain and loss
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # Subsequent average gain and loss
    # Convert to pandas Series if it's a numpy array
    if isinstance(avg_gain, np.ndarray):
        avg_gain = pd.Series(avg_gain)
    if isinstance(avg_loss, np.ndarray):
        avg_loss = pd.Series(avg_loss)
        
    for i in range(period, len(gain)):
        if hasattr(avg_gain, 'iloc'):
            avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period-1) + gain.iloc[i]) / period
            avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period-1) + loss.iloc[i]) / period
        else:
            # Fallback for numpy arrays
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_macd(
    data: pd.DataFrame, 
    column: str = 'close', 
    fast: int = 12, 
    slow: int = 26, 
    signal: int = 9
) -> Tuple[Union[pd.Series, pd.DataFrame, np.ndarray, Any], 
           Union[pd.Series, pd.DataFrame, np.ndarray, Any], 
           Union[pd.Series, pd.DataFrame, np.ndarray, Any]]:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        data: Price data with OHLCV columns
        column: Column name to use for calculation
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line period
        
    Returns:
        Tuple containing (MACD line, Signal line, Histogram)
    """
    if data is None or data.empty or len(data) < slow + signal:
        empty_series = pd.Series(dtype=float)
        return empty_series, empty_series, empty_series
        
    # Calculate EMAs
    fast_ema = data[column].ewm(span=fast, adjust=False).mean()
    slow_ema = data[column].ewm(span=slow, adjust=False).mean()
    
    # Calculate MACD line and signal line
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    
    # Calculate histogram
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(
    data: pd.DataFrame, 
    column: str = 'close', 
    period: int = 20, 
    std_dev: float = 2.0
) -> Tuple[Union[pd.Series, pd.DataFrame, np.ndarray, Any], 
           Union[pd.Series, pd.DataFrame, np.ndarray, Any], 
           Union[pd.Series, pd.DataFrame, np.ndarray, Any]]:
    """
    Calculate Bollinger Bands.
    
    Args:
        data: Price data with OHLCV columns
        column: Column name to use for calculation
        period: Period for moving average
        std_dev: Number of standard deviations
        
    Returns:
        Tuple containing (Upper Band, Middle Band, Lower Band)
    """
    if data is None or data.empty or len(data) < period:
        empty_series = pd.Series(dtype=float)
        return empty_series, empty_series, empty_series
        
    # Calculate middle band (SMA)
    middle_band = data[column].rolling(window=period).mean()
    
    # Calculate standard deviation
    rolling_std = data[column].rolling(window=period).std()
    
    # Calculate upper and lower bands
    upper_band = middle_band + (rolling_std * std_dev)
    lower_band = middle_band - (rolling_std * std_dev)
    
    return upper_band, middle_band, lower_band

def calculate_atr(
    data: pd.DataFrame, 
    period: int = 14
) -> Union[pd.Series, pd.DataFrame, np.ndarray, float, Any]:
    """
    Calculate Average True Range.
    
    Args:
        data: Price data with 'high', 'low', 'close' columns
        period: ATR period
        
    Returns:
        Series containing ATR values
    """
    if data is None or data.empty or len(data) < period + 1:
        return pd.Series(dtype=float)
    
    # Calculate true range
    high_low = data['high'] - data['low']
    high_close_prev = abs(data['high'] - data['close'].shift(1))
    low_close_prev = abs(data['low'] - data['close'].shift(1))
    
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    
    # Calculate ATR
    atr = true_range.rolling(window=period).mean()
    
    return atr

def calculate_stochastic(
    data: pd.DataFrame, 
    k_period: int = 14, 
    d_period: int = 3, 
    slowing: int = 3
) -> Tuple[Union[pd.Series, pd.DataFrame, np.ndarray, Any], 
           Union[pd.Series, pd.DataFrame, np.ndarray, Any]]:
    """
    Calculate Stochastic Oscillator.
    
    Args:
        data: Price data with 'high', 'low', 'close' columns
        k_period: K period
        d_period: D period
        slowing: Slowing period
        
    Returns:
        Tuple containing (K line, D line)
    """
    if data is None or data.empty or len(data) < k_period + d_period:
        empty_series = pd.Series(dtype=float)
        return empty_series, empty_series
    
    # Calculate %K
    lowest_low = data['low'].rolling(window=k_period).min()
    highest_high = data['high'].rolling(window=k_period).max()
    
    k_fast = 100 * ((data['close'] - lowest_low) / (highest_high - lowest_low))
    
    # Apply slowing if needed
    if slowing > 1:
        k = k_fast.rolling(window=slowing).mean()
    else:
        k = k_fast
    
    # Calculate %D
    d = k.rolling(window=d_period).mean()
    
    return k, d

def calculate_adx(
    data: pd.DataFrame, 
    period: int = 14
) -> Tuple[Union[pd.Series, pd.DataFrame, np.ndarray, Any], 
           Union[pd.Series, pd.DataFrame, np.ndarray, Any], 
           Union[pd.Series, pd.DataFrame, np.ndarray, Any]]:
    """
    Calculate Average Directional Index.
    
    Args:
        data: Price data with 'high', 'low', 'close' columns
        period: ADX period
        
    Returns:
        Tuple containing (ADX, +DI, -DI)
    """
    if data is None or data.empty or len(data) < period * 2:
        empty_series = pd.Series(dtype=float)
        return empty_series, empty_series, empty_series
        
    # Ensure we're working with pandas DataFrame/Series
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
        
    # Calculate True Range
    data['tr0'] = abs(data['high'] - data['low'])
    data['tr1'] = abs(data['high'] - data['close'].shift(1))
    data['tr2'] = abs(data['low'] - data['close'].shift(1))
    data['tr'] = data[['tr0', 'tr1', 'tr2']].max(axis=1)
    
    # Calculate +DM and -DM
    data['up_move'] = data['high'] - data['high'].shift(1)
    data['down_move'] = data['low'].shift(1) - data['low']
    
    data['+dm'] = ((data['up_move'] > data['down_move']) & (data['up_move'] > 0)) * data['up_move']
    data['-dm'] = ((data['down_move'] > data['up_move']) & (data['down_move'] > 0)) * data['down_move']
    
    # Calculate smoothed TR, +DM, and -DM
    data['smoothed_tr'] = data['tr'].rolling(window=period).sum()
    data['smoothed_+dm'] = data['+dm'].rolling(window=period).sum()
    data['smoothed_-dm'] = data['-dm'].rolling(window=period).sum()
    
    # Calculate +DI and -DI
    data['+di'] = 100 * data['smoothed_+dm'] / data['smoothed_tr']
    data['-di'] = 100 * data['smoothed_-dm'] / data['smoothed_tr']
    
    # Calculate DX and ADX
    data['dx'] = 100 * abs(data['+di'] - data['-di']) / (data['+di'] + data['-di'])
    
    # If dx is a numpy array, convert to Series for rolling()
    if isinstance(data['dx'], np.ndarray):
        data['dx'] = pd.Series(data['dx'])
        
    data['adx'] = data['dx'].rolling(window=period).mean()
    
    return data['adx'], data['+di'], data['-di']

def calculate_ichimoku(
    data: pd.DataFrame,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_span_b_period: int = 52,
    displacement: int = 26
) -> Tuple[Union[pd.Series, pd.DataFrame, np.ndarray, Any], 
           Union[pd.Series, pd.DataFrame, np.ndarray, Any], 
           Union[pd.Series, pd.DataFrame, np.ndarray, Any], 
           Union[pd.Series, pd.DataFrame, np.ndarray, Any], 
           Union[pd.Series, pd.DataFrame, np.ndarray, Any]]:
    """
    Calculate Ichimoku Cloud.
    
    Args:
        data: Price data with 'high', 'low', 'close' columns
        tenkan_period: Tenkan-sen (Conversion Line) period
        kijun_period: Kijun-sen (Base Line) period
        senkou_span_b_period: Senkou Span B (Leading Span B) period
        displacement: Displacement for Senkou Span A/B forward and Chikou Span backward
        
    Returns:
        Tuple containing (Tenkan-sen, Kijun-sen, Senkou Span A, Senkou Span B, Chikou Span)
    """
    if data is None or data.empty or len(data) < max(tenkan_period, kijun_period, senkou_span_b_period) + displacement:
        empty_series = pd.Series(dtype=float)
        return empty_series, empty_series, empty_series, empty_series, empty_series
    
    # Tenkan-sen (Conversion Line): (Highest High + Lowest Low) / 2 for tenkan_period
    tenkan_sen = (data['high'].rolling(window=tenkan_period).max() + 
                 data['low'].rolling(window=tenkan_period).min()) / 2
    
    # Kijun-sen (Base Line): (Highest High + Lowest Low) / 2 for kijun_period
    kijun_sen = (data['high'].rolling(window=kijun_period).max() + 
                data['low'].rolling(window=kijun_period).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2, shifted forward by displacement
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
    
    # Senkou Span B (Leading Span B): (Highest High + Lowest Low) / 2 for senkou_span_b_period, shifted forward by displacement
    senkou_span_b = ((data['high'].rolling(window=senkou_span_b_period).max() + 
                     data['low'].rolling(window=senkou_span_b_period).min()) / 2).shift(displacement)
    
    # Chikou Span (Lagging Span): Close price, shifted backward by displacement
    chikou_span = data['close'].shift(-displacement)
    
    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span

def calculate_pivot_points(
    data: pd.DataFrame,
    method: str = 'standard'
) -> Tuple[Union[pd.Series, float], Union[pd.Series, float], Union[pd.Series, float],
          Union[pd.Series, float], Union[pd.Series, float], Union[pd.Series, float], 
          Union[pd.Series, float]]:
    """
    Calculate pivot points for the given data.
    
    Args:
        data: Price data with OHLCV columns from the previous period (day, week, month)
        method: Pivot point calculation method ('standard', 'fibonacci', 'woodie', 'classic')
        
    Returns:
        Tuple containing (Pivot Point, Support 1, Support 2, Support 3, Resistance 1, Resistance 2, Resistance 3)
    """
    if data is None or data.empty:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    # Ensure we have a DataFrame
    if not isinstance(data, pd.DataFrame):
        try:
            data = pd.DataFrame(data)
        except Exception as e:
            print(f"Error converting data to DataFrame: {e}")
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    # Get last row values
    if isinstance(data.index, pd.DatetimeIndex):
        # Sort by date if datetime index
        data = data.sort_index()
    
    # Initialize default value for open_price
    open_price = 0.0
        
    try:
        high = float(data['high'].iloc[-1])
        low = float(data['low'].iloc[-1])
        close = float(data['close'].iloc[-1])
        
        # Add open price for some methods
        if method.lower() in ['woodie']:
            if 'open' in data.columns:
                open_price = float(data['open'].iloc[-1])
            else:
                # Fallback to close if open is not available
                open_price = close
    except (IndexError, KeyError, ValueError, AttributeError) as e:
        print(f"Error accessing data values: {e}")
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    # Calculate pivot point based on method
    if method.lower() == 'standard':
        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        r3 = high + 2 * (pivot - low)
        s3 = low - 2 * (high - pivot)
        
    elif method.lower() == 'fibonacci':
        pivot = (high + low + close) / 3
        r1 = pivot + 0.382 * (high - low)
        s1 = pivot - 0.382 * (high - low)
        r2 = pivot + 0.618 * (high - low)
        s2 = pivot - 0.618 * (high - low)
        r3 = pivot + 1.0 * (high - low)
        s3 = pivot - 1.0 * (high - low)
        
    elif method.lower() == 'woodie':
        pivot = (high + low + 2 * open_price) / 4
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        r3 = r1 + (high - low)
        s3 = s1 - (high - low)
        
    elif method.lower() == 'classic':
        pivot = (high + low + close) / 3
        r1 = (2 * pivot) - low
        s1 = (2 * pivot) - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        r3 = r1 + (high - low)
        s3 = s1 - (high - low)
        
    else:
        # Default to standard if invalid method
        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        r3 = high + 2 * (pivot - low)
        s3 = low - 2 * (high - pivot)
    
    return pivot, s1, s2, s3, r1, r2, r3 