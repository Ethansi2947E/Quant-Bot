"""
This module provides a collection of functions for detecting various
candlestick patterns in financial time series data. Each function typically
takes a pandas DataFrame with OHLC (Open, High, Low, Close) data as input
and returns a pandas Series of booleans indicating the occurrence of the
specified pattern.

These functions are designed to be vectorized for efficiency and can be
used as building blocks in trading strategies or technical analysis tools.
"""
import pandas as pd
import numpy as np
from typing import Optional

def detect_doji(df: pd.DataFrame, body_max_percent: float = 0.05) -> pd.Series:
    """Detects Doji candles.

    A Doji is a candle where the open and close prices are very close,
    indicating indecision in the market.

    Args:
        df (pd.DataFrame): DataFrame with 'open', 'high', 'low', 'close' columns.
        body_max_percent (float): Maximum percentage of the total range
                                  that the body can occupy to be considered a Doji.
                                  Defaults to 0.05 (5%).

    Returns:
        pd.Series: Boolean series indicating Doji occurrences.
    """
    total_range = df['high'] - df['low']
    body_size = abs(df['close'] - df['open'])
    
    # Avoid division by zero for zero-range candles
    is_doji = np.where(total_range > 0.00001, (body_size / total_range) <= body_max_percent, False)
    
    return pd.Series(is_doji, index=df.index)

def detect_hammer(
    df: pd.DataFrame, 
    body_max_percent: float = 0.33, 
    lower_wick_min_ratio: float = 2.0, 
    upper_wick_max_ratio: float = 1.0,
    trend_lookback: int = 0
) -> pd.Series:
    """Detects Hammer candles.

    A Hammer is a bullish reversal pattern that forms during a downtrend.
    It has a small body, a long lower wick, and a short or absent upper wick.

    Args:
        df (pd.DataFrame): DataFrame with 'open', 'high', 'low', 'close' columns.
        body_max_percent (float): Maximum percentage of the total range for the body.
                                  Defaults to 0.33.
        lower_wick_min_ratio (float): Minimum ratio of the lower wick to the body size.
                                      Defaults to 2.0.
        upper_wick_max_ratio (float): Maximum ratio of the upper wick to the body size.
                                      Defaults to 1.0.
        trend_lookback (int): Number of preceding candles to check for a downtrend.
                              If 0, trend is not considered. Defaults to 0.

    Returns:
        pd.Series: Boolean series indicating Hammer occurrences.
    """
    body_size = abs(df['close'] - df['open'])
    total_range = df['high'] - df['low']
    
    lower_wick = np.minimum(df['open'], df['close']) - df['low']
    upper_wick = df['high'] - np.maximum(df['open'], df['close'])

    is_small_body = np.where(total_range > 0.00001, (body_size / total_range) <= body_max_percent, False)
    is_long_lower_wick = np.where(body_size > 0.00001, lower_wick >= (body_size * lower_wick_min_ratio), lower_wick > 0.00001) # If body is tiny, any lower wick counts
    is_short_upper_wick = np.where(body_size > 0.00001, upper_wick <= (body_size * upper_wick_max_ratio), True) # If body is tiny, upper wick condition is relaxed
    
    condition_hammer = is_small_body & is_long_lower_wick & is_short_upper_wick
    
    if trend_lookback > 0 and len(df) > trend_lookback:
        # Simple downtrend: current low is lower than the low 'trend_lookback' bars ago
        # More sophisticated trend detection could be added here or handled by the strategy
        is_downtrend = df['low'].shift(1) < df['low'].shift(trend_lookback + 1)
        # Ensure is_downtrend is aligned and has same length, fill NaNs from shift
        is_downtrend = is_downtrend.fillna(False)
        condition_hammer = condition_hammer & is_downtrend
        
    return pd.Series(condition_hammer, index=df.index)

def detect_shooting_star(
    df: pd.DataFrame, 
    body_max_percent: float = 0.33, 
    upper_wick_min_ratio: float = 2.0, 
    lower_wick_max_ratio: float = 1.0,
    trend_lookback: int = 0
) -> pd.Series:
    """Detects Shooting Star candles.

    A Shooting Star is a bearish reversal pattern that forms during an uptrend.
    It has a small body, a long upper wick, and a short or absent lower wick.

    Args:
        df (pd.DataFrame): DataFrame with 'open', 'high', 'low', 'close' columns.
        body_max_percent (float): Maximum percentage of the total range for the body.
                                  Defaults to 0.33.
        upper_wick_min_ratio (float): Minimum ratio of the upper wick to the body size.
                                      Defaults to 2.0.
        lower_wick_max_ratio (float): Maximum ratio of the lower wick to the body size.
                                      Defaults to 1.0.
        trend_lookback (int): Number of preceding candles to check for an uptrend.
                              If 0, trend is not considered. Defaults to 0.

    Returns:
        pd.Series: Boolean series indicating Shooting Star occurrences.
    """
    body_size = abs(df['close'] - df['open'])
    total_range = df['high'] - df['low']
    
    lower_wick = np.minimum(df['open'], df['close']) - df['low']
    upper_wick = df['high'] - np.maximum(df['open'], df['close'])

    is_small_body = np.where(total_range > 0.00001, (body_size / total_range) <= body_max_percent, False)
    is_long_upper_wick = np.where(body_size > 0.00001, upper_wick >= (body_size * upper_wick_min_ratio), upper_wick > 0.00001)
    is_short_lower_wick = np.where(body_size > 0.00001, lower_wick <= (body_size * lower_wick_max_ratio), True)

    condition_shooting_star = is_small_body & is_long_upper_wick & is_short_lower_wick
    
    if trend_lookback > 0 and len(df) > trend_lookback:
        # Simple uptrend: current high is higher than the high 'trend_lookback' bars ago
        is_uptrend = df['high'].shift(1) > df['high'].shift(trend_lookback + 1)
        is_uptrend = is_uptrend.fillna(False)
        condition_shooting_star = condition_shooting_star & is_uptrend
        
    return pd.Series(condition_shooting_star, index=df.index)

def detect_bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    """Detects Bullish Engulfing patterns.

    A Bullish Engulfing pattern occurs when a small bearish candle is
    followed by a larger bullish candle that completely engulfs the
    previous candle's body.

    Args:
        df (pd.DataFrame): DataFrame with 'open', 'high', 'low', 'close' columns.

    Returns:
        pd.Series: Boolean series indicating Bullish Engulfing occurrences.
    """
    prev_close = df['close'].shift(1)
    prev_open = df['open'].shift(1)
    
    curr_close = df['close']
    curr_open = df['open']
    
    # Previous candle is bearish
    prev_is_bearish = prev_close < prev_open
    # Current candle is bullish
    curr_is_bullish = curr_close > curr_open
    
    # Current candle engulfs previous candle's body
    engulfs_body = (curr_open < prev_close) & (curr_close > prev_open)
    
    condition_bullish_engulfing = prev_is_bearish & curr_is_bullish & engulfs_body
    
    return pd.Series(condition_bullish_engulfing, index=df.index).fillna(False)

def detect_bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    """Detects Bearish Engulfing patterns.

    A Bearish Engulfing pattern occurs when a small bullish candle is
    followed by a larger bearish candle that completely engulfs the
    previous candle's body.

    Args:
        df (pd.DataFrame): DataFrame with 'open', 'high', 'low', 'close' columns.

    Returns:
        pd.Series: Boolean series indicating Bearish Engulfing occurrences.
    """
    prev_close = df['close'].shift(1)
    prev_open = df['open'].shift(1)
    
    curr_close = df['close']
    curr_open = df['open']
    
    # Previous candle is bullish
    prev_is_bullish = prev_close > prev_open
    # Current candle is bearish
    curr_is_bearish = curr_close < curr_open
    
    # Current candle engulfs previous candle's body
    engulfs_body = (curr_open > prev_close) & (curr_close < prev_open)
    
    condition_bearish_engulfing = prev_is_bullish & curr_is_bearish & engulfs_body
    
    return pd.Series(condition_bearish_engulfing, index=df.index).fillna(False)

def detect_inside_bar(df: pd.DataFrame) -> pd.Series:
    """Detects Inside Bar patterns.

    An Inside Bar is a candle whose entire range (high to low) is
    contained within the range of the previous candle.

    Args:
        df (pd.DataFrame): DataFrame with 'open', 'high', 'low', 'close' columns.

    Returns:
        pd.Series: Boolean series indicating Inside Bar occurrences.
    """
    prev_high = df['high'].shift(1)
    prev_low = df['low'].shift(1)
    
    curr_high = df['high']
    curr_low = df['low']
    
    is_inside = (curr_high < prev_high) & (curr_low > prev_low)
    
    return pd.Series(is_inside, index=df.index).fillna(False)

def detect_bullish_harami(df: pd.DataFrame, c1_body_min_percent: float = 0.3, c2_body_max_percent: float = 0.5) -> pd.Series:
    """Detects Bullish Harami patterns.

    A Bullish Harami is a two-candle pattern where a large bearish candle (C1)
    is followed by a small bullish candle (C2) whose body is contained
    within the body of C1. It suggests a potential bullish reversal.

    Args:
        df (pd.DataFrame): DataFrame with 'open', 'high', 'low', 'close' columns.
        c1_body_min_percent (float): Minimum percentage of C1's range that its body must occupy.
                                     Defaults to 0.3.
        c2_body_max_percent (float): Maximum percentage of C1's body that C2's body can occupy.
                                     Defaults to 0.5.

    Returns:
        pd.Series: Boolean series indicating Bullish Harami occurrences.
    """
    c1_open = df['open'].shift(1)
    c1_close = df['close'].shift(1)
    c1_high = df['high'].shift(1)
    c1_low = df['low'].shift(1)
    c1_body = abs(c1_open - c1_close)
    c1_range = c1_high - c1_low
    
    c2_open = df['open']
    c2_close = df['close']
    c2_body = abs(c2_open - c2_close)
    
    # C1 is bearish and has a significant body
    c1_is_bearish = c1_close < c1_open
    c1_has_min_body = np.where(c1_range > 0.00001, (c1_body / c1_range) >= c1_body_min_percent, False)
    
    # C2 is bullish and has a small body relative to C1's body
    c2_is_bullish = c2_close > c2_open
    c2_is_small_body = np.where(c1_body > 0.00001, (c2_body / c1_body) <= c2_body_max_percent, False) 
                                 # If C1 body is tiny, this check becomes less meaningful, might need refinement
    
    # C2 body is contained within C1 body
    c2_body_inside_c1_body = (c2_close <= c1_open) & (c2_open >= c1_close)
    
    condition_bullish_harami = c1_is_bearish & c1_has_min_body & \
                               c2_is_bullish & c2_is_small_body & \
                               c2_body_inside_c1_body
                               
    return pd.Series(condition_bullish_harami, index=df.index).fillna(False)

def detect_bearish_harami(df: pd.DataFrame, c1_body_min_percent: float = 0.3, c2_body_max_percent: float = 0.5) -> pd.Series:
    """Detects Bearish Harami patterns.

    A Bearish Harami is a two-candle pattern where a large bullish candle (C1)
    is followed by a small bearish candle (C2) whose body is contained
    within the body of C1. It suggests a potential bearish reversal.

    Args:
        df (pd.DataFrame): DataFrame with 'open', 'high', 'low', 'close' columns.
        c1_body_min_percent (float): Minimum percentage of C1's range that its body must occupy.
                                     Defaults to 0.3.
        c2_body_max_percent (float): Maximum percentage of C1's body that C2's body can occupy.
                                     Defaults to 0.5.

    Returns:
        pd.Series: Boolean series indicating Bearish Harami occurrences.
    """
    c1_open = df['open'].shift(1)
    c1_close = df['close'].shift(1)
    c1_high = df['high'].shift(1)
    c1_low = df['low'].shift(1)
    c1_body = abs(c1_open - c1_close)
    c1_range = c1_high - c1_low
    
    c2_open = df['open']
    c2_close = df['close']
    c2_body = abs(c2_open - c2_close)
    
    # C1 is bullish and has a significant body
    c1_is_bullish = c1_close > c1_open
    c1_has_min_body = np.where(c1_range > 0.00001, (c1_body / c1_range) >= c1_body_min_percent, False)
    
    # C2 is bearish and has a small body relative to C1's body
    c2_is_bearish = c2_close < c2_open
    c2_is_small_body = np.where(c1_body > 0.00001, (c2_body / c1_body) <= c2_body_max_percent, False)
    
    # C2 body is contained within C1 body
    c2_body_inside_c1_body = (c2_close >= c1_open) & (c2_open <= c1_close) # Corrected logic for bearish harami
    
    condition_bearish_harami = c1_is_bullish & c1_has_min_body & \
                               c2_is_bearish & c2_is_small_body & \
                               c2_body_inside_c1_body
                               
    return pd.Series(condition_bearish_harami, index=df.index).fillna(False)

def detect_morning_star(
    df: pd.DataFrame, 
    star_body_max_percent_of_range: float = 0.3, # Max body of star relative to its own range
    c1_body_min_percent_of_range: float = 0.3, # Min body of C1 relative to its own range
    c3_body_min_percent_of_range: float = 0.3, # Min body of C3 relative to its own range
    c3_closes_into_c1_body_min_percent: float = 0.5, # C3 must close at least this much into C1's body
    c1_c2_gap_down_percent: float = 0.0, # Min % C2 must gap down from C1 (0 for no gap needed)
    c2_c3_gap_up_percent: float = 0.0,    # Min % C3 must gap up from C2 (0 for no gap needed)
    star_body_max_percent_c1_body: Optional[float] = None # Optional: Max body of C2 relative to C1's body
) -> pd.Series:
    """Detects Morning Star patterns.

    A Morning Star is a three-candle bullish reversal pattern:
    1. C1: A bearish candle (preferably long).
    2. C2: A small-bodied candle (star) that gaps down from C1.
       The star can be bullish, bearish, or neutral (Doji).
    3. C3: A bullish candle (preferably long) that gaps up from C2
       and closes well into C1's body.

    Args:
        df (pd.DataFrame): DataFrame with 'open', 'high', 'low', 'close' columns.
        star_body_max_percent_of_range (float): Max body of C2 (star) relative to C2's range.
                                                Defaults to 0.3.
        c1_body_min_percent_of_range (float): Min body of C1 relative to C1's range.
                                              Defaults to 0.3.
        c3_body_min_percent_of_range (float): Min body of C3 relative to C3's range.
                                              Defaults to 0.3.
        c3_closes_into_c1_body_min_percent (float): Min percentage C3 must close into C1's body.
                                                    Defaults to 0.5 (midpoint).
        c1_c2_gap_down_percent (float): Minimum percentage C2's high must be below C1's low.
                                        Defaults to 0.0 (no strict gap required).
        c2_c3_gap_up_percent (float): Minimum percentage C3's low must be above C2's high.
                                        Defaults to 0.0 (no strict gap required).
        star_body_max_percent_c1_body (Optional[float]): Optional. Max body of C2 (star) relative to C1's body.
                                                        If provided, this overrides star_body_max_percent_of_range.
                                                        Defaults to None.

    Returns:
        pd.Series: Boolean series indicating Morning Star occurrences.
    """
    c3_open, c3_high, c3_low, c3_close = df['open'], df['high'], df['low'], df['close']
    c2_open, c2_high, c2_low, c2_close = df['open'].shift(1), df['high'].shift(1), df['low'].shift(1), df['close'].shift(1)
    c1_open, c1_high, c1_low, c1_close = df['open'].shift(2), df['high'].shift(2), df['low'].shift(2), df['close'].shift(2)

    c1_body = abs(c1_open - c1_close)
    c1_range = c1_high - c1_low
    c2_body = abs(c2_open - c2_close)
    c2_range = c2_high - c2_low
    c3_body = abs(c3_open - c3_close)
    c3_range = c3_high - c3_low

    # C1 is bearish and has a decent body
    c1_is_bearish = c1_close < c1_open
    c1_decent_body = np.where(c1_range > 0.00001, (c1_body / c1_range) >= c1_body_min_percent_of_range, False)

    # C2 (star) has a small body
    if star_body_max_percent_c1_body is not None:
        c2_is_small_body = np.where(c1_body > 0.00001, (c2_body / c1_body) <= star_body_max_percent_c1_body, c2_body < 0.00001) # if c1_body is zero, star must also be zero/tiny
    else:
        c2_is_small_body = np.where(c2_range > 0.00001, (c2_body / c2_range) <= star_body_max_percent_of_range, True) # True if range is 0 (like a perfect Doji)
    
    # C3 is bullish and has a decent body
    c3_is_bullish = c3_close > c3_open
    c3_decent_body = np.where(c3_range > 0.00001, (c3_body / c3_range) >= c3_body_min_percent_of_range, False)

    # Gap conditions (optional based on parameters)
    # C2 gaps down from C1: C2_high < C1_low
    c1_c2_gaps_down = c2_high < (c1_low - c1_range * c1_c2_gap_down_percent)
    # C3 gaps up from C2: C3_low > C2_high
    c2_c3_gaps_up = c3_low > (c2_high + c2_range * c2_c3_gap_up_percent)

    # C3 closes well into C1's body
    c3_closes_in_c1 = c3_close > (c1_close + c1_body * c3_closes_into_c1_body_min_percent)
    
    condition_morning_star = c1_is_bearish & c1_decent_body & \
                             c2_is_small_body & \
                             c1_c2_gaps_down & \
                             c3_is_bullish & c3_decent_body & \
                             c2_c3_gaps_up & \
                             c3_closes_in_c1

    return pd.Series(condition_morning_star, index=df.index).fillna(False)

def detect_evening_star(
    df: pd.DataFrame, 
    star_body_max_percent_of_range: float = 0.3,
    c1_body_min_percent_of_range: float = 0.3,
    c3_body_min_percent_of_range: float = 0.3,
    c3_closes_into_c1_body_min_percent: float = 0.5,
    c1_c2_gap_up_percent: float = 0.0,
    c2_c3_gap_down_percent: float = 0.0,
    star_body_max_percent_c1_body: Optional[float] = None # Optional: Max body of C2 relative to C1's body
) -> pd.Series:
    """Detects Evening Star patterns.

    An Evening Star is a three-candle bearish reversal pattern:
    1. C1: A bullish candle (preferably long).
    2. C2: A small-bodied candle (star) that gaps up from C1.
       The star can be bullish, bearish, or neutral (Doji).
    3. C3: A bearish candle (preferably long) that gaps down from C2
       and closes well into C1's body.

    Args:
        df (pd.DataFrame): DataFrame with 'open', 'high', 'low', 'close' columns.
        star_body_max_percent_of_range (float): Max body of C2 (star) relative to C2's range.
                                                Defaults to 0.3.
        c1_body_min_percent_of_range (float): Min body of C1 relative to C1's range.
                                              Defaults to 0.3.
        c3_body_min_percent_of_range (float): Min body of C3 relative to C3's range.
                                              Defaults to 0.3.
        c3_closes_into_c1_body_min_percent (float): Min percentage C3 must close into C1's body.
                                                    Defaults to 0.5 (midpoint).
        c1_c2_gap_up_percent (float): Minimum percentage C2's low must be above C1's high.
                                      Defaults to 0.0 (no strict gap required).
        c2_c3_gap_down_percent (float): Minimum percentage C3's high must be below C2's low.
                                        Defaults to 0.0 (no strict gap required).
        star_body_max_percent_c1_body (Optional[float]): Optional. Max body of C2 (star) relative to C1's body.
                                                        If provided, this overrides star_body_max_percent_of_range.
                                                        Defaults to None.

    Returns:
        pd.Series: Boolean series indicating Evening Star occurrences.
    """
    c3_open, c3_high, c3_low, c3_close = df['open'], df['high'], df['low'], df['close']
    c2_open, c2_high, c2_low, c2_close = df['open'].shift(1), df['high'].shift(1), df['low'].shift(1), df['close'].shift(1)
    c1_open, c1_high, c1_low, c1_close = df['open'].shift(2), df['high'].shift(2), df['low'].shift(2), df['close'].shift(2)

    c1_body = abs(c1_open - c1_close)
    c1_range = c1_high - c1_low
    c2_body = abs(c2_open - c2_close)
    c2_range = c2_high - c2_low
    c3_body = abs(c3_open - c3_close)
    c3_range = c3_high - c3_low

    # C1 is bullish and has a decent body
    c1_is_bullish = c1_close > c1_open
    c1_decent_body = np.where(c1_range > 0.00001, (c1_body / c1_range) >= c1_body_min_percent_of_range, False)

    # C2 (star) has a small body
    if star_body_max_percent_c1_body is not None:
        c2_is_small_body = np.where(c1_body > 0.00001, (c2_body / c1_body) <= star_body_max_percent_c1_body, c2_body < 0.00001) # if c1_body is zero, star must also be zero/tiny
    else:
        c2_is_small_body = np.where(c2_range > 0.00001, (c2_body / c2_range) <= star_body_max_percent_of_range, True)

    # C3 is bearish and has a decent body
    c3_is_bearish = c3_close < c3_open
    c3_decent_body = np.where(c3_range > 0.00001, (c3_body / c3_range) >= c3_body_min_percent_of_range, False)

    # Gap conditions (optional based on parameters)
    # C2 gaps up from C1: C2_low > C1_high
    c1_c2_gaps_up = c2_low > (c1_high + c1_range * c1_c2_gap_up_percent)
    # C3 gaps down from C2: C3_high < C2_low
    c2_c3_gaps_down = c3_high < (c2_low - c2_range * c2_c3_gap_down_percent)

    # C3 closes well into C1's body
    c3_closes_in_c1 = c3_close < (c1_close - c1_body * c3_closes_into_c1_body_min_percent)
    
    condition_evening_star = c1_is_bullish & c1_decent_body & \
                             c2_is_small_body & \
                             c1_c2_gaps_up & \
                             c3_is_bearish & c3_decent_body & \
                             c2_c3_gaps_down & \
                             c3_closes_in_c1

    return pd.Series(condition_evening_star, index=df.index).fillna(False)

def detect_false_breakout_level(
    df: pd.DataFrame, 
    level: float,
    direction: str, 
    lookback_candle_idx: int = 1 # Relative index of candle that made the initial break (1 for prev, 2 for one before prev etc.)
) -> pd.Series:
    """Detects a False Breakout of a given level.

    A False Breakout occurs when price breaks a level but then reverses
    and closes back on the original side of the level.

    Args:
        df (pd.DataFrame): DataFrame with 'open', 'high', 'low', 'close' columns.
        level (float): The price level to check for a false breakout.
        direction (str): 'bullish' for false breakout below support (expects price to recover above),
                         'bearish' for false breakout above resistance (expects price to fall below).
        lookback_candle_idx (int): The relative index (from current candle) of the candle
                                   that is assumed to have made the initial break.
                                   Defaults to 1 (previous candle broke, current candle confirms false break).

    Returns:
        pd.Series: Boolean series indicating False Breakout occurrences at the current candle.
    """
    if lookback_candle_idx <= 0:
        raise ValueError("lookback_candle_idx must be a positive integer.")

    # Ensure enough data for the lookback
    if len(df) <= lookback_candle_idx:
        return pd.Series([False] * len(df), index=df.index)

    # Candle that potentially broke the level
    break_candle_high = df['high'].shift(lookback_candle_idx)
    break_candle_low = df['low'].shift(lookback_candle_idx)
    # Current candle that confirms the false breakout by closing back
    current_close = df['close']

    is_false_breakout = pd.Series([False] * len(df), index=df.index)

    if direction == 'bullish': # False break below a support level, current closes back above
        # Previous candle broke below the level
        broke_below = break_candle_low < level
        # Current candle closed back above the level
        closed_back_above = current_close > level
        is_false_breakout = broke_below & closed_back_above
        
    elif direction == 'bearish': # False break above a resistance level, current closes back below
        # Previous candle broke above the level
        broke_above = break_candle_high > level
        # Current candle closed back below the level
        closed_back_below = current_close < level
        is_false_breakout = broke_above & closed_back_below
    else:
        raise ValueError("direction must be 'bullish' or 'bearish'.")

    return is_false_breakout.fillna(False)

def detect_morning_star_complex(
    df: pd.DataFrame,
    avg_body_lookback: int = 10,
    star_body_max_c1_avg_body_ratio: float = 0.3,
    c1_c2_gap_flex_ratio: float = 0.3, # Flexible part of gap based on c1_avg_body
    c3_recovery_min_percent_c1_avg_body: float = 0.618
) -> pd.Series:
    """Detects a complex Morning Star pattern variant.

    As found in breakout_trading_strategy.py:
    1. C1: Bearish candle.
    2. C2: Small body relative to C1's average body, gaps down (flexible definition).
    3. C3: Bullish candle, closing above a certain recovery point of C1's average body.

    Args:
        df (pd.DataFrame): DataFrame with 'open', 'high', 'low', 'close' columns.
        avg_body_lookback (int): Lookback period for C1 average body. Defaults to 10.
        star_body_max_c1_avg_body_ratio (float): Max ratio of C2 body to C1 avg body. Defaults to 0.3.
        c1_c2_gap_flex_ratio (float): Flexible component of gap based on C1 avg body. Defaults to 0.3.
        c3_recovery_min_percent_c1_avg_body (float): Min recovery of C3 into C1 avg body. Defaults to 0.618.

    Returns:
        pd.Series: Boolean series indicating complex Morning Star occurrences.
    """
    if len(df) < max(3, avg_body_lookback + 2):
        return pd.Series([False] * len(df), index=df.index)

    c1_open = df['open'].shift(2)
    c1_close = df['close'].shift(2)
    c1_body = abs(c1_open - c1_close)
    c1_avg_body = c1_body.rolling(window=avg_body_lookback, min_periods=1).mean() # Avg body of C1 type candles

    c2_open = df['open'].shift(1)
    c2_close = df['close'].shift(1)
    c2_high = df['high'].shift(1)
    c2_low = df['low'].shift(1)
    c2_body = abs(c2_open - c2_close)

    c3_open = df['open']
    c3_close = df['close']

    # C1 is bearish
    c1_is_bearish = c1_close < c1_open

    # C2 (star) has small body relative to C1's average body
    c2_is_small_body = c2_body < (c1_avg_body * star_body_max_c1_avg_body_ratio)
    
    # C2 gaps down from C1 (flexible definition)
    gap_cond1 = np.maximum(c2_open, c2_close) < np.minimum(c1_open, c1_close)
    gap_cond2 = np.maximum(c2_open, c2_close) < (np.minimum(c1_open, c1_close) + c1_avg_body * c1_c2_gap_flex_ratio)
    c1_c2_gaps_down = gap_cond1 | gap_cond2

    # C3 is bullish
    c3_is_bullish = c3_close > c3_open
    
    # C3 closes above 61.8% recovery of C1's average body (from C1 open)
    c3_recovers_c1 = c3_close > (c1_open - c1_avg_body * c3_recovery_min_percent_c1_avg_body)

    condition = c1_is_bearish & c2_is_small_body & c1_c2_gaps_down & c3_is_bullish & c3_recovers_c1
    return pd.Series(condition, index=df.index).fillna(False)

def detect_evening_star_complex(
    df: pd.DataFrame,
    avg_body_lookback: int = 10,
    star_body_max_c1_avg_body_ratio: float = 0.3,
    c1_c2_gap_flex_ratio: float = 0.3, 
    c3_recovery_min_percent_c1_avg_body: float = 0.618
) -> pd.Series:
    """Detects a complex Evening Star pattern variant.

    Symmetrical to detect_morning_star_complex. As found in breakout_trading_strategy.py:
    1. C1: Bullish candle.
    2. C2: Small body relative to C1's average body, gaps up (flexible definition).
    3. C3: Bearish candle, closing below a certain recovery point of C1's average body.

    Args:
        df (pd.DataFrame): DataFrame with 'open', 'high', 'low', 'close' columns.
        avg_body_lookback (int): Lookback period for C1 average body. Defaults to 10.
        star_body_max_c1_avg_body_ratio (float): Max ratio of C2 body to C1 avg body. Defaults to 0.3.
        c1_c2_gap_flex_ratio (float): Flexible component of gap based on C1 avg body. Defaults to 0.3.
        c3_recovery_min_percent_c1_avg_body (float): Min recovery of C3 into C1 avg body (measured from C1 open).
                                                    Defaults to 0.618.

    Returns:
        pd.Series: Boolean series indicating complex Evening Star occurrences.
    """
    if len(df) < max(3, avg_body_lookback + 2):
        return pd.Series([False] * len(df), index=df.index)

    c1_open = df['open'].shift(2)
    c1_close = df['close'].shift(2)
    c1_body = abs(c1_open - c1_close)
    c1_avg_body = c1_body.rolling(window=avg_body_lookback, min_periods=1).mean()

    c2_open = df['open'].shift(1)
    c2_close = df['close'].shift(1)
    c2_high = df['high'].shift(1)
    c2_low = df['low'].shift(1)
    c2_body = abs(c2_open - c2_close)

    c3_open = df['open']
    c3_close = df['close']

    # C1 is bullish
    c1_is_bullish = c1_close > c1_open

    # C2 (star) has small body relative to C1's average body
    c2_is_small_body = c2_body < (c1_avg_body * star_body_max_c1_avg_body_ratio)
    
    # C2 gaps up from C1 (flexible definition)
    gap_cond1 = np.minimum(c2_open, c2_close) > np.maximum(c1_open, c1_close)
    gap_cond2 = np.minimum(c2_open, c2_close) > (np.maximum(c1_open, c1_close) - c1_avg_body * c1_c2_gap_flex_ratio)
    c1_c2_gaps_up = gap_cond1 | gap_cond2

    # C3 is bearish
    c3_is_bearish = c3_close < c3_open
    
    # C3 closes below 61.8% recovery of C1's average body (from C1 open)
    c3_recovers_c1 = c3_close < (c1_open + c1_avg_body * c3_recovery_min_percent_c1_avg_body)

    condition = c1_is_bullish & c2_is_small_body & c1_c2_gaps_up & c3_is_bearish & c3_recovers_c1
    return pd.Series(condition, index=df.index).fillna(False)

def detect_strong_reversal_candle(
    df: pd.DataFrame, 
    direction: str, 
    wick_min_percent_of_range: float = 0.5,
    # volume_series: Optional[pd.Series] = None, # Decided against direct volume integration here
    # volume_threshold: Optional[float] = None
) -> pd.Series:
    """Detects a strong reversal candle pattern.

    Based on breakout_trading_strategy.py's detect_false_breakout logic.
    This is not about breaking a specific level, but a strong two-candle reversal structure.

    For a 'bullish' reversal (potential bottom):
    1. Previous candle (C1) was bearish.
    2. Current candle (C2) is bullish.
    3. C2's low is below C1's low.
    4. C2 closes above C1's close.
    5. C2 has a long lower wick (signifying rejection of lower prices).

    For a 'bearish' reversal (potential top):
    1. Previous candle (C1) was bullish.
    2. Current candle (C2) is bearish.
    3. C2's high is above C1's high.
    4. C2 closes below C1's close.
    5. C2 has a long upper wick (signifying rejection of higher prices).

    Args:
        df (pd.DataFrame): DataFrame with 'open', 'high', 'low', 'close' columns.
        direction (str): 'bullish' or 'bearish' reversal.
        wick_min_percent_of_range (float): Minimum percentage of the candle's total range
                                           that the relevant wick must occupy. Defaults to 0.5.

    Returns:
        pd.Series: Boolean series indicating strong reversal candle occurrences.
    """
    if len(df) < 2:
        return pd.Series([False] * len(df), index=df.index)

    c1_open = df['open'].shift(1)
    c1_high = df['high'].shift(1)
    c1_low = df['low'].shift(1)
    c1_close = df['close'].shift(1)

    c2_open = df['open']
    c2_high = df['high']
    c2_low = df['low']
    c2_close = df['close']
    c2_range = c2_high - c2_low

    results = pd.Series([False] * len(df), index=df.index)

    if direction.lower() == 'bullish':
        c1_is_bearish = c1_close < c1_open
        c2_is_bullish = c2_close > c2_open
        c2_low_below_c1_low = c2_low < c1_low
        c2_closes_above_c1_close = c2_close > c1_close
        lower_wick = np.where(c2_is_bullish, c2_open - c2_low, c2_close - c2_low)
        has_long_lower_wick = np.where(c2_range > 0.00001, (lower_wick / c2_range) >= wick_min_percent_of_range, False)
        
        results = c1_is_bearish & c2_is_bullish & c2_low_below_c1_low & \
                  c2_closes_above_c1_close & has_long_lower_wick
                  
    elif direction.lower() == 'bearish':
        c1_is_bullish = c1_close > c1_open
        c2_is_bearish = c2_close < c2_open
        c2_high_above_c1_high = c2_high > c1_high
        c2_closes_below_c1_close = c2_close < c1_close
        upper_wick = np.where(c2_is_bearish, c2_high - c2_open, c2_high - c2_close)
        has_long_upper_wick = np.where(c2_range > 0.00001, (upper_wick / c2_range) >= wick_min_percent_of_range, False)

        results = c1_is_bullish & c2_is_bearish & c2_high_above_c1_high & \
                  c2_closes_below_c1_close & has_long_upper_wick
    else:
        raise ValueError("Direction must be 'bullish' or 'bearish'.")

    return results.fillna(False)

def detect_morning_star_v2(df: pd.DataFrame) -> pd.Series:
    """Detects Morning Star patterns (Variant 2).

    Based on breakout_reversal_strategy.py:
    1. C1: A bearish candle.
    2. C2: A small-bodied candle (body < 50% of C1's body) that strictly gaps down
       (C2 high < C1 low).
    3. C3: A bullish candle that closes above the midpoint of C1's body.

    Args:
        df (pd.DataFrame): DataFrame with 'open', 'high', 'low', 'close' columns.

    Returns:
        pd.Series: Boolean series indicating Morning Star V2 occurrences.
    """
    if len(df) < 3:
        return pd.Series([False] * len(df), index=df.index)

    c1_open = df['open'].shift(2)
    c1_close = df['close'].shift(2)
    c1_high = df['high'].shift(2) # Not used in this version's gap logic but good for context
    c1_low = df['low'].shift(2)
    c1_body = abs(c1_open - c1_close)

    c2_open = df['open'].shift(1)
    c2_close = df['close'].shift(1)
    c2_high = df['high'].shift(1)
    # c2_low = df['low'].shift(1) # Not used in this version
    c2_body = abs(c2_open - c2_close)

    c3_open = df['open']
    c3_close = df['close']

    # C1 is bearish
    c1_is_bearish = c1_close < c1_open
    
    # C2 has small body relative to C1's body
    # Ensure c1_body is not zero to avoid division by zero or meaningless comparisons
    c2_is_small_body = np.where(c1_body > 0.00001, c2_body < (c1_body * 0.5), False)
    
    # C2 gaps down from C1 (max of C2 open/close is below min of C1 open/close)
    c2_gaps_down = np.maximum(c2_open, c2_close) < np.minimum(c1_open, c1_close)
    # More precise gap: C2 high < C1 low
    c2_gaps_down_strict = c2_high < c1_low


    # C3 is bullish
    c3_is_bullish = c3_close > c3_open
    
    # C3 closes above midpoint of C1's body
    c3_closes_in_c1_mid = c3_close > (c1_open + c1_close) / 2 # Midpoint of C1 body

    condition = c1_is_bearish & c2_is_small_body & c2_gaps_down_strict & c3_is_bullish & c3_closes_in_c1_mid
    return pd.Series(condition, index=df.index).fillna(False)

def detect_evening_star_v2(df: pd.DataFrame) -> pd.Series:
    """Detects Evening Star patterns (Variant 2).

    Based on breakout_reversal_strategy.py, symmetrical to Morning Star V2:
    1. C1: A bullish candle.
    2. C2: A small-bodied candle (body < 50% of C1's body) that strictly gaps up
       (C2 low > C1 high).
    3. C3: A bearish candle that closes below the midpoint of C1's body.

    Args:
        df (pd.DataFrame): DataFrame with 'open', 'high', 'low', 'close' columns.

    Returns:
        pd.Series: Boolean series indicating Evening Star V2 occurrences.
    """
    if len(df) < 3:
        return pd.Series([False] * len(df), index=df.index)

    c1_open = df['open'].shift(2)
    c1_close = df['close'].shift(2)
    c1_high = df['high'].shift(2)
    # c1_low = df['low'].shift(2) # Not used in this version's gap logic but good for context
    c1_body = abs(c1_open - c1_close)

    c2_open = df['open'].shift(1)
    c2_close = df['close'].shift(1)
    # c2_high = df['high'].shift(1) # Not used
    c2_low = df['low'].shift(1)
    c2_body = abs(c2_open - c2_close)

    c3_open = df['open']
    c3_close = df['close']

    # C1 is bullish
    c1_is_bullish = c1_close > c1_open
    
    # C2 has small body relative to C1's body
    c2_is_small_body = np.where(c1_body > 0.00001, c2_body < (c1_body * 0.5), False)
    
    # C2 gaps up from C1 (min of C2 open/close is above max of C1 open/close)
    c2_gaps_up = np.minimum(c2_open, c2_close) > np.maximum(c1_open, c1_close)
    # More precise gap: C2 low > C1 high
    c2_gaps_up_strict = c2_low > c1_high

    # C3 is bearish
    c3_is_bearish = c3_close < c3_open
    
    # C3 closes below midpoint of C1's body
    c3_closes_in_c1_mid = c3_close < (c1_open + c1_close) / 2 # Midpoint of C1 body

    condition = c1_is_bullish & c2_is_small_body & c2_gaps_up_strict & c3_is_bearish & c3_closes_in_c1_mid
    return pd.Series(condition, index=df.index).fillna(False)

def detect_key_reversal_bar(
    df: pd.DataFrame, 
    direction: str,
    close_percent_of_range: float = 0.3
) -> pd.Series:
    """Detects a Key Reversal Bar pattern.

    Based on breakout_reversal_strategy.py's detect_false_breakout logic.
    This is a strong two-candle reversal structure.

    For a 'bullish' reversal (potential bottom):
    1. Previous candle (C1) was bearish.
    2. Current candle (C2) is bullish.
    3. C2's low is below C1's low.
    4. C2 closes above C1's close.
    5. C2 closes in the top X% of its own range.

    For a 'bearish' reversal (potential top):
    1. Previous candle (C1) was bullish.
    2. Current candle (C2) is bearish.
    3. C2's high is above C1's high.
    4. C2 closes below C1's close.
    5. C2 closes in the bottom X% of its own range.

    Args:
        df (pd.DataFrame): DataFrame with 'open', 'high', 'low', 'close' columns.
        direction (str): 'bullish' or 'bearish' reversal.
        close_percent_of_range (float): The percentile of the candle's range where
                                         the close must occur (e.g., 0.3 for top/bottom 30%).
                                         Defaults to 0.3.

    Returns:
        pd.Series: Boolean series indicating Key Reversal Bar occurrences.
    """
    if len(df) < 2:
        return pd.Series([False] * len(df), index=df.index)

    c1_open = df['open'].shift(1)
    c1_low = df['low'].shift(1)
    c1_close = df['close'].shift(1)
    c1_high = df['high'].shift(1) # Needed for bearish case

    c2_open = df['open']
    c2_high = df['high']
    c2_low = df['low']
    c2_close = df['close']
    c2_range = c2_high - c2_low

    results = pd.Series([False] * len(df), index=df.index)

    if direction.lower() == 'bullish':
        c1_is_bearish = c1_close < c1_open
        c2_is_bullish = c2_close > c2_open
        c2_low_below_c1_low = c2_low < c1_low
        c2_closes_above_c1_close = c2_close > c1_close
        # Closes in the top X% of its range
        closes_in_upper_range = np.where(c2_range > 0.00001, 
                                         c2_close >= (c2_high - c2_range * close_percent_of_range),
                                         False) # If no range, cannot be in upper part
        
        results = c1_is_bearish & c2_is_bullish & c2_low_below_c1_low & \
                  c2_closes_above_c1_close & closes_in_upper_range
                  
    elif direction.lower() == 'bearish':
        c1_is_bullish = c1_close > c1_open
        c2_is_bearish = c2_close < c2_open
        c2_high_above_c1_high = c2_high > c1_high
        c2_closes_below_c1_close = c2_close < c1_close
        # Closes in the bottom X% of its range
        closes_in_lower_range = np.where(c2_range > 0.00001,
                                         c2_close <= (c2_low + c2_range * close_percent_of_range),
                                         False) # If no range, cannot be in lower part

        results = c1_is_bullish & c2_is_bearish & c2_high_above_c1_high & \
                  c2_closes_below_c1_close & closes_in_lower_range
    else:
        raise ValueError("Direction must be 'bullish' or 'bearish'.")

    return results.fillna(False)

def detect_key_reversal_bar_after_breakout(
    df: pd.DataFrame,
    level: float,
    break_direction: str, # 'bullish' if price broke above resistance, 'bearish' if price broke below support
    reversal_close_percent_of_range: float = 0.3,
    body_min_percent_of_range: float = 0.3, # Minimum body size for the reversal candle
    volume_series: Optional[pd.Series] = None,
    volume_threshold_quantile: Optional[float] = None
) -> pd.Series:
    """Detects a Key Reversal Bar after a breakout of a specified level.

    This pattern signifies a potential false breakout. 
    - For a bullish break (of resistance): The previous candle (C-1) breaks above the level.
      The current candle (C0) is a bearish key reversal (closes in bottom X% of range).
    - For a bearish break (of support): The previous candle (C-1) breaks below the level.
      The current candle (C0) is a bullish key reversal (closes in top X% of range).

    Args:
        df (pd.DataFrame): DataFrame with 'open', 'high', 'low', 'close' columns.
                           Optionally 'volume' if volume_series is None and volume_threshold_quantile is set.
        level (float): The support/resistance level that was broken.
        break_direction (str): 'bullish' if resistance was broken upwards,
                               'bearish' if support was broken downwards.
        reversal_close_percent_of_range (float): The current candle must close within this top/bottom
                                                 percentage of its own range to be a key reversal.
                                                 Defaults to 0.3 (30%).
        body_min_percent_of_range (float): Minimum percentage of the current candle's range
                                           that its body must occupy. Defaults to 0.3.
        volume_series (Optional[pd.Series]): External volume series. If None and 
                                             volume_threshold_quantile is set, df['volume'] is used.
        volume_threshold_quantile (Optional[float]): If set (e.g., 0.75 for 75th percentile),
                                                     the reversal candle's volume must be above
                                                     this quantile of recent volume.

    Returns:
        pd.Series: Boolean series indicating pattern occurrences on the current candle (C0).
    """
    if not isinstance(df, pd.DataFrame) or df.empty or len(df) < 2:
        return pd.Series([False] * len(df), index=df.index if isinstance(df, pd.DataFrame) else None)

    c0_open = df['open']
    c0_high = df['high']
    c0_low = df['low']
    c0_close = df['close']
    c0_body = abs(c0_close - c0_open)
    c0_range = c0_high - c0_low

    c1_open = df['open'].shift(1)
    c1_high = df['high'].shift(1)
    c1_low = df['low'].shift(1)
    c1_close = df['close'].shift(1)

    # Conditions for the current candle (C0) being a key reversal bar
    is_strong_body_c0 = np.where(c0_range > 0.00001, (c0_body / c0_range) >= body_min_percent_of_range, False)

    # Conditions for the previous candle (C-1) breaking the level
    if break_direction == 'bullish': # Previous candle broke resistance (level)
        c1_broke_level = (c1_close > level) & (c1_open < level) # Crossed from below
        # Current candle C0 is bearish key reversal
        c0_is_reversal_shape = (c0_close < c0_open) & \
                               (c0_close <= c0_low + c0_range * reversal_close_percent_of_range)
    elif break_direction == 'bearish': # Previous candle broke support (level)
        c1_broke_level = (c1_close < level) & (c1_open > level) # Crossed from above
        # Current candle C0 is bullish key reversal
        c0_is_reversal_shape = (c0_close > c0_open) & \
                               (c0_close >= c0_high - c0_range * reversal_close_percent_of_range)
    else:
        raise ValueError("break_direction must be 'bullish' or 'bearish'")

    # Volume condition for C0 (optional)
    volume_condition_c0 = pd.Series(True, index=df.index) # Default to True if no volume check
    if volume_threshold_quantile is not None:
        if volume_series is None and 'volume' in df.columns:
            volume_series = df['volume']
        
        if volume_series is not None and not volume_series.empty:
            # Calculate rolling quantile for volume threshold
            # Lookback for quantile calculation, e.g., 50 periods
            volume_lookback = 50 
            if len(volume_series) > volume_lookback:
                rolling_quantile_threshold = volume_series.rolling(window=volume_lookback, min_periods=max(1, volume_lookback // 2)).quantile(volume_threshold_quantile)
                actual_volume_c0 = volume_series
                volume_condition_c0 = actual_volume_c0 > rolling_quantile_threshold.shift(1) # Compare C0 volume with threshold from C-1
            else:
                volume_condition_c0 = pd.Series(False, index=df.index) # Not enough data for rolling quantile
        else:
            volume_condition_c0 = pd.Series(False, index=df.index) # No volume data provided

    # Combine all conditions
    # Ensure c1_broke_level is aligned and NaNs (from shift) are False
    final_condition = c1_broke_level.fillna(False) & \
                      is_strong_body_c0 & \
                      c0_is_reversal_shape & \
                      volume_condition_c0.fillna(False)

    return pd.Series(final_condition, index=df.index).fillna(False) 