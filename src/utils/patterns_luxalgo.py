import numpy as np
import pandas as pd

# Pattern Classifications by Directional Bias
BULLISH_PATTERNS = [
    'hammer',
    'bullish_engulfing',
    'white_marubozu',
    'bullish_harami',
    'morning_star',
    'pin_bar_bullish'  # Context: at support, long lower wick
]

BEARISH_PATTERNS = [
    'shooting_star',
    'hanging_man',
    'bearish_engulfing',
    'black_marubozu',
    'bearish_harami',
    'evening_star',
    'pin_bar_bearish'  # Context: at resistance, long upper wick
]

NEUTRAL_PATTERNS = [
    'inside_bar',
    'inverted_hammer'  # Can be a reversal, but often needs more confirmation.
                       # Its position as neutral implies it's a warning/setup candle.
]

# All patterns combined for easy iteration
ALL_PATTERNS = BULLISH_PATTERNS + BEARISH_PATTERNS + NEUTRAL_PATTERNS

def get_pattern_type(pattern_name: str) -> str:
    """
    Get the classification type of a pattern.
    
    Args:
        pattern_name (str): Name of the pattern
        
    Returns:
        str: 'bullish', 'bearish', 'neutral', or 'unknown'
    """
    if pattern_name in BULLISH_PATTERNS:
        return 'bullish'
    elif pattern_name in BEARISH_PATTERNS:
        return 'bearish'
    elif pattern_name in NEUTRAL_PATTERNS:
        return 'neutral'
    else:
        return 'unknown'

def filter_patterns_by_bias(detected_patterns: list, bias: str) -> list:
    """
    Filter detected patterns by their directional bias.
    
    Args:
        detected_patterns (list): List of detected pattern names
        bias (str): 'bullish', 'bearish', or 'neutral'
        
    Returns:
        list: Filtered patterns matching the bias
    """
    if bias.lower() == 'bullish':
        return [p for p in detected_patterns if p in BULLISH_PATTERNS]
    elif bias.lower() == 'bearish':
        return [p for p in detected_patterns if p in BEARISH_PATTERNS]
    elif bias.lower() == 'neutral':
        return [p for p in detected_patterns if p in NEUTRAL_PATTERNS]
    else:
        return detected_patterns

def get_atr(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Return ATR/2 as in the LuxAlgo script."""
    high, low, close = df['high'], df['low'], df['close']
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean() / 2
    return atr

def detect_hammer(df: pd.DataFrame) -> pd.Series:
    o, c, h, l = df['open'], df['close'], df['high'], df['low']
    d = (c - o).abs()
    condition = (
        ((np.minimum(o, c) - l) > 2 * d) &
        ((h - np.maximum(c, o)) < d / 4)
    )
    return pd.Series(condition, index=df.index).fillna(False)

def detect_inverted_hammer(df: pd.DataFrame) -> pd.Series:
    o, c, h, l = df['open'], df['close'], df['high'], df['low']
    d = (c - o).abs()
    condition = (
        ((h - np.maximum(o, c)) > 2 * d) &
        ((np.minimum(c, o) - l) < d / 4)
    )
    return pd.Series(condition, index=df.index).fillna(False)

def detect_shooting_star(df: pd.DataFrame) -> pd.Series:
    o, c, h, l = df['open'], df['close'], df['high'], df['low']
    d = (c - o).abs()
    condition = (
        ((h - np.maximum(o, c)) > 2 * d) &
        ((np.minimum(c, o) - l) < d / 4)
    )
    return pd.Series(condition, index=df.index).fillna(False)

def detect_hanging_man(df: pd.DataFrame) -> pd.Series:
    o, c, h, l = df['open'], df['close'], df['high'], df['low']
    d = (c - o).abs()
    condition = (
        ((np.minimum(o, c) - l) > 2 * d) &
        ((h - np.maximum(c, o)) < d / 4)
    )
    return pd.Series(condition, index=df.index).fillna(False)

def detect_bullish_engulfing(df: pd.DataFrame, atr: pd.Series) -> pd.Series:
    c, o = df['close'], df['open']
    d = (c - o).abs()
    condition = (
        (c > o) &
        (c.shift(1) < o.shift(1)) &
        (c > o.shift(1)) &
        (d > atr)
    )
    return pd.Series(condition, index=df.index).fillna(False)

def detect_bearish_engulfing(df: pd.DataFrame, atr: pd.Series) -> pd.Series:
    c, o = df['close'], df['open']
    d = (c - o).abs()
    condition = (
        (c < o) &
        (c.shift(1) > o.shift(1)) &
        (c < o.shift(1)) &
        (d > atr)
    )
    return pd.Series(condition, index=df.index).fillna(False)

def detect_white_marubozu(df: pd.DataFrame, atr: pd.Series) -> pd.Series:
    c, o, h, l = df['close'], df['open'], df['high'], df['low']
    d = (c - o).abs()
    condition = (
        (c > o) &
        ((h - np.maximum(o, c) + np.minimum(o, c) - l) < d / 10) &
        (d > atr)
    )
    return pd.Series(condition, index=df.index).fillna(False)

def detect_black_marubozu(df: pd.DataFrame, atr: pd.Series) -> pd.Series:
    c, o, h, l = df['close'], df['open'], df['high'], df['low']
    d = (c - o).abs()
    condition = (
        (c < o) &
        ((h - np.maximum(o, c) + np.minimum(o, c) - l) < d / 10) &
        (d > atr)
    )
    return pd.Series(condition, index=df.index).fillna(False)

# Add any additional patterns from price_action_sr_strategy.py that are not in the LuxAlgo script, using similar logic.
# For example, if you have a pin bar or inside bar, add them here with vectorized logic.

def detect_pin_bar_bullish(df: pd.DataFrame) -> pd.Series:
    """
    Detect bullish pin bars (hammer-like with long lower wick).
    Bullish pin bars typically form at support levels and signal potential upward reversal.
    """
    o, c, h, l = df['open'], df['close'], df['high'], df['low']
    body = (c - o).abs()
    upper_wick = h - np.maximum(c, o)
    lower_wick = np.minimum(c, o) - l
    total_range = h - l

    # Criteria for bullish pin bar
    min_wick_ratio = 2.0  # Lower wick should be at least 2x the body
    max_body_ratio = 0.3  # Body should be small relative to total range
    max_upper_wick_ratio = 0.2  # Upper wick should be small

    valid_range = total_range > 0

    condition = (
        valid_range &
        (lower_wick >= min_wick_ratio * body) &  # Long lower wick
        (upper_wick <= max_upper_wick_ratio * body) &  # Small upper wick
        (body / total_range < max_body_ratio)  # Small body relative to range
    )

    return pd.Series(condition, index=df.index).fillna(False)

def detect_pin_bar_bearish(df: pd.DataFrame) -> pd.Series:
    """
    Detect bearish pin bars (shooting star-like with long upper wick).
    Bearish pin bars typically form at resistance levels and signal potential downward reversal.
    """
    o, c, h, l = df['open'], df['close'], df['high'], df['low']
    body = (c - o).abs()
    upper_wick = h - np.maximum(c, o)
    lower_wick = np.minimum(c, o) - l
    total_range = h - l

    # Criteria for bearish pin bar
    min_wick_ratio = 2.0  # Upper wick should be at least 2x the body
    max_body_ratio = 0.3  # Body should be small relative to total range
    max_lower_wick_ratio = 0.2  # Lower wick should be small

    valid_range = total_range > 0

    condition = (
        valid_range &
        (upper_wick >= min_wick_ratio * body) &  # Long upper wick
        (lower_wick <= max_lower_wick_ratio * body) &  # Small lower wick
        (body / total_range < max_body_ratio)  # Small body relative to range
    )

    return pd.Series(condition, index=df.index).fillna(False)

def detect_bullish_harami(df: pd.DataFrame) -> pd.Series:
    # Bullish Harami: Downtrend, previous candle is long bearish, current is small bullish inside previous body
    c, o = df['close'], df['open']
    d = (c - o).abs()
    c1, o1 = c.shift(1), o.shift(1)
    d1 = (c1 - o1).abs()
    condition = (
        (c1 < o1) &  # Previous candle bearish
        (c > o) &  # Current candle bullish
        (c < o1) & (o > c1) &  # Current body inside previous body
        (d < d1)  # Current body smaller
    )
    return pd.Series(condition, index=df.index).fillna(False)

def detect_bearish_harami(df: pd.DataFrame) -> pd.Series:
    # Bearish Harami: Uptrend, previous candle is long bullish, current is small bearish inside previous body
    c, o = df['close'], df['open']
    d = (c - o).abs()
    c1, o1 = c.shift(1), o.shift(1)
    d1 = (c1 - o1).abs()
    condition = (
        (c1 > o1) &  # Previous candle bullish
        (c < o) &  # Current candle bearish
        (c > o1) & (o < c1) &  # Current body inside previous body
        (d < d1)  # Current body smaller
    )
    return pd.Series(condition, index=df.index).fillna(False)

def detect_morning_star(df: pd.DataFrame) -> pd.Series:
    # Morning Star: Downtrend, three candles: long bearish, small body (gap down), long bullish closing into first
    c, o = df['close'], df['open']
    d = (c - o).abs()
    c1, o1, d1 = c.shift(1), o.shift(1), (c.shift(1) - o.shift(1)).abs()
    c2, o2, d2 = c.shift(2), o.shift(2), (c.shift(2) - o.shift(2)).abs()
    condition = (
        (c2 < o2) &  # First candle: long bearish
        (d1 < d2) &  # Second candle: small body
        (c > o) &  # Third candle: long bullish
        (c > ((o2 + c2) / 2))  # Third closes into first candle's body
    )
    return pd.Series(condition, index=df.index).fillna(False)

def detect_evening_star(df: pd.DataFrame) -> pd.Series:
    # Evening Star: Uptrend, three candles: long bullish, small body (gap up), long bearish closing into first
    c, o = df['close'], df['open']
    d = (c - o).abs()
    c1, o1, d1 = c.shift(1), o.shift(1), (c.shift(1) - o.shift(1)).abs()
    c2, o2, d2 = c.shift(2), o.shift(2), (c.shift(2) - o.shift(2)).abs()
    condition = (
        (c2 > o2) &  # First candle: long bullish
        (d1 < d2) &  # Second candle: small body
        (c < o) &  # Third candle: long bearish
        (c < ((o2 + c2) / 2))  # Third closes into first candle's body
    )
    return pd.Series(condition, index=df.index).fillna(False)

def detect_inside_bar(df: pd.DataFrame) -> pd.Series:
    # Inside Bar: Current high < previous high AND current low > previous low
    condition = (df['high'] < df['high'].shift(1)) & (df['low'] > df['low'].shift(1))
    return pd.Series(condition, index=df.index).fillna(False)

def add_luxalgo_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Add all LuxAlgo-style pattern columns to the DataFrame."""
    atr = get_atr(df)
    df['hammer'] = detect_hammer(df)
    df['inverted_hammer'] = detect_inverted_hammer(df)
    df['shooting_star'] = detect_shooting_star(df)
    df['hanging_man'] = detect_hanging_man(df)
    df['bullish_engulfing'] = detect_bullish_engulfing(df, atr)
    df['bearish_engulfing'] = detect_bearish_engulfing(df, atr)
    df['white_marubozu'] = detect_white_marubozu(df, atr)
    df['black_marubozu'] = detect_black_marubozu(df, atr)
    df['bullish_harami'] = detect_bullish_harami(df)
    df['bearish_harami'] = detect_bearish_harami(df)
    df['morning_star'] = detect_morning_star(df)
    df['evening_star'] = detect_evening_star(df)
    df['inside_bar'] = detect_inside_bar(df)
    df['pin_bar_bullish'] = detect_pin_bar_bullish(df)
    df['pin_bar_bearish'] = detect_pin_bar_bearish(df)
    return df 