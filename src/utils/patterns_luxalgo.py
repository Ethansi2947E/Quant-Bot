import numpy as np
import pandas as pd

def get_trend_flags(df: pd.DataFrame, length: int = 14) -> tuple:
    """Return uptrend and downtrend boolean Series using a stochastic-like oscillator."""
    stoch = (df['close'] - df['close'].rolling(length).min()) / (df['close'].rolling(length).max() - df['close'].rolling(length).min())
    stoch = stoch.fillna(0.5) * 100
    uptrend = stoch > 50
    downtrend = stoch < 50
    return uptrend, downtrend

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

def detect_hammer(df: pd.DataFrame, downtrend: pd.Series, atr: pd.Series) -> pd.Series:
    o, c, h, l = df['open'], df['close'], df['high'], df['low']
    d = (c - o).abs()
    return (
        downtrend &
        ((np.minimum(o, c) - l) > 2 * d) &
        ((h - np.maximum(c, o)) < d / 4)
    )

def detect_inverted_hammer(df: pd.DataFrame, downtrend: pd.Series, atr: pd.Series) -> pd.Series:
    o, c, h, l = df['open'], df['close'], df['high'], df['low']
    d = (c - o).abs()
    return (
        downtrend &
        ((h - np.maximum(o, c)) > 2 * d) &
        ((np.minimum(c, o) - l) < d / 4)
    )

def detect_shooting_star(df: pd.DataFrame, uptrend: pd.Series, atr: pd.Series) -> pd.Series:
    o, c, h, l = df['open'], df['close'], df['high'], df['low']
    d = (c - o).abs()
    return (
        uptrend &
        ((h - np.maximum(o, c)) > 2 * d) &
        ((np.minimum(c, o) - l) < d / 4)
    )

def detect_hanging_man(df: pd.DataFrame, uptrend: pd.Series, atr: pd.Series) -> pd.Series:
    o, c, h, l = df['open'], df['close'], df['high'], df['low']
    d = (c - o).abs()
    return (
        uptrend &
        ((np.minimum(o, c) - l) > 2 * d) &
        ((h - np.maximum(c, o)) < d / 4)
    )

def detect_bullish_engulfing(df: pd.DataFrame, downtrend: pd.Series, atr: pd.Series) -> pd.Series:
    c, o = df['close'], df['open']
    d = (c - o).abs()
    return (
        downtrend &
        (c > o) &
        (c.shift(1) < o.shift(1)) &
        (c > o.shift(1)) &
        (d > atr)
    )

def detect_bearish_engulfing(df: pd.DataFrame, uptrend: pd.Series, atr: pd.Series) -> pd.Series:
    c, o = df['close'], df['open']
    d = (c - o).abs()
    return (
        uptrend &
        (c < o) &
        (c.shift(1) > o.shift(1)) &
        (c < o.shift(1)) &
        (d > atr)
    )

def detect_white_marubozu(df: pd.DataFrame, downtrend: pd.Series, atr: pd.Series) -> pd.Series:
    c, o, h, l = df['close'], df['open'], df['high'], df['low']
    d = (c - o).abs()
    return (
        (c > o) &
        ((h - np.maximum(o, c) + np.minimum(o, c) - l) < d / 10) &
        (d > atr) &
        downtrend.shift(1)
    )

def detect_black_marubozu(df: pd.DataFrame, uptrend: pd.Series, atr: pd.Series) -> pd.Series:
    c, o, h, l = df['close'], df['open'], df['high'], df['low']
    d = (c - o).abs()
    return (
        (c < o) &
        ((h - np.maximum(o, c) + np.minimum(o, c) - l) < d / 10) &
        (d > atr) &
        uptrend.shift(1)
    )

# Add any additional patterns from price_action_sr_strategy.py that are not in the LuxAlgo script, using similar logic.
# For example, if you have a pin bar or inside bar, add them here with vectorized logic.

def detect_pin_bar(df: pd.DataFrame) -> pd.Series:
    # Example: Pin bar logic (customize as needed)
    o, c, h, l = df['open'], df['close'], df['high'], df['low']
    body = (c - o).abs()
    upper_wick = h - np.maximum(c, o)
    lower_wick = np.minimum(c, o) - l
    total_range = h - l
    min_wick_ratio = 1.5
    max_body_ratio = 0.7
    bullish = (lower_wick >= min_wick_ratio * body) & (body / total_range < max_body_ratio)
    bearish = (upper_wick >= min_wick_ratio * body) & (body / total_range < max_body_ratio)
    return pd.Series(bullish | bearish, index=df.index)

# Add more as needed...

def detect_bullish_harami(df: pd.DataFrame, downtrend: pd.Series, atr: pd.Series) -> pd.Series:
    # Bullish Harami: Downtrend, previous candle is long bearish, current is small bullish inside previous body
    c, o = df['close'], df['open']
    d = (c - o).abs()
    c1, o1 = c.shift(1), o.shift(1)
    d1 = (c1 - o1).abs()
    return (
        downtrend &
        (c1 < o1) &  # Previous candle bearish
        (d1 > atr) &  # Previous candle long
        (c > o) &  # Current candle bullish
        (c < o1) & (o > c1) &  # Current body inside previous body
        (d < d1)  # Current body smaller
    )

def detect_bearish_harami(df: pd.DataFrame, uptrend: pd.Series, atr: pd.Series) -> pd.Series:
    # Bearish Harami: Uptrend, previous candle is long bullish, current is small bearish inside previous body
    c, o = df['close'], df['open']
    d = (c - o).abs()
    c1, o1 = c.shift(1), o.shift(1)
    d1 = (c1 - o1).abs()
    return (
        uptrend &
        (c1 > o1) &  # Previous candle bullish
        (d1 > atr) &  # Previous candle long
        (c < o) &  # Current candle bearish
        (c > o1) & (o < c1) &  # Current body inside previous body
        (d < d1)  # Current body smaller
    )

def detect_morning_star(df: pd.DataFrame, downtrend: pd.Series, atr: pd.Series) -> pd.Series:
    # Morning Star: Downtrend, three candles: long bearish, small body (gap down), long bullish closing into first
    c, o = df['close'], df['open']
    d = (c - o).abs()
    c1, o1, d1 = c.shift(1), o.shift(1), (c.shift(1) - o.shift(1)).abs()
    c2, o2, d2 = c.shift(2), o.shift(2), (c.shift(2) - o.shift(2)).abs()
    return (
        downtrend.shift(2) &
        (c2 < o2) & (d2 > atr.shift(2)) &  # First candle: long bearish
        (d1 < d2) &  # Second candle: small body
        (c > o) & (d > atr) &  # Third candle: long bullish
        (c > ((o2 + c2) / 2))  # Third closes into first candle's body
    )

def detect_evening_star(df: pd.DataFrame, uptrend: pd.Series, atr: pd.Series) -> pd.Series:
    # Evening Star: Uptrend, three candles: long bullish, small body (gap up), long bearish closing into first
    c, o = df['close'], df['open']
    d = (c - o).abs()
    c1, o1, d1 = c.shift(1), o.shift(1), (c.shift(1) - o.shift(1)).abs()
    c2, o2, d2 = c.shift(2), o.shift(2), (c.shift(2) - o.shift(2)).abs()
    return (
        uptrend.shift(2) &
        (c2 > o2) & (d2 > atr.shift(2)) &  # First candle: long bullish
        (d1 < d2) &  # Second candle: small body
        (c < o) & (d > atr) &  # Third candle: long bearish
        (c < ((o2 + c2) / 2))  # Third closes into first candle's body
    )

def detect_inside_bar(df: pd.DataFrame) -> pd.Series:
    # Inside Bar: Current high < previous high AND current low > previous low
    return (df['high'] < df['high'].shift(1)) & (df['low'] > df['low'].shift(1))

def add_luxalgo_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Add all LuxAlgo-style pattern columns to the DataFrame."""
    atr = get_atr(df)
    uptrend, downtrend = get_trend_flags(df)
    df['hammer'] = detect_hammer(df, downtrend, atr)
    df['inverted_hammer'] = detect_inverted_hammer(df, downtrend, atr)
    df['shooting_star'] = detect_shooting_star(df, uptrend, atr)
    df['hanging_man'] = detect_hanging_man(df, uptrend, atr)
    df['bullish_engulfing'] = detect_bullish_engulfing(df, downtrend, atr)
    df['bearish_engulfing'] = detect_bearish_engulfing(df, uptrend, atr)
    df['white_marubozu'] = detect_white_marubozu(df, downtrend, atr)
    df['black_marubozu'] = detect_black_marubozu(df, uptrend, atr)
    df['bullish_harami'] = detect_bullish_harami(df, downtrend, atr)
    df['bearish_harami'] = detect_bearish_harami(df, uptrend, atr)
    df['morning_star'] = detect_morning_star(df, downtrend, atr)
    df['evening_star'] = detect_evening_star(df, uptrend, atr)
    df['inside_bar'] = detect_inside_bar(df)
    # Add more as needed...
    return df 