import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from loguru import logger
from datetime import datetime

# Core framework imports
from src.trading_bot import SignalGenerator
from config.config import get_risk_manager_config
from src.risk_manager import RiskManager

class PremiumLuxAlgo:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with OHLCV data
        data should have columns: 'open', 'high', 'low', 'close', 'volume'
        """
        self.data = data.copy()
        self.data.columns = self.data.columns.str.lower()
        
        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in self.data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Initialize results dataframe
        self.results = self.data.copy()
        
    def ema(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate EMA"""
        return series.ewm(span=period, adjust=False).mean()
    
    def sma(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate SMA"""
        return series.rolling(window=period).mean()
    
    def atr(self, period: int) -> pd.Series:
        """Calculate ATR"""
        high = self.data['high']
        low = self.data['low']
        close = self.data['close']
        
        tr1 = high - low
        tr2 = np.abs(high - close.shift(1))
        tr3 = np.abs(low - close.shift(1))
        
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr_series = pd.Series(tr, index=self.data.index).fillna(0)
        return tr_series.ewm(alpha=1/period, adjust=False).mean()
    
    def supertrend(self, factor: float = 5.5, atr_period: int = 11) -> Tuple[pd.Series, pd.Series]:
        """Calculate Supertrend"""
        close = self.data['close']
        atr = self.atr(atr_period)
        
        # Calculate basic upper and lower bands
        upper_band = close + (factor * atr)
        lower_band = close - (factor * atr)
        
        # Initialize arrays
        final_upper = pd.Series(index=close.index, dtype=float)
        final_lower = pd.Series(index=close.index, dtype=float)
        supertrend = pd.Series(index=close.index, dtype=float)
        direction = pd.Series(index=close.index, dtype=int)
        
        for i in range(len(close)):
            if i == 0:
                final_upper.iloc[i] = upper_band.iloc[i]
                final_lower.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
            else:
                # Final upper band
                if upper_band.iloc[i] < final_upper.iloc[i-1] or close.iloc[i-1] > final_upper.iloc[i-1]:
                    final_upper.iloc[i] = upper_band.iloc[i]
                else:
                    final_upper.iloc[i] = final_upper.iloc[i-1]
                
                # Final lower band
                if lower_band.iloc[i] > final_lower.iloc[i-1] or close.iloc[i-1] < final_lower.iloc[i-1]:
                    final_lower.iloc[i] = lower_band.iloc[i]
                else:
                    final_lower.iloc[i] = final_lower.iloc[i-1]
                
                # Direction
                if direction.iloc[i-1] == -1 and close.iloc[i] > final_lower.iloc[i]:
                    direction.iloc[i] = 1
                elif direction.iloc[i-1] == 1 and close.iloc[i] < final_upper.iloc[i]:
                    direction.iloc[i] = -1
                else:
                    direction.iloc[i] = direction.iloc[i-1]
        
        # Supertrend line
        supertrend = np.where(direction == 1, final_lower, final_upper)
        
        return pd.Series(supertrend, index=close.index), pd.Series(direction, index=close.index)
    
    def calculate_tp_points(self, maj_qual: int = 13, maj_len: int = 40, 
                           min_qual: int = 5, min_len: int = 5) -> Tuple[pd.Series, pd.Series]:
        """Calculate TP (Take Profit) Points"""
        close = self.data['close']
        open_price = self.data['open']
        high = self.data['high']
        low = self.data['low']
        
        def lele_function(qual: int, length: int) -> pd.Series:
            bindex = pd.Series(0.0, index=close.index)
            sindex = pd.Series(0.0, index=close.index)
            ret = pd.Series(0, index=close.index)
            
            for i in range(4, len(close)):
                if close.iloc[i] > close.iloc[i-4]:
                    bindex.iloc[i] = bindex.iloc[i-1] + 1
                else:
                    bindex.iloc[i] = bindex.iloc[i-1]
                
                if close.iloc[i] < close.iloc[i-4]:
                    sindex.iloc[i] = sindex.iloc[i-1] + 1
                else:
                    sindex.iloc[i] = sindex.iloc[i-1]
                
                # Check for sell signal
                if (bindex.iloc[i] > qual and close.iloc[i] < open_price.iloc[i] and 
                    high.iloc[i] >= high.iloc[max(0, i-length):i].max()):
                    bindex.iloc[i] = 0
                    ret.iloc[i] = -1
                
                # Check for buy signal
                if (sindex.iloc[i] > qual and close.iloc[i] > open_price.iloc[i] and 
                    low.iloc[i] <= low.iloc[max(0, i-length):i].min()):
                    sindex.iloc[i] = 0
                    ret.iloc[i] = 1
            
            return ret
        
        major = lele_function(maj_qual, maj_len)
        minor = lele_function(min_qual, min_len)
        
        return major, minor
    
    def heikin_ashi_trend(self, ha_len: int = 100) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """Calculate Heikin Ashi trend"""
        o = self.ema(self.data['open'], ha_len)
        c = self.ema(self.data['close'], ha_len)
        h = self.ema(self.data['high'], ha_len)
        l = self.ema(self.data['low'], ha_len)
        
        haclose = (o + h + l + c) / 4
        xhaopen = (o + c) / 2
        haopen = pd.Series(index=self.data.index, dtype=float)
        
        haopen.iloc[0] = xhaopen.iloc[0]
        for i in range(1, len(haopen)):
            haopen.iloc[i] = (haopen.iloc[i-1] + haclose.iloc[i-1]) / 2
        
        hahigh = pd.concat([h, haopen, haclose], axis=1).max(axis=1)
        halow = pd.concat([l, haopen, haclose], axis=1).min(axis=1)
        
        # Second level smoothing
        o2 = self.ema(haopen, ha_len)
        c2 = self.ema(haclose, ha_len)
        h2 = self.ema(hahigh, ha_len)
        l2 = self.ema(halow, ha_len)
        
        osc_bias = 100 * (c2 - o2)
        osc_smooth = self.ema(osc_bias, 7)
        
        return osc_bias, osc_smooth, h2, l2
    
    def range_filter(self, rng_qty: float = 2.618, rng_per: int = 14,
                    smooth_per: int = 27) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Range Filter using 'Type 2' stepping logic from PineScript.
        """
        high = self.data['high']
        low = self.data['low']
        close = self.data['close']
        
        # Calculate Average Change using EMA to match PineScript's Cond_EMA
        avg_change_series = pd.Series(np.abs(close - close.shift(1)), index=close.index).fillna(0)
        avg_change = self.ema(avg_change_series, rng_per)
        rng_size = rng_qty * avg_change
        
        # Smooth the range
        rng_smooth = self.ema(rng_size, smooth_per)
        
        # Calculate filter using 'Type 2' logic
        filt = pd.Series(index=close.index, dtype=float)
        if not filt.empty:
             filt.iloc[0] = (high.iloc[0] + low.iloc[0]) / 2
        
        for i in range(1, len(close)):
            prev_filt = filt.iloc[i-1]
            r = rng_smooth.iloc[i]
            
            # Avoid division by zero and ensure r is positive
            if pd.isna(r) or r <= 0:
                filt.iloc[i] = prev_filt
                continue

            if high.iloc[i] >= prev_filt + r:
                # Step up
                filt.iloc[i] = prev_filt + np.floor(abs(high.iloc[i] - prev_filt) / r) * r
            elif low.iloc[i] <= prev_filt - r:
                # Step down
                filt.iloc[i] = prev_filt - np.floor(abs(low.iloc[i] - prev_filt) / r) * r
            else:
                # No change
                filt.iloc[i] = prev_filt
        
        # Calculate direction
        direction = pd.Series(0, index=close.index, dtype=int)
        for i in range(1, len(filt)):
            if filt.iloc[i] > filt.iloc[i-1]:
                direction.iloc[i] = 1
            elif filt.iloc[i] < filt.iloc[i-1]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = direction.iloc[i-1]
        
        return filt, direction
    
    def super_ichimoku(self, tenkan_len: int = 6, tenkan_mult: float = 2,
                      kijun_len: int = 5, kijun_mult: float = 3,
                      spanb_len: int = 26, spanb_mult: float = 4) -> dict:
        """Calculate Super Ichimoku"""
        high = self.data['high']
        low = self.data['low']
        close = self.data['close']
        hl2 = (high + low) / 2
        
        def avg_calculation(src, length, mult):
            atr = self.atr(length) * mult
            up = hl2 + atr
            dn = hl2 - atr
            
            upper = pd.Series(index=src.index, dtype=float)
            lower = pd.Series(index=src.index, dtype=float)
            
            upper.iloc[0] = up.iloc[0]
            lower.iloc[0] = dn.iloc[0]
            
            for i in range(1, len(src)):
                if src.iloc[i-1] < upper.iloc[i-1]:
                    upper.iloc[i] = min(up.iloc[i], upper.iloc[i-1])
                else:
                    upper.iloc[i] = up.iloc[i]
                
                if src.iloc[i-1] > lower.iloc[i-1]:
                    lower.iloc[i] = max(dn.iloc[i], lower.iloc[i-1])
                else:
                    lower.iloc[i] = dn.iloc[i]
            
            # Calculate oscillator state
            os = pd.Series(0, index=src.index)
            for i in range(1, len(src)):
                if src.iloc[i] > upper.iloc[i]:
                    os.iloc[i] = 1
                elif src.iloc[i] < lower.iloc[i]:
                    os.iloc[i] = 0
                else:
                    os.iloc[i] = os.iloc[i-1]
            
            spt = np.where(os == 1, lower, upper)
            
            # Calculate max and min
            max_val = pd.Series(index=src.index, dtype=float)
            min_val = pd.Series(index=src.index, dtype=float)
            
            for i in range(len(src)):
                if i == 0:
                    max_val.iloc[i] = src.iloc[i]
                    min_val.iloc[i] = src.iloc[i]
                else:
                    if os.iloc[i] == 1:
                        max_val.iloc[i] = max(src.iloc[i], max_val.iloc[i-1])
                        min_val.iloc[i] = spt[i]
                    else:
                        max_val.iloc[i] = spt[i]
                        min_val.iloc[i] = min(src.iloc[i], min_val.iloc[i-1])
            
            return (max_val + min_val) / 2
        
        tenkan = avg_calculation(close, tenkan_len, tenkan_mult)
        kijun = avg_calculation(close, kijun_len, kijun_mult)
        senkou_a = (kijun + tenkan) / 2
        senkou_b = avg_calculation(close, spanb_len, spanb_mult)
        
        return {
            'tenkan': tenkan,
            'kijun': kijun,
            'senkou_a': senkou_a,
            'senkou_b': senkou_b
        }
    
    def tbo_signals(self, fast_len: int = 20, medium_len: int = 40,
                   medfast_len: int = 50, slow_len: int = 150) -> Tuple[pd.Series, pd.Series]:
        """Calculate TBO (The Better Traders) signals"""
        close = self.data['close']
        
        fast_tbo = self.ema(close, fast_len)
        medium_tbo = self.ema(close, medium_len)
        medfast_tbo = self.sma(close, medfast_len)
        slow_tbo = self.sma(close, slow_len)
        
        # Calculate crossovers
        open_long = (fast_tbo > medium_tbo) & (fast_tbo.shift(1) <= medium_tbo.shift(1))
        open_short = (fast_tbo < medium_tbo) & (fast_tbo.shift(1) >= medium_tbo.shift(1))
        
        return open_long, open_short
    
    def smart_trail(self, atr_period: int = 13, atr_factor: int = 4,
                   smoothing: int = 8) -> pd.Series:
        """Calculate Smart Trail"""
        high = self.data['high']
        low = self.data['low']
        close = self.data['close']
        
        # Calculate True Range (Wilder's method)
        tr = pd.concat([
            pd.Series(high - low, index=close.index),
            pd.Series(np.abs(high - close.shift(1)), index=close.index),
            pd.Series(np.abs(low - close.shift(1)), index=close.index)
        ], axis=1).max(axis=1)
        
        # Wilder's moving average
        wild_atr = tr.ewm(alpha=1/atr_period, adjust=False).mean()
        loss = atr_factor * wild_atr
        
        up = close - loss
        dn = close + loss
        
        trend_up = pd.Series(index=close.index, dtype=float)
        trend_down = pd.Series(index=close.index, dtype=float)
        trend = pd.Series(1, index=close.index)
        
        trend_up.iloc[0] = up.iloc[0]
        trend_down.iloc[0] = dn.iloc[0]
        
        for i in range(1, len(close)):
            if close.iloc[i-1] > trend_up.iloc[i-1]:
                trend_up.iloc[i] = max(up.iloc[i], trend_up.iloc[i-1])
            else:
                trend_up.iloc[i] = up.iloc[i]
            
            if close.iloc[i-1] < trend_down.iloc[i-1]:
                trend_down.iloc[i] = min(dn.iloc[i], trend_down.iloc[i-1])
            else:
                trend_down.iloc[i] = dn.iloc[i]
            
            if close.iloc[i] > trend_down.iloc[i-1]:
                trend.iloc[i] = 1
            elif close.iloc[i] < trend_up.iloc[i-1]:
                trend.iloc[i] = -1
            else:
                trend.iloc[i] = trend.iloc[i-1]
        
        trail = np.where(trend == 1, trend_up, trend_down)
        return self.sma(pd.Series(trail, index=close.index), smoothing)
    
    def reversal_signals(self, rsi_period: int = 14, overbought: float = 75,
                        oversold: float = 25) -> Tuple[pd.Series, pd.Series]:
        """Calculate Reversal Signals"""
        close = self.data['close']
        
        # Calculate RSI manually
        delta = close.diff().astype(float)
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        
        avg_gain = gain.ewm(alpha=1/rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/rsi_period, adjust=False).mean()
        
        # Avoid division by zero
        rs = avg_gain / avg_loss.replace(0, 1)
        rsi = 100 - (100 / (1 + rs))
        
        rev_up = (rsi > oversold) & (rsi.shift(1) <= oversold)
        rev_down = (rsi < overbought) & (rsi.shift(1) >= overbought)
        
        return rev_up, rev_down
    
    def support_resistance(self, strength: int = 4, lookback: int = 284) -> dict:
        """Calculate Support and Resistance levels"""
        high = self.data['high']
        low = self.data['low']
        close = self.data['close']
        
        # Find pivot highs and lows
        pivot_high = pd.Series(index=high.index, dtype=float)
        pivot_low = pd.Series(index=low.index, dtype=float)
        
        rb = 10  # bars left and right
        
        for i in range(rb, len(high) - rb):
            if high.iloc[i] == high.iloc[i-rb:i+rb+1].max():
                pivot_high.iloc[i] = high.iloc[i]
            if low.iloc[i] == low.iloc[i-rb:i+rb+1].min():
                pivot_low.iloc[i] = low.iloc[i]
        
        # Calculate support and resistance levels
        sr_levels = []
        recent_pivots = []
        
        for i in range(len(pivot_high)):
            if not pd.isna(pivot_high.iloc[i]):
                recent_pivots.append(pivot_high.iloc[i])
            if not pd.isna(pivot_low.iloc[i]):
                recent_pivots.append(pivot_low.iloc[i])
        
        # Group similar levels
        tolerance = (high.max() - low.min()) * 0.01  # 1% tolerance
        
        for level in recent_pivots[-50:]:  # Consider last 50 pivots
            count = sum(1 for p in recent_pivots if abs(p - level) <= tolerance)
            if count >= strength:
                sr_levels.append(level)
        
        return {
            'levels': list(set(sr_levels)),  # Remove duplicates
            'pivot_highs': pivot_high,
            'pivot_lows': pivot_low
        }
    
    def reversal_cloud(self, length: int = 50, bd1: int = 9, bd2: int = 11, bd3: int = 14, kama_fast: int = 2, kama_slow: int = 30) -> dict:
        """Calculate Reversal Cloud (Lux Algo style)"""
        close = self.data['close']
        
        # KAMA calculation
        def kama(src, length, fast, slow):
            change = np.abs(src.diff())
            volatility = change.rolling(window=length).sum()
            direction = np.abs(src - src.shift(length))
            
            efficiency = (direction / volatility.replace(0, 1)).fillna(0)
            
            fast_sc = 2.0 / (fast + 1)
            slow_sc = 2.0 / (slow + 1)
            sc = (efficiency * (fast_sc - slow_sc) + slow_sc) ** 2
            
            kama_val = pd.Series(index=src.index, dtype=float)
            kama_val.iloc[0] = src.iloc[0]
            
            for i in range(1, len(src)):
                kama_val.iloc[i] = kama_val.iloc[i-1] + sc.iloc[i] * (src.iloc[i] - kama_val.iloc[i-1])
            
            return kama_val
        
        # Calculate True Range and KAMA
        tr = pd.concat([
            pd.Series(self.data['high'] - self.data['low'], index=close.index),
            pd.Series(np.abs(self.data['high'] - close.shift(1)), index=close.index),
            pd.Series(np.abs(self.data['low'] - close.shift(1)), index=close.index)
        ], axis=1).max(axis=1)
        
        rg = kama(tr, length, kama_fast, kama_slow)
        basis = kama(close, length, kama_fast, kama_slow)
        
        # Calculate bands
        upper1 = basis + rg * bd1
        upper2 = basis + rg * bd2
        upper3 = basis + rg * bd3
        lower1 = basis - rg * bd1
        lower2 = basis - rg * bd2
        lower3 = basis - rg * bd3
        
        return {
            'basis': basis,
            'upper1': upper1, 'upper2': upper2, 'upper3': upper3,
            'lower1': lower1, 'lower2': lower2, 'lower3': lower3
        }
    
    def macd_candle_coloring(self, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """Calculate MACD-based candle coloring"""
        close = self.data['close']
        
        # Calculate MACD
        exp1 = close.ewm(span=fast).mean()
        exp2 = close.ewm(span=slow).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        
        # Color classification
        colors = pd.Series('yellow', index=close.index)  # Default
        
        # Green conditions
        green_condition = (histogram > 0) & (histogram > histogram.shift(1)) & (histogram.shift(1) > 0)
        colors[green_condition] = 'green'
        
        # Red conditions
        red_condition = (histogram < 0) & (histogram < histogram.shift(1)) & (histogram.shift(1) < 0)
        colors[red_condition] = 'red'
        
        # Stronger green
        strong_green = (macd > 0) & (histogram > 0) & (histogram > histogram.shift(1))
        colors[strong_green] = 'strong_green'
        
        # Stronger red
        strong_red = (macd < 0) & (histogram < 0) & (histogram < histogram.shift(1))
        colors[strong_red] = 'strong_red'
        
        return colors
    
    def calculate_all_indicators(self, sensitivity: float = 5.5) -> pd.DataFrame:
        """Calculate all indicators and return results dataframe"""
        
        # Basic calculations
        self.results['oc_avg'] = (self.data['open'] + self.data['close']) / 2
        
        # Supertrend
        supertrend, st_direction = self.supertrend(sensitivity)
        self.results['supertrend'] = supertrend
        self.results['st_direction'] = st_direction
        
        # SMA calculations
        close = self.data['close']
        self.results['sma9'] = self.sma(close, 9)
        
        # Buy/Sell signals
        bull_signal = (close > supertrend) & (close.shift(1) <= supertrend.shift(1)) & (close >= self.results['sma9'])
        bear_signal = (close < supertrend) & (close.shift(1) >= supertrend.shift(1)) & (close <= self.results['sma9'])
        
        self.results['bull_signal'] = bull_signal
        self.results['bear_signal'] = bear_signal
        
        # TP Points
        major_tp, minor_tp = self.calculate_tp_points()
        self.results['major_tp'] = major_tp
        self.results['minor_tp'] = minor_tp
        
        # Heikin Ashi trend
        osc_bias, osc_smooth, h2, l2 = self.heikin_ashi_trend()
        self.results['ha_bias'] = osc_bias
        self.results['ha_smooth'] = osc_smooth
        self.results['ha_high'] = h2
        self.results['ha_low'] = l2
        
        # Range Filter
        filt, filt_direction = self.range_filter()
        self.results['range_filter'] = filt
        self.results['range_filter_direction'] = filt_direction
        
        # Super Ichimoku
        ichimoku = self.super_ichimoku()
        for key, value in ichimoku.items():
            self.results[f'ichi_{key}'] = value
        
        # TBO Signals
        tbo_long, tbo_short = self.tbo_signals()
        self.results['tbo_long'] = tbo_long
        self.results['tbo_short'] = tbo_short
        
        # Smart Trail
        self.results['smart_trail'] = self.smart_trail()
        
        # Reversal Signals
        rev_up, rev_down = self.reversal_signals()
        self.results['reversal_up'] = rev_up
        self.results['reversal_down'] = rev_down
        
        # Support/Resistance
        sr_data = self.support_resistance()
        self.results['sr_levels'] = sr_data['levels']
        
        # Reversal Cloud
        cloud = self.reversal_cloud()
        for key, value in cloud.items():
            self.results[f'cloud_{key}'] = value
        
        # MACD Candle Colors
        self.results['candle_color'] = self.macd_candle_coloring()
        
        return self.results
    
    def get_signals(self) -> pd.DataFrame:
        """Get clean signals summary"""
        signals = pd.DataFrame(index=self.results.index)
        
        signals['price'] = self.data['close']
        signals['supertrend_signal'] = np.where(self.results['bull_signal'], 1, 
                                               np.where(self.results['bear_signal'], -1, 0))
        signals['tp_major_signal'] = np.where(self.results['major_tp'] == 1, 1,
                                             np.where(self.results['major_tp'] == -1, -1, 0))
        signals['tbo_signal'] = np.where(self.results['tbo_long'], 1,
                                        np.where(self.results['tbo_short'], -1, 0))
        signals['reversal_signal'] = np.where(self.results['reversal_up'], 1,
                                             np.where(self.results['reversal_down'], -1, 0))
        signals['range_filter_trend'] = self.results['range_filter_direction']
        
        return signals

# --- Best Practice: Timeframe-Specific Parameter Profiles ---
# Define different parameter sets for various timeframes to make your strategy adaptable.
TIMEFRAME_PROFILES = {
    "M5": {"lookback": 284},
    "M15": {"lookback": 284},
    "H1": {"lookback": 284},
    # Add profiles for other timeframes your strategy might use
}
# A default profile is crucial as a fallback.
DEFAULT_PROFILE = {"lookback": 284}


class PremiumLuxAlgoStrategy(SignalGenerator):
    """
    Strategy based on a Python port of a popular PineScript indicator set.
    This strategy uses the PremiumLuxAlgo class to calculate a variety of indicators
    and generates trading signals based on their confluence.
    """

    def __init__(self,
                 primary_timeframe: str = "M1",
                 risk_percent: float = 0.01,
                 min_risk_reward: float = 2.0,
                 atr_multiplier: float = 2.0,
                 # --- Indicator Parameters (with defaults from PremiumLuxAlgo) ---
                 use_major_tp: bool = True,
                 use_minor_tp: bool = False,
                 supertrend_factor: float = 5.5,
                 supertrend_atr_period: int = 11,
                 tp_maj_qual: int = 13,
                 tp_maj_len: int = 40,
                 tp_min_qual: int = 5,
                 tp_min_len: int = 5,
                 ha_len: int = 100,
                 rng_qty: float = 2.618,
                 rng_per: int = 14,
                 rng_smooth_per: int = 27,
                 sr_strength: int = 4,
                 sr_lookback: int = 284,
                 **kwargs):
        """
        Initializes the PineScript-based strategy.
        """
        super().__init__(**kwargs)

        self.name = "PremiumLuxAlgoStrategy"
        self.description = "A strategy using a collection of PineScript-ported indicators."
        self.version = "1.0.0"

        self.primary_timeframe = primary_timeframe
        self.min_risk_reward = min_risk_reward
        self.atr_multiplier = atr_multiplier

        # --- Store Indicator Parameters ---
        self.params = {
            'use_major_tp': use_major_tp,
            'use_minor_tp': use_minor_tp,
            'supertrend_factor': supertrend_factor,
            'supertrend_atr_period': supertrend_atr_period,
            'tp_maj_qual': tp_maj_qual,
            'tp_maj_len': tp_maj_len,
            'tp_min_qual': tp_min_qual,
            'tp_min_len': tp_min_len,
            'ha_len': ha_len,
            'rng_qty': rng_qty,
            'rng_per': rng_per,
            'rng_smooth_per': rng_smooth_per,
            'sr_strength': sr_strength,
            'sr_lookback': sr_lookback,
        }

        rm_conf = get_risk_manager_config()
        self.risk_percent = rm_conf.get('max_risk_per_trade', risk_percent)

        self.lookback = None
        self._load_timeframe_profile()

        self.processed_bars = {}

    def _load_timeframe_profile(self):
        """
        Loads parameters from TIMEFRAME_PROFILES based on the primary timeframe.
        """
        profile = TIMEFRAME_PROFILES.get(self.primary_timeframe, DEFAULT_PROFILE)
        self.lookback = profile.get('lookback', DEFAULT_PROFILE['lookback'])
        self.params['sr_lookback'] = self.lookback # Sync lookback with SR lookback

        logger.info(
            f"🔄 [{self.name}] Profile loaded for {self.primary_timeframe}: lookback={self.lookback}"
        )

    @property
    def required_timeframes(self) -> List[str]:
        """
        Specifies the timeframes this strategy needs. Since we are only using
        a primary timeframe, this list contains just one element.
        """
        return [self.primary_timeframe]

    async def generate_signals(
        self,
        market_data: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None,
        **kwargs
    ) -> List[Dict]:
        """
        The core method where trading signals are generated.
        This will be implemented in the next step.
        """
        if market_data is None:
            market_data = {}

        signals = []
        rm = RiskManager.get_instance()
        balance = kwargs.get("balance", rm.daily_stats.get('starting_balance', 10000))

        for sym, frames in market_data.items():
            primary_df = frames.get(self.primary_timeframe)

            if not isinstance(primary_df, pd.DataFrame) or primary_df.empty or self.lookback is None or len(primary_df) < self.lookback:
                logger.debug(f"[{sym}] Missing, empty, or insufficient market data for {self.primary_timeframe}.")
                continue

            # --- Prevent Re-processing the Same Bar ---
            try:
                last_timestamp = str(primary_df.index[-1])
                bar_key = (sym, self.primary_timeframe)
                if self.processed_bars.get(bar_key) == last_timestamp:
                    continue
                self.processed_bars[bar_key] = last_timestamp
            except IndexError:
                logger.warning(f"[{sym}] Could not get last timestamp from primary_df.")
                continue
            
            # 1. Instantiate PremiumLuxAlgo and calculate indicators
            try:
                lux_algo = PremiumLuxAlgo(primary_df.copy())
                # Use a simplified signal set for clarity
                st, st_dir = lux_algo.supertrend(
                    factor=self.params['supertrend_factor'],
                    atr_period=self.params['supertrend_atr_period']
                )
                major_tp, minor_tp = lux_algo.calculate_tp_points(
                    maj_qual=self.params['tp_maj_qual'],
                    maj_len=self.params['tp_maj_len'],
                    min_qual=self.params['tp_min_qual'],
                    min_len=self.params['tp_min_len']
                )
            except Exception as e:
                logger.error(f"[{sym}] Error calculating indicators: {e}")
                continue

            # 2. Define entry conditions based on indicator confluence
            last_close = primary_df['close'].iloc[-1]
            prev_close = primary_df['close'].iloc[-2]
            
            last_st = st.iloc[-1]
            prev_st = st.iloc[-2]
            sma9 = lux_algo.sma(primary_df['close'], 9).iloc[-1]

            # Bullish crossover: close crosses above Supertrend and is above SMA(9)
            bull_cond = (last_close > last_st) and (prev_close <= prev_st) and (last_close >= sma9)
            # Bearish crossover: close crosses below Supertrend and is below SMA(9)
            bear_cond = (last_close < last_st) and (prev_close >= prev_st) and (last_close <= sma9)

            direction = None
            if bull_cond:
                direction = 'buy'
            elif bear_cond:
                direction = 'sell'
            
            if direction is None:
                continue # No signal

            # 3. Calculate TP score and confidence
            last_major_tp = major_tp.iloc[-1]
            last_minor_tp = minor_tp.iloc[-1]
            tp_score = 0
            
            if direction == 'buy':
                if self.params['use_minor_tp'] and last_minor_tp == 1:
                    tp_score = 1
                if self.params['use_major_tp'] and last_major_tp == 1:
                    tp_score = 2
                if (self.params['use_major_tp'] and last_major_tp == 1) and \
                   (self.params['use_minor_tp'] and last_minor_tp == 1):
                    tp_score = 3
            elif direction == 'sell':
                if self.params['use_minor_tp'] and last_minor_tp == -1:
                    tp_score = 1
                if self.params['use_major_tp'] and last_major_tp == -1:
                    tp_score = 2
                if (self.params['use_major_tp'] and last_major_tp == -1) and \
                   (self.params['use_minor_tp'] and last_minor_tp == -1):
                    tp_score = 3
            
            confidence = 0.75  # Base confidence
            if tp_score == 2:
                confidence = 0.85
            elif tp_score == 3:
                confidence = 0.95

            # 4. Construct and validate signal dictionary
            last_candle = primary_df.iloc[-1]
            entry = last_candle['close']
            atr = lux_algo.atr(period=self.params['supertrend_atr_period']).iloc[-1]
            
            if direction == 'buy':
                stop_loss = entry - (atr * self.atr_multiplier)
                take_profit = entry + ((entry - stop_loss) * self.min_risk_reward)
            else: # sell
                stop_loss = entry + (atr * self.atr_multiplier)
                take_profit = entry - ((stop_loss - entry) * self.min_risk_reward)

            signal_details = {
                "symbol": sym,
                "direction": direction,
                "entry_price": entry,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "timeframe": self.primary_timeframe,
                "strategy_name": self.name,
                "confidence": confidence,
                "description": f"Signal based on Supertrend crossover and SMA(9) confirmation.",
                "detailed_reasoning": [
                    f"Supertrend Crossover: {direction.upper()}",
                    f"SMA(9) Confirmation: Close ({last_close:.5f}) >= SMA(9) ({sma9:.5f})" if direction == 'buy' else f"SMA(9) Confirmation: Close ({last_close:.5f}) <= SMA(9) ({sma9:.5f})",
                    f"TP Score: {tp_score}"
                ],
                "signal_timestamp": str(last_candle.name),
            }

            # 5. Validate the Trade with the RiskManager
            validation_result = rm.validate_and_size_trade(signal_details)

            if validation_result['is_valid']:
                logger.info(f"✅ [{sym}] Valid signal generated. Direction: {direction}, Entry: {entry:.5f}")
                final_trade_params = validation_result['final_trade_params']
                signals.append(final_trade_params)
            else:
                logger.warning(f"Signal for {sym} rejected by RiskManager: {validation_result['reason']}")

        return signals