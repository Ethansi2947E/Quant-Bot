import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from loguru import logger
from datetime import datetime

# Core framework imports
from src.trading_bot import SignalGenerator
from config.config import RISK_MANAGER_CONFIG
from src.risk_manager import RiskManager

class Supertrend:
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
    
    def rma(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate RMA (Wilder's Moving Average)"""
        return series.ewm(alpha=1/period, adjust=False).mean()
    
    def atr(self, period: int) -> pd.Series:
        """Calculate ATR"""
        high = self.data['high']
        low = self.data['low']
        close = self.data['close']
        
        tr1 = high - low
        tr2 = np.abs(high - close.shift(1))
        tr3 = np.abs(low - close.shift(1))
        
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        return self.rma(tr, period) # PineScript's ta.atr uses RMA
    
    def supertrend(self, factor: float = 5.5, atr_period: int = 11) -> Tuple[pd.Series, pd.Series]:
        """
        Calculates Supertrend, ported from the stateful PineScript logic.
        """
        close = self.data['close']
        high = self.data['high']
        low = self.data['low']
        atr = self.atr(atr_period)
        
        # PineScript's direction logic is a bit unusual.
        # We'll use 1 for uptrend and -1 for downtrend.
        direction = pd.Series(1, index=close.index) 
        supertrend = pd.Series(np.nan, index=close.index)
        
        # Initial ATR value is needed for the first calculation
        first_valid_atr_index = atr.first_valid_index()
        if first_valid_atr_index is None:
            return supertrend, direction # Not enough data
        
        # Robustly get the integer position for the first valid ATR value.
        first_valid_loc = int(np.atleast_1d(atr.index.searchsorted(first_valid_atr_index))[0])

        # Vectorized calculation of upper/lower bands
        upper_band = high + factor * atr
        lower_band = low - factor * atr

        # Start the stateful calculation loop
        for i in range(first_valid_loc, len(close)):
            # On the first bar, we can't look back. Initialize based on close vs upper band.
            if i == 0:
                if close.iloc[i] > upper_band.iloc[i]:
                    direction.iloc[i] = 1
                else:
                    direction.iloc[i] = -1
                continue

            # Carry over the previous direction
            direction.iloc[i] = direction.iloc[i-1]

            # Determine the previous supertrend value to compare against
            prev_supertrend = supertrend.iloc[i-1]

            # Main state logic from PineScript
            if direction.iloc[i-1] == 1: # Previous was an uptrend
                if close.iloc[i] < prev_supertrend:
                    direction.iloc[i] = -1 # Flip to downtrend
            else: # Previous was a downtrend
                if close.iloc[i] > prev_supertrend:
                    direction.iloc[i] = 1 # Flip to uptrend

            # Update the supertrend value for the current bar
            if direction.iloc[i] == 1:
                supertrend.iloc[i] = max(lower_band.iloc[i], supertrend.iloc[i-1]) if not pd.isna(supertrend.iloc[i-1]) else lower_band.iloc[i]
            else:
                supertrend.iloc[i] = min(upper_band.iloc[i], supertrend.iloc[i-1]) if not pd.isna(supertrend.iloc[i-1]) else upper_band.iloc[i]
                
        return supertrend, direction
    
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
                bindex.iloc[i] = bindex.iloc[i-1] + 1 if close.iloc[i] > close.iloc[i-4] else bindex.iloc[i-1]
                sindex.iloc[i] = sindex.iloc[i-1] + 1 if close.iloc[i] < close.iloc[i-4] else sindex.iloc[i-1]
                
                if (bindex.iloc[i] > qual and close.iloc[i] < open_price.iloc[i] and 
                    high.iloc[i] >= high.iloc[max(0, i-length):i].max()):
                    bindex.iloc[i] = 0
                    ret.iloc[i] = -1
                
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
        
        haopen.iloc[0] = xhaopen.iloc[0] if not xhaopen.empty else np.nan
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
        close_prev = close.shift(1)
        high_prev = high.shift(1)

        hilo = high - low
        href = np.where(low <= high_prev, high - close_prev, high - close_prev - 0.5 * (low - high_prev))
        lref = np.where(high >= low.shift(1), close_prev - low, close_prev - low - 0.5 * (low.shift(1) - high))
        true_range = pd.DataFrame({'hilo': hilo, 'href': href, 'lref': lref}).max(axis=1)
        
        wild_atr = self.rma(true_range, atr_period)
        loss = atr_factor * wild_atr
        
        up = close - loss
        dn = close + loss
        
        trend_up = pd.Series(index=close.index, dtype=float)
        trend_down = pd.Series(index=close.index, dtype=float)
        trend = pd.Series(1, index=close.index)
        
        trend_up.iloc[0] = up.iloc[0]
        trend_down.iloc[0] = dn.iloc[0]
        
        for i in range(1, len(close)):
            if close.iloc[i-1] > trend_up.iloc[i-1]: trend_up.iloc[i] = max(up.iloc[i], trend_up.iloc[i-1])
            else: trend_up.iloc[i] = up.iloc[i]
            
            if close.iloc[i-1] < trend_down.iloc[i-1]: trend_down.iloc[i] = min(dn.iloc[i], trend_down.iloc[i-1])
            else: trend_down.iloc[i] = dn.iloc[i]
            
            if close.iloc[i] > trend_down.iloc[i-1]: trend.iloc[i] = 1
            elif close.iloc[i] < trend_up.iloc[i-1]: trend.iloc[i] = -1
            else: trend.iloc[i] = trend.iloc[i-1]
        
        trail = np.where(trend == 1, trend_up, trend_down)
        return self.sma(pd.Series(trail, index=close.index), smoothing)
    
    def reversal_signals(self, rsi_period: int = 14, overbought: float = 75,
                        oversold: float = 25) -> Tuple[pd.Series, pd.Series]:
        """Calculate Reversal Signals"""
        close = self.data['close']
        
        delta = close.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        
        avg_gain = self.rma(up, rsi_period)
        avg_loss = self.rma(down, rsi_period)
        
        rs = avg_gain / avg_loss.replace(0, 1)
        rsi = 100 - (100 / (1 + rs))
        
        rev_up = (rsi > oversold) & (rsi.shift(1) <= oversold)
        rev_down = (rsi < overbought) & (rsi.shift(1) >= overbought)
        
        return rev_up, rev_down
    
    def support_resistance(self, strength: int = 4, lookback: int = 284) -> dict:
        """Calculate Support and Resistance levels"""
        high = self.data['high']
        low = self.data['low']
        rb = 10  # bars left and right
        
        is_pivot_high = (high == high.rolling(2 * rb + 1, center=True).max())
        is_pivot_low = (low == low.rolling(2 * rb + 1, center=True).min())
        
        pivot_high_vals = high[is_pivot_high].dropna()
        pivot_low_vals = low[is_pivot_low].dropna()

        all_pivots = pd.concat([pivot_high_vals, pivot_low_vals]).sort_index().iloc[-lookback:]
        
        if all_pivots.empty:
            return {'levels': [], 'pivot_highs': high[is_pivot_high], 'pivot_lows': low[is_pivot_low]}

        tolerance = (high.max() - low.min()) * 0.01
        
        sr_levels = []
        pivots_list = all_pivots.to_list()
        
        while len(pivots_list) > 0:
            level_group = [p for p in pivots_list if abs(p - pivots_list[0]) <= tolerance]
            if len(level_group) >= strength:
                sr_levels.append(np.mean(level_group))
            
            pivots_list = [p for p in pivots_list if p not in level_group]

        return {
            'levels': sorted(list(set(sr_levels))),
            'pivot_highs': high[is_pivot_high],
            'pivot_lows': low[is_pivot_low]
        }
    
    def reversal_cloud(self, length: int = 50, bd1: int = 9, bd2: int = 11, bd3: int = 14, kama_fast: int = 2, kama_slow: int = 30) -> dict:
        """Calculate Reversal Cloud (Lux Algo style)"""
        close = self.data['close']
        
        def kama(src, length, fast, slow):
            change = np.abs(src.diff(1))
            volatility = change.rolling(window=length).sum()
            direction = np.abs(src - src.shift(length))
            
            efficiency_ratio = (direction / volatility.replace(0, 1)).fillna(0)
            
            fast_alpha = 2.0 / (fast + 1)
            slow_alpha = 2.0 / (slow + 1)
            smoothing_constant = (efficiency_ratio * (fast_alpha - slow_alpha) + slow_alpha) ** 2
            
            kama_val = pd.Series(index=src.index, dtype=float)
            kama_val.iloc[0] = src.iloc[0]
            
            for i in range(1, len(src)):
                kama_val.iloc[i] = kama_val.iloc[i-1] + smoothing_constant.iloc[i] * (src.iloc[i] - kama_val.iloc[i-1])
            return kama_val
        
        tr = pd.DataFrame({
            'a': self.data['high'] - self.data['low'],
            'b': np.abs(self.data['high'] - close.shift(1)),
            'c': np.abs(self.data['low'] - close.shift(1))
        }).max(axis=1)
        
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
    
    def psar(self, initial_af=0.02, max_af=0.2, af_step=0.02) -> pd.Series:
        high, low = self.data['high'], self.data['low']
        psar = low.copy()
        bull = True
        af = initial_af
        ep = high.iloc[0]

        for i in range(2, len(self.data)):
            if bull:
                psar.iloc[i] = psar.iloc[i-1] + af * (ep - psar.iloc[i-1])
                if low.iloc[i] < psar.iloc[i]:
                    bull = False
                    psar.iloc[i] = ep
                    ep = low.iloc[i]
                    af = initial_af
                else:
                    if high.iloc[i] > ep:
                        ep = high.iloc[i]
                        af = min(af + af_step, max_af)
            else:
                psar.iloc[i] = psar.iloc[i-1] - af * (psar.iloc[i-1] - ep)
                if high.iloc[i] > psar.iloc[i]:
                    bull = True
                    psar.iloc[i] = ep
                    ep = high.iloc[i]
                    af = initial_af
                else:
                    if low.iloc[i] < ep:
                        ep = low.iloc[i]
                        af = min(af + af_step, max_af)
        return psar

    def calculate_all_indicators(self, sensitivity: float = 5.5) -> pd.DataFrame:
        """Calculate all indicators and return results dataframe"""
        
        # Initialize the sr_data attribute to None
        self.sr_data = None

        # Basic calculations
        self.results['oc_avg'] = (self.data['open'] + self.data['close']) / 2
        
        # Supertrend
        supertrend, st_direction = self.supertrend(sensitivity)
        self.results['supertrend'] = supertrend
        self.results['st_direction'] = st_direction
        
        # SMA calculations (FIX: Use 13 period as per original script, not 9)
        close = self.data['close']
        self.results['sma13'] = self.sma(close, 13)
        
        # Buy/Sell signals (REFINEMENT: Decoupled from SMA filter)
        bull_signal = (close > supertrend) & (close.shift(1) <= supertrend.shift(1))
        bear_signal = (close < supertrend) & (close.shift(1) >= supertrend.shift(1))
        
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
        self.results['reversal_up'], self.results['reversal_down'] = rev_up, rev_down
        
        # Support/Resistance
        self.sr_data = self.support_resistance()
        self.results['sr_pivot_highs'] = self.sr_data['pivot_highs']
        self.results['sr_pivot_lows'] = self.sr_data['pivot_lows']
        
        # Reversal Cloud
        cloud = self.reversal_cloud()
        for key, value in cloud.items(): self.results[f'cloud_{key}'] = value
        
        self.results['psar'] = self.psar()
        
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


class SuperT(SignalGenerator):
    def __init__(self,
                 primary_timeframe: str = "M1", risk_percent: float = 0.01,
                 min_risk_reward: float = 2.0, atr_multiplier: float = 2.0,
                 use_major_tp: bool = True, use_minor_tp: bool = False,
                 supertrend_factor: float = 5.5, supertrend_atr_period: int = 11,
                 **kwargs):
        super().__init__(**kwargs)
        self.name = "PremiumLuxAlgoStrategy"
        self.description = "A strategy using a collection of PineScript-ported indicators."
        self.version = "1.1.0" # Updated version
        self.primary_timeframe = primary_timeframe
        self.min_risk_reward = min_risk_reward
        self.atr_multiplier = atr_multiplier
        self.params = {
            'use_major_tp': use_major_tp, 'use_minor_tp': use_minor_tp,
            'supertrend_factor': supertrend_factor,
            'supertrend_atr_period': supertrend_atr_period,
        }
        self.risk_percent = RISK_MANAGER_CONFIG.get('max_risk_per_trade', risk_percent)
        self.lookback = None
        self._load_timeframe_profile()
        self.processed_bars = {}

    def _load_timeframe_profile(self):
        profile = TIMEFRAME_PROFILES.get(self.primary_timeframe, DEFAULT_PROFILE)
        self.lookback = profile.get('lookback', DEFAULT_PROFILE['lookback'])
        logger.info(f"🔄 [{self.name}] Profile loaded for {self.primary_timeframe}: lookback={self.lookback}")

    @property
    def lookback_periods(self) -> Dict[str, int]:
        """
        Exposes the lookback period required for each timeframe.
        The trading_bot uses this to fetch the correct amount of historical data.
        """
        assert self.lookback is not None, "Lookback must be initialized before being accessed."
        return {self.primary_timeframe: self.lookback}
        
    @property
    def required_timeframes(self) -> List[str]:
        return [self.primary_timeframe]

    async def generate_signals(
        self, market_data: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None, **kwargs
    ) -> List[Dict]:
        if market_data is None: market_data = {}

        signals = []
        rm = RiskManager.get_instance()
        
        for sym, frames in market_data.items():
            primary_df = frames.get(self.primary_timeframe)
            if not isinstance(primary_df, pd.DataFrame) or primary_df.empty or self.lookback is None or len(primary_df) < self.lookback:
                logger.debug(f"[{sym}] Insufficient data for {self.primary_timeframe}.")
                continue
            
            try:
                last_timestamp = str(primary_df.index[-1])
                if self.processed_bars.get((sym, self.primary_timeframe)) == last_timestamp: continue
                self.processed_bars[(sym, self.primary_timeframe)] = last_timestamp
            except IndexError: continue

            logger.debug(f"[{sym}] Analyzing data for timestamp: {last_timestamp}")

            try:
                lux_algo = Supertrend(primary_df.copy())
                # Pass strategy params to the indicator calculation
                results = lux_algo.calculate_all_indicators(sensitivity=self.params['supertrend_factor'])
            except Exception as e:
                logger.error(f"[{sym}] Error calculating indicators: {e}")
                continue
            
            # --- CORRECTION: Use the pre-calculated results from the LuxAlgo class ---
            last = results.iloc[-1]
            
            logger.debug(f"[{sym}] Last candle data: Close={last['close']:.5f}, SMA13={last['sma13']:.5f}, "
                         f"Supertrend={last['supertrend']:.5f}, ST_Dir={last['st_direction']}, "
                         f"Bull={last['bull_signal']}, Bear={last['bear_signal']}, "
                         f"MajorTP={last['major_tp']}, MinorTP={last['minor_tp']}")

            # REFINEMENT: Apply the SMA(13) filter here, not during signal calculation
            bull_cond = last['bull_signal'] and last['close'] >= last['sma13']
            bear_cond = last['bear_signal'] and last['close'] <= last['sma13']

            direction = 'buy' if bull_cond else 'sell' if bear_cond else None
            if not direction:
                logger.trace(f"[{sym}] No signal condition met at {last.name}. Bull: {bull_cond}, Bear: {bear_cond}")
                continue

            tp_score = 0
            if direction == 'buy':
                if self.params['use_minor_tp'] and last['minor_tp'] == 1: tp_score += 1
                if self.params['use_major_tp'] and last['major_tp'] == 1: tp_score += 2
            else: # sell
                if self.params['use_minor_tp'] and last['minor_tp'] == -1: tp_score += 1
                if self.params['use_major_tp'] and last['major_tp'] == -1: tp_score += 2
            
            logger.debug(f"[{sym}] Direction: {direction}, TP Score: {tp_score}")

            confidence = 0.75 + (tp_score * 0.1) # Base confidence 0.75, max 0.95

            entry = last['close']
            atr = lux_algo.atr(period=self.params['supertrend_atr_period']).iloc[-1]
            
            # --- FEATURE: Multiple Take-Profit Levels ---
            # Generate a list of TP levels instead of a single one.
            take_profits = []
            if direction == 'buy':
                stop_loss = entry - (atr * self.atr_multiplier)
                # TP1: Standard risk/reward
                tp1 = entry + ((entry - stop_loss) * self.min_risk_reward)
                # TP2: Double risk/reward
                tp2 = entry + ((entry - stop_loss) * self.min_risk_reward * 2)
                take_profits.extend([tp1, tp2])
            else: # sell
                stop_loss = entry + (atr * self.atr_multiplier)
                # TP1: Standard risk/reward
                tp1 = entry - ((stop_loss - entry) * self.min_risk_reward)
                # TP2: Double risk/reward
                tp2 = entry - ((stop_loss - entry) * self.min_risk_reward * 2)
                take_profits.extend([tp1, tp2])

            logger.debug(f"[{sym}] Calculated SL: {stop_loss:.5f}, TPs: {[f'{tp:.5f}' for tp in take_profits]}, ATR: {atr:.5f}")

            signal_details = {
                "symbol": sym, "direction": direction, "entry_price": entry,
                "stop_loss": stop_loss, "take_profits": take_profits, # Use plural form for list
                "timeframe": self.primary_timeframe, "strategy_name": self.name,
                "confidence": confidence, "signal_timestamp": str(last.name),
                "detailed_reasoning": [
                    f"Supertrend Crossover: {direction.upper()}",
                    f"SMA(13) Confirmation: Close ({last['close']:.5f}) {' >=' if direction == 'buy' else ' <='} SMA(13) ({last['sma13']:.5f})",
                    f"TP Score: {tp_score}"
                ]
            }

            validation_result = rm.validate_and_size_trade(signal_details)

            if validation_result['is_valid']:
                logger.info(f"✅ [{sym}] Valid signal generated. Dir: {direction}, Entry: {entry:.5f}, Conf: {confidence:.2f}")
                signals.append(validation_result['final_trade_params'])
            else:
                logger.warning(f"Signal for {sym} rejected by RiskManager: {validation_result['reason']}")

        return signals