import pandas as pd
import numpy as np
import math
from typing import Tuple, List, Optional, Dict, Any
from collections import deque
from loguru import logger
from numba import njit

from src.trading_bot import SignalGenerator
from src.risk_manager import RiskManager
from config.config import RISK_MANAGER_CONFIG


# --- Numba-Optimized Helper Functions ---
@njit
def rngfilt_numba(x: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Numba-accelerated version of the rngfilt function."""
    rngfilt = np.zeros_like(x)
    rngfilt[0] = x[0]
    for i in range(1, len(x)):
        prev_filt = rngfilt[i - 1]
        if x[i] > prev_filt:
            rngfilt[i] = max(prev_filt, x[i] - r[i])
        else:
            rngfilt[i] = min(prev_filt, x[i] + r[i])
    return rngfilt

@njit
def find_pivots_numba(series: np.ndarray, left: int, right: int) -> Tuple[np.ndarray, np.ndarray]:
    """ Numba-accelerated pivot finder, matching Pine's ta.pivothigh/low logic. """
    highs = np.full(series.shape[0], np.nan)
    lows = np.full(series.shape[0], np.nan)
    
    # Pine's pivot is centered, so we need to look at a window of size left+right+1
    # The pivot point is at index 'left' within that window.
    for i in range(left, len(series) - right):
        window_high = series[i - left : i + right + 1]
        window_low = series[i - left : i + right + 1]
        
        if series[i] == np.max(window_high):
            highs[i] = series[i]
        if series[i] == np.min(window_low):
            lows[i] = series[i]
            
    return highs, lows

# --- Encapsulated Indicator Calculation Logic ---
class _GarbageAlgoCalculator:
    """
    Internal class to handle all indicator calculations for the GarbageAlgo.
    This keeps the main SignalGenerator class clean and focused on strategy logic.
    """
    def __init__(self, data: pd.DataFrame, params: Dict[str, Any]):
        self.data = data.copy()
        self.data.columns = self.data.columns.str.lower()
        self.params = params

        # Ensure numeric dtype – prevents pandas-object comparison errors
        for col in ("open", "high", "low", "close"):
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors="coerce")

        self.results = self.data.copy()
        
        # State-dependent objects for Supply/Demand
        self.supply_zones = deque(maxlen=self.params['history_of_demand_to_keep'])
        self.demand_zones = deque(maxlen=self.params['history_of_demand_to_keep'])
        self.box_id_counter = 0

    # --- TA Helper Methods ---
    def _rma(self, series: pd.Series, period: int) -> pd.Series:
        alpha = 1.0 / period
        return series.ewm(alpha=alpha, adjust=False).mean()

    def _wma(self, series: pd.Series, period: int) -> pd.Series:
        weights = np.arange(1, period + 1)
        return series.rolling(window=period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    def _alma(self, series: pd.Series, windowsize: int, offset: float, sigma: float) -> pd.Series:
        m = offset * (windowsize - 1)
        s = windowsize / sigma
        w = np.exp(-((np.arange(windowsize) - m) ** 2) / (2 * s * s))
        return series.rolling(window=windowsize).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)

    # --- Core Calculation Methods ---
    def _calculate_supertrend(self):
        atr = self._rma(self.results['high'] - self.results['low'], 11)
        factor = self.params['nsensitivity'] * 2
        
        upper_band = self.results['close'] + factor * atr
        lower_band = self.results['close'] - factor * atr
        
        supertrend = np.zeros(len(self.results))
        direction = np.ones(len(self.results))

        for i in range(1, len(self.results)):
            # Update bands based on previous values (direct Pine translation)
            if lower_band.iloc[i] > lower_band.iloc[i-1] or self.results['close'].iloc[i-1] < lower_band.iloc[i-1]:
                pass # Use current calculation
            else:
                lower_band.iloc[i] = lower_band.iloc[i-1]
            
            if upper_band.iloc[i] < upper_band.iloc[i-1] or self.results['close'].iloc[i-1] > upper_band.iloc[i-1]:
                pass # Use current calculation
            else:
                upper_band.iloc[i] = upper_band.iloc[i-1]

            # Determine direction and supertrend value
            if supertrend[i-1] == upper_band.iloc[i-1]:
                direction[i] = -1 if self.results['close'].iloc[i] > upper_band.iloc[i] else 1
            else: # supertrend[i-1] == lower_band.iloc[i-1]
                direction[i] = 1 if self.results['close'].iloc[i] < lower_band.iloc[i] else -1

            supertrend[i] = lower_band.iloc[i] if direction[i] == 1 else upper_band.iloc[i]

        self.results['supertrend'] = supertrend
        self.results['direction'] = direction

    def _calculate_adx(self):
        dilen, adxlen = self.params['dilen'], self.params['adxlen']
        tr = (self.results['high'] - self.results['low']).combine_first(abs(self.results['high'] - self.results['close'].shift())).combine_first(abs(self.results['low'] - self.results['close'].shift()))
        truerange = self._rma(tr, dilen)

        up, down = self.results['high'].diff(), -self.results['low'].diff()

        # Coerce to numeric right before use to satisfy linter and prevent type errors
        up = pd.to_numeric(up, errors='coerce').fillna(0)
        down = pd.to_numeric(down, errors='coerce').fillna(0)

        plusDM = self._rma(up.where((up > down) & (up > 0), 0), dilen)
        minusDM = self._rma(down.where((down > up) & (down > 0), 0), dilen)

        plus, minus = 100 * plusDM / truerange, 100 * minusDM / truerange
        sum_val = (plus + minus).replace(0, 1)
        self.results['adx'] = 100 * self._rma(abs(plus - minus) / sum_val, adxlen)

    def _calculate_trend_features(self):
        # Trend Follower (using numba for rngfilt)
        smrng = self._rma(abs(self.results['close'].diff()), 22).ewm(span=43, adjust=False).mean() * 6
        self.results['TrendFollower_filt'] = rngfilt_numba(self.results['close'].to_numpy(), smrng.to_numpy())
        
        # Long Trend Average (Hull MA)
        wma_half = self._wma(self.results['close'], 300)
        wma_full = self._wma(self.results['close'], 600)
        self.results['LongTrendAverage_hullma'] = self._wma(2 * wma_half - wma_full, int(np.sqrt(600)))
        
        # Clouds
        self.results['Comulus_candle'] = self._alma(self.results['close'], 310, 0.85, 32)
        self.results['Comulus_reach'] = self._alma(self.results['close'], 100, 0.9, 6)

    def _calculate_supply_demand(self):
        if not self.params['showsr']: return

        swing_len = self.params['swing_length']
        pivot_highs_np, pivot_lows_np = find_pivots_numba(self.results['high'].to_numpy(), swing_len, swing_len)
        
        self.results['pivot_high'] = pd.Series(pivot_highs_np, index=self.results.index)
        self.results['pivot_low'] = pd.Series(pivot_lows_np, index=self.results.index)

        atr50 = self._rma(self.results['high'] - self.results['low'], 50)
        
        for i in range(1, len(self.results)):
            # Break existing zones - Use .iloc for positional access
            self.supply_zones = deque([box for box in self.supply_zones if self.results['close'].iloc[i] < box['top']])
            self.demand_zones = deque([box for box in self.demand_zones if self.results['close'].iloc[i] > box['bottom']])

            # Create new zones from pivots - Use .iloc for positional access
            if pd.notna(self.results['pivot_high'].iloc[i]):
                atr_buffer = atr50.iloc[i] * (self.params['box_width'] / 10)
                box_top, box_bottom = self.results['pivot_high'].iloc[i], self.results['pivot_high'].iloc[i] - atr_buffer
                poi = (box_top + box_bottom) / 2
                
                if all(abs(poi - (b['top'] + b['bottom'])/2) > atr50.iloc[i] * 2 for b in self.supply_zones):
                    self.box_id_counter += 1
                    self.supply_zones.append({"id": self.box_id_counter, "left_idx": i, "top": box_top, "bottom": box_bottom})

            if pd.notna(self.results['pivot_low'].iloc[i]):
                atr_buffer = atr50.iloc[i] * (self.params['box_width'] / 10)
                box_bottom, box_top = self.results['pivot_low'].iloc[i], self.results['pivot_low'].iloc[i] + atr_buffer
                poi = (box_top + box_bottom) / 2

                if all(abs(poi - (b['top'] + b['bottom'])/2) > atr50.iloc[i] * 2 for b in self.demand_zones):
                    self.box_id_counter += 1
                    self.demand_zones.append({"id": self.box_id_counter, "left_idx": i, "top": box_top, "bottom": box_bottom})

    def calculate_all_indicators(self) -> pd.DataFrame:
        self.results['sma9'] = self.results['close'].rolling(window=13).mean()
        self._calculate_supertrend()
        self._calculate_adx()
        self._calculate_trend_features()
        self._calculate_supply_demand()

        # Signal Conditions
        self.results['bull'] = (self.results['close'] > self.results['supertrend']) & (self.results['close'].shift(1) <= self.results['supertrend'].shift(1)) & (self.results['close'] >= self.results['sma9'])
        self.results['bear'] = (self.results['close'] < self.results['supertrend']) & (self.results['close'].shift(1) >= self.results['supertrend'].shift(1)) & (self.results['close'] <= self.results['sma9'])
        
        # Risk Management Levels calculation
        self.results['atr_for_risk'] = self._rma(self.results['high'] - self.results['low'], self.params['atrLen'])
        
        return self.results


# --- Timeframe-Specific Parameter Profiles ---
TIMEFRAME_PROFILES = { "default": {"lookback": 650} } # 600 for Hull MA + 50 buffer

# --- The Signal Generator Class ---
class GarbageAlgoStrategy(SignalGenerator):
    def __init__(self,
                 primary_timeframe: str = "M5",
                 risk_percent: float = 0.01,
                 # Pine Script Inputs
                 nbuysell: bool = True, nsensitivity: float = 2.0,
                 LongTrendAverage: bool = False, TrendFollower: bool = False,
                 ShowComulus: bool = False, CirrusCloud: bool = False,
                 showsr: bool = True, swing_length: int = 8, box_width: float = 4.0,
                 sidewaysThreshold: int = 15, atrRisk: int = 1, atrLen: int = 14,
                 dilen: int = 15, adxlen: int = 15,
                 history_of_demand_to_keep: int = 20,
                 **kwargs):
        super().__init__(**kwargs)
        self.name = "GarbageAlgoStrategy"
        self.description = "Python implementation of the 'Garbage Algo' TradingView script."
        self.version = "1.0.0"
        
        self.primary_timeframe = primary_timeframe
        self.risk_percent = RISK_MANAGER_CONFIG.get('max_risk_per_trade', risk_percent)
        
        self.params = {
            'nbuysell': nbuysell, 'nsensitivity': nsensitivity,
            'LongTrendAverage': LongTrendAverage, 'TrendFollower': TrendFollower,
            'ShowComulus': ShowComulus, 'CirrusCloud': CirrusCloud,
            'showsr': showsr, 'swing_length': swing_length, 'box_width': box_width,
            'sidewaysThreshold': sidewaysThreshold, 'atrRisk': atrRisk, 'atrLen': atrLen,
            'dilen': dilen, 'adxlen': adxlen,
            'history_of_demand_to_keep': history_of_demand_to_keep
        }
        
        self.lookback = None
        self._load_timeframe_profile()
        self.processed_bars = {}

    def _load_timeframe_profile(self):
        self.lookback = TIMEFRAME_PROFILES.get(self.primary_timeframe, TIMEFRAME_PROFILES['default'])['lookback']
        logger.info(f"🔄 [{self.name}] Profile loaded for {self.primary_timeframe}: lookback={self.lookback}")

    @property
    def lookback_periods(self) -> Dict[str, int]:
        assert self.lookback is not None, "Lookback must be initialised"
        return {self.primary_timeframe: self.lookback}
        
    @property
    def required_timeframes(self) -> List[str]:
        return [self.primary_timeframe]

    async def generate_signals(self, market_data: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None, **kwargs) -> List[Dict]:
        if market_data is None: market_data = {}
        signals = []
        rm = RiskManager.get_instance()
        
        for sym, frames in market_data.items():
            primary_df = frames.get(self.primary_timeframe)
            if not isinstance(primary_df, pd.DataFrame):
                logger.trace(f"[{sym}] No DataFrame available for {self.primary_timeframe}.")
                continue

            if self.lookback is None or len(primary_df) < self.lookback:
                logger.trace(f"[{sym}] Insufficient data for {self.primary_timeframe}. Have {len(primary_df)}, need {self.lookback}.")
                continue
            
            try:
                last_timestamp = str(primary_df.index[-1])
                if self.processed_bars.get((sym, self.primary_timeframe)) == last_timestamp:
                    continue
                self.processed_bars[(sym, self.primary_timeframe)] = last_timestamp
            except IndexError:
                continue

            try:
                calculator = _GarbageAlgoCalculator(primary_df, self.params)
                results = calculator.calculate_all_indicators()
            except Exception as e:
                logger.error(f"[{sym}] Error calculating indicators for {self.name}: {e}")
                continue
            
            last = results.iloc[-1]
            
            # --- STRATEGY LOGIC & Structured Logging ---
            is_bull_signal = last['bull']
            is_bear_signal = last['bear']
            is_sideways = last['adx'] < self.params['sidewaysThreshold']
            
            is_long_candidate = self.params['nbuysell'] and is_bull_signal and not is_sideways
            is_short_candidate = self.params['nbuysell'] and is_bear_signal and not is_sideways
            
            log_msg = f"[{sym}][{self.name}] Condition Analysis for Bar {last.name}:\n"
            log_msg += f"  - LONG -> {'CANDIDATE' if is_long_candidate else 'REJECTED'}\n"
            log_msg += f"    {'✅' if is_bull_signal else '❌'} [Entry]     SuperTrend Bullish Crossover\n"
            log_msg += f"    {'✅' if not is_sideways else '❌'} [Filter]    ADX > Threshold: {last['adx']:.2f} > {self.params['sidewaysThreshold']}\n"
            
            log_msg += f"  - SHORT -> {'CANDIDATE' if is_short_candidate else 'REJECTED'}\n"
            log_msg += f"    {'✅' if is_bear_signal else '❌'} [Entry]     SuperTrend Bearish Crossover\n"
            log_msg += f"    {'✅' if not is_sideways else '❌'} [Filter]    ADX > Threshold: {last['adx']:.2f} > {self.params['sidewaysThreshold']}"

            logger.info(log_msg)

            direction, detailed_reasoning = None, []
            if is_long_candidate:
                direction = 'buy'
                detailed_reasoning = ["SuperTrend crossover bullish", "Close >= SMA(13)"]
            elif is_short_candidate:
                direction = 'sell'
                detailed_reasoning = ["SuperTrend crossover bearish", "Close <= SMA(13)"]

            if not direction:
                continue
                
            # --- RISK AND TRADE PARAMETER CALCULATION ---
            entry = last['close']
            atr_band = last['atr_for_risk'] * self.params['atrRisk']
            
            if direction == 'buy':
                stop_loss = last['low'] - atr_band
                risk_amount = entry - stop_loss
                if risk_amount <= 0: continue
                take_profits = [entry + risk_amount * rr for rr in [1, 2, 3]]
            else: # sell
                stop_loss = last['high'] + atr_band
                risk_amount = stop_loss - entry
                if risk_amount <= 0: continue
                take_profits = [entry - risk_amount * rr for rr in [1, 2, 3]]
            
            signal_details = {
                "symbol": sym, "direction": direction, "entry_price": entry,
                "stop_loss": stop_loss, "take_profits": take_profits,
                "timeframe": self.primary_timeframe, "strategy_name": self.name,
                "confidence": 0.75, # Base confidence
                "signal_timestamp": str(last.name),
                "detailed_reasoning": detailed_reasoning
            }

            validation_result = rm.validate_and_size_trade(signal_details)

            if validation_result['is_valid']:
                logger.info(f"✅ [{sym}] Valid signal from {self.name}. Dir: {direction}, Entry: {entry:.5f}, SL: {stop_loss:.5f}")
                signals.append(validation_result['final_trade_params'])
            else:
                logger.warning(f"Signal for {sym} from {self.name} rejected by RiskManager: {validation_result['reason']}")

        return signals