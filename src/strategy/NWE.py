import pandas as pd
import numpy as np
import math
from typing import Tuple, List, Optional, Dict, Any
from loguru import logger
from numba import njit

# Core framework imports (assuming these are in your project structure)
from src.trading_bot import SignalGenerator
from src.risk_manager import RiskManager
from config.config import RISK_MANAGER_CONFIG


# --- Numba-Optimized Helper Function (for performance) ---
@njit
def find_pivots_numba(series: np.ndarray, left: int, right: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numba-accelerated function to find pivot points, matching Pine Script's logic.
    A pivot is found at index `i - right` if that point is the extreme in the window.
    """
    highs = np.full(series.shape[0], np.nan)
    lows = np.full(series.shape[0], np.nan)
    
    for i in range(left + right, len(series)):
        window = series[i - left - right : i + 1]
        pivot_index = i - right
        pivot_val = series[pivot_index]
        
        # Pivot High Check
        if pivot_val == np.max(window):
            is_true_pivot = True
            for k in range(i - left - right, pivot_index):
                if series[k] == pivot_val:
                    is_true_pivot = False
                    break
            if is_true_pivot:
                highs[pivot_index] = pivot_val

        # Pivot Low Check
        if pivot_val == np.min(window):
            is_true_pivot = True
            for k in range(i - left - right, pivot_index):
                if series[k] == pivot_val:
                    is_true_pivot = False
                    break
            if is_true_pivot:
                lows[pivot_index] = pivot_val
                
    return highs, lows


# --- Encapsulated Indicator Calculation Logic ---
class _ExhaustionReversalCalculator:
    """
    Internal class to handle all indicator calculations. This keeps the main
    SignalGenerator class clean and focused on strategy logic.
    """
    def __init__(self, data: pd.DataFrame, params: Dict[str, Any]):
        self.data = data.copy()
        self.data.columns = self.data.columns.str.lower()
        self.params = params
        self.results = self.data.copy()

    def _calculate_rsi(self, series: pd.Series, length: int) -> pd.Series:
        numeric_series = pd.to_numeric(series, errors='coerce')
        delta = numeric_series.diff()

        # Use clip to prevent comparison errors with non-numeric types
        gain = delta.clip(lower=0).fillna(0)
        loss = -delta.clip(upper=0).fillna(0)

        avg_gain = gain.ewm(alpha=1/length, min_periods=length).mean()
        avg_loss = loss.ewm(alpha=1/length, min_periods=length).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _find_pivots(self, series: pd.Series, left: int, right: int) -> Tuple[pd.Series, pd.Series]:
        highs, lows = find_pivots_numba(series.to_numpy(dtype=np.float64), left, right)
        return pd.Series(highs, index=series.index), pd.Series(lows, index=series.index)

    def _valuewhen(self, condition: pd.Series, source: pd.Series) -> pd.Series:
        return source.where(condition).ffill()

    def _calculate_divergence(self):
        rsi = self._calculate_rsi(self.results['close'], self.params['rsi_len'])
        lookback = self.params['div_len']
        
        rsi_ph, rsi_pl = self._find_pivots(rsi, lookback, lookback)
        pl_found = rsi_pl.notna()
        ph_found = rsi_ph.notna()

        prev_rsi_pl = self._valuewhen(pl_found, rsi).shift(1)
        prev_price_pl = self._valuewhen(pl_found, self.results['low']).shift(1)
        prev_rsi_ph = self._valuewhen(ph_found, rsi).shift(1)
        prev_price_ph = self._valuewhen(ph_found, self.results['high']).shift(1)

        self.results['bullish_divergence'] = (self.results['low'] < prev_price_pl) & (rsi > prev_rsi_pl) & pl_found
        self.results['bearish_divergence'] = (self.results['high'] > prev_price_ph) & (rsi < prev_rsi_ph) & ph_found

    def _calculate_nadaraya_watson(self):
        h = self.params['nwe_h']
        mult = self.params['nwe_mult']
        src = self.results['close']
        lookback = self.params['nwe_lookback']

        weights = np.array([math.exp(-(i**2) / (h**2 * 2)) for i in range(lookback)])
        src_np = src.to_numpy()
        nwe_values = np.convolve(src_np, weights, mode='full')[:len(src_np)]
        sum_weights_convolve = np.convolve(np.ones_like(src_np), weights, mode='full')[:len(src_np)]
        sum_weights_convolve[sum_weights_convolve == 0] = 1
        
        nwe = pd.Series(nwe_values / sum_weights_convolve, index=src.index)
        mae = (src - nwe).abs().rolling(window=lookback, min_periods=1).mean() * mult
        
        self.results['nwe_upper'] = nwe + mae
        self.results['nwe_lower'] = nwe - mae

    def _calculate_filters(self):
        self.results['ema_filter'] = self.results['close'].ewm(span=self.params['ema_period'], adjust=False).mean()

    def calculate_all_indicators(self) -> pd.DataFrame:
        self._calculate_nadaraya_watson()
        self._calculate_divergence()
        self._calculate_filters()
        return self.results


# --- Timeframe-Specific Parameter Profiles ---
TIMEFRAME_PROFILES = {
    "H4": {"lookback": 250}, # 200 for EMA + extra buffer
    "D1": {"lookback": 250},
    "H1": {"lookback": 250},
}
DEFAULT_PROFILE = {"lookback": 250}


# --- The Signal Generator Class ---
class ExhaustionReversalStrategy(SignalGenerator):
    def __init__(self,
                 primary_timeframe: str = "M5",
                 risk_percent: float = 0.01,
                 min_risk_reward: float = 2.0,
                 ema_period: int = 200,
                 rsi_len: int = 24,
                 div_len: int = 10,
                 div_lookback: int = 5,
                 nwe_h: float = 8.0,
                 nwe_mult: float = 3.0,
                 nwe_lookback: int = 499,
                 **kwargs):
        super().__init__(**kwargs)
        self.name = "ExhaustionReversalStrategy"
        self.description = "A mean-reversion strategy using NWE and RSI Divergence with a trend filter."
        self.version = "1.0.0"
        
        # Strategy parameters
        self.primary_timeframe = primary_timeframe
        self.min_risk_reward = min_risk_reward
        self.risk_percent = RISK_MANAGER_CONFIG.get('max_risk_per_trade', risk_percent)
        
        # Indicator parameters are stored in a dictionary for easy passing
        self.params = {
            'ema_period': ema_period,
            'rsi_len': rsi_len,
            'div_len': div_len,
            'div_lookback': div_lookback,
            'nwe_h': nwe_h,
            'nwe_mult': nwe_mult,
            'nwe_lookback': nwe_lookback,
        }
        
        self.lookback = None
        self._load_timeframe_profile()
        self.processed_bars = {}

    def _load_timeframe_profile(self):
        profile = TIMEFRAME_PROFILES.get(self.primary_timeframe, DEFAULT_PROFILE)
        # Ensure lookback is sufficient for the longest calculation (EMA)
        self.lookback = max(profile.get('lookback', DEFAULT_PROFILE['lookback']), self.params['ema_period'] + 50)
        logger.info(f"🔄 [{self.name}] Profile loaded for {self.primary_timeframe}: lookback={self.lookback}")

    @property
    def lookback_periods(self) -> Dict[str, int]:
        assert self.lookback is not None, "Lookback must be initialized."
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
            if not isinstance(primary_df, pd.DataFrame):
                logger.trace(f"[{sym}] Insufficient data for {self.primary_timeframe}: no data frame. Skipping.")
                continue

            if self.lookback is None or len(primary_df) < self.lookback:
                logger.trace(f"[{sym}] Insufficient data for {self.primary_timeframe} to generate signal. Have {len(primary_df)}, need {self.lookback}.")
                continue
            
            last_timestamp = str(primary_df.index[-1])

            logger.trace(f"[{sym}] Analyzing data for timestamp: {last_timestamp}")

            try:
                calculator = _ExhaustionReversalCalculator(primary_df.copy(), self.params)
                results = calculator.calculate_all_indicators()
            except Exception as e:
                logger.error(f"[{sym}] Error calculating indicators for {self.name}: {e}")
                continue
            
            last = results.iloc[-1]
            
            # --- Strategy Logic & Structured Logging ---
            # LONG conditions
            is_above_ema = last['close'] > last['ema_filter']
            is_touching_lower_nwe = last['low'] <= last['nwe_lower']
            has_recent_bull_div = results['bullish_divergence'].iloc[-self.params['div_lookback']:].any()
            is_long_candidate = is_above_ema and is_touching_lower_nwe and has_recent_bull_div

            # SHORT conditions
            is_below_ema = last['close'] < last['ema_filter']
            is_touching_upper_nwe = last['high'] >= last['nwe_upper']
            has_recent_bear_div = results['bearish_divergence'].iloc[-self.params['div_lookback']:].any()
            is_short_candidate = is_below_ema and is_touching_upper_nwe and has_recent_bear_div

            # Structured Logging
            log_msg = f"[{sym}][{self.name}] Condition Analysis for Bar {last.name}:\n"
            log_msg += f"  - LONG -> {'CANDIDATE' if is_long_candidate else 'REJECTED'}\n"
            log_msg += f"    {'✅' if is_above_ema else '❌'} [Trend]     Close > EMA({self.params['ema_period']}): {last['close']:.2f} > {last['ema_filter']:.2f}\n"
            log_msg += f"    {'✅' if is_touching_lower_nwe else '❌'} [Entry]     Low <= Lower NWE: {last['low']:.2f} <= {last['nwe_lower']:.2f}\n"
            log_msg += f"    {'✅' if has_recent_bull_div else '❌'} [Momentum]  Recent Bullish Divergence\n"
            
            log_msg += f"  - SHORT -> {'CANDIDATE' if is_short_candidate else 'REJECTED'}\n"
            log_msg += f"    {'✅' if is_below_ema else '❌'} [Trend]     Close < EMA({self.params['ema_period']}): {last['close']:.2f} < {last['ema_filter']:.2f}\n"
            log_msg += f"    {'✅' if is_touching_upper_nwe else '❌'} [Entry]     High >= Upper NWE: {last['high']:.2f} >= {last['nwe_upper']:.2f}\n"
            log_msg += f"    {'✅' if has_recent_bear_div else '❌'} [Momentum]  Recent Bearish Divergence"

            logger.info(log_msg)

            direction, detailed_reasoning = None, []
            
            if is_long_candidate:
                direction = 'buy'
                detailed_reasoning = [
                    "Trend Filter: Close > 200 EMA",
                    "Mean Reversion: Price touched Lower NWE Band",
                    "Momentum Confirmation: Recent Bullish RSI Divergence"
                ]

            elif is_short_candidate:
                direction = 'sell'
                detailed_reasoning = [
                    "Trend Filter: Close < 200 EMA",
                    "Mean Reversion: Price touched Upper NWE Band",
                    "Momentum Confirmation: Recent Bearish RSI Divergence"
                ]

            if not direction:
                logger.trace(f"[{sym}] No signal condition met for {self.name} at {last.name}.")
                continue
            
            # --- RISK AND TRADE PARAMETER CALCULATION ---
            entry = last['close'] # Signal confirmed on close, assume entry at close
            take_profits = []
            
            if direction == 'buy':
                stop_loss = last['low'] # Place SL at the low of the signal candle
                risk_amount = entry - stop_loss
                if risk_amount <= 0: continue # Invalid risk
                
                # Generate multiple TP levels
                tp1 = entry + (risk_amount * self.min_risk_reward)
                tp2 = entry + (risk_amount * self.min_risk_reward * 1.5) # Example for a second TP
                take_profits.extend([tp1, tp2])
            else: # sell
                stop_loss = last['high'] # Place SL at the high of the signal candle
                risk_amount = stop_loss - entry
                if risk_amount <= 0: continue # Invalid risk

                tp1 = entry - (risk_amount * self.min_risk_reward)
                tp2 = entry - (risk_amount * self.min_risk_reward * 1.5)
                take_profits.extend([tp1, tp2])

            signal_details = {
                "symbol": sym, "direction": direction, "entry_price": entry,
                "stop_loss": stop_loss, "take_profits": take_profits,
                "timeframe": self.primary_timeframe, "strategy_name": self.name,
                "confidence": 0.85, # Base confidence, can be made dynamic later
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