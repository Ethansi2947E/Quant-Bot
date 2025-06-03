"""
Confluence Price Action Strategy

This strategy identifies the prevailing trend on a higher timeframe, marks key support/resistance levels, waits for pullbacks on a lower timeframe, and then looks for confluence of price-action signals (pin bars, engulfing bars, inside bars, false breakouts), Fibonacci retracements, and moving-average support/resistance. Risk management enforces fixed fractional sizing and minimum R:R.
"""

import pandas as pd
import numpy as np
from loguru import logger
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from src.trading_bot import SignalGenerator
import talib # Added talib import
from config.config import TRADING_CONFIG,get_risk_manager_config
from src.risk_manager import RiskManager
from src.utils.patterns_luxalgo import add_luxalgo_patterns, BULLISH_PATTERNS, BEARISH_PATTERNS, NEUTRAL_PATTERNS, ALL_PATTERNS, filter_patterns_by_bias

# Timeframe-specific profiles for dynamic parameter scaling
TIMEFRAME_PROFILES = {
    "M5": {"pivot_lookback": 140, "pullback_bars": 12, "pattern_bars": 6, "ma_period": 21, "price_tolerance": 0.002, "max_sl_atr_mult": 2.0, "max_sl_pct": 0.01},
    "M15": {"pivot_lookback": 96, "pullback_bars": 18, "pattern_bars": 6, "ma_period": 34, "price_tolerance": 0.002, "max_sl_atr_mult": 2.0, "max_sl_pct": 0.01},
    "H1": {"pivot_lookback": 50, "pullback_bars": 6, "pattern_bars": 2, "ma_period": 55, "price_tolerance": 0.002, "max_sl_atr_mult": 2.5, "max_sl_pct": 0.015},
    "H4": {"pivot_lookback": 30, "pullback_bars": 4, "pattern_bars": 2, "ma_period": 89, "price_tolerance": 0.002, "max_sl_atr_mult": 2.5, "max_sl_pct": 0.015},
    "D1": {"pivot_lookback": 20, "pullback_bars": 3, "pattern_bars": 1, "ma_period": 144, "price_tolerance": 0.002, "max_sl_atr_mult": 3.0, "max_sl_pct": 0.02}
}
DEFAULT_PROFILE = {"pivot_lookback": 50, "pullback_bars": 5, "pattern_bars": 3, "ma_period": 21, "price_tolerance": 0.002, "max_sl_atr_mult": 2.0, "max_sl_pct": 0.01}

# Add fallback max lookback bars by timeframe
MAX_LOOKBACK_BARS = {
    "M5": 200,
    "M15": 150,
    "H1": 100, 
    "H4": 80,
    "D1": 50
}

# NEW: Added tick size constant
TICK_SIZE = 0.0001

class ConfluencePriceActionStrategy(SignalGenerator):
    """
    Confluence-based price action strategy with pullbacks, candlestick confirmations,
    Fibonacci and moving-average confluence, plus strict risk management.
    """

    @staticmethod
    def _detect_inside_bar_vectorized(df: pd.DataFrame) -> pd.Series:
        """Detects inside bars (current high < prev high AND current low > prev low)."""
        if len(df) < 2:
            return pd.Series([False] * len(df), index=df.index)
        # Current high < previous high AND current low > previous low
        inside_bar = (df['high'] < df['high'].shift(1)) & (df['low'] > df['low'].shift(1))
        return inside_bar.fillna(False)

    def __init__(self,
                 primary_timeframe: str = "M15",
                 higher_timeframe: str = "H1",
                 ma_period: int = 21,
                 fib_levels=(0.5, 0.618),
                 use_fibonacci: bool = False,  # Added: Make Fibonacci optional
                 risk_percent: float = 0.01,
                 min_risk_reward: float = 2.0,
                 partial_profit_rr: float = 1.0,
                 partial_profit_pct: float = 0.5,
                 max_bars_till_exit: int = 5,
                 trailing_stop_atr_mult: float = 1.5,
                 # Regime filter kwargs (parameterized for easy tuning)
                 use_adx_filter: bool = True,
                 adx_threshold: float = 15.0,  # Lowered from 18.0 to 15.0
                 use_range_filter: bool = True,
                 range_ratio_threshold: float = 0.4,
                 lookback_period: int = 300,
                 **kwargs):
        """
        Args:
            ...
            use_fibonacci (bool): Whether to use Fibonacci retracement levels for confluence. Default True.
            adx_threshold (float): ADX threshold for trend regime. Default 15.0 (lowered for flexibility).
            range_ratio_threshold (float): Threshold for range_ratio to define ranging. Default 0.4.
        """
        super().__init__(**kwargs)
        self.use_fibonacci = use_fibonacci  # Move this assignment early
        self.name = "ConfluencePriceActionStrategy"
        self.description = (
            "Trend-follow pullbacks at key levels with candlestick confirmation, "
            f"{'Fibonacci + ' if self.use_fibonacci else ''}MA confluence, fixed-fraction risk sizing"
        )
        self.version = "1.0.0"

        # Timeframes
        self.primary_timeframe = primary_timeframe
        self.higher_timeframe = higher_timeframe
        # self.required_timeframes = [higher_timeframe, primary_timeframe]  # Removed, use property

        # Strategy parameters (base values; overridden by timeframe profile)
        self.ma_period = ma_period
        self.fib_levels = fib_levels
        self.risk_percent = risk_percent
        self.min_risk_reward = min_risk_reward
        # Dynamic exit parameters
        self.partial_profit_rr = partial_profit_rr  # Take partial profit at this R:R
        self.partial_profit_pct = partial_profit_pct  # Percentage of position to close at first target
        self.max_bars_till_exit = max_bars_till_exit  # Exit if no target hit within this many bars
        self.trailing_stop_atr_mult = trailing_stop_atr_mult  # ATR multiplier for trailing stop
        # Override price tolerance from global config if set
        data_mgmt = TRADING_CONFIG.get('data_management', {})
        self.price_tolerance = data_mgmt.get('price_tolerance', kwargs.get('price_tolerance', 0.002))
        # Override risk percent from RiskManager config for this timeframe
        rm_conf = get_risk_manager_config()
        self.risk_percent = rm_conf.get('max_risk_per_trade', self.risk_percent)
        # Regime filter config (parameterized for iterative tuning)
        self.use_adx_filter = kwargs.get('use_adx_filter', use_adx_filter)
        self.adx_threshold = kwargs.get('adx_threshold', adx_threshold)
        self.use_range_filter = kwargs.get('use_range_filter', use_range_filter)
        self.range_ratio_threshold = kwargs.get('range_ratio_threshold', range_ratio_threshold)

        self.max_sl_atr_mult = None
        self.max_sl_pct = None

        # Load dynamic profile based on primary timeframe
        self._load_timeframe_profile()

        # State tracking to prevent duplicate/conflicting signals
        self.processed_bars = {}  # {(symbol, timeframe): last_processed_timestamp}
        self.processed_zones = {}  # {(symbol, zone_type, zone_price): last_processed_timestamp}
        self.signal_cooldown = 86400  # 24 hours in seconds

        self.lookback_period = lookback_period

    def _load_timeframe_profile(self):
        """
        Load parameters specific to the selected primary timeframe from TIMEFRAME_PROFILES.
        Overrides: pivot_lookback, pullback_bars, pattern_bars, ma_period, price_tolerance
        """
        profile = TIMEFRAME_PROFILES.get(self.primary_timeframe, DEFAULT_PROFILE)
        # Core dynamic settings
        self.pivot_lookback = profile.get('pivot_lookback', DEFAULT_PROFILE['pivot_lookback'])
        self.pullback_bars = profile.get('pullback_bars', DEFAULT_PROFILE['pullback_bars'])
        self.pattern_bars = profile.get('pattern_bars', DEFAULT_PROFILE['pattern_bars'])
        # Override MA period and tolerance if provided
        self.ma_period = profile.get('ma_period', self.ma_period)
        self.price_tolerance = profile.get('price_tolerance', self.price_tolerance)
        
        # Override pattern bars from global pattern detector config

        self.max_sl_atr_mult = profile.get('max_sl_atr_mult', DEFAULT_PROFILE['max_sl_atr_mult'])
        self.max_sl_pct = profile.get('max_sl_pct', DEFAULT_PROFILE['max_sl_pct'])

        logger.info(
            f"🔄 Timeframe profile loaded for {self.primary_timeframe}: "
            f"pivot_lookback={self.pivot_lookback}, pullback_bars={self.pullback_bars}, "
            f"pattern_bars={self.pattern_bars}, ma_period={self.ma_period}, "
            f"price_tolerance={self.price_tolerance}, max_sl_atr_mult={self.max_sl_atr_mult}, max_sl_pct={self.max_sl_pct}"
        )

    async def initialize(self) -> bool:
        logger.info(f"🔌 Initializing {self.name}")
        return True

    async def generate_signals(
        self,
        market_data: Optional[dict] = None,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        **kwargs
    ) -> list:
        """
        Generate long/short signals based on confluence price action.
        1. Determine up/down trend on higher timeframe
        2. Identify support/resistance levels on higher timeframe
        3. On lower timeframe, wait for pullback to those levels
        4. Spot candlestick patterns (pin bar, engulfing, inside bar, false break)
        5. Check confluence: Fibonacci retrace, MA support/resistance, volume
        6. Place orders with stop-loss and take-profit (min RR)
        """
        if market_data is None:
            market_data = {}
        if symbol is None:
            symbol = ""
        if timeframe is None:
            timeframe = ""

        signals = []
        if not market_data:
            logger.warning(f"No market_data provided to {self.name}")
            return signals
        # initialize RiskManager and account balance
        rm = RiskManager.get_instance()
        balance = kwargs.get("balance", rm.daily_stats.get('starting_balance', 0) or 10000)

        # For detailed debugging
        debug_visualize = kwargs.get("debug_visualize", True)

        current_time = datetime.now().timestamp()
        for sym, frames in market_data.items():
            higher = frames.get(self.higher_timeframe)
            primary = frames.get(self.primary_timeframe)
            if not isinstance(higher, pd.DataFrame) or not isinstance(primary, pd.DataFrame):
                logger.debug(f"Missing data for {sym} - higher: {type(higher)}, primary: {type(primary)}")
                continue

            # More detailed information about the data received
            logger.debug(
                f"Analyzing {sym} - Higher TF: {self.higher_timeframe} ({len(higher)} bars), Primary TF: {self.primary_timeframe} ({len(primary)} bars)"
            )

            primary = add_luxalgo_patterns(primary.copy())  # Use .copy()

            bar_key = (sym, self.primary_timeframe)
            try:
                last_timestamp = pd.to_datetime(primary.index[-1])
                last_timestamp_str = str(last_timestamp)
            except Exception:
                logger.warning(f"Could not extract timestamp from dataframe for {sym}")
                last_timestamp_str = str(current_time)
            if (
                bar_key in self.processed_bars
                and self.processed_bars[bar_key] == last_timestamp_str
            ):
                logger.debug(
                    f"Already processed latest bar for {sym}/{self.primary_timeframe} at {last_timestamp_str}"
                )
                continue
            self.processed_bars[bar_key] = last_timestamp_str
            logger.debug(
                f"Processing new bar for {sym}/{self.primary_timeframe} at {last_timestamp_str}"
            )

            # --- Step 1: Cache ATR and ADX series for this symbol's primary timeframe ---
            atr_series = None
            adx_series = None
            if isinstance(primary, pd.DataFrame) and len(primary) >= 14:
                try:
                    if (
                        primary is not None
                        and not primary.empty
                        and all(col in primary.columns for col in ['high', 'low', 'close'])
                    ):
                        high_np = np.asarray(primary['high'].values, dtype=np.float64)
                        low_np = np.asarray(primary['low'].values, dtype=np.float64)
                        close_np = np.asarray(primary['close'].values, dtype=np.float64)
                        atr_series = talib.ATR(high_np, low_np, close_np, timeperiod=14)
                    else:
                        logger.warning(
                            f"ATR calculation failed for {sym}: DataFrame missing required columns or is empty"
                        )
                except Exception as e:
                    logger.warning(f"ATR calculation failed for {sym}: {e}")
                try:
                    if (
                        primary is not None
                        and not primary.empty
                        and all(col in primary.columns for col in ['high', 'low', 'close'])
                    ):
                        high_np = np.asarray(primary['high'].values, dtype=np.float64)
                        low_np = np.asarray(primary['low'].values, dtype=np.float64)
                        close_np = np.asarray(primary['close'].values, dtype=np.float64)
                        adx_series = talib.ADX(high_np, low_np, close_np, timeperiod=14)
                except Exception as e:
                    logger.warning(f"ADX calculation failed for {sym}: {e}")

            # 1. Trend on higher timeframe
            trend = self._determine_trend(higher)
            logger.debug(f"{sym}: Higher timeframe trend is {trend}")
            if trend == 'neutral':
                logger.debug(f"{sym}: Skipping due to neutral trend")
                continue

            # 2. Key levels on higher timeframe
            supports, resistances = self._find_key_levels(higher)
            # Filter out weak levels (strength < 2)
            supports = [lvl for lvl in supports if lvl['strength'] >= 1]
            resistances = [lvl for lvl in resistances if lvl['strength'] >= 1]
            logger.debug(
                f"{sym}: Found {len(supports)} support levels and {len(resistances)} resistance levels (strength >= 1)"
            )
            levels_to_check = supports if trend == 'bullish' else resistances
            level_type_str = "support" if trend == 'bullish' else "resistance"

            # 3. For each level, check pullback on primary TF
            level_info_strs = [
                f"Level {i+1}: {lvl['level']:.5f} (strength: {lvl['strength']})"
                for i, lvl in enumerate(levels_to_check)
            ]
            logger.debug(
                f"{sym}: Checking {len(levels_to_check)} {level_type_str} levels: {', '.join(level_info_strs)}"
            )

            symbol_signals = []
            for level_data in levels_to_check:
                current_level_price = level_data['level']
                zone_type_str = 'support' if trend == 'bullish' else 'resistance'
                zone_key = (sym, zone_type_str, round(current_level_price, 5))
                if zone_key in self.processed_zones:
                    last_used_time = self.processed_zones[zone_key]
                    # Ensure both are datetime objects for correct subtraction
                    if isinstance(current_time, pd.Timestamp):
                        current_time_dt = current_time.to_pydatetime()
                    elif not isinstance(current_time, datetime):
                        # Attempt to convert if not already datetime, log error if fails
                        try:
                            current_time_dt = pd.to_datetime(current_time).to_pydatetime()
                        except Exception as e_ct:
                            logger.error(f"Error converting current_time ({current_time}) to datetime: {e_ct}")
                            continue # Skip this level if current_time is problematic
                    else:
                        current_time_dt = current_time

                    if isinstance(last_used_time, pd.Timestamp):
                        last_used_time_dt = last_used_time.to_pydatetime()
                    elif not isinstance(last_used_time, datetime):
                        try:
                            last_used_time_dt = pd.to_datetime(last_used_time).to_pydatetime()
                        except Exception as e_lut:
                            logger.error(f"Error converting last_used_time ({last_used_time}) for zone {zone_key} to datetime: {e_lut}")
                            self.processed_zones.pop(zone_key, None) # Remove potentially corrupt entry
                            continue # Skip this level
                    else:
                        last_used_time_dt = last_used_time
                    
                    if current_time_dt and last_used_time_dt:
                        time_since_use = current_time_dt - last_used_time_dt
                        # Assuming self.signal_cooldown is in seconds
                        if time_since_use.total_seconds() < self.signal_cooldown:
                            logger.debug(
                                f"Skipping {zone_type_str} zone {current_level_price:.5f} - on cooldown ({time_since_use.total_seconds():.0f}s < {self.signal_cooldown}s)"
                            )
                            continue
                    else:
                        logger.warning(f"Could not compare times for zone {zone_key} due to invalid datetime objects.")
                        continue
                # --- Rejection-based (reversal) logic (existing) ---
                logger.debug(
                    f"Checking pullback for {sym} at {level_type_str} level {current_level_price:.5f} (trend: {trend})"
                )
                if self._is_pullback(primary, current_level_price, trend):
                    logger.debug(
                        f"Found pullback to {level_type_str} level {current_level_price:.5f} for {sym}"
                    )
                    idx = len(primary) - 1
                    if idx < 0:
                        continue  # Should not happen if len(primary) is checked
                    candle = primary.iloc[idx]

                    # --- Updated Pattern Detection & Scoring ---
                    detected_pattern_name = None
                    pattern_score = 0.0
                    patterns_to_evaluate = (
                        BULLISH_PATTERNS if trend == 'bullish' else BEARISH_PATTERNS
                    )
                    patterns_to_evaluate += NEUTRAL_PATTERNS  # Always consider neutral for confluence

                    # Check for specific Breakout/Rejection conditions first
                    is_breakout_acceptance = False
                    is_rejection_reversal = False

                    volume_score_for_pattern, _ = self._analyze_volume_quality(
                        primary, idx, trend
                    )  # Analyze volume for the current candle
                    volume_confirmed_for_pattern = volume_score_for_pattern >= 0.3

                    if trend == 'bullish':
                        if (
                            candle['open'] < current_level_price
                            and candle['close'] > current_level_price
                            and (candle['high'] - candle['close'])
                            < (current_level_price * self.price_tolerance)
                            and volume_confirmed_for_pattern
                        ):
                            is_breakout_acceptance = True
                            detected_pattern_name = 'Breakout Acceptance'
                            pattern_score = 1.0  # High score for confirmed breakout
                        elif (
                            candle['open'] < current_level_price
                            and candle['high'] > current_level_price
                            and candle['close'] < current_level_price
                        ):
                            is_rejection_reversal = True
                            detected_pattern_name = 'Rejection Reversal (Bullish)'  # More specific
                            pattern_score = 0.9  # High score for clear rejection
                    else:  # Bearish trend
                        if (
                            candle['open'] > current_level_price
                            and candle['close'] < current_level_price
                            and (candle['close'] - candle['low'])
                            < (current_level_price * self.price_tolerance)
                            and volume_confirmed_for_pattern
                        ):
                            is_breakout_acceptance = True
                            detected_pattern_name = 'Breakout Acceptance'
                            pattern_score = 1.0
                        elif (
                            candle['open'] > current_level_price
                            and candle['low'] < current_level_price
                            and candle['close'] > current_level_price
                        ):
                            is_rejection_reversal = True
                            detected_pattern_name = 'Rejection Reversal (Bearish)'  # More specific
                            pattern_score = 0.9

                    # If not a breakout/rejection, check standard LuxAlgo patterns
                    if not detected_pattern_name:
                        for p_col in patterns_to_evaluate:
                            if p_col in primary.columns:
                                pattern_series = primary[p_col]
                                if pattern_series.iloc[idx]:
                                    detected_pattern_name = f"{p_col.replace('_', ' ').title()} (LuxAlgo)"
                                    # Assign scores based on pattern type from your rubric
                                    if p_col in ['hammer', 'shooting_star', 'pin_bar']:
                                        pattern_score = 1.0
                                    elif p_col in ['bullish_engulfing', 'bearish_engulfing']:
                                        pattern_score = 0.95
                                    elif p_col in ['inverted_hammer', 'hanging_man']:
                                        pattern_score = 0.9
                                    elif p_col in ['morning_star', 'evening_star']:
                                        pattern_score = 0.9
                                    elif p_col in ['bullish_harami', 'bearish_harami']:
                                        pattern_score = 0.8
                                    elif p_col in ['white_marubozu', 'black_marubozu']:
                                        pattern_score = 0.7
                                    elif p_col in ['inside_bar']:
                                        pattern_score = 0.7  # Neutral, but can be part of a setup
                                    else:
                                        pattern_score = 0.6  # Default for other/new patterns
                                    break  # Found a pattern

                    if not detected_pattern_name:
                        logger.debug(
                            f"No valid pattern detected at idx={idx} for {sym} at level {current_level_price:.5f}"
                        )
                        continue
                    logger.debug(
                        f"{sym}: Found {detected_pattern_name} pattern at {candle.name} with score {pattern_score:.2f}"
                    )

                    # Confluence Checks (Fibonacci, MA)
                    fib_ok = self._check_fibonacci(primary, current_level_price, self.use_fibonacci)
                    ma_ok = self._check_ma(primary, current_level_price)
                    fib_details = {}  # Populate if fib_ok
                    ma_details = {}  # Populate if ma_ok
                    # ... (Populate fib_details and ma_details as before) ...

                    # Confluence Scoring (as per your existing logic, adapt as needed)
                    htf_trend_score = 1.0  # Already filtered by trend
                    htf_sr_score = 1.0 if level_data['strength'] >= 1 else 0.0
                    # pattern_score is already set above
                    fib_score_bonus = 0.3 if fib_ok else 0.0
                    ma_score_bonus = 0.3 if ma_ok else 0.0
                    volume_score_confluence = min(
                        max(volume_score_for_pattern, 0.0), 0.2
                    )  # Use earlier volume score
                    level_strength_bonus = min(0.2, level_data['strength'] / 5.0)

                    confluence_total_score = (
                        htf_trend_score
                        + htf_sr_score
                        + pattern_score
                        + fib_score_bonus
                        + ma_score_bonus
                        + volume_score_confluence
                        + level_strength_bonus
                    )
                    min_score_threshold = 2.2

                    # Max possible score: 1(trend) + 1(sr) + 1(pattern) + 0.3(fib) + 0.3(ma) + 0.2(vol) + 0.2(strength) = 4.0
                    normalization_divisor = 4.0 # Changed from 3.0

                    if confluence_total_score < min_score_threshold:
                        logger.debug(
                            f"[ConfluenceScoring] Signal for {sym} at {current_level_price:.5f} rejected: total score {confluence_total_score:.2f} < {min_score_threshold}"
                        )
                        continue
                    
                    # --- Inserted Missing Block: Signal Assembly (entry, SL, TP, etc.) ---
                    entry = candle['close']
                    direction_str = "buy" if trend == 'bullish' else "sell"

                    # Robust SL tolerance: use max of ATR and pattern candle's range
                    atr_val_sl_tp = (
                        atr_series[-1]
                        if atr_series is not None and len(atr_series) > 0 and pd.notna(atr_series[-1])
                        else current_level_price * 0.001 # Fallback if ATR is not available
                    )
                    candle_range_sl_tp = candle['high'] - candle['low']
                    tol_val_sl_tp = max(
                        current_level_price * self.price_tolerance,
                        atr_val_sl_tp if pd.notna(atr_val_sl_tp) and atr_val_sl_tp > 0 else current_level_price * 0.001, # Ensure tol_val is sensible
                        candle_range_sl_tp
                    )

                    if direction_str == 'buy':
                        stop = candle['low'] - tol_val_sl_tp
                        # Ensure stop is not ridiculously far or positive if candle['low'] is very small
                        stop = min(stop, entry - (TICK_SIZE * 2)) # Must be below entry
                        reward_calc = entry - stop 
                        tp = entry + reward_calc * self.min_risk_reward
                    else:  # sell
                        stop = candle['high'] + tol_val_sl_tp
                        # Ensure stop is not ridiculously far or negative if candle['high'] is very high
                        stop = max(stop, entry + (TICK_SIZE * 2)) # Must be above entry
                        reward_calc = stop - entry
                        tp = entry - reward_calc * self.min_risk_reward

                    # CAP SL/TP DISTANCE (as before, from the deleted block)
                    max_sl_dist_val = None
                    if pd.notna(atr_val_sl_tp) and atr_val_sl_tp > 0 and self.max_sl_atr_mult is not None:
                        max_sl_dist_val = max(atr_val_sl_tp * self.max_sl_atr_mult, 2 * TICK_SIZE)
                    
                    if self.max_sl_pct is not None:
                        max_sl_dist_pct_val = entry * self.max_sl_pct
                        if max_sl_dist_val is not None and pd.notna(max_sl_dist_val):
                            max_sl_dist_val = min(max_sl_dist_val, max_sl_dist_pct_val)
                        else:
                            max_sl_dist_val = max_sl_dist_pct_val
                    
                    if max_sl_dist_val is None or not np.isfinite(max_sl_dist_val) or max_sl_dist_val <= (TICK_SIZE * 2):
                        max_sl_dist_val = max(entry * 0.002, 2 * TICK_SIZE) # Default sensible cap

                    if abs(entry - stop) > max_sl_dist_val:
                        if direction_str == 'buy':
                            stop = entry - max_sl_dist_val
                        else:
                            stop = entry + max_sl_dist_val
                        reward_calc = abs(entry - stop)
                        tp = (
                            entry + reward_calc * self.min_risk_reward
                            if direction_str == 'buy'
                            else entry - reward_calc * self.min_risk_reward
                        )
                    
                    # Final check for valid SL/TP relative to entry
                    if direction_str == 'buy' and (stop >= entry or tp <= entry):
                        logger.warning(f"{sym} BUY signal: Invalid SL({stop:.5f})/TP({tp:.5f}) relative to Entry({entry:.5f}). Adjusting SL/TP or skipping.")
                        # Attempt a minimal adjustment or skip
                        if stop >= entry: stop = entry - (2 * TICK_SIZE)
                        if tp <= entry: tp = entry + (abs(entry-stop) * self.min_risk_reward if abs(entry-stop) > 0 else 4 * TICK_SIZE)
                        if stop >= entry or tp <= entry: # If still invalid, skip
                            logger.error(f"{sym} BUY signal: Could not correct SL/TP. Skipping signal.")
                            continue
                    elif direction_str == 'sell' and (stop <= entry or tp >= entry):
                        logger.warning(f"{sym} SELL signal: Invalid SL({stop:.5f})/TP({tp:.5f}) relative to Entry({entry:.5f}). Adjusting SL/TP or skipping.")
                        if stop <= entry: stop = entry + (2 * TICK_SIZE)
                        if tp >= entry: tp = entry - (abs(stop-entry) * self.min_risk_reward if abs(stop-entry) > 0 else 4 * TICK_SIZE)
                        if stop <= entry or tp >= entry: # If still invalid, skip
                            logger.error(f"{sym} SELL signal: Could not correct SL/TP. Skipping signal.")
                            continue

                    risk_pips = abs(entry - stop)
                    reward_pips = abs(tp - entry)
                    min_pip_reward_val = 5 * TICK_SIZE
                    if risk_pips < TICK_SIZE: # Prevent zero or too small risk
                        logger.debug(f"{sym}: Skipping signal due to extremely small risk_pips {risk_pips:.5f}")
                        continue
                    if reward_pips < min_pip_reward_val:
                        logger.debug(
                            f"{sym}: Skipping signal due to reward_pips {reward_pips:.5f} < minimum {min_pip_reward_val:.5f}"
                        )
                        continue
                    # --- End of Inserted Missing Block ---

                    signal_quality_norm = min(
                        max(confluence_total_score / normalization_divisor, 0.0), 1.0 
                    )
                    concise_analysis = (
                        f"📝 Analysis ({sym} - {self.primary_timeframe}): {direction_str.capitalize()} signal based on {detected_pattern_name} at {level_type_str} {current_level_price:.5f}."
                    )
                    # ... more details for rationale ...
                    reasoning_list = [concise_analysis]

                    signal_to_add = {
                        "symbol": sym,
                        "direction": direction_str,
                        "entry_price": entry,
                        "stop_loss": stop,
                        "take_profit": tp,
                        "timeframe": self.primary_timeframe,
                        "confidence": signal_quality_norm,
                        "strategy_name": self.name,
                        "pattern": detected_pattern_name,
                        "pattern_details": {},
                        "fib_details": fib_details if fib_ok else {},
                        "ma_details": ma_details if ma_ok else {},
                        "volume_details": {},
                        "risk_pips": risk_pips,
                        "reward_pips": reward_pips,
                        "risk_reward_ratio": reward_pips / risk_pips if risk_pips > 0 else 0,
                        "signal_quality": signal_quality_norm,
                        "technical_metrics": {},
                        "pattern_score": pattern_score,
                        "confluence_score": confluence_total_score / normalization_divisor, # Changed from 3.0
                        "volume_score": volume_score_confluence,
                        "recency_score": 0.0,
                        "level_strength": level_data['strength'],
                        "level_strength_score": level_strength_bonus,
                        "description": concise_analysis,
                        "detailed_reasoning": reasoning_list,
                        "signal_bar_index": idx,
                        "signal_timestamp": str(candle.name),
                    }
                    result_validation = rm.validate_and_size_trade(
                        signal_to_add
                    )  # Use the correctly named dict
                    if not result_validation['is_valid']:
                        logger.info(
                            f"Signal for {sym} rejected by RiskManager: {result_validation['reason']}"
                        )
                        continue
                    adjusted_signal_final = result_validation['final_trade_params']
                    for k_sig, v_sig in signal_to_add.items():  # Corrected loop variable
                        if k_sig not in adjusted_signal_final:
                            adjusted_signal_final[k_sig] = v_sig
                    symbol_signals.append(adjusted_signal_final)
                    self.processed_zones[zone_key] = current_time
                    break  # Process one signal per zone
            if len(symbol_signals) > 1:
                symbol_signals = self._prioritize_signals(symbol_signals)
            signals.extend(symbol_signals)
        
        # Ensure current_time is a datetime object before timedelta operations
        if isinstance(current_time, (float, int)):
            try:
                current_time_dt = datetime.fromtimestamp(current_time)
            except ValueError: # Handle cases like milliseconds timestamp
                current_time_dt = datetime.fromtimestamp(current_time / 1000)
        elif isinstance(current_time, pd.Timestamp):
            current_time_dt = current_time.to_pydatetime()
        elif isinstance(current_time, datetime):
            current_time_dt = current_time
        else:
            logger.error(f"Cannot determine cleanup_time: current_time is of unexpected type {type(current_time)}")
            current_time_dt = datetime.now() # Fallback, though ideally this path isn't hit

        cleanup_delta = timedelta(seconds=(self.signal_cooldown * 2))
        cleanup_time = current_time_dt - cleanup_delta
        
        old_keys = [k for k, v in self.processed_zones.items() if isinstance(v, (datetime, pd.Timestamp)) and pd.to_datetime(v) < cleanup_time]
        for k_del in old_keys:  # Corrected loop variable
            del self.processed_zones[k_del]
        if old_keys:
            logger.debug(f"Cleaned up {len(old_keys)} old zone records")
        return signals

    def _prioritize_signals(self, signals: list) -> list:
        """
        Prioritize signals when there are conflicts for the same symbol/bar.
        Returns only the highest-priority signal for each symbol/bar.
        """
        if not signals:
            return []
        signals_by_symbol = {}
        for signal in signals:
            symbol = signal.get('symbol')
            bar_index = signal.get('signal_bar_index', None)
            key = (symbol, bar_index)
            if key not in signals_by_symbol:
                signals_by_symbol[key] = []
            signals_by_symbol[key].append(signal)
        prioritized = []
        for key, sigs in signals_by_symbol.items():
            if len(sigs) == 1:
                prioritized.append(sigs[0])
            else:
                # Score by risk-reward, pattern, volume, etc.
                for s in sigs:
                    rr = s.get('risk_reward_ratio', 0)
                    pattern = s.get('pattern', '')
                    pattern_score = 5 if 'Engulfing' in pattern else 4 if 'Pin Bar' in pattern else 3
                    volume_score = s.get('volume_score', 0)
                    s['priority_score'] = rr * 10 + pattern_score * 10 + volume_score * 10
                best = sorted(sigs, key=lambda x: x['priority_score'], reverse=True)[0]
                prioritized.append(best)
        return prioritized

    # -- Trend and level detection --
    def _determine_trend(self, df: pd.DataFrame) -> str:
        """Return 'bullish', 'bearish' or 'neutral' based on MA and price structure (higher highs/lows).
        Loosened: Allow signal if either MA trend or price structure trend is clear.
        Assign trend confidence: both agree=1.0, one clear=0.7, both neutral=0.0 (skip).
        Logs the decision process for debugging.
        """
        if df is None or 'close' not in df.columns or len(df) < self.ma_period + 3:
            logger.debug("[Trend] Insufficient data for trend determination.")
            return 'neutral'
        close = np.asarray(df['close'].values, dtype=np.float64)
        ma = talib.SMA(close, timeperiod=self.ma_period)[-1]
        last = close[-1]
        ma_trend = 'bullish' if last > ma else 'bearish' if last < ma else 'neutral'
        highs = np.asarray(df['high'].values, dtype=np.float64)
        lows = np.asarray(df['low'].values, dtype=np.float64)
        # Use TA-Lib for swing high/low detection
        w = 2
        swing_highs = [highs[i] for i in range(w, len(highs) - w)
                       if highs[i] == talib.MAX(highs, timeperiod=2 * w + 1)[i] and not np.isnan(talib.MAX(highs, timeperiod=2 * w + 1)[i])]
        swing_lows = [lows[i] for i in range(w, len(lows) - w)
                      if lows[i] == talib.MIN(lows, timeperiod=2 * w + 1)[i] and not np.isnan(talib.MIN(lows, timeperiod=2 * w + 1)[i])]
        last_highs = swing_highs[-3:]
        last_lows = swing_lows[-3:]
        price_trend = 'neutral'
        if len(last_highs) == 3 and len(last_lows) == 3:
            if last_highs[2] > last_highs[1] > last_highs[0] and last_lows[2] > last_lows[1] > last_lows[0]:
                price_trend = 'bullish'
            elif last_highs[2] < last_highs[1] < last_highs[0] and last_lows[2] < last_lows[1] < last_lows[0]:
                price_trend = 'bearish'
        logger.debug(f"[Trend] MA trend: {ma_trend}, Price structure trend: {price_trend}")
        if ma_trend == price_trend and ma_trend != 'neutral':
            logger.info(f"[Trend] Confirmed {ma_trend} trend (MA + price structure)")
            return ma_trend
        elif ma_trend != 'neutral' or price_trend != 'neutral':
            logger.info(f"[Trend] Loosened: Accepting {ma_trend if ma_trend != 'neutral' else price_trend} trend (one clear)")
            return ma_trend if ma_trend != 'neutral' else price_trend
        logger.info("[Trend] No clear trend (both neutral)")
        return 'neutral'

    def _find_key_levels(self, df: pd.DataFrame) -> tuple:
        """Return (supports, resistances) as lists of dicts with 'level' and 'strength' (touch count)"""
        lows, highs = [], []
        # Limit to profile pivot_lookback bars
        subset = df.copy()
        if len(df) > self.pivot_lookback:
            subset = df.iloc[-self.pivot_lookback:]
        highs_arr = np.asarray(subset['high'].values, dtype=np.float64)
        lows_arr = np.asarray(subset['low'].values, dtype=np.float64)
        w = 2
        for i in range(w, len(subset) - w):
            if lows_arr[i] == talib.MIN(lows_arr, timeperiod=2 * w + 1)[i] and not np.isnan(talib.MIN(lows_arr, timeperiod=2 * w + 1)[i]):
                lows.append(lows_arr[i])
            if highs_arr[i] == talib.MAX(highs_arr, timeperiod=2 * w + 1)[i] and not np.isnan(talib.MAX(highs_arr, timeperiod=2 * w + 1)[i]):
                highs.append(highs_arr[i])
        logger.debug(f"[KeyLevels] Raw pivot lows: {len(lows)}, pivot highs: {len(highs)} (before clustering)")
        # Clustering tolerance is the maximum of 5 ticks and the minimum of (mean price * price_tolerance, 15 ticks),
        # but is further widened to at least ATR*0.1 if ATR is available. This ensures clusters are not too loose, but still adapt to volatility.
        clustering_tol = max(5 * TICK_SIZE, min(subset['close'].mean() * self.price_tolerance, 15 * TICK_SIZE))
        atr_val = None
        if len(subset) >= 14:
            high = np.asarray(subset['high'].values, dtype=np.float64)
            low = np.asarray(subset['low'].values, dtype=np.float64)
            close = np.asarray(subset['close'].values, dtype=np.float64)
            atr_series = talib.ATR(high, low, close, timeperiod=14)
            if len(atr_series) > 0:
                atr_val = float(atr_series[-1])
        if atr_val:
            clustering_tol = max(clustering_tol, atr_val * 0.1)
        support_levels = self._cluster_levels_with_strength(sorted(lows), clustering_tol, subset, is_support=True)
        resistance_levels = self._cluster_levels_with_strength(sorted(highs), clustering_tol, subset, is_support=False)
        logger.debug(f"[KeyLevels] Support clusters: {len(support_levels)}, Resistance clusters: {len(resistance_levels)} (before filtering by touches)")
        # --- Require at least 2 touches for a level to be considered ---
        last_50 = df.iloc[-min(50, len(df)):]
        min_touches = 2
        support_levels = [lvl for lvl in support_levels if self._count_level_touches(last_50, lvl['level'], clustering_tol, is_support=True) >= min_touches]
        resistance_levels = [lvl for lvl in resistance_levels if self._count_level_touches(last_50, lvl['level'], clustering_tol, is_support=False) >= min_touches]
        logger.debug(f"[KeyLevels] Support levels after filtering: {len(support_levels)}, Resistance levels after filtering: {len(resistance_levels)}")
        return support_levels, resistance_levels

    def _cluster_levels_with_strength(self, levels: list, tol: float, df: pd.DataFrame, is_support: bool) -> list:
        """Cluster nearby price levels within tolerance and count touches for each cluster."""
        if not levels:
            return []
        clusters, current = [], [levels[0]]
        for lv in levels[1:]:
            if abs(lv - current[-1]) <= tol:
                current.append(lv)
            else:
                clusters.append(current)
                current = [lv]
        clusters.append(current)
        # For each cluster, compute average and count touches
        clustered = []
        for cluster in clusters:
            avg_level = sum(cluster) / len(cluster)
            strength = self._count_level_touches(df, avg_level, tol, is_support)
            clustered.append({"level": avg_level, "strength": strength})
        return clustered

    def _count_level_touches(self, df: pd.DataFrame, level: float, tol: float, is_support: bool) -> int:
        """Count how many times price has touched a level within tolerance."""
        count = 0
        if is_support:
            for i in range(len(df)):
                if abs(df['low'].iat[i] - level) <= tol:
                    count += 1
        else:
            for i in range(len(df)):
                if abs(df['high'].iat[i] - level) <= tol:
                    count += 1
        return count

    # -- Pullback detection --
    def _is_pullback(self, df: pd.DataFrame, level: float, direction: str) -> bool:
        """Check for a pullback to the level in the direction of the trend.
        Loosened: Allow touch (penetration=0) as valid, bars_since_excursion up to 3, min_penetration set to 0.
        """
        if df is None or len(df) < self.pullback_bars:
            logger.debug("[Pullback] Not enough bars for pullback check.")
            return False
        atr_val = None
        if len(df) >= 14:
            high = np.asarray(df['high'].values, dtype=np.float64)
            low = np.asarray(df['low'].values, dtype=np.float64)
            close = np.asarray(df['close'].values, dtype=np.float64)
            atr_series = talib.ATR(high, low, close, timeperiod=14)
            if len(atr_series) > 0:
                atr_val = float(atr_series[-1])
        pullback_tolerance = max(level * self.price_tolerance, (atr_val * 0.5) if atr_val else 0)
        # Loosened min_penetration
        min_penetration = 0.0  # Loosened: allow touch
        min_bars_since_excursion = 1  # was 2
        recent = df.iloc[-self.pullback_bars:]
        last_candle = recent.iloc[-1]
        closes = recent['close']
        penetration = 0.0
        bars_since_excursion = None
        if direction == 'bullish':
            prior_beyond_idx = (closes.iloc[:-1] > level)[::-1].idxmax() if (closes.iloc[:-1] > level).any() else None
            if prior_beyond_idx is not None:
                bars_since_excursion = len(closes) - 1 - list(closes.index).index(prior_beyond_idx)
            retest = abs(last_candle['low'] - level) <= pullback_tolerance
            penetration = max(0.0, level - last_candle['low'])
            logger.debug(f"[Pullback] Bullish: penetration={penetration}, min_penetration={min_penetration}, bars_since_excursion={bars_since_excursion}")
            if retest and penetration >= min_penetration and (bars_since_excursion is None or bars_since_excursion >= min_bars_since_excursion):
                return True
        else:
            prior_beyond_idx = (closes.iloc[:-1] < level)[::-1].idxmax() if (closes.iloc[:-1] < level).any() else None
            if prior_beyond_idx is not None:
                bars_since_excursion = len(closes) - 1 - list(closes.index).index(prior_beyond_idx)
            retest = abs(last_candle['high'] - level) <= pullback_tolerance
            penetration = max(0.0, last_candle['high'] - level)
            logger.debug(f"[Pullback] Bearish: penetration={penetration}, min_penetration={min_penetration}, bars_since_excursion={bars_since_excursion}")
            if retest and penetration >= min_penetration and (bars_since_excursion is None or bars_since_excursion >= min_bars_since_excursion):
                return True
        logger.debug(f"[Pullback] No valid pullback detected (penetration={penetration}, bars_since_excursion={bars_since_excursion})")
        return False

    def _is_false_breakout(self, candles: pd.DataFrame, idx: int, level: float, direction: str) -> bool:
        """Detect a quick reversal after a breakout around `level` with wick and volume analysis.
        Relaxed: volume > 1.0x avg, wick > 1.2x body, allow touch-breaks (not just full clean breakouts).
        This method's logic is highly specific to this strategy and represents a sequence.
        It will remain largely as is.
        """
        if idx <= 0 or idx >= len(candles):
            return False
        prev = candles.iloc[idx - 1]
        curr = candles.iloc[idx]
        # tol_val = level * self.price_tolerance # Original tol_val not used here, price_tolerance is class member
        vol_col = 'volume' if 'volume' in candles.columns else 'tick_volume'
        
        avg_vol = np.nan # Default to NaN
        if vol_col in candles.columns and len(candles.iloc[max(0, idx-20):idx][vol_col]) > 0 : # Ensure slice isn't empty
             avg_vol = candles.iloc[max(0, idx-20):idx][vol_col].mean() # Look back on candles up to prev bar

        # Require volume at least 1.0x average (was 1.2x)
        # Handle avg_vol being NaN if not enough data
        current_vol = curr.get(vol_col, 0)
        vol_ok = current_vol > 1.0 * avg_vol if pd.notna(avg_vol) and avg_vol > 0 else False
        
        if not vol_ok and pd.notna(avg_vol) : # Log if vol not ok but avg_vol was calculable
            logger.debug(f"[_is_false_breakout] Vol check fail: current={current_vol}, avg={avg_vol:.2f}")
        elif not pd.notna(avg_vol):
             logger.debug(f"[_is_false_breakout] Vol check cannot be performed: avg_vol is NaN")


        if direction == 'bullish': # Looking for bullish reversal after a bearish break of support `level`
            body = abs(curr['close'] - curr['open'])
            if body == 0: return False # Avoid division by zero if body is zero
            wick = curr['close'] - curr['low'] 
            wick_ok = wick > 1.2 * body if body > 0 else wick > 0 # was 1.5x. Handle zero body.

            # Breakout: prev broke below level, curr came back above and closed above.
            breakout_condition = prev['low'] < level and curr['close'] > level 
            # Original code had: prev['low'] < level and curr['high'] > level. Switched to curr['close'] for confirmation.

            if breakout_condition and wick_ok and vol_ok:
                logger.debug(f"[_is_false_breakout] Bullish detected: prev_low={prev['low']:.5f}, level={level:.5f}, curr_close={curr['close']:.5f}, wick={wick:.5f}, body={body:.5f}, vol_ok={vol_ok}")
                return True
            else:
                logger.debug(f"[_is_false_breakout] Bullish rejected: breakout={breakout_condition}, wick_ok={wick_ok} (wick={wick:.5f}, body={body:.5f}), vol_ok={vol_ok}")
                return False
        else: # 'bearish' - Looking for bearish reversal after a bullish break of resistance `level`
            body = abs(curr['close'] - curr['open'])
            if body == 0: return False
            wick = curr['high'] - curr['close'] # For bearish reversal, upper wick is not the rejection.
                                             # It's the current candle's bearish move after breaking high.
                                             # Original logic: wick = curr['high'] - curr['close']
            
            wick_ok = wick > 1.2 * body if body > 0 else wick > 0 # was 1.5x

            # Breakout: prev broke above level, curr came back below and closed below.
            breakout_condition = prev['high'] > level and curr['close'] < level
            # Original code had: prev['high'] > level and curr['low'] < level. Switched to curr['close'] for confirmation.

            if breakout_condition and wick_ok and vol_ok:
                logger.debug(f"[_is_false_breakout] Bearish detected: prev_high={prev['high']:.5f}, level={level:.5f}, curr_close={curr['close']:.5f}, wick={wick:.5f}, body={body:.5f}, vol_ok={vol_ok}")
                return True
            else:
                logger.debug(f"[_is_false_breakout] Bearish rejected: breakout={breakout_condition}, wick_ok={wick_ok} (wick={wick:.5f}, body={body:.5f}), vol_ok={vol_ok}")
                return False
        return False # Should not be reached if logic above is exhaustive

    # -- Confluence checks --
    def _find_recent_swing(self, df: pd.DataFrame, lookback: int = 50) -> tuple:
        """Find the most recent swing high and low in the last `lookback` bars."""
        if df is None or len(df) < lookback:
            return None, None
        recent = df.iloc[-lookback:]
        swing_high = recent['high'].max()
        swing_low = recent['low'].min()
        return swing_high, swing_low

    def _check_fibonacci(self, df: pd.DataFrame, level: float, use_fibonacci: bool = True) -> bool:
        """Check if `level` is near a standard Fibonacci retracement of the most recent swing.
        Now allows a flexible zone: Fib +/- 0.15% of price or 0.5x ATR, whichever is greater.
        """
        if not use_fibonacci:
            logger.debug(f"[Fib] use_fibonacci=False, skipping Fib confluence check for level {level}")
            return False
        swing_high, swing_low = self._find_recent_swing(df, lookback=50)
        if swing_high is None or swing_low is None:
            logger.debug(f"[Fib] No valid swing points for Fib check (level={level})")
            return False
        fib_tolerance = getattr(self, 'fib_tolerance', self.price_tolerance if hasattr(self, 'price_tolerance') else 0.001)
        # Flexible zone: max(0.0015 * price, 0.5 * ATR)
        atr_val = None
        if len(df) >= 14:
            high = np.asarray(df['high'].values, dtype=np.float64)
            low = np.asarray(df['low'].values, dtype=np.float64)
            close = np.asarray(df['close'].values, dtype=np.float64)
            atr_series = talib.ATR(high, low, close, timeperiod=14)
            if len(atr_series) > 0:
                atr_val = float(atr_series[-1])
        zone = max(swing_high * 0.0015, (atr_val * 0.5) if atr_val else 0)
        for f in self.fib_levels:
            fib_lv = swing_low + (swing_high - swing_low) * f
            if abs(level - fib_lv) <= zone:
                logger.info(f"[Fib] Level {level} matches Fib {f:.3f} ({fib_lv}) within flexible zone {zone}")
                return True
        logger.debug(f"[Fib] Level {level} does not match any Fib retracement within flexible zone {zone}")
        return False

    def _check_ma(self, df: pd.DataFrame, level: float) -> bool:
        """Return True if `level` is near the moving-average support/resistance (within a flexible zone)."""
        if df is None or len(df) < self.ma_period:
            return False
        close = np.asarray(df['close'].values, dtype=np.float64)
        ma = talib.SMA(close, timeperiod=self.ma_period)[-1]
        # Flexible zone: max(0.002 * price, 0.5 * ATR)
        atr_val = None
        if len(df) >= 14:
            high = np.asarray(df['high'].values, dtype=np.float64)
            low = np.asarray(df['low'].values, dtype=np.float64)
            close = np.asarray(df['close'].values, dtype=np.float64)
            atr_series = talib.ATR(high, low, close, timeperiod=14)
            if len(atr_series) > 0:
                atr_val = float(atr_series[-1])
        zone = max(ma * 0.002, (atr_val * 0.5) if atr_val else 0)
        if abs(level - ma) > zone:
            return False
        # Check MA slope: positive for bullish, negative for bearish
        ma_series = talib.SMA(close, timeperiod=self.ma_period)
        slope = ma_series[-1] - ma_series[-2] if len(ma_series) >= 2 else 0
        if slope > 0:
            return True  # Bullish
        elif slope < 0:
            return True  # Bearish
        return False 
    
    def _evaluate_signal_alignment(self, 
                                  primary_df: pd.DataFrame, 
                                  higher_df: pd.DataFrame, 
                                  direction: str) -> dict:
        """
        Evaluate how well the signal aligns with multiple factors
        """
        alignment = {}
        
        # Check alignment with primary timeframe trend
        if len(primary_df) >= self.ma_period:
            primary_trend = self._determine_trend(primary_df)
            alignment['primary_tf_trend_aligned'] = primary_trend == direction
            
        # Check alignment with higher timeframe trend
        if len(higher_df) >= self.ma_period:
            higher_trend = self._determine_trend(higher_df)
            alignment['higher_tf_trend_aligned'] = higher_trend == direction
            
        # Calculate overall alignment score (0.0 to 1.0)
        aligned_count = sum(1 for v in alignment.values() if v)
        alignment['alignment_score'] = aligned_count / len(alignment) if alignment else 0.0
        
        return alignment 

    def _find_next_key_level(self, df: pd.DataFrame, price: float, direction: str) -> dict:
        """Find the next key support/resistance zone beyond the entry price (returns zone, not just a price)"""
        supports, resistances = self._find_key_levels(df)
        if direction == 'buy':
            # For buy signals, find the next resistance zone above entry
            higher_resistances = [r for r in resistances if r['level'] > price]
            if higher_resistances:
                # Find the cluster (zone) for the next resistance
                next_level = min(higher_resistances, key=lambda x: x['level'])
                # Find all levels in the same cluster (within tolerance)
                tol = df['close'].mean() * self.price_tolerance
                zone_levels = [r['level'] for r in resistances if abs(r['level'] - next_level['level']) <= tol]
                zone_min = min(zone_levels)
                zone_max = max(zone_levels)
                zone_width = zone_max - zone_min
                distance = next_level['level'] - price
                return {
                    "level": next_level['level'],
                    "zone_min": zone_min,
                    "zone_max": zone_max,
                    "zone_width": zone_width,
                    "distance": distance,
                    "distance_r": distance / ((price - next_level['level']) / 2) if price != next_level['level'] else 0,
                    "type": "resistance"
                }
        else:
            # For sell signals, find the next support zone below entry
            lower_supports = [s for s in supports if s['level'] < price]
            if lower_supports:
                next_level = max(lower_supports, key=lambda x: x['level'])
                tol = df['close'].mean() * self.price_tolerance
                zone_levels = [s['level'] for s in supports if abs(s['level'] - next_level['level']) <= tol]
                zone_min = min(zone_levels)
                zone_max = max(zone_levels)
                zone_width = zone_max - zone_min
                distance = price - next_level['level']
                return {
                    "level": next_level['level'],
                    "zone_min": zone_min,
                    "zone_max": zone_max,
                    "zone_width": zone_width,
                    "distance": distance,
                    "distance_r": distance / ((next_level['level'] - price) / 2) if price != next_level['level'] else 0,
                    "type": "support"
                }
        return {}

    def _analyze_volume_quality(self, df: pd.DataFrame, idx: int, direction: str) -> tuple:
        """Analyze volume quality for a given candle index and direction. Returns (score, details)."""
        if df is None or len(df) == 0 or idx >= len(df) or idx < -len(df):
            return 0.0, {}
        candle = df.iloc[idx]
        vol_col = 'volume' if 'volume' in df.columns else 'tick_volume'
        # Use rolling median as threshold (robust to outliers)
        lookback = min(20, len(df))
        threshold = df[vol_col].iloc[-lookback:].median() if lookback > 0 else 1.0
        tick_volume = candle[vol_col] if vol_col in candle else threshold * 0.8
        volume_ratio = tick_volume / threshold if threshold > 0 else 1.0
        is_bullish = candle['close'] > candle['open']
        total_range = candle['high'] - candle['low']
        body = abs(candle['close'] - candle['open'])
        details = {
            'tick_volume': tick_volume,
            'threshold': threshold,
            'volume_ratio': volume_ratio,
            'body': body,
            'total_range': total_range
        }
        # Add debug logging for volume analysis
        logger.debug(f"[VolumeAnalysis] idx={idx}, direction={direction}, tick_volume={tick_volume}, threshold={threshold}, volume_ratio={volume_ratio:.2f}, total_range={total_range}, body={body}")
        if total_range == 0 or total_range < 0.00001:
            return 0.0, details
        if volume_ratio < 0.15:  # Reduced threshold from 0.2 to 0.15 for more lenient volume filtering
            return 0.0, details
        score = 0.0
        if direction == 'buy':
            upper_wick = candle['high'] - candle['close']
            lower_wick = candle['open'] - candle['low']
            upper_wick_ratio = upper_wick / total_range
            lower_wick_ratio = lower_wick / total_range
            body_ratio = body / total_range
            details.update({'upper_wick_ratio': upper_wick_ratio, 'lower_wick_ratio': lower_wick_ratio, 'body_ratio': body_ratio})
            logger.debug(f"[VolumeAnalysis] buy: upper_wick_ratio={upper_wick_ratio:.2f}, lower_wick_ratio={lower_wick_ratio:.2f}, body_ratio={body_ratio:.2f}")
            if upper_wick_ratio > 0.4:
                score = -1.0
            elif body_ratio > 0.6 and lower_wick_ratio < 0.2:
                score = 2.0
            elif body_ratio > 0.4 and lower_wick_ratio < upper_wick_ratio:
                score = 1.0
            elif upper_wick_ratio > 0.6:
                score = -0.5
            else:
                score = 0.5
        else:
            upper_wick = candle['high'] - candle['open']
            lower_wick = candle['close'] - candle['low']
            upper_wick_ratio = upper_wick / total_range
            lower_wick_ratio = lower_wick / total_range
            body_ratio = body / total_range
            details.update({'upper_wick_ratio': upper_wick_ratio, 'lower_wick_ratio': lower_wick_ratio, 'body_ratio': body_ratio})
            logger.debug(f"[VolumeAnalysis] sell: upper_wick_ratio={upper_wick_ratio:.2f}, lower_wick_ratio={lower_wick_ratio:.2f}, body_ratio={body_ratio:.2f}")
            if lower_wick_ratio > 0.4:
                score = 1.0
            elif body_ratio > 0.6 and upper_wick_ratio < 0.2:
                score = -2.0
            elif body_ratio > 0.4 and upper_wick_ratio < lower_wick_ratio:
                score = -1.0
            elif lower_wick_ratio > 0.6:
                score = 0.5
            else:
                score = -0.5
        # Clamp score to [0, 1] for positive, else 0
        if score >= 1.0:
            norm_score = 1.0
        elif score > 0:
            norm_score = max(score, 0.2)  # Allow a minimum score of 0.2 for low but positive volume
        else:
            norm_score = 0.0
        # Soft pass: allow 0.1 if moderate volume and body ratio
        if norm_score == 0.0:
            if 0.7 < volume_ratio < 1.0 and 0.4 < details.get('body_ratio', 0) < 0.6:
                norm_score = 0.1
        return norm_score, details

    @property
    def required_timeframes(self) -> list:
        """List of timeframes required by this strategy (for orchestrator data fetching)."""
        # Remove duplicates if higher and primary are the same
        return list({self.higher_timeframe, self.primary_timeframe})
