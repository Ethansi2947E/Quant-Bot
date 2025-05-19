"""
Confluence Price Action Strategy

This strategy identifies the prevailing trend on a higher timeframe, marks key support/resistance levels, waits for pullbacks on a lower timeframe, and then looks for confluence of price-action signals (pin bars, engulfing bars, inside bars, false breakouts), Fibonacci retracements, and moving-average support/resistance. Risk management enforces fixed fractional sizing and minimum R:R.
"""

import pandas as pd
import numpy as np
from loguru import logger
from datetime import datetime
from typing import Optional, List, Dict, Any

from src.trading_bot import SignalGenerator
from src.utils.indicators import calculate_atr, calculate_adx
from config.config import TRADING_CONFIG,get_risk_manager_config
from src.risk_manager import RiskManager

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

    def __init__(self,
                 primary_timeframe: str = "M15",
                 higher_timeframe: str = "H1",
                 ma_period: int = 21,
                 fib_levels=(0.5, 0.618),
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
                 **kwargs):
        """
        Args:
            ...
            adx_threshold (float): ADX threshold for trend regime. Default 15.0 (lowered for flexibility).
            range_ratio_threshold (float): Threshold for range_ratio to define ranging. Default 0.4.
        """
        super().__init__(**kwargs)
        self.name = "ConfluencePriceActionStrategy"
        self.description = (
            "Trend-follow pullbacks at key levels with candlestick confirmation, "
            "Fibonacci + MA confluence, fixed-fraction risk sizing"
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

        self.lookback_period = max(self.ma_period, 50) + 50  # Dynamic lookback

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

    async def generate_signals(self,
                               market_data: Optional[dict] = None,
                               symbol: Optional[str] = None,
                               timeframe: Optional[str] = None,
                               **kwargs) -> list:
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
            logger.debug(f"Analyzing {sym} - Higher TF: {self.higher_timeframe} ({len(higher)} bars), Primary TF: {self.primary_timeframe} ({len(primary)} bars)")
            
            # Bar tracking
            bar_key = (sym, self.primary_timeframe)
            try:
                last_timestamp = pd.to_datetime(primary.index[-1])
                last_timestamp_str = str(last_timestamp)
            except Exception:
                logger.warning(f"Could not extract timestamp from dataframe for {sym}")
                last_timestamp_str = str(current_time)
            if bar_key in self.processed_bars and self.processed_bars[bar_key] == last_timestamp_str:
                logger.debug(f"Already processed latest bar for {sym}/{self.primary_timeframe} at {last_timestamp_str}")
                continue
            self.processed_bars[bar_key] = last_timestamp_str
            logger.debug(f"Processing new bar for {sym}/{self.primary_timeframe} at {last_timestamp_str}")
            
            # --- Step 1: Cache ATR and ADX series for this symbol's primary timeframe ---
            atr_series = None
            adx_series = None
            if isinstance(primary, pd.DataFrame) and len(primary) >= 14:
                try:
                    atr_series = calculate_atr(primary, period=14)
                except Exception as e:
                    logger.warning(f"ATR calculation failed for {sym}: {e}")
                try:
                    adx_series, _, _ = calculate_adx(primary, period=14)
                except Exception as e:
                    logger.warning(f"ADX calculation failed for {sym}: {e}")
            
            # 1. Trend on higher timeframe
            trend = self._determine_trend(higher)
            logger.debug(f"{sym}: Higher timeframe trend is {trend}")
            if trend == 'neutral':
                logger.debug(f"{sym}: Skipping due to neutral trend")
                continue
            
            # Regime filter - skip if market regime is not favorable
            if not self._is_favorable_regime(primary):
                logger.debug(f"{sym}: Unfavorable market regime, skipping.")
                continue
            
            # 2. Key levels on higher timeframe
            supports, resistances = self._find_key_levels(higher)
            # Filter out weak levels (strength < 2)
            supports = [lvl for lvl in supports if lvl['strength'] >= 1]
            resistances = [lvl for lvl in resistances if lvl['strength'] >= 1]
            logger.debug(f"{sym}: Found {len(supports)} support levels and {len(resistances)} resistance levels (strength >= 1)")
            levels = supports if trend == 'bullish' else resistances
            level_type = "support" if trend == 'bullish' else "resistance"
            
            # 3. For each level, check pullback on primary TF
            level_info = []
            for i, level in enumerate(levels):
                level_info.append(f"Level {i+1}: {level['level']:.5f} (strength: {level['strength']})")
            
            logger.debug(f"{sym}: Checking {len(levels)} {level_type} levels: {', '.join(level_info)}")
            
            symbol_signals = []
            for level in levels:
                zone_type = 'support' if trend == 'bullish' else 'resistance'
                zone_key = (sym, zone_type, round(level['level'], 5))
                if zone_key in self.processed_zones:
                    last_used_time = self.processed_zones[zone_key]
                    time_since_use = current_time - last_used_time
                    if time_since_use < self.signal_cooldown:
                        cooldown_remaining = self.signal_cooldown - time_since_use
                        hours_remaining = cooldown_remaining / 3600
                        logger.debug(f"Skipping {zone_type} zone {level['level']:.5f} - on cooldown for {hours_remaining:.1f} more hours")
                        continue
                # --- Rejection-based (reversal) logic (existing) ---
                logger.debug(f"Checking pullback for {sym} at {level_type} level {level['level']:.5f} (trend: {trend})")
                signal_generated = False
                if self._is_pullback(primary, level['level'], trend):
                    logger.debug(f"Found pullback to {level_type} level {level['level']:.5f} for {sym}")
                    # Only check the latest candle for patterns
                    idx = len(primary) - 1
                    candle = primary.iloc[idx]
                    pattern = None
                    pattern_details = {}
                    logger.debug(f"Checking for patterns at idx={idx} ({candle.name}) for {sym} at level {level['level']:.5f} ({level_type})")
                    
                    # --- Explicit Breakout (Acceptance) vs. Rejection (False Break) Logic ---
                    acceptance = False
                    rejection = False
                    acceptance_tol = level['level'] * self.price_tolerance
                    # Acceptance: strong close through level, small wick
                    if trend == 'bullish':
                        if (
                            candle['open'] < level['level'] and
                            candle['close'] > level['level'] and
                            (candle['high'] - candle['close']) < acceptance_tol
                        ):
                            acceptance = True
                    else:
                        if (
                            candle['open'] > level['level'] and
                            candle['close'] < level['level'] and
                            (candle['close'] - candle['low']) < acceptance_tol
                        ):
                            acceptance = True
                    # Rejection: false break then close back inside
                    if trend == 'bullish':
                        if (
                            candle['open'] < level['level'] and
                            candle['high'] > level['level'] and
                            candle['close'] < level['level']
                        ):
                            rejection = True
                    else:
                        if (
                            candle['open'] > level['level'] and
                            candle['low'] < level['level'] and
                            candle['close'] > level['level']
                        ):
                            rejection = True
                    # Add explicit pattern/rationale
                    if acceptance:
                        pattern = 'Breakout Acceptance'
                        pattern_details = {
                            'open': candle['open'],
                            'close': candle['close'],
                            'level': level['level'],
                            'wick_size': (candle['high'] - candle['close']) if trend == 'bullish' else (candle['close'] - candle['low']),
                            'acceptance_tol': acceptance_tol
                        }
                        logger.debug(f"Breakout Acceptance detected at idx={idx} for {sym}")
                    elif rejection:
                        pattern = 'Rejection Reversal'
                        pattern_details = {
                            'open': candle['open'],
                            'close': candle['close'],
                            'level': level['level'],
                            'wick_size': (candle['high'] - candle['close']) if trend == 'bullish' else (candle['close'] - candle['low']),
                            'acceptance_tol': acceptance_tol
                        }
                        logger.debug(f"Rejection Reversal detected at idx={idx} for {sym}")
                    
                    if not pattern:
                        logger.debug(f"No valid pattern detected at idx={idx} for {sym} at level {level['level']:.5f}")
                        continue
                    
                    logger.debug(f"{sym}: Found {pattern} pattern at {candle.name}")
                    
                    # 5. Confluence: Fibonacci or MA
                    fib_ok = self._check_fibonacci(primary, level['level'], True)
                    fib_details = {}
                    if fib_ok:
                        high = primary['high'].max()
                        low = primary['low'].min()
                        for f in self.fib_levels:
                            fib_lv = low + (high - low) * f
                            if abs(level['level'] - fib_lv) <= level['level'] * self.price_tolerance:
                                fib_details = {
                                    'fib_level': f,
                                    'calculated_value': fib_lv,
                                    'price_to_fib_distance': abs(level['level'] - fib_lv)
                                }
                                break
                    ma_ok = self._check_ma(primary, level['level'])
                    ma_details = {}
                    if ma_ok:
                        ma = primary['close'].rolling(self.ma_period).mean().iloc[-1]
                        ma_details = {
                            'ma_period': self.ma_period,
                            'ma_value': ma,
                            'price_to_ma_distance': abs(level['level'] - ma)
                        }
                    # --- Step 4: Flexible confluence stacking and early scoring ---
                    # New: Confluence Scoring System (see .cursor/scratchpad.md for rubric)
                    # Core factors: HTF Trend, HTF S/R, Pattern (all must be present)
                    htf_trend_score = 1.0 if trend in ('bullish', 'bearish') else 0.0
                    htf_sr_score = 1.0 if level['strength'] >= 1 else 0.0
                    pattern_score = 0.0
                    # Expanded pattern recognition with more tolerance and new patterns
                    if self._is_pin_bar(primary, idx, level['level'], trend):
                        pattern = 'Pin Bar'
                        pattern_score = 1.0
                    elif self._is_engulfing(primary, idx, trend, level['level']):
                        pattern = 'Engulfing'
                        pattern_score = 0.9
                    elif self._is_inside_bar(primary, idx, level['level']):
                        pattern = 'Inside Bar'
                        pattern_score = 0.7
                    elif self._is_two_bar_reversal(primary, idx, trend, level['level']):
                        pattern = 'Two-Bar Reversal'
                        pattern_score = 0.8
                    elif self._is_three_bar_reversal(primary, idx, trend, level['level']):
                        pattern = 'Three-Bar Reversal'
                        pattern_score = 0.85
                    elif pattern == 'Rejection Reversal' or pattern == 'Breakout Acceptance':
                        pattern_score = 0.8
                    # Optional/bonus factors
                    fib_score = 0.3 if fib_ok else 0.0
                    ma_score = 0.3 if ma_ok else 0.0
                    volume_score, volume_details = self._analyze_volume_quality(primary, idx=-1, direction=trend)
                    volume_score = min(max(volume_score, 0.0), 0.2)  # Clamp to [0, 0.2]
                    level_strength_score = min(0.2, level['strength'] / 5.0)  # Scaled bonus
                    # Total score
                    confluence_total = htf_trend_score + htf_sr_score + pattern_score + fib_score + ma_score + volume_score + level_strength_score
                    # Minimum score to trigger trade
                    min_score = 2.2
                    # Allow slightly weaker S/R or filter if total score >= 2.5
                    if htf_trend_score < 1.0 or htf_sr_score < 1.0 or pattern_score < 0.7:
                        if confluence_total < 2.5:
                            logger.debug(f"[ConfluenceScoring] Signal for {sym} at {level['level']:.5f} rejected: missing core factor and total score {confluence_total:.2f} < 2.5")
                            continue
                    else:
                        if confluence_total < min_score:
                            logger.debug(f"[ConfluenceScoring] Signal for {sym} at {level['level']:.5f} rejected: total score {confluence_total:.2f} < {min_score}")
                            continue
                    logger.info(f"[ConfluenceScoring] {sym} {pattern} at {level_type} {level['level']:.5f}: htf_trend={htf_trend_score}, htf_sr={htf_sr_score}, pattern={pattern_score}, fib={fib_score}, ma={ma_score}, volume={volume_score}, sr_strength={level_strength_score}, total={confluence_total}")
                    
                    # 6. Assemble signal
                    # Use the pattern's close as entry, not the latest bar
                    entry = primary.iloc[idx]['close']
                    # --- ATR-based SL tolerance ---
                    atr_val = None
                    if len(primary) >= 14:
                        atr_series = calculate_atr(primary, period=14)
                        if isinstance(atr_series, pd.Series):
                            atr_val = float(atr_series.iloc[idx]) if idx < len(atr_series) else float(atr_series.iloc[-1])
                        else:
                            try:
                                if isinstance(atr_series, (float, int, np.floating, np.integer)):
                                    atr_val = float(atr_series)
                                else:
                                    atr_val = None
                            except Exception:
                                atr_val = None
                    else:
                        atr_val = level['level'] * self.price_tolerance
                    # Ensure atr_val is a valid float, fallback to sane default if not
                    if not atr_val or not np.isfinite(atr_val) or atr_val == 0:
                        atr_val = entry * 0.001  # fallback default
                    # --- Robust SL tolerance: use max of ATR and pattern candle's range ---
                    candle_range = candle['high'] - candle['low']
                    tol_val = max(level['level'] * self.price_tolerance, atr_val if atr_val is not None else 0, candle_range)
                    logger.debug(f"[SL] ATR: {atr_val}, Candle range: {candle_range}, tol_val: {tol_val}")
                    if trend == 'bullish':
                        stop = candle['low'] - tol_val
                        reward = entry - stop
                        tp = entry + reward * self.min_risk_reward
                        direction = 'buy'
                    else:
                        stop = candle['high'] + tol_val
                        reward = stop - entry
                        tp = entry - reward * self.min_risk_reward
                        direction = 'sell'
                    # --- CAP SL/TP DISTANCE ---
                    max_sl_dist = None
                    if atr_val is not None and self.max_sl_atr_mult is not None:
                        max_sl_dist = max(atr_val * self.max_sl_atr_mult, 2 * TICK_SIZE)
                    if self.max_sl_pct is not None:
                        max_sl_dist_pct = entry * self.max_sl_pct
                        max_sl_dist = min(max_sl_dist, max_sl_dist_pct) if max_sl_dist is not None else max_sl_dist_pct
                    # Always ensure max_sl_dist is valid
                    if max_sl_dist is None or not np.isfinite(max_sl_dist) or max_sl_dist == 0:
                        max_sl_dist = max(entry * 0.002, 2 * TICK_SIZE)  # fallback default
                    logger.debug(f"[SL] max_sl_dist: {max_sl_dist}, abs(entry-stop): {abs(entry-stop)}")
                    if abs(entry - stop) > max_sl_dist:
                        if direction == 'buy':
                            stop = entry - max_sl_dist
                        else:
                            stop = entry + max_sl_dist
                        reward = abs(entry - stop)
                        tp = entry + reward * self.min_risk_reward if direction == 'buy' else entry - reward * self.min_risk_reward
                    # Calculate risk-reward and other trade metrics
                    risk_pips = abs(entry - stop)
                    reward_pips = abs(tp - entry)
                    # Enforce minimum pip reward (e.g., 5 ticks)
                    min_pip_reward = 5 * TICK_SIZE
                    if reward_pips < min_pip_reward:
                        logger.debug(f"{sym}: Skipping signal due to reward_pips {reward_pips:.5f} < minimum {min_pip_reward:.5f}")
                        continue
                    
                    # Calculate dynamic exit targets
                    exits = self._calculate_dynamic_exits(
                        primary=primary,
                        direction=direction,
                        entry=entry, 
                        stop=stop,
                        risk_pips=risk_pips,
                        level=level['level'],
                        candle_idx=idx
                    )
                    
                    # Build concise analysis string
                    volume_desc = "strong volume" if volume_score > 1 else ("adequate volume" if volume_score > 0 else "weak volume")
                    rationale = f"Detected a {pattern} at {level_type}, suggesting a potential {direction} reversal. Volume is {volume_desc}, supporting the signal."
                    concise_analysis = f"📝 Analysis:\n{direction.capitalize()} reversal ({pattern}) at {level_type} {level['level']:.5f} with {volume_desc}.\nRationale: {rationale}"
                    reasoning = [concise_analysis]
                   
                    # Technical metrics for entry
                    technical_metrics = {}
                    if len(primary) >= 14:
                        delta = primary['close'].diff().astype(float)
                        gain = delta.where(delta > 0, 0)
                        loss = -delta.where(delta < 0, 0)
                        avg_gain = gain.rolling(window=14).mean()
                        avg_loss = loss.rolling(window=14).mean()
                        rs = avg_gain / avg_loss
                        rsi = 100 - (100 / (1 + rs))
                        current_rsi = rsi.iloc[-1]
                        technical_metrics['rsi'] = current_rsi
                        reasoning.append(f"RSI: {current_rsi:.1f}")
                    if len(primary) >= 14:
                        atr_series = calculate_atr(primary, period=14)
                        if isinstance(atr_series, pd.Series):
                            current_atr = atr_series.iloc[-1]
                        else:
                            try:
                                if isinstance(atr_series, (float, int, np.floating, np.integer)):
                                    current_atr = float(atr_series)
                                else:
                                    current_atr = None
                            except Exception:
                                current_atr = None
                        if current_atr is not None and current_atr > 0:
                            stop_distance_atr = risk_pips / current_atr
                            technical_metrics['atr'] = current_atr
                            technical_metrics['stop_distance_atr'] = stop_distance_atr
                            reasoning.append(f"Stop distance: {stop_distance_atr:.2f} ATR")
                    # Score breakdown
                   
                    # Only allow buy reversals if higher timeframe trend is bullish, sell reversals if bearish
                    if (direction == 'buy' and trend != 'bullish') or (direction == 'sell' and trend != 'bearish'):
                        logger.debug(f"{sym}: Skipping reversal signal due to higher timeframe trend filter (trend: {trend}, direction: {direction})")
                        continue
                    
                    # --- RiskManager integration for validation and sizing ---
                    # --- Scoring weights and normalization ---
                    pattern_weight = 0.4
                    confluence_weight = 0.4
                    volume_weight = 0.1
                    recency_weight = 0.1
                    # Normalize sub-scores to 0-1
                    norm_pattern_score = min(max(pattern_score, 0.0), 1.0)
                    # Confluence: sum of htf_trend_score, htf_sr_score, fib_score, ma_score, level_strength_score (max 4.5)
                    norm_confluence_score = min(max((htf_trend_score + htf_sr_score + fib_score + ma_score + level_strength_score) / 4.5, 0.0), 1.0)
                    norm_volume_score = min(max(volume_score / 0.2, 0.0), 1.0)  # 0.2 is max volume score
                    norm_recency_score = 0.0  # Placeholder, set to 0 unless recency logic is added
                    # Weighted sum for signal quality/confidence
                    signal_quality = (
                        norm_pattern_score * pattern_weight +
                        norm_confluence_score * confluence_weight +
                        norm_volume_score * volume_weight +
                        norm_recency_score * recency_weight
                    )
                    # Store normalized (0-1) value in signal dict
                    signal = {
                        "symbol": sym,
                        "direction": direction,
                        "entry_price": entry,
                        "stop_loss": stop,
                        "take_profit": tp,
                        "dynamic_exits": exits,
                        "timeframe": self.primary_timeframe,
                        "confidence": signal_quality,  # Normalized 0-1
                        "source": self.name,
                        "pattern": pattern,
                        "confluence": {"fib": fib_ok, "ma": ma_ok},
                        "pattern_details": pattern_details,
                        "fib_details": fib_details if fib_ok else {},
                        "ma_details": ma_details if ma_ok else {},
                        "volume_details": volume_details,
                        "risk_pips": risk_pips,
                        "reward_pips": reward_pips,
                        "risk_reward_ratio": reward_pips / risk_pips if risk_pips > 0 else 0,
                        "signal_quality": signal_quality,  # Normalized 0-1
                        "technical_metrics": technical_metrics,
                        "pattern_score": norm_pattern_score,
                        "confluence_score": norm_confluence_score,
                        "volume_score": norm_volume_score,
                        "recency_score": norm_recency_score,
                        "level_strength": level['strength'],
                        "level_strength_score": level_strength_score,
                        "description": concise_analysis,
                        "detailed_reasoning": reasoning
                    }
                    # Add signal freshness metadata
                    signal['signal_bar_index'] = idx
                    signal['signal_timestamp'] = str(candle.name)
                    result = rm.validate_and_size_trade(signal)
                    if not result['valid']:
                        logger.info(f"Signal for {sym} rejected by RiskManager: {result['reason']}")
                        continue
                    adjusted_signal = result['adjusted_trade']
                    for k in signal:
                        if k not in adjusted_signal:
                            adjusted_signal[k] = signal[k]
                    symbol_signals.append(adjusted_signal)
                    # After signal is generated and appended:
                    self.processed_zones[zone_key] = current_time
                    # Only generate one signal per zone per bar
                    break
            # Prioritize signals if there are conflicting ones
            if len(symbol_signals) > 1:
                symbol_signals = self._prioritize_signals(symbol_signals)
            signals.extend(symbol_signals)
        # Cleanup old processed zones entries (older than 48 hours)
        cleanup_time = current_time - (self.signal_cooldown * 2)
        old_keys = [k for k, v in self.processed_zones.items() if v < cleanup_time]
        for k in old_keys:
            del self.processed_zones[k]
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
        ma = df['close'].rolling(window=self.ma_period).mean().iloc[-1]
        last = df['close'].iloc[-1]
        ma_trend = 'bullish' if last > ma else 'bearish' if last < ma else 'neutral'
        highs = df['high']
        lows = df['low']
        swing_highs = highs[(highs.shift(1) < highs) & (highs.shift(-1) < highs)]
        swing_lows = lows[(lows.shift(1) > lows) & (lows.shift(-1) > lows)]
        last_highs = swing_highs.tail(3)
        last_lows = swing_lows.tail(3)
        price_trend = 'neutral'
        if len(last_highs) == 3 and len(last_lows) == 3:
            if last_highs.iloc[2] > last_highs.iloc[1] > last_highs.iloc[0] and last_lows.iloc[2] > last_lows.iloc[1] > last_lows.iloc[0]:
                price_trend = 'bullish'
            elif last_highs.iloc[2] < last_highs.iloc[1] < last_highs.iloc[0] and last_lows.iloc[2] < last_lows.iloc[1] < last_lows.iloc[0]:
                price_trend = 'bearish'
        logger.debug(f"[Trend] MA trend: {ma_trend}, Price structure trend: {price_trend}")
        if ma_trend == price_trend and ma_trend != 'neutral':
            logger.info(f"[Trend] Confirmed {ma_trend} trend (MA + price structure)")
            return ma_trend
        elif ma_trend != 'neutral' or price_trend != 'neutral':
            # Loosened: allow if either is clear
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
        # Pivot detection (use subset for all indexing)
        for i in range(2, len(subset) - 2):
            if (subset['low'].iat[i] < subset['low'].iat[i-1] and subset['low'].iat[i] < subset['low'].iat[i-2]
                    and subset['low'].iat[i] < subset['low'].iat[i+1] and subset['low'].iat[i] < subset['low'].iat[i+2]):
                lows.append(subset['low'].iat[i])
            if (subset['high'].iat[i] > subset['high'].iat[i-1] and subset['high'].iat[i] > subset['high'].iat[i-2]
                    and subset['high'].iat[i] > subset['high'].iat[i+1] and subset['high'].iat[i] > subset['high'].iat[i+2]):
                highs.append(subset['high'].iat[i])
        logger.debug(f"[KeyLevels] Raw pivot lows: {len(lows)}, pivot highs: {len(highs)} (before clustering)")
        # Clustering tolerance is the maximum of 5 ticks and the minimum of (mean price * price_tolerance, 15 ticks),
        # but is further widened to at least ATR*0.1 if ATR is available. This ensures clusters are not too loose, but still adapt to volatility.
        clustering_tol = max(5 * TICK_SIZE, min(subset['close'].mean() * self.price_tolerance, 15 * TICK_SIZE))
        atr_val = None
        if len(subset) >= 14:
            from src.utils.indicators import calculate_atr
            atr_series = calculate_atr(subset, period=14)
            if isinstance(atr_series, pd.Series):
                atr_val = float(atr_series.iloc[-1])
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
            from src.utils.indicators import calculate_atr
            atr_series = calculate_atr(df, period=14)
            if isinstance(atr_series, pd.Series):
                atr_val = float(atr_series.iloc[-1])
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

    # -- Candlestick pattern checks --
    def _is_pin_bar(self, df: pd.DataFrame, idx: int, level: float, direction: str) -> bool:
        """Loosened: More tolerant pin bar detection. Accept near-miss, log reason if rejected."""
        candle = df.iloc[idx]
        body = abs(candle['close'] - candle['open'])
        upper_wick = candle['high'] - max(candle['close'], candle['open'])
        lower_wick = min(candle['close'], candle['open']) - candle['low']
        total_range = candle['high'] - candle['low']
        # Loosened ratios
        min_wick_ratio = 1.5  # was 2.0
        max_body_ratio = 0.7  # was 0.5
        if direction == 'bullish':
            if lower_wick >= min_wick_ratio * body and body / total_range < max_body_ratio:
                logger.debug(f"[Pattern] Pin Bar (bullish) detected at idx={idx}")
                return True
            elif lower_wick >= (min_wick_ratio - 0.3) * body:
                logger.debug(f"[Pattern] Near-miss Pin Bar (bullish) at idx={idx}: lower_wick={lower_wick}, body={body}, total_range={total_range}")
                return True  # Accept near-miss
            else:
                logger.debug(f"[Pattern] Pin Bar (bullish) rejected at idx={idx}: lower_wick={lower_wick}, body={body}, total_range={total_range}")
                return False
        else:
            if upper_wick >= min_wick_ratio * body and body / total_range < max_body_ratio:
                logger.debug(f"[Pattern] Pin Bar (bearish) detected at idx={idx}")
                return True
            elif upper_wick >= (min_wick_ratio - 0.3) * body:
                logger.debug(f"[Pattern] Near-miss Pin Bar (bearish) at idx={idx}: upper_wick={upper_wick}, body={body}, total_range={total_range}")
                return True  # Accept near-miss
            else:
                logger.debug(f"[Pattern] Pin Bar (bearish) rejected at idx={idx}: upper_wick={upper_wick}, body={body}, total_range={total_range}")
                return False

    def _is_engulfing(self, candles: pd.DataFrame, idx: int, direction: str, level: float) -> bool:
        """Loosened: More tolerant engulfing detection. Accept near-miss, log reason if rejected."""
        if idx < 1:
            return False
        prev = candles.iloc[idx - 1]
        curr = candles.iloc[idx]
        # Loosened: allow 90% engulf
        if direction == 'bullish':
            if curr['close'] > curr['open'] and prev['close'] < prev['open']:
                if curr['close'] >= prev['open'] and curr['open'] <= prev['close']:
                    logger.debug(f"[Pattern] Engulfing (bullish) detected at idx={idx}")
                    return True
                elif curr['close'] >= prev['open'] * 0.98:
                    logger.debug(f"[Pattern] Near-miss Engulfing (bullish) at idx={idx}")
                    return True
                else:
                    logger.debug(f"[Pattern] Engulfing (bullish) rejected at idx={idx}")
                    return False
        else:
            if curr['close'] < curr['open'] and prev['close'] > prev['open']:
                if curr['close'] <= prev['open'] and curr['open'] >= prev['close']:
                    logger.debug(f"[Pattern] Engulfing (bearish) detected at idx={idx}")
                    return True
                elif curr['close'] <= prev['open'] * 1.02:
                    logger.debug(f"[Pattern] Near-miss Engulfing (bearish) at idx={idx}")
                    return True
                else:
                    logger.debug(f"[Pattern] Engulfing (bearish) rejected at idx={idx}")
                    return False
        return False

    def _is_inside_bar(self, candles: pd.DataFrame, idx: int, level: float) -> bool:
        """Detect an inside bar where the current candle is fully contained within the previous candle.
        Allow mother candle to be within max(tol, ATR*0.2) of the S/R level (not child close).
        """
        if idx <= 0 or idx >= len(candles):
            return False
        mother = candles.iloc[idx - 1]
        child = candles.iloc[idx]
        tol = child['close'] * self.price_tolerance
        atr_val = None
        if len(candles) >= 14:
            from src.utils.indicators import calculate_atr
            atr_series = calculate_atr(candles, period=14)
            if isinstance(atr_series, pd.Series):
                atr_val = float(atr_series.iloc[idx]) if idx < len(atr_series) else float(atr_series.iloc[-1])
        offset = max(tol, (atr_val * 0.2) if atr_val else 0)
        if not isinstance(level, (float, int)):
            return False  # Level must be provided for correct anchoring
        level = float(level)
        if child['high'] < mother['high'] and child['low'] > mother['low']:
            if abs(mother['low'] - level) <= offset or abs(mother['high'] - level) <= offset:
                return True
        return False

    def _is_hammer(self, df: pd.DataFrame, idx: int, level: float) -> bool:
        """Detect a Hammer pattern (bullish reversal) near support."""
        candle = df.iloc[idx]
        body = abs(candle['close'] - candle['open'])
        total = candle['high'] - candle['low']
        lower_wick = min(candle['open'], candle['close']) - candle['low']
        upper_wick = candle['high'] - max(candle['open'], candle['close'])
        if total <= 0:
            return False
        wick_req = max(1.2 * body, 0)
        return (
            body / total < 0.3 and
            lower_wick > wick_req and
            upper_wick < body and
            abs(candle['low'] - level) < level * self.price_tolerance
        )

    def _is_shooting_star(self, df: pd.DataFrame, idx: int, level: float) -> bool:
        """Detect a Shooting Star pattern (bearish reversal) near resistance."""
        candle = df.iloc[idx]
        body = abs(candle['close'] - candle['open'])
        total = candle['high'] - candle['low']
        upper_wick = candle['high'] - max(candle['open'], candle['close'])
        lower_wick = min(candle['open'], candle['close']) - candle['low']
        if total <= 0:
            return False
        return (
            body / total < 0.3 and
            upper_wick > 2 * body and
            lower_wick < body and
            abs(candle['high'] - level) < level * self.price_tolerance
        )

    def _is_morning_star(self, candles: pd.DataFrame, idx: int, level: float) -> bool:
        """Detect a Morning Star (bullish 3-bar reversal) near support, allow pattern only in last 5 bars."""
        if idx < 2:
            return False
        # Only require pattern occurs anywhere in last 5 bars
        if idx < len(candles) - 5:
            return False
        c1, c2, c3 = candles.iloc[idx-2], candles.iloc[idx-1], candles.iloc[idx]
        return (
            c1['close'] < c1['open'] and
            abs(c1['close'] - c1['open']) > (c1['high'] - c1['low']) * 0.5 and
            abs(c2['close'] - c2['open']) < (c2['high'] - c2['low']) * 0.3 and
            c3['close'] > c3['open'] and
            c3['close'] > c1['open'] and
            abs(c3['low'] - level) < level * self.price_tolerance
        )

    def _is_evening_star(self, candles: pd.DataFrame, idx: int, level: float) -> bool:
        """Detect an Evening Star (bearish 3-bar reversal) near resistance, allow pattern only in last 5 bars."""
        if idx < 2:
            return False
        if idx < len(candles) - 5:
            return False
        c1, c2, c3 = candles.iloc[idx-2], candles.iloc[idx-1], candles.iloc[idx]
        return (
            c1['close'] > c1['open'] and
            abs(c1['close'] - c1['open']) > (c1['high'] - c1['low']) * 0.5 and
            abs(c2['close'] - c2['open']) < (c2['high'] - c2['low']) * 0.3 and
            c3['close'] < c3['open'] and
            c3['close'] < c1['open'] and
            abs(c3['high'] - level) < level * self.price_tolerance
        )

    def _is_false_breakout(self, candles: pd.DataFrame, idx: int, level: float, direction: str) -> bool:
        """Detect a quick reversal after a breakout around `level` with wick and volume analysis.
        Relaxed: volume > 1.0x avg, wick > 1.2x body, allow touch-breaks (not just full clean breakouts).
        """
        if idx <= 0 or idx >= len(candles):
            return False
        prev = candles.iloc[idx - 1]
        curr = candles.iloc[idx]
        tol_val = level * self.price_tolerance
        vol_col = 'volume' if 'volume' in candles.columns else 'tick_volume'
        avg_vol = candles[vol_col].rolling(window=20).mean().iloc[idx]
        # Require volume at least 1.0x average (was 1.2x)
        vol_ok = curr[vol_col] > 1.0 * avg_vol
        if direction == 'bullish':
            wick = curr['close'] - curr['low']
            body = abs(curr['close'] - curr['open'])
            wick_ok = wick > 1.2 * body  # was 1.5x
            # Allow touch-breaks: prev['low'] < level and curr['high'] > level
            breakout = prev['low'] < level and curr['high'] > level
            return breakout and wick_ok and vol_ok
        else:
            wick = curr['high'] - curr['close']
            body = abs(curr['close'] - curr['open'])
            wick_ok = wick > 1.2 * body  # was 1.5x
            breakout = prev['high'] > level and curr['low'] < level
            return breakout and wick_ok and vol_ok

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
            from src.utils.indicators import calculate_atr
            atr_series = calculate_atr(df, period=14)
            if isinstance(atr_series, pd.Series):
                atr_val = float(atr_series.iloc[-1])
        zone = max(swing_high * 0.0015, (atr_val * 0.5) if atr_val else 0)
        for f in self.fib_levels:
            fib_lv = swing_low + (swing_high - swing_low) * f
            if abs(level - fib_lv) <= zone:
                logger.info(f"[Fib] Level {level} matches Fib {f:.3f} ({fib_lv}) within flexible zone {zone}")
                return True
        logger.debug(f"[Fib] Level {level} does not match any Fib retracement within flexible zone {zone}")
        return False

    def _check_ma(self, df: pd.DataFrame, level: float) -> bool:
        """Return True if `level` is near the moving-average support/resistance (within a flexible zone).
        Now allows MA +/- 0.2% of price or 0.5x ATR, whichever is greater.
        """
        if df is None or len(df) < self.ma_period:
            return False
        ma = df['close'].rolling(self.ma_period).mean().iloc[-1]
        # Flexible zone: max(0.002 * price, 0.5 * ATR)
        atr_val = None
        if len(df) >= 14:
            from src.utils.indicators import calculate_atr
            atr_series = calculate_atr(df, period=14)
            if isinstance(atr_series, pd.Series):
                atr_val = float(atr_series.iloc[-1])
        zone = max(ma * 0.002, (atr_val * 0.5) if atr_val else 0)
        if abs(level - ma) > zone:
            return False
        # Check MA slope: positive for bullish, negative for bearish
        ma_series = df['close'].rolling(self.ma_period).mean()
        slope = ma_series.iloc[-1] - ma_series.iloc[-2] if len(ma_series) >= 2 else 0
        if slope > 0:
            return True  # Bullish
        elif slope < 0:
            return True  # Bearish
        return False

    # -- Position sizing --
    def _calculate_position_size(self, entry: float, stop: float, balance: float) -> float:
        """Return lot size based on risk percentage and price difference"""
        risk_amount = balance * self.risk_percent
        pip_risk = abs(entry - stop)
        if pip_risk <= 0:
            return 0
        return risk_amount / pip_risk 

    # -- Additional utility methods for trade analysis --
    def _analyze_market_context(self, df: pd.DataFrame) -> dict:
        """
        Analyze current market context to provide additional insight
        """
        if df is None or len(df) < 20:
            return {}
            
        # Get recent price action characteristics
        recent = df.iloc[-20:]
        price_range = recent['high'].max() - recent['low'].min()
        close_range = recent['close'].max() - recent['close'].min()
        range_ratio = close_range / price_range if price_range > 0 else 0
        
        # Calculate recent volatility
        volatility = recent['close'].pct_change().std() * 100  # as percentage
        
        # Identify if in range or trending recently
        is_ranging = False
        if hasattr(self, 'use_range_filter') and self.use_range_filter:
            is_ranging = range_ratio < (self.range_ratio_threshold if hasattr(self, 'range_ratio_threshold') else 0.5)
        
        # Calculate momentum
        momentum = (recent['close'].iloc[-1] - recent['close'].iloc[0]) / recent['close'].iloc[0] * 100
        
        return {
            'price_range': price_range,
            'close_range': close_range,
            'range_ratio': range_ratio,
            'volatility': volatility,
            'is_ranging': is_ranging,
            'momentum': momentum,
            'bars_analyzed': len(recent)
        }
    
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

    def _is_favorable_regime(self, df: pd.DataFrame) -> bool:
        """Check if market regime is favorable (trending, not ranging). Add debug logs for ADX and range ratio."""
        if df is None or len(df) < 20:
            logger.debug("[Regime] Not enough data for regime check.")
            return False
        adx_series, _, _ = calculate_adx(df, period=14)
        adx = adx_series.iloc[-1] if isinstance(adx_series, pd.Series) and not adx_series.empty else None
        context = self._analyze_market_context(df)
        adx_ok = True
        if hasattr(self, 'use_adx_filter') and self.use_adx_filter:
            adx_ok = adx is not None and adx > (self.adx_threshold if hasattr(self, 'adx_threshold') else 15.0)
        range_ok = True
        if hasattr(self, 'use_range_filter') and self.use_range_filter:
            range_ok = not context.get('is_ranging', False)
        logger.debug(f"[Regime] ADX={adx}, threshold={self.adx_threshold}, adx_ok={adx_ok}, range_ratio={context.get('range_ratio', None)}, threshold={self.range_ratio_threshold}, range_ok={range_ok}")
        return adx_ok and range_ok

    def _calculate_dynamic_exits(self, primary: pd.DataFrame, direction: str, entry: float, 
                                stop: float, risk_pips: float, level: float, candle_idx: int) -> dict:
        """
        Calculate dynamic exit strategies including:
        1. Partial profit target (at 1:1 R:R)
        2. Time-based exit (after max_bars_till_exit)
        3. Trailing stop parameters (based on ATR)
        4. Market Profile range extension TP (avg high-low range over 20 bars)
        """
        # Find the next support/resistance level for extended target
        higher_levels = self._find_next_key_level(primary, entry, direction)
        # Calculate partial profit target at 1:1 R:R
        partial_tp = None
        if direction == 'buy':
            partial_tp = entry + (risk_pips * self.partial_profit_rr)
        else:
            partial_tp = entry - (risk_pips * self.partial_profit_rr)
        # Calculate time-based exit (bar index to exit if targets not reached)
        time_exit_bar = candle_idx + self.max_bars_till_exit
        # Calculate ATR-based trailing stop
        atr_value = 0
        if len(primary) >= 14:
            atr_series = calculate_atr(primary, period=14)
            if isinstance(atr_series, pd.Series) and not atr_series.empty:
                atr_value = float(atr_series.iloc[-1])
            else:
                try:
                    if isinstance(atr_series, (float, int, np.floating, np.integer)):
                        atr_value = float(atr_series)
                    else:
                        atr_value = None
                except Exception:
                    atr_value = None
        else:
            atr_value = risk_pips / 2  # If not enough data for ATR, use half risk_pips
        # Only activate trailing stop after partial profit hit
        trailing_activation = partial_tp
        trailing_distance = (atr_value if atr_value is not None else 0.0) * self.trailing_stop_atr_mult
        # Market Profile range extension TP
        lookback = min(20, len(primary))
        avg_range = (primary['high'].iloc[-lookback:] - primary['low'].iloc[-lookback:]).mean() if lookback > 0 else 0
        range_tp = entry + avg_range if direction == 'buy' else entry - avg_range
        return {
            "partial_profit": {
                "price": partial_tp,
                "pct": self.partial_profit_pct,
                "r_multiple": self.partial_profit_rr
            },
            "time_exit": {
                "bar_idx": time_exit_bar,
                "bars_from_entry": self.max_bars_till_exit,
                "timeframe": self.primary_timeframe
            },
            "trailing_stop": {
                "activation_price": trailing_activation,
                "atr_value": atr_value,
                "distance": trailing_distance,
                "atr_multiple": self.trailing_stop_atr_mult
            },
            "next_key_level": higher_levels,
            "range_extension": {
                "price": range_tp,
                "type": "market_profile"
            }
        }

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

    def _is_acceptance(self, df: pd.DataFrame, level: float, direction: str, bars: int = 2, tol: float = 0.001) -> bool:
        """Check if price has accepted above (bullish) or below (bearish) a level for N bars."""
        if df is None or len(df) < bars:
            return False
        closes = df['close'].iloc[-bars:]
        if direction == 'bullish':
            # Using inclusive comparison to address borderline cases
            return all(close >= level + level * tol for close in closes)
        else:
            return all(close <= level - level * tol for close in closes)

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
        if volume_ratio < 0.2:
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

    def _is_two_bar_reversal(self, candles: pd.DataFrame, idx: int, direction: str, level: float) -> bool:
        """Detect a two-bar reversal pattern (bullish or bearish) near a level."""
        if idx <= 0 or idx >= len(candles):
            return False
        curr = candles.iloc[idx]
        prev = candles.iloc[idx - 1]
        tol = curr['close'] * self.price_tolerance
        atr_val = None
        if len(candles) >= 14:
            from src.utils.indicators import calculate_atr
            atr_series = calculate_atr(candles, period=14)
            if isinstance(atr_series, pd.Series):
                atr_val = float(atr_series.iloc[idx]) if idx < len(atr_series) else float(atr_series.iloc[-1])
        offset = max(tol, (atr_val * 0.2) if atr_val else 0)
        if direction == 'bullish':
            if prev['close'] < prev['open'] and curr['close'] > curr['open'] and curr['close'] > prev['high']:
                if abs(prev['low'] - level) <= offset:
                    return True
        else:
            if prev['close'] > prev['open'] and curr['close'] < curr['open'] and curr['close'] < prev['low']:
                if abs(prev['high'] - level) <= offset:
                    return True
        return False

    def _is_three_bar_reversal(self, candles: pd.DataFrame, idx: int, direction: str, level: float) -> bool:
        """Detect a three-bar reversal pattern (bullish or bearish) near a level."""
        if idx < 2 or idx >= len(candles):
            return False
        c1, c2, c3 = candles.iloc[idx-2], candles.iloc[idx-1], candles.iloc[idx]
        tol = c3['close'] * self.price_tolerance
        atr_val = None
        if len(candles) >= 14:
            from src.utils.indicators import calculate_atr
            atr_series = calculate_atr(candles, period=14)
            if isinstance(atr_series, pd.Series):
                atr_val = float(atr_series.iloc[idx]) if idx < len(atr_series) else float(atr_series.iloc[-1])
        offset = max(tol, (atr_val * 0.2) if atr_val else 0)
        if direction == 'bullish':
            if c1['close'] < c1['open'] and c2['close'] < c2['open'] and c3['close'] > c3['open'] and c3['close'] > c1['open']:
                if abs(c1['low'] - level) <= offset:
                    return True
        else:
            if c1['close'] > c1['open'] and c2['close'] > c2['open'] and c3['close'] < c3['open'] and c3['close'] < c1['open']:
                if abs(c1['high'] - level) <= offset:
                    return True
        return False

    @property
    def required_timeframes(self):
        return [self.primary_timeframe]

    @property
    def lookback_periods(self):
        return {self.primary_timeframe: self.lookback_period}