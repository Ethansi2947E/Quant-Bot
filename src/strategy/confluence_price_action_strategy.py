"""
Confluence Price Action Strategy

This strategy identifies the prevailing trend on a higher timeframe, marks key support/resistance levels, waits for pullbacks on a lower timeframe, and then looks for confluence of price-action signals (pin bars, engulfing bars, inside bars, false breakouts), Fibonacci retracements, and moving-average support/resistance. Risk management enforces fixed fractional sizing and minimum R:R.
"""

import pandas as pd
import numpy as np
from loguru import logger
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple

from src.trading_bot import SignalGenerator
import talib # Added talib import
from config.config import TRADING_CONFIG,get_risk_manager_config
from src.risk_manager import RiskManager
from src.utils.patterns_luxalgo import add_luxalgo_patterns, BULLISH_PATTERNS, BEARISH_PATTERNS, NEUTRAL_PATTERNS, ALL_PATTERNS, filter_patterns_by_bias

# Timeframe-specific profiles for dynamic parameter scaling
TIMEFRAME_PROFILES = {
    "M5": {"pivot_lookback": 140, "pullback_bars": 12, "pattern_bars": 6, "ma_period": 21, "price_tolerance": 0.002, "max_sl_atr_mult": 2.0, "max_sl_pct": 0.01, "pivot_window": 3},
    "M15": {"pivot_lookback": 96, "pullback_bars": 18, "pattern_bars": 6, "ma_period": 34, "price_tolerance": 0.002, "max_sl_atr_mult": 2.0, "max_sl_pct": 0.01, "pivot_window": 3},
    "H1": {"pivot_lookback": 50, "pullback_bars": 6, "pattern_bars": 2, "ma_period": 55, "price_tolerance": 0.002, "max_sl_atr_mult": 2.5, "max_sl_pct": 0.015, "pivot_window": 2},
    "H4": {"pivot_lookback": 30, "pullback_bars": 4, "pattern_bars": 2, "ma_period": 89, "price_tolerance": 0.002, "max_sl_atr_mult": 2.5, "max_sl_pct": 0.015, "pivot_window": 2},
    "D1": {"pivot_lookback": 20, "pullback_bars": 3, "pattern_bars": 1, "ma_period": 144, "price_tolerance": 0.002, "max_sl_atr_mult": 3.0, "max_sl_pct": 0.02, "pivot_window": 2}
}
DEFAULT_PROFILE = {"pivot_lookback": 50, "pullback_bars": 5, "pattern_bars": 3, "ma_period": 21, "price_tolerance": 0.002, "max_sl_atr_mult": 2.0, "max_sl_pct": 0.01, "pivot_window": 3}

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
                 use_fibonacci: bool = True,  # Changed: Enable Fibonacci by default
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
        self.pivot_window = None  # Will be set by timeframe profile

        # Load dynamic profile based on primary timeframe
        self._load_timeframe_profile()

        # State tracking to prevent duplicate/conflicting signals
        self.processed_bars = {}  # {(symbol, timeframe): last_processed_timestamp}
        self.processed_zones = {}  # {(symbol, zone_type, zone_price): last_processed_timestamp}
        
        # Dynamic cooldown based on timeframe - number of bars to wait before re-entering same zone
        self.cooldown_bars = self._get_dynamic_cooldown_bars()
        
        # Zone distance tracking for cooldown reset logic
        self.zone_distance_tracking = {}  # {zone_key: {'last_distance': float, 'max_distance_away': float}}

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
        self.pivot_window = profile.get('pivot_window', DEFAULT_PROFILE['pivot_window'])

        logger.info(
            f"🔄 Timeframe profile loaded for {self.primary_timeframe}: "
            f"pivot_lookback={self.pivot_lookback}, pullback_bars={self.pullback_bars}, "
            f"pattern_bars={self.pattern_bars}, ma_period={self.ma_period}, "
            f"price_tolerance={self.price_tolerance}, max_sl_atr_mult={self.max_sl_atr_mult}, max_sl_pct={self.max_sl_pct}, "
            f"pivot_window={self.pivot_window}"
        )

    # ===== FRACTAL PIVOT DETECTION FUNCTIONS =====
    def _find_fractal_pivots(self, df: pd.DataFrame, window: int = 3) -> Tuple[List[Dict], List[Dict]]:
        """
        Find fractal pivot highs and lows using proper fractal definition.
        A fractal high: current bar's high > N bars to left AND > N bars to right
        A fractal low: current bar's low < N bars to left AND < N bars to right
        
        Args:
            df: DataFrame with OHLC data
            window: Number of bars on each side to check (default 3)
            
        Returns:
            Tuple of (pivot_highs, pivot_lows) as lists of dicts with 'price', 'index', 'timestamp'
        """
        if df is None or len(df) < (2 * window + 1):
            logger.debug(f"[Fractals] Insufficient data for fractal detection: need {2*window+1}, have {len(df) if df is not None else 0}")
            return [], []
            
        pivot_highs = []
        pivot_lows = []
        
        high_values = df['high'].values
        low_values = df['low'].values
        
        # Check each potential pivot point (excluding edges)
        for i in range(window, len(df) - window):
            current_high = high_values[i]
            current_low = low_values[i]
            
            # Check for fractal high: current high > all surrounding highs
            is_fractal_high = True
            for j in range(1, window + 1):
                if current_high <= high_values[i - j] or current_high <= high_values[i + j]:
                    is_fractal_high = False
                    break
                    
            if is_fractal_high:
                pivot_highs.append({
                    'price': current_high,
                    'index': i,
                    'timestamp': df.index[i]
                })
                
            # Check for fractal low: current low < all surrounding lows
            is_fractal_low = True
            for j in range(1, window + 1):
                if current_low >= low_values[i - j] or current_low >= low_values[i + j]:
                    is_fractal_low = False
                    break
                    
            if is_fractal_low:
                pivot_lows.append({
                    'price': current_low,
                    'index': i,
                    'timestamp': df.index[i]
                })
                
        logger.debug(f"[Fractals] Found {len(pivot_highs)} fractal highs and {len(pivot_lows)} fractal lows (window={window})")
        return pivot_highs, pivot_lows

    def _get_swing_points_for_trend(self, df: pd.DataFrame, min_swings: int = 3) -> Tuple[List[float], List[float]]:
        """
        Extract swing highs and lows for trend analysis using fractal pivots.
        
        Args:
            df: DataFrame with OHLC data
            min_swings: Minimum number of swing points needed for trend analysis
            
        Returns:
            Tuple of (swing_highs, swing_lows) as lists of prices
        """
        window = self.pivot_window or 3  # Default to 3 if None
        pivot_highs, pivot_lows = self._find_fractal_pivots(df, window)
        
        # Extract just the prices and sort by recency (most recent last)
        swing_highs = [pivot['price'] for pivot in pivot_highs[-min_swings*2:]]  # Get more than needed
        swing_lows = [pivot['price'] for pivot in pivot_lows[-min_swings*2:]]   # Get more than needed
        
        # Take the most recent significant swings
        swing_highs = swing_highs[-min_swings:] if len(swing_highs) >= min_swings else swing_highs
        swing_lows = swing_lows[-min_swings:] if len(swing_lows) >= min_swings else swing_lows
        
        logger.debug(f"[SwingTrend] Extracted {len(swing_highs)} swing highs and {len(swing_lows)} swing lows for trend analysis")
        return swing_highs, swing_lows

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
        
        current_time = self._to_datetime(datetime.now())  # Standardize from the start
        
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
                    
                    # Check if we should reset cooldown due to price movement
                    should_reset = self._should_reset_zone_cooldown(primary, current_level_price, zone_key)
                    if should_reset:
                        logger.info(f"🔄 Cooldown reset for {zone_type_str} zone {current_level_price:.5f}")
                        # Remove from processed zones to allow re-entry
                        del self.processed_zones[zone_key]
                    else:
                        # Check normal cooldown timing using standardized datetime objects
                        cooldown_seconds = self._convert_bars_to_seconds(self.cooldown_bars)
                        
                        # Convert both timestamps to datetime objects
                        current_time_dt = current_time  # Already standardized
                        last_used_time_dt = self._to_datetime(last_used_time)
                        
                        time_since_use = current_time_dt - last_used_time_dt
                        
                        if time_since_use.total_seconds() < cooldown_seconds:
                            bars_elapsed = int(time_since_use.total_seconds() / self._convert_bars_to_seconds(1))
                            logger.debug(
                                f"Skipping {zone_type_str} zone {current_level_price:.5f} - on cooldown ({bars_elapsed}/{self.cooldown_bars} bars elapsed)"
                            )
                            continue
                # --- Rejection-based (reversal) logic (existing) ---
                logger.debug(
                    f"Checking pullback for {sym} at {level_type_str} level {current_level_price:.5f} (trend: {trend})"
                )
                if self._is_pullback(primary, current_level_price, trend):
                    logger.debug(
                        f"Found pullback to {level_type_str} level {current_level_price:.5f} for {sym}"
                    )
                    
                    # Ensure we have at least 2 candles (one to check pattern on, one to enter on)
                    if len(primary) < 2:
                        logger.debug(f"{sym}: Insufficient candles for pattern timing analysis")
                        continue
                    
                    # Check for patterns on the PREVIOUS (completed) candle
                    pattern_candle_idx = len(primary) - 2  # Previous completed candle
                    current_candle_idx = len(primary) - 1   # Current candle (where we might enter)
                    
                    pattern_candle = primary.iloc[pattern_candle_idx]
                    current_candle = primary.iloc[current_candle_idx]
                    
                    # Add tracking for processed patterns to ensure we only trade once per pattern
                    pattern_key = (sym, level_type_str, round(current_level_price, 5), str(pattern_candle.name))
                    if pattern_key in self.processed_zones:
                        logger.debug(f"Pattern at {pattern_candle.name} already processed for {sym}")
                        continue

                    # --- Updated Pattern Detection & Scoring (on PREVIOUS candle) ---
                    # Priority order: 1) LuxAlgo patterns (most reliable), 2) Custom rejection patterns (fallback)
                    detected_pattern_name = None
                    pattern_score = 0.0
                    patterns_to_evaluate = (
                        BULLISH_PATTERNS if trend == 'bullish' else BEARISH_PATTERNS
                    )
                    patterns_to_evaluate += NEUTRAL_PATTERNS  # Always consider neutral for confluence

                    volume_score_for_pattern, _ = self._analyze_volume_quality(
                        primary, pattern_candle_idx, trend
                    )  # Analyze volume for the PREVIOUS candle
                    volume_confirmed_for_pattern = volume_score_for_pattern >= 0.3

                    # PRIORITY 1: Check for LuxAlgo patterns first (most reliable)
                    for p_col in patterns_to_evaluate:
                        if p_col in primary.columns:
                            pattern_series = primary[p_col]
                            if pattern_series.iloc[pattern_candle_idx]:  # Check PREVIOUS candle
                                detected_pattern_name = f"{p_col.replace('_', ' ').title()} (LuxAlgo)"
                                # Assign scores based on pattern type reliability
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
                                    pattern_score = 0.6  # Default for other patterns
                                logger.info(f"✅ {sym}: Found LuxAlgo pattern '{detected_pattern_name}' on previous candle (score: {pattern_score:.2f})")
                                break  # Use first LuxAlgo pattern found (highest priority)

                    # PRIORITY 2: If no LuxAlgo pattern, check custom rejection patterns (fallback)
                    if not detected_pattern_name:
                        logger.debug(f"No LuxAlgo patterns found, checking custom rejection patterns...")
                        
                        if trend == 'bullish':
                            # Bullish rejection: candle wicked above level but closed below (bearish rejection for bullish entry)
                            if (
                                pattern_candle['open'] < current_level_price
                                and pattern_candle['high'] > current_level_price
                                and pattern_candle['close'] < current_level_price
                                and volume_confirmed_for_pattern
                            ):
                                detected_pattern_name = 'Level Rejection (Bullish Setup)'
                                pattern_score = 0.75  # Lower than LuxAlgo patterns
                        else:  # Bearish trend
                            # Bearish rejection: candle wicked below level but closed above (bullish rejection for bearish entry)
                            if (
                                pattern_candle['open'] > current_level_price
                                and pattern_candle['low'] < current_level_price
                                and pattern_candle['close'] > current_level_price
                                and volume_confirmed_for_pattern
                            ):
                                detected_pattern_name = 'Level Rejection (Bearish Setup)'
                                pattern_score = 0.75  # Lower than LuxAlgo patterns

                        if detected_pattern_name:
                            logger.info(f"✅ {sym}: Found custom pattern '{detected_pattern_name}' on previous candle (score: {pattern_score:.2f})")

                    if not detected_pattern_name:
                        logger.debug(
                            f"No valid patterns detected on previous candle (idx={pattern_candle_idx}) for {sym} at level {current_level_price:.5f}"
                        )
                        continue
                    
                    # Log pattern found on previous candle
                    logger.info(
                        f"{sym}: Found {detected_pattern_name} pattern on PREVIOUS candle ({pattern_candle.name}) with score {pattern_score:.2f}. Considering entry on CURRENT candle ({current_candle.name})"
                    )

                    # Confluence Checks (Fibonacci, MA) - still use current data
                    fib_ok = self._check_fibonacci(primary, current_level_price, self.use_fibonacci)
                    ma_ok = self._check_ma(primary, current_level_price)
                    fib_details = {}  # Populate if fib_ok
                    ma_details = {}  # Populate if ma_ok
                    
                    # Log confluence results for debugging
                    logger.info(f"[Confluence] {sym}: Fib check: {'✅ PASS' if fib_ok else '❌ FAIL'}, MA check: {'✅ PASS' if ma_ok else '❌ FAIL'}")

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
                    
                    # Log detailed confluence scoring breakdown
                    logger.info(f"[Confluence] {sym} Scoring Breakdown:")
                    logger.info(f"  • HTF Trend: {htf_trend_score:.1f}")
                    logger.info(f"  • S/R Level: {htf_sr_score:.1f} (strength: {level_data['strength']})")
                    logger.info(f"  • Pattern: {pattern_score:.1f} ({detected_pattern_name}) - on PREVIOUS candle")
                    logger.info(f"  • Fibonacci: {fib_score_bonus:.1f} ({'✅' if fib_ok else '❌'})")
                    logger.info(f"  • MA Support: {ma_score_bonus:.1f} ({'✅' if ma_ok else '❌'})")
                    logger.info(f"  • Volume: {volume_score_confluence:.2f}")
                    logger.info(f"  • Level Strength: {level_strength_bonus:.2f}")
                    logger.info(f"  • TOTAL: {confluence_total_score:.2f} / 4.0")
                    
                    min_score_threshold = 2.2

                    # Max possible score: 1(trend) + 1(sr) + 1(pattern) + 0.3(fib) + 0.3(ma) + 0.2(vol) + 0.2(strength) = 4.0
                    normalization_divisor = 4.0 # Changed from 3.0

                    if confluence_total_score < min_score_threshold:
                        logger.info(
                            f"[ConfluenceScoring] Signal for {sym} at {current_level_price:.5f} rejected: total score {confluence_total_score:.2f} < {min_score_threshold}"
                        )
                        continue
                    
                    # --- Signal Assembly: Use CURRENT candle for entry ---
                    entry = current_candle['open']  # Enter on current candle's open price
                    direction_str = "buy" if trend == 'bullish' else "sell"

                    # Proper stop-loss placement with ATR buffer
                    atr_buffer_multiplier = 1.5  # Configurable ATR buffer multiplier
                    atr_val = (
                        atr_series[-1]
                        if atr_series is not None and len(atr_series) > 0 and pd.notna(atr_series[-1])
                        else current_level_price * 0.002  # Fallback: 0.2% of price
                    )
                    
                    # Clean ATR buffer calculation
                    atr_buffer = atr_buffer_multiplier * atr_val if pd.notna(atr_val) and atr_val > 0 else entry * 0.002

                    if direction_str == 'buy':
                        # Stop below pattern candle's low with ATR buffer
                        stop = pattern_candle['low'] - atr_buffer
                        # Ensure stop is below entry with minimum distance
                        stop = min(stop, entry - (TICK_SIZE * 5))  # Must be at least 5 ticks below entry
                        reward_calc = entry - stop 
                        tp = entry + reward_calc * self.min_risk_reward
                    else:  # sell
                        # Stop above pattern candle's high with ATR buffer
                        stop = pattern_candle['high'] + atr_buffer
                        # Ensure stop is above entry with minimum distance
                        stop = max(stop, entry + (TICK_SIZE * 5))  # Must be at least 5 ticks above entry
                        reward_calc = stop - entry
                        tp = entry - reward_calc * self.min_risk_reward

                    # Cap excessive stop distances (keep existing logic but cleaner)
                    max_sl_dist_val = None
                    if pd.notna(atr_val) and atr_val > 0 and self.max_sl_atr_mult is not None:
                        max_sl_dist_val = max(atr_val * self.max_sl_atr_mult, TICK_SIZE * 10)
                    
                    if self.max_sl_pct is not None:
                        max_sl_dist_pct_val = entry * self.max_sl_pct
                        if max_sl_dist_val is not None:
                            max_sl_dist_val = min(max_sl_dist_val, max_sl_dist_pct_val)
                        else:
                            max_sl_dist_val = max_sl_dist_pct_val
                    
                    # Apply maximum distance cap if needed
                    if max_sl_dist_val and abs(entry - stop) > max_sl_dist_val:
                        logger.debug(f"Capping SL distance from {abs(entry - stop):.5f} to {max_sl_dist_val:.5f}")
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

                    signal_quality_norm = min(
                        max(confluence_total_score / normalization_divisor, 0.0), 1.0 
                    )
                    concise_analysis = (
                        f"📝 Analysis ({sym} - {self.primary_timeframe}): {direction_str.capitalize()} signal based on {detected_pattern_name} pattern on previous candle at {level_type_str} {current_level_price:.5f}. Entering on current candle open."
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
                        "pattern_details": {"pattern_candle": str(pattern_candle.name), "entry_candle": str(current_candle.name)},
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
                        "signal_bar_index": current_candle_idx,  # Entry on current candle
                        "pattern_bar_index": pattern_candle_idx,  # Pattern was on previous candle
                        "signal_timestamp": str(current_candle.name),
                        "pattern_timestamp": str(pattern_candle.name),
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
                    
                    # Mark this pattern as processed so we don't trade it again
                    self.processed_zones[pattern_key] = current_time
                    
                    logger.info(f"✅ {sym}: Pattern-based trade signal generated - Pattern on {pattern_candle.name}, Entry on {current_candle.name}")
                    break  # Process one signal per zone
            signals.extend(symbol_signals)
        
        # Cleanup old zone records using dynamic cooldown (2x the cooldown period)
        cleanup_seconds = self._convert_bars_to_seconds(self.cooldown_bars * 2)
        cleanup_delta = timedelta(seconds=cleanup_seconds)
        cleanup_time = current_time - cleanup_delta
        
        old_keys = [k for k, v in self.processed_zones.items() if isinstance(v, (datetime, pd.Timestamp)) and pd.to_datetime(v) < cleanup_time]
        for k_del in old_keys:  # Corrected loop variable
            del self.processed_zones[k_del]
            # Also cleanup zone distance tracking
            if k_del in self.zone_distance_tracking:
                del self.zone_distance_tracking[k_del]
        if old_keys:
            logger.debug(f"Cleaned up {len(old_keys)} old zone records (older than {self.cooldown_bars * 2} bars)")
        return signals

    # -- Trend and level detection --
    def _determine_trend(self, df: pd.DataFrame) -> str:
        """Return 'bullish', 'bearish' or 'neutral' based on MA and fractal-based price structure.
        Uses proper fractal pivots instead of talib.MAX/MIN for more reliable trend identification.
        """
        window = self.pivot_window or 3  # Default to 3 if None
        if df is None or 'close' not in df.columns or len(df) < self.ma_period + (2 * window + 1):
            logger.debug("[Trend] Insufficient data for trend determination.")
            return 'neutral'
            
        # MA trend analysis (unchanged)
        close = np.asarray(df['close'].values, dtype=np.float64)
        ma = talib.SMA(close, timeperiod=self.ma_period)[-1]
        last = close[-1]
        ma_trend = 'bullish' if last > ma else 'bearish' if last < ma else 'neutral'
        
        # Fractal-based price structure trend analysis
        swing_highs, swing_lows = self._get_swing_points_for_trend(df, min_swings=3)
        
        price_trend = 'neutral'
        if len(swing_highs) >= 3 and len(swing_lows) >= 3:
            # Check for higher highs and higher lows (bullish trend)
            hh_condition = swing_highs[-1] > swing_highs[-2] > swing_highs[-3]
            hl_condition = swing_lows[-1] > swing_lows[-2] > swing_lows[-3]
            
            # Check for lower highs and lower lows (bearish trend)
            lh_condition = swing_highs[-1] < swing_highs[-2] < swing_highs[-3]
            ll_condition = swing_lows[-1] < swing_lows[-2] < swing_lows[-3]
            
            if hh_condition and hl_condition:
                price_trend = 'bullish'
                logger.debug(f"[Trend] Fractal structure: HH + HL = bullish trend")
            elif lh_condition and ll_condition:
                price_trend = 'bearish'
                logger.debug(f"[Trend] Fractal structure: LH + LL = bearish trend")
            else:
                logger.debug(f"[Trend] Fractal structure: mixed signals = neutral")
        else:
            logger.debug(f"[Trend] Insufficient fractal swings for structure analysis: highs={len(swing_highs)}, lows={len(swing_lows)}")
        
        logger.debug(f"[Trend] MA trend: {ma_trend}, Fractal price structure: {price_trend}")
        
        # Combined trend decision
        if ma_trend == price_trend and ma_trend != 'neutral':
            logger.info(f"[Trend] Confirmed {ma_trend} trend (MA + fractal structure)")
            return ma_trend
        elif ma_trend != 'neutral' or price_trend != 'neutral':
            logger.info(f"[Trend] Loosened: Accepting {ma_trend if ma_trend != 'neutral' else price_trend} trend (one clear)")
            return ma_trend if ma_trend != 'neutral' else price_trend
            
        logger.info("[Trend] No clear trend (both neutral)")
        return 'neutral'

    def _find_key_levels(self, df: pd.DataFrame) -> tuple:
        """Return (supports, resistances) using fractal pivots for more reliable S/R levels."""
        window = self.pivot_window or 3  # Default to 3 if None
        if df is None or len(df) < (2 * window + 1):
            logger.debug("[KeyLevels] Insufficient data for fractal pivot detection")
            return [], []
            
        # Limit to profile pivot_lookback bars
        subset = df.copy()
        if len(df) > self.pivot_lookback:
            subset = df.iloc[-self.pivot_lookback:]
            
        # Find fractal pivots
        pivot_highs, pivot_lows = self._find_fractal_pivots(subset, window)
        
        # Extract price levels
        resistance_prices = [pivot['price'] for pivot in pivot_highs]
        support_prices = [pivot['price'] for pivot in pivot_lows]
        
        logger.debug(f"[KeyLevels] Raw fractal pivots: {len(support_prices)} lows, {len(resistance_prices)} highs (before clustering)")
        
        # Calculate dynamic clustering tolerance
        clustering_tol = max(5 * TICK_SIZE, min(subset['close'].mean() * self.price_tolerance, 15 * TICK_SIZE))
        atr_val = None
        if len(subset) >= 14:
            try:
                high = np.asarray(subset['high'].values, dtype=np.float64)
                low = np.asarray(subset['low'].values, dtype=np.float64)
                close = np.asarray(subset['close'].values, dtype=np.float64)
                atr_series = talib.ATR(high, low, close, timeperiod=14)
                if len(atr_series) > 0 and pd.notna(atr_series[-1]):
                    atr_val = float(atr_series[-1])
                    clustering_tol = max(clustering_tol, atr_val * 0.1)
            except Exception as e:
                logger.debug(f"[KeyLevels] ATR calculation failed: {e}")
        
        # Cluster levels and calculate strength
        support_levels = self._cluster_levels_with_strength(sorted(support_prices), clustering_tol, subset, is_support=True)
        resistance_levels = self._cluster_levels_with_strength(sorted(resistance_prices), clustering_tol, subset, is_support=False)
        
        logger.debug(f"[KeyLevels] Support clusters: {len(support_levels)}, Resistance clusters: {len(resistance_levels)} (before filtering)")
        
        # Filter by minimum touches (require at least 2 touches for significance)
        last_50 = df.iloc[-min(50, len(df)):]
        min_touches = 2
        support_levels = [lvl for lvl in support_levels if self._count_level_touches(last_50, lvl['level'], clustering_tol, is_support=True) >= min_touches]
        resistance_levels = [lvl for lvl in resistance_levels if self._count_level_touches(last_50, lvl['level'], clustering_tol, is_support=False) >= min_touches]
        
        logger.debug(f"[KeyLevels] Final levels: {len(support_levels)} supports, {len(resistance_levels)} resistances (after {min_touches}+ touch filter)")
        
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

    # -- Confluence checks --
    def _find_recent_swing(self, df: pd.DataFrame, lookback: int = 50) -> tuple:
        """Find the most recent significant swing high and low using fractal pivot logic."""
        window = self.pivot_window or 3  # Default to 3 if None
        if df is None or len(df) < max(lookback, 2 * window + 1):
            logger.debug(f"[Swing] Insufficient data for swing detection")
            return None, None
        
        # Use recent data for swing detection
        recent = df.iloc[-lookback:] if len(df) > lookback else df.copy()
        
        # Find fractal pivots
        pivot_highs, pivot_lows = self._find_fractal_pivots(recent, window)
        
        # Get the most recent significant swing points
        swing_high = None
        swing_low = None
        
        if pivot_highs:
            # Find the highest fractal high (most significant resistance)
            swing_high = max(pivot['price'] for pivot in pivot_highs)
            
        if pivot_lows:
            # Find the lowest fractal low (most significant support)
            swing_low = min(pivot['price'] for pivot in pivot_lows)
        
        # Fallback to simple max/min if no fractals found
        if swing_high is None or swing_low is None:
            logger.debug(f"[Swing] No fractal pivots found, using fallback max/min")
            if swing_high is None:
                swing_high = recent['high'].max()
            if swing_low is None:
                swing_low = recent['low'].min()
        
        logger.debug(f"[Swing] Found fractal swing high: {swing_high:.5f}, swing low: {swing_low:.5f}")
        return swing_high, swing_low

    def _check_fibonacci(self, df: pd.DataFrame, level: float, use_fibonacci: bool = True) -> bool:
        """Check if `level` is near a standard Fibonacci retracement of the most recent swing.
        Improved: Uses proper swing detection and more reasonable tolerance zones.
        """
        if not use_fibonacci:
            logger.debug(f"[Fib] use_fibonacci=False, skipping Fib confluence check for level {level:.5f}")
            return False
            
        swing_high, swing_low = self._find_recent_swing(df, lookback=50)
        if swing_high is None or swing_low is None:
            logger.debug(f"[Fib] No valid swing points for Fib check (level={level:.5f})")
            return False
            
        # Ensure we have a meaningful swing range
        swing_range = swing_high - swing_low
        if swing_range < (level * 0.001):  # Range must be at least 0.1% of current price
            logger.debug(f"[Fib] Swing range too small: {swing_range:.5f} for level {level:.5f}")
            return False
            
        # Calculate ATR for dynamic tolerance
        atr_val = None
        if len(df) >= 14:
            try:
                high = np.asarray(df['high'].values, dtype=np.float64)
                low = np.asarray(df['low'].values, dtype=np.float64)
                close = np.asarray(df['close'].values, dtype=np.float64)
                atr_series = talib.ATR(high, low, close, timeperiod=14)
                if len(atr_series) > 0 and pd.notna(atr_series[-1]):
                    atr_val = float(atr_series[-1])
            except Exception as e:
                logger.debug(f"[Fib] ATR calculation failed: {e}")
        
        # Dynamic tolerance: max of percentage-based and ATR-based
        percentage_tolerance = level * 0.003  # 0.3% of price (was 0.15%)
        atr_tolerance = (atr_val * 0.75) if atr_val else 0  # 0.75x ATR (was 0.5x)
        tolerance = max(percentage_tolerance, atr_tolerance)
        
        # Check standard Fibonacci levels
        fib_matches = []
        for fib_ratio in self.fib_levels:
            fib_price = swing_low + (swing_high - swing_low) * fib_ratio
            distance = abs(level - fib_price)
            if distance <= tolerance:
                fib_matches.append((fib_ratio, fib_price, distance))
                logger.info(f"[Fib] ✅ Level {level:.5f} matches Fib {fib_ratio:.1%} ({fib_price:.5f}) - distance: {distance:.5f}, tolerance: {tolerance:.5f}")
        
        if fib_matches:
            # Log the best match
            best_match = min(fib_matches, key=lambda x: x[2])
            logger.info(f"[Fib] Best Fib match: {best_match[0]:.1%} level at {best_match[1]:.5f}")
            return True
        else:
            logger.debug(f"[Fib] No Fib matches for level {level:.5f} within tolerance {tolerance:.5f}")
            logger.debug(f"[Fib] Swing: {swing_low:.5f} to {swing_high:.5f}, Checked levels: {[swing_low + (swing_high - swing_low) * f for f in self.fib_levels]}")
        return False

    def _check_ma(self, df: pd.DataFrame, level: float) -> bool:
        """Return True if `level` is near the moving-average support/resistance (within a flexible zone).
        Improved: More flexible tolerance and removed overly strict slope requirements.
        """
        if df is None or len(df) < self.ma_period:
            logger.debug(f"[MA] Insufficient data: have {len(df) if df is not None else 0}, need {self.ma_period}")
            return False
            
        try:
            close = np.asarray(df['close'].values, dtype=np.float64)
            ma_series = talib.SMA(close, timeperiod=self.ma_period)
            
            if len(ma_series) == 0 or pd.isna(ma_series[-1]):
                logger.debug(f"[MA] MA calculation failed or returned NaN")
                return False
                
            current_ma = float(ma_series[-1])
            
            # Calculate ATR for dynamic tolerance
            atr_val = None
            if len(df) >= 14:
                try:
                    high = np.asarray(df['high'].values, dtype=np.float64)
                    low = np.asarray(df['low'].values, dtype=np.float64)
                    atr_series = talib.ATR(high, low, close, timeperiod=14)
                    if len(atr_series) > 0 and pd.notna(atr_series[-1]):
                        atr_val = float(atr_series[-1])
                except Exception as e:
                    logger.debug(f"[MA] ATR calculation failed: {e}")
            
            # More flexible tolerance zone: max of percentage-based and ATR-based
            percentage_tolerance = current_ma * 0.004  # 0.4% of MA price (was 0.2%)
            atr_tolerance = (atr_val * 0.8) if atr_val else 0  # 0.8x ATR (was 0.5x)
            tolerance = max(percentage_tolerance, atr_tolerance)
            
            distance = abs(level - current_ma)
            
            # Check if level is within tolerance of MA
            if distance <= tolerance:
                logger.info(f"[MA] ✅ Level {level:.5f} near MA {current_ma:.5f} - distance: {distance:.5f}, tolerance: {tolerance:.5f}")
                return True
            else:
                logger.debug(f"[MA] Level {level:.5f} too far from MA {current_ma:.5f} - distance: {distance:.5f}, tolerance: {tolerance:.5f}")
            return False
                
        except Exception as e:
            logger.warning(f"[MA] Error in MA confluence check: {e}")
        return False  

    def _analyze_volume_quality(self, df: pd.DataFrame, idx: int, direction: str) -> tuple:
        """Analyze volume quality for a given candle index and direction. Returns (score, details).
        Improved: More lenient volume requirements and better fallbacks.
        """
        if df is None or len(df) == 0 or idx >= len(df) or idx < -len(df):
            return 0.0, {}
            
        candle = df.iloc[idx]
        vol_col = 'volume' if 'volume' in df.columns else 'tick_volume'
        
        # Get volume data with fallbacks
        if vol_col not in candle:
            logger.debug(f"[Volume] No volume data available, using neutral score")
            return 0.1, {'reason': 'no_volume_data'}  # Give small positive score when no volume data
            
        # Use rolling median as threshold (robust to outliers)
        lookback = min(20, len(df))
        if lookback < 5:  # If very little data, be more lenient
            threshold = 1.0
            logger.debug(f"[Volume] Limited data ({lookback} bars), using default threshold")
        else:
            vol_series = df[vol_col].iloc[-lookback:]
            threshold = vol_series.median() if not vol_series.empty else 1.0
            
        tick_volume = float(candle[vol_col]) if pd.notna(candle[vol_col]) else threshold * 0.8
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
        
        logger.debug(f"[Volume] idx={idx}, direction={direction}, vol_ratio={volume_ratio:.2f}, range={total_range:.5f}")
        
        # Basic validation
        if total_range == 0 or total_range < 0.00001:
            return 0.0, details
            
        # More lenient volume threshold - reduced from 0.15 to 0.1 (10% above median)
        if volume_ratio < 0.1:
            logger.debug(f"[Volume] Low volume ratio {volume_ratio:.2f}, but allowing small positive score")
            return 0.05, details  # Still give small score instead of 0
        
        # Calculate wick ratios for pattern analysis
        if direction == 'bullish':
            upper_wick = candle['high'] - max(candle['close'], candle['open'])
            lower_wick = min(candle['close'], candle['open']) - candle['low']
        else:  # bearish
            upper_wick = candle['high'] - max(candle['close'], candle['open'])
            lower_wick = min(candle['close'], candle['open']) - candle['low']
            
        upper_wick_ratio = upper_wick / total_range if total_range > 0 else 0
        lower_wick_ratio = lower_wick / total_range if total_range > 0 else 0
        body_ratio = body / total_range if total_range > 0 else 0
        
        details.update({
            'upper_wick_ratio': upper_wick_ratio,
            'lower_wick_ratio': lower_wick_ratio,
            'body_ratio': body_ratio
        })
        
        # Simplified scoring - focus on volume and basic candle structure
        base_score = 0.0
        
        if direction == 'bullish':
            # For bullish signals, prefer lower wicks (buying support) and good body
            if lower_wick_ratio > upper_wick_ratio and body_ratio > 0.3:
                base_score = 0.8
            elif body_ratio > 0.5:
                base_score = 0.6
            else:
                base_score = 0.4
        else:  # bearish
            # For bearish signals, prefer upper wicks (selling resistance) and good body
            if upper_wick_ratio > lower_wick_ratio and body_ratio > 0.3:
                base_score = 0.8
            elif body_ratio > 0.5:
                base_score = 0.6
            else:
                base_score = 0.4
        
        # Volume boost - higher volume increases score
        if volume_ratio >= 1.5:
            volume_multiplier = 1.2  # 20% boost for high volume
        elif volume_ratio >= 1.0:
            volume_multiplier = 1.1  # 10% boost for above-average volume
        elif volume_ratio >= 0.5:
            volume_multiplier = 1.0  # No penalty for moderate volume
        else:
            volume_multiplier = 0.8  # Small penalty for low volume
            
        final_score = min(base_score * volume_multiplier, 1.0)
        
        logger.debug(f"[Volume] Final score: {final_score:.2f} (base: {base_score:.2f}, vol_mult: {volume_multiplier:.2f})")
        
        return final_score, details

    @property
    def required_timeframes(self) -> list:
        """List of timeframes required by this strategy (for orchestrator data fetching)."""
        # Remove duplicates if higher and primary are the same
        return list({self.higher_timeframe, self.primary_timeframe})

    def _get_dynamic_cooldown_bars(self) -> int:
        """Get dynamic cooldown period in number of bars based on primary timeframe."""
        timeframe_cooldowns = {
            "M1": 30,   # 30 minutes
            "M5": 12,   # 1 hour  
            "M15": 8,   # 2 hours
            "M30": 6,   # 3 hours
            "H1": 4,    # 4 hours
            "H4": 3,    # 12 hours
            "D1": 2,    # 2 days
        }
        cooldown_bars = timeframe_cooldowns.get(self.primary_timeframe, 6)  # Default to 6 bars
        cooldown_duration = self._convert_bars_to_seconds(cooldown_bars)
        cooldown_hours = cooldown_duration / 3600
        logger.info(f"📊 Dynamic cooldown: {cooldown_bars} bars (~{cooldown_hours:.1f}h) for {self.primary_timeframe} timeframe")
        return cooldown_bars
    
    def _convert_bars_to_seconds(self, bars: int) -> int:
        """Convert number of bars to approximate seconds based on timeframe."""
        timeframe_seconds = {
            "M1": 60,
            "M5": 300,
            "M15": 900,
            "M30": 1800,
            "H1": 3600,
            "H4": 14400,
            "D1": 86400,
        }
        base_seconds = timeframe_seconds.get(self.primary_timeframe, 3600)  # Default to 1 hour
        return bars * base_seconds
    
    def _should_reset_zone_cooldown(self, df: pd.DataFrame, level: float, zone_key: tuple) -> bool:
        """Check if price moved significantly away from zone and should reset cooldown."""
        if len(df) < 5:  # Need minimum data
            return False
            
        current_price = df['close'].iloc[-1]
        distance_from_level = abs(current_price - level)
        distance_ratio = distance_from_level / level if level > 0 else 0
        
        # Track zone distances
        if zone_key not in self.zone_distance_tracking:
            self.zone_distance_tracking[zone_key] = {
                'last_distance': distance_ratio,
                'max_distance_away': distance_ratio
            }
            return False
        
        tracking = self.zone_distance_tracking[zone_key]
        
        # Update tracking
        tracking['last_distance'] = distance_ratio
        if distance_ratio > tracking['max_distance_away']:
            tracking['max_distance_away'] = distance_ratio
        
        # Reset cooldown if price moved significantly away (>0.5% for most instruments) and came back
        significant_distance_threshold = 0.005  # 0.5%
        close_distance_threshold = 0.001       # 0.1%
        
        # If price was far away and is now close again, reset cooldown
        if (tracking['max_distance_away'] > significant_distance_threshold and 
            distance_ratio < close_distance_threshold):
            logger.info(f"🔄 Resetting cooldown for zone {level:.5f} - price moved away {tracking['max_distance_away']:.3%} and returned to {distance_ratio:.3%}")
            # Reset tracking
            tracking['max_distance_away'] = distance_ratio
            return True
            
        return False

    def _to_datetime(self, timestamp) -> datetime:
        """Safely convert various timestamp formats to datetime object."""
        if isinstance(timestamp, datetime):
            return timestamp
        elif isinstance(timestamp, pd.Timestamp):
            return timestamp.to_pydatetime()
        elif isinstance(timestamp, (int, float)):
            try:
                # Handle both seconds and milliseconds timestamps
                if timestamp > 1e10:  # Likely milliseconds
                    return datetime.fromtimestamp(timestamp / 1000)
                else:  # Likely seconds
                    return datetime.fromtimestamp(timestamp)
            except (ValueError, OSError) as e:
                logger.warning(f"Failed to convert timestamp {timestamp}: {e}")
                return datetime.now()
        elif isinstance(timestamp, str):
            try:
                return pd.to_datetime(timestamp).to_pydatetime()
            except Exception as e:
                logger.warning(f"Failed to parse timestamp string {timestamp}: {e}")
                return datetime.now()
        else:
            logger.warning(f"Unknown timestamp type {type(timestamp)}: {timestamp}")
            return datetime.now()
