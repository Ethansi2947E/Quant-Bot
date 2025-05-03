"""
Confluence Price Action Strategy

This strategy identifies the prevailing trend on a higher timeframe, marks key support/resistance levels, waits for pullbacks on a lower timeframe, and then looks for confluence of price-action signals (pin bars, engulfing bars, inside bars, false breakouts), Fibonacci retracements, and moving-average support/resistance. Risk management enforces fixed fractional sizing and minimum R:R.
"""

import pandas as pd
import numpy as np
from loguru import logger
from datetime import datetime
from typing import Optional

from src.trading_bot import SignalGenerator
from src.utils.indicators import calculate_atr, calculate_adx
from config.config import TRADING_CONFIG,get_risk_manager_config
from src.risk_manager import RiskManager

# Timeframe-specific profiles for dynamic parameter scaling
TIMEFRAME_PROFILES = {
    "M5": {"pivot_lookback": 140, "pullback_bars": 12, "pattern_bars": 6, "ma_period": 21, "price_tolerance": 0.002, "max_sl_atr_mult": 2.0, "max_sl_pct": 0.01},
    "M15": {"pivot_lookback": 96, "pullback_bars": 6, "pattern_bars": 6, "ma_period": 34, "price_tolerance": 0.002, "max_sl_atr_mult": 2.0, "max_sl_pct": 0.01},
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
                 **kwargs):
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
        self.required_timeframes = [higher_timeframe, primary_timeframe]

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

        self.max_sl_atr_mult = None
        self.max_sl_pct = None

        # Load dynamic profile based on primary timeframe
        self._load_timeframe_profile()

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
        
        # Process each symbol's data
        for sym, frames in market_data.items():
            higher = frames.get(self.higher_timeframe)
            primary = frames.get(self.primary_timeframe)
            if not isinstance(higher, pd.DataFrame) or not isinstance(primary, pd.DataFrame):
                logger.debug(f"Missing data for {sym} - higher: {type(higher)}, primary: {type(primary)}")
                continue
            
            # More detailed information about the data received
            logger.debug(f"Analyzing {sym} - Higher TF: {self.higher_timeframe} ({len(higher)} bars), Primary TF: {self.primary_timeframe} ({len(primary)} bars)")
            
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
            
            for level in levels:
                # --- Rejection-based (reversal) logic (existing) ---
                logger.debug(f"Checking pullback for {sym} at {level_type} level {level['level']:.5f} (trend: {trend})")
                if self._is_pullback(primary, level['level'], trend):
                    logger.debug(f"Found pullback to {level_type} level {level['level']:.5f} for {sym}")
                    # Only check the latest candle for patterns
                    idx = len(primary) - 1
                    candle = primary.iloc[idx]
                    pattern = None
                    pattern_details = {}
                    logger.debug(f"Checking for patterns at idx={idx} ({candle.name}) for {sym} at level {level['level']:.5f} ({level_type})")
                    
                    if self._is_pin_bar(candle, level['level'], trend):
                        pattern = 'Pin Bar'
                        body = abs(candle['close'] - candle['open'])
                        total = candle['high'] - candle['low']
                        body_to_range_ratio = body / total if total > 0 else 0
                        pattern_details = {
                            'body_size': body,
                            'wick_size': total - body,
                            'body_to_range_ratio': body_to_range_ratio,
                            'price_to_level_distance': abs(candle['low' if trend == 'bullish' else 'high'] - level['level'])
                        }
                        logger.debug(f"Pin Bar detected at idx={idx} for {sym}")
                        
                    elif self._is_engulfing(primary, idx, trend):
                        pattern = 'Engulfing'
                        prev = primary.iloc[idx-1]
                        pattern_details = {
                            'current_candle_range': candle['high'] - candle['low'],
                            'previous_candle_range': prev['high'] - prev['low'],
                            'engulfing_ratio': (candle['high'] - candle['low']) / (prev['high'] - prev['low']) if (prev['high'] - prev['low']) > 0 else 0
                        }
                        logger.debug(f"Engulfing detected at idx={idx} for {sym}")
                        
                    elif self._is_inside_bar(primary, idx):
                        pattern = 'Inside Bar'
                        mother = primary.iloc[idx-1]
                        pattern_details = {
                            'mother_candle_range': mother['high'] - mother['low'],
                            'child_candle_range': candle['high'] - candle['low'],
                            'containment_ratio': (candle['high'] - candle['low']) / (mother['high'] - mother['low']) if (mother['high'] - mother['low']) > 0 else 0
                        }
                        logger.debug(f"Inside Bar detected at idx={idx} for {sym}")
                        
                    elif self._is_hammer(candle, level['level']):
                        pattern = 'Hammer'
                        body = abs(candle['close'] - candle['open'])
                        total = candle['high'] - candle['low']
                        lower_wick = min(candle['open'], candle['close']) - candle['low']
                        pattern_details = {
                            'body_size': body,
                            'lower_wick': lower_wick,
                            'body_to_range_ratio': body / total if total > 0 else 0
                        }
                        logger.debug(f"Hammer detected at idx={idx} for {sym}")
                    elif self._is_shooting_star(candle, level['level']):
                        pattern = 'Shooting Star'
                        body = abs(candle['close'] - candle['open'])
                        total = candle['high'] - candle['low']
                        upper_wick = candle['high'] - max(candle['open'], candle['close'])
                        pattern_details = {
                            'body_size': body,
                            'upper_wick': upper_wick,
                            'body_to_range_ratio': body / total if total > 0 else 0
                        }
                        logger.debug(f"Shooting Star detected at idx={idx} for {sym}")
                    elif self._is_morning_star(primary, idx, level['level']):
                        pattern = 'Morning Star'
                        c1, c2, c3 = primary.iloc[idx-2], primary.iloc[idx-1], primary.iloc[idx]
                        pattern_details = {
                            'c1_body': abs(c1['close'] - c1['open']),
                            'c2_body': abs(c2['close'] - c2['open']),
                            'c3_body': abs(c3['close'] - c3['open'])
                        }
                        logger.debug(f"Morning Star detected at idx={idx} for {sym}")
                    elif self._is_evening_star(primary, idx, level['level']):
                        pattern = 'Evening Star'
                        c1, c2, c3 = primary.iloc[idx-2], primary.iloc[idx-1], primary.iloc[idx]
                        pattern_details = {
                            'c1_body': abs(c1['close'] - c1['open']),
                            'c2_body': abs(c2['close'] - c2['open']),
                            'c3_body': abs(c3['close'] - c3['open'])
                        }
                        logger.debug(f"Evening Star detected at idx={idx} for {sym}")
                    elif self._is_false_breakout(primary, idx, level['level'], trend):
                        pattern = 'False Breakout'
                        prev = primary.iloc[idx-1]
                        breakout_size = abs(prev['close'] - level['level'])
                        reversal_size = abs(candle['close'] - level['level'])
                        wick = (candle['close'] - candle['low']) if trend == 'bullish' else (candle['high'] - candle['close'])
                        vol_col = 'volume' if 'volume' in primary.columns else 'tick_volume'
                        avg_vol = primary[vol_col].rolling(window=20).mean().iloc[idx]
                        pattern_details = {
                            'breakout_size': breakout_size,
                            'reversal_size': reversal_size,
                            'wick': wick,
                            'volume': candle[vol_col],
                            'avg_volume': avg_vol,
                            'wick_to_body': wick / (abs(candle['close'] - candle['open']) + 1e-6)
                        }
                        logger.debug(f"False Breakout detected at idx={idx} for {sym}")
                    
                    if not pattern:
                        logger.debug(f"No valid pattern detected at idx={idx} for {sym} at level {level['level']:.5f}")
                        continue
                    
                    logger.debug(f"{sym}: Found {pattern} pattern at {candle.name}")
                    
                    # 5. Confluence: Fibonacci or MA
                    fib_ok = self._check_fibonacci(primary, level['level'])
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
                        
                    if not (fib_ok or ma_ok):
                        logger.debug(f"Pattern {pattern} at idx={idx} for {sym} skipped due to missing confluence (fib_ok={fib_ok}, ma_ok={ma_ok})")
                        continue
                    
                    logger.info(f"{sym}: Strong confluence signal found - {pattern} with {'Fibonacci' if fib_ok else ''}{' and ' if fib_ok and ma_ok else ''}{'MA' if ma_ok else ''} confluence")
                    
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
                    # Ensure atr_val is a float or None
                    atr_val_val = float(atr_val) if isinstance(atr_val, (float, int, np.floating, np.integer)) else 0.0
                    tol_val = max(level['level'] * self.price_tolerance, atr_val_val if atr_val_val is not None else 0)
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
                        max_sl_dist = atr_val * self.max_sl_atr_mult
                    if self.max_sl_pct is not None:
                        max_sl_dist_pct = entry * self.max_sl_pct
                        max_sl_dist = min(max_sl_dist, max_sl_dist_pct) if max_sl_dist is not None else max_sl_dist_pct
                    if max_sl_dist is not None and abs(entry - stop) > max_sl_dist:
                        if direction == 'buy':
                            stop = entry - max_sl_dist
                        else:
                            stop = entry + max_sl_dist
                        reward = abs(entry - stop)
                        tp = entry + reward * self.min_risk_reward if direction == 'buy' else entry - reward * self.min_risk_reward
                    
                    # Calculate risk-reward and other trade metrics
                    risk_pips = abs(entry - stop)
                    reward_pips = abs(tp - entry)
                    risk_reward_ratio = reward_pips / risk_pips if risk_pips > 0 else 0
                    
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
                    
                    # Calculate advanced volume analysis with wick-based confirmation
                    volume_score, volume_details = self._analyze_volume_quality(primary, idx=-1, direction=direction)
                    
                    # Score pattern strength
                    pattern_score = 0.0
                    if pattern == 'Pin Bar':
                        # Pin bars stronger when body/range ratio is smaller
                        pattern_score = 1.0 - min(1.0, pattern_details['body_to_range_ratio'] * 2)
                    elif pattern == 'Engulfing':
                        # Engulfing stronger when engulfing ratio is higher
                        pattern_score = min(1.0, pattern_details['engulfing_ratio'] / 2.0)
                    elif pattern == 'Inside Bar':
                        # Inside bars stronger when containment ratio is smaller
                        pattern_score = 1.0 - min(1.0, pattern_details['containment_ratio'] * 2)
                    elif pattern == 'Hammer':
                        # Hammers stronger when lower wick is longer
                        pattern_score = min(1.0, pattern_details['lower_wick'] / (pattern_details['body_size'] + 1e-6))
                    elif pattern == 'Shooting Star':
                        # Shooting Stars stronger when upper wick is longer
                        pattern_score = min(1.0, pattern_details['upper_wick'] / (pattern_details['body_size'] + 1e-6))
                    elif pattern == 'Morning Star':
                        # Morning Stars stronger when 2nd body is smaller
                        pattern_score = min(1.0, pattern_details['c2_body'] / (pattern_details['c1_body'] + pattern_details['c3_body'] + 1e-6))
                    elif pattern == 'Evening Star':
                        # Evening Stars stronger when 2nd body is smaller
                        pattern_score = min(1.0, pattern_details['c2_body'] / (pattern_details['c1_body'] + pattern_details['c3_body'] + 1e-6))
                    elif pattern == 'False Breakout':
                        # False breakouts stronger when reversal ratio is higher
                        pattern_score = min(1.0, pattern_details['reversal_size'] / (pattern_details['breakout_size'] + 1e-6))
                    
                    # Score confluence factors
                    confluence_score = 0.0
                    if fib_ok and ma_ok:
                        confluence_score = 1.0  # Both fib and MA is maximum confluence
                    elif fib_ok or ma_ok:
                        confluence_score = 0.6  # Single confluence factor
                    
                    # Calculate level strength score
                    level_strength_score = min(1.0, level['strength'] / 5.0)  # Cap at 5 touches
                    # Calculate overall signal quality 
                    signal_quality = (
                        (pattern_score * 0.35) +
                        (confluence_score * 0.35) +
                        (0.2 if volume_score >= 1.0 else 0.0) +
                        (level_strength_score * 0.1)
                    )
                    # Standardized confidence: direct mapping, clamped to [0, 1]
                    confidence = max(0.0, min(1.0, signal_quality))
                    
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
                    signal = {
                        "symbol": sym,
                        "direction": direction,
                        "entry_price": entry,
                        "stop_loss": stop,
                        "take_profit": tp,
                        "dynamic_exits": exits,
                        "timeframe": self.primary_timeframe,
                        "confidence": confidence,
                        "source": self.name,
                        "pattern": pattern,
                        "confluence": {"fib": fib_ok, "ma": ma_ok},
                        "pattern_details": pattern_details,
                        "fib_details": fib_details if fib_ok else {},
                        "ma_details": ma_details if ma_ok else {},
                        "volume_details": volume_details,
                        "risk_pips": risk_pips,
                        "reward_pips": reward_pips,
                        "risk_reward_ratio": risk_reward_ratio,
                        "signal_quality": signal_quality,
                        "technical_metrics": technical_metrics,
                        "pattern_score": pattern_score,
                        "confluence_score": confluence_score,
                        "volume_score": volume_score,
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
                    signals.append(adjusted_signal)
                    continue  # allow other patterns per level

            # --- Acceptance-based (trend continuation) logic (new) ---
            # Only check for acceptance if price has broken the level
            acceptance_bars = 3
            # --- ATR-based SL tolerance for acceptance logic ---
            atr_val = None
            if len(primary) >= 14:
                atr_series = calculate_atr(primary, period=14)
                if isinstance(atr_series, pd.Series):
                    atr_val = float(atr_series.iloc[-1])
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
            # Ensure atr_val is a float or None
            atr_val_val = float(atr_val) if isinstance(atr_val, (float, int, np.floating, np.integer)) else 0.0
            tol_val = max(level['level'] * self.price_tolerance, atr_val_val if atr_val_val is not None else 0)
            # --- Trend continuation (acceptance) scoring ---
            # Use multi-factor scoring: pattern (momentum), confluence (MA), volume, recency, level strength, trend alignment
            def _is_momentum_candle(df, idx, direction):
                if df is None or len(df) == 0 or idx >= len(df) or idx < -len(df):
                    return False
                candle = df.iloc[idx]
                total_range = candle['high'] - candle['low']
                body = abs(candle['close'] - candle['open'])
                if total_range == 0:
                    return False
                # Strong body, closes near high/low, large relative to ATR
                if direction == 'buy':
                    return (
                        body / total_range > 0.6 and
                        (candle['close'] - candle['low']) / total_range > 0.6
                    )
                else:
                    return (
                        body / total_range > 0.6 and
                        (candle['high'] - candle['close']) / total_range > 0.6
                    )
            if trend == 'bullish':
                # Look for closes above resistance (trend continuation)
                if primary['close'].iloc[-1] > level['level'] + tol_val and \
                   (primary['close'].iloc[-2:] > level['level']).sum() >= 1:
                    if self._is_acceptance(primary, level['level'], 'bullish', bars=acceptance_bars, tol=self.price_tolerance):
                        logger.info(f"{sym}: Price acceptance above {level_type} level {level['level']:.5f} for {acceptance_bars} bars (trend continuation)")
                        entry = primary['close'].iloc[-1]
                        stop = primary['low'].iloc[-1] - tol_val
                        reward = entry - stop
                        tp = entry + reward * self.min_risk_reward
                        direction_str = 'buy'
                        # --- Volume confirmation for acceptance ---
                        volume_score, volume_details = self._analyze_volume_quality(primary, idx=-1, direction=direction_str)
                        # --- Pattern: momentum candle ---
                        pattern_score = 1.0 if _is_momentum_candle(primary, -1, direction_str) else 0.0
                        # --- Confluence: MA ---
                        ma_ok = self._check_ma(primary, entry)
                        confluence_score = 1.0 if ma_ok else 0.0
                        # --- Level strength ---
                        level_strength_score = min(1.0, level['strength'] / 5.0)
                        # --- Trend alignment ---
                        alignment = self._evaluate_signal_alignment(primary, higher, direction_str)
                        trend_score = alignment.get('alignment_score', 0.0)
                        # --- Weighted scoring ---
                        signal_quality = (
                            (pattern_score * 0.25) +
                            (confluence_score * 0.2) +
                            (0.2 if volume_score >= 1.0 else 0.0) +
                            (level_strength_score * 0.15) +
                            (trend_score * 0.15)
                        )
                        confidence = max(0.0, min(1.0, signal_quality))
                        # Only accept if score is at least 0.2
                        if confidence < 0.2:
                            logger.debug(f"{sym}: Acceptance signal rejected due to low score {confidence:.2f}")
                            continue
                        risk_pips = abs(entry - stop)
                        reward_pips = abs(tp - entry)
                        risk_reward_ratio = reward_pips / risk_pips if risk_pips > 0 else 0
                        # --- RiskManager integration for validation and sizing (trend continuation) ---
                        signal = {
                            "symbol": sym,
                            "direction": direction_str,
                            "entry_price": entry,
                            "stop_loss": stop,
                            "take_profit": tp,
                            "dynamic_exits": {},
                            "timeframe": self.primary_timeframe,
                            "confidence": confidence,
                            "source": self.name,
                            "pattern": "Acceptance Breakout",
                            "confluence": {"ma": ma_ok},
                            "pattern_details": {"momentum": pattern_score > 0},
                            "fib_details": {},
                            "ma_details": {"ma_period": self.ma_period, "ma_ok": ma_ok},
                            "volume_details": volume_details,
                            "risk_pips": risk_pips,
                            "reward_pips": reward_pips,
                            "risk_reward_ratio": risk_reward_ratio,
                            "signal_quality": signal_quality,
                            "technical_metrics": {},
                            "pattern_score": pattern_score,
                            "confluence_score": confluence_score,
                            "volume_score": volume_score,
                            "level_strength": level['strength'],
                            "level_strength_score": level_strength_score,
                            "trend_score": trend_score,
                            "description": f"Price acceptance breakout above {level_type} {level['level']:.5f} (Trend: {trend.capitalize()}, Score: {confidence:.2f})",
                            "detailed_reasoning": [f"Breakout confirmed with {acceptance_bars} bars above level {level['level']:.5f}. Volume score: {volume_score:.1f}, Trend score: {trend_score:.1f}"]
                        }
                        result = rm.validate_and_size_trade(signal)
                        if not result['valid']:
                            logger.info(f"Trend continuation signal for {sym} rejected by RiskManager: {result['reason']}")
                            continue
                        adjusted_signal = result['adjusted_trade']
                        for k in signal:
                            if k not in adjusted_signal:
                                adjusted_signal[k] = signal[k]
                        signals.append(adjusted_signal)
                        continue
            else:
                # Look for closes below support (trend continuation)
                if primary['close'].iloc[-1] < level['level'] - tol_val and \
                   (primary['close'].iloc[-2:] < level['level']).sum() >= 1:
                    if self._is_acceptance(primary, level['level'], 'bearish', bars=acceptance_bars, tol=self.price_tolerance):
                        logger.info(f"{sym}: Price acceptance below {level_type} level {level['level']:.5f} for {acceptance_bars} bars (trend continuation)")
                        entry = primary['close'].iloc[-1]
                        stop = primary['high'].iloc[-1] + tol_val
                        reward = stop - entry
                        tp = entry - reward * self.min_risk_reward
                        direction_str = 'sell'
                        # --- Volume confirmation for acceptance ---
                        volume_score, volume_details = self._analyze_volume_quality(primary, idx=-1, direction=direction_str)
                        # --- Pattern: momentum candle ---
                        pattern_score = 1.0 if _is_momentum_candle(primary, -1, direction_str) else 0.0
                        # --- Confluence: MA ---
                        ma_ok = self._check_ma(primary, entry)
                        confluence_score = 1.0 if ma_ok else 0.0
                        # --- Level strength ---
                        level_strength_score = min(1.0, level['strength'] / 5.0)
                        # --- Trend alignment ---
                        alignment = self._evaluate_signal_alignment(primary, higher, direction_str)
                        trend_score = alignment.get('alignment_score', 0.0)
                        # --- Weighted scoring ---
                        signal_quality = (
                            (pattern_score * 0.25) +
                            (confluence_score * 0.2) +
                            (0.2 if volume_score >= 1.0 else 0.0) +
                            (level_strength_score * 0.15) +
                            (trend_score * 0.15)
                        )
                        confidence = max(0.0, min(1.0, signal_quality))
                        # Only accept if score is at least 0.2
                        if confidence < 0.2:
                            logger.debug(f"{sym}: Acceptance signal rejected due to low score {confidence:.2f}")
                            continue
                        risk_pips = abs(entry - stop)
                        reward_pips = abs(tp - entry)
                        risk_reward_ratio = reward_pips / risk_pips if risk_pips > 0 else 0
                        # --- RiskManager integration for validation and sizing (trend continuation) ---
                        signal = {
                            "symbol": sym,
                            "direction": direction_str,
                            "entry_price": entry,
                            "stop_loss": stop,
                            "take_profit": tp,
                            "dynamic_exits": {},
                            "timeframe": self.primary_timeframe,
                            "confidence": confidence,
                            "source": self.name,
                            "pattern": "Acceptance Breakout",
                            "confluence": {"ma": ma_ok},
                            "pattern_details": {"momentum": pattern_score > 0},
                            "fib_details": {},
                            "ma_details": {"ma_period": self.ma_period, "ma_ok": ma_ok},
                            "volume_details": volume_details,
                            "risk_pips": risk_pips,
                            "reward_pips": reward_pips,
                            "risk_reward_ratio": risk_reward_ratio,
                            "signal_quality": signal_quality,
                            "technical_metrics": {},
                            "pattern_score": pattern_score,
                            "confluence_score": confluence_score,
                            "volume_score": volume_score,
                            "level_strength": level['strength'],
                            "level_strength_score": level_strength_score,
                            "trend_score": trend_score,
                            "description": f"Price acceptance breakout below {level_type} {level['level']:.5f} (Trend: {trend.capitalize()}, Score: {confidence:.2f})",
                            "detailed_reasoning": [f"Breakout confirmed with {acceptance_bars} bars below level {level['level']:.5f}. Volume score: {volume_score:.1f}, Trend score: {trend_score:.1f}"]
                        }
                        result = rm.validate_and_size_trade(signal)
                        if not result['valid']:
                            logger.info(f"Trend continuation signal for {sym} rejected by RiskManager: {result['reason']}")
                            continue
                        adjusted_signal = result['adjusted_trade']
                        for k in signal:
                            if k not in adjusted_signal:
                                adjusted_signal[k] = signal[k]
                        signals.append(adjusted_signal)
                        continue
        return signals

    # -- Trend and level detection --
    def _determine_trend(self, df: pd.DataFrame) -> str:
        """Return 'bullish', 'bearish' or 'neutral' based on moving average"""
        if df is None or 'close' not in df.columns or len(df) < self.ma_period:
            return 'neutral'
        # Simple MA-based trend
        ma = df['close'].rolling(window=self.ma_period).mean().iloc[-1]
        last = df['close'].iloc[-1]
        if last > ma:
            return 'bullish'
        elif last < ma:
            return 'bearish'
        return 'neutral'

    def _find_key_levels(self, df: pd.DataFrame) -> tuple:
        """Return (supports, resistances) as lists of dicts with 'level' and 'strength' (touch count)"""
        lows, highs = [], []
        # Limit to profile pivot_lookback bars
        subset = df.copy()
        if len(df) > self.pivot_lookback:
            subset = df.iloc[-self.pivot_lookback:]
        # Pivot detection
        for i in range(2, len(subset) - 2):
            if (df['low'].iat[i] < df['low'].iat[i-1] and df['low'].iat[i] < df['low'].iat[i-2]
                    and df['low'].iat[i] < df['low'].iat[i+1] and df['low'].iat[i] < df['low'].iat[i+2]):
                lows.append(subset['low'].iat[i])
            if (df['high'].iat[i] > df['high'].iat[i-1] and df['high'].iat[i] > df['high'].iat[i-2]
                    and df['high'].iat[i] > df['high'].iat[i+1] and df['high'].iat[i] > df['high'].iat[i+2]):
                highs.append(subset['high'].iat[i])
        # Debug: log number of raw pivots
        logger.debug(f"[KeyLevels] Raw pivot lows: {len(lows)}, pivot highs: {len(highs)} (before clustering)")
        # Calculate clustering tolerance using tightened parameters: use min(average * price_tolerance, 10 ticks)
        clustering_tol = min(subset['close'].mean() * self.price_tolerance, 10 * TICK_SIZE)
        # Cluster levels to avoid duplicates
        support_levels = self._cluster_levels_with_strength(sorted(lows), clustering_tol, subset, is_support=True)
        resistance_levels = self._cluster_levels_with_strength(sorted(highs), clustering_tol, subset, is_support=False)
        # Debug: log number of clusters before filtering
        logger.debug(f"[KeyLevels] Support clusters: {len(support_levels)}, Resistance clusters: {len(resistance_levels)} (before filtering by touches)")
        # Filter levels: only keep those with at least 1 touch in the last 50 bars
        last_50 = df.iloc[-min(50, len(df)):] 
        support_levels = [lvl for lvl in support_levels if self._count_level_touches(last_50, lvl['level'], clustering_tol, is_support=True) >= 1]
        resistance_levels = [lvl for lvl in resistance_levels if self._count_level_touches(last_50, lvl['level'], clustering_tol, is_support=False) >= 1]
        # Debug: log number of levels after filtering
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
        """Check if price pulled back to `level` within recent bars
        Only flag a pullback if a prior bar in the window closed beyond the level, and the last bar retests the level within tolerance.
        """
        if df is None or len(df) < self.pullback_bars:
            return False
        tol_val = level * self.price_tolerance
        recent_closes = df['close'].iloc[-self.pullback_bars:]
        last_candle = df.iloc[-1]
        if direction == 'bullish':
            # Ensure that at least one prior bar (excluding the last) closed above the level
            if recent_closes.iloc[:-1].max() > level and abs(last_candle['low'] - level) <= tol_val:
                return True
        else:
            # For bearish, ensure that at least one prior bar closed below the level
            if recent_closes.iloc[:-1].min() < level and abs(last_candle['high'] - level) <= tol_val:
                return True
        return False

    # -- Candlestick pattern checks --
    def _is_pin_bar(self, candle: pd.Series, level: float, direction: str) -> bool:
        """Detect a pin bar touching `level` with a long wick and body in the correct third of the range"""
        body = abs(candle['close'] - candle['open'])
        total = candle['high'] - candle['low']
        if total <= 0:
            return False
        tol_val = level * self.price_tolerance
        if direction == 'bullish':
            lower_wick = candle['open'] - candle['low']
            # Ensure lower wick is >2x body, touches level, and body is in upper third
            body_top = max(candle['open'], candle['close'])
            if (lower_wick > body * 2 and
                abs(candle['low'] - level) <= tol_val and
                body_top > candle['low'] + 2 * total / 3):
                return True
        else:
            upper_wick = candle['high'] - candle['open']
            # Ensure upper wick is >2x body, touches level, and body is in lower third
            body_bottom = min(candle['open'], candle['close'])
            if (upper_wick > body * 2 and
                abs(candle['high'] - level) <= tol_val and
                body_bottom < candle['high'] - 2 * total / 3):
                return True
        return False

    def _is_engulfing(self, candles: pd.DataFrame, idx: int, direction: str) -> bool:
        """Detect bullish or bearish engulfing at index"""
        if idx <= 0 or idx >= len(candles):
            return False
        curr = candles.iloc[idx]
        prev = candles.iloc[idx - 1]
        if direction == 'bullish':
            return (curr['close'] > curr['open'] and prev['close'] < prev['open'] and
                    curr['open'] < prev['close'] and curr['close'] > prev['open'])
        else:
            return (curr['close'] < curr['open'] and prev['close'] > prev['open'] and
                    curr['open'] > prev['close'] and curr['close'] < prev['open'])

    def _is_inside_bar(self, candles: pd.DataFrame, idx: int) -> bool:
        """Detect an inside bar where the current candle is fully contained within the previous candle."""
        if idx <= 0 or idx >= len(candles):
            return False
        mother = candles.iloc[idx - 1]
        child = candles.iloc[idx]
        return child['high'] < mother['high'] and child['low'] > mother['low']

    def _is_hammer(self, candle: pd.Series, level: float) -> bool:
        """Detect a Hammer pattern (bullish reversal) near support."""
        body = abs(candle['close'] - candle['open'])
        total = candle['high'] - candle['low']
        lower_wick = min(candle['open'], candle['close']) - candle['low']
        upper_wick = candle['high'] - max(candle['open'], candle['close'])
        if total <= 0:
            return False
        # Hammer: small body, long lower wick, small upper wick, near support
        return (
            body / total < 0.3 and
            lower_wick > 2 * body and
            upper_wick < body and
            abs(candle['low'] - level) < level * self.price_tolerance
        )

    def _is_shooting_star(self, candle: pd.Series, level: float) -> bool:
        """Detect a Shooting Star pattern (bearish reversal) near resistance."""
        body = abs(candle['close'] - candle['open'])
        total = candle['high'] - candle['low']
        upper_wick = candle['high'] - max(candle['open'], candle['close'])
        lower_wick = min(candle['open'], candle['close']) - candle['low']
        if total <= 0:
            return False
        # Shooting Star: small body, long upper wick, small lower wick, near resistance
        return (
            body / total < 0.3 and
            upper_wick > 2 * body and
            lower_wick < body and
            abs(candle['high'] - level) < level * self.price_tolerance
        )

    def _is_morning_star(self, candles: pd.DataFrame, idx: int, level: float) -> bool:
        """Detect a Morning Star (bullish 3-bar reversal) near support, anchored within the last pattern_bars bars."""
        if idx < 2:
            return False
        # Ensure the entire 3-bar pattern is within the most recent pattern_bars bars
        if (len(candles) - idx) > self.pattern_bars:
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
        """Detect an Evening Star (bearish 3-bar reversal) near resistance, anchored within the last pattern_bars bars."""
        if idx < 2:
            return False
        # Ensure the entire 3-bar pattern is within the most recent pattern_bars bars
        if (len(candles) - idx) > self.pattern_bars:
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
        """Enhanced: Detect a quick reversal after a breakout around `level` with wick and volume analysis."""
        if idx <= 0 or idx >= len(candles):
            return False
        prev = candles.iloc[idx - 1]
        curr = candles.iloc[idx]
        tol_val = level * self.price_tolerance
        # Wick analysis
        if direction == 'bullish':
            wick = curr['close'] - curr['low']
            body = abs(curr['close'] - curr['open'])
            wick_ok = wick > 2 * body
            # Volume analysis
            vol_col = 'volume' if 'volume' in candles.columns else 'tick_volume'
            avg_vol = candles[vol_col].rolling(window=20).mean().iloc[idx]
            vol_ok = curr[vol_col] > avg_vol
            return (
                prev['close'] < level - tol_val and
                curr['close'] > level + tol_val and
                wick_ok and
                vol_ok
            )
        else:
            wick = curr['high'] - curr['close']
            body = abs(curr['close'] - curr['open'])
            wick_ok = wick > 2 * body
            vol_col = 'volume' if 'volume' in candles.columns else 'tick_volume'
            avg_vol = candles[vol_col].rolling(window=20).mean().iloc[idx]
            vol_ok = curr[vol_col] > avg_vol
            return (
                prev['close'] > level + tol_val and
                curr['close'] < level - tol_val and
                wick_ok and
                vol_ok
            )

    # -- Confluence checks --
    def _find_recent_swing(self, df: pd.DataFrame, lookback: int = 50) -> tuple:
        """Find the most recent swing high and low in the last `lookback` bars."""
        if df is None or len(df) < lookback:
            return None, None
        recent = df.iloc[-lookback:]
        swing_high = recent['high'].max()
        swing_low = recent['low'].min()
        return swing_high, swing_low

    def _check_fibonacci(self, df: pd.DataFrame, level: float) -> bool:
        """Return True if `level` is near a standard Fibonacci retracement of the most recent swing."""
        # Use the most recent swing high/low (last 5-day pivot or 50 bars)
        swing_high, swing_low = self._find_recent_swing(df, lookback=50)
        if swing_high is None or swing_low is None:
            return False
        tol_val = swing_high * 0.002  # 0.2% tolerance (was 0.1%)
        for f in self.fib_levels:
            fib_lv = swing_low + (swing_high - swing_low) * f
            if abs(level - fib_lv) <= tol_val:
                return True
        return False

    def _check_ma(self, df: pd.DataFrame, level: float) -> bool:
        """Return True if `level` is near the moving-average support/resistance (within 0.2%) and price has bounced off it at least once."""
        if df is None or len(df) < self.ma_period:
            return False
        ma = df['close'].rolling(self.ma_period).mean().iloc[-1]
        tol_val = ma * 0.002  # 0.2% tolerance (was 0.1%)
        # Check if level is within 0.2% of MA
        if abs(level - ma) > tol_val:
            return False
        # Check for price bounce: price crossed MA in recent window
        recent = df['close'].iloc[-self.ma_period:]
        ma_series = df['close'].rolling(self.ma_period).mean().iloc[-self.ma_period:]
        crossed = ((recent > ma_series) & (recent.shift(1) < ma_series)) | ((recent < ma_series) & (recent.shift(1) > ma_series))
        if crossed.any():
            return True
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
        is_ranging = range_ratio < 0.7  # If close range is less than 70% of total range
        
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
        """Check if market regime is favorable (trending, not ranging)."""
        if df is None or len(df) < 20:
            return False
        # ADX filter for trend strength
        adx_series, _, _ = calculate_adx(df, period=14)
        adx = adx_series.iloc[-1] if isinstance(adx_series, pd.Series) and not adx_series.empty else None
        # Range filter
        context = self._analyze_market_context(df)
        # Check all conditions (ATR/volatility check removed)
        return (adx is not None and adx > 25 and
                not context.get('is_ranging', False))

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