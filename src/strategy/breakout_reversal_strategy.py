"""
Breakout and Reversal Hybrid Strategy

This signal generator implements a price action strategy focused on trading breakouts
and reversals at key support and resistance levels. It's based on Indrazith Shantharaj's
Price Action Trading principles.

Key features:
- Support and resistance level identification
- Trend line detection and analysis
- Breakout detection with volume confirmation
- Reversal pattern recognition at key levels
- Risk management integration with proper position sizing
- No technical indicators, just pure price action
"""

# pyright: reportArgumentType=false

import pandas as pd
import numpy as np
from loguru import logger
from typing import Dict, List, Any, Optional, Tuple
import math
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pathlib import Path
import time

from src.trading_bot import SignalGenerator
from src.risk_manager import RiskManager
import talib 
from src.utils.patterns_luxalgo import add_luxalgo_patterns, BULLISH_PATTERNS, BEARISH_PATTERNS, NEUTRAL_PATTERNS, ALL_PATTERNS, filter_patterns_by_bias, get_pattern_type

# Strategy parameter profiles for different timeframes
TIMEFRAME_PROFILES = {
    "M1": {
        "lookback_period": 300,  # ~5 hours to cover a full trading session
        "max_retest_bars": 30,   # 30 minutes for retest windows
        "level_update_hours": 4,
        "consolidation_bars": 60,
        "candles_to_check": 10,
        "consolidation_update_hours": 2,
        "atr_multiplier": 0.5,   # Lower multiplier for noisy timeframe
        "volume_percentile": 80,  # Changed from 85
        "min_risk_reward": 1.8   # Changed from 2.0
    },
    "M5": {
        "lookback_period": 140,  # ~12 hours
        "max_retest_bars": 12,   # 60 minutes
        "level_update_hours": 6,
        "consolidation_bars": 40, 
        "candles_to_check": 6,
        "consolidation_update_hours": 3,
        "atr_multiplier": 0.7,   # Medium multiplier
        "volume_percentile": 80,  # Changed from 85
        "min_risk_reward": 1.8   # Changed from 2.0
    },
    "M15": {
        "lookback_period": 96,   # ~24 hours
        "max_retest_bars": 6,    # 90 minutes
        "level_update_hours": 12,
        "consolidation_bars": 20,
        "candles_to_check": 3,
        "consolidation_update_hours": 6,
        "atr_multiplier": 1.0,   # Standard multiplier
        "volume_percentile": 80,  # Changed from 85
        "min_risk_reward": 1.8   # Changed from 2.0
    },
    "H1": {
        "lookback_period": 50,   # ~2 days
        "max_retest_bars": 6,    # 6 hours
        "level_update_hours": 24,
        "consolidation_bars": 10,
        "candles_to_check": 2,
        "consolidation_update_hours": 12,
        "atr_multiplier": 1.2,   # Higher multiplier for more significant movements
        "volume_percentile": 80,  # Changed from 85
        "min_risk_reward": 2.5   # Changed from 3.0
    },
    "H4": {
        "lookback_period": 30,   # ~5 days
        "max_retest_bars": 4,    # 16 hours
        "level_update_hours": 48,
        "consolidation_bars": 7,
        "candles_to_check": 2,
        "consolidation_update_hours": 24,
        "atr_multiplier": 1.5,   # Higher multiplier for more significant movements
        "volume_percentile": 80,  # Changed from 85
        "min_risk_reward": 2.5   # Changed from 3.0
    }
}
class _TrendLineAnalyzer:
    """Encapsulate trend-line machinery into a reusable analyzer."""
    def __init__(self, df: pd.DataFrame, strategy: 'BreakoutReversalStrategy'):
        self.df = df
        self.strategy = strategy
        self.params = {
            'min_points': strategy.trend_line_min_points,
            'max_angle': strategy.trend_line_max_angle,
            'r_squared_threshold': 0.5,
            'touches_threshold': 2,
            'cluster': {
                'angle_tolerance': 5.0,
                'intercept_pct_tolerance': 0.0015,
                'slope_tolerance': 0.00005
            },
            'max_trend_lines': 12
        }

    def find_swings(self, swing_window: int = 5) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        highs = self.strategy._find_swing_highs(self.df, window=swing_window)
        lows = self.strategy._find_swing_lows(self.df, window=swing_window)
        return highs, lows

    def fit_lines(self, swing_highs: List[Tuple[int, float]], swing_lows: List[Tuple[int, float]], skip_plots: bool=False) -> Tuple[List[Dict], List[Dict]]:
        bullish = self.strategy._identify_trend_lines(self.df, swing_lows, 'bullish', skip_plots)
        bearish = self.strategy._identify_trend_lines(self.df, swing_highs, 'bearish', skip_plots)
        return bullish, bearish

    def validate_and_cluster(self, lines: List[Dict]) -> List[Dict]:
        clustered = self.strategy._cluster_trend_lines(lines)
        return sorted(clustered, key=lambda x: x['quality_score'], reverse=True)[:self.params['max_trend_lines']]

    def get_trend_lines(self, skip_plots: bool=False, swing_window: int = 5) -> Tuple[List[Dict], List[Dict]]:
        swing_highs, swing_lows = self.find_swings(swing_window)
        bullish, bearish = self.fit_lines(swing_highs, swing_lows, skip_plots)
        bullish = self.validate_and_cluster(bullish)
        bearish = self.validate_and_cluster(bearish)
        return bullish, bearish

    def get_support_lines(self, skip_plots: bool=False, **kwargs) -> List[Dict]:
        swing_window = kwargs.get('swing_window', 5)
        _, swing_lows = self.find_swings(swing_window)
        bullish, _ = self.fit_lines([], swing_lows, skip_plots)
        return self.validate_and_cluster(bullish)

    def get_resistance_lines(self, skip_plots: bool=False, **kwargs) -> List[Dict]:
        swing_window = kwargs.get('swing_window', 5)
        swing_highs, _ = self.find_swings(swing_window)
        _, bearish = self.fit_lines(swing_highs, [], skip_plots)
        return self.validate_and_cluster(bearish)

class _SignalScorer:
    """Encapsulate volume analysis and signal scoring."""
    def __init__(self, strategy: 'BreakoutReversalStrategy'):
        self.strategy = strategy

    def analyze_volume_quality(self, candle: pd.Series, threshold: float, df: pd.DataFrame = None) -> float:
        """
        Simplified volume analysis based on threshold and candle decisiveness.
        Returns +1.0 (Strong), 0.0 (Neutral), or -1.0 (Weak/Contradictory).
        """
        try:
            tick_volume = candle.get('tick_volume', candle.get('volume'))
            if tick_volume is None:
                self.strategy.logger.debug("Volume data missing, returning weak score.")
                return -1.0

            # Compute threshold if df is provided (for dynamic thresholds)
            if df is not None and not df.empty:
                lookback = min(50, len(df))
                if 'tick_volume' in df.columns:
                    if self.strategy.volume_threshold_type == 'mean':
                        dynamic_threshold = df['tick_volume'].iloc[-lookback:].mean()
                    elif self.strategy.volume_threshold_type == 'median':
                        dynamic_threshold = df['tick_volume'].iloc[-lookback:].median()
                    else: # Default to percentile
                        dynamic_threshold = np.percentile(df['tick_volume'].iloc[-lookback:], self.strategy.volume_percentile)
                    threshold = dynamic_threshold # Override static threshold if df is available
                    self.strategy.logger.debug(f"[VOLUME] Using dynamic threshold ({self.strategy.volume_threshold_type}): {threshold:.1f}")
                else:
                    self.strategy.logger.debug("[VOLUME] 'tick_volume' not in df, using static threshold.")
            else:
                self.strategy.logger.debug(f"[VOLUME] No df for dynamic threshold, using static threshold: {threshold:.1f}")

            volume_above_threshold = tick_volume >= threshold

            if not volume_above_threshold:
                self.strategy.logger.debug(f"Volume {tick_volume:.1f} < threshold {threshold:.1f} -> Weak Volume")
                return -1.0 # Weak if below threshold

            # Volume is above threshold, now check candle structure
            is_bullish_candle = candle['close'] > candle['open']
            is_bearish_candle = candle['close'] < candle['open']
            total_range = candle['high'] - candle['low']
            body_size = abs(candle['close'] - candle['open'])

            if total_range == 0: # Doji or no range
                self.strategy.logger.debug("Doji or zero range candle with high volume -> Neutral Volume")
                return 0.0 

            body_ratio = body_size / total_range
            upper_wick = candle['high'] - max(candle['open'], candle['close'])
            lower_wick = min(candle['open'], candle['close']) - candle['low']
            upper_wick_ratio = upper_wick / total_range
            lower_wick_ratio = lower_wick / total_range

            # Strong Bullish Implication
            if is_bullish_candle and body_ratio > 0.6 and lower_wick_ratio < 0.3:
                self.strategy.logger.debug("Strong bullish candle (large body, small lower wick) -> Bullish Volume (+1.0)")
                return 1.0
            if lower_wick_ratio > 0.5 and body_ratio < 0.3: # Strong lower wick rejection
                self.strategy.logger.debug("Strong lower wick rejection -> Bullish Volume (+1.0)")
                return 1.0

            # Strong Bearish Implication
            if is_bearish_candle and body_ratio > 0.6 and upper_wick_ratio < 0.3:
                self.strategy.logger.debug("Strong bearish candle (large body, small upper wick) -> Bearish Volume (-1.0)")
                return -1.0
            if upper_wick_ratio > 0.5 and body_ratio < 0.3: # Strong upper wick rejection
                self.strategy.logger.debug("Strong upper wick rejection -> Bearish Volume (-1.0)")
                return -1.0

            # If volume is high but candle is not decisive or shows mixed signals
            self.strategy.logger.debug("High volume but indecisive candle -> Neutral Volume")
            return 0.0

        except Exception as e:
            self.strategy.logger.error(f"Error in analyze_volume_quality: {str(e)}")
            return -1.0 # Default to weak on error

    def score_signal(
        self,
        signal: dict,
        df: pd.DataFrame,
        higher_df: pd.DataFrame,
        *,
        support_levels=None,
        resistance_levels=None,
        last_consolidation_ranges=None,
        atr_value=None,
        consolidation_info=None,
        risk_manager=None,
        account_balance=None,
        extra_context=None
    ) -> dict:
        """
        Score a signal dictionary based on multiple weighted factors, matching the old system's explicit weights and bonus logic.
        Adds a bonus if the entry/level price is near a high-volume node in the most recent consolidation's volume profile.
        """
        symbol = signal.get('symbol')
        direction = signal['direction']
        level = signal.get('level')
        entry = signal.get('entry_price')
        stop = signal.get('stop_loss')
        tp = signal.get('take_profit')
        reason = signal.get('reason', '').lower()
        # Use context or fallback to strategy
        price_tolerance = (extra_context or {}).get('price_tolerance', getattr(self.strategy, 'price_tolerance', 0.001))
        min_risk_reward = (extra_context or {}).get('min_risk_reward', getattr(self.strategy, 'min_risk_reward', 1.5))
        volume_percentile = (extra_context or {}).get('volume_percentile', getattr(self.strategy, 'volume_percentile', 80))
        # Determine higher timeframe trend
        higher_trend = self.strategy._determine_higher_timeframe_trend(higher_df)
        # 1. Level Strength (30%)
        level_strength_score = 0
        levels = support_levels if direction == 'buy' else resistance_levels
        if level is not None and levels:
            closest = min(levels, key=lambda x: abs(x-level))
            tol = level * price_tolerance * 1.5
            if abs(closest-level) < tol:
                touches = self.strategy._count_level_touches(df, closest, 'support' if direction=='buy' else 'resistance')
                level_strength_score = max(0.3, min(touches/5,1.0))
                # recency bonus
                recent = df.iloc[-20:]
                if direction == 'buy':
                    if any((abs(recent['low']-closest) <= closest*price_tolerance)):
                        level_strength_score = min(level_strength_score+0.2,1.0)
                        self.strategy.logger.debug(f"[LEVEL] Added recency bonus for {symbol} BUY: {level_strength_score:.2f}")
                else:
                    if any((abs(recent['high']-closest) <= closest*price_tolerance)):
                        level_strength_score = min(level_strength_score+0.2,1.0)
                        self.strategy.logger.debug(f"[LEVEL] Added recency bonus for {symbol} SELL: {level_strength_score:.2f}")
        # 2. Volume Quality (20%)
        volume_quality_score = 0.5 # Default to neutral
        if 'strong' in reason and 'volume' in reason: # Legacy reason check, can be phased out
            volume_quality_score = 1.0
            self.strategy.logger.debug(f"[VOLUME] {symbol}: Detected 'strong volume' in reason, setting score to 1.0")
        elif 'adequate' in reason and 'volume' in reason: # Legacy reason check
            volume_quality_score = 0.7
            self.strategy.logger.debug(f"[VOLUME] {symbol}: Detected 'adequate volume' in reason, setting score to 0.7")
        else:
            try:
                # Ensure df is available and has enough data for dynamic threshold calculation if used
                df_for_vol_threshold = df if df is not None and len(df) >= 20 else None
                vol_thresh_static = np.percentile(df['tick_volume'].iloc[-min(50,len(df)-1):], volume_percentile) if df is not None and not df.empty and 'tick_volume' in df.columns else 1.0
                
                candle = df.iloc[-1] # Assuming signal is for the last candle
                vol_quality = self.analyze_volume_quality(candle, vol_thresh_static, df_for_vol_threshold)

                # Map new vol_quality score (-1, 0, +1) to volume_quality_score (0 to 1)
                if direction == 'buy':
                    volume_quality_score = (vol_quality + 1) / 2
                else: # direction == 'sell'
                    volume_quality_score = (-vol_quality + 1) / 2
                
                self.strategy.logger.debug(f"[VOLUME] {symbol} {direction}: Raw vol_quality={vol_quality:.1f}, Mapped score={volume_quality_score:.2f}")

            except Exception as e:
                volume_quality_score = 0.5 # Default to neutral in case of error
                self.strategy.logger.warning(f"[VOLUME] {symbol}: Error calculating volume quality: {e}. Using default 0.5")
        # 3. Pattern Reliability (20%)
        pattern_reliability = {
            'bullish engulfing': 0.8,
            'bearish engulfing': 0.8,
            'morning star': 0.9,
            'evening star': 0.9,
            'hammer': 0.7,
            'shooting star': 0.7,
            'breakout': 0.6,
            'breakdown': 0.6,
            'trend line breakout': 0.75,
            'trend line breakdown': 0.75,
            'retest': 0.85
        }
        pattern_reliability_score = 0.5
        # Prefer explicit pattern_type in signal dict if provided
        explicit_pattern = signal.get('pattern_type', '').lower()
        if explicit_pattern and explicit_pattern in pattern_reliability:
            pattern_reliability_score = pattern_reliability[explicit_pattern]
            self.strategy.logger.debug(f"[PATTERN] {symbol}: Explicit pattern '{explicit_pattern}' reliability {pattern_reliability_score:.2f}")
        else:
            for pat, score in pattern_reliability.items():
                if pat in reason:
                    pattern_reliability_score = score
                    self.strategy.logger.debug(f"[PATTERN] {symbol}: Detected '{pat}' pattern with reliability score {score:.2f}")
                    break
        # 4. Trend Alignment (20%)
        trend_alignment_score = 0
        if direction=='buy':
            if higher_trend=='bullish':
                trend_alignment_score = 1.0
                self.strategy.logger.debug(f"[TREND] {symbol}: BUY perfectly aligned with BULLISH higher timeframe")
            elif higher_trend=='neutral':
                trend_alignment_score = 0.5
                self.strategy.logger.debug(f"[TREND] {symbol}: BUY with NEUTRAL higher timeframe (moderate alignment)")
            else:
                trend_alignment_score = 0.0
                self.strategy.logger.debug(f"[TREND] {symbol}: BUY against BEARISH higher timeframe (counter-trend)")
        else:
            if higher_trend=='bearish':
                trend_alignment_score = 1.0
                self.strategy.logger.debug(f"[TREND] {symbol}: SELL perfectly aligned with BEARISH higher timeframe")
            elif higher_trend=='neutral':
                trend_alignment_score = 0.5
                self.strategy.logger.debug(f"[TREND] {symbol}: SELL with NEUTRAL higher timeframe (moderate alignment)")
            else:
                trend_alignment_score = 0.0
                self.strategy.logger.debug(f"[TREND] {symbol}: SELL against BULLISH higher timeframe (counter-trend)")
        # 5. Risk-Reward (10%)
        if entry is None or stop is None or tp is None:
            risk = 0
            reward = 0
            risk_reward_score = 0
            rr_ratio = 0
        else:
            if direction=='buy':
                risk = entry-stop
                reward = tp-entry
            else:
                risk = stop-entry
                reward = entry-tp
            if risk > 0:
                rr_ratio = reward/risk
                risk_reward_score = min(rr_ratio/3, 1.0)
                self.strategy.logger.debug(f"[R:R] {symbol}: Risk-reward ratio {rr_ratio:.2f}:1, score: {risk_reward_score:.2f}")
            else:
                risk_reward_score = 0
                self.strategy.logger.warning(f"[R:R] {symbol}: Invalid risk calculation (risk={risk})")
        
        # Reset bonus tracking
        signal['_volume_profile_bonus'] = False
        signal['_atr_bonus'] = 0
        signal['consolidation_bonus'] = False
        
        # ATR bonus (±0.1)
        atr_bonus = 0
        try:
            high = np.asarray(df['high'].values, dtype=np.float64)
            low = np.asarray(df['low'].values, dtype=np.float64)
            close = np.asarray(df['close'].values, dtype=np.float64)
            atr_series = talib.ATR(high, low, close, timeperiod=getattr(self.strategy, 'atr_period', 14))
            atr = atr_series[-1] if len(atr_series) > 0 else None
            if atr is not None and not pd.isna(atr) and float(atr) > 0:
                stop_atr_ratio = abs(risk) / float(atr)
                if 0.5 <= stop_atr_ratio <= 3.0:
                    atr_bonus = 0.1
                    self.strategy.logger.debug(f"[BONUS] {symbol}: +0.1 ATR bonus for optimal stop placement (ratio: {stop_atr_ratio:.2f})")
                    signal['_atr_bonus'] = 0.1
                else:
                    atr_bonus = -0.1
                    self.strategy.logger.debug(f"[PENALTY] {symbol}: -0.1 ATR penalty for suboptimal stop placement (ratio: {stop_atr_ratio:.2f})")
                    signal['_atr_bonus'] = -0.1
        except Exception as e:
            atr_bonus = 0
            self.strategy.logger.debug(f"[ATR] {symbol}: Error calculating ATR bonus: {e}")
        
        # Final weighted score
        final = (
            level_strength_score * 0.3 +
            volume_quality_score * 0.2 +
            pattern_reliability_score * 0.2 +
            trend_alignment_score * 0.2 +
            risk_reward_score * 0.1
        )
        final = max(0, min(1, final + atr_bonus))
        
        # Consolidation bonus (+0.05 for reversals in consolidation)
        try:
            if 'reversal' in reason and last_consolidation_ranges and symbol in last_consolidation_ranges:
                is_consolidation = last_consolidation_ranges[symbol].get('is_consolidation', False)
                if is_consolidation:
                    final = min(1, final + 0.05)
                    self.strategy.logger.debug(f"[BONUS] {symbol}: +0.05 consolidation bonus for reversal inside consolidation range")
                    signal['consolidation_bonus'] = True
        except Exception as e:
            self.strategy.logger.debug(f"[CONSOLIDATION] {symbol}: Error checking consolidation: {e}")
        
        # --- VOLUME PROFILE NODE BONUS ---
        try:
            if last_consolidation_ranges and symbol in last_consolidation_ranges:
                vprof = last_consolidation_ranges[symbol].get('volume_profile', [])
                if vprof:
                    # Find the highest-volume node(s)
                    max_vol = max(node['total_volume'] for node in vprof)
                    high_nodes = [node for node in vprof if node['total_volume'] >= 0.9 * max_vol]
                    # Use entry or level price
                    price = entry if entry is not None else level
                    if price is not None:
                        for node in high_nodes:
                            tol = max(abs(price) * 0.0015, (node['bin_max'] - node['bin_min']) / 2)
                            if abs(price - node['center']) <= tol:
                                final = min(1, final + 0.07)
                                self.strategy.logger.debug(f"[VOLUME PROFILE BONUS] {symbol}: Entry/level {price:.5f} near high-volume node {node['center']:.5f} (tol={tol:.5f}) → score +0.07")
                                signal['_volume_profile_bonus'] = True
                                break
        except Exception as e:
            self.strategy.logger.debug(f"[VOLUME PROFILE BONUS] Exception: {e}")
            
        # Comprehensive score breakdown log
        self.strategy.logger.info(f"📊 SCORE COMPONENTS for {symbol} {direction}:")
        self.strategy.logger.info(f"  • Level Strength: {level_strength_score:.2f} × 0.3 = {level_strength_score * 0.3:.2f}")
        self.strategy.logger.info(f"  • Volume Quality: {volume_quality_score:.2f} × 0.2 = {volume_quality_score * 0.2:.2f}")
        self.strategy.logger.info(f"  • Pattern Reliability: {pattern_reliability_score:.2f} × 0.2 = {pattern_reliability_score * 0.2:.2f}")
        self.strategy.logger.info(f"  • Trend Alignment: {trend_alignment_score:.2f} × 0.2 = {trend_alignment_score * 0.2:.2f}")
        self.strategy.logger.info(f"  • Risk-Reward: {risk_reward_score:.2f} × 0.1 = {risk_reward_score * 0.1:.2f}")
        
        # Log bonuses
        if atr_bonus != 0:
            self.strategy.logger.info(f"  • ATR Bonus/Penalty: {atr_bonus:.2f}")
        if signal.get('consolidation_bonus', False):
            self.strategy.logger.info(f"  • Consolidation Bonus: +0.05")
        if signal.get('_volume_profile_bonus', False):
            self.strategy.logger.info(f"  • Volume Profile Bonus: +0.07")
            
        self.strategy.logger.info(f"  📈 FINAL SCORE: {final:.2f}")
        
        signal['score'] = final
        signal['score_details'] = {
            'level_strength': level_strength_score,
            'volume_quality': volume_quality_score,
            'pattern_reliability': pattern_reliability_score,
            'trend_alignment': trend_alignment_score,
            'risk_reward': risk_reward_score,
            'final_score': final
        }
        if volume_quality_score < 0.5:
            signal['skip_due_to_volume'] = True
        return signal

class BreakoutReversalStrategy(SignalGenerator):
    """
    Breakout and Reversal Hybrid Strategy based on price action principles.
    Uses support/resistance levels, candlestick patterns, and volume analysis
    to generate high-probability trading signals.
    """
    
    def __init__(self, primary_timeframe="M15", higher_timeframe="H1", use_range_extension_tp=False, backtest_mode=False, **kwargs):
        """
        Initialize the Breakout and Reversal strategy.
        
        Args:
            primary_timeframe: Primary timeframe to analyze
            higher_timeframe: Higher timeframe for trend confirmation
            use_range_extension_tp: Whether to use Market Profile range extension for TP
            **kwargs: Additional parameters
        """
        # Call parent constructor to set up logger
        super().__init__(**kwargs)
        
        # Add logger instance here for reference in the class
        self.logger = logger
        
        # Strategy metadata
        self.name = "BreakoutReversalStrategy"
        self.description = "A hybrid strategy based on price action principles"
        self.version = "1.0.0"
        
        # Timeframes
        self.primary_timeframe = primary_timeframe
        self.higher_timeframe = higher_timeframe
        # self.required_timeframes = [primary_timeframe, higher_timeframe]  # Removed, use property
        
        # Load appropriate timeframe profile
        if primary_timeframe == "M1":
            self.timeframe_profile = "scalping"
        elif primary_timeframe in ["M5", "M15"]:
            self.timeframe_profile = "intraday"
        elif primary_timeframe in ["H1", "H4"]:
            self.timeframe_profile = "intraday_swing"
        else:
            self.timeframe_profile = "swing"
        
        logger.info(f"🔍 Using '{self.timeframe_profile}' profile for {primary_timeframe} timeframe")
        
        # General parameters
        self.lookback_period = kwargs.get("lookback_period", 100)  # Will be overridden by profile
        self.price_tolerance = kwargs.get("price_tolerance", 0.001)  # 0.1% tolerance for levels
        
        # ATR period for dynamic parameter scaling
        self.atr_period = kwargs.get("atr_period", 14)
        
        # Scale update intervals based on primary timeframe - these will be overridden by profile
        # Just set defaults based on timeframe for now
        if primary_timeframe == "M5":
            default_level_update = 1  # 1 hour for M5
            default_trend_line_update = 1  # 1 hour for M5
            default_range_update = 0.5  # 30 minutes for M5
            default_max_retest_time = 4  # 4 hours maximum to wait for retest on M5
        elif primary_timeframe == "M15":
            default_level_update = 2  # 2 hours for M15
            default_trend_line_update = 2  # 2 hours for M15
            default_range_update = 1  # 1 hour for M15
            default_max_retest_time = 8  # 8 hours maximum to wait for retest on M15
        elif primary_timeframe == "H1":
            default_level_update = 4  # 4 hours for H1
            default_trend_line_update = 4  # 4 hours for H1
            default_range_update = 2  # 2 hours for H1
            default_max_retest_time = 12  # 12 hours maximum to wait for retest on H1
        else:
            default_level_update = 8  # 8 hours for higher timeframes
            default_trend_line_update = 8  # 8 hours for higher timeframes
            default_range_update = 4  # 4 hours for higher timeframes
            default_max_retest_time = 24  # 24 hours maximum to wait for retest
        
        # Key level parameters
        self.min_level_touches = kwargs.get("min_level_touches", 2)
        self.level_recency_weight = kwargs.get("level_recency_weight", 0.5)
        self.level_update_interval = kwargs.get("level_update_interval", default_level_update)  # Hours
        
        # Trend line parameters
        self.trend_line_min_points = kwargs.get("trend_line_min_points", 3)
        self.trend_line_max_angle = kwargs.get("trend_line_max_angle", 45)  # degrees
        self.trend_line_update_interval = kwargs.get("trend_line_update_interval", default_trend_line_update)  # Hours
        
        # Breakout parameters
        self.retest_required = kwargs.get("retest_required", False)  # Force default to False for more signals
        self.max_retest_time = kwargs.get("max_retest_time", default_max_retest_time)  # Max hours to wait for retest
        self.candles_to_check = kwargs.get("candles_to_check", 5)  # How many recent candles to analyze
        
        # Consolidation parameters
        self.consolidation_length = kwargs.get("consolidation_length", 12)  # Minimum number of candles
        self.consolidation_range_max = kwargs.get("consolidation_range_max", 0.02)  # 2% max range
        self.range_update_interval = kwargs.get("range_update_interval", default_range_update)  # Hours
        
        # Ensure consolidation_bars is initialized (fix for attribute error)
        self.consolidation_bars = kwargs.get("consolidation_bars", 20)  # Default value
        
        # Risk management
        self.min_risk_reward = kwargs.get("min_risk_reward", 1.5)  # Minimum R:R ratio
        self.max_stop_pct = kwargs.get("max_stop_pct", 0.02)  # Maximum stop loss (% of price)
        
        # Volume analysis
        self.volume_threshold = kwargs.get("volume_threshold", 0.8)  # Volume spike threshold (multiplier of average) - lowered from 1.5 to 0.8
        
        # Initialize ATR and volume percentile settings (new)
        self.atr_multiplier = kwargs.get("atr_multiplier", 1.0)
        self.volume_percentile = kwargs.get("volume_percentile", 80)
        self.volume_threshold_type = kwargs.get("volume_threshold_type", "percentile")  # 'percentile', 'mean', or 'median'
        
        # Load timeframe-specific parameters from the profile
        # This will override any default values set above
        self._load_timeframe_profile()
        
        # Initialize storage for key levels, trend lines, and signals
        self.support_levels = {}
        self.resistance_levels = {}
        self.bullish_trend_lines = {}
        self.bearish_trend_lines = {}
        self.last_consolidation_ranges = {}
        self.retest_tracking = {}
        
        # Timetracking for updates
        self.last_updated = {
            'key_levels': {},
            'trend_lines': {},
            'consolidation_ranges': {}
        }
        
        current_time_init = datetime.now()
        logger.debug(f"⏰ Initializing time tracking with current time: {current_time_init}")
        
        logger.info(f"🔧 Initialized {self.name} with primary TF: {primary_timeframe}, higher TF: {higher_timeframe}")
        
        # Log all parameters for reference
        params_log = {
            'lookback_period': self.lookback_period,
            'price_tolerance': self.price_tolerance,
            'min_level_touches': self.min_level_touches,
            'level_update_interval': self.level_update_interval,
            'trend_line_min_points': self.trend_line_min_points,
            'retest_required': self.retest_required,
            'volume_threshold': self.volume_threshold,
            'min_risk_reward': self.min_risk_reward,
            'atr_multiplier': self.atr_multiplier,
            'volume_percentile': self.volume_percentile,
            'consolidation_bars': self.consolidation_bars
        }
        logger.debug(f"📊 Strategy parameters: {params_log}")
        self._scorer = _SignalScorer(self)
        self.risk_manager = RiskManager.get_instance()
        self.use_range_extension_tp = use_range_extension_tp
        self.swing_window = kwargs.get('swing_window', self.candles_to_check)
        self.use_simple_entry = kwargs.get("use_simple_entry", True)
        # --- Deduplication tracking ---
        self.processed_bars: Dict[Tuple[str, str], str] = {}  # Ensure type hint matches value type (str)
        self.processed_zones: Dict[Tuple[str, str, float, str], float] = {}
        self.signal_cooldown = kwargs.get('signal_cooldown', 86400)  # 24h default
        self.bar_expiry_hours = kwargs.get('bar_expiry_hours', 24) # Added bar_expiry_hours
        self.debug_plots_enabled = kwargs.get('debug_plots_enabled', False)
        self.plot_save_path = Path(kwargs.get('plot_save_path', 'debug_plots'))
        if self.debug_plots_enabled:
            self.plot_save_path.mkdir(parents=True, exist_ok=True)
        # Initialize state attributes
        self.key_levels: Dict[str, Dict[str, List[Dict]]] = {}  # E.g. {'EURUSD': {'support': [], 'resistance': []}}
        self.trend_lines: Dict[str, Dict[str, List[Dict]]] = {} # E.g. {'EURUSD': {'support': [], 'resistance': []}}
        self.consolidation_ranges: Dict[str, List[Dict]] = {} # E.g. {'EURUSD': []}
        self.last_level_update: Dict[str, datetime] = {}
        self.last_consolidation_update: Dict[str, datetime] = {}
        self.active_trades: Dict[str, List[Dict]] = {}
        self.retest_conditions: Dict[str, List[Dict]] = {}
        self._scorer = _SignalScorer(self)
        self.backtest_mode = backtest_mode
        self._load_timeframe_profile() # Ensure tf_profile is set during __init__

    def _load_timeframe_profile(self):
        """Loads strategy parameters based on the primary timeframe."""
        # Get profile for current timeframe or use default
        profile = TIMEFRAME_PROFILES.get(self.primary_timeframe)
        
        # Define a default profile in case the requested timeframe is not found
        default_profile = {
            "lookback_period": 100,
            "max_retest_bars": 20,
            "level_update_hours": 24,
            "consolidation_bars": 5,
            "candles_to_check": 10,
            "consolidation_update_hours": 4,
            "atr_multiplier": 1.0,
            "volume_percentile": 80,
            "min_risk_reward": 2.0
        }
        
        # Use the profile if it exists, otherwise use default
        if profile is None:
            logger.warning(f"⚠️ No profile found for {self.primary_timeframe}, using default profile")
            profile = default_profile
            profile_name = "default"
        else:
            profile_name = self.primary_timeframe
            logger.info(f"✅ Found profile for {self.primary_timeframe} timeframe")
        
        # Set all parameters from the profile
        self.lookback_period = profile["lookback_period"]
        self.max_retest_bars = profile["max_retest_bars"]
        self.level_update_hours = profile.get("level_update_hours", default_profile["level_update_hours"])
        self.consolidation_bars = profile["consolidation_bars"]
        self.candles_to_check = profile["candles_to_check"]
        self.consolidation_update_hours = profile.get("consolidation_update_hours", default_profile["consolidation_update_hours"])
        
        # Set ATR multiplier and volume percentile
        self.atr_multiplier = profile["atr_multiplier"]
        self.volume_percentile = profile["volume_percentile"]
        
        # Set min_risk_reward from profile (important for proper risk management)
        self.min_risk_reward = profile.get("min_risk_reward", default_profile["min_risk_reward"])
        
        logger.info(f"⚙️ Loaded {profile_name} profile for {self.primary_timeframe} timeframe")
        logger.debug(f"📊 Profile settings: lookback={self.lookback_period}, consolidation_bars={self.consolidation_bars}, " +
                    f"candles_to_check={self.candles_to_check}, atr_multiplier={self.atr_multiplier}, " +
                    f"min_risk_reward={self.min_risk_reward}, volume_percentile={self.volume_percentile}, " +
                    f"max_retest_bars={self.max_retest_bars}, level_update_hours={self.level_update_hours}")
    
    async def initialize(self):
        """Initialize resources needed by the strategy."""
        logger.info(f"🔌 Initializing {self.name}")
        # No specific initialization needed
        return True
    
    async def generate_signals(self, market_data: Dict[str, Any], symbol: Optional[str] = None, **kwargs) -> List[Dict]:
        start_time_gs = time.time() # Renamed to avoid conflict
        current_time_gs_ts = time.time() # Renamed to avoid conflict, gs for generate_signals, ts for timestamp
        signals = []
        all_signals = []

        logger.debug(f"[StrategyInit] {self.__class__.__name__}: required_timeframes={self.required_timeframes}")
        logger.debug(f"[BreakoutReversalStrategy] Analyzing symbol(s): {list(market_data.keys()) if market_data else symbol} | primary_timeframe={self.primary_timeframe}, higher_timeframe={self.higher_timeframe}")
        logger.info(f"🚀 SIGNAL GENERATION START: {self.name} strategy")
        
        if not market_data:
            logger.warning("⚠️ No market data provided to generate signals")
            return []
            
        debug_visualize = kwargs.get('debug_visualize', False)
        force_trendlines = kwargs.get('force_trendlines', False)
        skip_plots = kwargs.get('skip_plots', False)
        process_immediately = kwargs.get('process_immediately', False)
        
        if debug_visualize:
            logger.info("🔍 Debug visualization mode enabled - will force trendline updates with plots")
        elif force_trendlines:
            logger.info("🔄 Forcing trendline updates without plots")
            
        logger.info(f"🔍 Generating signals with {self.name} strategy for {len(market_data)} symbols")
        
        for symbol_loop_var in market_data:
            symbol_start_time_loop = time.time() # Renamed
            logger.debug(f"📊 Market data for {symbol_loop_var} contains timeframes: {list(market_data[symbol_loop_var].keys())}")
            
            if not all(tf in market_data[symbol_loop_var] for tf in self.required_timeframes):
                missing_tfs = [tf for tf in self.required_timeframes if tf not in market_data[symbol_loop_var]]
                logger.debug(f"⏩ Missing required timeframes for {symbol_loop_var}: {missing_tfs}, skipping")
                continue
                
            primary_raw = market_data[symbol_loop_var].get(self.primary_timeframe)
            primary_df = _to_dataframe(primary_raw, self.primary_timeframe)
            primary_df = _ensure_datetime_index(primary_df, self.primary_timeframe)

            if primary_df is not None and not primary_df.empty:
                last_bar_timestamp_str = str(primary_df.index[-1])
                bar_key = (symbol_loop_var, self.primary_timeframe)
                if self.processed_bars.get(bar_key) == last_bar_timestamp_str:
                    logger.debug(f"[{symbol_loop_var}/{self.primary_timeframe}] Bar at {last_bar_timestamp_str} already processed in this session. Skipping symbol.")
                    continue 
                self.processed_bars[bar_key] = last_bar_timestamp_str 
            else:
                logger.debug(f"[{symbol_loop_var}/{self.primary_timeframe}] No valid primary_df. Skipping analysis for this symbol.")
                continue

            higher_df = None
            if self.higher_timeframe and self.higher_timeframe != self.primary_timeframe:
                higher_raw = market_data[symbol_loop_var].get(self.higher_timeframe)
                higher_df = _to_dataframe(higher_raw, self.higher_timeframe)
                higher_df = _ensure_datetime_index(higher_df, self.higher_timeframe)

            primary_df = add_luxalgo_patterns(primary_df)
            if higher_df is not None:
                higher_df = add_luxalgo_patterns(higher_df)
            
            if higher_df is None or higher_df.empty: # primary_df already checked
                logger.debug(f"⏩ Higher timeframe DataFrame is None or empty for {symbol_loop_var}, skipping.")
                continue
            
            self._update_key_levels(symbol_loop_var, primary_df, debug_force_update=(debug_visualize or force_trendlines))
            self._find_trend_lines(symbol_loop_var, primary_df, debug_force_update=(debug_visualize or force_trendlines), skip_plots=skip_plots)
            self._identify_consolidation_ranges(symbol_loop_var, primary_df)
            self._process_retest_conditions(symbol_loop_var, primary_df)
            
            breakout_signals = self._check_breakout_signals(symbol_loop_var, primary_df, higher_df, skip_plots, processed_zones=self.processed_zones, signal_cooldown=self.signal_cooldown, current_time=current_time_gs_ts)
            reversal_signals = self._check_reversal_signals(symbol_loop_var, primary_df, higher_df, skip_plots, processed_zones=self.processed_zones, signal_cooldown=self.signal_cooldown, current_time=current_time_gs_ts)
            
            symbol_signals_loop = [] # Renamed
            if breakout_signals:
                symbol_signals_loop.extend(breakout_signals)
            if reversal_signals:
                symbol_signals_loop.extend(reversal_signals)
                
            symbol_signals_loop = self._score_signals(symbol_signals_loop, primary_df, higher_df)
            
            if process_immediately and symbol_signals_loop:
                best_signal = max(symbol_signals_loop, key=lambda x: x.get('score', 0))
                for signal_item_loop in symbol_signals_loop: # Renamed
                    logger.debug(f"Signal {signal_item_loop['direction']} for {symbol_loop_var}: {signal_item_loop.get('reason', 'No reason')} - Score: {signal_item_loop.get('score', 0):.2f}")
                logger.info(f"🌟 Selected best signal for {symbol_loop_var}: {best_signal['direction']} {best_signal.get('reason', 'No reason')} with score {best_signal.get('score', 0):.2f}")
                if 'original_symbol' in best_signal: del best_signal['original_symbol']
                if 'score_details' in best_signal: del best_signal['score_details']
                symbol_time_loop = time.time() - symbol_start_time_loop # Renamed
                logger.info(f"📊 Generated signal for {symbol_loop_var} in {symbol_time_loop:.2f}s: {best_signal['direction']} at {best_signal['entry_price']:.5f} | confidence: {best_signal['confidence']:.2f}")
                logger.info(f"👉 RETURNING IMMEDIATE SIGNAL FOR {symbol_loop_var}")
                return [best_signal]
            
            all_signals.extend(symbol_signals_loop)
        
        if all_signals:
            logger.info(f"👉 Found {len(all_signals)} potential signals before scoring and selection")
            for signal_as in all_signals: # Renamed
                symbol_as = signal_as.get('symbol') # Renamed
                consolidation_info = self.last_consolidation_ranges.get(symbol_as, {})
                is_consolidation = consolidation_info.get('is_consolidation', False)
                range_high = consolidation_info.get('high')
                range_low = consolidation_info.get('low')
                if signal_as.get('type', '').upper() == 'BREAKOUT' and is_consolidation:
                    entry = signal_as.get('entry_price')
                    direction = signal_as.get('direction', '').lower()
                    if direction == 'buy' and range_high is not None and entry is not None and entry > range_high:
                        signal_as['consolidation_breakout'] = True
                        signal_as['reason'] += ' | Breakout from consolidation zone'
                        signal_as['score'] = min(1.0, signal_as.get('score', 0) + 0.1)
                    elif direction == 'sell' and range_low is not None and entry is not None and entry < range_low:
                        signal_as['consolidation_breakout'] = True
                        signal_as['reason'] += ' | Breakout from consolidation zone'
                        signal_as['score'] = min(1.0, signal_as.get('score', 0) + 0.1)
            signals_by_symbol = {}
            for signal_sbs in all_signals: # Renamed
                symbol_sbs = signal_sbs['original_symbol'] # Renamed
                if symbol_sbs not in signals_by_symbol:
                    signals_by_symbol[symbol_sbs] = []
                signals_by_symbol[symbol_sbs].append(signal_sbs)
            for symbol_final, symbol_signals_final in signals_by_symbol.items(): # Renamed
                if not symbol_signals_final: continue
                best_signal_final = max(symbol_signals_final, key=lambda x: x.get('score', 0)) # Renamed
                for signal_item_final in symbol_signals_final: # Renamed
                    logger.debug(f"Signal {signal_item_final['direction']} for {symbol_final}: {signal_item_final.get('reason', 'No reason')} - Score: {signal_item_final.get('score', 0):.2f}")
                logger.info(f"🌟 Selected best signal for {symbol_final}: {best_signal_final['direction']} {best_signal_final.get('reason', 'No reason')} with score {best_signal_final.get('score', 0):.2f}")
                if 'original_symbol' in best_signal_final: del best_signal_final['original_symbol']
                if 'score_details' in best_signal_final: del best_signal_final['score_details']
                signals.append(best_signal_final)
        
        generation_time_gs = time.time() - start_time_gs # Renamed
        logger.info(f"✅ Generation completed in {generation_time_gs:.2f}s - Produced {len(signals)} final signals")
        if signals:
            for i, signal_ret in enumerate(signals): # Renamed
                logger.info(f"📊 Final Signal #{i+1}: {signal_ret['symbol']} {signal_ret['direction']} at {signal_ret['entry_price']:.5f} | confidence: {signal_ret['confidence']:.2f}")
            logger.info(f"👉 RETURNING {len(signals)} SIGNALS FOR PROCESSING")
        else:
            logger.info("📭 No signals generated - returning empty list")
        
        cleanup_delay_seconds = self.signal_cooldown * 2
        cleanup_threshold_time = current_time_gs_ts - cleanup_delay_seconds
        old_zone_keys = [k for k, v in self.processed_zones.items() if v < cleanup_threshold_time]
        for k_zone_cleanup in old_zone_keys: # Renamed
            del self.processed_zones[k_zone_cleanup]
        if old_zone_keys:
            logger.debug(f"[DEDUP] Cleaned up {len(old_zone_keys)} old zone records")

        bar_expiry_seconds = self.bar_expiry_hours * 3600
        keys_to_delete_bars = [] # Renamed
        current_dt_for_cleanup = datetime.fromtimestamp(current_time_gs_ts)
        for bar_key_cleanup, timestamp_str_cleanup in self.processed_bars.items(): # Renamed
            try:
                processed_dt = pd.to_datetime(timestamp_str_cleanup).to_pydatetime()
                if (current_dt_for_cleanup - processed_dt).total_seconds() > bar_expiry_seconds:
                    keys_to_delete_bars.append(bar_key_cleanup)
            except Exception as e:
                logger.warning(f"[DEDUP] Error parsing timestamp for processed_bars key {bar_key_cleanup} ('{timestamp_str_cleanup}'): {e}. This entry might not be cleaned correctly based on time.")
        for k_bar_delete in keys_to_delete_bars: # Renamed
            if k_bar_delete in self.processed_bars: 
                 del self.processed_bars[k_bar_delete]
        if keys_to_delete_bars:
            logger.debug(f"[DEDUP] Cleaned up {len(keys_to_delete_bars)} bar processing records older than {self.bar_expiry_hours} hours.")
        
        return signals
    
    def _update_key_levels(self, symbol: str, df: pd.DataFrame, debug_force_update: bool = False) -> None:
        """
        Update support and resistance levels for a symbol.
        
        Args:
            symbol: Trading symbol
            df: Price dataframe
            debug_force_update: Force an update regardless of the time interval (for debugging)
        """
        # Check if we need to update levels (limit computation)
        current_time = df.index[-1]
        
        # Ensure current_time is a datetime object
        if not isinstance(current_time, datetime):
            logger.debug(f"Converting current_time from {type(current_time)} to datetime")
            try:
                if isinstance(current_time, (int, np.integer, float)):
                    try:
                        current_time = datetime.fromtimestamp(current_time)
                    except (ValueError, OverflowError):
                        try:
                            current_time = datetime.fromtimestamp(current_time / 1000)
                        except:
                            current_time = datetime.now()
                elif isinstance(current_time, pd.Timestamp):
                    current_time = current_time.to_pydatetime()
                else:
                    current_time = pd.to_datetime(str(current_time)).to_pydatetime()
            except Exception as e:
                logger.debug(f"Failed to convert timestamp: {e}, using current time instead")
                current_time = datetime.now()
        
        last_update = self.last_updated['key_levels'].get(symbol)
        if last_update is not None:
            if not isinstance(last_update, datetime):
                logger.debug(f"Converting last_update from {type(last_update)} to datetime")
                try:
                    if isinstance(last_update, (int, np.integer, float)):
                        try:
                            last_update = datetime.fromtimestamp(last_update)
                        except (ValueError, OverflowError):
                            try:
                                last_update = datetime.fromtimestamp(last_update / 1000)
                            except:
                                last_update = datetime.now() - timedelta(hours=self.level_update_interval + 1)
                    elif isinstance(last_update, pd.Timestamp):
                        last_update = last_update.to_pydatetime()
                    else:
                        last_update = pd.to_datetime(str(last_update)).to_pydatetime()
                except Exception as e:
                    logger.debug(f"Failed to convert last_update: {e}, forcing update")
                    last_update = datetime.now() - timedelta(hours=self.level_update_interval + 1)
                self.last_updated['key_levels'][symbol] = last_update
            try:
                time_diff = (current_time - last_update).total_seconds()
                if time_diff < self.level_update_interval * 3600:
                    logger.debug(f"🕒 Skipping level update for {symbol}, last update was {time_diff/3600:.1f} hours ago")
                    return
            except Exception as e:
                logger.warning(f"Error calculating time difference: {e}. Forcing update.")
        logger.debug(f"🔄 Updating key levels for {symbol} with {len(df)} candles")
        support_levels = self._find_support_levels(df, symbol)
        resistance_levels = self._find_resistance_levels(df, symbol)
        self.support_levels[symbol] = support_levels
        self.resistance_levels[symbol] = resistance_levels
        self.last_updated['key_levels'][symbol] = current_time
        logger.info(f"🔄 Updated key levels for {symbol} - Support: {len(support_levels)}, Resistance: {len(resistance_levels)}")
        if support_levels:
            logger.debug(f"📉 Support levels for {symbol}: {[round(level['zone_max'], 5) for level in support_levels]}")
        if resistance_levels:
            logger.debug(f"📈 Resistance levels for {symbol}: {[round(level['zone_min'], 5) for level in resistance_levels]}")
            
    def _find_trend_lines(self, symbol: str, df: pd.DataFrame, debug_force_update: bool = False, skip_plots: bool = False) -> None:
        """
        Find and validate trend lines for a given symbol.
        Only plot trendlines and raw price series if skip_plots is False.
        """
        
        current_time_tl = datetime.now() # Renamed
        last_update_time_tl = self.last_updated['trend_lines'].get(symbol, None) # Renamed
        force_update_tl = debug_force_update # Renamed

        if (not force_update_tl and last_update_time_tl is not None and 
            (current_time_tl - last_update_time_tl).total_seconds() < self.trend_line_update_interval * 3600):
            logger.debug(f"⏭️ Skipping trend line update for {symbol} - last update: {last_update_time_tl}")
            return

        logger.info(f"🔍 Finding trend lines for {symbol}")
        analyzer = _TrendLineAnalyzer(df, self)
        bullish_trend_lines = analyzer.get_support_lines(skip_plots, swing_window=getattr(self, 'swing_window', 5))
        bearish_trend_lines = analyzer.get_resistance_lines(skip_plots, swing_window=getattr(self, 'swing_window', 5))
        self.bullish_trend_lines[symbol] = bullish_trend_lines
        self.bearish_trend_lines[symbol] = bearish_trend_lines

        if bullish_trend_lines:
            logger.info(f"📈 Found {len(bullish_trend_lines)} bullish trend lines for {symbol}")
        if bearish_trend_lines:
            logger.info(f"📉 Found {len(bearish_trend_lines)} bearish trend lines for {symbol}")

        if skip_plots:
            if bullish_trend_lines:
                logger.debug(f"📈 BULLISH TREND LINES for {symbol} (skipping plots)")
                for i, line_bull in enumerate(bullish_trend_lines): # Renamed
                    logger.debug(f"  📈 Bullish Line #{{i+1}}: Angle={{line_bull['angle']:.2f}}°, r²={{line_bull['r_squared']:.3f}}, Touches={{line_bull['touches']}}")
            if bearish_trend_lines:
                logger.debug(f"📉 BEARISH TREND LINES for {symbol} (skipping plots)")
                for i, line_bear in enumerate(bearish_trend_lines): # Renamed
                    logger.debug(f"  📉 Bearish Line #{{i+1}}: Angle={{line_bear['angle']:.2f}}°, r²={{line_bear['r_squared']:.3f}}, Touches={{line_bear['touches']}}")
            self.last_updated['trend_lines'][symbol] = current_time_tl
            return
        
        debug_dir = Path("debug_plots")
        debug_dir.mkdir(exist_ok=True)
        plt.figure(figsize=(15, 10))
        plot_range = min(200, len(df))
        x_dates = df.index[-plot_range:]
        plt.plot(x_dates, df['close'].iloc[-plot_range:], color='blue', alpha=0.5, label='Close Price')
        plt.plot(x_dates, df['high'].iloc[-plot_range:], color='green', alpha=0.3, label='High')
        plt.plot(x_dates, df['low'].iloc[-plot_range:], color='red', alpha=0.3, label='Low')
        
        swing_highs, swing_lows = analyzer.find_swings()
        if swing_highs:
            high_x = [df.index[x_val] for x_val, y_val in swing_highs if x_val >= len(df) - plot_range] # Renamed
            high_y = [y_val for x_val, y_val in swing_highs if x_val >= len(df) - plot_range] # Renamed
            plt.scatter(high_x, high_y, color='green', marker='^', s=50, label='Swing Highs')
            if len(high_x) >= 2:
                for i_sh in range(len(high_x) - 1): # Renamed
                    plt.plot([high_x[i_sh], high_x[i_sh+1]], [high_y[i_sh], high_y[i_sh+1]], color='lightgreen', linestyle='--', alpha=0.5)
        if swing_lows:
            low_x = [df.index[x] for x, y in swing_lows if x >= len(df) - plot_range]
            low_y = [y for x, y in swing_lows if x >= len(df) - plot_range]
            plt.scatter(low_x, low_y, color='red', marker='v', s=50, label='Swing Lows')
            if len(low_x) >= 2:
                for i in range(len(low_x) - 1):
                    plt.plot([low_x[i], low_x[i+1]], [low_y[i], low_y[i+1]], color='lightcoral', linestyle='--', alpha=0.5)
        # Plot bullish trend lines (support)
        for i, line in enumerate(bullish_trend_lines[:8]):
            # Use timestamps for x-axis
            start_idx = max(line['start_idx'], len(df) - plot_range)
            end_idx = min(line['end_idx'], len(df) - 1)
            if start_idx >= end_idx:
                continue
            x_start = df.index[start_idx].timestamp()
            x_end = df.index[end_idx].timestamp()
            x_line = np.linspace(x_start, x_end, 100)
            line_values = np.multiply(line['slope'], x_line) + line['intercept']
            x_line_dt = [datetime.fromtimestamp(x) for x in x_line]
            plt.plot(x_line_dt, line_values, color='green', linewidth=2, alpha=0.7, label=f"Support: Angle={line['angle']:.1f}°, Touches={line['touches']}")
            if i < 3:
                midpoint_x = (x_start + x_end) / 2
                midpoint_y = line['slope'] * midpoint_x + line['intercept']
                # Stagger annotation y-offsets to reduce overlap
                y_offset = -20 if i % 2 == 0 else 20
                y_offset += (i // 2) * 10 * (-1 if i % 2 == 0 else 1)
                plt.annotate(f"Support #{i+1}: {line['touches']} touches", xy=(datetime.fromtimestamp(midpoint_x), midpoint_y), xytext=(-30, y_offset), textcoords="offset points", bbox=dict(boxstyle="round", fc="white", alpha=0.7), arrowprops=dict(arrowstyle="->"))
        # Plot bearish trend lines (resistance)
        for i, line in enumerate(bearish_trend_lines[:8]):
            start_idx = max(line['start_idx'], len(df) - plot_range)
            end_idx = min(line['end_idx'], len(df) - 1)
            if start_idx >= end_idx:
                continue
            x_start = df.index[start_idx].timestamp()
            x_end = df.index[end_idx].timestamp()
            x_line = np.linspace(x_start, x_end, 100)
            line_values = np.multiply(line['slope'], x_line) + line['intercept']
            x_line_dt = [datetime.fromtimestamp(x) for x in x_line]
            plt.plot(x_line_dt, line_values, color='red', linewidth=2, alpha=0.7, label=f"Resistance: Angle={line['angle']:.1f}°, Touches={line['touches']}")
            if i < 3:
                midpoint_x = (x_start + x_end) / 2
                midpoint_y = line['slope'] * midpoint_x + line['intercept']
                # Stagger annotation y-offsets to reduce overlap
                y_offset = 20 if i % 2 == 0 else -20
                y_offset += (i // 2) * 10 * (1 if i % 2 == 0 else -1)
                plt.annotate(f"Resistance #{i+1}: {line['touches']} touches", xy=(datetime.fromtimestamp(midpoint_x), midpoint_y), xytext=(-30, y_offset), textcoords="offset points", bbox=dict(boxstyle="round", fc="white", alpha=0.7), arrowprops=dict(arrowstyle="->"))
        support_levels = self.support_levels.get(symbol, [])
        for level in support_levels:
            plt.axhline(y=level['zone_max'], color='green', linestyle='-', alpha=0.3)
        resistance_levels = self.resistance_levels.get(symbol, [])
        for level in resistance_levels:
            plt.axhline(y=level['zone_min'], color='red', linestyle='-', alpha=0.3)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.title(f'Trend Line Analysis for {symbol}')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper left', fontsize='small', title='Legend\nTrendline: Angle, Touches')
        file_path = debug_dir / f"{symbol}_trend_lines_{timestamp}.png"
        plt.savefig(file_path)
        plt.close()
        logger.info(f"📊 Saved trend line visualization to {file_path}")
        self.last_updated['trend_lines'][symbol] = current_time_tl
    
    def _find_swing_highs(self, df: pd.DataFrame, window: int = 5) -> list:
        """Classic Bill Williams fractal swing-high detection with ATR significance filter."""
        if df is None or len(df) < 2 * window + 1:
            return []
        highs = np.asarray(df['high'].values, dtype=np.float64)
        lows = np.asarray(df['low'].values, dtype=np.float64)
        closes = np.asarray(df['close'].values, dtype=np.float64)
        atr = talib.ATR(highs, lows, closes, timeperiod=14)
        sig_thresh = atr[-1] * 0.25 if len(atr) > 0 and not np.isnan(atr[-1]) else 0
        pivots = []
        last_pivot_price = None
        for i in range(window, len(df) - window):
            left = highs[i - window:i]
            right = highs[i + 1:i + window + 1]
            if all(highs[i] > h for h in left) and all(highs[i] > h for h in right):
                if last_pivot_price is None or abs(highs[i] - last_pivot_price) >= sig_thresh:
                    pivots.append((i, highs[i]))
                    last_pivot_price = highs[i]
        return pivots

    def _find_swing_lows(self, df: pd.DataFrame, window: int = 5) -> list:
        """Classic Bill Williams fractal swing-low detection with ATR significance filter."""
        if df is None or len(df) < 2 * window + 1:
            return []
        highs = np.asarray(df['high'].values, dtype=np.float64)
        lows = np.asarray(df['low'].values, dtype=np.float64)
        closes = np.asarray(df['close'].values, dtype=np.float64)
        atr = talib.ATR(highs, lows, closes, timeperiod=14)
        sig_thresh = atr[-1] * 0.25 if len(atr) > 0 and not np.isnan(atr[-1]) else 0
        pivots = []
        last_pivot_price = None
        for i in range(window, len(df) - window):
            left = lows[i - window:i]
            right = lows[i + 1:i + window + 1]
            if all(lows[i] < l for l in left) and all(lows[i] < l for l in right):
                if last_pivot_price is None or abs(lows[i] - last_pivot_price) >= sig_thresh:
                    pivots.append((i, lows[i]))
                    last_pivot_price = lows[i]
        return pivots
    
    def _identify_trend_lines(self, df: pd.DataFrame, swing_points: list, line_type: str, skip_plots: bool = False) -> list:
        """
        Fit multiple trend lines to recent combinations of swing points (e.g., last 3, 4, 5).
        Return a list of valid trend lines with metadata (start/end idx, r_squared, angle, touches, recency, quality_score).
        """
        import numpy as np
        from scipy.stats import linregress
        import math
        min_points = getattr(self, 'trend_line_min_points', 3)
        max_angle = getattr(self, 'trend_line_max_angle', 45)
        r2_threshold = getattr(self, 'trend_line_r2_threshold', 0.6)
        max_lines = 8
        if len(swing_points) < min_points:
            self.logger.debug(f"Not enough swing points ({len(swing_points)}) to identify {line_type} trend lines. Need at least {min_points}.")
            return []
        lines = []
        # Try fitting lines to last 3, 4, 5, ... up to all points (but prioritize recency)
        for n in range(min_points, min(len(swing_points), min_points + 3) + 1):
            pts = swing_points[-n:]
            x = np.array([p[0] for p in pts])
            y = np.array([p[1] for p in pts])
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            r_squared = float(r_value) ** 2
            angle = math.degrees(math.atan(slope)) if slope is not None else 0.0
            start_idx = int(x[0])
            end_idx = int(x[-1])
            # Count touches: how many swing points are within tolerance of the line
            tol = np.std(y) * 0.25 if len(y) > 1 else 0.0001
            line_y = slope * x + intercept
            touches = int(np.sum(np.abs(y - line_y) <= tol))
            # Recency: average index of points used, normalized (closer to end = more recent)
            recency_score = np.mean(x) / len(df) if len(df) > 0 else 0
            # Only keep lines that meet minimum criteria
            if touches >= min_points and abs(angle) < max_angle and r_squared >= r2_threshold:
                quality_score = r_squared * touches * (1 + recency_score)
                lines.append({
                    'slope': slope,
                    'intercept': intercept,
                    'r_squared': r_squared,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'line_type': line_type,
                    'points': pts,
                    'quality_score': quality_score,
                    'angle': angle,
                    'x_start': float(x[0]),
                    'x_end': float(x[-1]),
                    'touches': touches,
                    'recency_score': recency_score,
                })
        # Sort by quality_score descending, return up to max_lines
        lines = sorted(lines, key=lambda l: l['quality_score'], reverse=True)[:max_lines]
        return lines
    
    def _count_trend_line_touches(self, df: pd.DataFrame, slope: float, intercept: float, 
                                  line_type: str, atr: float = None, start_idx: int = None, end_idx: int = None) -> int:
        """
        Fully vectorized: Count how many times price has touched a trend line, using ATR-based tolerance and only between start_idx and end_idx.
        """
        import numpy as np
        if atr is None or pd.isna(atr) or atr == 0:
            # Ensure the calculation result is explicitly float
            std_val = df['close'].rolling(14).std().iloc[-1] if len(df) >= 14 else df['close'].std()
            atr = float(std_val) if pd.notna(std_val) else 0.0
        if atr is None or pd.isna(atr) or atr == 0: # Check again in case previous assignment resulted in 0 or NaN
            atr = 1e-4
        price_series = df['low'] if line_type == 'bullish' else df['high']
        if start_idx is None or end_idx is None:
            start_idx = 0
            end_idx = len(df) - 1
        idx_range = np.arange(start_idx, end_idx + 1)
        # Vectorized x values (timestamps or index)
        index_slice = df.index[idx_range]
        if hasattr(index_slice, 'to_numpy'):
            x_vals = np.array([ts.timestamp() if hasattr(ts, 'timestamp') else float(i) for i, ts in zip(idx_range, index_slice)])
        else:
            x_vals = np.array(idx_range, dtype=float)
        line_values = np.multiply(slope, x_vals) + intercept
        prices = price_series.iloc[idx_range].to_numpy(dtype=float)
        diffs = np.abs(prices - line_values)
        touches = np.count_nonzero(diffs <= atr * 0.25)
        return int(min(touches, 20))
    
    def _is_near_trend_line(self, df: pd.DataFrame, idx: int, trend_lines: List[Dict], 
                           line_type: str) -> Optional[Dict]:
        """
        Check if price is near a trend line, only within the valid segment (between x_start and x_end).
        """
        if not trend_lines:
            return None
        current_candle = df.iloc[idx]
        ts = df.index[idx]
        if isinstance(ts, pd.Timestamp):
            x = ts.timestamp()
        elif isinstance(ts, datetime):
            x = ts.timestamp()
        else:
            x = float(idx)
        for trend_line in trend_lines:
            # Only consider within valid segment
            if not (trend_line['x_start'] <= x <= trend_line['x_end']):
                continue
            line_value = trend_line['slope'] * x + trend_line['intercept']
            tolerance = current_candle['close'] * self.price_tolerance
            if line_type == 'bullish':
                if abs(current_candle['low'] - line_value) <= tolerance:
                    return trend_line
            else:
                if abs(current_candle['high'] - line_value) <= tolerance:
                    return trend_line
        return None
    
    def _calculate_trend_line_value(self, trend_line: Dict, idx: int) -> float:
        """
        Calculate the y-value of a trend line at a given index.
        
        Args:
            trend_line: Trend line dictionary
            idx: Index to calculate value for
            
        Returns:
            Price value of trend line at index
        """
        return trend_line['slope'] * idx + trend_line['intercept']
    
    def _identify_consolidation_ranges(self, symbol: str, df: pd.DataFrame) -> None:

        # Always set current_time from the last index
        current_time = df.index[-1]
        # Ensure current_time is a datetime object
        if not isinstance(current_time, datetime):
            logger.debug(f"Converting current_time from {type(current_time)} to datetime in consolidation ranges")
            if isinstance(current_time, (int, np.integer, float)):
                try:
                    current_time = pd.to_datetime(int(current_time) if isinstance(current_time, np.integer) else float(current_time), unit='s')
                except:
                    try:
                        current_time = pd.to_datetime(int(current_time) if isinstance(current_time, np.integer) else float(current_time), unit='ms')
                    except:
                        current_time = datetime.now()
                        logger.debug(f"Failed to convert timestamp, using current time instead")
        
        if symbol in self.last_consolidation_ranges:
            last_update = self.last_updated['consolidation_ranges'].get(symbol, datetime.min)
            
            # Ensure last_update is a datetime object
            if not isinstance(last_update, datetime):
                logger.debug(f"Converting last_update from {type(last_update)} to datetime in consolidation ranges")
                if isinstance(last_update, (int, np.integer, float)):
                    try:
                        # Fix: Explicitly cast to int or float for pd.to_datetime
                        last_update = pd.to_datetime(int(last_update) if isinstance(last_update, np.integer) else float(last_update), unit='s')
                    except:
                        try:
                            # Fix: Explicitly cast to int or float for pd.to_datetime
                            last_update = pd.to_datetime(int(last_update) if isinstance(last_update, np.integer) else float(last_update), unit='ms')
                        except:
                            # If conversion fails, use a time far in the past to force update
                            last_update = datetime.now() - timedelta(hours=self.range_update_interval + 1)
                            logger.debug(f"Failed to convert last_update, forcing update in consolidation ranges")
                # Update the stored value
                self.last_updated['consolidation_ranges'][symbol] = last_update
            
            # Only update after significant time has passed based on timeframe
            try:
                # Initialize time_diff to ensure it's always defined
                time_diff = 0
                # Fix: Ensure both values are datetime objects before subtraction
                if isinstance(current_time, datetime) and isinstance(last_update, datetime):
                    time_diff = (current_time - last_update).total_seconds()
                if time_diff < self.range_update_interval * 3600:
                    logger.debug(f"🕒 Skipping consolidation range update for {symbol}, last update was {time_diff/3600:.1f} hours ago")
                    return
                else:
                    # Force update if types are incompatible
                    logger.debug(f"Time comparison types incompatible: {type(current_time)} vs {type(last_update)}. Forcing update.")
            except Exception as e:
                logger.warning(f"Error calculating time difference in consolidation ranges: {e}. Forcing update.")
                # Continue with update in case of error
        
        # Calculate volatility metrics for dynamic consolidation detection
        try:
            # Calculate ATR for volatility reference
            high = np.asarray(df['high'].values, dtype=np.float64)
            low = np.asarray(df['low'].values, dtype=np.float64)
            close = np.asarray(df['close'].values, dtype=np.float64)
            atr_series = talib.ATR(high, low, close, timeperiod=self.atr_period)
            if not isinstance(atr_series, pd.Series) and len(atr_series) == 0:
                atr = None
            else:
                atr = atr_series[-1] if len(atr_series) > 0 else None
            if self._is_invalid_or_zero(atr):
                logger.warning(f"ATR calculation failed for {symbol}, using traditional approach")
                # Fallback to traditional approach with fixed bar count
                recent_bars = df.iloc[-self.consolidation_bars:]
                if len(recent_bars) > 0:  # Check if DataFrame is not empty
                    range_high = float(recent_bars['high'].max())
                    range_low = float(recent_bars['low'].min())
                    range_size = range_high - range_low
                    is_consolidation = True  # Assume it's consolidation with traditional approach
                else:
                    logger.warning(f"Recent bars DataFrame is empty for {symbol}")
                    range_high = 0
                    range_low = 0
                    range_size = 0
                    is_consolidation = False
            else:
                # Use rolling calculations for better consolidation detection
                # First, calculate the range of each candle
                lookback = min(30, len(df) - 5)  # Use last 30 bars or as many as available
                df_subset = df.iloc[-lookback:].copy()
                df_subset['candle_range'] = df_subset['high'] - df_subset['low']
                
                # Calculate the rolling standard deviation of price
                df_subset['close_std'] = df_subset['close'].rolling(window=10).std()
                
                # Calculate average range
                if len(df_subset) > 0 and 'candle_range' in df_subset.columns:  # Check if DataFrame is not empty
                    avg_range = float(df_subset['candle_range'].mean())
                else:
                    logger.warning(f"DataFrame subset is empty or missing candle_range column for {symbol}")
                    avg_range = 0
                
                # Get recent bars (potentially in consolidation)
                recent_period = min(self.consolidation_bars, len(df_subset) - 1)
                recent_bars = df_subset.iloc[-recent_period:]
                
                # Calculate recent volatility metrics
                if len(recent_bars) > 0 and 'close_std' in recent_bars.columns and 'candle_range' in recent_bars.columns:
                    recent_std = float(recent_bars['close_std'].mean())
                    recent_range_avg = float(recent_bars['candle_range'].mean())
                    
                    # Calculate the range high and low
                    range_high = float(recent_bars['high'].max())
                    range_low = float(recent_bars['low'].min())
                    range_size = range_high - range_low
                    
                    # Compare recent volatility to ATR
                    if avg_range > 0:  # Prevent division by zero
                        volatility_ratio = recent_range_avg / avg_range
                    else:
                        volatility_ratio = 1.0
                    
                    cond1 = bool(volatility_ratio < 0.5)
                    
                    # Make sure atr is not None before multiplying
                    if atr is not None and not pd.isna(atr):
                        cond2 = bool(recent_std < atr * 0.5)
                        cond3 = bool(range_size < atr * 3)  # Range shouldn't be more than 3x ATR
                    else:
                        # Fallback using recent standard deviation as reference
                        cond2 = bool(recent_std < recent_range_avg * 0.5)
                        cond3 = bool(range_size < recent_range_avg * 5)  # Use recent range as fallback
                    
                    is_consolidation = cond1 and cond2 and cond3
                    
                    logger.debug(f"📊 {symbol}: Volatility analysis - Avg range: {avg_range:.5f}, Recent avg: {recent_range_avg:.5f}, " +
                               f"Ratio: {volatility_ratio:.2f}, ATR: {atr:.5f}, Is consolidation: {is_consolidation}")
                else:
                    logger.warning(f"Recent bars DataFrame is empty or missing columns for {symbol}")
                    range_high = 0
                    range_low = 0
                    range_size = 0
                    is_consolidation = False
            
            clusters = []
            volume_profile = []
            try:
                from sklearn.cluster import KMeans
                use_kmeans = True
            except ImportError:
                use_kmeans = False
            # Use closes for clustering, fallback to highs/lows if needed
            price_values = recent_bars['close'].values if 'close' in recent_bars else recent_bars['high'].values
            price_values = np.array(price_values, dtype=float)
            volumes = recent_bars['tick_volume'].values if 'tick_volume' in recent_bars else np.ones_like(price_values)
            n_clusters = min(4, max(2, len(price_values)//5))
            if use_kmeans and len(price_values) >= n_clusters:
                # KMeans clustering on price
                kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
                price_reshape = price_values.reshape(-1, 1)
                labels = kmeans.fit_predict(price_reshape)
                for c in range(n_clusters):
                    mask = labels == c
                    if np.sum(mask.astype(int)) == 0:
                        continue
                    cluster_prices = price_values[mask]
                    cluster_vols = volumes[mask]
                    clusters.append({
                        'min': float(np.min(cluster_prices)),
                        'max': float(np.max(cluster_prices)),
                        'mean': float(np.mean(cluster_prices)),
                        'total_volume': float(np.sum(cluster_vols)),
                        'count': int(np.sum(mask))
                    })
            else:
                # Fallback: use _cluster_1d
                tol = np.std(price_values) * 0.5 if len(price_values) > 1 else 0.0001
                cluster_lists = self._cluster_1d(list(price_values), tol)
                for cl in cluster_lists:
                    cl = np.array(cl, dtype=float)
                    mask = np.isin(price_values, cl)
                    clusters.append({
                        'min': float(np.min(cl)),
                        'max': float(np.max(cl)),
                        'mean': float(np.mean(cl)),
                        'total_volume': float(np.sum(volumes[mask])),
                        'count': int(np.sum(mask))
                    })
            # Volume-by-price histogram (simple)
            n_bins = min(10, max(3, len(price_values)//2))
            if n_bins > 1:
                hist, bin_edges = np.histogram(price_values, bins=n_bins, weights=volumes)
                for i in range(len(hist)):
                    volume_profile.append({
                        'bin_min': float(bin_edges[i]),
                        'bin_max': float(bin_edges[i+1]),
                        'center': float((bin_edges[i]+bin_edges[i+1])/2),
                        'total_volume': float(hist[i])
                    })
            # Add to last_consolidation_ranges
            self.last_consolidation_ranges[symbol] = {
                'high': range_high,
                'low': range_low,
                'size': range_size,
                'is_consolidation': is_consolidation,
                'clusters': clusters,
                'volume_profile': volume_profile
            }
            logger.debug(f"[CLUSTER] {symbol}: Consolidation clusters: {clusters}")
            logger.debug(f"[VOLUME PROFILE] {symbol}: Volume-by-price: {volume_profile}")
            if is_consolidation:
                logger.info(f"📏 Identified consolidation range for {symbol}: High={range_high:.5f}, Low={range_low:.5f}, Size={range_size:.5f}")
            else:
                logger.debug(f"📏 Detected non-consolidation range for {symbol}: High={range_high:.5f}, Low={range_low:.5f}, Size={range_size:.5f}")
                
        except Exception as e:
            logger.warning(f"Error in consolidation detection for {symbol}: {str(e)}")
            # In case of error, use a simple fallback
            try:
                recent_bars = df.iloc[-self.consolidation_bars:]
                if len(recent_bars) > 0:  # Check if DataFrame is not empty
                    range_high = float(recent_bars['high'].max())
                    range_low = float(recent_bars['low'].min())
                    range_size = range_high - range_low
                else:
                    logger.warning(f"Recent bars DataFrame is empty for {symbol} in fallback")
                    range_high = 0
                    range_low = 0
                    range_size = 0
                
                # Store the range without additional metrics
                self.last_consolidation_ranges[symbol] = {
                    'high': range_high,
                    'low': range_low,
                    'size': range_size,
                    'is_consolidation': False  # Conservative approach
                }
                
                logger.debug(f"📏 Fallback consolidation calculation for {symbol}: High={range_high:.5f}, Low={range_low:.5f}, Size={range_size:.5f}")
            except Exception as e2:
                logger.error(f"Fallback consolidation calculation also failed for {symbol}: {str(e2)}")
                # Initialize with empty values to avoid errors later
                self.last_consolidation_ranges[symbol] = {
                    'high': 0,
                    'low': 0,
                    'size': 0,
                    'is_consolidation': False
                }
        self.last_updated['consolidation_ranges'][symbol] = current_time
    
    def _process_retest_conditions(self, symbol: str, df: pd.DataFrame) -> None:
        """
        Process any pending retest conditions for breakout trades using ATR-based dynamic window.
        
        Args:
            symbol: Trading symbol
            df: Price dataframe
        """
        if not self.retest_required or symbol not in self.retest_tracking:
            return
            
        # Get current tracking info
        retest_info = self.retest_tracking[symbol]
        if not retest_info:
            return
            
        current_time = df.index[-1]
        level = retest_info.get('level')
        direction = retest_info.get('direction')
        start_time = retest_info.get('start_time')
        
        # Ensure current_time is a datetime object
        if not isinstance(current_time, datetime):
            logger.debug(f"Converting current_time from {type(current_time)} to datetime in retest conditions")
            if isinstance(current_time, (int, np.integer, float)):
                try:
                    # Fix: Explicitly cast to int or float for pd.to_datetime
                    current_time = pd.to_datetime(int(current_time) if isinstance(current_time, np.integer) else float(current_time), unit='s')
                except:
                    try:
                        # Fix: Explicitly cast to int or float for pd.to_datetime
                        current_time = pd.to_datetime(int(current_time) if isinstance(current_time, np.integer) else float(current_time), unit='ms')
                    except:
                        # If conversion fails, use current time
                        current_time = datetime.now()
                        logger.debug(f"Failed to convert timestamp, using current time instead")
        
        # Ensure start_time is a datetime object
        if not isinstance(start_time, datetime):
            logger.debug(f"Converting start_time from {type(start_time)} to datetime in retest conditions")
            if isinstance(start_time, (int, np.integer, float)):
                try:
                    # Fix: Explicitly cast to int or float for pd.to_datetime
                    start_time = pd.to_datetime(int(start_time) if isinstance(start_time, np.integer) else float(start_time), unit='s')
                except:
                    try:
                        # Fix: Explicitly cast to int or float for pd.to_datetime
                        start_time = pd.to_datetime(int(start_time) if isinstance(start_time, np.integer) else float(start_time), unit='ms')
                    except:
                        # If conversion fails, use a default value
                        start_time = current_time - timedelta(hours=1)  # Arbitrary 1 hour
                        logger.debug(f"Failed to convert start_time, using default value")
                # Update in the tracking dictionary
                retest_info['start_time'] = start_time
        
        if not level or not direction or not start_time:
            logger.warning(f"Missing retest tracking info for {symbol}, clearing")
            self.retest_tracking[symbol] = None
            return
            
        # Check if it's been too long since the level was identified (timeframe dependent)
        try:
            # Initialize time_diff to ensure it's always defined
            time_diff = 0
            # Fix: Ensure both values are datetime objects before subtraction
            if isinstance(current_time, datetime) and isinstance(start_time, datetime):
                time_diff = (current_time - start_time).total_seconds()
            max_time_allowed = self.max_retest_time * 3600  # Convert hours to seconds
            
            if time_diff > max_time_allowed:
                logger.debug(f"⌛ Retest condition expired for {symbol} after {time_diff/3600:.1f} hours (max: {max_time_allowed/3600:.1f})")
                self.retest_tracking[symbol] = None
                return
            else:
                # Cannot perform comparison, log and continue
                logger.debug(f"Time comparison types incompatible: {type(current_time)} vs {type(start_time)}. Continuing anyway.")
        except Exception as e:
            logger.warning(f"Error calculating time difference in retest condition: {e}")
            # Continue processing despite the error
            
        # Calculate ATR for dynamic retest window
        try:
            # Calculate ATR for the current timeframe
            high = np.asarray(df['high'].values, dtype=np.float64)
            low = np.asarray(df['low'].values, dtype=np.float64)
            close = np.asarray(df['close'].values, dtype=np.float64)
            atr_series = talib.ATR(high, low, close, timeperiod=self.atr_period)
            atr = atr_series[-1] if len(atr_series) > 0 else None
            if self._is_invalid_or_zero(atr):
                # Fallback to a price-based tolerance if ATR calculation fails
                logger.warning(f"ATR calculation failed for {symbol}, using price-based tolerance")
                price_tolerance = float(df['close'].iloc[-1]) * self.max_stop_pct
            else:
                # Use ATR with multiplier for dynamic tolerance
                price_tolerance = atr * self.atr_multiplier
                logger.debug(f"Using ATR-based retest window: {price_tolerance:.5f} (ATR: {atr:.5f}, multiplier: {self.atr_multiplier})")
        except Exception as e:
            logger.warning(f"Error calculating ATR for {symbol}: {e}. Using price-based tolerance.")
            # Fallback to price-based tolerance
            price_tolerance = float(df['close'].iloc[-1]) * self.max_stop_pct
            
        # Check if the price has retested the level using ATR-based window
        current_price = df['close'].iloc[-1]
        current_high = df['high'].iloc[-1]

        # For breakout above resistance, we're looking for a retest from above
        if direction == 'bullish' and abs(current_price - level) <= price_tolerance and current_price > level:
            logger.info(f"✅ Confirmed bullish retest of {level:.5f} for {symbol} (ATR window: {price_tolerance:.5f})")
            # Update breakout tracking to indicate retest is confirmed
            retest_info['retest_confirmed'] = True
            self.retest_tracking[symbol] = retest_info
        
        # For breakout below support, we're looking for a retest from below
        elif direction == 'bearish' and abs(current_high - level) <= price_tolerance and current_price < level:
            logger.info(f"✅ Confirmed bearish retest of {level:.5f} for {symbol} (ATR window: {price_tolerance:.5f})")
            # Update breakout tracking to indicate retest is confirmed
            retest_info['retest_confirmed'] = True
            self.retest_tracking[symbol] = retest_info
    
    def _find_support_levels(self, df: pd.DataFrame, symbol: str = None) -> List[dict]:
        """
        Find significant support levels using swing lows.
        Returns a list of support zones (dicts with min, max, avg, width).
        """
        levels = []
        for i in range(2, len(df) - 2):
            if (df['low'].iloc[i] < df['low'].iloc[i-1] and 
                df['low'].iloc[i] < df['low'].iloc[i-2] and
                df['low'].iloc[i] < df['low'].iloc[i+1] and 
                df['low'].iloc[i] < df['low'].iloc[i+2]):
                level = df['low'].iloc[i]
                touches = self._count_level_touches(df, level, 'support')
                if touches >= self.min_level_touches:
                    levels.append(level)
        # --- Integrate volume-by-price clusters as dynamic support zones ---
        volume_profile_zones = []
        if symbol and symbol in self.last_consolidation_ranges:
            vprof = self.last_consolidation_ranges[symbol].get('volume_profile', [])
            if vprof:
                sorted_nodes = sorted(vprof, key=lambda n: n['total_volume'], reverse=True)
                for node in sorted_nodes[:2]:
                    volume_profile_zones.append(node['center'])
        clustered = self._cluster_levels(levels + volume_profile_zones)
        # Mark volume-profile derived zones for traceability
        for zone in clustered:
            if any(abs(zone['zone_avg'] - vp) < zone['zone_width']*2 for vp in volume_profile_zones):
                zone['source'] = 'volume_profile'
        return clustered

    def _find_resistance_levels(self, df: pd.DataFrame, symbol: str = None) -> List[dict]:
        """
        Find significant resistance levels using swing highs.
        Returns a list of resistance zones (dicts with min, max, avg, width).
        """
        levels = []
        for i in range(2, len(df) - 2):
            if (df['high'].iloc[i] > df['high'].iloc[i-1] and 
                df['high'].iloc[i] > df['high'].iloc[i-2] and
                df['high'].iloc[i] > df['high'].iloc[i+1] and 
                df['high'].iloc[i] > df['high'].iloc[i+2]):
                level = df['high'].iloc[i]
                touches = self._count_level_touches(df, level, 'resistance')
                if touches >= self.min_level_touches:
                    levels.append(level)
        # --- Integrate volume-by-price clusters as dynamic resistance zones ---
        volume_profile_zones = []
        if symbol and symbol in self.last_consolidation_ranges:
            vprof = self.last_consolidation_ranges[symbol].get('volume_profile', [])
            if vprof:
                sorted_nodes = sorted(vprof, key=lambda n: n['total_volume'], reverse=True)
                for node in sorted_nodes[:2]:
                    volume_profile_zones.append(node['center'])
        clustered = self._cluster_levels(levels + volume_profile_zones)
        for zone in clustered:
            if any(abs(zone['zone_avg'] - vp) < zone['zone_width']*2 for vp in volume_profile_zones):
                zone['source'] = 'volume_profile'
        return clustered

    def _count_level_touches(self, df: pd.DataFrame, level: float, level_type: str) -> int:
        """
        Vectorized: Count how many times price has touched a level.
        """
        import numpy as np
        tolerance = level * self.price_tolerance
        if level_type == 'support':
            prices = df['low'].to_numpy(dtype=float)
        else:
            prices = df['high'].to_numpy(dtype=float)
        diffs = np.abs(prices - level)
        count = np.sum(diffs <= tolerance)
        return int(count)
    
    def _cluster_levels(self, levels: List[float]) -> List[dict]:
        """Cluster nearby levels to avoid duplicates, returning zones (min, max, avg, width)."""
        tol = self.price_tolerance * max(levels) if levels else 0
        clusters = self.cluster_items(levels, metric=lambda x: x, tol=tol)
        result = []
        for cluster in clusters:
            zone_min = min(cluster)
            zone_max = max(cluster)
            zone_avg = sum(cluster) / len(cluster)
            zone_width = zone_max - zone_min
            result.append({
                'zone_min': zone_min,
                'zone_max': zone_max,
                'zone_avg': zone_avg,
                'zone_width': zone_width
            })
        return result
    
    def _check_breakout_signals(self, symbol: str, df: pd.DataFrame, h1_df: pd.DataFrame, skip_plots: bool = False, processed_zones=None, signal_cooldown=86400, current_time=None) -> List[Dict]:
        """
        Check for breakout signals across support/resistance levels and trend lines.
        
        Args:
            symbol: Trading symbol
            df: Price dataframe for primary timeframe
            h1_df: Price dataframe for higher timeframe
            skip_plots: Whether to skip creating debug plots
            
        Returns:
            List of breakout signal dictionaries
        """
        signals = []
        
        # Skip if no levels available
        if symbol not in self.resistance_levels or symbol not in self.support_levels:
            logger.debug(f"⏩ {symbol}: No resistance or support levels available, skipping breakout check")
            return signals
            
        resistance_levels = self.resistance_levels[symbol]
        support_levels = self.support_levels[symbol]
        
        # Prepare volume and debug information
        self._ensure_tick_volume(df, symbol)
        volume_threshold = self._compute_volume_threshold(df)
        
        # Get trend lines if available
        trend_lines = self.bullish_trend_lines.get(symbol, []) + self.bearish_trend_lines.get(symbol, [])
        bullish_trend_lines = [line for line in trend_lines if line['angle'] < 60]  # Increased from 45
        bearish_trend_lines = [line for line in trend_lines if line['angle'] > -60]  # Increased from -45
        
        logger.debug(f"🔍 {symbol}: Found {len(bullish_trend_lines)} bullish and {len(bearish_trend_lines)} bearish trend lines")
        
        # Get recent candles - use candles_to_check from timeframe profile
        candles_to_check = min(self.candles_to_check, len(df) - 1)
        
        # Get higher timeframe trend
        h1_trend = self._determine_higher_timeframe_trend(h1_df)
        logger.info(f"📈 {symbol}: {self.higher_timeframe} trend is {h1_trend}")
        
        # Check for retest confirmations first
        if (symbol in self.retest_tracking and 
            self.retest_tracking[symbol] and  # Check if entry exists and is not None
            self.retest_tracking[symbol].get('retest_confirmed', False)):
            retest_info = self.retest_tracking[symbol]
            retest_entry = retest_info.get('entry_price')
            retest_direction = retest_info.get('direction')
            retest_level = retest_info.get('level')
            retest_stop = retest_info.get('stop_loss')
            retest_reason = retest_info.get('reason')
            logger.info(f"✅ {symbol}: Retest confirmed for {retest_direction} at level {retest_level:.5f}")
            current_price = df['close'].iloc[-1]
            if retest_direction == 'buy':
                risk = retest_entry - retest_stop
                if symbol in self.last_consolidation_ranges:
                    range_size = self.last_consolidation_ranges[symbol]['size']
                    calculated_target = retest_level + range_size
                    min_target = retest_entry + (risk * self.min_risk_reward)
                    take_profit = max(calculated_target, min_target)
                else:
                    take_profit = retest_entry + (risk * self.min_risk_reward)
                # --- BUY RETEST: Ensure TP is above entry ---
                assert take_profit > retest_entry, "TP for buy must be above entry!"
                signal = {
                    "symbol": symbol,
                    "direction": "buy",
                    "entry_price": retest_entry,
                    "stop_loss": retest_stop,
                    "take_profit": take_profit,
                    "timeframe": self.primary_timeframe,
                    "strategy_name": self.name,  # Changed from "source" to "strategy_name"
                    "generator": self.name,
                    "reason": f"Retest confirmed: {retest_reason}",
                    "size": self.risk_manager.calculate_position_size(
                        account_balance=self.risk_manager.get_account_balance(),
                        risk_per_trade=self.risk_manager.max_risk_per_trade * 100,
                        entry_price=retest_entry,
                        stop_loss_price=retest_stop,
                        symbol=symbol
                    ),
                    "signal_bar_index": len(df) - 1,
                    "signal_timestamp": str(df.index[-1])
                }
                result = self.risk_manager.validate_and_size_trade(signal)
                if result['is_valid']:
                    adjusted_signal = result['final_trade_params']
                    for k in signal:
                        if k not in adjusted_signal:
                            adjusted_signal[k] = signal[k]
                    signals.append(adjusted_signal)
                    logger.info(f"🟢 RETEST BUY: {symbol} at {retest_entry:.5f} | SL: {retest_stop:.5f} | TP: {take_profit:.5f}")
                # Clear retest tracking after emission
                self.retest_tracking[symbol] = {}
            elif retest_direction == 'sell':
                risk = retest_stop - retest_entry
                if symbol in self.last_consolidation_ranges:
                    range_size = self.last_consolidation_ranges[symbol]['size']
                    calculated_target = retest_level - range_size
                    min_target = retest_entry - (risk * self.min_risk_reward)
                    take_profit = min(calculated_target, min_target)
                else:
                    take_profit = retest_entry - (risk * self.min_risk_reward)
                # --- SELL RETEST: Ensure TP is below entry ---
                assert take_profit < retest_entry, "TP for sell must be below entry!"
                signal = {
                    "symbol": symbol,
                    "direction": "sell",
                    "entry_price": retest_entry,
                    "stop_loss": retest_stop,
                    "take_profit": take_profit,
                    "timeframe": self.primary_timeframe,
                    "strategy_name": self.name,  # Changed from "source" to "strategy_name"
                    "generator": self.name,
                    "reason": f"Retest confirmed: {retest_reason}",
                    "size": self.risk_manager.calculate_position_size(
                        account_balance=self.risk_manager.get_account_balance(),
                        risk_per_trade=self.risk_manager.max_risk_per_trade * 100,
                        entry_price=retest_entry,
                        stop_loss_price=retest_stop,
                        symbol=symbol
                    ),
                    "signal_bar_index": len(df) - 1,
                    "signal_timestamp": str(df.index[-1])
                }
                result = self.risk_manager.validate_and_size_trade(signal)
                if result['is_valid']:
                    adjusted_signal = result['final_trade_params']
                    for k in signal:
                        if k not in adjusted_signal:
                            adjusted_signal[k] = signal[k]
                    signals.append(adjusted_signal)
                    logger.info(f"🔴 RETEST SELL: {symbol} at {retest_entry:.5f} | SL: {retest_stop:.5f} | TP: {take_profit:.5f}")
                # Clear retest tracking after emission
                self.retest_tracking[symbol] = {}
            # After handling, skip further breakout signal generation for this symbol in this tick
            return signals
        # If not confirmed, do not emit a signal yet
        # (rest of the breakout/retest logic remains unchanged)

        # Check for resistance breakouts (horizontal levels)
        for i in range(-candles_to_check, 0):
            current_candle = df.iloc[i]
            previous_candle = df.iloc[i-1]
            
            logger.debug(f"📊 {symbol}: Checking candle at {df.index[i]}: O={current_candle['open']:.5f} H={current_candle['high']:.5f} L={current_candle['low']:.5f} C={current_candle['close']:.5f} V={current_candle['tick_volume']}")
            
            # Volume analysis with wick structure
            volume_quality = self._scorer.analyze_volume_quality(current_candle, volume_threshold)
            logger.debug(f"📊 {symbol}: Volume quality score: {volume_quality:.1f} (>0 = bullish, <0 = bearish)")
            
            # Check each resistance level
            for level in resistance_levels:
                logger.debug(f"🔄 {symbol}: Checking resistance level {level['zone_min']:.5f}-{level['zone_max']:.5f}")
                # Use relaxed breakout logic
                if self._is_breakout_candle(current_candle, df, level['zone_max'], 'buy'):
                    
                    entry_price = current_candle['close']
                    
                    # Place stop under the breakout candle's low
                    stop_loss = min(current_candle['low'], previous_candle['low'])
                    
                    # Log the breakout regardless of whether we generate a signal
                    logger.info(f"👀 Detected potential breakout for {symbol} at level {level['zone_max']:.5f}")
                    
                    # RELAXED conditions: allow signals with neutral H1 trend, don't require strong volume
                    if h1_trend != 'bearish':  
                        # Add detailed logging
                        logger.debug(f"Breakout details: Close={current_candle['close']:.5f}, " +
                                   f"Level={level['zone_max']:.5f}, Volume quality={volume_quality:.2f}, " +
                                   f"H1 trend={h1_trend}")
            for trend_line in bearish_trend_lines:
                # Calculate trend line value at current and previous candle
                prev_line_value = self._calculate_trend_line_value(trend_line, i-1)
                curr_line_value = self._calculate_trend_line_value(trend_line, i)
                
                # DEBUG: Log the trendline values
                logger.debug(f"Trendline values: prev={prev_line_value:.5f}, curr={curr_line_value:.5f}, " +
                           f"close={current_candle['close']:.5f}, candle index={i}")
            
                if (previous_candle['close'] <= prev_line_value * (1 + self.price_tolerance) and
                    current_candle['close'] > curr_line_value * (1 + self.price_tolerance) and
                    h1_trend != 'bearish'):  # Just avoid counter-trend signals

                    entry_price = current_candle['close']
                    
                    # Place stop under the breakout candle's low
                    stop_loss = min(current_candle['low'], previous_candle['low'])
                    
                    # Calculate risk and take profit
                    risk = entry_price - stop_loss
                    take_profit = entry_price + (risk * self.min_risk_reward)
                    
                    # Reason with volume quality description
                    volume_desc = "strong bullish volume" if volume_quality > 1 else "adequate volume"
                    reason = f"Bullish breakout above bearish trend line with {volume_desc}"
                    
                    # Add detailed logging
                    logger.info(f"💡 TRENDLINE BREAKOUT DETECTED: {symbol} at {entry_price:.5f}")
                    logger.debug(f"Breakout details: Close={current_candle['close']:.5f}, " +
                               f"Trendline value={curr_line_value:.5f}, Volume quality={volume_quality:.2f}, " +
                               f"H1 trend={h1_trend}, r²={trend_line['r_squared']:.2f}, angle={trend_line['angle']:.2f}°")
                    
                    # If retest is required, don't generate signal now but track for retest
                    if self.retest_required:
                        # Store breakout info for retest tracking
                        self.retest_tracking[symbol] = {
                            'level': curr_line_value,
                            'direction': 'buy',
                            'entry_price': entry_price,
                            'stop_loss': stop_loss,
                            'start_time': df.index[i],
                            'retest_confirmed': False,
                            'reason': reason
                        }
                        logger.info(f"👀 TRACKING RETEST: {symbol} bullish trend line breakout at {curr_line_value:.5f}")
                    else:
                        # Create immediate signal if retest not required
                        signal = {
                            "symbol": symbol,
                            "direction": "buy",
                            "entry_price": entry_price,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "timeframe": self.primary_timeframe,
                            "confidence": 0.0,  # placeholder, will update after scoring
                            "strategy_name": self.name,  # Changed from "source" to "strategy_name"
                            "generator": self.name,
                            "reason": reason,
                            "size": self.risk_manager.calculate_position_size(
                                account_balance=self.risk_manager.get_account_balance(),
                                risk_per_trade=self.risk_manager.max_risk_per_trade * 100,
                                entry_price=entry_price,
                                stop_loss_price=stop_loss,
                                symbol=symbol
                            ),
                            "signal_bar_index": len(df) - 1,
                            "signal_timestamp": str(df.index[i])
                        }
                        
                        scored = self._scorer.score_signal(signal, df, df)
                        signal["confidence"] = max(0.0, min(1.0, scored.get("score", 0)))
                        logger.info(f"🟢 TREND LINE BREAKOUT BUY: {symbol} at {entry_price:.5f} | SL: {stop_loss:.5f} | TP: {take_profit:.5f}")
                        signals.append(signal)
        
        # Check for support breakdowns (horizontal levels)
        for i in range(-candles_to_check, 0):
            current_candle = df.iloc[i]
            previous_candle = df.iloc[i-1]
            
            # Volume analysis with wick structure
            volume_quality = self._scorer.analyze_volume_quality(current_candle, volume_threshold)
            
            # Check each support level
            for level in support_levels:
                logger.debug(f"🔄 {symbol}: Checking support level {level['zone_min']:.5f}-{level['zone_max']:.5f}")
                # Use relaxed breakout logic
                if self._is_breakout_candle(current_candle, df, level['zone_min'], 'sell'):
                    entry_price = current_candle['close']
                    stop_loss = max(current_candle['high'], previous_candle['high'])
                    logger.info(f"👀 Detected potential breakdown for {symbol} at level {level['zone_min']:.5f}")
                    if h1_trend != 'bullish':
                        logger.debug(f"Breakdown details: Close={current_candle['close']:.5f}, Level={level['zone_min']:.5f}, Vol={current_candle['tick_volume']}, H1 trend={h1_trend}")
                        # ... rest of the sell signal logic ...
            
            # Check trend line breakdowns (bearish) - RELAXED conditions
            for trend_line in bullish_trend_lines:
                prev_line_value = self._calculate_trend_line_value(trend_line, i-1)
                curr_line_value = self._calculate_trend_line_value(trend_line, i)
                # RELAXED: allow signal if higher timeframe is not bullish (remove strong candle/volume requirements)
                if (previous_candle['close'] >= prev_line_value * (1 - self.price_tolerance) and
                    current_candle['close'] < curr_line_value * (1 - self.price_tolerance) and
                    h1_trend != 'bullish'):
                    entry_price = current_candle['close']
                    stop_loss = max(current_candle['high'], previous_candle['high'])
                    risk = stop_loss - entry_price
                    take_profit = entry_price - (risk * self.min_risk_reward)
                    volume_desc = "trendline breakdown (relaxed)"
                    reason = f"Bearish breakdown below bullish trend line with {volume_desc}"
                    if self.retest_required:
                        self.retest_tracking[symbol] = {
                            'level': curr_line_value,
                            'direction': 'sell',
                            'entry_price': entry_price,
                            'stop_loss': stop_loss,
                            'start_time': df.index[i],
                            'retest_confirmed': False,
                            'reason': reason
                        }
                        logger.info(f"👀 TRACKING RETEST: {symbol} bearish trend line breakdown (relaxed)")
                    else:
                        signal = {
                            "symbol": symbol,
                            "direction": "sell",
                            "entry_price": entry_price,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "timeframe": self.primary_timeframe,
                            "confidence": 0.0,
                            "strategy_name": self.name,  # Changed from "source" to "strategy_name"
                            "generator": self.name,
                            "reason": reason,
                            "size": self.risk_manager.calculate_position_size(
                                account_balance=self.risk_manager.get_account_balance(),
                                risk_per_trade=self.risk_manager.max_risk_per_trade * 100,
                                entry_price=entry_price,
                                stop_loss_price=stop_loss,
                                symbol=symbol
                            ),
                            "signal_bar_index": len(df) - 1,
                            "signal_timestamp": str(df.index[i])
                        }
                        scored = self._scorer.score_signal(signal, df, df)
                        signal["confidence"] = max(0.0, min(1.0, scored.get("score", 0)))
                        logger.info(f"🔴 TREND LINE BREAKDOWN SELL (relaxed): {symbol} at {entry_price:.5f} | SL: {stop_loss:.5f} | TP: {take_profit:.5f}")
                        signals.append(signal)
        
        return signals
    
    def _check_reversal_signals(self, symbol: str, df: pd.DataFrame, h1_df: pd.DataFrame, skip_plots: bool = False, processed_zones=None, signal_cooldown=86400, current_time=None) -> List[Dict]:
        self.logger.debug(f"[{symbol}/{self.primary_timeframe}] Checking for REVERSAL signals...")
        signals = []
        if df is None or len(df) < max(20, self.lookback_period): # Ensure enough data for patterns and lookbacks
            self.logger.warning(f"[{symbol}/{self.primary_timeframe}] Not enough data for reversal checks (need {max(20, self.lookback_period)}, have {len(df) if df is not None else 0}).")
            return signals

        atr_value = self._compute_atr(df)
        volume_threshold = self._compute_volume_threshold(df)
        
        # Use LuxAlgo pattern columns
        hammer_series = df['hammer']
        shooting_star_series = df['shooting_star']
        bullish_engulfing_series = df['bullish_engulfing']
        bearish_engulfing_series = df['bearish_engulfing']
        bullish_harami_series = df['bullish_harami']
        bearish_harami_series = df['bearish_harami']
        morning_star_series = df['morning_star']
        evening_star_series = df['evening_star']
        white_marubozu_series = df['white_marubozu']
        black_marubozu_series = df['black_marubozu']
        inside_bar_series = df['inside_bar']

        pin_bar_bullish_series = df['pin_bar_bullish']
        pin_bar_bearish_series = df['pin_bar_bearish']

        start_idx = max(1, len(df) - self.candles_to_check -1) # Ensure we have at least one prior bar for setup
        end_idx = len(df) -1 # Current candle is T+1 (confirmation)

        self.logger.debug(f"[{symbol}/{self.primary_timeframe}] Reversal Check Loop: df len={len(df)}, start_idx={start_idx}, end_idx={end_idx}")

        for confirmation_candle_idx in range(start_idx, end_idx + 1):
            if confirmation_candle_idx == 0: # Need a previous candle for setup
                continue
            
            setup_candle_idx = confirmation_candle_idx - 1
            
            # Ensure DataFrame has enough rows before iloc
            if setup_candle_idx < 0 or confirmation_candle_idx >= len(df):
                self.logger.debug(f"[{symbol}/{self.primary_timeframe}] Index out of bounds: setup_idx={setup_candle_idx}, confirm_idx={confirmation_candle_idx}, df_len={len(df)}")
                continue

            setup_candle = df.iloc[setup_candle_idx]
            confirmation_candle = df.iloc[confirmation_candle_idx]

            # Timestamp for cooldown and logging
            signal_timestamp_dt = df.index[confirmation_candle_idx] # Signal event is on confirmation
            signal_timestamp_str = str(signal_timestamp_dt)

            # Check 1: Higher Timeframe Trend Alignment
            htf_trend = self._determine_higher_timeframe_trend(h1_df)
            self.logger.debug(f"[{symbol}/{self.primary_timeframe}] Reversal check at {signal_timestamp_str}: HTF trend is {htf_trend}")

            for level_type in ['support', 'resistance']:
                key_levels = self.key_levels.get(symbol, {}).get(level_type, [])
                trend_lines_to_check = self.trend_lines.get(symbol, {}).get(level_type, [])
                
                levels_to_evaluate = []
                for lvl_dict in key_levels:
                    levels_to_evaluate.append({'price': lvl_dict['level'], 'type': 'horizontal', 'touches': lvl_dict.get('touches', 1)})
                for tl_dict in trend_lines_to_check:
                    # Calculate trend line value at setup_candle_idx
                    tl_price_at_setup = self._calculate_trend_line_value(tl_dict, setup_candle_idx)
                    if tl_price_at_setup is not None:
                         levels_to_evaluate.append({'price': tl_price_at_setup, 'type': 'trendline', 'slope': tl_dict.get('slope'), 'touches': tl_dict.get('touches',1)})
                
                if not levels_to_evaluate:
                    # self.logger.debug(f"[{symbol}/{self.primary_timeframe}] No {level_type} levels to evaluate for reversal at setup_candle_idx {setup_candle_idx}.")
                    continue

                for level_info in levels_to_evaluate:
                    level_price = level_info['price']
                    level_source = level_info['type'] # 'horizontal' or 'trendline'
                    
                    # Cooldown check
                    zone_key = (symbol, level_type, round(level_price, 5), level_source)
                    if processed_zones is not None and zone_key in processed_zones:
                        last_used_time = processed_zones[zone_key]
                        if current_time is not None and (current_time - last_used_time < signal_cooldown):
                            self.logger.debug(f"[{symbol}/{self.primary_timeframe}] Zone {zone_key} on cooldown. Skipping.")
                            continue
                    
                    # Determine expected reversal direction based on level type
                    expected_direction = 'buy' if level_type == 'support' else 'sell'

                    # HTF Trend alignment check
                    if (expected_direction == 'buy' and htf_trend == 'DOWNTREND') or \
                       (expected_direction == 'sell' and htf_trend == 'UPTREND'):
                        self.logger.debug(f"[{symbol}/{self.primary_timeframe}] Skipping {expected_direction} reversal at {level_price:.5f} ({level_source}) due to conflicting HTF trend ({htf_trend}).")
                        continue
                    
                    # ---- Step 1: Analyze Setup Candle (T) at setup_candle_idx ----
                    is_near_level_setup, actual_level_price_setup = self._is_near_level_dynamic(setup_candle, level_price, level_type, atr_value, context="setup_candle")
                    if not is_near_level_setup:
                        continue
                    
                    rejection_details = self._check_rejection_on_setup_candle(
                        setup_candle, actual_level_price_setup, expected_direction, atr_value
                    )
                    
                    is_false_break_rejection = rejection_details['is_rejection']
                    pattern_on_setup = rejection_details.get('pattern', "Unknown Rejection") # e.g. "Lower Wick Rejection", "Upper Wick Rejection"
                    
                    # B. Standard Candlestick Reversal Patterns on Setup Candle (T)
                    # Check these if not already a strong false_break_rejection, or to add confluence
                    if not is_false_break_rejection: # Prioritize false break detection
                        if expected_direction == 'buy':
                            if hammer_series.iloc[setup_candle_idx]: pattern_on_setup = "Hammer"
                            elif bullish_engulfing_series.iloc[setup_candle_idx]: pattern_on_setup = "Bullish Engulfing"
                            elif morning_star_series.iloc[setup_candle_idx]: pattern_on_setup = "Morning Star"
                            elif bullish_harami_series.iloc[setup_candle_idx]: pattern_on_setup = "Bullish Harami"
                            elif white_marubozu_series.iloc[setup_candle_idx]: pattern_on_setup = "White Marubozu"
                            elif inside_bar_series.iloc[setup_candle_idx]: pattern_on_setup = "Inside Bar"
                            elif pin_bar_bullish_series.iloc[setup_candle_idx]: pattern_on_setup = "Bullish Pin Bar"
                        else: # expected_direction == 'sell'
                            if shooting_star_series.iloc[setup_candle_idx]: pattern_on_setup = "Shooting Star"
                            elif bearish_engulfing_series.iloc[setup_candle_idx]: pattern_on_setup = "Bearish Engulfing"
                            elif evening_star_series.iloc[setup_candle_idx]: pattern_on_setup = "Evening Star"
                            elif bearish_harami_series.iloc[setup_candle_idx]: pattern_on_setup = "Bearish Harami"
                            elif black_marubozu_series.iloc[setup_candle_idx]: pattern_on_setup = "Black Marubozu"
                            elif inside_bar_series.iloc[setup_candle_idx]: pattern_on_setup = "Inside Bar"
                            elif pin_bar_bearish_series.iloc[setup_candle_idx]: pattern_on_setup = "Bearish Pin Bar"
                    
                    if pattern_on_setup is None or pattern_on_setup == "Unknown Rejection" and not is_false_break_rejection : # No clear rejection or pattern on setup candle
                        self.logger.debug(f"[{symbol}/{self.primary_timeframe}] No strong rejection or pattern on setup candle {setup_candle_idx} at {level_price:.4f} for {expected_direction}.")
                        continue

                    self.logger.info(f"[{symbol}/{self.primary_timeframe}] Potential Reversal Setup: Candle {setup_candle_idx} ({df.index[setup_candle_idx]}) shows {pattern_on_setup} for {expected_direction} at {level_type} {level_price:.4f} ({level_source}).")
                    
                    # Volume on Setup Candle
                    setup_volume_quality = self._scorer.analyze_volume_quality(setup_candle, volume_threshold)
                    if setup_volume_quality < 0.3 and not is_false_break_rejection: # Allow false breaks on potentially lower setup vol
                         self.logger.debug(f"[{symbol}/{self.primary_timeframe}] Low volume (below threshold) on setup candle {setup_candle_idx} for {pattern_on_setup}. Skipping.")
                         continue


                    # ---- Step 2: Analyze Confirmation Candle (T+1) at confirmation_candle_idx ----
                    self.logger.debug(f"[{symbol}/{self.primary_timeframe}] Analyzing confirmation candle {confirmation_candle_idx} ({df.index[confirmation_candle_idx]}) for {expected_direction} setup from {pattern_on_setup} at {level_price:.4f}.")

                    confirmation_logic = self._validate_reversal_confirmation(
                        confirmation_candle, setup_candle, expected_direction, actual_level_price_setup, atr_value, volume_threshold, df.iloc[:confirmation_candle_idx+1]
                    )

                    if not confirmation_logic['is_confirmed']:
                        self.logger.info(f"[{symbol}/{self.primary_timeframe}] Reversal ({expected_direction} from {pattern_on_setup}) NOT CONFIRMED by candle {confirmation_candle_idx}. Reason: {confirmation_logic.get('reason', 'Failed criteria')}")
                        continue
                        
                    self.logger.info(f"[{symbol}/{self.primary_timeframe}] REVERSAL CONFIRMED for {expected_direction} from {pattern_on_setup} by candle {confirmation_candle_idx}. Reason: {confirmation_logic.get('reason', 'Passed criteria')}")

                    # ---- Step 3: Signal Generation ----
                    entry_price = confirmation_candle['close'] # Example: Entry on close of confirmation candle
                    
                    if expected_direction == 'buy':
                        # SL below the low of the setup candle or the rejection point
                        sl_level = min(setup_candle['low'], rejection_details.get('rejection_low', setup_candle['low']))
                        stop_loss = sl_level - atr_value * self.atr_multiplier * 0.5 # Tighter SL for reversals
                        # TP to next resistance or R:R
                        take_profit = self._find_next_resistance(df.iloc[:confirmation_candle_idx+1], entry_price, self.key_levels.get(symbol, {}).get('resistance', []))
                        if take_profit is None or (take_profit - entry_price) < atr_value: # Basic R:R
                            take_profit = entry_price + (entry_price - stop_loss) * self.min_risk_reward
                    else: # expected_direction == 'sell'
                        sl_level = max(setup_candle['high'], rejection_details.get('rejection_high', setup_candle['high']))
                        stop_loss = sl_level + atr_value * self.atr_multiplier * 0.5
                        take_profit = self._find_next_support(df.iloc[:confirmation_candle_idx+1], entry_price, self.key_levels.get(symbol, {}).get('support', []))
                        if take_profit is None or (entry_price - take_profit) < atr_value:
                            take_profit = entry_price - (stop_loss - entry_price) * self.min_risk_reward

                    if self._is_invalid_or_zero(entry_price) or self._is_invalid_or_zero(stop_loss) or self._is_invalid_or_zero(take_profit):
                        self.logger.warning(f"[{symbol}/{self.primary_timeframe}] Invalid E/SL/TP for reversal signal: E={entry_price}, SL={stop_loss}, TP={take_profit}")
                        continue
                    
                    # Ensure SL is not too close or on the wrong side
                    if (expected_direction == 'buy' and entry_price <= stop_loss) or \
                       (expected_direction == 'sell' and entry_price >= stop_loss):
                        self.logger.warning(f"[{symbol}/{self.primary_timeframe}] Invalid SL for {expected_direction} signal: Entry={entry_price}, SL={stop_loss}. Skipping.")
                        continue
                    
                    risk_reward = abs(take_profit - entry_price) / max(abs(entry_price - stop_loss), 1e-9) # Avoid zero division
                    if risk_reward < self.min_risk_reward:
                         self.logger.info(f"[{symbol}/{self.primary_timeframe}] Reversal signal R:R ({risk_reward:.2f}) less than min ({self.min_risk_reward}). Skipping.")
                         continue

                    signal_reason = (f"{pattern_on_setup} at {level_type} {level_price:.4f} ({level_source}), "
                                     f"confirmed by T+1 bar. HTF: {htf_trend}. "
                                     f"SetupVol: {setup_volume_quality:.2f}, ConfirmVol: {confirmation_logic.get('volume_score',0):.2f}. "
                                     f"ConfirmReason: {confirmation_logic.get('reason', '')}")
                    
                    signal = {
                        'symbol': symbol,
                        'timestamp': signal_timestamp_dt,
                        'signal_timestamp_str': signal_timestamp_str,
                        'direction': expected_direction,
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'pattern': pattern_on_setup,
                        'confirmation_pattern': confirmation_logic.get('pattern_on_confirmation', "N/A"),
                        'level_price': actual_level_price_setup,
                        'level_type': level_type,
                        'level_source': level_source,
                        'strategy_name': self.name,
                        'type': 'REVERSAL',
                        'reason': signal_reason,
                        'risk_reward': risk_reward,
                        'atr_at_signal': atr_value,
                        'htf_trend': htf_trend,
                        'setup_candle_idx': setup_candle_idx,
                        'confirmation_candle_idx': confirmation_candle_idx,
                        # Add more context for scoring if needed
                    }
                    signals.append(signal)
                    
                    if processed_zones is not None and current_time is not None:
                         processed_zones[zone_key] = current_time
                    
                    # Only one signal per level per confirmation candle to avoid flood
                    break # Break from level_info loop
            # End level_type loop
        # End candle_idx loop
        
        # Score signals before returning
        if signals:
            signals = self._score_signals(signals, df, h1_df)

        self.logger.info(f"[{symbol}/{self.primary_timeframe}] Reversal signal check complete. Found {len(signals)} signals.")
        return signals
    
    def _check_rejection_on_setup_candle(self, setup_candle: pd.Series, level_price: float, expected_direction: str, atr_value: float) -> dict:
        """
        Analyzes the setup_candle for signs of strong wick rejection or false breakout patterns.
        Returns a dictionary with 'is_rejection' (bool) and 'pattern' (str), 'rejection_low', 'rejection_high'.
        """
        details = {'is_rejection': False, 'pattern': None, 'rejection_low': setup_candle['low'], 'rejection_high': setup_candle['high']}
        candle_open = setup_candle['open']
        candle_high = setup_candle['high']
        candle_low = setup_candle['low']
        candle_close = setup_candle['close']
        body_size = abs(candle_close - candle_open)
        total_range = candle_high - candle_low
        
        if total_range == 0: # Doji or invalid data
            return details

        upper_wick = candle_high - max(candle_open, candle_close)
        lower_wick = min(candle_open, candle_close) - candle_low
        
        # Dynamic wick threshold based on ATR or a fixed percentage
        # For example, wick must be at least 0.5 * ATR or 40% of total range
        min_wick_size_abs = max(0.5 * atr_value, 0.001 * level_price) # e.g. 0.5 ATR or 0.1% of level
        wick_threshold_ratio = 0.4 # Wick should be at least 40% of total candle range

        if expected_direction == 'buy': # Testing for rejection of lower prices at support
            # 1. False break below support, then close back above
            is_false_break = candle_low < level_price and candle_close > level_price
            # 2. Strong lower wick rejection
            is_strong_lower_wick = lower_wick >= min_wick_size_abs and (lower_wick / total_range >= wick_threshold_ratio)
            
            if is_false_break and is_strong_lower_wick:
                details['is_rejection'] = True
                details['pattern'] = "False Break & Strong Lower Wick"
                details['rejection_low'] = candle_low # The actual point of rejection
            elif is_false_break:
                details['is_rejection'] = True
                details['pattern'] = "False Break Below Support"
                details['rejection_low'] = candle_low
            elif is_strong_lower_wick and candle_close >= candle_open : # Must be bullish or neutral close
                details['is_rejection'] = True
                details['pattern'] = "Strong Lower Wick (Hammer-like)"
                details['rejection_low'] = candle_low
                
        elif expected_direction == 'sell': # Testing for rejection of higher prices at resistance
            # 1. False break above resistance, then close back below
            is_false_break = candle_high > level_price and candle_close < level_price
            # 2. Strong upper wick rejection
            is_strong_upper_wick = upper_wick >= min_wick_size_abs and (upper_wick / total_range >= wick_threshold_ratio)

            if is_false_break and is_strong_upper_wick:
                details['is_rejection'] = True
                details['pattern'] = "False Break & Strong Upper Wick"
                details['rejection_high'] = candle_high # The actual point of rejection
            elif is_false_break:
                details['is_rejection'] = True
                details['pattern'] = "False Break Above Resistance"
                details['rejection_high'] = candle_high
            elif is_strong_upper_wick and candle_close <= candle_open: # Must be bearish or neutral close
                details['is_rejection'] = True
                details['pattern'] = "Strong Upper Wick (ShootingStar-like)"
                details['rejection_high'] = candle_high
        
        if details['is_rejection']:
            self.logger.debug(f"[{setup_candle.name}] Setup Rejection Check: {details} for {expected_direction} at level {level_price:.4f}")
        return details

    def _validate_reversal_confirmation(self, confirmation_candle: pd.Series, setup_candle: pd.Series, 
                                      expected_direction: str, level_price: float, atr_value: float, 
                                      volume_threshold: float, df_for_vol_analysis: pd.DataFrame) -> dict:
        """
        Simplified: First check for strong patterns (engulfing, marubozu, pin bar). If found, confirm immediately.
        If not, apply minimal secondary checks (body size, wick, volume).
        """
        results = {'is_confirmed': False, 'reason': "No confirmation criteria met", 'volume_score': 0.0, 'pattern_on_confirmation': None}
        try:
            if confirmation_candle.name in df_for_vol_analysis.index:
                confirmation_candle_idx_int = df_for_vol_analysis.index.get_loc(confirmation_candle.name)
                if isinstance(confirmation_candle_idx_int, slice):
                    confirmation_candle_idx_int = confirmation_candle_idx_int.stop - 1
                elif not isinstance(confirmation_candle_idx_int, int):
                    true_indices = np.where(confirmation_candle_idx_int)[0]
                    if len(true_indices) > 0:
                        confirmation_candle_idx_int = true_indices[0]
                    else:
                        self.logger.error(f"Could not resolve confirmation_candle_idx for {confirmation_candle.name}")
                        return results
            else:
                self.logger.error(f"Confirmation candle timestamp {confirmation_candle.name} not found in df_for_vol_analysis index.")
                return results
        except Exception as e:
            self.logger.error(f"Error getting integer index for confirmation_candle: {e}")
            return results

        conf_open, conf_high, conf_low, conf_close = confirmation_candle[['open', 'high', 'low', 'close']]
        conf_body = conf_close - conf_open
        conf_total_range = conf_high - conf_low
        if conf_total_range == 0:
            results['reason'] = "Confirmation candle has zero range."
            return results
        conf_upper_wick = conf_high - max(conf_open, conf_close)
        conf_lower_wick = min(conf_open, conf_close) - conf_low
        conf_vol_score = self._scorer.analyze_volume_quality(confirmation_candle, volume_threshold)
        results['volume_score'] = conf_vol_score

        # --- 1. STRONG PATTERN CONFIRMATION ---
        strong_pattern = None
        if expected_direction == 'buy':
            if 'bullish_engulfing' in df_for_vol_analysis.columns and bool(df_for_vol_analysis['bullish_engulfing'].iloc[confirmation_candle_idx_int]):
                strong_pattern = "Bullish Engulfing"
            elif 'white_marubozu' in df_for_vol_analysis.columns and bool(df_for_vol_analysis['white_marubozu'].iloc[confirmation_candle_idx_int]):
                strong_pattern = "White Marubozu"
            elif 'pin_bar' in df_for_vol_analysis.columns and bool(df_for_vol_analysis['pin_bar'].iloc[confirmation_candle_idx_int]) and conf_close > conf_open:
                strong_pattern = "Bullish Pin Bar"
        elif expected_direction == 'sell':
            if 'bearish_engulfing' in df_for_vol_analysis.columns and bool(df_for_vol_analysis['bearish_engulfing'].iloc[confirmation_candle_idx_int]):
                strong_pattern = "Bearish Engulfing"
            elif 'black_marubozu' in df_for_vol_analysis.columns and bool(df_for_vol_analysis['black_marubozu'].iloc[confirmation_candle_idx_int]):
                strong_pattern = "Black Marubozu"
            elif 'pin_bar' in df_for_vol_analysis.columns and bool(df_for_vol_analysis['pin_bar'].iloc[confirmation_candle_idx_int]) and conf_close < conf_open:
                strong_pattern = "Bearish Pin Bar"
        if strong_pattern:
            results['is_confirmed'] = True
            results['reason'] = f"Strong pattern confirmation: {strong_pattern}"
            results['pattern_on_confirmation'] = strong_pattern
            self.logger.info(f"[CONFIRM] {strong_pattern} found on confirmation candle. Immediate confirmation.")
            return results

        # --- 2. MINIMAL SECONDARY CHECKS ---
        min_body_for_confirmation = max(0.2 * atr_value, 0.1 * conf_total_range, 0.0005 * conf_close)
        if expected_direction == 'buy':
            is_bullish_candle = conf_close > conf_open
            has_significant_body = conf_body >= min_body_for_confirmation
            closes_above_setup = conf_close > max(setup_candle['high'], setup_candle['close'])
            manageable_upper_wick = (conf_upper_wick / conf_total_range) < 0.6 and conf_upper_wick < conf_body if conf_body > 0 else (conf_upper_wick / conf_total_range) < 0.6
            volume_supports = conf_vol_score >= 0.4
            did_not_close_below_level = conf_close > level_price
            if is_bullish_candle and has_significant_body and closes_above_setup and manageable_upper_wick and volume_supports and did_not_close_below_level:
                results['is_confirmed'] = True
                results['reason'] = "Minimal bullish confirmation: body, close, wick, volume, level respected."
                self.logger.info(f"[CONFIRM] Minimal bullish confirmation met.")
            else:
                fail_reasons = []
                if not is_bullish_candle: fail_reasons.append("not_bullish_candle")
                if not has_significant_body: fail_reasons.append("insignificant_body")
                if not closes_above_setup: fail_reasons.append("not_above_setup_close/high")
                if not manageable_upper_wick: fail_reasons.append("long_upper_wick")
                if not volume_supports: fail_reasons.append("low_volume")
                if not did_not_close_below_level: fail_reasons.append("closed_below_support_level")
                results['reason'] = f"Bullish confirmation failed: {', '.join(fail_reasons)}"
        elif expected_direction == 'sell':
            is_bearish_candle = conf_close < conf_open
            has_significant_body = abs(conf_body) >= min_body_for_confirmation
            closes_below_setup = conf_close < min(setup_candle['low'], setup_candle['close'])
            manageable_lower_wick = (conf_lower_wick / conf_total_range) < 0.6 and conf_lower_wick < abs(conf_body) if conf_body != 0 else (conf_lower_wick / conf_total_range) < 0.6
            volume_supports = conf_vol_score >= 0.4
            did_not_close_above_level = conf_close < level_price
            if is_bearish_candle and has_significant_body and closes_below_setup and manageable_lower_wick and volume_supports and did_not_close_above_level:
                results['is_confirmed'] = True
                results['reason'] = "Minimal bearish confirmation: body, close, wick, volume, level respected."
                self.logger.info(f"[CONFIRM] Minimal bearish confirmation met.")
            else:
                fail_reasons = []
                if not is_bearish_candle: fail_reasons.append("not_bearish_candle")
                if not has_significant_body: fail_reasons.append("insignificant_body")
                if not closes_below_setup: fail_reasons.append("not_below_setup_close/low")
                if not manageable_lower_wick: fail_reasons.append("long_lower_wick")
                if not volume_supports: fail_reasons.append("low_volume")
                if not did_not_close_above_level: fail_reasons.append("closed_above_resistance_level")
                results['reason'] = f"Bearish confirmation failed: {', '.join(fail_reasons)}"
        
        self.logger.debug(f"[{confirmation_candle.name}] Confirmation Check for {expected_direction}: {results}")
        return results

    def _is_near_level_dynamic(self, candle: pd.Series, level_price: float, level_type: str, atr_value: float, context: str = "") -> Tuple[bool, float]:
        """
        Checks if a candle is 'near' a given level, using a dynamic tolerance based on ATR.
        Returns (is_near, actual_level_price_used_for_check).
        For trend lines, level_price is the calculated value at candle's index.
        """
        # Dynamic tolerance: e.g., 0.3 * ATR or a small fixed percentage of the price
        tolerance = max(0.3 * atr_value, 0.001 * level_price) # Use constants for level proximity
        
        is_near = False
        if level_type == 'support':
            # Candle low is within tolerance of the support level.
            # And candle close is above or very near the support level (respecting it).
            is_near = (candle['low'] <= level_price + tolerance) and \
                      (candle['low'] >= level_price - tolerance) 
                      # (candle['close'] >= level_price - tolerance * 0.5) # Close respects level
            
        elif level_type == 'resistance':
            is_near = (candle['high'] >= level_price - tolerance) and \
                      (candle['high'] <= level_price + tolerance)
                      # (candle['close'] <= level_price + tolerance * 0.5) # Close respects level

        # self.logger.debug(f"[{candle.name}/{context}] Near level check: Level {level_price:.4f} ({level_type}), Candle L/H: {candle['low']:.4f}/{candle['high']:.4f}, Close: {candle['close']:.4f}, Tol: {tolerance:.4f}, Near: {is_near}")
        return is_near, level_price # actual_level_price is same as input here, but could differ if level was dynamically adjusted
    
    def _find_next_resistance(self, df: pd.DataFrame, current_price: float, resistance_levels: List[dict], use_range_extension=None) -> Optional[float]:
        """Finds the closest resistance level above the current_price."""
        if use_range_extension is None:
            use_range_extension = getattr(self, 'use_range_extension_tp', False)
        if use_range_extension:
            tp = self._calculate_range_extension(df, current_price, direction="buy")
            self.logger.debug(f"[MarketProfile] Using range extension TP: {tp:.5f} for buy breakout")
            return tp
        if not resistance_levels:
            return None
        # Find the first zone where zone_min > current_price
        levels_above = [zone for zone in resistance_levels if zone['zone_min'] > current_price]
        if not levels_above:
            return None
        # Return the zone_min of the nearest zone above
        return min(levels_above, key=lambda z: z['zone_min'])['zone_min']
    
    def _find_next_support(self, df: pd.DataFrame, current_price: float, support_levels: List[dict], use_range_extension=None) -> Optional[float]:
        if use_range_extension is None:
            use_range_extension = getattr(self, 'use_range_extension_tp', False)
        if use_range_extension:
            tp = self._calculate_range_extension(df, current_price, direction="sell")
            self.logger.debug(f"[MarketProfile] Using range extension TP: {tp:.5f} for sell breakout")
            return tp
        if not support_levels:
            return None
        # Find the first zone where zone_max < current_price
        levels_below = [zone for zone in support_levels if zone['zone_max'] < current_price]
        if not levels_below:
            return None
        # Return the zone_max of the nearest zone below
        return max(levels_below, key=lambda z: z['zone_max'])['zone_max']    
    
    def _determine_higher_timeframe_trend(self, higher_df: pd.DataFrame):
        """
        Determine the trend on the higher timeframe using TA-Lib's LINEARREG_SLOPE (trendline slope) on close prices.
        If unavailable or insufficient data, fallback to price action swing logic.
        Returns: 'bullish', 'bearish', or 'neutral'
        """
        if len(higher_df) < 20:
            logger.debug(f"⚠️ Not enough data for trend determination, need 20 candles but got {len(higher_df)}")
            return 'neutral'
        try:
            close = np.asarray(higher_df['close'].values, dtype=np.float64)
            if len(close) >= 20:
                slope = talib.LINEARREG_SLOPE(close, timeperiod=20)[-1]
                logger.debug(f"[TA-Lib Trend] LINEARREG_SLOPE over 20 bars: {slope:.5f}")
                if slope > 0:
                    return 'bullish'
                elif slope < 0:
                    return 'bearish'
                else:
                    return 'neutral'
        except Exception as e:
            logger.warning(f"[TA-Lib Trend] Error using LINEARREG_SLOPE: {e}, falling back to swing logic")

    
    async def close(self):
        """Close and clean up resources."""
        logger.info(f"🔌 Closing {self.name}")
        # No specific cleanup needed
        return True
    
    def _is_invalid_or_zero(self, value):
        """
        Helper function to safely check if a value is zero, None or invalid.
        
        Args:
            value: The value to check
            
        Returns:
            bool: True if the value is None, NaN, a pandas Series/DataFrame with no valid data, or zero
        """
        if value is None:
            return True
            
        # Handle pandas Series by using the last value if possible
        if isinstance(value, pd.Series):
            if value.empty:
                return True
            try:
                # Try to get the last value from the series
                value = value.iloc[-1]
            except:
                return True
                
        # Handle DataFrame
        if isinstance(value, pd.DataFrame):
            return True
            
        # Check for NaN
        if pd.isna(value):
            return True
            
        # Check for zero value
        try:
            return float(value) == 0
        except (TypeError, ValueError):
            return True
    
    def _score_signals(self, raw_signals: List[Dict], primary_df: pd.DataFrame, higher_df: pd.DataFrame) -> List[Dict]:
        """Score a list of raw signals using SignalScorer and standardize confidence."""
        scored = []
        for sig in raw_signals:
            symbol = sig.get('symbol')
            scored_sig = self._scorer.score_signal(
                sig,
                primary_df,
                higher_df,
                support_levels=self.support_levels.get(symbol, []),
                resistance_levels=self.resistance_levels.get(symbol, []),
                last_consolidation_ranges=self.last_consolidation_ranges,
                atr_value=None if primary_df is None else self._compute_atr(primary_df),
                consolidation_info=self.last_consolidation_ranges.get(symbol, {}),
                risk_manager=getattr(self, 'risk_manager', None),
                account_balance=getattr(self, 'account_balance', None),
                extra_context={
                    "price_tolerance": self.price_tolerance,
                    "min_risk_reward": self.min_risk_reward,
                    "volume_percentile": self.volume_percentile,
                    "volume_threshold": self.volume_threshold,
                }
            )
            # Standardized confidence: direct mapping, clamped to [0, 1]
            scored_sig['confidence'] = max(0.0, min(1.0, scored_sig.get('score', 0)))
            # Add detailed explanations about why this signal received its score
            score_details = scored_sig.get('score_details', {})
            detailed_reasons = []
            direction = sig.get('direction', '').lower()
            h1_trend = self._determine_higher_timeframe_trend(higher_df)
            if score_details:
                # Level strength explanation
                level_strength = score_details.get('level_strength', 0)
                if level_strength > 0.7:
                    detailed_reasons.append(f"Very strong {direction} level with multiple touches (Level score: {level_strength:.2f})")
                elif level_strength > 0.5:
                    detailed_reasons.append(f"Strong {direction} level with good validation (Level score: {level_strength:.2f})")
                elif level_strength > 0.3:
                    detailed_reasons.append(f"Moderate {direction} level support/resistance (Level score: {level_strength:.2f})")
                elif level_strength > 0:
                    detailed_reasons.append(f"Weak level structure (Level score: {level_strength:.2f})")
                
                # Volume quality explanation
                volume_quality = score_details.get('volume_quality', 0)
                if volume_quality > 0.7:
                    detailed_reasons.append(f"Exceptional volume confirming {direction} move (Volume score: {volume_quality:.2f})")
                elif volume_quality > 0.5:
                    detailed_reasons.append(f"Strong volume confirmation (Volume score: {volume_quality:.2f})")
                elif volume_quality > 0.3:
                    detailed_reasons.append(f"Adequate volume support (Volume score: {volume_quality:.2f})")
                elif volume_quality > 0:
                    detailed_reasons.append(f"Low volume, limited confirmation (Volume score: {volume_quality:.2f})")
                
                # Pattern reliability explanation
                pattern_reliability = score_details.get('pattern_reliability', 0)
                if pattern_reliability > 0.7:
                    detailed_reasons.append(f"High-reliability {direction} pattern (Pattern score: {pattern_reliability:.2f})")
                elif pattern_reliability > 0.5:
                    detailed_reasons.append(f"Reliable {direction} pattern structure (Pattern score: {pattern_reliability:.2f})")
                elif pattern_reliability > 0.3:
                    detailed_reasons.append(f"Moderate pattern reliability (Pattern score: {pattern_reliability:.2f})")
                elif pattern_reliability > 0:
                    detailed_reasons.append(f"Basic pattern with limited reliability (Pattern score: {pattern_reliability:.2f})")
                
                # Trend alignment explanation
                trend_alignment = score_details.get('trend_alignment', 0)
                if trend_alignment > 0.7:
                    detailed_reasons.append(f"Strong alignment with {h1_trend} higher timeframe trend (H1 trend score: {trend_alignment:.2f})")
                elif trend_alignment > 0.5:
                    detailed_reasons.append(f"Good alignment with {h1_trend} market direction (H1 trend score: {trend_alignment:.2f})")
                elif trend_alignment > 0.3:
                    detailed_reasons.append(f"Neutral trend alignment (H1 trend score: {trend_alignment:.2f})")
                elif trend_alignment >= 0:
                    detailed_reasons.append(f"Counter-trend signal against {h1_trend} direction (H1 trend score: {trend_alignment:.2f})")
                
                # Risk-reward explanation
                risk_reward = score_details.get('risk_reward', 0)
                if risk_reward > 0.7:
                    detailed_reasons.append(f"Excellent risk-reward ratio > 2:1 (R:R score: {risk_reward:.2f})")
                elif risk_reward > 0.5:
                    detailed_reasons.append(f"Good risk-reward profile (R:R score: {risk_reward:.2f})")
                elif risk_reward > 0.3:
                    detailed_reasons.append(f"Acceptable risk-reward (R:R score: {risk_reward:.2f})")
                elif risk_reward >= 0:
                    detailed_reasons.append(f"Minimum acceptable risk-reward ratio (R:R score: {risk_reward:.2f})")
                
                # Add special bonuses that might have been applied
                if hasattr(self, '_volume_profile_bonus') and getattr(self, '_volume_profile_bonus', False):
                    detailed_reasons.append("Bonus: Entry near high-volume node (+0.07)")
                
                if hasattr(self, '_atr_bonus') and getattr(self, '_atr_bonus', 0) > 0:
                    detailed_reasons.append(f"Bonus: Optimal stop-loss placement relative to ATR (+0.1)")
                elif hasattr(self, '_atr_bonus') and getattr(self, '_atr_bonus', 0) < 0:
                    detailed_reasons.append(f"Penalty: Suboptimal stop-loss placement (-0.1)")
                
                # Add consolidation bonus information if applicable
                if sig.get('consolidation_bonus', False):
                    detailed_reasons.append("Bonus: Signal within consolidation zone (+0.05)")
            
            # Add the detailed reasons to the signal
            scored_sig['detailed_reasoning'] = detailed_reasons
            
            # Log detailed reasoning for this signal
            logger.info(f"🔍 SIGNAL SCORE BREAKDOWN - {symbol} {direction}:")
            for reason in detailed_reasons:
                logger.info(f"  • {reason}")
            logger.info(f"  📊 Final score: {scored_sig.get('score', 0):.2f} → Confidence: {scored_sig.get('confidence', 0):.2f}")
            scored_sig['original_symbol'] = symbol
            # HARD FILTER: Only add if not skip_due_to_volume
            if not scored_sig.get('skip_due_to_volume', False):
                scored.append(scored_sig)
        return scored

    def _compute_atr(self, df: pd.DataFrame) -> float:
        # Helper to compute ATR for scoring context using TA-Lib
        try:
            high = np.asarray(df['high'].values, dtype=np.float64)
            low = np.asarray(df['low'].values, dtype=np.float64)
            close = np.asarray(df['close'].values, dtype=np.float64)
            atr = talib.ATR(high, low, close, timeperiod=14)
            return float(atr[-1]) if len(atr) > 0 else 0.0
        except Exception:
            return 0.0

    def _ensure_tick_volume(self, df: pd.DataFrame, symbol: str) -> None:
        """Ensure df has 'tick_volume' column, falling back to 'volume' or default."""
        if 'tick_volume' not in df.columns:
            if 'volume' in df.columns:
                logger.debug(f"Using 'volume' as 'tick_volume' for {symbol}")
                df['tick_volume'] = df['volume']
            else:
                logger.debug(f"Setting default tick_volume=1 for {symbol}")
                df['tick_volume'] = 1

    def _compute_volume_threshold(self, df: pd.DataFrame) -> float:
        """Compute volume threshold based on percentile or fallback to average."""
        try:
            lookback = min(50, len(df) - 1)
            vol = df['tick_volume'].iloc[-lookback:].copy()
            thresh = float(np.percentile(vol, self.volume_percentile))
            logger.debug(f"Volume threshold (percentile {self.volume_percentile}): {thresh:.1f}")
        except Exception as e:
            logger.warning(f"Volume threshold percentile failed: {e}")
            try:
                avg_vol = float(df['tick_volume'].rolling(window=20).mean().iloc[-1])
                thresh = avg_vol * self.volume_threshold
                logger.debug(f"Fallback avg volume threshold: {thresh:.1f}")
            except Exception as e2:
                logger.warning(f"Volume threshold fallback failed: {e2}")
                thresh = 1.0
        return thresh

    def _cluster_1d(self, values: List[float], tolerance: float) -> List[List[float]]:
        """Cluster 1D values using a given tolerance."""
        if not values:
            return []
        sorted_vals = sorted(values)
        clusters = []
        current_cluster = [sorted_vals[0]]
        for i in range(1, len(sorted_vals)):
            if sorted_vals[i] - sorted_vals[i-1] <= tolerance:
                current_cluster.append(sorted_vals[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [sorted_vals[i]]
        clusters.append(current_cluster)
        return clusters
    
    def _cluster_trend_lines(self, trend_lines: List[dict]) -> List[dict]:
        """Cluster similar trend lines to reduce redundancy."""
        if not trend_lines:
            return []
        # Cluster by intercept (using a tolerance based on price)
        tol = np.mean([line['intercept'] for line in trend_lines]) * 0.0015 if trend_lines else 0
        clusters = self.cluster_items(trend_lines, metric=lambda x: x['intercept'], tol=tol)
        clustered_lines = []
        for cluster in clusters:
            best_line = max(cluster, key=lambda x: x['quality_score'])
            clustered_lines.append(best_line)
        return clustered_lines

    def _calculate_range_extension(self, df, entry_price, direction):
        if not self.use_range_extension_tp:
            return None

        if df is None or df.empty:
            self.logger.warning(f"[{self.name}] DataFrame is empty, cannot calculate range extension TP.")
            return None

        lookback_period = getattr(self, 'range_extension_lookback', 20) # Add a configurable lookback
        
        # Ensure lookback does not exceed available data
        actual_lookback = min(lookback_period, len(df))

        if actual_lookback <= 0:
            self.logger.warning(f"[{self.name}] Not enough data for range extension lookback (actual_lookback: {actual_lookback}).")
            return None
            
        try:
            # Calculate average range (high - low) over the lookback period
            avg_range = (df['high'].iloc[-actual_lookback:] - df['low'].iloc[-actual_lookback:]).mean()
            
            if pd.isna(avg_range) or avg_range <= 0:
                self.logger.warning(f"[{self.name}] Could not calculate a valid average range (avg_range: {avg_range}).")
                return None

            if direction == "buy":
                take_profit = entry_price + avg_range
            elif direction == "sell":
                take_profit = entry_price - avg_range
            else:
                self.logger.warning(f"[{self.name}] Invalid direction '{direction}' for range extension TP.")
                return None
            
            self.logger.debug(f"[{self.name}] Calculated range extension TP: {take_profit:.5f} (entry: {entry_price:.5f}, avg_range: {avg_range:.5f}, direction: {direction})")
            return take_profit
            
        except Exception as e:
            self.logger.error(f"[{self.name}] Error calculating range extension TP: {e}")
            return None

    @staticmethod
    def cluster_items(items, metric, tol):
        """
        Generic 1D clustering utility. Groups items by proximity using a metric and tolerance.
        Args:
            items: list of items to cluster
            metric: function to extract a float value from each item
            tol: float, maximum distance for clustering
        Returns:
            List of clusters (each a list of items)
        """
        if not items:
            return []
        sorted_items = sorted(items, key=metric)
        clusters = []
        current_cluster = [sorted_items[0]]
        for item in sorted_items[1:]:
            if abs(metric(item) - metric(current_cluster[-1])) <= tol:
                current_cluster.append(item)
            else:
                clusters.append(current_cluster)
                current_cluster = [item]
        clusters.append(current_cluster)
        return clusters

    @property
    def required_timeframes(self):
        if self.primary_timeframe == self.higher_timeframe or not self.higher_timeframe:
            return [self.primary_timeframe]
        return list(set([self.primary_timeframe, self.higher_timeframe]))

    @property
    def lookback_periods(self):
        return {self.primary_timeframe: self.lookback_period}

    def _is_breakout_candle(self, candle: pd.Series, df: pd.DataFrame, level: float, direction: str) -> bool:
        """
        Relaxed breakout confirmation: 1-bar close above/below S/R with volume > configurable percentile.
        """
        if len(df) < 50 or 'tick_volume' not in df.columns:
            return False
        # Use self.volume_percentile instead of hardcoded 80
        vol_threshold = np.percentile(df['tick_volume'].iloc[-50:], self.volume_percentile)
        if direction == 'buy':
            return (candle['close'] > level and candle['tick_volume'] >= vol_threshold)
        else:
            return (candle['close'] < level and candle['tick_volume'] >= vol_threshold)

# --- Module-level DataFrame helpers (restored) ---
def _to_dataframe(raw_data, timeframe: str) -> Optional[pd.DataFrame]:
    """Convert raw market data to a pandas DataFrame with proper columns and datetime index."""
    import pandas as pd
    if raw_data is None:
        return None
    if isinstance(raw_data, pd.DataFrame):
        return raw_data.copy()
    try:
        df = pd.DataFrame(raw_data)
        if df.empty:
            return None
        # Try to set datetime index if possible
        if 'time' in df.columns:
            df.index = pd.DatetimeIndex(pd.to_datetime(df['time']))        
        return df
    except Exception as e:
        logger.debug(f"_to_dataframe: Failed to convert raw_data to DataFrame: {e}")
        return None

def _ensure_datetime_index(df: Optional[pd.DataFrame], timeframe: str) -> Optional[pd.DataFrame]:
    """Ensure the DataFrame has a DatetimeIndex. Return None if not possible."""
    import pandas as pd
    if df is None or df.empty:
        return None
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'time' in df.columns:
            try:
                df.index = pd.DatetimeIndex(pd.to_datetime(df['time']))
            except Exception as e:
                logger.debug(f"_ensure_datetime_index: Failed to set DatetimeIndex: {e}")
                return None
        else:
            return None
    return df