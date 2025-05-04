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
from scipy import stats
import matplotlib.pyplot as plt
import os
from pathlib import Path
import time
import traceback

from src.trading_bot import SignalGenerator
from src.utils.indicators import calculate_atr
from src.risk_manager import RiskManager

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
        "volume_percentile": 85,  # 85th percentile for volume threshold
        "min_risk_reward": 2.0   # Align with book
    },
    "M5": {
        "lookback_period": 140,  # ~12 hours
        "max_retest_bars": 12,   # 60 minutes
        "level_update_hours": 6,
        "consolidation_bars": 40, 
        "candles_to_check": 6,
        "consolidation_update_hours": 3,
        "atr_multiplier": 0.7,   # Medium multiplier
        "volume_percentile": 85,  # 85th percentile for volume threshold
        "min_risk_reward": 2.0   # Align with book
    },
    "M15": {
        "lookback_period": 96,   # ~24 hours
        "max_retest_bars": 6,    # 90 minutes
        "level_update_hours": 12,
        "consolidation_bars": 20,
        "candles_to_check": 3,
        "consolidation_update_hours": 6,
        "atr_multiplier": 1.0,   # Standard multiplier
        "volume_percentile": 85,  # 85th percentile for volume threshold
        "min_risk_reward": 2.0   # Align with book
    },
    "H1": {
        "lookback_period": 50,   # ~2 days
        "max_retest_bars": 6,    # 6 hours
        "level_update_hours": 24,
        "consolidation_bars": 10,
        "candles_to_check": 2,
        "consolidation_update_hours": 12,
        "atr_multiplier": 1.2,   # Higher multiplier for more significant movements
        "volume_percentile": 85,  # 85th percentile for volume threshold
        "min_risk_reward": 3.0   # More conservative for higher TF
    },
    "H4": {
        "lookback_period": 30,   # ~5 days
        "max_retest_bars": 4,    # 16 hours
        "level_update_hours": 48,
        "consolidation_bars": 7,
        "candles_to_check": 2,
        "consolidation_update_hours": 24,
        "atr_multiplier": 1.5,   # Higher multiplier for more significant movements
        "volume_percentile": 85,  # 85th percentile for volume threshold
        "min_risk_reward": 3.0   # More conservative for higher TF
    }
}

# Module-level helper: dict→DataFrame conversion logic
def _to_dataframe(raw_data: Any, timeframe: str) -> Optional[pd.DataFrame]:
    """
    Centralize all of your dict→DataFrame conversion logic in one place.
    Now includes debug logging and validation for data quality.
    """
    df: Optional[pd.DataFrame] = None
    try:
        if isinstance(raw_data, pd.DataFrame):
            df = raw_data.copy()
        elif isinstance(raw_data, dict):
            for key in [timeframe] + ['data','candles','ohlc'] + ['M1','M5','M15','H1','1m','5m','15m','1h']:
                if key in raw_data and isinstance(raw_data[key], pd.DataFrame):
                    df = raw_data[key].copy()
                    break
            if df is None and all(k in raw_data for k in ['open','high','low','close']):
                df = pd.DataFrame({
                    'open': raw_data['open'],
                    'high': raw_data['high'],
                    'low': raw_data['low'],
                    'close': raw_data['close'],
                })
                vol = raw_data.get('tick_volume', raw_data.get('volume', None))
                if vol is not None:
                    df['tick_volume'] = vol
                if 'time' in raw_data:
                    df.index = pd.to_datetime(raw_data['time'])
        if df is not None and 'tick_volume' not in df.columns:
            df['tick_volume'] = df.get('volume', 1)
        # --- Debug logging and validation ---
        import numpy as np
        from loguru import logger
        if df is not None:
            logger.debug(f"[DF-DEBUG] DataFrame created for timeframe {timeframe} - shape: {df.shape}")
            try:
                logger.debug(f"[DF-DEBUG] Head:\n{df.head(3)}")
                logger.debug(f"[DF-DEBUG] Describe:\n{df.describe(include='all').T}")
                # Check for all-zero or all-NaN columns
                for col in ['open','high','low','close']:
                    if col in df.columns:
                        if np.all(df[col] == 0):
                            logger.debug(f"[DF-DEBUG] Column '{col}' is all zeros for timeframe {timeframe}")
                        if df[col].isnull().all():
                            logger.debug(f"[DF-DEBUG] Column '{col}' is all NaN for timeframe {timeframe}")
                # Check for extreme values
                for col in ['open','high','low','close']:
                    if col in df.columns:
                        max_val = df[col].max()
                        min_val = df[col].min()
                        if max_val > 1e5 or min_val < -1e5:
                            logger.debug(f"[DF-DEBUG] Column '{col}' has extreme values: min={min_val}, max={max_val} for timeframe {timeframe}")
            except Exception as e:
                logger.debug(f"[DF-DEBUG] Exception during DataFrame debug logging: {e}")
    except Exception as e:
        df = None
        from loguru import logger
        logger.error(f"[DF-DEBUG] Exception in _to_dataframe for timeframe {timeframe}: {e}")
    return df

# Module-level helper: ensure DatetimeIndex logic
def _ensure_datetime_index(df: Optional[pd.DataFrame], timeframe: str) -> Optional[pd.DataFrame]:
    """
    Centralize synthetic-index or 'time'→DatetimeIndex logic.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'time' in df.columns:
            try:
                df['time'] = pd.to_datetime(df['time'])
                df.set_index('time', inplace=True)
            except Exception:
                pass
        if not isinstance(df.index, pd.DatetimeIndex):
            now = datetime.now()
            minutes = int(timeframe[1:]) if timeframe.startswith('M') and timeframe[1:].isdigit() else 1
            df.index = pd.DatetimeIndex([
                now - timedelta(minutes=minutes * i)
                for i in range(len(df)-1, -1, -1)
            ])
    return df

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

    def analyze_volume_quality(self, candle: pd.Series, threshold: float) -> float:
        """
        Analyze the quality of volume based on candle structure and wick analysis.
        Returns a score indicating volume quality (-2 to +2).
        """
        try:
            # Check if 'tick_volume' column exists, if not use 'volume', if neither exists use a default
            if 'tick_volume' not in candle:
                if 'volume' in candle:
                    tick_volume = candle['volume']
                    self.strategy.logger.debug(f"Using 'volume' instead of missing 'tick_volume' for volume analysis")
                else:
                    self.strategy.logger.debug(f"Using default volume value as neither 'tick_volume' nor 'volume' exists")
                    tick_volume = threshold * 0.8  # Default to 80% of threshold as a reasonable value
            else:
                tick_volume = candle['tick_volume']
                
            # First check if volume is even significant - using a stricter threshold
            volume_ratio = tick_volume / threshold
            self.strategy.logger.debug(f"Volume ratio: {volume_ratio:.2f} (volume: {tick_volume}, threshold: {threshold:.1f})")
            
            if volume_ratio < 0.8:  # More strict check
                self.strategy.logger.debug(f"Insufficient volume: {tick_volume} < 80% of threshold {threshold:.1f}")
                return 0  # Insufficient volume
                
            # Calculate components
            is_bullish = candle['close'] > candle['open']
            total_range = candle['high'] - candle['low']
            body = abs(candle['close'] - candle['open'])
            
            if total_range == 0 or total_range < 0.00001:  # Guard against division by zero
                self.strategy.logger.debug("Doji or very small candle - neutral volume")
                return 0  # Doji or similar
                
            # Analyze wick structure
            if is_bullish:
                upper_wick = candle['high'] - candle['close']
                lower_wick = candle['open'] - candle['low']
                
                upper_wick_ratio = upper_wick / total_range
                lower_wick_ratio = lower_wick / total_range
                body_ratio = body / total_range
                
                # Debug information
                self.strategy.logger.debug(f"Bullish candle - body ratio: {body_ratio:.2f}, upper wick: {upper_wick_ratio:.2f}, lower wick: {lower_wick_ratio:.2f}")
                
                # Bullish cases
                if body_ratio > 0.6 and lower_wick_ratio < 0.2:
                    # Strong buying pressure - high quality bullish volume
                    return 2.0
                elif body_ratio > 0.4 and lower_wick_ratio < upper_wick_ratio:
                    # Good buying pressure - moderate quality bullish volume
                    return 1.0
                elif upper_wick_ratio > 0.6:
                    # Large upper wick - poor quality for bulls despite green candle
                    return -0.5
                else:
                    # Average quality bullish volume
                    return 0.5
            else:
                # Bearish candle
                upper_wick = candle['high'] - candle['open']
                lower_wick = candle['close'] - candle['low']
                
                upper_wick_ratio = upper_wick / total_range
                lower_wick_ratio = lower_wick / total_range
                body_ratio = body / total_range
                
                # Debug information
                self.strategy.logger.debug(f"Bearish candle - body ratio: {body_ratio:.2f}, upper wick: {upper_wick_ratio:.2f}, lower wick: {lower_wick_ratio:.2f}")
                
                # Bearish cases
                if body_ratio > 0.6 and upper_wick_ratio < 0.2:
                    # Strong selling pressure - high quality bearish volume
                    return -2.0
                elif body_ratio > 0.4 and upper_wick_ratio < lower_wick_ratio:
                    # Good selling pressure - moderate quality bearish volume
                    return -1.0
                elif lower_wick_ratio > 0.6:
                    # Large lower wick - poor quality for bears despite red candle
                    return 0.5
                else:
                    # Average quality bearish volume
                    return -0.5
        except Exception as e:
            self.strategy.logger.error(f"Error in volume analysis: {str(e)}")
            return 0  # Safe default

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
        """
        symbol = signal.get('symbol')
        direction = signal['direction']
        level = signal.get('level')
        entry = signal['entry_price']
        stop = signal['stop_loss']
        tp = signal['take_profit']
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
                else:
                    if any((abs(recent['high']-closest) <= closest*price_tolerance)):
                        level_strength_score = min(level_strength_score+0.2,1.0)
        # 2. Volume Quality (20%)
        volume_quality_score = 0
        if 'strong' in reason and 'volume' in reason:
            volume_quality_score = 1.0
        elif 'adequate' in reason and 'volume' in reason:
            volume_quality_score = 0.7
        else:
            try:
                lookback = min(50,len(df)-1)
                vol_thresh = np.percentile(df['tick_volume'].iloc[-lookback:], volume_percentile)
                candle = df.iloc[-1]
                vol_quality = self.analyze_volume_quality(candle, vol_thresh)
                if direction == 'buy':
                    volume_quality_score = max(0, vol_quality/2)
                else:
                    volume_quality_score = max(0, -vol_quality/2)
            except Exception:
                volume_quality_score = 0.5
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
        for pat, score in pattern_reliability.items():
            if pat in reason:
                pattern_reliability_score = score
                break
        # 4. Trend Alignment (20%)
        trend_alignment_score = 0
        if direction=='buy':
            if higher_trend=='bullish':
                trend_alignment_score = 1.0
            elif higher_trend=='neutral':
                trend_alignment_score = 0.5
            else:
                trend_alignment_score = 0.0
        else:
            if higher_trend=='bearish':
                trend_alignment_score = 1.0
            elif higher_trend=='neutral':
                trend_alignment_score = 0.5
            else:
                trend_alignment_score = 0.0
        # 5. Risk-Reward (10%)
        if direction=='buy':
            risk = entry-stop
            reward = tp-entry
        else:
            risk = stop-entry
            reward = entry-tp
        if risk > 0:
            rr_ratio = reward/risk
            risk_reward_score = min(rr_ratio/3, 1.0)
        else:
            risk_reward_score = 0
        # ATR bonus (±0.1)
        atr_bonus = 0
        try:
            atr_series = calculate_atr(df, getattr(self.strategy, 'atr_period', 14))
            atr = atr_series.iloc[-1] if isinstance(atr_series, pd.Series) and not atr_series.empty else None
            if atr is not None and not pd.isna(atr) and float(atr) > 0:
                stop_atr_ratio = abs(risk) / float(atr)
                if 0.5 <= stop_atr_ratio <= 3.0:
                    atr_bonus = 0.1
                else:
                    atr_bonus = -0.1
        except Exception:
            atr_bonus = 0
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
        except Exception:
            pass
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

def plot_raw_price_series(df, symbol, timeframe):
    """Plot and save the raw price series for debugging."""
    if df is None or df.empty:
        return
    plt.figure(figsize=(12, 6))
    plt.plot(df['close'], label='Close', color='blue')
    plt.plot(df['high'], label='High', color='green', alpha=0.3)
    plt.plot(df['low'], label='Low', color='red', alpha=0.3)
    plt.title(f'Raw Price Series for {symbol} ({timeframe})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    import os
    os.makedirs('debug_plots', exist_ok=True)
    plt.savefig(f'debug_plots/{symbol}_{timeframe}_raw.png')
    plt.close()

class BreakoutReversalStrategy(SignalGenerator):
    """
    Breakout and Reversal Hybrid Strategy based on price action principles.
    Uses support/resistance levels, candlestick patterns, and volume analysis
    to generate high-probability trading signals.
    """
    
    def __init__(self, primary_timeframe="M1", higher_timeframe="H1", use_range_extension_tp=False, **kwargs):
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
        self.required_timeframes = [primary_timeframe, higher_timeframe]
        
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
        
        current_time = datetime.now()
        logger.debug(f"⏰ Initializing time tracking with current time: {current_time}")
        
        logger.info(f"🔧 Initialized {self.name} with primary TF: {primary_timeframe}, higher TF: {higher_timeframe}")
        
        # Log all parameters for reference
        params = {
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
        logger.debug(f"📊 Strategy parameters: {params}")
        self._scorer = _SignalScorer(self)
        self.risk_manager = RiskManager.get_instance()
        self.use_range_extension_tp = use_range_extension_tp
        self.swing_window = kwargs.get('swing_window', self.candles_to_check)
    
    def _load_timeframe_profile(self):
        """Load timeframe-specific parameters from the appropriate profile."""
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
    
    def _to_dataframe(self, raw_data, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Convert raw market_data (dict or DataFrame) into a standardized DataFrame with
        columns ['open','high','low','close','tick_volume'] and return None if conversion fails.
        """
        # Delegate conversion to module-level helper
        return _to_dataframe(raw_data, timeframe)
    
    def _ensure_datetime_index(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Ensure df.index is a DatetimeIndex; if not, use 'time' column or synthesize it.
        Also logs structure and sample rows once.
        """
        # Retain logging for shape and sample rows
        if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
            logger.debug(f"📊 {timeframe} DataFrame for {symbol}: shape={df.shape}, cols={list(df.columns)}")
            try:
                sample_n = min(3, len(df))
                for i in range(-sample_n, 0):
                    c = df.iloc[i]
                    logger.debug(f"   {i}: O={c['open']:.5f},H={c['high']:.5f},L={c['low']:.5f},C={c['close']:.5f},Vol={c['tick_volume']}")
            except Exception:
                pass
        # Delegate index normalization to module-level helper
        return _ensure_datetime_index(df, timeframe)
    
    async def generate_signals(self, market_data=None, symbol=None, timeframe=None, debug_visualize=False, force_trendlines=False, skip_plots=False, **kwargs):
        logger.debug(f"[BreakoutReversalStrategy] Analyzing symbol(s): {list(market_data.keys()) if market_data else symbol} | primary_timeframe={self.primary_timeframe}, higher_timeframe={self.higher_timeframe}")
        start_time = time.time()
        logger.info(f"🚀 SIGNAL GENERATION START: {self.name} strategy")
        
        if not market_data:
            logger.warning("⚠️ No market data provided to generate signals")
            return []
            
        # Check if we should force visualization for debugging
        debug_visualize = kwargs.get('debug_visualize', debug_visualize)
        force_trendlines = kwargs.get('force_trendlines', force_trendlines)
        skip_plots = kwargs.get('skip_plots', skip_plots)
        process_immediately = kwargs.get('process_immediately', False)
        
        # Debug logging
        if debug_visualize:
            logger.info("🔍 Debug visualization mode enabled - will force trendline updates with plots")
        elif force_trendlines:
            logger.info("🔄 Forcing trendline updates without plots")
            
        signals = []
        all_signals = []  # To collect all potential signals for scoring
        logger.info(f"🔍 Generating signals with {self.name} strategy for {len(market_data)} symbols")
        
        # Process symbols one by one, potentially returning signals immediately
        for symbol in market_data:
            symbol_start_time = time.time()
            logger.debug(f"📊 Market data for {symbol} contains timeframes: {list(market_data[symbol].keys())}")
            
            # Skip if we don't have all required timeframes
            if not all(tf in market_data[symbol] for tf in self.required_timeframes):
                missing_tfs = [tf for tf in self.required_timeframes if tf not in market_data[symbol]]
                logger.debug(f"⏩ Missing required timeframes for {symbol}: {missing_tfs}, skipping")
                continue
                
            # Prepare dataframes for primary and higher timeframes
            primary_df, higher_df = self._prepare_dataframes(market_data[symbol], symbol)
            
            # Check if DataFrames are None or empty
            if primary_df is None or len(primary_df) == 0 or higher_df is None or len(higher_df) == 0:
                logger.debug(f"⏩ DataFrame is None or empty for {symbol}, skipping.")
                continue
            # Warn if not enough bars for lookback, but proceed anyway
            if len(primary_df) < self.lookback_period:
                logger.debug(f"[PERMISSIVE] {symbol}: primary_df has only {len(primary_df)} rows (lookback required: {self.lookback_period}) – proceeding anyway.")
            if len(higher_df) < 10:
                logger.debug(f"[PERMISSIVE] {symbol}: higher_df has only {len(higher_df)} rows (min 10 recommended) – proceeding anyway.")
            
            # Verify we have proper DataFrames before processing
            if not isinstance(primary_df, pd.DataFrame) or not isinstance(higher_df, pd.DataFrame):
                logger.warning(f"Expected DataFrames but got: primary={type(primary_df)}, higher={type(higher_df)}")
                continue
            
            # Update key levels and trend lines
            try:
                if primary_df is not None and len(primary_df) > 0:
                    # Force trend line updates if debug_visualize or force_trendlines is True
                    self._update_key_levels(symbol, primary_df, debug_force_update=(debug_visualize or force_trendlines))
                    self._find_trend_lines(symbol, primary_df, debug_force_update=(debug_visualize or force_trendlines), skip_plots=skip_plots)
                    self._identify_consolidation_ranges(symbol, primary_df)
                    self._process_retest_conditions(symbol, primary_df)
                else:
                    logger.warning(f"⚠️ Empty primary DataFrame for {symbol}, skipping level and trendline detection")
            except Exception as e:
                logger.exception(f"Error during level detection for {symbol}: {str(e)}")
            
            # Check for breakout signals
            breakout_signals = self._check_breakout_signals(symbol, primary_df, higher_df, skip_plots)
            
            # Check for reversal signals
            reversal_signals = self._check_reversal_signals(symbol, primary_df, higher_df, skip_plots)
            
            # Collect all signals for this symbol
            symbol_signals = []
            if breakout_signals:
                symbol_signals.extend(breakout_signals)
            
            if reversal_signals:
                symbol_signals.extend(reversal_signals)
                
            # Score signals using helper
            symbol_signals = self._score_signals(symbol_signals, primary_df, higher_df)
            
            # For each symbol, return the best signal immediately if requested
            if process_immediately and symbol_signals:
                # Find best signal for this symbol
                best_signal = max(symbol_signals, key=lambda x: x.get('score', 0))
                
                # Log all signals with their scores for debugging
                for signal in symbol_signals:
                    logger.debug(f"Signal {signal['direction']} for {symbol}: {signal.get('reason', 'No reason')} - Score: {signal.get('score', 0):.2f}")
                
                logger.info(f"🌟 Selected best signal for {symbol}: {best_signal['direction']} {best_signal.get('reason', 'No reason')} with score {best_signal.get('score', 0):.2f}")
                
                # Remove scoring metadata before returning
                if 'original_symbol' in best_signal:
                    del best_signal['original_symbol']
                if 'score_details' in best_signal:
                    del best_signal['score_details']
                
                # Return this single signal in a list for immediate processing
                symbol_time = time.time() - symbol_start_time
                logger.info(f"📊 Generated signal for {symbol} in {symbol_time:.2f}s: {best_signal['direction']} at {best_signal['entry_price']:.5f} | confidence: {best_signal['confidence']:.2f}")
                logger.info(f"👉 RETURNING IMMEDIATE SIGNAL FOR {symbol}")
                return [best_signal]
            
            # Add to all signals collection for batch processing
            all_signals.extend(symbol_signals)
        
        # After processing all symbols, add this log before signal selection:
        if all_signals:
            logger.info(f"👉 Found {len(all_signals)} potential signals before scoring and selection")
            
            # Group signals by symbol
            signals_by_symbol = {}
            for signal in all_signals:
                symbol = signal['original_symbol']
                if symbol not in signals_by_symbol:
                    signals_by_symbol[symbol] = []
                signals_by_symbol[symbol].append(signal)
            
            # For each symbol, only select highest scoring signal
            for symbol, symbol_signals in signals_by_symbol.items():
                if not symbol_signals:
                    continue
                    
                # Find highest scoring signal
                best_signal = max(symbol_signals, key=lambda x: x.get('score', 0))
                
                # Log all signals with their scores for debugging
                for signal in symbol_signals:
                    logger.debug(f"Signal {signal['direction']} for {symbol}: {signal.get('reason', 'No reason')} - Score: {signal.get('score', 0):.2f}")
                
                logger.info(f"🌟 Selected best signal for {symbol}: {best_signal['direction']} {best_signal.get('reason', 'No reason')} with score {best_signal.get('score', 0):.2f}")
                
                # Remove scoring metadata before returning
                if 'original_symbol' in best_signal:
                    del best_signal['original_symbol']
                if 'score_details' in best_signal:
                    del best_signal['score_details']
                
                signals.append(best_signal)
        
        generation_time = time.time() - start_time
        logger.info(f"✅ Generation completed in {generation_time:.2f}s - Produced {len(signals)} final signals")
        if signals:
            for i, signal in enumerate(signals):
                logger.info(f"📊 Final Signal #{i+1}: {signal['symbol']} {signal['direction']} at {signal['entry_price']:.5f} | confidence: {signal['confidence']:.2f}")
            logger.info(f"👉 RETURNING {len(signals)} SIGNALS FOR PROCESSING")
        else:
            logger.info("📭 No signals generated - returning empty list")
        
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
            # If it's a timestamp (integer), convert to datetime
            try:
                if isinstance(current_time, (int, np.integer, float)):
                    # Try different units for timestamp conversion
                    try:
                        current_time = datetime.fromtimestamp(current_time)
                    except (ValueError, OverflowError):
                        try:
                            current_time = datetime.fromtimestamp(current_time / 1000)  # Try milliseconds
                        except:
                            current_time = datetime.now()
                elif isinstance(current_time, pd.Timestamp):
                    current_time = current_time.to_pydatetime()
                else:
                    # For other types, use string conversion
                    current_time = pd.to_datetime(str(current_time)).to_pydatetime()
            except Exception as e:
                # If all conversions fail, use current time
                logger.debug(f"Failed to convert timestamp: {e}, using current time instead")
                current_time = datetime.now()
        
        if symbol in self.last_updated:
            last_update = self.last_updated[symbol]
            # Ensure last_update is a datetime object too
            if not isinstance(last_update, datetime):
                logger.debug(f"Converting last_update from {type(last_update)} to datetime")
                try:
                    if isinstance(last_update, (int, np.integer, float)):
                        # Try different units for timestamp conversion
                        try:
                            last_update = datetime.fromtimestamp(last_update)
                        except (ValueError, OverflowError):
                            try:
                                last_update = datetime.fromtimestamp(last_update / 1000)  # Try milliseconds
                            except:
                                # Force update in case of conversion error
                                last_update = datetime.now() - timedelta(hours=self.level_update_interval + 1)
                    elif isinstance(last_update, pd.Timestamp):
                        last_update = last_update.to_pydatetime()
                    else:
                        # For other types, use string conversion
                        last_update = pd.to_datetime(str(last_update)).to_pydatetime()
                except Exception as e:
                    # If all conversions fail, force an update
                    logger.debug(f"Failed to convert last_update: {e}, forcing update")
                    last_update = datetime.now() - timedelta(hours=self.level_update_interval + 1)
                
                # Update the stored value
                self.last_updated[symbol] = last_update
            
            # Only update every X hours depending on timeframe
            try:
                time_diff = (current_time - last_update).total_seconds()
                if time_diff < self.level_update_interval * 3600:
                    logger.debug(f"🕒 Skipping level update for {symbol}, last update was {time_diff/3600:.1f} hours ago")
                    return
            except Exception as e:
                logger.warning(f"Error calculating time difference: {e}. Forcing update.")
                # Force update in case of error
        
        logger.debug(f"🔄 Updating key levels for {symbol} with {len(df)} candles")
        
        # Find swing highs and lows
        support_levels = self._find_support_levels(df)
        resistance_levels = self._find_resistance_levels(df)
        
        # Store levels
        self.support_levels[symbol] = support_levels
        self.resistance_levels[symbol] = resistance_levels
        self.last_updated[symbol] = current_time
        
        logger.info(f"🔄 Updated key levels for {symbol} - Support: {len(support_levels)}, Resistance: {len(resistance_levels)}")
        
        # Log actual levels for debugging
        if support_levels:
            logger.debug(f"📉 Support levels for {symbol}: {[round(level['zone_max'], 5) for level in support_levels]}")
        if resistance_levels:
            logger.debug(f"📈 Resistance levels for {symbol}: {[round(level['zone_min'], 5) for level in resistance_levels]}")
            
    def _find_trend_lines(self, symbol: str, df: pd.DataFrame, debug_force_update: bool = False, skip_plots: bool = False) -> None:
        """
        Find and validate trend lines for a given symbol.
        Only plot trendlines and raw price series if skip_plots is False.
        """
        # Only plot raw price series for debugging if skip_plots is False
        if not skip_plots:
            plot_raw_price_series(df, symbol, self.primary_timeframe)
        current_time = datetime.now()
        last_update_time = self.last_updated.get('trend_lines', {}).get(symbol, None)
        force_update = debug_force_update

        if (not force_update and last_update_time is not None and 
            (current_time - last_update_time).total_seconds() < self.trend_line_update_interval * 3600):
            logger.debug(f"⏭️ Skipping trend line update for {symbol} - last update: {last_update_time}")
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
                for i, line in enumerate(bullish_trend_lines):
                    logger.debug(f"  📈 Bullish Line #{{i+1}}: Angle={{line['angle']:.2f}}°, r²={{line['r_squared']:.3f}}, Touches={{line['touches']}}")
            if bearish_trend_lines:
                logger.debug(f"📉 BEARISH TREND LINES for {symbol} (skipping plots)")
                for i, line in enumerate(bearish_trend_lines):
                    logger.debug(f"  📉 Bearish Line #{{i+1}}: Angle={{line['angle']:.2f}}°, r²={{line['r_squared']:.3f}}, Touches={{line['touches']}}")
            self.last_updated['trend_lines'][symbol] = current_time
            return
        # Create debug plots directory if it doesn't exist
        debug_dir = Path("debug_plots")
        debug_dir.mkdir(exist_ok=True)
        plt.figure(figsize=(15, 10))
        plot_range = min(200, len(df))
        # Use datetime index for x-axis
        x_dates = df.index[-plot_range:]
        plt.plot(x_dates, df['close'].iloc[-plot_range:], color='blue', alpha=0.5, label='Close Price')
        plt.plot(x_dates, df['high'].iloc[-plot_range:], color='green', alpha=0.3, label='High')
        plt.plot(x_dates, df['low'].iloc[-plot_range:], color='red', alpha=0.3, label='Low')
        # Plot swing highs and lows
        swing_highs, swing_lows = analyzer.find_swings()
        if swing_highs:
            high_x = [df.index[x] for x, y in swing_highs if x >= len(df) - plot_range]
            high_y = [y for x, y in swing_highs if x >= len(df) - plot_range]
            plt.scatter(high_x, high_y, color='green', marker='^', s=50, label='Swing Highs')
            if len(high_x) >= 2:
                for i in range(len(high_x) - 1):
                    plt.plot([high_x[i], high_x[i+1]], [high_y[i], high_y[i+1]], color='lightgreen', linestyle='--', alpha=0.5)
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
            y_line = line['slope'] * x_line + line['intercept']
            x_line_dt = [datetime.fromtimestamp(x) for x in x_line]
            plt.plot(x_line_dt, y_line, color='green', linewidth=2, alpha=0.7, label=f"Support: Angle={line['angle']:.1f}°, Touches={line['touches']}")
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
            y_line = line['slope'] * x_line + line['intercept']
            x_line_dt = [datetime.fromtimestamp(x) for x in x_line]
            plt.plot(x_line_dt, y_line, color='red', linewidth=2, alpha=0.7, label=f"Resistance: Angle={line['angle']:.1f}°, Touches={line['touches']}")
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
        self.last_updated['trend_lines'][symbol] = current_time
    
    def _find_swing_highs(self, df: pd.DataFrame, window: int = 5) -> list:
        """
        Find swing highs (pivot highs) using a windowed approach.
        Returns a list of (index, high) tuples.
        """
        if df is None or len(df) < 2 * window + 1:
            return []
        pivots_high = []
        for i in range(window, len(df) - window):
            if df['high'].iloc[i] == max(df['high'].iloc[i-window:i+window+1]):
                pivots_high.append((i, df['high'].iloc[i]))
        return pivots_high
    
    def _find_swing_lows(self, df: pd.DataFrame, window: int = 5) -> list:
        """
        Find swing lows (pivot lows) using a windowed approach.
        Returns a list of (index, low) tuples.
        """
        if df is None or len(df) < 2 * window + 1:
            return []
        pivots_low = []
        for i in range(window, len(df) - window):
            if df['low'].iloc[i] == min(df['low'].iloc[i-window:i+window+1]):
                pivots_low.append((i, df['low'].iloc[i]))
        return pivots_low
    
    def _identify_trend_lines(self, df: pd.DataFrame, swing_points: list, line_type: str, skip_plots: bool = False) -> list:
        """
        Fit a trendline to the given swing points using linear regression.
        Returns a list with a single trendline dict if valid, else empty list.
        """
        import numpy as np
        from scipy.stats import linregress
        min_points = getattr(self, 'trend_line_min_points', 3)
        r2_threshold = getattr(self, 'trend_line_r2_threshold', 0.7)
        if len(swing_points) < min_points:
            self.logger.debug(f"Not enough swing points ({len(swing_points)}) to identify {line_type} trend lines. Need at least {min_points}.")
            return []
        x = np.array([p[0] for p in swing_points])
        y = np.array([p[1] for p in swing_points])
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        r_squared = float(r_value) ** 2
        if r_squared < r2_threshold:
            self.logger.debug(f"Rejected {line_type} trendline: r_squared={r_squared:.2f} < threshold {r2_threshold}")
            return []
        import math
        trendline = {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'start_idx': int(x[0]),
            'end_idx': int(x[-1]),
            'line_type': line_type,
            'points': swing_points,
            'quality_score': r_squared * len(swing_points),
            # Newly added metadata for downstream processing
            'angle': math.degrees(math.atan(slope)) if slope is not None else 0.0,
            'x_start': float(x[0]),
            'x_end': float(x[-1]),
            'touches': len(swing_points),
        }
        return [trendline]
    
    def _count_trend_line_touches(self, df: pd.DataFrame, slope: float, intercept: float, 
                                  line_type: str, atr: float = None, start_idx: int = None, end_idx: int = None) -> int:
        """
        Count how many times price has touched a trend line, using ATR-based tolerance and only between start_idx and end_idx.
        """
        if atr is None or pd.isna(atr) or atr == 0:
            atr = df['close'].rolling(14).std().iloc[-1] if len(df) >= 14 else df['close'].std()
        if atr is None or pd.isna(atr) or atr == 0:
            atr = 1e-4
        touches = 0
        price_series = df['low'] if line_type == 'bullish' else df['high']
        # Only check between start_idx and end_idx
        if start_idx is None or end_idx is None:
            start_idx = 0
            end_idx = len(df) - 1
        for i in range(start_idx, end_idx + 1):
            ts = df.index[i]
            if isinstance(ts, pd.Timestamp):
                x = ts.timestamp()
            elif isinstance(ts, datetime):
                x = ts.timestamp()
            else:
                x = float(i)
            line_value = slope * x + intercept
            price = price_series.iloc[i]
            if abs(price - line_value) <= atr * 0.25:  # 0.25 ATR as touch tolerance
                    touches += 1
        return min(touches, 20)
    
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
        """
        Identify recent consolidation ranges for target calculation using volatility metrics.
        Detects periods where price range is less than 50% of the average range over recent bars.
        
        Args:
            symbol: Trading symbol
            df: Price dataframe
        """
        # Look for periods where price is ranging (not trending strongly)
        if symbol in self.last_consolidation_ranges:
            last_update = self.last_updated.get(symbol, datetime.min)
            current_time = df.index[-1]
            
            # Ensure current_time is a datetime object
            if not isinstance(current_time, datetime):
                logger.debug(f"Converting current_time from {type(current_time)} to datetime in consolidation ranges")
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
                self.last_updated[symbol] = last_update
            
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
            atr_series = calculate_atr(df, self.atr_period)
            if not isinstance(atr_series, pd.Series) or atr_series.empty:
                # Handle error case
                atr = None
            else:
                atr = atr_series.iloc[-1]  # Get the most recent ATR value

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
                    
                    # Define consolidation as period where:
                    # 1. Recent range average is less than 50% of overall average range
                    # 2. Standard deviation is low relative to price
                    # 3. Total range is reasonable (not too wide)
                    # Fix: Use explicit boolean checks instead of Series comparison
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
            
            # Store the range information
            self.last_consolidation_ranges[symbol] = {
                'high': range_high,
                'low': range_low,
                'size': range_size,
                'is_consolidation': is_consolidation
            }
            
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
            atr = calculate_atr(df, self.atr_period)
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

        # For breakout above resistance, we're looking for a retest from above
        if direction == 'bullish' and abs(current_price - level) <= price_tolerance and current_price > level:
            logger.info(f"✅ Confirmed bullish retest of {level:.5f} for {symbol} (ATR window: {price_tolerance:.5f})")
            # Update breakout tracking to indicate retest is confirmed
            self.retest_tracking[symbol]['retest_confirmed'] = True
            
        # For breakout below support, we're looking for a retest from below
        elif direction == 'bearish' and abs(current_price - level) <= price_tolerance and current_price < level:
            logger.info(f"✅ Confirmed bearish retest of {level:.5f} for {symbol} (ATR window: {price_tolerance:.5f})")
            # Update breakout tracking to indicate retest is confirmed
            self.retest_tracking[symbol]['retest_confirmed'] = True
    
    def _find_support_levels(self, df: pd.DataFrame) -> List[dict]:
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
        return self._cluster_levels(levels)

    def _find_resistance_levels(self, df: pd.DataFrame) -> List[dict]:
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
        return self._cluster_levels(levels)

    def _count_level_touches(self, df: pd.DataFrame, level: float, level_type: str) -> int:
        """
        Count how many times price has touched a level.
        
        Args:
            df: Price dataframe
            level: Price level to check
            level_type: 'support' or 'resistance'
            
        Returns:
            Number of touches
        """
        tolerance = level * self.price_tolerance
        count = 0
        
        if level_type == 'support':
            # Price approached from above and bounced
            for i in range(len(df)):
                if abs(df['low'].iloc[i] - level) <= tolerance:
                    count += 1
        else:  # resistance
            # Price approached from below and bounced
            for i in range(len(df)):
                if abs(df['high'].iloc[i] - level) <= tolerance:
                    count += 1
                    
        return count
    
    def _cluster_levels(self, levels: List[float]) -> List[dict]:
        """
        Cluster nearby levels to avoid duplicates, returning zones (min, max, avg, width).
        Args:
            levels: List of price levels
        Returns:
            List of dicts: {'zone_min', 'zone_max', 'zone_avg', 'zone_width'}
        """
        if not levels:
            return []
        sorted_levels = sorted(levels)
        clusters = []
        current_cluster = [sorted_levels[0]]
        for i in range(1, len(sorted_levels)):
            if sorted_levels[i] - sorted_levels[i-1] <= sorted_levels[i] * self.price_tolerance:
                current_cluster.append(sorted_levels[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [sorted_levels[i]]
        clusters.append(current_cluster)
        # For each cluster, return zone info
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
    
    def _check_breakout_signals(self, symbol: str, df: pd.DataFrame, h1_df: pd.DataFrame, skip_plots: bool = False) -> List[Dict]:
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
        self._log_candle_samples(df, symbol, count=5)
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
                signal = {
                    "symbol": symbol,
                    "direction": "buy",
                    "entry_price": retest_entry,
                    "stop_loss": retest_stop,
                    "take_profit": take_profit,
                    "timeframe": self.primary_timeframe,
                    "source": self.name,
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
                if result['valid']:
                    adjusted_signal = result['adjusted_trade']
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
                signal = {
                    "symbol": symbol,
                    "direction": "sell",
                    "entry_price": retest_entry,
                    "stop_loss": retest_stop,
                    "take_profit": take_profit,
                    "timeframe": self.primary_timeframe,
                    "source": self.name,
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
                if result['valid']:
                    adjusted_signal = result['adjusted_trade']
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
            volume_quality = self._analyze_volume_quality(current_candle, volume_threshold)
            logger.debug(f"📊 {symbol}: Volume quality score: {volume_quality:.1f} (>0 = bullish, <0 = bearish)")
            
            # Check each resistance level
            for level in resistance_levels:
                logger.debug(f"🔄 {symbol}: Checking resistance level {level['zone_min']:.5f}-{level['zone_max']:.5f}")
                # Updated breakout condition: require close > zone_max + price_tolerance*zone_max
                if (
                    previous_candle['close'] <= level['zone_max'] * (1 + self.price_tolerance)
                    and current_candle['close'] > level['zone_max'] * (1 + self.price_tolerance)
                ):
                    # False-breakout filter for bullish breakouts
                    if self.detect_false_breakout(df.iloc[:i+1], direction='bullish', price_tolerance=self.price_tolerance).iloc[-1]:
                        logger.debug(f"🚫 False breakout detected for {symbol} at resistance breakout, skipping signal.")
                        continue  # skip this candle—likely a fakeout
                    # Generate buy signal
                    entry_price = current_candle['close']
                    
                    # Place stop under the breakout candle's low
                    stop_loss = min(current_candle['low'], previous_candle['low'])
                    
                    # Log the breakout regardless of whether we generate a signal
                    logger.info(f"👀 Detected potential breakout for {symbol} at level {level['zone_max']:.5f}")
                    
                    # RELAXED conditions: allow signals with neutral H1 trend, don't require strong volume
                    if h1_trend != 'bearish':  # Just avoid counter-trend signals
                        # Generate buy signal
                        # (existing code for generating signals)
                        
                        # Add detailed logging
                        logger.debug(f"Breakout details: Close={current_candle['close']:.5f}, " +
                                   f"Level={level['zone_max']:.5f}, Volume quality={volume_quality:.2f}, " +
                                   f"H1 trend={h1_trend}")
                        
                        # Generate trade signals...
                        # (Rest of the existing code)
                
            # Check trend line breakouts (bullish) - RELAXED conditions
            for trend_line in bearish_trend_lines:
                # Calculate trend line value at current and previous candle
                prev_line_value = self._calculate_trend_line_value(trend_line, i-1)
                curr_line_value = self._calculate_trend_line_value(trend_line, i)
                
                # DEBUG: Log the trendline values
                logger.debug(f"Trendline values: prev={prev_line_value:.5f}, curr={curr_line_value:.5f}, " +
                           f"close={current_candle['close']:.5f}, candle index={i}")
                
                # RELAXED: Breakout condition - now doesn't require strong candle, allows neutral trend
                # Previous candle below trend line, current candle closing above
                if (previous_candle['close'] <= prev_line_value * (1 + self.price_tolerance) and
                    current_candle['close'] > curr_line_value * (1 + self.price_tolerance) and
                    h1_trend != 'bearish'):  # Just avoid counter-trend signals
                    
                    # Generate buy signal
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
                            'confirmed': False,
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
                            "source": self.name,
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
                        signals.append(signal)
                        logger.info(f"🟢 TREND LINE BREAKOUT BUY: {symbol} at {entry_price:.5f} | SL: {stop_loss:.5f} | TP: {take_profit:.5f}")
        
        # Check for support breakdowns (horizontal levels)
        for i in range(-candles_to_check, 0):
            current_candle = df.iloc[i]
            previous_candle = df.iloc[i-1]
            
            # Volume analysis with wick structure
            volume_quality = self._analyze_volume_quality(current_candle, volume_threshold)
            
            # Check each support level
            for level in support_levels:
                logger.debug(f"🔄 {symbol}: Checking support level {level['zone_min']:.5f}-{level['zone_max']:.5f}")
                # Updated breakdown condition: require close < zone_min - price_tolerance*zone_min
                if (
                    previous_candle['close'] >= level['zone_min'] * (1 - self.price_tolerance)
                    and current_candle['close'] < level['zone_min'] * (1 - self.price_tolerance)
                    and self._is_strong_candle(current_candle)
                    and volume_quality < 0
                    and h1_trend == 'bearish'
                ):
                    # False-breakout filter for bearish breakdowns
                    if self.detect_false_breakout(df.iloc[:i+1], direction='bearish', price_tolerance=self.price_tolerance).iloc[-1]:
                        logger.debug(f"🚫 False breakdown detected for {symbol} at support breakdown, skipping signal.")
                        continue  # skip this candle—likely a fakeout
                    # Generate sell signal
                    entry_price = current_candle['close']
                    
                    # Place stop above the breakdown candle's high
                    stop_loss = max(current_candle['high'], previous_candle['high'])
                    
                    # Advanced target calculation
                    if symbol in self.last_consolidation_ranges:
                        range_size = self.last_consolidation_ranges[symbol]['size']
                        risk = stop_loss - entry_price
                        calculated_target = level['zone_max'] - range_size
                        min_target = entry_price - (risk * self.min_risk_reward)
                        take_profit = min(calculated_target, min_target)
                    else:
                        # Fallback to minimum risk-reward
                        risk = stop_loss - entry_price
                        take_profit = entry_price - (risk * self.min_risk_reward)
                    
                    # Reason with volume quality description
                    volume_desc = "strong bearish volume" if volume_quality < -1 else "adequate volume"
                    reason = f"Bearish breakdown below support at {level['zone_max']:.5f} with {volume_desc}"
                    
                    # If retest is required, don't generate signal now but track for retest
                    if self.retest_required:
                        # Store breakout info for retest tracking
                        self.retest_tracking[symbol] = {
                            'level': level,
                            'direction': 'sell',
                            'entry_price': entry_price,
                            'stop_loss': stop_loss,
                            'start_time': df.index[i],
                            'confirmed': False,
                            'reason': reason
                        }
                        logger.info(f"👀 TRACKING RETEST: {symbol} bearish breakdown at {level['zone_max']:.5f}")
                    else:
                        # Create immediate signal if retest not required
                        signal = {
                            "symbol": symbol,
                            "direction": "sell",
                            "entry_price": entry_price,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "timeframe": self.primary_timeframe,
                            "confidence": 0.0,  # placeholder, will update after scoring
                            "source": self.name,
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
                        signals.append(signal)
                        logger.info(f"🔴 BREAKDOWN SELL: {symbol} at {entry_price:.5f} | Level: {level['zone_max']:.5f} | SL: {stop_loss:.5f} | TP: {take_profit:.5f}")
            
            # Check trend line breakdowns (bearish) - RELAXED conditions
            for trend_line in bullish_trend_lines:
                prev_line_value = self._calculate_trend_line_value(trend_line, i-1)
                curr_line_value = self._calculate_trend_line_value(trend_line, i)
                # RELAXED: allow signal if higher timeframe is not bullish (remove strong candle/volume requirements)
                if (previous_candle['close'] >= prev_line_value * (1 - self.price_tolerance) and
                    current_candle['close'] < curr_line_value * (1 - self.price_tolerance) and
                    h1_trend != 'bullish'):
                    # Generate sell signal
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
                            'confirmed': False,
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
                            "source": self.name,
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
                        signals.append(signal)
                        logger.info(f"🔴 TREND LINE BREAKDOWN SELL (relaxed): {symbol} at {entry_price:.5f} | SL: {stop_loss:.5f} | TP: {take_profit:.5f}")
        
        return signals
    
    def _check_reversal_signals(self, symbol: str, df: pd.DataFrame, h1_df: pd.DataFrame, skip_plots: bool = False) -> List[Dict]:
        """
        Check for reversal signals at key support and resistance levels.
        
        Args:
            symbol: Trading symbol
            df: Price dataframe for primary timeframe
            h1_df: Price dataframe for higher timeframe
            skip_plots: Whether to skip creating debug plots
            
        Returns:
            List of reversal signal dictionaries
        """
        signals = []
        
        # Get support and resistance levels
        if symbol not in self.resistance_levels or symbol not in self.support_levels:
            logger.debug(f"❓ No support/resistance levels available for {symbol}")
            return signals
            
        resistance_levels = self.resistance_levels[symbol]
        support_levels = self.support_levels[symbol]
        
        logger.debug(f"🔍 {symbol}: Checking reversals with {len(resistance_levels)} resistance and {len(support_levels)} support levels")
        
        # Determine the trend context from the higher timeframe
        h1_trend = self._determine_higher_timeframe_trend(h1_df)
        is_downtrend = h1_trend == 'bearish'
        
        # Prepare volume data and debug logs
        self._ensure_tick_volume(df, symbol)
        self._log_candle_samples(df, symbol, count=5)
        volume_threshold = self._compute_volume_threshold(df)
        
        # Define number of recent candles to check for patterns
        candles_to_check = 10
        
        # Prepare trend line lists
        trend_lines = self.bullish_trend_lines.get(symbol, []) + self.bearish_trend_lines.get(symbol, [])
        bullish_trend_lines = [line for line in trend_lines if line['angle'] < 60]
        bearish_trend_lines = [line for line in trend_lines if line['angle'] > -60]

        # Precompute vectorized pattern detections for efficiency
        hammer_pattern = self.detect_hammer(df, self.price_tolerance)
        shooting_star_pattern = self.detect_shooting_star(df, self.price_tolerance)
        bullish_engulfing_pattern = self.detect_bullish_engulfing(df)
        bearish_engulfing_pattern = self.detect_bearish_engulfing(df)
        morning_star_pattern = self.detect_morning_star(df, self.price_tolerance)
        evening_star_pattern = self.detect_evening_star(df, self.price_tolerance)
        # False breakout is more custom, so keep as is for now

        # Check for reversal at support (bullish patterns)
        for i in range(-candles_to_check+1, 0):
            current_candle = df.iloc[i]
            previous_candle = df.iloc[i-1]
            logger.debug(f"📊 {symbol}: Checking reversal at candle {df.index[i]}: O={current_candle['open']:.5f} H={current_candle['high']:.5f} L={current_candle['low']:.5f} C={current_candle['close']:.5f}")
            volume_quality = self._analyze_volume_quality(current_candle, volume_threshold)
            logger.debug(f"📊 {symbol}: Volume quality score: {volume_quality:.1f} (>0 = bullish, <0 = bearish)")
            for level in support_levels:
                logger.debug(f"🔄 {symbol}: Checking support level {level['zone_max']:.5f}")
                is_near_support = abs(current_candle['low'] - level['zone_max']) <= self._get_dynamic_tolerance(df, i, fallback_price=level['zone_max'])
                logger.debug(f"✓ {symbol}: Price near support: {is_near_support} (Low: {current_candle['low']:.5f}, Support: {level['zone_max']:.5f}, Tolerance: {self._get_dynamic_tolerance(df, i, fallback_price=level['zone_max']):.5f})")
                if is_near_support:
                    # Use only precomputed vectorized pattern detection
                    pattern_types = []
                    idx = i if i >= 0 else len(df) + i
                    if hammer_pattern.iloc[idx]:
                        pattern_types.append("Hammer")
                    if bullish_engulfing_pattern.iloc[idx]:
                        pattern_types.append("Bullish Engulfing")
                    if morning_star_pattern.iloc[idx]:
                        pattern_types.append("Morning Star")
                    pattern_type = ", ".join(pattern_types) if pattern_types else None
                    volume_desc = "strong bullish volume" if volume_quality > 1 else "adequate volume"
                    if pattern_type:
                        logger.info(f"⚡ {symbol}: Detected bullish reversal pattern ({pattern_type}) at support {level['zone_max']:.5f}")
                        # Use the pattern's close as entry, not the latest bar
                        entry_price = df.iloc[i]['close']
                        stop_loss = current_candle['low'] - max(level['zone_max'] * self.price_tolerance, self.atr_multiplier * self.atr_period)
                        risk = entry_price - stop_loss
                        next_resistance = self._find_next_resistance(df, entry_price, resistance_levels)
                        if next_resistance:
                            reward_to_resistance = next_resistance - entry_price
                            min_reward = risk * self.min_risk_reward
                            if reward_to_resistance >= min_reward:
                                take_profit = next_resistance
                            else:
                                take_profit = entry_price + min_reward
                        else:
                            take_profit = entry_price + (risk * self.min_risk_reward)
                        signal = {
                            "symbol": symbol,
                            "direction": "buy",
                            "entry_price": entry_price,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "timeframe": self.primary_timeframe,
                            "confidence": 0.0,
                            "source": self.name,
                            "generator": self.name,
                            "reason": f"Bullish reversal ({pattern_type}) at support {level['zone_max']:.5f} with {volume_desc}",
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
                        signals.append(signal)
                        logger.info(f"🟢 REVERSAL BUY: {symbol} at {entry_price:.5f} | Pattern: {pattern_type} | Level: {level['zone_max']:.5f} | SL: {stop_loss:.5f} | TP: {take_profit:.5f}")
                    else:
                        if not pattern_type:
                            logger.debug(f"❌ {symbol}: No bullish pattern detected")
                        if volume_quality <= 0:
                            logger.debug(f"❌ {symbol}: Insufficient bullish volume (quality: {volume_quality:.1f})")
            # Check for reversal at bullish trend lines
            for trend_line in bullish_trend_lines:
                line_value = self._calculate_trend_line_value(trend_line, i)
                
                logger.debug(f"🔄 {symbol}: Checking bullish trend line at price {line_value:.5f}")
                
                # Price near trend line
                is_near_trendline = abs(current_candle['low'] - line_value) <= self._get_dynamic_tolerance(df, i, fallback_price=line_value)
                logger.debug(f"✓ {symbol}: Price near trend line: {is_near_trendline} (Low: {current_candle['low']:.5f}, Trend line: {line_value:.5f})")
                
                if is_near_trendline:
                    # Detect bullish reversal pattern
                    pattern_type = self._detect_bullish_reversal_pattern(df, i, is_downtrend)
                    volume_desc = "strong bullish volume" if volume_quality > 1 else "adequate volume"
                    
                    if pattern_type:
                        logger.info(f"⚡ {symbol}: Detected bullish reversal pattern ({pattern_type}) at trend line with {volume_desc}")
                        
                        # Generate buy signal
                        entry_price = current_candle['close']
                        
                        # Stop loss below the reversal candle low
                        stop_loss = current_candle['low'] - max(line_value * self.price_tolerance, self.atr_multiplier * self.atr_period)
                        
                        # Target: Either next resistance or at least 2x risk
                        risk = entry_price - stop_loss
                        
                        logger.debug(f"📐 {symbol}: Entry: {entry_price:.5f}, Stop: {stop_loss:.5f}, Risk: {risk:.5f}")
                        
                        # Advanced target calculation - find nearest resistance above
                        next_resistance = self._find_next_resistance(df, entry_price, resistance_levels)
                        
                        if next_resistance:
                            logger.debug(f"🎯 {symbol}: Found next resistance at {next_resistance:.5f}")
                            
                            # Check if next resistance provides enough reward
                            reward_to_resistance = next_resistance - entry_price
                            min_reward = risk * self.min_risk_reward
                            
                            logger.debug(f"📊 {symbol}: Reward to resistance: {reward_to_resistance:.5f}, Min required: {min_reward:.5f}")
                            
                            if reward_to_resistance >= min_reward:
                                take_profit = next_resistance
                                logger.debug(f"✅ {symbol}: Using next resistance as target: {take_profit:.5f}")
                            else:
                                take_profit = entry_price + min_reward
                                logger.debug(f"⚠️ {symbol}: Resistance too close, using min RR target: {take_profit:.5f}")
                        else:
                            take_profit = entry_price + (risk * self.min_risk_reward)
                            logger.debug(f"ℹ️ {symbol}: No resistance found, using min RR target: {take_profit:.5f}")
                        
                        # Volume description
                        volume_desc = "strong bullish volume" if volume_quality > 1 else "adequate volume"
                        
                        # Create signal
                        signal = {
                            "symbol": symbol,
                            "direction": "buy",
                            "entry_price": entry_price,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "timeframe": self.primary_timeframe,
                            "confidence": 0.0,  # placeholder, will update after scoring
                            "source": self.name,
                            "generator": self.name,
                            "reason": f"Bullish reversal ({pattern_type}) at trend line with {volume_desc}",
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
                        signals.append(signal)
                        logger.info(f"🟢 TREND LINE REVERSAL BUY: {symbol} at {entry_price:.5f} | Pattern: {pattern_type} | SL: {stop_loss:.5f} | TP: {take_profit:.5f}")
                    else:
                        if not pattern_type:
                            logger.debug(f"❌ {symbol}: No bullish pattern detected")
                        if volume_quality <= 0:
                            logger.debug(f"❌ {symbol}: Insufficient bullish volume (quality: {volume_quality:.1f})")
        
        # Check for reversal at resistance (bearish patterns)
        for i in range(-candles_to_check+1, 0):
            current_candle = df.iloc[i]
            previous_candle = df.iloc[i-1]
            
            # Volume analysis with wick structure
            volume_quality = self._analyze_volume_quality(current_candle, volume_threshold)
            
            # Check each resistance level
            for level in resistance_levels:
                logger.debug(f"🔄 {symbol}: Checking resistance level {level['zone_min']:.5f}")
                if abs(current_candle['high'] - level['zone_min']) <= self._get_dynamic_tolerance(df, i, fallback_price=level['zone_min']):
                    # Use only precomputed vectorized pattern detection
                    pattern_types = []
                    idx = i if i >= 0 else len(df) + i
                    if shooting_star_pattern.iloc[idx]:
                        pattern_types.append("Shooting Star")
                    if bearish_engulfing_pattern.iloc[idx]:
                        pattern_types.append("Bearish Engulfing")
                    if evening_star_pattern.iloc[idx]:
                        pattern_types.append("Evening Star")
                    pattern_type = ", ".join(pattern_types) if pattern_types else None
                    volume_desc = "strong bearish volume" if volume_quality < -1 else "adequate volume"

                    if pattern_type:
                        # Generate sell signal
                        entry_price = current_candle['close']
                        # Stop loss above the reversal candle high
                        stop_loss = current_candle['high'] + max(level['zone_min'] * self.price_tolerance, self.atr_multiplier * self.atr_period)
                        # Target: Either next support or at least 2x risk
                        risk = stop_loss - entry_price
                        # Advanced target calculation - find nearest support below
                        next_support = self._find_next_support(df, entry_price, support_levels)
                        if next_support and (entry_price - next_support) >= (risk * self.min_risk_reward):
                            take_profit = next_support
                        else:
                            take_profit = entry_price - (risk * self.min_risk_reward)
                        # Volume description
                        volume_desc = "strong bearish volume" if volume_quality < -1 else "adequate volume"
                        # Create signal
                        signal = {
                            "symbol": symbol,
                            "direction": "sell",
                            "entry_price": entry_price,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "timeframe": self.primary_timeframe,
                            "confidence": 0.0,  # placeholder, will update after scoring
                            "source": self.name,
                            "generator": self.name,
                            "reason": f"Bearish reversal ({pattern_type}) at resistance {level['zone_min']:.5f} with {volume_desc}",
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
                        signals.append(signal)
                        logger.info(f"🔴 REVERSAL SELL: {symbol} at {entry_price:.5f} | Pattern: {pattern_type} | Level: {level['zone_min']:.5f} | SL: {stop_loss:.5f} | TP: {take_profit:.5f}")
                        signals.append(signal)
                    else:
                        if not pattern_type:
                            logger.debug(f"❌ {symbol}: No bearish pattern detected")
                        if volume_quality >= 0:
                            logger.debug(f"❌ {symbol}: Insufficient bearish volume (quality: {volume_quality:.1f})")
            
            # Check for reversal at bearish trend lines
            for trend_line in bearish_trend_lines:
                # Calculate trend line value at current position
                line_value = self._calculate_trend_line_value(trend_line, i)
                # Price near trend line
                if abs(current_candle['high'] - line_value) <= self._get_dynamic_tolerance(df, i, fallback_price=line_value):
                    # Detect bearish reversal pattern
                    # Get the uptrend state instead of downtrend for bearish pattern
                    is_uptrend = self._determine_higher_timeframe_trend(h1_df) == 'bullish'
                    pattern_type = self._detect_bearish_reversal_pattern(df, i, is_uptrend)
                    volume_desc = "strong bearish volume" if volume_quality < -1 else "adequate volume"

                    if pattern_type:
                        # Generate sell signal
                        entry_price = current_candle['close']
                        # Stop loss above the reversal candle high
                        stop_loss = current_candle['high'] + max(line_value * self.price_tolerance, self.atr_multiplier * self.atr_period)
                        # Target: Either next support or at least 2x risk
                        risk = stop_loss - entry_price
                        # Advanced target calculation - find nearest support below
                        next_support = self._find_next_support(df, entry_price, support_levels)
                        if next_support and (entry_price - next_support) >= (risk * self.min_risk_reward):
                            take_profit = next_support
                        else:
                            take_profit = entry_price - (risk * self.min_risk_reward)
                        # Volume description
                        volume_desc = "strong bearish volume" if volume_quality < -1 else "adequate volume"
                        # Create signal
                        signal = {
                            "symbol": symbol,
                            "direction": "sell",
                            "entry_price": entry_price,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "timeframe": self.primary_timeframe,
                            "confidence": 0.0,  # placeholder, will update after scoring
                            "source": self.name,
                            "generator": self.name,
                            "reason": f"Bearish reversal ({pattern_type}) at trend line with {volume_desc}",
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
                        signals.append(signal)
                        logger.info(f"🔴 TREND LINE REVERSAL SELL: {symbol} at {entry_price:.5f} | {pattern_type} | SL: {stop_loss:.5f} | TP: {take_profit:.5f}")
                        signals.append(signal)
                    else:
                        if not pattern_type:
                            logger.debug(f"❌ {symbol}: No bearish pattern detected")
                        if volume_quality >= 0:
                            logger.debug(f"❌ {symbol}: Insufficient bearish volume (quality: {volume_quality:.1f})")
        
        return signals
    
    def _find_next_resistance(self, df: pd.DataFrame, current_price: float, resistance_levels: List[dict], use_range_extension=None) -> Optional[float]:
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
    
    
    def _is_strong_candle(self, candle: pd.Series) -> bool:
        """
        Check if a candle is strong (body > 50% of range).
        
        Args:
            candle: Candle data
            
        Returns:
            True if it's a strong candle
        """
        total_range = candle['high'] - candle['low']
        body = abs(candle['close'] - candle['open'])
        
        if total_range == 0:
            return False
            
        body_percentage = body / total_range
        
        return bool(body_percentage > 0.5)
    
    def _determine_higher_timeframe_trend(self, higher_df: pd.DataFrame) -> str:
        """
        Determine the trend on the higher timeframe using price action instead of EMA.
        Uses swing highs and lows to identify the trend direction.
        
        Args:
            higher_df: Higher timeframe dataframe
            
        Returns:
            'bullish', 'bearish', or 'neutral'
        """
        if len(higher_df) < 20:
            logger.debug(f"⚠️ Not enough data for trend determination, need 20 candles but got {len(higher_df)}")
            return 'neutral'
        
        try:
            # Get a subset of recent data
            lookback = min(30, len(higher_df))
            df_subset = higher_df.iloc[-lookback:].copy()
            
            # Find swing highs and lows
            swing_highs = []
            swing_lows = []
            
            # We need at least 5 candles to establish a good pattern of swings
            if len(df_subset) < 5:
                logger.debug(f"⚠️ Insufficient data for swing analysis, using last 2 candles for simple trend")
                # Simple trend based on last 2 candles
                if float(df_subset['close'].iloc[-1]) > float(df_subset['close'].iloc[-2]):
                    return 'bullish'
                elif float(df_subset['close'].iloc[-1]) < float(df_subset['close'].iloc[-2]):
                    return 'bearish'
                else:
                    return 'neutral'
            
            # Use a rolling window to find swing points
            for i in range(2, len(df_subset) - 2):
                # Check for swing high
                if (float(df_subset['high'].iloc[i]) > float(df_subset['high'].iloc[i-1]) and 
                    float(df_subset['high'].iloc[i]) > float(df_subset['high'].iloc[i-2]) and
                    float(df_subset['high'].iloc[i]) > float(df_subset['high'].iloc[i+1]) and 
                    float(df_subset['high'].iloc[i]) > float(df_subset['high'].iloc[i+2])):
                    swing_highs.append((i, float(df_subset['high'].iloc[i])))
                
                # Check for swing low
                if (float(df_subset['low'].iloc[i]) < float(df_subset['low'].iloc[i-1]) and 
                    float(df_subset['low'].iloc[i]) < float(df_subset['low'].iloc[i-2]) and
                    float(df_subset['low'].iloc[i]) < float(df_subset['low'].iloc[i+1]) and 
                    float(df_subset['low'].iloc[i]) < float(df_subset['low'].iloc[i+2])):
                    swing_lows.append((i, float(df_subset['low'].iloc[i])))
            
            # Need at least two swing points of each type to determine trend
            if len(swing_highs) >= 2 and len(swing_lows) >= 2:
                # Get the last two swing highs and lows
                last_two_highs = sorted(swing_highs, key=lambda x: x[0])[-2:]
                last_two_lows = sorted(swing_lows, key=lambda x: x[0])[-2:]
                
                # Extract the values
                high1, high2 = last_two_highs[0][1], last_two_highs[1][1]
                low1, low2 = last_two_lows[0][1], last_two_lows[1][1]
                
                # Higher highs and higher lows = bullish trend
                # Lower highs and lower lows = bearish trend
                if high2 > high1 and low2 > low1:
                    trend = 'bullish'
                    logger.debug(f"📈 Bullish trend detected: Higher highs ({high1:.5f} → {high2:.5f}) and higher lows ({low1:.5f} → {low2:.5f})")
                elif high2 < high1 and low2 < low1:
                    trend = 'bearish'
                    logger.debug(f"📉 Bearish trend detected: Lower highs ({high1:.5f} → {high2:.5f}) and lower lows ({low1:.5f} → {low2:.5f})")
                else:
                    # Conflicting signals - check the most recent swing points
                    # If the most recent swing is a high, check if it's higher than previous
                    # If the most recent swing is a low, check if it's lower than previous
                    latest_swings = swing_highs + swing_lows
                    if not latest_swings:
                        trend = 'neutral'
                    else:
                        latest_swing = max(latest_swings, key=lambda x: x[0])
                        is_high = latest_swing in swing_highs
                        
                        if is_high:
                            # Latest swing is a high, compare to previous high
                            if len(swing_highs) >= 2:
                                prev_high = sorted(swing_highs, key=lambda x: x[0])[-2][1]
                                if latest_swing[1] > prev_high:
                                    trend = 'bullish'
                                else:
                                    trend = 'bearish'
                            else:
                                trend = 'neutral'
                        else:
                            # Latest swing is a low, compare to previous low
                            if len(swing_lows) >= 2:
                                prev_low = sorted(swing_lows, key=lambda x: x[0])[-2][1]
                                if latest_swing[1] < prev_low:
                                    trend = 'bearish'
                                else:
                                    trend = 'bullish'
                            else:
                                trend = 'neutral'
            else:
                # Not enough swing points, use price action from the last 5 candles
                recent_close = df_subset['close'].iloc[-5:].values
                # Convert to regular Python list if necessary
                if hasattr(recent_close, 'tolist'):
                    recent_close = recent_close.tolist()
                # Compare first and last price in the window
                if float(recent_close[-1]) > float(recent_close[0]):
                    trend = 'bullish'
                    logger.debug(f"📈 Bullish trend based on recent price movement: {float(recent_close[0]):.5f} → {float(recent_close[-1]):.5f}")
                elif float(recent_close[-1]) < float(recent_close[0]):
                    trend = 'bearish'
                    logger.debug(f"📉 Bearish trend based on recent price movement: {float(recent_close[0]):.5f} → {float(recent_close[-1]):.5f}")
                else:
                    trend = 'neutral'
                    logger.debug(f"📊 Neutral trend detected (no clear direction)")
            
            logger.debug(f"📊 Trend determined as {trend} using price action (swing highs/lows)")
            return trend
            
        except Exception as e:
            logger.warning(f"Error in trend determination: {str(e)}, falling back to simple method")
            # Simple fallback: compare current close to N periods ago
            try:
                periods_ago = min(10, len(higher_df) - 1)
                current_close = float(higher_df['close'].iloc[-1])
                past_close = float(higher_df['close'].iloc[-periods_ago])
                
                if current_close > past_close * 1.005:  # 0.5% higher
                    return 'bullish'
                elif current_close < past_close * 0.995:  # 0.5% lower
                    return 'bearish'
                else:
                    return 'neutral'
            except Exception as e2:
                logger.warning(f"Error in fallback trend determination: {str(e2)}")
                return 'neutral'  # Ultimate fallback
    
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
    
    
    def _detect_bullish_reversal_pattern(self, df: pd.DataFrame, idx: int, in_downtrend: bool) -> Optional[str]:
        """
        Detect bullish reversal patterns with trend context consideration.
        
        Args:
            df: DataFrame with price data
            idx: Index to check for pattern
            in_downtrend: Whether we're in a downtrend (makes bullish reversal more reliable)
            
        Returns:
            String describing detected pattern(s) or None
        """
        # Precompute vectorized patterns for the DataFrame
        hammer_pattern = self.detect_hammer(df, self.price_tolerance)
        bullish_engulfing_pattern = self.detect_bullish_engulfing(df)
        morning_star_pattern = self.detect_morning_star(df, self.price_tolerance)
        
        pattern_types = []
        
        # Apply trend context filter - bullish patterns are more reliable in downtrends
        if not in_downtrend:
            # If not in downtrend, require stronger confirmation of pattern quality
            # This is similar to the original implementation which checked confirmation
            
            # For hammer, check post-pattern confirmation by looking at next candle (if available)
            if hammer_pattern.iloc[idx] and idx < len(df) - 1:
                next_candle = df.iloc[idx + 1]
                # Only include hammer if next candle confirms with higher close
                if next_candle['close'] > df.iloc[idx]['close']:
                    pattern_types.append("Hammer (Confirmed)")
                else:
                    # Weaker evidence - still include but note it's unconfirmed
                    pattern_types.append("Hammer (Unconfirmed)")
            elif hammer_pattern.iloc[idx]:
                pattern_types.append("Hammer (Unconfirmed)")
                
            # For engulfing, check volume confirmation
            if bullish_engulfing_pattern.iloc[idx]:
                current_candle = df.iloc[idx]
                prev_candle = df.iloc[idx - 1] if idx > 0 else None
                
                if prev_candle is not None:
                    # Check if engulfing candle has higher volume than previous
                    current_vol = current_candle.get('tick_volume', current_candle.get('volume', 0))
                    prev_vol = prev_candle.get('tick_volume', prev_candle.get('volume', 0))
                    
                    if current_vol > prev_vol:
                        pattern_types.append("Bullish Engulfing (Volume Confirmed)")
                    else:
                        pattern_types.append("Bullish Engulfing")
                else:
                    pattern_types.append("Bullish Engulfing")
                    
            # Morning star has built-in confirmation in its 3-candle structure
            if morning_star_pattern.iloc[idx]:
                pattern_types.append("Morning Star")
        else:
            # In a downtrend, patterns are more reliable, so include as is
            if hammer_pattern.iloc[idx]:
                pattern_types.append("Hammer")
            if bullish_engulfing_pattern.iloc[idx]:
                pattern_types.append("Bullish Engulfing")
            if morning_star_pattern.iloc[idx]:
                pattern_types.append("Morning Star")
        
        if pattern_types:
            return ", ".join(pattern_types)
        return None

    def _detect_bearish_reversal_pattern(self, df: pd.DataFrame, idx: int, in_uptrend: bool) -> Optional[str]:
        """
        Detect bearish reversal patterns with trend context consideration.
        
        Args:
            df: DataFrame with price data
            idx: Index to check for pattern
            in_uptrend: Whether we're in an uptrend (makes bearish reversal more reliable)
            
        Returns:
            String describing detected pattern(s) or None
        """
        shooting_star_pattern = self.detect_shooting_star(df, self.price_tolerance)
        bearish_engulfing_pattern = self.detect_bearish_engulfing(df)
        evening_star_pattern = self.detect_evening_star(df, self.price_tolerance)
        
        pattern_types = []
        
        # Apply trend context filter - bearish patterns are more reliable in uptrends
        if not in_uptrend:
            # If not in uptrend, require stronger confirmation 
            
            # For shooting star, check post-pattern confirmation
            if shooting_star_pattern.iloc[idx] and idx < len(df) - 1:
                next_candle = df.iloc[idx + 1]
                # Only include shooting star if next candle confirms with lower close
                if next_candle['close'] < df.iloc[idx]['close']:
                    pattern_types.append("Shooting Star (Confirmed)")
                else:
                    pattern_types.append("Shooting Star (Unconfirmed)")
            elif shooting_star_pattern.iloc[idx]:
                pattern_types.append("Shooting Star (Unconfirmed)")
                
            # For engulfing, check volume confirmation
            if bearish_engulfing_pattern.iloc[idx]:
                current_candle = df.iloc[idx]
                prev_candle = df.iloc[idx - 1] if idx > 0 else None
                
                if prev_candle is not None:
                    # Check if engulfing candle has higher volume than previous
                    current_vol = current_candle.get('tick_volume', current_candle.get('volume', 0))
                    prev_vol = prev_candle.get('tick_volume', prev_candle.get('volume', 0))
                    
                    if current_vol > prev_vol:
                        pattern_types.append("Bearish Engulfing (Volume Confirmed)")
                    else:
                        pattern_types.append("Bearish Engulfing")
                else:
                    pattern_types.append("Bearish Engulfing")
                    
            # Evening star has built-in confirmation in its 3-candle structure
            if evening_star_pattern.iloc[idx]:
                pattern_types.append("Evening Star")
        else:
            # In an uptrend, patterns are more reliable, so include as is
            if shooting_star_pattern.iloc[idx]:
                pattern_types.append("Shooting Star")
            if bearish_engulfing_pattern.iloc[idx]:
                pattern_types.append("Bearish Engulfing")
            if evening_star_pattern.iloc[idx]:
                pattern_types.append("Evening Star")
        
        if pattern_types:
            return ", ".join(pattern_types)
        return None
    
    def _score_signals(self, raw_signals: List[Dict], primary_df: pd.DataFrame, higher_df: pd.DataFrame) -> List[Dict]:
        """Score a list of raw signals using SignalScorer and standardize confidence."""
        scored = []
        for sig in raw_signals:
            # preserve original symbol for post-processing
            symbol = sig.get('symbol')
            scored_sig = self._scorer.score_signal(
                sig,
                primary_df,
                higher_df,
                support_levels=self.support_levels.get(symbol),
                resistance_levels=self.resistance_levels.get(symbol),
                last_consolidation_ranges=self.last_consolidation_ranges,
                atr_value=None if primary_df is None else self._compute_atr(primary_df),
                consolidation_info=self.last_consolidation_ranges.get(symbol),
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
            # *** Add the original symbol back for grouping later ***
            scored_sig['original_symbol'] = symbol  # Add this line
            scored.append(scored_sig)
        return scored

    def _compute_atr(self, df: pd.DataFrame) -> float:
        # Helper to compute ATR for scoring context
        try:
            from src.utils.indicators import calculate_atr
            import numpy as np
            import pandas as pd
            atr_series = calculate_atr(df, self.atr_period)
            if isinstance(atr_series, pd.Series):
                return float(atr_series.iloc[-1])
            elif isinstance(atr_series, np.ndarray):
                return float(atr_series[-1])
            else:
                return float(atr_series)
        except Exception:
            return 0.0

    def _prepare_dataframes(self, data: Dict[str, Any], symbol: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        # Delegate conversion and indexing
        primary = self._to_dataframe(data.get(self.primary_timeframe), symbol, self.primary_timeframe)
        primary = self._ensure_datetime_index(primary, symbol, self.primary_timeframe)
        higher = self._to_dataframe(data.get(self.higher_timeframe), symbol, self.higher_timeframe)
        higher = self._ensure_datetime_index(higher, symbol, self.higher_timeframe)
        return primary, higher

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

    def _log_candle_samples(self, df: pd.DataFrame, symbol: str, count: int = 5) -> None:
        """Log recent candle data for debugging."""
        n = min(count, len(df))
        if n <= 0:
            return
        logger.debug(f"🕯️ {symbol}: Last {n} candles data sample:")
        for i in range(-n, 0):
            c = df.iloc[i]
            logger.debug(f"   {df.index[i]}: O={c['open']:.5f},H={c['high']:.5f},L={c['low']:.5f},C={c['close']:.5f},Vol={c['tick_volume']}")
    
    def _analyze_volume_quality(self, candle: pd.Series, threshold: float) -> float:
        """Wrapper that delegates volume quality analysis to the shared _SignalScorer instance."""
        return self._scorer.analyze_volume_quality(candle, threshold)
    
    def _cluster_trend_lines(self, trend_lines: List[dict]) -> List[dict]:
        """
        Cluster similar trend lines to reduce redundancy.
        """
        if not trend_lines:
            return []
        angle_tolerance = 5.0  # Degrees
        intercept_pct_tolerance = 0.0015  # 0.15% of price
        slope_tolerance = 0.00005

        avg_intercept = np.mean([line['intercept'] for line in trend_lines])
        intercept_tolerance = avg_intercept * intercept_pct_tolerance

        clustered_lines = []
        used_indices = set()

        for i, line1 in enumerate(trend_lines):
            if i in used_indices:
                continue
            cluster = [line1]
            used_indices.add(i)
            for j, line2 in enumerate(trend_lines):
                if j in used_indices or i == j:
                    continue
                angle_diff = abs(line1['angle'] - line2['angle'])
                intercept_diff = abs(line1['intercept'] - line2['intercept'])
                slope_diff = abs(line1['slope'] - line2['slope'])
                if (angle_diff <= angle_tolerance and
                    intercept_diff <= intercept_tolerance and
                    slope_diff <= slope_tolerance):
                    cluster.append(line2)
                    used_indices.add(j)
            # Choose the best line from the cluster
            if len(cluster) > 1:
                best_line = max(cluster, key=lambda x: x['quality_score'])
            else:
                best_line = cluster[0]
            clustered_lines.append(best_line)
        return clustered_lines

    def _calculate_range_extension(self, df, entry_price, direction):
        """
        Calculate a Market Profile-based range extension TP.
        Uses the average range of the last 50 bars as the extension distance.
        """
        lookback = min(50, len(df))
        if lookback == 0:
            return entry_price  # fallback
        ranges = df['high'].iloc[-lookback:] - df['low'].iloc[-lookback:]
        avg_range = ranges.mean()
        if direction == "buy":
            return entry_price + avg_range
        else:
            return entry_price - avg_range

    # --- VECTORIZE CANDLESTICK PATTERNS (industry standard) ---
    @staticmethod
    def detect_hammer(df: pd.DataFrame, price_tolerance: float = 0.002) -> pd.Series:
        """Vectorized detection of Hammer pattern (bullish reversal) for all candles."""
        body = (df['close'] - df['open']).abs()
        total = df['high'] - df['low']
        lower_wick = df[['open', 'close']].min(axis=1) - df['low']
        upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
        return (
            (total > 0) &
            (body / total < 0.3) &
            (lower_wick > 2 * body) &
            (upper_wick < body)
        )

    @staticmethod
    def detect_shooting_star(df: pd.DataFrame, price_tolerance: float = 0.002) -> pd.Series:
        """Vectorized detection of Shooting Star pattern (bearish reversal) for all candles."""
        body = (df['close'] - df['open']).abs()
        total = df['high'] - df['low']
        upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
        lower_wick = df[['open', 'close']].min(axis=1) - df['low']
        return (
            (total > 0) &
            (body / total < 0.3) &
            (upper_wick > 2 * body) &
            (lower_wick < body)
        )

    @staticmethod
    def detect_bullish_engulfing(df: pd.DataFrame) -> pd.Series:
        """Vectorized detection of Bullish Engulfing pattern for all candles."""
        prev_open = df['open'].shift(1)
        prev_close = df['close'].shift(1)
        is_prev_bearish = prev_close < prev_open
        is_curr_bullish = df['close'] > df['open']
        engulfs = (df['open'] < prev_close) & (df['close'] > prev_open)
        return is_prev_bearish & is_curr_bullish & engulfs

    @staticmethod
    def detect_bearish_engulfing(df: pd.DataFrame) -> pd.Series:
        """Vectorized detection of Bearish Engulfing pattern for all candles."""
        prev_open = df['open'].shift(1)
        prev_close = df['close'].shift(1)
        is_prev_bullish = prev_close > prev_open
        is_curr_bearish = df['close'] < df['open']
        engulfs = (df['open'] > prev_close) & (df['close'] < prev_open)
        return is_prev_bullish & is_curr_bearish & engulfs

    @staticmethod
    def detect_inside_bar(df: pd.DataFrame) -> pd.Series:
        """Vectorized detection of Inside Bar pattern for all candles."""
        prev_high = df['high'].shift(1)
        prev_low = df['low'].shift(1)
        return (df['high'] < prev_high) & (df['low'] > prev_low)

    @staticmethod
    def detect_morning_star(df: pd.DataFrame, price_tolerance: float = 0.002) -> pd.Series:
        """Vectorized detection of Morning Star (bullish 3-bar reversal) for all candles."""
        c1 = df.shift(2)
        c2 = df.shift(1)
        c3 = df
        c1_body = (c1['close'] - c1['open']).abs()
        c2_body = (c2['close'] - c2['open']).abs()
        c3_body = (c3['close'] - c3['open']).abs()
        is_first_bearish = c1['close'] < c1['open']
        is_last_bullish = c3['close'] > c3['open']
        is_middle_small = c2_body < 0.3 * c1_body.rolling(15, min_periods=1).mean()
        is_gap_down = c2[['open', 'close']].max(axis=1) <= c1[['open', 'close']].min(axis=1)
        has_minimal_overlap = c2[['open', 'close']].max(axis=1) <= c1[['open', 'close']].min(axis=1) + 0.3 * c1_body
        first_61_8_level = c1['open'] - 0.618 * c1_body
        good_recovery = c3['close'] > first_61_8_level
        return (
            is_first_bearish & is_last_bullish & is_middle_small & (is_gap_down | has_minimal_overlap) & good_recovery
        )

    @staticmethod
    def detect_evening_star(df: pd.DataFrame, price_tolerance: float = 0.002) -> pd.Series:
        """Vectorized detection of Evening Star (bearish 3-bar reversal) for all candles."""
        c1 = df.shift(2)
        c2 = df.shift(1)
        c3 = df
        c1_body = (c1['close'] - c1['open']).abs()
        c2_body = (c2['close'] - c2['open']).abs()
        c3_body = (c3['close'] - c3['open']).abs()
        is_first_bullish = c1['close'] > c1['open']
        is_last_bearish = c3['close'] < c3['open']
        is_middle_small = c2_body < 0.3 * c1_body.rolling(15, min_periods=1).mean()
        is_gap_up = c2[['open', 'close']].min(axis=1) >= c1[['open', 'close']].max(axis=1)
        has_minimal_overlap = c2[['open', 'close']].min(axis=1) >= c1[['open', 'close']].max(axis=1) - 0.3 * c1_body
        first_61_8_level = c1['open'] + 0.618 * c1_body
        good_decline = c3['close'] < first_61_8_level
        return (
            is_first_bullish & is_last_bearish & is_middle_small & (is_gap_up | has_minimal_overlap) & good_decline
        )

    @staticmethod
    def detect_false_breakout(df: pd.DataFrame, direction: str, price_tolerance: float = 0.002) -> pd.Series:
        """Vectorized detection of False Breakout pattern (custom, based on wick and volume)."""
        prev_close = df['close'].shift(1)
        tol_val = df['close'] * price_tolerance
        if direction == 'bullish':
            wick = df['close'] - df['low']
            body = (df['close'] - df['open']).abs()
            wick_ok = wick > 2 * body
            # Volume analysis omitted for vectorized version; can be added if needed
            return (
                (prev_close < df['close'] - tol_val) &
                (wick_ok)
            )
        else:
            wick = df['high'] - df['close']
            body = (df['close'] - df['open']).abs()
            wick_ok = wick > 2 * body
            return (
                (prev_close > df['close'] + tol_val) &
                (wick_ok)
            )

    def _get_dynamic_tolerance(self, df: pd.DataFrame, idx: int, fallback_price: float = None) -> float:
        """
        Compute a dynamic tolerance using ATR * atr_multiplier, fallback to price_tolerance if ATR is unavailable.
        Args:
            df: DataFrame with price data
            idx: Index for the current candle
            fallback_price: Price to use for fallback (e.g., level or close)
        Returns:
            Tolerance value (float)
        """
        try:
            from src.utils.indicators import calculate_atr
            atr_series = calculate_atr(df.iloc[:idx+1], self.atr_period)
            if isinstance(atr_series, pd.Series) and not atr_series.empty:
                atr = float(atr_series.iloc[-1])
                if not pd.isna(atr) and atr > 0:
                    return atr * self.atr_multiplier
        except Exception:
            pass
        # Fallback to price_tolerance * fallback_price
        if fallback_price is not None:
            return fallback_price * self.price_tolerance
        # Fallback to close price if nothing else
        close = df['close'].iloc[idx] if idx < len(df) else df['close'].iloc[-1]
        return close * self.price_tolerance