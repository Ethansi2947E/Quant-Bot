"""
trend_following_strategy.py

Trend Following Strategy (Price Action Only)

A rules-based strategy to trade in the direction of an established trend,
entering on pullbacks at support/resistance and using price/volume action for confirmation.

Features:
- Trend identification using higher highs/lows or lower highs/lows (no indicators).
- Entry on pullbacks to support/resistance zones.
- Confirmation with candlestick patterns and advanced volume-wick analysis.
- Stop-loss based on support/resistance, not ATR.
- Risk management via fixed fractional position sizing.
"""

import pandas as pd
import numpy as np
from loguru import logger
from typing import Dict, List, Any, Optional
import talib # Added TA-Lib import

from src.trading_bot import SignalGenerator
from src.risk_manager import RiskManager
from src.utils.patterns_luxalgo import add_luxalgo_patterns, BULLISH_PATTERNS, BEARISH_PATTERNS, NEUTRAL_PATTERNS, ALL_PATTERNS, filter_patterns_by_bias, get_pattern_type

# Helper function, should be defined outside the class or as a static method properly.
def is_progressive_swing_check(seq: List[float], direction: str = "up", logger_instance=None) -> bool:
    """Analyze last 2-3 swings for trend."""
    if logger_instance:
        logger_instance.debug(f"[is_progressive_swing_check] Analyzing seq: {seq}, direction: {direction}, len: {len(seq)}")
    if len(seq) < 3:
        if logger_instance:
            logger_instance.debug(f"[is_progressive_swing_check] Sequence too short ({len(seq)} < 3), returning False.")
        return False
    
    progressive = False
    if direction == "up":
        progressive = seq[-1] > seq[-2] and seq[-2] > seq[-3]
    else: # direction == "down"
        progressive = seq[-1] < seq[-2] and seq[-2] < seq[-3]
    if logger_instance:
        logger_instance.debug(f"[is_progressive_swing_check] Result: {progressive}")
    return progressive

class TrendFollowingStrategy(SignalGenerator):
    """Trend Following Strategy: Pure price action (no indicators, no EMAs/ADX/ATR)."""

    def __init__(
        self,
        primary_timeframe: str = "M15",
        secondary_timeframe: Optional[str] = "H1",
        risk_per_trade: float = 0.01,
        wick_threshold: float = 0.4,
        volume_confirmation_enabled: bool = False,
        debug_disable_pattern: bool = False,
        debug_disable_volume: bool = True,
        lookback_period: int = 300,
        pivot_window: int = 10,
        max_zones: int = 2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.logger = logger
        self.name = "TrendFollowingStrategy"
        self.description = "Trades price action pullbacks in established trends, using support/resistance, candlestick patterns, and price/volume analysis."
        self.version = "2.0.0"
        self.primary_timeframe = primary_timeframe
        self.secondary_timeframe = secondary_timeframe
        self.risk_per_trade = risk_per_trade
        self.wick_threshold = wick_threshold
        self.volume_confirmation_enabled = not debug_disable_volume and volume_confirmation_enabled
        self.debug_disable_pattern = debug_disable_pattern
        self.debug_disable_volume = debug_disable_volume
        self.lookback_period = lookback_period
        self.pivot_window = pivot_window
        self.max_zones = max_zones
        self.risk_manager = RiskManager.get_instance() if hasattr(RiskManager, "get_instance") else RiskManager()
        self.processed_bars = {}
        self.active_trades = {}
        self.params = kwargs
        
        # Configure pivot window based on timeframe (similar to PriceActionSRStrategy)
        if self.primary_timeframe == "M5":
            self.pivot_window = self.params.get("pivot_window_m5", 30)
        elif self.primary_timeframe == "H1":
            self.pivot_window = self.params.get("pivot_window_h1", 5)
        
        logger.info(f"Initialized {self.name} v{self.version}")
        logger.info(f"  Primary Timeframe: {self.primary_timeframe}, Secondary Timeframe: {self.secondary_timeframe}, Lookback: {self.lookback_period}")
        logger.info(f"  Risk: Risk Per Trade={self.risk_per_trade}, Pivot Window: {self.pivot_window}, Max Zones: {self.max_zones}")

    def get_trend_direction(self, df: pd.DataFrame) -> Optional[str]:
        self.logger.debug(f"[get_trend_direction START] df len: {len(df)}, lookback_period: {self.lookback_period}, pivot_window: {self.pivot_window}")

        lookback = min(self.lookback_period, len(df))
        if lookback < 2 * self.pivot_window + 1:
            self.logger.warning(f"[get_trend_direction] Not enough data for robust swing analysis: need {2 * self.pivot_window + 1}, have {lookback}. Returning None.")
            return None

        df_lookback = df.iloc[-lookback:]
        swing_highs_raw = []
        swing_lows_raw = []
        w = self.pivot_window
        highs_vals = df_lookback['high'].values
        lows_vals = df_lookback['low'].values
        for i in range(w, len(df_lookback) - w):
            is_high = all(highs_vals[i] > highs_vals[j] for j in range(i - w, i)) and all(highs_vals[i] > highs_vals[j] for j in range(i + 1, i + w + 1))
            is_low = all(lows_vals[i] < lows_vals[j] for j in range(i - w, i)) and all(lows_vals[i] < lows_vals[j] for j in range(i + 1, i + w + 1))
            if is_high:
                swing_highs_raw.append((i, highs_vals[i]))
            if is_low:
                swing_lows_raw.append((i, lows_vals[i]))
        self.logger.debug(f"[get_trend_direction] Raw swings found: {len(swing_highs_raw)} highs, {len(swing_lows_raw)} lows")

        atr = self._calculate_atr(df_lookback)
        current_close_for_min_dist = df_lookback['close'].iloc[-1] if not df_lookback.empty else 0
        min_dist = max(atr, 0.002 * current_close_for_min_dist) if current_close_for_min_dist > 0 else atr
        self.logger.debug(f"[get_trend_direction] Significance filter: ATR={atr:.5f}, current_close={current_close_for_min_dist:.5f}, min_dist={min_dist:.5f}")

        def filter_significant(swings, swing_type):
            filtered = []
            for idx, price in swings:
                if not filtered:
                    filtered.append((idx, price))
                else:
                    _, last_price = filtered[-1]
                    if abs(price - last_price) > min_dist:
                        filtered.append((idx, price))
                    else:
                        self.logger.debug(f"[filter_significant] Filtering out {swing_type} swing at index {idx} (price {price:.5f}) due to min_dist (last price {last_price:.5f})")
            return filtered
            
        swing_highs_filtered = filter_significant(swing_highs_raw, "high")
        swing_lows_filtered = filter_significant(swing_lows_raw, "low")
        self.logger.debug(f"[get_trend_direction] Filtered significant swings: {len(swing_highs_filtered)} highs, {len(swing_lows_filtered)} lows")

        if len(swing_highs_filtered) < 2 or len(swing_lows_filtered) < 2:
            self.logger.warning(f"[get_trend_direction] Insufficient significant swings for 2-swing sequence analysis: {len(swing_highs_filtered)} highs, {len(swing_lows_filtered)} lows. Need at least 2 of each. Returning None.")
            return None
        
        # Get the prices of the last two significant highs and lows
        last_high_1 = swing_highs_filtered[-1][1]
        last_high_2 = swing_highs_filtered[-2][1]
        last_low_1 = swing_lows_filtered[-1][1]
        last_low_2 = swing_lows_filtered[-2][1]

        self.logger.debug(f"[get_trend_direction] Last two highs: {last_high_1:.5f} (latest), {last_high_2:.5f}")
        self.logger.debug(f"[get_trend_direction] Last two lows:  {last_low_1:.5f} (latest), {last_low_2:.5f}")

        is_HH = last_high_1 > last_high_2  # Higher High
        is_HL = last_low_1 > last_low_2    # Higher Low
        is_LH = last_high_1 < last_high_2  # Lower High
        is_LL = last_low_1 < last_low_2    # Lower Low
        
        self.logger.debug(f"[get_trend_direction] Trend components: HH={is_HH}, HL={is_HL}, LH={is_LH}, LL={is_LL}")
        
        # Basic Dow Theory trend identification
        basic_uptrend = is_HH and is_HL
        basic_downtrend = is_LH and is_LL
        
        # Additional progressive swing analysis for enhanced confirmation
        # Extract just the prices for progressive analysis
        swing_high_prices = [price for _, price in swing_highs_filtered]
        swing_low_prices = [price for _, price in swing_lows_filtered]
        
        # Use progressive swing check if we have enough data (3+ swings)
        progressive_uptrend_confirmed = False
        progressive_downtrend_confirmed = False
        
        if len(swing_high_prices) >= 3:
            progressive_uptrend_confirmed = is_progressive_swing_check(swing_high_prices, "up", self.logger)
            self.logger.debug(f"[get_trend_direction] Progressive uptrend in highs: {progressive_uptrend_confirmed}")
            
        if len(swing_low_prices) >= 3:
            progressive_uptrend_low_confirmed = is_progressive_swing_check(swing_low_prices, "up", self.logger)
            progressive_downtrend_confirmed = is_progressive_swing_check(swing_low_prices, "down", self.logger)
            self.logger.debug(f"[get_trend_direction] Progressive uptrend in lows: {progressive_uptrend_low_confirmed}")
            self.logger.debug(f"[get_trend_direction] Progressive downtrend in lows: {progressive_downtrend_confirmed}")
            
            # For uptrend, we want progressive highs AND progressive lows
            if progressive_uptrend_confirmed and progressive_uptrend_low_confirmed:
                progressive_uptrend_confirmed = True
            else:
                progressive_uptrend_confirmed = False
        
        # Enhanced trend determination with progressive confirmation
        if basic_uptrend:
            if progressive_uptrend_confirmed:
                self.logger.info(f"[get_trend_direction] STRONG UP trend identified (HH+HL + progressive swings confirmed).")
                return "UP"
            else:
                self.logger.info(f"[get_trend_direction] UP trend identified (HH+HL, progressive confirmation: {progressive_uptrend_confirmed}).")
                return "UP"
        elif basic_downtrend:
            if progressive_downtrend_confirmed:
                self.logger.info(f"[get_trend_direction] STRONG DOWN trend identified (LH+LL + progressive swings confirmed).")
                return "DOWN"
            else:
                self.logger.info(f"[get_trend_direction] DOWN trend identified (LH+LL, progressive confirmation: {progressive_downtrend_confirmed}).")
                return "DOWN"
        else:
            self.logger.info(f"[get_trend_direction] No clear UP or DOWN trend identified based on swing analysis. Returning None.")
            return None

    def _build_signal(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        pattern: str,
        confidence: float,
        size: float,
        position_size: float,
        timeframe: str,
        reason: str,
        signal_timestamp: str,
        take_profit: Optional[float] = None,
        **kwargs
    ) -> dict:
        """Assemble a signal dictionary for output. Trailing stop is managed by PositionManager."""
        signal = {
            "symbol": symbol,
            "direction": direction,
            "entry_price": float(entry_price),
            "stop_loss": float(stop_loss),
            "pattern": pattern,
            "confidence": float(confidence),
            "size": float(size),
            "position_size": float(position_size),
            "timeframe": timeframe,
            "reason": reason,
            "strategy_name": self.name,
            "signal_timestamp": signal_timestamp,
            "take_profit": take_profit if take_profit is not None else 'None',
        }
        signal.update(kwargs)
        self.logger.info(
            f"[Signal] {symbol} {direction.upper()} @ {entry_price:.5f}, SL={stop_loss:.5f}, TP={take_profit if take_profit is not None else 'None'}, Size={size:.3f}, Pattern={pattern}, Conf={confidence:.2f}, Reason={reason}"
        )
        return signal

    def _log_debug_info(
        self,
        symbol: str,
        trend: str,
        entry: float,
        stop: float,
        size: float,
        pattern: str,
        confidence: float,
        reason: str,
        **kwargs
    ) -> None:
        """Log detailed debug information about the signal and context.

        Args:
            symbol (str): Trading symbol
            trend (str): Trend direction
            entry (float): Entry price
            stop (float): Stop-loss price
            size (float): Position size
            pattern (str): Entry pattern
            confidence (float): Signal confidence
            reason (str): Reason for signal
            **kwargs: Additional context
        """
        self.logger.info(
            f"[Debug] {symbol} | Trend: {trend} | Entry: {entry:.5f} | SL: {stop:.5f} | Size: {size:.3f} | Pattern: {pattern} | Conf: {confidence:.2f}"
        )
        self.logger.info(f"[Debug] Reason: {reason}")
        for k, v in kwargs.items():
            self.logger.debug(f"[Debug] {k}: {v}")

    async def initialize(self):
        """Initialize the strategy with any necessary setup."""
        logger.info(f"Initializing {self.__class__.__name__}")
        return True

    @property
    def required_timeframes(self):
        tfs = [self.primary_timeframe]
        if self.secondary_timeframe and self.secondary_timeframe != self.primary_timeframe:
            tfs.append(self.secondary_timeframe)
        return tfs

    @property
    def lookback_periods(self):
        periods = {self.primary_timeframe: self.lookback_period}
        if self.secondary_timeframe and self.secondary_timeframe != self.primary_timeframe:
            periods[self.secondary_timeframe] = self.lookback_period
        return periods

    def is_near_support_resistance(self, df: pd.DataFrame, tolerance: float = 0.005) -> bool:
        """
        Check if price is near any significant support or resistance level.
        This is a general check - use specific methods for directional bias.
        """
        zones = self.get_support_resistance_zones(df)
        current_price = df['close'].iloc[-1]
        
        # Check proximity to any significant level
        all_levels = zones["support"] + zones["resistance"]
        near_any_level = any(abs(current_price - level) / level < tolerance for level in all_levels)
        
        # Determine which type of level we're closest to
        closest_support = min(zones["support"], key=lambda x: abs(current_price - x), default=None) if zones["support"] else None
        closest_resistance = min(zones["resistance"], key=lambda x: abs(current_price - x), default=None) if zones["resistance"] else None
        
        near_support_dist = abs(current_price - closest_support) / closest_support if closest_support else float('inf')
        near_resistance_dist = abs(current_price - closest_resistance) / closest_resistance if closest_resistance else float('inf')
        
        self.logger.debug(f"[S/R] {current_price=}, near_any_level={near_any_level}, "
                         f"closest_support={closest_support}, closest_resistance={closest_resistance}")
        return near_any_level

    def is_near_support(self, df: pd.DataFrame, tolerance: float = 0.005) -> bool:
        """
        Check if price is near support. Only returns True if support is the closest significant level.
        """
        zones = self.get_support_resistance_zones(df)
        current_price = df['close'].iloc[-1]
        
        if not zones["support"]:
            return False
            
        # Find closest support level
        closest_support = min(zones["support"], key=lambda x: abs(current_price - x))
        support_distance = abs(current_price - closest_support) / closest_support
        
        # Only consider near support if within tolerance AND support is below current price
        near_support = (support_distance < tolerance and current_price >= closest_support * 0.995)
        
        # Additional check: ensure no resistance is closer
        if zones["resistance"] and near_support:
            closest_resistance = min(zones["resistance"], key=lambda x: abs(current_price - x))
            resistance_distance = abs(current_price - closest_resistance) / closest_resistance
            
            # If resistance is significantly closer, we're not "near support"
            if resistance_distance < support_distance * 0.7:
                near_support = False
                
        self.logger.debug(f"[Support] {current_price=}, closest_support={closest_support}, "
                         f"distance={support_distance:.4f}, near_support={near_support}")
        return near_support

    def is_near_resistance(self, df: pd.DataFrame, tolerance: float = 0.005) -> bool:
        """
        Check if price is near resistance. Only returns True if resistance is the closest significant level.
        """
        zones = self.get_support_resistance_zones(df)
        current_price = df['close'].iloc[-1]
        
        if not zones["resistance"]:
            return False
            
        # Find closest resistance level
        closest_resistance = min(zones["resistance"], key=lambda x: abs(current_price - x))
        resistance_distance = abs(current_price - closest_resistance) / closest_resistance
        
        # Only consider near resistance if within tolerance AND resistance is above current price
        near_resistance = (resistance_distance < tolerance and current_price <= closest_resistance * 1.005)
        
        # Additional check: ensure no support is closer
        if zones["support"] and near_resistance:
            closest_support = min(zones["support"], key=lambda x: abs(current_price - x))
            support_distance = abs(current_price - closest_support) / closest_support
            
            # If support is significantly closer, we're not "near resistance"
            if support_distance < resistance_distance * 0.7:
                near_resistance = False
                
        self.logger.debug(f"[Resistance] {current_price=}, closest_resistance={closest_resistance}, "
                         f"distance={resistance_distance:.4f}, near_resistance={near_resistance}")
        return near_resistance
    
    def check_price_acceptance_rejection(self, df: pd.DataFrame) -> Dict[str, bool]:
        """
        Enhanced Price Acceptance/Rejection Logic:
        - Breakout: Requires 3-bar close rule AND a high-volume, strong-bodied breakout candle.
        - Rejection: Uses dynamic wick/body threshold (e.g., based on ATR or recent volatility) and close location.
        - Explicitly links TA-Lib patterns (Hammer, Shooting Star) to rejection flags.
        - Modular for reuse in other strategies.
        """
        SR_WINDOW = 20
        MIN_BARS_FOR_LOGIC = 3
        result = {
            "breakout_resistance_confirmed": False,
            "breakout_support_confirmed": False,
            "rejection_at_resistance": False,
            "rejection_at_support": False
        }
        if len(df) < SR_WINDOW + MIN_BARS_FOR_LOGIC - 1:
            self.logger.debug(f"[PriceAcceptance] Not enough data: Need {SR_WINDOW + MIN_BARS_FOR_LOGIC -1}, have {len(df)}")
            return result
        
        # Ensure LuxAlgo patterns are present
        if not all(col in df.columns for col in ['hammer', 'shooting_star']):
            self.logger.warning(f"[PriceAcceptance] LuxAlgo pattern columns ('hammer', 'shooting_star') not found in DataFrame. Adding them now.")
            df_copy = df.copy()
            df_with_patterns = add_luxalgo_patterns(df_copy) 
        else:
            df_with_patterns = df

        highs = np.asarray(df_with_patterns['high'].values, dtype=np.float64)
        lows = np.asarray(df_with_patterns['low'].values, dtype=np.float64)
        closes = np.asarray(df_with_patterns['close'].values, dtype=np.float64)
        opens = np.asarray(df_with_patterns['open'].values, dtype=np.float64)

        # --- S/R level for breakout: use up to T-4 (exclude last 3 bars) ---
        if len(df_with_patterns) > SR_WINDOW + 3:
            highs_sr = highs[:-3]
            lows_sr = lows[:-3]
        else:
            highs_sr = highs
            lows_sr = lows
        resistance_level_T_minus_2 = talib.MAX(highs_sr, timeperiod=SR_WINDOW)[-1]
        support_level_T_minus_2 = talib.MIN(lows_sr, timeperiod=SR_WINDOW)[-1]
        self.logger.debug(f"[PriceAcceptance] S/R for breakout: resistance_level_T_minus_2={resistance_level_T_minus_2:.4f}, support_level_T_minus_2={support_level_T_minus_2:.4f} (computed up to T-4)")

        current_close_T = closes[-1]
        prev_close_T_minus_1 = closes[-2]
        prev_prev_close_T_minus_2 = closes[-3]
        current_open_T = opens[-1]
        # --- Breakout Confirmation: 3-bar close rule + volume/body check ---
        breakout_body = abs(current_close_T - current_open_T)
        breakout_range = abs(df_with_patterns['high'].iloc[-1] - df_with_patterns['low'].iloc[-1])
        atr = talib.ATR(highs, lows, closes, timeperiod=14)[-1] if len(df_with_patterns) >= 14 else breakout_range
        strong_body = breakout_body > 0.5 * atr if atr and not np.isnan(atr) else breakout_body > 0.5 * breakout_range
        high_volume = self.is_valid_volume_spike(df_with_patterns, mode='breakout')
        # Resistance breakout
        result["breakout_resistance_confirmed"] = (
            prev_prev_close_T_minus_2 < resistance_level_T_minus_2 and
            prev_close_T_minus_1 > resistance_level_T_minus_2 and
            current_close_T > resistance_level_T_minus_2 and
            strong_body and high_volume
        )
        # Support breakout
        result["breakout_support_confirmed"] = (
            prev_prev_close_T_minus_2 > support_level_T_minus_2 and
            prev_close_T_minus_1 < support_level_T_minus_2 and
            current_close_T < support_level_T_minus_2 and
            strong_body and high_volume
        )
        # --- Rejection Logic: Dynamic wick/body threshold, close location, TA-Lib pattern ---
        wick_atr = atr if atr and not np.isnan(atr) else breakout_range
        wick_threshold = 0.4 * wick_atr
        # Support rejection
        if self.is_near_support(df_with_patterns):
            lower_wick = min(df_with_patterns['open'].iloc[-1], df_with_patterns['close'].iloc[-1]) - df_with_patterns['low'].iloc[-1]
            body_size = abs(df_with_patterns['close'].iloc[-1] - df_with_patterns['open'].iloc[-1])
            close_location = abs(df_with_patterns['close'].iloc[-1] - df_with_patterns['open'].iloc[-1]) < 0.3 * breakout_range
            hammer_luxalgo = df_with_patterns['hammer'].iloc[-1] if 'hammer' in df_with_patterns.columns else False
            result["rejection_at_support"] = (
                (lower_wick > wick_threshold and close_location) or hammer_luxalgo
            )
        # Resistance rejection
        if self.is_near_resistance(df_with_patterns):
            upper_wick = df_with_patterns['high'].iloc[-1] - max(df_with_patterns['open'].iloc[-1], df_with_patterns['close'].iloc[-1])
            body_size = abs(df_with_patterns['close'].iloc[-1] - df_with_patterns['open'].iloc[-1])
            close_location = abs(df_with_patterns['close'].iloc[-1] - df_with_patterns['open'].iloc[-1]) < 0.3 * breakout_range
            shooting_star_luxalgo = df_with_patterns['shooting_star'].iloc[-1] if 'shooting_star' in df_with_patterns.columns else False
            result["rejection_at_resistance"] = (
                (upper_wick > wick_threshold and close_location) or shooting_star_luxalgo
            )
        self.logger.debug(f"[PriceAcceptance] ResLvl(T-2): {resistance_level_T_minus_2:.4f}, SupLvl(T-2): {support_level_T_minus_2:.4f}")
        self.logger.debug(f"[PriceAcceptance] Closes: T-2={prev_prev_close_T_minus_2:.4f}, T-1={prev_close_T_minus_1:.4f}, T={current_close_T:.4f}")
        self.logger.debug(f"[PriceAcceptance] Results: {result}")
        return result

    def is_valid_volume_spike(self, df: pd.DataFrame, mode: str = 'breakout') -> bool:
        """
        Detects a valid volume spike for the given context.
        Args:
            df (pd.DataFrame): DataFrame with 'volume', 'open', 'close', 'high', 'low'.
            mode (str): 'breakout' (default) for strong body, 'rejection' for large wick.
        Returns:
            bool: True if valid volume spike for the context.
        """
        if 'volume' not in df.columns:
            self.logger.debug(f"[Volume] 'volume' column missing in DataFrame. Mode={mode}")
            return False
        current_volume = df['volume'].iloc[-1]
        vol_mean = df['volume'].rolling(20).mean().iloc[-1]
        if vol_mean == 0 or np.isnan(vol_mean):
            self.logger.debug(f"[Volume] Rolling mean is zero or NaN. Mode={mode}")
            return False
        is_spike = current_volume > 2 * vol_mean
        if not is_spike:
            self.logger.debug(f"[Volume] No spike: current={current_volume}, mean={vol_mean}, Mode={mode}")
            return False
        body = abs(df['close'].iloc[-1] - df['open'].iloc[-1])
        total_range = df['high'].iloc[-1] - df['low'].iloc[-1]
        if total_range == 0:
            self.logger.debug(f"[Volume] Total range is zero, cannot compute wick ratios. Mode={mode}")
            return False
        wick = (df['high'].iloc[-1] - df['low'].iloc[-1]) - body
        wick_body_ratio = wick / body if body > 0 else 0
        upper_wick = df['high'].iloc[-1] - max(df['open'].iloc[-1], df['close'].iloc[-1])
        lower_wick = min(df['open'].iloc[-1], df['close'].iloc[-1]) - df['low'].iloc[-1]
        upper_wick_ratio = upper_wick / total_range
        lower_wick_ratio = lower_wick / total_range
        if mode == 'breakout':
            valid = wick_body_ratio < 0.5
        elif mode == 'rejection':
            valid = wick_body_ratio > 1
        else:
            self.logger.debug(f"[Volume] Unknown mode: {mode}")
            return False
        self.logger.debug(f"[Volume] {current_volume=}, {vol_mean=}, is_spike={is_spike}, wick_body_ratio={wick_body_ratio:.3f}, valid={valid}, mode={mode}")
        return valid

    def _find_pivot_highs(self, df: pd.DataFrame) -> List[float]:
        """
        Find true fractal/pivot highs using proper pivot point logic.
        A pivot high must be higher than N bars on both left and right sides.
        
        Args:
            df (pd.DataFrame): Price data with 'high' column
        Returns:
            List[float]: List of pivot high prices
        """
        w = self.pivot_window
        if len(df) < 2 * w + 1:
            return []
        
        highs = df['high'].values
        pivots = []
        
        # Check each potential pivot point (excluding edges)
        for i in range(w, len(highs) - w):
            current_high = highs[i]
            is_pivot = True
            
            # Check if current high is higher than all bars to the left
            for j in range(i - w, i):
                if current_high <= highs[j]:
                    is_pivot = False
                    break
            
            # Check if current high is higher than all bars to the right
            if is_pivot:
                for j in range(i + 1, i + w + 1):
                    if current_high <= highs[j]:
                        is_pivot = False
                        break
            
            if is_pivot:
                pivots.append(current_high)
                self.logger.debug(f"Found pivot high at index {i}: {current_high:.5f}")
        
        return pivots

    def _find_pivot_lows(self, df: pd.DataFrame) -> List[float]:
        """
        Find true fractal/pivot lows using proper pivot point logic.
        A pivot low must be lower than N bars on both left and right sides.
        
        Args:
            df (pd.DataFrame): Price data with 'low' column
        Returns:
            List[float]: List of pivot low prices
        """
        w = self.pivot_window
        if len(df) < 2 * w + 1:
            return []
        
        lows = df['low'].values
        pivots = []
        
        # Check each potential pivot point (excluding edges)
        for i in range(w, len(lows) - w):
            current_low = lows[i]
            is_pivot = True
            
            # Check if current low is lower than all bars to the left
            for j in range(i - w, i):
                if current_low >= lows[j]:
                    is_pivot = False
                    break
            
            # Check if current low is lower than all bars to the right
            if is_pivot:
                for j in range(i + 1, i + w + 1):
                    if current_low >= lows[j]:
                        is_pivot = False
                        break
            
            if is_pivot:
                pivots.append(current_low)
                self.logger.debug(f"Found pivot low at index {i}: {current_low:.5f}")
        
        return pivots

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate the Average True Range (ATR) over the given period using TA-Lib.
        Args:
            df (pd.DataFrame): Price data with 'high', 'low', 'close'
            period (int): ATR period (default 14)
        Returns:
            float: ATR value
        """
        if len(df) < period + 1:
            return 0.0
        high = np.asarray(df['high'].values, dtype=np.float64)
        low = np.asarray(df['low'].values, dtype=np.float64)
        close = np.asarray(df['close'].values, dtype=np.float64)
        atr = talib.ATR(high, low, close, timeperiod=period)
        return float(atr[-1]) if not np.isnan(atr[-1]) else 0.0

    def _cluster_levels(self, levels: List[float], tol: float = 0.003, df: Optional[pd.DataFrame] = None) -> List[float]:
        """
        Cluster price levels into horizontal zones within ±tol (as a fraction of price).
        If df is provided, tol is dynamically set as max(0.003, 0.5 * ATR / latest_price).
        Args:
            levels (List[float]): List of price levels
            tol (float): Tolerance as a fraction of price (default 0.003 = 0.3%)
            df (pd.DataFrame, optional): Price data for ATR-based tolerance
        Returns:
            List[float]: Clustered zone center prices
        """
        if not levels:
            return []
        
        if df is not None and len(df) > 15:
            latest_price = df['close'].iloc[-1]
            atr = self._calculate_atr(df)
            dynamic_tol = max(0.003, 0.5 * atr / latest_price) if latest_price > 0 else 0.003
            tol = dynamic_tol
            self.logger.debug(f"Dynamic zone tolerance set to {tol:.5f} (ATR={atr:.5f}, price={latest_price:.5f})")
        
            levels = sorted(levels)
            clusters = []
            current = [levels[0]]
        
        for price in levels[1:]:
            if abs(price - current[-1]) <= current[-1] * tol:
                current.append(price)
            else:
                clusters.append(current)
                current = [price]
        
        clusters.append(current)
        # Use mean of each cluster as zone center
        return [float(np.mean(cluster)) for cluster in clusters]

    def get_support_resistance_zones(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """
        Identify key support and resistance zones based on pivot points and ATR-based clustering.
        Improved version from PriceActionSRStrategy with dynamic tolerance and zone strength evaluation.
        Args:
            df (pd.DataFrame): Price data with OHLC
        Returns:
            Dict[str, List[float]]: Dictionary with support and resistance levels
        """
        if len(df) < 2 * self.pivot_window + 1:
            self.logger.debug(f"Not enough data for pivot detection: need {2 * self.pivot_window + 1}, have {len(df)}")
            return {"support": [], "resistance": []}
        
        current_price = df['close'].iloc[-1]
        
        # Find pivot highs and lows using the improved methods
        highs = self._find_pivot_highs(df)
        lows = self._find_pivot_lows(df)
        
        # Cluster levels with ATR-based dynamic tolerance
        res_zones = self._cluster_levels(highs, df=df)
        sup_zones = self._cluster_levels(lows, df=df)
        
        # Vectorized count touches for each zone (strength evaluation)
        def count_touches(prices, zones):
            if not zones or not prices:
                return []
            prices_arr = np.array(prices)
            touches = []
            for zone in zones:
                # Count how many prices are within the zone tolerance
                zone_tolerance = 0.003  # Use base tolerance for touch counting
                in_zone_mask = np.abs(prices_arr - zone) <= zone * zone_tolerance
                touches.append(np.sum(in_zone_mask))
            return touches
        
        res_counts = count_touches(highs, res_zones)
        sup_counts = count_touches(lows, sup_zones)
        
        # Dynamic max_zones based on volatility (like PriceActionSRStrategy)
        close_std = df['close'].rolling(50).std().iloc[-1] if len(df) >= 50 else df['close'].std()
        close_mean = df['close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else df['close'].mean()
        volatility = close_std / close_mean if close_mean > 0 else 0
        dynamic_max_zones = 4 if volatility > 0.01 else self.max_zones  # 1% threshold
        
        self.logger.debug(f"Volatility: {volatility:.4f}, using max_zones={dynamic_max_zones}")
        
        # Select top zones by strength (touch count)
        top_res = [z for _, z in sorted(zip(res_counts, res_zones), reverse=True)[:dynamic_max_zones]] if res_counts else []
        top_sup = [z for _, z in sorted(zip(sup_counts, sup_zones), reverse=True)[:dynamic_max_zones]] if sup_counts else []
        
        # Separate levels by current price (support below, resistance above)
        filtered_support = [level for level in top_sup if level < current_price]
        filtered_resistance = [level for level in top_res if level > current_price]
        
        # Sort by proximity to current price for easier selection in SL/TP logic
        filtered_support = sorted(filtered_support, key=lambda x: abs(current_price - x))
        filtered_resistance = sorted(filtered_resistance, key=lambda x: abs(current_price - x))
        
        self.logger.debug(f"[S/R Zones] Current: {current_price}, Support: {filtered_support}, Resistance: {filtered_resistance}")
        return {"support": filtered_support, "resistance": filtered_resistance}

    def calculate_stop_loss(self, df: pd.DataFrame, direction: str) -> float:
        """
        Calculate stop loss based on S/R zone with buffer.
        
        Args:
            df (pd.DataFrame): Price data
            direction (str): Trade direction ("buy" or "sell")
            
        Returns:
            float: Calculated stop loss price
        """
        zones = self.get_support_resistance_zones(df)
        buffer = 0.001 * df['close'].iloc[-1]  # 0.1% buffer
        
        if direction == "buy":
            # For buy, use nearest support level
            support_levels = sorted(zones["support"], reverse=True)  # Sort descending
            if not support_levels:
                # Fallback to recent low
                support = df['low'].rolling(20).min().iloc[-1]
            else:
                # Find nearest support below current price
                current_price = df['close'].iloc[-1]
                support = support_levels[0]
                for level in support_levels:
                    if level < current_price:
                        support = level
                        break
            
            stop = support - buffer
            self.logger.debug(f"[StopLoss] BUY: support={support}, buffer={buffer}, SL={stop}")
            return stop
        else:
            # For sell, use nearest resistance level
            resistance_levels = sorted(zones["resistance"])  # Sort ascending
            if not resistance_levels:
                # Fallback to recent high
                resistance = df['high'].rolling(20).max().iloc[-1]
            else:
                # Find nearest resistance above current price
                current_price = df['close'].iloc[-1]
                resistance = resistance_levels[0]
                for level in resistance_levels:
                    if level > current_price:
                        resistance = level
                        break
            
            stop = resistance + buffer
            self.logger.debug(f"[StopLoss] SELL: resistance={resistance}, buffer={buffer}, SL={stop}")
            return stop

    def calculate_take_profit(self, df: pd.DataFrame, direction: str, entry_price_override: Optional[float] = None, stop_loss_override: Optional[float] = None) -> Optional[float]:
        """
        Calculate take profit based on opposing S/R zones, with 2R fallback.
        
        Args:
            df (pd.DataFrame): Price data
            direction (str): Trade direction ("buy" or "sell")
            entry_price_override (Optional[float]): If provided, use this entry for 2R calculation.
            stop_loss_override (Optional[float]): If provided, use this SL for 2R calculation.
            
        Returns:
            Optional[float]: Calculated take profit price or None if not possible
        """
        zones = self.get_support_resistance_zones(df)
        current_price = df['close'].iloc[-1] # Still use current_price for S/R zone selection
        
        entry_for_risk_calc = entry_price_override if entry_price_override is not None else current_price
        
        if direction == "buy":
            # For buy, use resistance zones as take profit
            resistance_levels = sorted(zones["resistance"])  # Sort ascending
            
            # Find next resistance above current price
            tp = None
            for level in resistance_levels:
                if level > entry_for_risk_calc: # Check against entry, not current_price
                    tp = level
                    break
                    
            if tp is None:
                # Fallback to 2R
                if stop_loss_override is not None and entry_price_override is not None:
                    risk = abs(entry_price_override - stop_loss_override)
                else: # Fallback to original S/R based SL if overrides not provided
                    risk = abs(entry_for_risk_calc - self.calculate_stop_loss(df, direction))
                
                if risk == 0: # Avoid division by zero or no profit
                    self.logger.debug(f"[TP] BUY: Risk is zero, cannot calculate 2R TP. Entry: {entry_for_risk_calc}, SL: {stop_loss_override if stop_loss_override else self.calculate_stop_loss(df, direction)}")
                    return None 
                tp = entry_for_risk_calc + 2 * risk
                self.logger.debug(f"[TP] BUY: Using fallback 2R TP={tp} (risk={risk}, entry={entry_for_risk_calc})")
            else:
                self.logger.debug(f"[TP] BUY: Using resistance zone TP={tp}")
            
            return tp
        else: # direction == "sell"
            # For sell, use support zones as take profit
            support_levels = sorted(zones["support"], reverse=True)  # Sort descending
            
            # Find next support below current price
            tp = None
            for level in support_levels:
                if level < entry_for_risk_calc: # Check against entry, not current_price
                    tp = level
                    break
                    
            if tp is None:
                # Fallback to 2R
                if stop_loss_override is not None and entry_price_override is not None:
                    risk = abs(entry_price_override - stop_loss_override)
                else: # Fallback to original S/R based SL if overrides not provided
                    risk = abs(entry_for_risk_calc - self.calculate_stop_loss(df, direction))

                if risk == 0: # Avoid division by zero or no profit
                    self.logger.debug(f"[TP] SELL: Risk is zero, cannot calculate 2R TP. Entry: {entry_for_risk_calc}, SL: {stop_loss_override if stop_loss_override else self.calculate_stop_loss(df, direction)}")
                    return None
                tp = entry_for_risk_calc - 2 * risk
                self.logger.debug(f"[TP] SELL: Using fallback 2R TP={tp} (risk={risk}, entry={entry_for_risk_calc})")
            else:
                self.logger.debug(f"[TP] SELL: Using support zone TP={tp}")
            
            return tp

    async def generate_signals(self, market_data: Dict[str, Any], symbol: Optional[str] = None, **kwargs) -> List[Dict]:
        logger.debug(f"[StrategyInit] {self.__class__.__name__}: required_timeframes={self.required_timeframes}, lookback_periods={self.lookback_periods}")
        signals = []
        min_bars = 25 # TA-Lib patterns might need a certain lookback
        for sym, data in market_data.items():
            if isinstance(data, dict):
                df_primary = data.get(self.primary_timeframe)
                df_secondary = data.get(self.secondary_timeframe) if self.secondary_timeframe else None
            elif isinstance(data, pd.DataFrame):
                df_primary = data
                df_secondary = None
            else:
                self.logger.debug(f"[Data] Invalid data structure for {sym}")
                continue
            if not isinstance(df_primary, pd.DataFrame) or len(df_primary) < min_bars:
                self.logger.debug(f"[Data] Not enough data for {sym} (need {min_bars}, got {len(df_primary) if isinstance(df_primary, pd.DataFrame) else 0})")
                continue
            df_primary = add_luxalgo_patterns(df_primary.copy())
            ht_trend_ok = True
            trend_ht = None
            if self.secondary_timeframe and self.secondary_timeframe != self.primary_timeframe:
                if df_secondary is None or len(df_secondary) < self.lookback_period:
                    self.logger.warning(f"Insufficient data for {sym} secondary timeframe {self.secondary_timeframe}: Need {self.lookback_period} bars, got {len(df_secondary) if df_secondary is not None else 0}")
                    ht_trend_ok = False
                else:
                    trend_ht = self.get_trend_direction(df_secondary)
            trend = self.get_trend_direction(df_primary)
            self.logger.debug(f"[Trend] {sym} idx={len(df_primary)-1} trend={trend} HTF trend={trend_ht}")
            if self.secondary_timeframe and self.secondary_timeframe != self.primary_timeframe:
                if not ht_trend_ok or trend_ht not in ("UP", "DOWN") or trend != trend_ht:
                    self.logger.info(f"[Skip] {sym} idx={len(df_primary)-1} HTF trend filter: primary={trend}, secondary={trend_ht}")
                    continue
            if trend not in ("UP", "DOWN"):
                self.logger.info(f"[Skip] {sym} idx={len(df_primary)-1} No valid trend detected.")
                continue
            price_action = self.check_price_acceptance_rejection(df_primary)
            is_at_pullback_sr = self.is_near_support_resistance(df_primary) 
            is_confirmed_breakout = price_action["breakout_resistance_confirmed"] or price_action["breakout_support_confirmed"]
            if not (is_at_pullback_sr or is_confirmed_breakout):
                self.logger.info(f"[Skip] {sym} idx={len(df_primary)-1} Price not near S/R for pullback AND no confirmed breakout. PullbackSR: {is_at_pullback_sr}, ConfBreakout: {is_confirmed_breakout}")
                continue
            detected_patterns_list = []
            relevant_patterns_for_trend = []
            if trend == "UP":
                relevant_patterns_for_trend = BULLISH_PATTERNS + NEUTRAL_PATTERNS
            elif trend == "DOWN":
                relevant_patterns_for_trend = BEARISH_PATTERNS + NEUTRAL_PATTERNS
            if is_at_pullback_sr:
                if trend == "UP" and self.is_near_support(df_primary):
                    for p_col in relevant_patterns_for_trend:
                        if p_col in df_primary.columns and df_primary[p_col].iloc[-1]:
                            detected_patterns_list.append(p_col)
                    if price_action["rejection_at_support"]:
                        detected_patterns_list.append("Support Rejection")
                elif trend == "DOWN" and self.is_near_resistance(df_primary):
                    for p_col in relevant_patterns_for_trend:
                        if p_col in df_primary.columns and df_primary[p_col].iloc[-1]:
                            detected_patterns_list.append(p_col)
                    if price_action["rejection_at_resistance"]:
                        detected_patterns_list.append("Resistance Rejection")
            if trend == "UP" and price_action["breakout_resistance_confirmed"]:
                detected_patterns_list.append("Resistance Breakout Confirmed")
                for p_col in ['bullish_engulfing', 'white_marubozu']:
                    if p_col in df_primary.columns and df_primary[p_col].iloc[-1]:
                        detected_patterns_list.append(p_col + "_on_breakout")
            elif trend == "DOWN" and price_action["breakout_support_confirmed"]:
                detected_patterns_list.append("Support Breakdown Confirmed")
                for p_col in ['bearish_engulfing', 'black_marubozu']:
                    if p_col in df_primary.columns and df_primary[p_col].iloc[-1]:
                        detected_patterns_list.append(p_col + "_on_breakdown")
            formatted_patterns = []
            for p_name in detected_patterns_list:
                if p_name.endswith(("_on_breakout", "_on_breakdown")):
                    base_name, suffix = p_name.rsplit('_', 2)
                    formatted_patterns.append(f"{base_name.replace('_', ' ').title()} on {suffix.capitalize()} (LuxAlgo)")
                elif p_name in ALL_PATTERNS:
                    formatted_patterns.append(f"{p_name.replace('_', ' ').title()} (LuxAlgo)")
                else:
                    formatted_patterns.append(p_name) 
            final_pattern_str_elements = list(dict.fromkeys(formatted_patterns))
            self.logger.debug(f"[Pattern] {sym} idx={len(df_primary)-1} detected_patterns={final_pattern_str_elements}")
            pattern_ok = bool(final_pattern_str_elements) or self.debug_disable_pattern
            if not pattern_ok:
                self.logger.info(f"[Skip] {sym} idx={len(df_primary)-1} No valid pattern or condition detected.")
                continue
            vol_ok = self.debug_disable_volume or self.is_valid_volume_spike(df_primary, mode='breakout')
            self.logger.debug(f"[VolumeCheck] {sym} idx={len(df_primary)-1} vol_ok={vol_ok}")
            if not vol_ok:
                self.logger.info(f"[Skip] {sym} idx={len(df_primary)-1} Volume-wick confirmation failed.")
                continue
            pattern_display_string = ", ".join(final_pattern_str_elements)
            direction = "buy" if trend == "UP" else "sell"
            # --- Entry and Stop Loss Logic ---
            # Entry price logic remains as breakout stop order (high/low of signal bar)
            if direction == "buy":
                entry_price = df_primary['high'].iloc[-1]
            else:
                entry_price = df_primary['low'].iloc[-1]
            # S/R-based stop loss (primary)
            stop_loss_sr = self.calculate_stop_loss(df_primary, direction)
            # Signal bar's extremity (secondary/tighter SL, for logging)
            if direction == "buy":
                stop_loss_signal = df_primary['low'].iloc[-1]
            else:
                stop_loss_signal = df_primary['high'].iloc[-1]
            # Choose SL based on trade reason
            if (direction == "buy" and ("Support Rejection" in detected_patterns_list or price_action["rejection_at_support"])):
                stop_loss = stop_loss_sr
                sl_reason = "S/R-based SL below support (rejection)"
            elif (direction == "sell" and ("Resistance Rejection" in detected_patterns_list or price_action["rejection_at_resistance"])):
                stop_loss = stop_loss_sr
                sl_reason = "S/R-based SL above resistance (rejection)"
            elif (direction == "buy" and price_action["breakout_resistance_confirmed"]):
                stop_loss = stop_loss_sr
                sl_reason = "S/R-based SL below broken resistance (breakout)"
            elif (direction == "sell" and price_action["breakout_support_confirmed"]):
                stop_loss = stop_loss_sr
                sl_reason = "S/R-based SL above broken support (breakout)"
            else:
                stop_loss = stop_loss_signal
                sl_reason = "Signal bar extremity SL (fallback)"
            self.logger.info(f"[SL] {sym} idx={len(df_primary)-1} SL={stop_loss} Reason: {sl_reason} (SR={stop_loss_sr}, Signal={stop_loss_signal})")
            take_profit = self.calculate_take_profit(df_primary, direction, entry_price_override=entry_price, stop_loss_override=stop_loss)
            if not take_profit or take_profit == 0:
                self.logger.info(f"[Skip] {sym} idx={len(df_primary)-1} No valid take-profit found (TP: {take_profit}), skipping signal.")
                continue
            if direction == "buy" and entry_price <= stop_loss:
                self.logger.info(f"[Skip] {sym} idx={len(df_primary)-1} Invalid SL for BUY signal (Entry: {entry_price}, SL: {stop_loss}), skipping.")
                continue
            if direction == "sell" and entry_price >= stop_loss:
                self.logger.info(f"[Skip] {sym} idx={len(df_primary)-1} Invalid SL for SELL signal (Entry: {entry_price}, SL: {stop_loss}), skipping.")
                continue
            risk_manager = self.risk_manager if hasattr(self, 'risk_manager') else RiskManager.get_instance()
            account_balance = risk_manager.get_account_balance()
            risk_per_trade = getattr(self, 'risk_per_trade', 0.01)
            try:
                position_size = risk_manager.calculate_position_size(
                    account_balance=account_balance,
                    risk_per_trade=risk_per_trade * 100.0,
                    entry_price=entry_price,
                    stop_loss_price=stop_loss,
                    symbol=sym
                )
            except Exception as e:
                self.logger.warning(f"[RiskManager] Position sizing failed for {sym}: {e}")
                position_size = 0.0
            size = position_size
            confidence = 0.7
            reason = f"Trend: {trend}, Patterns: {pattern_display_string}, Vol Conf: {vol_ok}"
            if price_action["breakout_resistance_confirmed"]:
                reason += ", BreakoutResistanceConfirmed"
            if price_action["breakout_support_confirmed"]:
                reason += ", BreakoutSupportConfirmed"
            if price_action["rejection_at_resistance"]:
                reason += ", RejectionAtResistance"
            if price_action["rejection_at_support"]:
                reason += ", RejectionAtSupport"
            if is_at_pullback_sr and not (price_action["breakout_resistance_confirmed"] or price_action["breakout_support_confirmed"]):
                 reason += ", PullbackToSR"
            signal_timestamp = str(df_primary.index[-1])
            self.logger.info(f"[SignalCandidate] {sym} idx={len(df_primary)-1} direction={direction} entry={entry_price} stop={stop_loss} tp={take_profit} pattern={pattern_display_string} vol_ok={vol_ok}")
            signal = self._build_signal(
                symbol=sym,
                direction=direction,
                entry_price=entry_price,
                stop_loss=stop_loss,
                pattern=pattern_display_string,
                confidence=confidence,
                size=size,
                position_size=size,
                timeframe=self.primary_timeframe,
                reason=reason,
                signal_timestamp=signal_timestamp,
                take_profit=take_profit
            )
            signals.append(signal)
        return signals