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

class TrendFollowingStrategy(SignalGenerator):
    """Trend Following Strategy: Pure price action (no indicators, no EMAs/ADX/ATR)."""

    @staticmethod
    def _detect_inside_bar_vectorized(df: pd.DataFrame) -> pd.Series:
        if len(df) < 2:
            return pd.Series([False] * len(df), index=df.index)
        # Current high < previous high AND current low > previous low
        inside_bar = (df['high'] < df['high'].shift(1)) & (df['low'] > df['low'].shift(1))
        return inside_bar.fillna(False)

    def __init__(
        self,
        primary_timeframe: str = "M15",
        secondary_timeframe: Optional[str] = "H1",
        risk_per_trade: float = 0.01,
        wick_threshold: float = 0.4,
        volume_confirmation_enabled: bool = True,
        debug_disable_pattern: bool = True,
        debug_disable_volume: bool = False,
        lookback_period: int = 300,
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
        self.risk_manager = RiskManager.get_instance() if hasattr(RiskManager, "get_instance") else RiskManager()
        self.processed_bars = {}
        self.active_trades = {}
        self.params = kwargs
        logger.info(f"Initialized {self.name} v{self.version}")
        logger.info(f"  Primary Timeframe: {self.primary_timeframe}, Secondary Timeframe: {self.secondary_timeframe}, Lookback: {self.lookback_period}")
        logger.info(f"  Risk: Risk Per Trade={self.risk_per_trade}")

    def _get_trend_direction(self, df: pd.DataFrame, window: int = 20) -> Optional[str]:
        if len(df) < window:
            self.logger.debug(f"[_get_trend_direction] Not enough data for window {window}, have {len(df)}")
            return None

        # Divide the window into two halves
        prior_half_df = df.iloc[-window : -window//2]
        recent_half_df = df.iloc[-window//2:]

        if prior_half_df.empty or recent_half_df.empty:
            self.logger.debug(f"[_get_trend_direction] One of the halves is empty for window {window}.")
            return None

        prior_high = prior_half_df['high'].max()
        prior_low = prior_half_df['low'].min()
        recent_high = recent_half_df['high'].max()
        recent_low = recent_half_df['low'].min()
        latest_close = df['close'].iloc[-1]

        is_uptrend_progression = recent_high > prior_high and recent_low > prior_low
        is_downtrend_progression = recent_high < prior_high and recent_low < prior_low
        
        # Check current position within the recent segment's range
        recent_segment_midpoint = (recent_high + recent_low) / 2
        is_current_strength = latest_close > recent_segment_midpoint
        is_current_weakness = latest_close < recent_segment_midpoint

        if is_uptrend_progression and is_current_strength:
            self.logger.debug(f"[_get_trend_direction] UP trend: recent_high={recent_high:.4f} > prior_high={prior_high:.4f}, recent_low={recent_low:.4f} > prior_low={prior_low:.4f}, latest_close={latest_close:.4f} > midpoint={recent_segment_midpoint:.4f}")
            return "UP"
        elif is_downtrend_progression and is_current_weakness:
            self.logger.debug(f"[_get_trend_direction] DOWN trend: recent_high={recent_high:.4f} < prior_high={prior_high:.4f}, recent_low={recent_low:.4f} < prior_low={prior_low:.4f}, latest_close={latest_close:.4f} < midpoint={recent_segment_midpoint:.4f}")
            return "DOWN"
        
        self.logger.debug(f"[_get_trend_direction] No clear trend: RH={recent_high:.4f}, PH={prior_high:.4f}, RL={recent_low:.4f}, PL={prior_low:.4f}, Close={latest_close:.4f}, Mid={recent_segment_midpoint:.4f}")
        return None

    def _get_trend_direction_long(self, df: pd.DataFrame, window: int = 300) -> Optional[str]:
        if df is None or len(df) < window:
            self.logger.debug(f"[_get_trend_direction_long] Not enough data for window {window}, have {len(df) if df is not None else 0}")
            return None
            
        highs = df['high'].iloc[-window:]
        lows = df['low'].iloc[-window:]
        # closes = df['close'].iloc[-window:] # No longer needed for recent_strong_move

        # Rely on comparison of highs/lows at the start and end of the window
        end_high = highs.iloc[-1]
        start_high = highs.iloc[0]
        end_low = lows.iloc[-1]
        start_low = lows.iloc[0]

        is_clear_uptrend = end_high > start_high and end_low > start_low
        is_clear_downtrend = end_high < start_high and end_low < start_low
        
        if is_clear_uptrend:
            self.logger.debug(f"[_get_trend_direction_long] UP trend: end_high={end_high:.4f} > start_high={start_high:.4f}, end_low={end_low:.4f} > start_low={start_low:.4f}")
            return "UP"
        elif is_clear_downtrend:
            self.logger.debug(f"[_get_trend_direction_long] DOWN trend: end_high={end_high:.4f} < start_high={start_high:.4f}, end_low={end_low:.4f} < start_low={start_low:.4f}")
            return "DOWN"
            
        self.logger.debug(f"[_get_trend_direction_long] No clear long trend: EH={end_high:.4f}, SH={start_high:.4f}, EL={end_low:.4f}, SL={start_low:.4f}")
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
        highs = np.asarray(df['high'].values, dtype=np.float64)
        lows = np.asarray(df['low'].values, dtype=np.float64)
        closes = np.asarray(df['close'].values, dtype=np.float64)
        opens = np.asarray(df['open'].values, dtype=np.float64)
        resistance_level_T_minus_2 = talib.MAX(highs, timeperiod=SR_WINDOW)[-3]
        support_level_T_minus_2 = talib.MIN(lows, timeperiod=SR_WINDOW)[-3]
        current_close_T = closes[-1]
        prev_close_T_minus_1 = closes[-2]
        prev_prev_close_T_minus_2 = closes[-3]
        current_open_T = opens[-1]
        # --- Breakout Confirmation: 3-bar close rule + volume/body check ---
        breakout_body = abs(current_close_T - current_open_T)
        breakout_range = abs(df['high'].iloc[-1] - df['low'].iloc[-1])
        # Use ATR for dynamic body threshold
        atr = talib.ATR(highs, lows, closes, timeperiod=14)[-1] if len(df) >= 14 else breakout_range
        strong_body = breakout_body > 0.5 * atr if atr and not np.isnan(atr) else breakout_body > 0.5 * breakout_range
        high_volume = self.is_valid_volume_spike(df)
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
        # Use ATR for dynamic wick threshold
        wick_atr = atr if atr and not np.isnan(atr) else breakout_range
        wick_threshold = 0.4 * wick_atr
        # Support rejection
        if self.is_near_support(df):
            lower_wick = min(df['open'].iloc[-1], df['close'].iloc[-1]) - df['low'].iloc[-1]
            body_size = abs(df['close'].iloc[-1] - df['open'].iloc[-1])
            close_location = abs(df['close'].iloc[-1] - df['open'].iloc[-1]) < 0.3 * breakout_range
            # TA-Lib Hammer pattern
            open_prices = np.array(df['open'].values, dtype=np.float64)
            high_prices = np.array(df['high'].values, dtype=np.float64)
            low_prices = np.array(df['low'].values, dtype=np.float64)
            close_prices = np.array(df['close'].values, dtype=np.float64)
            hammer_talib = talib.CDLHAMMER(open_prices, high_prices, low_prices, close_prices)[-1] > 0
            # Rejection if wick is large, close is near open, and/or hammer pattern
            result["rejection_at_support"] = (
                (lower_wick > wick_threshold and close_location) or hammer_talib
            )
        # Resistance rejection
        if self.is_near_resistance(df):
            upper_wick = df['high'].iloc[-1] - max(df['open'].iloc[-1], df['close'].iloc[-1])
            body_size = abs(df['close'].iloc[-1] - df['open'].iloc[-1])
            close_location = abs(df['close'].iloc[-1] - df['open'].iloc[-1]) < 0.3 * breakout_range
            # TA-Lib Shooting Star pattern
            open_prices = np.array(df['open'].values, dtype=np.float64)
            high_prices = np.array(df['high'].values, dtype=np.float64)
            low_prices = np.array(df['low'].values, dtype=np.float64)
            close_prices = np.array(df['close'].values, dtype=np.float64)
            shooting_star_talib = talib.CDLSHOOTINGSTAR(open_prices, high_prices, low_prices, close_prices)[-1] < 0
            # Rejection if wick is large, close is near open, and/or shooting star pattern
            result["rejection_at_resistance"] = (
                (upper_wick > wick_threshold and close_location) or shooting_star_talib
            )
        self.logger.debug(f"[PriceAcceptance] ResLvl(T-2): {resistance_level_T_minus_2:.4f}, SupLvl(T-2): {support_level_T_minus_2:.4f}")
        self.logger.debug(f"[PriceAcceptance] Closes: T-2={prev_prev_close_T_minus_2:.4f}, T-1={prev_close_T_minus_1:.4f}, T={current_close_T:.4f}")
        self.logger.debug(f"[PriceAcceptance] Results: {result}")
        return result

    def is_valid_volume_spike(self, df: pd.DataFrame) -> bool:
        if 'volume' not in df.columns:
            self.logger.debug("[Volume] 'volume' column missing in DataFrame.")
            return False
        current_volume = df['volume'].iloc[-1]
        vol_mean = df['volume'].rolling(20).mean().iloc[-1]
        if vol_mean == 0 or np.isnan(vol_mean):
            self.logger.debug("[Volume] Rolling mean is zero or NaN.")
            return False
        # --- Simple, robust volume spike definition ---
        is_spike = current_volume > 2 * vol_mean
        if not is_spike:
            self.logger.debug(f"[Volume] No spike: current={current_volume}, mean={vol_mean}")
            return False
        # --- Candle shape analysis after spike confirmed ---
        body = abs(df['close'].iloc[-1] - df['open'].iloc[-1])
        total_range = df['high'].iloc[-1] - df['low'].iloc[-1]
        if total_range == 0:
            self.logger.debug(f"[Volume] Total range is zero, cannot compute wick ratios.")
            return False
        wick = (df['high'].iloc[-1] - df['low'].iloc[-1]) - body
        wick_body_ratio = wick / body if body > 0 else 0
        upper_wick = df['high'].iloc[-1] - max(df['open'].iloc[-1], df['close'].iloc[-1])
        lower_wick = min(df['open'].iloc[-1], df['close'].iloc[-1]) - df['low'].iloc[-1]
        upper_wick_ratio = upper_wick / total_range
        lower_wick_ratio = lower_wick / total_range
        # --- Simple pattern: prefer small wick/body ratio for conviction ---
        if wick_body_ratio < 0.5:
            valid = True
        else:
            valid = False
        self.logger.debug(f"[Volume] {current_volume=}, {vol_mean=}, is_spike={is_spike}, wick_body_ratio={wick_body_ratio:.3f}, valid={valid}")
        # --- VSA-style enhancement (future): ---
        # High volume + small range + close in middle: possible absorption/indecision
        # High volume + small range + close near low (in uptrend): selling pressure
        # Low volume + large range (breakout): weak breakout, likely to fail
        # These can be added as advanced filters if needed.
        return valid

    def get_support_resistance_zones(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """
        Identify key support and resistance zones based on recent price action.
        Uses TA-Lib for rolling max/min.
        Args:
            df (pd.DataFrame): Price data with OHLC
        Returns:
            Dict[str, List[float]]: Dictionary with support and resistance levels
        """
        if len(df) < 20:
            return {"support": [], "resistance": []}
        current_price = df['close'].iloc[-1]
        lows = np.asarray(df['low'].values, dtype=np.float64)
        highs = np.asarray(df['high'].values, dtype=np.float64)
        # Use TA-Lib MIN/MAX for swing detection
        swing_lows = []
        swing_highs = []
        w = 10
        min_lows = talib.MIN(lows, timeperiod=2 * w + 1)
        max_highs = talib.MAX(highs, timeperiod=2 * w + 1)
        for i in range(w, len(df) - w):
            if lows[i] == min_lows[i] and not np.isnan(min_lows[i]):
                swing_lows.append(lows[i])
            if highs[i] == max_highs[i] and not np.isnan(max_highs[i]):
                swing_highs.append(highs[i])
        # Cluster similar levels (within 0.3% of each other)
        def cluster_levels(levels, tolerance=0.003):
            if not levels:
                return []
            levels = sorted(levels)
            clusters = []
            current_cluster = [levels[0]]
            for price in levels[1:]:
                if abs(price - current_cluster[-1]) / current_cluster[-1] <= tolerance:
                    current_cluster.append(price)
                else:
                    clusters.append(current_cluster)
                    current_cluster = [price]
            clusters.append(current_cluster)
            return [sum(cluster) / len(cluster) for cluster in clusters]
        support_zones = cluster_levels(swing_lows)
        resistance_zones = cluster_levels(swing_highs)
        # Separate levels by current price (support below, resistance above)
        filtered_support = [level for level in support_zones if level < current_price]
        filtered_resistance = [level for level in resistance_zones if level > current_price]
        # Use recent min/max as backup if no zones found
        if not filtered_support:
            recent_low = talib.MIN(lows, timeperiod=20)[-1]
            if recent_low < current_price:
                filtered_support = [recent_low]
        if not filtered_resistance:
            recent_high = talib.MAX(highs, timeperiod=20)[-1]
            if recent_high > current_price:
                filtered_resistance = [recent_high]
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

            open_prices = np.array(df_primary['open'].values, dtype=np.float64)
            high_prices = np.array(df_primary['high'].values, dtype=np.float64)
            low_prices = np.array(df_primary['low'].values, dtype=np.float64)
            close_prices = np.array(df_primary['close'].values, dtype=np.float64)

            # Higher timeframe trend filter
            ht_trend_ok = True
            trend_ht = None # Initialize trend_ht
            if self.secondary_timeframe and self.secondary_timeframe != self.primary_timeframe:
                if df_secondary is None or len(df_secondary) < self.lookback_period:
                    self.logger.warning(f"Insufficient data for {sym} secondary timeframe {self.secondary_timeframe}: Need {self.lookback_period} bars, got {len(df_secondary) if df_secondary is not None else 0}")
                    ht_trend_ok = False
                else:
                    trend_ht = self._get_trend_direction_long(df_secondary, window=self.lookback_period)
            
            # Precompute vectorized patterns using TA-Lib
            # TA-Lib functions return integer arrays (-100 for bearish, 100 for bullish, 0 for none)
            hammer_talib = talib.CDLHAMMER(open_prices, high_prices, low_prices, close_prices)
            shooting_star_talib = talib.CDLSHOOTINGSTAR(open_prices, high_prices, low_prices, close_prices)
            # TA-Lib CDLENGULFING covers both Bullish and Bearish
            engulfing_talib = talib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices)
            morning_star_talib = talib.CDLMORNINGSTAR(open_prices, high_prices, low_prices, close_prices) # Default penetration = 0.3
            evening_star_talib = talib.CDLEVENINGSTAR(open_prices, high_prices, low_prices, close_prices) # Default penetration = 0.3

            # Convert TA-Lib output to boolean Series, aligned with df_primary.index
            hammer = pd.Series(hammer_talib > 0, index=df_primary.index)
            shooting_star = pd.Series(shooting_star_talib < 0, index=df_primary.index) # Shooting star is bearish
            bullish_engulfing = pd.Series(engulfing_talib > 0, index=df_primary.index)
            bearish_engulfing = pd.Series(engulfing_talib < 0, index=df_primary.index)
            morning_star = pd.Series(morning_star_talib > 0, index=df_primary.index)
            evening_star = pd.Series(evening_star_talib < 0, index=df_primary.index)
            inside_bar = TrendFollowingStrategy._detect_inside_bar_vectorized(df_primary)

            idx = len(df_primary) - 1
            if idx < 0: continue # Should not happen if min_bars check passed

            trend = self._get_trend_direction(df_primary, window=20)
            self.logger.debug(f"[Trend] {sym} idx={idx} trend={trend} HTF trend={trend_ht}")
            
            if self.secondary_timeframe and self.secondary_timeframe != self.primary_timeframe:
                if not ht_trend_ok or trend_ht not in ("UP", "DOWN") or trend != trend_ht:
                    self.logger.info(f"[Skip] {sym} idx={idx} HTF trend filter: primary={trend}, secondary={trend_ht}")
                    continue
            
            if trend not in ("UP", "DOWN"):
                self.logger.info(f"[Skip] {sym} idx={idx} No valid trend detected.")
                continue
            
            price_action = self.check_price_acceptance_rejection(df_primary)
            
            # Check if price is either near S/R for pullback/rejection, OR a confirmed breakout has occurred.
            is_at_pullback_sr = self.is_near_support_resistance(df_primary) 
            is_confirmed_breakout = price_action["breakout_resistance_confirmed"] or price_action["breakout_support_confirmed"]

            if not (is_at_pullback_sr or is_confirmed_breakout):
                self.logger.info(f"[Skip] {sym} idx={idx} Price not near S/R for pullback AND no confirmed breakout. PullbackSR: {is_at_pullback_sr}, ConfBreakout: {is_confirmed_breakout}")
                continue
            
            detected_patterns = []
            if trend == "UP":
                if is_at_pullback_sr and self.is_near_support(df_primary): # Pullback to support
                    if hammer.iloc[idx]: detected_patterns.append("Hammer (TA-Lib)")
                    if bullish_engulfing.iloc[idx]: detected_patterns.append("Bullish Engulfing (TA-Lib)")
                    if morning_star.iloc[idx]: detected_patterns.append("Morning Star (TA-Lib)")
                    if inside_bar.iloc[idx]: detected_patterns.append("Inside Bar (cp)")
                    if price_action["rejection_at_support"]: detected_patterns.append("Support Rejection (Bullish)")
                
                if price_action["breakout_resistance_confirmed"]: # Confirmed breakout above resistance
                    detected_patterns.append("Resistance Breakout Confirmed")
                    # Potentially add breakout-specific patterns here if needed, e.g. strong close on breakout bar T
                    if bullish_engulfing.iloc[idx]: detected_patterns.append("Bullish Engulfing on Breakout (TA-Lib)")


            elif trend == "DOWN":
                if is_at_pullback_sr and self.is_near_resistance(df_primary): # Pullback to resistance
                    if shooting_star.iloc[idx]: detected_patterns.append("Shooting Star (TA-Lib)")
                    if bearish_engulfing.iloc[idx]: detected_patterns.append("Bearish Engulfing (TA-Lib)")
                    if evening_star.iloc[idx]: detected_patterns.append("Evening Star (TA-Lib)")
                    if inside_bar.iloc[idx]: detected_patterns.append("Inside Bar (cp)")
                    if price_action["rejection_at_resistance"]: detected_patterns.append("Resistance Rejection (Bearish)")

                if price_action["breakout_support_confirmed"]: # Confirmed breakdown below support
                    detected_patterns.append("Support Breakdown Confirmed")
                    if bearish_engulfing.iloc[idx]: detected_patterns.append("Bearish Engulfing on Breakdown (TA-Lib)")
                    
            self.logger.debug(f"[Pattern] {sym} idx={idx} detected_patterns={detected_patterns}")
            pattern_ok = bool(detected_patterns) or self.debug_disable_pattern
            if not pattern_ok:
                self.logger.info(f"[Skip] {sym} idx={idx} No valid pattern at S/R zone.")
                continue
            
            vol_ok = self.debug_disable_volume or self.is_valid_volume_spike(df_primary)
            self.logger.debug(f"[VolumeCheck] {sym} idx={idx} vol_ok={vol_ok}")
            if not vol_ok:
                self.logger.info(f"[Skip] {sym} idx={idx} Volume-wick confirmation failed.")
                continue
                
            pattern = ", ".join(detected_patterns)
            # Get signal candle data
            signal_candle_open = df_primary['open'].iloc[idx]
            signal_candle_high = df_primary['high'].iloc[idx]
            signal_candle_low = df_primary['low'].iloc[idx]
            signal_candle_close = df_primary['close'].iloc[idx] # Keep for reference, entry is candle H/L

            direction = "buy" if trend == "UP" else "sell"

            if direction == "buy":
                entry_price = signal_candle_high
                stop_loss = signal_candle_low
            else: # direction == "sell"
                entry_price = signal_candle_low
                stop_loss = signal_candle_high
            
            # Original S/R based SL is now only used by TP logic if candle-based SL is not suitable for TP calc, or for S/R TP targets.
            # For consistency, TP calculation will now use the candle-based entry and SL for its 2R fallback.
            take_profit = self.calculate_take_profit(df_primary, direction, entry_price_override=entry_price, stop_loss_override=stop_loss)

            if not take_profit or take_profit == 0: # Ensure take_profit is valid
                self.logger.info(f"[Skip] {sym} idx={idx} No valid take-profit found (TP: {take_profit}), skipping signal.")
                continue
            
            # Ensure stop_loss leads to a positive risk amount
            if direction == "buy" and entry_price <= stop_loss:
                self.logger.info(f"[Skip] {sym} idx={idx} Invalid SL for BUY signal (Entry: {entry_price}, SL: {stop_loss}), skipping.")
                continue
            if direction == "sell" and entry_price >= stop_loss:
                self.logger.info(f"[Skip] {sym} idx={idx} Invalid SL for SELL signal (Entry: {entry_price}, SL: {stop_loss}), skipping.")
                continue

            # --- Position Sizing: Use RiskManager for modular, robust sizing ---
            risk_manager = self.risk_manager if hasattr(self, 'risk_manager') else RiskManager.get_instance()
            account_balance = risk_manager.get_account_balance()
            risk_per_trade = getattr(self, 'risk_per_trade', 0.01)
            try:
                position_size = risk_manager.calculate_position_size(
                    account_balance=account_balance,
                    risk_per_trade=risk_per_trade * 100.0,  # RiskManager expects percent, e.g., 1.0 for 1%
                    entry_price=entry_price,
                    stop_loss_price=stop_loss,
                    symbol=sym
                )
            except Exception as e:
                self.logger.warning(f"[RiskManager] Position sizing failed for {sym}: {e}")
                position_size = 0.0
            size = position_size

            confidence = 0.7 # Base confidence for TA-Lib patterns, adjust as needed
            reason = f"Trend: {trend}, Patterns: {pattern}, Vol Conf: {vol_ok}"
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
            
            signal_timestamp = str(df_primary.index[idx])
            self.logger.info(f"[SignalCandidate] {sym} idx={idx} direction={direction} entry={entry_price} stop={stop_loss} tp={take_profit} pattern={pattern} vol_ok={vol_ok}")
            
            signal = self._build_signal(
                symbol=sym,
                direction=direction,
                entry_price=entry_price,
                stop_loss=stop_loss,
                pattern=pattern,
                confidence=confidence,
                size=size, # Now sized by RiskManager
                position_size=size, # For compatibility with downstream consumers
                timeframe=self.primary_timeframe,
                reason=reason,
                signal_timestamp=signal_timestamp,
                take_profit=take_profit
            )
            signals.append(signal)
        return signals