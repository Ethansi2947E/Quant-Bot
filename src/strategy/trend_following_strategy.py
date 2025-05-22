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

from src.trading_bot import SignalGenerator
from src.risk_manager import RiskManager

class TrendFollowingStrategy(SignalGenerator):
    """Trend Following Strategy: Pure price action (no indicators, no EMAs/ADX/ATR)."""

    def __init__(
        self,
        primary_timeframe: str = "M15",
        risk_per_trade: float = 0.01,
        wick_threshold: float = 0.4,
        volume_confirmation_enabled: bool = True,
        debug_disable_pattern: bool = True,
        debug_disable_volume: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.logger = logger
        self.name = "TrendFollowingStrategy"
        self.description = "Trades price action pullbacks in established trends, using support/resistance, candlestick patterns, and price/volume analysis."
        self.version = "2.0.0"
        self.primary_timeframe = primary_timeframe
        self.risk_per_trade = risk_per_trade
        self.wick_threshold = wick_threshold
        self.volume_confirmation_enabled = not debug_disable_volume and volume_confirmation_enabled
        self.debug_disable_pattern = debug_disable_pattern
        self.debug_disable_volume = debug_disable_volume
        self.lookback_period = 100
        self.risk_manager = RiskManager.get_instance() if hasattr(RiskManager, "get_instance") else RiskManager()
        self.processed_bars = {}
        self.active_trades = {}
        self.params = kwargs
        logger.info(f"Initialized {self.name} v{self.version}")
        logger.info(f"  Primary Timeframe: {self.primary_timeframe}, Lookback: {self.lookback_period}")
        logger.info(f"  Risk: Risk Per Trade={self.risk_per_trade}")

    def _get_trend_direction(self, df: pd.DataFrame, window: int = 5) -> Optional[str]:
        if len(df) < window + 1:
            return None
        highs = df['high'].iloc[-window:]
        lows = df['low'].iloc[-window:]
        latest_high = highs.iloc[-1]
        latest_low = lows.iloc[-1]
        is_uptrend = (latest_high == highs.max()) and (latest_low == lows.max())
        is_downtrend = (latest_high == highs.min()) and (latest_low == lows.min())
        self.logger.debug(f"[TrendWindow] highs={highs.tolist()}, lows={lows.tolist()}, latest_high={latest_high}, latest_low={latest_low}, is_uptrend={is_uptrend}, is_downtrend={is_downtrend}")
        if is_uptrend:
            return "UP"
        elif is_downtrend:
            return "DOWN"
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
        return [self.primary_timeframe]

    @property
    def lookback_periods(self):
        return {self.primary_timeframe: self.lookback_period}

    def is_near_support_resistance(self, df: pd.DataFrame, tolerance: float = 0.005) -> bool:
        current_price = df['close'].iloc[-1]
        swing_highs = df['high'].rolling(20).max().dropna()
        swing_lows = df['low'].rolling(20).min().dropna()
        near_swing_high = any(abs(high - current_price) / high < tolerance for high in swing_highs)
        near_swing_low = any(abs(current_price - low) / low < tolerance for low in swing_lows)
        self.logger.debug(f"[S/R] {current_price=}, near_swing_high={near_swing_high}, near_swing_low={near_swing_low}")
        return near_swing_high or near_swing_low

    def is_near_support(self, df: pd.DataFrame, tolerance: float = 0.005) -> bool:
        current_price = df['close'].iloc[-1]
        swing_lows = df['low'].rolling(20).min().dropna()
        near_support = any(abs(current_price - low) / low < tolerance for low in swing_lows)
        self.logger.debug(f"[Support] {current_price=}, near_support={near_support}")
        return near_support

    def is_near_resistance(self, df: pd.DataFrame, tolerance: float = 0.005) -> bool:
        current_price = df['close'].iloc[-1]
        swing_highs = df['high'].rolling(20).max().dropna()
        near_resistance = any(abs(high - current_price) / high < tolerance for high in swing_highs)
        self.logger.debug(f"[Resistance] {current_price=}, near_resistance={near_resistance}")
        return near_resistance
    
    def check_price_acceptance_rejection(self, df: pd.DataFrame) -> Dict[str, bool]:
        """Check if price has broken through and accepted/rejected S/R levels.
        
        This function detects if price has closed above resistance (bullish) or 
        below support (bearish), confirming zone acceptance/rejection.
        
        Args:
            df (pd.DataFrame): Price data with OHLC columns
            
        Returns:
            Dict[str, bool]: Dictionary with acceptance/rejection flags
        """
        result = {
            "close_above_resistance": False,
            "close_below_support": False,
            "rejection_at_resistance": False,
            "rejection_at_support": False
        }
        
        if len(df) < 3:
            return result
            
        # Find recent S/R levels
        resistance_level = df['high'].rolling(20).max().iloc[-2]
        support_level = df['low'].rolling(20).min().iloc[-2]
        
        current_close = df['close'].iloc[-1]
        prev_close = df['close'].iloc[-2]
        
        # Price acceptance: Close above resistance or below support
        result["close_above_resistance"] = (prev_close < resistance_level) & (current_close > resistance_level)
        result["close_below_support"] = (prev_close > support_level) & (current_close < support_level)
        
        # Price rejection: Long wick indicating rejection
        if self.is_near_resistance(df):
            upper_wick = df['high'].iloc[-1] - max(df['open'].iloc[-1], df['close'].iloc[-1])
            body_size = abs(df['close'].iloc[-1] - df['open'].iloc[-1])
            result["rejection_at_resistance"] = (upper_wick > body_size) & (df['close'].iloc[-1] < df['open'].iloc[-1])
            
        if self.is_near_support(df):
            lower_wick = min(df['open'].iloc[-1], df['close'].iloc[-1]) - df['low'].iloc[-1]
            body_size = abs(df['close'].iloc[-1] - df['open'].iloc[-1])
            result["rejection_at_support"] = (lower_wick > body_size) & (df['close'].iloc[-1] > df['open'].iloc[-1])
            
        self.logger.debug(f"[PriceAcceptance] {result}")
        return result

    def is_valid_volume_spike(self, df: pd.DataFrame) -> bool:
        if 'volume' not in df.columns:
            self.logger.debug("[Volume] 'volume' column missing in DataFrame.")
            return False
        current_volume = df['volume'].iloc[-1]
        vol_mean = df['volume'].rolling(20).mean().iloc[-1]
        body = abs(df['close'].iloc[-1] - df['open'].iloc[-1])
        wick = (df['high'].iloc[-1] - df['low'].iloc[-1]) - body
        if body == 0:
            self.logger.debug(f"[Volume] Body is zero, cannot compute wick/body ratio.")
            return False
        wick_body_ratio = wick / body
        spike = current_volume > 1.5 * vol_mean
        
        # Enhanced volume context with wick location analysis
        upper_wick = df['high'].iloc[-1] - max(df['open'].iloc[-1], df['close'].iloc[-1])
        lower_wick = min(df['open'].iloc[-1], df['close'].iloc[-1]) - df['low'].iloc[-1]
        total_range = df['high'].iloc[-1] - df['low'].iloc[-1]
        
        if total_range == 0:
            self.logger.debug(f"[Volume] Total range is zero, cannot compute wick ratios.")
            return False
            
        upper_wick_ratio = upper_wick / total_range if total_range > 0 else 0
        lower_wick_ratio = lower_wick / total_range if total_range > 0 else 0
        
        # Volume spike with long upper wick at resistance = bearish
        is_bearish_volume = (upper_wick_ratio > 0.5) and self.is_near_resistance(df) and spike
        
        # Volume spike with long lower wick at support = bullish
        is_bullish_volume = (lower_wick_ratio > 0.5) and self.is_near_support(df) and spike
        
        # General volume spike condition modified to consider context
        valid = spike and (wick_body_ratio < 0.3 or is_bearish_volume or is_bullish_volume)
        
        self.logger.debug(f"[Volume] {current_volume=}, {vol_mean=}, {spike=}, {wick_body_ratio=:.3f}, "
                          f"upper_wick_ratio={upper_wick_ratio:.3f}, lower_wick_ratio={lower_wick_ratio:.3f}, "
                          f"is_bearish_volume={is_bearish_volume}, is_bullish_volume={is_bullish_volume}, valid={valid}")
        return valid

    def get_support_resistance_zones(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """
        Identify key support and resistance zones based on recent price action.
        
        Args:
            df (pd.DataFrame): Price data with OHLC
            
        Returns:
            Dict[str, List[float]]: Dictionary with support and resistance levels
        """
        if len(df) < 20:
            return {"support": [], "resistance": []}
            
        # Find support zones (recent lows)
        swing_lows = []
        for i in range(10, len(df) - 10):
            if df['low'].iloc[i] <= min(df['low'].iloc[i-10:i]) and df['low'].iloc[i] <= min(df['low'].iloc[i+1:i+10]):
                swing_lows.append(df['low'].iloc[i])
                
        # Find resistance zones (recent highs)
        swing_highs = []
        for i in range(10, len(df) - 10):
            if df['high'].iloc[i] >= max(df['high'].iloc[i-10:i]) and df['high'].iloc[i] >= max(df['high'].iloc[i+1:i+10]):
                swing_highs.append(df['high'].iloc[i])
                
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
            
            # Return average of each cluster
            return [sum(cluster) / len(cluster) for cluster in clusters]
            
        support_zones = cluster_levels(swing_lows)
        resistance_zones = cluster_levels(swing_highs)
        
        # Use recent min/max as backup if no zones found
        if not support_zones:
            support_zones = [df['low'].rolling(20).min().iloc[-1]]
        if not resistance_zones:
            resistance_zones = [df['high'].rolling(20).max().iloc[-1]]
            
        self.logger.debug(f"[S/R Zones] Support: {support_zones}, Resistance: {resistance_zones}")
        return {"support": support_zones, "resistance": resistance_zones}

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

    def calculate_take_profit(self, df: pd.DataFrame, direction: str) -> Optional[float]:
        """
        Calculate take profit based on opposing S/R zones, with 2R fallback.
        
        Args:
            df (pd.DataFrame): Price data
            direction (str): Trade direction ("buy" or "sell")
            
        Returns:
            Optional[float]: Calculated take profit price or None if not possible
        """
        zones = self.get_support_resistance_zones(df)
        current_price = df['close'].iloc[-1]
        
        if direction == "buy":
            # For buy, use resistance zones as take profit
            resistance_levels = sorted(zones["resistance"])  # Sort ascending
            
            # Find next resistance above current price
            tp = None
            for level in resistance_levels:
                if level > current_price:
                    tp = level
                    break
                    
            if tp is None:
                # Fallback to 2R
                risk = abs(current_price - self.calculate_stop_loss(df, direction))
                tp = current_price + 2 * risk
                self.logger.debug(f"[TP] BUY: Using fallback 2R TP={tp} (no suitable resistance found)")
            else:
                self.logger.debug(f"[TP] BUY: Using resistance zone TP={tp}")
            
            return tp
        else:
            # For sell, use support zones as take profit
            support_levels = sorted(zones["support"], reverse=True)  # Sort descending
            
            # Find next support below current price
            tp = None
            for level in support_levels:
                if level < current_price:
                    tp = level
                    break
                    
            if tp is None:
                # Fallback to 2R
                risk = abs(current_price - self.calculate_stop_loss(df, direction))
                tp = current_price - 2 * risk
                self.logger.debug(f"[TP] SELL: Using fallback 2R TP={tp} (no suitable support found)")
            else:
                self.logger.debug(f"[TP] SELL: Using support zone TP={tp}")
            
            return tp

    async def generate_signals(self, market_data: Dict[str, Any], symbol: Optional[str] = None, **kwargs) -> List[Dict]:
        logger.debug(f"[StrategyInit] {self.__class__.__name__}: required_timeframes={self.required_timeframes}, lookback_periods={self.lookback_periods}")
        signals = []
        min_bars = 25
        for sym, data in market_data.items():
            if isinstance(data, dict) and self.primary_timeframe in data:
                df = data[self.primary_timeframe]
            elif isinstance(data, pd.DataFrame):
                df = data
            else:
                self.logger.debug(f"[Data] Invalid data structure for {sym}")
                continue
            if not isinstance(df, pd.DataFrame) or len(df) < min_bars:
                self.logger.debug(f"[Data] Not enough data for {sym} (need {min_bars}, got {len(df) if isinstance(df, pd.DataFrame) else 0})")
                continue
            # Precompute vectorized patterns
            hammer = self.detect_hammer(df)
            shooting_star = self.detect_shooting_star(df)
            bullish_engulfing = self.detect_bullish_engulfing(df)
            bearish_engulfing = self.detect_bearish_engulfing(df)
            inside_bar = self.detect_inside_bar(df)
            morning_star = self.detect_morning_star(df)
            evening_star = self.detect_evening_star(df)
            false_breakout_buy = self.detect_false_breakout(df, 'buy')
            false_breakout_sell = self.detect_false_breakout(df, 'sell')

            idx = len(df) - 1
            trend = self._get_trend_direction(df, window=5)
            self.logger.debug(f"[Trend] {sym} idx={idx} trend={trend}")
            if trend not in ("UP", "DOWN"):
                self.logger.info(f"[Skip] {sym} idx={idx} No valid trend detected.")
                continue
                
            # Check for price acceptance/rejection patterns
            price_action = self.check_price_acceptance_rejection(df)
            
            # First, check if price is near any support/resistance level or has broken through one
            if not (self.is_near_support_resistance(df) or price_action["close_above_resistance"] or price_action["close_below_support"]):
                self.logger.info(f"[Skip] {sym} idx={idx} Price not near any S/R level and no breakout detected.")
                continue
            
            detected_patterns = []
            if trend == "UP" and (self.is_near_support(df) or price_action["close_above_resistance"]):
                if hammer.iloc[idx]:
                    detected_patterns.append("Hammer")
                if bullish_engulfing.iloc[idx]:
                    detected_patterns.append("Bullish Engulfing")
                if morning_star.iloc[idx]:
                    detected_patterns.append("Morning Star")
                if inside_bar.iloc[idx]:
                    detected_patterns.append("Inside Bar")
                if false_breakout_buy.iloc[idx]:
                    detected_patterns.append("False Breakout (Buy)")
                if price_action["close_above_resistance"]:
                    detected_patterns.append("Resistance Breakout")
                if price_action["rejection_at_support"]:
                    detected_patterns.append("Support Rejection (Bullish)")
            elif trend == "DOWN" and (self.is_near_resistance(df) or price_action["close_below_support"]):
                if shooting_star.iloc[idx]:
                    detected_patterns.append("Shooting Star")
                if bearish_engulfing.iloc[idx]:
                    detected_patterns.append("Bearish Engulfing")
                if evening_star.iloc[idx]:
                    detected_patterns.append("Evening Star")
                if inside_bar.iloc[idx]:
                    detected_patterns.append("Inside Bar")
                if false_breakout_sell.iloc[idx]:
                    detected_patterns.append("False Breakout (Sell)")
                if price_action["close_below_support"]:
                    detected_patterns.append("Support Breakdown")
                if price_action["rejection_at_resistance"]:
                    detected_patterns.append("Resistance Rejection (Bearish)")
                    
            self.logger.debug(f"[Pattern] {sym} idx={idx} detected_patterns={detected_patterns}")
            pattern_ok = bool(detected_patterns) or self.debug_disable_pattern
            if not pattern_ok:
                self.logger.info(f"[Skip] {sym} idx={idx} No valid pattern at S/R zone.")
                continue
            # Volume-wick confirmation
            vol_ok = self.debug_disable_volume or self.is_valid_volume_spike(df)
            self.logger.debug(f"[VolumeCheck] {sym} idx={idx} vol_ok={vol_ok}")
            if not vol_ok:
                self.logger.info(f"[Skip] {sym} idx={idx} Volume-wick confirmation failed.")
                continue
            pattern = ", ".join(detected_patterns)
            entry_price = df['close'].iloc[idx]
            direction = "buy" if trend == "UP" else "sell"
            stop_loss = self.calculate_stop_loss(df, direction)
            take_profit = self.calculate_take_profit(df, direction)
            if not take_profit or take_profit == 0:
                self.logger.info(f"[Skip] {sym} idx={idx} No valid take-profit found, skipping signal.")
                continue
            size = 0.0
            confidence = 1.0
            reason = f"Trend: {trend}, S/R, Patterns: {pattern}, Volume Confirmed: {vol_ok}"
            if any(key for key, val in price_action.items() if val):
                reason += f", Price Action: {[key for key, val in price_action.items() if val]}"
            signal_timestamp = str(df.index[idx])
            self.logger.info(f"[SignalCandidate] {sym} idx={idx} direction={direction} entry={entry_price} stop={stop_loss} tp={take_profit} pattern={pattern} vol_ok={vol_ok}")
            signal = self._build_signal(
                symbol=sym,
                direction=direction,
                entry_price=entry_price,
                stop_loss=stop_loss,
                pattern=pattern,
                confidence=confidence,
                size=size,
                timeframe=self.primary_timeframe,
                reason=reason,
                signal_timestamp=signal_timestamp,
                take_profit=take_profit
            )
            signals.append(signal)
        return signals

    # --- VECTORIZE CANDLESTICK PATTERNS (industry standard) ---
    @staticmethod
    def detect_hammer(df: pd.DataFrame) -> pd.Series:
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
    def detect_shooting_star(df: pd.DataFrame) -> pd.Series:
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
        prev_open = df['open'].shift(1)
        prev_close = df['close'].shift(1)
        is_prev_bearish = prev_close < prev_open
        is_curr_bullish = df['close'] > df['open']
        engulfs = (df['open'] < prev_close) & (df['close'] > prev_open)
        return is_prev_bearish & is_curr_bullish & engulfs

    @staticmethod
    def detect_bearish_engulfing(df: pd.DataFrame) -> pd.Series:
        prev_open = df['open'].shift(1)
        prev_close = df['close'].shift(1)
        is_prev_bullish = prev_close > prev_open
        is_curr_bearish = df['close'] < df['open']
        engulfs = (df['open'] > prev_close) & (df['close'] < prev_open)
        return is_prev_bullish & is_curr_bearish & engulfs

    @staticmethod
    def detect_inside_bar(df: pd.DataFrame) -> pd.Series:
        prev_high = df['high'].shift(1)
        prev_low = df['low'].shift(1)
        return (df['high'] < prev_high) & (df['low'] > prev_low)

    @staticmethod
    def detect_morning_star(df: pd.DataFrame) -> pd.Series:
        c1 = df.shift(2)
        c2 = df.shift(1)
        c3 = df
        c1_body = (c1['close'] - c1['open']).abs()
        c2_body = (c2['close'] - c2['open']).abs()
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
    def detect_evening_star(df: pd.DataFrame) -> pd.Series:
        c1 = df.shift(2)
        c2 = df.shift(1)
        c3 = df
        c1_body = (c1['close'] - c1['open']).abs()
        c2_body = (c2['close'] - c2['open']).abs()
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
    def detect_false_breakout(df: pd.DataFrame, direction: str, price_tolerance: float = 0.002, volume_threshold: Optional[float] = None) -> pd.Series:
        if len(df) < 2:
            return pd.Series(dtype=bool, index=df.index)
        prev_close = df['close'].shift(1)
        tol_val = df['close'] * price_tolerance
        total_range = df['high'] - df['low']
        if direction == 'buy':
            wick = df['close'] - df['low']
            wick_ok = wick > 0.5 * total_range
            if volume_threshold is not None and 'tick_volume' in df.columns:
                vol_ok = df['tick_volume'] > volume_threshold
                result = (
                    (prev_close < df['close'] - tol_val) &
                    wick_ok &
                    vol_ok
                )
            else:
                result = (
                    (prev_close < df['close'] - tol_val) &
                    wick_ok
                )
        else:
            wick = df['high'] - df['close']
            wick_ok = wick > 0.5 * total_range
            if volume_threshold is not None and 'tick_volume' in df.columns:
                vol_ok = df['tick_volume'] > volume_threshold
                result = (
                    (prev_close > df['close'] + tol_val) &
                    wick_ok &
                    vol_ok
                )
            else:
                result = (
                    (prev_close > df['close'] + tol_val) &
                    wick_ok
                )
        return result 