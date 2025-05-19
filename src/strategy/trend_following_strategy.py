"""
trend_following_strategy.py

Trend Following Strategy

A rules-based strategy to trade in the direction of an established trend,
entering on pullbacks and using trailing stops to let profits run.

Features:
- Trend identification using ADX and Moving Average alignment (EMAs).
- Entry on pullbacks to dynamic support/resistance (EMA20).
- Confirmation with candlestick patterns and optional volume.
- ATR-based initial stop-loss.
- ATR-based trailing stop-loss for profit protection and capturing trend moves.
- Risk management via fixed fractional position sizing.
"""

import pandas as pd
import numpy as np
from loguru import logger
from typing import Dict, List, Any, Optional

from src.trading_bot import SignalGenerator
from src.risk_manager import RiskManager
from src.utils.indicators import calculate_adx, calculate_moving_average, calculate_atr

class TrendFollowingStrategy(SignalGenerator):
    """Trend Following Strategy: Identifies trends and trades pullbacks."""

    TIMEFRAME_PRESETS = {
        "M1": {
            "ema_short_period": 5,
            "ema_long_period": 20,
            "adx_period": 14,
            "adx_threshold_trending": 20,
            "pullback_tolerance": 0.0015,
            "trend_confirmation_bars": 2,
            "atr_period": 14,
            "initial_stop_loss_atr_multiplier": 0.8,
            "pullback_ema_period": 5,
        },
        "M5": {
            "ema_short_period": 10,
            "ema_long_period": 50,
            "adx_period": 14,
            "adx_threshold_trending": 25,
            "pullback_tolerance": 0.002,
            "trend_confirmation_bars": 3,
            "atr_period": 14,
            "initial_stop_loss_atr_multiplier": 1.2,
        },
    }

    def apply_timeframe_preset(self, timeframe: str):
        """Apply parameter preset for a given timeframe if available."""
        preset = self.TIMEFRAME_PRESETS.get(timeframe)
        if preset:
            for k, v in preset.items():
                setattr(self, k, v)

    def __init__(
        self,
        primary_timeframe: str = "M5",
        risk_per_trade: float = 0.01,
        adx_period: int = 14,
        adx_threshold_trending: float = 25.0,
        ema_short_period: int = 20,
        ema_long_period: int = 50,
        trend_confirmation_bars: int = 3,
        pullback_ema_period: int = 20,
        wick_threshold: float = 0.4,
        volume_confirmation_enabled: bool = True,
        atr_period: int = 14,
        initial_stop_loss_atr_multiplier: float = 1.0,
        pullback_tolerance: float = 0.002,
        debug_disable_pattern: bool = False,
        debug_disable_volume: bool = False,
        **kwargs
    ):
        """Initialize the TrendFollowingStrategy for M15 timeframe by default."""
        super().__init__(**kwargs)
        self.logger = logger
        self.name = "TrendFollowingStrategy"
        self.description = "Trades pullbacks in established trends, using EMAs, ADX, and ATR-based stop-loss. Trailing stop is managed by PositionManager."
        self.version = "1.0.0"
        self.primary_timeframe = primary_timeframe
        self.risk_per_trade = risk_per_trade
        self.adx_period = adx_period
        self.adx_threshold_trending = adx_threshold_trending
        self.ema_short_period = ema_short_period
        self.ema_long_period = ema_long_period
        self.trend_confirmation_bars = trend_confirmation_bars
        self.pullback_ema_period = pullback_ema_period
        self.wick_threshold = wick_threshold
        self.volume_confirmation_enabled = not debug_disable_volume and volume_confirmation_enabled
        self.atr_period = atr_period
        self.initial_stop_loss_atr_multiplier = initial_stop_loss_atr_multiplier
        self.pullback_tolerance = pullback_tolerance
        self.debug_disable_pattern = debug_disable_pattern
        self.debug_disable_volume = debug_disable_volume
        # Apply preset if available for the selected timeframe
        self.apply_timeframe_preset(self.primary_timeframe)
        self.lookback_period = max(100, self.ema_long_period + 20, self.adx_period + 20, self.atr_period + 20)
        self.risk_manager = RiskManager.get_instance() if hasattr(RiskManager, "get_instance") else RiskManager()
        self.processed_bars = {}
        self.active_trades = {}
        self.params = kwargs
        logger.info(f"Initialized {self.name} v{self.version}")
        logger.info(f"  Primary Timeframe: {self.primary_timeframe}, Lookback: {self.lookback_period}")
        logger.info(f"  Trend ID: ADX Period={self.adx_period}, ADX Threshold={self.adx_threshold_trending}")
        logger.info(f"    EMAs: {self.ema_short_period}/{self.ema_long_period}, Confirm Bars={self.trend_confirmation_bars}")
        logger.info(f"  Entry: Pullback EMA={self.pullback_ema_period}, Wick Threshold={self.wick_threshold}, Volume Confirm={self.volume_confirmation_enabled}")
        logger.info(f"  Exits: ATR Period={self.atr_period}, Initial SL ATR Mult={self.initial_stop_loss_atr_multiplier}")
        logger.info(f"  Risk: Risk Per Trade={self.risk_per_trade}")

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate and append ADX, EMA, and ATR indicators to the DataFrame.

        Args:
            df (pd.DataFrame): Price data with 'high', 'low', 'close' columns
        Returns:
            pd.DataFrame: DataFrame with indicator columns added
        """
        min_bars = max(self.ema_long_period, self.adx_period, self.atr_period)
        if df is None or df.empty or len(df) < min_bars:
            self.logger.debug(f"[Indicators] Not enough data for indicator calculation (need {min_bars}, got {len(df) if df is not None else 0})")
            # Always append indicator columns with NaN so downstream code works
            for col in [
                f'EMA_{self.ema_short_period}', f'EMA_{self.ema_long_period}',
                f'ATR_{self.atr_period}', f'ADX_{self.adx_period}',
                f'DI+_{self.adx_period}', f'DI-_{self.adx_period}'
            ]:
                df[col] = np.nan
            return df

        # Calculate EMAs
        ema_short_col = f"EMA_{self.ema_short_period}"
        ema_long_col = f"EMA_{self.ema_long_period}"
        df[ema_short_col] = calculate_moving_average(df, column="close", period=self.ema_short_period, ma_type="exponential")
        df[ema_long_col] = calculate_moving_average(df, column="close", period=self.ema_long_period, ma_type="exponential")

        # Calculate ATR
        atr_col = f"ATR_{self.atr_period}"
        df[atr_col] = calculate_atr(df, period=self.atr_period)

        # Calculate ADX and DI
        adx_col = f"ADX_{self.adx_period}"
        dip_col = f"DI+_{self.adx_period}"
        dim_col = f"DI-_{self.adx_period}"
        adx, dip, dim = calculate_adx(df, period=self.adx_period)
        df[adx_col] = adx
        df[dip_col] = dip
        df[dim_col] = dim

        # Log last row of indicators for debug
        last_idx = df.index[-1]
        self.logger.debug(
            f"[Indicators] Last row: "
            f"close={df['close'].iloc[-1]}, "
            f"{ema_short_col}={df[ema_short_col].iloc[-1]}, "
            f"{ema_long_col}={df[ema_long_col].iloc[-1]}, "
            f"{atr_col}={df[atr_col].iloc[-1]}, "
            f"{adx_col}={df[adx_col].iloc[-1]}, "
            f"{dip_col}={df[dip_col].iloc[-1]}, "
            f"{dim_col}={df[dim_col].iloc[-1]}"
        )
        return df

    def _get_trend_direction(self, df: pd.DataFrame, idx: int) -> Optional[str]:
        """Determine the current trend direction based on ADX, EMA alignment, and price position."""
        if idx < self.trend_confirmation_bars - 1:
            self.logger.debug("[Trend] Not enough bars for trend confirmation.")
            return None

        adx_col = f"ADX_{self.adx_period}"
        dip_col = f"DI+_{self.adx_period}"
        dim_col = f"DI-_{self.adx_period}"
        ema_short_col = f"EMA_{self.ema_short_period}"
        ema_long_col = f"EMA_{self.ema_long_period}"

        # Only check once, at the latest bar
        current_idx = idx
        if any(pd.isna(df[col].iloc[current_idx]) for col in [adx_col, dip_col, dim_col, ema_short_col, ema_long_col]):
            self.logger.debug(f"[Trend] Missing indicator data at idx {current_idx}.")
            return None
        adx_val = df[adx_col].iloc[current_idx]
        dip_val = df[dip_col].iloc[current_idx]
        dim_val = df[dim_col].iloc[current_idx]
        ema_short = df[ema_short_col].iloc[current_idx]
        ema_long = df[ema_long_col].iloc[current_idx]
        price_close = df["close"].iloc[current_idx]
        # Slope checks over the period
        start_idx = current_idx - self.trend_confirmation_bars + 1
        if start_idx < 0:
            self.logger.debug(f"[Trend] Not enough data for EMA slope at idx {current_idx}.")
            return None
        ema_short_start = df[ema_short_col].iloc[start_idx]
        ema_long_start = df[ema_long_col].iloc[start_idx]
        ema_short_slope_up = ema_short > ema_short_start
        ema_long_slope_up = ema_long > ema_long_start
        ema_short_slope_down = ema_short < ema_short_start
        ema_long_slope_down = ema_long < ema_long_start
        # Uptrend conditions
        is_uptrend = (
            adx_val >= self.adx_threshold_trending and
            dip_val > dim_val and
            ema_short > ema_long and
            ema_short_slope_up and ema_long_slope_up and
            price_close > ema_long  # Relaxed: only require price > EMA_long
        )
        # Downtrend conditions
        is_downtrend = (
            adx_val >= self.adx_threshold_trending and
            dim_val > dip_val and
            ema_short < ema_long and
            ema_short_slope_down and ema_long_slope_down and
            price_close < ema_long
        )
        if is_uptrend:
            self.logger.info(f"[Trend] Confirmed UP trend at idx {idx}.")
            return "UP"
        if is_downtrend:
            self.logger.info(f"[Trend] Confirmed DOWN trend at idx {idx}.")
            return "DOWN"
        self.logger.debug(f"[Trend] No trend at idx {idx}.")
        return None

    def _calculate_initial_stop_loss(self, direction: str, candle: pd.Series, atr: float, ema_long: float) -> float:
        """Calculate the initial stop-loss for a trade.

        Args:
            direction (str): 'buy' or 'sell'
            candle (pd.Series): Confirmation candle
            atr (float): ATR value
            ema_long (float): EMA(50) value
        Returns:
            float: Stop-loss price
        """
        if direction == "buy":
            sl_candle = candle["low"] - self.initial_stop_loss_atr_multiplier * atr
            sl_ema = ema_long if ema_long < candle["low"] else sl_candle
            stop_loss = max(sl_candle, sl_ema) if abs(sl_ema - candle["close"]) < 2 * atr else sl_candle
        else:
            sl_candle = candle["high"] + self.initial_stop_loss_atr_multiplier * atr
            sl_ema = ema_long if ema_long > candle["high"] else sl_candle
            stop_loss = min(sl_candle, sl_ema) if abs(sl_ema - candle["close"]) < 2 * atr else sl_candle
        self.logger.debug(f"[Stops] Initial stop-loss for {direction}: {stop_loss:.5f} (Candle: {sl_candle:.5f}, EMA: {ema_long:.5f}, ATR: {atr:.5f})")
        return float(stop_loss) if stop_loss is not None else 0.0

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
            "take_profit": float(take_profit) if take_profit is not None else None,
        }
        signal.update(kwargs)
        self.logger.info(
            f"[Signal] {symbol} {direction.upper()} @ {entry_price:.5f}, SL={stop_loss:.5f}, Size={size:.3f}, Pattern={pattern}, Conf={confidence:.2f}, Reason={reason}"
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

    def _score_signal(self, idx, df, pattern, vol_mean, vol_std, atr_percentile_series):
        # Trend strength: (ADX - 25) / 25, clipped to [0,1]
        adx_col = f"ADX_{self.adx_period}"
        adx = df[adx_col].iloc[idx]
        trend_score = np.clip((adx - 25) / 25, 0, 1) if not pd.isna(adx) else 0.0
        # MA alignment: (EMA_short - EMA_long) / ATR, clipped to [0,1]
        ema_short_col = f"EMA_{self.ema_short_period}"
        ema_long_col = f"EMA_{self.ema_long_period}"
        atr_col = f"ATR_{self.atr_period}"
        ema_short = df[ema_short_col].iloc[idx]
        ema_long = df[ema_long_col].iloc[idx]
        atr = df[atr_col].iloc[idx]
        ma_score = np.clip((ema_short - ema_long) / atr, 0, 1) if atr and not pd.isna(atr) else 0.0
        # Pullback quality: 1 - abs(close - EMA_short) / EMA_short, clipped
        close = df['close'].iloc[idx]
        pullback = 1 - abs(close - ema_short) / ema_short if ema_short else 0.0
        pullback_score = np.clip(pullback, 0, 1)
        # Pattern power
        pattern_map = {
            'Bullish Engulfing': 1.0,
            'Bearish Engulfing': 1.0,
            'Hammer': 0.8,
            'Shooting Star': 0.8,
            'Pin-bar': 0.7,
            'Doji': 0.4,
            'Morning Star': 0.9,
            'Evening Star': 0.9,
            'Inside Bar': 0.5,
            'False Breakout (Buy)': 0.7,
            'False Breakout (Sell)': 0.7,
            '': 0.0,
            None: 0.0
        }
        # If multiple patterns, take the max score
        if pattern:
            if isinstance(pattern, str) and ',' in pattern:
                pattern_score = max(pattern_map.get(p.strip(), 0.0) for p in pattern.split(','))
            else:
                pattern_score = pattern_map.get(pattern, 0.0)
        else:
            pattern_score = 0.0
        # Volume confirmation: Z-score normalized, clipped
        volume = df['tick_volume'].iloc[idx] if 'tick_volume' in df.columns else 0.0
        vol_zscore = (volume - vol_mean) / vol_std if vol_std > 0 else 0.0
        vol_score = np.clip(vol_zscore / 3, 0, 1)
        # ATR regime: percentile rank of ATR in the year-to-date series
        atr_score = atr_percentile_series.iloc[idx] if atr_percentile_series is not None else 0.0
        # Weights
        weights = [0.25, 0.10, 0.20, 0.20, 0.15, 0.10]
        scores = [trend_score, ma_score, pullback_score, pattern_score, vol_score, atr_score]
        score = float(np.dot(weights, scores))
        breakdown = {
            'trend_score': trend_score,
            'ma_score': ma_score,
            'pullback_score': pullback_score,
            'pattern_score': pattern_score,
            'vol_score': vol_score,
            'atr_score': atr_score,
            'weights': weights,
            'scores': scores,
            'final_score': score
        }
        return score, breakdown

    async def generate_signals(self, market_data: Dict[str, Any], symbol: Optional[str] = None, **kwargs) -> List[Dict]:
        logger.debug(f"[StrategyInit] {self.__class__.__name__}: required_timeframes={self.required_timeframes}, lookback_periods={self.lookback_periods}")
        signals = []
        min_bars = max(self.ema_long_period, self.adx_period, self.atr_period)
        take_profit_rr_ratio = 2.0  # Default R:R for TP
        for sym, data in market_data.items():
            # Handle nested timeframe structure
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
            df = self._calculate_indicators(df.copy())
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

            # Compute volume mean/std for last 50 bars
            vol_mean = df['tick_volume'].rolling(50, min_periods=10).mean().iloc[-1] if 'tick_volume' in df.columns else 1.0
            vol_std = df['tick_volume'].rolling(50, min_periods=10).std().iloc[-1] if 'tick_volume' in df.columns else 1.0
            # Compute ATR percentile rank (year-to-date or available data)
            atr_col = f"ATR_{self.atr_period}"
            if atr_col in df.columns:
                atr_series = df[atr_col]
                atr_percentile_series = atr_series.rank(pct=True)
            else:
                atr_percentile_series = pd.Series(0.0, index=df.index)

            # Only check the latest bar (most recent actionable bar)
            idx = len(df) - 1
            trend = self._get_trend_direction(df, idx - 1)
            if trend not in ("UP", "DOWN"):
                continue
            # Use the correct EMA for pullback
            ema_col = f"EMA_{self.pullback_ema_period}"
            if ema_col not in df.columns:
                self.logger.debug(f"[Pullback] EMA column {ema_col} not found for {sym}")
                continue
            ema_val = df[ema_col].iloc[idx]
            low = df['low'].iloc[idx]
            high = df['high'].iloc[idx]
            tolerance = self.pullback_tolerance * df['close'].iloc[idx]
            pullback_ok = (low - tolerance <= ema_val <= high + tolerance)
            self.logger.debug(f"[Pullback] {sym} idx={idx} ema_col={ema_col} ema_val={ema_val}, low={low}, high={high}, tolerance={tolerance:.5f}, pullback_ok={pullback_ok}")
            if not pullback_ok:
                continue
            # Pattern check (can be disabled for debug)
            detected_patterns = []
            if trend == "UP":
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
            elif trend == "DOWN":
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
            pattern_ok = bool(detected_patterns) or self.debug_disable_pattern
            # Log pattern check
            self.logger.debug(f"[DEBUG] {sym} idx={idx} pattern_ok={pattern_ok} (patterns={detected_patterns}, debug_disable_pattern={self.debug_disable_pattern})")
            if not pattern_ok:
                # Log when pullback would have signaled if pattern filter was off
                self.logger.info(f"[DEBUG] {sym} idx={idx} Would signal on pullback if pattern filter was off.")
                continue
            # Volume confirmation (can be disabled for debug)
            vol_ok = True
            if self.volume_confirmation_enabled:
                # Example: require volume above rolling mean
                vol_ok = df['tick_volume'].iloc[idx] > vol_mean if 'tick_volume' in df.columns else True
            # Log volume check
            self.logger.debug(f"[DEBUG] {sym} idx={idx} vol_ok={vol_ok} (volume={df['tick_volume'].iloc[idx] if 'tick_volume' in df.columns else 'N/A'}, mean={vol_mean}, enabled={self.volume_confirmation_enabled})")
            if not vol_ok:
                continue
            pattern = ", ".join(detected_patterns)
            self.logger.debug(f"[Pattern] {sym} idx={idx} Detected: {pattern}")
            atr_col = f"ATR_{self.atr_period}"
            atr = df[atr_col].iloc[idx] if atr_col in df.columns else None
            ema_long_col = f"EMA_{self.ema_long_period}"
            ema_long = df[ema_long_col].iloc[idx] if ema_long_col in df.columns else None
            direction = "buy" if trend == "UP" else "sell"
            stop_loss = self._calculate_initial_stop_loss(direction, df.iloc[idx], atr, ema_long) if atr is not None and ema_long is not None else 0.0
            entry_price = df['close'].iloc[idx]
            # Calculate take_profit using R:R ratio
            take_profit = None
            if stop_loss is not None and stop_loss != entry_price:
                if direction == "buy":
                    take_profit = entry_price + abs(entry_price - stop_loss) * take_profit_rr_ratio
                else:
                    take_profit = entry_price - abs(entry_price - stop_loss) * take_profit_rr_ratio
            size = 0.0
            risk_amt = self.risk_per_trade * kwargs.get('balance', 10000)
            size = risk_amt / abs(entry_price - stop_loss)
            # --- Signal scoring ---
            score, score_breakdown = self._score_signal(idx, df, pattern, vol_mean, vol_std, atr_percentile_series)
            confidence = min(1.0, max(0.1, score))
            reason = f"Trend: {trend}, Pullback to EMA{self.pullback_ema_period}, Patterns: {pattern}, Score: {score:.2f}"
            signal_timestamp = str(df.index[idx])
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
                take_profit=take_profit,
                score=score,
                score_breakdown=score_breakdown
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