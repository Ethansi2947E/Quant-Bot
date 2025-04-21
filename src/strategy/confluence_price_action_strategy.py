"""
Confluence Price Action Strategy

This strategy identifies the prevailing trend on a higher timeframe, marks key support/resistance levels, waits for pullbacks on a lower timeframe, and then looks for confluence of price-action signals (pin bars, engulfing bars, inside bars, false breakouts), Fibonacci retracements, and moving-average support/resistance. Risk management enforces fixed fractional sizing and minimum R:R.
"""

import pandas as pd
import numpy as np
from loguru import logger
from datetime import datetime

from src.trading_bot import SignalGenerator
from src.utils.indicators import calculate_atr
from config.config import TRADING_CONFIG, get_pattern_detector_config, get_risk_manager_config
from src.risk_manager import RiskManager

# Timeframe-specific profiles for dynamic parameter scaling
TIMEFRAME_PROFILES = {
    "M5": {"pivot_lookback": 140, "pullback_bars": 12, "pattern_bars": 6, "ma_period": 21, "price_tolerance": 0.002},
    "M15": {"pivot_lookback": 96, "pullback_bars": 6, "pattern_bars": 3, "ma_period": 34, "price_tolerance": 0.002},
    "H1": {"pivot_lookback": 50, "pullback_bars": 6, "pattern_bars": 2, "ma_period": 55, "price_tolerance": 0.002},
    "H4": {"pivot_lookback": 30, "pullback_bars": 4, "pattern_bars": 2, "ma_period": 89, "price_tolerance": 0.002},
    "D1": {"pivot_lookback": 20, "pullback_bars": 3, "pattern_bars": 1, "ma_period": 144, "price_tolerance": 0.002}
}
DEFAULT_PROFILE = {"pivot_lookback": 50, "pullback_bars": 5, "pattern_bars": 3, "ma_period": 21, "price_tolerance": 0.002}

# Add fallback max lookback bars by timeframe
MAX_LOOKBACK_BARS = {
    "M5": 200,
    "M15": 150,
    "H1": 100, 
    "H4": 80,
    "D1": 50
}

class ConfluencePriceActionStrategy(SignalGenerator):
    """
    Confluence-based price action strategy with pullbacks, candlestick confirmations,
    Fibonacci and moving-average confluence, plus strict risk management.
    """

    def __init__(self,
                 primary_timeframe: str = "M5",
                 higher_timeframe: str = "H1",
                 ma_period: int = 21,
                 fib_levels=(0.5, 0.618),
                 risk_percent: float = 0.01,
                 min_risk_reward: float = 2.0,
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
        # Override price tolerance from global config if set
        data_mgmt = TRADING_CONFIG.get('data_management', {})
        self.price_tolerance = data_mgmt.get('price_tolerance', kwargs.get('price_tolerance', 0.002))
        # Override risk percent from RiskManager config for this timeframe
        rm_conf = get_risk_manager_config(self.primary_timeframe)
        self.risk_percent = rm_conf.get('max_risk_per_trade', self.risk_percent)

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
        
        # Use local MAX_LOOKBACK_BARS instead of TIMEFRAME_CONFIG
        self.pivot_lookback = MAX_LOOKBACK_BARS.get(self.primary_timeframe, self.pivot_lookback)
        
        # Override pattern bars from global pattern detector config
        pattern_conf = get_pattern_detector_config(self.primary_timeframe)
        self.pattern_bars = pattern_conf.get('lookback_period', self.pattern_bars)

        logger.info(
            f"🔄 Timeframe profile loaded for {self.primary_timeframe}: "
            f"pivot_lookback={self.pivot_lookback}, pullback_bars={self.pullback_bars}, "
            f"pattern_bars={self.pattern_bars}, ma_period={self.ma_period}, "
            f"price_tolerance={self.price_tolerance}"
        )

    async def initialize(self) -> bool:
        logger.info(f"🔌 Initializing {self.name}")
        return True

    async def generate_signals(self,
                               market_data: dict = None,
                               symbol: str = None,
                               timeframe: str = None,
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
        signals = []
        if not market_data:
            logger.warning(f"No market_data provided to {self.name}")
            return signals
        # initialize RiskManager and account balance
        rm = RiskManager.get_instance()
        balance = kwargs.get("balance", rm.daily_stats.get('starting_balance', 0) or 10000)
        
        # For detailed debugging
        debug_visualize = kwargs.get("debug_visualize", False)
        
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
            
            # 2. Key levels on higher timeframe
            supports, resistances = self._find_key_levels(higher)
            logger.debug(f"{sym}: Found {len(supports)} support levels and {len(resistances)} resistance levels")
            
            levels = supports if trend == 'bullish' else resistances
            level_type = "support" if trend == 'bullish' else "resistance"
            
            # 3. For each level, check pullback on primary TF
            level_info = []
            for i, level in enumerate(levels):
                level_info.append(f"Level {i+1}: {level:.5f}")
            
            logger.debug(f"{sym}: Checking {len(levels)} {level_type} levels: {', '.join(level_info)}")
            
            for level in levels:
                if not self._is_pullback(primary, level, trend):
                    continue
                
                logger.debug(f"{sym}: Found pullback to {level_type} level {level:.5f}")
                
                # 4. Look for candlestick patterns in recent bars
                start = max(0, len(primary) - self.pattern_bars)
                for idx in range(start, len(primary)):
                    candle = primary.iloc[idx]
                    pattern = None
                    pattern_details = {}
                    
                    if self._is_pin_bar(candle, level, trend):
                        pattern = 'Pin Bar'
                        body = abs(candle['close'] - candle['open'])
                        total = candle['high'] - candle['low']
                        body_to_range_ratio = body / total if total > 0 else 0
                        pattern_details = {
                            'body_size': body,
                            'wick_size': total - body,
                            'body_to_range_ratio': body_to_range_ratio,
                            'price_to_level_distance': abs(candle['low' if trend == 'bullish' else 'high'] - level)
                        }
                        
                    elif self._is_engulfing(primary, idx, trend):
                        pattern = 'Engulfing'
                        prev = primary.iloc[idx-1]
                        pattern_details = {
                            'current_candle_range': candle['high'] - candle['low'],
                            'previous_candle_range': prev['high'] - prev['low'],
                            'engulfing_ratio': (candle['high'] - candle['low']) / (prev['high'] - prev['low']) if (prev['high'] - prev['low']) > 0 else 0
                        }
                        
                    elif self._is_inside_bar(primary, idx):
                        pattern = 'Inside Bar'
                        mother = primary.iloc[idx-1]
                        pattern_details = {
                            'mother_candle_range': mother['high'] - mother['low'],
                            'child_candle_range': candle['high'] - candle['low'],
                            'containment_ratio': (candle['high'] - candle['low']) / (mother['high'] - mother['low']) if (mother['high'] - mother['low']) > 0 else 0
                        }
                        
                    elif self._is_false_breakout(primary, idx, level, trend):
                        pattern = 'False Breakout'
                        prev = primary.iloc[idx-1]
                        breakout_size = abs(prev['close'] - level)
                        reversal_size = abs(candle['close'] - level)
                        pattern_details = {
                            'breakout_size': breakout_size,
                            'reversal_size': reversal_size,
                            'reversal_ratio': reversal_size / breakout_size if breakout_size > 0 else 0
                        }
                        
                    if not pattern:
                        continue
                    
                    logger.debug(f"{sym}: Found {pattern} pattern at {candle.name}")
                    
                    # 5. Confluence: Fibonacci or MA
                    fib_ok = self._check_fibonacci(primary, level)
                    fib_details = {}
                    if fib_ok:
                        high = primary['high'].max()
                        low = primary['low'].min()
                        for f in self.fib_levels:
                            fib_lv = low + (high - low) * f
                            if abs(level - fib_lv) <= level * self.price_tolerance:
                                fib_details = {
                                    'fib_level': f,
                                    'calculated_value': fib_lv,
                                    'price_to_fib_distance': abs(level - fib_lv)
                                }
                                break
                        
                    ma_ok = self._check_ma(primary, level)
                    ma_details = {}
                    if ma_ok:
                        ma = primary['close'].rolling(self.ma_period).mean().iloc[-1]
                        ma_details = {
                            'ma_period': self.ma_period,
                            'ma_value': ma,
                            'price_to_ma_distance': abs(level - ma)
                        }
                        
                    if not (fib_ok or ma_ok):
                        logger.debug(f"{sym}: Pattern found but no Fibonacci or MA confluence")
                        continue
                    
                    logger.info(f"{sym}: Strong confluence signal found - {pattern} with {'Fibonacci' if fib_ok else ''}{' and ' if fib_ok and ma_ok else ''}{'MA' if ma_ok else ''} confluence")
                    
                    # 6. Assemble signal
                    entry = candle['close']
                    if trend == 'bullish':
                        stop = candle['low'] - level * self.price_tolerance
                        reward = entry - stop
                        tp = entry + reward * self.min_risk_reward
                        direction = 'buy'
                    else:
                        stop = candle['high'] + level * self.price_tolerance
                        reward = stop - entry
                        tp = entry - reward * self.min_risk_reward
                        direction = 'sell'
                        
                    # Calculate risk-reward and other trade metrics
                    risk_pips = abs(entry - stop)
                    reward_pips = abs(tp - entry)
                    risk_reward_ratio = reward_pips / risk_pips if risk_pips > 0 else 0
                    
                    # Delegate position sizing to RiskManager
                    size = rm.calculate_position_size(
                        account_balance=balance,
                        risk_per_trade=self.risk_percent * 100,
                        entry_price=entry,
                        stop_loss_price=stop,
                        symbol=sym
                    )
                    
                    # Calculate additional metrics for score/ranking
                    recency_score = (idx - start + 1) / (len(primary) - start) if len(primary) > start else 0.5
                    
                    # Calculate volume analysis if volume column exists
                    volume_score = 0.0
                    volume_details = {}
                    if 'volume' in primary.columns or 'tick_volume' in primary.columns:
                        vol_col = 'volume' if 'volume' in primary.columns else 'tick_volume'
                        current_vol = candle[vol_col]
                        avg_vol = primary[vol_col].rolling(window=20).mean().iloc[-1]
                        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
                        volume_score = min(1.0, vol_ratio / 2.0)  # Scale 0-2x average to 0-1 score
                        volume_details = {
                            'current_volume': current_vol,
                            'average_volume': avg_vol,
                            'volume_ratio': vol_ratio
                        }
                    
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
                    elif pattern == 'False Breakout':
                        # False breakouts stronger when reversal ratio is higher
                        pattern_score = min(1.0, pattern_details['reversal_ratio'] / 2.0)
                    
                    # Score confluence factors
                    confluence_score = 0.0
                    if fib_ok and ma_ok:
                        confluence_score = 1.0  # Both fib and MA is maximum confluence
                    elif fib_ok or ma_ok:
                        confluence_score = 0.6  # Single confluence factor
                    
                    # Calculate overall signal quality 
                    signal_quality = (
                        (pattern_score * 0.4) +
                        (confluence_score * 0.4) +
                        (volume_score * 0.1) +
                        (recency_score * 0.1)
                    )
                    
                    # Scale to confidence value between 0.5-0.95
                    confidence = 0.5 + (signal_quality * 0.45)
                    
                    # Build detailed reasoning
                    reasoning = [
                        f"{self.higher_timeframe} trend: {trend.upper()}",
                        f"Price pullback to {level_type} level at {level:.5f}",
                        f"Pattern: {pattern} detected on {self.primary_timeframe}",
                    ]
                    
                    # Add specific pattern details to reasoning
                    if pattern == 'Pin Bar':
                        reasoning.append(f"Pin bar with {pattern_details['body_to_range_ratio']:.2f} body/range ratio")
                    elif pattern == 'Engulfing':
                        reasoning.append(f"Engulfing pattern {pattern_details['engulfing_ratio']:.2f}x previous candle")
                    elif pattern == 'Inside Bar':
                        reasoning.append(f"Inside bar with containment ratio {pattern_details['containment_ratio']:.2f}")
                    elif pattern == 'False Breakout':
                        reasoning.append(f"False breakout with reversal ratio {pattern_details['reversal_ratio']:.2f}")
                    
                    if fib_ok:
                        reasoning.append(f"Fibonacci confluence: {fib_details['fib_level']:.3f} retracement level")
                    if ma_ok:
                        reasoning.append(f"MA{self.ma_period} confluence: price near moving average")
                        
                    reasoning.append(f"Risk-reward ratio: {risk_reward_ratio:.2f} (min required: {self.min_risk_reward:.2f})")
                    
                    # Add volume information if available
                    if volume_details:
                        reasoning.append(f"Volume: {volume_details['volume_ratio']:.2f}x average")
                    
                    # Add signal quality info
                    reasoning.append(f"Signal quality score: {signal_quality:.2f}")
                    
                    # Technical metrics for entry
                    technical_metrics = {}
                    
                    # Check RSI if enough data
                    if len(primary) >= 14:
                        delta = primary['close'].diff()
                        gain = delta.where(delta > 0, 0)
                        loss = -delta.where(delta < 0, 0)
                        avg_gain = gain.rolling(window=14).mean()
                        avg_loss = loss.rolling(window=14).mean()
                        rs = avg_gain / avg_loss
                        rsi = 100 - (100 / (1 + rs))
                        current_rsi = rsi.iloc[-1]
                        technical_metrics['rsi'] = current_rsi
                        reasoning.append(f"RSI: {current_rsi:.1f}")
                    
                    # Calculate ATR if possible 
                    if len(primary) >= 14:
                        atr = calculate_atr(primary, period=14)
                        if isinstance(atr, pd.Series):
                            current_atr = atr.iloc[-1]
                        else:
                            current_atr = atr
                            
                        if current_atr > 0:
                            stop_distance_atr = risk_pips / current_atr
                            technical_metrics['atr'] = current_atr
                            technical_metrics['stop_distance_atr'] = stop_distance_atr
                            reasoning.append(f"Stop distance: {stop_distance_atr:.2f} ATR")
                    
                    signal = {
                        "symbol": sym,
                        "direction": direction,
                        "entry_price": entry,
                        "stop_loss": stop,
                        "take_profit": tp,
                        "size": size,
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
                        "recency_score": recency_score,
                        "description": f"{direction.upper()} signal on {sym} {self.primary_timeframe} ({pattern})",
                        "detailed_reasoning": reasoning
                    }
                    
                    # Log detailed signal info
                    logger.info(f"Generated signal for {sym}: {direction.upper()} at {entry:.5f}, SL: {stop:.5f}, TP: {tp:.5f}, Risk:Reward = 1:{risk_reward_ratio:.2f}")
                    logger.info(f"Reasoning: {' | '.join(reasoning)}")
                    
                    signals.append(signal)
                    break  # one pattern per level
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
        """Return (supports, resistances) based on pivot detection over recent bars"""
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
        # Cluster levels to avoid duplicates
        tol = subset['close'].mean() * self.price_tolerance
        supports = self._cluster_levels(sorted(lows), tol)
        resistances = self._cluster_levels(sorted(highs), tol)
        return supports, resistances
    
    def _cluster_levels(self, levels: list, tol: float) -> list:
        """Cluster nearby price levels within tolerance"""
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
        # Average each cluster
        return [sum(c)/len(c) for c in clusters]

    # -- Pullback detection --
    def _is_pullback(self, df: pd.DataFrame, level: float, direction: str) -> bool:
        """Check if price pulled back to `level` within recent bars"""
        if df is None or len(df) < self.pullback_bars:
            return False
        recent = df['close'].iloc[-self.pullback_bars:]
        tol_val = level * self.price_tolerance
        if direction == 'bullish':
            return recent.max() > level and abs(df['low'].iloc[-1] - level) <= tol_val
        else:
            return recent.min() < level and abs(df['high'].iloc[-1] - level) <= tol_val

    # -- Candlestick pattern checks --
    def _is_pin_bar(self, candle: pd.Series, level: float, direction: str) -> bool:
        """Detect a pin bar touching `level` with a long wick"""
        body = abs(candle['close'] - candle['open'])
        total = candle['high'] - candle['low']
        if total <= 0:
            return False
        tol_val = level * self.price_tolerance
        if direction == 'bullish':
            lower_wick = candle['open'] - candle['low']
            if lower_wick > body * 2 and abs(candle['low'] - level) <= tol_val:
                return True
        else:
            upper_wick = candle['high'] - candle['open']
            if upper_wick > body * 2 and abs(candle['high'] - level) <= tol_val:
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
        """Detect breakout of an inside bar"""
        if idx <= 0 or idx >= len(candles):
            return False
        mother = candles.iloc[idx - 1]
        child = candles.iloc[idx]
        # child inside mother range but then breaks out
        if child['high'] < mother['high'] and child['low'] > mother['low']:
            # breakout candle is child
            return child['close'] > mother['high'] or child['close'] < mother['low']
        return False

    def _is_false_breakout(self, candles: pd.DataFrame, idx: int, level: float, direction: str) -> bool:
        """Detect a quick reversal after a breakout around `level`"""
        if idx <= 0 or idx >= len(candles):
            return False
        prev = candles.iloc[idx - 1]
        curr = candles.iloc[idx]
        tol_val = level * self.price_tolerance
        if direction == 'bullish':
            return prev['close'] < level - tol_val and curr['close'] > level + tol_val
        else:
            return prev['close'] > level + tol_val and curr['close'] < level - tol_val

    # -- Confluence checks --
    def _check_fibonacci(self, df: pd.DataFrame, level: float) -> bool:
        """Return True if `level` is near a standard Fibonacci retracement"""
        high = df['high'].max()
        low = df['low'].min()
        tol_val = high * self.price_tolerance
        for f in self.fib_levels:
            fib_lv = low + (high - low) * f
            if abs(level - fib_lv) <= tol_val:
                return True
        return False

    def _check_ma(self, df: pd.DataFrame, level: float) -> bool:
        """Return True if `level` is near the moving-average support/resistance"""
        ma = df['close'].rolling(self.ma_period).mean().iloc[-1]
        tol_val = ma * self.price_tolerance
        return abs(level - ma) <= tol_val

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