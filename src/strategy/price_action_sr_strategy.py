"""
Price Action Support/Resistance Strategy

A fully rules-based price-action strategy using S/R zones, candlestick confirmation, wick rejection, and volume spikes.
Implements the core logic from Chapters 3–6 of the referenced PDF, with simple risk management.

Features:
- Pivot-based S/R zone detection (parameterized)
- Candlestick pattern and wick rejection confirmation
- Volume spike filter
- Parameterized for M15 by default, but supports M5/H1
- Signal output compatible with signal_processor.py
"""

import pandas as pd
import numpy as np
from loguru import logger
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from src.trading_bot import SignalGenerator
from src.risk_manager import RiskManager

class PriceActionSRStrategy(SignalGenerator):
    """
    Price Action S/R Strategy: Generates signals based on S/R zones, candlestick pattern, wick rejection, and volume spike.
    Parameterized for timeframe and all key thresholds.
    """
 
    def __init__(
        self,
        primary_timeframe: str = "M15",
        pivot_window: int = 10,
        wick_threshold: float = 0.5,
        volume_multiplier: float = 1.3,
        max_zones: int = 2,
        risk_per_trade: float = 0.01,
        **kwargs
    ):
        """
        Initialize the PriceActionSRStrategy.

        Args:
            primary_timeframe (str): Timeframe for analysis (default M15)
            pivot_window (int): Window for pivot high/low detection
            wick_threshold (float): Wick % threshold for rejection
            volume_multiplier (float): Volume spike threshold
            max_zones (int): Number of S/R zones to keep
            risk_per_trade (float): Fraction of equity to risk per trade
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.logger = logger
        self.name = "PriceActionSRStrategy"
        self.description = "Rules-based price action S/R strategy"
        self.version = "1.0.0"
        self.primary_timeframe = primary_timeframe
        self.pivot_window = pivot_window
        self.wick_threshold = wick_threshold
        self.volume_multiplier = volume_multiplier
        self.max_zones = max_zones
        self.risk_per_trade = risk_per_trade
        self.risk_manager = RiskManager.get_instance() if hasattr(RiskManager, 'get_instance') else RiskManager()
        
        # State tracking to prevent signal duplication
        self.processed_bars = {}  # {(symbol, timeframe): last_processed_timestamp}
        self.processed_zones = {}  # {(symbol, zone_type, zone_price): last_processed_timestamp}
        self.signal_cooldown = 86400  # 24 hours in seconds - don't reuse the same zone for this duration
        
        # For future extension: allow parameterization for M5/H1
        self.params = kwargs

    async def initialize(self):
        """
        Initialize the strategy with any necessary setup.
        Called by TradingBot during startup.
        """
        logger.info(f"Initializing {self.name} v{self.version}")
        
        # Set up required timeframes (allow primary or parameters to override)
        timeframe_map = {
            "M5": 5,     # 5-minute bars
            "M15": 15,   # 15-minute bars
            "H1": 60     # 1-hour bars
        }
        
        # Log important parameters
        logger.info(f"Primary timeframe: {self.primary_timeframe}")
        logger.info(f"Pivot window: {self.pivot_window}")
        logger.info(f"Wick threshold: {self.wick_threshold}")
        logger.info(f"Volume multiplier: {self.volume_multiplier}")
        
        # Configure internal parameters based on timeframe
        if self.primary_timeframe == "M5":
            # For M5, use a larger pivot window (30 bars) to approximate the same lookback depth
            self.pivot_window = self.params.get("pivot_window_m5", 30)
            # Higher volume multiplier for faster timeframe
            self.volume_multiplier = self.params.get("volume_multiplier_m5", 1.2)
        elif self.primary_timeframe == "H1":
            # For H1, use smaller pivot window (5 bars)
            self.pivot_window = self.params.get("pivot_window_h1", 5)
            # Smaller wick threshold for H1
            self.wick_threshold = self.params.get("wick_threshold_h1", 0.4)
        
        logger.info(f"{self.name} initialization complete")
        return True

    def _find_pivot_highs(self, df: pd.DataFrame) -> List[float]:
        """
        Find pivot highs over the configured window.
        Args:
            df (pd.DataFrame): Price data with 'high' column
        Returns:
            List[float]: List of pivot high prices
        """
        pivots = []
        w = self.pivot_window
        for i in range(w, len(df) - w):
            window = df['high'].iloc[i-w:i+w+1]
            if df['high'].iloc[i] == window.max():
                pivots.append(df['high'].iloc[i])
        return pivots

    def _find_pivot_lows(self, df: pd.DataFrame) -> List[float]:
        """
        Find pivot lows over the configured window.
        Args:
            df (pd.DataFrame): Price data with 'low' column
        Returns:
            List[float]: List of pivot low prices
        """
        pivots = []
        w = self.pivot_window
        for i in range(w, len(df) - w):
            window = df['low'].iloc[i-w:i+w+1]
            if df['low'].iloc[i] == window.min():
                pivots.append(df['low'].iloc[i])
        return pivots

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate the Average True Range (ATR) over the given period.
        Args:
            df (pd.DataFrame): Price data with 'high', 'low', 'close'
            period (int): ATR period (default 14)
        Returns:
            float: ATR value
        """
        if len(df) < period + 1:
            return 0.0
        high = df['high']
        low = df['low']
        close = df['close']
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]
        return float(atr) if not np.isnan(atr) else 0.0

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

    def _is_in_zone(self, price, zone: float, tol: float = 0.003):
        """
        Vectorized: Check if price(s) is/are inside a zone (±tol).
        Args:
            price: float, pd.Series, or np.ndarray
            zone: float
            tol: float
        Returns:
            bool or np.ndarray: True/False or boolean mask
        """
        if isinstance(price, (np.ndarray, pd.Series)):
            return np.abs(price - zone) <= zone * tol
        return abs(price - zone) <= zone * tol

    def get_sr_zones(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """
        Compute and return the top N support and resistance zones.
        Dynamically adjust max_zones based on volatility and use ATR-based tolerance for clustering.
        Uses vectorized zone touch counting for performance.
        Args:
            df (pd.DataFrame): Price data with 'high' and 'low' columns
        Returns:
            Dict[str, List[float]]: {'support': [..], 'resistance': [..]}
        """
        highs = self._find_pivot_highs(df)
        lows = self._find_pivot_lows(df)
        res_zones = self._cluster_levels(highs, df=df)
        sup_zones = self._cluster_levels(lows, df=df)
        # Vectorized count touches for each zone
        def count_touches(prices, zones):
            prices_arr = np.array(prices)
            return [self._is_in_zone(prices_arr, z).sum() for z in zones]
        res_counts = count_touches(highs, res_zones)
        sup_counts = count_touches(lows, sup_zones)
        # Dynamic max_zones based on volatility
        close_std = df['close'].rolling(50).std().iloc[-1] if len(df) >= 50 else df['close'].std()
        close_mean = df['close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else df['close'].mean()
        volatility = close_std / close_mean if close_mean > 0 else 0
        dynamic_max_zones = 4 if volatility > 0.01 else self.max_zones  # 1% threshold, else fallback
        logger.info(f"Volatility: {volatility:.4f}, using max_zones={dynamic_max_zones}")
        top_res = [z for _, z in sorted(zip(res_counts, res_zones), reverse=True)[:dynamic_max_zones]]
        top_sup = [z for _, z in sorted(zip(sup_counts, sup_zones), reverse=True)[:dynamic_max_zones]]
        return {'support': top_sup, 'resistance': top_res}

    def _bar_touches_zone(self, candle, zone: float, direction: str, tol: float = 0.003) -> bool:
        """
        Check if the candle's close OR (low for support, high for resistance) touches the zone.
        Args:
            candle: pd.Series with open, high, low, close
            zone: float, zone price
            direction: 'buy' or 'sell'
            tol: float, tolerance
        Returns:
            bool: True if candle touches zone
        """
        close_in_zone = bool(self._is_in_zone(candle['close'], zone, tol))
        if direction == 'buy':
            low_in_zone = bool(self._is_in_zone(candle['low'], zone, tol))
            return close_in_zone or low_in_zone
        else:
            high_in_zone = bool(self._is_in_zone(candle['high'], zone, tol))
            return close_in_zone or high_in_zone

    def _is_bullish_engulfing(self, prev, curr) -> bool:
        return (
            prev['close'] < prev['open'] and
            curr['close'] > curr['open'] and
            curr['close'] > prev['open'] and
            curr['open'] < prev['close']
        )

    def _is_bearish_engulfing(self, prev, curr) -> bool:
        return (
            prev['close'] > prev['open'] and
            curr['close'] < curr['open'] and
            curr['open'] > prev['close'] and
            curr['close'] < prev['open']
        )

    def _is_hammer(self, candle) -> bool:
        body = abs(candle['close'] - candle['open'])
        total = candle['high'] - candle['low']
        lower_wick = min(candle['open'], candle['close']) - candle['low']
        upper_wick = candle['high'] - max(candle['open'], candle['close'])
        return (
            total > 0 and
            body / total < 0.3 and
            lower_wick > 2 * body and
            upper_wick < body
        )

    def _is_shooting_star(self, candle) -> bool:
        body = abs(candle['close'] - candle['open'])
        total = candle['high'] - candle['low']
        upper_wick = candle['high'] - max(candle['open'], candle['close'])
        lower_wick = min(candle['open'], candle['close']) - candle['low']
        return (
            total > 0 and
            body / total < 0.3 and
            upper_wick > 2 * body and
            lower_wick < body
        )

    def _is_bullish_harami(self, prev, curr) -> bool:
        # Bullish Harami: large bearish candle, followed by small bullish candle inside previous body
        return (
            prev['close'] < prev['open'] and
            curr['close'] > curr['open'] and
            curr['open'] > prev['close'] and curr['close'] < prev['open']
        )

    def _is_bearish_harami(self, prev, curr) -> bool:
        # Bearish Harami: large bullish candle, followed by small bearish candle inside previous body
        return (
            prev['close'] > prev['open'] and
            curr['close'] < curr['open'] and
            curr['open'] < prev['close'] and curr['close'] > prev['open']
        )

    def _is_morning_star(self, df: pd.DataFrame, idx: int) -> bool:
        # Morning Star: bearish candle, small body (gap down), bullish candle closing into first
        if idx < 2:
            return False
        c1 = df.iloc[idx-2]
        c2 = df.iloc[idx-1]
        c3 = df.iloc[idx]
        return (
            c1['close'] < c1['open'] and
            abs(c2['close'] - c2['open']) < 0.5 * abs(c1['close'] - c1['open']) and
            c3['close'] > c3['open'] and
            c3['close'] > ((c1['open'] + c1['close']) / 2)
        )

    def _is_evening_star(self, df: pd.DataFrame, idx: int) -> bool:
        # Evening Star: bullish candle, small body (gap up), bearish candle closing into first
        if idx < 2:
            return False
        c1 = df.iloc[idx-2]
        c2 = df.iloc[idx-1]
        c3 = df.iloc[idx]
        return (
            c1['close'] > c1['open'] and
            abs(c2['close'] - c2['open']) < 0.5 * abs(c1['close'] - c1['open']) and
            c3['close'] < c3['open'] and
            c3['close'] < ((c1['open'] + c1['close']) / 2)
        )

    def _wick_rejection(self, candle, direction: str) -> bool:
        total = candle['high'] - candle['low']
        if total == 0:
            return False
        if direction == 'buy':
            lower_wick = min(candle['open'], candle['close']) - candle['low']
            return lower_wick / total >= self.wick_threshold
        else:
            upper_wick = candle['high'] - max(candle['open'], candle['close'])
            return upper_wick / total >= self.wick_threshold

    def _volume_spike(self, df: pd.DataFrame, idx: int) -> bool:
        """
        Adaptive volume spike: True if current volume is above 85th percentile of last 40 bars (or 2x mean if not enough data).
        Uses rolling quantile and mean columns for speed.
        """
        if 'vol_85q' not in df.columns or 'vol_rolling_mean' not in df.columns:
            df['vol_85q'] = df['tick_volume'].rolling(40, min_periods=10).quantile(0.85)
            df['vol_rolling_mean'] = df['tick_volume'].rolling(40, min_periods=10).mean()
        if idx < 20:
            return False
        current_vol = df['tick_volume'].iloc[idx]
        threshold = df['vol_85q'].iloc[idx]
        avg_vol = df['vol_rolling_mean'].iloc[idx]
        if not np.isnan(threshold):
            self.logger.debug(f"Volume spike check: current={current_vol}, 85th percentile={threshold}")
            return current_vol >= threshold
        else:
            self.logger.debug(f"Volume spike fallback: current={current_vol}, 2x mean={2*avg_vol}")
            return current_vol >= 2 * avg_vol

    def is_valid_volume_spike(self, candle, volume_spike: bool) -> bool:
        if not volume_spike:
            return False
        body = abs(candle['close'] - candle['open'])
        total_range = candle['high'] - candle['low']
        if total_range == 0:
            self.logger.debug("[VolumeWick] Total range is zero, cannot calculate wick ratios")
            return False
        upper_wick = candle['high'] - max(candle['open'], candle['close'])
        lower_wick = min(candle['open'], candle['close']) - candle['low']
        upper_wick_ratio = upper_wick / total_range
        lower_wick_ratio = lower_wick / total_range
        # Relaxed thresholds
        is_bullish_volume = upper_wick_ratio < 0.6
        is_bearish_volume = lower_wick_ratio < 0.6
        combined_valid = body > 0 and ((upper_wick + lower_wick) / body < 2.0)
        self.logger.debug(f"[VolumeWick] Stats: upper_ratio={upper_wick_ratio:.2f}, lower_ratio={lower_wick_ratio:.2f}, body={body}")
        self.logger.debug(f"[VolumeWick] Analysis: bullish={is_bullish_volume}, bearish={is_bearish_volume}, combined_valid={combined_valid}")
        return is_bullish_volume or is_bearish_volume or combined_valid

    def calculate_stop_loss(self, zone: float, direction: str, buffer: float = 0.001) -> float:
        if direction == "buy":
            sl = zone - buffer
            self.logger.debug(f"[SL] BUY: zone={zone}, buffer={buffer}, SL={sl}")
            return sl
        else:
            sl = zone + buffer
            self.logger.debug(f"[SL] SELL: zone={zone}, buffer={buffer}, SL={sl}")
            return sl

    def _pattern_match(self, df: pd.DataFrame, idx: int, direction: str) -> Optional[str]:
        prev = df.iloc[idx-1]
        curr = df.iloc[idx]
        if direction == 'buy':
            if self._is_bullish_engulfing(prev, curr):
                return 'Bullish Engulfing'
            if self._is_hammer(curr):
                return 'Hammer'
            if self._is_bullish_harami(prev, curr):
                return 'Bullish Harami'
            if self._is_morning_star(df, idx):
                return 'Morning Star'
            # Pin-bar: hammer with long lower wick
            if self._wick_rejection(curr, 'buy') and abs(curr['close'] - curr['open']) < (curr['high'] - curr['low']) * 0.3:
                return 'Pin-bar'
        else:
            if self._is_bearish_engulfing(prev, curr):
                return 'Bearish Engulfing'
            if self._is_shooting_star(curr):
                return 'Shooting Star'
            if self._is_bearish_harami(prev, curr):
                return 'Bearish Harami'
            if self._is_evening_star(df, idx):
                return 'Evening Star'
            # Pin-bar: shooting star with long upper wick
            if self._wick_rejection(curr, 'sell') and abs(curr['close'] - curr['open']) < (curr['high'] - curr['low']) * 0.3:
                return 'Pin-bar'
        return None

    def _score_signal_01(self, pattern: str, wick: bool, volume_score: float, risk_reward: float, zone_touches: int, other_confluence: float = 0.0) -> Tuple[float, dict]:
        """
        Compute a normalized 0-1 score for a trading signal based on pattern, wick rejection, volume, risk-reward, zone strength, and optional other confluence.

        Args:
            pattern (str): Candlestick pattern name
            wick (bool): Whether wick rejection is present
            volume_score (float): Volume spike score (0-1, e.g. 1 if strong spike, 0 if not)
            risk_reward (float): Risk-reward ratio (e.g. 2.5 for 2.5:1)
            zone_touches (int): Number of times price has touched/respected the zone
            other_confluence (float): Reserved for future use (0-1)
        Returns:
            (float, dict): Tuple of (score, breakdown dict)
        """
        # Pattern reliability mapping
        pattern_map = {
            'Bullish Engulfing': 1.0,
            'Bearish Engulfing': 1.0,
            'Hammer': 0.7,
            'Shooting Star': 0.7,
            'Pin-bar': 0.5,
            '': 0.0,
            None: 0.0
        }
        pattern_score = pattern_map.get(pattern, 0.0)
        wick_score = 1.0 if wick else 0.0
        # Volume: allow float for partial spike (e.g. 0.5 if just above threshold)
        volume_score = max(0.0, min(volume_score, 1.0))
        # Risk-reward: 0 if <1, 0.5 at 2:1, 1.0 at 3:1 or higher
        if risk_reward < 1.0:
            risk_reward_score = 0.0
        elif risk_reward >= 3.0:
            risk_reward_score = 1.0
        elif risk_reward >= 2.0:
            risk_reward_score = 0.5 + 0.5 * (risk_reward - 2.0)
            risk_reward_score = min(risk_reward_score, 1.0)
        else:
            risk_reward_score = 0.5 * (risk_reward - 1.0)
        risk_reward_score = max(0.0, min(risk_reward_score, 1.0))
        # Zone strength: 0 if 1 touch, 0.5 at 3, 1.0 at 5+
        if zone_touches <= 1:
            zone_strength_score = 0.0
        elif zone_touches >= 5:
            zone_strength_score = 1.0
        elif zone_touches >= 3:
            zone_strength_score = 0.5 + 0.5 * (zone_touches - 3) / 2
        else:
            zone_strength_score = 0.5 * (zone_touches - 1) / 2
        zone_strength_score = max(0.0, min(zone_strength_score, 1.0))
        # Other confluence (future)
        other_confluence_score = max(0.0, min(other_confluence, 1.0))
        # Weights
        w_pattern = 0.30
        w_wick = 0.15
        w_volume = 0.15
        w_risk = 0.20
        w_zone = 0.10
        w_other = 0.10
        score = (
            pattern_score * w_pattern +
            wick_score * w_wick +
            volume_score * w_volume +
            risk_reward_score * w_risk +
            zone_strength_score * w_zone +
            other_confluence_score * w_other
        )
        breakdown = {
            'pattern_score': pattern_score,
            'wick_score': wick_score,
            'volume_score': volume_score,
            'risk_reward_score': risk_reward_score,
            'zone_strength_score': zone_strength_score,
            'other_confluence_score': other_confluence_score,
            'weights': {
                'pattern': w_pattern,
                'wick': w_wick,
                'volume': w_volume,
                'risk_reward': w_risk,
                'zone_strength': w_zone,
                'other_confluence': w_other
            },
            'final_score': score
        }
        self.logger.debug(f"[Scoring01] pattern={pattern}({pattern_score}), wick={wick_score}, volume={volume_score}, risk_reward={risk_reward_score}, zone={zone_strength_score}, other={other_confluence_score} -> score={score:.2f}")
        return score, breakdown

    def _log_debug_info(self, symbol: str, df: pd.DataFrame, zones: Dict, signals: List[Dict]) -> None:
        """
        Log detailed debug information about S/R zones and signals.
        
        Args:
            symbol: Trading symbol
            df: Price data
            zones: Support/resistance zones
            signals: Generated signals
        """
        # Log S/R zones
        support_zones = zones.get('support', [])
        resistance_zones = zones.get('resistance', [])
        
        logger.info(f"[{symbol}] Analysis on {len(df)} candles, primary timeframe: {self.primary_timeframe}")
        logger.info(f"[{symbol}] Current price: {df['close'].iloc[-1]}")
        
        # Log support zones
        if support_zones:
            support_str = ", ".join([f"{z:.5f}" for z in support_zones])
            logger.info(f"[{symbol}] Support zones: {support_str}")
        else:
            logger.info(f"[{symbol}] No support zones detected")
            
        # Log resistance zones
        if resistance_zones:
            resistance_str = ", ".join([f"{z:.5f}" for z in resistance_zones])
            logger.info(f"[{symbol}] Resistance zones: {resistance_str}")
        else:
            logger.info(f"[{symbol}] No resistance zones detected")
        
        # Log signals
        if signals:
            for i, signal in enumerate(signals):
                direction = signal.get('direction', 'unknown')
                pattern = signal.get('pattern', 'unknown')
                entry = signal.get('entry_price', 0)
                sl = signal.get('stop_loss', 0)
                tp = signal.get('take_profit', 0)
                rr = abs(tp - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 0
                
                logger.info(f"[{symbol}] Signal #{i+1}: {direction.upper()} - {pattern}")
                logger.info(f"[{symbol}] Entry: {entry:.5f}, SL: {sl:.5f}, TP: {tp:.5f}, R:R = {rr:.2f}")
        else:
            logger.info(f"[{symbol}] No signals generated")

    def _rank_patterns(self, pattern: str) -> int:
        """
        Rank candlestick patterns by reliability.
        Higher number = more reliable pattern.
        
        Args:
            pattern: Candlestick pattern name
            
        Returns:
            int: Pattern rank (1-5)
        """
        pattern_ranks = {
            "Bullish Engulfing": 5,
            "Bearish Engulfing": 5,
            "Hammer": 4,
            "Shooting Star": 4,
            "Pin-bar": 3,
            # Add more patterns with their ranks as needed
        }
        return pattern_ranks.get(pattern, 1)  # Default rank = 1
        
    def _prioritize_signals(self, signals: List[Dict]) -> List[Dict]:
        """
        Prioritize signals when there are conflicts for the same symbol.
        Returns only the highest-priority signal for each symbol.
        Now uses the normalized 0-1 score ('score_01') for ranking, with tie-breakers:
        1. score_01 (higher is better)
        2. risk_reward (higher is better)
        3. bar_index (most recent is better)
        4. pattern reliability (higher is better)
        """
        if not signals:
            return []
        signals_by_symbol = {}
        for signal in signals:
            symbol = signal.get('symbol')
            if symbol not in signals_by_symbol:
                signals_by_symbol[symbol] = []
            signals_by_symbol[symbol].append(signal)
        prioritized_signals = []
        for symbol, symbol_signals in signals_by_symbol.items():
            if len(symbol_signals) == 1:
                prioritized_signals.append(symbol_signals[0])
                continue
            logger.warning(f"Found {len(symbol_signals)} conflicting signals for {symbol}, prioritizing best one")
            # Use multi-level tie-breaker
            def pattern_reliability(s):
                pattern = s.get('pattern', '')
                pattern_map = {
                    'Bullish Engulfing': 1.0,
                    'Bearish Engulfing': 1.0,
                    'Hammer': 0.7,
                    'Shooting Star': 0.7,
                    'Pin-bar': 0.5,
                    '': 0.0,
                    None: 0.0
                }
                return pattern_map.get(pattern, 0.0)
            # Sort by: score_01, risk_reward, bar_index, pattern reliability
            sorted_signals = sorted(
                symbol_signals,
                key=lambda s: (
                    s.get('score_01', 0),
                    s.get('risk_reward', 0),
                    s.get('bar_index', 0),
                    pattern_reliability(s)
                ),
                reverse=True
            )
            # Log tie-breaker if needed
            top_score = sorted_signals[0].get('score_01', 0)
            tied = [s for s in sorted_signals if s.get('score_01', 0) == top_score]
            if len(tied) > 1:
                logger.info(f"Tie on score_01 for {symbol}: {len(tied)} signals. Applying tie-breakers (risk_reward, bar_index, pattern reliability).")
                for s in tied:
                    logger.info(f"  Signal: risk_reward={s.get('risk_reward', 0):.2f}, bar_index={s.get('bar_index', 0)}, pattern={s.get('pattern', '')}")
            best_signal = sorted_signals[0]
            logger.info(f"Selected {best_signal.get('direction')} {best_signal.get('pattern')} as best signal for {symbol} with score_01={best_signal.get('score_01', 0):.2f}")
            prioritized_signals.append(best_signal)
        return prioritized_signals

    @property
    def required_timeframes(self):
        return [self.primary_timeframe]

    @property
    def lookback_periods(self):
        # Use the minimum bars needed for your logic, e.g. pivot_window + 21
        return {self.primary_timeframe: max(self.pivot_window + 21, 100)}

    def is_uptrend(self, df: pd.DataFrame) -> bool:
        if len(df) < 6:
            return False
        higher_high = df['high'].iloc[-1] > df['high'].iloc[-5]
        higher_low = df['low'].iloc[-1] > df['low'].iloc[-5]
        recent_strong_move = df['close'].iloc[-1] > df['close'].iloc[-3]
        self.logger.debug(f"[TrendCheck] Uptrend analysis: higher_high={higher_high}, higher_low={higher_low}, recent_strong_move={recent_strong_move}")
        self.logger.debug(f"[TrendCheck] Last 5 bars: highs={df['high'].iloc[-5:].tolist()}, lows={df['low'].iloc[-5:].tolist()}")
        # Relaxed: pass if any of the three is true
        return higher_high or higher_low or recent_strong_move

    def is_downtrend(self, df: pd.DataFrame) -> bool:
        if len(df) < 6:
            return False
        lower_high = df['high'].iloc[-1] < df['high'].iloc[-5]
        lower_low = df['low'].iloc[-1] < df['low'].iloc[-5]
        recent_strong_move = df['close'].iloc[-1] < df['close'].iloc[-3]
        self.logger.debug(f"[TrendCheck] Downtrend analysis: lower_high={lower_high}, lower_low={lower_low}, recent_strong_move={recent_strong_move}")
        self.logger.debug(f"[TrendCheck] Last 5 bars: highs={df['high'].iloc[-5:].tolist()}, lows={df['low'].iloc[-5:].tolist()}")
        # Relaxed: pass if any of the three is true
        return lower_high or lower_low or recent_strong_move

    async def generate_signals(self, market_data: Dict[str, Any], symbol: Optional[str] = None, **kwargs) -> List[Dict]:
        """
        Generate trading signals based on S/R zone entry, candlestick pattern, wick rejection, and volume spike.
        Uses a normalized 0-1 scoring system for signal quality and confidence.
        """
        logger.debug(f"[StrategyInit] {self.__class__.__name__}: required_timeframes={self.required_timeframes}, lookback_periods={self.lookback_periods}")
        signals = []
        current_time = datetime.now().timestamp()
        symbols = [symbol] if symbol else list(market_data.keys())

        for sym in symbols:
            logger.info(f"Analyzing {sym} with {self.name}")
            df = market_data[sym]
            if isinstance(df, dict):
                df = df.get(self.primary_timeframe)
            if df is None or len(df) < self.pivot_window + 21:
                logger.warning(f"Insufficient data for {sym}: Need at least {self.pivot_window + 21} bars, got {len(df) if df is not None else 0}")
                continue

            bar_key = (sym, self.primary_timeframe)
            last_timestamp = None

            try:
                last_timestamp = pd.to_datetime(df.index[-1])
                last_timestamp_str = str(last_timestamp)
            except:
                logger.warning(f"Could not extract timestamp from dataframe for {sym}")
                last_timestamp_str = str(current_time)

            if bar_key in self.processed_bars and self.processed_bars[bar_key] == last_timestamp_str:
                logger.debug(f"Already processed latest bar for {sym}/{self.primary_timeframe} at {last_timestamp_str}")
                continue

            self.processed_bars[bar_key] = last_timestamp_str
            logger.debug(f"Processing new bar for {sym}/{self.primary_timeframe} at {last_timestamp_str}")

            df = df.copy()
            if 'tick_volume' not in df.columns:
                df['tick_volume'] = df.get('volume', 1)

            zones = self.get_sr_zones(df)
            symbol_signals = []

            for direction, zone_list in [('buy', zones['support']), ('sell', zones['resistance'])]:
                zone_type = 'support' if direction == 'buy' else 'resistance'
                logger.debug(f"Checking {len(zone_list)} {zone_type} zones for {sym}")

                # PDF-aligned trend validation
                if direction == 'buy':
                    uptrend = self.is_uptrend(df)
                    self.logger.debug(f"[Trend] {sym} uptrend={uptrend}")
                    if not uptrend:
                        self.logger.debug(f"[Skip] {sym} buy: not in uptrend.")
                        continue
                else:
                    downtrend = self.is_downtrend(df)
                    self.logger.debug(f"[Trend] {sym} downtrend={downtrend}")
                    if not downtrend:
                        self.logger.debug(f"[Skip] {sym} sell: not in downtrend.")
                        continue

                for zone in zone_list:
                    zone_key = (sym, zone_type, round(zone, 5))
                    if zone_key in self.processed_zones:
                        last_used_time = self.processed_zones[zone_key]
                        time_since_use = current_time - last_used_time
                        if time_since_use < self.signal_cooldown:
                            cooldown_remaining = self.signal_cooldown - time_since_use
                            hours_remaining = cooldown_remaining / 3600
                            logger.debug(f"Skipping {zone_type} zone {zone:.5f} - on cooldown for {hours_remaining:.1f} more hours")
                            continue
                    zone_touches = int(self._is_in_zone(df['close'], zone).sum())
                    idx = len(df) - 1
                    process_idx = idx - 1
                    if process_idx < self.pivot_window + 20:
                        logger.debug(f"Not enough bars to check for patterns at index {process_idx}")
                        continue
                    candle = df.iloc[process_idx]
                    in_zone = self._bar_touches_zone(candle, zone, direction)
                    pattern = self._pattern_match(df, process_idx, direction)
                    wick = self._wick_rejection(candle, direction)
                    vol_spike = self._volume_spike(df, process_idx)
                    vol_wick_ok = self.is_valid_volume_spike(candle, vol_spike)
                    # Granular logging for all filters
                    logger.debug(f"[Filter] {sym} idx={process_idx} direction={direction} in_zone={in_zone} pattern={pattern} wick={wick} vol_spike={vol_spike} vol_wick_ok={vol_wick_ok}")
                    if not in_zone:
                        logger.debug(f"[Skip] {sym} {direction} at zone {zone:.5f}: not in zone.")
                        continue
                    if not pattern:
                        logger.debug(f"[Skip] {sym} {direction} at zone {zone:.5f}: no valid pattern.")
                        continue
                    if not vol_wick_ok:
                        logger.debug(f"[Skip] {sym} {direction} at zone {zone:.5f}: volume-wick confirmation failed.")
                        continue
                    # Calculate volume score (how much above threshold)
                    avg_vol = df['tick_volume'].iloc[process_idx-20:process_idx].mean()
                    current_vol = df['tick_volume'].iloc[process_idx]
                    volume_score = min((current_vol / avg_vol) / self.volume_multiplier, 1.0) if avg_vol > 0 else 0.0
                    entry = df['open'].iloc[idx] if idx < len(df) else candle['close']
                    stop = self.calculate_stop_loss(zone, direction)
                    opp_zones = zones['resistance'] if direction == 'buy' else zones['support']
                    tp = None
                    for opp in (sorted(opp_zones) if direction == 'buy' else sorted(opp_zones, reverse=True)):
                        if (direction == 'buy' and opp > entry) or (direction == 'sell' and opp < entry):
                            tp = opp
                            break
                    if tp is None:
                        risk = abs(entry - stop)
                        tp = entry + 2 * risk if direction == 'buy' else entry - 2 * risk
                        logger.debug(f"Using fallback 2R TP: {tp:.5f}")
                    equity = self.risk_manager.get_account_balance() if hasattr(self.risk_manager, 'get_account_balance') else 10000
                    size = (equity * self.risk_per_trade) / max(abs(entry - stop), 1e-6)
                    risk_reward = abs(tp - entry) / abs(entry - stop) if abs(entry - stop) > 0 else 0
                    # Use new 0-1 scoring system
                    score_01, score_breakdown = self._score_signal_01(
                        pattern=pattern or "",
                        wick=wick,
                        volume_score=volume_score,
                        risk_reward=risk_reward,
                        zone_touches=zone_touches,
                        other_confluence=0.0
                    )
                    # Set confidence directly from score_01, with min/max cap
                    confidence = min(max(score_01, 0.1), 0.95)
                    reason = (
                        f"Confluence: pattern={pattern}, wick={wick}, volume_spike={vol_spike} at {zone_type} zone {zone:.5f}. "
                        f"Risk:Reward = {risk_reward:.2f}, Score01 = {score_01:.2f}"
                    )
                    signal = {
                        'symbol': sym,
                        'direction': direction,
                        'entry_price': float(entry),
                        'stop_loss': float(stop),
                        'take_profit': float(tp),
                        'confidence': confidence,
                        'size': float(size),
                        'timeframe': self.primary_timeframe,
                        'reason': reason,
                        'pattern': pattern if pattern else '',
                        'zone': float(zone),
                        'zone_touches': zone_touches,
                        'volume_score': volume_score,
                        'bar_index': int(process_idx),
                        'risk_reward': float(risk_reward),
                        'signal_timestamp': str(df.index[process_idx]),
                        'source': self.name,
                        'score_01': score_01,
                        'score_breakdown': score_breakdown
                    }
                    logger.info(f"Signal generated for {sym} {direction.upper()}: {reason}")
                    logger.debug(f"Score breakdown: {score_breakdown}")
                    self.processed_zones[zone_key] = current_time
                    symbol_signals.append(signal)
                    break
            if len(symbol_signals) > 1:
                all_signals = symbol_signals.copy()
                symbol_signals = self._prioritize_signals(symbol_signals)
                logger.info(f"Prioritized {len(all_signals)} signals down to {len(symbol_signals)} for {sym}")
            signals.extend(symbol_signals)
            self._log_debug_info(sym, df, zones, symbol_signals)
        cleanup_time = current_time - (self.signal_cooldown * 2)
        old_keys = [k for k, v in self.processed_zones.items() if v < cleanup_time]
        for k in old_keys:
            del self.processed_zones[k]
        if old_keys:
            logger.debug(f"Cleaned up {len(old_keys)} old zone records")
        return signals