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
import talib # Added TA-Lib import
from src.utils.patterns_luxalgo import add_luxalgo_patterns, BULLISH_PATTERNS, BEARISH_PATTERNS, NEUTRAL_PATTERNS, ALL_PATTERNS, filter_patterns_by_bias

class PriceActionSRStrategy(SignalGenerator):
    """
    Price Action S/R Strategy: Generates signals based on S/R zones, candlestick pattern, wick rejection, and volume spike.
    Parameterized for timeframe and all key thresholds.
    """
 
    def __init__(
        self,
        primary_timeframe: str = "M15",
        secondary_timeframe: Optional[str] = "H1",
        trend_lookback: int = 300,
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
            secondary_timeframe (str): Higher timeframe for trend filter (default H1)
            trend_lookback (int): Number of bars to use for trend detection (default 300)
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
        self.secondary_timeframe = secondary_timeframe
        self.trend_lookback = trend_lookback
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
        logger.info(f"Secondary timeframe: {self.secondary_timeframe}")
        logger.info(f"Trend lookback: {self.trend_lookback}")
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
        Find true fractal/pivot highs using classic pivot point logic.
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
        for i in range(w, len(highs) - w):
            current_high = highs[i]
            is_pivot = True
            for j in range(i - w, i):
                if current_high <= highs[j]:
                    is_pivot = False
                    break
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
        Find true fractal/pivot lows using classic pivot point logic.
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
        for i in range(w, len(lows) - w):
            current_low = lows[i]
            is_pivot = True
            for j in range(i - w, i):
                if current_low >= lows[j]:
                    is_pivot = False
                    break
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

    def get_sr_zones(self, df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
        """
        Compute and return the top N support and resistance zones.
        Dynamically adjust max_zones based on volatility and use ATR-based tolerance for clustering.
        Uses vectorized zone touch counting for performance.
        Args:
            df (pd.DataFrame): Price data with 'high' and 'low' columns
        Returns:
            Dict[str, List[Dict[str, Any]]]: {'support': [{'level': price, 'strength': count}, ...], 'resistance': [...]
        """
        highs = self._find_pivot_highs(df)
        lows = self._find_pivot_lows(df)
        res_zones_prices = self._cluster_levels(highs, df=df)
        sup_zones_prices = self._cluster_levels(lows, df=df)

        # Vectorized count touches for each zone
        def count_touches(prices, zones_prices):
            if not prices or not zones_prices: # Add check for empty lists
                return [0] * len(zones_prices)
            prices_arr = np.array(prices)
            return [self._is_in_zone(prices_arr, z_price).sum() for z_price in zones_prices]
        res_counts = count_touches(highs, res_zones_prices)
        sup_counts = count_touches(lows, sup_zones_prices)

        # Dynamic max_zones based on volatility
        close_std = df['close'].rolling(50).std().iloc[-1] if len(df) >= 50 else df['close'].std()
        close_mean = df['close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else df['close'].mean()
        volatility = close_std / close_mean if close_mean > 0 else 0
        dynamic_max_zones = 4 if volatility > 0.01 else self.max_zones  # 1% threshold, else fallback
        logger.info(f"Volatility: {volatility:.4f}, using max_zones={dynamic_max_zones}")

        # Create list of dicts with level and strength
        res_zone_data = [{'level': price, 'strength': count} for price, count in zip(res_zones_prices, res_counts)]
        sup_zone_data = [{'level': price, 'strength': count} for price, count in zip(sup_zones_prices, sup_counts)]

        # Sort by strength and take top N
        top_res = sorted(res_zone_data, key=lambda x: x['strength'], reverse=True)[:dynamic_max_zones]
        top_sup = sorted(sup_zone_data, key=lambda x: x['strength'], reverse=True)[:dynamic_max_zones]
        
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
        Enhanced adaptive volume spike: Multiple thresholds for more flexible volume validation.
        Primary: 85th percentile, Secondary: 70th percentile, Fallback: 1.5× mean (reduced from 2×).
        """
        if 'vol_85q' not in df.columns or 'vol_rolling_mean' not in df.columns:
            df['vol_85q'] = df['tick_volume'].rolling(40, min_periods=10).quantile(0.85)
            df['vol_70q'] = df['tick_volume'].rolling(40, min_periods=10).quantile(0.70)  # Added: Secondary threshold
            df['vol_rolling_mean'] = df['tick_volume'].rolling(40, min_periods=10).mean()
        if idx < 20:
            return False
        current_vol = df['tick_volume'].iloc[idx]
        threshold_85 = df['vol_85q'].iloc[idx]
        threshold_70 = df.get('vol_70q', pd.Series([np.nan])).iloc[idx] if 'vol_70q' in df.columns else np.nan
        avg_vol = df['vol_rolling_mean'].iloc[idx]
        
        # Multi-tier volume validation
        if not np.isnan(threshold_85) and current_vol >= threshold_85:
            self.logger.debug(f"Volume spike check: current={current_vol}, 85th percentile={threshold_85} - STRONG SPIKE")
            return True
        elif not np.isnan(threshold_70) and current_vol >= threshold_70:
            self.logger.debug(f"Volume spike check: current={current_vol}, 70th percentile={threshold_70} - MODERATE SPIKE")
            return True
        elif not np.isnan(avg_vol) and current_vol >= 1.5 * avg_vol:  # Reduced from 2× to 1.5×
            self.logger.debug(f"Volume spike fallback: current={current_vol}, 1.5x mean={1.5*avg_vol} - BASIC SPIKE")
            return True
        else:
            self.logger.debug(f"Volume spike check: current={current_vol}, 85th percentile={threshold_85}, 70th percentile={threshold_70}, 1.5x mean={1.5*avg_vol if not np.isnan(avg_vol) else 'N/A'} - NO SPIKE")
            return False
    
    def calculate_stop_loss(self, zone: float, direction: str, candle_extremity: float, buffer: float = 0.001) -> float:
        if direction == "buy":
            # SL for buy is below the lower of the zone or the candle's low
            sl = min(zone, candle_extremity) - buffer
            self.logger.debug(f"[SL] BUY: zone={zone}, candle_low={candle_extremity}, buffer={buffer}, SL={sl}")
            return sl
        else: # direction == "sell"
            # SL for sell is above the higher of the zone or the candle's high
            sl = max(zone, candle_extremity) + buffer
            self.logger.debug(f"[SL] SELL: zone={zone}, candle_high={candle_extremity}, buffer={buffer}, SL={sl}")
            return sl

    def _score_signal_01(self, pattern: str, wick: bool, volume_score: float, risk_reward: float, zone_touches: int, other_confluence: float = 0.0) -> Tuple[float, dict]:
        """
        Compute a normalized 0-1 score for a trading signal based on pattern, wick rejection, volume, risk-reward, zone strength, and optional other confluence.

        Args:
            pattern (str): Candlestick pattern name (e.g., 'Hammer (TA-Lib)')
            wick (bool): Whether wick rejection is present (from _wick_rejection method)
            volume_score (float): Volume spike score (0-1, e.g. 1 if strong spike, 0 if not)
            risk_reward (float): Risk-reward ratio (e.g. 2.5 for 2.5:1)
            zone_touches (int): Number of times price has touched/respected the zone
            other_confluence (float): Reserved for future use (0-1)
        Returns:
            (float, dict): Tuple of (score, breakdown dict)
        """
        # Pattern reliability mapping - updated to include LuxAlgo patterns
        pattern_score = 0.0
        pattern_lower = pattern.lower() if pattern else ""
        
        # High-reliability patterns (0.8-1.0)
        if any(p in pattern_lower for p in ['bullish engulfing', 'bearish engulfing']):
            pattern_score = 1.0
        elif any(p in pattern_lower for p in ['hammer', 'shooting star']):
            pattern_score = 0.8
        elif any(p in pattern_lower for p in ['pin bar', 'pin_bar_bullish', 'pin_bar_bearish']):
            pattern_score = 0.8  # Pin bars are strong reversal patterns at S/R levels
        elif any(p in pattern_lower for p in ['morning star', 'evening star']):
            pattern_score = 0.9  # Morning/Evening stars are strong 3-candle patterns
        
        # Medium-reliability patterns (0.5-0.7)
        elif any(p in pattern_lower for p in ['bullish harami', 'bearish harami']):
            pattern_score = 0.6
        elif any(p in pattern_lower for p in ['inverted hammer', 'hanging man']):
            pattern_score = 0.7
        elif any(p in pattern_lower for p in ['white marubozu', 'black marubozu']):
            pattern_score = 0.7
        
        # Lower-reliability patterns (0.3-0.5)
        elif any(p in pattern_lower for p in ['inside bar']):
            pattern_score = 0.4  # Inside bars are more of a consolidation pattern
        
        # If no pattern matched, check for empty/None
        elif not pattern or pattern.strip() == '':
            pattern_score = 0.0
        else:
            # Default score for unrecognized patterns
            pattern_score = 0.3
            self.logger.debug(f"[Scoring] Unrecognized pattern '{pattern}', using default score {pattern_score}")
       
        wick_score = 1.0 if wick else 0.0
        # Volume: allow float for partial spike (e.g. 0.5 if just above threshold)
        volume_score = max(0.0, min(volume_score, 1.0))
        # Risk-reward: Enhanced scoring for better R:R ratios
        if risk_reward < 1.0:
            risk_reward_score = 0.0
        elif risk_reward >= 3.0:
            risk_reward_score = 1.0
        elif risk_reward >= 2.0:
            risk_reward_score = 0.5 + 0.5 * (risk_reward - 2.0)
            risk_reward_score = min(risk_reward_score, 1.0)
        else: # 1.0 <= risk_reward < 2.0
            risk_reward_score = 0.5 * (risk_reward - 1.0)
        risk_reward_score = max(0.0, min(risk_reward_score, 1.0))
        
        # Zone strength: Enhanced scoring for zone touches
        if zone_touches <= 1:
            zone_strength_score = 0.0
        elif zone_touches >= 5:
            zone_strength_score = 1.0
        elif zone_touches >= 3: # 3 or 4 touches
            zone_strength_score = 0.5 + 0.5 * (zone_touches - 3) / 2.0 # Scale from 0.5 to 1.0
        else: # zone_touches == 2
            zone_strength_score = 0.25 # Halfway to 0.5 for 2 touches
        zone_strength_score = max(0.0, min(zone_strength_score, 1.0))
        
        # Other confluence (future)
        other_confluence_score = max(0.0, min(other_confluence, 1.0))

        score = (
            pattern_score * 0.30 +
            wick_score * 0.15 +
            volume_score * 0.15 +
            risk_reward_score * 0.20 +
            zone_strength_score * 0.10 +
            other_confluence_score * 0.10
        )
        breakdown = {
            'pattern_score': pattern_score,
            'wick_score': wick_score,
            'volume_score': volume_score,
            'risk_reward_score': risk_reward_score,
            'zone_strength_score': zone_strength_score,
            'other_confluence_score': other_confluence_score,
            'weights': {
                'pattern': 0.30,
                'wick': 0.15,
                'volume': 0.15,
                'risk_reward': 0.20,
                'zone_strength': 0.10,
                'other_confluence': 0.10
            },
            'final_score': score
        }
        self.logger.debug(f"[Scoring01] pattern='{pattern}'({pattern_score}), wick={wick_score}, volume={volume_score}, risk_reward={risk_reward_score}, zone={zone_strength_score}, other={other_confluence_score} -> score={score:.2f}")
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
            support_str = ", ".join([f"{z['level']:.5f} (strength: {z['strength']})" for z in support_zones])
            logger.info(f"[{symbol}] Support zones: {support_str}")
        else:
            logger.info(f"[{symbol}] No support zones detected")
            
        # Log resistance zones
        if resistance_zones:
            resistance_str = ", ".join([f"{z['level']:.5f} (strength: {z['strength']})" for z in resistance_zones])
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
            "Bullish Harami": 2,
            "Bearish Harami": 2,
            "Morning Star": 1,
            "Evening Star": 1,
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
        tfs = [self.primary_timeframe]
        if self.secondary_timeframe and self.secondary_timeframe != self.primary_timeframe:
            tfs.append(self.secondary_timeframe)
        return tfs

    @property
    def lookback_periods(self):
        periods = {self.primary_timeframe: max(self.pivot_window + 21, self.trend_lookback, 100)}
        if self.secondary_timeframe and self.secondary_timeframe != self.primary_timeframe:
            periods[self.secondary_timeframe] = self.trend_lookback
        return periods

    def is_uptrend(self, df: pd.DataFrame) -> bool:
        """
        Determine if market is in uptrend using robust swing analysis.
        Analyzes sequence of significant swing highs and lows over trend_lookback period.
        """
        lookback = min(self.trend_lookback, len(df))
        if lookback < 2 * self.pivot_window + 1:
            self.logger.debug(f"[Uptrend] Not enough data for swing analysis: need {2 * self.pivot_window + 1}, have {lookback}")
            return False

        df_lookback = df.iloc[-lookback:]
        
        # Find all swing highs and lows with their indices
        swing_highs = []
        swing_lows = []
        w = self.pivot_window
        highs = df_lookback['high'].values
        lows = df_lookback['low'].values
        
        for i in range(w, len(df_lookback) - w):
            is_high = all(highs[i] > highs[j] for j in range(i - w, i)) and all(highs[i] > highs[j] for j in range(i + 1, i + w + 1))
            is_low = all(lows[i] < lows[j] for j in range(i - w, i)) and all(lows[i] < lows[j] for j in range(i + 1, i + w + 1))
            
            if is_high:
                swing_highs.append((i, highs[i]))
            if is_low:
                swing_lows.append((i, lows[i]))

        # Filter for significant swings using ATR
        atr = self._calculate_atr(df_lookback)
        min_dist = max(atr, 0.002 * df_lookback['close'].iloc[-1])  # 0.2% or ATR
        
        def filter_significant(swings):
            filtered = []
            for idx, price in swings:
                if not filtered:
                    filtered.append((idx, price))
                else:
                    last_idx, last_price = filtered[-1]
                    if abs(price - last_price) > min_dist:
                        filtered.append((idx, price))
            return filtered
        
        swing_highs = filter_significant(swing_highs)
        swing_lows = filter_significant(swing_lows)
        
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            self.logger.debug(f"[Uptrend] Insufficient significant swings: {len(swing_highs)} highs, {len(swing_lows)} lows")
            return False

        # Analyze last 2-3 swings for uptrend
        def is_progressive_up(seq):
            if len(seq) < 3:
                return len(seq) >= 2 and seq[-1] > seq[-2]  # At least 2 swings, last higher than previous
            return seq[-1] > seq[-2] and seq[-2] > seq[-3]

        last_highs = [price for idx, price in swing_highs[-3:]]
        last_lows = [price for idx, price in swing_lows[-3:]]
        
        is_higher_high = is_progressive_up(last_highs)
        is_higher_low = is_progressive_up(last_lows)
        
        uptrend = is_higher_high and is_higher_low
        
        self.logger.debug(f"[Uptrend] last_highs={last_highs}, last_lows={last_lows}, is_higher_high={is_higher_high}, is_higher_low={is_higher_low}, uptrend={uptrend}")
        return uptrend

    def is_downtrend(self, df: pd.DataFrame) -> bool:        
        """
        Determine if market is in downtrend using robust swing analysis.
        Analyzes sequence of significant swing highs and lows over trend_lookback period.
        """
        lookback = min(self.trend_lookback, len(df))
        if lookback < 2 * self.pivot_window + 1:
            self.logger.debug(f"[Downtrend] Not enough data for swing analysis: need {2 * self.pivot_window + 1}, have {lookback}")
            return False

        df_lookback = df.iloc[-lookback:]
        
        # Find all swing highs and lows with their indices
        swing_highs = []
        swing_lows = []
        w = self.pivot_window
        highs = df_lookback['high'].values
        lows = df_lookback['low'].values
        
        for i in range(w, len(df_lookback) - w):
            is_high = all(highs[i] > highs[j] for j in range(i - w, i)) and all(highs[i] > highs[j] for j in range(i + 1, i + w + 1))
            is_low = all(lows[i] < lows[j] for j in range(i - w, i)) and all(lows[i] < lows[j] for j in range(i + 1, i + w + 1))
            
            if is_high:
                swing_highs.append((i, highs[i]))
            if is_low:
                swing_lows.append((i, lows[i]))

        # Filter for significant swings using ATR
        atr = self._calculate_atr(df_lookback)
        min_dist = max(atr, 0.002 * df_lookback['close'].iloc[-1])  # 0.2% or ATR
        
        def filter_significant(swings):
            filtered = []
            for idx, price in swings:
                if not filtered:
                    filtered.append((idx, price))
                else:
                    last_idx, last_price = filtered[-1]
                    if abs(price - last_price) > min_dist:
                        filtered.append((idx, price))
            return filtered
        
        swing_highs = filter_significant(swing_highs)
        swing_lows = filter_significant(swing_lows)
        
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            self.logger.debug(f"[Downtrend] Insufficient significant swings: {len(swing_highs)} highs, {len(swing_lows)} lows")
            return False

        # Analyze last 2-3 swings for downtrend
        def is_progressive_down(seq):
            if len(seq) < 3:
                return len(seq) >= 2 and seq[-1] < seq[-2]  # At least 2 swings, last lower than previous
            return seq[-1] < seq[-2] and seq[-2] < seq[-3]

        last_highs = [price for idx, price in swing_highs[-3:]]
        last_lows = [price for idx, price in swing_lows[-3:]]
        
        is_lower_high = is_progressive_down(last_highs)
        is_lower_low = is_progressive_down(last_lows)
        
        downtrend = is_lower_high and is_lower_low
        
        self.logger.debug(f"[Downtrend] last_highs={last_highs}, last_lows={last_lows}, is_lower_high={is_lower_high}, is_lower_low={is_lower_low}, downtrend={downtrend}")
        return downtrend

    async def generate_signals(self, market_data: Dict[str, Any], symbol: Optional[str] = None, **kwargs) -> List[Dict]:
        """
        Generate trading signals based on S/R zone entry, candlestick pattern, wick rejection, and volume spike.
        Uses a normalized 0-1 scoring system for signal quality and confidence.
        Volume confirmation uses _volume_spike (quantile-based, adaptive) as the primary filter.
        """
        logger.debug(f"[StrategyInit] {self.__class__.__name__}: required_timeframes={self.required_timeframes}, lookback_periods={self.lookback_periods}")
        signals = []
        current_time = datetime.now().timestamp()
        symbols = [symbol] if symbol else list(market_data.keys())

        for sym in symbols:
            logger.info(f"Analyzing {sym} with {self.name}")
            df_data = market_data[sym]
            if isinstance(df_data, dict):
                df_primary = df_data.get(self.primary_timeframe)
                df_secondary = df_data.get(self.secondary_timeframe) if self.secondary_timeframe else None
            else:
                df_primary = df_data
                df_secondary = None
            if df_primary is None or len(df_primary) < self.pivot_window + 21:
                logger.warning(f"Insufficient data for {sym}: Need at least {self.pivot_window + 21} bars, got {len(df_primary) if df_primary is not None else 0}")
                continue
            ht_trend_ok = True
            if self.secondary_timeframe and self.secondary_timeframe != self.primary_timeframe:
                if df_secondary is None or len(df_secondary) < self.trend_lookback:
                    logger.warning(f"Insufficient data for {sym} secondary timeframe {self.secondary_timeframe}: Need {self.trend_lookback} bars, got {len(df_secondary) if df_secondary is not None else 0}")
                    ht_trend_ok = False
                else:
                    uptrend_ht = self.is_uptrend(df_secondary)
                    downtrend_ht = self.is_downtrend(df_secondary)
            else:
                uptrend_ht = downtrend_ht = True

            bar_key = (sym, self.primary_timeframe)
            last_timestamp = None

            try:
                last_timestamp = pd.to_datetime(df_primary.index[-1])
                last_timestamp_str = str(last_timestamp)
            except Exception as e:
                logger.warning(f"Could not extract timestamp from dataframe for {sym}: {e}")
                last_timestamp_str = str(current_time)

            if bar_key in self.processed_bars and self.processed_bars[bar_key] == last_timestamp_str:
                logger.debug(f"Already processed latest bar for {sym}/{self.primary_timeframe} at {last_timestamp_str}")
                continue

            self.processed_bars[bar_key] = last_timestamp_str
            logger.debug(f"Processing new bar for {sym}/{self.primary_timeframe} at {last_timestamp_str}")

            df_primary = df_primary.copy()
            if 'tick_volume' not in df_primary.columns:
                df_primary['tick_volume'] = df_primary.get('volume', 1)

            df_primary = add_luxalgo_patterns(df_primary.copy())

            zones = self.get_sr_zones(df_primary)
            symbol_signals = []

            for direction, zone_list in [('buy', zones['support']), ('sell', zones['resistance'])]:
                zone_type = 'support' if direction == 'buy' else 'resistance'
                if not ht_trend_ok:
                    self.logger.debug(f"[HTF Trend Filter] Skipping {direction} signal for {sym} due to insufficient secondary timeframe data.")
                    continue
                if direction == 'buy' and not uptrend_ht:
                    self.logger.debug(f"[HTF Trend Filter] Skipping BUY signal for {sym}: HTF trend is not UP.")
                    continue
                if direction == 'sell' and not downtrend_ht:
                    self.logger.debug(f"[HTF Trend Filter] Skipping SELL signal for {sym}: HTF trend is not DOWN.")
                    continue
                logger.debug(f"Checking {len(zone_list)} {zone_type} zones for {sym}")

                for zone_price_map in zone_list:
                    zone = zone_price_map['level']
                    zone_touches = zone_price_map['strength']

                    zone_key = (sym, zone_type, round(zone, 5))
                    if zone_key in self.processed_zones:
                        last_used_time = self.processed_zones[zone_key]
                        time_since_use = current_time - last_used_time
                        if time_since_use < self.signal_cooldown:
                            cooldown_remaining = self.signal_cooldown - time_since_use
                            hours_remaining = cooldown_remaining / 3600
                            logger.debug(f"Skipping {zone_type} zone {zone:.5f} - on cooldown for {hours_remaining:.1f} more hours")
                            continue
                    idx = len(df_primary) - 1
                    process_idx = idx - 1
                    if process_idx < 0 or process_idx < self.pivot_window + 20:
                        logger.debug(f"Not enough bars to check for patterns at index {process_idx}, need at least {self.pivot_window + 20}")
                        continue
                    candle = df_primary.iloc[process_idx]
                    in_zone = self._bar_touches_zone(candle, zone, direction)

                    matched_pattern_name = None
                    patterns_to_check = []
                    if direction == 'buy':
                        patterns_to_check = BULLISH_PATTERNS + NEUTRAL_PATTERNS
                    elif direction == 'sell':
                        patterns_to_check = BEARISH_PATTERNS + NEUTRAL_PATTERNS

                    for p_col in patterns_to_check:
                        if p_col in df_primary.columns and df_primary[p_col].iloc[process_idx]:
                            matched_pattern_name = p_col
                            break

                    pattern_display_name = f"{matched_pattern_name.replace('_', ' ').title()} (LuxAlgo)" if matched_pattern_name else None

                    wick = self._wick_rejection(candle, direction)
                    vol_spike = self._volume_spike(df_primary, process_idx) # Primary volume confirmation
                    # If you want to add wick/body analysis as a secondary filter after a volume spike, do it here (currently not used)
                    # Example: if vol_spike and self.is_valid_volume_spike(df_primary): ...

                    logger.debug(f"[Filter] {sym} idx={process_idx} direction={direction} in_zone={in_zone} pattern={pattern_display_name} wick={wick} vol_spike={vol_spike}")

                    if not in_zone:
                        logger.debug(f"[Skip] {sym} {direction} at zone {zone:.5f}: not in zone.")
                        continue
                    if not matched_pattern_name:
                        logger.debug(f"[Skip] {sym} {direction} at zone {zone:.5f}: no valid pattern.")
                        continue
                    if not vol_spike:
                        logger.debug(f"[Skip] {sym} {direction} at zone {zone:.5f}: volume_spike confirmation failed.")
                        continue

                    avg_vol = df_primary['tick_volume'].iloc[max(0, process_idx-20):process_idx].mean() if process_idx > 0 else 0
                    current_vol = df_primary['tick_volume'].iloc[process_idx] if process_idx < len(df_primary) else 0
                    volume_score = min((current_vol / avg_vol) / self.volume_multiplier, 1.0) if avg_vol > 0 else 0.0

                    entry = df_primary['open'].iloc[idx] if idx < len(df_primary) else candle['close']

                    candle_low = candle['low']
                    candle_high = candle['high']
                    stop = self.calculate_stop_loss(
                        zone, 
                        direction, 
                        candle_low if direction == 'buy' else candle_high
                    )

                    opp_zones_map = zones['resistance'] if direction == 'buy' else zones['support']
                    opp_zones_levels = [z['level'] for z in opp_zones_map]
                    tp = None
                    for opp_level in (sorted(opp_zones_levels) if direction == 'buy' else sorted(opp_zones_levels, reverse=True)):
                        if (direction == 'buy' and opp_level > entry) or (direction == 'sell' and opp_level < entry):
                            tp = opp_level
                            break
                    if tp is None:
                        risk = abs(entry - stop)
                        if risk == 0:
                            logger.warning(f"Risk is zero for {sym} {direction} at {entry}, SL {stop}. Skipping TP calc.")
                            continue
                        tp = entry + 2 * risk if direction == 'buy' else entry - 2 * risk
                        logger.debug(f"Using fallback 2R TP: {tp:.5f}")

                    equity = self.risk_manager.get_account_balance() if hasattr(self.risk_manager, 'get_account_balance') else 10000

                    if ((direction == 'buy' and entry <= stop) or \
                       (direction == 'sell' and entry >= stop) or \
                       abs(entry - stop) == 0):
                        logger.warning(f"Invalid entry/stop for position sizing: {sym} {direction}, entry={entry}, stop={stop}. Skipping signal.")
                        continue

                    size = (equity * self.risk_per_trade) / abs(entry - stop)
                    risk_reward = abs(tp - entry) / abs(entry - stop) if abs(entry - stop) > 0 else 0

                    score_01, score_breakdown = self._score_signal_01(
                        pattern=pattern_display_name or "",
                        wick=wick,
                        volume_score=volume_score,
                        risk_reward=risk_reward,
                        zone_touches=zone_touches,
                        other_confluence=0.0
                    )
                    confidence = min(max(score_01, 0.1), 0.95)
                    reason = (
                        f"Confluence: pattern={pattern_display_name}, wick={wick}, volume_spike={vol_spike} at {zone_type} zone {zone:.5f}. "
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
                        'pattern': pattern_display_name if pattern_display_name else '',
                        'zone': float(zone),
                        'zone_touches': zone_touches,
                        'volume_score': volume_score,
                        'bar_index': int(process_idx),
                        'risk_reward': float(risk_reward),
                        'signal_timestamp': str(df_primary.index[process_idx]),
                        'strategy_name': self.name,
                        'score_01': score_01,
                        'score_breakdown': score_breakdown
                    }
                    logger.info(f"Signal generated for {sym} {direction.upper()}: {reason}")
                    logger.debug(f"Score breakdown: {score_breakdown}")
                    self.processed_zones[zone_key] = current_time
                    symbol_signals.append(signal)
                    break # Process one signal per zone
            if len(symbol_signals) > 1:
                all_signals = symbol_signals.copy()
                symbol_signals = self._prioritize_signals(symbol_signals)
                logger.info(f"Prioritized {len(all_signals)} signals down to {len(symbol_signals)} for {sym}")
            signals.extend(symbol_signals)
            self._log_debug_info(sym, df_primary, zones, symbol_signals)
        cleanup_time = current_time - (self.signal_cooldown * 2)
        old_keys = [k for k, v in self.processed_zones.items() if v < cleanup_time]
        for k in old_keys:
            del self.processed_zones[k]
        if old_keys:
            logger.debug(f"Cleaned up {len(old_keys)} old zone records")
        return signals