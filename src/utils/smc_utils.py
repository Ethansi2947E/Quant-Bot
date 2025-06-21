"""
This work is licensed under a Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
https://creativecommons.org/licenses/by-nc-sa/4.0/
© LuxAlgo

This module provides a Python implementation of the 'Smart Money Concepts [LuxAlgo]'
TradingView indicator.

It is designed to be used as a utility within a trading bot framework, processing
OHLCV data to identify various market structures and patterns, including:
- Swing and Internal Structure (BOS, CHoCH)
- Order Blocks
- Fair Value Gaps
- Equal Highs/Lows
- Premium & Discount Zones
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple
import pandas as pd
import numpy as np
from loguru import logger
import talib
from scipy.signal import find_peaks

# --- Constants ---
BULLISH = 1
BEARISH = -1
FVG_STATE_ACTIVE = "active"
FVG_STATE_PARTIAL = "partial"
FVG_STATE_MITIGATED = "mitigated"
LIQUIDITY_BUYSIDE = 1
LIQUIDITY_SELLSIDE = -1


# --- Data Classes for SMC Components ---

@dataclass
class Pivot:
    """Represents a pivot point in market structure."""
    price: float = 0.0
    bar_index: int = 0
    time: Optional[pd.Timestamp] = None
    crossed: bool = False
    
@dataclass
class Structure:
    """Represents a market structure break (BOS or CHoCH)."""
    pivot: Pivot
    break_bar_index: int
    break_time: pd.Timestamp
    type: str  # 'BOS' or 'CHoCH'
    direction: int  # BULLISH or BEARISH
    originating_ob: Optional['OrderBlock'] = None
    
@dataclass
class OrderBlock:
    """Represents an order block."""
    high: float
    low: float
    open: float
    time: pd.Timestamp
    bar_index: int
    direction: int  # BULLISH or BEARISH
    is_swing: bool
    ob_type: str  # NEW: 'Extreme' or 'Decisional'
    mitigated: bool = False
    
@dataclass
class FairValueGap:
    """Represents a Fair Value Gap (FVG)."""
    high: float
    low: float
    time: pd.Timestamp
    bar_index: int
    direction: int  # BULLISH or BEARISH
    state: str = FVG_STATE_ACTIVE # Can be 'active', 'partial', or 'mitigated'
    
@dataclass
class EqualHighLow:
    """Represents Equal Highs or Lows."""
    price1: float
    time1: pd.Timestamp
    price2: float
    time2: pd.Timestamp
    type: str  # 'EQH' or 'EQL'
    bar_index: int

@dataclass
class Liquidity:
    """Represents a liquidity pool (e.g., equal highs/lows)."""
    price: float
    time: pd.Timestamp
    bar_index: int
    direction: int  # LIQUIDITY_BUYSIDE or LIQUIDITY_SELLSIDE
    num_pivots: int # How many pivots form this pool
    mitigated: bool = False

@dataclass
class BreakerMitigationBlock:
    """Represents a Breaker Block (BB) or Mitigation Block (MB)."""
    high: float
    low: float
    time: pd.Timestamp
    bar_index: int
    direction: int  # BULLISH or BEARISH
    block_type: str # 'BB' or 'MB'
    mitigated: bool = False

@dataclass
class PremiumDiscountZone:
    """Represents a premium, discount, or equilibrium zone."""
    premium_top: float
    premium_bottom: float
    equilibrium_top: float
    equilibrium_bottom: float
    discount_top: float
    discount_bottom: float
    
@dataclass
class StrongWeakHighLow:
    """Labels the last swing high/low as strong or weak."""
    high: Pivot
    high_type: str # 'Strong' or 'Weak'
    low: Pivot
    low_type: str # 'Strong' or 'Weak'

class SmartMoneyConcepts:
    """
    A utility class to detect Smart Money Concepts from OHLCV data.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the SmartMoneyConcepts utility.

        Args:
            config (Optional[Dict]): A configuration dictionary to override default settings.
        """
        self.config = self._get_default_config()
        if config:
            self.config.update(config)
        
        self.df: pd.DataFrame = pd.DataFrame()
        self._reset_state()

    def _get_default_config(self) -> Dict:
        """Returns the default configuration dictionary."""
        return {
            "swing_length": 50,
            "internal_length": 5,
            "equal_high_low_length": 3,
            "equal_high_low_threshold": 0.1,
            "ob_filter_mode": "ATR",  # ATR or RANGE
            "ob_mitigation_mode": "High/Low",  # High/Low or Close
            "fvg_auto_threshold": True,
            "atr_period": 200,
            "internal_filter_confluence": False,
            "decisional_ob_search_length": 10,
            # --- NEW: Result Count Configuration ---
            "swing_ob_count": 2,
            "internal_ob_count": 2,
            "fvg_count": 3,
            "liquidity_count": 2,
            "bb_mb_count": 2
        }

    def _reset_state(self):
        """Resets the internal state of the detector."""
        self.swing_high = Pivot()
        self.swing_low = Pivot()
        self.internal_high = Pivot()
        self.internal_low = Pivot()
        self.swing_trend = BULLISH
        self.internal_trend = BULLISH
        
        # Previous pivots for EQH/EQL detection
        self.prev_swing_high = Pivot()
        self.prev_swing_low = Pivot()
        
        # Trailing extremes for premium/discount zones
        self.trailing_high: Optional[Pivot] = None
        self.trailing_low: Optional[Pivot] = None
        
        self.swing_order_blocks: List[OrderBlock] = []
        self.internal_order_blocks: List[OrderBlock] = []
        self.fair_value_gaps: List[FairValueGap] = []
        self.structures: List[Structure] = []
        self.equal_highs_lows: List[EqualHighLow] = []
        self.liquidity_pools: List[Liquidity] = []
        self.breaker_mitigation_blocks: List[BreakerMitigationBlock] = []

    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Analyzes the given OHLCV data for Smart Money Concepts.

        Args:
            df (pd.DataFrame): DataFrame with columns ['time', 'open', 'high', 'low', 'close', 'volume']
                               The 'time' column should be pandas Timestamps.

        Returns:
            Dict: A dictionary containing lists of detected SMC components.
        """
        if df.empty or len(df) < self.config['swing_length']:
            logger.warning("DataFrame is too short for SMC analysis.")
            return {}

        self._reset_state()
        
        df_copy = df.copy().reset_index()

        # --- FIX for KeyError: 'time' ---
        # After reset_index(), the timestamp column might be named 'index' instead of 'time'.
        # We'll rename it here to ensure compatibility.
        logger.debug(f"DataFrame columns after reset_index: {list(df_copy.columns)}")
        if 'time' not in df_copy.columns:
            if 'index' in df_copy.columns:
                df_copy.rename(columns={'index': 'time'}, inplace=True)
                logger.debug("Renamed 'index' column to 'time'")
            else:
                # If neither 'time' nor 'index' exists, use the first column as time
                first_col = df_copy.columns[0]
                df_copy.rename(columns={first_col: 'time'}, inplace=True)
                logger.debug(f"Renamed '{first_col}' column to 'time'")

        # --- 1. Pre-calculate Indicators & Pivots ---
        if self.config['ob_filter_mode'] == 'ATR':
            if 'atr' not in df_copy.columns:
                df_copy['atr'] = talib.ATR(
                    np.array(df_copy['high']),
                    np.array(df_copy['low']),
                    np.array(df_copy['close']),
                    timeperiod=self.config['atr_period']
                )
            volatility_measure = df_copy['atr']
        else:  # 'RANGE' mode
            true_range_np = talib.TRANGE(
                np.array(df_copy['high']),
                np.array(df_copy['low']),
                np.array(df_copy['close'])
            )
            # Convert numpy array to pandas Series to use .expanding()
            true_range_series = pd.Series(true_range_np)
            # Cumulative mean of true range
            volatility_measure = true_range_series.expanding().mean()

        high_volatility_bar = (df_copy['high'] - df_copy['low']) >= (2 * volatility_measure)
        df_copy['parsed_high'] = np.where(high_volatility_bar, df_copy['low'], df_copy['high'])
        df_copy['parsed_low'] = np.where(high_volatility_bar, df_copy['high'], df_copy['low'])

        swing_pivots = self._find_pivots(df_copy, self.config['swing_length'])
        internal_pivots = self._find_pivots(df_copy, self.config['internal_length'])
        
        if not df_copy['atr'].isnull().all():
            self._find_liquidity(df_copy)

        # --- 2. Process bar-by-bar for stateful analysis ---
        for i in range(len(df_copy)):
            # Order of operations is important
            self._check_ob_mitigation(i, df_copy)
            self._check_fvg_mitigation(i, df_copy)
            self._check_liquidity_mitigation(i, df_copy)
            self._check_bb_mb_mitigation(i, df_copy)
            self._process_bar(i, df_copy, swing_pivots, internal_pivots)
            self._update_trailing_extremes(i, df_copy)
            self._find_fair_value_gaps(i, df_copy, volatility_measure)

        # --- 3. Finalize results ---
        zones = self._get_premium_discount_zones()
        strong_weak_pivots = self._get_strong_weak_highs_lows()
        
        # Filter out mitigated OBs and FVGs for a cleaner final result
        unmitigated_swing_obs = [ob for ob in self.swing_order_blocks if not ob.mitigated]
        unmitigated_internal_obs = [ob for ob in self.internal_order_blocks if not ob.mitigated]
        unmitigated_fvgs = [fvg for fvg in self.fair_value_gaps if fvg.state != FVG_STATE_MITIGATED]
        unmitigated_liquidity = [pool for pool in self.liquidity_pools if not pool.mitigated]
        unmitigated_bb_mb = [b for b in self.breaker_mitigation_blocks if not b.mitigated]

        # --- NEW: Default Filtering Logic ---
        # Sort each list by bar_index in descending order (most recent first)
        unmitigated_swing_obs.sort(key=lambda x: x.bar_index, reverse=True)
        unmitigated_internal_obs.sort(key=lambda x: x.bar_index, reverse=True)
        unmitigated_fvgs.sort(key=lambda x: x.bar_index, reverse=True)
        unmitigated_liquidity.sort(key=lambda x: x.bar_index, reverse=True)
        unmitigated_bb_mb.sort(key=lambda x: x.bar_index, reverse=True)
        
        # Slice the lists to keep only the N most recent items
        swing_obs_to_keep = unmitigated_swing_obs[:self.config['swing_ob_count']]
        internal_obs_to_keep = unmitigated_internal_obs[:self.config['internal_ob_count']]
        fvgs_to_keep = unmitigated_fvgs[:self.config['fvg_count']]
        liquidity_to_keep = unmitigated_liquidity[:self.config['liquidity_count']]
        bb_mb_to_keep = unmitigated_bb_mb[:self.config['bb_mb_count']]

        logger.info(f"SMC analysis completed. Returning {len(swing_obs_to_keep) + len(internal_obs_to_keep)} OBs, "
                    f"{len(fvgs_to_keep)} FVGs, {len(liquidity_to_keep)} liquidity pools.")

        analysis_result = {
            "structures": self.structures,
            "swing_order_blocks": swing_obs_to_keep,
            "internal_order_blocks": internal_obs_to_keep,
            "fair_value_gaps": fvgs_to_keep,
            "equal_highs_lows": self.equal_highs_lows,
            "liquidity_pools": liquidity_to_keep,
            "breaker_mitigation_blocks": bb_mb_to_keep,
            "premium_discount_zones": zones,
            "strong_weak_highs_lows": strong_weak_pivots
        }
        
        return analysis_result
        
    def _find_pivots(self, df: pd.DataFrame, length: int) -> Dict[str, pd.Series]:
        """
        Finds pivot highs and lows based on the lagging PineScript logic.
        PineScript: leg() => newLegHigh = high[size] > ta.highest(size)
        This translates to: at bar `i`, is high[i-size] > max(high[i], ..., high[i-size+1])?
        This is a lagging pivot detector. The pivot occurs at `i-size` and is confirmed at `i`.
        """
        highs = df['high']
        lows = df['low']
        
        # PineScript: ta.highest(length) at bar `i` is max(high[i], high[i-1], ..., high[i-length+1])
        # This is equivalent to a rolling max over `length` bars.
        rolling_max = highs.rolling(window=length).max()
        rolling_min = lows.rolling(window=length).min()

        # A new bullish leg starts (a swing low is confirmed) if the low `length` bars ago
        # was lower than all lows in the subsequent `length` bars.
        is_pivot_low_confirmed = lows.shift(length) < rolling_min

        # A new bearish leg starts (a swing high is confirmed) if the high `length` bars ago
        # was higher than all highs in the subsequent `length` bars.
        is_pivot_high_confirmed = highs.shift(length) > rolling_max

        return {"highs": is_pivot_high_confirmed, "lows": is_pivot_low_confirmed}

    def _process_bar(self, i: int, df: pd.DataFrame, swing_pivots: Dict, internal_pivots: Dict):
        """Processes a single bar for SMC analysis."""
        
        swing_length = self.config['swing_length']
        internal_length = self.config['internal_length']

        # --- 1. Update Pivots & Check for EQH/EQL ---
        # Since pivots are lagging, when a pivot is confirmed at `i`, it occurred at `i-length`.
        # Swing Pivots
        if i >= swing_length and swing_pivots['highs'][i]:
            pivot_idx = i - swing_length
            try:
                price = float(df.iloc[pivot_idx]['high'])
                time_val = pd.Timestamp(df.iloc[pivot_idx]['time'])
                atr_val = float(df.iloc[i]['atr'])
                new_pivot = Pivot(price=price, bar_index=pivot_idx, time=time_val, crossed=False)
                self._check_eqh_eql(new_pivot, self.swing_high, atr_val, 'EQH')
                self.prev_swing_high = self.swing_high
                self.swing_high = new_pivot
                self.trailing_high = new_pivot # Reset the top of the range
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not create pivot at index {pivot_idx} due to data type issue: {e}")

        if i >= swing_length and swing_pivots['lows'][i]:
            pivot_idx = i - swing_length
            try:
                price = float(df.iloc[pivot_idx]['low'])
                time_val = pd.Timestamp(df.iloc[pivot_idx]['time'])
                atr_val = float(df.iloc[i]['atr'])
                new_pivot = Pivot(price=price, bar_index=pivot_idx, time=time_val, crossed=False)
                self._check_eqh_eql(new_pivot, self.swing_low, atr_val, 'EQL')
                self.prev_swing_low = self.swing_low
                self.swing_low = new_pivot
                self.trailing_low = new_pivot # Reset the bottom of the range
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not create pivot at index {pivot_idx} due to data type issue: {e}")

        # Internal Pivots
        if i >= internal_length and internal_pivots['highs'][i]:
            pivot_idx = i - internal_length
            try:
                price = float(df.iloc[pivot_idx]['high'])
                time_val = pd.Timestamp(df.iloc[pivot_idx]['time'])
                self.internal_high = Pivot(price=price, bar_index=pivot_idx, time=time_val, crossed=False)
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not create internal pivot at index {pivot_idx} due to data type issue: {e}")

        if i >= internal_length and internal_pivots['lows'][i]:
            pivot_idx = i - internal_length
            try:
                price = float(df.iloc[pivot_idx]['low'])
                time_val = pd.Timestamp(df.iloc[pivot_idx]['time'])
                self.internal_low = Pivot(price=price, bar_index=pivot_idx, time=time_val, crossed=False)
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not create internal pivot at index {pivot_idx} due to data type issue: {e}")

        # --- 2. Detect Structure Breaks ---
        if i == 0: return # Need previous bar for crossover check
        
        close = df['close'][i]
        
        # Swing Structure (check for body close above/below)
        if self.swing_high.price > 0 and not self.swing_high.crossed and close > self.swing_high.price:
            structure_type = 'CHoCH' if self.swing_trend == BEARISH else 'BOS'
            new_structure = Structure(
                pivot=self.swing_high,
                break_bar_index=i,
                break_time=df['time'][i],
                type=structure_type,
                direction=BULLISH
            )
            # Find the OB and assign it directly
            origin_ob = self._find_and_store_order_block_for_leg(new_structure, df, is_swing=True)
            new_structure.originating_ob = origin_ob
            self.structures.append(new_structure)

            self.swing_high.crossed = True
            self.swing_trend = BULLISH
            self._store_breaker_or_mitigation_block(df, BULLISH)

        if self.swing_low.price > 0 and not self.swing_low.crossed and close < self.swing_low.price:
            structure_type = 'CHoCH' if self.swing_trend == BULLISH else 'BOS'
            new_structure = Structure(
                pivot=self.swing_low,
                break_bar_index=i,
                break_time=df['time'][i],
                type=structure_type,
                direction=BEARISH
            )
            origin_ob = self._find_and_store_order_block_for_leg(new_structure, df, is_swing=True)
            new_structure.originating_ob = origin_ob
            self.structures.append(new_structure)

            self.swing_low.crossed = True
            self.swing_trend = BEARISH
            self._store_breaker_or_mitigation_block(df, BEARISH)

        # Internal Structure (check for body close above/below)
        if self.internal_high.price > 0 and not self.internal_high.crossed and close > self.internal_high.price:
            structure_type = 'CHoCH' if self.internal_trend == BEARISH else 'BOS'
            new_structure = Structure(
                pivot=self.internal_high,
                break_bar_index=i,
                break_time=df['time'][i],
                type=structure_type,
                direction=BULLISH
            )
            origin_ob = self._find_and_store_order_block_for_leg(new_structure, df, is_swing=False)
            new_structure.originating_ob = origin_ob
            self.structures.append(new_structure)

            self.internal_high.crossed = True
            self.internal_trend = BULLISH
            
        if self.internal_low.price > 0 and not self.internal_low.crossed and close < self.internal_low.price:
            structure_type = 'CHoCH' if self.internal_trend == BULLISH else 'BOS'
            new_structure = Structure(
                pivot=self.internal_low,
                break_bar_index=i,
                break_time=df['time'][i],
                type=structure_type,
                direction=BEARISH
            )
            origin_ob = self._find_and_store_order_block_for_leg(new_structure, df, is_swing=False)
            new_structure.originating_ob = origin_ob
            self.structures.append(new_structure)

            self.internal_low.crossed = True
            self.internal_trend = BEARISH

    def _find_and_store_order_block_for_leg(self, structure: Structure, df: pd.DataFrame, is_swing: bool) -> Optional[OrderBlock]:
        """Finds the extreme order block for a given leg, stores it, and returns it."""
        
        # Search range from the start of the leg to the break
        search_range = df.iloc[structure.pivot.bar_index : structure.break_bar_index + 1]
        
        if structure.direction == BULLISH: # Bullish break, look for bearish OB
            # We want the lowest point of the leg that started the move
            ob_candle_idx = int(search_range['low'].idxmin())
            ob_direction = BEARISH
        else: # Bearish break, look for bullish OB
            # We want the highest point of the leg that started the move
            ob_candle_idx = int(search_range['high'].idxmax())
            ob_direction = BULLISH

        ob_candle = df.iloc[ob_candle_idx]
        
        new_ob = OrderBlock(
            high=ob_candle['high'],
            low=ob_candle['low'],
            open=ob_candle['open'],
            time=ob_candle['time'],
            bar_index=ob_candle_idx,
            direction=ob_direction,
            is_swing=is_swing,
            ob_type='Extreme' # We can simplify to just find the extreme for now
        )
        
        if is_swing:
            self.swing_order_blocks.append(new_ob)
        else:
            self.internal_order_blocks.append(new_ob)
            
        return new_ob

    def _check_ob_mitigation(self, i: int, df: pd.DataFrame):
        """Checks and flags mitigated order blocks."""
        mitigation_mode = self.config['ob_mitigation_mode']
        
        for ob_list in [self.swing_order_blocks, self.internal_order_blocks]:
            for ob in ob_list:
                if ob.mitigated:
                    continue
                
                if ob.direction == BULLISH: # Bullish OB (support)
                    source = df['low'][i] if mitigation_mode == 'High/Low' else df['close'][i]
                    if source < ob.low:
                        ob.mitigated = True
                elif ob.direction == BEARISH: # Bearish OB (resistance)
                    source = df['high'][i] if mitigation_mode == 'High/Low' else df['close'][i]
                    if source > ob.high:
                        ob.mitigated = True

    def _check_fvg_mitigation(self, i: int, df: pd.DataFrame):
        """
        Checks if price has filled a Fair Value Gap, updating its state.
        'partial' means the gap has been touched.
        'mitigated' means the gap has been fully passed through.
        """
        for fvg in self.fair_value_gaps:
            # Skip FVGs that are already fully mitigated (invalidated)
            if fvg.state == FVG_STATE_MITIGATED:
                continue

            if fvg.direction == BULLISH: # Bullish FVG (acts as support)
                # Check for full mitigation (price trades completely through the gap)
                if df['low'][i] < fvg.low:
                    fvg.state = FVG_STATE_MITIGATED
                # Check for partial mitigation (price trades into the gap)
                elif df['low'][i] < fvg.high:
                    fvg.state = FVG_STATE_PARTIAL
            
            elif fvg.direction == BEARISH: # Bearish FVG (acts as resistance)
                # Check for full mitigation (price trades completely through the gap)
                if df['high'][i] > fvg.high:
                    fvg.state = FVG_STATE_MITIGATED
                # Check for partial mitigation (price trades into the gap)
                elif df['high'][i] > fvg.low:
                    fvg.state = FVG_STATE_PARTIAL
                
    def _find_fair_value_gaps(self, i: int, df: pd.DataFrame, volatility: pd.Series):
        """Identifies Fair Value Gaps, including auto-thresholding."""
        if i < 2: return
        
        # --- FVG Threshold Logic ---
        last_close = df['close'][i-1]
        last_open = df['open'][i-1]
        
        threshold = 0.0
        if self.config['fvg_auto_threshold'] and last_open > 0:
            # The dynamic threshold is a fraction of the instrument's recent volatility (ATR or Range).
            # This makes FVG detection adaptive to market conditions.
            # We use 25% of the average volatility as the minimum candle body size.
            instrument_volatility = volatility.iloc[i-1]
            if pd.notna(instrument_volatility):
                threshold = instrument_volatility * 0.25
        
        is_strong_bullish_move = (last_close > last_open) and ((last_close - last_open) > threshold)

        if df['low'][i] > df['high'][i-2] and is_strong_bullish_move:
            new_fvg = FairValueGap(
                high=df['low'][i],
                low=df['high'][i-2],
                time=df['time'][i],
                bar_index=i,
                direction=BULLISH
            )
            self.fair_value_gaps.append(new_fvg)
            
        is_strong_bearish_move = (last_close < last_open) and ((last_open - last_close) > threshold)
        if df['high'][i] < df['low'][i-2] and is_strong_bearish_move:
            new_fvg = FairValueGap(
                high=df['low'][i-2],
                low=df['high'][i],
                time=df['time'][i],
                bar_index=i,
                direction=BEARISH
            )
            self.fair_value_gaps.append(new_fvg)

    def _check_eqh_eql(self, new_pivot: Pivot, prev_pivot: Pivot, atr: float, type: str):
        """Checks for Equal Highs or Lows."""
        if prev_pivot.price == 0.0 or prev_pivot.time is None or new_pivot.time is None:
            return
        
        threshold = self.config['equal_high_low_threshold'] * atr
        
        if abs(new_pivot.price - prev_pivot.price) < threshold:
            eqh_eql = EqualHighLow(
                price1=prev_pivot.price,
                time1=prev_pivot.time,
                price2=new_pivot.price,
                time2=new_pivot.time,
                type=type,
                bar_index=new_pivot.bar_index
            )
            self.equal_highs_lows.append(eqh_eql)

    def _find_liquidity(self, df: pd.DataFrame):
        """
        Identifies liquidity pools (buyside and sellside) based on swing points.
        This replicates the "Liquidity" feature from the ICT Concepts script.
        It looks for multiple swing highs/lows that are close in price.
        """
        # --- Configuration ---
        # These values are inspired by the PineScript but can be tuned.
        margin_atr_multiplier = 0.4  # PineScript: 10 / 4 = 2.5; we use 1/2.5 = 0.4 for multiplication
        min_pivots_for_pool = 3      # PineScript: `count > 2`

        # --- 1. Find all significant swing highs and lows using SciPy ---
        # `distance` ensures pivots are not too close to each other
        # `prominence` ensures the pivot "stands out" from its neighbors
        high_peaks, _ = find_peaks(df['high'], distance=5, prominence=df['atr'].mean() * 0.5)
        low_peaks, _ = find_peaks(-df['low'], distance=5, prominence=df['atr'].mean() * 0.5)

        # --- 2. Detect Buyside Liquidity (Pools of Highs) ---
        # Sort peaks by price to easily group them
        sorted_high_peaks = sorted(high_peaks, key=lambda i: float(df['high'].iloc[i]))
        
        i = 0
        while i < len(sorted_high_peaks):
            current_peak_idx = sorted_high_peaks[i]
            price_level = float(df['high'].iloc[current_peak_idx])
            atr_at_level = float(df['atr'].iloc[current_peak_idx])
            
            # Define the price tolerance range (margin)
            price_margin = atr_at_level * margin_atr_multiplier
            price_range_top = price_level + price_margin
            
            # Group all subsequent peaks that fall within this range
            pool_indices = [current_peak_idx]
            j = i + 1
            while j < len(sorted_high_peaks):
                next_peak_idx = sorted_high_peaks[j]
                if float(df['high'].iloc[next_peak_idx]) <= price_range_top:
                    pool_indices.append(next_peak_idx)
                    j += 1
                else:
                    break
            
            # If we found enough pivots to form a pool, store it
            if len(pool_indices) >= min_pivots_for_pool:
                # The pool is defined by its highest price and most recent time
                highest_price = max(float(df['high'].iloc[k]) for k in pool_indices)
                latest_index = max(pool_indices)
                
                self.liquidity_pools.append(Liquidity(
                    price=highest_price,
                    time=pd.Timestamp(df['time'].iloc[latest_index]),
                    bar_index=latest_index,
                    direction=LIQUIDITY_BUYSIDE,
                    num_pivots=len(pool_indices)
                ))
            
            # Move the main loop past the group we just processed
            i = j
            
        # --- 3. Detect Sellside Liquidity (Pools of Lows) ---
        # The logic is the inverse of buyside liquidity
        sorted_low_peaks = sorted(low_peaks, key=lambda i: float(df['low'].iloc[i]), reverse=True)

        i = 0
        while i < len(sorted_low_peaks):
            current_peak_idx = sorted_low_peaks[i]
            price_level = float(df['low'].iloc[current_peak_idx])
            atr_at_level = float(df['atr'].iloc[current_peak_idx])
            
            price_margin = atr_at_level * margin_atr_multiplier
            price_range_bottom = price_level - price_margin

            pool_indices = [current_peak_idx]
            j = i + 1
            while j < len(sorted_low_peaks):
                next_peak_idx = sorted_low_peaks[j]
                if float(df['low'].iloc[next_peak_idx]) >= price_range_bottom:
                    pool_indices.append(next_peak_idx)
                    j += 1
                else:
                    break
            
            if len(pool_indices) >= min_pivots_for_pool:
                lowest_price = min(float(df['low'].iloc[k]) for k in pool_indices)
                latest_index = max(pool_indices)
                
                self.liquidity_pools.append(Liquidity(
                    price=lowest_price,
                    time=pd.Timestamp(df['time'].iloc[latest_index]),
                    bar_index=latest_index,
                    direction=LIQUIDITY_SELLSIDE,
                    num_pivots=len(pool_indices)
                ))
            
            i = j

    def _check_liquidity_mitigation(self, i: int, df: pd.DataFrame):
        """Checks if a liquidity pool has been taken."""
        for pool in self.liquidity_pools:
            if pool.mitigated:
                continue
            
            # Buyside liquidity is taken if price moves above it
            if pool.direction == LIQUIDITY_BUYSIDE and df['high'][i] > pool.price:
                pool.mitigated = True
                
            # Sellside liquidity is taken if price moves below it
            if pool.direction == LIQUIDITY_SELLSIDE and df['low'][i] < pool.price:
                pool.mitigated = True

    def _store_breaker_or_mitigation_block(self, df: pd.DataFrame, direction: int):
        """
        Identifies and stores a Breaker or Mitigation Block after a structure break.
        This is triggered when a major swing high/low is broken.
        """
        if direction == BULLISH:
            # We just had a bullish MSB, breaking swing_high.
            # The relevant pivots are the two most recent swing lows.
            # l0 = self.swing_low, l1 = self.prev_swing_low
            if self.swing_low.price == 0.0 or self.prev_swing_low.price == 0.0:
                return

            # Determine if it's a Breaker or Mitigation block
            block_type = 'BB' if self.swing_low.price < self.prev_swing_low.price else 'MB'
            
            # Find the last bearish candle (down-candle) before the swing_low (l0) was formed.
            # A simple approach is to look back a fixed number of candles from the low.
            search_start = max(0, self.swing_low.bar_index - self.config['internal_length'])
            search_end = self.swing_low.bar_index + 1
            
            search_range = df.iloc[search_start:search_end]
            down_candles = search_range[search_range['open'] > search_range['close']]
            
            if not down_candles.empty:
                block_candle = down_candles.iloc[-1] # The last down-candle
                new_block = BreakerMitigationBlock(
                    high=block_candle['high'],
                    low=block_candle['low'],
                    time=block_candle['time'],
                    bar_index=int(block_candle.name) if isinstance(block_candle.name, (int, float)) else 0,
                    direction=BULLISH, # A bullish BB/MB acts as support
                    block_type=block_type
                )
                self.breaker_mitigation_blocks.append(new_block)

        elif direction == BEARISH:
            # We just had a bearish MSB, breaking swing_low.
            # The relevant pivots are the two most recent swing highs.
            # h0 = self.swing_high, h1 = self.prev_swing_high
            if self.swing_high.price == 0.0 or self.prev_swing_high.price == 0.0:
                return

            block_type = 'BB' if self.swing_high.price > self.prev_swing_high.price else 'MB'

            search_start = max(0, self.swing_high.bar_index - self.config['internal_length'])
            search_end = self.swing_high.bar_index + 1
            
            search_range = df.iloc[search_start:search_end]
            up_candles = search_range[search_range['open'] < search_range['close']]
            
            if not up_candles.empty:
                block_candle = up_candles.iloc[-1] # The last up-candle
                new_block = BreakerMitigationBlock(
                    high=block_candle['high'],
                    low=block_candle['low'],
                    time=block_candle['time'],
                    bar_index=int(block_candle.name) if isinstance(block_candle.name, (int, float)) else 0,
                    direction=BEARISH, # A bearish BB/MB acts as resistance
                    block_type=block_type
                )
                self.breaker_mitigation_blocks.append(new_block)

    def _check_bb_mb_mitigation(self, i: int, df: pd.DataFrame):
        """Checks if a Breaker or Mitigation Block has been mitigated."""
        for block in self.breaker_mitigation_blocks:
            if block.mitigated:
                continue
            
            # A bullish block is mitigated if price trades below its low
            if block.direction == BULLISH and df['low'][i] < block.low:
                block.mitigated = True
                
            # A bearish block is mitigated if price trades above its high
            if block.direction == BEARISH and df['high'][i] > block.high:
                block.mitigated = True

    def _update_trailing_extremes(self, i: int, df: pd.DataFrame):
        """Updates the highest high and lowest low of the current swing leg."""
        # After a swing low is confirmed, the trend is bullish. We track the highest high of the leg.
        if self.swing_trend == BULLISH and self.trailing_high is not None:
            if df['high'][i] > self.trailing_high.price:
                try:
                    price = float(df.iloc[i]['high'])
                    time_val = pd.Timestamp(df.iloc[i]['time'])
                    self.trailing_high = Pivot(price=price, bar_index=i, time=time_val)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not update trailing high at index {i} due to data type issue: {e}")

        # After a swing high is confirmed, the trend is bearish. We track the lowest low of the leg.
        elif self.swing_trend == BEARISH and self.trailing_low is not None:
            if df['low'][i] < self.trailing_low.price:
                try:
                    price = float(df.iloc[i]['low'])
                    time_val = pd.Timestamp(df.iloc[i]['time'])
                    self.trailing_low = Pivot(price=price, bar_index=i, time=time_val)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not update trailing low at index {i} due to data type issue: {e}")

    def _get_premium_discount_zones(self) -> Optional[PremiumDiscountZone]:
        """Calculates the Premium/Discount zones based on the current trailing extremes."""
        # The range is defined by the last confirmed swing pivot and the extreme of the current leg.
        top_pivot = self.trailing_high if self.swing_trend == BEARISH else self.swing_high
        bottom_pivot = self.trailing_low if self.swing_trend == BULLISH else self.swing_low

        if self.swing_trend == BULLISH:
            if not (self.swing_low and self.trailing_high and self.swing_low.price > 0):
                return None
            top = self.trailing_high.price
            bottom = self.swing_low.price
        else:  # BEARISH
            if not (self.swing_high and self.trailing_low and self.swing_high.price > 0):
                return None
            top = self.swing_high.price
            bottom = self.trailing_low.price

        price_range = top - bottom
        if price_range <= 0: return None
        
        equilibrium = bottom + price_range * 0.5
        
        return PremiumDiscountZone(
            premium_top=top,
            premium_bottom=equilibrium,
            equilibrium_top=equilibrium + price_range * 0.025, # 2.5% buffer around equilibrium
            equilibrium_bottom=equilibrium - price_range * 0.025,
            discount_top=equilibrium,
            discount_bottom=bottom
        )

    def _get_strong_weak_highs_lows(self) -> StrongWeakHighLow:
        """Labels the last swing pivots as strong or weak based on the final trend."""
        # A high is strong if the trend is bearish (it caused a break of structure down)
        high_type = 'Strong' if self.swing_trend == BEARISH else 'Weak'
        # A low is strong if the trend is bullish (it caused a break of structure up)
        low_type = 'Strong' if self.swing_trend == BULLISH else 'Weak'
        
        return StrongWeakHighLow(
            high=self.swing_high,
            high_type=high_type,
            low=self.swing_low,
            low_type=low_type
        )