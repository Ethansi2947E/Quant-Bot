"""
Breakout Trading Strategy

A rules-based strategy to trade breakouts from consolidation patterns or key S/R levels,
confirmed by volume and price action.

Features:
- Consolidation identification using price structure (narrowing range bars) instead of Bollinger Bands.
- Breakout from identified S/R levels.
- Entry on decisive breakout candle with significant, directional volume (buy/sell spike differentiation).
- Filters for false breakouts using price rejection logic (pin bars/long wicks at breakout levels).
- Stop-loss placement based on consolidation structure or ATR.
- Take-profit targets based on measured moves, R:R, or ATR multiples.
- Risk management via fixed fractional position sizing.
"""

# Import pandas and numpy directly without pandas_ta
import pandas as pd
import numpy as np
from loguru import logger
from typing import Dict, List, Any, Optional
from src.trading_bot import SignalGenerator
from src.risk_manager import RiskManager


# Define custom implementations of the indicators we need
def calculate_bollinger_bands(df, length=20, std=2.0):
    """Calculate Bollinger Bands without pandas_ta"""
    # Initialize an empty DataFrame with the same index
    df_bb = pd.DataFrame(index=df.index)
    
    if len(df) < length:
        # Return empty DataFrame with required columns
        df_bb[f"BBM_{length}_{std}"] = np.nan
        df_bb[f"BBU_{length}_{std}"] = np.nan
        df_bb[f"BBL_{length}_{std}"] = np.nan
        df_bb[f"BBB_{length}_{std}"] = np.nan
        return df_bb
    
    # Calculate middle band (SMA)
    df_bb[f"BBM_{length}_{std}"] = df["close"].rolling(length).mean()
    
    # Calculate standard deviation
    stdev = df["close"].rolling(length).std()
    
    # Calculate upper and lower bands
    df_bb[f"BBU_{length}_{std}"] = df_bb[f"BBM_{length}_{std}"] + (std * stdev)
    df_bb[f"BBL_{length}_{std}"] = df_bb[f"BBM_{length}_{std}"] - (std * stdev)
    
    # Calculate bandwidth
    df_bb[f"BBB_{length}_{std}"] = (
        (df_bb[f"BBU_{length}_{std}"] - df_bb[f"BBL_{length}_{std}"]) / 
        df_bb[f"BBM_{length}_{std}"] * 100
    )
    
    return df_bb

def calculate_atr(df, length=14):
    """Calculate ATR without pandas_ta"""
    if len(df) < length + 1:
        # Return a Series of NaNs with the same index as df
        return pd.Series(np.nan, index=df.index)
    
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)
    
    # True Range calculation
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # ATR calculation
    atr = tr.rolling(window=length).mean()
        
    return atr

def is_consolidating(df: pd.DataFrame, window=10, tolerance=0.02) -> bool:
    """
    Simple consolidation detection based on price range.
    
    Args:
        df: DataFrame with OHLC data
        window: Number of bars to check for consolidation
        tolerance: Maximum price range as percentage of min price
        
    Returns:
        bool: True if price is consolidating
    """
    if len(df) < window:
        return False
        
    recent = df.iloc[-window:]
    max_close = recent['close'].max()
    min_close = recent['close'].min()
    return (max_close - min_close) / min_close < tolerance

def is_near_support_resistance(df: pd.DataFrame, pivot_window=20, tolerance=0.005) -> bool:
    """
    Check if price is near any significant support or resistance level.
    This ensures levels are properly separated and mutually exclusive.
    
    Args:
        df: DataFrame with OHLC data
        pivot_window: Window to look for pivot points
        tolerance: How close price needs to be to S/R (as percentage)
        
    Returns:
        bool: True if price is near any S/R level
    """
    if len(df) < pivot_window * 2:
        return False
        
    current_price = df['close'].iloc[-1]
        
    # Find pivot highs and lows
    pivot_highs = []
    pivot_lows = []
    
    for i in range(pivot_window, len(df) - pivot_window):
        window_high = df['high'].iloc[i-pivot_window:i+pivot_window+1]
        window_low = df['low'].iloc[i-pivot_window:i+pivot_window+1]
        
        if df['high'].iloc[i] == window_high.max():
            pivot_highs.append(df['high'].iloc[i])
            
        if df['low'].iloc[i] == window_low.min():
            pivot_lows.append(df['low'].iloc[i])
    
    # Separate levels by current price (support below, resistance above)
    support_levels = []
    resistance_levels = []
    
    for level in pivot_lows:
        # Support must be below current price
        if level < current_price:
            support_levels.append(level)
            
    for level in pivot_highs:
        # Resistance must be above current price
        if level > current_price:
            resistance_levels.append(level)
    
    # Check proximity to properly separated levels
    near_support = any(abs(current_price - pl) / pl < tolerance for pl in support_levels)
    near_resistance = any(abs(ph - current_price) / ph < tolerance for ph in resistance_levels)
    
    # Return True if near any properly separated level
    return near_support or near_resistance

def is_valid_breakout(candle, breakout_level, direction):
    """
    Validate breakout quality based on candle structure.
    
    Args:
        candle: Single row of OHLC data (as dict or Series)
        breakout_level: The S/R level being broken
        direction: 'BUY' or 'SELL'
        
    Returns:
        bool: True if breakout is valid based on price rejection
    """
    body = abs(candle['close'] - candle['open'])
    if body == 0:  # Avoid division by zero
        return False
        
    if direction == "BUY":
        wick_ratio = (candle['high'] - candle['close']) / body if body > 0 else 0
        return wick_ratio < 0.3  # Minimal upper wick = strong acceptance
    else:
        wick_ratio = (candle['open'] - candle['low']) / body if body > 0 else 0
        return wick_ratio < 0.3  # Minimal lower wick = strong acceptance

class BreakoutTradingStrategy(SignalGenerator):
    """
    Breakout Trading Strategy: Identifies consolidation and trades breakouts.
    """
    def __init__(
        self,
        primary_timeframe: str = "M15",
        risk_per_trade: float = 0.01,
        # Consolidation Identification Parameters
        bb_period: int = 20,
        bb_std_dev: float = 2.0,
        bb_squeeze_lookback: int = 20,
        atr_period_consolidation: int = 14,
        min_consolidation_bars: int = 12,
        # Breakout Parameters
        breakout_confirmation_atr_multiplier: float = 0.15,
        volume_confirmation_multiplier: float = 1.3,
        volume_avg_period: int = 20,
        stop_loss_atr_multiplier: float = 1.5,
        take_profit_rr_ratio: float = 2.0,
        wait_for_confirmation_candle: int = 0, # 0: no delay, 1: wait 1 bar, 2: wait 2 bars, etc.
        pivot_window: int = 10,
        max_zones: int = 2,
        take_profit_mode: str = 'rr',
        require_retest: bool = False,
        retest_threshold_pct: float = 0.3,
        lookback_period: int = 300,
        **kwargs
    ):
        """
        Breakout Trading Strategy: Identifies consolidation and trades breakouts.
        Args:
            min_consolidation_bars (int): Minimum bars for consolidation. Recommended 20+ for M5/M15 to match real squeeze tension (see StockCharts/ChartSchool best practices).
            wait_for_confirmation_candle (int): Number of bars to wait after breakout before entry. 0 = immediate, 1 = wait 1 bar, etc. Most guides recommend 1–2 bars max.
        """
        super().__init__(**kwargs)
        self.logger = logger
        self.name = "BreakoutTradingStrategy"
        self.description = "Trades breakouts from consolidation, confirmed by volume and volatility."
        self.version = "1.0.0"
        self.primary_timeframe = primary_timeframe
        self.risk_per_trade = risk_per_trade

        self.bb_period = bb_period
        self.bb_std_dev = bb_std_dev
        self.bb_squeeze_lookback = bb_squeeze_lookback
        self.atr_period_consolidation = atr_period_consolidation
        self.min_consolidation_bars = min_consolidation_bars
        if self.min_consolidation_bars < 15:
            self.logger.warning(f"[Config] min_consolidation_bars={self.min_consolidation_bars} is low. For M5/M15, 20+ bars is recommended for meaningful squeezes.")

        self.breakout_confirmation_atr_multiplier = breakout_confirmation_atr_multiplier
        self.volume_confirmation_multiplier = volume_confirmation_multiplier
        self.volume_avg_period = volume_avg_period

        self.stop_loss_atr_multiplier = stop_loss_atr_multiplier
        self.take_profit_rr_ratio = take_profit_rr_ratio
        self.wait_for_confirmation_candle = wait_for_confirmation_candle
        self.pivot_window = pivot_window
        self.max_zones = max_zones
        self.take_profit_mode = take_profit_mode
        self.require_retest = require_retest
        self.retest_threshold_pct = retest_threshold_pct
        self.lookback_period = lookback_period
        self.params = kwargs

        self.risk_manager = RiskManager.get_instance() if hasattr(RiskManager, "get_instance") else RiskManager()
        self.processed_bars = {}
        self.consolidation_state = {}
        self.breakout_levels = {}

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            self.logger.debug("[Indicators] DataFrame is empty, skipping indicator calculation.")
            return df
        if not all(col in df.columns for col in ["high", "low", "close"]):
            self.logger.error("[Indicators] DataFrame missing HLC columns for indicator calculation.")
            return df
            
        # Calculate ATR using our custom function instead of df.ta
        df[f"ATR_{self.atr_period_consolidation}"] = calculate_atr(df, self.atr_period_consolidation)
        
        # Calculate Bollinger Bands using our custom function
        bbands = calculate_bollinger_bands(df, self.bb_period, self.bb_std_dev)
        if bbands is not None and not bbands.empty:
            df[f"BB_LOWER_{self.bb_period}_{self.bb_std_dev}"] = bbands[f"BBL_{self.bb_period}_{self.bb_std_dev}"]
            df[f"BB_MID_{self.bb_period}_{self.bb_std_dev}"] = bbands[f"BBM_{self.bb_period}_{self.bb_std_dev}"]
            df[f"BB_UPPER_{self.bb_period}_{self.bb_std_dev}"] = bbands[f"BBU_{self.bb_period}_{self.bb_std_dev}"]
            df[f"BB_WIDTH_{self.bb_period}_{self.bb_std_dev}"] = bbands[f"BBB_{self.bb_period}_{self.bb_std_dev}"] / 100
        else:
            for col_suffix in ["LOWER", "MID", "UPPER", "WIDTH"]:
                df[f"BB_{col_suffix}_{self.bb_period}_{self.bb_std_dev}"] = np.nan
                
        # Calculate volume moving average, backfill to avoid NaN
        if "tick_volume" in df.columns:
            df[f"volume_ma_{self.volume_avg_period}"] = df["tick_volume"].rolling(window=self.volume_avg_period).mean().bfill()
        else:
            df[f"volume_ma_{self.volume_avg_period}"] = np.nan
            
        # Log last row of indicators for debug
        last_idx = df.index[-1]
        self.logger.debug(f"[Indicators] Last row: {df.loc[last_idx][['close', f'ATR_{self.atr_period_consolidation}', f'BB_LOWER_{self.bb_period}_{self.bb_std_dev}', f'BB_MID_{self.bb_period}_{self.bb_std_dev}', f'BB_UPPER_{self.bb_period}_{self.bb_std_dev}', f'BB_WIDTH_{self.bb_period}_{self.bb_std_dev}', f'volume_ma_{self.volume_avg_period}']].to_dict()}")
        return df

    def _detect_consolidation(self, df: pd.DataFrame) -> dict:
        result = {
            'is_consolidating': False,
            'consolidation_start': None,
            'consolidation_end': None,
            'breakout_high': None,
            'breakout_low': None,
            'bars_in_consolidation': 0,
            'is_squeeze': False,  # Deprecated, kept for compatibility
            'inside_sr': False
        }
        if df is None or len(df) < 30:
            self.logger.debug(f"[Consolidation] Not enough data: len(df)={len(df) if df is not None else 'None'}, required=30 for robust detection")
            return result

        # --- Dynamic Bar Counting Based on Volatility ---
        atr_col = f"ATR_{self.atr_period_consolidation}"
        atr_value = df[atr_col].iloc[-1] if atr_col in df.columns else 0
        close_value = df['close'].iloc[-1] if 'close' in df.columns else 1
        volatility_ratio = atr_value / close_value if close_value > 0 else 0
        
        # Dynamic window based on volatility (more volatile = shorter window)
        dynamic_window = max(5, int(10 * (1 - volatility_ratio * 100)))
        self.logger.debug(f"[Consolidation] ATR={atr_value:.5f}, close={close_value:.5f}, volatility_ratio={volatility_ratio:.5f}, dynamic_window={dynamic_window}")
        
        # Use simpler consolidation detection
        is_consolidating_simple = is_consolidating(df, window=dynamic_window, tolerance=0.02)
        is_near_sr = is_near_support_resistance(df, pivot_window=self.pivot_window, tolerance=0.005)
        
        # Combined criteria - must be consolidating AND near S/R for good breakout setup
        is_in_breakout_zone = is_consolidating_simple and is_near_sr
        
        # Get S/R zones for breakout levels
        sr_zones = self.get_sr_zones(df)
        support_zone = max([z for z in sr_zones.get('support', []) if z <= close_value], default=None)
        resistance_zone = min([z for z in sr_zones.get('resistance', []) if z >= close_value], default=None)
        
        self.logger.info(f"[Consolidation] is_consolidating={is_consolidating_simple}, is_near_sr={is_near_sr}, is_in_breakout_zone={is_in_breakout_zone}, S/R: {support_zone}-{resistance_zone}")

        result['dynamic_min_bars'] = dynamic_window
        result['inside_sr'] = is_near_sr
        result['bars_in_consolidation'] = dynamic_window if is_consolidating_simple else 0
        result['is_consolidating'] = is_in_breakout_zone
        result['consolidation_start'] = df.index[-dynamic_window] if is_in_breakout_zone else None
        result['consolidation_end'] = df.index[-1] if is_in_breakout_zone else None
        result['breakout_high'] = resistance_zone
        result['breakout_low'] = support_zone
        return result

    def _find_pivot_highs(self, df: pd.DataFrame) -> list:
        """Find pivot highs over the configured window."""
        pivots = []
        w = self.pivot_window
        for i in range(w, len(df) - w):
            window = df['high'].iloc[i-w:i+w+1]
            if df['high'].iloc[i] == window.max():
                pivots.append(df['high'].iloc[i])
        return pivots

    def _find_pivot_lows(self, df: pd.DataFrame) -> list:
        """Find pivot lows over the configured window."""
        pivots = []
        w = self.pivot_window
        for i in range(w, len(df) - w):
            window = df['low'].iloc[i-w:i+w+1]
            if df['low'].iloc[i] == window.min():
                pivots.append(df['low'].iloc[i])
        return pivots

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate the Average True Range (ATR) over the given period."""
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

    def _cluster_levels(self, levels: list, tol: float = 0.003, df: Optional[pd.DataFrame] = None) -> list:
        """Cluster price levels into horizontal zones within ±tol (as a fraction of price)."""
        if not levels:
            return []
        if df is not None and len(df) > 15:
            latest_price = df['close'].iloc[-1]
            atr = self._calculate_atr(df)
            tol = max(0.003, 0.5 * atr / latest_price) if latest_price > 0 else 0.003
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
        return [float(np.mean(cluster)) for cluster in clusters]

    def _is_in_zone(self, price, zone: float, tol: float = 0.003):
        """Check if price(s) is/are inside a zone (±tol)."""
        if isinstance(price, (np.ndarray, pd.Series)):
            return np.abs(price - zone) <= zone * tol
        return abs(price - zone) <= zone * tol

    def get_sr_zones(self, df: pd.DataFrame) -> dict:
        """Compute and return the top N support and resistance zones."""
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
        dynamic_max_zones = 4 if volatility > 0.01 else self.max_zones
        self.logger.info(f"Volatility: {volatility:.4f}, using max_zones={dynamic_max_zones}")
        top_res = [z for _, z in sorted(zip(res_counts, res_zones), reverse=True)[:dynamic_max_zones]]
        top_sup = [z for _, z in sorted(zip(sup_counts, sup_zones), reverse=True)[:dynamic_max_zones]]
        return {'support': top_sup, 'resistance': top_res}

    @property
    def required_timeframes(self):
        return [self.primary_timeframe]

    @property
    def lookback_periods(self):
        return {self.primary_timeframe: self.lookback_period}

    async def generate_signals(self, market_data: Dict[str, Any], symbol: Optional[str] = None, **kwargs) -> List[Dict]:
        logger.debug(f"[StrategyInit] {self.__class__.__name__}: required_timeframes={self.required_timeframes}, lookback_periods={self.lookback_periods}")
        signals = []
        self.processed_bars = {}
        symbols = [symbol] if symbol else list(market_data.keys())
        for sym in symbols:
            if self.primary_timeframe not in market_data[sym] or market_data[sym][self.primary_timeframe].empty:
                self.logger.debug(f"[Signal] {sym}: No data for timeframe {self.primary_timeframe} or DataFrame is empty.")
                continue
            df = market_data[sym][self.primary_timeframe].copy()
            if 'tick_volume' not in df.columns:
                if 'volume' in df.columns:
                    self.logger.debug(f"[Signal] {sym}: Using 'volume' as 'tick_volume' fallback.")
                    df['tick_volume'] = df['volume']
                else:
                    self.logger.debug(f"[Signal] {sym}: No 'tick_volume' or 'volume' column, using price-based volume proxy.")
                    df['tick_volume'] = df['close'].rolling(20).mean().bfill()
            if len(df) < self.lookback_periods[self.primary_timeframe]:
                self.logger.debug(f"[Signal] {sym}: Not enough bars ({len(df)}) for lookback ({self.lookback_periods[self.primary_timeframe]}). Skipping.")
                continue
            df = self._calculate_indicators(df)
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
            n_delay = self.wait_for_confirmation_candle
            idx_breakout_candle = idx - n_delay
            if idx_breakout_candle < 0:
                self.logger.debug(f"[Signal] {sym}: idx_breakout_candle < 0 ({idx_breakout_candle}). Skipping.")
                continue
            entry_bar_idx = idx
            confirmation_bar_idx = idx_breakout_candle
            current_bar_ts = df.index[entry_bar_idx]
            self.logger.debug(f"[EntryDelay] {sym}: wait_for_confirmation_candle={n_delay}, confirmation_bar_idx={confirmation_bar_idx}, entry_bar_idx={entry_bar_idx}")
            if self.processed_bars.get((sym, self.primary_timeframe)) == current_bar_ts:
                self.logger.debug(f"[Signal] {sym}: Already processed bar {current_bar_ts}. Skipping.")
                continue
            self.logger.debug(f"[Breakout] {sym}: close={df.iloc[confirmation_bar_idx]['close']}, high={df.iloc[confirmation_bar_idx]['high']}, low={df.iloc[confirmation_bar_idx]['low']}, ATR={df[f'ATR_{self.atr_period_consolidation}'].iloc[confirmation_bar_idx]:.5f}")

            atr_col = f"ATR_{self.atr_period_consolidation}"
            volatility_ratio = 0.0
            if atr_col in df.columns and confirmation_bar_idx >= 0 and df[atr_col].iloc[confirmation_bar_idx] > 0 and df['close'].iloc[confirmation_bar_idx] > 0:
                volatility_ratio = df[atr_col].iloc[confirmation_bar_idx] / df['close'].iloc[confirmation_bar_idx]
            threshold_multiplier = self._get_dynamic_atr_multiplier(df)
            self.logger.debug(f"[Threshold] {sym}: Dynamic threshold_multiplier={threshold_multiplier} (volatility_ratio={volatility_ratio:.5f})")
            wait_for_confirmation_candle = 1 if volatility_ratio < 0.015 else 0
            n_delay = wait_for_confirmation_candle
            idx_breakout_candle = idx - n_delay
            if idx_breakout_candle < 0:
                self.logger.debug(f"[Signal] {sym}: idx_breakout_candle < 0 ({idx_breakout_candle}). Skipping.")
                continue
            entry_bar_idx = idx
            confirmation_bar_idx = idx_breakout_candle
            current_bar_ts = df.index[entry_bar_idx]
            self.logger.debug(f"[EntryDelay] {sym}: wait_for_confirmation_candle={n_delay}, confirmation_bar_idx={confirmation_bar_idx}, entry_bar_idx={entry_bar_idx}")
            if self.processed_bars.get((sym, self.primary_timeframe)) == current_bar_ts:
                self.logger.debug(f"[Signal] {sym}: Already processed bar {current_bar_ts}. Skipping.")
                continue
            self.logger.debug(f"[Breakout] {sym}: close={df.iloc[confirmation_bar_idx]['close']}, high={df.iloc[confirmation_bar_idx]['high']}, low={df.iloc[confirmation_bar_idx]['low']}, ATR={df[f'ATR_{self.atr_period_consolidation}'].iloc[confirmation_bar_idx]:.5f}")

            consolidation = self._detect_consolidation(df)
            bars_in_consolidation = consolidation['bars_in_consolidation']
            dynamic_min = consolidation.get('dynamic_min_bars', self.min_consolidation_bars)
            if bars_in_consolidation < dynamic_min:
                self.logger.info(f"[Signal] {sym}: Not enough bars in consolidation ({bars_in_consolidation} < {dynamic_min}). Skipping signal.")
                continue
            breakout_high = consolidation['breakout_high']
            breakout_low = consolidation['breakout_low']
            atr_col = f"ATR_{self.atr_period_consolidation}"
            atr_val = df[atr_col].iloc[confirmation_bar_idx] if atr_col in df.columns and not pd.isna(df[atr_col].iloc[confirmation_bar_idx]) else 0.0001
            volume_ma_col = f"volume_ma_{self.volume_avg_period}"
            candle = df.iloc[confirmation_bar_idx]
            entry_price, stop_loss, take_profit = None, None, None
            signal_direction = None

            # --- Liquidity Check: Volume Z-Score ---
            volume_rolling = df['tick_volume'].rolling(50)
            volume_mean = volume_rolling.mean().iloc[confirmation_bar_idx]
            volume_std = volume_rolling.std().iloc[confirmation_bar_idx]
            volume_z_score = (candle['tick_volume'] - volume_mean) / (volume_std if volume_std > 0 else 1)
            min_z = 0.5  # Lowered from 1.0 to 0.5 for all volatility regimes
            # Volume check temporarily disabled
            # if volume_z_score < min_z:
            #     self.logger.info(f"[Signal] {sym}: Skipping - insufficient volume z-score: {volume_z_score:.2f} (min required: {min_z})")
            #     continue

            up_close = candle['close'] > candle['open']
            down_close = candle['close'] < candle['open']
            high_buying_volume = up_close and candle['tick_volume'] > volume_mean + volume_std
            high_selling_volume = down_close and candle['tick_volume'] > volume_mean + volume_std
            if signal_direction == "BUY" and not high_buying_volume:
                self.logger.info(f"[Signal] {sym}: No high buying volume on up close. Skipping BUY signal.")
                continue
            if signal_direction == "SELL" and not high_selling_volume:
                self.logger.info(f"[Signal] {sym}: No high selling volume on down close. Skipping SELL signal.")
                continue

            atr_col = f"ATR_{self.atr_period_consolidation}"
            volatility_ratio = 0.0
            if atr_col in df.columns and confirmation_bar_idx >= 0 and df[atr_col].iloc[confirmation_bar_idx] > 0 and df['close'].iloc[confirmation_bar_idx] > 0:
                volatility_ratio = df[atr_col].iloc[confirmation_bar_idx] / df['close'].iloc[confirmation_bar_idx]
            threshold_multiplier = self._get_dynamic_atr_multiplier(df)

            breakout_high_str = f"{breakout_high:.5f}" if breakout_high is not None else "None"
            breakout_low_str = f"{breakout_low:.5f}" if breakout_low is not None else "None"
            threshold_high_str = f"{(breakout_high + (threshold_multiplier * atr_val)):.5f}" if breakout_high is not None else "None"
            threshold_low_str = f"{(breakout_low - (threshold_multiplier * atr_val)):.5f}" if breakout_low is not None else "None"

            self.logger.debug(
                f"[Breakout] {sym}: close={candle['close']:.5f}, "
                f"breakout_high={breakout_high_str}, "
                f"breakout_low={breakout_low_str}, "
                f"threshold_high={threshold_high_str}, "
                f"threshold_low={threshold_low_str}, "
                f"ATR={atr_val:.5f}, breakout_mult={threshold_multiplier}"
            )

            pinbar_rejection = False
            if breakout_high is not None and hammer.iloc[confirmation_bar_idx] and candle['low'] < consolidation['breakout_low']:
                pinbar_rejection = True
                self.logger.info(f"[Signal] {sym}: Pin bar/hammer detected at resistance breakout level outside consolidation. Filtering out false breakout.")
            if (
                breakout_low is not None and
                shooting_star.iloc[confirmation_bar_idx] and
                consolidation['breakout_high'] is not None and
                candle['high'] > consolidation['breakout_high']
            ):
                pinbar_rejection = True
                self.logger.info(f"[Signal] {sym}: Shooting star detected at support breakout level outside consolidation. Filtering out false breakout.")
            if pinbar_rejection:
                self.logger.info(f"[Signal] {sym}: Pinbar rejection filter triggered. Skipping signal.")
                continue
            # Upside breakout
            if breakout_high is not None and candle["close"] >= breakout_high + (threshold_multiplier * atr_val):
                # Hybrid retest logic: only require retest for add-ons, not initial breakout
                retest_satisfied = True if not self.require_retest else False
                if self.require_retest:
                    breakout_key = (sym, "BUY", breakout_high)
                    if breakout_key not in self.breakout_levels:
                        self.breakout_levels[breakout_key] = {
                            "level": breakout_high,
                            "timestamp": current_bar_ts,
                            "retest_detected": False
                        }
                        self.logger.info(f"BUY Breakout recorded for {sym} at level {breakout_high:.5f}, waiting for retest...")
                        retest_satisfied = False
                    else:
                        if self.breakout_levels[breakout_key]["retest_detected"]:
                            retest_satisfied = True
                            self.logger.info(f"BUY Retest already detected for {sym} at level {breakout_high:.5f}")
                        else:
                            retest_satisfied = self._detect_retest(df, sym, breakout_high, "BUY")
                            if retest_satisfied:
                                self.breakout_levels[breakout_key]["retest_detected"] = True
                                self.logger.info(f"BUY Retest detected for {sym} at level {breakout_high:.5f}")
                            else:
                                self.logger.debug(f"BUY Retest not yet detected for {sym} at level {breakout_high:.5f}")
                if retest_satisfied:
                    avg_vol = df["tick_volume"].iloc[max(0, confirmation_bar_idx - self.volume_avg_period):confirmation_bar_idx].mean()
                    strong_break = candle["close"] >= breakout_high + (2 * threshold_multiplier * atr_val)
                    # Volume confirmation relaxation for crypto/volatile indices
                    volume_confirmation_multiplier = 1.1 if volatility_ratio > 0.01 else 1.3
                    if strong_break:
                        vol_ok = candle["tick_volume"] >= avg_vol * volume_confirmation_multiplier
                    else:
                        vol_ok = candle["tick_volume"] >= avg_vol * max(volume_confirmation_multiplier, 1.3)
                    self.logger.debug(f"[Signal] {sym}: Volume check: tick_volume={candle['tick_volume']}, avg_vol={avg_vol}, required={avg_vol * (volume_confirmation_multiplier if strong_break else max(volume_confirmation_multiplier, 1.3))}, strong_break={strong_break}, vol_ok={vol_ok}")
                    
                    # Price action confirmation (bullish close above S/R)
                    price_action_ok = candle["close"] > candle["open"]
                    
                    # Check wick structure for breakout quality using new function
                    breakout_structure_ok = is_valid_breakout(candle, breakout_high, "BUY")
                    
                    # Volume must be high AND price action must confirm
                    if vol_ok and price_action_ok and breakout_structure_ok:
                        signal_direction = "BUY"
                        entry_price = df["open"].iloc[entry_bar_idx]
                        # Stop-loss optimization (structural)
                        stop_loss = min(breakout_low, entry_price - 1.0 * atr_val) if breakout_low is not None else entry_price - 1.0 * atr_val
                        
                        # Use measured move for target (height of consolidation)
                        if consolidation['is_consolidating'] and breakout_high is not None and breakout_low is not None:
                            consolidation_height = breakout_high - breakout_low
                            take_profit = entry_price + consolidation_height
                        else:
                            # Fallback to ATR-based target if consolidation not well defined
                            if volatility_ratio > 0.025:
                                take_profit = entry_price + 1.8 * atr_val
                            else:
                                take_profit = entry_price + 2.5 * atr_val
                        
                        self.logger.info(f"BUY Breakout for {sym} at {current_bar_ts}. Level={breakout_high:.5f}, Vol={vol_ok}, Price Action={price_action_ok}, Structure={breakout_structure_ok}, TP={take_profit}")
                    else:
                        failed_reasons = []
                        if not vol_ok: failed_reasons.append("insufficient volume")
                        if not price_action_ok: failed_reasons.append("bearish close")
                        if not breakout_structure_ok: failed_reasons.append("poor breakout structure")
                        self.logger.info(f"[Signal] {sym}: Upside breakout filters failed: {', '.join(failed_reasons)}")
                else:
                    self.logger.info(f"[Signal] {sym}: Upside breakout waiting for retest of level {breakout_high:.5f}")
            # Downside breakout
            elif breakout_low is not None and candle["close"] <= breakout_low - (threshold_multiplier * atr_val):
                retest_satisfied = True if not self.require_retest else False
                if self.require_retest:
                    breakout_key = (sym, "SELL", breakout_low)
                    if breakout_key not in self.breakout_levels:
                        self.breakout_levels[breakout_key] = {
                            "level": breakout_low,
                            "timestamp": current_bar_ts,
                            "retest_detected": False
                        }
                        self.logger.info(f"SELL Breakout recorded for {sym} at level {breakout_low:.5f}, waiting for retest...")
                        retest_satisfied = False
                    else:
                        if self.breakout_levels[breakout_key]["retest_detected"]:
                            retest_satisfied = True
                            self.logger.info(f"SELL Retest already detected for {sym} at level {breakout_low:.5f}")
                        else:
                            retest_satisfied = self._detect_retest(df, sym, breakout_low, "SELL")
                            if retest_satisfied:
                                self.breakout_levels[breakout_key]["retest_detected"] = True
                                self.logger.info(f"SELL Retest detected for {sym} at level {breakout_low:.5f}")
                            else:
                                self.logger.debug(f"SELL Retest not yet detected for {sym} at level {breakout_low:.5f}")
                if retest_satisfied:
                    avg_vol = df["tick_volume"].iloc[max(0, confirmation_bar_idx - self.volume_avg_period):confirmation_bar_idx].mean()
                    strong_break = candle["close"] <= breakout_low - (2 * threshold_multiplier * atr_val)
                    # Volume confirmation relaxation for crypto/volatile indices
                    volume_confirmation_multiplier = 1.1 if volatility_ratio > 0.01 else 1.3
                    if strong_break:
                        vol_ok = candle["tick_volume"] >= avg_vol * volume_confirmation_multiplier
                    else:
                        vol_ok = candle["tick_volume"] >= avg_vol * max(volume_confirmation_multiplier, 1.3)
                    self.logger.debug(f"[Signal] {sym}: Volume check: tick_volume={candle['tick_volume']}, avg_vol={avg_vol}, required={avg_vol * (volume_confirmation_multiplier if strong_break else max(volume_confirmation_multiplier, 1.3))}, strong_break={strong_break}, vol_ok={vol_ok}")
                    
                    # Price action confirmation (bearish close below S/R)
                    price_action_ok = candle["close"] < candle["open"]
                    
                    # Check wick structure for breakout quality
                    breakout_structure_ok = is_valid_breakout(candle, breakout_low, "SELL")
                    
                    # Volume must be high AND price action must confirm
                    if vol_ok and price_action_ok and breakout_structure_ok:
                        signal_direction = "SELL"
                        entry_price = df["open"].iloc[entry_bar_idx]
                        stop_loss = max(breakout_high, entry_price + 1.0 * atr_val) if breakout_high is not None else entry_price + 1.0 * atr_val
                        
                        # Use measured move for target (height of consolidation)
                        if consolidation['is_consolidating'] and breakout_high is not None and breakout_low is not None:
                            consolidation_height = breakout_high - breakout_low
                            take_profit = entry_price - consolidation_height
                        else:
                            # Fallback to ATR-based target if consolidation not well defined
                            if volatility_ratio > 0.025:
                                take_profit = entry_price - 1.8 * atr_val
                            else:
                                take_profit = entry_price - 2.5 * atr_val
                                
                        self.logger.info(f"SELL Breakout for {sym} at {current_bar_ts}. Level={breakout_low:.5f}, Vol={vol_ok}, Price Action={price_action_ok}, Structure={breakout_structure_ok}, TP={take_profit}")
                    else:
                        failed_reasons = []
                        if not vol_ok: failed_reasons.append("insufficient volume")
                        if not price_action_ok: failed_reasons.append("bullish close")
                        if not breakout_structure_ok: failed_reasons.append("poor breakout structure")
                        self.logger.info(f"[Signal] {sym}: Downside breakout filters failed: {', '.join(failed_reasons)}")
                else:
                    self.logger.info(f"[Signal] {sym}: Downside breakout waiting for retest of level {breakout_low:.5f}")
            else:
                self.logger.info(f"[Signal] {sym}: No breakout detected. close={candle['close']}, breakout_high={breakout_high}, breakout_low={breakout_low}, ATR={atr_val:.5f}")
            # --- Step 3: Signal creation ---
            if signal_direction:
                if entry_price is None:
                    self.logger.warning(f"[Signal] {sym}: No entry_price, skipping signal.")
                    continue
                min_stop_distance = 0.5 * atr_val
                if signal_direction == "BUY" and (stop_loss is None or abs(entry_price - stop_loss) < min_stop_distance):
                    stop_loss = entry_price - self.stop_loss_atr_multiplier * atr_val
                    self.logger.info(f"[Signal] {sym}: BUY - Enforcing minimum SL distance ({min_stop_distance:.5f}), new SL: {stop_loss:.5f}")
                elif signal_direction == "SELL" and (stop_loss is None or abs(entry_price - stop_loss) < min_stop_distance):
                    stop_loss = entry_price + self.stop_loss_atr_multiplier * atr_val
                    self.logger.info(f"[Signal] {sym}: SELL - Enforcing minimum SL distance ({min_stop_distance:.5f}), new SL: {stop_loss:.5f}")
                if take_profit is None or abs(entry_price - stop_loss) < 1e-9:
                    if signal_direction == "BUY":
                        take_profit = entry_price + abs(entry_price - stop_loss) * self.take_profit_rr_ratio
                    else:
                        take_profit = entry_price - abs(entry_price - stop_loss) * self.take_profit_rr_ratio
                    self.logger.info(f"[Signal] {sym}: Fallback TP used: {take_profit:.5f}")
                size = 1.0
                risk_reward = abs(take_profit - entry_price) / max(abs(entry_price - stop_loss), 1e-9)
                reason = f"Breakout {signal_direction} from consolidation. SL: {stop_loss:.5f}, TP: {take_profit:.5f}, R:R={risk_reward:.2f}"
                if hasattr(self.risk_manager, 'calculate_position_size'):
                    try:
                        account_equity = self.risk_manager.get_account_balance() if hasattr(self.risk_manager, 'get_account_balance') else 10000
                        pips_per_unit = 1
                        instrument_price = entry_price
                        stop_loss_pips = abs(entry_price - stop_loss)
                        symbol_for_risk = sym if isinstance(sym, str) else ""
                        size = self.risk_manager.calculate_position_size(
                            account_equity, self.risk_per_trade, stop_loss_pips, pips_per_unit, instrument_price
                        )
                        self.logger.info(f"[Signal] {sym}: Position size: {size}")
                    except Exception as e:
                        self.logger.warning(f"[Signal] {sym}: RiskManager sizing failed: {e}")
                        size = 1.0
                assert entry_price is not None and stop_loss is not None and take_profit is not None, f"Invalid signal values for {sym}"
                if abs(entry_price - stop_loss) > 1e-9:
                    self.logger.info(f"[Signal] {sym}: Signal created: direction={signal_direction}, entry={entry_price}, SL={stop_loss}, TP={take_profit}, size={size}, R:R={risk_reward:.2f}, reason={reason}")
                    signals.append({
                        "symbol": sym, "timestamp": df.index[entry_bar_idx], "direction": signal_direction,
                        "entry_price": entry_price, "stop_loss": stop_loss, "take_profit": take_profit,
                        "pattern": f"Breakout from Consolidation ({bars_in_consolidation} bars)",
                        "strategy_name": self.name,
                        "size": size,
                        "risk_reward": risk_reward,
                        "reason": reason
                    })
                else:
                    self.logger.warning(f"[Signal] {sym}: Signal skipped due to zero/invalid stop distance. entry={entry_price}, stop={stop_loss}")
            if signals:
                self.logger.debug(f"[Debug] Setting processed_bars for ({sym}, {self.primary_timeframe}) = {current_bar_ts}")
                self.processed_bars[(sym, self.primary_timeframe)] = current_bar_ts
        self.require_retest = True
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

    def _detect_retest(self, df: pd.DataFrame, symbol: str, breakout_level: float, direction: str) -> bool:
        """
        Detects if price has retested a broken level after a breakout.
        
        Args:
            df: DataFrame with price data
            symbol: Symbol being traded
            breakout_level: The level that was broken (resistance for upside breakout, support for downside)
            direction: 'BUY' for upside breakout, 'SELL' for downside breakout
            
        Returns:
            bool: True if a valid retest has been detected, False otherwise
        """
        # Need at least 3 bars after breakout to detect retest
        if len(df) < 5:
            return False
            
        # Get the most recent bars (after breakout)
        recent_bars = df.iloc[-5:]
        
        # Determine retest threshold based on ATR
        atr_col = f"ATR_{self.atr_period_consolidation}"
        atr = recent_bars[atr_col].iloc[-1] if atr_col in recent_bars.columns and not pd.isna(recent_bars[atr_col].iloc[-1]) else 0
        
        # Default threshold as percentage of breakout level
        threshold = breakout_level * self.retest_threshold_pct / 100
        
        # If ATR is available, use it to refine the threshold (min 0.25 ATR, max 1.0 ATR)
        if atr > 0:
            threshold = min(max(threshold, 0.25 * atr), 1.0 * atr)
        
        # For upside breakout (BUY): price must pull back down to near the breakout level
        if direction == "BUY":
            # Initial breakout size (from breakout bar)
            initial_breakout_size = df['high'].iloc[-5] - breakout_level
            
            # Retest threshold - how close price must come back to the breakout level
            # Default: 50% of initial breakout size, but adjusted by parameter
            retest_zone_high = breakout_level + (initial_breakout_size * self.retest_threshold_pct)
            
            # Check if any of the recent bars pulled back to the retest zone
            lowest_low = recent_bars['low'].min()
            retest_detected = lowest_low <= retest_zone_high
            
            # Also check if price has moved back up after retest (confirmation)
            if retest_detected and len(recent_bars) >= 2:
                last_close = recent_bars['close'].iloc[-1]
                return last_close > lowest_low  # Price moved back up after retest
            
            return False
            
        # For downside breakout (SELL): price must pull back up to near the breakout level
        elif direction == "SELL":
            # Initial breakout size (from breakout bar)
            initial_breakout_size = breakout_level - df['low'].iloc[-5]
            
            # Retest threshold - how close price must come back to the breakout level
            # Default: 50% of initial breakout size, but adjusted by parameter
            retest_zone_low = breakout_level - (initial_breakout_size * self.retest_threshold_pct)
            
            # Check if any of the recent bars pulled back to the retest zone
            highest_high = recent_bars['high'].max()
            retest_detected = highest_high >= retest_zone_low
            
            # Also check if price has moved back down after retest (confirmation)
            if retest_detected and len(recent_bars) >= 2:
                last_close = recent_bars['close'].iloc[-1]
                return last_close < highest_high  # Price moved back down after retest
                
            return False
            
        return False 

    def _get_dynamic_atr_multiplier(self, df):
        """Return a dynamic ATR multiplier for breakout confirmation based on volatility."""
        atr_col = f"ATR_{self.atr_period_consolidation}"
        if atr_col in df.columns and df[atr_col].iloc[-1] > 0 and df['close'].iloc[-1] > 0:
            volatility_ratio = df[atr_col].iloc[-1] / df['close'].iloc[-1]
            
            # More granular adjustments based on volatility
            if volatility_ratio > 0.03:  # Extremely volatile
                return 0.20  # Wider threshold for confirmation
            elif volatility_ratio > 0.02:  # Highly volatile
                return 0.15
            elif volatility_ratio > 0.01:  # Moderately volatile
                return 0.12
            else:  # Low volatility
                return 0.08  # Tighter threshold for confirmation
                
        return self.breakout_confirmation_atr_multiplier 