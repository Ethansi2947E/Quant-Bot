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
        logger.debug(f"[{self.name}] Entered _detect_consolidation. DF shape: {df.shape if df is not None else 'None'}")
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
        if df is None or len(df) < 30: # Increased min length for robustness
            logger.warning(f"[{self.name}/_detect_consolidation] Not enough data: len(df)={len(df) if df is not None else 'None'}, required=30 for robust detection")
            return result

        # --- Dynamic Bar Counting Based on Volatility ---
        atr_col = f"ATR_{self.atr_period_consolidation}"
        atr_value = df[atr_col].iloc[-1] if atr_col in df.columns else 0
        close_value = df['close'].iloc[-1] if 'close' in df.columns and len(df['close']) > 0 else 1.0 # Added check for empty series
        volatility_ratio = atr_value / close_value if close_value > 0 else 0
        
        # Dynamic window based on volatility (more volatile = shorter window)
        # Adjusted calculation for dynamic_window for more sensitivity
        if volatility_ratio == 0: # Avoid issues if ATR is zero
            dynamic_window = self.min_consolidation_bars # Fallback to configured min
        elif volatility_ratio < 0.005: # Very low volatility
            dynamic_window = int(self.min_consolidation_bars * 1.5) # Longer window for very tight ranges
        elif volatility_ratio > 0.02: # High volatility
            dynamic_window = max(5, int(self.min_consolidation_bars * 0.5)) # Shorter window
        else: # Normal volatility
            dynamic_window = self.min_consolidation_bars
        
        dynamic_window = max(5, min(dynamic_window, len(df) -1, 30)) # Clamp window size

        logger.debug(f"[{self.name}/_detect_consolidation] ATR={atr_value:.5f}, Close={close_value:.5f}, VolatilityRatio={volatility_ratio:.5f}, DynamicWindowForConsolidation={dynamic_window}")
        
        # Ensure dynamic_window doesn't exceed available data
        if len(df) < dynamic_window:
            logger.warning(f"[{self.name}/_detect_consolidation] Dynamic window ({dynamic_window}) exceeds available data ({len(df)}). Adjusting window to {len(df)}.")
            dynamic_window = len(df)
            if dynamic_window < 5 : # If still too small
                 logger.warning(f"[{self.name}/_detect_consolidation] Adjusted dynamic window ({dynamic_window}) is less than 5. No consolidation detected.")
                 return result


        # Use simpler consolidation detection
        is_consolidating_simple = is_consolidating(df, window=dynamic_window, tolerance=0.02) # is_consolidating uses tolerance on min_close, so 0.02 means 2% range
        is_near_sr = is_near_support_resistance(df, pivot_window=self.pivot_window, tolerance=0.005) # is_near_support_resistance uses tolerance on level itself
        
        # Combined criteria - must be consolidating AND near S/R for good breakout setup
        is_in_breakout_zone = is_consolidating_simple and is_near_sr
        
        # Get S/R zones for breakout levels
        sr_zones = self.get_sr_zones(df)
        support_zone = max([z for z in sr_zones.get('support', []) if z <= close_value], default=None) if sr_zones.get('support') else None
        resistance_zone = min([z for z in sr_zones.get('resistance', []) if z >= close_value], default=None) if sr_zones.get('resistance') else None
        
        logger.info(f"[{self.name}/_detect_consolidation] SimpleConsolidating={is_consolidating_simple}, NearSR={is_near_sr}, BreakoutZoneCandidate={is_in_breakout_zone}, NearestSupport={support_zone}, NearestResistance={resistance_zone}")

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
        logger.debug(f"[{self.name}] Entered get_sr_zones. DF shape: {df.shape if df is not None else 'None'}")
        highs = self._find_pivot_highs(df)
        lows = self._find_pivot_lows(df)
        logger.debug(f"[{self.name}/get_sr_zones] Found {len(highs)} raw pivot highs, {len(lows)} raw pivot lows.")

        res_zones = self._cluster_levels(highs, tol=0.003, df=df) # Pass df for dynamic tolerance
        sup_zones = self._cluster_levels(lows, tol=0.003, df=df) # Pass df for dynamic tolerance
        logger.debug(f"[{self.name}/get_sr_zones] Clustered into {len(res_zones)} resistance zones, {len(sup_zones)} support zones.")

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
        
        # Dynamic max_zones: more zones for higher volatility (wider price swings)
        if volatility > 0.015: # High volatility
            dynamic_max_zones = self.max_zones + 2 
        elif volatility < 0.005: # Low volatility
            dynamic_max_zones = max(1, self.max_zones -1)
        else: # Medium volatility
            dynamic_max_zones = self.max_zones

        logger.info(f"[{self.name}/get_sr_zones] Volatility={volatility:.4f} (StdDev={close_std:.5f}, Mean={close_mean:.5f}). Using max_zones={dynamic_max_zones} (config max_zones={self.max_zones}).")
        
        top_res = sorted([z for _, z in sorted(zip(res_counts, res_zones), key=lambda x: x[0], reverse=True)[:dynamic_max_zones]], reverse=False) # Sort from low to high price
        top_sup = sorted([z for _, z in sorted(zip(sup_counts, sup_zones), key=lambda x: x[0], reverse=True)[:dynamic_max_zones]], reverse=True) # Sort from high to low price

        logger.debug(f"[{self.name}/get_sr_zones] Top Resistance Zones ({len(top_res)}): {top_res}")
        logger.debug(f"[{self.name}/get_sr_zones] Top Support Zones ({len(top_sup)}): {top_sup}")
        return {'support': top_sup, 'resistance': top_res}

    @property
    def required_timeframes(self):
        return [self.primary_timeframe]

    @property
    def lookback_periods(self):
        return {self.primary_timeframe: self.lookback_period}

    async def generate_signals(self, market_data: Dict[str, Any], symbol: Optional[str] = None, **kwargs) -> List[Dict]:
        logger.info(f"🚀 [{self.name}] Starting signal generation. Primary TF: {self.primary_timeframe}")
        logger.debug(f"[{self.name}] Strategy Params: BB Period={self.bb_period}, BB StdDev={self.bb_std_dev}, Squeeze Lookback={self.bb_squeeze_lookback}, Min Consolidation Bars={self.min_consolidation_bars}, Breakout ATR Mult={self.breakout_confirmation_atr_multiplier}")

        signals = []
        self.processed_bars = {} # Reset processed bars for each run, or manage state externally if needed
        symbols_to_process = [symbol] if symbol else list(market_data.keys())
        logger.debug(f"[{self.name}] Symbols to process: {symbols_to_process}")

        for sym in symbols_to_process:
            logger.info(f"▶️ [{self.name}/{sym}] Processing symbol.")
            if self.primary_timeframe not in market_data.get(sym, {}):
                logger.warning(f"[{self.name}/{sym}] Data for primary timeframe '{self.primary_timeframe}' not found. Skipping.")
                continue
            
            df = market_data[sym][self.primary_timeframe].copy()
            if df.empty:
                logger.warning(f"[{self.name}/{sym}] DataFrame for '{self.primary_timeframe}' is empty. Skipping.")
                continue

            logger.debug(f"[{self.name}/{sym}] Initial DataFrame shape for {self.primary_timeframe}: {df.shape}")

            if 'tick_volume' not in df.columns:
                if 'volume' in df.columns:
                    logger.debug(f"[{self.name}/{sym}] Using 'volume' column as 'tick_volume'.")
                    df['tick_volume'] = df['volume']
                else:
                    logger.warning(f"[{self.name}/{sym}] Neither 'tick_volume' nor 'volume' column found. Volume-based checks might be unreliable. Using price-based proxy.")
                    # Using a rolling mean of close as a proxy if no volume data is available
                    df['tick_volume'] = df['close'].rolling(self.volume_avg_period, min_periods=1).mean().bfill()

            required_lookback = self.lookback_periods.get(self.primary_timeframe, self.lookback_period)
            if len(df) < required_lookback:
                logger.warning(f"[{self.name}/{sym}] Insufficient data: {len(df)} bars, required {required_lookback}. Skipping.")
                continue
            
            logger.debug(f"[{self.name}/{sym}] Calculating indicators...")
            df = self._calculate_indicators(df)
            if df.empty or df.isnull().all().all(): # Check if df became all NaNs after indicators
                 logger.error(f"[{self.name}/{sym}] DataFrame became empty or all NaNs after indicator calculation. Skipping.")
                 continue
            logger.debug(f"[{self.name}/{sym}] Indicators calculated. DataFrame shape: {df.shape}")

            # Log a sample of the latest data with indicators
            if len(df) > 0:
                indicator_cols_to_log = [
                    'close', f"ATR_{self.atr_period_consolidation}",
                    f"BB_LOWER_{self.bb_period}_{self.bb_std_dev}",
                    f"BB_MID_{self.bb_period}_{self.bb_std_dev}",
                    f"BB_UPPER_{self.bb_period}_{self.bb_std_dev}",
                    f"BB_WIDTH_{self.bb_period}_{self.bb_std_dev}",
                    f"volume_ma_{self.volume_avg_period}"
                ]
                sample_log_cols = [col for col in indicator_cols_to_log if col in df.columns]
                logger.debug(f"[{self.name}/{sym}] Latest data with indicators: {df[sample_log_cols].iloc[-1].to_dict()}")


            # Candlestick patterns
            logger.debug(f"[{self.name}/{sym}] Detecting candlestick patterns...")
            hammer = self.detect_hammer(df)
            shooting_star = self.detect_shooting_star(df)
            bullish_engulfing = self.detect_bullish_engulfing(df)
            bearish_engulfing = self.detect_bearish_engulfing(df)
            inside_bar = self.detect_inside_bar(df)
            morning_star = self.detect_morning_star(df)
            evening_star = self.detect_evening_star(df)
            false_breakout_buy = self.detect_false_breakout(df, 'buy')
            false_breakout_sell = self.detect_false_breakout(df, 'sell')
            idx = len(df) - 1 # Current bar index for decision making
            
            # Determine entry bar and confirmation bar indices
            # Adaptive wait_for_confirmation_candle based on volatility
            atr_col_name = f"ATR_{self.atr_period_consolidation}"
            current_atr = df[atr_col_name].iloc[idx] if atr_col_name in df.columns and not pd.isna(df[atr_col_name].iloc[idx]) else 0.0001
            current_close = df['close'].iloc[idx]
            volatility_ratio_entry = current_atr / current_close if current_close > 0 else 0
            
            # More dynamic wait period: shorter for very low vol, longer for high vol
            if volatility_ratio_entry < 0.005: # Very low volatility (e.g., <0.5% ATR/Price)
                n_delay = 0 # Enter immediately
            elif volatility_ratio_entry > 0.02: # High volatility (e.g., >2% ATR/Price)
                n_delay = min(self.wait_for_confirmation_candle, 1) # Wait max 1 bar
            else:
                n_delay = self.wait_for_confirmation_candle # Use configured default

            logger.debug(f"[{self.name}/{sym}] Volatility for entry delay: ATR={current_atr:.5f}, Close={current_close:.5f}, Ratio={volatility_ratio_entry:.4f}, n_delay set to {n_delay} bars.")

            idx_breakout_candle = idx - n_delay # The actual candle that broke out
            
            if idx_breakout_candle < 0:
                logger.warning(f"[{self.name}/{sym}] Not enough bars for entry delay. idx_breakout_candle={idx_breakout_candle}. Skipping.")
                continue
            
            entry_bar_idx = idx # The bar on which the entry decision is made (current bar)
            confirmation_bar_idx = idx_breakout_candle # The bar that confirmed the breakout

            current_bar_ts = df.index[entry_bar_idx]
            logger.debug(f"[{self.name}/{sym}] Entry Bar (idx={entry_bar_idx}, ts={current_bar_ts}), Confirmation/Breakout Bar (idx={confirmation_bar_idx}, ts={df.index[confirmation_bar_idx]})")

            if self.processed_bars.get((sym, self.primary_timeframe)) == current_bar_ts:
                logger.debug(f"[{self.name}/{sym}] Bar {current_bar_ts} already processed. Skipping.")
                continue
            
            logger.debug(f"[{self.name}/{sym}] Processing Breakout Conditions: Confirmation Candle Close={df.iloc[confirmation_bar_idx]['close']:.5f}, High={df.iloc[confirmation_bar_idx]['high']:.5f}, Low={df.iloc[confirmation_bar_idx]['low']:.5f}, ATR={df[f'ATR_{self.atr_period_consolidation}'].iloc[confirmation_bar_idx]:.5f}")

            logger.info(f"🔄 [{self.name}/{sym}] Detecting consolidation...")
            consolidation = self._detect_consolidation(df)
            logger.info(f"📊 [{self.name}/{sym}] Consolidation Results: {consolidation}")
            
            bars_in_consolidation = consolidation.get('bars_in_consolidation', 0)
            dynamic_min_bars_consol = consolidation.get('dynamic_min_bars', self.min_consolidation_bars)

            if bars_in_consolidation < dynamic_min_bars_consol:
                logger.info(f"[{self.name}/{sym}] Not in valid consolidation or not enough bars ({bars_in_consolidation} < {dynamic_min_bars_consol}). Skipping signal.")
                self.processed_bars[(sym, self.primary_timeframe)] = current_bar_ts # Mark as processed even if no signal
                continue
            
            breakout_high = consolidation.get('breakout_high')
            breakout_low = consolidation.get('breakout_low')
            
            logger.debug(f"[{self.name}/{sym}] Breakout Levels from Consolidation: High={breakout_high}, Low={breakout_low}")

            atr_col = f"ATR_{self.atr_period_consolidation}"
            atr_val = df[atr_col].iloc[confirmation_bar_idx] if atr_col in df.columns and not pd.isna(df[atr_col].iloc[confirmation_bar_idx]) else 0.0001
            if atr_val <= 0:
                logger.warning(f"[{self.name}/{sym}] ATR value is zero or invalid ({atr_val:.5f}). Using small default for calculations.")
                atr_val = 0.0001 * df['close'].iloc[confirmation_bar_idx] if df['close'].iloc[confirmation_bar_idx] > 0 else 0.0001


            volume_ma_col = f"volume_ma_{self.volume_avg_period}"
            candle = df.iloc[confirmation_bar_idx] # Candle that caused the breakout
            entry_price, stop_loss, take_profit = None, None, None
            signal_direction = None
            
            logger.debug(f"[{self.name}/{sym}] Confirmation Candle ({df.index[confirmation_bar_idx]}): O={candle['open']:.5f}, H={candle['high']:.5f}, L={candle['low']:.5f}, C={candle['close']:.5f}, V={candle['tick_volume'] if 'tick_volume' in candle else 'N/A'}")


            # --- Liquidity Check: Volume Z-Score ---
            if 'tick_volume' in df.columns:
                volume_rolling = df['tick_volume'].rolling(window=50, min_periods=10) # Ensure min_periods
                volume_mean = volume_rolling.mean().iloc[confirmation_bar_idx]
                volume_std = volume_rolling.std().iloc[confirmation_bar_idx]
                
                if pd.isna(volume_mean) or pd.isna(volume_std) or volume_std <= 1e-9: # Check for NaN or zero std
                    volume_z_score = 0 # Cannot compute Z-score
                    logger.warning(f"[{self.name}/{sym}] Could not compute valid Z-score (Mean={volume_mean}, Std={volume_std}). Defaulting Z-score to 0.")
                else:
                    volume_z_score = (candle['tick_volume'] - volume_mean) / volume_std
                
                min_z = 0.5
                logger.info(f"[{self.name}/{sym}] Volume Z-score: {volume_z_score:.2f} (Volume={candle['tick_volume']}, Mean={volume_mean:.2f}, Std={volume_std:.2f})")
                # if volume_z_score < min_z: # Re-enable this if needed after testing
                #     logger.info(f"[{self.name}/{sym}] Skipping due to low volume Z-score: {volume_z_score:.2f} (min required: {min_z})")
                #     self.processed_bars[(sym, self.primary_timeframe)] = current_bar_ts
                #     continue
            else:
                logger.warning(f"[{self.name}/{sym}] 'tick_volume' not available for Z-score calculation.")


            up_close = candle['close'] > candle['open']
            down_close = candle['close'] < candle['open']
            
            # More robust volume check - comparing candle volume to its own rolling MA
            volume_ma_value = df[volume_ma_col].iloc[confirmation_bar_idx] if volume_ma_col in df.columns and not pd.isna(df[volume_ma_col].iloc[confirmation_bar_idx]) else 0
            
            # Tiered volume confirmation based on market type (e.g., crypto vs forex)
            # Assuming sym is a string like "BTCUSD" or "EURUSD"
            is_crypto_or_fx_major = any(p in sym.upper() for p in ["BTC", "ETH", "XAU", "EURUSD", "GBPUSD", "USDJPY"])
            
            # Stricter for less volatile, more lenient for crypto/fx
            base_volume_mult = 1.1 if is_crypto_or_fx_major else self.volume_confirmation_multiplier 
            
            # Further adjust based on volatility ratio
            if volatility_ratio_entry > 0.02: # High vol
                current_vol_mult = max(1.0, base_volume_mult * 0.8) # Slightly more lenient
            elif volatility_ratio_entry < 0.005: # Low vol
                current_vol_mult = base_volume_mult * 1.2 # Slightly stricter
            else: # Medium vol
                current_vol_mult = base_volume_mult

            logger.debug(f"[{self.name}/{sym}] Volume multiplier determination: Base={base_volume_mult}, VolRatio={volatility_ratio_entry:.4f}, FinalMult={current_vol_mult:.2f}")

            high_buying_volume = up_close and candle['tick_volume'] > volume_ma_value * current_vol_mult
            high_selling_volume = down_close and candle['tick_volume'] > volume_ma_value * current_vol_mult
            
            logger.debug(f"[{self.name}/{sym}] Volume Confirmation: CandleVol={candle['tick_volume'] if 'tick_volume' in candle else 'N/A'}, VolMA={volume_ma_value:.2f}, RequiredVol={volume_ma_value * current_vol_mult:.2f}, HighBuyingVol={high_buying_volume}, HighSellingVol={high_selling_volume}")


            # Dynamic ATR multiplier for breakout confirmation
            threshold_multiplier = self._get_dynamic_atr_multiplier(df.iloc[:confirmation_bar_idx+1]) # Pass df up to confirmation candle
            logger.info(f"[{self.name}/{sym}] Dynamic ATR Multiplier for Breakout: {threshold_multiplier:.3f}")


            breakout_high_str = f"{breakout_high:.5f}" if breakout_high is not None else "N/A"
            breakout_low_str = f"{breakout_low:.5f}" if breakout_low is not None else "N/A"
            
            threshold_high = (breakout_high + (threshold_multiplier * atr_val)) if breakout_high is not None else float('inf')
            threshold_low = (breakout_low - (threshold_multiplier * atr_val)) if breakout_low is not None else float('-inf')

            threshold_high_str = f"{threshold_high:.5f}" if breakout_high is not None else "N/A"
            threshold_low_str = f"{threshold_low:.5f}" if breakout_low is not None else "N/A"

            self.logger.debug(
                f"[Breakout] {sym}: close={candle['close']:.5f}, "
                f"breakout_high={breakout_high_str}, "
                f"breakout_low={breakout_low_str}, "
                f"threshold_high={threshold_high_str}, "
                f"threshold_low={threshold_low_str}, "
                f"ATR={atr_val:.5f}, breakout_mult={threshold_multiplier}"
            )

            pinbar_rejection = False
            if breakout_high is not None and hammer.iloc[confirmation_bar_idx] and consolidation['breakout_low'] is not None and candle['low'] < consolidation['breakout_low']:
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
                    volume_confirmation_multiplier = 1.1 if volatility_ratio_entry > 0.01 else 1.3
                    if strong_break:
                        vol_ok = candle["tick_volume"] >= avg_vol * current_vol_mult
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
                            if volatility_ratio_entry > 0.025:
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
                    volume_confirmation_multiplier = 1.1 if volatility_ratio_entry > 0.01 else 1.3
                    if strong_break:
                        vol_ok = candle["tick_volume"] >= avg_vol * current_vol_mult
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
                            if volatility_ratio_entry > 0.025:
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
                self.logger.info(f"[{self.name}/{sym}] No breakout detected. Close={candle['close']:.5f}, BreakoutHigh={breakout_high_str}, ThresholdHigh={threshold_high_str}, BreakoutLow={breakout_low_str}, ThresholdLow={threshold_low_str}, ATR={atr_val:.5f}")
            
            # --- Step 3: Signal creation ---
            if signal_direction:
                if entry_price is None:
                    self.logger.warning(f"[{self.name}/{sym}] No entry_price, skipping signal.")
                    continue
                min_stop_distance = 0.5 * atr_val
                if signal_direction == "BUY" and (stop_loss is None or abs(entry_price - stop_loss) < min_stop_distance):
                    stop_loss = entry_price - self.stop_loss_atr_multiplier * atr_val
                    self.logger.info(f"[{self.name}/{sym}] BUY - Enforcing minimum SL distance ({min_stop_distance:.5f}), new SL: {stop_loss:.5f}")
                elif signal_direction == "SELL" and (stop_loss is None or abs(entry_price - stop_loss) < min_stop_distance):
                    stop_loss = entry_price + self.stop_loss_atr_multiplier * atr_val
                    self.logger.info(f"[{self.name}/{sym}] SELL - Enforcing minimum SL distance ({min_stop_distance:.5f}), new SL: {stop_loss:.5f}")
                if take_profit is None or abs(entry_price - stop_loss) < 1e-9:
                    if signal_direction == "BUY":
                        take_profit = entry_price + abs(entry_price - stop_loss) * self.take_profit_rr_ratio
                    else:
                        take_profit = entry_price - abs(entry_price - stop_loss) * self.take_profit_rr_ratio
                    self.logger.info(f"[{self.name}/{sym}] Fallback TP used: {take_profit:.5f}")
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
                        self.logger.info(f"[{self.name}/{sym}] Position size calculated by RiskManager: {size:.4f} lots (Equity={account_equity}, Risk/Trade={self.risk_per_trade*100}%, SL pips={stop_loss_pips}, PipValue={pips_per_unit}, Price={instrument_price})")
                    except Exception as e:
                        self.logger.warning(f"[{self.name}/{sym}] RiskManager sizing failed: {e}. Defaulting size to 1.0.")
                        size = 1.0 # Fallback size
                
                # Final check for valid signal parameters
                if not all(v is not None for v in [entry_price, stop_loss, take_profit]):
                    logger.error(f"[{self.name}/{sym}] Critical error: Signal parameters are None before creation. EP={entry_price}, SL={stop_loss}, TP={take_profit}. Skipping signal.")
                    self.processed_bars[(sym, self.primary_timeframe)] = current_bar_ts
                    continue

                if abs(entry_price - stop_loss) > 1e-9: # Ensure stop loss is not at entry price
                    logger.info(f"📢 [{self.name}/{sym}] SIGNAL CREATED: Direction={signal_direction}, Entry={entry_price:.5f}, SL={stop_loss:.5f}, TP={take_profit:.5f}, Size={size:.4f}, R:R={risk_reward:.2f}, Reason: {reason}")
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
                    self.logger.warning(f"[{self.name}/{sym}] Signal skipped due to zero/invalid stop distance. Entry={entry_price}, Stop={stop_loss}")
            
            # Mark bar as processed even if no signal was generated to avoid redundant checks on the same bar
            if not signals or (signals and signals[-1]['symbol'] != sym): # If no signal for this symbol was added
                 self.processed_bars[(sym, self.primary_timeframe)] = current_bar_ts
                 logger.debug(f"[{self.name}/{sym}] Marked bar {current_bar_ts} as processed (no signal generated or signal for different symbol).")
            elif signals and signals[-1]['symbol'] == sym:
                 self.processed_bars[(sym, self.primary_timeframe)] = current_bar_ts
                 logger.debug(f"[{self.name}/{sym}] Marked bar {current_bar_ts} as processed (signal generated for this symbol).")


        self.require_retest = True # Reset retest requirement if it was changed for a specific signal
        logger.info(f"🏁 [{self.name}] Signal generation finished. Total signals: {len(signals)}")
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
        # Need at least 5 bars after breakout to detect retest
        if len(df) < 5:
            logger.debug(f"[{self.name}/{symbol}] Not enough bars ({len(df)}) for retest detection. Need at least 5.")
            return False
            
        # Get the most recent bars (after breakout)
        recent_bars_count = 5 # Define how many bars to check for retest
        recent_bars = df.iloc[-recent_bars_count:]
        logger.debug(f"[{self.name}/{symbol}] Detecting retest. Recent {recent_bars_count} bars: {recent_bars[['open', 'high', 'low', 'close']].to_dict('records')}")
        
        # Determine retest threshold based on ATR
        atr_col = f"ATR_{self.atr_period_consolidation}"
        atr = recent_bars[atr_col].iloc[-1] if atr_col in recent_bars.columns and not pd.isna(recent_bars[atr_col].iloc[-1]) else 0
        
        # Default threshold as percentage of breakout level
        threshold = breakout_level * self.retest_threshold_pct / 100
        
        # If ATR is available, use it to refine the threshold (min 0.25 ATR, max 1.0 ATR)
        if atr > 0:
            threshold = min(max(threshold, 0.25 * atr), 1.0 * atr) # Clamped ATR-based threshold
            logger.debug(f"[{self.name}/{symbol}] Retest threshold refined by ATR ({atr:.5f}): {threshold:.5f}")
        else:
            logger.warning(f"[{self.name}/{symbol}] ATR is zero or invalid for retest threshold calculation. Using percentage-based: {threshold:.5f}")
        
        # For upside breakout (BUY): price must pull back down to near the breakout level
        if direction == "BUY":
            # Initial breakout size (from breakout bar)
            initial_breakout_size = df['high'].iloc[-5] - breakout_level
            
            # Retest threshold - how close price must come back to the breakout level
            # Default: 50% of initial breakout size, but adjusted by parameter
            retest_zone_high = breakout_level + (initial_breakout_size * self.retest_threshold_pct)
            
            # Check if any of the recent bars pulled back to the retest zone
            lowest_low_in_retest_window = recent_bars['low'].min()
            retest_condition_met = lowest_low_in_retest_window <= retest_zone_high
            logger.debug(f"[{self.name}/{symbol}/BUY_RETEST] BreakoutLevel={breakout_level:.5f}, InitialBreakoutSize={initial_breakout_size:.5f}, RetestZoneHigh={retest_zone_high:.5f}, LowestLowInWindow={lowest_low_in_retest_window:.5f}, RetestConditionMet={retest_condition_met}")
            
            # Also check if price has moved back up after retest (confirmation)
            if retest_condition_met and len(recent_bars) >= 2:
                last_close = recent_bars['close'].iloc[-1]
                confirmation = last_close > lowest_low_in_retest_window  # Price moved back up after retest
                logger.debug(f"[{self.name}/{symbol}/BUY_RETEST] LastClose={last_close:.5f}, ConfirmationOfRetestBounce={confirmation}")
                return confirmation
            
            logger.debug(f"[{self.name}/{symbol}/BUY_RETEST] Retest condition not met or not enough bars for bounce confirmation.")
            return False
            
        # For downside breakout (SELL): price must pull back up to near the breakout level
        elif direction == "SELL":
            # Initial breakout size (from breakout bar)
            initial_breakout_size = breakout_level - df['low'].iloc[-5]
            
            # Retest threshold - how close price must come back to the breakout level
            # Default: 50% of initial breakout size, but adjusted by parameter
            retest_zone_low = breakout_level - (initial_breakout_size * self.retest_threshold_pct)
            
            # Check if any of the recent bars pulled back to the retest zone
            highest_high_in_retest_window = recent_bars['high'].max()
            retest_condition_met = highest_high_in_retest_window >= retest_zone_low
            logger.debug(f"[{self.name}/{symbol}/SELL_RETEST] BreakoutLevel={breakout_level:.5f}, InitialBreakoutSize={initial_breakout_size:.5f}, RetestZoneLow={retest_zone_low:.5f}, HighestHighInWindow={highest_high_in_retest_window:.5f}, RetestConditionMet={retest_condition_met}")
            
            # Also check if price has moved back down after retest (confirmation)
            if retest_condition_met and len(recent_bars) >= 2:
                last_close = recent_bars['close'].iloc[-1]
                confirmation = last_close < highest_high_in_retest_window  # Price moved back down after retest
                logger.debug(f"[{self.name}/{symbol}/SELL_RETEST] LastClose={last_close:.5f}, ConfirmationOfRetestRejection={confirmation}")
                return confirmation
            
            logger.debug(f"[{self.name}/{symbol}/SELL_RETEST] Retest condition not met or not enough bars for rejection confirmation.")
            return False
            
        return False 

    def _get_dynamic_atr_multiplier(self, df_subset: pd.DataFrame) -> float:
        """Return a dynamic ATR multiplier for breakout confirmation based on volatility of the provided DataFrame subset."""
        # Ensure df_subset is not empty and has the required columns and length
        if df_subset is None or df_subset.empty or len(df_subset) < self.atr_period_consolidation +1 : # Need enough data for ATR
            logger.warning(f"[{self.name}/_get_dynamic_atr_multiplier] DataFrame subset too small or None. Using default multiplier: {self.breakout_confirmation_atr_multiplier}")
            return self.breakout_confirmation_atr_multiplier

        atr_col = f"ATR_{self.atr_period_consolidation}"
        # Recalculate ATR on the subset if not present or to ensure it's up-to-date for this specific segment
        if atr_col not in df_subset.columns or df_subset[atr_col].isnull().all():
            df_subset = df_subset.copy() # Avoid SettingWithCopyWarning
            df_subset[atr_col] = calculate_atr(df_subset, self.atr_period_consolidation)
            logger.debug(f"[{self.name}/_get_dynamic_atr_multiplier] Recalculated ATR for subset. Last ATR: {df_subset[atr_col].iloc[-1] if not df_subset[atr_col].empty else 'N/A'}")


        last_atr = df_subset[atr_col].iloc[-1] if not df_subset[atr_col].empty and not pd.isna(df_subset[atr_col].iloc[-1]) else 0
        last_close = df_subset['close'].iloc[-1] if not df_subset['close'].empty and not pd.isna(df_subset['close'].iloc[-1]) else 0
        
        logger.debug(f"[{self.name}/_get_dynamic_atr_multiplier] Subset Last ATR={last_atr}, Last Close={last_close}")

        if last_atr > 0 and last_close > 0:
            volatility_ratio = last_atr / last_close
            logger.debug(f"[{self.name}/_get_dynamic_atr_multiplier] Subset VolatilityRatio={volatility_ratio:.4f}")
            
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