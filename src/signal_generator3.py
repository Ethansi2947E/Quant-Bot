import asyncio
from typing import Dict, Optional, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import json
import traceback
import sys
import os
from loguru import logger

from src.mtf_analysis import MTFAnalysis
from src.divergence_analysis import DivergenceAnalysis
from src.poi_detector import POIDetector
from src.risk_manager import RiskManager
from src.market_analysis import MarketAnalysis
from src.smc_analysis import SMCAnalysis
from src.volume_analysis import VolumeAnalysis
from src.mt5_handler import MT5Handler

# Create logs directory if it doesn't exist
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Configure loguru logger
logger_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>SG3:{function}:{line}</cyan> | <level>{message}</level>"

# Configure loguru with custom format (no need to call remove() as we'll use a fresh logger)
# Add file handler
logger.configure(handlers=[
    {"sink": sys.stdout, "format": logger_format, "level": "INFO", "colorize": True},
    {"sink": os.path.join(log_dir, "signal_generator3_detailed.log"), 
     "format": logger_format, "level": "DEBUG", "rotation": "10 MB", 
     "retention": 5, "compression": "zip"}
])

# Add context to differentiate this logger
logger = logger.bind(name="signal_generator3")

logger.info("[SG3] SignalGenerator3 logger initialized")
logger.info(f"[SG3] Detailed logs will be written to {os.path.join(log_dir, 'signal_generator3_detailed.log')}")

# Add a custom JSON encoder class to handle Pandas Timestamp objects
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.Timestamp):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(obj, np.bool_):
            return bool(obj)  # Convert numpy.bool_ to Python bool
        elif isinstance(obj, np.integer):
            return int(obj)  # Convert numpy int types to Python int
        elif isinstance(obj, np.floating):
            return float(obj)  # Convert numpy float types to Python float
        elif isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert numpy arrays to Python lists
        elif hasattr(obj, 'to_json'):
            return obj.to_json()
        elif hasattr(obj, 'dtype') and hasattr(obj, 'item'):
            return obj.item()  # Handle other numpy scalar types with .item()
        return super().default(obj)

class SignalGenerator3:
    """
    Signal Generator 3: Implements an institutional-grade trading strategy with
    comprehensive market analysis and relaxed validation rules for more frequent signals.
    """
    
    def __init__(self, mt5_handler: Optional[MT5Handler] = None, risk_manager: Optional[RiskManager] = None, debug_mode: bool = False):
        """Initialize SignalGenerator3 with analysis components."""
        logger.info("[INIT] Initializing SignalGenerator3 with strategy components")
        
        self.mt5_handler = mt5_handler
        self.market_analysis = MarketAnalysis()
        self.divergence_analysis = DivergenceAnalysis()
        self.mtf_analysis = MTFAnalysis()
        self.poi_detector = POIDetector()
        self.smc_analysis = SMCAnalysis()
        self.volume_analysis = VolumeAnalysis()
        self.risk_manager = risk_manager if risk_manager else RiskManager()
        self.debug_mode = debug_mode
        
        # Relaxed configuration parameters
        self.min_volume_momentum = 0.3  # Reduced from 0.5
        self.min_confidence = 0.4  # Reduced from 0.5
        self.risk_per_trade = 0.01  # 1% risk per trade
        self.min_rr_ratio = 1.5  # Reduced from 2.0
        
        # New parameters for flexible validation
        self.require_liquidity_sweeps = False  # Make liquidity sweeps optional
        self.min_poi_zones = 1  # Minimum number of POI zones required
        self.ignore_divergence = True  # Make divergence check optional
        self.volatility_threshold = 0.15  # Reduced from 0.2
        
        # Price level validation parameters
        self.max_stop_distance_percent = 0.8  # Increased from 0.5
        self.min_stop_distance_percent = 0.08  # Reduced from 0.1
        self.price_precision = 5  # Default price precision
        
        # ATR standardization
        self.atr_period = 14  # Standard ATR period for all calculations
        
        logger.info(f"Configuration: min_volume_momentum={self.min_volume_momentum}, " 
                    f"min_confidence={self.min_confidence}, risk_per_trade={self.risk_per_trade}, "
                    f"min_rr_ratio={self.min_rr_ratio}, atr_period={self.atr_period}")
        logger.info(f"Validation settings: require_liquidity_sweeps={self.require_liquidity_sweeps}, "
                    f"min_poi_zones={self.min_poi_zones}, ignore_divergence={self.ignore_divergence}, "
                    f"volatility_threshold={self.volatility_threshold}")
        logger.info(f"Price validation: max_stop_distance_percent={self.max_stop_distance_percent}%, " 
                    f"min_stop_distance_percent={self.min_stop_distance_percent}%")

    def _validate_price_levels(self, symbol: str, current_price: float, stop_loss: float, 
                             direction: str, risk_distance: float) -> tuple[bool, str]:
        """Validate price levels for trade setup."""
        logger.info(f"[{symbol}] Validating price levels for {direction} trade")
        try:
            # Calculate price distances as percentages
            stop_distance_percent = abs(current_price - stop_loss) / current_price * 100
            
            logger.debug(f"[{symbol}] Price level details - Direction: {direction}")
            logger.debug(f"[{symbol}] Current price: {current_price:.5f}, Stop loss: {stop_loss:.5f}")
            logger.debug(f"[{symbol}] Risk distance: {risk_distance:.5f} pips, ({stop_distance_percent:.2f}%)")
            
            # Basic price level checks
            if current_price <= 0 or stop_loss <= 0:
                logger.warning(f"[{symbol}] Invalid price levels - Current: {current_price}, Stop: {stop_loss}")
                return False, "Invalid price levels (zero or negative)"
            
            # Direction-specific validation
            if direction == "BUY":
                if stop_loss >= current_price:
                    logger.warning(f"[{symbol}] Invalid stop loss for BUY: Stop ({stop_loss:.5f}) >= Entry ({current_price:.5f})")
                    return False, f"Invalid stop loss for BUY ({stop_loss:.5f} >= {current_price:.5f})"
            elif direction == "SELL":
                if stop_loss <= current_price:
                    logger.warning(f"[{symbol}] Invalid stop loss for SELL: Stop ({stop_loss:.5f}) <= Entry ({current_price:.5f})")
                    return False, f"Invalid stop loss for SELL ({stop_loss:.5f} <= {current_price:.5f})"
            
            # Stop distance validation
            if stop_distance_percent > self.max_stop_distance_percent:
                logger.warning(f"[{symbol}] Stop loss too far: {stop_distance_percent:.2f}% > {self.max_stop_distance_percent}%")
                return False, f"Stop loss too far ({stop_distance_percent:.2f}% > {self.max_stop_distance_percent}%)"
            if stop_distance_percent < self.min_stop_distance_percent:
                logger.warning(f"[{symbol}] Stop loss too close: {stop_distance_percent:.2f}% < {self.min_stop_distance_percent}%")
                return False, f"Stop loss too close ({stop_distance_percent:.2f}% < {self.min_stop_distance_percent}%)"
            
            logger.info(f"[{symbol}] Price levels validated successfully - Distance: {stop_distance_percent:.2f}%")
            return True, "Price levels valid"
            
        except Exception as e:
            logger.error(f"[{symbol}] Error validating price levels: {str(e)}")
            logger.debug(f"[{symbol}] Price validation exception: {traceback.format_exc()}")
            return False, f"Price validation error: {str(e)}"
    

    def _get_default_stop_loss(self, df: pd.DataFrame, direction: str, current_price: float) -> float:
        """Calculate a default stop loss when swing points are not available."""
        lookback = 20  # Look back period
        logger.info(f"Calculating default stop loss for {direction} trade using {lookback} bars lookback")
        try:
            # Use recent price action to determine stop loss
            recent_data = df.iloc[-lookback:]
            
            logger.debug(f"Recent price range: High={recent_data['high'].max():.5f}, Low={recent_data['low'].min():.5f}")
            logger.debug(f"Current price: {current_price:.5f}")
            
            if direction == 'up':
                # For buy signals, use recent low with buffer
                raw_stop = recent_data['low'].min()
                buffer_factor = 0.999  # 0.1% buffer
                stop_loss = raw_stop * buffer_factor
                logger.debug(f"BUY stop calculation: Recent low={raw_stop:.5f} -> {buffer_factor} = {stop_loss:.5f}")
            else:
                # For sell signals, use recent high with buffer
                raw_stop = recent_data['high'].max()
                buffer_factor = 1.001  # 0.1% buffer
                stop_loss = raw_stop * buffer_factor
                logger.debug(f"SELL stop calculation: Recent high={raw_stop:.5f} -> {buffer_factor} = {stop_loss:.5f}")
            
            # Ensure minimum distance from current price
            min_distance = current_price * self.min_stop_distance_percent / 100
            logger.debug(f"Minimum required stop distance: {min_distance:.5f}")
            
            if direction == 'up':
                adjusted_stop = min(stop_loss, current_price - min_distance)
                if adjusted_stop != stop_loss:
                    logger.debug(f"Adjusting stop loss to ensure minimum distance: {stop_loss:.5f} -> {adjusted_stop:.5f}")
                stop_loss = adjusted_stop
            else:
                adjusted_stop = max(stop_loss, current_price + min_distance)
                if adjusted_stop != stop_loss:
                    logger.debug(f"Adjusting stop loss to ensure minimum distance: {stop_loss:.5f} -> {adjusted_stop:.5f}")
                stop_loss = adjusted_stop
            
            logger.info(f"Default stop loss calculated: {stop_loss:.5f} ({abs(current_price - stop_loss) / current_price * 100:.2f}% from price)")
            return stop_loss
            
        except Exception as e:
            logger.error(f"Error calculating default stop loss: {str(e)}")
            logger.debug(f"Default stop loss exception: {traceback.format_exc()}")
            
            # Fallback to percentage-based stop loss
            fallback_stop = current_price * (0.997 if direction == 'up' else 1.003)
            logger.warning(f"Using fallback stop loss calculation: {fallback_stop:.5f}")
            return fallback_stop
        
    def _calculate_stop_loss(self, df: pd.DataFrame, turtle_soup_pattern: Dict) -> float:
        """
        Calculate stop loss based on the false breakout candle.
        For long trades: below the low of the breakout candle.
        For short trades: above the high of the breakout candle.
        """
        logger.info(f"Calculating Turtle Soup stop loss based on breakout candle")
        try:
            breakout_candle_idx = turtle_soup_pattern['breakout_candle']
            breakout_candle = df.loc[breakout_candle_idx]
            pattern_type = turtle_soup_pattern['type']  # 'long' or 'short'
            
            logger.debug(f"Breakout candle details: Index={breakout_candle_idx}, " 
                         f"OHLC=[{breakout_candle['open']:.5f}, {breakout_candle['high']:.5f}, "
                         f"{breakout_candle['low']:.5f}, {breakout_candle['close']:.5f}]")
            logger.debug(f"Pattern type: {pattern_type.upper()}, Breakout level: {turtle_soup_pattern['level']:.5f}")
            
            # Calculate distance from current price to breakout level
            current_price = df['close'].iloc[-1]
            breakout_level = turtle_soup_pattern['level']
            distance_to_level = abs(current_price - breakout_level)
            percent_distance = (distance_to_level / current_price) * 100
            
            logger.debug(f"Current price: {current_price:.5f}, Distance to breakout level: {distance_to_level:.5f} ({percent_distance:.2f}%)")
            
            # Get the symbol to apply specific adjustments
            symbol = df.iloc[-1].get('symbol', '')
            
            # More flexible buffer sizing based on price and volatility
            # Calculate ATR using standardized method with self.atr_period instead of hardcoded 10
            recent_atr = self._calculate_atr(df)
            
            # Apply symbol-specific buffer adjustments
            buffer_multiplier = 1.0
            
            if 'XAU' in symbol:  # Gold
                buffer_multiplier = 2.0
            elif 'XAG' in symbol:  # Silver
                buffer_multiplier = 2.5
            elif 'JPY' in symbol:  # JPY pairs
                buffer_multiplier = 1.5
            
            # Calculate dynamic buffer (15% of ATR, adjusted by symbol)
            dynamic_buffer = recent_atr * 0.15 * buffer_multiplier
            
            # Ensure minimum buffer based on price
            min_buffer = current_price * 0.0005  # 0.05% minimum buffer
            buffer = max(dynamic_buffer, min_buffer)
            
            logger.debug(f"Stop loss buffer calculation: ATR={recent_atr:.5f}, Multiplier={buffer_multiplier}")
            logger.debug(f"Dynamic buffer={dynamic_buffer:.5f}, Minimum buffer={min_buffer:.5f}")
            logger.debug(f"Final buffer={buffer:.5f} ({buffer/current_price*100:.3f}% of price)")
            
            if pattern_type == 'long':
                # For long trade, place stop below the breakout candle's low (with variable buffer)
                is_full_breakout = turtle_soup_pattern.get('is_full_breakout', True)
                
                if is_full_breakout:
                    # For full breakouts, use the breakout candle's low
                    stop_basis = breakout_candle['low']
                    logger.debug(f"Using full breakout candle low as stop basis: {stop_basis:.5f}")
                else:
                    # For near breakouts, use the breakout level
                    stop_basis = turtle_soup_pattern['level'] * 0.999  # Slightly below level
                    logger.debug(f"Using breakout level as stop basis (near-breakout): {stop_basis:.5f}")
                
                stop_loss = stop_basis - buffer
                
                logger.debug(f"LONG stop calculation:")
                logger.debug(f"  - Base level: {stop_basis:.5f}")
                logger.debug(f"  - Buffer amount: -{buffer:.5f}")
                logger.debug(f"  - Final stop loss: {stop_loss:.5f}")
                
                # Calculate stop distance metrics
                stop_distance = current_price - stop_loss
                stop_percent = (stop_distance / current_price) * 100
                logger.info(f"LONG stop loss distance: {stop_distance:.5f} ({stop_percent:.2f}%)")
                
                # Adjust stop if it's too wide
                max_stop_percent = 0.8  # Increased from 0.5% to 0.8%
                if stop_percent > max_stop_percent:
                    old_stop = stop_loss
                    stop_loss = current_price * (1 - max_stop_percent/100)
                    logger.info(f"Stop loss too wide ({stop_percent:.2f}%), adjusted from {old_stop:.5f} to {stop_loss:.5f}")
                
            elif pattern_type == 'short':
                # For short trade, place stop above the breakout candle's high (with variable buffer)
                is_full_breakout = turtle_soup_pattern.get('is_full_breakout', True)
                
                if is_full_breakout:
                    # For full breakouts, use the breakout candle's high
                    stop_basis = breakout_candle['high']
                    logger.debug(f"Using full breakout candle high as stop basis: {stop_basis:.5f}")
                else:
                    # For near breakouts, use the breakout level
                    stop_basis = turtle_soup_pattern['level'] * 1.001  # Slightly above level
                    logger.debug(f"Using breakout level as stop basis (near-breakout): {stop_basis:.5f}")
                
                stop_loss = stop_basis + buffer
                
                logger.debug(f"SHORT stop calculation:")
                logger.debug(f"  - Base level: {stop_basis:.5f}")
                logger.debug(f"  - Buffer amount: +{buffer:.5f}")
                logger.debug(f"  - Final stop loss: {stop_loss:.5f}")
                
                # Calculate stop distance metrics
                stop_distance = stop_loss - current_price
                stop_percent = (stop_distance / current_price) * 100
                logger.info(f"SHORT stop loss distance: {stop_distance:.5f} ({stop_percent:.2f}%)")
                
                # Adjust stop if it's too wide
                max_stop_percent = 0.8  # Increased from 0.5% to 0.8%
                if stop_percent > max_stop_percent:
                    old_stop = stop_loss
                    stop_loss = current_price * (1 + max_stop_percent/100)
                    logger.info(f"Stop loss too wide ({stop_percent:.2f}%), adjusted from {old_stop:.5f} to {stop_loss:.5f}")
                
            else:
                logger.warning(f"Unknown pattern type: {pattern_type}")
                return None
            
            # Check if stop distance is reasonable
            max_stop_percent = 0.8  # Increased from 0.5% to 0.8%
            min_stop_percent = 0.05  # Reduced from 0.1% to 0.05%
            
            stop_percent = (abs(current_price - stop_loss) / current_price) * 100
            logger.debug(f"Final stop distance check: {stop_percent:.2f}% (Min: {min_stop_percent}%, Max: {max_stop_percent}%)")
            
            # Ensure minimum stop distance
            if pattern_type == 'long' and stop_percent < min_stop_percent:
                old_stop = stop_loss
                stop_loss = current_price * (1 - min_stop_percent/100)
                logger.info(f"Stop loss too tight ({stop_percent:.2f}%), adjusted from {old_stop:.5f} to {stop_loss:.5f}")
            elif pattern_type == 'short' and stop_percent < min_stop_percent:
                old_stop = stop_loss
                stop_loss = current_price * (1 + min_stop_percent/100)
                logger.info(f"Stop loss too tight ({stop_percent:.2f}%), adjusted from {old_stop:.5f} to {stop_loss:.5f}")
            
            logger.info(f"Turtle Soup stop loss calculated: {stop_loss:.5f}")
            return stop_loss
            
        except Exception as e:
            logger.error(f"Error calculating Turtle Soup stop loss: {str(e)}")
            logger.error(f"Stop loss calculation exception: {traceback.format_exc()}")
            
            # Fallback to simple percentage-based stop loss
            current_price = df['close'].iloc[-1]
            pattern_type = turtle_soup_pattern.get('type', 'unknown')
            
            if pattern_type == 'long':
                stop_loss = current_price * 0.995  # 0.5% below current price
            else:
                stop_loss = current_price * 1.005  # 0.5% above current price
                
            logger.warning(f"Using fallback stop loss due to error: {stop_loss:.5f}")
            return stop_loss

    def _calculate_atr(self, df: pd.DataFrame) -> float:
        """
        Calculate ATR using standard period.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            float: ATR value
        """
        try:
            # Use the standard period defined in the class
            lookback = min(self.atr_period, len(df) - 1)
            
            # Calculate true ranges
            true_ranges = []
            for i in range(1, lookback + 1):
                idx = -i
                high = df['high'].iloc[idx]
                low = df['low'].iloc[idx]
                prev_close = df['close'].iloc[idx-1]
                true_range = max(high-low, abs(high-prev_close), abs(low-prev_close))
                true_ranges.append(true_range)
            
            # Return mean of true ranges
            atr = np.mean(true_ranges) if true_ranges else 0
            return atr
            
        except Exception as e:
            logger.error(f"Error calculating ATR: {str(e)}")
            # Fallback to high-low range
            if len(df) > 1:
                return (df['high'].max() - df['low'].min()) / len(df)
            return 0

    def _detect_turtle_soup(self, df: pd.DataFrame, lookback: int = 20, confirmation_bars: int = 3) -> Optional[Dict]:
        """
        Detect Turtle Soup patterns with more lenient conditions.
        Dynamically adjusts lookback period based on timeframe and ATR.
        Returns a dictionary with pattern details or None if no pattern is found.
        """
        # Get timeframe from dataframe if available
        timeframe = df.iloc[-1].get('timeframe', '') if 'timeframe' in df.columns else ''
        
        # Adjust lookback dynamically based on timeframe
        if timeframe == '15m':
            # Use adjusted lookback for 15-minute timeframe
            lookback = 30  # Increased from 20 to detect more patterns
        elif timeframe == '5m':
            lookback = 40  # More lookback for faster timeframes
        elif timeframe == '1h':
            lookback = 15  # Fewer bars for slower timeframes
        elif timeframe == '4h':
            lookback = 10  # Even fewer for very slow timeframes
            
        logger.info(f"SCANNING FOR TURTLE SOUP PATTERNS (lookback={lookback}, confirmation={confirmation_bars}, timeframe={timeframe})")
        try:
            # Identify the lowest low and highest high in the lookback period
            lowest_low = df['low'].iloc[-lookback:].min()
            highest_high = df['high'].iloc[-lookback:].max()
            
            # Add more detailed logging of price levels
            logger.debug(f"Price range in lookback period: Low={lowest_low:.5f}, High={highest_high:.5f}, Range={highest_high-lowest_low:.5f}")
            logger.debug(f"Current candle - Low: {df['low'].iloc[-1]:.5f}, High: {df['high'].iloc[-1]:.5f}, Close: {df['close'].iloc[-1]:.5f}")
            
            # Calculate ATR using standardized method
            atr = self._calculate_atr(df)
            
            # Calculate momentum for directional bias (3-candle ROC)
            momentum = 0
            if len(df) >= 3:
                momentum = (df['close'].iloc[-1] / df['close'].iloc[-3] - 1) * 100  # 3-candle momentum
                logger.debug(f"Current momentum (3-candle ROC): {momentum:.2f}%")
            
            # Symbol-specific thresholds - higher for volatile instruments
            symbol = df.iloc[-1].get('symbol', '')
            multiplier = 1.0
            if 'XAU' in symbol:  # Gold
                multiplier = 2.0
            elif 'XAG' in symbol:  # Silver
                multiplier = 2.5
            elif 'JPY' in symbol:  # JPY pairs
                multiplier = 1.5
            elif any(s in symbol for s in ['NAS', 'SPX', 'US30']):  # Indices
                multiplier = 1.8
            
            # Calculate threshold with symbol-specific multiplier
            base_threshold_percent = 0.25  # Increased from 0.15
            threshold = atr * base_threshold_percent * multiplier
            
            # Ensure minimum threshold based on price and ATR
            current_price = df['close'].iloc[-1]
            min_threshold = max(atr * 0.25, current_price * 0.001)  # Scale with ATR or use 0.1% minimum
            threshold = max(threshold, min_threshold)
            
            logger.debug(f"ATR: {atr:.5f}, Base threshold: {atr * base_threshold_percent:.5f}, Symbol multiplier: {multiplier}")
            logger.debug(f"Final threshold: {threshold:.5f}, Threshold as percent of price: {threshold/current_price*100:.2f}%")
            
            # Log the key breakout levels we're looking for
            logger.debug(f"Looking for false breakouts below {lowest_low:.5f} (LONG setup) or above {highest_high:.5f} (SHORT setup)")
            
            # NEW: Track indices of breakout candles for recency validation
            breakout_candidates = []
            
            # Check for false breakouts in the last 6 candles instead of just 5
            for i in range(1, min(7, len(df))):
                check_candle = df.iloc[-i]
                
                logger.debug(f"Checking candle at position -{i}: Open={check_candle['open']:.5f}, High={check_candle['high']:.5f}, Low={check_candle['low']:.5f}, Close={check_candle['close']:.5f}")
                
                # False breakout to the downside (for long trades)
                # Widen near-breakout tolerance from 0.05% to 0.1% or ATR-based
                near_breakout_tolerance = max(0.001, atr * 0.1 / lowest_low)  # 0.1% or ATR-based
                if check_candle['low'] <= lowest_low * (1 + near_breakout_tolerance):
                    full_breakout = check_candle['low'] < lowest_low
                    
                    # Allow more near-breakouts to be considered
                    if not full_breakout:
                        logger.debug(f"Near-breakout at {check_candle['low']:.5f} - using relaxed criteria")
                    
                    breakout_type = "FULL BREAKOUT" if full_breakout else "NEAR BREAKOUT"
                    logger.info(f"POTENTIAL DOWNSIDE {breakout_type} DETECTED (candle -{i})")
                    logger.debug(f"   Price {'broke below' if full_breakout else 'approached'} {lowest_low:.5f} "
                               f"by {abs(lowest_low - check_candle['low']):.5f}")
                    
                    # Use wider close threshold
                    close_threshold = threshold * 1.2  # Increased from 1.0
                    if check_candle['close'] > lowest_low + close_threshold:  # Must close above level + threshold
                        logger.info(f"   Close {check_candle['close']:.5f} exceeds threshold ({lowest_low + close_threshold:.5f})")
                        
                        # Require less reversal evidence
                        reversal_strength = 0
                        for j in range(1, confirmation_bars + 1):
                            if i - j >= 0:  # Ensure we don't go beyond available data
                                reversal_candle = df.iloc[-(i-j)]
                                logger.debug(f"   Checking reversal candle at position -{i-j}: Close={reversal_candle['close']:.5f}")
                                if reversal_candle['close'] > check_candle['close'] + threshold * 0.4:  # Reduced from 0.5
                                    logger.debug(f"   Strong reversal evidence: candle close {reversal_candle['close']:.5f} > breakout close + threshold {check_candle['close'] + threshold * 0.4:.5f}")
                                    reversal_strength += 1
                        
                        # Check for a bullish candle with relaxed criteria
                        bullish_candle = check_candle['close'] > check_candle['open'] + threshold * 0.25  # Reduced from 0.3
                        logger.debug(f"   Bullish candle check: {bullish_candle} (close-open: {check_candle['close'] - check_candle['open']:.5f}, threshold: {threshold * 0.25:.5f})")
                        
                        # Only proceed if we have sufficient reversal evidence or a bullish candle
                        if reversal_strength >= 1 or bullish_candle:
                            logger.debug(f"   Reversal evidence found: Strong reversal = {reversal_strength}, Bullish candle = {bullish_candle}")
                            
                            # Lower the base score as requested
                            pattern_score = 2  # Reduced from 3 to be more lenient
                            if reversal_strength >= 1:
                                pattern_score += 1  # Higher score with reversal evidence
                            if bullish_candle:
                                pattern_score += 1  # Higher score for bullish candle
                            
                            # Check for positive momentum for long setups
                            momentum_aligned = momentum > 0.1  # Reduced from 0.2
                            if momentum_aligned:
                                pattern_score += 1  # Bonus for aligned momentum
                                logger.debug(f"   Momentum aligned for LONG: {momentum:.2f}% > 0.1%, adding score point")
                            
                            # Calculate how much the pattern has already moved
                            current_close = df['close'].iloc[-1]
                            pattern_range = abs(current_close - lowest_low)
                            range_percent = (pattern_range / lowest_low) * 100
                            
                            # NEW: Track this candidate for later recency validation
                            breakout_candidates.append({
                                'type': 'long',
                                'breakout_candle': df.index[-i],
                                'level': lowest_low,
                                'threshold': threshold,
                                'confirmation_close': check_candle['close'],
                                'score': pattern_score,
                                'is_full_breakout': bool(full_breakout),
                                'pattern_age': i,
                                'played_out_percent': range_percent,
                                'momentum': momentum,
                                'momentum_aligned': momentum_aligned
                            })
                
                # False breakout to the upside (for short trades)
                # Widen near-breakout tolerance from 0.05% to 0.1% or ATR-based
                near_breakout_tolerance = max(0.001, atr * 0.1 / highest_high)  # 0.1% or ATR-based
                if check_candle['high'] >= highest_high * (1 - near_breakout_tolerance):
                    full_breakout = check_candle['high'] > highest_high
                    
                    # Allow more near-breakouts to be considered
                    wick_rejection = check_candle['high'] > highest_high and check_candle['close'] < check_candle['high'] - threshold * 0.4  # Upper wick rejection, reduced from 0.5
                    
                    # Allow more near-breakouts with relaxed criteria
                    if not full_breakout and not wick_rejection:
                        logger.debug(f"Near-breakout at {check_candle['high']:.5f} - using relaxed criteria")
                    
                    breakout_type = "FULL BREAKOUT" if full_breakout else ("WICK REJECTION" if wick_rejection else "NEAR BREAKOUT")
                    logger.info(f"POTENTIAL UPSIDE {breakout_type} DETECTED (candle -{i})")
                    logger.debug(f"   Price {'broke above' if full_breakout else 'rejected at'} {highest_high:.5f} "
                               f"by {abs(check_candle['high'] - highest_high):.5f}")
                    
                    # Use wider close threshold
                    close_threshold = threshold * 1.2  # Increased from 1.0
                    if check_candle['close'] < highest_high - close_threshold or wick_rejection:  # Must close below level - threshold or show wick rejection
                        logger.info(f"   Close {check_candle['close']:.5f} below threshold ({highest_high - close_threshold:.5f}) or wick rejection: {wick_rejection}")
                        
                        # Require less reversal evidence
                        reversal_strength = 0
                        for j in range(1, confirmation_bars + 1):
                            if i - j >= 0:  # Ensure we don't go beyond available data
                                reversal_candle = df.iloc[-(i-j)]
                                logger.debug(f"   Checking reversal candle at position -{i-j}: Close={reversal_candle['close']:.5f}")
                                if reversal_candle['close'] < check_candle['close'] - threshold * 0.4:  # Reduced from 0.5
                                    logger.debug(f"   Strong reversal evidence: candle close {reversal_candle['close']:.5f} < breakout close - threshold {check_candle['close'] - threshold * 0.4:.5f}")
                                    reversal_strength += 1
                        
                        # Check for a bearish candle with relaxed criteria
                        bearish_candle = check_candle['close'] < check_candle['open'] - threshold * 0.25  # Reduced from 0.3
                        logger.debug(f"   Bearish candle check: {bearish_candle} (open-close: {check_candle['open'] - check_candle['close']:.5f}, threshold: {threshold * 0.25:.5f})")
                        logger.debug(f"   Wick rejection check: {wick_rejection} (upper wick: {check_candle['high'] - check_candle['close']:.5f})")
                        
                        # Allow more patterns with relaxed criteria
                        if reversal_strength >= 1 or bearish_candle or wick_rejection:
                            logger.debug(f"   Reversal evidence found: Strong reversal = {reversal_strength}, Bearish candle = {bearish_candle}, Wick rejection = {wick_rejection}")
                            
                            # Lower the base score as requested
                            pattern_score = 2  # Reduced from 3 to be more lenient
                            if reversal_strength >= 1:
                                pattern_score += 1  # Higher score with reversal evidence
                            if bearish_candle:
                                pattern_score += 1  # Reduced from +2
                            if wick_rejection:
                                pattern_score += 1  # Higher score for wick rejection
                            
                            # Check for negative momentum for short setups with relaxed criteria
                            momentum_aligned = momentum < -0.03  # Reduced from -0.05
                            if momentum_aligned:
                                pattern_score += 1  # Bonus for aligned momentum
                                logger.debug(f"   Momentum aligned for SHORT: {momentum:.2f}% < -0.03%, adding score point")
                            
                            # Calculate how much the pattern has already moved
                            current_close = df['close'].iloc[-1]
                            pattern_range = abs(current_close - highest_high)
                            range_percent = (pattern_range / highest_high) * 100
                            
                            # NEW: Track this candidate for later recency validation
                            breakout_candidates.append({
                                'type': 'short',
                                'breakout_candle': df.index[-i],
                                'level': highest_high,
                                'threshold': threshold,
                                'confirmation_close': check_candle['close'],
                                'score': pattern_score,
                                'is_full_breakout': bool(full_breakout),
                                'pattern_age': i,
                                'played_out_percent': range_percent,
                                'wick_rejection': wick_rejection,
                                'momentum': momentum,
                                'momentum_aligned': momentum_aligned
                            })
            
            # NEW: If we have multiple candidates, select the best one
            if breakout_candidates:
                # First, filter out patterns that have played out too much
                # For crypto, this is higher (0.7%) than for forex (0.5%)
                is_crypto = 'USD' in symbol and ('BTC' in symbol or 'ETH' in symbol)
                max_played_out = 0.8 if is_crypto else 0.6  # Increased from 0.7/0.5 to allow more signals
                
                valid_candidates = [c for c in breakout_candidates if c['played_out_percent'] <= max_played_out]
                
                if valid_candidates:
                    logger.info(f"Found {len(valid_candidates)} valid pattern candidates that haven't played out too much")
                    
                    # Sort by score (higher is better) and then by age (newer is better)
                    valid_candidates.sort(key=lambda x: (-x['score'], x['pattern_age']))
                    
                    best_candidate = valid_candidates[0]
                    logger.info(f"Selected best pattern: {best_candidate['type'].upper()} with score {best_candidate['score']}/5, played out: {best_candidate['played_out_percent']:.2f}%")
                    logger.info(f"   Breakout and reversal at level: {best_candidate['level']:.5f}")
                    logger.debug(f"Pattern details: {json.dumps(best_candidate, default=str)}")
                    return best_candidate
                else:
                    logger.info(f"All {len(breakout_candidates)} pattern candidates have played out too much (>{max_played_out}%)")
                    for c in breakout_candidates:
                        logger.debug(f"Rejected {c['type']} pattern (played out {c['played_out_percent']:.2f}%)")
                    return None
            
            logger.debug("No Turtle Soup pattern detected in current price action")
            return None
            
        except Exception as e:
            logger.error(f"ERROR detecting Turtle Soup pattern: {str(e)}")
            logger.error(f"Exception traceback: {traceback.format_exc()}")
            return None

    def _calculate_fibonacci_levels(self, swing_high: float, swing_low: float) -> Dict[str, float]:
        """
        Calculate Fibonacci retracement levels between swing high and swing low.
        Returns a dictionary with key levels (38.2%, 50%, 61.8%).
        """
        logger.info(f"Calculating Fibonacci retracement levels between swing high={swing_high:.5f} and swing low={swing_low:.5f}")
        try:
            # Validate swing points
            if swing_high <= swing_low:
                logger.warning(f"Invalid swing points: high ({swing_high:.5f}) <= low ({swing_low:.5f})")
                logger.info("Using default range calculation")
                # Calculate a default range based on recent volatility
                range_size = abs(swing_high - swing_low)
                logger.debug(f"Original range size: {range_size:.5f}")
                
                if range_size == 0:
                    range_size = swing_high * 0.01  # 1% of price if range is zero
                    logger.debug(f"Range was zero, set to 1% of price: {range_size:.5f}")
                
                swing_high = max(swing_high, swing_low) + (range_size * 0.5)
                swing_low = min(swing_high, swing_low) - (range_size * 0.5)
                logger.info(f"Adjusted range - High: {swing_high:.5f}, Low: {swing_low:.5f}")
                logger.debug(f"New range size: {swing_high - swing_low:.5f}")
            
            diff = swing_high - swing_low
            logger.debug(f"Price range size: {diff:.5f}, Percentage of price: {(diff/swing_high*100):.2f}%")
            
            # Standard Fibonacci calculations
            fib_levels = {
                '23.6%': swing_high - (diff * 0.236),
                '38.2%': swing_high - (diff * 0.382),
                '50%': swing_high - (diff * 0.5),
                '61.8%': swing_high - (diff * 0.618),
                '78.6%': swing_high - (diff * 0.786)
            }
            
            # Identify current trend to weight Fibonacci levels appropriately
            # We'll need the DataFrame to determine the trend direction
            # For now, we'll use a placeholder that will be updated when _identify_trend is called
            current_trend = getattr(self, 'current_trend', 'neutral')
            
            # Create a weighted set of key levels based on trend bias
            key_levels = {}
            
            if current_trend == 'bearish':
                # In a bearish trend, favor shallower retracements (38.2%)
                key_levels = {
                    '38.2%': fib_levels['38.2%'],
                    '50%': fib_levels['50%'],
                    '61.8%': fib_levels['61.8%']
                }
                logger.debug(f"Using bearish-biased Fibonacci levels with 38.2% weighted higher")
            elif current_trend == 'bullish':
                # In a bullish trend, favor deeper retracements (61.8%)
                key_levels = {
                    '38.2%': fib_levels['38.2%'],
                    '50%': fib_levels['50%'],
                    '61.8%': fib_levels['61.8%']
                }
                logger.debug(f"Using bullish-biased Fibonacci levels with 61.8% weighted higher")
            else:
                # Neutral trend, use standard levels
                key_levels = {
                    '38.2%': fib_levels['38.2%'],
                    '50%': fib_levels['50%'],
                    '61.8%': fib_levels['61.8%']
                }
                logger.debug(f"Using standard Fibonacci levels (neutral trend)")
            
            return key_levels
            
        except Exception as e:
            logger.error(f"Error calculating Fibonacci levels: {str(e)}")
            logger.error(f"Fibonacci calculation exception: {traceback.format_exc()}")
            
            # Return default levels based on midpoint
            midpoint = (swing_high + swing_low) / 2
            range_size = abs(swing_high - swing_low) or midpoint * 0.01  # Use 1% if range is zero
            
            default_levels = {
                '38.2%': midpoint + (range_size * 0.382),
                '50%': midpoint,
                '61.8%': midpoint - (range_size * 0.382)
            }
            
            logger.warning(f"Using default Fibonacci levels based on midpoint {midpoint:.5f}")
            logger.debug(f"Default fib levels: 38.2%={default_levels['38.2%']:.5f}, 50%={default_levels['50%']:.5f}, 61.8%={default_levels['61.8%']:.5f}")
            return default_levels
        
    def _confirm_reversal_candle(self, df: pd.DataFrame, fib_level: float, direction: str) -> bool:
        """
        Confirm entry with a more lenient reversal candle pattern at the Fibonacci level.
        Simplified conditions allow confirmation with just a directional candle if near level.
        Returns True if a valid pattern is detected.
        """
        logger.info(f"Checking for reversal candle confirmation at fib level {fib_level:.5f} for {direction} direction")
        try:
            # Get the last 3 candles for analysis
            last_candles = df.iloc[-3:].copy()
            last_candle = df.iloc[-1]
            
            # Get current price and symbol
            current_price = last_candle['close']
            symbol = df.iloc[-1].get('symbol', '')
            
            # Log detailed candle information
            logger.debug(f"Last 3 candles for reversal confirmation:")
            for i, candle in last_candles.iterrows():
                logger.debug(f"Candle {i} - O:{candle['open']:.5f}, H:{candle['high']:.5f}, "
                           f"L:{candle['low']:.5f}, C:{candle['close']:.5f}, "
                           f"Size:{abs(candle['close']-candle['open']):.5f}, "
                           f"Bullish:{candle['close'] > candle['open']}")
            
            # More detailed logging of the main candle we're checking
            candle_body_size = abs(last_candle['close'] - last_candle['open'])
            candle_total_range = last_candle['high'] - last_candle['low']
            body_percent = (candle_body_size / candle_total_range) * 100 if candle_total_range > 0 else 0
            
            logger.debug(f"Last candle details: Body size: {candle_body_size:.5f}, "
                       f"Total range: {candle_total_range:.5f}, Body %: {body_percent:.1f}%")
            logger.debug(f"Last candle direction: {'Bullish' if last_candle['close'] > last_candle['open'] else 'Bearish'}")
            
            # Calculate ATR using standardized method
            recent_atr = self._calculate_atr(df)
            
            # Extract pattern score if available
            pattern_score = 0
            try:
                if 'pattern_score' in df.columns:
                    pattern_score = float(df.iloc[-1]['pattern_score'])
                # Fallback to checking for columns containing 'score' if specific column not found
                elif any('score' in col.lower() for col in df.columns):
                    for col in df.columns:
                        if 'score' in col.lower():
                            pattern_score = float(df.iloc[-1][col])
                            logger.debug(f"Using score from column: {col}")
                            break
            except Exception as e:
                logger.debug(f"Could not extract pattern score: {str(e)}")
                pattern_score = 0
            
            # Log the pattern score explicitly
            logger.debug(f"Pattern score detected: {pattern_score}")
            
            # Higher score = more flexible entry
            score_multiplier = 1.0
            if pattern_score >= 4:  # High quality pattern
                score_multiplier = 2.0
                logger.debug(f"High quality pattern (score {pattern_score}) - using 2x more flexible thresholds")
            elif pattern_score >= 3:  # Good quality pattern
                score_multiplier = 1.5
                logger.debug(f"Good quality pattern (score {pattern_score}) - using 1.5x more flexible thresholds")
            
            # EXPANDED FIBONACCI PROXIMITY THRESHOLD
            # Determine if this is a volatile pair
            is_volatile = any(s in symbol for s in ['XAU', 'XAG', 'GBP', 'JPY', 'NAS', 'SPX', 'US30'])
            
            # Set base threshold as percentage of price (increased from 0.3% to 0.5%)
            base_threshold_percent = 0.5 if is_volatile else 0.3
            
            # Apply multipliers based on instrument type
            if 'JPY' in symbol:
                logger.debug(f"Detected JPY pair, using specialized threshold calculations")
                multiplier = 5.0 * score_multiplier
                
            elif any(s in symbol for s in ['XAU', 'XAG']):
                logger.debug(f"Detected volatile metal pair, using expanded thresholds")
                # Use 2x ATR for volatile pairs (up from previous values)
                multiplier = 4.0 * score_multiplier
                
            elif 'GBP' in symbol:
                logger.debug(f"Detected GBP pair, using expanded thresholds")
                multiplier = 3.5 * score_multiplier
                
            elif any(s in symbol for s in ['NAS', 'SPX', 'US30']):
                logger.debug(f"Detected index, using expanded thresholds")
                multiplier = 3.0 * score_multiplier
                
            else:
                # Standard FX pairs
                multiplier = 2.5 * score_multiplier
                logger.debug(f"Standard FX pair with multiplier: {multiplier:.1f}")
            
            # Calculate ATR-based threshold
            atr_threshold = recent_atr * base_threshold_percent * multiplier
            
            # For volatile pairs, use 2*ATR as minimum threshold
            if is_volatile:
                min_threshold = max(current_price * 0.005, recent_atr * 2.0)  # 0.5% of price or 2x ATR
            else:
                min_threshold = max(current_price * 0.003, recent_atr * 1.5)  # 0.3% of price or 1.5x ATR
            
            # Use the larger of the two thresholds
            distance_threshold = max(atr_threshold, min_threshold)
            logger.debug(f"Expanded Fibonacci proximity threshold: {distance_threshold:.5f} ({distance_threshold/current_price*100:.2f}% of price)")
            
            # Calculate distance to Fibonacci level
            distance_to_fib = abs(current_price - fib_level)
            logger.debug(f"Distance to fib level: {distance_to_fib:.5f}, Within threshold?: {distance_to_fib <= distance_threshold}")
            
            # Calculate momentum with lower threshold (±0.05% instead of ±0.1%)
            if len(df) >= 3:
                momentum = (df['close'].iloc[-1] / df['close'].iloc[-3] - 1) * 100  # 3-candle ROC
            else:
                # Fallback to single candle momentum
                momentum = (last_candle['close'] / last_candle['open'] - 1) * 100
            
            # Add a smaller neutral zone of ±0.05% to require clearer momentum
            momentum_direction = "Neutral"
            if momentum > 0.05:  # Reduced from 0.1 to 0.05
                momentum_direction = "Bullish"
            elif momentum < -0.05:  # Reduced from -0.1 to -0.05
                momentum_direction = "Bearish"
            
            logger.debug(f"Momentum (3-candle ROC): {momentum:.2f}% ({momentum_direction})")
            
            # Allow early confirmation for high-quality patterns
            if pattern_score >= 4:
                if direction == 'up' and last_candle['close'] > last_candle['open']:
                    logger.info(f"EARLY CONFIRM: Bullish entry confirmed for high quality pattern (score {pattern_score})")
                    return True
                elif direction == 'down' and last_candle['close'] < last_candle['open']:
                    logger.info(f"EARLY CONFIRM: Bearish entry confirmed for high quality pattern (score {pattern_score})")
                    return True
            
            # SIMPLIFIED CONFIRMATION CRITERIA
            if direction == 'up':
                # Simplified bullish conditions - just need a directional candle if near level
                is_bullish_candle = last_candle['close'] > last_candle['open']
                near_fib_level = distance_to_fib <= distance_threshold
                
                logger.debug(f"Bullish conditions check:")
                logger.debug(f"  - Is bullish candle: {is_bullish_candle}")
                logger.debug(f"  - Near fib level: {near_fib_level} (Distance: {distance_to_fib:.5f}, Threshold: {distance_threshold:.5f})")
                logger.debug(f"  - Momentum: {momentum:.2f}% ({momentum_direction})")
                
                # Confirm entry if bullish candle near Fibonacci level OR bullish momentum
                if is_bullish_candle and near_fib_level:
                    logger.info(f"Bullish reversal confirmed: Bullish candle near Fibonacci level")
                    return True
                elif momentum > 0.05 and near_fib_level:
                    logger.info(f"Bullish reversal confirmed: Positive momentum near Fibonacci level")
                    return True
                elif is_bullish_candle and momentum > 0.1:
                    logger.info(f"Bullish reversal confirmed: Bullish candle with strong positive momentum")
                    return True
                else:
                    logger.debug(f"Failed bullish confirmation: Bullish={is_bullish_candle}, Near={near_fib_level}, Momentum={momentum:.2f}%")
            
            elif direction == 'down':
                # Simplified bearish conditions - just need a directional candle if near level
                is_bearish_candle = last_candle['close'] < last_candle['open']
                wick_rejection = last_candle['high'] > fib_level and last_candle['close'] < last_candle['high'] - recent_atr * 0.3
                near_fib_level = distance_to_fib <= distance_threshold
                
                logger.debug(f"Bearish conditions check:")
                logger.debug(f"  - Is bearish candle: {is_bearish_candle}")
                logger.debug(f"  - Wick rejection: {wick_rejection}")
                logger.debug(f"  - Near fib level: {near_fib_level} (Distance: {distance_to_fib:.5f}, Threshold: {distance_threshold:.5f})")
                logger.debug(f"  - Momentum: {momentum:.2f}% ({momentum_direction})")
                
                # Confirm entry if bearish candle near Fibonacci level OR bearish momentum
                if (is_bearish_candle or wick_rejection) and near_fib_level:
                    logger.info(f"Bearish reversal confirmed: Bearish candle or wick rejection near Fibonacci level")
                    return True
                elif momentum < -0.05 and near_fib_level:
                    logger.info(f"Bearish reversal confirmed: Negative momentum near Fibonacci level")
                    return True
                elif (is_bearish_candle or wick_rejection) and momentum < -0.1:
                    logger.info(f"Bearish reversal confirmed: Bearish candle or wick rejection with strong negative momentum")
                    return True
                else:
                    logger.debug(f"Failed bearish confirmation: Bearish={is_bearish_candle}, Rejection={wick_rejection}, Near={near_fib_level}, Momentum={momentum:.2f}%")
            
            logger.info(f"No reversal candle confirmation found at fibonacci level")
            return False
            
        except Exception as e:
            logger.error(f"Error confirming reversal candle: {str(e)}")
            logger.error(f"Reversal candle confirmation exception: {traceback.format_exc()}")
            return False

    async def generate_signals(self, market_data: Dict, symbol: str, timeframe: str, 
                          account_info: Optional[Dict] = None) -> List[Dict]:
        """Generate trading signals based on the enhanced Turtle Soup strategy."""
        logger.info(f"[{symbol}] STARTING SIGNAL GENERATION on {timeframe} timeframe")
        logger.debug(f"[{symbol}] Signal generation parameters: risk_per_trade={self.risk_per_trade}, min_rr_ratio={self.min_rr_ratio}")
        
        signals = []
        try:
            df = market_data.get(timeframe)
            if df is None or df.empty:
                logger.warning(f"[{symbol}] No data available for {timeframe} timeframe")
                return []

            # Add symbol to dataframe for reference in pattern detection
            df['symbol'] = symbol

            logger.info(f"[{symbol}] Analyzing {len(df)} candles of {timeframe} data")
            logger.info(f"[{symbol}] Last close price: {df['close'].iloc[-1]:.5f}")
            logger.info(f"[{symbol}] Price range: High={df['high'].max():.5f}, Low={df['low'].min():.5f}")
            
            # Additional market context logging
            logger.debug(f"[{symbol}] Recent price action (last 5 candles):")
            for i in range(min(5, len(df))):
                candle = df.iloc[-(i+1)]
                logger.debug(f"[{symbol}] Candle -{i+1}: Open={candle['open']:.5f}, High={candle['high']:.5f}, Low={candle['low']:.5f}, Close={candle['close']:.5f}")
            
            # VOLATILITY CHECK - Skip signals in low volatility environments
            current_price = df['close'].iloc[-1]
            atr = self._calculate_atr(df)
            normalized_atr = atr / current_price  # ATR as percentage of price
            
            # Define minimum volatility threshold (0.05% of price)
            min_volatility = 0.0005  # 0.05% of price
            
            # Symbol-specific volatility adjustments
            if 'JPY' in symbol:
                min_volatility = 0.0006  # 0.06% for JPY pairs
            elif any(x in symbol for x in ['XAU', 'XAG']):
                min_volatility = 0.0008  # 0.08% for metals
            elif any(x in symbol for x in ['NAS', 'SPX', 'US30']):
                min_volatility = 0.0007  # 0.07% for indices
            
            logger.info(f"[{symbol}] Volatility check: ATR={atr:.5f}, Normalized ATR={normalized_atr*100:.3f}%, Minimum={min_volatility*100:.3f}%")
            
            if normalized_atr < min_volatility:
                logger.warning(f"[{symbol}] MARKET TOO QUIET - Normalized ATR ({normalized_atr*100:.3f}%) below threshold ({min_volatility*100:.3f}%)")
                logger.info(f"[{symbol}] Skipping signal generation in low volatility environment")
                return []
            
            # Check higher timeframe trend before pattern detection
            logger.info(f"[{symbol}] Analyzing higher timeframe trend")
            
            # Use MTFAnalysis class to properly analyze timeframe alignment
            mtf_data = {}
            if 'H1' in market_data and market_data['H1'] is not None and not market_data['H1'].empty:
                mtf_data['H1'] = market_data['H1']
            if 'H4' in market_data and market_data['H4'] is not None and not market_data['H4'].empty:
                mtf_data['H4'] = market_data['H4']
            if 'D1' in market_data and market_data['D1'] is not None and not market_data['D1'].empty:
                mtf_data['D1'] = market_data['D1']
                
            # Add current timeframe data to MTF analysis
            mtf_data[timeframe] = df
            
            # Perform comprehensive MTF analysis
            mtf_analysis_result = self.mtf_analysis.analyze_mtf(mtf_data, timeframe)
            
            # Extract alignment information
            timeframe_alignment = mtf_analysis_result.get('is_aligned', {'is_aligned': False})
            overall_bias = mtf_analysis_result.get('overall_bias', {'bias': 'neutral'})
            
            # Log detailed MTF analysis results
            alignment_ratio = timeframe_alignment.get('alignment_ratio', 0)
            required_ratio = timeframe_alignment.get('required_ratio', 0)
            logger.info(f"[{symbol}] MTF Analysis - Overall bias: {overall_bias['bias']}, Alignment ratio: {alignment_ratio:.2f}/{required_ratio:.2f}")
            
            # Determine if current timeframe aligns with higher timeframes
            higher_tf_bias = overall_bias['bias']
            is_aligned_with_higher_timeframes = timeframe_alignment.get('is_aligned', False)
            
            logger.info(f"[{symbol}] Higher timeframe bias: {higher_tf_bias} (Alignment: {'✅' if is_aligned_with_higher_timeframes else '❌'})")
            
            # Legacy code for backward compatibility - keep this to maintain logging
            htf_trends = {}
            if 'H1' in market_data and market_data['H1'] is not None and not market_data['H1'].empty:
                h1_trend = self._identify_trend(market_data['H1'].tail(20))
                htf_trends['H1'] = h1_trend
                logger.info(f"[{symbol}] H1 trend: {h1_trend}")
            
            if 'H4' in market_data and market_data['H4'] is not None and not market_data['H4'].empty:
                h4_trend = self._identify_trend(market_data['H4'].tail(20))
                htf_trends['H4'] = h4_trend
                logger.info(f"[{symbol}] H4 trend: {h4_trend}")
            
            if 'D1' in market_data and market_data['D1'] is not None and not market_data['D1'].empty:
                d1_trend = self._identify_trend(market_data['D1'].tail(20))
                htf_trends['D1'] = d1_trend
                logger.info(f"[{symbol}] D1 trend: {d1_trend}")
            
            # Get the dominant higher timeframe trend
            if htf_trends:
                # Prioritize higher timeframes (D1 > H4 > H1)
                if 'D1' in htf_trends:
                    htf_trend = htf_trends['D1']
                elif 'H4' in htf_trends:
                    htf_trend = htf_trends['H4']
                elif 'H1' in htf_trends:
                    htf_trend = htf_trends['H1']
                else:
                    htf_trend = 'neutral'
            else:
                htf_trend = 'neutral'
                
            # Store for other methods to use
            self.current_trend = htf_trend if htf_trend != 'neutral' else higher_tf_bias
            
            logger.info(f"[{symbol}] DOMINANT HTF TREND: {self.current_trend}")
            
            # Step 1: Detect Turtle Soup patterns
            logger.info(f"[{symbol}] STEP 1: Detecting Turtle Soup patterns")
            
            # Get current candle details for log
            current_candle = df.iloc[-1]
            logger.debug(f"[{symbol}] Current candle: Open={current_candle['open']:.5f}, High={current_candle['high']:.5f}, Low={current_candle['low']:.5f}, Close={current_candle['close']:.5f}")
            
            # Detect pattern
            logger.debug(f"[{symbol}] Searching for Turtle Soup patterns")
            # Add timeframe to df for dynamic lookback
            df['timeframe'] = timeframe
            turtle_soup_pattern = self._detect_turtle_soup(df)
            
            if not turtle_soup_pattern:
                logger.info(f"[{symbol}] No Turtle Soup pattern detected")
                return []
            
            logger.info(f"[{symbol}] Found Turtle Soup pattern: {turtle_soup_pattern['type'].upper()}")
            direction = 'long' if turtle_soup_pattern['type'] == 'long' else 'short'
            logger.debug(f"[{symbol}] Pattern details: {json.dumps(turtle_soup_pattern, cls=CustomJSONEncoder)}")
            
            pattern_score = turtle_soup_pattern.get('score', 0)
            
            # TREND FILTER - Only accept with-trend signals unless pattern is very strong
            trend_aligned = False
            
            if direction == 'long' and (self.current_trend == 'bullish' or higher_tf_bias == 'bullish'):
                logger.info(f"[{symbol}] LONG signal aligned with bullish higher timeframe trend ✅")
                trend_aligned = True
            elif direction == 'short' and (self.current_trend == 'bearish' or higher_tf_bias == 'bearish'):
                logger.info(f"[{symbol}] SHORT signal aligned with bearish higher timeframe trend ✅")
                trend_aligned = True
            else:
                if pattern_score >= 5:
                    logger.info(f"[{symbol}] Counter-trend signal allowed due to ultra-high pattern quality (score: {pattern_score}) ⚠️")
                    trend_aligned = True
                else:
                    logger.warning(f"[{symbol}] COUNTER-TREND SIGNAL REJECTED - {direction.upper()} against {self.current_trend} trend ❌")
                    logger.info(f"[{symbol}] Pattern score {pattern_score} below threshold (5) required for counter-trend signals")
                    return []
                
            # Store HTF alignment for later use in direct entry conditions
            htf_aligned = trend_aligned
            
            # Continue with signal generation now that trend filter is passed
            current_price = df['close'].iloc[-1]

            # NEW: Double-check pattern hasn't played out too much (for safety)
            direction = turtle_soup_pattern['type']  # 'long' or 'short'
            signal_type = 'BUY' if direction == 'long' else 'SELL'
            current_price = df['close'].iloc[-1]
            pattern_level = turtle_soup_pattern['level']
            played_out_percent = turtle_soup_pattern.get('played_out_percent', 0)
            
            logger.info(f"[{symbol}] TURTLE SOUP PATTERN DETECTED: {direction.upper()} at price {current_price:.5f}")
            logger.info(f"[{symbol}] Pattern level: {pattern_level:.5f}, Current distance: {abs(current_price - pattern_level):.5f} ({played_out_percent:.2f}%)")

            # Check if pattern direction aligns with higher timeframe trend
            htf_aligned = False
            confidence_boost = 0.0
            if higher_tf_bias:
                htf_aligned = (direction == 'long' and higher_tf_bias == 'bullish') or \
                             (direction == 'short' and higher_tf_bias == 'bearish')
                
                # Log alignment status
                if htf_aligned:
                    logger.info(f"[{symbol}] Pattern direction ({direction}) ALIGNS with higher timeframe bias ({higher_tf_bias})")
                    confidence_boost = 0.1  # Boost confidence for trend-aligned trades
                    if direction == 'short' and higher_tf_bias == 'bearish':
                        confidence_boost = 0.15  # Extra boost for bearish-aligned sells
                else:
                    logger.warning(f"[{symbol}] Pattern direction ({direction}) CONFLICTS with higher timeframe bias ({higher_tf_bias})")
                    
                # Get pattern score and check if it's high quality
                pattern_score = turtle_soup_pattern.get('score', 0)
                
                # CRITICAL CHANGE: Apply proper MTF alignment rules
                # Only allow counter-trend trades if:
                # 1. The pattern is extremely high quality (score 5+), AND
                # 2. We're not going against a strong higher timeframe trend
                if not htf_aligned:
                    # Only allow counter-trend trades with score 5 AND some alignment or strong confirmation
                    if pattern_score < 5 or (not is_aligned_with_higher_timeframes and alignment_ratio < 0.5):
                        logger.warning(f"[{symbol}] Rejecting counter-trend signal: Score {pattern_score} < 5 or insufficient alignment ({alignment_ratio:.2f})")
                        return []
                    else:
                        logger.info(f"[{symbol}] Allowing rare counter-trend trade: Score 5 with partial alignment ({alignment_ratio:.2f})")
                elif direction == 'short' and higher_tf_bias == 'bearish':
                    # Lower threshold for trend-aligned sells
                    if pattern_score >= 4:
                        logger.info(f"[{symbol}] Favoring trend-aligned SELL: Score {pattern_score} >= 4 with bearish bias")
                    elif pattern_score < 3:  # Still require a minimum quality for trend-aligned sells
                        logger.warning(f"[{symbol}] Trend-aligned SELL rejected: Score {pattern_score} < 3")
                        return []

            # Fix JSON serialization error by using the custom encoder
            try:
                # Convert potential Timestamp object to string first
                if isinstance(turtle_soup_pattern['breakout_candle'], pd.Timestamp):
                    turtle_soup_pattern_safe = turtle_soup_pattern.copy()
                    turtle_soup_pattern_safe['breakout_candle'] = turtle_soup_pattern_safe['breakout_candle'].strftime('%Y-%m-%d %H:%M:%S')
                    logger.info(f"[{symbol}] Pattern details: {json.dumps(turtle_soup_pattern_safe)}")
                else:
                    logger.info(f"[{symbol}] Pattern details: {json.dumps(turtle_soup_pattern, cls=CustomJSONEncoder)}")
            except Exception as e:
                # Fallback in case of any JSON serialization issues
                logger.info(f"[{symbol}] Pattern detected but details cannot be serialized to JSON: {str(e)}")
                logger.debug(f"[{symbol}] Pattern type: {direction}, level: {turtle_soup_pattern.get('level')}")
            
            # CRITICAL FIX: Add the pattern score to the dataframe 
            # so it can be referenced by confirmation methods
            if 'score' in turtle_soup_pattern:
                pattern_score = turtle_soup_pattern['score']
                df.loc[df.index[-1], 'pattern_score'] = pattern_score
                logger.debug(f"[{symbol}] Added pattern score {pattern_score} to last row of dataframe")
            
            # Log the exact breakout candle details
            breakout_idx = turtle_soup_pattern['breakout_candle']
            breakout_candle = df.loc[breakout_idx]
            logger.debug(f"[{symbol}] Breakout candle details: Index={breakout_idx}, Open={breakout_candle['open']:.5f}, "
                        f"High={breakout_candle['high']:.5f}, Low={breakout_candle['low']:.5f}, Close={breakout_candle['close']:.5f}")

            # Step 2: Identify swing points for Fibonacci calculation - without sensitivity parameter
            logger.info(f"[{symbol}] STEP 2: Identifying swing points for Fibonacci calculation")
            try:
                # Remove the sensitivity parameter since it's not supported
                swing_points = self.market_analysis.detect_swing_points(df)
                logger.info(f"[{symbol}] Found {len(swing_points['highs'])} swing highs and {len(swing_points['lows'])} swing lows")
                
                # Log detailed swing point information
                logger.debug(f"[{symbol}] All swing highs: {[f'{p['price']:.5f}@{p['index']}' for p in swing_points['highs'][-10:]]}")
                logger.debug(f"[{symbol}] All swing lows: {[f'{p['price']:.5f}@{p['index']}' for p in swing_points['lows'][-10:]]}")
                
                # More flexible swing point selection with fallback options
                if direction == 'long':
                    # For longs, use the recent lows and either recent or significant swing highs
                    recent_lows = [point['price'] for point in swing_points['lows'][-7:]] if swing_points['lows'] else []  # Increased from 5 to 7
                    if not recent_lows:
                        recent_lows = [df['low'].min()]  # Fallback to absolute low
                    
                    # Find previous significant swing high for target - look at more points
                    swing_high = max([point['price'] for point in swing_points['highs'][-10:]] if swing_points['highs'] else [df['high'].max()])
                    
                    # Use turtle soup level or best low point if available
                    swing_low = turtle_soup_pattern['level']
                    logger.info(f"[{symbol}] Using swing high={swing_high:.5f}, swing low={swing_low:.5f} for LONG pattern")
                else:
                    # For shorts, use the recent highs and either recent or significant swing lows
                    recent_highs = [point['price'] for point in swing_points['highs'][-7:]] if swing_points['highs'] else []  # Increased from 5 to 7
                    if not recent_highs:
                        recent_highs = [df['high'].max()]  # Fallback to absolute high
                    
                    # Find previous significant swing low for target - look at more points
                    swing_low = min([point['price'] for point in swing_points['lows'][-10:]] if swing_points['lows'] else [df['low'].min()])
                    
                    # Use turtle soup level or best high point if available
                    swing_high = turtle_soup_pattern['level']
                    logger.info(f"[{symbol}] Using swing high={swing_high:.5f}, swing low={swing_low:.5f} for SHORT pattern")

                # If swing high and low are too close, expand the range
                min_range_percent = 0.003  # 0.3% minimum range
                if (swing_high - swing_low) / current_price < min_range_percent:
                    logger.info(f"[{symbol}] Swing range too narrow ({(swing_high - swing_low) / current_price * 100:.2f}%), expanding range")
                    range_amount = current_price * min_range_percent
                    if direction == 'long':
                        swing_high = swing_low + range_amount
                    else:
                        swing_low = swing_high - range_amount
                    logger.info(f"[{symbol}] Adjusted swing range: High={swing_high:.5f}, Low={swing_low:.5f}")

            except Exception as e:
                logger.warning(f"[{symbol}] Error detecting swing points: {str(e)}")
                # Create fallback swing points if detection fails
                logger.info(f"[{symbol}] Using fallback swing point calculation")
                
                # Calculate dynamic lookback based on ATR
                atr = self._calculate_atr(df)
                current_price = df['close'].iloc[-1]
                
                # Dynamic volatility-adjusted lookback (higher volatility = shorter lookback)
                dynamic_lookback = int(max(10, min(100, atr / current_price * 1000)))
                logger.info(f"[{symbol}] Using dynamic lookback of {dynamic_lookback} bars (ATR: {atr:.5f}, Price: {current_price:.5f})")
                
                # Use recent data with dynamic lookback 
                recent_data = df.iloc[-dynamic_lookback:] if len(df) > dynamic_lookback else df
                
                # Use zigzag-like algorithm for higher quality swing points
                if len(recent_data) >= 5:  # Need at least 5 bars for basic detection
                    # Simple deviation threshold based on ATR
                    dev_threshold = max(0.001, atr / current_price * 0.3)  # 0.3 * ATR as percentage of price
                    
                    # Initialize for zigzag detection
                    is_uptrend = recent_data['close'].iloc[1] > recent_data['close'].iloc[0]
                    last_high_idx = 0
                    last_low_idx = 0
                    last_high = recent_data['high'].iloc[0]
                    last_low = recent_data['low'].iloc[0]
                    swing_highs = []
                    swing_lows = []
                    
                    # Simple zigzag detection
                    for i in range(1, len(recent_data)):
                        idx = recent_data.index[i]
                        if is_uptrend:
                            if recent_data['high'].iloc[i] > last_high:
                                # New higher high
                                last_high = recent_data['high'].iloc[i]
                                last_high_idx = idx
                            elif recent_data['low'].iloc[i] < last_low * (1 - dev_threshold):
                                # Trend reversal - add swing high
                                swing_highs.append({'index': last_high_idx, 'price': last_high})
                                
                                # Switch to downtrend
                                is_uptrend = False
                                last_low = recent_data['low'].iloc[i]
                                last_low_idx = idx
                        else:
                            if recent_data['low'].iloc[i] < last_low:
                                # New lower low
                                last_low = recent_data['low'].iloc[i]
                                last_low_idx = idx
                            elif recent_data['high'].iloc[i] > last_high * (1 + dev_threshold):
                                # Trend reversal - add swing low
                                swing_lows.append({'index': last_low_idx, 'price': last_low})
                                
                                # Switch to uptrend
                                is_uptrend = True
                                last_high = recent_data['high'].iloc[i]
                                last_high_idx = idx
                    
                    # Add the final swing point if needed
                    if is_uptrend and last_high_idx not in [h['index'] for h in swing_highs]:
                        swing_highs.append({'index': last_high_idx, 'price': last_high})
                    elif not is_uptrend and last_low_idx not in [l['index'] for l in swing_lows]:
                        swing_lows.append({'index': last_low_idx, 'price': last_low})
                    
                    # If we found swing points, use them
                    if swing_highs and swing_lows:
                        swing_points = {
                            'highs': [{'price': p['price'], 'index': p['index']} for p in swing_highs],
                            'lows': [{'price': p['price'], 'index': p['index']} for p in swing_lows]
                        }
                        logger.info(f"[{symbol}] Found {len(swing_highs)} swing highs and {len(swing_lows)} swing lows using zigzag fallback")
                    else:
                        # Simple fallback if zigzag produces no points
                        swing_points = {
                            'highs': [{'price': recent_data['high'].max(), 'index': recent_data['high'].idxmax()}],
                            'lows': [{'price': recent_data['low'].min(), 'index': recent_data['low'].idxmin()}]
                        }
                else:
                    # Very simple fallback with min/max for very small datasets
                    swing_points = {
                        'highs': [{'price': recent_data['high'].max(), 'index': recent_data['high'].idxmax()}],
                        'lows': [{'price': recent_data['low'].min(), 'index': recent_data['low'].idxmin()}]
                    }
                
                logger.info(f"[{symbol}] Fallback swing points - High: {swing_points['highs'][0]['price']:.5f}, Low: {swing_points['lows'][0]['price']:.5f}")
            
            # Step 3: Calculate Fibonacci levels
            logger.info(f"[{symbol}] STEP 3: Calculating Fibonacci retracement levels")
            logger.debug(f"[{symbol}] Calculating Fibonacci between: High={swing_high:.5f}, Low={swing_low:.5f}, Range={swing_high-swing_low:.5f}")
            
            fib_levels = self._calculate_fibonacci_levels(swing_high, swing_low)
            if not fib_levels:
                logger.warning(f"[{symbol}] Unable to calculate Fibonacci levels, using simple price projections")
                # Create simple levels based on recent price action
                midpoint = (swing_high + swing_low) / 2
                range_size = swing_high - swing_low
                
                fib_levels = {
                    '38.2%': swing_low + range_size * 0.382,
                    '50%': midpoint,
                    '61.8%': swing_low + range_size * 0.618
                }
                logger.debug(f"[{symbol}] Created fallback Fibonacci levels from midpoint {midpoint:.5f}")
            
            logger.info(f"[{symbol}] Fibonacci levels: 38.2%={fib_levels['38.2%']:.5f}, " 
                        f"50%={fib_levels['50%']:.5f}, 61.8%={fib_levels['61.8%']:.5f}")

            # Step 4: Check for potential entry based on current price - MADE MORE FLEXIBLE
            logger.info(f"[{symbol}] STEP 4: Checking entry conditions")
            
            # Consider all Fibonacci levels valid but prioritize directionally relevant ones
            potential_levels = []
            for level_name, level_price in fib_levels.items():
                distance = abs(current_price - level_price)
                logger.debug(f"[{symbol}] Distance to {level_name} level ({level_price:.5f}): {distance:.5f}")
                
                # Determine relevance score based on direction and level position
                relevance_score = 0
                if direction == 'long':
                    # For longs, prefer levels below current price
                    if level_price < current_price:
                        # Look for pullback to these levels (50% and 61.8% are often best)
                        relevance_score = 5
                        if level_name == '61.8%':
                            relevance_score = 10  # Best retracement for longs
                        elif level_name == '50%':
                            relevance_score = 8   # Second best
                    else:
                        # Level above price - less relevant for entry
                        relevance_score = 2
                else:  # direction == 'short'
                    # For shorts, prefer levels above current price
                    if level_price > current_price:
                        # Look for pullback to these levels (50% and 61.8% are often best)
                        relevance_score = 5
                        if level_name == '61.8%':
                            relevance_score = 10  # Best retracement for shorts
                        elif level_name == '50%':
                            relevance_score = 8   # Second best
                    else:
                        # Level below price - less relevant for entry
                        relevance_score = 2
                
                potential_levels.append((level_name, level_price, distance, relevance_score))
            
            # Sort by relevance score (higher is better) and then by distance (closer is better)
            potential_levels.sort(key=lambda x: (-x[3], x[2]))
            
            if potential_levels:
                # Use best level based on relevance and distance
                best_level = potential_levels[0]
                retracement_level = best_level[1]
                
                # Harmonize Fibonacci tolerance calculation with _confirm_reversal_candle method
                # Determine if this is a volatile pair
                is_volatile = any(s in symbol for s in ['XAU', 'XAG', 'GBP', 'JPY', 'NAS', 'SPX', 'US30'])
                tolerance = current_price * 0.005 if is_volatile else current_price * 0.003
                
                logger.info(f"[{symbol}] Selected retracement level: {best_level[0]} at {retracement_level:.5f} with tolerance ±{tolerance:.5f}")
                logger.info(f"[{symbol}] Current price: {current_price:.5f}, Distance to level: {abs(current_price - retracement_level):.5f}, Relevance: {best_level[3]}/10")
            else:
                # Fallback to using the current price as the level
                logger.warning(f"[{symbol}] No valid retracement levels found, using current price")
                retracement_level = current_price

            # Step 5: Confirm entry with reversal candle (or skip if price is already in favorable position)
            logger.info(f"[{symbol}] STEP 5: Confirming entry setup")
            entry_confirmed = False
            
            # Check if price is already in a favorable position
            if direction == 'long' and current_price > retracement_level:
                logger.info(f"[{symbol}] Price already above retracement level, considering favorable for LONG")
                logger.debug(f"[{symbol}] Price {current_price:.5f} > Level {retracement_level:.5f} by {current_price - retracement_level:.5f}")
                entry_confirmed = True
            elif direction == 'short' and current_price < retracement_level:
                logger.info(f"[{symbol}] Price already below retracement level, considering favorable for SHORT")
                logger.debug(f"[{symbol}] Price {current_price:.5f} < Level {retracement_level:.5f} by {retracement_level - current_price:.5f}")
                entry_confirmed = True
            # Otherwise check for reversal candle
            else:
                logger.info(f"[{symbol}] Price not in favorable position, checking for reversal candle")
                entry_confirmed = self._confirm_reversal_candle(df, retracement_level, direction)
                
            # NEW: Direct Market Entry option for high-quality patterns
            # If entry not confirmed but we have a high-quality pattern (score 4+), consider immediate entry
            pattern_score = turtle_soup_pattern.get('score', 0)
            if not entry_confirmed and pattern_score >= 5:  # Increased from 4 to 5
                # For ultra-high-quality patterns, verify if the current candle aligns with the pattern direction
                logger.info(f"[{symbol}] Entry not confirmed via standard methods, but ultra-high-quality pattern (score {pattern_score}) detected")
                
                last_candle = df.iloc[-1]
                
                # Require a stronger alignment: clear bullish/bearish candle
                direction_aligned = (direction == 'long' and last_candle['close'] > last_candle['open'] * 1.001) or \
                                   (direction == 'short' and last_candle['close'] < last_candle['open'] * 0.999)
                
                # Check for wick rejection in the correct direction for shorts
                wick_rejection = False
                if direction == 'short':
                    wick_size = last_candle['high'] - last_candle['close']
                    wick_rejection = wick_size > (last_candle['high'] - last_candle['low']) * 0.5
                    if wick_rejection:
                        logger.debug(f"[{symbol}] Detected wick rejection for short: Upper wick {wick_size:.5f} is significant")
                
                # Calculate 3-candle momentum
                momentum = 0
                if len(df) >= 3:
                    momentum = (df['close'].iloc[-1] / df['close'].iloc[-3] - 1) * 100  # 3-candle momentum
                
                # Require actual momentum in the trade direction
                momentum_confirmed = (momentum > 0.2 if direction == 'long' else momentum < -0.2)
                
                # For trend-aligned shorts, be more lenient with momentum requirement
                if direction == 'short' and htf_aligned and momentum < -0.05:
                    momentum_confirmed = True
                    logger.debug(f"[{symbol}] Using relaxed momentum threshold for trend-aligned short: {momentum:.2f}%")
                
                logger.debug(f"[{symbol}] Direct entry check - Candle aligned: {direction_aligned}, Wick rejection: {wick_rejection}, Momentum: {momentum:.2f}%")
                
                if direction_aligned and (momentum_confirmed or wick_rejection):
                    logger.info(f"[{symbol}] DIRECT MARKET ENTRY: High quality pattern with aligned candle and momentum/rejection")
                    entry_confirmed = True
                else:
                    logger.info(f"[{symbol}] Direct entry rejected: Insufficient confirmation (aligned: {direction_aligned}, momentum: {momentum:.2f}%, rejection: {wick_rejection})")
            
            # Add debug logging after Step 5 if in debug mode
            if self.debug_mode and not entry_confirmed:
                logger.info(f"[{symbol}] DEBUG: Potential signal rejected at confirmation - Pattern: {turtle_soup_pattern['type']}, Score: {pattern_score}")
            
            if entry_confirmed:
                logger.info(f"[{symbol}] ENTRY CONFIRMED!")
                
                # Step 6: Calculate stop loss beyond breakout candle
                logger.info(f"[{symbol}] STEP 6: Calculating stop loss")
                stop_loss = self._calculate_stop_loss(df, turtle_soup_pattern)
                if not stop_loss:
                    # Fallback to default stop loss based on ATR
                    logger.info(f"[{symbol}] Using fallback stop loss calculation")
                    stop_loss = self._get_default_stop_loss(df, direction, current_price)
                
                logger.info(f"[{symbol}] Stop loss calculated: {stop_loss:.5f}")
                logger.debug(f"[{symbol}] Stop details - Distance from entry: {abs(current_price - stop_loss):.5f}, "
                            f"Percent: {abs(current_price - stop_loss) / current_price * 100:.2f}%")

                # FIX: Use reasonable minimum stop distance (0.3% instead of 1%)
                # Calculate minimum stop percentage based on instrument type
                is_crypto = any(s in symbol for s in ['BTC', 'ETH', 'XRP', 'USDT'])
                min_stop_percent = 0.003 if is_crypto else 0.002  # 0.3% for crypto, 0.2% for others (increased from 0.25% and 0.15%)
                
                min_stop_distance = current_price * min_stop_percent
                logger.debug(f"[{symbol}] Minimum stop distance check: {min_stop_distance:.5f} ({min_stop_percent*100:.2f}% of price)")
                
                # Apply minimum stop distance if necessary
                if direction == 'long' and (current_price - stop_loss) < min_stop_distance:
                    old_stop = stop_loss
                    stop_loss = current_price - min_stop_distance
                    logger.info(f"[{symbol}] Adjusted stop loss to ensure minimum distance: {old_stop:.5f} -> {stop_loss:.5f}")
                elif direction == 'short' and (stop_loss - current_price) < min_stop_distance:
                    old_stop = stop_loss
                    stop_loss = current_price + min_stop_distance
                    logger.info(f"[{symbol}] Adjusted stop loss to ensure minimum distance: {old_stop:.5f} -> {stop_loss:.5f}")

                # Step 7: Calculate take profit (minimum 1.5:1 risk-reward)
                logger.info(f"[{symbol}] STEP 7: Calculating take profit")
                risk = abs(current_price - stop_loss)
                logger.info(f"[{symbol}] Risk distance: {risk:.5f} ({risk/current_price*100:.2f}%)")
                
                # Use at least 1.5:1 reward/risk ratio
                min_reward_ratio = 1.5
                basic_take_profit = current_price + (min_reward_ratio * risk) if direction == 'long' else current_price - (min_reward_ratio * risk)
                logger.debug(f"[{symbol}] Basic take profit at {min_reward_ratio}:1 R:R: {basic_take_profit:.5f}")
                
                # Start with the basic take profit
                take_profit = basic_take_profit
                
                # Also check for key levels that might serve as better targets
                if direction == 'long':
                    # For longs, consider recent swing highs as potential targets
                    potential_targets = sorted([h['price'] for h in swing_points['highs'][-10:] if h['price'] > current_price])
                    logger.debug(f"[{symbol}] Potential LONG targets (swing highs): {[f'{p:.5f}' for p in potential_targets]}")
                    
                    for high in potential_targets:
                        if high > current_price + (risk * min_reward_ratio):
                            take_profit = high
                            logger.info(f"[{symbol}] Using swing high as take profit target: {take_profit:.5f}")
                            logger.debug(f"[{symbol}] Target distance: {take_profit - current_price:.5f}, R:R: {(take_profit - current_price) / risk:.2f}")
                            break
                else:
                    # For shorts, consider recent swing lows as potential targets
                    potential_targets = sorted([l['price'] for l in swing_points['lows'][-10:] if l['price'] < current_price], reverse=True)
                    logger.debug(f"[{symbol}] Potential SHORT targets (swing lows): {[f'{p:.5f}' for p in potential_targets]}")
                    
                    for low in potential_targets:
                        if low < current_price - (risk * min_reward_ratio):
                            take_profit = low
                            logger.info(f"[{symbol}] Using swing low as take profit target: {take_profit:.5f}")
                            logger.debug(f"[{symbol}] Target distance: {current_price - take_profit:.5f}, R:R: {(current_price - take_profit) / risk:.2f}")
                            break
                
                logger.info(f"[{symbol}] Take profit calculated: {take_profit:.5f}")
                logger.debug(f"[{symbol}] Final R:R ratio: {abs(take_profit - current_price) / risk:.2f}")

                # Step 8: Calculate position size
                logger.info(f"[{symbol}] STEP 8: Calculating position size")
                account_balance = account_info.get('balance', 10000) if account_info else 10000
                logger.info(f"[{symbol}] Account balance: {account_balance}")
                
                # Use risk_manager to calculate position size instead of manual calculation
                position_size = self.risk_manager.calculate_position_size(
                    account_balance=account_balance,
                    risk_per_trade=self.risk_per_trade,
                    entry_price=current_price,
                    stop_loss_price=stop_loss,
                    symbol=symbol
                )
                
                # Check if a valid position size was returned
                if position_size <= 0:
                    # Fallback to minimum position size
                    position_size = 0.01
                    logger.warning(f"[{symbol}] Invalid position size returned from risk manager - using minimum")
                
                # Calculate actual risk for reporting
                risk = abs(current_price - stop_loss)
                actual_risk_amount = position_size * risk
                risk_percent = actual_risk_amount / account_balance if account_balance > 0 else 0
                
                # Define maximum risk percent for trade validation (this was missing)
                max_risk_percent = 0.01 # Maximum 1% risk per trade
                
                logger.info(f"[{symbol}] Position size calculated: {position_size}")
                logger.info(f"[{symbol}] Actual risk: {actual_risk_amount:.2f} ({risk_percent*100:.2f}% of account)")

                # Step 9: Validate trade (more lenient validation)
                logger.info(f"[{symbol}] STEP 9: Validating trade parameters")
                
                # Calculate final risk-reward ratio
                risk_reward_ratio = abs(take_profit - current_price) / risk if risk > 0 else 0
                
                # Use more strict validation than before
                valid_trade = True
                reason = "Valid trade setup"
                
                # Perform rigorous validation
                if position_size <= 0:
                    valid_trade = False
                    reason = "Invalid position size"
                    logger.warning(f"[{symbol}] Invalid position size: {position_size}")
                elif risk <= 0:
                    valid_trade = False
                    reason = "Invalid risk amount"
                    logger.warning(f"[{symbol}] Invalid risk amount: {risk}")
                elif direction == 'long' and stop_loss >= current_price:
                    valid_trade = False
                    reason = "Invalid stop loss for long trade"
                    logger.warning(f"[{symbol}] Invalid stop for LONG: {stop_loss:.5f} >= {current_price:.5f}")
                elif direction == 'short' and stop_loss <= current_price:
                    valid_trade = False
                    reason = "Invalid stop loss for short trade"
                    logger.warning(f"[{symbol}] Invalid stop for SHORT: {stop_loss:.5f} <= {current_price:.5f}")
                elif risk_reward_ratio < self.min_rr_ratio:
                    valid_trade = False
                    reason = f"Insufficient risk-reward ratio: {risk_reward_ratio:.2f} < {self.min_rr_ratio}"
                    logger.warning(f"[{symbol}] {reason}")
                # FIX: Add validation for maximum risk percent
                elif risk_percent > max_risk_percent:
                    valid_trade = False
                    reason = f"Excessive risk: {risk_percent*100:.2f}% > {max_risk_percent*100:.1f}%"
                    logger.warning(f"[{symbol}] {reason}")
                    
                # Accept more trades by allowing reasonable adjustments
                if not valid_trade and "risk-reward" in reason:
                    logger.info(f"[{symbol}] Attempting to fix risk-reward by adjusting take profit")
                    
                    # Adjust take profit to ensure minimum R:R of 1.5
                    new_tp = current_price + (risk * self.min_rr_ratio) if direction == 'long' else current_price - (risk * self.min_rr_ratio)
                    logger.info(f"[{symbol}] Adjusted take profit from {take_profit:.5f} to {new_tp:.5f}")
                    take_profit = new_tp
                    risk_reward_ratio = self.min_rr_ratio  # Fixed ratio after adjustment
                    
                    valid_trade = True
                    reason = "Valid trade after take profit adjustment"
                    logger.info(f"[{symbol}] {reason}")
                
                # Add debug logging after Step 9 if in debug mode
                if self.debug_mode and not valid_trade:
                    logger.info(f"[{symbol}] DEBUG: Potential signal rejected at validation - Reason: {reason}")
                
                if valid_trade:
                    logger.info(f"[{symbol}] TRADE VALIDATION PASSED: {reason}")
                
                    # Step 10: Generate signal
                    logger.info(f"[{symbol}] STEP 10: Generating final signal")
                    
                    # NEW: Add entry type field based on how the entry was confirmed
                    entry_type = "standard"
                    if pattern_score >= 4 and not (
                        (direction == 'long' and current_price > retracement_level) or 
                        (direction == 'short' and current_price < retracement_level)
                    ):
                        entry_type = "direct_market"
                        logger.info(f"[{symbol}] Using direct market entry due to high-quality pattern")
                    
                    signal = {
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'direction': signal_type,
                        'entry_price': current_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'position_size': position_size,
                        'risk_amount': actual_risk_amount,  # FIX: Use accurate risk amount
                        'risk_percent': risk_percent,  # FIX: Add actual risk percentage
                        'risk_reward_ratio': risk_reward_ratio,
                        'confidence': 0.7 + confidence_boost,
                        'timestamp': datetime.now().isoformat(),
                        'strategy': 'SG3_TurtleSoup',
                        'setup_type': 'turtle_soup',
                        'entry_type': entry_type,
                        'pattern_score': pattern_score,
                        'analysis': {
                            'turtle_soup_pattern': turtle_soup_pattern,
                            'fib_level': retracement_level
                        }
                    }
                    signals.append(signal)
                    logger.info(f"[{symbol}] GENERATED {signal_type} SIGNAL at {current_price:.5f}")
                    logger.info(f"[{symbol}] Entry: {current_price:.5f} | Stop: {stop_loss:.5f} | Target: {take_profit:.5f}")
                    logger.info(f"[{symbol}] R:R = 1:{risk_reward_ratio:.1f} | Size: {position_size}")
                    logger.info(f"[{symbol}] Risk: {actual_risk_amount:.2f} ({risk_percent*100:.2f}% of account)")
                    logger.debug(f"[{symbol}] Complete signal details: {json.dumps(signal, default=str)}")
                else:
                    logger.warning(f"[{symbol}] TRADE VALIDATION FAILED: {reason}")
            else:
                logger.info(f"[{symbol}] No valid entry confirmation found")

            # FIX: Add debug logging to show what's being returned
            if signals:
                logger.info(f"[{symbol}] Returning {len(signals)} signals")
            else:
                logger.info(f"[{symbol}] No signals generated")
                
            return signals

        except Exception as e:
            logger.error(f"[{symbol}] ERROR generating signals: {str(e)}")
            logger.error(f"[{symbol}] Exception traceback: {traceback.format_exc()}")
            logger.debug(f"[{symbol}] DataFrame head at time of error: {df.head().to_dict() if df is not None and not df.empty else 'DataFrame empty'}")
            return []

    def _identify_trend(self, df: pd.DataFrame) -> str:
        """
        Identify the trend in a given dataframe using multiple indicators.
        Returns 'bullish', 'bearish', or 'neutral'.
        """
        try:
            if df is None or len(df) < 10:
                return 'neutral'  # Not enough data
                
            # Calculate simple moving averages
            fast_ma = df['close'].rolling(5).mean().iloc[-1]
            slow_ma = df['close'].rolling(10).mean().iloc[-1]
            
            # Calculate closing price position
            current_close = df['close'].iloc[-1]
            recent_high = df['high'].iloc[-5:].max()
            recent_low = df['low'].iloc[-5:].min()
            price_range = recent_high - recent_low
            
            # Calculate momentum using rate of change
            roc = ((current_close / df['close'].iloc[-4]) - 1) * 100
            
            # Identify basic trend signals
            ma_bullish = fast_ma > slow_ma
            ma_bearish = fast_ma < slow_ma
            
            price_bullish = current_close > (recent_low + price_range * 0.66)
            price_bearish = current_close < (recent_high - price_range * 0.66)
            
            momentum_bullish = roc > 0.5
            momentum_bearish = roc < -0.5
            
            # Count bullish and bearish signals
            bullish_signals = sum([ma_bullish, price_bullish, momentum_bullish])
            bearish_signals = sum([ma_bearish, price_bearish, momentum_bearish])
            
            # Determine overall trend
            if bullish_signals > bearish_signals + 1:
                return 'bullish'
            elif bearish_signals > bullish_signals + 1:
                return 'bearish'
            else:
                return 'neutral'
                
        except Exception as e:
            logger.error(f"Error identifying trend: {str(e)}")
            return 'neutral'  # Default to neutral on error