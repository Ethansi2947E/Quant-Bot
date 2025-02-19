import pandas as pd
import numpy as np
from datetime import datetime, time, UTC
from typing import Dict, List, Tuple, Optional, Any
from loguru import logger
from config.config import SESSION_CONFIG, MARKET_STRUCTURE_CONFIG

class MarketAnalysis:
    def __init__(self, ob_threshold=0.0015):
        self.session_config = SESSION_CONFIG
        self.structure_config = MARKET_STRUCTURE_CONFIG
        # Timeframe-specific lookback periods
        self.lookback_periods = {
            'M5': 200,    # ~17 hours
            'M15': 288,   # 3 days
            'H1': 168,    # 1 week
            'H4': 180,    # 1 month
            'D1': 90      # 3 months
        }
        self.ob_threshold = ob_threshold
        self.fvg_threshold = 0.00015  # Adjusted for M15
        self.swing_detection_lookback = 6  # Optimized for M15
        self.swing_detection_threshold = 0.00015  # Adjusted for M15
        self.min_swing_size = 0.0003  # Adjusted for M15
        self.bos_threshold = 0.00015  # Adjusted for M15
        self.min_swing_points = 2  # Reduced for faster structure detection
        self.structure_break_threshold = 0.0004  # Adjusted for M15
        
        # Statistics tracking
        self.stats = {
            'swing_points': {'highs': 0, 'lows': 0, 'rejected': 0},
            'structure_breaks': {'bullish': 0, 'bearish': 0, 'invalid': 0},
            'fvg_gaps': {'bullish': 0, 'bearish': 0, 'filled': 0},
            'order_blocks': {'bullish': 0, 'bearish': 0, 'tested': 0}
        }
        
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        try:
            if len(df) < period:
                logger.warning(
                    f"Not enough data for ATR. Required: {period}, Got: {len(df)}"
                )
                return pd.Series(np.nan, index=df.index)

            if not all(col in df.columns for col in ['high', 'low', 'close']):
                logger.error("Missing required columns for ATR calculation")
                return pd.Series(np.nan, index=df.index)

            high = df['high'].astype(float)
            low = df['low'].astype(float)
            close = df['close'].astype(float).shift(1)

            tr1 = high - low
            tr2 = abs(high - close)
            tr3 = abs(low - close)
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            atr = tr.ewm(span=period, adjust=False).mean()
            atr = atr.bfill().ffill()

            if atr.isnull().any():
                logger.warning("Some ATR values could not be calculated")
                atr = atr.fillna(atr.mean())

            typical_price = (df['high'] + df['low'] + df['close']) / 3
            atr_pct = (atr / typical_price) * 100

            logger.debug(
                f"ATR Stats - Mean: {atr.mean():.5f}, Max: {atr.max():.5f}, "
                f"Min: {atr.min():.5f}"
            )
            logger.debug(
                f"ATR% Stats - Mean: {atr_pct.mean():.2f}%, Max: {atr_pct.max():.2f}%, "
                f"Min: {atr_pct.min():.2f}%"
            )

            return atr
        except Exception as e:
            logger.error(f"Error calculating ATR: {str(e)}")
            return pd.Series(np.nan, index=df.index)
        
    def get_current_session(self) -> Tuple[str, Dict]:
        """Determine the current trading session based on UTC time."""
        current_time = datetime.now(UTC).time()
        
        # Define standard session names
        SESSION_NAMES = {
            'london': 'london',
            'new_york': 'new_york',
            'asia': 'asia',
            'no_session': 'no_session'
        }
        
        for session_name, session_data in self.session_config.items():
            session_start = datetime.strptime(session_data['start'], '%H:%M').time()
            session_end = datetime.strptime(session_data['end'], '%H:%M').time()
            
            if session_start <= current_time <= session_end:
                return SESSION_NAMES.get(session_name, session_name), session_data
                
        return SESSION_NAMES['no_session'], {}
        
    def detect_swing_points(self, df: pd.DataFrame, window_size: int = 5) -> Dict:
        """
        Detect swing high and low points in the price data.
        
        Args:
            df (pd.DataFrame): Price data
            window_size (int): Window size for detecting swings
            
        Returns:
            Dict: Dictionary containing swing highs and lows
        """
        try:
            highs = []
            lows = []
            
            for i in range(window_size, len(df) - window_size):
                # Get the window of data
                window = df.iloc[i-window_size:i+window_size+1]
                current_price = df.iloc[i]['close']
                
                # Check for swing high
                if all(current_price >= window.iloc[j]['high'] for j in range(len(window)) if j != window_size):
                    highs.append({
                            'index': i,
                        'price': current_price,
                            'timestamp': df.index[i],
                        'size': current_price - min(window['low'])
                    })
                
                # Check for swing low
                if all(current_price <= window.iloc[j]['low'] for j in range(len(window)) if j != window_size):
                    lows.append({
                            'index': i,
                        'price': current_price,
                            'timestamp': df.index[i],
                        'size': max(window['high']) - current_price
                    })
            
            return {
                'highs': highs,
                'lows': lows
            }
            
        except Exception as e:
            logger.error(f"Error detecting swing points: {str(e)}")
            return {'highs': [], 'lows': []}
            
    def _validate_swing_high(self, df: pd.DataFrame, index: int, lookback: int, threshold: float) -> bool:
        """Validate a potential swing high point."""
        try:
            # Check left side (previous bars)
            left_valid = all(df['high'].iloc[index] > df['high'].iloc[j] 
                           for j in range(index - lookback, index))
            
            # Check right side (following bars)
            right_valid = all(df['high'].iloc[index] > df['high'].iloc[j] 
                            for j in range(index + 1, min(index + lookback + 1, len(df))))
            
            # Check minimum price movement
            price_valid = (df['high'].iloc[index] - df['low'].iloc[index-lookback:index+lookback+1].min()) >= threshold
            
            # Check for clean swing (no noise)
            noise_ratio = abs(df['close'].iloc[index] - df['open'].iloc[index]) / \
                         (df['high'].iloc[index] - df['low'].iloc[index])
            clean_swing = noise_ratio <= 0.5  # Body should not be more than 50% of the range
            
            return left_valid and right_valid and price_valid and clean_swing
            
        except Exception as e:
            logger.error(f"Error validating swing high at index {index}: {str(e)}")
            return False
            
    def _validate_swing_low(self, df: pd.DataFrame, index: int, lookback: int, threshold: float) -> bool:
        """Validate a potential swing low point."""
        try:
            # Check left side (previous bars)
            left_valid = all(df['low'].iloc[index] < df['low'].iloc[j] 
                           for j in range(index - lookback, index))
            
            # Check right side (following bars)
            right_valid = all(df['low'].iloc[index] < df['low'].iloc[j] 
                            for j in range(index + 1, min(index + lookback + 1, len(df))))
            
            # Check minimum price movement
            price_valid = (df['high'].iloc[index-lookback:index+lookback+1].max() - df['low'].iloc[index]) >= threshold
            
            # Check for clean swing (no noise)
            noise_ratio = abs(df['close'].iloc[index] - df['open'].iloc[index]) / \
                         (df['high'].iloc[index] - df['low'].iloc[index])
            clean_swing = noise_ratio <= 0.5  # Body should not be more than 50% of the range
            
            return left_valid and right_valid and price_valid and clean_swing
            
        except Exception as e:
            logger.error(f"Error validating swing low at index {index}: {str(e)}")
            return False
        
    def detect_order_blocks(
        self,
        df: pd.DataFrame,
        swing_points: Tuple[List[Dict], List[Dict]]
    ) -> Dict[str, List[Dict]]:
        """Detect bullish and bearish order blocks."""
        try:
            bullish_obs = []
            bearish_obs = []
            
            for i in range(3, len(df)):
                # Initialize ob_size to None at the start of each iteration
                ob_size = None
                
                # Bullish order blocks
                if df['close'].iloc[i-2] < df['open'].iloc[i-2] and \
                   df['high'].iloc[i] > df['high'].iloc[i-2]:
                    
                    # Calculate OB zone
                    ob_high = df['high'].iloc[i-2]
                    ob_low = min(df['open'].iloc[i-2], df['close'].iloc[i-2])
                    ob_size = ob_high - ob_low
                    
                    # Check if OB is significant
                    if ob_size and ob_size >= self.ob_threshold:
                        bullish_obs.append({
                            'index': i-2,
                            'high': ob_high,
                            'low': ob_low,
                            'size': ob_size,
                            'timestamp': df.index[i-2]
                        })
                
                # Bearish order blocks
                if df['close'].iloc[i-2] > df['open'].iloc[i-2] and \
                   df['low'].iloc[i] < df['low'].iloc[i-2]:
                    
                    # Calculate OB zone
                    ob_high = max(df['open'].iloc[i-2], df['close'].iloc[i-2])
                    ob_low = df['low'].iloc[i-2]
                    ob_size = ob_high - ob_low
                    
                    # Check if OB is significant
                    if ob_size and ob_size >= self.ob_threshold:
                        bearish_obs.append({
                            'index': i-2,
                            'high': ob_high,
                            'low': ob_low,
                            'size': ob_size,
                            'timestamp': df.index[i-2]
                        })
            
            return {
                'bullish': bullish_obs,
                'bearish': bearish_obs
            }
            
        except Exception as e:
            logger.error(f"Error detecting order blocks: {str(e)}")
            return {'bullish': [], 'bearish': []}
        
    def detect_fair_value_gaps(
        self,
        df: pd.DataFrame
    ) -> Dict[str, List[Dict]]:
        """Detect fair value gaps in price action."""
        try:
            bullish_fvgs = []
            bearish_fvgs = []
            
            for i in range(2, len(df)):
                # Bullish FVG (gap up)
                if df['low'].iloc[i] > df['high'].iloc[i-2]:
                    gap_size = df['low'].iloc[i] - df['high'].iloc[i-2]
                    
                    if gap_size >= self.fvg_threshold:
                        bullish_fvgs.append({
                            'index': i-1,
                            'top': df['low'].iloc[i],
                            'bottom': df['high'].iloc[i-2],
                            'size': gap_size,
                            'timestamp': df.index[i-1]
                        })
                        
                # Bearish FVG (gap down)
                if df['high'].iloc[i] < df['low'].iloc[i-2]:
                    gap_size = df['low'].iloc[i-2] - df['high'].iloc[i]
                    
                    if gap_size >= self.fvg_threshold:
                        bearish_fvgs.append({
                            'index': i-1,
                            'top': df['low'].iloc[i-2],
                            'bottom': df['high'].iloc[i],
                            'size': gap_size,
                            'timestamp': df.index[i-1]
                        })
            
            return {
                'bullish': bullish_fvgs,
                'bearish': bearish_fvgs
            }
            
        except Exception as e:
            logger.error(f"Error detecting FVGs: {str(e)}")
            return {'bullish': [], 'bearish': []}
        
    def detect_structure_breaks(
        self,
        df: pd.DataFrame,
        swing_points: Dict
    ) -> Dict:
        """
        Detect structure breaks in the market with improved detection logic.
        
        Args:
            df (pd.DataFrame): Price data with OHLCV
            swing_points (Dict): Dictionary containing swing highs and lows
            
        Returns:
            Dict: Dictionary containing structure breaks information
        """
        structure_breaks = {
            'bullish': [],
            'bearish': [],
            'strength': [],
            'timestamps': []
        }
        
        try:
            if not swing_points or 'highs' not in swing_points or 'lows' not in swing_points:
                logger.warning("No swing points provided for structure break detection")
                return structure_breaks
            
            # Sort swing points by index
            highs = sorted(swing_points['highs'], key=lambda x: x['index'])
            lows = sorted(swing_points['lows'], key=lambda x: x['index'])
            
            # Need at least 2 swing points of each type
            if len(highs) < 2 or len(lows) < 2:
                logger.warning(f"Insufficient swing points for structure break detection. Highs: {len(highs)}, Lows: {len(lows)}")
                return structure_breaks
            
            logger.info(f"Analyzing structure breaks with {len(highs)} highs and {len(lows)} lows")
            
            # Analyze recent swing points (last 5)
            recent_highs = highs[-5:] if len(highs) >= 5 else highs
            recent_lows = lows[-5:] if len(lows) >= 5 else lows
            
            # Detect bullish structure breaks
            for i in range(1, len(recent_highs)):
                curr_high = recent_highs[i]
                prev_high = recent_highs[i-1]
                
                # Find any lows between these highs
                intermediate_lows = [l for l in recent_lows 
                                   if l['index'] > prev_high['index'] 
                                   and l['index'] < curr_high['index']]
                
                # Bullish break conditions:
                # 1. Current high is higher than previous high
                # 2. We have at least one higher low between the highs
                if curr_high['price'] > prev_high['price'] and intermediate_lows:
                    # Check if any intermediate low is higher than previous low
                    prev_low = next((l for l in recent_lows if l['index'] < prev_high['index']), None)
                    if prev_low and any(l['price'] > prev_low['price'] for l in intermediate_lows):
                        strength = ((curr_high['price'] - prev_high['price']) / prev_high['price']) * 100
                        
                        structure_breaks['bullish'].append({
                            'price': curr_high['price'],
                            'timestamp': curr_high['timestamp'],
                            'strength': strength,
                            'index': curr_high['index']
                        })
                        structure_breaks['strength'].append(strength)
                        structure_breaks['timestamps'].append(curr_high['timestamp'])
                        logger.info(f"Detected bullish break at {curr_high['price']:.5f} with strength {strength:.2f}%")
            
            # Detect bearish structure breaks
            for i in range(1, len(recent_lows)):
                curr_low = recent_lows[i]
                prev_low = recent_lows[i-1]
                
                # Find any highs between these lows
                intermediate_highs = [h for h in recent_highs 
                                    if h['index'] > prev_low['index'] 
                                    and h['index'] < curr_low['index']]
                
                # Bearish break conditions:
                # 1. Current low is lower than previous low
                # 2. We have at least one lower high between the lows
                if curr_low['price'] < prev_low['price'] and intermediate_highs:
                    # Check if any intermediate high is lower than previous high
                    prev_high = next((h for h in recent_highs if h['index'] < prev_low['index']), None)
                    if prev_high and any(h['price'] < prev_high['price'] for h in intermediate_highs):
                        strength = ((prev_low['price'] - curr_low['price']) / prev_low['price']) * 100
                        
                        structure_breaks['bearish'].append({
                            'price': curr_low['price'],
                            'timestamp': curr_low['timestamp'],
                            'strength': strength,
                            'index': curr_low['index']
                        })
                        structure_breaks['strength'].append(strength)
                        structure_breaks['timestamps'].append(curr_low['timestamp'])
                        logger.info(f"Detected bearish break at {curr_low['price']:.5f} with strength {strength:.2f}%")
            
            # Log structure break statistics
            logger.info(f"Structure Breaks Found - Bullish: {len(structure_breaks['bullish'])}, "
                       f"Bearish: {len(structure_breaks['bearish'])}")
            
            if structure_breaks['strength']:
                avg_strength = sum(structure_breaks['strength']) / len(structure_breaks['strength'])
                logger.info(f"Average Break Strength: {avg_strength:.2f}%")
            
        except Exception as e:
            logger.error(f"Error in structure break detection: {str(e)}")
            logger.exception("Detailed error trace:")
            
        return structure_breaks
        
    def analyze_session_conditions(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> Dict:
        """Analyze current session-specific trading conditions."""
        try:
            session_name, session_data = self.get_current_session()
            
            if not session_data:
                return {
                    'session': session_name,
                    'suitable_for_trading': False,
                    'volatility_factor': 0.0,
                    'reason': 'Outside main trading sessions'
                }
            
            # Calculate ATR
            if 'atr' not in df.columns:
                # Calculate True Range directly without using apply
                high_low = df['high'] - df['low']
                high_close = abs(df['high'] - df['close'].shift(1))
                low_close = abs(df['low'] - df['close'].shift(1))
                
                # Combine into TR
                df['tr'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                df['atr'] = df['tr'].rolling(window=14).mean()
            
            # Get current ATR and calculate volatility state
            current_atr = df['atr'].iloc[-1]
            atr_mean = df['atr'].mean()
            atr_std = df['atr'].std()
            
            # Calculate volatility factor based on standard deviations from mean
            if atr_mean > 0 and atr_std > 0:
                z_score = (current_atr - atr_mean) / atr_std
                volatility_factor = 1.0 + (z_score * 0.2)  # Scale factor based on z-score
            else:
                volatility_factor = 1.0
            
            # Ensure volatility factor is within reasonable bounds
            volatility_factor = max(0.5, min(2.0, volatility_factor))
            
            logger.info(f"Volatility Analysis:")
            logger.info(f"Current ATR: {current_atr:.5f}")
            logger.info(f"Mean ATR: {atr_mean:.5f}")
            logger.info(f"ATR Std Dev: {atr_std:.5f}")
            logger.info(f"Volatility Factor: {volatility_factor:.2f}")
            
            return {
                'session': session_name,
                'suitable_for_trading': True,
                'session_range': session_data.get('trading_hours', {}),
                'volatility_factor': volatility_factor,
                'reason': 'Conditions suitable for trading'
            }
            
        except Exception as e:
            logger.error(f"Error analyzing session conditions: {str(e)}")
            logger.exception("Detailed error trace:")
            return {
                'session': 'unknown',
                'suitable_for_trading': False,
                'volatility_factor': 1.0,
                'reason': f'Error in analysis: {str(e)}'
            }
        
    def analyze_market_structure(self, df: pd.DataFrame, symbol: str = None, timeframe: str = None) -> Dict:
        """Analyze market structure and return key levels and bias."""
        try:
            # Use existing swing points if available from previous analysis
            if hasattr(self, '_current_swing_points'):
                swing_points = self._current_swing_points
            else:
                # Detect swing points
                swing_points = self.detect_swing_points(df)
                self._current_swing_points = swing_points
            
            # Detect structure breaks
            logger.info("Starting market structure analysis...")
            try:
                structure_breaks = self.detect_structure_breaks(df, swing_points)
                logger.info(f"Structure breaks detected - Bullish: {len(structure_breaks['bullish'])}, "
                          f"Bearish: {len(structure_breaks['bearish'])}")
            except Exception as e:
                logger.error(f"Error in market structure analysis: {str(e)}")
                structure_breaks = {'bullish': [], 'bearish': [], 'strength': [], 'timestamps': []}
            
            # Determine structure type
            structure_type = self._determine_structure_type(swing_points, structure_breaks)
            
            # Find key levels
            key_levels = self._find_key_levels(df, swing_points, structure_type)
            
            # Determine market bias
            market_bias = "neutral"
            if structure_type in ["Uptrend", "Accumulation"]:
                market_bias = "bullish"
            elif structure_type in ["Downtrend", "Distribution"]:
                market_bias = "bearish"
            
            # Calculate momentum as percentage change over last 5 bars
            momentum = 0.0
            if len(df) >= 6:
                momentum = ((df['close'].iloc[-1] - df['close'].iloc[-6]) / df['close'].iloc[-6]) * 100
            
            # Assess structure quality
            quality_assessment = self._assess_structure_quality(df, swing_points.get('highs', []), swing_points.get('lows', []))
            quality_score = quality_assessment.get('quality_score', 0.0)
            
            # Log market structure analysis once
            logger.info("🏗️ Market Structure Analysis:")
            logger.info(f"    Structure Type: {structure_type}")
            logger.info(f"    Market Bias: {market_bias}")
            logger.info(f"    Key Levels: {[round(level, 3) for level in key_levels[:10]]}")
            logger.info(f"    Swing Points: {len(swing_points.get('highs', []))} highs, {len(swing_points.get('lows', []))} lows")
            logger.info(f"    Structure Breaks: {len(structure_breaks.get('bullish', []))} bullish, {len(structure_breaks.get('bearish', []))} bearish")
            logger.info(f"    Momentum: {momentum:.2f}%")
            logger.info(f"    Quality Score: {quality_score:.2f}")
            
            return {
                'structure_type': structure_type,
                'market_bias': market_bias,
                'key_levels': key_levels[:10],  # Store only top 10 levels
                'swing_points': swing_points,
                'structure_breaks': structure_breaks,
                'momentum': momentum,
                'quality_score': quality_score
            }
            
        except Exception as e:
            logger.error(f"Error in market structure analysis: {str(e)}")
            return {
                'structure_type': "Unknown",
                'market_bias': "neutral",
                'key_levels': [],
                'swing_points': {'highs': [], 'lows': []},
                'structure_breaks': {'bullish': [], 'bearish': []},
                'momentum': 0.0,
                'quality_score': 0.0
            }
        
    def _reset_stats(self):
        """Reset statistics tracking."""
        self.stats = {
            'swing_points': {'highs': 0, 'lows': 0, 'rejected': 0},
            'structure_breaks': {'bullish': 0, 'bearish': 0, 'invalid': 0},
            'fvg_gaps': {'bullish': 0, 'bearish': 0, 'filled': 0},
            'order_blocks': {'bullish': 0, 'bearish': 0, 'tested': 0}
        }
        
    def _log_analysis_stats(self):
        """Log detailed analysis statistics."""
        logger.info("Market Structure Analysis Statistics:")
        logger.info("Swing Points:")
        logger.info(f"- Highs detected: {self.stats['swing_points']['highs']}")
        logger.info(f"- Lows detected: {self.stats['swing_points']['lows']}")
        logger.info(f"- Rejected points: {self.stats['swing_points']['rejected']}")
        
        logger.info("Structure Breaks:")
        logger.info(f"- Bullish breaks: {self.stats['structure_breaks']['bullish']}")
        logger.info(f"- Bearish breaks: {self.stats['structure_breaks']['bearish']}")
        logger.info(f"- Invalid breaks: {self.stats['structure_breaks']['invalid']}")
        
        logger.info("Fair Value Gaps:")
        logger.info(f"- Upward gaps: {self.stats['fvg_gaps']['bullish']}")
        logger.info(f"- Downward gaps: {self.stats['fvg_gaps']['bearish']}")
        logger.info(f"- Filled gaps: {self.stats['fvg_gaps']['filled']}")
        
        logger.info("Order Blocks:")
        logger.info(f"- Bullish blocks: {self.stats['order_blocks']['bullish']}")
        logger.info(f"- Bearish blocks: {self.stats['order_blocks']['bearish']}")
        logger.info(f"- Tested blocks: {self.stats['order_blocks']['tested']}")
        
    def _assess_structure_quality(self, df: pd.DataFrame, swing_highs: List[Dict], swing_lows: List[Dict]) -> Dict:
        """Assess the quality of detected market structure with stricter criteria."""
        try:
            quality_score = 0.5  # Start with neutral score
            reasons = []
            
            # Check minimum number of swing points with higher requirement
            min_points = max(self.min_swing_points * 2, 4)  # Increased minimum points
            if len(swing_highs) < min_points or len(swing_lows) < min_points:
                quality_score -= 0.2
                reasons.append("Insufficient swing points for reliable structure")
            
            # Check swing point distribution
            if len(swing_highs) > 0 and len(swing_lows) > 0:
                # Calculate average distances
                high_distances = [h2['index'] - h1['index'] 
                                for h1, h2 in zip(swing_highs[:-1], swing_highs[1:])]
                low_distances = [l2['index'] - l1['index'] 
                               for l1, l2 in zip(swing_lows[:-1], swing_lows[1:])]
                
                if high_distances and low_distances:
                    avg_distance = (sum(high_distances) + sum(low_distances)) / (len(high_distances) + len(low_distances))
                    
                    # Check for clustering
                    max_cluster_size = avg_distance * 0.4  # Stricter clustering threshold
                    cluster_penalty = sum(1 for d in high_distances + low_distances if d < max_cluster_size)
                    quality_score -= 0.1 * cluster_penalty
                    if cluster_penalty > 0:
                        reasons.append(f"Found {cluster_penalty} clustered swing points")
                    
                    # Check for irregular spacing
                    std_dev = np.std(high_distances + low_distances)
                    if std_dev > avg_distance * 0.5:  # Check for irregular spacing
                        quality_score -= 0.15
                        reasons.append("Irregular swing point spacing")
            
            # Check swing point magnitudes
            if swing_highs and swing_lows:
                magnitudes = [p['size'] for p in swing_highs + swing_lows]
                avg_magnitude = sum(magnitudes) / len(magnitudes)
                min_magnitude = avg_magnitude * 0.4  # Stricter minimum size
                
                small_swings = sum(1 for m in magnitudes if m < min_magnitude)
                if small_swings > 0:
                    quality_score -= 0.1 * (small_swings / len(magnitudes))
                    reasons.append(f"Found {small_swings} undersized swing points")
            
            # Check trend consistency
            if len(df) >= 50:
                ma20 = df['close'].rolling(window=20).mean()
                ma50 = df['close'].rolling(window=50).mean()
                
                # Calculate trend consistency
                trend_changes = sum(1 for i in range(1, len(ma20)) if 
                                  (ma20.iloc[i] > ma50.iloc[i]) != (ma20.iloc[i-1] > ma50.iloc[i-1]))
                
                if trend_changes > 3:  # More than 3 trend changes in the period
                    quality_score -= 0.2
                    reasons.append("Inconsistent trend direction")
            
            # Add momentum check
            if len(df) >= 20:
                momentum = ((df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20]) * 100
                if abs(momentum) < 0.1:  # Less than 0.1% price change
                    quality_score -= 0.15
                    reasons.append("Low momentum")
            
            # Ensure score is between 0 and 1
            quality_score = max(0.0, min(1.0, quality_score))
            
            return {
                'quality_score': quality_score,
                'reasons': reasons
            }
            
        except Exception as e:
            logger.error(f"Error assessing structure quality: {str(e)}")
            return {
                'quality_score': 0.0,
                'reasons': ["Error in quality assessment"]
            }
            
    def _validate_swing_sequence(self, swing_highs: List[Dict], swing_lows: List[Dict]) -> bool:
        """Validate the sequence of swing points."""
        try:
            if not swing_highs or not swing_lows:
                return False
                
            # Combine and sort all swing points by index
            all_swings = [(h['index'], 'high') for h in swing_highs] + \
                        [(l['index'], 'low') for l in swing_lows]
            all_swings.sort()
            
            # Check alternation (high-low-high or low-high-low)
            for i in range(1, len(all_swings)):
                if all_swings[i][1] == all_swings[i-1][1]:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating swing sequence: {str(e)}")
            return False

    def analyze(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Dict:
        """Analyze market conditions and structure."""
        try:
            logger.info(f"🔍 Starting market analysis for {symbol} on {timeframe}")
            
            if df is None or len(df) < 100:
                logger.error("Insufficient data for market analysis")
                return {}
            
            current_price = df['close'].iloc[-1]
            daily_range = ((df['high'].max() - df['low'].min()) / df['close'].mean()) * 100
            
            logger.info(f"📊 Market Stats for {symbol}:")
            logger.info(f"    Current Price: {current_price:.5f}")
            logger.info(f"    Daily Range: {daily_range:.2f}%")
            logger.info(f"    Analyzing {len(df)} candles ({timeframe} timeframe)")
            
            # Reset statistics
            self._reset_stats()
            
            # Detect swing points with enhanced logging
            logger.info("Starting swing point detection...")
            swing_points = self.detect_swing_points(df)
            if not swing_points['highs'] and not swing_points['lows']:
                logger.warning("No swing points detected in the current market")
            else:
                logger.info(f"Detected {len(swing_points['highs'])} swing highs and {len(swing_points['lows'])} swing lows")
            
            # Store swing points for reuse
            self._current_swing_points = swing_points
            
            # Detect structure breaks
            logger.info("Starting market structure analysis...")
            try:
                structure_breaks = self.detect_structure_breaks(df, swing_points)
                logger.info(f"Structure breaks detected - Bullish: {len(structure_breaks['bullish'])}, "
                          f"Bearish: {len(structure_breaks['bearish'])}")
            except Exception as e:
                logger.error(f"Error in market structure analysis: {str(e)}")
                structure_breaks = {'bullish': [], 'bearish': [], 'strength': [], 'timestamps': []}
            
            # Volume Analysis
            logger.info("Starting volume analysis...")
            volume_data = self._calculate_volume_trend(df)
            logger.info("📊 Volume Analysis Details:")
            logger.info(f"    Average Volume: {volume_data.get('average_volume', 0):.2f}")
            logger.info(f"    Recent Volume: {volume_data.get('recent_volume', 0):.2f}")
            logger.info(f"    Volume Trend: {volume_data.get('trend', 'Unknown')}")
            logger.info(f"    Volume Strength: {volume_data.get('strength', 0):.2f}x average")
            
            # Market Structure Analysis
            market_structure = self.analyze_market_structure(df, symbol, timeframe)
            logger.info("🏗️ Market Structure:")
            logger.info(f"    Structure Type: {market_structure.get('structure_type', 'Unknown')}")
            logger.info(f"    Key Levels: {[round(level, 3) for level in market_structure.get('key_levels', [])[:10]]}")  # Show only top 10 levels
            
            # Session Analysis
            session_conditions = self.analyze_session_conditions(df, symbol)
            logger.info(f"📈 Session Analysis:")
            logger.info(f"    Current Session: {session_conditions.get('session', 'Unknown')}")
            logger.info(f"    Volatility State: {session_conditions.get('volatility_factor', 0.0):.2f}")
            logger.info(f"    Trading Allowed: {'✅' if session_conditions.get('suitable_for_trading', False) else '❌'}")
            
            # Final Analysis Summary
            analysis_result = {
                'trend': market_structure.get('market_bias', 'neutral'),
                'momentum': market_structure.get('momentum', 0.0),
                'volume_analysis': volume_data,
                'structure_breaks': structure_breaks,
                'session_conditions': session_conditions,
                'key_levels': market_structure.get('key_levels', [])[:10],  # Store only top 10 levels
                'quality_score': market_structure.get('quality_score', 0.0)
            }
            
            logger.info(f"✅ Market analysis completed for {symbol} on {timeframe}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error in market analysis: {str(e)}")
            return {
                'trend': 'neutral',
                'momentum': 0.0,
                'volume_analysis': {},
                'structure_breaks': {'bullish': [], 'bearish': []},
                'session_conditions': {},
                'key_levels': [],
                'quality_score': 0.0
            }

    def _calculate_volume_trend(self, df: pd.DataFrame) -> Dict:
        """Calculate volume trend and related metrics with stricter criteria."""
        try:
            if 'volume' not in df.columns:
                logger.error("Volume data not found in DataFrame")
                return self._get_default_volume_result()
            
            if df['volume'].isnull().all() or (df['volume'] == 0).all():
                logger.error("No valid volume data available")
                return self._get_default_volume_result()
            
            # Calculate volume moving averages with more periods
            vol_sma_20 = df['volume'].rolling(window=20, min_periods=1).mean()
            vol_sma_50 = df['volume'].rolling(window=50, min_periods=1).mean()
            vol_sma_100 = df['volume'].rolling(window=100, min_periods=1).mean()
            
            # Get recent and average volumes
            recent_volume = float(vol_sma_20.iloc[-1]) if not pd.isna(vol_sma_20.iloc[-1]) else 0.0
            medium_volume = float(vol_sma_50.iloc[-1]) if not pd.isna(vol_sma_50.iloc[-1]) else 0.0
            long_term_volume = float(vol_sma_100.iloc[-1]) if not pd.isna(vol_sma_100.iloc[-1]) else 0.0
            
            if long_term_volume == 0:
                logger.warning("Long-term average volume is zero, using simple volume calculations")
                recent_volume = float(df['volume'].tail(20).mean())
                medium_volume = float(df['volume'].tail(50).mean())
                long_term_volume = float(df['volume'].mean())
            
            # Calculate volume trend with stricter criteria
            volume_trend = (recent_volume / long_term_volume) - 1 if long_term_volume > 0 else 0
            medium_trend = (medium_volume / long_term_volume) - 1 if long_term_volume > 0 else 0
            
            # Calculate volume consistency
            volume_std = df['volume'].tail(50).std()
            volume_consistency = 1 - (volume_std / medium_volume) if medium_volume > 0 else 0
            
            # Determine trend description with stricter classification
            if volume_trend > 0.25 and medium_trend > 0.15 and volume_consistency > 0.7:
                trend_desc = "Strongly Increasing"
            elif volume_trend > 0.15 and medium_trend > 0.1 and volume_consistency > 0.6:
                trend_desc = "Increasing"
            elif volume_trend < -0.25 and medium_trend < -0.15 and volume_consistency > 0.7:
                trend_desc = "Strongly Decreasing"
            elif volume_trend < -0.15 and medium_trend < -0.1 and volume_consistency > 0.6:
                trend_desc = "Decreasing"
            elif volume_consistency > 0.8:
                trend_desc = "Stable"
            else:
                trend_desc = "Inconsistent"
            
            # Calculate volume strength with consistency factor
            strength = (recent_volume / long_term_volume) * volume_consistency if long_term_volume > 0 else 1.0
            
            logger.info(f"Volume Analysis Details:")
            logger.info(f"Recent Volume (20-period MA): {recent_volume:.2f}")
            logger.info(f"Medium Volume (50-period MA): {medium_volume:.2f}")
            logger.info(f"Long-term Volume (100-period MA): {long_term_volume:.2f}")
            logger.info(f"Volume Trend: {trend_desc} ({volume_trend:.2%})")
            logger.info(f"Volume Consistency: {volume_consistency:.2f}")
            logger.info(f"Volume Strength: {strength:.2f}x average")
            
            return {
                'average_volume': float(long_term_volume),
                'recent_volume': float(recent_volume),
                'trend': trend_desc,
                'strength': float(strength),
                'trend_value': float(volume_trend),
                'consistency': float(volume_consistency)
            }
            
        except Exception as e:
            logger.error(f"Error calculating volume trend: {str(e)}")
            return self._get_default_volume_result()

    def _get_default_volume_result(self) -> Dict:
        """Return default volume analysis result when calculation fails."""
        return {
            'average_volume': 0.0,
            'recent_volume': 0.0,
            'trend': "Unknown",
            'strength': 0.0,
            'trend_value': 0.0,
            'consistency': 0.0
        }

    def _calculate_smc_score(self, market_structure: Dict) -> float:
        """Calculate Smart Money Concepts score."""
        try:
            # Extract components from market structure
            order_blocks = market_structure.get('order_blocks', {})
            fair_value_gaps = market_structure.get('fair_value_gaps', {})
            
            # Count recent structure elements
            bullish_blocks = len(order_blocks.get('bullish', []))
            bearish_blocks = len(order_blocks.get('bearish', []))
            bullish_fvgs = len(fair_value_gaps.get('bullish', []))
            bearish_fvgs = len(fair_value_gaps.get('bearish', []))
            
            # Calculate bias scores
            block_bias = (bullish_blocks - bearish_blocks) / max(bullish_blocks + bearish_blocks, 1)
            fvg_bias = (bullish_fvgs - bearish_fvgs) / max(bullish_fvgs + bearish_fvgs, 1)
            
            # Combine into final score (-1 to 1)
            score = np.clip((block_bias + fvg_bias) / 2, -1, 1)
            
            return float(score)
            
        except Exception as e:
            logger.error(f"Error calculating SMC score: {str(e)}")
            return 0.0
            
    def _calculate_mtf_score(self, df: pd.DataFrame) -> float:
        """Calculate Multi-Timeframe score."""
        try:
            # Calculate EMAs for different timeframes
            ema_fast = df['close'].ewm(span=20).mean()
            ema_med = df['close'].ewm(span=50).mean()
            ema_slow = df['close'].ewm(span=200).mean()
            
            # Calculate current positions
            curr_price = df['close'].iloc[-1]
            curr_fast = ema_fast.iloc[-1]
            curr_med = ema_med.iloc[-1]
            curr_slow = ema_slow.iloc[-1]
            
            # Calculate alignment score
            if curr_fast > curr_med > curr_slow:
                base_score = 1.0
            elif curr_fast < curr_med < curr_slow:
                base_score = -1.0
            else:
                # Calculate partial alignment
                fast_med = 1 if curr_fast > curr_med else -1
                med_slow = 1 if curr_med > curr_slow else -1
                base_score = (fast_med + med_slow) / 3
            
            # Adjust score based on price position
            price_position = (curr_price - curr_med) / curr_med
            
            # Combine into final score (-1 to 1)
            score = np.clip(base_score * (1 + abs(price_position)), -1, 1)
            
            return float(score)
            
        except Exception as e:
            logger.error(f"Error calculating MTF score: {str(e)}")
            return 0.0

    def _calculate_market_quality(
        self,
        trend_strength: float,
        momentum: float,
        volume_trend: float,
        volatility_state: str,
        market_state: str
    ) -> float:
        """Calculate overall market quality score."""
        try:
            # Base score starts at 0.5
            quality_score = 0.5
            
            # Add trend component (max ±0.2)
            quality_score += trend_strength * 0.2
            
            # Add momentum component (max ±0.15)
            quality_score += momentum * 0.15
            
            # Add volume component (max ±0.15)
            quality_score += volume_trend * 0.15
            
            # Adjust for volatility state
            if volatility_state == 'normal':
                quality_score *= 1.1
            elif volatility_state == 'high':
                quality_score *= 0.8
            elif volatility_state == 'low':
                quality_score *= 0.9
                
            # Adjust for market state
            if market_state == 'trending':
                quality_score *= 1.2
            elif market_state == 'ranging':
                quality_score *= 0.9
            elif market_state == 'transitioning':
                quality_score *= 0.7
                
            # Ensure score is between 0 and 1
            return max(min(quality_score, 1.0), 0.0)
            
        except Exception as e:
            logger.error(f"Error calculating market quality: {str(e)}")
            return 0.5

    def _get_recent_swings(self, swing_points: Dict, lookback: int = 5) -> Dict:
        """Get the most recent swing points for analysis.
        
        Args:
            swing_points (Dict): Dictionary containing 'highs' and 'lows' lists of swing points
            lookback (int): Number of most recent swing points to return
            
        Returns:
            Dict: Dictionary containing most recent swing highs and lows
        """
        try:
            # Get highs and lows from swing points
            highs = swing_points.get('highs', [])
            lows = swing_points.get('lows', [])
            
            # Sort by index/timestamp if available
            if highs and 'index' in highs[0]:
                highs = sorted(highs, key=lambda x: x['index'], reverse=True)
            elif highs and 'timestamp' in highs[0]:
                highs = sorted(highs, key=lambda x: x['timestamp'], reverse=True)
                
            if lows and 'index' in lows[0]:
                lows = sorted(lows, key=lambda x: x['index'], reverse=True)
            elif lows and 'timestamp' in lows[0]:
                lows = sorted(lows, key=lambda x: x['timestamp'], reverse=True)
            
            # Get most recent points
            recent_highs = highs[:lookback]
            recent_lows = lows[:lookback]
            
            return {
                'highs': recent_highs,
                'lows': recent_lows
            }
            
        except Exception as e:
            logger.error(f"Error getting recent swing points: {str(e)}")
            return {'highs': [], 'lows': []}

    def _determine_market_bias(self, recent_swings: Dict) -> str:
        """Determine market bias based on recent swing points.
        
        Args:
            recent_swings (Dict): Dictionary containing recent swing highs and lows
            
        Returns:
            str: Market bias ('bullish', 'bearish', or 'neutral')
        """
        try:
            highs = recent_swings.get('highs', [])
            lows = recent_swings.get('lows', [])
            
            if not highs or not lows:
                return 'neutral'
            
            # Get the most recent high and low
            latest_high = highs[0] if highs else None
            latest_low = lows[0] if lows else None
            
            # Get the previous high and low for comparison
            prev_high = highs[1] if len(highs) > 1 else None
            prev_low = lows[1] if len(lows) > 1 else None
            
            # Determine bias based on swing point sequence
            if latest_high and prev_high and latest_low and prev_low:
                # Higher highs and higher lows = bullish
                if latest_high['price'] > prev_high['price'] and latest_low['price'] > prev_low['price']:
                    return 'bullish'
                # Lower highs and lower lows = bearish
                elif latest_high['price'] < prev_high['price'] and latest_low['price'] < prev_low['price']:
                    return 'bearish'
            
            # If we can't determine a clear bias, return neutral
            return 'neutral'
            
        except Exception as e:
            logger.error(f"Error determining market bias: {str(e)}")
            return 'neutral'

    def _find_order_blocks(self, df: pd.DataFrame) -> Dict:
        """Find bullish and bearish order blocks in price action.
        
        Args:
            df (pd.DataFrame): Price data with OHLC columns
            
        Returns:
            Dict: Dictionary containing bullish and bearish order blocks
        """
        try:
            bullish_blocks = []
            bearish_blocks = []
            
            for i in range(2, len(df)-1):
                # Bullish order block
                if df['close'].iloc[i-1] < df['open'].iloc[i-1] and \
                   df['high'].iloc[i] > df['high'].iloc[i-1]:
                    block = {
                        'high': df['high'].iloc[i-1],
                        'low': min(df['open'].iloc[i-1], df['close'].iloc[i-1]),
                        'index': i-1,
                        'timestamp': df.index[i-1]
                    }
                    bullish_blocks.append(block)
                
                # Bearish order block
                if df['close'].iloc[i-1] > df['open'].iloc[i-1] and \
                   df['low'].iloc[i] < df['low'].iloc[i-1]:
                    block = {
                        'high': max(df['open'].iloc[i-1], df['close'].iloc[i-1]),
                        'low': df['low'].iloc[i-1],
                        'index': i-1,
                        'timestamp': df.index[i-1]
                    }
                    bearish_blocks.append(block)
            
            return {
                'bullish': bullish_blocks,
                'bearish': bearish_blocks
            }
            
        except Exception as e:
            logger.error(f"Error finding order blocks: {str(e)}")
            return {'bullish': [], 'bearish': []}
    
    def _find_fair_value_gaps(self, df: pd.DataFrame) -> Dict:
        """Find fair value gaps in price action.
        
        Args:
            df (pd.DataFrame): Price data with OHLC columns
            
        Returns:
            Dict: Dictionary containing bullish and bearish fair value gaps
        """
        try:
            bullish_fvgs = []
            bearish_fvgs = []
            
            for i in range(2, len(df)-1):
                # Bullish FVG
                if df['low'].iloc[i] > df['high'].iloc[i-2]:
                    fvg = {
                        'top': df['low'].iloc[i],
                        'bottom': df['high'].iloc[i-2],
                        'index': i-1,
                        'timestamp': df.index[i-1]
                    }
                    bullish_fvgs.append(fvg)
                
                # Bearish FVG
                if df['high'].iloc[i] < df['low'].iloc[i-2]:
                    fvg = {
                        'top': df['low'].iloc[i-2],
                        'bottom': df['high'].iloc[i],
                        'index': i-1,
                        'timestamp': df.index[i-1]
                    }
                    bearish_fvgs.append(fvg)
            
            return {
                'bullish': bullish_fvgs,
                'bearish': bearish_fvgs
            }
            
        except Exception as e:
            logger.error(f"Error finding fair value gaps: {str(e)}")
            return {'bullish': [], 'bearish': []}

    def _determine_structure_type(self, swing_points: Dict, structure_breaks: Dict) -> str:
        """Determine market structure type based on swing points and recent breaks."""
        try:
            if not swing_points or 'highs' not in swing_points or 'lows' not in swing_points:
                logger.warning("Insufficient swing points data.")
                return "Sideways"  # Changed from "Insufficient Data"

            highs = swing_points.get('highs') or []
            lows = swing_points.get('lows') or []

            if len(highs) < 2 or len(lows) < 2:
                return "Forming"  # Not enough points to determine structure
                
            # Safely sort swing points using .get() to avoid KeyError
            last_highs = sorted(highs, key=lambda x: x.get('index', 0))
            last_lows = sorted(lows, key=lambda x: x.get('index', 0))
            
            # Take only the last 3 points
            last_highs = last_highs[-3:] if len(last_highs) >= 3 else last_highs
            last_lows = last_lows[-3:] if len(last_lows) >= 3 else last_lows
            
            if len(last_highs) < 2 or len(last_lows) < 2:
                return "Forming"
            
            # Calculate price movements
            high_movement = last_highs[-1]['price'] - last_highs[0]['price']
            low_movement = last_lows[-1]['price'] - last_lows[0]['price']
            
            # Calculate movement thresholds based on ATR or price range
            price_range = max(h['price'] for h in last_highs) - min(l['price'] for l in last_lows)
            movement_threshold = price_range * 0.1  # 10% of the price range
            
            # Check higher highs and higher lows with threshold
            higher_highs = high_movement > movement_threshold
            higher_lows = low_movement > movement_threshold
            
            # Check lower highs and lower lows with threshold
            lower_highs = high_movement < -movement_threshold
            lower_lows = low_movement < -movement_threshold
            
            # Get recent breaks from both bullish and bearish lists
            bullish_breaks = structure_breaks.get('bullish', [])
            bearish_breaks = structure_breaks.get('bearish', [])
            
            # Sort all breaks by index and take the most recent ones
            all_breaks = []
            for break_data in bullish_breaks:
                all_breaks.append({'type': 'bullish', **break_data})
            for break_data in bearish_breaks:
                all_breaks.append({'type': 'bearish', **break_data})
                
            # Sort by index and take last 3
            recent_breaks = sorted(all_breaks, key=lambda x: x['index'])[-3:] if all_breaks else []
            
            # Count break types
            bullish_count = sum(1 for b in recent_breaks if b['type'] == 'bullish')
            bearish_count = sum(1 for b in recent_breaks if b['type'] == 'bearish')
            
            # Calculate price volatility
            price_volatility = price_range / min(l['price'] for l in last_lows) * 100
            
            # Log structure detection details
            logger.info(f"Structure Detection Details:")
            logger.info(f"Higher Highs: {higher_highs}, Higher Lows: {higher_lows}")
            logger.info(f"Lower Highs: {lower_highs}, Lower Lows: {lower_lows}")
            logger.info(f"Recent Bullish Breaks: {bullish_count}, Recent Bearish Breaks: {bearish_count}")
            logger.info(f"Price Volatility: {price_volatility:.2f}%")
            logger.info(f"Movement Threshold: {movement_threshold:.5f}")
            
            # Enhanced structure type determination
            structure_type = "Unknown"
            
            # Clear trend conditions
            if higher_highs and higher_lows:
                structure_type = "Uptrend"
            elif lower_highs and lower_lows:
                structure_type = "Downtrend"
            # Accumulation/Distribution conditions
            elif abs(high_movement) < movement_threshold and higher_lows:
                structure_type = "Accumulation"
            elif abs(low_movement) < movement_threshold and lower_highs:
                structure_type = "Distribution"
            # Ranging conditions
            elif abs(high_movement) < movement_threshold and abs(low_movement) < movement_threshold:
                if price_volatility > 0.5:  # More than 0.5% range
                    structure_type = "Ranging"
                else:
                    structure_type = "Consolidation"
            # Transition conditions
            elif (higher_highs and lower_lows) or (lower_highs and higher_lows):
                structure_type = "Transition"
            
            # If still unknown, use swing point analysis
            if structure_type == "Unknown":
                # Compare last two swing points
                if len(last_highs) >= 2 and len(last_lows) >= 2:
                    last_high = last_highs[-1]['price']
                    prev_high = last_highs[-2]['price']
                    last_low = last_lows[-1]['price']
                    prev_low = last_lows[-2]['price']

                    if last_high > prev_high and last_low > prev_low:
                        structure_type = "Uptrend"
                    elif last_high < prev_high and last_low < prev_low:
                        structure_type = "Downtrend"
                    else:
                        # Check if price is in a tight range
                        price_range_percent = (last_high - last_low) / last_low * 100
                        if price_range_percent < 0.3:  # Less than 0.3% range
                            structure_type = "Consolidation"
                        else:
                            structure_type = "Ranging"
            
            logger.info(f"Determined Structure Type: {structure_type}")
            return structure_type
                
        except Exception as e:
            logger.error(f"Error determining structure type: {str(e)}")
            logger.exception("Detailed error trace:")
            return "Ranging"  # Changed from "Unknown" to provide a more useful default

    def _find_key_levels(self, df: pd.DataFrame, swing_points: Dict, structure_type: str) -> List[float]:
        """Find key price levels based on market structure."""
        try:
            key_levels = set()
            
            # Add swing point levels
            for high in swing_points.get('highs', []):
                key_levels.add(round(high['price'], 3))
            for low in swing_points.get('lows', []):
                key_levels.add(round(low['price'], 3))
            
            # Add structure-specific levels
            if structure_type in ["Uptrend", "Accumulation"]:
                # Add recent higher lows
                lows = sorted(swing_points.get('lows', []), key=lambda x: x['index'])[-3:]
                for low in lows:
                    key_levels.add(round(low['price'], 3))
                    
            elif structure_type in ["Downtrend", "Distribution"]:
                # Add recent lower highs
                highs = sorted(swing_points.get('highs', []), key=lambda x: x['index'])[-3:]
                for high in highs:
                    key_levels.add(round(high['price'], 3))
                    
            elif structure_type == "Ranging":
                # Add range boundaries
                if swing_points.get('highs') and swing_points.get('lows'):
                    recent_high = max(h['price'] for h in swing_points['highs'][-3:])
                    recent_low = min(l['price'] for l in swing_points['lows'][-3:])
                    key_levels.add(round(recent_high, 3))
                    key_levels.add(round(recent_low, 3))
            
            return sorted(list(key_levels))
            
        except Exception as e:
            logger.error(f"Error finding key levels: {str(e)}")
            return []

    def classify_trend(self, df: pd.DataFrame) -> str:
        """
        Classify the market trend based on recent price data.

        Args:
            df (pd.DataFrame): DataFrame with a 'close' column.

        Returns:
            str: 'uptrend', 'downtrend', 'sideways', or 'unknown'.
        """
        try:
            ma = df['close'].rolling(window=20).mean()
            last_close = df['close'].iloc[-1]
            last_ma = ma.iloc[-1]

            if last_close > last_ma:
                return "uptrend"
            elif last_close < last_ma:
                return "downtrend"
            else:
                return "sideways"
        except Exception as e:
            logger.error(f"Error classifying trend: {str(e)}")
            return "unknown"

    def classify_volatility(self, atr: pd.Series) -> str:
        """
        Classify market volatility based on ATR values.

        Args:
            atr (pd.Series): Series of ATR values.

        Returns:
            str: 'low', 'medium', 'high', or 'unknown'.
        """
        try:
            avg_atr = atr.mean()
            if avg_atr < 0.001:
                return "low"
            elif avg_atr < 0.003:
                return "medium"
            else:
                return "high"
        except Exception as e:
            logger.error(f"Error classifying volatility: {str(e)}")
            return "unknown"