import pandas as pd
import numpy as np
from datetime import datetime, time, UTC, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from loguru import logger
from config.config import SESSION_CONFIG, MARKET_STRUCTURE_CONFIG

from .mtf_analysis import MTFAnalysis
from .mt5_handler import MT5Handler

class MarketAnalysis:
    def __init__(self, ob_threshold: float = 0.0015):
        """
        Initialize MarketAnalysis with configuration parameters.
        
        Args:
            ob_threshold: Threshold for order block detection
        """
        # Constants
        self.SWING_SIZE = 10
        self.ATR_PERIOD = 14
        self.RSI_PERIOD = 14
        self.MIN_SWING_SEQUENCE = 3
        self.OB_THRESHOLD = ob_threshold
        
        # Session configurations
        self.SESSIONS = {
            'Asian': {'start': time(0, 0), 'end': time(8, 0)},
            'London': {'start': time(8, 0), 'end': time(16, 0)},
            'New York': {'start': time(13, 0), 'end': time(21, 0)}
        }
        
        self.KILLZONES = {
            'Asian': {'start': time(0, 0), 'end': time(3, 0)},
            'London': {'start': time(8, 0), 'end': time(11, 0)},
            'New York': {'start': time(13, 0), 'end': time(16, 0)}
        }

        # Market schedule for holidays and partial trading days
        self.market_schedule = {
            "holidays": {
                "2024": {
                    "new_years": "2024-01-01",
                    "good_friday": "2024-03-29",
                    "easter_monday": "2024-04-01",
                    "memorial_day": "2024-05-27",
                    "independence_day": "2024-07-04",
                    "labor_day": "2024-09-02",
                    "thanksgiving": "2024-11-28",
                    "christmas": "2024-12-25",
                    "boxing_day": "2024-12-26"
                }
            },
            "partial_trading_days": {
                "2024": {
                    "christmas_eve": {
                        "date": "2024-12-24",
                        "close_time": "13:00"
                    },
                    "new_years_eve": {
                        "date": "2024-12-31",
                        "close_time": "13:00"
                    }
                }
            }
        }
        
        self._reset_stats()
        
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
        
        self.mtf_analysis = MTFAnalysis()
        
        # Constants for market analysis
        self.SWING_SIZE = 10
        self.ATR_PERIOD = 14
        self.RSI_PERIOD = 14
        self.MIN_SWING_SEQUENCE = 3
        
        # Session times (UTC)
        self.SESSIONS = {
            'Asian': {'start': time(0, 0), 'end': time(8, 0)},
            'London': {'start': time(8, 0), 'end': time(16, 0)},
            'New York': {'start': time(13, 0), 'end': time(21, 0)}
        }
        
        # Killzone times (UTC)
        self.KILLZONES = {
            'Asian': {'start': time(0, 0), 'end': time(3, 0)},
            'London': {'start': time(8, 0), 'end': time(11, 0)},
            'New York': {'start': time(13, 0), 'end': time(16, 0)}
        }
        
        self.mt5_handler = MT5Handler()
        
    def calculate_atr(self, df: pd.DataFrame, period: int = None) -> pd.Series:
        """Calculate Average True Range."""
        try:
            period = period or self.ATR_PERIOD
            
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            
            return true_range.rolling(period).mean()
            
        except Exception as e:
            logger.error(f"Error calculating ATR: {str(e)}")
            return pd.Series(index=df.index)
        
    def get_current_session(self) -> Tuple[str, Dict]:
        """Determine the current trading session based on UTC time."""
        current_time = datetime.now(UTC).time()
        
        # Define standard session names
        SESSION_NAMES = {
            'asia_session': 'asia',
            'london_session': 'london',
            'new_york_session': 'new_york',
            'no_session': 'no_session'
        }
        
        # Add buffer time around sessions (15 minutes)
        buffer_minutes = 15
        
        logger.info(f"Checking current session at UTC time: {current_time.strftime('%H:%M:%S')}")
        
        for session_name, session_data in self.session_config.items():
            if not session_data.get('enabled', True):
                logger.debug(f"Session {session_name} is disabled, skipping")
                continue
                
            start_time = datetime.strptime(session_data['start'], '%H:%M').time()
            end_time = datetime.strptime(session_data['end'], '%H:%M').time()
            
            logger.debug(f"Evaluating {session_name}: {start_time.strftime('%H:%M')} - {end_time.strftime('%H:%M')}")
            
            # Adjust times with buffer
            start_with_buffer = (
                datetime.combine(datetime.today(), start_time) - 
                timedelta(minutes=buffer_minutes)
            ).time()
            
            end_with_buffer = (
                datetime.combine(datetime.today(), end_time) + 
                timedelta(minutes=buffer_minutes)
            ).time()
            
            # Handle sessions that cross midnight
            if start_time > end_time:
                if current_time >= start_with_buffer or current_time <= end_with_buffer:
                    logger.info(f"In session: {session_name} (crosses midnight)")
                    return SESSION_NAMES.get(session_name, session_name), session_data
            else:
                if start_with_buffer <= current_time <= end_with_buffer:
                    logger.info(f"In session: {session_name}")
                    return SESSION_NAMES.get(session_name, session_name), session_data
        
        # Check if we're close to the next session
        for session_name, session_data in self.session_config.items():
            if not session_data.get('enabled', True):
                continue
                
            start_time = datetime.strptime(session_data['start'], '%H:%M').time()
            time_to_session = (
                datetime.combine(datetime.today(), start_time) - 
                datetime.combine(datetime.today(), current_time)
            ).total_seconds() / 60
            
            if 0 <= time_to_session <= 30:  # Within 30 minutes of next session
                logger.info(f"Approaching session: {session_name} (within 30 minutes)")
                return SESSION_NAMES.get(session_name, session_name), session_data
        
        logger.info("No active trading session detected")
        return SESSION_NAMES['no_session'], {}
        
    def detect_swing_points(
        self,
        df: pd.DataFrame,
        swing_size: int = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Detect swing highs and lows in price data.

        Args:
            df: DataFrame with OHLC data
            swing_size: Number of bars to look back/forward (default: self.SWING_SIZE)

        Returns:
            Dict with 'highs' and 'lows' containing swing points
        """
        try:
            swing_size = swing_size or self.SWING_SIZE
            
            # Calculate ATR for dynamic thresholds
            atr = self.calculate_atr(df)
            
            swing_highs = []
            swing_lows = []

            for i in range(swing_size, len(df) - swing_size):
                # Get price window
                window = df.iloc[i - swing_size:i + swing_size + 1]
                current_price = df['high'].iloc[i]
                
                # Dynamic threshold based on ATR
                threshold = atr.iloc[i] * 0.5
                
                # Check for swing high
                if self._validate_swing_high(df, i, swing_size, threshold):
                    swing_highs.append({
                        'index': i,
                        'price': current_price,
                        'time': df.index[i]
                    })
                
                # Check for swing low
                current_price = df['low'].iloc[i]
                if self._validate_swing_low(df, i, swing_size, threshold):
                    swing_lows.append({
                        'index': i,
                        'price': current_price,
                        'time': df.index[i]
                    })

            return {
                'highs': swing_highs,
                'lows': swing_lows
            }

        except Exception as e:
            logger.error(f"Error detecting swing points: {str(e)}")
            return {'highs': [], 'lows': []}
            
    def _validate_swing_high(
        self,
        df: pd.DataFrame,
        index: int,
        lookback: int,
        threshold: float
    ) -> bool:
        """
        Validate if a point is a swing high.
        
        Args:
            df: DataFrame with price data
            index: Current index to check
            lookback: Number of bars to look back/forward
            threshold: Minimum price difference for valid swing
            
        Returns:
            bool: True if valid swing high
        """
        try:
            current_high = df['high'].iloc[index]
            
            # Check left side
            left_window = df['high'].iloc[index - lookback:index]
            if not all(current_high > price + threshold for price in left_window):
                return False
                
            # Check right side
            right_window = df['high'].iloc[index + 1:index + lookback + 1]
            if not all(current_high > price + threshold for price in right_window):
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating swing high: {str(e)}")
            return False
            
    def _validate_swing_low(
        self,
        df: pd.DataFrame,
        index: int,
        lookback: int,
        threshold: float
    ) -> bool:
        """
        Validate if a point is a swing low.
        
        Args:
            df: DataFrame with price data
            index: Current index to check
            lookback: Number of bars to look back/forward
            threshold: Minimum price difference for valid swing
            
        Returns:
            bool: True if valid swing low
        """
        try:
            current_low = df['low'].iloc[index]
            
            # Check left side
            left_window = df['low'].iloc[index - lookback:index]
            if not all(current_low < price - threshold for price in left_window):
                return False
                
            # Check right side
            right_window = df['low'].iloc[index + 1:index + lookback + 1]
            if not all(current_low < price - threshold for price in right_window):
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating swing low: {str(e)}")
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
                   df['low'].iloc[i] < df['low'].iloc[i-1]:
                    block = {
                        'high': max(df['open'].iloc[i-1], df['close'].iloc[i-1]),
                        'low': df['low'].iloc[i-1],
                        'index': i-1,
                        'timestamp': df.index[i-1]
                    }
                    bearish_obs.append(block)
            
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
            
            # Check for breaks
            for fvg in bullish_fvgs:
                if df['low'].iloc[i] < fvg['bottom']:
                    fvg['broken_bottom'] = True
                if df['low'].iloc[i] < fvg['top']:
                    fvg['broken_top'] = True
            
            for fvg in bearish_fvgs:
                if df['high'].iloc[i] > fvg['top']:
                    fvg['broken_top'] = True
                if df['high'].iloc[i] > fvg['bottom']:
                    fvg['broken_bottom'] = True
            
            return {
                'bullish': bullish_fvgs,
                'bearish': bearish_fvgs
            }
            
        except Exception as e:
            logger.error(f"Error detecting FVGs: {str(e)}")
            return {'bullish': [], 'bearish': []}
        
        
        
    def analyze_market_structure(
        self,
        df: pd.DataFrame,
        symbol: str = None,
        timeframe: str = None
    ) -> Dict[str, Any]:
        """
        Analyze market structure including swing points, breaks, and order blocks.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Optional trading symbol for logging
            timeframe: Optional timeframe for logging
            
        Returns:
            Dict containing structure analysis:
            {
                'swing_points': Detected swing points,
                'structure_breaks': Structure break points,
                'order_blocks': Detected order blocks,
                'fair_value_gaps': Fair value gaps,
                'bias': Market bias,
                'key_levels': Important price levels,
                'quality': Structure quality metrics
            }
        """
        try:
            # Get swing points
            swing_points = self.detect_swing_points(df)
            
            # Detect structure breaks
            structure_breaks = self.detect_structure_breaks(df, swing_points)
            
            # Detect order blocks
            order_blocks = self.detect_order_blocks(df, (swing_points['highs'], swing_points['lows']))
            
            # Detect fair value gaps
            fair_value_gaps = self.detect_fair_value_gaps(df)
            
            # Determine market bias
            bias = self._determine_market_bias(swing_points)
            
            # Find key levels
            key_levels = self._find_key_levels(df, swing_points, bias)
            
            # Assess structure quality
            quality = self._assess_structure_quality(
                df,
                swing_points['highs'],
                swing_points['lows']
            )
            
            return {
                'swing_points': swing_points,
                'structure_breaks': structure_breaks,
                'order_blocks': order_blocks,
                'fair_value_gaps': fair_value_gaps,
                'bias': bias,
                'key_levels': key_levels,
                'quality': quality
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market structure: {str(e)}")
            return {
                'swing_points': {'highs': [], 'lows': []},
                'structure_breaks': {'bullish': [], 'bearish': []},
                'order_blocks': {'bullish': [], 'bearish': []},
                'fair_value_gaps': {'bullish': [], 'bearish': []},
                'bias': 'neutral',
                'key_levels': [],
                'quality': {'score': 0.0}
            }
    
    def _determine_market_bias(self, swing_points: Dict[str, List[Dict[str, Any]]]) -> str:
        """
        Determine market bias based on swing points.
        
        Args:
            swing_points: Dict containing swing highs and lows
            
        Returns:
            str: Market bias ('Bullish', 'Bearish', or 'Neutral')
        """
        try:
            # Get recent swings
            recent_swings = self._get_recent_swings(swing_points)
            highs = recent_swings['highs']
            lows = recent_swings['lows']
            
            if len(highs) < 2 or len(lows) < 2:
                return 'Neutral'
            
            # Check for higher highs and higher lows (bullish)
            higher_highs = all(highs[i]['price'] > highs[i-1]['price'] 
                             for i in range(1, len(highs)))
            higher_lows = all(lows[i]['price'] > lows[i-1]['price'] 
                            for i in range(1, len(lows)))
            
            # Check for lower highs and lower lows (bearish)
            lower_highs = all(highs[i]['price'] < highs[i-1]['price'] 
                            for i in range(1, len(highs)))
            lower_lows = all(lows[i]['price'] < lows[i-1]['price'] 
                           for i in range(1, len(lows)))
            
            if higher_highs and higher_lows:
                return 'Bullish'
            elif lower_highs and lower_lows:
                return 'Bearish'
            else:
                return 'Neutral'
                
        except Exception as e:
            logger.error(f"Error determining market bias: {str(e)}")
            return 'Neutral'
    
    def _find_key_levels(
        self,
        df: pd.DataFrame,
        swing_points: Dict[str, List[Dict[str, Any]]],
        bias: str
    ) -> List[float]:
        """
        Find key price levels based on swing points and market bias.
        
        Args:
            df: OHLC price data
            swing_points: Dictionary containing swing highs and lows
            bias: Current market bias ('bullish' or 'bearish')
            
        Returns:
            List of key price levels
        """
        try:
            key_levels = set()
            
            # Add swing high/low levels
            for high in swing_points['highs'][-5:]:  # Consider last 5 swing highs
                key_levels.add(round(high['price'], 5))
            for low in swing_points['lows'][-5:]:    # Consider last 5 swing lows
                key_levels.add(round(low['price'], 5))
                
            # Add psychological levels
            current_price = df['close'].iloc[-1]
            psych_levels = self._get_psychological_levels(current_price)
            key_levels.update(psych_levels)
            
            # Add recent session high/low
            session_high = df['high'].tail(24).max()  # Last 24 candles
            session_low = df['low'].tail(24).min()
            key_levels.add(round(session_high, 5))
            key_levels.add(round(session_low, 5))
            
            return sorted(list(key_levels))
            
        except Exception as e:
            logger.error(f"Error finding key levels: {str(e)}")
            return []
    
    def _get_psychological_levels(self, price: float) -> List[float]:
        """
        Get psychological price levels around current price.
        
        Args:
            price: Current price
            
        Returns:
            List[float]: Psychological levels
        """
        try:
            levels = []
            
            # Round price to determine magnitude
            magnitude = 10 ** (len(str(int(price))) - 1)
            
            # Add levels at 0.0, 0.25, 0.5, 0.75 intervals
            base = int(price / magnitude) * magnitude
            for i in range(-2, 3):
                level = base + (i * magnitude)
                levels.append(level)
                levels.append(level + 0.25 * magnitude)
                levels.append(level + 0.5 * magnitude)
                levels.append(level + 0.75 * magnitude)
            
            return sorted(levels)
            
        except Exception as e:
            logger.error(f"Error getting psychological levels: {str(e)}")
            return []
    
    async def analyze_market(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """
        Perform comprehensive market analysis including structure, indicators, and session data.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe to analyze
            
        Returns:
            Dict containing complete market analysis or None if error:
            {
                'structure': Market structure analysis,
                'indicators': Technical indicators,
                'session': Current trading session info,
                'trend': Trend analysis,
                'killzones': Trading killzones analysis,
                'quality': Market quality score
            }
        """
        try:
            # Get market data
            df = await self.get_market_data(symbol, timeframe)
            if df is None or df.empty:
                logger.error("No market data available for analysis")
                return None
                
            # Analyze market structure
            structure = self.analyze_market_structure(df)
            
            # Calculate indicators
            indicators = self.calculate_indicators(df)
            
            # Get current session
            session = self.get_current_session()
            
            # Analyze trend
            trend = self.analyze_trend(df)
            
            # Check killzones
            killzones = self.detect_killzones(df)
            
            # Calculate market quality
            quality = self._calculate_market_quality(
                trend_strength=structure['quality'].get('trend_strength', 0.0),
                momentum=self._calculate_momentum(df),
                volume_trend=self._calculate_volume_trend(df).get('strength', 0.0),
                volatility_state=self.classify_volatility(df['atr']),
                market_state=trend.get('state', 'neutral')
            )
            
            return {
                'structure': structure,
                'indicators': indicators,
                'session': session,
                'trend': trend,
                'killzones': killzones,
                'quality': quality
            }
            
        except Exception as e:
            logger.error(f"Error in market analysis: {str(e)}")
            return None
                
    def calculate_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate technical indicators.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            Dict containing calculated indicators
        """
        try:
            # Calculate ATR
            atr = self.calculate_atr(df)
            
            # Calculate RSI
            rsi = self.calculate_rsi(df['close'])
            
            # Calculate volatility state
            volatility = self.classify_volatility(atr)
            
            return {
                'atr': atr.iloc[-1],
                'rsi': rsi.iloc[-1],
                'volatility': volatility
            }
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
        return {}
    
    def calculate_rsi(self, prices: pd.Series, period: int = None) -> pd.Series:
        """
        Calculate Relative Strength Index.
        
        Args:
            prices: Price series
            period: RSI period (default: self.RSI_PERIOD)
            
        Returns:
            pd.Series: RSI values
        """
        try:
            period = period or self.RSI_PERIOD
            
            # Calculate price changes
            delta = prices.diff()
            
            # Separate gains and losses
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            
            # Calculate average gains and losses
            avg_gains = gains.rolling(window=period).mean()
            avg_losses = losses.rolling(window=period).mean()
            
            # Calculate RS and RSI
            rs = avg_gains / avg_losses
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {str(e)}")
            return pd.Series(index=prices.index)
    
    def analyze_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze price trend using multiple indicators and swing points.
        
        Args:
            df: OHLC price data
            
        Returns:
            Dictionary containing trend analysis results including direction,
            strength, momentum and consistency
        """
        try:
            # Get swing points
            swing_points = self.detect_swing_points(df)
            recent_swings = self._get_recent_swings(swing_points)
            
            # Initialize trend metrics
            trend_metrics = {
                'direction': 'Ranging',
                'strength': 0.0,
                'momentum': 0.0,
                'consistency': 0.0
            }
            
            # Calculate trend direction and strength
            if len(recent_swings['highs']) >= 2 and len(recent_swings['lows']) >= 2:
                highs = recent_swings['highs']
                lows = recent_swings['lows']
                
                # Check trend patterns
                higher_highs = highs[-1]['price'] > highs[-2]['price']
                higher_lows = lows[-1]['price'] > lows[-2]['price']
                lower_highs = highs[-1]['price'] < highs[-2]['price']
                lower_lows = lows[-1]['price'] < lows[-2]['price']
                
                if higher_highs and higher_lows:
                    trend_metrics['direction'] = 'Uptrend'
                    trend_metrics['strength'] = self._calculate_trend_strength(highs, lows)
                elif lower_highs and lower_lows:
                    trend_metrics['direction'] = 'Downtrend'
                    trend_metrics['strength'] = self._calculate_trend_strength(highs, lows)
                
                # Calculate additional metrics
                trend_metrics['momentum'] = self._calculate_momentum(df)
                trend_metrics['consistency'] = self._calculate_trend_consistency(df)
            
            return trend_metrics
            
        except Exception as e:
            logger.error(f"Error analyzing trend: {str(e)}")
            return {
                'direction': 'Ranging',
                'strength': 0.0,
                'momentum': 0.0,
                'consistency': 0.0
            }

    def _calculate_momentum(self, df: pd.DataFrame) -> float:
        """Calculate price momentum using RSI and price changes."""
        try:
            # Calculate RSI
            rsi = self.calculate_rsi(df['close'])
            
            # Calculate recent price changes
            price_changes = df['close'].pct_change()
            recent_momentum = price_changes.tail(5).mean()
            
            # Combine RSI and price momentum
            rsi_score = (rsi.iloc[-1] - 50) / 50  # Normalize RSI to [-1, 1]
            momentum = (rsi_score + recent_momentum) / 2
            
            return float(momentum)
            
        except Exception as e:
            logger.error(f"Error calculating momentum: {str(e)}")
            return 0.0

    def _calculate_trend_consistency(self, df: pd.DataFrame) -> float:
        """Calculate consistency of price movements."""
        try:
            # Get price changes
            changes = df['close'].pct_change()
            
            # Calculate directional consistency
            positive_moves = (changes > 0).sum()
            total_moves = len(changes)
            
            if total_moves > 0:
                consistency = abs((positive_moves / total_moves) - 0.5) * 2
                return float(consistency)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating trend consistency: {str(e)}")
            return 0.0

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

    def detect_killzones(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect active trading killzones and their characteristics.
        
        Args:
            df: OHLC price data
            
        Returns:
            Dictionary containing:
            - active_zones: List of currently active killzones
            - next_zone: Name of next upcoming killzone
            - time_to_next: Minutes until next killzone
            - volatility: Volatility metrics for each zone
        """
        try:
            current_time = datetime.now().time()
            
            killzones = {
                'active_zones': [],
                'next_zone': None,
                'time_to_next': None,
                'volatility': {}
            }
            
            min_time_to_next = float('inf')
            
            for zone_name, zone_times in self.KILLZONES.items():
                # Check if in killzone
                in_zone = self.timeinrange(
                    current_time,
                    zone_times['start'],
                    zone_times['end']
                )
                
                if in_zone:
                    zone_data = {
                        'name': zone_name,
                        'start': zone_times['start'].strftime('%H:%M'),
                        'end': zone_times['end'].strftime('%H:%M'),
                        'volatility': self._calculate_zone_volatility(df, zone_times)
                    }
                    killzones['active_zones'].append(zone_data)
                    killzones['volatility'][zone_name] = zone_data['volatility']
            else:
                    # Check if next upcoming zone
                    time_to_zone = self._calculate_time_to_zone(
                        current_time,
                        zone_times['start']
                    )
                    if 0 < time_to_zone < min_time_to_next:
                        min_time_to_next = time_to_zone
                        killzones['next_zone'] = zone_name
                        killzones['time_to_next'] = time_to_zone
            
            return killzones
            
        except Exception as e:
            logger.error(f"Error detecting killzones: {str(e)}")
            return {
                'active_zones': [],
                'next_zone': None,
                'time_to_next': None,
                'volatility': {}
            }

    def _calculate_zone_volatility(
        self,
        df: pd.DataFrame,
        zone_times: Dict[str, time]
    ) -> float:
        """
        Calculate volatility during a specific trading zone.
        
        Args:
            df: OHLC price data
            zone_times: Dictionary with zone start and end times
            
        Returns:
            float: Volatility score for the zone
        """
        try:
            # Filter data for zone times
            zone_data = df[
                (df.index.time >= zone_times['start']) & 
                (df.index.time <= zone_times['end'])
            ]
            
            if len(zone_data) > 0:
                # Calculate normalized ATR
                atr = self.calculate_atr(zone_data)
                avg_price = zone_data['close'].mean()
                return float(atr.mean() / avg_price * 100)  # As percentage
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating zone volatility: {str(e)}")
            return 0.0

    def _calculate_time_to_zone(
        self,
        current_time: time,
        zone_start: time
    ) -> float:
        """
        Calculate minutes until the start of a trading zone.
        
        Args:
            current_time: Current time
            zone_start: Zone start time
            
        Returns:
            float: Minutes until zone starts, accounting for day rollover
        """
        try:
            current_minutes = current_time.hour * 60 + current_time.minute
            zone_minutes = zone_start.hour * 60 + zone_start.minute
            
            if zone_minutes > current_minutes:
                return float(zone_minutes - current_minutes)
            else:
                return float((24 * 60) - (current_minutes - zone_minutes))
                
        except Exception as e:
            logger.error(f"Error calculating time to zone: {str(e)}")
            return float('inf')

    def detect_displacement(self, df: pd.DataFrame) -> Dict[str, bool]:
        """
        Detect price displacement from moving averages.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            Dict indicating displacement states
        """
        try:
            # Calculate moving averages
            ma20 = df['close'].rolling(20).mean()
            ma50 = df['close'].rolling(50).mean()
            
            current_price = df['close'].iloc[-1]
            current_ma20 = ma20.iloc[-1]
            current_ma50 = ma50.iloc[-1]
            
            # Calculate displacement percentages
            ma20_displacement = (current_price - current_ma20) / current_ma20 * 100
            ma50_displacement = (current_price - current_ma50) / current_ma50 * 100
            
            return {
                'ma20_displaced': abs(ma20_displacement) > 1.0,  # More than 1% away
                'ma50_displaced': abs(ma50_displacement) > 2.0,  # More than 2% away
                'ma20_displacement': ma20_displacement,
                'ma50_displacement': ma50_displacement
            }
            
        except Exception as e:
            logger.error(f"Error detecting displacement: {str(e)}")
            return {
                'ma20_displaced': False,
                'ma50_displaced': False,
                'ma20_displacement': 0.0,
                'ma50_displacement': 0.0
            }
    
    def timeinrange(self, current_time: time, start_time: time, end_time: time) -> bool:
        """
        Check if current time is within a given range.
        
        Args:
            current_time: Time to check
            start_time: Start of range
            end_time: End of range
            
        Returns:
            bool: True if time is in range
        """
        try:
            if start_time <= end_time:
                return start_time <= current_time <= end_time
            else:  # Handle ranges that cross midnight
                return current_time >= start_time or current_time <= end_time
            
        except Exception as e:
            logger.error(f"Error checking time range: {str(e)}")
            return False
        
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
        """Assess the quality of detected market structure with improved criteria."""
        try:
            quality_score = 0.5  # Start with neutral score
            reasons = []
            
            # 1. Check swing point quantity and distribution (25% weight)
            min_points = max(self.min_swing_points * 2, 4)
            if len(swing_highs) >= min_points and len(swing_lows) >= min_points:
                swing_score = min(1.0, (len(swing_highs) + len(swing_lows)) / (min_points * 2))
                quality_score += swing_score * 0.25
                reasons.append(f"Found {len(swing_highs)} highs and {len(swing_lows)} lows")
            else:
                quality_score -= 0.25
                reasons.append("Insufficient swing points")
            
            # 2. Check trend consistency (25% weight)
            if len(df) >= 20:
                ma20 = df['close'].rolling(window=20).mean()
                ma50 = df['close'].rolling(window=50).mean()
                current_price = df['close'].iloc[-1]
                
                # Calculate trend direction consistency
                trend_changes = sum(1 for i in range(1, len(ma20)) 
                                  if (ma20.iloc[i] > ma50.iloc[i]) != (ma20.iloc[i-1] > ma50.iloc[i-1]))
                trend_score = max(0, 1 - (trend_changes / 10))  # Penalize frequent trend changes
                
                # Add trend strength component
                trend_strength = abs(current_price - ma20.iloc[-1]) / ma20.iloc[-1]
                trend_score = min(1.0, trend_score + trend_strength)
                
                quality_score += trend_score * 0.25
                reasons.append(f"Trend consistency score: {trend_score:.2f}")
            
            # 3. Check swing point magnitudes (25% weight)
            if swing_highs and swing_lows:
                # Calculate average swing size
                high_sizes = [abs(h['price'] - min(l['price'] for l in swing_lows if l['index'] < h['index'])) 
                            for h in swing_highs if any(l['index'] < h['index'] for l in swing_lows)]
                low_sizes = [abs(max(h['price'] for h in swing_highs if h['index'] < l['index']) - l['price']) 
                           for l in swing_lows if any(h['index'] < l['index'] for h in swing_highs)]
                
                if high_sizes and low_sizes:
                    avg_size = sum(high_sizes + low_sizes) / len(high_sizes + low_sizes)
                    size_score = min(1.0, avg_size / (df['atr'].mean() * 2))
                    quality_score += size_score * 0.25
                    reasons.append(f"Swing magnitude score: {size_score:.2f}")
            
            # 4. Check volume confirmation (25% weight)
            if 'volume' in df.columns:
                recent_volume = df['volume'].tail(20).mean()
                overall_volume = df['volume'].mean()
                volume_ratio = recent_volume / overall_volume
                
                volume_score = min(1.0, volume_ratio)
                quality_score += volume_score * 0.25
                reasons.append(f"Volume confirmation score: {volume_score:.2f}")
            
            # Additional context-based adjustments
            # Penalize if price is in a very tight range
            price_range = (df['high'].tail(20).max() - df['low'].tail(20).min()) / df['close'].iloc[-1]
            if price_range < 0.001:  # Less than 0.1% range
                quality_score *= 0.8
                reasons.append("Price range too tight")
            
            # Bonus for clear structure type
            structure_type = self._determine_structure_type(
                {'highs': swing_highs, 'lows': swing_lows}, 
                {'bullish': [], 'bearish': []}
            )
            if structure_type in ['Uptrend', 'Downtrend', 'Distribution', 'Accumulation']:
                quality_score *= 1.2
                reasons.append(f"Clear {structure_type} structure")
            
            # Ensure score is between 0 and 1
            quality_score = max(0.0, min(1.0, quality_score))
            
            logger.debug("Structure Quality Assessment:")
            logger.debug(f"Base Quality Score: {quality_score:.2f}")
            for reason in reasons:
                logger.debug(f"- {reason}")
            
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

    def _determine_structure_type(self, swing_points: Dict, structure_breaks: Dict) -> Dict:
        """Determine market structure type based on swing points and recent breaks."""
        try:
            if not swing_points or 'highs' not in swing_points or 'lows' not in swing_points:
                logger.warning("Insufficient swing points data.")
                return {
                    "structure_type": "Sideways",
                    "higher_highs": False,
                    "higher_lows": False,
                    "lower_highs": False,
                    "lower_lows": False,
                    "bullish_breaks": 0,
                    "bearish_breaks": 0,
                    "price_volatility": 0,
                    "movement_threshold": 0
                }

            highs = swing_points.get('highs') or []
            lows = swing_points.get('lows') or []

            if len(highs) < 2 or len(lows) < 2:
                return {
                    "structure_type": "Forming",
                    "higher_highs": False,
                    "higher_lows": False,
                    "lower_highs": False,
                    "lower_lows": False,
                    "bullish_breaks": 0,
                    "bearish_breaks": 0,
                    "price_volatility": 0,
                    "movement_threshold": 0
                }
                
            # Safely sort swing points using .get() to avoid KeyError
            last_highs = sorted(highs, key=lambda x: x.get('index', 0))
            last_lows = sorted(lows, key=lambda x: x.get('index', 0))
            
            # Take only the last 3 points
            last_highs = last_highs[-3:] if len(last_highs) >= 3 else last_highs
            last_lows = last_lows[-3:] if len(last_lows) >= 3 else last_lows
            
            if len(last_highs) < 2 or len(last_lows) < 2:
                return {
                    "structure_type": "Forming",
                    "higher_highs": False,
                    "higher_lows": False,
                    "lower_highs": False,
                    "lower_lows": False,
                    "bullish_breaks": 0,
                    "bearish_breaks": 0,
                    "price_volatility": 0,
                    "movement_threshold": 0
                }
            
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
                
            # Sort by time instead of index, or fall back to sorting by price if time not available
            recent_breaks = sorted(all_breaks, key=lambda x: x.get('time', pd.Timestamp.min))[-3:] if all_breaks else []
            
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
            return {
                "structure_type": structure_type,
                "higher_highs": higher_highs,
                "higher_lows": higher_lows,
                "lower_highs": lower_highs,
                "lower_lows": lower_lows,
                "bullish_breaks": bullish_count,
                "bearish_breaks": bearish_count,
                "price_volatility": price_volatility,
                "movement_threshold": movement_threshold
            }
                
        except Exception as e:
            logger.error(f"Error determining structure type: {str(e)}")
            logger.exception("Detailed error trace:")
            return {
                "structure_type": "Ranging",
                "higher_highs": False,
                "higher_lows": False,
                "lower_highs": False,
                "lower_lows": False,
                "bullish_breaks": 0,
                "bearish_breaks": 0,
                "price_volatility": 0,
                "movement_threshold": 0
            }

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

    def detect_bos(self, df: pd.DataFrame, swing_points: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Detect Break of Structure (BOS) with improved validation and error handling.
        
        Args:
            df (pd.DataFrame): Price data
            swing_points (Dict): Dictionary containing swing highs and lows
            
        Returns:
            Dict: Dictionary containing bullish and bearish BOS points
        """
        try:
            if not swing_points or not swing_points.get('highs') or not swing_points.get('lows'):
                logger.warning("Insufficient swing points for BOS detection")
                return {'bullish': [], 'bearish': []}
                
            # Sort swing points by index
            highs = sorted(swing_points['highs'], key=lambda x: x['index'])
            lows = sorted(swing_points['lows'], key=lambda x: x['index'])
            
            if len(highs) < 2 or len(lows) < 2:
                logger.warning("Need at least 2 swing highs and 2 swing lows for BOS detection")
                return {'bullish': [], 'bearish': []}
            
            bullish_bos = []
            bearish_bos = []
            
            # Calculate ATR for dynamic threshold if not present
            if 'atr' not in df.columns:
                df['atr'] = self.calculate_atr(df)
            avg_atr = df['atr'].mean()
            bos_threshold = avg_atr * 0.2  # Reduced threshold for more sensitive detection
            
            # Detect Bullish BOS
            for i in range(2, len(highs)):
                try:
                    current_high = highs[i]
                    prev_high = highs[i-1]
                    prev_prev_high = highs[i-2]
                    
                    # Find relevant lows between these highs
                    relevant_lows = [l for l in lows if prev_prev_high['index'] < l['index'] < current_high['index']]
                    if not relevant_lows:
                        continue
                        
                    lowest_low = min(relevant_lows, key=lambda x: x['price'])
                    
                    # Check for bullish BOS pattern:
                    # 1. Previous structure was making lower highs
                    # 2. Current high breaks above the previous high
                    if (prev_high['price'] < prev_prev_high['price'] and  # Lower high
                        current_high['price'] > prev_prev_high['price']):  # Breaks above previous structure
                        
                        # Calculate strength based on the break size
                        break_size = current_high['price'] - prev_prev_high['price']
                        strength = break_size / bos_threshold
                        
                        bullish_bos.append({
                            'index': current_high['index'],
                            'price': float(current_high['price']),
                            'timestamp': current_high['timestamp'],
                            'strength': float(strength),
                            'prev_high': float(prev_prev_high['price']),
                            'break_level': float(prev_high['price']),
                            'low_point': float(lowest_low['price'])
                        })
                        logger.debug(f"Bullish BOS detected at index {current_high['index']}, "
                                   f"price: {current_high['price']:.5f}, strength: {strength:.2f}")
                
                except Exception as e:
                    logger.warning(f"Error processing bullish BOS at index {i}: {str(e)}")
                    continue
            
            # Detect Bearish BOS
            for i in range(2, len(lows)):
                try:
                    current_low = lows[i]
                    prev_low = lows[i-1]
                    prev_prev_low = lows[i-2]
                    
                    # Find relevant highs between these lows
                    relevant_highs = [h for h in highs if prev_prev_low['index'] < h['index'] < current_low['index']]
                    if not relevant_highs:
                        continue
                        
                    highest_high = max(relevant_highs, key=lambda x: x['price'])
                    
                    # Check for bearish BOS pattern:
                    # 1. Previous structure was making higher lows
                    # 2. Current low breaks below the previous low
                    if (prev_low['price'] > prev_prev_low['price'] and  # Higher low
                        current_low['price'] < prev_prev_low['price']):  # Breaks below previous structure
                        
                        # Calculate strength based on the break size
                        break_size = prev_prev_low['price'] - current_low['price']
                        strength = break_size / bos_threshold
                        
                        bearish_bos.append({
                            'index': current_low['index'],
                            'price': float(current_low['price']),
                            'timestamp': current_low['timestamp'],
                            'strength': float(strength),
                            'prev_low': float(prev_prev_low['price']),
                            'break_level': float(prev_low['price']),
                            'high_point': float(highest_high['price'])
                        })
                        logger.debug(f"Bearish BOS detected at index {current_low['index']}, "
                                   f"price: {current_low['price']:.5f}, strength: {strength:.2f}")
                
                except Exception as e:
                    logger.warning(f"Error processing bearish BOS at index {i}: {str(e)}")
                    continue
            
            logger.info(f"Detected {len(bullish_bos)} bullish and {len(bearish_bos)} bearish BOS points")
            if bullish_bos:
                logger.debug(f"Latest bullish BOS: price={bullish_bos[-1]['price']:.5f}, "
                           f"strength={bullish_bos[-1]['strength']:.2f}")
            if bearish_bos:
                logger.debug(f"Latest bearish BOS: price={bearish_bos[-1]['price']:.5f}, "
                           f"strength={bearish_bos[-1]['strength']:.2f}")
            
                return {
                'bullish': bullish_bos,
                'bearish': bearish_bos
            }
            
        except Exception as e:
            logger.error(f"Error in BOS detection: {str(e)}")
            return {'bullish': [], 'bearish': []}

    def detect_bos_and_choch(self, df: pd.DataFrame, swing_points: Dict[str, List[Dict[str, Any]]], confirmation_type: str = 'Candle Close') -> Dict[str, List[Dict[str, Any]]]:
        """
        Detect Break of Structure (BOS) and Change of Character (CHoCH) points.

        Args:
            df: OHLC price data
            swing_points: Dictionary containing swing highs and lows
            confirmation_type: Type of confirmation ('Candle Close' or 'Price Action')

        Returns:
            Dictionary containing BOS and CHoCH points
        """
        try:
            bos_points = {'bullish': [], 'bearish': []}
            choch_points = {'bullish': [], 'bearish': []}
            
            recent_swings = self._get_recent_swings(swing_points)
            
            # Detect BOS points
            for i in range(len(df) - 1):
                # BOS detection logic
                if self._is_bullish_bos(df, i, recent_swings, confirmation_type):
                    bos_points['bullish'].append({
                        'index': i,
                        'price': df['low'].iloc[i],
                        'type': 'BOS'
                    })
                elif self._is_bearish_bos(df, i, recent_swings, confirmation_type):
                    bos_points['bearish'].append({
                        'index': i,
                        'price': df['high'].iloc[i],
                        'type': 'BOS'
                    })
            
            # Detect CHoCH points based on BOS points
            if confirmation_type == 'Candle Close':
                choch_points = self._detect_choch_points(df, bos_points)
            
            return {
                'bos': bos_points,
                'choch': choch_points
            }
            
        except Exception as e:
            logger.error(f"Error detecting BOS and CHoCH: {str(e)}")
            return {'bos': {'bullish': [], 'bearish': []}, 'choch': {'bullish': [], 'bearish': []}}

    def detect_structure_breaks(
        self,
        df: pd.DataFrame,
        swing_points: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Detect structure breaks in the market.
        
        Args:
            df: DataFrame with OHLC data
            swing_points: Dict containing swing highs and lows
            
        Returns:
            Dict containing bullish and bearish structure breaks
        """
        try:
            breaks = {
                'bullish': [],
                'bearish': []
            }
            
            # Get recent swings
            recent_swings = self._get_recent_swings(swing_points)
            highs = recent_swings['highs']
            lows = recent_swings['lows']
            
            # Need at least 2 swings to detect breaks
            if len(highs) < 2 or len(lows) < 2:
                return breaks
            
            # Check for bullish breaks (breaking above previous structure)
            for i in range(1, len(highs)):
                if highs[i]['price'] > highs[i-1]['price']:
                    breaks['bullish'].append({
                        'price': highs[i]['price'],
                        'time': highs[i]['time'],
                        'index': highs[i].get('index', i),
                        'prev_price': highs[i-1]['price'],
                        'prev_time': highs[i-1]['time']
                    })
            
            # Check for bearish breaks (breaking below previous structure)
            for i in range(1, len(lows)):
                if lows[i]['price'] < lows[i-1]['price']:
                    breaks['bearish'].append({
                        'price': lows[i]['price'],
                        'time': lows[i]['time'],
                        'index': lows[i].get('index', i),
                        'prev_price': lows[i-1]['price'],
                        'prev_time': lows[i-1]['time']
                    })
            
            return breaks
            
        except Exception as e:
            logger.error(f"Error detecting structure breaks: {str(e)}")
            return {'bullish': [], 'bearish': []}

    def detect_market_shift(
        self,
        df: pd.DataFrame,
        swing_points: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Detect market structure shifts (combines bullish and bearish MSS detection).
        
        Args:
            df: DataFrame with OHLC data
            swing_points: Dict containing swing highs and lows
            
        Returns:
            Dict containing bullish and bearish market structure shifts
        """
        try:
            shifts = {
                    'bullish': [],
                'bearish': []
            }
            
            # Get recent swings
            recent_swings = self._get_recent_swings(swing_points)
            highs = recent_swings['highs']
            lows = recent_swings['lows']
            
            # Need at least 3 swings to detect shifts
            if len(highs) < 3 or len(lows) < 3:
                return shifts
            
            # Detect bullish shifts (higher lows forming)
            for i in range(2, len(lows)):
                if (lows[i]['price'] > lows[i-1]['price'] and 
                    lows[i-1]['price'] > lows[i-2]['price']):
                    shifts['bullish'].append({
                        'price': lows[i]['price'],
                        'time': lows[i]['time'],
                        'prev_price': lows[i-1]['price'],
                        'prev_time': lows[i-1]['time']
                    })
            
            # Detect bearish shifts (lower highs forming)
            for i in range(2, len(highs)):
                if (highs[i]['price'] < highs[i-1]['price'] and 
                    highs[i-1]['price'] < highs[i-2]['price']):
                    shifts['bearish'].append({
                        'price': highs[i]['price'],
                        'time': highs[i]['time'],
                        'prev_price': highs[i-1]['price'],
                        'prev_time': highs[i-1]['time']
                    })
            
            return shifts
            
        except Exception as e:
            logger.error(f"Error detecting market shifts: {str(e)}")
            return {'bullish': [], 'bearish': []}
    
    def _get_recent_swings(
        self,
        swing_points: Dict[str, List[Dict[str, Any]]],
        lookback: int = 5
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get the most recent swing points.
        
        Args:
            swing_points: Dict containing all swing points
            lookback: Number of recent swings to return
            
        Returns:
            Dict containing recent swing highs and lows
        """
        try:
            recent_highs = swing_points['highs'][-lookback:] if swing_points['highs'] else []
            recent_lows = swing_points['lows'][-lookback:] if swing_points['lows'] else []
            
            return {
                'highs': recent_highs,
                'lows': recent_lows
            }
            
        except Exception as e:
            logger.error(f"Error getting recent swings: {str(e)}")
            return {'highs': [], 'lows': []}
    
    def _assess_structure_quality(
        self,
        df: pd.DataFrame,
        swing_highs: List[Dict[str, Any]],
        swing_lows: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Assess the quality of market structure.
        
        Args:
            df: DataFrame with OHLC data
            swing_highs: List of swing high points
            swing_lows: List of swing low points
            
        Returns:
            Dict containing structure quality metrics
        """
        try:
            # Need minimum number of swings
            if len(swing_highs) < self.MIN_SWING_SEQUENCE or len(swing_lows) < self.MIN_SWING_SEQUENCE:
                return {
                        'quality': 0.0,
                        'trend_strength': 0.0,
                        'swing_regularity': 0.0
                    }
            
            # Calculate trend strength
            trend_strength = self._calculate_trend_strength(swing_highs, swing_lows)
            
            # Calculate swing regularity
            swing_regularity = self._calculate_swing_regularity(swing_highs, swing_lows)
            
            # Overall quality score
            quality = (trend_strength + swing_regularity) / 2
            
            return {
                'quality': quality,
                'trend_strength': trend_strength,
                'swing_regularity': swing_regularity
            }
            
        except Exception as e:
            logger.error(f"Error assessing structure quality: {str(e)}")
            return {
                'quality': 0.0,
                'trend_strength': 0.0,
                'swing_regularity': 0.0
            }
    
    def _calculate_trend_strength(
        self,
        swing_highs: List[Dict[str, Any]],
        swing_lows: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate trend strength based on swing point alignment.
        
        Args:
            swing_highs: List of swing high points
            swing_lows: List of swing low points
            
        Returns:
            float: Trend strength score (0.0 to 1.0)
        """
        try:
            # Check higher highs and higher lows for uptrend
            higher_highs = all(swing_highs[i]['price'] > swing_highs[i-1]['price'] 
                             for i in range(1, len(swing_highs)))
            higher_lows = all(swing_lows[i]['price'] > swing_lows[i-1]['price'] 
                            for i in range(1, len(swing_lows)))
            
            # Check lower highs and lower lows for downtrend
            lower_highs = all(swing_highs[i]['price'] < swing_highs[i-1]['price'] 
                            for i in range(1, len(swing_highs)))
            lower_lows = all(swing_lows[i]['price'] < swing_lows[i-1]['price'] 
                           for i in range(1, len(swing_lows)))
            
            # Calculate strength score
            if (higher_highs and higher_lows) or (lower_highs and lower_lows):
                return 1.0
            elif higher_highs or higher_lows or lower_highs or lower_lows:
                return 0.5
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating trend strength: {str(e)}")
            return 0.0
    
    def _calculate_swing_regularity(
        self,
        swing_highs: List[Dict[str, Any]],
        swing_lows: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate regularity of swing point formation.
        
        Args:
            swing_highs: List of swing high points
            swing_lows: List of swing low points
            
        Returns:
            float: Swing regularity score (0.0 to 1.0)
        """
        try:
            # Calculate time between swings
            high_intervals = [swing_highs[i]['time'] - swing_highs[i-1]['time'] 
                            for i in range(1, len(swing_highs))]
            low_intervals = [swing_lows[i]['time'] - swing_lows[i-1]['time'] 
                           for i in range(1, len(swing_lows))]
            
            # Convert Timedelta objects to seconds (float values)
            high_intervals_seconds = [interval.total_seconds() for interval in high_intervals]
            low_intervals_seconds = [interval.total_seconds() for interval in low_intervals]
            
            # Calculate standard deviation of intervals
            if high_intervals_seconds and low_intervals_seconds:
                high_std = np.std(high_intervals_seconds)
                low_std = np.std(low_intervals_seconds)
                avg_std = (high_std + low_std) / 2
                
                # Convert to regularity score (lower std = higher regularity)
                # Use a scaling factor to normalize the result
                scaling_factor = 3600.0  # 1 hour in seconds
                regularity = 1.0 / (1.0 + (avg_std / scaling_factor))
                return min(regularity, 1.0)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating swing regularity: {str(e)}")
            return 0.0

    def detect_mss_bullish(self, df: pd.DataFrame, swing_points: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Detect bullish Market Structure Shift (MSS) patterns.
        
        Args:
            df (pd.DataFrame): Price data
            swing_points (Dict): Dictionary containing swing highs and lows
            
        Returns:
            List[Dict[str, Any]]: List of detected bullish MSS points
        """
        try:
            if not swing_points or not swing_points.get('highs') or not swing_points.get('lows'):
                logger.warning("Insufficient swing points for bullish MSS detection")
                return []
            
            # Sort swing points by index
            highs = sorted(swing_points['highs'], key=lambda x: x['index'])
            lows = sorted(swing_points['lows'], key=lambda x: x['index'])
            
            if len(highs) < 3 or len(lows) < 3:
                logger.warning("Need at least 3 swing highs and lows for MSS detection")
                return []
            
            mss_points = []
            
            # Calculate ATR for dynamic threshold if not present
            if 'atr' not in df.columns:
                df['atr'] = self.calculate_atr(df)
            avg_atr = df['atr'].mean()
            mss_threshold = avg_atr * 0.3  # 30% of ATR for MSS confirmation
            
            # Look for MSS pattern in recent swing points
            for i in range(2, len(highs)):
                try:
                    current_high = highs[i]
                    prev_high = highs[i-1]
                    prev_prev_high = highs[i-2]
                    
                    # Find relevant lows between these highs
                    relevant_lows = [l for l in lows if prev_prev_high['index'] < l['index'] < current_high['index']]
                    if not relevant_lows:
                        continue
                    
                    lowest_low = min(relevant_lows, key=lambda x: x['price'])
                    
                    # Get previous lows for comparison
                    previous_lows = [l for l in lows if l['index'] < lowest_low['index']]
                    if not previous_lows:
                        # No previous lows to compare with, can't confirm higher low
                        continue
                    
                    # Check for bullish MSS pattern:
                    # 1. Previous structure was making lower highs and lower lows
                    # 2. Current high breaks above the previous high significantly
                    # 3. Recent low is higher than the previous low
                    previous_low_prices = [l['price'] for l in previous_lows]
                    if not previous_low_prices:
                        continue
                        
                    if (prev_high['price'] < prev_prev_high['price'] and  # Lower high
                        current_high['price'] > prev_prev_high['price'] and  # Breaks above previous structure
                        lowest_low['price'] > min(previous_low_prices)):  # Higher low
                        
                        # Calculate strength based on the break size
                        break_size = current_high['price'] - prev_prev_high['price']
                        if break_size >= mss_threshold:
                            strength = break_size / mss_threshold
                            
                            mss_points.append({
                                'index': current_high['index'],
                                'price': float(current_high['price']),
                                'timestamp': current_high['timestamp'],
                                'strength': float(strength),
                                'prev_high': float(prev_prev_high['price']),
                                'low_point': float(lowest_low['price']),
                                'confirmation': True if df['close'].iloc[current_high['index']] > prev_prev_high['price'] else False
                            })
                            logger.debug(f"Bullish MSS detected at index {current_high['index']}, "
                                       f"price: {current_high['price']:.5f}, strength: {strength:.2f}")
                
                except Exception as e:
                    logger.warning(f"Error processing bullish MSS at index {i}: {str(e)}")
                    continue
            
            logger.info(f"Detected {len(mss_points)} bullish MSS points")
            if mss_points:
                logger.debug(f"Latest bullish MSS: price={mss_points[-1]['price']:.5f}, "
                           f"strength={mss_points[-1]['strength']:.2f}")
            
            return mss_points
            
        except Exception as e:
            logger.error(f"Error in bullish MSS detection: {str(e)}")
            return []

    def detect_mss_bearish(self, df: pd.DataFrame, swing_points: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Detect bearish Market Structure Shift (MSS) patterns.
        
        Args:
            df (pd.DataFrame): Price data
            swing_points (Dict): Dictionary containing swing highs and lows
            
        Returns:
            List[Dict[str, Any]]: List of detected bearish MSS points
        """
        try:
            if not swing_points or not swing_points.get('highs') or not swing_points.get('lows'):
                logger.warning("Insufficient swing points for bearish MSS detection")
                return []
            
            # Sort swing points by index
            highs = sorted(swing_points['highs'], key=lambda x: x['index'])
            lows = sorted(swing_points['lows'], key=lambda x: x['index'])
            
            if len(highs) < 3 or len(lows) < 3:
                logger.warning("Need at least 3 swing highs and lows for MSS detection")
                return []
            
            mss_points = []
            
            # Calculate ATR for dynamic threshold if not present
            if 'atr' not in df.columns:
                df['atr'] = self.calculate_atr(df)
            avg_atr = df['atr'].mean()
            mss_threshold = avg_atr * 0.3  # 30% of ATR for MSS confirmation
            
            # Look for MSS pattern in recent swing points
            for i in range(2, len(lows)):
                try:
                    current_low = lows[i]
                    prev_low = lows[i-1]
                    prev_prev_low = lows[i-2]
                    
                    # Find relevant highs between these lows
                    relevant_highs = [h for h in highs if prev_prev_low['index'] < h['index'] < current_low['index']]
                    if not relevant_highs:
                        continue
                    
                    highest_high = max(relevant_highs, key=lambda x: x['price'])
                    
                    # Get previous highs for comparison
                    previous_highs = [h for h in highs if h['index'] < highest_high['index']]
                    if not previous_highs:
                        # No previous highs to compare with, can't confirm lower high
                        continue
                    
                    # Check for bearish MSS pattern:
                    # 1. Previous structure was making higher highs and higher lows
                    # 2. Current low breaks below the previous low significantly
                    # 3. Recent high is lower than the previous high
                    previous_high_prices = [h['price'] for h in previous_highs]
                    if not previous_high_prices:
                        continue
                        
                    if (prev_low['price'] > prev_prev_low['price'] and  # Higher low
                        current_low['price'] < prev_prev_low['price'] and  # Breaks below previous structure
                        highest_high['price'] < max(previous_high_prices)):  # Lower high
                        
                        # Calculate strength based on the break size
                        break_size = prev_prev_low['price'] - current_low['price']
                        if break_size >= mss_threshold:
                            strength = break_size / mss_threshold
                            
                            mss_points.append({
                                'index': current_low['index'],
                                'price': float(current_low['price']),
                                'timestamp': current_low['timestamp'],
                                'strength': float(strength),
                                'prev_low': float(prev_prev_low['price']),
                                'high_point': float(highest_high['price']),
                                'confirmation': True if df['close'].iloc[current_low['index']] < prev_prev_low['price'] else False
                            })
                            logger.debug(f"Bearish MSS detected at index {current_low['index']}, "
                                       f"price: {current_low['price']:.5f}, strength: {strength:.2f}")
                
                except Exception as e:
                    logger.warning(f"Error processing bearish MSS at index {i}: {str(e)}")
                    continue
            
            logger.info(f"Detected {len(mss_points)} bearish MSS points")
            if mss_points:
                logger.debug(f"Latest bearish MSS: price={mss_points[-1]['price']:.5f}, "
                           f"strength={mss_points[-1]['strength']:.2f}")
            
            return mss_points
            
        except Exception as e:
            logger.error(f"Error in bearish MSS detection: {str(e)}")
            return []

    def detect_killzones(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect active killzones based on current time.
        Returns a dictionary indicating which killzones are currently active.
        This function is currently deactivated and will always return False for all killzones.
        """
        try:
            # Killzones are deactivated
            logger.info("Killzones detection is currently deactivated")
            
            # Return all killzones as inactive
            return {
                'london_open': False,
                'london_close': False,
                'new_york': False,
                'asian': False
            }
            
        except Exception as e:
            logger.error(f"Error in deactivated killzones function: {str(e)}")
            return {name: False for name in ['london_open', 'london_close', 'new_york', 'asian']}

    def timeinrange(self, current_time: time, start_time: time, end_time: time) -> bool:
        """Check if current time is within a given range, handling midnight crossing correctly."""
        # Normal range where start time is before end time
        if start_time <= end_time:
            return start_time <= current_time <= end_time
        
        # Range crosses midnight
        else:
            return current_time >= start_time or current_time <= end_time
    
    def analyze_trend(self, df: pd.DataFrame) -> str:
        """Analyze trend using moving averages and multiple confirmations.
        
        Returns:
            str: 'bullish', 'bearish', or 'neutral'
        """
        try:
            # Calculate moving averages
            df['MA20'] = df['close'].rolling(window=20).mean()
            df['MA50'] = df['close'].rolling(window=50).mean()
            df['MA200'] = df['close'].rolling(window=200).mean()
            
            # Get current values
            current_close = df['close'].iloc[-1]
            current_ma20 = df['MA20'].iloc[-1]
            current_ma50 = df['MA50'].iloc[-1]
            current_ma200 = df['MA200'].iloc[-1]
            
            # Calculate short-term momentum (last 5 candles)
            short_term_change = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5] * 100
            
            # Calculate medium-term momentum (last 20 candles)
            medium_term_change = (df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20] * 100
            
            # Calculate price position relative to MAs
            above_ma20 = current_close > current_ma20
            above_ma50 = current_close > current_ma50
            above_ma200 = current_close > current_ma200
            
            # Calculate MA alignments
            bullish_alignment = current_ma20 > current_ma50 and current_ma50 > current_ma200
            bearish_alignment = current_ma20 < current_ma50 and current_ma50 < current_ma200
            
            # Define trend thresholds
            MOMENTUM_THRESHOLD = 0.1  # 0.1% change
            
            # Determine trend with multiple confirmations
            bullish_conditions = [
                above_ma20,
                above_ma50,
                short_term_change > MOMENTUM_THRESHOLD,
                medium_term_change > 0,
                bullish_alignment
            ]
            
            bearish_conditions = [
                not above_ma20,
                not above_ma50,
                short_term_change < -MOMENTUM_THRESHOLD,
                medium_term_change < 0,
                bearish_alignment
            ]
            
            # Count confirmations
            bullish_count = sum(bullish_conditions)
            bearish_count = sum(bearish_conditions)
            
            # Require at least 3 confirmations for a trend
            if bullish_count >= 3:
                return 'bullish'
            elif bearish_count >= 3:
                return 'bearish'
            else:
                # Check if price is showing strong momentum in either direction
                if abs(short_term_change) > MOMENTUM_THRESHOLD * 2:
                    return 'bullish' if short_term_change > 0 else 'bearish'
                return 'neutral'
                
        except Exception as e:
            logger.error(f"Error analyzing trend: {str(e)}")
            return 'neutral'
        
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def detect_displacement(self, df: pd.DataFrame) -> Dict[str, bool]:
        displacement = {'bullish': False, 'bearish': False}
        
        if len(df) >= 2:
            current_candle = df.iloc[-1]
            prev_candle = df.iloc[-2]
            
            body_size = abs(current_candle['close'] - current_candle['open'])
            upper_wick = current_candle['high'] - max(current_candle['close'], current_candle['open'])
            lower_wick = min(current_candle['close'], current_candle['open']) - current_candle['low']
            
            if body_size > 0:
                wick_body_ratio = (upper_wick + lower_wick) / body_size
                
                if wick_body_ratio < 0.1:
                    if current_candle['close'] > prev_candle['close']:
                        displacement['bullish'] = True
                    elif current_candle['close'] < prev_candle['close']:
                        displacement['bearish'] = True
        
        return displacement
    
    def analyze_session(self) -> str:
        """Analyze the current trading session based on current UTC time.
        
        Returns:
            str: One of 'asian', 'london', 'new_york', 'overlap', 'evening' or 'no_session'
                indicating the current active trading session.
        """
        try:
            # Get current UTC time
            current_time = datetime.now(UTC)
            current_hour = current_time.hour
            
            logger.debug(f"Current UTC time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}, Hour: {current_hour}")
            
            # Define session hours in UTC
            asian_session = range(0, 8)      # 00:00-08:00 UTC
            london_session = range(8, 16)    # 08:00-16:00 UTC  
            ny_session = range(13, 21)       # 13:00-21:00 UTC
            evening_session = range(21, 24)  # 21:00-00:00 UTC (NY evening/Asian pre-session)
            
            # Check for session overlaps first
            if current_hour in range(13, 16):  # London-NY overlap
                return "overlap"
                
            # Check individual sessions
            if current_hour in asian_session:
                return "asian"
            elif current_hour in london_session:
                return "london"
            elif current_hour in ny_session:
                return "new_york"
            elif current_hour in evening_session:
                return "evening"
                
            # Outside of main sessions
            logger.debug(f"Current hour {current_hour} is outside all defined sessions")
            return "evening"  # Instead of no_session, consider evening hours as part of trading
            
        except Exception as e:
            logger.error(f"Error analyzing session: {str(e)}")
            return "no_session"
    
    def calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate technical indicators for momentum analysis with enhanced type safety."""
        try:
            indicators = {}
            
            # Calculate RSI with NaN handling
            delta = df['close'].diff().ffill()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean().fillna(0)
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean().fillna(0)
            rs = np.where(loss != 0, gain / loss, 1)  # Avoid division by zero
            indicators['rsi'] = float(100 - (100 / (1 + rs[-1])))
            
            # Calculate MACD with explicit type conversion
            exp1 = df['close'].ewm(span=12, adjust=False).mean().astype('float64')
            exp2 = df['close'].ewm(span=26, adjust=False).mean().astype('float64')
            macd_line = (exp1 - exp2).astype('float64')
            signal_line = macd_line.ewm(span=9, adjust=False).mean().astype('float64')
            
            indicators['macd'] = {
                'macd_line': float(macd_line.iloc[-1]),
                'signal_line': float(signal_line.iloc[-1])
            }
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            return {'rsi': 50, 'macd': {'macd_line': 0.0, 'signal_line': 0.0}}
        

    def is_market_open(self, symbol: str = None) -> bool:
        """
        Check if the market is currently open based on schedule and holidays.
        Returns True if market is open, False otherwise.
        
        Forex market is open 24/5 - from Sunday evening to Friday evening local time,
        except for holidays.
        
        Cryptocurrency markets are open 24/7, so we'll bypass day/time restrictions for crypto symbols.
        
        Args:
            symbol: Optional symbol to check. If provided, used to determine if it's a crypto symbol.
        """
        try:
            # Get current local time
            local_time = datetime.now()
            current_date = local_time.date()
            current_weekday = current_date.weekday()  # 0 = Monday, 6 = Sunday
            
            logger.info(f"Market open check for {symbol}: Local time: {local_time.strftime('%Y-%m-%d %H:%M:%S')}, Weekday: {current_weekday}")
            
            # Check if it's a cryptocurrency symbol
            is_crypto = False
            if symbol is not None:
                # True cryptocurrency symbols typically contain BTC, ETH, etc.
                crypto_identifiers = ["BTC", "ETH", "XBT", "LTC", "DOT", "SOL", "ADA", "DOGE", "CRYPTO"]
                is_crypto = any(identifier in symbol for identifier in crypto_identifiers)
                
                # Check for explicit cryptocurrency pairs that might not have the above identifiers
                crypto_pairs = ["BTCUSD", "ETHUSD"]
                if any(pair in symbol for pair in crypto_pairs):
                    is_crypto = True
            
            if is_crypto:
                logger.debug(f"Cryptocurrency symbol {symbol} detected - market is always open")
                return True
            
            # Check if it's a forex or metal symbol (if not crypto)
            # Default to forex if we can't determine
            market_type = "forex"
            logger.debug(f"Symbol {symbol} identified as {market_type}")
            
            # Check if it's a holiday
            current_year = str(current_date.year)
            if current_year in self.market_schedule["holidays"]:
                holiday_dates = [
                    datetime.strptime(date, "%Y-%m-%d").date()
                    for date in self.market_schedule["holidays"][current_year].values()
                ]
                if current_date in holiday_dates:
                    logger.info(f"Market is closed for holiday on {current_date}")
                    return False
            
            # Check if it's a partial trading day
            if current_year in self.market_schedule["partial_trading_days"]:
                for day_info in self.market_schedule["partial_trading_days"][current_year].values():
                    if datetime.strptime(day_info["date"], "%Y-%m-%d").date() == current_date:
                        close_time = datetime.strptime(day_info["close_time"], "%H:%M").time()
                        if local_time.time() >= close_time:
                            logger.info(f"Market is closed for partial trading day at {close_time} local time")
                            return False
            
            # Market is closed on Saturday for Forex
            if current_weekday == 5:  # Saturday
                logger.info(f"Forex market is closed for {symbol} (Saturday)")
                return False
            
            # Market is closed on Sunday until 11 PM local time for Forex
            if current_weekday == 6:  # Sunday
                market_open = local_time.replace(hour=23, minute=0, second=0, microsecond=0)
                if local_time < market_open:
                    logger.info(f"Forex market is closed for {symbol} (Sunday before 11 PM: {local_time} < {market_open})")
                    return False
            
            # Market is closed Friday after 11 PM local time for Forex
            if current_weekday == 4:  # Friday
                market_close = local_time.replace(hour=23, minute=0, second=0, microsecond=0)
                if local_time >= market_close:
                    logger.info(f"Forex market is closed for {symbol} (Friday after 11 PM: {local_time} >= {market_close})")
                    return False
            
            # If we got here, market is open
            logger.info(f"Market is open for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error checking market schedule for {symbol}: {str(e)}")
            # If there's an error checking the schedule, assume market is closed for safety
            return False

    def _is_bullish_bos(
        self,
        df: pd.DataFrame,
        index: int,
        recent_swings: Dict[str, List[Dict[str, Any]]],
        confirmation_type: str
    ) -> bool:
        """
        Check if a bullish Break of Structure (BOS) occurs at the given index.
        
        Args:
            df: OHLC price data
            index: Candle index to check
            recent_swings: Dictionary of recent swing highs and lows
            confirmation_type: Type of confirmation ('Candle Close' or 'Price Action')
            
        Returns:
            bool: True if bullish BOS is detected, False otherwise
        """
        try:
            if index < 2 or index >= len(df):
                return False
                
            # Get recent swing lows for reference
            swing_lows = recent_swings['lows']
            if not swing_lows:
                return False
                
            # Get the most recent swing low
            recent_low = swing_lows[-1]
            
            # Check if price breaks above the recent structure
            if confirmation_type == 'Candle Close':
                # Confirm with candle close
                return (df['close'].iloc[index] > recent_low['price'] and
                       df['low'].iloc[index-1] <= recent_low['price'])
            else:
                # Confirm with price action (wicks)
                return (df['high'].iloc[index] > recent_low['price'] and
                       df['low'].iloc[index-1] <= recent_low['price'])
                       
        except Exception as e:
            logger.error(f"Error checking bullish BOS: {str(e)}")
            return False
            
    def _is_bearish_bos(
        self,
        df: pd.DataFrame,
        index: int,
        recent_swings: Dict[str, List[Dict[str, Any]]],
        confirmation_type: str
    ) -> bool:
        """
        Check if a bearish Break of Structure (BOS) occurs at the given index.
        
        Args:
            df: OHLC price data
            index: Candle index to check
            recent_swings: Dictionary of recent swing highs and lows
            confirmation_type: Type of confirmation ('Candle Close' or 'Price Action')
            
        Returns:
            bool: True if bearish BOS is detected, False otherwise
        """
        try:
            if index < 2 or index >= len(df):
                return False
                
            # Get recent swing highs for reference
            swing_highs = recent_swings['highs']
            if not swing_highs:
                return False
                
            # Get the most recent swing high
            recent_high = swing_highs[-1]
            
            # Check if price breaks below the recent structure
            if confirmation_type == 'Candle Close':
                # Confirm with candle close
                return (df['close'].iloc[index] < recent_high['price'] and
                       df['high'].iloc[index-1] >= recent_high['price'])
            else:
                # Confirm with price action (wicks)
                return (df['low'].iloc[index] < recent_high['price'] and
                       df['high'].iloc[index-1] >= recent_high['price'])
                       
        except Exception as e:
            logger.error(f"Error checking bearish BOS: {str(e)}")
            return False
            
    def _detect_choch_points(
        self,
        df: pd.DataFrame,
        bos_points: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Detect Change of Character (CHoCH) points based on BOS points.
        
        Args:
            df: OHLC price data
            bos_points: Dictionary containing bullish and bearish BOS points
            
        Returns:
            Dictionary containing bullish and bearish CHoCH points
        """
        try:
            choch_points = {'bullish': [], 'bearish': []}
            
            # Combine all BOS points and sort by index
            all_bos = []
            for point in bos_points['bullish']:
                all_bos.append(('bullish', point))
            for point in bos_points['bearish']:
                all_bos.append(('bearish', point))
                
            all_bos.sort(key=lambda x: x[1]['index'])
            
            # Find CHoCH points (first counter-trend BOS after a trend)
            prev_type = None
            for bos_type, point in all_bos:
                if prev_type and bos_type != prev_type:
                    # Counter-trend BOS found - mark as CHoCH
                    choch_point = point.copy()
                    choch_point['type'] = 'CHoCH'
                    choch_points[bos_type].append(choch_point)
                prev_type = bos_type
                
            return choch_points
            
        except Exception as e:
            logger.error(f"Error detecting CHoCH points: {str(e)}")
            return {'bullish': [], 'bearish': []}
                
    async def get_market_data(
        self,
        symbol: str,
        timeframe: str
    ) -> Optional[pd.DataFrame]:
        """
        Get market data for the specified symbol and timeframe using MT5Handler.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe to fetch data for
            
        Returns:
            DataFrame with OHLCV data or None if error
        """
        try:
            # Use MT5Handler to get market data
            df = await self.mt5_handler.get_rates(symbol, timeframe)
            if df is None:
                return None
                
            # Calculate basic indicators
            df['atr'] = self.calculate_atr(df)
            df['rsi'] = self.calculate_rsi(df['close'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting market data: {str(e)}")
            return None

    def detect_market_structure(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Detect and analyze market structure components.
        
        Args:
            df: DataFrame with OHLCV data containing:
                - time (index)
                - open, high, low, close
                - volume
                
        Returns:
            Dict containing structure analysis:
            {
                'swing_points': Dict with highs and lows,
                'structure_breaks': Dict with bullish/bearish breaks,
                'order_blocks': Dict with bullish/bearish blocks,
                'fair_value_gaps': Dict with bullish/bearish gaps,
                'structure_type': Dict with type and confidence,
                'quality': Dict with quality metrics
            }
        """
        try:
            # Get swing points
            swing_points = self.detect_swing_points(df)
            if not swing_points or not (swing_points.get('highs') and swing_points.get('lows')):
                logger.warning("No valid swing points detected")
                return self._get_default_structure_result()
            
            # Detect structure breaks
            structure_breaks = self.detect_structure_breaks(df, swing_points)
            
            # Detect order blocks
            order_blocks = self.detect_order_blocks(
                df, 
                (swing_points['highs'], swing_points['lows'])
            )
            
            # Detect fair value gaps
            fair_value_gaps = self.detect_fair_value_gaps(df)
            
            # Determine structure type
            structure_type = self._determine_structure_type(swing_points, structure_breaks)
            
            # Calculate quality metrics
            quality = self._assess_structure_quality(
                df,
                swing_points['highs'],
                swing_points['lows']
            )
            
            return {
                'swing_points': swing_points,
                'structure_breaks': structure_breaks,
                'order_blocks': order_blocks,
                'fair_value_gaps': fair_value_gaps,
                'structure_type': structure_type,
                'quality': quality
            }
            
        except Exception as e:
            logger.error(f"Error detecting market structure: {str(e)}")
            return {
                'swing_points': {'highs': [], 'lows': []},
                'structure_breaks': {'bullish': [], 'bearish': []},
                'order_blocks': {'bullish': [], 'bearish': []},
                'fair_value_gaps': {'bullish': [], 'bearish': []},
                'structure_type': {'type': 'Unknown', 'confidence': 0.0},
                'quality': {
                    'score': 0.0,
                    'trend_strength': 0.0,
                    'swing_regularity': 0.0,
                    'volume_confirmation': 0.0
                }
            }
            
    def _get_default_structure_result(self) -> Dict[str, Any]:
        """Get default empty structure analysis result."""
        return {
            'swing_points': {'highs': [], 'lows': []},
            'structure_breaks': {'bullish': [], 'bearish': []},
            'order_blocks': {'bullish': [], 'bearish': []},
            'fair_value_gaps': {'bullish': [], 'bearish': []},
            'structure_type': {
                'type': 'Unknown',
                'confidence': 0.0
            },
            'quality': {
                'score': 0.0,
                'trend_strength': 0.0,
                'swing_regularity': 0.0,
                'volume_confirmation': 0.0
            }
        }
