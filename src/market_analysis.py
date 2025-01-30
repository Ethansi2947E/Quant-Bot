import pandas as pd
import numpy as np
from datetime import datetime, time, UTC
from typing import Dict, List, Tuple, Optional
from loguru import logger
from config.config import SESSION_CONFIG, MARKET_STRUCTURE_CONFIG

class MarketAnalysis:
    def __init__(self, ob_threshold=0.0015):
        self.session_config = SESSION_CONFIG
        self.structure_config = MARKET_STRUCTURE_CONFIG
        self.ob_threshold = ob_threshold
        self.fvg_threshold = 0.0005  # 5 pips
        self.swing_detection_lookback = 15  # Increased from 10 for more reliable swings
        self.swing_detection_threshold = 0.0008  # Increased from 0.0005
        self.min_swing_size = 0.0020  # Increased from 0.0015
        self.bos_threshold = 0.0008  # Increased from 0.0005
        self.min_swing_points = 4  # Increased from 3
        self.structure_break_threshold = 0.0020  # Increased from 0.0015
        
    def get_current_session(self) -> Tuple[str, Dict]:
        """Determine the current trading session based on UTC time."""
        current_time = datetime.now(UTC).time()
        
        for session_name, session_data in self.session_config.items():
            session_start = datetime.strptime(session_data['start'], '%H:%M').time()
            session_end = datetime.strptime(session_data['end'], '%H:%M').time()
            
            if session_start <= current_time <= session_end:
                return session_name, session_data
                
        return "no_session", {}
        
    def detect_swing_points(
        self,
        df: pd.DataFrame,
        lookback: int = None
    ) -> Tuple[List[Dict], List[Dict]]:
        """Detect swing high and low points in the price action."""
        if lookback is None:
            lookback = self.structure_config['swing_detection']['lookback_periods']
            
        swing_highs = []
        swing_lows = []
        
        for i in range(lookback, len(df) - lookback):
            # Check for swing high
            if all(df['high'].iloc[i] > df['high'].iloc[i-j] for j in range(1, lookback+1)) and \
               all(df['high'].iloc[i] > df['high'].iloc[i+j] for j in range(1, lookback+1)):
                swing_highs.append({
                    'index': i,
                    'price': df['high'].iloc[i],
                    'timestamp': df.index[i]
                })
                
            # Check for swing low
            if all(df['low'].iloc[i] < df['low'].iloc[i-j] for j in range(1, lookback+1)) and \
               all(df['low'].iloc[i] < df['low'].iloc[i+j] for j in range(1, lookback+1)):
                swing_lows.append({
                    'index': i,
                    'price': df['low'].iloc[i],
                    'timestamp': df.index[i]
                })
                
        return swing_highs, swing_lows
        
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
        swing_points: Tuple[List[Dict], List[Dict]]
    ) -> List[Dict]:
        """Detect breaks of market structure with enhanced filtering."""
        try:
            breaks = []
            swing_highs, swing_lows = swing_points
            
            # Check if we have enough swing points to analyze
            if len(swing_highs) < 2 or len(swing_lows) < 2:
                logger.debug("Not enough swing points to detect structure breaks")
                return []
            
            # Convert lists to numpy arrays for easier comparison
            high_prices = np.array([h['price'] for h in swing_highs if isinstance(h, dict) and 'price' in h])
            high_times = [h['timestamp'] for h in swing_highs if isinstance(h, dict) and 'timestamp' in h]
            low_prices = np.array([l['price'] for l in swing_lows if isinstance(l, dict) and 'price' in l])
            low_times = [l['timestamp'] for l in swing_lows if isinstance(l, dict) and 'timestamp' in l]
            
            # Ensure we still have enough valid points after filtering
            if len(high_prices) < 2 or len(low_prices) < 2:
                logger.debug("Not enough valid swing points after filtering")
                return []
            
            # Calculate average swing size for dynamic thresholds
            avg_swing_size = np.mean(np.abs(np.diff(high_prices)))
            min_break_size = avg_swing_size * 0.5  # Dynamic threshold
            
            for i in range(1, len(high_prices)):
                # Bullish break with size confirmation
                if high_prices[i] > high_prices[i-1]:
                    break_size = high_prices[i] - high_prices[i-1]
                    if break_size >= min_break_size:
                        breaks.append({
                            'type': 'bullish',
                            'price': float(high_prices[i]),
                            'time': high_times[i] if i < len(high_times) else None,
                            'strength': float(break_size / avg_swing_size)  # Normalized strength
                        })
            
            for i in range(1, len(low_prices)):
                # Bearish break with size confirmation
                if low_prices[i] < low_prices[i-1]:
                    break_size = low_prices[i-1] - low_prices[i]
                    if break_size >= min_break_size:
                        breaks.append({
                            'type': 'bearish',
                            'price': float(low_prices[i]),
                            'time': low_times[i] if i < len(low_times) else None,
                            'strength': float(break_size / avg_swing_size)  # Normalized strength
                        })
            
            return breaks
            
        except Exception as e:
            logger.error(f"Error detecting structure breaks: {str(e)}")
            return []
        
    def analyze_session_conditions(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> Dict:
        """Analyze current session-specific trading conditions."""
        session_name, session_data = self.get_current_session()
        
        if not session_data:
            return {
                'session': 'no_session',
                'suitable_for_trading': False,
                'reason': 'Outside main trading sessions'
            }
            
        # Check if symbol is preferred for current session
        if symbol not in session_data['pairs']:
            return {
                'session': session_name,
                'suitable_for_trading': False,
                'reason': f'Symbol {symbol} not preferred in {session_name}'
            }
            
        # Calculate session range
        session_high = df['high'].tail(20).max()
        session_low = df['low'].tail(20).min()
        session_range = (session_high - session_low) * 10000  # Convert to pips
        
        # Check if range is within session parameters
        if session_range < session_data['min_range_pips']:
            return {
                'session': session_name,
                'suitable_for_trading': False,
                'reason': 'Range too small for session'
            }
            
        if session_range > session_data['max_range_pips']:
            return {
                'session': session_name,
                'suitable_for_trading': False,
                'reason': 'Range too large for session'
            }
            
        # Calculate recent volatility
        atr = df['high'].tail(14).max() - df['low'].tail(14).min()
        volatility_factor = atr * session_data['volatility_factor']
        
        return {
            'session': session_name,
            'suitable_for_trading': True,
            'session_range': session_range,
            'volatility_factor': volatility_factor,
            'reason': 'Conditions suitable for trading'
        }
        
    def analyze_market_structure(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Dict:
        """Analyze market structure including swings, breaks, and order blocks."""
        try:
            # Get session conditions first
            session_name, session_data = self.get_current_session()
            session_analysis = {
                'session': session_name,
                'suitable_for_trading': True,
                'reason': 'Session conditions met'
            }

            # Initialize result dictionary
            result = {
                'market_bias': 'neutral',
                'swing_points': {'highs': [], 'lows': []},
                'structure_breaks': [],
                'order_blocks': {'bullish': [], 'bearish': []},
                'fair_value_gaps': {'bullish': [], 'bearish': []},
                'liquidity_voids': [],
                'session_analysis': session_analysis
            }

            # Identify swing points
            swing_points = self._detect_swing_points(df['high'], 5)
            result['swing_points']['highs'] = swing_points['highs']
            result['swing_points']['lows'] = swing_points['lows']

            # Detect structure breaks
            breaks = self.detect_structure_breaks(df, (result['swing_points']['highs'], result['swing_points']['lows']))
            result['structure_breaks'] = breaks

            # Identify order blocks
            obs = self._identify_order_blocks(df)
            result['order_blocks'] = obs

            # Find fair value gaps
            fvgs = self._find_fair_value_gaps(df)
            result['fair_value_gaps'] = fvgs

            # Detect liquidity voids
            voids = self._detect_liquidity_voids(df)
            result['liquidity_voids'] = voids

            # Determine market bias
            result['market_bias'] = self._determine_market_bias(df, result['swing_points']['highs'], result['swing_points']['lows'], breaks)

            return result

        except Exception as e:
            logger.error(f"Error in market structure analysis: {str(e)}")
            return self._empty_analysis()

    def _detect_swing_points(self, price: pd.Series, window: int = 5) -> Dict[str, List[Dict]]:
        """Detect swing highs and lows."""
        try:
            highs = []
            lows = []
            
            for i in range(window, len(price) - window):
                # Check for swing high
                if price.iloc[i] == max(price.iloc[i-window:i+window+1]):
                    highs.append({
                        'index': i,
                        'price': price.iloc[i],
                        'time': price.index[i]
                    })
                
                # Check for swing low
                if price.iloc[i] == min(price.iloc[i-window:i+window+1]):
                    lows.append({
                        'index': i,
                        'price': price.iloc[i],
                        'time': price.index[i]
                    })
            
            return {
                'highs': highs,
                'lows': lows
            }
            
        except Exception as e:
            logger.error(f"Error detecting swing points: {str(e)}")
            return {'highs': [], 'lows': []}

    def _identify_order_blocks(self, df: pd.DataFrame) -> List[Dict]:
        """Identify order blocks in the price action."""
        try:
            order_blocks = []
            for i in range(1, len(df) - 1):
                # Check for bullish order block
                if (df['low'].iloc[i] < df['low'].iloc[i-1] and 
                    df['high'].iloc[i+1] > df['high'].iloc[i]):
                    order_blocks.append({
                        'type': 'bullish',
                        'price': df['low'].iloc[i],
                        'index': i
                    })
                # Check for bearish order block
                elif (df['high'].iloc[i] > df['high'].iloc[i-1] and 
                      df['low'].iloc[i+1] < df['low'].iloc[i]):
                    order_blocks.append({
                        'type': 'bearish',
                        'price': df['high'].iloc[i],
                        'index': i
                    })
            return order_blocks
        except Exception as e:
            logger.error(f"Error identifying order blocks: {str(e)}")
            return []

    def _find_fair_value_gaps(self, df: pd.DataFrame) -> Dict:
        """Identify fair value gaps in price action."""
        fvgs = {'bullish': [], 'bearish': []}
        
        for i in range(1, len(df)-1):
            # Bullish FVG
            if df['low'].iloc[i+1] > df['high'].iloc[i-1]:
                fvgs['bullish'].append({
                    'top': df['low'].iloc[i+1],
                    'bottom': df['high'].iloc[i-1],
                    'timestamp': df.index[i],
                    'size': df['low'].iloc[i+1] - df['high'].iloc[i-1]
                })
            
            # Bearish FVG
            if df['high'].iloc[i+1] < df['low'].iloc[i-1]:
                fvgs['bearish'].append({
                    'top': df['low'].iloc[i-1],
                    'bottom': df['high'].iloc[i+1],
                    'timestamp': df.index[i],
                    'size': df['low'].iloc[i-1] - df['high'].iloc[i+1]
                })
        
        return fvgs

    def _detect_liquidity_voids(self, df: pd.DataFrame) -> List[Dict]:
        """Detect areas of low trading activity (liquidity voids)."""
        voids = []
        volume_threshold = df['volume'].mean() * 0.3  # 30% of average volume
        
        for i in range(1, len(df)-1):
            if df['volume'].iloc[i] < volume_threshold:
                if abs(df['high'].iloc[i] - df['low'].iloc[i]) > self.structure_break_threshold:
                    voids.append({
                        'start_price': df['low'].iloc[i],
                        'end_price': df['high'].iloc[i],
                        'timestamp': df.index[i],
                        'volume': df['volume'].iloc[i]
                    })
        
        return voids

    def _determine_market_bias(self, df: pd.DataFrame, swing_highs: List[Dict], 
                             swing_lows: List[Dict], breaks: List[Dict]) -> str:
        """Determine overall market bias using multiple factors."""
        try:
            score = 0
            
            # Check recent structure breaks (last 3)
            recent_breaks = [b for b in breaks[-3:]]
            bullish_breaks = sum(1 for b in recent_breaks if b['type'] == 'bullish')
            bearish_breaks = sum(1 for b in recent_breaks if b['type'] == 'bearish')
            score += (bullish_breaks - bearish_breaks) * 2  # Doubled the weight
            
            # Check swing point progression (last 3 points instead of 2)
            if len(swing_highs) >= 3:
                if all(swing_highs[i]['price'] > swing_highs[i-1]['price'] for i in range(-1, -3, -1)):
                    score += 2
                elif all(swing_highs[i]['price'] < swing_highs[i-1]['price'] for i in range(-1, -3, -1)):
                    score -= 2
                    
            if len(swing_lows) >= 3:
                if all(swing_lows[i]['price'] > swing_lows[i-1]['price'] for i in range(-1, -3, -1)):
                    score += 2
                elif all(swing_lows[i]['price'] < swing_lows[i-1]['price'] for i in range(-1, -3, -1)):
                    score -= 2

            # Check moving averages with more weight
            ema20 = df['close'].ewm(span=20).mean()
            ema50 = df['close'].ewm(span=50).mean()
            ema200 = df['close'].ewm(span=200).mean()  # Added longer-term trend
            
            # Price vs EMAs with trend alignment
            if df['close'].iloc[-1] > ema20.iloc[-1] > ema50.iloc[-1] > ema200.iloc[-1]:
                score += 3  # Strong bullish alignment
            elif df['close'].iloc[-1] < ema20.iloc[-1] < ema50.iloc[-1] < ema200.iloc[-1]:
                score -= 3  # Strong bearish alignment
            
            # Check for momentum using RSI
            rsi = self._calculate_rsi(df['close'], 14)
            if rsi[-1] > 70:
                score += 1
            elif rsi[-1] < 30:
                score -= 1
                
            # Volume confirmation
            volume_sma = df['volume'].rolling(20).mean()
            if df['volume'].iloc[-1] > volume_sma.iloc[-1] * 1.5:
                if df['close'].iloc[-1] > df['open'].iloc[-1]:
                    score += 1
                else:
                    score -= 1

            # Determine bias with stricter thresholds
            if score >= 4:  # Increased from 2
                return 'bullish'
            elif score <= -4:  # Increased from -2
                return 'bearish'
            else:
                return 'neutral'

        except Exception as e:
            logger.error(f"Error determining market bias: {str(e)}")
            return 'neutral'

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> np.ndarray:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.values

    def _empty_analysis(self) -> Dict:
        """Return an empty analysis result structure."""
        return {
            "market_structure": {},
            "session_conditions": {},
            "swing_points": ([], []),
            "order_blocks": {"bullish": [], "bearish": []},
            "fair_value_gaps": {"bullish": [], "bearish": []},
            "structure_breaks": []
        }

    def analyze(self, df: pd.DataFrame, symbol: str = "UNKNOWN", timeframe: str = "UNKNOWN") -> Dict:
        """Perform comprehensive market analysis."""
        if df.empty:
            return self._empty_analysis()
            
        try:
            # Get swing points
            swing_points = self._detect_swing_points(df['high'], 5)
            
            # Detect order blocks
            order_blocks = self._identify_order_blocks(df)
            
            # Detect fair value gaps
            fair_value_gaps = self.detect_fair_value_gaps(df)
            
            # Detect structure breaks
            structure_breaks = self.detect_structure_breaks(df, swing_points)
            
            # Analyze market structure
            market_structure = self.analyze_market_structure(df, symbol, timeframe)
            
            # Analyze session conditions
            session_conditions = self.analyze_session_conditions(df, symbol)
            
            return {
                "market_structure": market_structure,
                "session_conditions": session_conditions,
                "swing_points": swing_points,
                "order_blocks": order_blocks,
                "fair_value_gaps": fair_value_gaps,
                "structure_breaks": structure_breaks
            }
            
        except Exception as e:
            logger.error(f"Error in market analysis: {str(e)}")
            return self._empty_analysis() 