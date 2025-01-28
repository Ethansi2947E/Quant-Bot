from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from loguru import logger

class SMCAnalysis:
    def __init__(self):
        self.equal_level_threshold = 0.0001  # 1 pip for equal high/low detection
        self.liquidity_threshold = 0.0020    # 20 pips for liquidity pool size
        self.manipulation_threshold = 0.0015  # 15 pips for manipulation moves
        self.ob_threshold = 0.0015           # 15 pips for order block size
        
    def analyze(self, df: pd.DataFrame) -> Dict:
        """Analyze Smart Money Concepts patterns and structures."""
        try:
            # Detect liquidity sweeps
            sweeps = self._detect_liquidity_sweeps(df)
            
            # Detect manipulation points
            manipulation = self._detect_manipulation(df)
            
            # Find breaker blocks
            breakers = self._find_breaker_blocks(df)
            
            # Find mitigation blocks
            mitigation = self._find_mitigation_blocks(df)
            
            # Detect premium/discount zones
            zones = self._detect_premium_discount_zones(df)
            
            # Find inefficient price moves
            inefficient_moves = self._find_inefficient_moves(df)
            
            # Analyze institutional order flow
            order_flow = self._analyze_order_flow(df)
            
            return {
                'liquidity_sweeps': sweeps,
                'manipulation_points': manipulation,
                'breaker_blocks': breakers,
                'mitigation_blocks': mitigation,
                'premium_discount_zones': zones,
                'inefficient_moves': inefficient_moves,
                'order_flow': order_flow
            }
            
        except Exception as e:
            logger.error(f"Error in SMC analysis: {str(e)}")
            return {
                'liquidity_sweeps': [],
                'manipulation_points': [],
                'breaker_blocks': {'bullish': [], 'bearish': []},
                'mitigation_blocks': {'bullish': [], 'bearish': []},
                'premium_discount_zones': {'premium': [], 'discount': []},
                'inefficient_moves': [],
                'order_flow': {'bias': 'neutral', 'strength': 0}
            }
    
    def _detect_liquidity_sweeps(self, df: pd.DataFrame) -> List[Dict]:
        """Detect liquidity sweeps (stop hunts)."""
        try:
            sweeps = []
            
            for i in range(4, len(df)):
                # Find equal highs/lows
                equal_highs = abs(df['high'].iloc[i-1] - df['high'].iloc[i-2]) < self.equal_level_threshold
                equal_lows = abs(df['low'].iloc[i-1] - df['low'].iloc[i-2]) < self.equal_level_threshold
                
                # Bullish sweep (sweep lows then move up)
                if equal_lows and df['low'].iloc[i] < df['low'].iloc[i-1] and \
                   df['close'].iloc[i] > df['open'].iloc[i]:
                    sweep_size = df['close'].iloc[i] - df['low'].iloc[i]
                    if sweep_size >= self.liquidity_threshold:
                        sweeps.append({
                            'index': i,
                            'type': 'bullish',
                            'entry': df['low'].iloc[i],
                            'exit': df['close'].iloc[i],
                            'size': sweep_size,
                            'timestamp': df.index[i]
                        })
                
                # Bearish sweep (sweep highs then move down)
                if equal_highs and df['high'].iloc[i] > df['high'].iloc[i-1] and \
                   df['close'].iloc[i] < df['open'].iloc[i]:
                    sweep_size = df['high'].iloc[i] - df['close'].iloc[i]
                    if sweep_size >= self.liquidity_threshold:
                        sweeps.append({
                            'index': i,
                            'type': 'bearish',
                            'entry': df['high'].iloc[i],
                            'exit': df['close'].iloc[i],
                            'size': sweep_size,
                            'timestamp': df.index[i]
                        })
            
            return sweeps
            
        except Exception as e:
            logger.error(f"Error detecting liquidity sweeps: {str(e)}")
            return []
    
    def _detect_manipulation(self, df: pd.DataFrame) -> List[Dict]:
        """Detect manipulation points (stop runs, liquidity grabs)."""
        try:
            manipulation = []
            
            for i in range(4, len(df)):
                # Calculate move sizes
                up_move = df['high'].iloc[i] - df['low'].iloc[i-1]
                down_move = df['high'].iloc[i-1] - df['low'].iloc[i]
                
                # Bullish manipulation (fake down move then strong up)
                if down_move > self.manipulation_threshold and \
                   df['close'].iloc[i] > df['high'].iloc[i-1]:
                    manipulation.append({
                        'index': i,
                        'type': 'bullish',
                        'entry': df['low'].iloc[i],
                        'exit': df['close'].iloc[i],
                        'size': down_move,
                        'timestamp': df.index[i]
                    })
                
                # Bearish manipulation (fake up move then strong down)
                if up_move > self.manipulation_threshold and \
                   df['close'].iloc[i] < df['low'].iloc[i-1]:
                    manipulation.append({
                        'index': i,
                        'type': 'bearish',
                        'entry': df['high'].iloc[i],
                        'exit': df['close'].iloc[i],
                        'size': up_move,
                        'timestamp': df.index[i]
                    })
            
            return manipulation
            
        except Exception as e:
            logger.error(f"Error detecting manipulation: {str(e)}")
            return []
    
    def _find_breaker_blocks(self, df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """Find breaker blocks (strong reversal points)."""
        try:
            bullish_breakers = []
            bearish_breakers = []
            
            for i in range(4, len(df)):
                # Bullish breaker (strong rejection of lows)
                if df['low'].iloc[i] < df['low'].iloc[i-1] and \
                   df['close'].iloc[i] > df['high'].iloc[i-1]:
                    block_size = df['close'].iloc[i] - df['low'].iloc[i]
                    if block_size >= self.manipulation_threshold:
                        bullish_breakers.append({
                            'index': i,
                            'high': df['close'].iloc[i],
                            'low': df['low'].iloc[i],
                            'size': block_size,
                            'timestamp': df.index[i]
                        })
                
                # Bearish breaker (strong rejection of highs)
                if df['high'].iloc[i] > df['high'].iloc[i-1] and \
                   df['close'].iloc[i] < df['low'].iloc[i-1]:
                    block_size = df['high'].iloc[i] - df['close'].iloc[i]
                    if block_size >= self.manipulation_threshold:
                        bearish_breakers.append({
                            'index': i,
                            'high': df['high'].iloc[i],
                            'low': df['close'].iloc[i],
                            'size': block_size,
                            'timestamp': df.index[i]
                        })
            
            return {
                'bullish': bullish_breakers,
                'bearish': bearish_breakers
            }
            
        except Exception as e:
            logger.error(f"Error finding breaker blocks: {str(e)}")
            return {'bullish': [], 'bearish': []}
    
    def _find_mitigation_blocks(self, df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """Find mitigation blocks (areas of unmitigated price)."""
        try:
            bullish_mitigation = []
            bearish_mitigation = []
            
            for i in range(4, len(df)):
                # Bullish mitigation (price returns to fill gap)
                if df['low'].iloc[i] > df['high'].iloc[i-2]:
                    block_size = df['low'].iloc[i] - df['high'].iloc[i-2]
                    if block_size >= self.manipulation_threshold:
                        bullish_mitigation.append({
                            'index': i,
                            'high': df['low'].iloc[i],
                            'low': df['high'].iloc[i-2],
                            'size': block_size,
                            'timestamp': df.index[i]
                        })
                
                # Bearish mitigation (price returns to fill gap)
                if df['high'].iloc[i] < df['low'].iloc[i-2]:
                    block_size = df['low'].iloc[i-2] - df['high'].iloc[i]
                    if block_size >= self.manipulation_threshold:
                        bearish_mitigation.append({
                            'index': i,
                            'high': df['low'].iloc[i-2],
                            'low': df['high'].iloc[i],
                            'size': block_size,
                            'timestamp': df.index[i]
                        })
            
            return {
                'bullish': bullish_mitigation,
                'bearish': bearish_mitigation
            }
            
        except Exception as e:
            logger.error(f"Error finding mitigation blocks: {str(e)}")
            return {'bullish': [], 'bearish': []}
    
    def _detect_premium_discount_zones(self, df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """Detect premium and discount zones."""
        try:
            premium_zones = []
            discount_zones = []
            
            # Calculate average true range
            atr = df['high'].rolling(14).max() - df['low'].rolling(14).min()
            avg_atr = atr.mean()
            
            for i in range(20, len(df)):
                # Calculate local high/low
                local_high = df['high'].iloc[i-20:i].max()
                local_low = df['low'].iloc[i-20:i].min()
                
                # Premium zone (price far above average)
                if df['close'].iloc[i] > local_high + avg_atr:
                    premium_zones.append({
                        'index': i,
                        'top': df['high'].iloc[i],
                        'bottom': local_high,
                        'size': df['high'].iloc[i] - local_high,
                        'timestamp': df.index[i]
                    })
                
                # Discount zone (price far below average)
                if df['close'].iloc[i] < local_low - avg_atr:
                    discount_zones.append({
                        'index': i,
                        'top': local_low,
                        'bottom': df['low'].iloc[i],
                        'size': local_low - df['low'].iloc[i],
                        'timestamp': df.index[i]
                    })
            
            return {
                'premium': premium_zones,
                'discount': discount_zones
            }
            
        except Exception as e:
            logger.error(f"Error detecting premium/discount zones: {str(e)}")
            return {'premium': [], 'discount': []}
    
    def _find_inefficient_moves(self, df: pd.DataFrame) -> List[Dict]:
        """Find inefficient price movements (gaps, strong moves)."""
        try:
            inefficient_moves = []
            
            for i in range(2, len(df)):
                # Calculate move efficiency
                price_change = abs(df['close'].iloc[i] - df['close'].iloc[i-1])
                high_low_range = df['high'].iloc[i] - df['low'].iloc[i]
                
                if high_low_range > 0:
                    efficiency = price_change / high_low_range
                    
                    # Highly efficient move (price moves strongly in one direction)
                    if efficiency > 0.8 and price_change >= self.manipulation_threshold:
                        inefficient_moves.append({
                            'index': i,
                            'start_price': df['close'].iloc[i-1],
                            'end_price': df['close'].iloc[i],
                            'efficiency': efficiency,
                            'size': price_change,
                            'timestamp': df.index[i]
                        })
            
            return inefficient_moves
            
        except Exception as e:
            logger.error(f"Error finding inefficient moves: {str(e)}")
            return []
    
    def _analyze_order_flow(self, df: pd.DataFrame) -> Dict:
        """Analyze institutional order flow based on volume and price action."""
        try:
            bullish_pressure = 0
            bearish_pressure = 0
            
            for i in range(1, len(df)):
                # Calculate candle metrics
                body_size = abs(df['close'].iloc[i] - df['open'].iloc[i])
                upper_wick = df['high'].iloc[i] - max(df['open'].iloc[i], df['close'].iloc[i])
                lower_wick = min(df['open'].iloc[i], df['close'].iloc[i]) - df['low'].iloc[i]
                
                # Analyze volume
                volume = df['volume'].iloc[i] if 'volume' in df else 1.0
                relative_volume = volume / df['volume'].iloc[i-20:i].mean() if 'volume' in df else 1.0
                
                # Bullish pressure
                if df['close'].iloc[i] > df['open'].iloc[i]:
                    score = body_size * relative_volume
                    if lower_wick < body_size * 0.3:  # Strong bullish candle
                        score *= 1.5
                    bullish_pressure += score
                
                # Bearish pressure
                else:
                    score = body_size * relative_volume
                    if upper_wick < body_size * 0.3:  # Strong bearish candle
                        score *= 1.5
                    bearish_pressure += score
            
            return {
                'bullish': bullish_pressure,
                'bearish': bearish_pressure,
                'bias': 'bullish' if bullish_pressure > bearish_pressure else 'bearish'
            }
            
        except Exception as e:
            logger.error(f"Error analyzing order flow: {str(e)}")
            return {'bullish': 0, 'bearish': 0, 'bias': 'neutral'} 