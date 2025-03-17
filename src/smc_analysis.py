from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from loguru import logger

class SMCAnalysis:
    def __init__(self, use_symbol_specific=True):
        # Base thresholds
        self.equal_level_threshold = 0.0001  # 1 pip for equal high/low detection
        self.liquidity_threshold = 0.0020    # 20 pips for liquidity pool size
        self.manipulation_threshold = 0.0015  # 15 pips for manipulation moves
        self.ob_threshold = 0.0015           # 15 pips for order block size
        
        # Store whether to use symbol-specific adjustments
        self.use_symbol_specific = use_symbol_specific
        
    def adjust_thresholds_for_symbol(self, symbol: str):
        """Adjust thresholds based on the symbol being analyzed."""
        if not self.use_symbol_specific:
            return
            
        # Default multiplier
        multiplier = 1.0
        
        # Adjust for JPY pairs
        if 'JPY' in symbol:
            multiplier = 100.0  # JPY pairs have different pip values
            
        # Adjust for precious metals
        elif 'XAU' in symbol:  # Gold
            multiplier = 2.0
        elif 'XAG' in symbol:  # Silver
            multiplier = 2.5
            
        # Adjust for crypto
        elif any(crypto in symbol for crypto in ['BTC', 'ETH', 'XRP']):
            multiplier = 3.0
            
        # Adjust for indices
        elif any(index in symbol for index in ['NAS', 'SPX', 'US30']):
            multiplier = 1.8
            
        # Apply the multiplier to all thresholds
        self.equal_level_threshold *= multiplier
        self.liquidity_threshold *= multiplier
        self.manipulation_threshold *= multiplier
        self.ob_threshold *= multiplier
        
    def analyze(self, df: pd.DataFrame, symbol: str = "") -> Dict:
        """Analyze Smart Money Concepts patterns and structures."""
        try:
            # Adjust thresholds for the symbol
            self.adjust_thresholds_for_symbol(symbol)
            
            # Detect liquidity sweeps
            sweeps = self._detect_liquidity_sweeps(df, symbol)
            
            # Detect manipulation points
            manipulation = self._detect_manipulation(df, symbol)
            
            # Find breaker blocks
            breakers = self._find_breaker_blocks(df, symbol)
            
            # Find mitigation blocks
            mitigation = self._find_mitigation_blocks(df, symbol)
            
            # Detect premium/discount zones
            zones = self._detect_premium_discount_zones(df, symbol)
            
            # Find inefficient price moves
            inefficient_moves = self._find_inefficient_moves(df, symbol)
            
            # Analyze institutional order flow
            order_flow = self._analyze_order_flow(df, symbol)
            
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
    
    def _detect_liquidity_sweeps(self, df: pd.DataFrame, symbol: str) -> List[Dict]:
        """Detect liquidity sweeps (stop hunts)."""
        try:
            sweeps = []
            
            # Make equal level threshold more lenient
            equal_level_threshold = self.equal_level_threshold * 1.5
            liquidity_threshold = self.liquidity_threshold * 0.7  # Reduce the threshold by 30%
            
            for i in range(4, len(df)):
                # Find near equal highs/lows (more lenient)
                equal_highs = abs(df['high'].iloc[i-1] - df['high'].iloc[i-2]) < equal_level_threshold
                equal_lows = abs(df['low'].iloc[i-1] - df['low'].iloc[i-2]) < equal_level_threshold
                
                # Also check for three-bar patterns (triple tops/bottoms)
                triple_top = (abs(df['high'].iloc[i-1] - df['high'].iloc[i-3]) < equal_level_threshold and
                             abs(df['high'].iloc[i-2] - df['high'].iloc[i-3]) < equal_level_threshold)
                triple_bottom = (abs(df['low'].iloc[i-1] - df['low'].iloc[i-3]) < equal_level_threshold and
                               abs(df['low'].iloc[i-2] - df['low'].iloc[i-3]) < equal_level_threshold)
                
                # Bullish sweep (sweep lows then move up)
                if (equal_lows or triple_bottom) and df['low'].iloc[i] < df['low'].iloc[i-1]:
                    # More lenient on the close condition - only need partial recovery
                    if df['close'].iloc[i] > (df['low'].iloc[i] + (df['open'].iloc[i] - df['low'].iloc[i]) * 0.3):
                        sweep_size = df['close'].iloc[i] - df['low'].iloc[i]
                        if sweep_size >= liquidity_threshold:
                            # Categorize as SSL/BSL for signal generator
                            sweep_level = df['low'].iloc[i-1]
                            
                            sweeps.append({
                                'index': i,
                                'type': 'bullish',
                                'entry': df['low'].iloc[i],
                                'exit': df['close'].iloc[i],
                                'size': sweep_size,
                                'timestamp': df.index[i],
                                'level': sweep_level,
                                'strength': min(1.0, sweep_size / (liquidity_threshold * 2)),
                                'label': 'SSL'  # Support Sweep Level
                            })
                
                # Bearish sweep (sweep highs then move down)
                if (equal_highs or triple_top) and df['high'].iloc[i] > df['high'].iloc[i-1]:
                    # More lenient on the close condition - only need partial recovery
                    if df['close'].iloc[i] < (df['high'].iloc[i] - (df['high'].iloc[i] - df['open'].iloc[i]) * 0.3):
                        sweep_size = df['high'].iloc[i] - df['close'].iloc[i]
                        if sweep_size >= liquidity_threshold:
                            # Categorize as BSL/SSL for signal generator
                            sweep_level = df['high'].iloc[i-1]
                            
                            sweeps.append({
                                'index': i,
                                'type': 'bearish',
                                'entry': df['high'].iloc[i],
                                'exit': df['close'].iloc[i],
                                'size': sweep_size,
                                'timestamp': df.index[i],
                                'level': sweep_level,
                                'strength': min(1.0, sweep_size / (liquidity_threshold * 2)),
                                'label': 'BSL'  # Resistance Sweep Level
                            })
            
            # Process the detected sweeps to assign BSL/SSL labels for signal generator use
            processed_sweeps = []
            for sweep in sweeps:
                if sweep['type'] == 'bearish':
                    processed_sweeps.append({
                        'type': 'BSL',  # Bearish Sweep Level (for Resistance)
                        'level': sweep['level'],
                        'strength': sweep['strength'],
                        'timestamp': sweep['timestamp']
                    })
                else:
                    processed_sweeps.append({
                        'type': 'SSL',  # Support Sweep Level
                        'level': sweep['level'],
                        'strength': sweep['strength'],
                        'timestamp': sweep['timestamp']
                    })
            
            return processed_sweeps
            
        except Exception as e:
            logger.error(f"Error detecting liquidity sweeps: {str(e)}")
            return []
    
    def _detect_manipulation(self, df: pd.DataFrame, symbol: str) -> List[Dict]:
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
    
    def _find_breaker_blocks(self, df: pd.DataFrame, symbol: str) -> Dict[str, List[Dict]]:
        """Find breaker blocks (strong reversal points)."""
        try:
            bullish_breakers = []
            bearish_breakers = []
            
            for i in range(4, len(df)):
                # Bullish breaker (strong rejection of lows)
                # More flexible condition: Allow partial rejections too
                if df['low'].iloc[i] < df['low'].iloc[i-1]:
                    # Check if price closed significantly higher
                    price_recovery = df['close'].iloc[i] - df['low'].iloc[i]
                    price_range = df['high'].iloc[i] - df['low'].iloc[i]
                    
                    # Consider it a breaker if price recovered at least 60% of the range
                    # or if it closed above the previous high
                    if (price_range > 0 and price_recovery / price_range >= 0.6) or \
                       df['close'].iloc[i] > df['high'].iloc[i-1]:
                        block_size = df['close'].iloc[i] - df['low'].iloc[i]
                        if block_size >= self.manipulation_threshold:
                            bullish_breakers.append({
                                'index': i,
                                'high': df['close'].iloc[i],
                                'low': df['low'].iloc[i],
                                'size': block_size,
                                'strength': min(1.0, price_recovery / (price_range * 1.5)) if price_range > 0 else 0.5,
                                'timestamp': df.index[i]
                            })
                
                # Bearish breaker (strong rejection of highs)
                # More flexible condition: Allow partial rejections too
                if df['high'].iloc[i] > df['high'].iloc[i-1]:
                    # Check if price closed significantly lower
                    price_rejection = df['high'].iloc[i] - df['close'].iloc[i]
                    price_range = df['high'].iloc[i] - df['low'].iloc[i]
                    
                    # Consider it a breaker if price rejected at least 60% of the range
                    # or if it closed below the previous low
                    if (price_range > 0 and price_rejection / price_range >= 0.6) or \
                       df['close'].iloc[i] < df['low'].iloc[i-1]:
                        block_size = df['high'].iloc[i] - df['close'].iloc[i]
                        if block_size >= self.manipulation_threshold:
                            bearish_breakers.append({
                                'index': i,
                                'high': df['high'].iloc[i],
                                'low': df['close'].iloc[i],
                                'size': block_size,
                                'strength': min(1.0, price_rejection / (price_range * 1.5)) if price_range > 0 else 0.5,
                                'timestamp': df.index[i]
                            })
            
            return {
                'bullish': bullish_breakers,
                'bearish': bearish_breakers
            }
            
        except Exception as e:
            logger.error(f"Error finding breaker blocks: {str(e)}")
            return {'bullish': [], 'bearish': []}
    
    def _find_mitigation_blocks(self, df: pd.DataFrame, symbol: str) -> Dict[str, List[Dict]]:
        """Find mitigation blocks (areas of unmitigated price)."""
        try:
            bullish_mitigation = []
            bearish_mitigation = []
            
            for i in range(4, len(df)):
                # Bullish mitigation (price returns to fill gap)
                if df['low'].iloc[i] < df['high'].iloc[i-2] and df['low'].iloc[i-1] > df['high'].iloc[i-2]:
                    block_size = df['high'].iloc[i-2] - df['low'].iloc[i]
                    if block_size >= self.manipulation_threshold:
                        bullish_mitigation.append({
                            'index': i,
                            'high': df['high'].iloc[i-2],
                            'low': df['low'].iloc[i],
                            'size': block_size,
                            'timestamp': df.index[i]
                        })
                
                # Bearish mitigation (price returns to fill gap)
                if df['high'].iloc[i] > df['low'].iloc[i-2] and df['high'].iloc[i-1] < df['low'].iloc[i-2]:
                    block_size = df['high'].iloc[i] - df['low'].iloc[i-2]
                    if block_size >= self.manipulation_threshold:
                        bearish_mitigation.append({
                            'index': i,
                            'high': df['high'].iloc[i],
                            'low': df['low'].iloc[i-2],
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
    
    def _detect_premium_discount_zones(self, df: pd.DataFrame, symbol: str) -> Dict[str, List[Dict]]:
        """Detect premium and discount zones."""
        try:
            premium_zones = []
            discount_zones = []
            
            # Calculate proper ATR using True Range formula
            true_ranges = []
            for i in range(1, min(14, len(df))):
                high = df['high'].iloc[-i]
                low = df['low'].iloc[-i]
                prev_close = df['close'].iloc[-(i+1)] if i < len(df)-1 else df['open'].iloc[-i]
                true_range = max(high-low, abs(high-prev_close), abs(low-prev_close))
                true_ranges.append(true_range)
            
            avg_atr = np.mean(true_ranges) if true_ranges else (df['high'].max() - df['low'].min()) / len(df)
            
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
    
    def _find_inefficient_moves(self, df: pd.DataFrame, symbol: str) -> List[Dict]:
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
    
    def _analyze_order_flow(self, df: pd.DataFrame, symbol: str) -> Dict:
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
                relative_volume = volume / df['volume'].iloc[max(0, i-20):i].mean() if 'volume' in df and i > 0 else 1.0
                
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

    def check_liquidity(self, analysis: Dict) -> bool:
        """Check for liquidity conditions based on market analysis.
        
        Args:
            analysis (Dict): Market analysis data containing OHLCV and other metrics
            
        Returns:
            bool: True if liquidity conditions are favorable, False otherwise
        """
        try:
            # Get recent liquidity sweeps
            sweeps = analysis.get('smc', {}).get('liquidity_sweeps', [])
            if not sweeps:
                logger.debug("No liquidity sweeps found in analysis")
                return False
                
            # Get the most recent sweep
            recent_sweep = sweeps[-1]
            sweep_type = recent_sweep.get('type', '').lower()
            
            # Get current market trend
            trend = analysis.get('trend', '').lower()
            
            # If no trend is provided, use the sweep type as a signal
            if not trend:
                logger.debug(f"No trend provided, using sweep type '{sweep_type}' as signal")
                return True
                
            # Check if sweep type aligns with trend
            if trend == 'bullish' and sweep_type == 'ssl':  # Support Sweep Level for bullish
                logger.debug(f"Bullish trend aligns with SSL sweep")
                return True
            elif trend == 'bearish' and sweep_type == 'bsl':  # Bearish Sweep Level for bearish
                logger.debug(f"Bearish trend aligns with BSL sweep")
                return True
            else:
                logger.debug(f"Sweep type '{sweep_type}' does not align with trend '{trend}'")
                return False
            
        except Exception as e:
            logger.error(f"Error checking liquidity: {str(e)}")
            return False 