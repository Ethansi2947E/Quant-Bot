from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from loguru import logger

class MTFAnalysis:
    def __init__(self):
        self.timeframes = ["M5", "M15", "H1", "H4", "D1"]
        self.weights = {
            "D1": 0.30,  # Higher timeframe has more weight
            "H4": 0.25,
            "H1": 0.20,
            "M15": 0.15,
            "M5": 0.10
        }
    
    def analyze(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], timeframe: Optional[str] = None) -> Dict:
        """
        Analyze price action for single or multiple timeframes.
        
        Args:
            data: Either a single DataFrame with OHLCV data or a dictionary of DataFrames for multiple timeframes
            timeframe: Optional timeframe identifier when passing a single DataFrame
            
        Returns:
            Dict: Analysis results with consistent structure
        """
        try:
            # Handle single timeframe analysis
            if isinstance(data, pd.DataFrame):
                score = self._analyze_single_timeframe(data)
                trend = 'bullish' if score > 0 else 'bearish' if score < 0 else 'neutral'
                return {
                    'trend': trend,
                    'score': score,
                    'timeframe': timeframe or 'unknown',
                    'key_levels': self._find_key_levels(data),
                    'confidence': abs(score),
                    'analysis_type': 'single_timeframe'
                }
            
            # Handle multi-timeframe analysis
            elif isinstance(data, dict):
                return self.analyze_mtf(data, timeframe)
            
            else:
                raise ValueError("Data must be either a DataFrame or a dictionary of DataFrames")
                
        except Exception as e:
            logger.error(f"Error in analysis: {str(e)}")
            return self._get_default_analysis()
            
    def _analyze_single_timeframe(self, df: pd.DataFrame) -> float:
        """
        Analyze a single timeframe's price action.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            float: Analysis score between -1 and 1
        """
        try:
            # Calculate EMAs for different timeframes
            ema_20 = df['close'].ewm(span=20).mean()
            ema_50 = df['close'].ewm(span=50).mean()
            ema_200 = df['close'].ewm(span=200).mean()
            
            # Calculate trend alignment score
            trend_score = 0
            
            # Check EMA alignment
            if ema_20.iloc[-1] > ema_50.iloc[-1] > ema_200.iloc[-1]:
                trend_score = 1  # Strong uptrend
            elif ema_20.iloc[-1] < ema_50.iloc[-1] < ema_200.iloc[-1]:
                trend_score = -1  # Strong downtrend
            else:
                # Calculate partial trend score based on 20 and 50 EMAs
                if ema_20.iloc[-1] > ema_50.iloc[-1]:
                    trend_score = 0.5
                elif ema_20.iloc[-1] < ema_50.iloc[-1]:
                    trend_score = -0.5
                    
            return trend_score
            
        except Exception as e:
            logger.error(f"Error in single timeframe analysis: {str(e)}")
            return 0.0
    
    def analyze_mtf(self, dataframes: Dict[str, Union[pd.DataFrame, pd.Series]], timeframe: Optional[str] = None) -> Dict:
        """
        Analyze price action across multiple timeframes.
        
        Args:
            dataframes: Dictionary of DataFrames/Series for multiple timeframes
            timeframe: Optional timeframe identifier for the current analysis
            
        Returns:
            Dict: Analysis results
        """
        try:
            # Validate input is a dictionary with proper timeframe keys
            valid_timeframes = set(self.timeframes)
            available_timeframes = [tf for tf in dataframes.keys() if tf in valid_timeframes]
            
            if not available_timeframes:
                logger.warning("No valid timeframe data available for MTF analysis")
                return self._get_default_analysis()
            
            # Convert any Series to DataFrame and validate data structure
            processed_data = {}
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            
            for tf in available_timeframes:
                data = dataframes[tf]
                
                # Convert Series to DataFrame if necessary
                if isinstance(data, pd.Series):
                    df = pd.DataFrame()
                    df[data.name] = data
                    # Fill other columns with the same data
                    for col in required_columns:
                        if col not in df.columns:
                            df[col] = data
                else:
                    df = data.copy()
                    # Check for required columns
                    for col in required_columns:
                        if col not in df.columns:
                            if col == 'volume':
                                df[col] = 1
                            else:
                                df[col] = df['close'] if 'close' in df.columns else df.iloc[:, 0]
                
                processed_data[tf] = df
            
            # Recalculate weights based on available timeframes
            total_weight = sum(self.weights[tf] for tf in available_timeframes)
            adjusted_weights = {
                tf: self.weights[tf] / total_weight 
                for tf in available_timeframes
            }
            
            logger.info(f"Analyzing {len(available_timeframes)} timeframes: {', '.join(available_timeframes)}")
            logger.debug(f"Adjusted weights: {adjusted_weights}")
            
            # Analyze available timeframes with processed data
            trend_analysis = self._analyze_trend_alignment(processed_data, adjusted_weights)
            structure_analysis = self._analyze_structure_alignment(processed_data, adjusted_weights)
            momentum_analysis = self._analyze_momentum_alignment(processed_data, adjusted_weights)
            
            # Calculate overall bias with confidence adjustment
            confidence_factor = len(available_timeframes) / len(self.timeframes)
            bias = self._calculate_mtf_bias(
                trend_analysis,
                structure_analysis,
                momentum_analysis,
                confidence_factor
            )
            
            return {
                'trend_alignment': trend_analysis,
                'structure_alignment': structure_analysis,
                'momentum_alignment': momentum_analysis,
                'overall_bias': bias,
                'available_timeframes': available_timeframes,
                'confidence_factor': confidence_factor,
                'current_timeframe': timeframe
            }
            
        except Exception as e:
            logger.error(f"Error in MTF analysis: {str(e)}")
            return self._get_default_analysis()
    
    def _analyze_trend_alignment(self, dataframes: Dict[str, pd.DataFrame], weights: Dict[str, float]) -> Dict:
        """Analyze trend alignment across timeframes."""
        try:
            trends = {}
            alignment_score = 0
            available_weights_sum = 0
            
            # Only analyze available timeframes
            for tf in dataframes.keys():
                if tf not in self.weights:
                    logger.warning(f"Skipping unknown timeframe: {tf}")
                    continue
                    
                df = dataframes[tf]
                available_weights_sum += self.weights[tf]
                
                # Calculate EMAs
                df['ema_20'] = df['close'].ewm(span=20).mean()
                df['ema_50'] = df['close'].ewm(span=50).mean()
                df['ema_200'] = df['close'].ewm(span=200).mean()
                
                # Determine trend
                current_price = df['close'].iloc[-1]
                if current_price > df['ema_20'].iloc[-1] > df['ema_50'].iloc[-1] > df['ema_200'].iloc[-1]:
                    trend = 'bullish'
                    score = 1
                elif current_price < df['ema_20'].iloc[-1] < df['ema_50'].iloc[-1] < df['ema_200'].iloc[-1]:
                    trend = 'bearish'
                    score = -1
                else:
                    trend = 'neutral'
                    score = 0
                
                trends[tf] = {
                    'trend': trend,
                    'score': score
                }
                
                alignment_score += score * self.weights[tf]
            
            # Normalize alignment score based on available weights
            if available_weights_sum > 0:
                alignment_score = alignment_score / available_weights_sum
            
            return {
                'timeframes': trends,
                'alignment_score': alignment_score,
                'aligned': abs(alignment_score) > 0.5,
                'available_timeframes': list(trends.keys())
            }
            
        except Exception as e:
            logger.error(f"Error analyzing trend alignment: {str(e)}")
            return {
                'timeframes': {},
                'alignment_score': 0,
                'aligned': False,
                'available_timeframes': []
            }
    
    def _analyze_structure_alignment(self, dataframes: Dict[str, pd.DataFrame], weights: Dict[str, float]) -> Dict:
        """Analyze market structure alignment across timeframes."""
        try:
            structures = {}
            alignment_score = 0
            available_weights_sum = 0
            
            # Only analyze available timeframes
            for tf in dataframes.keys():
                if tf not in self.weights:
                    logger.warning(f"Skipping unknown timeframe: {tf}")
                    continue
                    
                df = dataframes[tf]
                available_weights_sum += self.weights[tf]
                
                # Find swing points
                highs = []
                lows = []
                
                for i in range(2, len(df)-2):
                    # Swing high
                    if df['high'].iloc[i] > df['high'].iloc[i-1] and \
                       df['high'].iloc[i] > df['high'].iloc[i-2] and \
                       df['high'].iloc[i] > df['high'].iloc[i+1] and \
                       df['high'].iloc[i] > df['high'].iloc[i+2]:
                        highs.append(df['high'].iloc[i])
                    
                    # Swing low
                    if df['low'].iloc[i] < df['low'].iloc[i-1] and \
                       df['low'].iloc[i] < df['low'].iloc[i-2] and \
                       df['low'].iloc[i] < df['low'].iloc[i+1] and \
                       df['low'].iloc[i] < df['low'].iloc[i+2]:
                        lows.append(df['low'].iloc[i])
                
                # Determine structure
                if len(highs) >= 2 and len(lows) >= 2:
                    if highs[-1] > highs[-2] and lows[-1] > lows[-2]:
                        structure = 'bullish'
                        score = 1
                    elif highs[-1] < highs[-2] and lows[-1] < lows[-2]:
                        structure = 'bearish'
                        score = -1
                    else:
                        structure = 'neutral'
                        score = 0
                else:
                    structure = 'neutral'
                    score = 0
                
                structures[tf] = {
                    'structure': structure,
                    'score': score,
                    'swing_highs': len(highs),
                    'swing_lows': len(lows)
                }
                
                alignment_score += score * self.weights[tf]
            
            # Normalize alignment score based on available weights
            if available_weights_sum > 0:
                alignment_score = alignment_score / available_weights_sum
            
            return {
                'timeframes': structures,
                'alignment_score': alignment_score,
                'aligned': abs(alignment_score) > 0.5,
                'available_timeframes': list(structures.keys())
            }
            
        except Exception as e:
            logger.error(f"Error analyzing structure alignment: {str(e)}")
            return {
                'timeframes': {},
                'alignment_score': 0,
                'aligned': False,
                'available_timeframes': []
            }
    
    def _analyze_momentum_alignment(self, dataframes: Dict[str, pd.DataFrame], weights: Dict[str, float]) -> Dict:
        """Analyze momentum alignment across timeframes."""
        try:
            momentum = {}
            alignment_score = 0
            available_weights_sum = 0
            
            # Only analyze available timeframes
            for tf in dataframes.keys():
                if tf not in self.weights:
                    logger.warning(f"Skipping unknown timeframe: {tf}")
                    continue
                    
                df = dataframes[tf]
                available_weights_sum += self.weights[tf]
                
                # Calculate RSI
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['rsi'] = 100 - (100 / (1 + rs))
                
                # Calculate MACD
                exp1 = df['close'].ewm(span=12).mean()
                exp2 = df['close'].ewm(span=26).mean()
                df['macd'] = exp1 - exp2
                df['signal'] = df['macd'].ewm(span=9).mean()
                
                # Determine momentum
                rsi = df['rsi'].iloc[-1]
                macd_hist = df['macd'].iloc[-1] - df['signal'].iloc[-1]
                
                if rsi > 50 and macd_hist > 0:
                    mom = 'bullish'
                    score = 1
                elif rsi < 50 and macd_hist < 0:
                    mom = 'bearish'
                    score = -1
                else:
                    mom = 'neutral'
                    score = 0
                
                momentum[tf] = {
                    'momentum': mom,
                    'score': score,
                    'rsi': rsi,
                    'macd_hist': macd_hist
                }
                
                alignment_score += score * self.weights[tf]
            
            # Normalize alignment score based on available weights
            if available_weights_sum > 0:
                alignment_score = alignment_score / available_weights_sum
            
            return {
                'timeframes': momentum,
                'alignment_score': alignment_score,
                'aligned': abs(alignment_score) > 0.5,
                'available_timeframes': list(momentum.keys())
            }
            
        except Exception as e:
            logger.error(f"Error analyzing momentum alignment: {str(e)}")
            return {
                'timeframes': {},
                'alignment_score': 0,
                'aligned': False,
                'available_timeframes': []
            }
    
    def _calculate_mtf_bias(self, trend: Dict, structure: Dict, momentum: Dict, confidence_factor: float) -> Dict:
        """Calculate overall bias based on all MTF components with confidence adjustment."""
        try:
            # Component weights
            weights = {
                'trend': 0.4,
                'structure': 0.4,
                'momentum': 0.2
            }
            
            # Calculate weighted score
            total_score = (
                trend['alignment_score'] * weights['trend'] +
                structure['alignment_score'] * weights['structure'] +
                momentum['alignment_score'] * weights['momentum']
            )
            
            # Adjust score based on available timeframe confidence
            adjusted_score = total_score * confidence_factor
            
            # Determine bias strength with adjusted thresholds
            if abs(adjusted_score) >= 0.7 * confidence_factor:
                strength = 'strong'
            elif abs(adjusted_score) >= 0.4 * confidence_factor:
                strength = 'moderate'
            else:
                strength = 'weak'
            
            return {
                'bias': 'bullish' if adjusted_score > 0 else 'bearish' if adjusted_score < 0 else 'neutral',
                'strength': strength,
                'raw_score': total_score,
                'adjusted_score': adjusted_score,
                'confidence_factor': confidence_factor,
                'components': {
                    'trend': trend['alignment_score'],
                    'structure': structure['alignment_score'],
                    'momentum': momentum['alignment_score']
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating MTF bias: {str(e)}")
            return {
                'bias': 'neutral',
                'strength': 'weak',
                'raw_score': 0,
                'adjusted_score': 0,
                'confidence_factor': 0,
                'components': {
                    'trend': 0,
                    'structure': 0,
                    'momentum': 0
                }
            }
    
    def _get_default_analysis(self) -> Dict:
        """Return default analysis when data is insufficient."""
        return {
            'trend_alignment': {
                'bias': 'neutral',
                'strength': 0,
                'timeframes': {}
            },
            'structure_alignment': {
                'bias': 'neutral',
                'strength': 0,
                'timeframes': {}
            },
            'momentum_alignment': {
                'bias': 'neutral',
                'strength': 0,
                'timeframes': {}
            },
            'overall_bias': {
                'direction': 'neutral',
                'strength': 0,
                'confidence': 0
            }
        }
    
    def _find_key_levels(self, df: pd.DataFrame) -> List[float]:
        """Find key price levels in the data."""
        try:
            levels = []
            
            # Add recent swing highs and lows
            window = 10
            for i in range(window, len(df)-window):
                # Swing high
                if df['high'].iloc[i] == df['high'].iloc[i-window:i+window+1].max():
                    levels.append(df['high'].iloc[i])
                # Swing low
                if df['low'].iloc[i] == df['low'].iloc[i-window:i+window+1].min():
                    levels.append(df['low'].iloc[i])
            
            # Remove duplicates and sort
            levels = sorted(list(set(levels)))
            
            # Keep only the most recent levels (last 5)
            return levels[-5:] if levels else []
            
        except Exception as e:
            logger.error(f"Error finding key levels: {str(e)}")
            return [] 