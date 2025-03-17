from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from loguru import logger

class MTFAnalysis:
    def __init__(self):
        """Initialize the MTF Analysis module with improved timeframe hierarchy and weighting."""
        # Define timeframe hierarchy from lowest to highest
        self.timeframe_hierarchy = ["M1", "M5", "M15", "H1", "H4", "D1", "W1"]
        
        # Define timeframe weights (higher timeframes have more influence)
        self.weights = {
            "W1": 0.35,   # Weekly timeframe has highest weight
            "D1": 0.25,   # Daily timeframe
            "H4": 0.18,   # 4-hour timeframe
            "H1": 0.12,   # 1-hour timeframe
            "M15": 0.06,  # 15-minute timeframe
            "M5": 0.03,   # 5-minute timeframe
            "M1": 0.01    # 1-minute timeframe has least weight
        }
        
        # Define timeframe relationships for trading
        self.timeframe_relationships = {
            "M5": ["M15", "H1", "H4"],    # Trading on M5 requires confirmation from M15, H1, and H4
            "M15": ["H1", "H4", "D1"],    # Trading on M15 requires confirmation from H1, H4, and D1
            "H1": ["H4", "D1"],           # Trading on H1 requires confirmation from H4, D1
            "H4": ["D1"],                 # Trading on H4 requires confirmation from D1
            "D1": ["W1"],                 # Trading on D1 requires confirmation from W1
            "W1": []                      # No higher timeframe for W1
        }
        
        # Define minimum alignment requirements for each timeframe
        self.minimum_alignment = {
            "M1": 0.5,  # At least 50% of higher timeframes must align
            "M5": 0.6,  # At least 60% of higher timeframes must align
            "M15": 0.7, # At least 70% of higher timeframes must align
            "H1": 0.8,  # At least 80% of higher timeframes must align
            "H4": 0.9,  # At least 90% of higher timeframes must align
            "D1": 1.0,  # All higher timeframes must align
            "W1": 1.0   # All higher timeframes must align
        }

    def get_higher_timeframes(self, timeframe: str) -> List[str]:
        """Get all higher timeframes for a given timeframe."""
        try:
            if timeframe not in self.timeframe_hierarchy:
                logger.warning(f"Unknown timeframe: {timeframe}")
                return []
                
            current_index = self.timeframe_hierarchy.index(timeframe)
            return self.timeframe_hierarchy[current_index + 1:]
        except Exception as e:
            logger.error(f"Error getting higher timeframes: {str(e)}")
            return []

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

    def analyze_mtf(self, dataframes: Dict[str, Union[pd.DataFrame, pd.Series]], current_timeframe: Optional[str] = None) -> Dict:
        """
        Analyze price action across multiple timeframes with improved hierarchy and weighting.
        
        Args:
            dataframes: Dictionary of DataFrames/Series for multiple timeframes
            current_timeframe: The timeframe being traded on
            
        Returns:
            Dict: Analysis results with detailed timeframe alignment information
        """
        try:
            # Validate input is a dictionary with proper timeframe keys
            valid_timeframes = set(self.timeframe_hierarchy)
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
            
            # Analyze each timeframe individually
            individual_analyses = {}
            for tf in available_timeframes:
                individual_analyses[tf] = self._analyze_single_timeframe(processed_data[tf])
            
            # Determine trend for each timeframe
            timeframe_trends = {}
            for tf, score in individual_analyses.items():
                timeframe_trends[tf] = {
                    'trend': 'bullish' if score > 0 else 'bearish' if score < 0 else 'neutral',
                    'score': score,
                    'strength': self._determine_trend_strength(score)
                }
            
            # Calculate alignment scores
            alignment_scores = self._calculate_alignment_scores(timeframe_trends, current_timeframe)
            
            # Calculate overall bias with weighted timeframe influence
            overall_bias = self._calculate_overall_bias(timeframe_trends)
            
            # Determine if the current timeframe is aligned with higher timeframes
            is_aligned = self._check_timeframe_alignment(timeframe_trends, current_timeframe)
            
            # Calculate key levels from all timeframes
            key_levels = self._aggregate_key_levels(processed_data)
            
            # Prepare detailed analysis result
            result = {
                'timeframe_trends': timeframe_trends,
                'alignment_scores': alignment_scores,
                'overall_bias': overall_bias,
                'is_aligned': is_aligned,
                'key_levels': key_levels,
                'available_timeframes': available_timeframes,
                'current_timeframe': current_timeframe
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in MTF analysis: {str(e)}")
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
            
            # Calculate price position relative to EMAs
            current_price = df['close'].iloc[-1]
            price_vs_ema20 = 0.1 if current_price > ema_20.iloc[-1] else -0.1
            price_vs_ema50 = 0.2 if current_price > ema_50.iloc[-1] else -0.2
            price_vs_ema200 = 0.3 if current_price > ema_200.iloc[-1] else -0.3
            
            # Calculate momentum with safety check for short dataframes
            momentum_score = 0.0
            if len(df) >= 5:
                momentum = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]
                momentum_score = min(max(momentum * 10, -0.4), 0.4)  # Scale and cap momentum score
            else:
                # Use available data if less than 5 candles
                if len(df) > 1:
                    momentum = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]
                    momentum_score = min(max(momentum * 10, -0.4), 0.4) * (len(df) / 5)  # Scale by available data
                logger.debug(f"Short dataframe detected in momentum calculation, adjusted using {len(df)} candles")
            
            # Combine scores
            final_score = trend_score + price_vs_ema20 + price_vs_ema50 + price_vs_ema200 + momentum_score
            
            # Normalize to range [-1, 1]
            final_score = max(min(final_score, 1), -1)
            
            return final_score
            
        except Exception as e:
            logger.error(f"Error in single timeframe analysis: {str(e)}")
            return 0.0

    def _determine_trend_strength(self, score: float) -> str:
        """Determine trend strength based on score."""
        abs_score = abs(score)
        if abs_score >= 0.8:
            return 'strong'
        elif abs_score >= 0.5:
            return 'moderate'
        elif abs_score >= 0.2:
            return 'weak'
        else:
            return 'neutral'

    def _calculate_alignment_scores(self, timeframe_trends: Dict, current_timeframe: Optional[str]) -> Dict:
        """Calculate alignment scores between timeframes."""
        if not current_timeframe or current_timeframe not in timeframe_trends:
            return {}
        
        alignment_scores = {}
        current_trend = timeframe_trends[current_timeframe]['trend']
        
        # Get higher timeframes
        higher_timeframes = self.get_higher_timeframes(current_timeframe)
        available_higher_timeframes = [tf for tf in higher_timeframes if tf in timeframe_trends]
        
        # Calculate alignment with each higher timeframe
        for tf in available_higher_timeframes:
            higher_trend = timeframe_trends[tf]['trend']
            if higher_trend == 'neutral' or current_trend == 'neutral':
                alignment_scores[tf] = 0.5  # Neutral trends are partially aligned with anything
            elif higher_trend == current_trend:
                alignment_scores[tf] = 1.0  # Perfect alignment
            else:
                alignment_scores[tf] = 0.0  # Misalignment
        
        # Calculate overall alignment score
        if alignment_scores:
            weighted_scores = []
            total_weight = 0
            
            for tf, score in alignment_scores.items():
                weight = self.weights.get(tf, 0.1)
                weighted_scores.append(score * weight)
                total_weight += weight
            
            if total_weight > 0:
                alignment_scores['overall'] = sum(weighted_scores) / total_weight
            else:
                alignment_scores['overall'] = 0.5
        else:
            alignment_scores['overall'] = 1.0  # No higher timeframes to align with
        
        return alignment_scores

    def _calculate_overall_bias(self, timeframe_trends: Dict) -> Dict:
        """Calculate overall market bias with weighted timeframe influence."""
        if not timeframe_trends:
            return {'bias': 'neutral', 'strength': 'weak', 'score': 0.0}
        
        weighted_scores = []
        total_weight = 0
        
        for tf, analysis in timeframe_trends.items():
            score = analysis['score']
            weight = self.weights.get(tf, 0.1)
            weighted_scores.append(score * weight)
            total_weight += weight
        
        if total_weight > 0:
            overall_score = sum(weighted_scores) / total_weight
        else:
            overall_score = 0.0
        
        # Determine bias and strength
        bias = 'bullish' if overall_score > 0 else 'bearish' if overall_score < 0 else 'neutral'
        strength = self._determine_trend_strength(overall_score)
        
        return {
            'bias': bias,
            'strength': strength,
            'score': overall_score
        }

    def _check_timeframe_alignment(self, timeframe_trends: Dict, current_timeframe: Optional[str]) -> Dict:
        """Check if the current timeframe is aligned with higher timeframes."""
        if not current_timeframe or current_timeframe not in timeframe_trends:
            return {'is_aligned': False, 'alignment_ratio': 0, 'required_ratio': 0}
        
        # Get required timeframes for the current trading timeframe
        required_timeframes = self.timeframe_relationships.get(current_timeframe, [])
        available_required = [tf for tf in required_timeframes if tf in timeframe_trends]
        
        if not available_required:
            return {'is_aligned': True, 'alignment_ratio': 1.0, 'required_ratio': 0}
        
        # Get current timeframe trend
        current_trend = timeframe_trends[current_timeframe]['trend']
        if current_trend == 'neutral':
            return {'is_aligned': False, 'alignment_ratio': 0, 'required_ratio': self.minimum_alignment[current_timeframe]}
        
        # Count aligned timeframes
        aligned_count = 0
        for tf in available_required:
            higher_trend = timeframe_trends[tf]['trend']
            if higher_trend == current_trend or higher_trend == 'neutral':
                aligned_count += 1
        
        # Calculate alignment ratio
        alignment_ratio = aligned_count / len(available_required)
        required_ratio = self.minimum_alignment.get(current_timeframe, 0.7)
        
        return {
            'is_aligned': alignment_ratio >= required_ratio,
            'alignment_ratio': alignment_ratio,
            'required_ratio': required_ratio,
            'aligned_timeframes': aligned_count,
            'total_required': len(available_required)
        }

    def _find_key_levels(self, df: pd.DataFrame) -> List[float]:
        """Find key price levels in a single timeframe with optimized numpy operations."""
        try:
            if 'volume' in df.columns:
                # Create price bins
                price_range = df['high'].max() - df['low'].min()
                bin_size = price_range / 50  # 50 bins across the price range
                
                # Create bins with numpy
                min_price = df['low'].min()
                max_price = df['high'].max()
                bins = np.linspace(min_price, max_price, 51)  # 51 bin edges for 50 bins
                
                # Initialize volume profile array
                volume_profile = np.zeros(50)
                
                # Extract numpy arrays for vectorized operations
                lows = df['low'].values
                highs = df['high'].values
                volumes = df['volume'].values
                
                # For each bin, find candles that overlap and add weighted volume
                for bin_idx in range(len(volume_profile)):
                    bin_min = bins[bin_idx]
                    bin_max = bins[bin_idx + 1]
                    
                    # Create boolean mask for candles that overlap with this bin
                    # A candle overlaps if its high is above bin_min and its low is below bin_max
                    overlaps = (highs > bin_min) & (lows < bin_max)
                    
                    if np.any(overlaps):
                        # For overlapping candles, calculate the overlap portion
                        overlap_candles_low = lows[overlaps]
                        overlap_candles_high = highs[overlaps]
                        overlap_candles_volume = volumes[overlaps]
                        
                        # Calculate candle ranges (vectorized)
                        candle_ranges = np.maximum(0.0001, overlap_candles_high - overlap_candles_low)
                        
                        # Calculate overlaps (vectorized)
                        overlaps_size = np.minimum(overlap_candles_high, bin_max) - np.maximum(overlap_candles_low, bin_min)
                        
                        # Calculate weights and sum weighted volumes
                        weights = overlaps_size / candle_ranges
                        volume_profile[bin_idx] = np.sum(overlap_candles_volume * weights)
                
                # Find local maxima in volume profile (high volume nodes)
                # Use numpy's gradient to find peaks more efficiently
                volume_mean = np.mean(volume_profile)
                gradient = np.gradient(volume_profile)
                peak_indices = []
                
                for i in range(1, len(volume_profile) - 1):
                    # A peak has gradient sign change from positive to negative
                    if gradient[i-1] > 0 and gradient[i] < 0:
                        # Only include significant peaks
                        if volume_profile[i] > volume_mean * 1.5:
                            peak_indices.append(i)
                
                # Convert peak indices to price levels
                key_levels = [bins[i] for i in peak_indices]
                
                # Add recent swing highs and lows
                window = min(20, len(df) // 4)  # Look at recent price action
                if window > 2:  # Need at least 3 candles to find swings
                    recent_df = df.iloc[-window:]
                    
                    # Use numpy for faster swing point detection
                    recent_highs = recent_df['high'].values
                    recent_lows = recent_df['low'].values
                    
                    # Find swing highs (where a high is greater than both neighbors)
                    for i in range(1, len(recent_highs) - 1):
                        if recent_highs[i] > recent_highs[i-1] and recent_highs[i] > recent_highs[i+1]:
                            key_levels.append(recent_highs[i])
                    
                    # Find swing lows (where a low is less than both neighbors)
                    for i in range(1, len(recent_lows) - 1):
                        if recent_lows[i] < recent_lows[i-1] and recent_lows[i] < recent_lows[i+1]:
                            key_levels.append(recent_lows[i])
                
                # Add current price
                key_levels.append(df['close'].iloc[-1])
                
                # Remove duplicates and sort
                # Round to 6 decimal places to avoid floating-point comparison issues
                rounded_levels = np.round(key_levels, 6)
                unique_levels = np.unique(rounded_levels)
                
                return sorted(unique_levels.tolist())
            else:
                # Simple fallback if no volume data
                return [round(level, 6) for level in [
                    df['close'].iloc[-1],  # Current price
                    df['high'].max(),      # All-time high
                    df['low'].min()        # All-time low
                ]]
                
        except Exception as e:
            logger.error(f"Error finding key levels: {str(e)}")
            return []

    def _aggregate_key_levels(self, dataframes: Dict[str, pd.DataFrame]) -> Dict[str, List[float]]:
        """Aggregate key price levels from multiple timeframes."""
        all_levels = {}
        for tf, df in dataframes.items():
            levels = self._find_key_levels(df)
            all_levels[tf] = levels
        return all_levels

    def _get_default_analysis(self) -> Dict:
        """Return default analysis result when analysis fails."""
        return {
            'trend': 'neutral',
            'score': 0.0,
            'timeframe': 'unknown',
            'key_levels': [],
            'confidence': 0.0,
            'analysis_type': 'default',
            'overall_bias': {'bias': 'neutral', 'strength': 'weak', 'score': 0.0},
            'is_aligned': False
        }

    def analyze_timeframe_correlation(self, timeframes_data: Dict[str, pd.DataFrame]) -> Dict:
        """Analyze correlation between different timeframes."""
        correlations = {}
        timeframes = list(timeframes_data.keys())
        
        for i in range(len(timeframes)):
            for j in range(i+1, len(timeframes)):
                tf1, tf2 = timeframes[i], timeframes[j]
                correlation = self.calculate_timeframe_correlation(
                    timeframes_data[tf1], 
                    timeframes_data[tf2]
                )
                correlations[f"{tf1}_{tf2}"] = correlation
        
        return correlations

    def calculate_timeframe_correlation(self, df1: pd.DataFrame, df2: pd.DataFrame) -> float:
        """Calculate correlation between two timeframes."""
        try:
            # Resample higher timeframe data to match lower timeframe
            if len(df1) > len(df2):
                # df1 is lower timeframe (more data points)
                lower_tf_returns = df1['close'].pct_change().dropna()
                higher_tf_returns = df2['close'].pct_change().dropna()
                
                # Take last n values of lower_tf_returns where n = len(higher_tf_returns)
                if len(lower_tf_returns) > len(higher_tf_returns):
                    lower_tf_returns = lower_tf_returns.iloc[-len(higher_tf_returns):]
            else:
                # df2 is lower timeframe (more data points)
                lower_tf_returns = df2['close'].pct_change().dropna()
                higher_tf_returns = df1['close'].pct_change().dropna()
                
                # Take last n values of lower_tf_returns where n = len(higher_tf_returns)
                if len(lower_tf_returns) > len(higher_tf_returns):
                    lower_tf_returns = lower_tf_returns.iloc[-len(higher_tf_returns):]
            
            # Calculate correlation if we have enough data points
            if len(lower_tf_returns) > 5 and len(higher_tf_returns) > 5:
                # Make sure the series are the same length
                min_length = min(len(lower_tf_returns), len(higher_tf_returns))
                correlation = np.corrcoef(
                    lower_tf_returns.iloc[-min_length:],
                    higher_tf_returns.iloc[-min_length:]
                )[0, 1]
                return correlation
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating timeframe correlation: {str(e)}")
            return 0.0

    def analyze_multiple_timeframes(self, market_data: Dict) -> Dict:
        """
        Analyze market data across multiple timeframes.
        This is a compatibility method used by the trading bot that redirects to analyze_mtf.
        
        Args:
            market_data: Dictionary containing dataframes for different timeframes
            
        Returns:
            Dictionary with MTF analysis results
        """
        logger.debug("Running analyze_multiple_timeframes (redirecting to analyze_mtf)")
        
        # Extract the current timeframe from the market data (use the first timeframe as the current one)
        current_timeframe = None
        if market_data:
            # Find the first key that's a recognized timeframe
            for tf in self.timeframe_hierarchy:
                if tf in market_data:
                    current_timeframe = tf
                    break
            
            # If we didn't find a recognized timeframe, use the first key
            if current_timeframe is None and market_data:
                current_timeframe = next(iter(market_data.keys()))
        
        # Call the existing analyze_mtf method
        return self.analyze_mtf(market_data, current_timeframe) 