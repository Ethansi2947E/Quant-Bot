from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from loguru import logger

class VolumeAnalysis:
    def __init__(self):
        self.volume_ma_period = 20
        self.delta_threshold = 0.6  # 60% of average volume for significance
        self.profile_levels = 50    # Number of levels for volume profile
        
    def analyze(self, df: pd.DataFrame) -> Dict:
        """Analyze volume patterns and indicators with data quality assessment."""
        try:
            # Create a copy of the dataframe to avoid warnings
            df = df.copy()
            
            # Check volume data quality
            data_quality = self._assess_volume_data_quality(df)
            if data_quality == 0:
                logger.warning("Volume data not available or invalid, skipping volume analysis")
                return {
                    'trend': 'neutral',
                    'momentum': 0,
                    'patterns': [],
                    'data_quality': 0,
                    'reason': 'Missing or invalid volume data'
                }
            
            # Calculate volume indicators
            df.loc[:, 'volume_sma'] = df['volume'].rolling(window=self.volume_ma_period).mean()
            df.loc[:, 'vwap'] = (df['volume'] * ((df['high'] + df['low'] + df['close']) / 3)).cumsum() / df['volume'].cumsum()
            df.loc[:, 'relative_volume'] = df['volume'] / df['volume_sma']
            df.loc[:, 'up_volume'] = df['volume'].where(df['close'] > df['open'], 0)
            df.loc[:, 'down_volume'] = df['volume'].where(df['close'] < df['open'], 0)
            
            # Analyze volume patterns
            patterns = self._detect_volume_patterns(df)
            
            # Get latest volume trend and momentum
            latest_obv_trend = df['obv_trend'].iloc[-1] if 'obv_trend' in df.columns else 'neutral'
            latest_momentum = df['obv_momentum'].iloc[-1] if 'obv_momentum' in df.columns else 0
            
            return {
                'trend': latest_obv_trend,
                'momentum': latest_momentum,
                'patterns': patterns,
                'data_quality': data_quality,
                'indicators': {
                    'obv': df['obv'].iloc[-1] if 'obv' in df.columns else None,
                    'obv_ema': df['obv_ema'].iloc[-1] if 'obv_ema' in df.columns else None,
                    'volume_ma': df['volume_ma'].iloc[-1] if 'volume_ma' in df.columns else None
                }
            }
            
        except Exception as e:
            logger.error(f"Error in volume analysis: {str(e)}")
            return {
                'trend': 'neutral',
                'momentum': 0,
                'patterns': [],
                'data_quality': 0,
                'reason': f'Analysis error: {str(e)}'
            }
            
    def _assess_volume_data_quality(self, df: pd.DataFrame) -> float:
        """Assess the quality of volume data."""
        try:
            if 'volume' not in df.columns:
                return 0
                
            # Check for missing values
            missing_ratio = df['volume'].isna().mean()
            if missing_ratio > 0.1:  # More than 10% missing
                return 0
                
            # Check for zero values
            zero_ratio = (df['volume'] == 0).mean()
            if zero_ratio > 0.2:  # More than 20% zeros
                return 0
                
            # Calculate quality score based on data consistency
            quality_score = 1.0
            
            # Penalize for high percentage of zeros
            if zero_ratio > 0:
                quality_score *= (1 - zero_ratio)
                
            # Penalize for missing values
            if missing_ratio > 0:
                quality_score *= (1 - missing_ratio)
                
            # Check for sudden volume spikes
            volume_std = df['volume'].std()
            volume_mean = df['volume'].mean()
            if volume_std / volume_mean > 5:  # High volatility in volume
                quality_score *= 0.8
                
            return round(quality_score, 2)
            
        except Exception as e:
            logger.error(f"Error assessing volume data quality: {str(e)}")
            return 0
    
    def _calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate various volume-based indicators."""
        try:
            # Check if volume data is available and valid
            if 'volume' not in df.columns or df['volume'].sum() == 0:
                logger.warning("Volume data not available or invalid, skipping volume analysis")
                return df
            
            # Volume Moving Average
            df['volume_sma'] = df['volume'].rolling(window=self.volume_ma_period).mean()
            
            # Volume Weighted Average Price (VWAP)
            df['vwap'] = (df['volume'] * ((df['high'] + df['low'] + df['close']) / 3)).cumsum() / df['volume'].cumsum()
            
            # Relative Volume
            df['relative_volume'] = df['volume'] / df['volume_sma']
            
            # Up/Down Volume
            df['up_volume'] = df['volume'].where(df['close'] > df['open'], 0)
            df['down_volume'] = df['volume'].where(df['close'] < df['open'], 0)
            
            # Volume Force
            price_change = df['close'] - df['open']
            df['volume_force'] = price_change * df['volume']
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating volume indicators: {str(e)}")
            return df
    
    def _calculate_volume_profile(self, df: pd.DataFrame) -> Dict:
        """Calculate Volume Profile (TPO)."""
        try:
            price_range = df['high'].max() - df['low'].min()
            level_height = price_range / self.profile_levels
            
            levels = {}
            total_volume = 0
            
            # Calculate volume at each price level
            for i in range(self.profile_levels):
                level_price = df['low'].min() + (i * level_height)
                level_volume = df['volume'][
                    (df['low'] <= level_price + level_height) & 
                    (df['high'] >= level_price)
                ].sum()
                
                levels[level_price] = level_volume
                total_volume += level_volume
            
            # Find Point of Control (POC)
            poc_price = max(levels.items(), key=lambda x: x[1])[0]
            
            # Calculate Value Area (70% of total volume)
            value_area_volume = total_volume * 0.7
            cumulative_volume = 0
            value_area = {'high': poc_price, 'low': poc_price}
            
            # Expand value area above and below POC
            above_poc = {k: v for k, v in levels.items() if k > poc_price}
            below_poc = {k: v for k, v in levels.items() if k < poc_price}
            
            while cumulative_volume < value_area_volume:
                above_vol = max(above_poc.values()) if above_poc else 0
                below_vol = max(below_poc.values()) if below_poc else 0
                
                if above_vol > below_vol:
                    price = max(above_poc.items(), key=lambda x: x[1])[0]
                    cumulative_volume += above_vol
                    value_area['high'] = price
                    del above_poc[price]
                else:
                    price = max(below_poc.items(), key=lambda x: x[1])[0]
                    cumulative_volume += below_vol
                    value_area['low'] = price
                    del below_poc[price]
            
            return {
                'poc': poc_price,
                'value_area': value_area,
                'levels': levels
            }
            
        except Exception as e:
            logger.error(f"Error calculating volume profile: {str(e)}")
            return {
                'poc': None,
                'value_area': {'high': None, 'low': None},
                'levels': {}
            }
    
    def _calculate_cumulative_delta(self, df: pd.DataFrame) -> Dict:
        """Calculate Cumulative Volume Delta."""
        try:
            # Calculate buying/selling volume
            buying_volume = df['volume'].where(df['close'] > df['open'], 0)
            selling_volume = df['volume'].where(df['close'] < df['open'], 0)
            
            # Cumulative delta
            df['cvd'] = (buying_volume - selling_volume).cumsum()
            
            # Delta trend
            current_delta = df['cvd'].iloc[-1]
            delta_sma = df['cvd'].rolling(window=self.volume_ma_period).mean().iloc[-1]
            
            if current_delta > delta_sma * (1 + self.delta_threshold):
                trend = 'bullish'
            elif current_delta < delta_sma * (1 - self.delta_threshold):
                trend = 'bearish'
            else:
                trend = 'neutral'
            
            return {
                'current': current_delta,
                'sma': delta_sma,
                'trend': trend
            }
            
        except Exception as e:
            logger.error(f"Error calculating cumulative delta: {str(e)}")
            return {
                'current': 0,
                'sma': 0,
                'trend': 'neutral'
            }
    
    def _find_volume_levels(self, df: pd.DataFrame, profile: Dict) -> Dict:
        """Find support/resistance levels based on volume."""
        try:
            support = []
            resistance = []
            
            # Use volume profile levels
            levels = profile['levels']
            avg_volume = sum(levels.values()) / len(levels)
            
            for price, volume in levels.items():
                if volume > avg_volume * 1.5:  # Significant volume level
                    current_price = df['close'].iloc[-1]
                    
                    if price < current_price:
                        support.append({
                            'price': price,
                            'strength': volume / avg_volume,
                            'volume': volume
                        })
                    else:
                        resistance.append({
                            'price': price,
                            'strength': volume / avg_volume,
                            'volume': volume
                        })
            
            # Sort by strength
            support = sorted(support, key=lambda x: x['strength'], reverse=True)
            resistance = sorted(resistance, key=lambda x: x['strength'], reverse=True)
            
            return {
                'support': support[:3],  # Top 3 support levels
                'resistance': resistance[:3]  # Top 3 resistance levels
            }
            
        except Exception as e:
            logger.error(f"Error finding volume levels: {str(e)}")
            return {'support': [], 'resistance': []}
    
    def _detect_volume_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect volume-based trading patterns."""
        try:
            patterns = []
            
            # Check if required columns exist
            required_columns = ['volume', 'volume_sma', 'close', 'open']
            if not all(col in df.columns for col in required_columns):
                logger.warning("Missing required columns for volume pattern detection")
                return []
            
            for i in range(3, len(df)):
                window = df.iloc[i-3:i+1]
                
                # Volume Climax (high volume with price reversal)
                if df['volume'].iloc[i] > df['volume_sma'].iloc[i] * 2:
                    if df['close'].iloc[i] < df['open'].iloc[i] and \
                       df['close'].iloc[i-1] > df['open'].iloc[i-1]:
                        patterns.append({
                            'type': 'volume_climax',
                            'direction': 'bearish',
                            'index': i,
                            'volume': df['volume'].iloc[i],
                            'price': df['close'].iloc[i]
                        })
                    elif df['close'].iloc[i] > df['open'].iloc[i] and \
                         df['close'].iloc[i-1] < df['open'].iloc[i-1]:
                        patterns.append({
                            'type': 'volume_climax',
                            'direction': 'bullish',
                            'index': i,
                            'volume': df['volume'].iloc[i],
                            'price': df['close'].iloc[i]
                        })
                
                # Volume Dry-up (low volume after trend)
                if df['volume'].iloc[i] < df['volume_sma'].iloc[i] * 0.5:
                    # After uptrend
                    if all(df['close'].iloc[i-j] > df['close'].iloc[i-j-1] for j in range(3)):
                        patterns.append({
                            'type': 'volume_dryup',
                            'direction': 'bearish',
                            'index': i,
                            'volume': df['volume'].iloc[i],
                            'price': df['close'].iloc[i]
                        })
                    # After downtrend
                    elif all(df['close'].iloc[i-j] < df['close'].iloc[i-j-1] for j in range(3)):
                        patterns.append({
                            'type': 'volume_dryup',
                            'direction': 'bullish',
                            'index': i,
                            'volume': df['volume'].iloc[i],
                            'price': df['close'].iloc[i]
                        })
                
                # Smart Money Accumulation/Distribution
                if df['volume'].iloc[i] > df['volume_sma'].iloc[i] * 1.5:
                    try:
                        price_change = abs(df['close'].iloc[i] - df['open'].iloc[i])
                        avg_price_change = abs(df['close'] - df['open']).rolling(20).mean().iloc[i]
                        
                        # Accumulation (high volume, small price movement)
                        if price_change < avg_price_change * 0.5:
                            patterns.append({
                                'type': 'accumulation',
                                'index': i,
                                'volume': df['volume'].iloc[i],
                                'price': df['close'].iloc[i]
                            })
                        # Distribution (high volume, large price movement)
                        elif price_change > avg_price_change * 2:
                            patterns.append({
                                'type': 'distribution',
                                'index': i,
                                'volume': df['volume'].iloc[i],
                                'price': df['close'].iloc[i]
                            })
                    except Exception as e:
                        logger.debug(f"Error calculating price changes: {str(e)}")
                        continue
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting volume patterns: {str(e)}")
            return [] 