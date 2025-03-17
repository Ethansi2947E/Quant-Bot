from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
from loguru import logger
from src.mt5_handler import MT5Handler

class VolumeAnalysis:
    def __init__(self):
        self.volume_ma_period = 20
        self.delta_threshold = 0.4  # Decreased from 0.6
        self.profile_levels = 50    # Number of levels for volume profile
        
    def analyze(self, df: pd.DataFrame) -> Dict:
        """Analyze volume patterns and indicators with data quality assessment."""
        try:
            logger.info("Starting volume analysis...")
            # Create a copy of the dataframe to avoid warnings
            df = df.copy()
            
            # Check volume data quality
            logger.debug("Assessing volume data quality...")
            data_quality = self._assess_volume_data_quality(df)
            logger.info(f"Volume data quality score: {data_quality}")
            
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
            logger.debug("Calculating volume indicators...")
            df = self._calculate_volume_indicators(df)
            
            # Analyze indicator relationships
            logger.debug("Analyzing indicator relationships...")
            relationships = self._analyze_indicator_relationships(df)
            
            # Calculate weighted volume distribution for better accuracy
            logger.debug("Calculating weighted volume distribution...")
            volume_distribution = self.weighted_volume_distribution(df, num_levels=self.profile_levels)
            
            # Analyze volume patterns
            logger.debug("Detecting volume patterns...")
            patterns = self._detect_volume_patterns(df)
            logger.info(f"Found {len(patterns)} volume patterns")
            
            # Validate patterns against price action
            patterns = self._validate_patterns(df, patterns)
            logger.info(f"Validated patterns: {len(patterns)}")
            
            for pattern in patterns:
                logger.debug(f"Pattern detected: {pattern['type']} ({pattern['direction'] if 'direction' in pattern else 'n/a'})")
            
            # Calculate comprehensive momentum
            momentum = self._calculate_comprehensive_momentum(df)
            
            # Get latest volume trend
            latest_obv_trend = df['obv_trend'].iloc[-1] if 'obv_trend' in df.columns else 'neutral'
            
            # Find key volume levels (support/resistance)
            try:
                key_levels = []
                if volume_distribution['high_volume_nodes']:
                    current_price = df['close'].iloc[-1]
                    for node in volume_distribution['high_volume_nodes']:
                        level_type = 'support' if node['price'] < current_price else 'resistance'
                        strength = node['volume'] / (df['volume'].mean() * self.profile_levels / 10)
                        key_levels.append({
                            'price': node['price'],
                            'type': level_type,
                            'strength': min(1.0, strength)  # Cap strength at 1.0
                        })
            except Exception as e:
                logger.error(f"Error finding key volume levels: {str(e)}")
                key_levels = []
            
            # Calculate cumulative delta
            delta = self._calculate_cumulative_delta(df)
            
            logger.info(f"Analysis complete - Trend: {latest_obv_trend}, Momentum: {momentum:.2f}")
            
            return {
                'trend': latest_obv_trend,
                'momentum': momentum,
                'patterns': patterns,
                'data_quality': data_quality,
                'relationships': relationships,
                'indicators': {
                    'obv': df['obv'].iloc[-1] if 'obv' in df.columns else None,
                    'obv_ema': df['obv_ema'].iloc[-1] if 'obv_ema' in df.columns else None,
                    'volume_ma': df['volume_sma'].iloc[-1] if 'volume_sma' in df.columns else None,
                    'relative_volume': df['relative_volume'].iloc[-1] if 'relative_volume' in df.columns else None
                },
                'volume_distribution': {
                    'poc': volume_distribution['poc'],
                    'high_volume_nodes': volume_distribution['high_volume_nodes']
                },
                'key_levels': key_levels,
                'cumulative_delta': {
                    'value': delta['current'],
                    'trend': delta['trend']
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
        
    def calculate_momentum(self, data: Union[Dict, pd.DataFrame]) -> float:
        """Calculate momentum score based on multiple factors with enhanced accuracy.
        
        Args:
            data: Either a pandas DataFrame with OHLCV data or a dictionary containing
                 market data with OHLCV information.
                 
        Returns:
            float: A momentum score between -100 and 100, where:
                  - Positive values indicate bullish momentum
                  - Negative values indicate bearish momentum
                  - The magnitude indicates the strength of the momentum
        """
        try:
            # Get DataFrame from input
            df = self._prepare_dataframe(data)
            if df is None:
                return 0.0
                
            # Validate data requirements
            if not self._validate_momentum_data(df):
                return 0.0

            # Calculate technical indicators
            indicators = self._calculate_momentum_indicators(df)
            
            # Calculate component scores
            scores = self._calculate_component_scores(indicators)
            
            # Combine scores with weights
            weights = {
                'rsi': 0.25,
                'price': 0.30,
                'volume': 0.20,
                'macd': 0.15,
                'ma': 0.10
            }
            
            momentum_score = sum(score * weights[component] 
                               for component, score in scores.items())
            
            # Scale to percentage and round to 2 decimals
            final_score = round(momentum_score * 100, 2)
            
            # Log components for debugging
            self._log_momentum_components(scores, final_score)
            
            return final_score
            
        except Exception as e:
            logger.error(f"Error calculating momentum: {str(e)}")
            logger.debug(f"Input data structure: {data}")
            return 0.0
            
    def _prepare_dataframe(self, data: Union[Dict, pd.DataFrame]) -> Optional[pd.DataFrame]:
        """Prepare DataFrame from input data."""
        if isinstance(data, pd.DataFrame):
            return data.copy()
            
        if isinstance(data, dict):
            for key in ['ohlc', 'market_data', 'data']:
                if key in data:
                    df_data = data[key] if key != 'market_data' else data[key].get('ohlc')
                    if df_data is not None:
                        return pd.DataFrame(df_data).copy()
        
        logger.error(f"Could not find valid OHLCV data. Input type: {type(data)}")
        return None
    
    def calculate_volatility(self, df: pd.DataFrame) -> float:
        """Calculate current volatility using ATR-based method.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            float: Volatility value. Returns 0 if calculation fails.
        """
        try:
            if len(df) < 12:
                logger.warning(f"Insufficient data for volatility calculation: {len(df)} bars available, 12 required")
                return 0.0
                
            # Calculate ATR-based volatility using last 12 bars (or available data)
            n_bars = min(12, len(df))
            
            # Get required data
            high = df['high'].values[-n_bars:]
            low = df['low'].values[-n_bars:]
            close = df['close'].values[-n_bars:]
            
            # Create previous close array (shifted by 1 period)
            prev_close = np.zeros(n_bars)
            prev_close[1:] = close[:-1]
            
            # True Range calculation
            tr = np.zeros(n_bars)
            for i in range(1, n_bars):  # Skip first row as it doesn't have prev_close
                tr[i] = max(
                    high[i] - low[i],                 # Current high - low
                    abs(high[i] - prev_close[i]),     # Current high - previous close
                    abs(low[i] - prev_close[i])       # Current low - previous close
                )
            
            # Calculate ATR (average true range)
            atr = np.mean(tr[1:])  # Skip the first element (it's zero)
            
            # Normalize by current price
            if close[-1] > 0:
                volatility = atr / close[-1]
            else:
                logger.warning("Cannot normalize ATR: current price is zero or negative")
                return 0.0
            
            logger.debug(f"Calculated volatility: {volatility:.4f}")
            return volatility
            
        except Exception as e:
            logger.error(f"Error calculating volatility: {str(e)}")
            return 0.0
        
    def _validate_momentum_data(self, df: pd.DataFrame) -> bool:
        """Validate DataFrame for momentum calculation."""
        if df.empty:
            logger.warning("Empty DataFrame in momentum calculation")
            return False
            
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}")
            return False
            
        # Convert columns to numeric
        for col in required_columns:
            df.loc[:, col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)
        
        if len(df) < 20:
            logger.warning(f"Insufficient data: {len(df)} rows")
            return False
            
        return True
        
    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate technical indicators for momentum."""
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        
        # Price momentum
        price_changes = {
            5: (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5],
            10: (df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10],
            20: (df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20]
        }
        
        # Volume momentum
        volume_sma = {
            'short': df['volume'].rolling(window=10).mean(),
            'long': df['volume'].rolling(window=20).mean()
        }
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        
        # Moving averages
        ma20 = df['close'].rolling(window=20).mean()
        ma50 = df['close'].rolling(window=50).mean()
        
        return {
            'rsi': rsi.iloc[-1],
            'price_changes': price_changes,
            'volume_sma': volume_sma,
            'macd': macd.iloc[-1],
            'signal': signal.iloc[-1],
            'ma20': ma20.iloc[-1],
            'ma50': ma50.iloc[-1]
        }
        
    def _calculate_component_scores(self, indicators: Dict) -> Dict[str, float]:
        """Calculate individual component scores for momentum."""
        # RSI score (-1 to 1 scale)
        rsi_score = (indicators['rsi'] - 50) / 50
        
        # Price momentum score
        price_score = (
            0.5 * indicators['price_changes'][5] +
            0.3 * indicators['price_changes'][10] +
            0.2 * indicators['price_changes'][20]
        ) * 10
        
        # Volume momentum score
        volume_score = (
            (indicators['volume_sma']['short'].iloc[-1] -
             indicators['volume_sma']['long'].iloc[-1]) /
            indicators['volume_sma']['long'].iloc[-1]
        )
        
        # MACD score
        macd_score = (indicators['macd'] - indicators['signal']) / abs(indicators['macd'])
        
        # Moving average trend score
        ma_score = (
            (indicators['ma20'] - indicators['ma50']) /
            indicators['ma50'] * 5
        )
        
        return {
            'rsi': rsi_score,
            'price': price_score,
            'volume': volume_score,
            'macd': macd_score,
            'ma': ma_score
        }
        
    def _log_momentum_components(self, scores: Dict[str, float], final_score: float) -> None:
        """Log momentum calculation components for debugging."""
        logger.debug("Momentum Components:")
        for component, score in scores.items():
            logger.debug(f"    {component.upper()} Score: {score:.4f}")
        logger.debug(f"    Final Score: {final_score:.2f}")
            
    def _assess_volume_data_quality(self, df: pd.DataFrame) -> float:
        """Assess the quality of volume data."""
        try:
            if 'volume' not in df.columns:
                return 0
                
            # Check for missing values
            missing_ratio = df['volume'].isna().mean()
            if missing_ratio > 0.2:  # Increased from 0.1
                return 0
                
            # Check for zero values
            zero_ratio = (df['volume'] == 0).mean()
            if zero_ratio > 0.3:  # Increased from 0.2
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
            
            # Create a copy to avoid warnings
            df = df.copy()
            
            # Remove NaN and Inf values if any
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)
            
            # Volume Moving Average
            df['volume_sma'] = df['volume'].rolling(window=self.volume_ma_period).mean()
            
            # Volume Weighted Average Price (VWAP)
            try:
                typical_price = (df['high'] + df['low'] + df['close']) / 3
                df['vwap'] = (df['volume'] * typical_price).cumsum() / df['volume'].cumsum()
            except Exception as e:
                logger.warning(f"Error calculating VWAP: {str(e)}")
                df['vwap'] = np.nan
            
            # Relative Volume with safety
            df['relative_volume'] = np.nan  # Initialize with NaN
            mask = df['volume_sma'] > 0     # Only calculate where divisor > 0
            df.loc[mask, 'relative_volume'] = df.loc[mask, 'volume'] / df.loc[mask, 'volume_sma']
            
            # Up/Down Volume
            df['up_volume'] = df['volume'].where(df['close'] > df['open'], 0)
            df['down_volume'] = df['volume'].where(df['close'] < df['open'], 0)
            
            # Volume Force with protection
            price_change = df['close'] - df['open']
            df['volume_force'] = price_change * df['volume']
            
            # On-Balance Volume (OBV)
            try:
                # Initialize OBV column
                df['obv'] = 0
                
                # Calculate OBV using vectorized operations rather than .loc slicing
                if len(df) > 1:
                    # Get closing prices and volumes as numpy arrays
                    close_prices = df['close'].values
                    volumes = df['volume'].values
                    
                    # Create OBV values array
                    obv_values = np.zeros(len(df))
                    
                    # Calculate OBV changes
                    for i in range(1, len(df)):
                        if close_prices[i] > close_prices[i-1]:
                            obv_values[i] = obv_values[i-1] + volumes[i]
                        elif close_prices[i] < close_prices[i-1]:
                            obv_values[i] = obv_values[i-1] - volumes[i]
                        else:
                            obv_values[i] = obv_values[i-1]
                    
                    # Assign to dataframe
                    df['obv'] = obv_values
                
                # OBV EMA for trend
                df['obv_ema'] = df['obv'].ewm(span=20, adjust=False).mean()
                
                # OBV trend determination
                df['obv_trend'] = 'neutral'
                rising_mask = df['obv'] > df['obv_ema'] * 1.02  # 2% buffer
                falling_mask = df['obv'] < df['obv_ema'] * 0.98  # 2% buffer
                df.loc[rising_mask, 'obv_trend'] = 'bullish'
                df.loc[falling_mask, 'obv_trend'] = 'bearish'
            except Exception as e:
                logger.warning(f"Error calculating OBV: {str(e)}")
                df['obv'] = 0
                df['obv_ema'] = 0
                df['obv_trend'] = 'neutral'
            
            # Fill NaN values for indicators
            indicators = ['volume_sma', 'vwap', 'relative_volume', 'up_volume', 
                         'down_volume', 'volume_force', 'obv', 'obv_ema']
            for indicator in indicators:
                if indicator in df.columns and df[indicator].isna().any():
                    logger.debug(f"Filling NaN values in {indicator}")
                    # Fix deprecated fillna method warnings
                    df[indicator] = df[indicator].ffill()
                    df[indicator] = df[indicator].fillna(0)  # Fill any remaining NaNs
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating volume indicators: {str(e)}")
            return df
    
    def _calculate_volume_profile(self, df: pd.DataFrame) -> Dict:
        """Calculate Volume Profile (TPO)."""
        try:
            # Check for empty dataframe or invalid data
            if df.empty or 'volume' not in df.columns:
                logger.warning("Cannot calculate volume profile: empty dataframe or missing volume data")
                return {
                    'poc': None,
                    'value_area': {'high': None, 'low': None},
                    'levels': {}
                }
                
            # Check if we have price data
            if 'high' not in df.columns or 'low' not in df.columns:
                logger.warning("Cannot calculate volume profile: missing price data")
                return {
                    'poc': None,
                    'value_area': {'high': None, 'low': None},
                    'levels': {}
                }
                
            # Calculate price range
            price_high = df['high'].max()
            price_low = df['low'].min()
            
            # Check for valid price range
            if price_high <= price_low or not np.isfinite(price_high) or not np.isfinite(price_low):
                logger.warning("Cannot calculate volume profile: invalid price range")
                return {
                    'poc': None,
                    'value_area': {'high': None, 'low': None},
                    'levels': {}
                }
                
            price_range = price_high - price_low
            level_height = price_range / self.profile_levels
            
            # Check if level_height is too small
            if level_height <= 0 or not np.isfinite(level_height):
                logger.warning("Cannot calculate volume profile: level height is zero or invalid")
                return {
                    'poc': None,
                    'value_area': {'high': None, 'low': None},
                    'levels': {}
                }
            
            levels = {}
            total_volume = 0
            
            # Calculate volume at each price level
            for i in range(self.profile_levels):
                level_price = price_low + (i * level_height)
                
                # Get candles where price range overlaps with this level
                mask = ((df['low'] <= level_price + level_height) & 
                        (df['high'] >= level_price))
                
                # Sum volume for those candles
                level_volume = df.loc[mask, 'volume'].sum()
                
                levels[level_price] = level_volume
                total_volume += level_volume
            
            # Handle case where no volume was distributed
            if total_volume <= 0:
                logger.warning("Cannot calculate volume profile: no volume distributed to levels")
                return {
                    'poc': None,
                    'value_area': {'high': None, 'low': None},
                    'levels': {}
                }
                
            # Find Point of Control (POC)
            poc_price = max(levels.items(), key=lambda x: x[1])[0]
            
            # Calculate Value Area (70% of total volume)
            value_area_volume = total_volume * 0.7
            cumulative_volume = 0
            value_area = {'high': poc_price, 'low': poc_price}
            
            # Expand value area above and below POC
            above_poc = {k: v for k, v in levels.items() if k > poc_price}
            below_poc = {k: v for k, v in levels.items() if k < poc_price}
            
            # Sort above and below levels by their volume (highest to lowest)
            sorted_above = sorted(above_poc.items(), key=lambda x: x[1], reverse=True)
            sorted_below = sorted(below_poc.items(), key=lambda x: x[1], reverse=True)
            
            # Alternate adding high and low levels until we reach value area volume
            i_above = 0
            i_below = 0
            
            while cumulative_volume < value_area_volume:
                # Check if we have exhausted levels in either direction
                if i_above >= len(sorted_above) and i_below >= len(sorted_below):
                    break
                    
                # Determine which direction to add next
                add_above = False
                
                if i_above < len(sorted_above) and i_below < len(sorted_below):
                    # Both directions available, compare volumes
                    add_above = sorted_above[i_above][1] >= sorted_below[i_below][1]
                elif i_above < len(sorted_above):
                    # Only above levels remain
                    add_above = True
                    
                # Add the selected level
                if add_above:
                    price, volume = sorted_above[i_above]
                    cumulative_volume += volume
                    value_area['high'] = max(value_area['high'], price)
                    i_above += 1
                else:
                    price, volume = sorted_below[i_below]
                    cumulative_volume += volume
                    value_area['low'] = min(value_area['low'], price)
                    i_below += 1
            
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
            logger.info("Starting volume pattern detection...")
            patterns = []
            
            # Check if required columns exist
            required_columns = ['volume', 'volume_sma', 'close', 'open']
            if not all(col in df.columns for col in required_columns):
                logger.warning("Missing required columns for volume pattern detection")
                return []
            
            logger.debug(f"Analyzing {len(df)} candles for patterns...")
            pattern_stats = {'volume_climax': 0, 'volume_dryup': 0, 'accumulation': 0, 'distribution': 0}
            
            for i in range(3, len(df)):
                window = df.iloc[i-3:i+1]
                current_volume = df['volume'].iloc[i]
                current_sma = df['volume_sma'].iloc[i]
                
                # Volume Climax (high volume with price reversal)
                if current_volume > current_sma * 2:
                    logger.debug(f"High volume detected at index {i}: {current_volume:.2f} (SMA: {current_sma:.2f})")
                    
                    if df['close'].iloc[i] < df['open'].iloc[i] and \
                       df['close'].iloc[i-1] > df['open'].iloc[i-1]:
                        logger.debug(f"Bearish volume climax at index {i}")
                        pattern_stats['volume_climax'] += 1
                        patterns.append({
                            'type': 'volume_climax',
                            'direction': 'bearish',
                            'index': i,
                            'volume': current_volume,
                            'price': df['close'].iloc[i]
                        })
                    elif df['close'].iloc[i] > df['open'].iloc[i] and \
                         df['close'].iloc[i-1] < df['open'].iloc[i-1]:
                        logger.debug(f"Bullish volume climax at index {i}")
                        pattern_stats['volume_climax'] += 1
                        patterns.append({
                            'type': 'volume_climax',
                            'direction': 'bullish',
                            'index': i,
                            'volume': current_volume,
                            'price': df['close'].iloc[i]
                        })
                
                # Volume Dry-up (low volume after trend)
                if current_volume < current_sma * 0.5:
                    logger.debug(f"Low volume detected at index {i}: {current_volume:.2f} (SMA: {current_sma:.2f})")
                    
                    # After uptrend
                    if all(df['close'].iloc[i-j] > df['close'].iloc[i-j-1] for j in range(3)):
                        logger.debug(f"Bearish volume dry-up after uptrend at index {i}")
                        pattern_stats['volume_dryup'] += 1
                        patterns.append({
                            'type': 'volume_dryup',
                            'direction': 'bearish',
                            'index': i,
                            'volume': current_volume,
                            'price': df['close'].iloc[i]
                        })
                    # After downtrend
                    elif all(df['close'].iloc[i-j] < df['close'].iloc[i-j-1] for j in range(3)):
                        logger.debug(f"Bullish volume dry-up after downtrend at index {i}")
                        pattern_stats['volume_dryup'] += 1
                        patterns.append({
                            'type': 'volume_dryup',
                            'direction': 'bullish',
                            'index': i,
                            'volume': current_volume,
                            'price': df['close'].iloc[i]
                        })
                
                # Smart Money Accumulation/Distribution
                if current_volume > current_sma * 1.5:
                    try:
                        price_change = abs(df['close'].iloc[i] - df['open'].iloc[i])
                        avg_price_change = abs(df['close'] - df['open']).rolling(20).mean().iloc[i]
                        
                        logger.debug(f"Price change at index {i}: {price_change:.5f} (Avg: {avg_price_change:.5f})")
                        
                        # Accumulation (high volume, small price movement)
                        if price_change < avg_price_change * 0.5:
                            logger.debug(f"Accumulation pattern at index {i}")
                            pattern_stats['accumulation'] += 1
                            patterns.append({
                                'type': 'accumulation',
                                'direction': 'bullish',  # Accumulation indicates bullish sentiment
                                'index': i,
                                'volume': current_volume,
                                'price': df['close'].iloc[i]
                            })
                        # Distribution (high volume, large price movement)
                        elif price_change > avg_price_change * 2:
                            logger.debug(f"Distribution pattern at index {i}")
                            pattern_stats['distribution'] += 1
                            patterns.append({
                                'type': 'distribution',
                                'direction': 'bearish',  # Distribution indicates bearish sentiment
                                'index': i,
                                'volume': current_volume,
                                'price': df['close'].iloc[i]
                            })
                    except Exception as e:
                        logger.debug(f"Error calculating price changes: {str(e)}")
                        continue
            
            logger.info("Pattern detection complete. Statistics:")
            for pattern_type, count in pattern_stats.items():
                logger.info(f"- {pattern_type}: {count} occurrences")
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting volume patterns: {str(e)}")
            return []

    def _validate_patterns(self, df: pd.DataFrame, patterns: List[Dict]) -> List[Dict]:
        """Validate detected patterns against price action."""
        try:
            logger.info("Starting pattern validation...")
            validated_patterns = []
            
            for pattern in patterns:
                i = pattern['index']
                is_valid = True
                validation_reason = ""
                
                # Skip if we don't have enough bars after the pattern
                if i >= len(df) - 3:
                    continue
                
                # Get price action context
                pre_pattern = df.iloc[i-3:i]
                post_pattern = df.iloc[i:i+3]
                
                pattern_type = pattern['type']
                
                if pattern_type == 'volume_climax':
                    # Validate volume climax
                    avg_volume = df['volume'].iloc[i-5:i].mean()
                    if pattern['volume'] < avg_volume * 2:
                        is_valid = False
                        validation_reason = "Volume not significantly higher than previous average"
                    
                    # Check for price follow-through
                    if pattern.get('direction') == 'bullish':
                        if not all(post_pattern['close'] > post_pattern['open']):
                            is_valid = False
                            validation_reason = "No bullish follow-through after climax"
                    elif pattern.get('direction') == 'bearish':
                        if not all(post_pattern['close'] < post_pattern['open']):
                            is_valid = False
                            validation_reason = "No bearish follow-through after climax"
                
                elif pattern_type == 'volume_dryup':
                    # Validate volume dry-up
                    if not all(df['volume'].iloc[i-2:i+1] < df['volume_sma'].iloc[i-2:i+1] * 0.5):
                        is_valid = False
                        validation_reason = "Volume not consistently low enough"
                    
                    # Check for trend reversal
                    pre_trend = all(pre_pattern['close'].diff() > 0) if pattern.get('direction') == 'bearish' else all(pre_pattern['close'].diff() < 0)
                    if not pre_trend:
                        is_valid = False
                        validation_reason = "No clear trend before dry-up"
                
                elif pattern_type in ['accumulation', 'distribution']:
                    # Validate accumulation/distribution
                    price_range = (df['high'].iloc[i] - df['low'].iloc[i]) / df['close'].iloc[i]
                    avg_range = ((df['high'] - df['low']) / df['close']).rolling(20).mean().iloc[i]
                    
                    if pattern_type == 'accumulation' and price_range > avg_range * 0.7:
                        is_valid = False
                        validation_reason = "Price range too large for accumulation"
                    elif pattern_type == 'distribution' and price_range < avg_range * 1.5:
                        is_valid = False
                        validation_reason = "Price range too small for distribution"
                
                if is_valid:
                    validated_patterns.append(pattern)
                    logger.debug(f"Validated {pattern_type} pattern at index {i}")
                else:
                    logger.debug(f"Rejected {pattern_type} pattern at index {i}: {validation_reason}")
            
            logger.info(f"Pattern validation complete. {len(validated_patterns)}/{len(patterns)} patterns validated")
            return validated_patterns
            
        except Exception as e:
            logger.error(f"Error validating patterns: {str(e)}")
            return patterns 

    def _analyze_indicator_relationships(self, df: pd.DataFrame) -> Dict:
        """Analyze relationships between different indicators."""
        try:
            logger.info("Analyzing indicator relationships...")
            relationships = {}
            
            # Volume vs Price correlation
            price_changes = df['close'].pct_change()
            volume_changes = df['volume'].pct_change()
            vol_price_corr = price_changes.corr(volume_changes)
            relationships['volume_price_correlation'] = vol_price_corr
            logger.debug(f"Volume-Price correlation: {vol_price_corr:.3f}")
            
            # Volume trend analysis
            current_vol = df['volume'].iloc[-1]
            vol_sma = df['volume_sma'].iloc[-1]
            vol_trend = 'increasing' if current_vol > vol_sma * 1.1 else 'decreasing' if current_vol < vol_sma * 0.9 else 'neutral'
            relationships['volume_trend'] = vol_trend
            logger.debug(f"Volume trend: {vol_trend} (Current: {current_vol:.2f}, SMA: {vol_sma:.2f})")
            
            # Relative volume analysis
            rel_vol = df['relative_volume'].iloc[-1]
            vol_strength = 'high' if rel_vol > 1.5 else 'low' if rel_vol < 0.5 else 'normal'
            relationships['volume_strength'] = vol_strength
            logger.debug(f"Volume strength: {vol_strength} (Relative: {rel_vol:.2f})")
            
            # Price range vs volume relationship
            price_ranges = (df['high'] - df['low']) / df['close']
            vol_range_corr = price_ranges.corr(df['volume'])
            relationships['range_volume_correlation'] = vol_range_corr
            logger.debug(f"Range-Volume correlation: {vol_range_corr:.3f}")
            
            # Buying/Selling pressure
            up_vol_ratio = df['up_volume'].sum() / df['volume'].sum()
            relationships['buying_pressure'] = up_vol_ratio
            logger.debug(f"Buying pressure ratio: {up_vol_ratio:.3f}")
            
            # Volume consistency
            vol_std = df['volume'].std()
            vol_mean = df['volume'].mean()
            vol_cv = vol_std / vol_mean  # Coefficient of variation
            relationships['volume_consistency'] = 'consistent' if vol_cv < 0.5 else 'volatile'
            logger.debug(f"Volume consistency: {relationships['volume_consistency']} (CV: {vol_cv:.3f})")
            
            # VWAP relationship
            vwap = df['vwap'].iloc[-1]
            close = df['close'].iloc[-1]
            vwap_position = 'above' if close > vwap else 'below'
            relationships['price_to_vwap'] = vwap_position
            logger.debug(f"Price to VWAP: {vwap_position} (Price: {close:.5f}, VWAP: {vwap:.5f})")
            
            logger.info("Indicator relationship analysis complete")
            return relationships
            
        except Exception as e:
            logger.error(f"Error analyzing indicator relationships: {str(e)}")
            return {} 

    def _calculate_comprehensive_momentum(self, df: pd.DataFrame) -> float:
        """Calculate comprehensive momentum score based on multiple factors."""
        try:
            # Check if we have sufficient data
            if len(df) < 50:  # Need at least 50 bars for reliable calculation
                logger.warning(f"Insufficient data for momentum calculation: {len(df)} bars available")
                if len(df) >= 20:  # Can still calculate with limited data
                    logger.info("Computing momentum with limited data")
                else:
                    logger.error("Cannot calculate momentum: minimum 20 bars required")
                    return 0.0
            
            # Calculate RSI with protection against division by zero
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            
            # Safe division for RSI calculation
            with np.errstate(divide='ignore', invalid='ignore'):
                rs = gain / loss
            rs = np.where(np.isnan(rs) | np.isinf(rs), 100, rs)  # Handle inf/nan
            rsi = 100 - (100 / (1 + rs))
            rsi = pd.Series(rsi, index=delta.index)
            
            # Calculate price momentum over multiple periods with safety
            lookback_periods = [5, 10, 20]
            price_momentums = {}
            
            for period in lookback_periods:
                if len(df) > period:
                    if df['close'].iloc[-period] != 0:
                        price_momentums[period] = (df['close'].iloc[-1] - df['close'].iloc[-period]) / df['close'].iloc[-period] * 100
                    else:
                        price_momentums[period] = 0
                else:
                    price_momentums[period] = 0
                    
            # Calculate volume momentum safely
            try:
                volume_sma_short = df['volume'].rolling(window=min(10, len(df)-1)).mean()
                volume_sma_long = df['volume'].rolling(window=min(20, len(df)-1)).mean()
                
                if volume_sma_long.iloc[-1] > 0:
                    volume_momentum = ((volume_sma_short.iloc[-1] - volume_sma_long.iloc[-1]) / volume_sma_long.iloc[-1]) * 100
                else:
                    volume_momentum = 0
            except Exception as e:
                logger.warning(f"Error calculating volume momentum: {str(e)}")
                volume_momentum = 0
            
            # Calculate MACD for trend confirmation
            try:
                exp1 = df['close'].ewm(span=12, adjust=False).mean()
                exp2 = df['close'].ewm(span=26, adjust=False).mean()
                macd = exp1 - exp2
                signal = macd.ewm(span=9, adjust=False).mean()
                
                # Safe calculation of MACD momentum
                macd_mean = abs(macd).mean()
                if macd_mean > 0:
                    macd_momentum = (macd.iloc[-1] - signal.iloc[-1]) / macd_mean
                else:
                    macd_momentum = 0
            except Exception as e:
                logger.warning(f"Error calculating MACD: {str(e)}")
                macd_momentum = 0
            
            # Calculate moving average trend safely
            try:
                ma_len = min(50, len(df)-1)
                ma20 = df['close'].rolling(window=min(20, ma_len)).mean()
                ma50 = df['close'].rolling(window=ma_len).mean()
                
                if ma50.iloc[-1] > 0:
                    ma_trend = (ma20.iloc[-1] - ma50.iloc[-1]) / ma50.iloc[-1] * 100
                else:
                    ma_trend = 0
            except Exception as e:
                logger.warning(f"Error calculating MA trend: {str(e)}")
                ma_trend = 0
            
            # Normalize and combine all factors with weighted importance
            try:
                # RSI score (-1 to 1 scale)
                rsi_score = min(max((rsi.iloc[-1] - 50) / 50, -1), 1)
                
                # Price score combines different timeframes
                price_score = min(max(
                    (0.5 * price_momentums.get(5, 0) + 
                     0.3 * price_momentums.get(10, 0) + 
                     0.2 * price_momentums.get(20, 0)) / 10,
                    -1), 1)
                
                # Volume score (capped to prevent extreme values)
                volume_score = min(max(volume_momentum / 100, -1), 1)
                
                # MACD score (capped to prevent extreme values)
                macd_score = min(max(macd_momentum, -1), 1)
                
                # MA score (capped to prevent extreme values)
                ma_score = min(max(ma_trend / 5, -1), 1)
            except Exception as e:
                logger.warning(f"Error normalizing component scores: {str(e)}")
                return 0.0
            
            # Weighted combination with emphasis on recent price action and volume
            momentum_score = (
                0.25 * rsi_score +      # RSI weight
                0.30 * price_score +    # Recent price action weight
                0.20 * volume_score +   # Volume trend weight
                0.15 * macd_score +     # MACD confirmation weight
                0.10 * ma_score         # Moving average trend weight
            )
            
            # Scale to percentage and round to 2 decimals
            final_score = round(momentum_score * 100, 2)
            
            # Log component contributions for debugging
            logger.debug(f"Momentum components: RSI={rsi_score:.2f}, Price={price_score:.2f}, "
                         f"Volume={volume_score:.2f}, MACD={macd_score:.2f}, MA={ma_score:.2f}")
            logger.debug(f"Final momentum score: {final_score}")
            
            return final_score
            
        except Exception as e:
            logger.error(f"Error calculating comprehensive momentum: {str(e)}")
            return 0.0
        
    def analyze_volume_trend(self, df: pd.DataFrame, momentum_score: float) -> Dict[str, Any]:
        """Analyze volume trend with enhanced accuracy.
        
        Args:
            df: DataFrame with OHLCV data
            momentum_score: Current momentum score
            
        Returns:
            Dict containing trend analysis results:
                - trend: str ('bullish', 'bearish', or 'neutral')
                - strength: float (0-1)
                - signals: List[str] (reasons for the trend)
        """
        try:
            # Calculate volume moving averages
            volume_sma_10 = df['volume'].rolling(window=10).mean()
            volume_sma_20 = df['volume'].rolling(window=20).mean()
            
            # Get recent volume metrics
            recent_volume_avg = df['volume'].tail(5).mean()
            baseline_volume_avg = df['volume'].tail(20).mean()
            volume_ratio = recent_volume_avg / baseline_volume_avg if baseline_volume_avg > 0 else 1.0
            
            # Calculate price direction and range
            price_direction = df['close'].iloc[-1] > df['close'].iloc[-5]
            price_range = (df['high'] - df['low']) / df['close']
            avg_range = price_range.rolling(20).mean().iloc[-1]
            current_range = price_range.iloc[-1]
            
            # Initialize trend analysis
            signals = []
            trend = 'neutral'
            strength = 0.0
            
            # Volume trend analysis
            if volume_sma_10.iloc[-1] > volume_sma_20.iloc[-1] * 1.1:
                signals.append("Short-term volume above long-term average")
                strength += 0.2
            elif volume_sma_10.iloc[-1] < volume_sma_20.iloc[-1] * 0.9:
                signals.append("Short-term volume below long-term average")
                strength -= 0.2
                
            # Volume ratio analysis
            if volume_ratio > 1.2:
                signals.append("Recent volume significantly above average")
                strength += 0.3 if price_direction else -0.3
            elif volume_ratio < 0.8:
                signals.append("Recent volume significantly below average")
                strength -= 0.2
                
            # Price range analysis
            if current_range > avg_range * 1.5:
                signals.append("Increased price volatility")
                strength = strength * 1.2 if abs(strength) > 0 else strength
                
            # Momentum confirmation
            if abs(momentum_score) >= 15:
                if momentum_score > 0 and strength > 0:
                    signals.append("Positive momentum confirmation")
                    strength += 0.2
                elif momentum_score < 0 and strength < 0:
                    signals.append("Negative momentum confirmation")
                    strength -= 0.2
                    
            # Determine final trend
            if strength > 0.3:
                trend = 'bullish'
            elif strength < -0.3:
                trend = 'bearish'
                
            return {
                'trend': trend,
                'strength': abs(strength),
                'signals': signals
            }
            
        except Exception as e:
            logger.error(f"Error analyzing volume trend: {str(e)}")
            return {
                'trend': 'neutral',
                'strength': 0.0,
                'signals': []
            }

    def weighted_volume_distribution(self, df: pd.DataFrame, num_levels: int = 50) -> Dict:
        """Calculate weighted volume distribution with more accurate volume attribution.
        
        This method distributes volume across price levels proportional to the price overlap
        between each candle and the price level, instead of attributing all volume to any
        overlapping level.
        
        Args:
            df: DataFrame with OHLCV data
            num_levels: Number of price levels to divide the range into
            
        Returns:
            Dict containing volume distribution data
        """
        try:
            # Check for empty dataframe or invalid data
            if df.empty or 'volume' not in df.columns:
                logger.warning("Cannot calculate volume distribution: empty dataframe or missing volume data")
                return {
                    'levels': {},
                    'poc': None,
                    'high_volume_nodes': []
                }
                
            # Check if we have price data
            required_cols = ['high', 'low', 'volume']
            if not all(col in df.columns for col in required_cols):
                logger.warning(f"Missing columns for volume distribution: {[c for c in required_cols if c not in df.columns]}")
                return {
                    'levels': {},
                    'poc': None,
                    'high_volume_nodes': []
                }
                
            # Calculate price range and level height
            price_high = df['high'].max()
            price_low = df['low'].min()
            
            if price_high <= price_low or not np.isfinite(price_high) or not np.isfinite(price_low):
                logger.warning("Invalid price range for volume distribution")
                return {
                    'levels': {},
                    'poc': None,
                    'high_volume_nodes': []
                }
                
            price_range = price_high - price_low
            level_height = price_range / num_levels if price_range > 0 else 1.0
            
            # Create levels
            levels = {}
            level_prices = [price_low + i * level_height for i in range(num_levels + 1)]
            
            # Initialize volume at each level
            for i in range(num_levels):
                levels[level_prices[i]] = 0.0
                
            # Distribute volume across levels proportionally
            for idx, row in df.iterrows():
                candle_high = row['high']
                candle_low = row['low']
                candle_volume = row['volume']
                
                # Skip candles with zero volume or invalid range
                if candle_volume <= 0 or candle_high <= candle_low:
                    continue
                    
                candle_range = candle_high - candle_low
                
                # Find levels that overlap with this candle
                for i in range(num_levels):
                    level_low = level_prices[i]
                    level_high = level_prices[i + 1]
                    
                    # Calculate overlap between candle and level
                    overlap_low = max(candle_low, level_low)
                    overlap_high = min(candle_high, level_high)
                    
                    if overlap_high > overlap_low:  # There is an overlap
                        # Calculate proportion of candle in this level
                        overlap_range = overlap_high - overlap_low
                        proportion = overlap_range / candle_range
                        
                        # Distribute volume proportionally
                        levels[level_low] += candle_volume * proportion
            
            # Find point of control (POC) - price level with highest volume
            if not levels:
                return {
                    'levels': {},
                    'poc': None,
                    'high_volume_nodes': []
                }
                
            poc = max(levels.items(), key=lambda x: x[1])[0]
            
            # Find high volume nodes (significant volume levels)
            avg_volume = sum(levels.values()) / len(levels) if levels else 0
            high_volume_nodes = [
                {'price': price, 'volume': volume}
                for price, volume in levels.items()
                if volume > avg_volume * 1.5  # 50% above average
            ]
            
            # Sort by volume (highest first)
            high_volume_nodes.sort(key=lambda x: x['volume'], reverse=True)
            
            return {
                'levels': levels,
                'poc': poc,
                'high_volume_nodes': high_volume_nodes[:5]  # Top 5 high volume nodes
            }
            
        except Exception as e:
            logger.error(f"Error calculating weighted volume distribution: {str(e)}")
            return {
                'levels': {},
                'poc': None,
                'high_volume_nodes': []
            }