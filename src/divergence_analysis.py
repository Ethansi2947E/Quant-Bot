from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from loguru import logger

class DivergenceAnalysis:
    def __init__(self):
        self.lookback_period = 30  # Increased from 20 to get more context
        self.divergence_threshold = 0.005  # 0.5% threshold
        self.hidden_divergence_threshold = 0.2  
        self.momentum_divergence_threshold = 0.2  
        self.min_swing_size = 0.0003  # Decreased to be more sensitive to price movements
        self.min_rsi_swing = 5.0     # Minimum RSI swing size
        self.confirmation_bars = 2    
        self.min_data_points = 50
        
        # New parameters for dynamic threshold adjustment
        self.use_dynamic_thresholds = True
        self.volatility_adjustment_factor = 1.5
        self.min_volatility_threshold = 0.0001
        self.max_volatility_threshold = 0.05
    
    def _calculate_dynamic_thresholds(self, df: pd.DataFrame) -> None:
        """Calculate dynamic thresholds based on market volatility."""
        try:
            if not self.use_dynamic_thresholds or df is None or len(df) < 30:
                logger.info("Using static thresholds (dynamic thresholds disabled or insufficient data)")
                return
            
            # Calculate recent volatility using ATR (Average True Range)
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift(1))
            low_close = abs(df['low'] - df['close'].shift(1))
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr = true_range.rolling(14).mean().iloc[-1]
            
            # Calculate price volatility as percentage
            close_volatility = df['close'].pct_change().abs().rolling(14).std().iloc[-1]
            
            # Calculate RSI volatility
            if 'rsi' in df.columns:
                rsi_volatility = df['rsi'].diff().abs().rolling(14).std().iloc[-1]
            else:
                rsi_volatility = 5.0  # Default if RSI not available
            
            # Adjust thresholds based on volatility
            volatility_factor = max(0.5, min(2.0, (close_volatility / 0.01) * self.volatility_adjustment_factor))
            
            logger.info(f"Market volatility: ATR={atr:.6f}, Close={close_volatility:.6f}, RSI={rsi_volatility:.2f}")
            logger.info(f"Volatility adjustment factor: {volatility_factor:.2f}")
            
            # Update thresholds based on volatility
            self.min_swing_size = max(
                self.min_volatility_threshold,
                min(self.max_volatility_threshold, 0.0003 * volatility_factor)
            )
            self.min_rsi_swing = max(2.0, min(10.0, 5.0 * volatility_factor))
            self.divergence_threshold = max(0.001, min(0.02, 0.005 * volatility_factor))
            self.hidden_divergence_threshold = max(0.05, min(0.5, 0.2 * volatility_factor))
            self.momentum_divergence_threshold = max(0.05, min(0.5, 0.2 * volatility_factor))
            
            logger.info(f"Dynamic thresholds adjusted - min_swing_size: {self.min_swing_size:.6f}, min_rsi_swing: {self.min_rsi_swing:.2f}")
            logger.info(f"Divergence thresholds - regular: {self.divergence_threshold:.4f}, hidden: {self.hidden_divergence_threshold:.4f}, momentum: {self.momentum_divergence_threshold:.4f}")
            
        except Exception as e:
            logger.error(f"Error calculating dynamic thresholds: {str(e)}")
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        """Analyze multiple types of divergences with enhanced logging and validation."""
        try:
            if df is None or df.empty:
                logger.warning("Empty DataFrame provided for divergence analysis")
                return {}
                
            if len(df) < self.min_data_points:
                logger.warning(f"Insufficient data points for divergence analysis. Required: {self.min_data_points}, Got: {len(df)}")
                return {}
            
            logger.info("Starting divergence analysis...")
            
            # Create a copy of the DataFrame to prevent SettingWithCopyWarning
            df = df.copy()
            
            # Calculate indicators
            logger.debug("Calculating technical indicators...")
            df = self._calculate_indicators(df)
            
            # Calculate dynamic thresholds based on market volatility
            self._calculate_dynamic_thresholds(df)
            
            # Normalize RSI and MFI values to ensure they're within bounds
            if 'rsi' in df.columns:
                df['rsi'] = df['rsi'].clip(0, 100)
            if 'mfi' in df.columns:
                df['mfi'] = df['mfi'].clip(0, 100)
            
            # Add structural RSI if not present
            if 'structural_rsi' not in df.columns and 'rsi' in df.columns:
                df['structural_rsi'] = df['rsi']
            
            # Analyze indicator quality
            logger.debug("Analyzing indicator quality...")
            indicator_quality = self._assess_indicator_quality(df)
            if indicator_quality['quality_score'] < 0.7:
                logger.warning(f"Low indicator quality: {indicator_quality['reason']}")
            
            # Calculate lookback period
            max_lookback = len(df) - self.min_data_points
            lookback_period = min(self.lookback_period, max_lookback)
            
            # Create analysis window with proper indexing
            analysis_df = df.iloc[-lookback_period:].copy()
            analysis_df = analysis_df.reset_index(drop=True)  # Reset index for proper indexing
            
            # Find regular divergences
            logger.debug("Searching for regular divergences...")
            regular = self._find_regular_divergences(analysis_df)
            logger.info(f"Found {len(regular['bullish'])} bullish and {len(regular['bearish'])} bearish regular divergences")
            
            # Find hidden divergences
            logger.debug("Searching for hidden divergences...")
            hidden = self._find_hidden_divergences(analysis_df)
            logger.info(f"Found {len(hidden['bullish'])} bullish and {len(hidden['bearish'])} bearish hidden divergences")
            
            # Find structural divergences
            logger.debug("Searching for structural divergences...")
            structural = self._find_structural_divergences(analysis_df)
            logger.info(f"Found {len(structural['bullish'])} bullish and {len(structural['bearish'])} bearish structural divergences")
            
            # Find momentum divergences
            logger.debug("Searching for momentum divergences...")
            momentum = self._find_momentum_divergences(analysis_df)
            logger.info(f"Found {len(momentum['bullish'])} bullish and {len(momentum['bearish'])} bearish momentum divergences")
            
            # Validate divergences
            logger.debug("Validating divergence patterns...")
            regular = self._validate_divergences(analysis_df, regular, 'regular')
            hidden = self._validate_divergences(analysis_df, hidden, 'hidden')
            structural = self._validate_divergences(analysis_df, structural, 'structural')
            momentum = self._validate_divergences(analysis_df, momentum, 'momentum')
            
            # Analyze relationships between different divergence types
            logger.debug("Analyzing divergence relationships...")
            relationships = self._analyze_divergence_relationships(regular, hidden, structural, momentum)
            
            logger.info("Divergence analysis complete")
            return {
                'regular': regular,
                'hidden': hidden,
                'structural': structural,
                'momentum': momentum,
                'relationships': relationships,
                'indicator_quality': indicator_quality
            }
            
        except Exception as e:
            logger.error(f"Error in divergence analysis: {str(e)}")
            return {
                'regular': {'bullish': [], 'bearish': []},
                'hidden': {'bullish': [], 'bearish': []},
                'structural': {'bullish': [], 'bearish': []},
                'momentum': {'bullish': [], 'bearish': []},
                'relationships': {},
                'indicator_quality': {'quality_score': 0, 'reason': str(e)}
            }
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators for divergence analysis with improved edge case handling."""
        try:
            # RSI with improved calculation
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            # Use exponential moving average for smoother RSI
            avg_gain = gain.ewm(com=13, adjust=False).mean()
            avg_loss = loss.ewm(com=13, adjust=False).mean()
            
            # Ensure no division by zero
            rs = avg_gain / avg_loss.replace(0, 1e-10)
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Ensure RSI stays within bounds
            df['rsi'] = df['rsi'].clip(0, 100)
            
            # MACD with validation
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['signal']
            
            # Improved OBV calculation
            df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
            
            # Enhanced MFI calculation with dtype handling
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            money_flow = typical_price * df['volume']
            
            # Initialize flow series with proper dtype
            pos_flow = pd.Series(0.0, index=df.index)  # Explicitly use float
            neg_flow = pd.Series(0.0, index=df.index)  # Explicitly use float
            
            # Calculate positive and negative money flow with proper type conversion
            pos_mask = typical_price > typical_price.shift(1)
            neg_mask = typical_price < typical_price.shift(1)
            
            # Convert money_flow to float64 before assignment
            pos_flow[pos_mask] = money_flow[pos_mask].astype('float64')
            neg_flow[neg_mask] = money_flow[neg_mask].astype('float64')
            
            # Use exponential moving average for smoother MFI
            pos_mf = pos_flow.ewm(span=14, adjust=False).mean()
            neg_mf = neg_flow.ewm(span=14, adjust=False).mean()
            
            # Ensure no division by zero
            mf_ratio = pos_mf / neg_mf.replace(0, 1e-10)
            df['mfi'] = 100 - (100 / (1 + mf_ratio))
            
            # Ensure MFI stays within bounds
            df['mfi'] = df['mfi'].clip(0, 100)
            
            # Add validation flags
            df['indicators_valid'] = (
                (df['rsi'] >= 0) & (df['rsi'] <= 100) &
                (df['mfi'] >= 0) & (df['mfi'] <= 100) &
                df['macd'].notna() & df['signal'].notna()
            )
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            return df
    
    def _find_regular_divergences(self, df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """Find regular (classic) divergences with enhanced validation."""
        try:
            logger.debug("Starting regular divergence detection...")
            bullish = []
            bearish = []
            
            # Track statistics for debugging
            stats = {
                'total_checked': 0,
                'price_swings_found': 0,
                'indicator_swings_found': 0,
                'potential_divergences': 0,
                'confirmed_divergences': 0,
                'rejected_reasons': {
                    'small_price_swing': 0,
                    'small_indicator_swing': 0,
                    'no_confirmation': 0,
                    'invalid_pattern': 0
                }
            }
            
            # Reduce minimum swing size for more sensitive detection
            min_swing_size = self.min_swing_size * 0.5  # Make more sensitive
            min_rsi_swing = self.min_rsi_swing * 0.75   # Make more sensitive
            
            for i in range(2, len(df)-2):  # Reduced range for more frequent checks
                stats['total_checked'] += 1
                window = df.iloc[i-self.lookback_period:i+1]
                
                # Find price swings with smoothed data
                price_low = window['low'].rolling(2).mean().min()
                price_high = window['high'].rolling(2).mean().max()
                price_swing = abs(price_high - price_low)
                
                if price_swing >= min_swing_size:
                    stats['price_swings_found'] += 1
                    logger.debug(f"Found price swing at index {i}: {price_swing:.5f}")
                    
                    # Check RSI divergence with smoothed data
                    if 'rsi' in df.columns:
                        rsi_values = window['rsi'].rolling(2).mean()
                        rsi_low = rsi_values.min()
                        rsi_high = rsi_values.max()
                        rsi_swing = abs(rsi_high - rsi_low)
                        
                        if rsi_swing >= min_rsi_swing:
                            stats['indicator_swings_found'] += 1
                            
                            # Check for potential bullish divergence
                            if window['low'].iloc[-1] < window['low'].iloc[:-1].min():
                                stats['potential_divergences'] += 1
                                
                                if window['rsi'].iloc[-1] > window['rsi'].iloc[:-1].min():
                                    # Validate with volume
                                    volume_confirming = window['volume'].iloc[-1] > window['volume'].mean()
                                    
                                    if volume_confirming:
                                        stats['confirmed_divergences'] += 1
                                        logger.info(f"Found bullish RSI divergence at index {i}")
                                        
                                        # Calculate divergence angle
                                        price_change = (window['low'].iloc[-1] - window['low'].min()) / window['low'].min()
                                        indicator_change = (window['rsi'].iloc[-1] - window['rsi'].min()) / window['rsi'].min() if window['rsi'].min() != 0 else 0
                                        angle_strength = abs(indicator_change - price_change)
                                        
                                        bullish.append({
                                            'type': 'rsi',
                                            'start_index': i - self.lookback_period,
                                            'end_index': i,
                                            'price_start': window['low'].min(),
                                            'price_end': window['low'].iloc[-1],
                                            'indicator_start': window['rsi'].min(),
                                            'indicator_end': window['rsi'].iloc[-1],
                                            'strength': rsi_swing / 30,
                                            'angle_strength': angle_strength
                                        })
                            
                            # Check for potential bearish divergence
                            if window['high'].iloc[-1] > window['high'].iloc[:-1].max():
                                stats['potential_divergences'] += 1
                                
                                if window['rsi'].iloc[-1] < window['rsi'].iloc[:-1].max():
                                    # Validate with volume
                                    volume_confirming = window['volume'].iloc[-1] > window['volume'].mean()
                                    
                                    if volume_confirming:
                                        stats['confirmed_divergences'] += 1
                                        logger.info(f"Found bearish RSI divergence at index {i}")
                                        
                                        # Calculate divergence angle
                                        price_change = (window['high'].iloc[-1] - window['high'].max()) / window['high'].max()
                                        indicator_change = (window['rsi'].iloc[-1] - window['rsi'].max()) / window['rsi'].max() if window['rsi'].max() != 0 else 0
                                        angle_strength = abs(indicator_change - price_change)
                                        
                                        bearish.append({
                                            'type': 'rsi',
                                            'start_index': i - self.lookback_period,
                                            'end_index': i,
                                            'price_start': window['high'].max(),
                                            'price_end': window['high'].iloc[-1],
                                            'indicator_start': window['rsi'].max(),
                                            'indicator_end': window['rsi'].iloc[-1],
                                            'strength': rsi_swing / 30,
                                            'angle_strength': angle_strength
                                        })
            
            # Log statistics
            logger.info("Regular divergence detection statistics:")
            logger.info(f"Total periods checked: {stats['total_checked']}")
            logger.info(f"Price swings found: {stats['price_swings_found']}")
            logger.info(f"Indicator swings found: {stats['indicator_swings_found']}")
            logger.info(f"Potential divergences: {stats['potential_divergences']}")
            logger.info(f"Confirmed divergences: {stats['confirmed_divergences']}")
            
            return {
                'bullish': bullish,
                'bearish': bearish,
                'stats': stats
            }
            
        except Exception as e:
            logger.error(f"Error finding regular divergences: {str(e)}")
            return {'bullish': [], 'bearish': [], 'stats': {}}
    
    def _validate_bullish_divergence(self, window: pd.DataFrame) -> bool:
        """Validate a bullish divergence pattern with improved sensitivity."""
        try:
            # Calculate differences between the current swing and previous swings
            price_diff = window['low'].iloc[:-1].min() - window['low'].iloc[-1]
            rsi_diff = window['rsi'].iloc[-1] - window['rsi'].iloc[:-1].min()

            # Calculate trend context with shorter window
            price_trend = window['close'].diff().rolling(3).mean().iloc[-1]
            volume_trend = window['volume'].diff().rolling(3).mean().iloc[-1]
            
            # More sensitive price validation
            price_valid = price_diff >= self.min_swing_size * 0.75  # Reduced threshold
            
            # More sensitive RSI validation
            rsi_valid = rsi_diff >= 1.5  # Reduced from 2.0
            
            # Volume validation with reduced requirements
            volume_confirming = volume_trend > -0.1  # Allow slightly declining volume
            
            # More sensitive reversal detection
            reversal_forming = price_trend > -0.00005  # More sensitive to price changes
            
            logger.debug(f"Bullish validation - Price diff: {price_diff:.5f}, RSI diff: {rsi_diff:.2f}")
            logger.debug(f"Volume trend: {volume_trend:.2f}, Price trend: {price_trend:.5f}")
            
            # Return True if price and RSI conditions are met, plus either volume or reversal confirmation
            return price_valid and rsi_valid and (volume_confirming or reversal_forming)
            
        except Exception as e:
            logger.error(f"Error validating bullish divergence: {str(e)}")
            return False
    
    def _validate_bearish_divergence(self, window: pd.DataFrame) -> bool:
        """Validate a bearish divergence pattern with improved sensitivity."""
        try:
            # Calculate differences between the current swing and previous swings
            price_diff = window['high'].iloc[-1] - window['high'].iloc[:-1].max()
            rsi_diff = window['rsi'].iloc[:-1].max() - window['rsi'].iloc[-1]

            # Calculate trend context with shorter window
            price_trend = window['close'].diff().rolling(3).mean().iloc[-1]
            volume_trend = window['volume'].diff().rolling(3).mean().iloc[-1]
            
            # More sensitive price validation
            price_valid = price_diff >= self.min_swing_size * 0.75  # Reduced threshold
            
            # More sensitive RSI validation
            rsi_valid = rsi_diff >= 1.5  # Reduced from 2.0
            
            # Volume validation with reduced requirements
            volume_confirming = volume_trend > -0.1  # Allow slightly declining volume
            
            # More sensitive reversal detection
            reversal_forming = price_trend < 0.00005  # More sensitive to price changes
            
            logger.debug(f"Bearish validation - Price diff: {price_diff:.5f}, RSI diff: {rsi_diff:.2f}")
            logger.debug(f"Volume trend: {volume_trend:.2f}, Price trend: {price_trend:.5f}")
            
            # Return True if price and RSI conditions are met, plus either volume or reversal confirmation
            return price_valid and rsi_valid and (volume_confirming or reversal_forming)
            
        except Exception as e:
            logger.error(f"Error validating bearish divergence: {str(e)}")
            return False
    
    def _find_hidden_divergences(self, df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """Find hidden divergences with improved sensitivity."""
        try:
            bullish = []
            bearish = []
            
            for i in range(self.lookback_period, len(df)):
                window = df.iloc[i-self.lookback_period:i+1]
                
                # Find price swings with smoothed data
                price_low = window['low'].rolling(2).mean().min()
                price_high = window['high'].rolling(2).mean().max()
                price_low_idx = window['low'].rolling(2).mean().idxmin()
                price_high_idx = window['high'].rolling(2).mean().idxmax()
                
                # RSI divergence with smoothed data
                rsi_low = window['rsi'].rolling(2).mean().min()
                rsi_high = window['rsi'].rolling(2).mean().max()
                rsi_low_idx = window['rsi'].rolling(2).mean().idxmin()
                rsi_high_idx = window['rsi'].rolling(2).mean().idxmax()
                
                # Bullish hidden (price higher low, indicator lower low)
                if price_low > window['low'].iloc[-self.lookback_period] and \
                   rsi_low < window['rsi'].iloc[-self.lookback_period] and \
                   abs(price_low - window['low'].iloc[-self.lookback_period]) > self.hidden_divergence_threshold:
                    bullish.append({
                        'type': 'rsi',
                        'start_index': price_low_idx,
                        'end_index': i,
                        'price_start': price_low,
                        'price_end': window['close'].iloc[-1],
                        'indicator_start': rsi_low,
                        'indicator_end': window['rsi'].iloc[-1],
                        'strength': abs(rsi_low - window['rsi'].iloc[-1]) / 30  # Normalized strength
                    })
                
                # Bearish hidden (price lower high, indicator higher high)
                if price_high < window['high'].iloc[-self.lookback_period] and \
                   rsi_high > window['rsi'].iloc[-self.lookback_period] and \
                   abs(price_high - window['high'].iloc[-self.lookback_period]) > self.hidden_divergence_threshold:
                    bearish.append({
                        'type': 'rsi',
                        'start_index': price_high_idx,
                        'end_index': i,
                        'price_start': price_high,
                        'price_end': window['close'].iloc[-1],
                        'indicator_start': rsi_high,
                        'indicator_end': window['rsi'].iloc[-1],
                        'strength': abs(rsi_high - window['rsi'].iloc[-1]) / 30  # Normalized strength
                    })
            
            return {
                'bullish': bullish,
                'bearish': bearish
            }
            
        except Exception as e:
            logger.error(f"Error finding hidden divergences: {str(e)}")
            return {'bullish': [], 'bearish': []}
    
    def _find_structural_divergences(self, df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """Find structural divergences (between swing points)."""
        try:
            bullish = []
            bearish = []
            
            # Find swing points
            swing_highs = []
            swing_lows = []
            
            for i in range(2, len(df)-2):
                # Swing high
                if df['high'].iloc[i] > df['high'].iloc[i-1] and \
                   df['high'].iloc[i] > df['high'].iloc[i-2] and \
                   df['high'].iloc[i] > df['high'].iloc[i+1] and \
                   df['high'].iloc[i] > df['high'].iloc[i+2]:
                    swing_highs.append(i)
                
                # Swing low
                if df['low'].iloc[i] < df['low'].iloc[i-1] and \
                   df['low'].iloc[i] < df['low'].iloc[i-2] and \
                   df['low'].iloc[i] < df['low'].iloc[i+1] and \
                   df['low'].iloc[i] < df['low'].iloc[i+2]:
                    swing_lows.append(i)
            
            # Compare consecutive swing points
            for i in range(1, len(swing_lows)):
                # Price made lower low
                if df['low'].iloc[swing_lows[i]] < df['low'].iloc[swing_lows[i-1]]:
                    # RSI made higher low (bullish)
                    if df['rsi'].iloc[swing_lows[i]] > df['rsi'].iloc[swing_lows[i-1]]:
                        bullish.append({
                            'type': 'structural_rsi',
                            'start_index': swing_lows[i-1],
                            'end_index': swing_lows[i],
                            'price_start': df['low'].iloc[swing_lows[i-1]],
                            'price_end': df['low'].iloc[swing_lows[i]],
                            'indicator_start': df['rsi'].iloc[swing_lows[i-1]],
                            'indicator_end': df['rsi'].iloc[swing_lows[i]]
                        })
            
            for i in range(1, len(swing_highs)):
                # Price made higher high
                if df['high'].iloc[swing_highs[i]] > df['high'].iloc[swing_highs[i-1]]:
                    # RSI made lower high (bearish)
                    if df['rsi'].iloc[swing_highs[i]] < df['rsi'].iloc[swing_highs[i-1]]:
                        bearish.append({
                            'type': 'structural_rsi',
                            'start_index': swing_highs[i-1],
                            'end_index': swing_highs[i],
                            'price_start': df['high'].iloc[swing_highs[i-1]],
                            'price_end': df['high'].iloc[swing_highs[i]],
                            'indicator_start': df['rsi'].iloc[swing_highs[i-1]],
                            'indicator_end': df['rsi'].iloc[swing_highs[i]]
                        })
            
            return {
                'bullish': bullish,
                'bearish': bearish
            }
            
        except Exception as e:
            logger.error(f"Error finding structural divergences: {str(e)}")
            return {'bullish': [], 'bearish': []}
    
    def _find_momentum_divergences(self, df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """Find momentum divergences with improved sensitivity and edge case handling."""
        try:
            bullish = []
            bearish = []
            
            for i in range(self.lookback_period, len(df)):
                window = df.iloc[i-self.lookback_period:i+1]
                
                # Find price swings with reduced smoothing
                price_low = window['low'].min()
                price_high = window['high'].max()
                price_low_idx = window['low'].idxmin()
                price_high_idx = window['high'].idxmax()
                
                # MFI divergence with relaxed bounds checking
                if 'mfi' in window.columns:
                    mfi_values = window['mfi'].ffill()
                    if not mfi_values.empty and mfi_values.notna().any():
                        mfi_low = mfi_values.min()
                        mfi_high = mfi_values.max()
                
                # Bullish MFI divergence
                        if (price_low < window['low'].iloc[-self.lookback_period] and 
                            mfi_values.iloc[-1] > mfi_values.iloc[-self.lookback_period] and
                            abs(price_low - window['low'].iloc[-self.lookback_period]) > self.momentum_divergence_threshold):
                            
                            bullish.append({
                                'type': 'mfi',
                                'start_index': price_low_idx,
                                'end_index': i,
                                'price_start': price_low,
                                'price_end': window['close'].iloc[-1],
                                'indicator_start': mfi_low,
                                'indicator_end': mfi_values.iloc[-1],
                                'strength': abs(mfi_values.iloc[-1] - mfi_low) / 100
                            })
                
                # Bearish MFI divergence
                        if (price_high > window['high'].iloc[-self.lookback_period] and 
                            mfi_values.iloc[-1] < mfi_values.iloc[-self.lookback_period] and
                            abs(price_high - window['high'].iloc[-self.lookback_period]) > self.momentum_divergence_threshold):
                            
                            bearish.append({
                                'type': 'mfi',
                                'start_index': price_high_idx,
                                'end_index': i,
                                'price_start': price_high,
                                'price_end': window['close'].iloc[-1],
                                'indicator_start': mfi_high,
                                'indicator_end': mfi_values.iloc[-1],
                                'strength': abs(mfi_high - mfi_values.iloc[-1]) / 100
                            })
                
                # OBV divergence with improved volume analysis
                if 'obv' in window.columns and 'volume' in window.columns:
                    obv_values = window['obv'].ffill()
                    volume_mean = window['volume'].mean()
                    
                    if not obv_values.empty and obv_values.notna().any():
                        # Calculate rate of change for both price and OBV
                        price_roc = (window['close'].iloc[-1] - window['close'].iloc[0]) / window['close'].iloc[0]
                        obv_roc = (obv_values.iloc[-1] - obv_values.iloc[0]) / abs(obv_values.iloc[0]) if obv_values.iloc[0] != 0 else 0
                
                # Bullish OBV divergence
                        if (price_roc < -0.0005 and  # Reduced threshold
                            obv_roc > 0.0005 and     # Reduced threshold
                            window['volume'].iloc[-1] > volume_mean * 0.7):  # Reduced volume requirement
                            
                            bullish.append({
                                'type': 'obv',
                                'start_index': i - self.lookback_period,
                                'end_index': i,
                                'price_start': window['close'].iloc[0],
                                'price_end': window['close'].iloc[-1],
                                'indicator_start': obv_values.iloc[0],
                                'indicator_end': obv_values.iloc[-1],
                                'strength': abs(obv_roc)
                            })
                
                # Bearish OBV divergence
                        if (price_roc > 0.0005 and   # Reduced threshold
                            obv_roc < -0.0005 and    # Reduced threshold
                            window['volume'].iloc[-1] > volume_mean * 0.7):  # Reduced volume requirement
                            
                            bearish.append({
                                'type': 'obv',
                                'start_index': i - self.lookback_period,
                                'end_index': i,
                                'price_start': window['close'].iloc[0],
                                'price_end': window['close'].iloc[-1],
                                'indicator_start': obv_values.iloc[0],
                                'indicator_end': obv_values.iloc[-1],
                                'strength': abs(obv_roc)
                            })
            
            # Add MACD momentum divergence detection
            if 'macd' in df.columns and 'macd_hist' in df.columns:
                for i in range(self.lookback_period, len(df)):
                    window = df.iloc[i-self.lookback_period:i+1]
                    
                    # Calculate MACD momentum
                    macd_momentum = window['macd_hist'].diff().rolling(3).mean()
                    price_momentum = window['close'].diff().rolling(3).mean()
                    
                    # Bullish MACD divergence
                    if (price_momentum.iloc[-1] < -0.0001 and  # Reduced threshold
                        macd_momentum.iloc[-1] > 0.0001 and    # Reduced threshold
                        window['volume'].iloc[-1] > window['volume'].mean() * 0.7):
                        
                        bullish.append({
                            'type': 'macd',
                            'start_index': i - self.lookback_period,
                            'end_index': i,
                            'price_start': window['close'].iloc[0],
                            'price_end': window['close'].iloc[-1],
                            'indicator_start': window['macd'].iloc[0],
                            'indicator_end': window['macd'].iloc[-1],
                            'strength': abs(macd_momentum.iloc[-1] / price_momentum.iloc[-1])
                        })
                    
                    # Bearish MACD divergence
                    if (price_momentum.iloc[-1] > 0.0001 and   # Reduced threshold
                        macd_momentum.iloc[-1] < -0.0001 and   # Reduced threshold
                        window['volume'].iloc[-1] > window['volume'].mean() * 0.7):
                        
                        bearish.append({
                            'type': 'macd',
                            'start_index': i - self.lookback_period,
                            'end_index': i,
                            'price_start': window['close'].iloc[0],
                            'price_end': window['close'].iloc[-1],
                            'indicator_start': window['macd'].iloc[0],
                            'indicator_end': window['macd'].iloc[-1],
                            'strength': abs(macd_momentum.iloc[-1] / price_momentum.iloc[-1])
                        })
            
            return {
                'bullish': bullish,
                'bearish': bearish
            }
            
        except Exception as e:
            logger.error(f"Error finding momentum divergences: {str(e)}")
            return {'bullish': [], 'bearish': []} 
    
    def _assess_indicator_quality(self, df: pd.DataFrame) -> Dict:
        """Assess the quality and reliability of calculated indicators."""
        try:
            quality_score = 1.0
            issues = []
            
            # Check RSI bounds and validity
            if 'rsi' in df.columns:
                rsi_valid = (df['rsi'] >= 0) & (df['rsi'] <= 100)
                rsi_quality = rsi_valid.mean()
                quality_score *= rsi_quality
                
                if rsi_quality < 0.95:
                    issues.append("RSI values out of bounds")
            
            # Check MFI bounds and validity
            if 'mfi' in df.columns:
                mfi_valid = (df['mfi'] >= 0) & (df['mfi'] <= 100)
                mfi_quality = mfi_valid.mean()
                quality_score *= mfi_quality
                
                if mfi_quality < 0.95:
                    issues.append("MFI values out of bounds")
            
            # Check MACD validity
            if 'macd' in df.columns and 'signal' in df.columns:
                macd_valid = df['macd'].notna() & df['signal'].notna()
                macd_quality = macd_valid.mean()
                quality_score *= macd_quality
                
                if macd_quality < 0.95:
                    issues.append("MACD calculation issues")
            
            # Check for sufficient data points
            if len(df) < self.min_data_points:
                quality_score *= 0.5
                issues.append(f"Insufficient data points ({len(df)} < {self.min_data_points})")
            
            # Check for extreme values
            if 'close' in df.columns:
                price_std = df['close'].std()
                price_mean = df['close'].mean()
                if price_std > price_mean * 0.1:  # More than 10% standard deviation
                    quality_score *= 0.8
                    issues.append("High price volatility")
            
            quality_score = round(quality_score, 2)
            reason = '; '.join(issues) if issues else "All indicators valid"
            
            logger.info(f"Indicator quality score: {quality_score} - {reason}")
            return {
                'quality_score': quality_score,
                'reason': reason,
                'issues': issues
            }
            
        except Exception as e:
            logger.error(f"Error assessing indicator quality: {str(e)}")
            return {
                'quality_score': 0,
                'reason': str(e),
                'issues': ['Error assessing indicator quality']
            }
    
    def _validate_divergences(self, df: pd.DataFrame, divergences: Dict[str, List[Dict]], indicator_label: str) -> Dict[str, List[Dict]]:
        """Validate divergence patterns in the provided divergences dictionary."""
        try:
            validated_divergences = {'bullish': [], 'bearish': []}
            
            # Validate DataFrame
            if df is None or df.empty:
                logger.warning("Empty DataFrame provided for divergence validation")
                return validated_divergences
                
            df_length = len(df)
            
            for side in ['bullish', 'bearish']:
                for divergence in divergences.get(side, []):
                    try:
                        # Validate indices are within bounds
                        start_idx = divergence.get('start_index')
                        end_idx = divergence.get('end_index')
                        
                        if start_idx is None or end_idx is None:
                            logger.warning(f"Missing index in divergence: {divergence}")
                            continue
                            
                        if not (0 <= start_idx < df_length and 0 <= end_idx < df_length):
                            logger.warning(f"Index out of bounds - start: {start_idx}, end: {end_idx}, df length: {df_length}")
                            continue
                        
                        # Get indicator key
                        indicator_key = divergence.get('type', indicator_label)
                        if indicator_key not in df.columns:
                            logger.warning(f"Indicator {indicator_key} not found in DataFrame columns: {df.columns}")
                            continue
                        
                        # Validate price and indicator values
                        if (divergence['price_start'] < df['close'].iloc[start_idx] and
                            divergence['price_end'] > df['close'].iloc[end_idx] and
                            divergence['indicator_start'] < df[indicator_key].iloc[start_idx] and
                            divergence['indicator_end'] > df[indicator_key].iloc[end_idx]):
                            validated_divergences[side].append(divergence)
                            
                    except Exception as div_error:
                        logger.warning(f"Error validating individual divergence: {str(div_error)}")
                        continue
            
            return validated_divergences
            
        except Exception as e:
            logger.error(f"Error validating divergences for {indicator_label}: {str(e)}")
            return {'bullish': [], 'bearish': []}
    
    def _analyze_divergence_relationships(self, regular: Dict[str, List[Dict]], hidden: Dict[str, List[Dict]], structural: Dict[str, List[Dict]], momentum: Dict[str, List[Dict]]) -> Dict:
        """Analyze relationships between different divergence types."""
        try:
            relationships = {}
            
            # Regular to Hidden
            relationships['regular_to_hidden'] = self._find_divergence_relationship(regular, hidden, 'regular', 'hidden')
            
            # Regular to Structural
            relationships['regular_to_structural'] = self._find_divergence_relationship(regular, structural, 'regular', 'structural')
            
            # Regular to Momentum
            relationships['regular_to_momentum'] = self._find_divergence_relationship(regular, momentum, 'regular', 'momentum')
            
            # Hidden to Structural
            relationships['hidden_to_structural'] = self._find_divergence_relationship(hidden, structural, 'hidden', 'structural')
            
            # Hidden to Momentum
            relationships['hidden_to_momentum'] = self._find_divergence_relationship(hidden, momentum, 'hidden', 'momentum')
            
            # Structural to Momentum
            relationships['structural_to_momentum'] = self._find_divergence_relationship(structural, momentum, 'structural', 'momentum')
            
            return relationships
            
        except Exception as e:
            logger.error(f"Error analyzing divergence relationships: {str(e)}")
            return {}
    
    def _find_divergence_relationship(self, div_type1: Dict[str, List[Dict]], div_type2: Dict[str, List[Dict]], type1_name: str, type2_name: str) -> Dict:
        """Generic function to find relationships between two types of divergences."""
        try:
            relationships = {}
            
            # Check bullish relationships
            for div1 in div_type1['bullish']:
                for div2 in div_type2['bullish']:
                    # More flexible relationship criteria
                    if self._is_overlapping(div1, div2):
                        relationships[f"{type1_name}_to_{type2_name}_bullish"] = {
                            type1_name: div1,
                            type2_name: div2
                        }
            
            # Check bearish relationships
            for div1 in div_type1['bearish']:
                for div2 in div_type2['bearish']:
                    # More flexible relationship criteria
                    if self._is_overlapping(div1, div2):
                        relationships[f"{type1_name}_to_{type2_name}_bearish"] = {
                            type1_name: div1,
                            type2_name: div2
                        }
            
            return relationships
            
        except Exception as e:
            logger.error(f"Error finding {type1_name} to {type2_name} relationships: {str(e)}")
            return {}
    
    def _is_overlapping(self, div1: Dict, div2: Dict) -> bool:
        """Check if two divergences have a meaningful overlap based on more flexible criteria."""
        try:
            # Time overlap
            time_overlap = (div1['start_index'] <= div2['end_index'] and 
                            div1['end_index'] >= div2['start_index'])
            
            # Price movement agreement
            price_agreement = ((div1['price_end'] > div1['price_start'] and 
                                div2['price_end'] > div2['price_start']) or
                               (div1['price_end'] < div1['price_start'] and 
                                div2['price_end'] < div2['price_start']))
            
            # Indicator movement agreement
            indicator_agreement = ((div1['indicator_end'] > div1['indicator_start'] and 
                                   div2['indicator_end'] > div2['indicator_start']) or
                                  (div1['indicator_end'] < div1['indicator_start'] and 
                                   div2['indicator_end'] < div2['indicator_start']))
            
            # Return True if there is time overlap and either price or indicator agreement
            return time_overlap and (price_agreement or indicator_agreement)
            
        except Exception as e:
            logger.error(f"Error checking divergence overlap: {str(e)}")
            return False