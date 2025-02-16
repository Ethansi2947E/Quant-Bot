from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from loguru import logger

class DivergenceAnalysis:
    def __init__(self):
        self.lookback_period = 20  # Bars to look back for divergence
        self.divergence_threshold = 0.0010  # 10 pips minimum price movement
        self.min_swing_size = 0.0005  # Minimum swing size for divergence
        self.confirmation_bars = 2  # Number of bars to confirm divergence
        
    def analyze(self, df: pd.DataFrame) -> Dict:
        """Analyze multiple types of divergences with enhanced logging and validation."""
        try:
            logger.info("Starting divergence analysis...")
            
            # Calculate indicators
            logger.debug("Calculating technical indicators...")
            df = self._calculate_indicators(df)
            
            # Analyze indicator quality
            logger.debug("Analyzing indicator quality...")
            indicator_quality = self._assess_indicator_quality(df)
            if indicator_quality['quality_score'] < 0.7:
                logger.warning(f"Low indicator quality: {indicator_quality['reason']}")
            
            # Find regular divergences
            logger.debug("Searching for regular divergences...")
            regular = self._find_regular_divergences(df)
            logger.info(f"Found {len(regular['bullish'])} bullish and {len(regular['bearish'])} bearish regular divergences")
            
            # Find hidden divergences
            logger.debug("Searching for hidden divergences...")
            hidden = self._find_hidden_divergences(df)
            logger.info(f"Found {len(hidden['bullish'])} bullish and {len(hidden['bearish'])} bearish hidden divergences")
            
            # Find structural divergences
            logger.debug("Searching for structural divergences...")
            structural = self._find_structural_divergences(df)
            logger.info(f"Found {len(structural['bullish'])} bullish and {len(structural['bearish'])} bearish structural divergences")
            
            # Find momentum divergences
            logger.debug("Searching for momentum divergences...")
            momentum = self._find_momentum_divergences(df)
            logger.info(f"Found {len(momentum['bullish'])} bullish and {len(momentum['bearish'])} bearish momentum divergences")
            
            # Validate divergences
            logger.debug("Validating divergence patterns...")
            regular = self._validate_divergences(df, regular, 'regular')
            hidden = self._validate_divergences(df, hidden, 'hidden')
            structural = self._validate_divergences(df, structural, 'structural')
            momentum = self._validate_divergences(df, momentum, 'momentum')
            
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
        """Calculate indicators for divergence analysis."""
        try:
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['close'].ewm(span=12).mean()
            exp2 = df['close'].ewm(span=26).mean()
            df['macd'] = exp1 - exp2
            df['signal'] = df['macd'].ewm(span=9).mean()
            df['macd_hist'] = df['macd'] - df['signal']
            
            # OBV (On Balance Volume)
            df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
            
            # MFI (Money Flow Index)
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            raw_money_flow = typical_price * df['volume']
            
            positive_flow = pd.Series(0, index=df.index, dtype='float64')
            negative_flow = pd.Series(0, index=df.index, dtype='float64')
            
            # Calculate positive and negative money flow
            positive_mask = typical_price > typical_price.shift(1)
            negative_mask = typical_price < typical_price.shift(1)
            
            positive_flow[positive_mask] = raw_money_flow[positive_mask]
            negative_flow[negative_mask] = raw_money_flow[negative_mask]
            
            positive_mf = positive_flow.rolling(window=14).sum()
            negative_mf = negative_flow.rolling(window=14).sum()
            
            # Calculate MFI
            mfi_ratio = positive_mf / negative_mf
            df['mfi'] = 100 - (100 / (1 + mfi_ratio))
            
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
            
            for i in range(self.lookback_period, len(df)):
                stats['total_checked'] += 1
                window = df.iloc[i-self.lookback_period:i+1]
                
                # Find price swings
                price_low = window['low'].min()
                price_high = window['high'].max()
                price_swing = abs(price_high - price_low)
                
                if price_swing >= self.min_swing_size:
                    stats['price_swings_found'] += 1
                    logger.debug(f"Found price swing at index {i}: {price_swing:.5f}")
                    
                    # Check RSI divergence
                    if 'rsi' in df.columns:
                        rsi_low = window['rsi'].min()
                        rsi_high = window['rsi'].max()
                        rsi_swing = abs(rsi_high - rsi_low)
                        
                        if rsi_swing >= 5:  # Minimum RSI swing
                            stats['indicator_swings_found'] += 1
                            stats['potential_divergences'] += 1
                            
                            # Bullish divergence (lower price low but higher RSI low)
                            if window['low'].iloc[-1] < window['low'].min() and \
                               window['rsi'].iloc[-1] > window['rsi'].min():
                                
                                # Validate the pattern
                                if self._validate_bullish_divergence(window):
                                    stats['confirmed_divergences'] += 1
                                    logger.info(f"Found bullish RSI divergence at index {i}")
                                    logger.debug(f"Price: {window['low'].iloc[-1]:.5f} < {window['low'].min():.5f}")
                                    logger.debug(f"RSI: {window['rsi'].iloc[-1]:.2f} > {window['rsi'].min():.2f}")
                                    
                                    bullish.append({
                                        'type': 'rsi',
                                        'start_index': i - self.lookback_period,
                                        'end_index': i,
                                        'price_start': window['low'].min(),
                                        'price_end': window['low'].iloc[-1],
                                        'indicator_start': window['rsi'].min(),
                                        'indicator_end': window['rsi'].iloc[-1],
                                        'strength': rsi_swing / 30  # Normalized strength
                                    })
                                else:
                                    stats['rejected_reasons']['invalid_pattern'] += 1
                                    logger.debug(f"Rejected bullish divergence at index {i}: Invalid pattern")
                            
                            # Bearish divergence (higher price high but lower RSI high)
                            elif window['high'].iloc[-1] > window['high'].max() and \
                                 window['rsi'].iloc[-1] < window['rsi'].max():
                                
                                # Validate the pattern
                                if self._validate_bearish_divergence(window):
                                    stats['confirmed_divergences'] += 1
                                    logger.info(f"Found bearish RSI divergence at index {i}")
                                    logger.debug(f"Price: {window['high'].iloc[-1]:.5f} > {window['high'].max():.5f}")
                                    logger.debug(f"RSI: {window['rsi'].iloc[-1]:.2f} < {window['rsi'].max():.2f}")
                                    
                                    bearish.append({
                                        'type': 'rsi',
                                        'start_index': i - self.lookback_period,
                                        'end_index': i,
                                        'price_start': window['high'].max(),
                                        'price_end': window['high'].iloc[-1],
                                        'indicator_start': window['rsi'].max(),
                                        'indicator_end': window['rsi'].iloc[-1],
                                        'strength': rsi_swing / 30  # Normalized strength
                                    })
                                else:
                                    stats['rejected_reasons']['invalid_pattern'] += 1
                                    logger.debug(f"Rejected bearish divergence at index {i}: Invalid pattern")
                        else:
                            stats['rejected_reasons']['small_indicator_swing'] += 1
                            logger.debug(f"Rejected at index {i}: RSI swing too small ({rsi_swing:.2f})")
                else:
                    stats['rejected_reasons']['small_price_swing'] += 1
                    logger.debug(f"Rejected at index {i}: Price swing too small ({price_swing:.5f})")
            
            # Log statistics
            logger.info("Regular divergence detection statistics:")
            logger.info(f"Total periods checked: {stats['total_checked']}")
            logger.info(f"Price swings found: {stats['price_swings_found']}")
            logger.info(f"Indicator swings found: {stats['indicator_swings_found']}")
            logger.info(f"Potential divergences: {stats['potential_divergences']}")
            logger.info(f"Confirmed divergences: {stats['confirmed_divergences']}")
            logger.info("Rejection reasons:")
            for reason, count in stats['rejected_reasons'].items():
                logger.info(f"- {reason}: {count}")
            
            return {
                'bullish': bullish,
                'bearish': bearish,
                'stats': stats
            }
            
        except Exception as e:
            logger.error(f"Error finding regular divergences: {str(e)}")
            return {'bullish': [], 'bearish': [], 'stats': {}}
    
    def _validate_bullish_divergence(self, window: pd.DataFrame) -> bool:
        """Validate a bullish divergence pattern."""
        try:
            # Check for proper price action (lower lows)
            price_valid = window['low'].iloc[-1] < window['low'].iloc[-2] < window['low'].iloc[-3]
            
            # Check for proper indicator action (higher lows)
            indicator_valid = window['rsi'].iloc[-1] > window['rsi'].iloc[-2] > window['rsi'].iloc[-3]
            
            # Check for oversold condition
            oversold = window['rsi'].iloc[-1] < 30 or window['rsi'].min() < 30
            
            # Check for momentum
            momentum = window['rsi'].iloc[-1] - window['rsi'].iloc[-2] > 0
            
            return price_valid and indicator_valid and (oversold or momentum)
            
        except Exception as e:
            logger.error(f"Error validating bullish divergence: {str(e)}")
            return False
    
    def _validate_bearish_divergence(self, window: pd.DataFrame) -> bool:
        """Validate a bearish divergence pattern."""
        try:
            # Check for proper price action (higher highs)
            price_valid = window['high'].iloc[-1] > window['high'].iloc[-2] > window['high'].iloc[-3]
            
            # Check for proper indicator action (lower highs)
            indicator_valid = window['rsi'].iloc[-1] < window['rsi'].iloc[-2] < window['rsi'].iloc[-3]
            
            # Check for overbought condition
            overbought = window['rsi'].iloc[-1] > 70 or window['rsi'].max() > 70
            
            # Check for momentum
            momentum = window['rsi'].iloc[-1] - window['rsi'].iloc[-2] < 0
            
            return price_valid and indicator_valid and (overbought or momentum)
            
        except Exception as e:
            logger.error(f"Error validating bearish divergence: {str(e)}")
            return False
    
    def _find_hidden_divergences(self, df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """Find hidden divergences."""
        try:
            bullish = []
            bearish = []
            
            for i in range(self.lookback_period, len(df)):
                window = df.iloc[i-self.lookback_period:i+1]
                
                # Find price swings
                price_low = window['low'].min()
                price_high = window['high'].max()
                price_low_idx = window['low'].idxmin()
                price_high_idx = window['high'].idxmax()
                
                # RSI divergence
                rsi_low = window['rsi'].min()
                rsi_high = window['rsi'].max()
                rsi_low_idx = window['rsi'].idxmin()
                rsi_high_idx = window['rsi'].idxmax()
                
                # Bullish hidden (price higher low, indicator lower low)
                if price_low > window['low'].iloc[-self.lookback_period] and \
                   rsi_low < window['rsi'].iloc[-self.lookback_period] and \
                   abs(price_low - window['low'].iloc[-self.lookback_period]) > self.divergence_threshold:
                    bullish.append({
                        'type': 'rsi',
                        'start_index': price_low_idx,
                        'end_index': i,
                        'price_start': price_low,
                        'price_end': window['close'].iloc[-1],
                        'indicator_start': rsi_low,
                        'indicator_end': window['rsi'].iloc[-1]
                    })
                
                # Bearish hidden (price lower high, indicator higher high)
                if price_high < window['high'].iloc[-self.lookback_period] and \
                   rsi_high > window['rsi'].iloc[-self.lookback_period] and \
                   abs(price_high - window['high'].iloc[-self.lookback_period]) > self.divergence_threshold:
                    bearish.append({
                        'type': 'rsi',
                        'start_index': price_high_idx,
                        'end_index': i,
                        'price_start': price_high,
                        'price_end': window['close'].iloc[-1],
                        'indicator_start': rsi_high,
                        'indicator_end': window['rsi'].iloc[-1]
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
        """Find momentum divergences using multiple indicators."""
        try:
            bullish = []
            bearish = []
            
            for i in range(self.lookback_period, len(df)):
                window = df.iloc[i-self.lookback_period:i+1]
                
                # Find price swings
                price_low = window['low'].min()
                price_high = window['high'].max()
                price_low_idx = window['low'].idxmin()
                price_high_idx = window['high'].idxmax()
                
                # MFI divergence
                mfi_low = window['mfi'].min()
                mfi_high = window['mfi'].max()
                
                # Bullish MFI divergence
                if price_low < window['low'].iloc[-self.lookback_period] and \
                   mfi_low > window['mfi'].iloc[-self.lookback_period] and \
                   abs(price_low - window['low'].iloc[-self.lookback_period]) > self.divergence_threshold:
                    bullish.append({
                        'type': 'mfi',
                        'start_index': price_low_idx,
                        'end_index': i,
                        'price_start': price_low,
                        'price_end': window['close'].iloc[-1],
                        'indicator_start': mfi_low,
                        'indicator_end': window['mfi'].iloc[-1]
                    })
                
                # Bearish MFI divergence
                if price_high > window['high'].iloc[-self.lookback_period] and \
                   mfi_high < window['mfi'].iloc[-self.lookback_period] and \
                   abs(price_high - window['high'].iloc[-self.lookback_period]) > self.divergence_threshold:
                    bearish.append({
                        'type': 'mfi',
                        'start_index': price_high_idx,
                        'end_index': i,
                        'price_start': price_high,
                        'price_end': window['close'].iloc[-1],
                        'indicator_start': mfi_high,
                        'indicator_end': window['mfi'].iloc[-1]
                    })
                
                # OBV divergence
                obv_low = window['obv'].min()
                obv_high = window['obv'].max()
                
                # Bullish OBV divergence
                if price_low < window['low'].iloc[-self.lookback_period] and \
                   obv_low > window['obv'].iloc[-self.lookback_period]:
                    bullish.append({
                        'type': 'obv',
                        'start_index': price_low_idx,
                        'end_index': i,
                        'price_start': price_low,
                        'price_end': window['close'].iloc[-1],
                        'indicator_start': obv_low,
                        'indicator_end': window['obv'].iloc[-1]
                    })
                
                # Bearish OBV divergence
                if price_high > window['high'].iloc[-self.lookback_period] and \
                   obv_high < window['obv'].iloc[-self.lookback_period]:
                    bearish.append({
                        'type': 'obv',
                        'start_index': price_high_idx,
                        'end_index': i,
                        'price_start': price_high,
                        'price_end': window['close'].iloc[-1],
                        'indicator_start': obv_high,
                        'indicator_end': window['obv'].iloc[-1]
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
            
            # Check for missing values
            for indicator in ['rsi', 'macd', 'obv', 'mfi']:
                if indicator not in df.columns:
                    quality_score *= 0.7
                    issues.append(f"Missing {indicator} indicator")
                elif df[indicator].isna().mean() > 0.1:
                    quality_score *= 0.8
                    issues.append(f"High number of NaN values in {indicator}")
            
            # Check RSI bounds
            if 'rsi' in df.columns:
                if not ((df['rsi'] >= 0) & (df['rsi'] <= 100)).all():
                    quality_score *= 0.8
                    issues.append("RSI values out of bounds")
            
            # Check MFI bounds
            if 'mfi' in df.columns:
                if not ((df['mfi'] >= 0) & (df['mfi'] <= 100)).all():
                    quality_score *= 0.8
                    issues.append("MFI values out of bounds")
            
            # Check for indicator stability
            if 'macd' in df.columns:
                macd_std = df['macd'].std()
                if macd_std > df['close'].std() * 0.1:
                    quality_score *= 0.9
                    issues.append("High MACD volatility")
            
            # Check OBV consistency
            if 'obv' in df.columns and 'volume' in df.columns:
                obv_changes = df['obv'].diff().abs()
                volume_values = df['volume'].abs()
                if not (obv_changes <= volume_values + 1e-10).all():
                    quality_score *= 0.8
                    issues.append("OBV changes larger than volume")
            
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
    
    def _validate_divergences(self, df: pd.DataFrame, divergences: Dict[str, List[Dict]], divergence_type: str) -> Dict[str, List[Dict]]:
        """Validate divergence patterns."""
        try:
            validated_divergences = {'bullish': [], 'bearish': []}
            
            for divergence in divergences[divergence_type]:
                # Validate price and indicator consistency
                if divergence['price_start'] < df['close'].iloc[divergence['start_index']] and \
                   divergence['price_end'] > df['close'].iloc[divergence['end_index']] and \
                   divergence['indicator_start'] < df[divergence['type']].iloc[divergence['start_index']] and \
                   divergence['indicator_end'] > df[divergence['type']].iloc[divergence['end_index']]:
                    validated_divergences[divergence_type].append(divergence)
            
            return validated_divergences
            
        except Exception as e:
            logger.error(f"Error validating {divergence_type} divergences: {str(e)}")
            return {'bullish': [], 'bearish': []}
    
    def _analyze_divergence_relationships(self, regular: Dict[str, List[Dict]], hidden: Dict[str, List[Dict]], structural: Dict[str, List[Dict]], momentum: Dict[str, List[Dict]]) -> Dict:
        """Analyze relationships between different divergence types."""
        try:
            relationships = {}
            
            # Regular to Hidden
            relationships['regular_to_hidden'] = self._find_regular_to_hidden_relationships(regular, hidden)
            
            # Regular to Structural
            relationships['regular_to_structural'] = self._find_regular_to_structural_relationships(regular, structural)
            
            # Regular to Momentum
            relationships['regular_to_momentum'] = self._find_regular_to_momentum_relationships(regular, momentum)
            
            # Hidden to Structural
            relationships['hidden_to_structural'] = self._find_hidden_to_structural_relationships(hidden, structural)
            
            # Hidden to Momentum
            relationships['hidden_to_momentum'] = self._find_hidden_to_momentum_relationships(hidden, momentum)
            
            # Structural to Momentum
            relationships['structural_to_momentum'] = self._find_structural_to_momentum_relationships(structural, momentum)
            
            return relationships
            
        except Exception as e:
            logger.error(f"Error analyzing divergence relationships: {str(e)}")
            return {}
    
    def _find_regular_to_hidden_relationships(self, regular: Dict[str, List[Dict]], hidden: Dict[str, List[Dict]]) -> Dict:
        """Find relationships between regular and hidden divergences."""
        try:
            relationships = {}
            
            for regular_divergence in regular['bullish']:
                for hidden_divergence in hidden['bullish']:
                    if regular_divergence['price_start'] < hidden_divergence['price_start'] and \
                       regular_divergence['price_end'] > hidden_divergence['price_end'] and \
                       regular_divergence['indicator_start'] < hidden_divergence['indicator_start'] and \
                       regular_divergence['indicator_end'] > hidden_divergence['indicator_end']:
                        relationships['regular_to_hidden'] = {
                            'regular': regular_divergence,
                            'hidden': hidden_divergence
                        }
            
            for regular_divergence in regular['bearish']:
                for hidden_divergence in hidden['bearish']:
                    if regular_divergence['price_start'] < hidden_divergence['price_start'] and \
                       regular_divergence['price_end'] > hidden_divergence['price_end'] and \
                       regular_divergence['indicator_start'] < hidden_divergence['indicator_start'] and \
                       regular_divergence['indicator_end'] > hidden_divergence['indicator_end']:
                        relationships['regular_to_hidden'] = {
                            'regular': regular_divergence,
                            'hidden': hidden_divergence
                        }
            
            return relationships
            
        except Exception as e:
            logger.error(f"Error finding regular to hidden relationships: {str(e)}")
            return {}
    
    def _find_regular_to_structural_relationships(self, regular: Dict[str, List[Dict]], structural: Dict[str, List[Dict]]) -> Dict:
        """Find relationships between regular and structural divergences."""
        try:
            relationships = {}
            
            for regular_divergence in regular['bullish']:
                for structural_divergence in structural['bullish']:
                    if regular_divergence['price_start'] < structural_divergence['price_start'] and \
                       regular_divergence['price_end'] > structural_divergence['price_end'] and \
                       regular_divergence['indicator_start'] < structural_divergence['indicator_start'] and \
                       regular_divergence['indicator_end'] > structural_divergence['indicator_end']:
                        relationships['regular_to_structural'] = {
                            'regular': regular_divergence,
                            'structural': structural_divergence
                        }
            
            for regular_divergence in regular['bearish']:
                for structural_divergence in structural['bearish']:
                    if regular_divergence['price_start'] < structural_divergence['price_start'] and \
                       regular_divergence['price_end'] > structural_divergence['price_end'] and \
                       regular_divergence['indicator_start'] < structural_divergence['indicator_start'] and \
                       regular_divergence['indicator_end'] > structural_divergence['indicator_end']:
                        relationships['regular_to_structural'] = {
                            'regular': regular_divergence,
                            'structural': structural_divergence
                        }
            
            return relationships
            
        except Exception as e:
            logger.error(f"Error finding regular to structural relationships: {str(e)}")
            return {}
    
    def _find_regular_to_momentum_relationships(self, regular: Dict[str, List[Dict]], momentum: Dict[str, List[Dict]]) -> Dict:
        """Find relationships between regular and momentum divergences."""
        try:
            relationships = {}
            
            for regular_divergence in regular['bullish']:
                for momentum_divergence in momentum['bullish']:
                    if regular_divergence['price_start'] < momentum_divergence['price_start'] and \
                       regular_divergence['price_end'] > momentum_divergence['price_end'] and \
                       regular_divergence['indicator_start'] < momentum_divergence['indicator_start'] and \
                       regular_divergence['indicator_end'] > momentum_divergence['indicator_end']:
                        relationships['regular_to_momentum'] = {
                            'regular': regular_divergence,
                            'momentum': momentum_divergence
                        }
            
            for regular_divergence in regular['bearish']:
                for momentum_divergence in momentum['bearish']:
                    if regular_divergence['price_start'] < momentum_divergence['price_start'] and \
                       regular_divergence['price_end'] > momentum_divergence['price_end'] and \
                       regular_divergence['indicator_start'] < momentum_divergence['indicator_start'] and \
                       regular_divergence['indicator_end'] > momentum_divergence['indicator_end']:
                        relationships['regular_to_momentum'] = {
                            'regular': regular_divergence,
                            'momentum': momentum_divergence
                        }
            
            return relationships
            
        except Exception as e:
            logger.error(f"Error finding regular to momentum relationships: {str(e)}")
            return {}
    
    def _find_hidden_to_structural_relationships(self, hidden: Dict[str, List[Dict]], structural: Dict[str, List[Dict]]) -> Dict:
        """Find relationships between hidden and structural divergences."""
        try:
            relationships = {}
            
            for hidden_divergence in hidden['bullish']:
                for structural_divergence in structural['bullish']:
                    if hidden_divergence['price_start'] < structural_divergence['price_start'] and \
                       hidden_divergence['price_end'] > structural_divergence['price_end'] and \
                       hidden_divergence['indicator_start'] < structural_divergence['indicator_start'] and \
                       hidden_divergence['indicator_end'] > structural_divergence['indicator_end']:
                        relationships['hidden_to_structural'] = {
                            'hidden': hidden_divergence,
                            'structural': structural_divergence
                        }
            
            for hidden_divergence in hidden['bearish']:
                for structural_divergence in structural['bearish']:
                    if hidden_divergence['price_start'] < structural_divergence['price_start'] and \
                       hidden_divergence['price_end'] > structural_divergence['price_end'] and \
                       hidden_divergence['indicator_start'] < structural_divergence['indicator_start'] and \
                       hidden_divergence['indicator_end'] > structural_divergence['indicator_end']:
                        relationships['hidden_to_structural'] = {
                            'hidden': hidden_divergence,
                            'structural': structural_divergence
                        }
            
            return relationships
            
        except Exception as e:
            logger.error(f"Error finding hidden to structural relationships: {str(e)}")
            return {}
    
    def _find_hidden_to_momentum_relationships(self, hidden: Dict[str, List[Dict]], momentum: Dict[str, List[Dict]]) -> Dict:
        """Find relationships between hidden and momentum divergences."""
        try:
            relationships = {}
            
            for hidden_divergence in hidden['bullish']:
                for momentum_divergence in momentum['bullish']:
                    if hidden_divergence['price_start'] < momentum_divergence['price_start'] and \
                       hidden_divergence['price_end'] > momentum_divergence['price_end'] and \
                       hidden_divergence['indicator_start'] < momentum_divergence['indicator_start'] and \
                       hidden_divergence['indicator_end'] > momentum_divergence['indicator_end']:
                        relationships['hidden_to_momentum'] = {
                            'hidden': hidden_divergence,
                            'momentum': momentum_divergence
                        }
            
            for hidden_divergence in hidden['bearish']:
                for momentum_divergence in momentum['bearish']:
                    if hidden_divergence['price_start'] < momentum_divergence['price_start'] and \
                       hidden_divergence['price_end'] > momentum_divergence['price_end'] and \
                       hidden_divergence['indicator_start'] < momentum_divergence['indicator_start'] and \
                       hidden_divergence['indicator_end'] > momentum_divergence['indicator_end']:
                        relationships['hidden_to_momentum'] = {
                            'hidden': hidden_divergence,
                            'momentum': momentum_divergence
                        }
            
            return relationships
            
        except Exception as e:
            logger.error(f"Error finding hidden to momentum relationships: {str(e)}")
            return {}
    
    def _find_structural_to_momentum_relationships(self, structural: Dict[str, List[Dict]], momentum: Dict[str, List[Dict]]) -> Dict:
        """Find relationships between structural and momentum divergences."""
        try:
            relationships = {}
            
            for structural_divergence in structural['bullish']:
                for momentum_divergence in momentum['bullish']:
                    if structural_divergence['price_start'] < momentum_divergence['price_start'] and \
                       structural_divergence['price_end'] > momentum_divergence['price_end'] and \
                       structural_divergence['indicator_start'] < momentum_divergence['indicator_start'] and \
                       structural_divergence['indicator_end'] > momentum_divergence['indicator_end']:
                        relationships['structural_to_momentum'] = {
                            'structural': structural_divergence,
                            'momentum': momentum_divergence
                        }
            
            for structural_divergence in structural['bearish']:
                for momentum_divergence in momentum['bearish']:
                    if structural_divergence['price_start'] < momentum_divergence['price_start'] and \
                       structural_divergence['price_end'] > momentum_divergence['price_end'] and \
                       structural_divergence['indicator_start'] < momentum_divergence['indicator_start'] and \
                       structural_divergence['indicator_end'] > momentum_divergence['indicator_end']:
                        relationships['structural_to_momentum'] = {
                            'structural': structural_divergence,
                            'momentum': momentum_divergence
                        }
            
            return relationships
            
        except Exception as e:
            logger.error(f"Error finding structural to momentum relationships: {str(e)}")
            return {} 