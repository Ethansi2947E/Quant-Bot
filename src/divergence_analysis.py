from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from loguru import logger

class DivergenceAnalysis:
    def __init__(self):
        self.lookback_period = 20  # Bars to look back for divergence
        self.divergence_threshold = 0.0010  # 10 pips minimum price movement
        
    def analyze_divergences(self, df: pd.DataFrame) -> Dict:
        """Analyze multiple types of divergences."""
        try:
            # Calculate indicators
            df = self._calculate_indicators(df)
            
            # Find regular divergences
            regular = self._find_regular_divergences(df)
            
            # Find hidden divergences
            hidden = self._find_hidden_divergences(df)
            
            # Find structural divergences
            structural = self._find_structural_divergences(df)
            
            # Find momentum divergences
            momentum = self._find_momentum_divergences(df)
            
            return {
                'regular': regular,
                'hidden': hidden,
                'structural': structural,
                'momentum': momentum
            }
            
        except Exception as e:
            logger.error(f"Error in divergence analysis: {str(e)}")
            return {
                'regular': {'bullish': [], 'bearish': []},
                'hidden': {'bullish': [], 'bearish': []},
                'structural': {'bullish': [], 'bearish': []},
                'momentum': {'bullish': [], 'bearish': []}
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
        """Find regular (classic) divergences."""
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
                
                # Bullish regular (price lower low, indicator higher low)
                if price_low < window['low'].iloc[-self.lookback_period] and \
                   rsi_low > window['rsi'].iloc[-self.lookback_period] and \
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
                
                # Bearish regular (price higher high, indicator lower high)
                if price_high > window['high'].iloc[-self.lookback_period] and \
                   rsi_high < window['rsi'].iloc[-self.lookback_period] and \
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
                
                # MACD divergence
                macd_low = window['macd'].min()
                macd_high = window['macd'].max()
                macd_low_idx = window['macd'].idxmin()
                macd_high_idx = window['macd'].idxmax()
                
                # Bullish MACD divergence
                if price_low < window['low'].iloc[-self.lookback_period] and \
                   macd_low > window['macd'].iloc[-self.lookback_period] and \
                   abs(price_low - window['low'].iloc[-self.lookback_period]) > self.divergence_threshold:
                    bullish.append({
                        'type': 'macd',
                        'start_index': price_low_idx,
                        'end_index': i,
                        'price_start': price_low,
                        'price_end': window['close'].iloc[-1],
                        'indicator_start': macd_low,
                        'indicator_end': window['macd'].iloc[-1]
                    })
                
                # Bearish MACD divergence
                if price_high > window['high'].iloc[-self.lookback_period] and \
                   macd_high < window['macd'].iloc[-self.lookback_period] and \
                   abs(price_high - window['high'].iloc[-self.lookback_period]) > self.divergence_threshold:
                    bearish.append({
                        'type': 'macd',
                        'start_index': price_high_idx,
                        'end_index': i,
                        'price_start': price_high,
                        'price_end': window['close'].iloc[-1],
                        'indicator_start': macd_high,
                        'indicator_end': window['macd'].iloc[-1]
                    })
            
            return {
                'bullish': bullish,
                'bearish': bearish
            }
            
        except Exception as e:
            logger.error(f"Error finding regular divergences: {str(e)}")
            return {'bullish': [], 'bearish': []}
    
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