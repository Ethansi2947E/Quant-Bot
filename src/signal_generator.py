import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from loguru import logger
import pandas_ta as ta

from config.config import TRADING_CONFIG
from src.market_analysis import MarketAnalysis
from src.smc_analysis import SMCAnalysis
from src.mtf_analysis import MTFAnalysis
from src.divergence_analysis import DivergenceAnalysis
from src.volume_analysis import VolumeAnalysis

class SignalGenerator:
    def __init__(self):
        self.required_periods = {
            "rsi": 14,
            "macd": 34,  # Longest period needed for MACD (26 + 8 buffer)
            "atr": 14,
            "ema": 200  # Longest EMA period
        }
        self.max_period = max(self.required_periods.values())
        self.min_confidence = 0.5  # Minimum confidence threshold
        
        # Initialize analysis components
        self.market_analysis = MarketAnalysis()
        self.smc_analysis = SMCAnalysis()
        self.mtf_analysis = MTFAnalysis()
        self.divergence_analysis = DivergenceAnalysis()
        self.volume_analysis = VolumeAnalysis()
        
        self.weights = {
            'trend': 0.15,            # Weight for trend analysis
            'structure': 0.20,        # Weight for market structure
            'smc': 0.20,             # Weight for Smart Money Concepts
            'mtf': 0.15,             # Weight for multi-timeframe analysis
            'divergence': 0.15,       # Weight for divergence analysis
            'volume': 0.15           # Weight for volume analysis
        }
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators."""
        try:
            df = df.copy()
            
            # Ensure numeric types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Calculate OBV
            df['price_change'] = df['close'].diff()
            df['obv'] = 0.0  # Initialize OBV column
            
            # First value of OBV is the same as volume
            df.loc[df.index[0], 'obv'] = df['volume'].iloc[0]
            
            # Calculate OBV for the rest of the data
            for i in range(1, len(df)):
                if df['price_change'].iloc[i] > 0:
                    df.loc[df.index[i], 'obv'] = df['obv'].iloc[i-1] + df['volume'].iloc[i]
                elif df['price_change'].iloc[i] < 0:
                    df.loc[df.index[i], 'obv'] = df['obv'].iloc[i-1] - df['volume'].iloc[i]
                else:
                    df.loc[df.index[i], 'obv'] = df['obv'].iloc[i-1]
            
            # Calculate OBV EMAs for trend analysis
            df['obv_ema20'] = df['obv'].ewm(span=20).mean()
            df['obv_ema50'] = df['obv'].ewm(span=50).mean()
            
            # Calculate OBV trend signals
            df['obv_trend'] = 'neutral'
            df.loc[(df['obv'] > df['obv_ema20']) & (df['obv_ema20'] > df['obv_ema50']), 'obv_trend'] = 'bullish'
            df.loc[(df['obv'] < df['obv_ema20']) & (df['obv_ema20'] < df['obv_ema50']), 'obv_trend'] = 'bearish'
            
            # Calculate OBV momentum (5-period rate of change)
            df['obv_momentum'] = df['obv'].pct_change(periods=5)
            
            # Calculate OBV divergence with price
            df['price_momentum'] = df['close'].pct_change(periods=5)
            df['obv_divergence'] = np.where(
                (df['price_momentum'] > 0) & (df['obv_momentum'] < 0), 'bearish',
                np.where((df['price_momentum'] < 0) & (df['obv_momentum'] > 0), 'bullish', 'none')
            )
            
            # Moving Averages
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['sma_200'] = df['close'].rolling(window=200).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).astype('float64')
            loss = (-delta.where(delta < 0, 0)).astype('float64')
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            df['bb_upper'] = df['bb_middle'] + 2 * df['close'].rolling(window=20).std()
            df['bb_lower'] = df['bb_middle'] - 2 * df['close'].rolling(window=20).std()
            
            # Money Flow Index (MFI)
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            money_flow = typical_price * df['volume']
            
            positive_flow = pd.Series(0, index=df.index, dtype='float64')
            negative_flow = pd.Series(0, index=df.index, dtype='float64')
            
            # Calculate positive and negative money flow
            positive_mask = typical_price > typical_price.shift(1)
            negative_mask = typical_price < typical_price.shift(1)
            
            positive_flow[positive_mask] = money_flow[positive_mask]
            negative_flow[negative_mask] = money_flow[negative_mask]
            
            positive_mf = positive_flow.rolling(window=14).sum()
            negative_mf = negative_flow.rolling(window=14).sum()
            
            mfi_ratio = positive_mf / negative_mf
            df['mfi'] = 100 - (100 / (1 + mfi_ratio))
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            return df
    
    def generate_signal(self, df: pd.DataFrame, symbol: str, timeframe: str, mtf_data: Dict) -> Dict:
        """Generate trading signal based on market analysis."""
        try:
            # Get current market conditions
            structure = self.market_analysis.analyze_market_structure(df, symbol, timeframe)
            volume = self.volume_analysis.analyze_volume(df)
            
            # Log market conditions
            logger.info(f"Market conditions for {symbol} ({timeframe}):")
            logger.info(f"Overall trend: {structure.get('market_bias', 'neutral')}")
            logger.info(f"OBV trend: {df['obv_trend'].iloc[-1] if 'obv_trend' in df.columns else 'neutral'}")
            logger.info(f"Volume momentum: {'Bullish' if df['obv_momentum'].iloc[-1] > 0 else 'Bearish' if 'obv_momentum' in df.columns else 'neutral'}")
            
            # Calculate component scores with MTF confidence adjustment
            mtf_confidence = mtf_data.get('confidence_factor', 1.0)
            logger.info(f"MTF Analysis confidence factor: {mtf_confidence:.2f}")
            
            structure_score = self._calculate_structure_score(structure)
            volume_score = self._calculate_volume_score(volume)
            smc_score = self._calculate_smc_score(structure.get('smc'), df)
            mtf_score = self._calculate_mtf_score(mtf_data)
            
            # Adjust score weights based on MTF confidence
            weights = {
                'structure': 0.3,
                'volume': 0.2,
                'smc': 0.2,
                'mtf': 0.3 * mtf_confidence  # Reduce MTF weight if fewer timeframes
            }
            
            # Normalize weights
            total_weight = sum(weights.values())
            weights = {k: v/total_weight for k, v in weights.items()}
            
            # Calculate final score
            final_score = (
                structure_score * weights['structure'] +
                volume_score * weights['volume'] +
                smc_score * weights['smc'] +
                mtf_score * weights['mtf']
            )
            
            # Adjust confidence thresholds based on available data
            confidence_factor = min(mtf_confidence, volume.get('data_quality', 1.0))
            signal_threshold = 0.3 * (1 + confidence_factor)
            strong_threshold = 0.6 * (1 + confidence_factor)
            
            # Determine signal and confidence
            if abs(final_score) >= strong_threshold:
                confidence = 80
                signal_type = 'BUY' if final_score > 0 else 'SELL'
            elif abs(final_score) >= signal_threshold:
                confidence = 60
                signal_type = 'BUY' if final_score > 0 else 'SELL'
            else:
                confidence = 50
                signal_type = 'HOLD'
            
            # Adjust confidence based on data quality
            confidence = int(confidence * confidence_factor)
            
            logger.info(f"Generated {signal_type} signal with {confidence}% confidence")
            logger.debug(f"Score components: structure={structure_score:.2f}, volume={volume_score:.2f}, "
                        f"smc={smc_score:.2f}, mtf={mtf_score:.2f}")
            
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'signal_type': signal_type,
                'confidence': confidence,
                'score': final_score,
                'market_bias': structure.get('market_bias', 'neutral'),
                'components': {
                    'structure': structure_score,
                    'volume': volume_score,
                    'smc': smc_score,
                    'mtf': mtf_score
                },
                'data_quality': {
                    'mtf_confidence': mtf_confidence,
                    'volume_quality': volume.get('data_quality', 1.0),
                    'overall_confidence': confidence_factor
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'signal_type': 'HOLD',
                'confidence': 0,
                'score': 0,
                'market_bias': 'neutral',
                'error': str(e)
            }
    
    def _calculate_trend_score(self, df: pd.DataFrame) -> float:
        """Calculate trend score based on multiple factors."""
        try:
            score = 0
            
            # EMA trend alignment (40%)
            if df['sma_20'].iloc[-1] > df['sma_50'].iloc[-1] > df['sma_200'].iloc[-1]:
                score += 0.4
            elif df['sma_20'].iloc[-1] < df['sma_50'].iloc[-1] < df['sma_200'].iloc[-1]:
                score -= 0.4
                
            # ADX strength (30%)
            adx = df['rsi'].iloc[-1]
            if adx > 25:
                score += 0.3 if df['sma_20'].iloc[-1] > df['sma_50'].iloc[-1] else -0.3
                
            # Price position relative to EMAs (30%)
            current_price = df['close'].iloc[-1]
            if current_price > df['sma_20'].iloc[-1]:
                score += 0.15
            if current_price > df['sma_50'].iloc[-1]:
                score += 0.15
                
            return score
            
        except Exception as e:
            logger.error(f"Error calculating trend score: {str(e)}")
            return 0
    
    def _calculate_structure_score(self, structure: Dict) -> float:
        """Calculate structure score based on market structure analysis."""
        try:
            score = 0
            
            # Market bias (40%)
            if structure['market_bias'] == 'bullish':
                score += 0.4
            elif structure['market_bias'] == 'bearish':
                score -= 0.4
            
            # Order blocks (30%)
            bull_obs = len(structure['order_blocks']['bullish'])
            bear_obs = len(structure['order_blocks']['bearish'])
            if bull_obs > bear_obs:
                score += 0.3
            elif bear_obs > bull_obs:
                score -= 0.3
            
            # Fair value gaps (30%)
            bull_fvgs = len(structure['fair_value_gaps']['bullish'])
            bear_fvgs = len(structure['fair_value_gaps']['bearish'])
            if bull_fvgs > bear_fvgs:
                score += 0.3
            elif bear_fvgs > bull_fvgs:
                score -= 0.3
                
            return score
            
        except Exception as e:
            logger.error(f"Error calculating structure score: {str(e)}")
            return 0
    
    def _calculate_smc_score(self, smc: Optional[Dict], df: pd.DataFrame) -> float:
        """Calculate score based on Smart Money Concepts analysis."""
        try:
            if not smc:
                return 0.0
                
            # Initialize score
            score = 0.0
            
            # Get SMC components with safe defaults
            ob_score = smc.get('order_block_score', 0)
            fvg_score = smc.get('fair_value_gap_score', 0)
            lq_score = smc.get('liquidity_score', 0)
            
            # Combine scores with weights
            score = (ob_score * 0.4 + fvg_score * 0.3 + lq_score * 0.3)
            
            # Ensure score is within [-1, 1]
            return max(min(score, 1.0), -1.0)
            
        except Exception as e:
            logger.error(f"Error calculating SMC score: {str(e)}")
            return 0.0
    
    def _calculate_mtf_score(self, mtf_data: Dict) -> float:
        """Calculate score based on multi-timeframe analysis."""
        try:
            if not mtf_data:
                return 0.0
                
            # Get bias information with safe defaults
            bias = mtf_data.get('overall_bias', {})
            if not bias:
                return 0.0
                
            # Get bias direction and strength
            direction = bias.get('bias', 'neutral')
            strength = bias.get('strength', 'weak')
            adjusted_score = bias.get('adjusted_score', 0.0)
            
            # Calculate base score from adjusted_score
            score = adjusted_score
            
            # Adjust based on strength if adjusted_score is not available
            if adjusted_score == 0.0:
                if direction == 'bullish':
                    score = 0.5 if strength == 'strong' else 0.3 if strength == 'moderate' else 0.1
                elif direction == 'bearish':
                    score = -0.5 if strength == 'strong' else -0.3 if strength == 'moderate' else -0.1
            
            # Ensure score is within [-1, 1]
            return max(min(score, 1.0), -1.0)
            
        except Exception as e:
            logger.error(f"Error calculating MTF score: {str(e)}")
            return 0.0
    
    def _calculate_divergence_score(self, divergences: Dict) -> float:
        """Calculate score based on divergence analysis."""
        try:
            score = 0
            
            # Regular divergences (strongest)
            if divergences['regular']['bullish']:
                score += 0.4
            if divergences['regular']['bearish']:
                score -= 0.4
            
            # Hidden divergences (trend continuation)
            if divergences['hidden']['bullish']:
                score += 0.3
            if divergences['hidden']['bearish']:
                score -= 0.3
            
            # Structural divergences
            if divergences['structural']['bullish']:
                score += 0.2
            if divergences['structural']['bearish']:
                score -= 0.2
            
            # Momentum divergences
            if divergences['momentum']['bullish']:
                score += 0.1
            if divergences['momentum']['bearish']:
                score -= 0.1
            
            return max(min(score, 1.0), -1.0)  # Clamp between -1 and 1
            
        except Exception as e:
            logger.error(f"Error calculating divergence score: {str(e)}")
            return 0
    
    def _calculate_volume_score(self, volume: Dict) -> float:
        """Calculate score based on volume analysis."""
        try:
            # Get volume trend and momentum with safe defaults
            trend = volume.get('trend', 'neutral')
            momentum = volume.get('momentum', 0)
            
            # Initialize score
            score = 0.0
            
            # Score based on trend
            if trend == 'bullish':
                score += 0.5
            elif trend == 'bearish':
                score -= 0.5
                
            # Add momentum component
            score += momentum * 0.5  # Scale momentum to [-0.5, 0.5]
            
            # Ensure score is within [-1, 1]
            return max(min(score, 1.0), -1.0)
            
        except Exception as e:
            logger.error(f"Error calculating volume score: {str(e)}")
            return 0.0
    
    def _get_key_levels(self, df: pd.DataFrame, structure: Dict, smc: Dict, volume: Dict) -> tuple:
        """Get key support and resistance levels from multiple sources."""
        try:
            levels = []
            
            # Add structure levels
            for high in structure['swing_points']['highs']:
                levels.append({
                    'price': high['price'],
                    'type': 'resistance',
                    'source': 'structure',
                    'strength': 1.0
                })
            for low in structure['swing_points']['lows']:
                levels.append({
                    'price': low['price'],
                    'type': 'support',
                    'source': 'structure',
                    'strength': 1.0
                })
            
            # Add SMC levels
            for ob in smc['breaker_blocks']['bullish'][-2:]:
                levels.append({
                    'price': ob['high'],
                    'type': 'resistance',
                    'source': 'breaker',
                    'strength': 1.2
                })
                levels.append({
                    'price': ob['low'],
                    'type': 'support',
                    'source': 'breaker',
                    'strength': 1.2
                })
            
            # Add volume levels
            for level in volume['levels']['support']:
                levels.append({
                    'price': level['price'],
                    'type': 'support',
                    'source': 'volume',
                    'strength': level['strength']
                })
            for level in volume['levels']['resistance']:
                levels.append({
                    'price': level['price'],
                    'type': 'resistance',
                    'source': 'volume',
                    'strength': level['strength']
                })
            
            if not levels:
                return None, None
            
            # Sort levels and find closest support/resistance
            current_price = df['close'].iloc[-1]
            
            # Group nearby levels (within 10 pips)
            grouped_levels = []
            for level in sorted(levels, key=lambda x: x['price']):
                added = False
                for group in grouped_levels:
                    if abs(group[0]['price'] - level['price']) < 0.0010:
                        group.append(level)
                        added = True
                        break
                if not added:
                    grouped_levels.append([level])
            
            # Calculate weighted average price for each group
            consolidated_levels = []
            for group in grouped_levels:
                total_strength = sum(l['strength'] for l in group)
                avg_price = sum(l['price'] * l['strength'] for l in group) / total_strength
                level_type = max(set(l['type'] for l in group), 
                               key=lambda x: sum(1 for l in group if l['type'] == x))
                consolidated_levels.append({
                    'price': avg_price,
                    'type': level_type,
                    'strength': total_strength
                })
            
            # Find closest levels
            support_levels = [l for l in consolidated_levels if l['type'] == 'support' and l['price'] < current_price]
            resistance_levels = [l for l in consolidated_levels if l['type'] == 'resistance' and l['price'] > current_price]
            
            support = max(support_levels, key=lambda x: x['price'])['price'] if support_levels else None
            resistance = min(resistance_levels, key=lambda x: x['price'])['price'] if resistance_levels else None
            
            return support, resistance
            
        except Exception as e:
            logger.error(f"Error getting key levels: {str(e)}")
            return None, None
    
    def _empty_signal(self, df: pd.DataFrame) -> Dict:
        """Return empty signal structure."""
        return {
            "signal_type": "HOLD",
            "confidence": 0.0,
            "current_price": df['close'].iloc[-1] if len(df) > 0 else None,
            "support": None,
            "resistance": None,
            "trend": "neutral",
            "analysis": {
                "reason": "Insufficient data for analysis"
            }
        }
    
    def _determine_signal(self, final_score: float) -> tuple:
        """Determine signal type and confidence based on final score."""
        try:
            abs_score = abs(final_score)
            
            # Strong signal thresholds
            if abs_score > 0.7:
                confidence = min(95, int(abs_score * 100))
                signal = 'BUY' if final_score > 0 else 'SELL'
            # Moderate signal thresholds
            elif abs_score > 0.5:
                confidence = min(75, int(abs_score * 100))
                signal = 'BUY' if final_score > 0 else 'SELL'
            # Weak signal thresholds
            elif abs_score > 0.3:
                confidence = min(50, int(abs_score * 100))
                signal = 'BUY' if final_score > 0 else 'SELL'
            else:
                confidence = int(abs_score * 100)
                signal = 'HOLD'
                
            return signal, confidence
            
        except Exception as e:
            logger.error(f"Error determining signal: {str(e)}")
            return 'HOLD', 0 