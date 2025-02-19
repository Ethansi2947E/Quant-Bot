import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from loguru import logger
import pandas_ta as ta
from datetime import datetime, UTC
import traceback

from config.config import (
    SIGNAL_CONFIG, RISK_CONFIG, POSITION_CONFIG,
    MARKET_FILTERS, TRADE_EXIT_CONFIG, VOLATILITY_CONFIG,
    RSI_CONFIG, CONFIRMATION_CONFIG
)
from src.market_analysis import MarketAnalysis
from src.smc_analysis import SMCAnalysis
from src.mtf_analysis import MTFAnalysis
from src.divergence_analysis import DivergenceAnalysis
from src.volume_analysis import VolumeAnalysis

class SignalGenerator:
    def __init__(self):
        """Initialize the signal generator with its analysis components."""
        try:
            self.market_analysis = MarketAnalysis()
            self.smc_analysis = SMCAnalysis()
            self.mtf_analysis = MTFAnalysis()
            self.divergence_analysis = DivergenceAnalysis()
            self.volume_analysis = VolumeAnalysis()
            
            # Load configurations from config file
            self.timeframe_thresholds = SIGNAL_CONFIG['timeframe_thresholds']
            self.timeframe_weights = SIGNAL_CONFIG['timeframe_weights']
            self.signal_thresholds = SIGNAL_CONFIG['signal_thresholds']
            self.timeframe_multipliers = SIGNAL_CONFIG['timeframe_multipliers']
            self.base_thresholds = SIGNAL_CONFIG['base_thresholds']
            self.component_weights = SIGNAL_CONFIG['component_weights']
            
            # Load risk and position sizing configs
            self.risk_config = RISK_CONFIG
            self.position_config = POSITION_CONFIG
            self.market_filters = MARKET_FILTERS
            self.trade_exits = TRADE_EXIT_CONFIG
            self.volatility_config = VOLATILITY_CONFIG
            self.rsi_config = RSI_CONFIG
            
        except Exception as e:
            logger.error(f"Error initializing SignalGenerator: {str(e)}")
            raise
        
        self.required_periods = {
            "rsi": 14,
            "macd": 34,
            "atr": 14,
            "ema": 200
        }
        self.max_period = max(self.required_periods.values())
        self.min_confidence = self.signal_thresholds["weak"]
        
        # Initialize analysis components
        self.market_analysis = MarketAnalysis()
        self.smc_analysis = SMCAnalysis()
        self.mtf_analysis = MTFAnalysis()
        self.divergence_analysis = DivergenceAnalysis()
        self.volume_analysis = VolumeAnalysis()
        
        self.scale_factor = 10.0
        
        # Load confirmation weights from config
        self.confirmation_weights = CONFIRMATION_CONFIG["weights"]
        self.min_required_confirmations = CONFIRMATION_CONFIG["min_required"]
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators."""
        try:
            df = df.copy()
            
            # Ensure numeric types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Calculate ATR first
            high = df['high']
            low = df['low']
            close = df['close']
            
            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate ATR with 14 period
            df['atr'] = tr.rolling(window=14).mean()
            df['atr'] = df['atr'].bfill()  # Replace fillna with bfill
            
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
    
    def generate_signal(self, symbol: str, timeframe: str, market_data: Dict) -> Optional[Dict]:
        """Generate trading signal based on market analysis with stricter criteria."""
        try:
            logger.info(f"Generating signals for {symbol} on {timeframe}")
            
            # Early exit conditions
            if market_data.get('structure_type') == 'Transition':
                logger.info("Rejecting signal - Transitional market structure")
                return None
            
            structure_quality = market_data.get('quality_score', 0)
            if structure_quality < MARKET_FILTERS['structure_quality_min']:
                logger.info(f"Rejecting signal - Low structure quality: {structure_quality:.2f}")
                return None
            
            volume_trend = market_data.get('volume_analysis', {}).get('trend_value', 0)
            if abs(volume_trend) < 0.15:  # Minimum volume trend requirement
                logger.info(f"Rejecting signal - Insufficient volume trend: {volume_trend:.2f}")
                return None
            
            # Extract scores with stricter validation
            structure_score = market_data.get('structure_score', 0)
            volume_score = market_data.get('volume_score', 0)
            smc_score = market_data.get('smc_score', 0)
            mtf_score = market_data.get('mtf_score', 0)

            # Require minimum scores for each component
            if any(score < 0.4 for score in [structure_score, smc_score]):  # Stricter minimum for key components
                logger.info("Rejecting signal - Key component scores below threshold")
                return None

            # Calculate final score with adjusted weights
            final_score = (
                structure_score * SIGNAL_CONFIG['component_weights']['structure'] +
                volume_score * SIGNAL_CONFIG['component_weights']['volume'] +
                smc_score * SIGNAL_CONFIG['component_weights']['smc'] +
                mtf_score * SIGNAL_CONFIG['component_weights']['mtf']
            )
            
            # Apply multipliers
            tf_multiplier = self.get_timeframe_multiplier(timeframe)
            final_score *= tf_multiplier
            
            # Determine signal type and confidence
            signal_type, confidence = self._determine_signal(final_score, market_data.get('mtf_analysis', {}))
            
            if confidence < 65:  # Increased minimum confidence threshold
                logger.info(f"Rejecting signal - Confidence too low: {confidence}")
                return None
            
            # Calculate dynamic RR based on market conditions
            volatility_ratio = market_data.get('volatility_ratio', 1.0)
            trend_strength = market_data.get('trend_strength', 0.5)
            base_rr = 2.0  # Increased from 1.5
            dynamic_rr = base_rr * (1 + trend_strength) * volatility_ratio
            dynamic_rr = max(2.0, min(dynamic_rr, 3.5))  # Increased RR range
            
            # Calculate entry, stop loss, and take profit levels
            current_price = market_data['data']['close'].iloc[-1]
            atr = market_data['data']['atr'].iloc[-1]
            
            # Set fixed risk-reward ratio of 2:1
            sl_distance = 2.0 * atr  # Fixed SL distance
            tp_distance = 4.0 * atr  # Fixed TP distance (2:1 RR)
            
            stop_loss = current_price - sl_distance if signal_type == 'BUY' else current_price + sl_distance
            take_profit = current_price + tp_distance if signal_type == 'BUY' else current_price - tp_distance
            
            # Create signal with all required fields
            signal = {
                'symbol': symbol,
                'timeframe': timeframe,
                'direction': signal_type,
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'confidence': confidence,
                'timestamp': datetime.now(UTC).isoformat(),
                'risk_reward_ratio': 2.0,  # Fixed RR ratio
                'structure_quality': structure_quality,
                'volume_trend': volume_trend,
                'atr': atr,
                'volatility_ratio': volatility_ratio,
                'trend_strength': trend_strength
            }
            
            # Add debug logging for signal validation
            logger.debug(f"Generated signal before validation: {signal}")
            
            # Validate all required fields are present and have valid values
            required_fields = [
                'symbol', 
                'timeframe', 
                'direction', 
                'entry_price', 
                'stop_loss', 
                'take_profit',
                'confidence'
            ]
            
            # Check for missing fields
            missing_fields = [field for field in required_fields if field not in signal]
            if missing_fields:
                logger.error(f"Generated signal missing required fields: {missing_fields}")
                logger.error(f"Signal content: {signal}")
                return None
                
            # Validate numeric fields
            numeric_fields = ['entry_price', 'stop_loss', 'take_profit', 'confidence']
            for field in numeric_fields:
                if not isinstance(signal[field], (int, float)) or signal[field] <= 0:
                    if field != 'confidence':  # Confidence can be between 0 and 1
                        logger.error(f"Invalid {field} value: {signal[field]}")
                        return None
                    elif not 0 <= signal[field] <= 1:  # Validate confidence range
                        logger.error(f"Invalid confidence value: {signal[field]}")
                        return None
            
            # Validate direction
            if signal['direction'] not in ['BUY', 'SELL']:
                logger.error(f"Invalid signal direction: {signal['direction']}")
                return None
                
            # Validate string fields are not empty
            string_fields = ['symbol', 'timeframe']
            for field in string_fields:
                if not signal[field] or not isinstance(signal[field], str):
                    logger.error(f"Invalid {field}: {signal[field]}")
                    return None
            
            # Log the complete signal for debugging
            logger.debug(f"Generated valid signal: {signal}")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
            logger.error(f"Stack trace:\n{traceback.format_exc()}")
            return None
    
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
            market_bias = structure.get('market_bias', 'neutral')
            if market_bias == 'bullish':
                score += 0.4
            elif market_bias == 'bearish':
                score -= 0.4
            
            # Order blocks (30%)
            order_blocks = structure.get('order_blocks', {'bullish': [], 'bearish': []})
            bull_obs = len(order_blocks.get('bullish', []))
            bear_obs = len(order_blocks.get('bearish', []))
            if bull_obs > bear_obs:
                score += 0.3
            elif bear_obs > bull_obs:
                score -= 0.3
            
            # Fair value gaps (30%)
            fvg = structure.get('fair_value_gaps', {'bullish': [], 'bearish': []})
            bull_fvgs = len(fvg.get('bullish', []))
            bear_fvgs = len(fvg.get('bearish', []))
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
    
    def _empty_signal(self, df, symbol, timeframe):
        """Generate an empty HOLD signal with default values."""
        return {
            'entry_time': df.index[-1],  # Changed from timestamp to entry_time
            'symbol': symbol,
            'timeframe': timeframe,
            'direction': 'HOLD',
            'confidence': 0,
            'entry_price': df['close'].iloc[-1],
            'stop_loss': None,
            'take_profit': None
        }
    
    def _determine_signal(self, final_score: float, mtf_analysis: Dict) -> tuple:
        """Determine signal type and confidence with MTF confirmation."""
        try:
            abs_score = abs(final_score)
            mtf_bias = mtf_analysis.get('overall_bias', {}).get('bias', 'neutral')
            mtf_strength = mtf_analysis.get('overall_bias', {}).get('strength', 'weak')
            
            # Adjust score based on MTF alignment
            if mtf_bias == 'bullish' and final_score > 0:
                abs_score *= 1.2  # Boost score for aligned bias
            elif mtf_bias == 'bearish' and final_score < 0:
                abs_score *= 1.2
            
            # Apply MTF strength modifier
            strength_multiplier = {
                'strong': 1.2,
                'moderate': 1.1,
                'weak': 1.0
            }.get(mtf_strength, 1.0)
            
            abs_score *= strength_multiplier
            
            if abs_score > SIGNAL_CONFIG['signal_thresholds']["strong"]:
                confidence = min(95, int(abs_score * 100))
                signal = 'BUY' if final_score > 0 else 'SELL'
            elif abs_score > SIGNAL_CONFIG['signal_thresholds']["moderate"]:
                confidence = min(85, int(abs_score * 100))
                signal = 'BUY' if final_score > 0 else 'SELL'
            elif abs_score > SIGNAL_CONFIG['signal_thresholds']["weak"]:
                confidence = min(75, int(abs_score * 100))
                signal = 'BUY' if final_score > 0 else 'SELL'
            else:
                confidence = int(abs_score * 100)
                signal = 'HOLD'
                
            return signal, confidence
            
        except Exception as e:
            logger.error(f"Error determining signal: {str(e)}")
            return 'HOLD', 0
    
    def get_timeframe_multiplier(self, timeframe: str) -> float:
        """Get the multiplier for a given timeframe.
        
        Higher timeframes get higher multipliers as they are more significant.
        """
        multipliers = {
            'M5': 0.75,   # Reduced weight for very short timeframe
            'M15': 0.85,  # Slightly reduced weight for short timeframe
            'H1': 1.0,   # Increased weight for medium timeframe
            'H4': 1.25    # Higher weight for longer timeframe
        }
        return multipliers.get(timeframe, 1.0)
    
    def get_symbol_multiplier(self, symbol: str) -> float:
        """Get the multiplier for a given symbol. All symbols are treated equally."""
        return 1.0  # Return constant value for all pairs

    def adjust_timeframe_weights(self, market_conditions: Dict) -> Dict:
        """Adjust timeframe weights based on market conditions."""
        base_weights = SIGNAL_CONFIG['timeframe_weights'].copy()
        volatility = market_conditions.get('volatility', 'normal')
        
        if volatility == 'high':
            # Increase weight of higher timeframes in high volatility
            base_weights['H4'] *= 1.2
            base_weights['H1'] *= 1.1
            base_weights['M5'] *= 0.8
        elif volatility == 'low':
            # Increase weight of lower timeframes in low volatility
            base_weights['M5'] *= 1.2
            base_weights['M15'] *= 1.1
            base_weights['H4'] *= 0.8
        
        return base_weights

def validate_signal(confirmations):
    """Validate signal with stricter confirmation requirements."""
    required = 3  # Increased from 2
    strong_confirmations = sum(1 for conf in confirmations if conf > 0.7)  # Count strong confirmations
    total_confirmations = sum(confirmations)
    
    # Require both minimum number of confirmations and at least one strong confirmation
    return total_confirmations >= required and strong_confirmations >= 1 