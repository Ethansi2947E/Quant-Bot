import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from loguru import logger
import pandas_ta as ta
from datetime import datetime, UTC

from config.config import TRADING_CONFIG, SIGNAL_THRESHOLDS, CONFIRMATION_CONFIG
from src.market_analysis import MarketAnalysis
from src.smc_analysis import SMCAnalysis
from src.mtf_analysis import MTFAnalysis
from src.divergence_analysis import DivergenceAnalysis
from src.volume_analysis import VolumeAnalysis

# Timeframe-specific thresholds
TIMEFRAME_THRESHOLDS = {
    'M5': {
        'base_score': 0.65,       # Decreased from 0.75
        'ranging_market': 0.45,   # Decreased from 0.50
        'trending_market': 0.75,  # Decreased from 0.85
        'volatility_filter': 1.2, # Increased from 1.1
        'min_trend_strength': 0.60, # Decreased from 0.70
        'rr_ratio': 2.0,         # Decreased from 2.2
        'min_confirmations': 2    # Decreased from 3
    },
    'M15': {
        'base_score': 0.15,       # Decreased from 0.35
        'ranging_market': 0.15,   # Decreased from 0.25
        'trending_market': 0.25,  # Decreased from 0.45
        'volatility_filter': 1.2, # Decreased from 1.5
        'min_trend_strength': 0.15, # Decreased from 0.25
        'rr_ratio': 1.0,         # Decreased from 1.2
        'min_confirmations': 1    # Keep at 1
    },
    'H1': {
        'base_score': 0.65,      # Decreased from 0.75
        'ranging_market': 0.40,  # Decreased from 0.45
        'trending_market': 0.75, # Decreased from 0.85
        'volatility_filter': 1.2, # Increased from 1.1
        'min_trend_strength': 0.60, # Decreased from 0.70
        'rr_ratio': 2.4,        # Decreased from 2.6
        'min_confirmations': 2   # Decreased from 3
    },
    'H4': {
        'base_score': 0.65,      # Decreased from 0.70
        'ranging_market': 0.40,  # Decreased from 0.45
        'trending_market': 0.70, # Decreased from 0.75
        'volatility_filter': 1.1, # Increased from 1.0
        'min_trend_strength': 0.60, # Decreased from 0.65
        'rr_ratio': 2.8,        # Decreased from 3.0
        'min_confirmations': 2   # Decreased from 3
    }
}

# Timeframe-specific component weights
TIMEFRAME_WEIGHTS = {
    'M5': {
        'structure': 0.35,  # Decreased from 0.40
        'volume': 0.25,    # Decreased from 0.30
        'smc': 0.25,       # Increased from 0.20
        'mtf': 0.15        # Increased from 0.10
    },
    'M15': {
        'structure': 0.35,  # Decreased from 0.40
        'volume': 0.25,    # Decreased from 0.30
        'smc': 0.25,       # Increased from 0.20
        'mtf': 0.15        # Increased from 0.10
    },
    'H1': {
        'structure': 0.30,
        'volume': 0.30,    # Decreased from 0.35
        'smc': 0.25,       # Increased from 0.20
        'mtf': 0.15
    },
    'H4': {
        'structure': 0.35,
        'volume': 0.25,    # Decreased from 0.30
        'smc': 0.25,       # Increased from 0.20
        'mtf': 0.15
    }
}

# Signal thresholds with more lenient criteria
SIGNAL_THRESHOLDS = {
    'strong': 0.45,    # Decreased from 0.65
    'moderate': 0.35,  # Decreased from 0.55
    'weak': 0.25,      # Decreased from 0.45
    'minimum': 0.15    # Decreased from 0.35
}

# Currency pair specific multipliers - Adjusted for EURUSD and AUDUSD
SYMBOL_MULTIPLIERS = {
    'EURUSD': 1.20,    # Increased from 1.10
    'GBPUSD': 0.90,
    'USDJPY': 0.85,
    'AUDUSD': 1.35,    # Increased from 1.25
}

# Modified timeframe multipliers
TIMEFRAME_MULTIPLIERS = {
    'H4': 1.15,  # Reduced from 1.20 for more conservative approach
    'H1': 0.85,
}

# Base thresholds - More lenient
BASE_SCORE_THRESHOLD = 0.15    # Decreased from 0.22
RANGING_MARKET_THRESHOLD = 0.15  # Decreased from 0.32
TRENDING_MARKET_THRESHOLD = 0.12  # Decreased from 0.18

# Adjusted component weights
COMPONENT_WEIGHTS = TIMEFRAME_WEIGHTS['H4']  # Default to H4 weights

# Volatility thresholds
VOLATILITY_THRESHOLDS = {
    'EURUSD': {
        'H4': 1.5,
        'H1': 1.4,
        'M15': 1.25,  # Added specific M15 threshold
        'M5': 1.2
    },
    'GBPUSD': 1.8,
    'USDJPY': 1.4,
    'AUDUSD': 1.35,  # Reduced from 1.45 for tighter volatility control
}

# RSI thresholds
RSI_THRESHOLDS = {
    'EURUSD': {'overbought': 70, 'oversold': 30},
    'GBPUSD': {'overbought': 78, 'oversold': 22},
    'USDJPY': {'overbought': 75, 'oversold': 25},
    'AUDUSD': {'overbought': 78, 'oversold': 22},  # More extreme levels for stronger confirmation
}

# Risk management parameters - Adjusted for more trades
RISK_MANAGEMENT = {
    'max_daily_trades': 4,          # Increased from 3
    'max_concurrent_trades': 2,      # Kept same
    'min_trades_spacing': 1,         # Decreased from 2
    'max_daily_loss': 0.015,        # Kept same
    'max_drawdown_pause': 0.05,     # Kept same
    'max_weekly_trades': 16,         # Increased from 12
    'min_win_rate_continue': 0.30,  # Decreased from 0.35
    'max_risk_per_trade': 0.01,     # Kept same
    'consecutive_loss_limit': 4,     # Increased from 3
    'volatility_scaling': True,      
    'partial_tp_enabled': True,      
    'recovery_mode': {
        'enabled': True,
        'drawdown_trigger': 0.05,    
        'position_size_reduction': 0.5,
        'min_wins_to_exit': 2
    },
    'M15': {
        'max_daily_trades': 5,       # Increased from 4
        'max_concurrent_trades': 2,   # Kept same
        'min_trades_spacing': 1,     # Kept same
        'max_daily_loss': 0.015,     
        'max_drawdown_pause': 0.05,  
        'max_weekly_trades': 20,     # Increased from 16
        'consecutive_loss_limit': 4   # Increased from 3
    }
}

# Updated position sizing based on volatility
VOLATILITY_POSITION_SCALING = {
    'high_volatility': 0.5,     # 50% size in high volatility
    'normal_volatility': 1.0,   # Normal position size
    'low_volatility': 0.75,     # 75% size in low volatility
    'atr_multipliers': {
        'high': 1.5,            # ATR threshold for high volatility
        'low': 0.5              # ATR threshold for low volatility
    }
}

# Market condition filters - More lenient
MARKET_CONDITION_FILTERS = {
    'min_daily_range': 0.0005,      # Decreased from 0.0008
    'max_daily_range': 0.0180,      # Increased from 0.0160
    'min_volume_threshold': 200,    # Decreased from 400
    'max_spread_threshold': 0.0006, # Increased from 0.0005
    'correlation_threshold': 0.60,   # Decreased from 0.70
    'trend_strength_min': 0.25,     # Decreased from 0.35
    'volatility_percentile': 0.05,  # Decreased from 0.10
    'momentum_threshold': 0.005,    # Decreased from 0.008
    'M15': {
        'min_daily_range': 0.0004,  # Decreased from 0.0006
        'max_daily_range': 0.0160,  # Increased from 0.0140
        'min_volume_threshold': 100, # Decreased from 200
        'max_spread_threshold': 0.0006, # Increased from 0.0005
        'min_confirmations': 1      # Keep at 1
    }
}

# Take profit and stop loss configuration
TRADE_EXITS = {
    'partial_tp_ratio': 0.5,        # Exit 50% at first target
    'tp_levels': [
        {'ratio': 1.0, 'size': 0.5}, # First TP at 1R with 50% size
        {'ratio': 2.0, 'size': 0.5}  # Second TP at 2R with remaining
    ],
    'trailing_stop': {
        'enabled': True,
        'activation_ratio': 1.0,     # Start trailing at 1R profit
        'trail_points': 0.5          # Trail by 0.5R
    }
}

class SignalGenerator:
    def __init__(self):
        """Initialize the signal generator with its analysis components."""
        try:
            self.market_analysis = MarketAnalysis()
            self.smc_analysis = SMCAnalysis()
            self.mtf_analysis = MTFAnalysis()
            self.divergence_analysis = DivergenceAnalysis()
            self.volume_analysis = VolumeAnalysis()
            self.config = TRADING_CONFIG
            self.thresholds = SIGNAL_THRESHOLDS
            self.confirmation = CONFIRMATION_CONFIG
        except Exception as e:
            logger.error(f"Error initializing SignalGenerator: {str(e)}")
            raise
        
        self.required_periods = {
            "rsi": 14,
            "macd": 34,  # Longest period needed for MACD (26 + 8 buffer)
            "atr": 14,
            "ema": 200  # Longest EMA period
        }
        self.max_period = max(self.required_periods.values())
        self.min_confidence = SIGNAL_THRESHOLDS["weak"]  # Use new threshold from config
        
        # Initialize analysis components
        self.market_analysis = MarketAnalysis()
        self.smc_analysis = SMCAnalysis()
        self.mtf_analysis = MTFAnalysis()
        self.divergence_analysis = DivergenceAnalysis()
        self.volume_analysis = VolumeAnalysis()
        
        # Adjusted weights to focus on SMC and structure
        self.weights = {
            'structure': 0.35,
            'volume': 0.25,
            'smc': 0.25,
            'mtf': 0.15
        }
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
        """Generate trading signal based on market analysis."""
        try:
            logger.info(f"Generating signals for {symbol} on {timeframe}")
            
            current_price = market_data['data']['close'].iloc[-1]
            atr = market_data['data']['atr'].iloc[-1]
            
            # Dynamic RR based on market conditions and signal strength
            volatility_ratio = market_data.get('volatility_ratio', 1.0)
            trend_strength = market_data.get('trend_strength', 0.5)
            
            # Base RR starts at 1.5:1 and can scale up to 3:1
            base_rr = 1.5
            dynamic_rr = base_rr * (1 + trend_strength) * volatility_ratio
            # Adjust the dynamic RR calculation to be more responsive to market conditions
            dynamic_rr = max(1.5, min(dynamic_rr, 3.0))  # Cap between 1.5 and 3.0
            logger.debug(f"Dynamic RR: {dynamic_rr:.2f}")
            
            # Calculate dynamic stop loss and take profit distances
            sl_distance = 2 * atr  # Base SL distance
            tp_distance = sl_distance * dynamic_rr  # TP distance based on dynamic RR
            
            # Calculate stop loss and take profit levels
            stop_loss = current_price - sl_distance if market_data.get('direction') == 'BUY' else current_price + sl_distance
            take_profit = current_price + tp_distance if market_data.get('direction') == 'BUY' else current_price - tp_distance
            
            signal = {
                'symbol': symbol,
                'timeframe': timeframe,
                'direction': 'HOLD',
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'timestamp': datetime.now(UTC).isoformat(),
                'confidence': 0,
                'risk_reward_ratio': dynamic_rr
            }
            
            # Extract scores from market data
            structure_score = market_data.get('structure_score', 0)
            volume_score = market_data.get('volume_score', 0)
            smc_score = market_data.get('smc_score', 0)
            mtf_score = market_data.get('mtf_score', 0)

            logger.debug(f"Component Scores: Structure={structure_score:.2f}, Volume={volume_score:.2f}, SMC={smc_score:.2f}, MTF={mtf_score:.2f}")
            
            # Apply timeframe and symbol multipliers
            tf_multiplier = self.get_timeframe_multiplier(timeframe)
            symbol_multiplier = self.get_symbol_multiplier(symbol)
            mtf_confidence = market_data.get('mtf_confidence', 1.0)

            logger.debug(f"Multipliers: Timeframe={tf_multiplier:.2f}, Symbol={symbol_multiplier:.2f}, MTF_Confidence={mtf_confidence:.2f}")
            
            # Calculate final score using weighted sum of components
            final_score = (
                structure_score * self.weights['structure'] +
                volume_score * self.weights['volume'] +
                smc_score * self.weights['smc'] +
                mtf_score * self.weights['mtf']
            ) * tf_multiplier * symbol_multiplier * mtf_confidence * self.scale_factor
            
            logger.info(f"Final Score: {final_score:.3f}")

            # More permissive base threshold
            base_threshold = 0.02
            
            # Adjust threshold based on market conditions
            if market_data.get('is_ranging', False):
                base_threshold *= 1.2
                logger.info(f"Ranging market detected - Increased threshold to {base_threshold:.2f}")
            
            # Check volatility
            logger.info(f"Volatility ratio: {volatility_ratio}")
            
            if abs(final_score) < base_threshold:
                logger.info(f"Weak signal - Score {final_score:.2f} below threshold {base_threshold:.2f}")
                return None
                
            # Generate signal
            signal['direction'] = 'buy' if final_score > 0 else 'sell'
            signal['strength'] = abs(final_score)
            signal['confidence'] = min(95, int(abs(final_score) * 100))
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
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
    
    def _determine_signal(self, final_score: float) -> tuple:
        """Determine signal type and confidence based on final score."""
        try:
            abs_score = abs(final_score)
            
            # Strong signal thresholds
            if abs_score > SIGNAL_THRESHOLDS["strong"]:
                confidence = min(95, int(abs_score * 100))
                signal = 'BUY' if final_score > 0 else 'SELL'
            # Moderate signal thresholds
            elif abs_score > SIGNAL_THRESHOLDS["moderate"]:
                confidence = min(75, int(abs_score * 100))
                signal = 'BUY' if final_score > 0 else 'SELL'
            # Weak signal thresholds
            elif abs_score > SIGNAL_THRESHOLDS["weak"]:
                confidence = min(50, int(abs_score * 100))
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
            'M5': 0.85,   # Reduced weight for very short timeframe
            'M15': 0.95,  # Slightly reduced weight for short timeframe
            'H1': 1.15,   # Increased weight for medium timeframe
            'H4': 1.25    # Higher weight for longer timeframe
        }
        return multipliers.get(timeframe, 1.0)
    
    def get_symbol_multiplier(self, symbol: str) -> float:
        """Get the multiplier for a given symbol.
        
        Different symbols may have different characteristics that affect signal reliability.
        """
        multipliers = {
            'EURUSD': 1.0,  # Base pair - standard multiplier
            'GBPUSD': 0.95, # Slightly reduced due to higher volatility
            'USDJPY': 0.85, # Reduced due to specific characteristics
            'AUDUSD': 0.9,  # Reduced due to commodity influence
            'USDCAD': 0.9,  # Reduced due to commodity influence
            'NZDUSD': 0.9,  # Reduced due to commodity influence
            'EURJPY': 0.85, # Cross rate with reduced multiplier
            'GBPJPY': 0.8,  # Most volatile cross - lowest multiplier
            'EURGBP': 0.9   # Cross rate with moderate multiplier
        }
        return multipliers.get(symbol, 0.9)  # Default to 0.9 for unknown symbols

def validate_signal(confirmations):
    required = 2  # Instead of 3/4
    return sum(confirmations) >= required 