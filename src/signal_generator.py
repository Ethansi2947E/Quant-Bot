import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from loguru import logger
import pandas_ta as ta

from config.config import TRADING_CONFIG, SIGNAL_THRESHOLDS, CONFIRMATION_CONFIG
from src.market_analysis import MarketAnalysis
from src.smc_analysis import SMCAnalysis
from src.mtf_analysis import MTFAnalysis
from src.divergence_analysis import DivergenceAnalysis
from src.volume_analysis import VolumeAnalysis

# Timeframe-specific thresholds
TIMEFRAME_THRESHOLDS = {
    'M5': {
        'base_score': 0.65,  # Decreased from 0.80
        'ranging_market': 0.40,  # Decreased from 0.45
        'trending_market': 0.75,  # Decreased from 0.85
        'volatility_filter': 1.3,  # Increased from 1.1
        'min_trend_strength': 0.60,  # Decreased from 0.70
        'rr_ratio': 2.0  # Decreased from 2.2
    },
    'M15': {
        'base_score': 0.75,  # Increased from 0.70 for stronger signals
        'ranging_market': 0.45,  # Increased from 0.40 for better ranging market detection
        'trending_market': 0.85,  # Increased from 0.80 for stronger trend confirmation
        'volatility_filter': 1.15,  # Reduced from 1.2 for tighter volatility control
        'min_trend_strength': 0.70,  # Increased from 0.65 for stronger trends
        'rr_ratio': 2.0  # Reduced from 2.2 for more realistic targets in M15
    },
    'H1': {
        'base_score': 0.65,  # Increased from 0.6
        'ranging_market': 0.35,  # Increased from 0.3
        'trending_market': 0.75,  # Increased from 0.7
        'volatility_filter': 1.3,  # Reduced from 1.4
        'min_trend_strength': 0.60,  # Increased from 0.55
        'rr_ratio': 2.4  # Increased from 2.2
    },
    'H4': {
        'base_score': 0.60,  # Increased from 0.54
        'ranging_market': 0.35,  # Increased from 0.32
        'trending_market': 0.65,  # Increased from 0.58
        'volatility_filter': 1.2,  # Slightly reduced from 1.25
        'min_trend_strength': 0.52,  # Increased from 0.48
        'rr_ratio': 2.8  # Increased from 2.5
    }
}

# Timeframe-specific component weights
TIMEFRAME_WEIGHTS = {
    'M5': {
        'structure': 0.40,  # Reduced from 0.45
        'volume': 0.30,    # Increased from 0.25
        'smc': 0.20,      # Unchanged
        'mtf': 0.10       # Unchanged
    },
    'M15': {
        'structure': 0.40,  # Increased from 0.35 for better price action focus
        'volume': 0.30,    # Reduced from 0.35 to balance
        'smc': 0.20,      # Unchanged
        'mtf': 0.10       # Unchanged
    },
    'H1': {
        'structure': 0.30,  # Reduced from 0.35
        'volume': 0.35,    # Increased from 0.30
        'smc': 0.20,      # Unchanged
        'mtf': 0.15       # Unchanged
    },
    'H4': {
        'structure': 0.35,  # Reduced from 0.40
        'volume': 0.30,    # Increased from 0.25
        'smc': 0.20,      # Unchanged
        'mtf': 0.15       # Unchanged
    }
}

# Signal thresholds
SIGNAL_THRESHOLDS = {
    'strong': 0.75,    # Increased from 0.70
    'moderate': 0.65,  # Increased from 0.60
    'weak': 0.55,      # Increased from 0.50
}

# Currency pair specific multipliers
SYMBOL_MULTIPLIERS = {
    'EURUSD': 1.00,
    'GBPUSD': 0.90,
    'USDJPY': 0.85,
    'AUDUSD': 1.15,  # Reduced from 1.20 to be more conservative
}

# Modified timeframe multipliers
TIMEFRAME_MULTIPLIERS = {
    'H4': 1.15,  # Reduced from 1.20 for more conservative approach
    'H1': 0.85,
}

# Base thresholds
BASE_SCORE_THRESHOLD = 0.32    # Increased from 0.28
RANGING_MARKET_THRESHOLD = 0.42  # Increased from 0.38
TRENDING_MARKET_THRESHOLD = 0.25  # Increased from 0.22

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

# Additional risk management parameters
RISK_MANAGEMENT = {
    'max_daily_trades': 2,          # Reduced from 3
    'max_concurrent_trades': 1,      # Reduced from 2
    'min_trades_spacing': 6,         # Increased from 4
    'max_daily_loss': 0.015,        # Reduced from 0.02
    'max_drawdown_pause': 0.10,     # Reduced from 0.15
    'max_weekly_trades': 8,         # New parameter
    'min_win_rate_continue': 0.40,  # New parameter
    'max_risk_per_trade': 0.015,    # New parameter
    'consecutive_loss_limit': 3,     # New parameter
    'volatility_scaling': True,      # New parameter
    'M15': {
        'max_daily_trades': 4,          # More trades allowed for M15
        'max_concurrent_trades': 1,      # Keep single trade limit
        'min_trades_spacing': 2,         # Reduced spacing for M15 (in hours)
        'max_daily_loss': 0.015,        # Same as default
        'max_drawdown_pause': 0.10,     # Same as default
        'max_weekly_trades': 16,        # Increased for M15
        'consecutive_loss_limit': 2      # More conservative for M15
    }
}

# Additional volatility-based position sizing
VOLATILITY_POSITION_SCALING = {
    'high_volatility': 0.5,    # Reduce position size by 50% in high volatility
    'normal_volatility': 1.0,  # Normal position size
    'low_volatility': 0.75     # Reduce position size by 25% in low volatility
}

# Market condition filters
MARKET_CONDITION_FILTERS = {
    'min_daily_range': 0.0020,  # Minimum daily range in pips
    'max_daily_range': 0.0150,  # Maximum daily range in pips
    'min_volume_threshold': 1000,  # Minimum volume requirement
    'max_spread_threshold': 0.0003,  # Maximum allowed spread
    'M15': {
        'min_daily_range': 0.0015,  # Adjusted for M15
        'max_daily_range': 0.0120,  # Adjusted for M15
        'min_volume_threshold': 800,  # Lower threshold for M15
        'max_spread_threshold': 0.0002  # Tighter spread for M15
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
        
        # Adjusted weights to focus on strong signals
        self.weights = {
            'trend': 0.25,            # Increased from 0.20
            'structure': 0.30,        # Increased from 0.25
            'smc': 0.20,             # Unchanged
            'mtf': 0.15,             # Decreased from 0.20
            'divergence': 0.05,       # Decreased from 0.10
            'volume': 0.05            # Unchanged
        }
        
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
            
            # Calculate ATR
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df['atr'] = true_range.rolling(window=14).mean()
            
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
            if df.empty:
                return self._empty_signal(df)
            
            # Get scores from different components
            structure_analysis = self.market_analysis.analyze(df)
            volume_analysis = self.volume_analysis.analyze(df)
            smc_analysis = self.smc_analysis.analyze(df)
            mtf_analysis = self.mtf_analysis.analyze(df, timeframe)
            
            # Extract numeric scores from analysis results
            structure_score = -1.0 if structure_analysis.get('market_structure', {}).get('market_bias') == 'bearish' else 1.0 if structure_analysis.get('market_structure', {}).get('market_bias') == 'bullish' else 0.0
            volume_score = -1.0 if isinstance(volume_analysis, dict) and volume_analysis.get('trend') == 'bearish' else 1.0 if isinstance(volume_analysis, dict) and volume_analysis.get('trend') == 'bullish' else 0.0
            smc_score = -1.0 if isinstance(smc_analysis, dict) and smc_analysis.get('order_flow', {}).get('bias') == 'bearish' else 1.0 if isinstance(smc_analysis, dict) and smc_analysis.get('order_flow', {}).get('bias') == 'bullish' else 0.0
            mtf_score = float(mtf_analysis) if isinstance(mtf_analysis, (int, float)) else 0.0
            
            # Log analysis scores without format strings
            logger.info("Analysis scores:")
            logger.info(f"Symbol: {symbol}")
            logger.info(f"Timeframe: {timeframe}")
            logger.info(f"Structure Score: {structure_score}")
            logger.info(f"Volume Score: {volume_score}")
            logger.info(f"SMC Score: {smc_score}")
            logger.info(f"MTF Score: {mtf_score}")
            
            # Additional M5-specific filters
            if timeframe == 'M5':
                # Check for high volatility periods
                atr = df['atr'].iloc[-1]
                atr_ma = df['atr'].rolling(window=14).mean().iloc[-1]
                volatility_ratio = atr / atr_ma
                
                # Reject trades during high volatility (increased threshold)
                if volatility_ratio > 1.8:  # Increased from 1.4
                    logger.info("M5 trade rejected: High volatility period")
                    return self._empty_signal(df)
                
                # Check for trend alignment (relaxed conditions)
                ema_20 = df['close'].ewm(span=20).mean().iloc[-1]
                ema_50 = df['close'].ewm(span=50).mean().iloc[-1]
                ema_200 = df['close'].ewm(span=200).mean().iloc[-1]
                
                # More lenient trend alignment check
                if not ((ema_20 > ema_50) or (ema_20 < ema_50)):  # Only check 20 and 50 EMAs
                    logger.info("M5 trade rejected: EMA misalignment")
                    return self._empty_signal(df)
                    
                # Add time-based filters for M5
                hour = df.index[-1].hour
                # Avoid trading during low liquidity hours
                if hour in [22, 23, 0, 1, 2]:
                    logger.info("M5 trade rejected: Low liquidity hours")
                    return self._empty_signal(df)
            
            # Get timeframe-specific weights
            weights = TIMEFRAME_WEIGHTS.get(timeframe, TIMEFRAME_WEIGHTS['H4'])
            
            # Calculate final score with weights
            final_score = (
                structure_score * weights['structure'] +
                volume_score * weights['volume'] +
                smc_score * weights['smc'] +
                mtf_score * weights['mtf']
            )
            
            # Get timeframe-specific thresholds
            thresholds = TIMEFRAME_THRESHOLDS.get(timeframe, TIMEFRAME_THRESHOLDS['H4'])
            
            # Adjusted timeframe-specific multipliers
            timeframe_multiplier = TIMEFRAME_MULTIPLIERS.get(timeframe, 1.0)
            logger.info(f"Timeframe multiplier: {timeframe_multiplier}")
            
            # Symbol-specific adjustments
            symbol_multiplier = SYMBOL_MULTIPLIERS.get(symbol, 1.0)
            logger.info(f"Symbol multiplier: {symbol_multiplier}")
            
            # Calculate MTF confidence with adjusted multiplier
            mtf_confidence = mtf_data.get('confidence_factor', 1.0) * timeframe_multiplier
            logger.info(f"MTF Confidence: {mtf_confidence}")
            
            # Calculate final score
            final_score = (
                structure_score * COMPONENT_WEIGHTS['structure'] +
                volume_score * COMPONENT_WEIGHTS['volume'] +
                smc_score * COMPONENT_WEIGHTS['smc'] +
                mtf_score * COMPONENT_WEIGHTS['mtf']
            ) * timeframe_multiplier * symbol_multiplier
            
            logger.info(f"Final Score: {final_score}")
            
            # Timeframe-specific filters
            if timeframe == 'H1':
                obv_trend = df['obv_trend'].iloc[-1]
                market_bias = self.market_analysis.analyze_market_structure(df, symbol, timeframe).get('market_bias')
                logger.info(f"H1 Filter Check - OBV Trend: {obv_trend}, Market Bias: {market_bias}")
                if not (obv_trend == 'bullish' and market_bias == 'bullish' and final_score > 0.7) and \
                   not (obv_trend == 'bearish' and market_bias == 'bearish' and final_score < -0.7):
                    logger.info("H1 trade rejected: Failed alignment check")
                    return self._empty_signal(df)
            
            # Base threshold
            score_threshold = thresholds['base_score']
            
            # Symbol-specific threshold adjustments
            if symbol in ['GBPUSD', 'GBPJPY', 'GBPCHF']:
                score_threshold *= 1.2  # Higher threshold for GBP pairs
                
                # Additional GBP pairs specific checks
                rsi = df['rsi'].iloc[-1]
                if (final_score > 0 and rsi > 65) or (final_score < 0 and rsi < 35):
                    logger.info(f"{symbol} trade rejected: RSI in extreme zone")
                    return self._empty_signal(df)
            elif symbol in ['USDJPY', 'EURJPY', 'AUDJPY']:
                score_threshold *= 1.1  # Slightly higher threshold for JPY pairs
            elif symbol in ['AUDUSD', 'NZDUSD']:
                score_threshold *= 1.15  # Higher threshold for commodity pairs
            
            # Market condition adjustments
            if abs(structure_score) < thresholds['ranging_market']:
                score_threshold *= 1.3  # More conservative in ranging markets
                logger.info(f"Ranging market detected - Increased threshold to {score_threshold:.2f}")
            
            # Only decrease threshold in very strong trending markets
            if abs(structure_score) > thresholds['trending_market']:
                score_threshold *= 0.85  # Smaller threshold reduction
                logger.info(f"Strong trend - Decreased threshold to {score_threshold:.2f}")
            
            # Volatility checks
            atr = df['atr'].iloc[-1]
            atr_ma = df['atr'].rolling(window=14).mean().iloc[-1]
            volatility_ratio = atr / atr_ma
            logger.info(f"Volatility ratio: {volatility_ratio:.2f}")
            
            if volatility_ratio > thresholds['volatility_filter']:
                score_threshold *= 1.4  # Increased from 1.3
                logger.info(f"High volatility - Increased threshold to {score_threshold:.2f}")
                
                # Additional volatility check for GBPUSD
                if symbol == 'GBPUSD' and volatility_ratio > 1.6:
                    logger.info("GBPUSD trade rejected: Excessive volatility")
                    return self._empty_signal(df)
            
            # Trend strength requirements
            trend_strength = abs(structure_score)
            min_trend_strength = thresholds['min_trend_strength']
            
            if trend_strength < min_trend_strength:
                logger.info(f"Weak trend - Signal rejected (Required: {min_trend_strength})")
                return self._empty_signal(df)
            
            # Additional confluence check for GBPUSD
            if symbol == 'GBPUSD':
                if not (abs(structure_score) > 0.7 and abs(volume_score) > 0.6 and abs(mtf_score) > 0.5):
                    logger.info("GBPUSD trade rejected: Insufficient confluence")
                    return self._empty_signal(df)
            
            # Adjusted confluence requirements for GBPUSD
            if symbol == 'GBPUSD':
                if structure_score > 0.5 and abs(volume_score) > 0.4:  # Reduced from 0.6 and 0.5
                    final_score *= 1.15  # Increased reward
            
            # Enhanced USDJPY signal generation
            if symbol == 'USDJPY':
                if abs(structure_score) > 0.6 and abs(volume_score) > 0.5:  # Increased requirements
                    final_score *= 1.1  # Reduced multiplier
            
            # Adjusted AUDUSD specific logic
            if symbol == 'AUDUSD':
                # More lenient requirements for AUDUSD
                if abs(structure_score) > 0.5 and abs(volume_score) > 0.4:  # Reduced from 0.6 and 0.5
                    final_score *= 1.2  # Increased reward multiplier
                
                # Relaxed volatility filter for AUDUSD
                if volatility_ratio > 1.6:  # Increased from 1.4
                    logger.info(f"High volatility detected for {symbol} - Score adjusted")
                    final_score *= 0.9  # Less penalty for volatility
            
            # Keep EURUSD's existing logic
            if symbol == 'EURUSD':
                if abs(structure_score) > 0.7 and abs(volume_score) > 0.6:
                    final_score *= 1.2
            
            # Relaxed trend strength requirement
            if abs(final_score) < thresholds['ranging_market']:
                logger.info(f"Weak trend - Signal rejected (Required: {thresholds['ranging_market']})")
                return self._empty_signal(df)
            
            # Generate signal based on scores
            signal = {
                'timestamp': df.index[-1],
                'symbol': symbol,
                'timeframe': timeframe,
                'direction': 'HOLD',  # Default to HOLD
                'confidence': 0  # Default confidence
            }
            
            # Signal generation based on final score
            try:
                if final_score > score_threshold:
                    signal['direction'] = 'BUY'
                    signal['entry_price'] = df['close'].iloc[-1]
                    
                    # Dynamic ATR multiplier based on volatility and trend strength
                    trend_strength = 1.0 if structure_score > 0 else -1.0 if structure_score < 0 else 0.0
                    
                    # Base ATR multiplier
                    atr_multiplier = thresholds['rr_ratio']
                    
                    # Adjust for volatility
                    if volatility_ratio > 1.5:
                        atr_multiplier *= 0.8  # Tighter stops in high volatility
                    elif volatility_ratio < 0.7:
                        atr_multiplier *= 1.2  # Wider stops in low volatility
                    
                    # Adjust for trend strength
                    if trend_strength > 0.7:
                        atr_multiplier *= 0.9  # Tighter stops in strong trends
                    
                    # Calculate stop loss and take profit
                    stop_distance = atr * atr_multiplier
                    signal['stop_loss'] = signal['entry_price'] + stop_distance
                    
                    # Dynamic risk:reward based on confidence
                    final_score_value = float(final_score) if isinstance(final_score, (int, float)) else 0.0
                    signal['confidence'] = min(abs(final_score_value) * 100, 100)
                    rr_ratio = 2 + (signal['confidence'] / 100)  # 2:1 to 3:1 based on confidence
                    signal['take_profit'] = signal['entry_price'] - (stop_distance * rr_ratio)
                elif final_score < -score_threshold:
                    signal['direction'] = 'SELL'
                    signal['entry_price'] = df['close'].iloc[-1]
                    
                    # Dynamic ATR multiplier based on volatility and trend strength
                    trend_strength = 1.0 if structure_score > 0 else -1.0 if structure_score < 0 else 0.0
                    
                    # Base ATR multiplier
                    atr_multiplier = thresholds['rr_ratio']
                    
                    # Adjust for volatility
                    if volatility_ratio > 1.5:
                        atr_multiplier *= 0.8  # Tighter stops in high volatility
                    elif volatility_ratio < 0.7:
                        atr_multiplier *= 1.2  # Wider stops in low volatility
                    
                    # Adjust for trend strength
                    if trend_strength > 0.7:
                        atr_multiplier *= 0.9  # Tighter stops in strong trends
                    
                    # Calculate stop loss and take profit
                    stop_distance = atr * atr_multiplier
                    signal['stop_loss'] = signal['entry_price'] + stop_distance
                    
                    # Dynamic risk:reward based on confidence
                    final_score_value = float(final_score) if isinstance(final_score, (int, float)) else 0.0
                    signal['confidence'] = min(abs(final_score_value) * 100, 100)
                    rr_ratio = 2 + (signal['confidence'] / 100)  # 2:1 to 3:1 based on confidence
                    signal['take_profit'] = signal['entry_price'] - (stop_distance * rr_ratio)
                else:
                    final_score_value = float(final_score) if isinstance(final_score, (int, float)) else 0.0
                    signal['confidence'] = min(abs(final_score_value) * 50, 50)  # Lower confidence for HOLD signals
            
            except Exception as e:
                logger.error(f"Error generating signal: {str(e)}")
                return self._empty_signal(df)
            
            logger.info(f"Generated {signal['direction']} signal with {signal['confidence']:.0f}% confidence")
            logger.debug(f"Score components: structure={structure_score:.2f}, volume={volume_score:.2f}, smc={smc_score:.2f}, mtf={mtf_score:.2f}")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error in signal generation: {str(e)}")
            return self._empty_signal(df)
    
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

def validate_signal(confirmations):
    required = 2  # Instead of 3/4
    return sum(confirmations) >= required 