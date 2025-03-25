import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import sys
import os
from loguru import logger
import traceback

# Import centralized configuration
from config.config import TRADING_CONFIG

# Importing existing modules with correct src. prefix
from src.risk_manager import RiskManager
from src.mt5_handler import MT5Handler
from src.analysis.mtf_analysis import MTFAnalysis
from src.poi_detector import POIDetector
from src.analysis.volume_analysis import VolumeAnalysis
from src.technical_indicators import TechnicalIndicators
from src.pattern_detector import PatternDetector
from src.utils.signal_processor import SignalProcessor
# Remove direct import of MarketAnalysis to avoid circular imports
# from src.analysis.market_analysis import MarketAnalysis

# Create logs directory if it doesn't exist
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Configure loguru logger
logger_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>SG123:{function}:{line}</cyan> | <level>{message}</level>"

# Configure loguru with custom format
logger.configure(handlers=[
    {"sink": sys.stdout, "format": logger_format, "level": "DEBUG", "colorize": True},
    {"sink": os.path.join(log_dir, "signal_generator123_detailed.log"), 
     "format": logger_format, "level": "DEBUG", "rotation": "10 MB", 
     "retention": "1 week", "compression": "zip"}
])

# Add context to differentiate this logger
logger = logger.bind(name="signal_generator123")

logger.info("[SG123] SignalGenerator123 logger initialized")
logger.info(f"[SG123] Detailed logs will be written to {os.path.join(log_dir, 'signal_generator123_detailed.log')}")

class SignalGenerator123:
    def __init__(self, mt5_handler=None, config=None, risk_manager=None):
        """Initialize SignalGenerator123."""
        logger.debug("Initializing SignalGenerator123")
        self.mt5_handler = mt5_handler if mt5_handler is not None else MT5Handler()
        self.risk_manager = risk_manager if risk_manager is not None else RiskManager(self.mt5_handler)
        
        # Import global config using centralized configuration
        global_config = TRADING_CONFIG
        logger.debug("Using centralized trading configuration")
            
        # Load default configuration
        self.config = {
            "symbols": TRADING_CONFIG.get("symbols", ["EURUSD", "GBPUSD", "USDJPY"]),
            "timeframes": TRADING_CONFIG.get("timeframes", ["M15", "H1", "H4"]),
            "max_risk_per_trade": TRADING_CONFIG.get("max_risk_per_trade", 0.01),
            "min_rr": TRADING_CONFIG.get("min_rr", 2.0),
            "confirmation_timeframe": TRADING_CONFIG.get("confirmation_timeframe", "H1"),
            "min_trend_strength": TRADING_CONFIG.get("min_trend_strength", 0.6),
            "max_correlation": TRADING_CONFIG.get("max_correlation", 0.7),
            "use_volume_filtering": TRADING_CONFIG.get("use_volume_filtering", True),
            "adaptive_parameters": TRADING_CONFIG.get("adaptive_parameters", True),
            "atr_period": TRADING_CONFIG.get("atr_period", 14),
            "williams_period": TRADING_CONFIG.get("williams_period", 14),
            "confirmation_bars": TRADING_CONFIG.get("confirmation_bars", 2),
            "min_volatility": TRADING_CONFIG.get("min_volatility", 10),  # Default minimum volatility in pips
            "fib_levels": TRADING_CONFIG.get("fib_levels", [0.236, 0.382, 0.5, 0.618, 0.786])  # Default Fibonacci levels
        }
        
        # Override with provided configuration if any
        if config:
            self.config.update(config)
            
        # Initialize risk manager
        self.risk_manager = risk_manager if risk_manager else RiskManager(mt5_handler=self.mt5_handler)
        
        # Lazy load market analysis to avoid circular imports
        self._market_analysis = None
        
        # Initialize other components as needed
        self.mtf_analyzer = MTFAnalysis()
        self.poi_detector = POIDetector()
        self.volume_analyzer = VolumeAnalysis()
        self.indicators = TechnicalIndicators()
        self.pattern_detector = PatternDetector(self.indicators, self.config)
        
        # Initialize default configuration values
        self.min_rr = self.config.get("min_rr", 2.0)
        self.min_trend_strength = self.config.get("min_trend_strength", 0.6)
        self.use_volume_filtering = self.config.get("use_volume_filtering", True)
        self.max_correlation = self.config.get("max_correlation", 0.7)
        self.adaptive_parameters = self.config.get("adaptive_parameters", True)
        
        # For tracking performance and statistics
        self.last_signal_time = {}
        self.signal_performance = {}
        self.signals_generated = 0
        self.signals_filtered = 0
        
        logger.info("SignalGenerator123 initialized successfully")

    @property
    def market_analysis(self):
        """Lazy load the MarketAnalysis to avoid circular imports."""
        if self._market_analysis is None:
            # Import MarketAnalysis here to avoid circular imports
            from src.analysis.market_analysis import MarketAnalysis
            self._market_analysis = MarketAnalysis()
        return self._market_analysis

    async def generate_signals(self, market_data: Dict, symbol: str, timeframe: str,
                              account_info: Optional[Dict] = None, **kwargs) -> List[Dict]:
        """Generate trading signals based on 123 Reversal/Continuation strategy."""
        logger.debug(f"Generating signals for {symbol} on {timeframe}")
        # Extract analyzed_data if available in kwargs
        analyzed_data = kwargs.get('analyzed_data')
        signals = []
        
        # Try to get market data from different sources
        df = None
        
        # 1. First check if we have pre-analyzed data we can use
        if analyzed_data and timeframe in analyzed_data:
            logger.debug(f"Using pre-analyzed data for {symbol} {timeframe}")
            # If analysis contains the data we need, extract it
            if 'df' in analyzed_data[timeframe]:
                df = analyzed_data[timeframe]['df']
            # Otherwise proceed with normal data sources
        
        # 2. Try using the provided market_data if no pre-analyzed data
        if df is None and market_data and symbol in market_data and timeframe in market_data[symbol]:
            df = market_data[symbol][timeframe]
            logger.debug(f"Using provided market data for {symbol} {timeframe}")
        
        # 3. If no data from market_data, try using mt5_handler directly
        elif df is None and self.mt5_handler:
            logger.debug(f"Fetching market data via mt5_handler for {symbol} {timeframe}")
            df = self.mt5_handler.get_market_data(symbol, timeframe, num_candles=1000)
        
        # 4. If no mt5_handler, try using the risk_manager's mt5_handler
        elif df is None and self.risk_manager and hasattr(self.risk_manager, 'mt5_handler') and self.risk_manager.mt5_handler:
            logger.debug(f"Fetching market data via risk_manager.mt5_handler for {symbol} {timeframe}")
            df = self.risk_manager.mt5_handler.get_market_data(symbol, timeframe, num_candles=1000)
        
        # No data available from any source
        if df is None or len(df) < 50:
            logger.warning(f"Insufficient or no data available for {symbol} on {timeframe}")
            return signals

        # Ensure symbol is stored in the dataframe
        if "symbol" not in df.columns:
            df["symbol"] = symbol
            df.name = symbol  # Some methods use df.name for symbol

        # Store current symbol in class context for other methods to use
        self.current_symbol = symbol
        
        # Calculate initial volatility metrics to determine adaptive parameters
        self._set_adaptive_parameters(df, symbol)

        # Add technical indicators using the TechnicalIndicators module
        df = self._add_indicators(df)
        
        # Update volatility context using calculated ATR
        if "atr" in df.columns:
            atr_value = df["atr"].iloc[-1]
            self.current_atr_pips = atr_value * 10000  # Convert to pips for easier use in other methods
            
            # Classify current volatility
            if self.current_atr_pips > 25:  # High volatility threshold
                self.recent_volatility = 'high'
            elif self.current_atr_pips < 10:  # Low volatility threshold
                self.recent_volatility = 'low'
            else:
                self.recent_volatility = 'normal'
                
            logger.debug(f"Current volatility for {symbol}: {self.recent_volatility} (ATR: {self.current_atr_pips:.1f} pips)")

        # Detect swing points for 123 pattern using PatternDetector instead of MarketAnalysis
        pattern_params = {"lookback": 20, "threshold": 0.0015}
        swing_points = self.pattern_detector.detect_swing_points(df, pattern_params)
        if not swing_points["highs"] or not swing_points["lows"]:
            logger.debug(f"No swing points detected for {symbol}")
            return signals

        # Multi-timeframe trend alignment (H1 for M15 entries)
        higher_tf = "H1" if timeframe == "M15" else None
        trend_aligned = await self._check_trend_alignment(symbol, higher_tf) if higher_tf else True
        if not trend_aligned:
            logger.debug(f"MTF trend not aligned for {symbol}")
            return signals

        # Check for strong trend with ADX
        if not self._validate_trend_strength(df):
            logger.debug(f"Trend strength insufficient for {symbol} (ADX below threshold)")
            return signals

        # Check volume and volatility
        if not self._validate_volume_and_volatility(df):
            logger.debug(f"Volume/volatility conditions not met for {symbol}")
            return signals

        # Detect 123 Reversal pattern
        reversal_signal = self._detect_123_reversal(df, swing_points)
        if reversal_signal:
            logger.info(f"123 Reversal pattern detected on {symbol} {timeframe}")
            # Ensure signal has the correct symbol
            if reversal_signal.get("symbol") == "UNKNOWN":
                reversal_signal["symbol"] = symbol
            signals.append(reversal_signal)

        # Detect 123 Continuation pattern
        continuation_signal = self._detect_123_continuation(df, swing_points)
        if continuation_signal:
            logger.info(f"123 Continuation pattern detected on {symbol} {timeframe}")
            # Ensure signal has the correct symbol
            if continuation_signal.get("symbol") == "UNKNOWN":
                continuation_signal["symbol"] = symbol
            signals.append(continuation_signal)

        logger.debug(f"Generated {len(signals)} signals for {symbol} on {timeframe}")
        
        # Process signals through SignalProcessor if available
        if signals:
            try:
                # Initialize SignalProcessor
                signal_processor = SignalProcessor(mt5_handler=self.mt5_handler, risk_manager=self.risk_manager)
                
                # Initialize the signal processor if needed
                await signal_processor.initialize()
                
                # Process the signals
                logger.info(f"Processing {len(signals)} signals via SignalProcessor")
                result = await signal_processor.process_signals(signals)
                logger.debug(f"Signal processing result: {result}")
            except Exception as e:
                logger.error(f"Error processing signals with SignalProcessor: {str(e)}")
                logger.error(traceback.format_exc())
        
        return signals

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the DataFrame using the TechnicalIndicators module.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            DataFrame with indicators added
        """
        logger.debug("Adding technical indicators")
        
        # Use the TechnicalIndicators module to add indicators
        indicators_to_add = ['rsi', 'adx', 'atr', 'williams', 'ema', 'macd', 'bollinger', 'stochastic']
        
        # Add indicators using the centralized module
        df = self.indicators.add_indicators(df, indicators_to_add)
        
        logger.debug("Technical indicators added successfully")
        return df

    async def _check_trend_alignment(self, symbol: str, higher_tf: str) -> bool:
        """Check if the higher timeframe trend aligns with the entry timeframe."""
        if not higher_tf:
            return True
            
        # If no MT5 handler is available, skip trend alignment check
        if not self.mt5_handler and not (self.risk_manager and hasattr(self.risk_manager, 'mt5_handler') and self.risk_manager.mt5_handler):
            logger.debug(f"Skipping trend alignment check for {symbol} (no MT5 handler available)")
            return True
            
        try:
            # Get handler
            handler = self.mt5_handler if self.mt5_handler else self.risk_manager.mt5_handler
            
            # Get higher timeframe data
            htf_df = handler.get_market_data(symbol, higher_tf, num_candles=100)
            
            # Skip if data is missing
            if htf_df is None or len(htf_df) < 20:
                logger.warning(f"Insufficient data for trend alignment check on {symbol} {higher_tf}")
                return True
                
            # Calculate SMAs and EMAs using TechnicalIndicators
            htf_df = self.indicators.add_indicators(htf_df, ['ema'])
            
            # Add SMA since it might not be in the standard indicator set
            htf_df['sma20'] = self.indicators.calculate_sma(htf_df, 20)
            htf_df['sma50'] = self.indicators.calculate_sma(htf_df, 50)
            
            # Check both SMA and EMA alignment
            sma_aligned = (htf_df['sma20'].iloc[-1] > htf_df['sma50'].iloc[-1]) == (htf_df['sma20'].iloc[-5] > htf_df['sma50'].iloc[-5])
            ema_aligned = (htf_df['ema9'].iloc[-1] > htf_df['ema21'].iloc[-1]) == (htf_df['ema9'].iloc[-3] > htf_df['ema21'].iloc[-3])
            
            # Check if price is above/below both EMAs (trend strength)
            price_above_emas = htf_df['close'].iloc[-1] > htf_df['ema21'].iloc[-1] and htf_df['close'].iloc[-1] > htf_df['ema50'].iloc[-1]
            price_below_emas = htf_df['close'].iloc[-1] < htf_df['ema21'].iloc[-1] and htf_df['close'].iloc[-1] < htf_df['ema50'].iloc[-1]
            strong_trend = price_above_emas or price_below_emas
            
            # Log trend analysis results
            logger.debug(f"Trend alignment: SMA={sma_aligned}, EMA={ema_aligned}, Strong={strong_trend}")
            
            # Consider trend aligned if either SMA or EMA shows alignment, with bonus for strong trends
            return sma_aligned or ema_aligned or strong_trend
            
        except Exception as e:
            logger.warning(f"Error in trend alignment check: {str(e)}")
            return False  # Changed to False to avoid false positives on error

    def _validate_volume_and_volatility(self, df: pd.DataFrame) -> bool:
        """Validate volume and volatility conditions."""
        try:
            # Get volume analysis from the volume analyzer
            volume_analysis = self.volume_analyzer.analyze(df)
            
            # Get ATR from the dataframe - added by TechnicalIndicators
            volatility = df["atr"].iloc[-1] * 10000  # Convert to pips
            
            # Get symbol for instrument-specific adjustments
            symbol = df["symbol"].iloc[-1] if "symbol" in df.columns else "UNKNOWN"
            price = df["close"].iloc[-1]
            
            # Default min volatility from config with fallback
            min_volatility = self.config.get("min_volatility", 10)  # Default to 10 pips if not in config
            
            # Adjust minimum volatility based on instrument type
            if symbol.startswith(("BTC", "ETH", "XRP", "LTC", "BCH", "DOGE", "ADA")):
                # For crypto, use percentage-based volatility (0.2% of price)
                min_volatility = max(5, price * 0.002 * 10000)
            elif "JPY" in symbol:
                min_volatility = 15  # Higher for JPY pairs
            elif price < 1.0:
                min_volatility = 5   # Lower for low-priced instruments
            
            # Check Average Daily Range (ADR) if available
            adr_ok = True
            if "adr" in df.columns and len(df) > 20:
                current_adr = df["adr"].iloc[-1]
                
                # Calculate minimum ADR based on instrument type and price
                min_adr = 30  # Default minimum (30 pips for major pairs)
                
                # Cryptocurrency adjustment
                if symbol.startswith(("BTC", "ETH", "XRP", "LTC", "BCH", "DOGE", "ADA")):
                    # For crypto, use percentage-based ADR (0.5% daily range minimum)
                    min_adr = price * 0.005 * 10000  # 0.5% of price in pips
                    logger.debug(f"Crypto asset detected: Using {min_adr:.1f} pips minimum (0.5% of price)")
                
                # Adjust based on price level for all instruments
                elif price < 1.0:  # Low-priced instruments
                    min_adr = 15  # Lower ADR requirement (15 pips)
                    logger.debug(f"Low-priced asset detected: Using 15 pips minimum")
                elif price > 100:  # High-priced instruments
                    # For high-price instruments, use percentage-based ADR (0.25% daily range minimum)
                    min_adr = price * 0.0025 * 10000  # 0.25% of price in pips
                    logger.debug(f"High-priced asset detected: Using {min_adr:.1f} pips minimum (0.25% of price)")
                
                # JPY pairs adjustment
                if "JPY" in symbol:
                    min_adr = 40  # Higher ADR requirement for JPY pairs (more volatile)
                    logger.debug(f"JPY pair detected: Using 40 pips minimum")
                
                adr_ok = current_adr >= min_adr
                logger.debug(f"ADR check: Current={current_adr:.1f} pips, Minimum={min_adr:.1f} pips, OK={adr_ok}")
            
            # Enhanced volume analysis - use both volume analyzer and pattern detector insights
            # Get volume momentum from volume analysis
            volume_momentum = volume_analysis.get('momentum', 0)
            
            # Get volume trend from volume analysis
            volume_trend = volume_analysis.get('trend', 'neutral')
            
            # Calculate relative volume using TechnicalIndicators if available
            relative_volume = 1.0
            if 'relative_volume' in df.columns:
                relative_volume = df['relative_volume'].iloc[-1]
            else:
                # Calculate it if not already in the dataframe
                if 'volume' in df.columns and len(df) > 20:
                    relative_volume = self.indicators.calculate_relative_volume(df).iloc[-1]
            
            # Get pattern quality from pattern detector if available
            pattern_analysis = self.pattern_detector.analyze_patterns(df, parameters={"min_quality": 0.4})
            pattern_score = 0
            
            if pattern_analysis and 'patterns' in pattern_analysis:
                # Average quality of detected patterns
                patterns = pattern_analysis.get('patterns', [])
                if patterns:
                    pattern_qualities = [p.get('quality', 0) for p in patterns]
                    if pattern_qualities:
                        pattern_score = sum(pattern_qualities) / len(pattern_qualities) * 5  # Scale to 0-5
            
            # More flexible volume condition with relative volume
            volume_ok = (volume_momentum > -0.5 or 
                         volume_trend != 'bearish' or 
                         volume_trend == 'neutral' or
                         relative_volume > 0.8)  # Accept if volume is at least 80% of average
            
            # Make volatility check dependent on pattern quality
            if pattern_score >= 4:  # High-quality pattern
                min_volatility *= 0.7  # 30% reduction in volatility requirement
                logger.debug(f"High-quality pattern (score {pattern_score:.1f}) detected - reducing volatility requirement")
            
            # Evaluate volatility condition
            volatility_ok = volatility >= min_volatility
            
            # More flexible overall validation for high-quality patterns
            if pattern_score >= 4:
                conditions_met = sum([volatility_ok, volume_ok, adr_ok])
                if conditions_met >= 2:
                    logger.debug(f"High-quality pattern with {conditions_met}/3 conditions met - allowing signal")
                    return True
            
            logger.debug(f"Volume/volatility validation: ATR={volatility:.1f} pips, Minimum={min_volatility:.1f}, " 
                         f"Momentum={volume_momentum:.2f}, Trend={volume_trend}, RelVol={relative_volume:.2f}, " 
                         f"PatternScore={pattern_score:.1f}")
            
            return volatility_ok and volume_ok and adr_ok
            
        except Exception as e:
            logger.warning(f"Error in volume validation: {str(e)}")
            return True  # On error, allow signal to pass through

    def _validate_trend_strength(self, df: pd.DataFrame) -> bool:
        """Validate trend strength using ADX."""
        try:
            # Check if ADX is available
            if 'adx' not in df.columns:
                logger.warning("ADX indicator not found in dataframe, skipping trend strength validation")
                return True
                
            # Get latest ADX value
            adx_value = df['adx'].iloc[-1]
            
            # Get symbol from dataframe for instrument-specific adjustments
            symbol = df["symbol"].iloc[-1] if "symbol" in df.columns else "UNKNOWN"
            
            # Default ADX threshold
            adx_threshold = 20  # Standard threshold
            
            # Adjust threshold based on instrument volatility
            if symbol.startswith(("BTC", "ETH", "XRP", "LTC", "BCH", "DOGE", "ADA")):
                # Crypto is naturally more volatile, use lower threshold
                adx_threshold = 15
            elif "JPY" in symbol:
                # JPY pairs tend to trend strongly, use higher threshold
                adx_threshold = 22
                
            # Check if there's a pattern score we can use to adjust threshold
            pattern_score = 0
            for col in df.columns:
                if 'score' in col.lower():
                    pattern_score = df[col].iloc[-1] if not pd.isna(df[col].iloc[-1]) else 0
                    break
                    
            # Reduce ADX threshold for high-quality patterns
            if pattern_score >= 4:  # High-quality pattern
                adx_threshold *= 0.8  # 20% reduction in ADX requirement
                logger.debug(f"High-quality pattern detected - reducing ADX threshold to {adx_threshold}")
                
            # Check if the market is trending based on ADX value
            is_trending = adx_value >= adx_threshold
            
            logger.debug(f"Trend strength validation: ADX={adx_value:.1f}, Threshold={adx_threshold:.1f}, Trending={is_trending}")
            
            return is_trending
        except Exception as e:
            logger.warning(f"Error in trend strength validation: {str(e)}")
            return True  # On error, allow signal to pass through

    def _detect_123_reversal(self, df: pd.DataFrame, swing_points: Dict) -> Optional[Dict]:
        """Detect 123 Reversal pattern."""
        try:
            # Get last few swing points, but check if there are enough
            highs = swing_points["highs"]
            lows = swing_points["lows"]
            
            # Log available swing points
            logger.debug(f"Available swing points: {len(highs)} highs, {len(lows)} lows")
            
            # Need at least 3 of each for the pattern
            if len(highs) < 3 or len(lows) < 3:
                logger.debug("Insufficient swing points for 123 pattern")
                return None
                
            # Get last 3 of each for analysis, ensuring they're unique
            # Filter duplicates with a small tolerance (0.0001)
            unique_highs = []
            unique_lows = []
            
            # Process highs to ensure uniqueness
            for high in reversed(highs):
                # Skip if too similar to already added high
                if not unique_highs or all(abs(high["price"] - h["price"]) > 0.0001 for h in unique_highs):
                    unique_highs.append(high)
                if len(unique_highs) >= 3:
                    break
                    
            # Process lows to ensure uniqueness
            for low in reversed(lows):
                # Skip if too similar to already added low
                if not unique_lows or all(abs(low["price"] - l["price"]) > 0.0001 for l in unique_lows):
                    unique_lows.append(low)
                if len(unique_lows) >= 3:
                    break
                    
            # Check if we have enough unique points
            if len(unique_highs) < 3 or len(unique_lows) < 3:
                logger.debug("Insufficient unique swing points for 123 pattern")
                return None
                
            # Reverse lists to maintain chronological order
            unique_highs.reverse()
            unique_lows.reverse()
            
            # Get last 3 of each for analysis
            last_3_highs = unique_highs[:3]
            last_3_lows = unique_lows[:3]

            # Bearish Reversal (123 Top)
            point_1 = last_3_highs[0]["price"]  # Swing high
            point_2 = last_3_lows[1]["price"]   # Pullback low
            point_3 = last_3_highs[2]["price"]  # Retracement high
            
            if point_3 < point_1 and self._is_fib_retracement(point_1, point_2, point_3):
                if df["close"].iloc[-1] < point_2 and self._confirm_break(df, point_2, "SELL"):
                    logger.debug(f"Bearish 123 pattern detected: {point_1:.5f} -> {point_2:.5f} -> {point_3:.5f}")
                    return self._create_signal("SELL", df["close"].iloc[-1], point_1, point_3, df)

            # Bullish Reversal (123 Bottom)
            point_1 = last_3_lows[0]["price"]   # Swing low
            point_2 = last_3_highs[1]["price"]  # Pullback high
            point_3 = last_3_lows[2]["price"]   # Retracement low
            
            if point_3 > point_1 and self._is_fib_retracement(point_1, point_2, point_3):
                if df["close"].iloc[-1] > point_2 and self._confirm_break(df, point_2, "BUY"):
                    logger.debug(f"Bullish 123 pattern detected: {point_1:.5f} -> {point_2:.5f} -> {point_3:.5f}")
                    return self._create_signal("BUY", df["close"].iloc[-1], point_1, point_3, df)

            return None
            
        except Exception as e:
            logger.warning(f"Error in 123 reversal detection: {str(e)}")
            return None

    def _detect_123_continuation(self, df: pd.DataFrame, swing_points: Dict) -> Optional[Dict]:
        """Detect 123 Continuation pattern."""
        try:
            # Use PatternDetector's market bias analysis instead of MarketAnalysis
            market_bias = self.pattern_detector.detect_market_bias(df, swing_points)
            trend = market_bias.get("direction", "neutral")
                
            logger.debug(f"Trend analysis result: {trend}")
            
            if trend == "neutral":
                logger.debug("No clear trend detected for continuation pattern")
                return None
                
            # Get swing points
            highs = swing_points["highs"]
            lows = swing_points["lows"]
            
            # Log available swing points
            logger.debug(f"Continuation pattern analysis: {len(highs)} highs, {len(lows)} lows, trend: {trend}")
            
            # Check for minimum required swing points
            if len(highs) < 2 or len(lows) < 2:
                logger.debug("Insufficient swing points for continuation pattern")
                return None
                
            # Filter duplicates with a small tolerance (0.0001)
            unique_highs = []
            unique_lows = []
            
            # Process highs to ensure uniqueness
            for high in reversed(highs):
                # Skip if too similar to already added high
                if not unique_highs or all(abs(high["price"] - h["price"]) > 0.0001 for h in unique_highs):
                    unique_highs.append(high)
                if len(unique_highs) >= 2:  # We need at least 2 for continuation
                    break
                    
            # Process lows to ensure uniqueness
            for low in reversed(lows):
                # Skip if too similar to already added low
                if not unique_lows or all(abs(low["price"] - l["price"]) > 0.0001 for l in unique_lows):
                    unique_lows.append(low)
                if len(unique_lows) >= 2:  # We need at least 2 for continuation
                    break
                    
            # Check if we have enough unique points
            if len(unique_highs) < 2 or len(unique_lows) < 2:
                logger.debug("Insufficient unique swing points for continuation pattern")
                return None
                
            # Reverse lists to maintain chronological order
            unique_highs.reverse()
            unique_lows.reverse()
            
            # Use unique points instead of original lists
            highs = unique_highs
            lows = unique_lows

            # Bullish Continuation
            if trend == "bullish" and len(lows) >= 2 and len(highs) >= 2:
                point_1 = highs[-2]["price"]
                point_2 = lows[-1]["price"]
                point_3 = df["close"].iloc[-1]
                
                if point_3 > point_2 and "williams_r" in df.columns and df["williams_r"].iloc[-1] > -80:
                    logger.debug(f"Bullish continuation pattern detected: {point_1:.5f} -> {point_2:.5f} -> {point_3:.5f}")
                    return self._create_signal("BUY", point_3, point_1, point_2, df)

            # Bearish Continuation
            if trend == "bearish" and len(highs) >= 2 and len(lows) >= 2:
                point_1 = lows[-2]["price"]
                point_2 = highs[-1]["price"]
                point_3 = df["close"].iloc[-1]
                
                if point_3 < point_2 and "williams_r" in df.columns and df["williams_r"].iloc[-1] < -20:
                    logger.debug(f"Bearish continuation pattern detected: {point_1:.5f} -> {point_2:.5f} -> {point_3:.5f}")
                    return self._create_signal("SELL", point_3, point_1, point_2, df)

            return None
            
        except Exception as e:
            logger.warning(f"Error in 123 continuation detection: {str(e)}")
            return None

    def _is_fib_retracement(self, point_1: float, point_2: float, point_3: float) -> bool:
        """Check if Point 3 is within Fibonacci retracement levels of Point 1 to Point 2."""
        diff = abs(point_1 - point_2)
        retracement = abs(point_3 - point_2)
        ratio = retracement / diff
        
        try:
            # Get current symbol from dataframe if possible
            symbol = getattr(self, 'current_symbol', 'UNKNOWN')
            
            # Determine volatility based on ATR
            recent_volatility = 'normal'
            
            # Try to access ATR from the current context
            atr_pips = getattr(self, 'current_atr_pips', None)
            
            # Classify volatility based on ATR if available
            if atr_pips is not None:
                if atr_pips > 25:  # High volatility threshold (25 pips)
                    recent_volatility = 'high'
                elif atr_pips < 10:  # Low volatility threshold (10 pips)
                    recent_volatility = 'low'
                    
            # Base tolerance - default is 0.01 (1%)
            base_tolerance = 0.01
            
            # Adjust tolerance based on symbol type
            if symbol.startswith(("BTC", "ETH", "XRP", "LTC", "BCH", "DOGE", "ADA")):
                # Crypto is more volatile, use wider tolerance
                base_tolerance = 0.015  # 1.5%
            elif "JPY" in symbol:
                # JPY pairs can also benefit from slightly wider tolerance
                base_tolerance = 0.012  # 1.2%
                
            # Adjust based on recent volatility
            if recent_volatility == 'high':
                base_tolerance *= 1.5  # 50% wider tolerance in high volatility
            elif recent_volatility == 'low':
                base_tolerance *= 0.8  # 20% tighter tolerance in low volatility
                
            logger.debug(f"Fibonacci validation with tolerance: {base_tolerance:.3f} for {symbol}")
            
            # Dynamic weighting of Fibonacci levels based on market conditions
            # During high volatility, prefer deeper retracements
            weighted_levels = self.config.get("fib_levels", [0.236, 0.382, 0.5, 0.618, 0.786])
            
            # In high volatility, give more weight to deeper retracements (0.618, 0.786)
            # In low volatility, give more weight to shallower retracements (0.382, 0.5)
            prioritized_levels = []
            
            if recent_volatility == 'high':
                # Prioritize deeper retracements in high volatility
                prioritized_levels = [level for level in weighted_levels if level >= 0.5]
                for level in prioritized_levels:
                    # Use tighter tolerance for prioritized levels to ensure quality
                    if abs(ratio - level) < base_tolerance * 0.8:
                        logger.debug(f"Fibonacci match at prioritized level: {level:.3f} (high volatility)")
                        return True
                
            elif recent_volatility == 'low':
                # Prioritize shallower retracements in low volatility
                prioritized_levels = [level for level in weighted_levels if level <= 0.5]
                for level in prioritized_levels:
                    # Use tighter tolerance for prioritized levels to ensure quality
                    if abs(ratio - level) < base_tolerance * 0.8:
                        logger.debug(f"Fibonacci match at prioritized level: {level:.3f} (low volatility)")
                        return True
            
            # Check all Fibonacci levels with the adjusted tolerance
            for level in weighted_levels:
                if abs(ratio - level) < base_tolerance:
                    # Additional check for a key Fibonacci level - require tighter match
                    is_key_level = level in [0.382, 0.5, 0.618]
                    if is_key_level:
                        is_match = abs(ratio - level) < (base_tolerance * 0.8)  # 20% tighter for key levels
                    else:
                        is_match = True  # Accept normal tolerance for other levels
                        
                    if is_match:
                        logger.debug(f"Fibonacci match at level: {level:.3f} (ratio: {ratio:.3f})")
                        return True
                
            logger.debug(f"No Fibonacci match found. Closest ratio: {ratio:.3f}")
            return False
            
        except Exception as e:
            logger.warning(f"Error in flexible Fibonacci validation: {str(e)}")
            # Fall back to standard Fibonacci check with default values
            default_fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
            return any(abs(ratio - level) < 0.01 for level in default_fib_levels)

    def _confirm_break(self, df: pd.DataFrame, level: float, direction: str) -> bool:
        """Confirm break with candle close beyond Point 2 and volume spike check."""
        # Get the recent candles for confirmation check
        recent_candles = df.tail(self.config["confirmation_bars"])
        
        # Price confirmation - all candles must close beyond the level
        if direction == "SELL":
            price_confirmed = all(candle["close"] < level for _, candle in recent_candles.iterrows())
        else:
            price_confirmed = all(candle["close"] > level for _, candle in recent_candles.iterrows())
            
        if not price_confirmed:
            return False
        
        # Volume spike check - look for increased volume during breakout
        try:
            # Calculate average volume over the last 10 candles before the recent ones
            lookback = 10
            if len(df) <= lookback + self.config["confirmation_bars"]:
                # Not enough data for volume comparison
                return price_confirmed
                
            # Get the volume data
            volume_history = df["volume"].iloc[-(lookback + self.config["confirmation_bars"]):-self.config["confirmation_bars"]]
            recent_volume = df["volume"].iloc[-self.config["confirmation_bars"]:]
            
            if len(volume_history) == 0 or len(recent_volume) == 0:
                return price_confirmed
                
            # Calculate the average volume
            avg_volume = volume_history.mean()
            
            # Check if any of the recent candles has volume spike (20% above average)
            volume_spike = any(vol > avg_volume * 1.2 for vol in recent_volume)
            
            # Log the volume analysis
            logger.debug(f"Volume analysis: Avg={avg_volume:.1f}, Recent={list(recent_volume)}, Spike={volume_spike}")
            
            # Return final confirmation (require both price and volume confirmation if possible)
            return price_confirmed and (volume_spike or "volume" not in df.columns)
            
        except Exception as e:
            logger.warning(f"Error in volume spike check: {str(e)}")
            # Fall back to price confirmation only on error
            return price_confirmed

    def _create_signal(self, direction: str, entry: float, point_1: float, point_3: float, df: pd.DataFrame) -> Dict:
        """Create a trading signal dictionary."""
        logger.debug(f"Creating {direction} signal at price {entry}")
        
        # Get symbol from dataframe or use the one from the current context
        symbol = df["symbol"].iloc[-1] if "symbol" in df.columns else "UNKNOWN"
        
        # Calculate ATR for dynamic stop loss and take profit
        atr_value = df["atr"].iloc[-1] if "atr" in df.columns else 0.0001
        
        # Calculate stop loss based on direction and swing points
        if direction == "BUY":
            # For BUY, stop loss should be below entry
            # Use point_1 (swing low) as reference, but ensure it's not too far
            if point_1 < entry:
                # Use swing low as stop loss, but not more than 2 ATR away
                sl_distance = min(entry - point_1, 2 * atr_value)
                stop_loss = entry - sl_distance
            else:
                # If point_1 is not below entry, use 1.5 ATR
                stop_loss = entry - (1.5 * atr_value)
        else:  # SELL
            # For SELL, stop loss should be above entry
            # Use point_1 (swing high) as reference, but ensure it's not too far
            if point_1 > entry:
                # Use swing high as stop loss, but not more than 2 ATR away
                sl_distance = min(point_1 - entry, 2 * atr_value)
                stop_loss = entry + sl_distance
            else:
                # If point_1 is not above entry, use 1.5 ATR
                stop_loss = entry + (1.5 * atr_value)
        
        # Calculate risk in price terms
        risk = abs(entry - stop_loss)
        
        # Use risk_manager to calculate the stop loss if available
        if self.risk_manager:
            try:
                # Use the risk manager's calculated stop loss
                pattern_type = "123_reversal" if direction == "BUY" else "123_continuation"
                sl_info = self.risk_manager.calculate_stop_loss(
                    entry_price=entry,
                    direction=direction,
                    atr_value=atr_value,
                    pattern_type=pattern_type,
                    recent_swing={"price": point_1}
                )
                if sl_info and "price" in sl_info:
                    stop_loss = sl_info["price"]
                    risk = abs(entry - stop_loss)
                    logger.debug(f"Using risk manager's stop loss: {stop_loss:.5f}")
            except Exception as e:
                logger.warning(f"Failed to use risk manager's stop loss calculation: {str(e)}")
        
        # Get dynamic take profit levels based on market structure
        take_profit_levels = self._calculate_dynamic_take_profits(df, direction, entry, stop_loss, risk)
        
        # Calculate position size if possible (required by trading bot)
        position_size = 0.01  # Default minimum size
        if self.risk_manager:
            try:
                # Get account info
                account_info = {"balance": 10000}  # Default if not available
                
                # Try to get actual account info
                if hasattr(self.risk_manager, "_get_account_info"):
                    actual_info = self.risk_manager._get_account_info()
                    if actual_info and "balance" in actual_info:
                        account_info = actual_info
                
                # Calculate position size based on risk
                position_size = self.risk_manager.calculate_position_size(
                    account_balance=account_info["balance"],
                    risk_per_trade=self.config["max_risk_per_trade"],
                    entry_price=entry,
                    stop_loss_price=stop_loss,
                    symbol=symbol,
                    market_condition='normal',  # Can be dynamically determined based on analysis
                    volatility_state=self.recent_volatility,
                    confidence_score=0.7  # Base confidence level
                )
            except Exception as e:
                logger.warning(f"Failed to calculate position size: {str(e)}")
        
        # Check for low volatility condition
        low_volatility = False
        if "atr" in df.columns:
            # Convert ATR to pips for easier interpretation
            atr_pips = atr_value * 10000
            low_volatility = atr_pips < self.config["min_volatility"]
        
        # Create the signal dictionary with multiple take profit levels
        signal = {
            "symbol": symbol,
            "timeframe": self.config["timeframes"][0],
            "direction": direction,
            # Fields needed by trading bot
            "entry_price": entry,  # Required by trading bot
            "entry": entry,        # Keep for backward compatibility
            "stop_loss": stop_loss,
            "take_profit": take_profit_levels[0]["price"],  # First take profit for backward compatibility
            "take_profit_levels": take_profit_levels,  # Multiple take profit levels
            "position_size": position_size,  # Required by trading bot
            "confidence": 0.7,  # Base confidence level
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
            "low_volatility": low_volatility,
            "generator": "SignalGenerator123"
        }
        
        # If risk manager is available, validate the trade for risk management rules
        if self.risk_manager and hasattr(self.risk_manager, 'validate_trade'):
            try:
                # Get account balance and open trades
                account_info = self.risk_manager._get_account_info() if hasattr(self.risk_manager, '_get_account_info') else {"balance": 10000}
                open_trades = []
                
                # Get open trades if method is available
                if hasattr(self.mt5_handler, 'get_open_positions'):
                    open_trades = self.mt5_handler.get_open_positions()
                elif hasattr(self.risk_manager, 'mt5_handler') and self.risk_manager.mt5_handler and hasattr(self.risk_manager.mt5_handler, 'get_open_positions'):
                    open_trades = self.risk_manager.mt5_handler.get_open_positions()
                
                # Validate trade against risk management rules
                validation = self.risk_manager.validate_trade(
                    trade=signal,
                    account_balance=account_info.get("balance", 10000),
                    open_trades=open_trades
                )
                
                # Adjust signal based on validation results if needed
                if validation and validation.get("valid", False):
                    # Update signal with any adjustments from risk manager
                    if "adjusted_position_size" in validation:
                        signal["position_size"] = validation["adjusted_position_size"]
                        logger.debug(f"Position size adjusted by risk manager: {signal['position_size']}")
                else:
                    logger.warning(f"Signal rejected by risk manager: {validation.get('reason', 'Unknown')}")
                    # You can decide to return None here to reject the signal
                    # For now, we'll still return it but log the warning
            except Exception as e:
                logger.warning(f"Error in trade validation: {str(e)}")
        
        logger.info(f"Created signal: {signal['symbol']} {signal['direction']} Entry: {signal['entry_price']:.5f} SL: {signal['stop_loss']:.5f}")
        logger.info(f"Take profits: TP1: {take_profit_levels[0]['price']:.5f} ({take_profit_levels[0]['ratio']:.1f}R)")
        if len(take_profit_levels) > 1:
            logger.info(f"             TP2: {take_profit_levels[1]['price']:.5f} ({take_profit_levels[1]['ratio']:.1f}R)")
        if len(take_profit_levels) > 2:
            logger.info(f"             TP3: {take_profit_levels[2]['price']:.5f} ({take_profit_levels[2]['ratio']:.1f}R)")
        
        return signal

    def _calculate_dynamic_take_profits(self, df: pd.DataFrame, direction: str, entry: float, stop_loss: float, risk: float) -> List[Dict]:
        """Calculate dynamic take profit levels based on market structure and key levels."""
        try:
            # Look for key levels in the market structure
            # Get swing points from the pattern detector
            parameters = {"lookback": 30, "threshold": 0.0010}
            swing_points = self.pattern_detector.detect_swing_points(df, parameters)
            highs = []
            lows = []
            
            if swing_points and "highs" in swing_points and "lows" in swing_points:
                highs = [h["price"] for h in swing_points["highs"]]
                lows = [l["price"] for l in swing_points["lows"]]
            
            # Get symbol for instrument-specific adjustments
            symbol = df["symbol"].iloc[-1] if "symbol" in df.columns else "UNKNOWN"
            
            # Current price and ATR
            current_price = df["close"].iloc[-1]
            atr_value = df["atr"].iloc[-1] if "atr" in df.columns else 0.0001
            
            # Calculate default R-multiple targets
            default_tp1 = entry + (risk * 1.5) if direction == "BUY" else entry - (risk * 1.5)
            default_tp2 = entry + (risk * 2.5) if direction == "BUY" else entry - (risk * 2.5)
            default_tp3 = entry + (risk * 3.5) if direction == "BUY" else entry - (risk * 3.5)
            
            # Try to find market structure based targets
            structure_targets = []
            
            if direction == "BUY":
                # For BUY signals, look for resistance levels above entry
                resistance_levels = sorted([h for h in highs if h > entry])
                
                # Add key moving averages if available (like 50 EMA or 200 SMA)
                if "ema50" in df.columns and df["ema50"].iloc[-1] > entry:
                    resistance_levels.append(df["ema50"].iloc[-1])
                
                # If we found resistance levels, use them as targets
                if resistance_levels:
                    for level in resistance_levels:
                        # Only consider levels that are at least 1R away
                        if level - entry >= risk:
                            structure_targets.append(level)
                            # Stop after finding 3 targets
                            if len(structure_targets) >= 3:
                                break
            else:  # SELL direction
                # For SELL signals, look for support levels below entry
                support_levels = sorted([l for l in lows if l < entry], reverse=True)
                
                # Add key moving averages if available
                if "ema50" in df.columns and df["ema50"].iloc[-1] < entry:
                    support_levels.append(df["ema50"].iloc[-1])
                
                # If we found support levels, use them as targets
                if support_levels:
                    for level in support_levels:
                        # Only consider levels that are at least 1R away
                        if entry - level >= risk:
                            structure_targets.append(level)
                            # Stop after finding 3 targets
                            if len(structure_targets) >= 3:
                                break
            
            # Determine final take profit levels
            tp1, tp2, tp3 = default_tp1, default_tp2, default_tp3
            
            # Use structure based targets if available
            if len(structure_targets) >= 3:
                tp1, tp2, tp3 = structure_targets[0], structure_targets[1], structure_targets[2]
            elif len(structure_targets) == 2:
                tp1, tp2 = structure_targets[0], structure_targets[1]
                # Calculate tp3 based on the pattern formed by tp1 and tp2
                target_interval = abs(tp2 - tp1)
                tp3 = tp2 + target_interval if direction == "BUY" else tp2 - target_interval
            elif len(structure_targets) == 1:
                tp1 = structure_targets[0]
                # Use default R-multiples for remaining targets but based off the first target
                target_interval = abs(tp1 - entry)
                tp2 = tp1 + target_interval if direction == "BUY" else tp1 - target_interval
                tp3 = tp2 + target_interval if direction == "BUY" else tp2 - target_interval
            
            # Calculate R-multiples for the targets
            tp1_r = abs(tp1 - entry) / risk
            tp2_r = abs(tp2 - entry) / risk
            tp3_r = abs(tp3 - entry) / risk
            
            # Adjust position sizes based on target distances
            # Further targets get smaller position sizes
            tp1_size = 0.5  # Default 50% for first target
            tp2_size = 0.3  # Default 30% for second target
            tp3_size = 0.2  # Default 20% for third target
            
            # Adjust for very close or very far targets
            if tp1_r < 1.2:  # Very close first target
                tp1_size = 0.4  # Take less off at the first target
                tp2_size = 0.4  # Take more off at the second target
                tp3_size = 0.2  # Keep third target the same
            elif tp1_r > 2.5:  # Very far first target
                tp1_size = 0.6  # Take more off at the far first target
                tp2_size = 0.3  # Keep second target the same
                tp3_size = 0.1  # Take less off at third target
            
            # Build take profit levels list
            take_profit_levels = [
                {"price": tp1, "size": tp1_size, "ratio": tp1_r},
                {"price": tp2, "size": tp2_size, "ratio": tp2_r},
                {"price": tp3, "size": tp3_size, "ratio": tp3_r}
            ]
            
            logger.debug(f"Dynamic take profits: TP1={tp1:.5f} ({tp1_r:.1f}R), TP2={tp2:.5f} ({tp2_r:.1f}R), TP3={tp3:.5f} ({tp3_r:.1f}R)")
            
            return take_profit_levels
            
        except Exception as e:
            logger.warning(f"Error in dynamic take profit calculation: {str(e)}")
            # Fall back to default take profit calculation
            default_tp1 = entry + (risk * 1.5) if direction == "BUY" else entry - (risk * 1.5)
            default_tp2 = entry + (risk * 2.5) if direction == "BUY" else entry - (risk * 2.5)
            default_tp3 = entry + (risk * 3.5) if direction == "BUY" else entry - (risk * 3.5)
            
            return [
                {"price": default_tp1, "size": 0.5, "ratio": 1.5},
                {"price": default_tp2, "size": 0.3, "ratio": 2.5},
                {"price": default_tp3, "size": 0.2, "ratio": 3.5}
            ]

    async def filter_signals(self, signals: List[Dict]) -> List[Dict]:
        """Apply additional filters to signals."""
        if not signals:
            logger.debug("No signals to filter")
            return []
            
        logger.debug(f"Filtering {len(signals)} signals")
        filtered_signals = []
        for signal in signals:
            # Correlation check (new function)
            if not await self._check_correlation(signal["symbol"]):
                logger.debug(f"Signal for {signal['symbol']} rejected due to correlation")
                continue

            # Get market data for POI analysis
            df = None
            
            # Try to get market data using the most reliable method available
            if self.mt5_handler:
                df = self.mt5_handler.get_market_data(signal["symbol"], signal["timeframe"])
            elif self.risk_manager and hasattr(self.risk_manager, 'mt5_handler') and self.risk_manager.mt5_handler:
                df = self.risk_manager.mt5_handler.get_market_data(signal["symbol"], signal["timeframe"])
                
            if df is not None:
                try:
                    # Get analysis from pattern detector for more sophisticated 
                    # pattern detection instead of just POI analysis
                    patterns_analysis = self.pattern_detector.analyze_patterns(df)
                    
                    # If there are patterns confirming our signal, boost confidence
                    if patterns_analysis.get("patterns"):
                        # Check if any pattern aligns with our signal direction
                        signal_direction = signal["direction"]
                        aligned_patterns = []
                        
                        for pattern in patterns_analysis.get("patterns", []):
                            pattern_direction = pattern.get("direction", "")
                            # Map BUY/SELL to bullish/bearish
                            is_aligned = (signal_direction == "BUY" and pattern_direction == "bullish") or \
                                          (signal_direction == "SELL" and pattern_direction == "bearish")
                            
                            if is_aligned:
                                aligned_patterns.append(pattern.get("type", "unknown"))
                        
                        if aligned_patterns:
                            # Boost confidence based on number of aligned patterns
                            confidence_boost = min(0.1 * len(aligned_patterns), 0.3)  # Cap at 0.3
                            signal["confidence"] += confidence_boost
                            logger.debug(f"Confidence boosted for {signal['symbol']} by {confidence_boost:.2f} due to aligned patterns: {aligned_patterns}")
                except Exception as e:
                    logger.warning(f"Error in pattern analysis: {str(e)}")
            else:
                logger.warning(f"Could not get market data for pattern analysis on {signal['symbol']}")

            filtered_signals.append(signal)
            
        logger.info(f"{len(filtered_signals)}/{len(signals)} signals passed filtering")
        return filtered_signals

    async def _check_correlation(self, symbol: str) -> bool:
        """Check correlation with other pairs."""
        # If no MT5 handler is available, skip correlation check
        if not self.mt5_handler and not (self.risk_manager and hasattr(self.risk_manager, 'mt5_handler') and self.risk_manager.mt5_handler):
            logger.debug(f"Skipping correlation check for {symbol} (no MT5 handler available)")
            return True
            
        correlated_pairs = {"EURUSDm": ["GBPUSDm", "USDCHFm"], "GBPUSDm": ["EURUSDm", "USDJPYm"]}
        if symbol not in correlated_pairs:
            return True
            
        # Get handler
        handler = self.mt5_handler if self.mt5_handler else self.risk_manager.mt5_handler
            
        for pair in correlated_pairs[symbol]:
            try:
                df1 = handler.get_market_data(symbol, "H1", 100)
                df2 = handler.get_market_data(pair, "H1", 100)
                
                # Skip if data is missing
                if df1 is None or df2 is None or len(df1) < 50 or len(df2) < 50:
                    logger.warning(f"Insufficient data for correlation check between {symbol} and {pair}")
                    continue
                    
                correlation = self.mtf_analyzer.calculate_timeframe_correlation(df1, df2)
                if abs(correlation) > 0.8:  # High correlation
                    logger.debug(f"High correlation detected between {symbol} and {pair}: {correlation:.2f}")
                    return False
            except Exception as e:
                logger.warning(f"Error in correlation check: {str(e)}")
                
        return True

    def _set_adaptive_parameters(self, df: pd.DataFrame, symbol: str) -> None:
        """Set adaptive parameters based on market conditions and symbol characteristics."""
        try:
            # Store baseline ATR to determine volatility
            initial_atr = 0
            
            # Calculate ATR using TechnicalIndicators module 
            if len(df) > 20:
                # Use TechnicalIndicators for ATR calculation directly
                initial_atr = self.indicators.calculate_atr(df, period=14).iloc[-1]
                
                if pd.isna(initial_atr):
                    logger.warning(f"ATR calculation returned NaN, using fallback calculation")
                    # Fallback calculation if ATR returns NaN
                    true_ranges = []
                    for i in range(1, min(20, len(df))):
                        high = df['high'].iloc[-i]
                        low = df['low'].iloc[-i]
                        prev_close = df['close'].iloc[-i-1]
                        true_range = max(high-low, abs(high-prev_close), abs(low-prev_close))
                        true_ranges.append(true_range)
                    initial_atr = np.mean(true_ranges)
                
            price = df['close'].iloc[-1]
            initial_atr_pips = initial_atr * 10000
            
            # Default parameters with fallbacks if not in config
            atr_period = self.config.get("atr_period", 14)  # Default to 14 if not in config
            williams_period = self.config.get("williams_period", 14)  # Default to 14 if not in config
            confirmation_bars = self.config.get("confirmation_bars", 2)  # Default to 2 if not in config
            
            # Adjust parameters based on symbol characteristics
            if symbol.startswith(("BTC", "ETH", "XRP", "LTC", "BCH", "DOGE", "ADA")):
                # Cryptocurrencies - more volatile, use shorter timeframes
                logger.debug(f"Adjusting parameters for cryptocurrency: {symbol}")
                atr_period = max(10, atr_period - 4)  # Shorter ATR period (min 10)
                williams_period = max(10, williams_period - 4)  # Shorter Williams period
                confirmation_bars = min(3, confirmation_bars + 1)  # May need extra confirmation
                
            elif "JPY" in symbol:
                # JPY pairs - different volatility profile
                logger.debug(f"Adjusting parameters for JPY pair: {symbol}")
                atr_period = max(12, atr_period - 2)  # Slightly shorter ATR period
                
            elif price < 1.0:
                # Low-priced instruments often need longer periods
                logger.debug(f"Adjusting parameters for low-priced instrument: {symbol}")
                atr_period = min(18, atr_period + 4)  # Longer ATR period
                confirmation_bars = max(1, confirmation_bars - 1)  # May need fewer confirmations
                
            # Adjust parameters based on current volatility from initial ATR calculation
            if initial_atr_pips > 25:  # High volatility
                # In high volatility, use shorter periods to be more responsive
                logger.debug(f"Adjusting parameters for high volatility: {initial_atr_pips:.1f} pips")
                atr_period = max(8, atr_period - 6)  # Much shorter ATR period
                williams_period = max(8, williams_period - 6)  # Shorter Williams period
                confirmation_bars = min(3, confirmation_bars + 1)  # Extra confirmation for safety
                
            elif initial_atr_pips < 10:  # Low volatility
                # In low volatility, use longer periods to filter noise
                logger.debug(f"Adjusting parameters for low volatility: {initial_atr_pips:.1f} pips")
                atr_period = min(20, atr_period + 6)  # Longer ATR period
                williams_period = min(20, williams_period + 6)  # Longer Williams period
                
            # Update configuration with adaptive parameters
            adapted_config = {
                "atr_period": atr_period,
                "williams_period": williams_period,
                "confirmation_bars": confirmation_bars
            }
            
            logger.debug(f"Adaptive parameters for {symbol}: ATR period={atr_period}, Williams period={williams_period}, Confirmation bars={confirmation_bars}")
            
            # Update the configuration (keeping other settings intact)
            self.config.update(adapted_config)
            
            # Store context for other methods to use
            self.current_atr_pips = initial_atr_pips
            
            # Classify volatility for other methods
            if initial_atr_pips > 25:
                self.recent_volatility = 'high'
            elif initial_atr_pips < 10:
                self.recent_volatility = 'low'
            else:
                self.recent_volatility = 'normal'
            
        except Exception as e:
            logger.warning(f"Error setting adaptive parameters: {str(e)}")
            # Default values retained if there's an error

# Create an alias for backward compatibility
SignalGenerator = SignalGenerator123