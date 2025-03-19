import os
from loguru import logger
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
import sys

# Importing essential modules
from src.market_analysis import MarketAnalysis
from src.risk_manager import RiskManager
from src.mt5_handler import MT5Handler
from src.mtf_analysis import MTFAnalysis
from src.poi_detector import POIDetector

# Create logs directory if it doesn't exist
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Configure loguru logger
logger_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>SG2:{function}:{line}</cyan> | <level>{message}</level>"

# Configure loguru with custom format
logger.configure(handlers=[
    {"sink": sys.stdout, "format": logger_format, "level": "DEBUG", "colorize": True},
    {"sink": os.path.join(log_dir, "signal_generator2_detailed.log"), 
     "format": logger_format, "level": "DEBUG", "rotation": "10 MB", 
     "retention": 5, "compression": "zip"}
])

# Add context to differentiate this logger
logger = logger.bind(name="signal_generator2")

class SignalGeneratorBankTrading:
    """Bank trading strategy signal generator with simplified implementation.
    
    This class detects trading opportunities based on bank trading strategies including:
    - Liquidity grabs (stop runs)
    - Order block reversals
    - Supply and demand zone interactions
    """
    
    def __init__(self, mt5_handler: Optional[MT5Handler] = None, config: Dict = None, risk_manager: Optional[RiskManager] = None):
        """Initialize with market analysis components and configuration.
        
        Args:
            mt5_handler: Optional MetaTrader 5 handler for market data
            config: Strategy configuration parameters 
            risk_manager: Risk manager for position sizing
        """
        self.mt5_handler = mt5_handler
        self.risk_manager = risk_manager or RiskManager()
        self.market_analyzer = MarketAnalysis()
        self.poi_detector = POIDetector()
        self.mtf_analysis = MTFAnalysis()
        
        # NEW: Add trade cooldown tracking to prevent rapid trading
        self.last_signal_time = {}  # Track last signal time by symbol
        self.min_signal_interval = 900  # Default 15 minutes (900 seconds) cooldown between signals for same symbol
        
        # Default configuration
        self.config = {
            "timeframe": "M15",
            "stop_run_threshold_pips": 5,
            "pullback_pips": 5,
            "min_stop_pips": 5,
            "min_rr_ratio": 2.0,  # Updated from 1.5 to 2.0
            "max_candles_for_setup": 10,
            "min_confidence_threshold": 0.6,
            "signal_cooldown_seconds": 900,  # 15 minutes cooldown between signals for same symbol
            "fixed_sl": False,  # New option for fixed stop loss
            "manipulation_points": {
                "recent_high_low": True,
                "previous_day_high_low": True,
                "sharp_reversals": True  # NEW: Enable sharp reversal detection
            },
            "reversal_detection": {
                "window_size": 5,  # Window size for trend direction calculation
                "body_size_threshold": 0.5,  # Minimum body size ratio to qualify as strong candle
                "close_position_threshold": 0.3  # Maximum distance from extreme for close price
            },
            "confidence_weights": {
                "base_confidence": 0.4,
                "volume_spike": 0.15,
                "rsi_divergence": 0.20,
                "high_quality_entry": 0.15,
                "htf_alignment": 0.25,
                "poi_alignment": 0.20,
                "volume_trend": 0.15
            }
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
            
        logger.info("SignalGeneratorBankTrading initialized with configuration")
        
    async def generate_signals(self, market_data: Dict, symbol: str, timeframe: str, account_info: Optional[Dict] = None) -> Dict:
        """Generate trading signals based on bank trading strategy.
        
        Args:
            market_data: Dictionary containing market data for different symbols and timeframes
            symbol: The trading symbol to generate signals for
            timeframe: The timeframe to use for signal generation
            account_info: Optional account information
            
        Returns:
            Dict containing:
                - signals: List of generated trading signals
                - status: Status of signal generation
                - message: Descriptive message about the result
        """
        logger.info(f"=== Starting signal generation for {symbol} on {timeframe} ===")
        
        # Enforce 15-minute timeframe for Bank Trading strategy
        if timeframe != "M15":
            logger.warning(f"Bank Trading strategy requires M15 timeframe, but received {timeframe}")
            return {
                "signals": [],
                "status": "invalid_timeframe",
                "message": f"Bank Trading strategy requires M15 timeframe, but received {timeframe}"
            }
        
        # Get market data
        df = None
        if market_data and symbol in market_data and timeframe in market_data[symbol]:
            df = market_data[symbol][timeframe]
            logger.debug(f"Using provided market data for {symbol} {timeframe}, shape: {df.shape}")
        elif self.mt5_handler:
            logger.debug(f"Fetching market data from MT5 for {symbol} {timeframe}")
            df = self.mt5_handler.get_market_data(symbol, timeframe, num_candles=200)
            if df is not None:
                logger.debug(f"Fetched {len(df)} candles via MT5 handler for {symbol} {timeframe}")
            else:
                logger.warning(f"MT5 handler returned None for {symbol} {timeframe}")
        else:
            logger.warning(f"No market data source available for {symbol} {timeframe}")
        
        # Require at least 15 candles instead of 20 (further reduced requirement)
        if df is None:
            logger.warning(f"No data available for {symbol} on {timeframe}")
            return {
                "signals": [],
                "status": "no_data",
                "message": f"No market data available for {symbol} on {timeframe}"
            }
        elif len(df) < 50:
            logger.warning(f"Insufficient data for {symbol} on {timeframe}: only {len(df)} candles available (need 15)")
            # Add more detailed logging about the available data
            if len(df) > 0:
                logger.debug(f"Available data range: {df.index[0]} to {df.index[-1]}")
                logger.debug(f"First candle: Open={df['open'].iloc[0]:.5f}, Close={df['close'].iloc[0]:.5f}")
                logger.debug(f"Last candle: Open={df['open'].iloc[-1]:.5f}, Close={df['close'].iloc[-1]:.5f}")
            return {
                "signals": [],
                "status": "insufficient_data",
                "message": f"Insufficient market data for {symbol} on {timeframe}: only {len(df)} candles"
            }
        
        logger.debug(f"Data summary for {symbol} {timeframe}: {len(df)} candles, latest close: {df['close'].iloc[-1]:.5f}")
        logger.debug(f"Data range: {df.index[0]} to {df.index[-1]}")
        
        # Ensure we have symbol in the dataframe
        df["symbol"] = symbol
        
        # Add basic indicators
        logger.debug(f"Adding basic indicators for {symbol}")
        df = self._add_basic_indicators(df)
        
        # Get higher timeframe data for confirmation
        logger.debug(f"Checking higher timeframe alignment for {symbol}")
        htf_data = await self._check_higher_timeframe_alignment(symbol)
        logger.debug(f"Higher timeframe data: {htf_data}")
        
        # 1. Identify manipulation points
        logger.debug(f"Identifying manipulation points for {symbol}")
        manipulation_points = self._identify_manipulation_points(df, symbol)
        if not manipulation_points:
            logger.debug(f"No manipulation points identified for {symbol}")
            return {
                "signals": [],
                "status": "no_setups",
                "message": f"No manipulation points identified for {symbol}"
            }
        
        logger.info(f"Found {len(manipulation_points)} manipulation points for {symbol}: {[f'{p['type']}:{p['price']:.5f}' for p in manipulation_points]}")
        
        # Check for signals using the bank trading three-step process
        signals = []
        
        for point in manipulation_points:
            logger.debug(f"===== Analyzing manipulation point: {point['type']} at {point['price']:.5f} =====")
            
            # Step 1: Find stop run candle
            logger.debug(f"Looking for stop run candle for {point['type']} at {point['price']:.5f}")
            stop_run = self._find_stop_run_candle(df, point, symbol)
            if not stop_run:
                logger.debug(f"No stop run found for {point['type']} at {point['price']:.5f}")
                continue
            
            logger.debug(f"Found stop run candle: direction={stop_run['direction']}, index={stop_run['candle_index']}, high={stop_run['high']:.5f}, low={stop_run['low']:.5f}")
                
            # Step 2: Find confirmation candle
            logger.debug(f"Looking for confirmation candle after stop run")
            confirmation = self._find_confirmation_candle(df, stop_run)
            if not confirmation:
                logger.debug(f"No confirmation candle found for stop run at {point['price']:.5f}")
                continue
            
            logger.debug(f"Found confirmation candle: index={confirmation['candle_index']}, close={confirmation['close']:.5f}")
                
            # Step 3: Find pullback entry
            logger.debug(f"Looking for pullback entry after confirmation")
            entry = self._find_pullback_entry(df, stop_run, confirmation, symbol)
            if not entry:
                logger.debug(f"No valid pullback entry found for {point['type']} at {point['price']:.5f}")
                continue
            
            logger.debug(f"Found entry: type={entry['entry_type']}, price={entry['entry_price']:.5f}")
                
            # Check higher timeframe alignment
            logger.debug(f"Checking higher timeframe alignment for {entry['direction']} direction")
            if not self._is_htf_aligned(htf_data, entry["direction"]):
                logger.debug(f"Setup not aligned with higher timeframe trends, skipping")
                continue
            
            logger.debug(f"Higher timeframe alignment confirmed for {entry['direction']} direction")
                
            # Create signal with dynamic Stop Loss and Take Profit
            direction = "BUY" if entry["direction"] == "long" else "SELL"
            entry_price = entry["entry_price"]
            
            # Calculate stop loss - use stop run extreme with buffer
            atr = df["atr"].iloc[-1]
            pip_size = self._get_pip_size_for_instrument(symbol)
            logger.debug(f"ATR: {atr:.5f}, Pip size: {pip_size:.5f}")
            
            if direction == "BUY":
                # For long trades, calculate buffer based on ATR and instrument type
                sl_buffer = 20 * pip_size if self.config.get("fixed_sl", False) else max(atr * 0.5, 5 * pip_size)
                
                # For crypto pairs, use larger buffer
                if symbol.endswith('USDm') or symbol.endswith('USDT') or any(crypto in symbol for crypto in ['BTC', 'ETH', 'LTC', 'XRP', 'DOGE']):
                    sl_buffer = 20 * pip_size if self.config.get("fixed_sl", False) else max(atr * 1.0, 10 * pip_size)
                
                # For long trades, stop loss is below the stop run low
                stop_loss = stop_run["low"] - sl_buffer
                logger.debug(f"BUY stop loss: {stop_loss:.5f} (stop run low {stop_run['low']:.5f} - buffer {sl_buffer:.5f})")
                
                # Calculate take profit using risk-reward ratio
                risk = entry_price - stop_loss
                take_profit = entry_price + (risk * self.config["min_rr_ratio"])
                logger.debug(f"BUY take profit: {take_profit:.5f} (risk: {risk:.5f}, RR: {self.config['min_rr_ratio']})")
            else:
                # For short trades, calculate buffer based on ATR and instrument type
                sl_buffer = 20 * pip_size if self.config.get("fixed_sl", False) else max(atr * 0.5, 5 * pip_size)
                
                # For crypto pairs, use larger buffer
                if symbol.endswith('USDm') or symbol.endswith('USDT') or any(crypto in symbol for crypto in ['BTC', 'ETH', 'LTC', 'XRP', 'DOGE']):
                    sl_buffer = 20 * pip_size if self.config.get("fixed_sl", False) else max(atr * 1.0, 10 * pip_size)
                
                # For short trades, stop loss is above the stop run high
                stop_loss = stop_run["high"] + sl_buffer
                logger.debug(f"SELL stop loss: {stop_loss:.5f} (stop run high {stop_run['high']:.5f} + buffer {sl_buffer:.5f})")
                
                # Calculate take profit using risk-reward ratio
                risk = stop_loss - entry_price
                take_profit = entry_price - (risk * self.config["min_rr_ratio"])
                logger.debug(f"SELL take profit: {take_profit:.5f} (risk: {risk:.5f}, RR: {self.config['min_rr_ratio']})")
            
            # Calculate position size
            position_size = 0.01  # Default
            if self.risk_manager and account_info:
                try:
                    logger.debug(f"Calculating position size with risk manager")
                    position_size = self.risk_manager.calculate_position_size(
                        account_balance=account_info.get("balance", 10000),
                        risk_per_trade=0.01,
                        entry_price=entry_price,
                        stop_loss_price=stop_loss,
                        symbol=symbol
                    )
                    logger.debug(f"Calculated position size: {position_size}")
                except Exception as e:
                    logger.error(f"Error calculating position size: {e}")
            
            # Create the signal
            signal = {
                "symbol": symbol,
                "timeframe": timeframe,
                "direction": direction,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "position_size": position_size,
                "strategy": f"Bank_{point['type']}",
                "manipulation_point": point["price"],
                "entry_type": entry["entry_type"],
                "risk_reward": round(self.config["min_rr_ratio"], 2),
                "timestamp": datetime.now().isoformat()
            }
            
            # FIXED: Add score data to help prioritize signals
            if "score" in stop_run:
                signal["stop_run_score"] = stop_run["score"]
                
            if "score" in entry:
                signal["entry_score"] = entry["score"]
                
            # Calculate overall quality score for signal prioritization
            quality_score = 0
            
            # Add points for entry type
            if entry["entry_type"] == "fib_618":
                quality_score += 3
            elif entry["entry_type"] == "fib_500":
                quality_score += 2
            elif entry["entry_type"] == "fib_382":
                quality_score += 1
                
            # Add points from stop run quality
            if "score" in stop_run:
                quality_score += stop_run["score"]
                
            # Add the quality score to the signal
            signal["quality_score"] = quality_score
            
            # Add time-based invalidation
            time_invalidation = self._calculate_time_invalidation(timeframe)
            signal["time_invalidation"] = time_invalidation
            
            # Add the signal to the signals list
            signals.append(signal)
            logger.info(f"Added {direction} signal for {symbol} at price {entry_price}")
        
        result = {
            "signals": signals,
            "status": "success" if signals else "no_signals",
            "message": f"Generated {len(signals)} signals for {symbol}"
        }
        
        # FIXED: Prioritize signals to prevent conflicting signals (BUY and SELL at the same time)
        # This fixes the issue of trading bot continuously switching between positions
        if len(signals) > 1:
            logger.info(f"Found {len(signals)} potentially conflicting signals - prioritizing the best one")
            
            # Check if we have signals with opposite directions
            has_buy = any(s["direction"] == "BUY" for s in signals)
            has_sell = any(s["direction"] == "SELL" for s in signals)
            
            if has_buy and has_sell:
                logger.warning(f"Detected conflicting BUY and SELL signals for {symbol}")
                
                # Score each signal for quality (if not already scored during signal creation)
                for signal in signals:
                    if "quality_score" not in signal:
                        score = 0
                        
                        # Factor 1: Entry type quality
                        if signal.get("entry_type") == "fib_618":
                            score += 3  # Deep retracements are high quality
                        elif signal.get("entry_type") == "fib_500":
                            score += 2  # Mid retracements are medium quality
                        elif signal.get("entry_type") == "fib_382":
                            score += 1  # Shallow retracements are lower quality
                            
                        # Factor 2: Stop run quality
                        if "stop_run_score" in signal:
                            score += signal["stop_run_score"]
                        
                        # Factor 3: Entry score
                        if "entry_score" in signal:
                            score += signal["entry_score"]
                            
                        # Factor 4: Higher timeframe alignment
                        direction = signal["direction"].lower()
                        if htf_data.get("daily_trend") and direction[0] == htf_data["daily_trend"][0]:
                            score += 3  # Daily alignment is highest value
                        if htf_data.get("h4_trend") and direction[0] == htf_data["h4_trend"][0]:
                            score += 2  # H4 alignment is medium value
                        if htf_data.get("h1_trend") and direction[0] == htf_data["h1_trend"][0]:
                            score += 1  # H1 alignment is lowest value
                            
                        # Store score in signal
                        signal["quality_score"] = score
                    
                    # Log the score details
                    logger.debug(f"Signal {signal['direction']} on {symbol} scored {signal.get('quality_score', 0)} [entry_type: {signal.get('entry_type', 'unknown')}, run_score: {signal.get('stop_run_score', 0)}, entry_score: {signal.get('entry_score', 0)}]")
                
                # Sort by quality score (higher is better)
                signals.sort(key=lambda s: s.get("quality_score", 0), reverse=True)
                
                # Keep only the highest quality signal
                best_signal = signals[0]
                logger.info(f"Selected highest quality signal: {best_signal['direction']} (score: {best_signal.get('quality_score', 0)})")
                
                # Replace signals list with only the best one
                signals = [best_signal]
                result["signals"] = signals
                result["message"] = f"Generated 1 signal for {symbol} (prioritized from {len(signals)} conflicting signals)"
                
            else:
                # Even if there are multiple signals with the same direction,
                # prioritize the highest quality one
                if len(signals) > 1:
                    # Sort by quality score (higher is better)
                    signals.sort(key=lambda s: s.get("quality_score", 0), reverse=True)
                    
                    # Keep only the highest quality signal
                    best_signal = signals[0]
                    logger.info(f"Selected highest quality signal among {len(signals)} {best_signal['direction']} signals (score: {best_signal.get('quality_score', 0)})")
                    
                    # Replace signals list with only the best one
                    signals = [best_signal]
                    result["signals"] = signals
        
        # NEW: Apply cooldown period to prevent rapid trading
        if signals:
            current_time = datetime.now().timestamp()
            cooldown_seconds = self.config.get("signal_cooldown_seconds", 900)  # Default 15 minutes
            
            # Check if we've generated a signal for this symbol recently
            last_signal_time = self.last_signal_time.get(symbol, 0)
            time_since_last_signal = current_time - last_signal_time
            
            if last_signal_time > 0 and time_since_last_signal < cooldown_seconds:
                # Signal is within cooldown period, skip it
                remaining_cooldown = cooldown_seconds - time_since_last_signal
                logger.warning(f"Signal for {symbol} rejected due to cooldown period ({remaining_cooldown:.0f} seconds remaining)")
                
                signals = []  # Clear signals list
                result["signals"] = []
                result["status"] = "cooldown"
                result["message"] = f"No signals generated for {symbol} due to cooldown period"
            else:
                # Update last signal time for this symbol
                self.last_signal_time[symbol] = current_time
                logger.debug(f"Updated last signal time for {symbol}: {current_time}")
        
        logger.info(f"=== Completed signal generation for {symbol}: {len(signals)} signals generated ===")
        return result
        
    async def _check_higher_timeframe_alignment(self, symbol: str) -> Dict:
        """
        Check higher timeframe alignment for market bias.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict with alignment information
        """
        htf_data = {
            "h1_trend": None,
            "h4_trend": None,
            "daily_trend": None,
            "alignment": False
        }
        
        if not self.mt5_handler:
            logger.warning("MT5 handler not available, skipping HTF analysis")
            return htf_data
            
        try:
            # Get H1 data
            h1_data = self.mt5_handler.get_market_data(symbol, "H1", num_candles=50)
            if h1_data is not None and len(h1_data) > 20:
                # Add EMA using pandas instead of talib
                h1_data["ema50"] = h1_data["close"].ewm(span=50, adjust=False).mean()
                h1_data["ema200"] = h1_data["close"].ewm(span=200, adjust=False).mean()
                
                # Determine trend
                current_close = h1_data["close"].iloc[-1]
                ema50 = h1_data["ema50"].iloc[-1]
                ema200 = h1_data["ema200"].iloc[-1]
                
                if current_close > ema50 and ema50 > ema200:
                    htf_data["h1_trend"] = "bullish"
                elif current_close < ema50 and ema50 < ema200:
                    htf_data["h1_trend"] = "bearish"
                else:
                    htf_data["h1_trend"] = "neutral"
                    
            # Get H4 data
            h4_data = self.mt5_handler.get_market_data(symbol, "H4", num_candles=50)
            if h4_data is not None and len(h4_data) > 20:
                # Add EMA using pandas instead of talib
                h4_data["ema50"] = h4_data["close"].ewm(span=50, adjust=False).mean()
                h4_data["ema200"] = h4_data["close"].ewm(span=200, adjust=False).mean()
                
                # Determine trend
                current_close = h4_data["close"].iloc[-1]
                ema50 = h4_data["ema50"].iloc[-1]
                ema200 = h4_data["ema200"].iloc[-1]
                
                if current_close > ema50 and ema50 > ema200:
                    htf_data["h4_trend"] = "bullish"
                elif current_close < ema50 and ema50 < ema200:
                    htf_data["h4_trend"] = "bearish"
                else:
                    htf_data["h4_trend"] = "neutral"
                    
            # Get daily data
            daily_data = self.mt5_handler.get_market_data(symbol, "D1", num_candles=50)
            if daily_data is not None and len(daily_data) > 20:
                # Add EMA using pandas instead of talib
                daily_data["ema50"] = daily_data["close"].ewm(span=50, adjust=False).mean()
                daily_data["ema200"] = daily_data["close"].ewm(span=200, adjust=False).mean()
                
                # Determine trend
                current_close = daily_data["close"].iloc[-1]
                ema50 = daily_data["ema50"].iloc[-1]
                ema200 = daily_data["ema200"].iloc[-1]
                
                if current_close > ema50 and ema50 > ema200:
                    htf_data["daily_trend"] = "bullish"
                elif current_close < ema50 and ema50 < ema200:
                    htf_data["daily_trend"] = "bearish"
                else:
                    htf_data["daily_trend"] = "neutral"
                    
            logger.info(f"Higher timeframe analysis: H1={htf_data['h1_trend']}, H4={htf_data['h4_trend']}, D1={htf_data['daily_trend']}")
            
        except Exception as e:
            logger.error(f"Error during higher timeframe analysis: {e}")
            
        return htf_data
        
    def _is_htf_aligned(self, htf_data: Dict, direction: str) -> bool:
        """
        Check if the signal direction is aligned with higher timeframe trends.
        More flexible alignment rules to allow more valid setups.
        
        Args:
            htf_data: Higher timeframe data dict
            direction: Signal direction ("long" or "short")
            
        Returns:
            True if aligned, False otherwise
        """
        logger.debug(f"Checking higher timeframe alignment for {direction} direction")
        
        # Default to True if no higher timeframe data is available
        if not htf_data or all(v is None for k, v in htf_data.items() if k != 'alignment'):
            logger.debug("No higher timeframe data available, assuming alignment")
            return True
            
        # Count how many timeframes align with our direction
        aligned_count = 0
        total_available = 0
        
        # Check H1 alignment
        if htf_data.get("h1_trend"):
            total_available += 1
            if (direction == "long" and htf_data["h1_trend"] == "bullish") or \
               (direction == "short" and htf_data["h1_trend"] == "bearish"):
                aligned_count += 1
                logger.debug(f"H1 trend ({htf_data['h1_trend']}) aligns with {direction} direction: +1")
            else:
                logger.debug(f"H1 trend ({htf_data['h1_trend']}) does not align with {direction} direction: +0")
                
        # Check H4 alignment (more weight)
        if htf_data.get("h4_trend"):
            total_available += 2  # Give more weight to H4
            if (direction == "long" and htf_data["h4_trend"] == "bullish") or \
               (direction == "short" and htf_data["h4_trend"] == "bearish"):
                aligned_count += 2
                logger.debug(f"H4 trend ({htf_data['h4_trend']}) aligns with {direction} direction: +2")
            else:
                logger.debug(f"H4 trend ({htf_data['h4_trend']}) does not align with {direction} direction: +0")
                
        # Check Daily alignment (most weight)
        if htf_data.get("daily_trend"):
            total_available += 3  # Give most weight to Daily
            if (direction == "long" and htf_data["daily_trend"] == "bullish") or \
               (direction == "short" and htf_data["daily_trend"] == "bearish"):
                aligned_count += 3
                logger.debug(f"Daily trend ({htf_data['daily_trend']}) aligns with {direction} direction: +3")
            else:
                logger.debug(f"Daily trend ({htf_data['daily_trend']}) does not align with {direction} direction: +0")
                
        # Calculate alignment percentage
        alignment_pct = (aligned_count / total_available) if total_available > 0 else 0
        
        # Consider aligned if at least 40% of available timeframes align (reduced from 50%)
        # OR if H1 aligns (for countertrend opportunities)
        is_aligned = alignment_pct >= 0.4 or (htf_data.get("h1_trend") and 
            ((direction == "long" and htf_data["h1_trend"] == "bullish") or 
             (direction == "short" and htf_data["h1_trend"] == "bearish")))
        
        # FIXED: If alignment is below 25%, require that H1 AND (H4 OR Daily) align with direction
        # This prevents accepting signals with extremely low alignment percentages like 17%
        if alignment_pct < 0.25:
            # Need stronger evidence for very low alignment
            h1_aligned = htf_data.get("h1_trend") and (
                (direction == "long" and htf_data["h1_trend"] == "bullish") or 
                (direction == "short" and htf_data["h1_trend"] == "bearish")
            )
            h4_aligned = htf_data.get("h4_trend") and (
                (direction == "long" and htf_data["h4_trend"] == "bullish") or 
                (direction == "short" and htf_data["h4_trend"] == "bearish")
            )
            daily_aligned = htf_data.get("daily_trend") and (
                (direction == "long" and htf_data["daily_trend"] == "bullish") or 
                (direction == "short" and htf_data["daily_trend"] == "bearish")
            )
            
            # Must have H1 AND (H4 OR Daily) aligned for very low alignment percentages
            if not (h1_aligned and (h4_aligned or daily_aligned)):
                is_aligned = False
                logger.debug(f"Rejecting low alignment signal ({alignment_pct:.2f}) without strong timeframe support")
        
        logger.debug(f"Higher timeframe alignment: {aligned_count}/{total_available} = {alignment_pct:.2f} for {direction} direction, aligned: {is_aligned}")
        
        return is_aligned
    
    def _create_signal_from_pattern(self, pattern: Dict, symbol: str, timeframe: str, account_info: Optional[Dict] = None) -> Optional[Dict]:
        """Create a trading signal from a detected pattern.
        
        Args:
            pattern: Pattern details
            symbol: Trading symbol
            timeframe: Trading timeframe
            account_info: Account information for position sizing
            
        Returns:
            Trading signal dictionary or None if creation fails
        """
        try:
            direction = "BUY" if pattern["type"] == "long" else "SELL"
            entry_price = pattern["entry"]
            stop_loss = pattern["stop_loss"]
            
            # Calculate take profit if not provided
            if "target" in pattern:
                take_profit = pattern["target"]
            else:
                risk = abs(entry_price - stop_loss)
                take_profit = entry_price + (risk * self.config["min_rr_ratio"]) if direction == "BUY" else \
                             entry_price - (risk * self.config["min_rr_ratio"])
            
            # Calculate position size
            position_size = 0.01  # Default
            if self.risk_manager and account_info:
                try:
                    position_size = self.risk_manager.calculate_position_size(
                        account_balance=account_info.get("balance", 10000),
                        risk_per_trade=0.01,
                        entry_price=entry_price,
                        stop_loss_price=stop_loss,
                        symbol=symbol
                    )
                except Exception as e:
                    logger.error(f"Error calculating position size: {e}")
            
            # Calculate risk-reward ratio
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            risk_reward = round(reward / risk, 2) if risk > 0 else 0
            
            signal = {
                "symbol": symbol,
                "timeframe": timeframe,
                "direction": direction,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "position_size": position_size,
                "strategy": f"Bank_{pattern['type'].capitalize()}",
                "confidence": pattern["confidence"],
                "risk_reward": risk_reward,
                "timestamp": datetime.now().isoformat()
            }
            
            return signal
            
        except Exception as e:
            logger.error(f"Error creating signal: {e}")
        return None

    def _add_basic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add essential indicators to the dataframe for trade analysis.
        
        Args:
            df: Price dataframe with OHLCV data
            
        Returns:
            Dataframe with added indicators
        """
        # Add ATR for volatility measurement
        df["atr"] = self.market_analyzer.calculate_atr(df, 14)
        
        # Add simple EMA-based trend
        df["ema20"] = df["close"].ewm(span=20).mean()
        df["ema50"] = df["close"].ewm(span=50).mean()
        
        # Add simple trend direction indicator
        df["trend"] = 0  # Neutral
        df.loc[df["ema20"] > df["ema50"], "trend"] = 1  # Bullish
        df.loc[df["ema20"] < df["ema50"], "trend"] = -1  # Bearish
        
        # Add RSI for divergence detection
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        logger.debug("Added basic indicators to dataframe")
        return df
    
    def _detect_liquidity_grab_pattern(self, df: pd.DataFrame, symbol: str) -> Optional[Dict]:
        """Detect liquidity grab patterns (stop hunts) with immediate reversal potential.
        
        Args:
            df: Dataframe with price data and indicators
            symbol: Trading symbol
        
        Returns:
            Dictionary with pattern details if found, None otherwise
        """
        # Ensure minimum data requirement
        if len(df) < 20:
            logger.warning(f"Insufficient data for liquidity grab pattern detection: {len(df)} candles")
            return None
            
        try:
            # Get recent high/low
            recent_high = df["high"].rolling(window=min(20, len(df))).max().iloc[-1]
            recent_low = df["low"].rolling(window=min(20, len(df))).min().iloc[-1]
            
            # Calculate ATR for dynamic levels
            atr = df["atr"].iloc[-1] if "atr" in df.columns else self.market_analyzer.calculate_atr(df, 14).iloc[-1]
            
            # Get pip size for this instrument
            pip_size = self._get_pip_size_for_instrument(symbol)
            
            # Look at last few candles (use min to prevent slicing beyond df length)
            last_n = min(3, len(df))
            last_candles = df.iloc[-last_n:]
            last_candle = last_candles.iloc[-1]
            
            logger.debug(f"Checking for liquidity grab pattern on {symbol}")
            logger.debug(f"Recent high: {recent_high:.5f}, Recent low: {recent_low:.5f}")
            
            # Detect bearish liquidity grab (stop run above recent high)
            if (last_candles["high"].max() > recent_high and 
                last_candle["close"] < recent_high):
                # Check if the price movement is significant
                if (last_candles["high"].max() - recent_high) / pip_size >= 3:
                    logger.info(f"Detected potential SHORT liquidity grab on {symbol}")
                    return {
                        "type": "short",
                        "level": recent_high,
                        "entry": last_candle["close"],
                        "stop_loss": last_candles["high"].max() + atr,
                        "confidence": self._calculate_pattern_strength(df, "short")
                    }
            
            # Detect bullish liquidity grab (stop run below recent low)
            if (last_candles["low"].min() < recent_low and 
                last_candle["close"] > recent_low):
                # Check if the price movement is significant
                if (recent_low - last_candles["low"].min()) / pip_size >= 3:
                    logger.info(f"Detected potential LONG liquidity grab on {symbol}")
                    return {
                        "type": "long",
                        "level": recent_low,
                        "entry": last_candle["close"],
                        "stop_loss": last_candles["low"].min() - atr,
                        "confidence": self._calculate_pattern_strength(df, "long")
                    }
            
            logger.debug(f"No liquidity grab pattern detected on {symbol}")
            return None
            
        except Exception as e:
            logger.error(f"Error in liquidity grab pattern detection: {e}")
            return None
    
    def _detect_ob_reversal(self, df: pd.DataFrame, symbol: str) -> Optional[Dict]:
        """Detect order block reversal patterns.
        
        Args:
            df: Dataframe with price data and indicators
            symbol: Trading symbol
            
        Returns:
            Dictionary with pattern details if found, None otherwise
        """
        logger.debug(f"Checking for order block reversal on {symbol}")
        
        # Look for bullish order blocks (bearish candles before price moves up)
        if df["close"].iloc[-1] > df["open"].iloc[-1]:  # Current candle is bullish
            # Find the most recent bearish candle
            for i in range(2, 10):
                if i >= len(df):
                    break
                candle = df.iloc[-i]
                if candle["close"] < candle["open"]:  # Bearish candle (potential order block)
                    # Check if we've retraced to this order block
                    if (df["low"].iloc[-1] <= candle["high"] and 
                        df["close"].iloc[-1] > candle["low"]):
                        logger.info(f"Detected potential LONG order block reversal on {symbol}")
                        
                        # Calculate dynamic stop and target
                        entry = df["close"].iloc[-1]
                        stop_loss = min(candle["low"], entry - self._get_pip_size_for_instrument(symbol) * 10)
                        risk = entry - stop_loss
                        target = entry + (risk * self.config["min_rr_ratio"])
                        
                        return {
                            "type": "long",
                            "entry": entry,
                            "stop_loss": stop_loss,
                            "target": target,
                            "confidence": self._calculate_pattern_strength(df, "long")
                        }
        
        # Look for bearish order blocks (bullish candles before price moves down)
        if df["close"].iloc[-1] < df["open"].iloc[-1]:  # Current candle is bearish
            # Find the most recent bullish candle
            for i in range(2, 10):
                if i >= len(df):
                    break
                candle = df.iloc[-i]
                if candle["close"] > candle["open"]:  # Bullish candle (potential order block)
                    # Check if we've retraced to this order block
                    if (df["high"].iloc[-1] >= candle["low"] and 
                        df["close"].iloc[-1] < candle["high"]):
                        logger.info(f"Detected potential SHORT order block reversal on {symbol}")
                        
                        # Calculate dynamic stop and target
                        entry = df["close"].iloc[-1]
                        stop_loss = max(candle["high"], entry + self._get_pip_size_for_instrument(symbol) * 10)
                        risk = stop_loss - entry
                        target = entry - (risk * self.config["min_rr_ratio"])
                        
                        return {
                            "type": "short",
                            "entry": entry,
                            "stop_loss": stop_loss,
                            "target": target,
                            "confidence": self._calculate_pattern_strength(df, "short")
                        }
        
        logger.debug(f"No order block reversal detected on {symbol}")
        return None
        
    def _calculate_pattern_strength(self, df: pd.DataFrame, direction: str) -> float:
        """Calculate pattern strength score (0-1) based on simple factors.
        
        Args:
            df: Dataframe with price data and indicators
            direction: Trade direction ('long' or 'short')
        
        Returns:
            Pattern strength score from 0.0 to 1.0
        """
        score = 0
        
        # Last 3 candles
        last_candles = df.iloc[-3:]
        
        # 1. Volatility factor - higher volatility = stronger pattern
        atr = df["atr"].iloc[-1] if "atr" in df.columns else self.market_analyzer.calculate_atr(df, 14).iloc[-1]
        avg_range = df["high"].iloc[-10:] - df["low"].iloc[-10:]
        if atr > avg_range.mean() * 1.2:
            score += 2
            logger.debug("Pattern strength: +2 for higher volatility")
        
        # 2. Volume confirmation
        if "volume" in df.columns and last_candles["volume"].iloc[-1] > last_candles["volume"].iloc[:-1].mean() * 1.2:
            score += 2
            logger.debug("Pattern strength: +2 for volume confirmation")
        
        # 3. Trend alignment
        if "trend" in df.columns:
            if (direction == "long" and df["trend"].iloc[-1] >= 0) or \
               (direction == "short" and df["trend"].iloc[-1] <= 0):
                score += 2
                logger.debug("Pattern strength: +2 for trend alignment")
        elif "ema20" in df.columns and "ema50" in df.columns:
            # Use EMAs if trend column not available
            if (direction == "long" and df["ema20"].iloc[-1] > df["ema50"].iloc[-1]) or \
               (direction == "short" and df["ema20"].iloc[-1] < df["ema50"].iloc[-1]):
                score += 2
                logger.debug("Pattern strength: +2 for EMA alignment")
        
        # 4. Candle pattern strength
        last_candle = df.iloc[-1]
        if (direction == "long" and last_candle["close"] > last_candle["open"] and
            (last_candle["close"] - last_candle["open"]) / (last_candle["high"] - last_candle["low"]) > 0.6):
            score += 2
            logger.debug("Pattern strength: +2 for strong bullish candle")
        elif (direction == "short" and last_candle["close"] < last_candle["open"] and
              (last_candle["open"] - last_candle["close"]) / (last_candle["high"] - last_candle["low"]) > 0.6):
            score += 2
            logger.debug("Pattern strength: +2 for strong bearish candle")
        
        # 5. RSI confirmation
        if "rsi" in df.columns:
            if (direction == "long" and df["rsi"].iloc[-1] < 40 and df["rsi"].iloc[-1] > df["rsi"].iloc[-2]) or \
               (direction == "short" and df["rsi"].iloc[-1] > 60 and df["rsi"].iloc[-1] < df["rsi"].iloc[-2]):
                score += 2
                logger.debug("Pattern strength: +2 for RSI confirmation")
        
        # Convert to 0-1 scale
        normalized_score = min(score / 10.0, 1.0)
        logger.debug(f"Final pattern strength: {normalized_score:.2f} (raw score: {score})")
        
        return normalized_score

    def _identify_manipulation_points(self, df: pd.DataFrame, symbol: str) -> List[Dict]:
        """Identify manipulation points based on recent and previous day's highs/lows.
        
        Args:
            df: Dataframe with price data
            symbol: Trading symbol
            
        Returns:
            List of manipulation points
        """
        manipulation_points = []
        unique_prices = set()  # Track unique price points to prevent duplicates
        logger.debug(f"Identifying manipulation points for {symbol}")
        
        # For M15 timeframe, we have 96 candles per day (4 candles per hour * 24 hours)
        day_candles = 96  # Fixed for M15 timeframe
        
        # Adjust for smaller datasets
        if len(df) < day_candles:
            logger.debug(f"Limited data available ({len(df)} candles), using all available data for recent high/low")
            day_candles = len(df)
        
        # Recent swing high/low (last 24 hours or all available data)
        if self.config["manipulation_points"]["recent_high_low"]:
            # Use a smaller window if we have limited data
            window_size = min(day_candles, len(df) - 1)
            if window_size > 0:
                # Use the last window_size candles for recent high/low
                recent_slice = df.iloc[-window_size:]
                recent_high = recent_slice["high"].max()
                recent_low = recent_slice["low"].min()
                
                logger.debug(f"Recent high/low calculated from last {window_size} candles")
                logger.debug(f"Recent high: {recent_high:.5f}, Recent low: {recent_low:.5f}")
                
                # Only add if the price is unique (with small tolerance)
                if not any(abs(recent_high - price) < 0.0001 for price in unique_prices):
                    manipulation_points.append({"type": "recent_high", "price": recent_high})
                    unique_prices.add(recent_high)
                    logger.debug(f"Added recent high: {recent_high:.5f}")
                    
                if not any(abs(recent_low - price) < 0.0001 for price in unique_prices):
                    manipulation_points.append({"type": "recent_low", "price": recent_low})
                    unique_prices.add(recent_low)
                    logger.debug(f"Added recent low: {recent_low:.5f}")
            else:
                logger.warning(f"Not enough data to calculate recent high/low for {symbol}")

        # Previous day's high/low (using MT5Handler.get_historical_data)
        if self.config["manipulation_points"]["previous_day_high_low"]:
            previous_day_df = None
            
            # Try to get data using whichever handler is available
            handler = None
            
            # 1. Try direct mt5_handler
            if self.mt5_handler:
                handler = self.mt5_handler
            # 2. Try risk_manager's mt5_handler
            elif self.risk_manager and hasattr(self.risk_manager, 'mt5_handler') and self.risk_manager.mt5_handler:
                handler = self.risk_manager.mt5_handler
                
            if handler:
                try:
                    # Get daily data for the past 2 days
                    previous_day_df = handler.get_market_data(symbol, "D1", num_candles=2)
                    if previous_day_df is not None and len(previous_day_df) > 0:
                        logger.debug(f"Successfully fetched previous day data: {len(previous_day_df)} candles")
                    else:
                        logger.warning(f"Failed to get previous day data for {symbol}")
                except Exception as e:
                    logger.warning(f"Failed to get historical data: {str(e)}")
                    
            # If we couldn't get historical data, try to extract daily high/low from the current dataframe
            if previous_day_df is None or len(previous_day_df) == 0:
                # Try to estimate previous day high/low from existing data
                logger.debug(f"Using existing dataframe to estimate previous day high/low")
                
                # If we have enough data, try to use the first half of the data for previous day
                if len(df) >= day_candles * 1.5:  # Need at least 1.5 days of data
                    # Use the first half of the data for previous day
                    prev_day_slice = df.iloc[:-day_candles]
                    
                    if len(prev_day_slice) > 0:
                        previous_high = prev_day_slice["high"].max()
                        previous_low = prev_day_slice["low"].min()
                        
                        logger.debug(f"Estimated previous day high: {previous_high:.5f}, Previous day low: {previous_low:.5f}")
                        
                        # Only add if the price is unique
                        if not any(abs(previous_high - price) < 0.0001 for price in unique_prices):
                            manipulation_points.append({"type": "previous_day_high", "price": previous_high})
                            unique_prices.add(previous_high)
                            logger.debug(f"Added previous day high: {previous_high:.5f}")
                            
                        if not any(abs(previous_low - price) < 0.0001 for price in unique_prices):
                            manipulation_points.append({"type": "previous_day_low", "price": previous_low})
                            unique_prices.add(previous_low)
                            logger.debug(f"Added previous day low: {previous_low:.5f}")
                    else:
                        logger.debug(f"Not enough data to estimate previous day high/low")
                else:
                    logger.debug(f"Not enough data to estimate previous day high/low: only {len(df)} candles available")
            else:
                # Use historical data as originally intended
                if len(previous_day_df) > 0:
                    previous_high = previous_day_df["high"].iloc[0]  # First candle is the previous day
                    previous_low = previous_day_df["low"].iloc[0]
                    
                    logger.debug(f"Previous day high: {previous_high:.5f}, Previous day low: {previous_low:.5f}")
                    
                    # Only add if the price is unique
                    if not any(abs(previous_high - price) < 0.0001 for price in unique_prices):
                        manipulation_points.append({"type": "previous_day_high", "price": previous_high})
                        unique_prices.add(previous_high)
                        logger.debug(f"Added previous day high: {previous_high:.5f}")
                        
                    if not any(abs(previous_low - price) < 0.0001 for price in unique_prices):
                        manipulation_points.append({"type": "previous_day_low", "price": previous_low})
                        unique_prices.add(previous_low)
                        logger.debug(f"Added previous day low: {previous_low:.5f}")

        # Add sharp reversal points as additional manipulation points
        logger.debug(f"Detecting sharp reversals for {symbol}")
        sharp_reversals = self._detect_sharp_reversal(df)
        
        for reversal in sharp_reversals:
            if not any(abs(reversal["price"] - price) < 0.0001 for price in unique_prices):
                # Convert the reversal to manipulation point format
                manipulation_points.append({
                    "type": reversal["type"],
                    "price": reversal["price"],
                    "strength": reversal.get("strength", 0.5)  # Include strength metric
                })
                unique_prices.add(reversal["price"])
                logger.debug(f"Added {reversal['type']} manipulation point at {reversal['price']:.5f}")

        # Filter points within ADR range from current price
        if "atr" in df.columns:
            atr = df["atr"].iloc[-1]
        else:
            atr = self.market_analyzer.calculate_atr(df, 14).iloc[-1]
            
        current_price = df["close"].iloc[-1]
        pip_size = self._get_pip_size_for_instrument(symbol)
        adr_pips = atr / pip_size
        
        logger.debug(f"Current price: {current_price:.5f}, ATR: {atr:.5f}, ADR: {adr_pips:.1f} pips")
        
        # Use ADR-based filtering (half of ADR above and below current price)
        max_distance_pips = adr_pips / 2
        
        # Ensure minimum range of 30 pips for low volatility pairs
        max_distance_pips = max(max_distance_pips, 30.0)
        
        logger.debug(f"Using maximum distance of {max_distance_pips:.1f} pips for manipulation points")
        
        filtered_points = []
        for point in manipulation_points:
            dist_pips = abs(point["price"] - current_price) / pip_size
            
            # Prioritize sharp reversal points by allowing them a greater distance
            if "type" in point and point["type"] in ["bullish_reversal", "bearish_reversal"]:
                # Allow up to 1.5x the normal distance for sharp reversals
                adjusted_max_distance = max_distance_pips * 1.5
                logger.debug(f"Using adjusted max distance of {adjusted_max_distance:.1f} pips for sharp reversal point")
                
                if dist_pips <= adjusted_max_distance:
                    filtered_points.append(point)
                    logger.debug(f"Kept sharp reversal point: {point['type']} at {point['price']:.5f}, distance: {dist_pips:.1f} pips (within {adjusted_max_distance:.1f} pips)")
                else:
                    logger.debug(f"Filtered out sharp reversal point: {point['type']} at {point['price']:.5f}, distance: {dist_pips:.1f} pips (exceeds {adjusted_max_distance:.1f} pips)")
            else:
                # Use standard distance for normal manipulation points
                if dist_pips <= max_distance_pips:
                    filtered_points.append(point)
                    logger.debug(f"Kept manipulation point: {point['type']} at {point['price']:.5f}, distance: {dist_pips:.1f} pips (within {max_distance_pips:.1f} pips)")
                else:
                    logger.debug(f"Filtered out manipulation point: {point['type']} at {point['price']:.5f}, distance: {dist_pips:.1f} pips (exceeds {max_distance_pips:.1f} pips)")
        
        logger.debug(f"Manipulation points after filtering: {len(filtered_points)}")
        
        return filtered_points
        
    def _find_stop_run_candle(self, df: pd.DataFrame, point: Dict, symbol: str) -> Optional[Dict]:
        """Find a stop run candle that breaks a manipulation point.
        
        Args:
            df: Price dataframe
            point: Manipulation point dictionary
            symbol: Trading symbol
            
        Returns:
            Dictionary with stop run candle details or None if not found
        """
        # Ensure we have enough data
        if len(df) < 5:  # Minimum required for basic analysis
            logger.warning(f"Insufficient data for stop run detection: {len(df)} candles")
            return None
            
        try:
            logger.debug(f"Looking for stop run candle for {point['type']} at {point['price']:.5f}")
            
            # Get pip size for this instrument
            pip_size = self._get_pip_size_for_instrument(symbol)
            
            # More flexible threshold for stop run (reduced from 5 pips)
            threshold_pips = 3  # Reduced from 5 to 3 pips
            threshold = threshold_pips * pip_size
            
            # Get current price
            current_price = df["close"].iloc[-1]
            
            # Determine direction based on manipulation point type and price
            if point["type"] in ["recent_high", "previous_day_high", "bearish_reversal"] and current_price < point["price"]:
                direction = "short"  # Looking for a bearish stop run
                logger.debug(f"Looking for SHORT stop run (current price {current_price:.5f} below point {point['price']:.5f})")
            elif point["type"] in ["recent_low", "previous_day_low", "bullish_reversal"] and current_price > point["price"]:
                direction = "long"  # Looking for a bullish stop run
                logger.debug(f"Looking for LONG stop run (current price {current_price:.5f} above point {point['price']:.5f})")
            else:
                logger.debug(f"Point {point['type']} at {point['price']:.5f} not valid for stop run (current price: {current_price:.5f})")
                return None
            
            # Look for stop run in the last 15 candles (increased from 10)
            max_lookback = min(15, len(df) - 1)  # Ensure we don't exceed dataframe length
            
            # Track the best candidate
            best_candidate = None
            best_score = 0
            
            for i in range(1, max_lookback + 1):
                candle_idx = len(df) - i
                if candle_idx < 0:
                    continue
                    
                candle = df.iloc[candle_idx]
                next_candle = df.iloc[candle_idx + 1] if candle_idx + 1 < len(df) else None
                
                # Implement 2-candle rule: if two consecutive candles close beyond the manipulation point, invalidate the setup
                if i > 1:  # Skip this check for the most recent candle
                    prev_candle = df.iloc[candle_idx - 1] if candle_idx > 0 else None
                    if prev_candle is not None:
                        if (direction == "short" and 
                            candle["close"] > point["price"] and 
                            prev_candle["close"] > point["price"]):
                            logger.debug(f"Potential stop run at index {candle_idx} invalidated by 2-candle rule (two consecutive closes above the level)")
                            continue
                        elif (direction == "long" and 
                              candle["close"] < point["price"] and 
                              prev_candle["close"] < point["price"]):
                            logger.debug(f"Potential stop run at index {candle_idx} invalidated by 2-candle rule (two consecutive closes below the level)")
                            continue
                
                if direction == "short":
                    # For short setup, we need a candle that breaks above the manipulation point
                    if candle["high"] > point["price"]:
                        # Calculate score based on various factors
                        score = 0
                        
                        # Factor 1: How far it broke above the level
                        break_distance = (candle["high"] - point["price"]) / pip_size
                        if break_distance >= threshold_pips:
                            score += 2
                        elif break_distance >= 1:
                            score += 1
                        
                        # Factor 2: Closed back below the level
                        if candle["close"] < point["price"]:
                            score += 2
                        elif candle["close"] < candle["open"]:  # At least bearish
                            score += 1
                        
                        # Factor 3: Next candle continuation
                        if next_candle is not None and next_candle["close"] < next_candle["open"]:
                            score += 1
                        
                        # Factor 4: Check for strong reversal signals (enhancing score)
                        if candle["high"] - candle["close"] > (candle["high"] - candle["low"]) * 0.6:
                            # Large upper wick showing strong rejection
                            score += 2
                            logger.debug(f"Added +2 to score for large upper wick (strong rejection)")
                        
                        # Volume-based scoring (if volume data is available)
                        if "volume" in df.columns:
                            vol_avg = df["volume"].iloc[max(0, candle_idx-5):candle_idx].mean()
                            if candle["volume"] > vol_avg * 1.5:
                                score += 1
                                logger.debug(f"Added +1 to score for high volume on stop run candle")
                        
                        if score > best_score:
                            best_score = score
                            best_candidate = {
                                "direction": "short",
                                "candle_index": candle_idx,
                                "high": candle["high"],
                                "low": candle["low"],
                                "close": candle["close"],
                                "manipulation_point": point["price"],
                                "score": score,  # Store the quality score
                                "stop_run_score": score  # FIXED: Added clearer name for score
                            }
                            logger.debug(f"Found potential SHORT stop run at index {candle_idx} with score {score}")
                        
                else:  # long
                    # For long setup, we need a candle that breaks below the manipulation point
                    if candle["low"] < point["price"]:
                        # Calculate score based on various factors
                        score = 0
                        
                        # Factor 1: How far it broke below the level
                        break_distance = (point["price"] - candle["low"]) / pip_size
                        if break_distance >= threshold_pips:
                            score += 2
                        elif break_distance >= 1:
                            score += 1
                        
                        # Factor 2: Closed back above the level
                        if candle["close"] > point["price"]:
                            score += 2
                        elif candle["close"] > candle["open"]:  # At least bullish
                            score += 1
                        
                        # Factor 3: Next candle continuation
                        if next_candle is not None and next_candle["close"] > next_candle["open"]:
                            score += 1
                        
                        # Factor 4: Check for strong reversal signals (enhancing score)
                        if candle["close"] - candle["low"] > (candle["high"] - candle["low"]) * 0.6:
                            # Large lower wick showing strong rejection
                            score += 2
                            logger.debug(f"Added +2 to score for large lower wick (strong rejection)")
                        
                        # Volume-based scoring (if volume data is available)
                        if "volume" in df.columns:
                            vol_avg = df["volume"].iloc[max(0, candle_idx-5):candle_idx].mean()
                            if candle["volume"] > vol_avg * 1.5:
                                score += 1
                                logger.debug(f"Added +1 to score for high volume on stop run candle")
                        
                        if score > best_score:
                            best_score = score
                            best_candidate = {
                                "direction": "long",
                                "candle_index": candle_idx,
                                "high": candle["high"],
                                "low": candle["low"],
                                "close": candle["close"],
                                "manipulation_point": point["price"],
                                "score": score,  # Store the quality score
                                "stop_run_score": score  # FIXED: Added clearer name for score
                            }
                            logger.debug(f"Found potential LONG stop run at index {candle_idx} with score {score}")
            
            # Accept candidates with a minimum score
            if best_candidate and best_score >= 2:
                logger.debug(f"Selected stop run candidate with score {best_score}")
                return best_candidate
            
            logger.debug(f"No stop run candle found for {point['type']} at {point['price']:.5f}")
            return None
            
        except Exception as e:
            logger.error(f"Error in stop run candle detection: {e}")
            return None
        
    def _find_confirmation_candle(self, df: pd.DataFrame, stop_run: Dict) -> Optional[Dict]:
        """Find a confirmation candle after the stop run candle that meets the strategy criteria."""
        try:
            logger.debug(f"Looking for confirmation candle after stop run at index {stop_run['candle_index']}")
            
            # Validate input parameters
            if not isinstance(stop_run, dict) or 'candle_index' not in stop_run or 'direction' not in stop_run:
                logger.warning(f"Invalid stop_run parameter: {stop_run}")
                return None
                
            # Ensure we have enough data
            if len(df) <= stop_run["candle_index"] + 1:
                logger.debug("No candles available after stop run")
                return None
            
            # Look for confirmation within next 3 candles instead of just the next one
            max_confirmation_candles = 3
            max_lookback = min(max_confirmation_candles, len(df) - stop_run["candle_index"] - 1)
            
            for i in range(1, max_lookback + 1):
                confirmation_idx = stop_run["candle_index"] + i
                if confirmation_idx >= len(df):
                    break
                    
                confirmation_candle = df.iloc[confirmation_idx]
                stop_run_candle = df.iloc[stop_run["candle_index"]]
                
                if stop_run["direction"] == "short":
                    # For short setup:
                    # 1. Must close below the body of stop run candle
                    # 2. Must be a bearish candle (close < open)
                    # 3. Must close in lower 33% of its range (strict requirement)
                    candle_range = confirmation_candle["high"] - confirmation_candle["low"]
                    if candle_range == 0:  # Avoid division by zero
                        continue
                        
                    close_position = (confirmation_candle["close"] - confirmation_candle["low"]) / candle_range
                    
                    # Stricter confirmation criteria
                    closes_below_body = confirmation_candle["close"] < min(stop_run_candle["open"], stop_run_candle["close"])
                    closes_in_lower_third = close_position <= 0.33  # Strict 1/3 rule
                    is_bearish_candle = confirmation_candle["close"] < confirmation_candle["open"]
                    
                    # Require ALL criteria to be met (stricter)
                    if closes_below_body and closes_in_lower_third and is_bearish_candle:
                        logger.debug(f"Found valid SHORT confirmation at index {confirmation_idx}: " +
                                    f"below_body={closes_below_body}, lower_third={closes_in_lower_third}, " +
                                    f"bearish={is_bearish_candle}, position={close_position:.2f}")
                        return {
                            "candle_index": confirmation_idx,
                            "high": confirmation_candle["high"],
                            "low": confirmation_candle["low"],
                            "close": confirmation_candle["close"],
                            "open": confirmation_candle["open"]
                        }
                
                else:  # long setup
                    # For long setup:
                    # 1. Must close above the body of stop run candle
                    # 2. Must be a bullish candle (close > open)
                    # 3. Must close in upper 33% of its range (strict requirement)
                    candle_range = confirmation_candle["high"] - confirmation_candle["low"]
                    if candle_range == 0:
                        continue
                        
                    close_position = (confirmation_candle["high"] - confirmation_candle["close"]) / candle_range
                    
                    # Stricter confirmation criteria
                    closes_above_body = confirmation_candle["close"] > max(stop_run_candle["open"], stop_run_candle["close"])
                    closes_in_upper_third = close_position <= 0.33  # Strict 1/3 rule
                    is_bullish_candle = confirmation_candle["close"] > confirmation_candle["open"]
                    
                    # Require ALL criteria to be met (stricter)
                    if closes_above_body and closes_in_upper_third and is_bullish_candle:
                        logger.debug(f"Found valid LONG confirmation at index {confirmation_idx}: " +
                                    f"above_body={closes_above_body}, upper_third={closes_in_upper_third}, " +
                                    f"bullish={is_bullish_candle}, position={close_position:.2f}")
                        return {
                            "candle_index": confirmation_idx,
                            "high": confirmation_candle["high"],
                            "low": confirmation_candle["low"],
                            "close": confirmation_candle["close"],
                            "open": confirmation_candle["open"]
                        }
            
            logger.debug("No valid confirmation candle found within 3 candles")
            return None
            
        except Exception as e:
            logger.error(f"Error in confirmation candle detection: {e}")
            return None
        
    def _find_pullback_entry(self, df: pd.DataFrame, stop_run: Dict, confirmation: Dict, symbol: str) -> Optional[Dict]:
        """Find a pullback entry after confirmation.
        
        Strategy uses a fixed pip range (5-15 pips) for pullback and looks for 
        reversal candle patterns (pin bars, engulfing) within 7 candles.
        """
        logger.debug(f"Looking for pullback entry after confirmation at index {confirmation['candle_index']}")
        
        # Get pip size for this instrument
        pip_size = self._get_pip_size_for_instrument(symbol)
        
        # Define pullback range in pips (5-15 pips)
        min_pullback_pips = self.config.get("pullback_pips", 5) * pip_size
        max_pullback_pips = min_pullback_pips * 3  # 15 pips
        
        logger.debug(f"Using pullback range: {min_pullback_pips/pip_size:.1f}-{max_pullback_pips/pip_size:.1f} pips")
        
        # Maximum 7 candles for pullback
        max_candles = 7
        max_lookback = min(max_candles, len(df) - confirmation["candle_index"] - 1)
        
        logger.debug(f"Checking {max_lookback} candles for pullback entry")
        
        # Get extreme price from stop run candle
        if stop_run["direction"] == "short":
            # For short, the extreme is the high of the stop run candle
            stop_run_extreme = stop_run["high"]
            logger.debug(f"SHORT stop run extreme: {stop_run_extreme:.5f}")
            
            # Look for pullback within pip range
            for i in range(1, max_lookback + 1):
                entry_idx = confirmation["candle_index"] + i
                if entry_idx >= len(df):
                    break
                    
                entry_candle = df.iloc[entry_idx]
                prev_candle = df.iloc[entry_idx-1] if entry_idx > 0 else None
                
                # Calculate pullback in pips
                pullback = entry_candle["high"] - stop_run_extreme
                
                # Check if pullback is within range
                if min_pullback_pips <= pullback <= max_pullback_pips:
                    logger.debug(f"Found pullback within range at index {entry_idx}: {pullback/pip_size:.1f} pips")
                    
                    # Score based on reversal candle quality
                    score = 0
                    is_reversal_candle = False
                    reversal_type = "standard"
                    
                    # Check for bearish reversal patterns
                    
                    # 1. Check for bearish engulfing
                    if (prev_candle is not None and
                        entry_candle["open"] > prev_candle["close"] and
                        entry_candle["close"] < prev_candle["open"] and
                        entry_candle["close"] < entry_candle["open"]):
                        score += 3
                        is_reversal_candle = True
                        reversal_type = "bearish_engulfing"
                        logger.debug(f"Detected bearish engulfing pattern at index {entry_idx}")
                    
                    # 2. Check for bearish pin bar (shooting star)
                    candle_range = entry_candle["high"] - entry_candle["low"]
                    if candle_range > 0:
                        body_size = abs(entry_candle["close"] - entry_candle["open"])
                        upper_wick = entry_candle["high"] - max(entry_candle["open"], entry_candle["close"])
                        lower_wick = min(entry_candle["open"], entry_candle["close"]) - entry_candle["low"]
                        
                        if (entry_candle["close"] < entry_candle["open"] and  # Bearish candle
                            body_size < candle_range * 0.4 and  # Small body
                            upper_wick > body_size * 2 and  # Long upper wick
                            lower_wick < body_size):  # Small or no lower wick
                            score += 3
                            is_reversal_candle = True
                            reversal_type = "shooting_star"
                            logger.debug(f"Detected shooting star pattern at index {entry_idx}")
                    
                    # 3. Standard reversal (just a bearish candle)
                    if entry_candle["close"] < entry_candle["open"]:
                        score += 1
                        is_reversal_candle = True
                        if reversal_type == "standard":  # Only set if not already set
                            reversal_type = "bearish_candle"
                        logger.debug(f"Detected bearish candle at index {entry_idx}")
                    
                    # Only accept entry if it's a reversal candle
                    if is_reversal_candle:
                        return {
                            "direction": "short",
                            "entry_price": entry_candle["close"],
                            "entry_type": reversal_type,
                            "candle_index": entry_idx,
                            "score": score
                        }
            
            # Alternative entry: If no pullback occurs but price continues in direction
            last_idx = min(confirmation["candle_index"] + max_lookback, len(df) - 1)
            last_candle = df.iloc[last_idx]
            
            # For SHORT: if price is continuing down strongly, take entry at current price
            if last_candle["close"] < confirmation["close"] and last_candle["close"] < last_candle["open"]:
                logger.debug(f"No pullback detected, but found continuation entry for SHORT at index {last_idx}")
                return {
                    "direction": "short",
                    "entry_price": last_candle["close"],
                    "entry_type": "continuation",
                    "candle_index": last_idx,
                    "score": 1
                }
                
        else:  # long setup
            # For long, the extreme is the low of the stop run candle
            stop_run_extreme = stop_run["low"]
            logger.debug(f"LONG stop run extreme: {stop_run_extreme:.5f}")
            
            # Look for pullback within pip range
            for i in range(1, max_lookback + 1):
                entry_idx = confirmation["candle_index"] + i
                if entry_idx >= len(df):
                    break
                    
                entry_candle = df.iloc[entry_idx]
                prev_candle = df.iloc[entry_idx-1] if entry_idx > 0 else None
                
                # Calculate pullback in pips
                pullback = stop_run_extreme - entry_candle["low"]
                
                # Check if pullback is within range
                if min_pullback_pips <= pullback <= max_pullback_pips:
                    logger.debug(f"Found pullback within range at index {entry_idx}: {pullback/pip_size:.1f} pips")
                    
                    # Score based on reversal candle quality
                    score = 0
                    is_reversal_candle = False
                    reversal_type = "standard"
                    
                    # Check for bullish reversal patterns
                    
                    # 1. Check for bullish engulfing
                    if (prev_candle is not None and
                        entry_candle["open"] < prev_candle["close"] and
                        entry_candle["close"] > prev_candle["open"] and
                        entry_candle["close"] > entry_candle["open"]):
                        score += 3
                        is_reversal_candle = True
                        reversal_type = "bullish_engulfing"
                        logger.debug(f"Detected bullish engulfing pattern at index {entry_idx}")
                    
                    # 2. Check for bullish pin bar (hammer)
                    candle_range = entry_candle["high"] - entry_candle["low"]
                    if candle_range > 0:
                        body_size = abs(entry_candle["close"] - entry_candle["open"])
                        upper_wick = entry_candle["high"] - max(entry_candle["open"], entry_candle["close"])
                        lower_wick = min(entry_candle["open"], entry_candle["close"]) - entry_candle["low"]
                        
                        if (entry_candle["close"] > entry_candle["open"] and  # Bullish candle
                            body_size < candle_range * 0.4 and  # Small body
                            lower_wick > body_size * 2 and  # Long lower wick
                            upper_wick < body_size):  # Small or no upper wick
                            score += 3
                            is_reversal_candle = True
                            reversal_type = "hammer"
                            logger.debug(f"Detected hammer pattern at index {entry_idx}")
                    
                    # 3. Standard reversal (just a bullish candle)
                    if entry_candle["close"] > entry_candle["open"]:
                        score += 1
                        is_reversal_candle = True
                        if reversal_type == "standard":  # Only set if not already set
                            reversal_type = "bullish_candle"
                        logger.debug(f"Detected bullish candle at index {entry_idx}")
                    
                    # Only accept entry if it's a reversal candle
                    if is_reversal_candle:
                        return {
                            "direction": "long",
                            "entry_price": entry_candle["close"],
                            "entry_type": reversal_type,
                            "candle_index": entry_idx,
                            "score": score
                        }
            
            # Alternative entry: If no pullback occurs but price continues in direction
            last_idx = min(confirmation["candle_index"] + max_lookback, len(df) - 1)
            last_candle = df.iloc[last_idx]
            
            # For LONG: if price is continuing up strongly, take entry at current price
            if last_candle["close"] > confirmation["close"] and last_candle["close"] > last_candle["open"]:
                logger.debug(f"No pullback detected, but found continuation entry for LONG at index {last_idx}")
                return {
                    "direction": "long",
                    "entry_price": last_candle["close"],
                    "entry_type": "continuation",
                    "candle_index": last_idx,
                    "score": 1
                }
        
        logger.debug(f"No valid pullback entry found within {max_lookback} candles")
        return None
        
    def _calculate_time_invalidation(self, timeframe: str) -> str:
        """Calculate time-based invalidation for a signal.
        
        Args:
            timeframe: Trading timeframe
            
        Returns:
            ISO format datetime string for invalidation
        """
        now = datetime.now()
        
        # Set invalidation based on timeframe
        if timeframe == "M5":
            invalidation = now + pd.Timedelta(hours=2)
        elif timeframe == "M15":
            invalidation = now + pd.Timedelta(hours=6)
        elif timeframe == "M30":
            invalidation = now + pd.Timedelta(hours=12)
        elif timeframe == "H1":
            invalidation = now + pd.Timedelta(days=1)
        elif timeframe == "H4":
            invalidation = now + pd.Timedelta(days=2)
        elif timeframe == "D1":
            invalidation = now + pd.Timedelta(days=5)
        else:
            invalidation = now + pd.Timedelta(days=1)  # Default to 1 day
            
        return invalidation.isoformat()
        
    def _get_pip_size_for_instrument(self, symbol: str) -> float:
        """Get the pip size for a given instrument.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Pip size as a decimal
        """
        # Default pip sizes for common forex pairs
        pip_sizes = {
            # Major pairs
            "EURUSD": 0.0001,
            "GBPUSD": 0.0001,
            "USDJPY": 0.01,
            "USDCHF": 0.0001,
            "AUDUSD": 0.0001,
            "NZDUSD": 0.0001,
            "USDCAD": 0.0001,
            
            # Cross pairs
            "EURGBP": 0.0001,
            "EURJPY": 0.01,
            "GBPJPY": 0.01,
            "CADJPY": 0.01,
            
            # Metals
            "XAUUSD": 0.01,  # Gold
            "XAGUSD": 0.001,  # Silver
        }
        
        # Remove suffix for MT5 symbols (e.g., EURUSDm -> EURUSD)
        base_symbol = symbol.replace("m", "").upper()
        
        # Try to get the pip size for the base symbol
        pip_size = pip_sizes.get(base_symbol)
        
        # If not found, determine based on symbol characteristics
        if pip_size is None:
            if "JPY" in base_symbol:
                pip_size = 0.01
            elif "XAU" in base_symbol or "GOLD" in base_symbol:
                pip_size = 0.01
            elif "XAG" in base_symbol or "SILVER" in base_symbol:
                pip_size = 0.001
            # Special handling for cryptocurrency pairs
            elif symbol.endswith('USDm') or symbol.endswith('USDT') or symbol.endswith('USD') and any(crypto in symbol for crypto in ['BTC', 'ETH', 'LTC', 'XRP', 'DOGE']):
                pip_size = 1.0  # For cryptocurrencies, use 1 unit as 1 pip
            else:
                pip_size = 0.0001  # Default for most forex pairs
                
        logger.debug(f"Using pip size {pip_size} for {symbol}")
        return pip_size

    def _detect_sharp_reversal(self, df: pd.DataFrame, window: int = None) -> List[Dict]:
        """Detect sharp reversals in price action within a given window.
        
        A sharp reversal is defined as:
        1. A significant price movement in one direction
        2. Followed by a sharp reversal in the opposite direction
        3. With significant volume (if volume data is available)
        
        Args:
            df: Dataframe with price data
            window: Window size to detect reversals (defaults to config value)
            
        Returns:
            List of sharp reversal points
        """
        # Skip if sharp reversal detection is disabled
        if not self.config["manipulation_points"].get("sharp_reversals", True):
            logger.debug("Sharp reversal detection is disabled in configuration")
            return []
        
        # Get configuration parameters
        window = window or self.config["reversal_detection"]["window_size"]
        body_size_threshold = self.config["reversal_detection"]["body_size_threshold"]
        close_position_threshold = self.config["reversal_detection"]["close_position_threshold"]
        
        sharp_reversals = []
        
        if len(df) < window * 2:
            logger.warning(f"Insufficient data for sharp reversal detection: {len(df)} candles")
            return sharp_reversals
            
        # Calculate price movements
        df['price_change'] = df['close'].diff()
        df['price_change_pct'] = df['close'].pct_change() * 100
        
        # Look for sharp reversals in the last 50 candles
        lookback = min(50, len(df) - window)
        
        for i in range(window, lookback):
            # Get current window
            current_window = df.iloc[i-window:i]
            next_candles = df.iloc[i:i+window]
            
            if len(current_window) < window or len(next_candles) < 1:
                continue
                
            # Calculate trend direction in current window
            price_changes = current_window['price_change'].sum()
            
            # Check for bullish reversal (downtrend followed by upward reversal)
            if price_changes < 0:  # Downtrend
                # Check if next candle has a strong bullish reversal
                next_candle = next_candles.iloc[0]
                
                # Conditions for strong bullish reversal:
                # 1. The candle is bullish (close > open)
                # 2. The candle's range is significant
                # 3. The close is in the upper half of the candle
                
                if next_candle['close'] > next_candle['open']:
                    candle_range = next_candle['high'] - next_candle['low']
                    if candle_range == 0:
                        continue
                        
                    body_size = abs(next_candle['close'] - next_candle['open'])
                    body_to_range_ratio = body_size / candle_range
                    top_close_ratio = (next_candle['high'] - next_candle['close']) / candle_range
                    
                    # Strong bullish reversal conditions using configured thresholds
                    if body_to_range_ratio > body_size_threshold and top_close_ratio < close_position_threshold:
                        # This candle has a strong bullish body and closed near its high
                        reversal_point = {
                            'type': 'bullish_reversal',
                            'price': next_candle['low'],
                            'candle_index': i,
                            'strength': body_to_range_ratio
                        }
                        sharp_reversals.append(reversal_point)
                        logger.debug(f"Detected bullish reversal at index {i}, price: {next_candle['low']:.5f}")
                        
            # Check for bearish reversal (uptrend followed by downward reversal)
            elif price_changes > 0:  # Uptrend
                # Check if next candle has a strong bearish reversal
                next_candle = next_candles.iloc[0]
                
                # Conditions for strong bearish reversal:
                # 1. The candle is bearish (close < open)
                # 2. The candle's range is significant
                # 3. The close is in the lower half of the candle
                
                if next_candle['close'] < next_candle['open']:
                    candle_range = next_candle['high'] - next_candle['low']
                    if candle_range == 0:
                        continue
                        
                    body_size = abs(next_candle['close'] - next_candle['open'])
                    body_to_range_ratio = body_size / candle_range
                    bottom_close_ratio = (next_candle['close'] - next_candle['low']) / candle_range
                    
                    # Strong bearish reversal conditions using configured thresholds
                    if body_to_range_ratio > body_size_threshold and bottom_close_ratio < close_position_threshold:
                        # This candle has a strong bearish body and closed near its low
                        reversal_point = {
                            'type': 'bearish_reversal',
                            'price': next_candle['high'],
                            'candle_index': i,
                            'strength': body_to_range_ratio
                        }
                        sharp_reversals.append(reversal_point)
                        logger.debug(f"Detected bearish reversal at index {i}, price: {next_candle['high']:.5f}")
                        
        return sharp_reversals

