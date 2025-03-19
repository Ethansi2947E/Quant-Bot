import asyncio
from typing import Dict, Optional, List
import pandas as pd
from datetime import datetime, time
import os
import json
import traceback
import sys
from loguru import logger

from src.mt5_handler import MT5Handler
from src.mtf_analysis import MTFAnalysis
from src.poi_detector import POIDetector
from src.risk_manager import RiskManager
from src.market_analysis import MarketAnalysis
from src.smc_analysis import SMCAnalysis
from src.volume_analysis import VolumeAnalysis

# Create logs directory if it doesn't exist
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Configure loguru logger
logger_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>SG1:{function}:{line}</cyan> | <level>{message}</level>"

# Configure loguru with custom format
logger.configure(handlers=[
    {"sink": sys.stdout, "format": logger_format, "level": "INFO", "colorize": True},
    {"sink": os.path.join(log_dir, "signal_generator1_detailed.log"), 
     "format": logger_format, "level": "DEBUG", "rotation": "10 MB", 
     "retention": 5, "compression": "zip"}
])

# Add context to differentiate this logger
logger = logger.bind(name="signal_generator1")

logger.info("[SG1] SignalGenerator1 logger initialized")
logger.info(f"[SG1] Detailed logs will be written to {os.path.join(log_dir, 'signal_generator1_detailed.log')}")

class SignalGenerator1:
    def __init__(self, mt5_handler: Optional[MT5Handler] = None, risk_manager: Optional[RiskManager] = None):
        """Initialize SignalGenerator1 with analysis components and strategy parameters."""
        logger.info("[INIT] Initializing SignalGenerator1 with strategy components")
        
        self.mt5_handler = mt5_handler
        self.market_analysis = MarketAnalysis()
        self.smc_analysis = SMCAnalysis()
        self.mtf_analysis = MTFAnalysis()
        self.poi_detector = POIDetector()
        self.risk_manager = risk_manager if risk_manager else RiskManager()
        
        # Strategy parameters - Relaxed conditions
        self.min_sweep_pips = 3  # Reduced from 5
        self.max_sweep_pips = 30  # Increased from 20
        self.min_stop_pips = 5   # Reduced from 10
        self.fib_extension = 1.5  # Reduced from 1.618
        self.min_rr_ratio = 1.5   # Reduced from 2.0
        self.price_precision = 5
        
        logger.info(f"[INIT] Strategy parameters configured: min_sweep={self.min_sweep_pips}, "
                   f"max_sweep={self.max_sweep_pips}, min_stop={self.min_stop_pips}, "
                   f"fib_ext={self.fib_extension}, min_rr={self.min_rr_ratio}")

    def _detect_turtle_soup(self, df: pd.DataFrame, bsl: float, ssl: float) -> Optional[Dict]:
        """Detect Turtle Soup patterns: false breakouts with quick reversals."""
        try:
            logger.info("[SCAN] Starting Turtle Soup pattern detection")
            logger.debug(f"[SCAN] BSL: {bsl:.5f}, SSL: {ssl:.5f}")
            
            # Look at last 3 candles instead of just last 2
            last_candles = df.iloc[-3:]
            last_candle = last_candles.iloc[-1]
            
            logger.debug(f"[DATA] Last candle - O:{last_candle['open']:.5f}, H:{last_candle['high']:.5f}, "
                        f"L:{last_candle['low']:.5f}, C:{last_candle['close']:.5f}")
            
            # Short entry: false breakout above BSL with more lenient conditions
            if (last_candles['high'].max() > bsl and last_candle['close'] < bsl):
                sweep_pips = (last_candles['high'].max() - bsl) * 10000
                logger.info(f"[DETECT] Potential SHORT setup - Sweep distance: {sweep_pips:.1f} pips")
                
                if self.min_sweep_pips <= sweep_pips <= self.max_sweep_pips:
                    logger.info(f"[SUCCESS] Valid SHORT Turtle Soup pattern detected at level {bsl:.5f}")
                    return {'type': 'short', 'level': bsl}
                logger.info("[FAIL] Sweep distance outside allowed range")
                
            # Long entry: false breakout below SSL with more lenient conditions
            elif (last_candles['low'].min() < ssl and last_candle['close'] > ssl):
                sweep_pips = (ssl - last_candles['low'].min()) * 10000
                logger.info(f"[DETECT] Potential LONG setup - Sweep distance: {sweep_pips:.1f} pips")
                
                if self.min_sweep_pips <= sweep_pips <= self.max_sweep_pips:
                    logger.info(f"[SUCCESS] Valid LONG Turtle Soup pattern detected at level {ssl:.5f}")
                    return {'type': 'long', 'level': ssl}
                logger.info("[FAIL] Sweep distance outside allowed range")
            
            logger.info("[FAIL] No valid Turtle Soup pattern detected")
            return None
            
        except Exception as e:
            logger.error(f"[ERROR] Error detecting Turtle Soup: {str(e)}")
            logger.error(f"[ERROR] {traceback.format_exc()}")
            return None

    def _detect_sh_bms_rto(self, df: pd.DataFrame, bsl: float, ssl: float) -> Optional[Dict]:
        """Detect SH + BMS + RTO sequence."""
        try:
            logger.info("[SCAN] Starting SH+BMS+RTO pattern detection")
            
            # Step 1: Detect Stop Hunt (SH)
            logger.info("[STEP1] Detecting Stop Hunt")
            sh_pattern = self._detect_turtle_soup(df, bsl, ssl)
            if not sh_pattern:
                logger.info("[FAIL] No Stop Hunt pattern detected")
                return None
                
            logger.info(f"[SUCCESS] Stop Hunt detected: {sh_pattern['type'].upper()}")
            
            # Step 2: Confirm BMS
            logger.info("[STEP2] Confirming Break of Market Structure")
            if sh_pattern['type'] == 'short' and df['close'].iloc[-1] < sh_pattern['level']:
                direction = 'short'
                logger.info(f"[SUCCESS] Bearish BMS confirmed below {sh_pattern['level']:.5f}")
            elif sh_pattern['type'] == 'long' and df['close'].iloc[-1] > sh_pattern['level']:
                direction = 'long'
                logger.info(f"[SUCCESS] Bullish BMS confirmed above {sh_pattern['level']:.5f}")
            else:
                logger.info("[FAIL] No Break of Market Structure confirmed")
                return None
            
            # Step 3: Find order block for RTO
            logger.info("[STEP3] Searching for Order Block")
            ob = self._find_order_block(df, direction)
            if not ob:
                logger.info("[FAIL] No suitable Order Block found")
                return None
                
            logger.info(f"[SUCCESS] Order Block found: Low={ob['low']:.5f}, High={ob['high']:.5f}")
            
            # Check for retracement to OB
            logger.info("[STEP4] Checking retracement to Order Block")
            if direction == 'long' and df['low'].iloc[-1] <= ob['high'] and df['close'].iloc[-1] >= ob['low']:
                logger.info("[SUCCESS] Price has retraced to bullish Order Block")
                return {'type': 'long', 'ob': ob}
            elif direction == 'short' and df['high'].iloc[-1] >= ob['low'] and df['close'].iloc[-1] <= ob['high']:
                logger.info("[SUCCESS] Price has retraced to bearish Order Block")
                return {'type': 'short', 'ob': ob}
                
            logger.info("[FAIL] No valid retracement to Order Block")
            return None
            
        except Exception as e:
            logger.error(f"[ERROR] Error detecting SH+BMS+RTO: {str(e)}")
            logger.error(f"[ERROR] {traceback.format_exc()}")
            return None

    def _find_order_block(self, df: pd.DataFrame, direction: str) -> Optional[Dict]:
        """Find the most recent order block for the given direction."""
        try:
            if direction == 'long':
                # Last bearish candle before bullish move
                bearish_candles = df[df['close'] < df['open']]
                if not bearish_candles.empty:
                    return {'low': bearish_candles.iloc[-1]['low'], 'high': bearish_candles.iloc[-1]['high']}
            elif direction == 'short':
                # Last bullish candle before bearish move
                bullish_candles = df[df['close'] > df['open']]
                if not bullish_candles.empty:
                    return {'low': bullish_candles.iloc[-1]['low'], 'high': bullish_candles.iloc[-1]['high']}
            return None
        except Exception as e:
            logger.error(f"Error finding order block: {str(e)}")
            return None

    def _detect_amd(self, df: pd.DataFrame) -> Optional[Dict]:
        """Detect AMD patterns based on price action with dynamic lookback and enhanced reversal detection."""
        try:
            # Calculate ATR for volatility-based stops and dynamic window sizing
            atr_series = self.market_analysis.calculate_atr(df, period=14)
            atr = atr_series.iloc[-1]
            
            # Determine relative volatility (ATR as a fraction of the average close)
            relative_volatility = atr / df['close'].mean()
            # Base window is 20 candles; adjust inversely with volatility:
            # Higher volatility -> shorter window; lower volatility -> longer window.
            dynamic_window = int(20 * (0.5 / relative_volatility)) if relative_volatility > 0 else 20
            # Limit the dynamic window between 15 and 40 candles
            dynamic_window = max(15, min(dynamic_window, 40))
            
            recent_data = df.iloc[-dynamic_window:]
            recent_high = recent_data['high'].max()
            recent_low = recent_data['low'].min()
            
            # Retrieve the last three candles
            last_candle = df.iloc[-1]
            prev_candle = df.iloc[-2]
            prev2_candle = df.iloc[-3] if len(df) >= 3 else None
            
            # Only proceed if we have at least three candles and the generic reversal condition is met
            if prev2_candle is not None and self._detect_reversal(df):
                # Calculate the average volume of the two preceding candles for a more robust volume confirmation
                average_prev_volume = (prev_candle['volume'] + prev2_candle['volume']) / 2

                # Bearish reversal (distribution) conditions:
                # - Last candle is bearish
                # - The two previous candles were bullish
                # - Last candle volume exceeds the average of the prior two candles by at least 20%
                if (last_candle['close'] < last_candle['open'] and
                    prev_candle['close'] > prev_candle['open'] and
                    prev2_candle['close'] > prev2_candle['open'] and
                    last_candle['volume'] > average_prev_volume * 1.2):
                    
                    stop_loss = max(recent_high, last_candle['high'] + 2 * atr)
                    risk = stop_loss - last_candle['close']
                    target = last_candle['close'] - (2 * risk)
                    
                    if target > recent_low:
                        return {
                            'type': 'distribution',
                            'direction': 'short',
                            'entry': last_candle['close'],
                            'stop_loss': stop_loss,
                            'target': target,
                            'volume_ratio': last_candle['volume'] / average_prev_volume
                        }
                
                # Bullish reversal (accumulation) conditions:
                # - Last candle is bullish
                # - The two previous candles were bearish
                # - Last candle volume exceeds the average of the prior two candles by at least 20%
                elif (last_candle['close'] > last_candle['open'] and
                    prev_candle['close'] < prev_candle['open'] and
                    prev2_candle['close'] < prev2_candle['open'] and
                    last_candle['volume'] > average_prev_volume * 1.2):
                    
                    stop_loss = min(recent_low, last_candle['low'] - 2 * atr)
                    risk = last_candle['close'] - stop_loss
                    target = last_candle['close'] + (2 * risk)
                    
                    if target < recent_high:
                        return {
                            'type': 'accumulation',
                            'direction': 'long',
                            'entry': last_candle['close'],
                            'stop_loss': stop_loss,
                            'target': target,
                            'volume_ratio': last_candle['volume'] / average_prev_volume
                        }
            
            return None

        except Exception as e:
            logger.error(f"Error detecting AMD: {str(e)}")
            return None

    def _detect_reversal(self, df: pd.DataFrame) -> bool:
        """
        Detect reversal patterns in price action based on candle patterns and indicators.
        
        Args:
            df: DataFrame with OHLCV data and indicators
            
        Returns:
            bool: True if a reversal pattern is detected, False otherwise
        """
        try:
            # Need at least 5 candles for a reliable reversal pattern
            if len(df) < 5:
                return False
            
            # Get the most recent candles
            last_candle = df.iloc[-1]
            prev_candle = df.iloc[-2]
            prev2_candle = df.iloc[-3]
            
            # Calculate ATR for volatility context
            atr_series = self.market_analysis.calculate_atr(df, period=14)
            atr = atr_series.iloc[-1]
            
            # Calculate the current momentum
            momentum = 0
            if 'rsi' in df.columns:
                # Use RSI for momentum
                momentum = df['rsi'].iloc[-1] - df['rsi'].iloc[-2]
            else:
                # Simple momentum calculation if RSI is not available
                momentum = ((df['close'].iloc[-1] / df['close'].iloc[-5]) - 1) * 100
            
            # Check for bullish reversal
            bullish_reversal = (
                # Bullish engulfing or strong bullish candle
                (last_candle['close'] > last_candle['open'] and 
                 prev_candle['close'] < prev_candle['open'] and
                 last_candle['close'] > prev_candle['open'] and
                 last_candle['open'] < prev_candle['close']) or
                
                # Strong momentum reversal upward
                (momentum > 5 and 
                 last_candle['close'] > last_candle['open'] and
                 last_candle['close'] - last_candle['open'] > 0.5 * atr) or
                
                # Price rejection from support (long lower wick)
                (last_candle['low'] < prev_candle['low'] and
                 last_candle['close'] > last_candle['open'] and
                 last_candle['close'] - last_candle['low'] > 2 * (last_candle['high'] - last_candle['close']))
            )
            
            # Check for bearish reversal
            bearish_reversal = (
                # Bearish engulfing or strong bearish candle
                (last_candle['close'] < last_candle['open'] and 
                 prev_candle['close'] > prev_candle['open'] and
                 last_candle['close'] < prev_candle['open'] and
                 last_candle['open'] > prev_candle['close']) or
                
                # Strong momentum reversal downward
                (momentum < -5 and 
                 last_candle['close'] < last_candle['open'] and
                 last_candle['open'] - last_candle['close'] > 0.5 * atr) or
                
                # Price rejection from resistance (long upper wick)
                (last_candle['high'] > prev_candle['high'] and
                 last_candle['close'] < last_candle['open'] and
                 last_candle['high'] - last_candle['close'] > 2 * (last_candle['close'] - last_candle['low']))
            )
            
            # Check for volume confirmation if available
            volume_confirmation = True
            if 'volume' in df.columns:
                avg_volume = df['volume'].iloc[-5:-1].mean()
                volume_confirmation = last_candle['volume'] > avg_volume * 1.1
            
            # Return true if either type of reversal is detected with volume confirmation
            return (bullish_reversal or bearish_reversal) and volume_confirmation
            
        except Exception as e:
            logger.error(f"Error in reversal detection: {str(e)}")
            return False

    async def generate_signals(self, market_data: Dict, symbol: str, timeframe: str,
                               account_info: Optional[Dict] = None) -> List[Dict]:
        """Generate trading signals based on the combined strategy with priority logic."""
        signals = []
        try:
            logger.info(f"[{symbol}] [START] Starting signal generation on {timeframe} timeframe")
            
            df = market_data.get(timeframe)
            if df is None or df.empty:
                logger.warning(f"[{symbol}] [FAIL] No data available for {timeframe} timeframe")
                return []
            
            logger.info(f"[{symbol}] [DATA] Analyzing {len(df)} candles")
            logger.info(f"[{symbol}] [DATA] Current price: {df['close'].iloc[-1]:.5f}")

            # Get liquidity zones
            logger.info(f"[{symbol}] [STEP1] Detecting liquidity zones")
            poi_zones = self.poi_detector.detect_supply_demand_zones(df, timeframe)
            bsl = max([zone.price_end for zone in poi_zones.get('supply', [])], default=df['high'].max())
            ssl = min([zone.price_start for zone in poi_zones.get('demand', [])], default=df['low'].min())
            logger.info(f"[{symbol}] [INFO] BSL: {bsl:.5f}, SSL: {ssl:.5f}")

            # Detect all setups
            logger.info(f"[{symbol}] [STEP2] Detecting pattern setups")
            turtle_soup = self._detect_turtle_soup(df, bsl, ssl)
            sh_bms_rto = self._detect_sh_bms_rto(df, bsl, ssl)
            amd = self._detect_amd(df)

            # HTF confirmation
            logger.info(f"[{symbol}] [STEP3] Checking higher timeframe trend")
            htf_df = market_data.get('D1', df)
            htf_trend = 'bullish' if htf_df['close'].iloc[-1] > htf_df['close'].rolling(200).mean().iloc[-1] else 'bearish'
            logger.info(f"[{symbol}] [INFO] HTF Trend: {htf_trend}")

            # Priority logic and confluence
            logger.info(f"[{symbol}] [STEP4] Applying priority logic")
            signal = None
            
            if sh_bms_rto and ((sh_bms_rto['type'] == 'long' and htf_trend == 'bullish') or
                               (sh_bms_rto['type'] == 'short' and htf_trend == 'bearish')):
                logger.info(f"[{symbol}] [PRIORITY1] Creating signal from SH+BMS+RTO pattern")
                signal = self._create_signal_from_sh_bms_rto(sh_bms_rto, df, symbol, timeframe, account_info)
            elif turtle_soup and \
                 ((turtle_soup['type'] == 'long' and htf_trend == 'bullish') or
                  (turtle_soup['type'] == 'short' and htf_trend == 'bearish')):
                logger.info(f"[{symbol}] [PRIORITY2] Creating signal from Turtle Soup pattern")
                signal = self._create_signal_from_turtle_soup(turtle_soup, df, symbol, timeframe, account_info)
            elif amd:
                logger.info(f"[{symbol}] [PRIORITY3] Creating signal from AMD pattern")
                signal = self._create_signal_from_amd(amd, df, symbol, timeframe, account_info)

            if signal:
                signals.append(signal)
                logger.info(f"[{symbol}] [SUCCESS] Generated {signal['direction']} signal at {signal['entry_price']:.5f}")
                logger.info(f"[{symbol}] [INFO] Stop: {signal['stop_loss']:.5f}, Target: {signal['take_profit']:.5f}")
                logger.info(f"[{symbol}] [INFO] Position Size: {signal['position_size']}, Strategy: {signal['strategy']}")
            else:
                logger.info(f"[{symbol}] [INFO] No valid signals generated")

            return signals
            
        except Exception as e:
            logger.error(f"[{symbol}] [ERROR] Error generating signals: {str(e)}")
            logger.error(f"[{symbol}] [ERROR] {traceback.format_exc()}")
            return []

    def _create_signal_from_sh_bms_rto(self, pattern: Dict, df: pd.DataFrame, symbol: str, timeframe: str, account_info: Dict) -> Dict:
        """Create a signal from SH + BMS + RTO pattern."""
        try:
            logger.info(f"[{symbol}] Creating signal from SH+BMS+RTO pattern")
            
            direction = 'BUY' if pattern['type'] == 'long' else 'SELL'
            entry_price = df['close'].iloc[-1]
            stop_loss = self._calculate_stop_loss(direction, entry_price, pattern['ob'])
            take_profit = self._calculate_take_profit(entry_price, stop_loss, direction)
            
            logger.info(f"[{symbol}] [CALC] Direction: {direction}")
            logger.info(f"[{symbol}] [CALC] Entry: {entry_price:.5f}")
            logger.info(f"[{symbol}] [CALC] Stop Loss: {stop_loss:.5f}")
            logger.info(f"[{symbol}] [CALC] Take Profit: {take_profit:.5f}")
            
            position_size = self.risk_manager.calculate_position_size(
                account_balance=account_info.get('balance', 10000),
                risk_per_trade=0.01,
                entry_price=entry_price,
                stop_loss_price=stop_loss,
                symbol=symbol
            )
            logger.info(f"[{symbol}] [CALC] Position Size: {position_size}")
            
            signal = {
                'symbol': symbol,
                'timeframe': timeframe,
                'direction': direction,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'position_size': position_size,
                'strategy': 'SH_BMS_RTO',
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"[{symbol}] [SUCCESS] SH+BMS+RTO signal created")
            return signal
            
        except Exception as e:
            logger.error(f"[{symbol}] [ERROR] Error creating SH+BMS+RTO signal: {str(e)}")
            logger.error(f"[{symbol}] [ERROR] {traceback.format_exc()}")
            raise

    def _create_signal_from_turtle_soup(self, pattern: Dict, df: pd.DataFrame, symbol: str, timeframe: str, account_info: Dict) -> Dict:
        """Create a signal from Turtle Soup pattern."""
        direction = 'BUY' if pattern['type'] == 'long' else 'SELL'
        entry_price = df['close'].iloc[-1]
        stop_loss = pattern['level'] - 0.001 if direction == 'BUY' else pattern['level'] + 0.001
        take_profit = self._calculate_take_profit(entry_price, stop_loss, direction)
        position_size = self.risk_manager.calculate_position_size(
            account_balance=account_info.get('balance', 10000),
            risk_per_trade=0.01,
            entry_price=entry_price,
            stop_loss_price=stop_loss,
            symbol=symbol
        )
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'direction': direction,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': position_size,
            'strategy': 'Turtle_Soup',
            'timestamp': datetime.now().isoformat()
        }

    def _create_signal_from_amd(self, pattern: Dict, df: pd.DataFrame, symbol: str, timeframe: str, account_info: Dict) -> Dict:
        """Create a signal from AMD distribution pattern with improved risk management."""
        direction = 'BUY' if pattern['direction'] == 'long' else 'SELL'
        entry_price = pattern['entry']
        stop_loss = pattern['stop_loss']
        take_profit = pattern['target']
        
        # Calculate position size with proper risk management
        position_size = self.risk_manager.calculate_position_size(
            account_balance=account_info.get('balance', 10000),
            risk_per_trade=0.01,
            entry_price=entry_price,
            stop_loss_price=stop_loss,
            symbol=symbol
        )
        
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'direction': direction,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': position_size,
            'strategy': 'AMD',
            'timestamp': datetime.now().isoformat(),
            'pattern_type': pattern['type'],
            'volume_ratio': pattern.get('volume_ratio', 1.0)
        }

    def _calculate_stop_loss(self, direction: str, entry_price: float, ob: Optional[Dict] = None) -> float:
        """Calculate stop loss based on order block or default distance."""
        try:
            logger.info(f"[CALC] Calculating stop loss - Direction: {direction}, Entry: {entry_price:.5f}")
            
            if ob:
                stop_loss = ob['low'] - 0.001 if direction == 'BUY' else ob['high'] + 0.001
                logger.info(f"[CALC] Using Order Block for stop loss: {stop_loss:.5f}")
            else:
                stop_loss = entry_price - (self.min_stop_pips / 10000) if direction == 'BUY' else entry_price + (self.min_stop_pips / 10000)
                logger.info(f"[CALC] Using default stop distance: {stop_loss:.5f}")
            
            return stop_loss
            
        except Exception as e:
            logger.error(f"[ERROR] Error calculating stop loss: {str(e)}")
            logger.error(f"[ERROR] {traceback.format_exc()}")
            raise

    def _calculate_take_profit(self, entry_price: float, stop_loss: float, direction: str) -> float:
        """Calculate take profit based on risk-reward ratio."""
        try:
            logger.info(f"[CALC] Calculating take profit - Direction: {direction}, Entry: {entry_price:.5f}, Stop: {stop_loss:.5f}")
            
            risk = abs(entry_price - stop_loss)
            take_profit = entry_price + (self.min_rr_ratio * risk) if direction == 'BUY' else entry_price - (self.min_rr_ratio * risk)
            
            logger.info(f"[CALC] Risk distance: {risk:.5f}")
            logger.info(f"[CALC] Take profit calculated: {take_profit:.5f}")
            logger.info(f"[CALC] R:R ratio: 1:{self.min_rr_ratio:.1f}")
            
            return take_profit
            
        except Exception as e:
            logger.error(f"[ERROR] Error calculating take profit: {str(e)}")
            logger.error(f"[ERROR] {traceback.format_exc()}")
            raise