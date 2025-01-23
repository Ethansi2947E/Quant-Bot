import asyncio
from typing import Dict, List, Optional
from loguru import logger
import pandas as pd
from datetime import datetime, timedelta
import json
import sys
import MetaTrader5 as mt5
import numpy as np
import threading

from config.config import TRADING_CONFIG, SESSION_CONFIG, MT5_CONFIG, MARKET_STRUCTURE_CONFIG
from src.mt5_handler import MT5Handler
from src.signal_generator import SignalGenerator
from src.risk_manager import RiskManager
from src.ai_analyzer import AIAnalyzer
from src.telegram_bot import TelegramBot
from src.models import Trade, Signal, MarketData, NewsEvent
from src.dashboard import init_app, app, add_signal
from src.market_analysis import MarketAnalysis
from src.smc_analysis import SMCAnalysis
from src.mtf_analysis import MTFAnalysis
from src.divergence_analysis import DivergenceAnalysis
from src.volume_analysis import VolumeAnalysis

class TradingBot:
    def __init__(self, config=None):
        """Initialize the trading bot."""
        self.config = config
        self.mt5 = None
        self.signal_generator = SignalGenerator()
        self.risk_manager = RiskManager()
        self.ai_analyzer = AIAnalyzer()
        self.telegram_bot = TelegramBot()
        self.dashboard = None
        self.session_config = SESSION_CONFIG  # Store session config as instance variable
        self.trading_config = TRADING_CONFIG if config is None else config.TRADING_CONFIG
        self.running = False
        self.trades: List[Trade] = []
        self.signals: List[Signal] = []
        self.market_data: Dict[str, MarketData] = {}
        self.news_events: List[NewsEvent] = []
        self.trade_counter = 0
        
        # Initialize analysis components
        self.market_analysis = MarketAnalysis(ob_threshold=MARKET_STRUCTURE_CONFIG['structure_levels']['ob_size'])
        self.smc_analysis = SMCAnalysis()
        self.mtf_analysis = MTFAnalysis()
        self.divergence_analysis = DivergenceAnalysis()
        self.volume_analysis = VolumeAnalysis()
    
    def setup_logging(self):
        logger.add(
            "logs/trading_bot.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            level="DEBUG",
            rotation="1 day",
            retention="1 month",
            compression="zip"
        )

    def initialize_mt5(self):
        """Initialize connection to MetaTrader 5."""
        try:
            # Initialize MT5
            if not mt5.initialize():
                logger.error("MT5 initialization failed")
                return False
            
            # Get MT5 config from either passed config or imported config
            mt5_config = self.config.MT5_CONFIG if self.config else MT5_CONFIG
            
            # Login to MT5
            if not mt5.login(
                login=mt5_config["login"],
                password=mt5_config["password"],
                server=mt5_config["server"]
            ):
                logger.error("MT5 login failed")
                return False
            
            # Initialize MT5Handler
            self.mt5 = MT5Handler()
            
            logger.info("MT5 initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing MT5: {str(e)}")
            return False

    def initialize_dashboard(self):
        """Initialize and start the dashboard server."""
        try:
            # Initialize the Flask app with bot reference
            self.dashboard = init_app(self)
            
            # Start the Flask server in a background thread
            def run_server():
                self.dashboard.run(host='localhost', port=5000, debug=False, use_reloader=False)
            
            dashboard_thread = threading.Thread(target=run_server, daemon=True)
            dashboard_thread.start()
            logger.info("Dashboard server started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start dashboard server: {str(e)}")
            raise Exception("Failed to start dashboard server")

    async def start(self):
        """Start the trading bot."""
        try:
            logger.info("Starting trading bot...")
            
            # Initialize MT5 first
            if not self.initialize_mt5():
                raise Exception("Failed to initialize MT5")
            logger.info("MT5 initialized successfully")
            
            # Initialize dashboard
            try:
                self.initialize_dashboard()
                logger.info("Dashboard initialized successfully")
            except Exception as e:
                logger.error(f"Dashboard initialization failed: {str(e)}")
                # Continue even if dashboard fails
            
            # Initialize Telegram bot with retry
            telegram_init_attempts = 3
            telegram_init_success = False
            
            for attempt in range(telegram_init_attempts):
                try:
                    logger.info(f"Attempting to initialize Telegram bot (attempt {attempt + 1}/{telegram_init_attempts})")
                    if await self.telegram_bot.initialize(self.trading_config):
                        telegram_init_success = True
                        logger.info("Telegram bot initialized successfully")
                        break
                    else:
                        if attempt < telegram_init_attempts - 1:
                            logger.warning(f"Telegram initialization attempt {attempt + 1} failed, retrying in 2 seconds...")
                            await asyncio.sleep(2)
                except Exception as e:
                    logger.error(f"Error during Telegram initialization attempt {attempt + 1}: {str(e)}")
                    if attempt < telegram_init_attempts - 1:
                        logger.warning("Retrying in 2 seconds...")
                        await asyncio.sleep(2)
            
            if not telegram_init_success:
                raise Exception("Failed to initialize Telegram bot after multiple attempts")
            
            self.running = True
            logger.info("Trading bot started successfully")
            
            # Start main trading loop
            while self.running:
                try:
                    if not self.telegram_bot.trading_enabled:
                        logger.debug("Trading is disabled. Waiting for /enable command...")
                        await asyncio.sleep(5)  # Check more frequently for enable command
                        continue
                    
                    logger.info("Starting market analysis cycle...")
                    
                    # Get current session
                    current_session = self.analyze_session()
                    logger.info(f"Current trading session: {current_session}")
                    
                    for symbol in self.trading_config["symbols"]:
                        if not self.running:  # Check if we should stop
                            break
                            
                        # Skip if symbol not allowed in current session
                        if not self._is_symbol_allowed_in_session(symbol, current_session):
                            logger.debug(f"Skipping {symbol} - not allowed in {current_session} session")
                            continue
                            
                        for timeframe in self.trading_config["timeframes"]:
                            if not self.running:  # Check if we should stop
                                break
                                
                            if not self.telegram_bot.trading_enabled:
                                logger.info("Trading disabled during analysis, stopping cycle")
                                break
                                
                            logger.info(f"Analyzing {symbol} on {timeframe}")
                            analysis = await self.analyze_market(symbol, timeframe)
                            
                            if analysis:
                                signals = await self.generate_signals(analysis, symbol, timeframe)
                                if signals:
                                    await self.process_signals(signals)
                            
                            await asyncio.sleep(1)  # Prevent overloading
                    
                    if self.running:  # Only log if we're still running
                        logger.info("Completed market analysis cycle")
                        await self.manage_open_trades()  # Check and manage existing trades
                        await asyncio.sleep(60)  # Wait before next cycle
                    
                except Exception as e:
                    logger.error(f"Error in main loop: {str(e)}")
                    if self.telegram_bot and self.telegram_bot.bot:
                        await self.telegram_bot.send_error_alert(f"Error in main loop: {str(e)}")
                    await asyncio.sleep(60)  # Wait longer on error
                    
        except Exception as e:
            logger.error(f"Bot error: {str(e)}")
            if self.telegram_bot and self.telegram_bot.bot:
                await self.telegram_bot.send_error_alert(f"Bot error: {str(e)}")
            self.running = False
        finally:
            # Ensure clean shutdown
            self.running = False
            if self.mt5:
                mt5.shutdown()
            if self.telegram_bot:
                await self.telegram_bot.stop()
            logger.info("Trading bot stopped")
    
    async def stop(self):
        """Stop the trading bot."""
        try:
            self.running = False
            logger.info("Stopping trading bot...")
            
            # Close all open trades
            positions = self.mt5.get_open_positions()
            for position in positions:
                if self.mt5.close_position(position["ticket"]):
                    await self.telegram_bot.send_trade_update(
                        position["ticket"],
                        position["symbol"],
                        "CLOSED (Bot Stop)",
                        position["price_current"],
                        position["profit"]
                    )
            
            # Close MT5 connection
            del self.mt5
            
            # Stop Telegram bot
            if self.telegram_bot.application:
                await self.telegram_bot.application.stop()
            
            logger.info("Trading bot stopped")
            
        except Exception as e:
            logger.error(f"Error stopping trading bot: {str(e)}")
    
    async def check_symbol(self, symbol: str, timeframe: str):
        """Check a symbol for trading opportunities."""
        try:
            # Get market data
            df = self.mt5.get_market_data(symbol, timeframe)
            if df is None or len(df) < self.signal_generator.max_period:
                logger.warning(f"Insufficient data for {symbol} {timeframe}")
                return
            
            # Generate signal based on technical indicators only
            signal = self.signal_generator.generate_signal(df)
            
            # Send setup alert if potential setup is forming
            if signal["confidence"] >= 0.3 and signal["signal_type"] != "HOLD":
                await self.telegram_bot.send_setup_alert(
                    symbol,
                    timeframe,
                    signal["signal_type"],
                    signal["confidence"] * 100
                )
            
            if signal["signal_type"] != "HOLD" and signal["confidence"] >= 0.5:
                # Get current price
                current_price = df['close'].iloc[-1]
                
                # Calculate stop loss and take profit
                stop_loss = self.risk_manager.calculate_stop_loss(
                    df,
                    signal["signal_type"],
                    current_price
                )
                
                take_profit = self.risk_manager.calculate_take_profit(
                    current_price,
                    stop_loss
                )
                
                # Validate trade parameters
                valid, reason = self.risk_manager.validate_trade(
                    signal["signal_type"],
                    current_price,
                    stop_loss,
                    take_profit,
                    signal["confidence"]
                )
                
                if valid:
                    # Calculate position size
                    account_info = self.mt5.get_account_info()
                    if not account_info:
                        return
                    
                    position_size = self.risk_manager.calculate_position_size(
                        account_info["balance"],
                        current_price,
                        stop_loss,
                        symbol
                    )
                    
                    # Check daily risk limit
                    if not self.risk_manager.check_daily_risk(account_info["balance"]):
                        await self.telegram_bot.send_error_alert(
                            f"Daily risk limit reached for {symbol}"
                        )
                        return
                    
                    # Place trade
                    self.trade_counter += 1
                    trade_result = self.mt5.place_market_order(
                        symbol,
                        signal["signal_type"],
                        position_size,
                        stop_loss,
                        take_profit,
                        f"Signal: {signal['reason']}"
                    )
                    
                    if trade_result:
                        # Send trade alert
                        await self.telegram_bot.send_trade_alert(
                            symbol,
                            signal["signal_type"],
                            current_price,
                            stop_loss,
                            take_profit,
                            signal["confidence"],
                            signal["reason"]
                        )
                        
                        # Update trade history
                        trade_info = {
                            'id': self.trade_counter,
                            'symbol': symbol,
                            'type': signal["signal_type"],
                            'entry': current_price,
                            'exit': None,
                            'pnl': 0
                        }
                        self.telegram_bot.trade_history.append(trade_info)
                    else:
                        await self.telegram_bot.send_error_alert(
                            f"Failed to place trade for {symbol}"
                        )
            
        except Exception as e:
            logger.error(f"Error checking symbol {symbol}: {str(e)}")
            await self.telegram_bot.send_error_alert(
                f"Error checking {symbol}: {str(e)}"
            )
    
    async def manage_open_trades(self):
        """Manage existing trades."""
        try:
            # Get open positions
            positions = self.mt5.get_open_positions()
            self.risk_manager.update_open_trades(positions)
            
            for position in positions:
                symbol = position["symbol"]
                
                # Get current market data
                df = self.mt5.get_market_data(symbol, "M5")  # Use M5 for monitoring
                if df is None:
                    continue
                
                current_price = df['close'].iloc[-1]
                
                # Calculate indicators
                df = self.signal_generator.calculate_indicators(df)
                
                # Check if trade should be closed
                should_close, reason = self.risk_manager.should_close_trade(
                    position,
                    current_price,
                    {
                        "structure": df.get("structure", "NEUTRAL"),
                        "strength": df.get("strength", 0.0)
                    }
                )
                
                # Check if stop loss should be adjusted
                new_sl = self.risk_manager.calculate_trailing_stop(position, current_price)
                if new_sl != position["sl"]:
                    if self.mt5.modify_position(position["ticket"], new_sl, position["tp"]):
                        await self.telegram_bot.send_management_alert(
                            position["ticket"],
                            symbol,
                            "Stop Loss Adjusted",
                            position["sl"],
                            new_sl,
                            "Trailing Stop Update"
                        )
                
                if should_close:
                    # Close position
                    if self.mt5.close_position(position["ticket"]):
                        # Update trade history
                        for trade in self.telegram_bot.trade_history:
                            if trade['id'] == position["ticket"]:
                                trade['exit'] = current_price
                                trade['pnl'] = position["profit"]
                                self.telegram_bot.update_metrics(trade)
                                break
                        
                        await self.telegram_bot.send_trade_update(
                            position["ticket"],
                            symbol,
                            "CLOSED",
                            current_price,
                            position["profit"]
                        )
                    else:
                        await self.telegram_bot.send_error_alert(
                            f"Failed to close position {position['ticket']}"
                        )
            
        except Exception as e:
            logger.error(f"Error managing open trades: {str(e)}")
            await self.telegram_bot.send_error_alert(
                f"Error managing trades: {str(e)}"
            )
    
    async def main_loop(self):
        """Main trading loop."""
        try:
            while self.running:
                try:
                    if not self.telegram_bot.trading_enabled:
                        await asyncio.sleep(60)
                        continue
                    
                    # Check each symbol and timeframe
                    for symbol in self.trading_config["symbols"]:
                        for timeframe in self.trading_config["timeframes"]:
                            await self.check_symbol(symbol, timeframe)
                    
                    # Manage open trades
                    await self.manage_open_trades()
                    
                    # Wait before next iteration
                    await asyncio.sleep(60)  # Check every minute
                    
                except Exception as e:
                    logger.error(f"Error in main loop iteration: {str(e)}")
                    if self.telegram_bot:
                        await self.telegram_bot.send_error_alert(f"Error in main loop: {str(e)}")
        except Exception as e:
            logger.error(f"Fatal error in main loop: {str(e)}")
            await self.stop()

    async def analyze_market(self, symbol: str, timeframe: str):
        """Analyze market conditions for a symbol."""
        try:
            # Get market data
            df = self.mt5.get_market_data(symbol, timeframe)
            if df is None or len(df) < self.signal_generator.max_period:
                logger.warning(f"Insufficient data for {symbol} {timeframe}")
                return None
            
            # Calculate indicators
            df = self.signal_generator.calculate_indicators(df)
            
            # Get MTF data
            mtf_data = {}
            for tf in self.trading_config["timeframes"]:
                if tf != timeframe:  # Skip current timeframe
                    mtf_df = self.mt5.get_market_data(symbol, tf)
                    if mtf_df is not None:
                        mtf_data[tf] = mtf_df
            
            # Generate signal
            signal = self.signal_generator.generate_signal(
                df=df,
                symbol=symbol,
                timeframe=timeframe,
                mtf_data=mtf_data
            )
            
            # Analyze trend and POIs
            trend = self.analyze_trend(df)
            pois = self.detect_pois(df)
            
            # Get current session
            session_name, session_data = self.market_analysis.get_current_session()
            session_analysis = self.market_analysis.analyze_session_conditions(df, symbol)
            
            analysis = {
                'symbol': symbol,
                'timeframe': timeframe,
                'trend': trend,
                'pois': pois,
                'signal': signal,
                'data': df,
                'mtf_data': mtf_data,
                'market_structure': {
                    'support_levels': [pois['support']],
                    'resistance_levels': [pois['resistance']],
                    'current_price': pois['current_price']
                },
                'session': session_name,
                'session_data': session_data,
                'session_analysis': session_analysis
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing market: {str(e)}")
            return None

    def analyze_trend(self, df):
        # Simple trend analysis using moving averages
        df['MA20'] = df['close'].rolling(window=20).mean()
        df['MA50'] = df['close'].rolling(window=50).mean()
        
        current_ma20 = df['MA20'].iloc[-1]
        current_ma50 = df['MA50'].iloc[-1]
        
        if current_ma20 > current_ma50:
            trend = "Bullish"
        elif current_ma20 < current_ma50:
            trend = "Bearish"
        else:
            trend = "Sideways"
            
        return trend

    def detect_pois(self, df):
        # Detect Points of Interest (support/resistance levels)
        highs = df['high'].rolling(window=20).max()
        lows = df['low'].rolling(window=20).min()
        
        current_price = df['close'].iloc[-1]
        nearby_high = highs.iloc[-1]
        nearby_low = lows.iloc[-1]
        
        pois = {
            "resistance": nearby_high,
            "support": nearby_low,
            "current_price": current_price
        }
        
        return pois

    def analyze_session(self):
        """Determine current trading session based on UTC time."""
        try:
            current_time = datetime.utcnow()
            hour = current_time.hour
            
            # Check each session's time range
            for session_name, session_data in self.session_config.items():
                session_start = datetime.strptime(session_data['start'], '%H:%M').time()
                session_end = datetime.strptime(session_data['end'], '%H:%M').time()
                current_time_only = current_time.time()
                
                # Handle session crossing midnight
                if session_start > session_end:
                    # Session spans across midnight
                    if current_time_only >= session_start or current_time_only <= session_end:
                        return session_name.replace('_session', '')
                else:
                    # Normal session check
                    if session_start <= current_time_only <= session_end:
                        return session_name.replace('_session', '')
            
            return "no_session"
            
        except Exception as e:
            logger.error(f"Error determining session: {str(e)}")
            return "no_session"

    async def generate_signals(self, analysis, symbol, timeframe):
        """Generate trading signals based on market analysis."""
        logger.info(f"Generating signals for {symbol} on {timeframe}")
        
        # Apply primary filters
        tf_alignment = await self.check_timeframe_alignment(analysis)
        session_check = self.check_session_conditions(analysis)
        
        logger.debug(f"Primary Filters: TF_Alignment={tf_alignment}, Session_Check={session_check}")
        
        if not (tf_alignment and session_check):
            logger.debug(f"Primary filters not passed for {symbol}")
            return None
            
        # Calculate confirmation score
        confirmation_score = 0
        
        # Check SMT divergence
        smt_divergence = self.check_smt_divergence(analysis)
        if smt_divergence:
            confirmation_score += 0.6  # 60% weight for SMT divergence
        
        # Check liquidity
        liquidity = self.check_liquidity(analysis)
        if liquidity:
            confirmation_score += 0.4  # 40% weight for liquidity
        
        logger.debug(f"Confirmation Score: {confirmation_score:.2f} (SMT={smt_divergence}, Liquidity={liquidity})")
        
        # Generate signal based on trend and confirmation score
        if analysis['trend'] == "Bullish":
            if confirmation_score >= 0.6:  # Require at least SMT divergence or liquidity + partial other signal
                signal = "BUY"
                confidence = int(70 + (confirmation_score * 30))  # Scale 70-100 based on confirmation
            else:
                signal = "HOLD"
                confidence = 50
        elif analysis['trend'] == "Bearish":
            if confirmation_score >= 0.6:
                signal = "SELL"
                confidence = int(70 + (confirmation_score * 30))
            else:
                signal = "HOLD"
                confidence = 50
        else:
            signal = "HOLD"
            confidence = 50
            
        logger.info(f"Generated {signal} signal for {symbol} with {confidence}% confidence")
        
        # Add signal to dashboard
        add_signal({
            'symbol': symbol,
            'type': signal,
            'price': analysis['pois']['current_price'],
            'confidence': confidence,
            'trend': analysis['trend'],
            'session': analysis['session']
        })
        
        return [{
            "signal": signal,
            "confidence": confidence,
            "current_price": analysis["pois"]["current_price"],
            "support": analysis["pois"]["support"],
            "resistance": analysis["pois"]["resistance"],
            "trend": analysis["trend"],
            "session": analysis["session"]
        }]

    async def check_timeframe_alignment(self, analysis):
        """Check if the trend aligns across timeframes."""
        try:
            if not analysis or "timeframe" not in analysis or "symbol" not in analysis:
                logger.error("Invalid analysis data for timeframe alignment check")
                return False
            
            # Get data for higher timeframes
            higher_tf_data = {}
            timeframes = self.trading_config["timeframes"]
            current_tf_idx = timeframes.index(analysis["timeframe"])
            
            # Check if we have higher timeframes to check
            if current_tf_idx >= len(timeframes) - 1:
                logger.debug(f"No higher timeframes available for {analysis['timeframe']}")
                return True  # Already at highest timeframe
            
            # Check alignment with higher timeframes
            for tf in timeframes[current_tf_idx + 1:]:
                tf_analysis = await self.analyze_market(analysis["symbol"], tf)
                if tf_analysis and tf_analysis.get("trend") == analysis.get("trend"):
                    higher_tf_data[tf] = tf_analysis
                    logger.debug(f"Timeframe {tf} aligned with trend")
            
            # Require at least one higher timeframe alignment
            is_aligned = len(higher_tf_data) > 0
            logger.debug(f"Timeframe alignment result: {is_aligned}")
            return is_aligned
            
        except Exception as e:
            logger.error(f"Error in timeframe alignment check: {str(e)}")
            return False

    def check_session_conditions(self, analysis):
        """Check if current session is suitable for trading."""
        try:
            if not analysis or "session" not in analysis or "symbol" not in analysis:
                logger.error("Invalid analysis data for session conditions check")
                return False
            
            session = analysis["session"]  # Get session name
            symbol = analysis["symbol"]
            
            # Add '_session' suffix if not present and not 'no_session'
            if session != "no_session" and not session.endswith("_session"):
                session_key = f"{session}_session"
            else:
                session_key = session
            
            # Get session rules from config
            if session_key not in self.session_config:
                logger.warning(f"No rules found for session: {session}")
                return False
            
            session_rules = self.session_config[session_key]
            
            # Check if symbol is allowed in current session
            if symbol not in session_rules["pairs"]:
                logger.debug(f"Symbol {symbol} not allowed in {session} session")
                return False
            
            # Get current volatility and spread
            try:
                volatility = self.calculate_volatility(symbol)
                spread = self.get_spread(symbol)
                
                # Check volatility and spread conditions
                volatility_ok = volatility >= session_rules["min_range_pips"]
                spread_ok = spread <= session_rules.get("max_spread", float('inf'))
                
                logger.debug(f"Session conditions for {symbol}: Volatility={volatility_ok}, Spread={spread_ok}")
                return volatility_ok and spread_ok
                
            except Exception as e:
                logger.error(f"Error checking market conditions: {str(e)}")
                return False
                
        except Exception as e:
            logger.error(f"Error in session conditions check: {str(e)}")
            return False

    def check_smt_divergence(self, analysis):
        """Check for Smart Money Concepts divergence."""
        try:
            df = analysis["data"]
            
            # Calculate RSI
            df['RSI'] = self.calculate_rsi(df['close'])
            
            # Get last few candles
            last_candles = df.tail(5)
            
            # Check for bullish divergence
            if analysis["trend"] == "Bullish":
                price_making_lower_low = last_candles['low'].iloc[-1] < last_candles['low'].min()
                rsi_making_higher_low = last_candles['RSI'].iloc[-1] > last_candles['RSI'].min()
                return price_making_lower_low and rsi_making_higher_low
                
            # Check for bearish divergence
            elif analysis["trend"] == "Bearish":
                price_making_higher_high = last_candles['high'].iloc[-1] > last_candles['high'].max()
                rsi_making_lower_high = last_candles['RSI'].iloc[-1] < last_candles['RSI'].max()
                return price_making_higher_high and rsi_making_lower_high
                
            return False
            
        except Exception as e:
            logger.error(f"Error in SMT divergence check: {str(e)}")
            return False

    def check_liquidity(self, analysis):
        """Check for liquidity conditions."""
        try:
            df = analysis["data"]
            
            # Calculate average volume
            avg_volume = df['tick_volume'].rolling(window=20).mean()
            current_volume = df['tick_volume'].iloc[-1]
            
            # Check for volume spike
            volume_spike = current_volume > (avg_volume.iloc[-1] * 1.5)
            
            # Check for price at key levels
            price = analysis["pois"]["current_price"]
            near_support = abs(price - analysis["pois"]["support"]) / price < 0.001
            near_resistance = abs(price - analysis["pois"]["resistance"]) / price < 0.001
            
            return volume_spike and (near_support or near_resistance)
            
        except Exception as e:
            logger.error(f"Error in liquidity check: {str(e)}")
            return False

    def calculate_position_size(self, symbol):
        """Calculate position size based on risk management rules."""
        try:
            # Get account info
            account_info = mt5.account_info()
            if not account_info:
                raise Exception("Failed to get account info")
            
            balance = account_info.balance
            risk_amount = balance * self.trading_config["risk_per_trade"]
            
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                raise Exception(f"Failed to get symbol info for {symbol}")
            
            # Calculate position size based on risk
            pip_value = symbol_info.trade_tick_value
            stop_loss_pips = 50  # Default SL distance in pips
            
            position_size = risk_amount / (stop_loss_pips * pip_value)
            
            # Round to valid lot size
            min_lot = symbol_info.volume_min
            lot_step = symbol_info.volume_step
            position_size = round(position_size / lot_step) * lot_step
            
            # Ensure within limits
            position_size = max(min_lot, min(position_size, symbol_info.volume_max))
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0.01  # Default minimum size

    def calculate_trade_levels(self, symbol, signal):
        """Calculate entry, stop loss, and take profit levels."""
        try:
            # Get current price
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                raise Exception(f"Failed to get tick data for {symbol}")
            
            if signal == "BUY":
                entry = tick.ask
                stop_loss = entry - (entry * 0.01)  # 1% SL
                take_profit = entry + (entry * 0.02)  # 2% TP (1:2 RR)
            else:  # SELL
                entry = tick.bid
                stop_loss = entry + (entry * 0.01)  # 1% SL
                take_profit = entry - (entry * 0.02)  # 2% TP (1:2 RR)
            
            return entry, stop_loss, take_profit
            
        except Exception as e:
            logger.error(f"Error calculating trade levels: {str(e)}")
            return None, None, None

    def execute_trade(self, trade_params):
        """Execute trade on MT5."""
        try:
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": trade_params['symbol'],
                "volume": trade_params['position_size'],
                "type": mt5.ORDER_TYPE_BUY if trade_params['signal_type'] == 'BUY' else mt5.ORDER_TYPE_SELL,
                "price": trade_params['entry_price'],
                "sl": trade_params['stop_loss'],
                "tp": trade_params['take_profit'],
                "deviation": 10,
                "magic": 234000,
                "comment": f"Python Trading Bot - {trade_params['signal_type']}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                raise Exception(f"Order failed: {result.comment}")
            
            return result.order
            
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            return None

    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_volatility(self, symbol):
        """Calculate current volatility for a symbol in pips."""
        try:
            # Get recent price data (last hour of M5 data = 12 bars)
            data = self.mt5.get_market_data(symbol, "M5", 12)
            if data is None or len(data) < 2:
                logger.error(f"Insufficient data to calculate volatility for {symbol}")
                return 0
                
            # Calculate volatility as high-low range in pips
            high = data['high'].max()
            low = data['low'].min()
            
            # Convert to pips based on symbol type
            multiplier = 100 if symbol.endswith('JPY') else 10000
            volatility = (high - low) * multiplier
            
            logger.debug(f"Current volatility for {symbol}: {volatility} pips")
            return volatility
            
        except Exception as e:
            logger.error(f"Error calculating volatility: {str(e)}")
            return 0

    def get_spread(self, symbol):
        """Get current spread for a symbol in pips."""
        try:
            # Get current symbol info using MT5 directly
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                logger.error(f"Failed to get tick data for {symbol}")
                return float('inf')
            
            # Calculate spread in pips
            multiplier = 10000 if not symbol.endswith('JPY') else 100
            spread = (tick.ask - tick.bid) * multiplier
            
            logger.debug(f"Current spread for {symbol}: {spread} pips")
            return spread
            
        except Exception as e:
            logger.error(f"Error getting spread: {str(e)}")
            return float('inf')

    async def process_signals(self, signals: List[Dict]) -> None:
        """Process trading signals and execute trades.
        
        Args:
            signals: List of signal dictionaries containing trade information
        """
        try:
            if not signals:
                return
            
            for signal in signals:
                try:
                    if signal['signal'] == 'HOLD':
                        continue
                    
                    # Check if signal meets minimum confidence threshold
                    if signal['confidence'] < self.min_confidence:
                        logger.info(f"Signal confidence too low for {signal['symbol']}: {signal['confidence']}")
                        continue
                    
                    # Calculate position size
                    risk_amount = self.calculate_risk_amount()
                    position_size = self.risk_manager.calculate_position_size(
                        symbol=signal['symbol'],
                        risk_amount=risk_amount,
                        entry_price=signal['current_price'],
                        stop_loss=signal['support'] if signal['signal'] == 'SELL' else signal['resistance']
                    )
                    
                    if position_size is None or position_size <= 0:
                        logger.warning(f"Invalid position size calculated for {signal['symbol']}")
                        continue
                    
                    # Prepare trade parameters
                    trade_params = {
                        'symbol': signal['symbol'],
                        'signal_type': signal['signal'],
                        'entry_price': signal['current_price'],
                        'stop_loss': signal['support'] if signal['signal'] == 'SELL' else signal['resistance'],
                        'take_profit': signal['resistance'] if signal['signal'] == 'BUY' else signal['support'],
                        'position_size': position_size,
                        'confidence': signal['confidence']
                    }
                    
                    # Execute trade
                    if self.execute_trade(trade_params):
                        # Send trade alert
                        if self.telegram_bot:
                            await self.telegram_bot.send_trade_alert(
                                symbol=signal['symbol'],
                                signal_type=signal['signal'],
                                entry=signal['current_price'],
                                sl=trade_params['stop_loss'],
                                tp=trade_params['take_profit'],
                                confidence=signal['confidence'],
                                reason=f"Analysis: Trend={signal['trend']}"
                            )
                except Exception as e:
                    logger.error(f"Error processing signals for {signal['symbol']}: {str(e)}")
        except Exception as e:
            logger.error(f"Error in process_signals: {str(e)}")

    async def run(self):
        """Main trading loop."""
        try:
            logger.info("Starting trading bot...")
            
            # Initialize MT5 connection
            if not self.mt5.connect():
                logger.error("Failed to connect to MT5")
                return
            
            # Initialize components
            self.running = True
            
            while self.running:
                try:
                    signals = []
                    
                    # Process each symbol
                    for symbol in self.trading_config["symbols"]:
                        try:
                            # Check if symbol is allowed in current session
                            session = self.analyze_session()
                            if symbol not in session['allowed_symbols']:
                                logger.info(f"{symbol} not allowed in {session['name']} session")
                                continue
                            
                            # Check volatility requirements
                            volatility = self.calculate_volatility(symbol)
                            if volatility < session['min_volatility']:
                                logger.info(f"{symbol} volatility ({volatility:.1f} pips) below minimum ({session['min_volatility']} pips)")
                                continue
                            
                            # Analyze market for each timeframe
                            timeframe_signals = []
                            for timeframe in self.trading_config["timeframes"]:
                                signal = await self.analyze_market(symbol, timeframe)
                                if signal:
                                    timeframe_signals.append(signal)
                            
                            if not timeframe_signals:
                                continue
                            
                            # Combine signals from different timeframes
                            combined_signal = self._combine_timeframe_signals(timeframe_signals)
                            if combined_signal:
                                signals.append(combined_signal)
                            
                        except Exception as e:
                            logger.error(f"Error processing {symbol}: {str(e)}")
                            if self.telegram_bot:
                                await self.telegram_bot.send_error_alert(
                                    f"Error analyzing {symbol}: {str(e)}"
                                )
                    
                    # Process signals
                    if signals:
                        await self.process_signals(signals)
                    
                    # Update dashboard
                    if self.dashboard:
                        self.dashboard.update_status({
                            'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'active_symbols': [signal['symbol'] for signal in signals],
                            'session': session['name'],
                            'signals': signals
                        })
                    
                    # Wait for next iteration
                    await asyncio.sleep(self.check_interval)
                    
                except Exception as e:
                    logger.error(f"Error in main loop: {str(e)}")
                    if self.telegram_bot:
                        await self.telegram_bot.send_error_alert(
                            f"Error in main trading loop: {str(e)}"
                        )
                    await asyncio.sleep(60)  # Wait longer on error
            
        except Exception as e:
            logger.error(f"Fatal error in trading bot: {str(e)}")
            if self.telegram_bot:
                await self.telegram_bot.send_error_alert(
                    f"Fatal error in trading bot: {str(e)}"
                )
        finally:
            await self.stop()

    def _combine_timeframe_signals(self, timeframe_signals: List[Dict]) -> Optional[Dict]:
        """Combine signals from multiple timeframes."""
        try:
            if not timeframe_signals:
                return None
            
            # Initialize scores
            combined_score = 0
            total_weight = 0
            
            # Weights for different timeframes
            weights = {
                "M5": 0.15,
                "M15": 0.20,
                "H1": 0.25,
                "H4": 0.25,
                "D1": 0.15
            }
            
            # Combine scores from different timeframes
            for signal in timeframe_signals:
                if signal['signal'] == 'HOLD':
                    continue
                
                # Calculate weighted score
                score = signal['confidence']
                if signal['signal'] == 'SELL':
                    score = -score
                
                combined_score += score * weights[signal['timeframe']]
                total_weight += weights[signal['timeframe']]
            
            if total_weight == 0:
                return None
            
            # Normalize combined score
            final_score = combined_score / total_weight
            
            # Determine final signal type and confidence
            if abs(final_score) < 0.3:
                signal_type = 'HOLD'
                confidence = 0
            else:
                signal_type = 'BUY' if final_score > 0 else 'SELL'
                confidence = abs(final_score)
            
            # Get primary timeframe signal for price levels
            primary_signal = next((signal for signal in timeframe_signals if signal['timeframe'] == 'H1'), None)
            if not primary_signal:
                return None
            
            return {
                'signal_type': signal_type,
                'confidence': confidence,
                'current_price': primary_signal['pois']['current_price'],
                'support': primary_signal['pois']['support'],
                'resistance': primary_signal['pois']['resistance'],
                'trend': primary_signal['trend'],
                'analysis': {
                    'timeframes': {signal['timeframe']: signal['analysis'] for signal in timeframe_signals},
                    'combined_score': final_score
                }
            }
            
        except Exception as e:
            logger.error(f"Error combining timeframe signals: {str(e)}")
            return None

    def _is_symbol_allowed_in_session(self, symbol: str, session: str) -> bool:
        """Check if symbol is allowed to trade in current session."""
        try:
            if session == "no_session":
                return False
                
            session_key = f"{session}_session"
            if session_key not in self.session_config:
                return False
                
            return symbol in self.session_config[session_key]["pairs"]
        except Exception as e:
            logger.error(f"Error checking symbol session allowance: {str(e)}")
            return False

if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        level="INFO"
    )
    logger.add(
        "logs/trading_bot.log",
        rotation="1 day",
        retention="1 month",
        compression="zip",
        level="INFO"
    )
    
    # Start the bot
    bot = TradingBot()
    asyncio.run(bot.start()) 