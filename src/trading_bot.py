import asyncio
from typing import Dict, List, Optional
from loguru import logger
import pandas as pd
from datetime import datetime, timedelta, UTC, time
import json
import sys
import MetaTrader5 as mt5
import numpy as np
import threading
import pytz
from pathlib import Path
import traceback
import time as tm
import math

from config.config import (
    TRADING_CONFIG, SESSION_CONFIG, MT5_CONFIG, MARKET_STRUCTURE_CONFIG,
    SIGNAL_THRESHOLDS, CONFIRMATION_CONFIG, MARKET_SCHEDULE_CONFIG, TELEGRAM_CONFIG
)
from src.mt5_handler import MT5Handler
from src.signal_generator import SignalGenerator
from src.risk_manager import RiskManager
from src.telegram_bot import TelegramBot
from src.models import Trade, Signal, MarketData, NewsEvent
from src.dashboard import init_app, app, add_signal
from src.market_analysis import MarketAnalysis
from src.smc_analysis import SMCAnalysis
from src.mtf_analysis import MTFAnalysis
from src.divergence_analysis import DivergenceAnalysis
from src.volume_analysis import VolumeAnalysis
from src.poi_detector import POIDetector, POI

BASE_DIR = Path(__file__).resolve().parent.parent

class TradingBot:
    def __init__(self, config=None):
        """Initialize the trading bot."""
        self.config = config
        self.mt5 = None
        self.signal_generator = SignalGenerator()
        self.risk_manager = RiskManager()
        self.telegram_bot = TelegramBot()
        self.dashboard = None
        self.session_config = SESSION_CONFIG
        self.market_schedule = MARKET_SCHEDULE_CONFIG
        self.trading_config = TRADING_CONFIG if config is None else config.TRADING_CONFIG
        self.running = False
        self.trading_enabled = False  # Add this line
        self.trades: List[Trade] = []
        self.signals: List[Signal] = []
        self.market_data: Dict[str, MarketData] = {}
        self.news_events: List[NewsEvent] = []
        self.trade_counter = 0
        self.last_signal = {}  # Dictionary to track last signal timestamp and direction per symbol
        self.ny_timezone = pytz.timezone('America/New_York')
        self.min_confidence = SIGNAL_THRESHOLDS["weak"]
        
        # Initialize analysis components with H4 timeframe thresholds as default
        self.market_analysis = MarketAnalysis(ob_threshold=MARKET_STRUCTURE_CONFIG['structure_levels']['H4']['ob_size'])
        self.smc_analysis = SMCAnalysis()
        self.mtf_analysis = MTFAnalysis()
        self.divergence_analysis = DivergenceAnalysis()
        self.volume_analysis = VolumeAnalysis()
        self.poi_detector = POIDetector()  # Add POIDetector initialization
    
    def setup_logging(self):
        """Set up detailed logging configuration."""
        # Define custom format for different log levels
        fmt = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "{message}"
        )
        
        logger.remove()  # Remove default handler
        logger.add(
            "logs/trading_bot.log",
            format=fmt,
            level="DEBUG",
            rotation="1 day",
            retention="1 month",
            compression="zip",
            enqueue=True,
            backtrace=True,
            diagnose=True,
            catch=True,
        )
        logger.add(sys.stderr, format=fmt, level="INFO", colorize=True)

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

    def is_market_open(self) -> bool:
        """
        Check if the forex market is currently open based on schedule and holidays.
        Returns True if market is open, False otherwise.
        
        Forex market is open 24/5 - from Sunday evening to Friday evening local time,
        except for holidays.
        """
        try:
            # Get current local time
            local_time = datetime.now()
            current_date = local_time.date()
            current_weekday = current_date.weekday()  # 0 = Monday, 6 = Sunday
            
            logger.info(f"Local time: {local_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Check if it's a holiday
            current_year = str(current_date.year)
            if current_year in self.market_schedule["holidays"]:
                holiday_dates = [
                    datetime.strptime(date, "%Y-%m-%d").date()
                    for date in self.market_schedule["holidays"][current_year].values()
                ]
                if current_date in holiday_dates:
                    logger.info(f"Market is closed for holiday on {current_date}")
                    return False
            
            # Check if it's a partial trading day
            if current_year in self.market_schedule["partial_trading_days"]:
                for day_info in self.market_schedule["partial_trading_days"][current_year].values():
                    if datetime.strptime(day_info["date"], "%Y-%m-%d").date() == current_date:
                        close_time = datetime.strptime(day_info["close_time"], "%H:%M").time()
                        if local_time.time() >= close_time:
                            logger.info(f"Market is closed for partial trading day at {close_time} local time")
                            return False
            
            # Market is closed on Saturday
            if current_weekday == 5:  # Saturday
                logger.debug("Market is closed (Saturday)")
                return False
            
            # Market is closed on Sunday until 23:00 (11 PM) local time
            if current_weekday == 6:  # Sunday
                market_open = local_time.replace(hour=23, minute=0, second=0, microsecond=0)
                if local_time < market_open:
                    logger.debug("Market is closed (Sunday before 11 PM)")
                    return False
            
            # Market is closed Friday after 22:00 (10 PM) local time
            if current_weekday == 4:  # Friday
                market_close = local_time.replace(hour=22, minute=0, second=0, microsecond=0)
                if local_time >= market_close:
                    logger.debug("Market is closed (Friday after 10 PM)")
                    return False
            
            # If we got here, market is open
            logger.debug("Market is open")
            return True
            
        except Exception as e:
            logger.error(f"Error checking market schedule: {str(e)}")
            # If there's an error checking the schedule, assume market is closed for safety
            return False

    async def start(self):
        """Start the trading bot."""
        try:
            # If already running, just enable trading
            if self.running:
                if self.telegram_bot and self.telegram_bot.is_running:
                    self.telegram_bot.trading_enabled = True
                    logger.info("Trading enabled on already running bot")
                return
            
            logger.info("Starting trading bot...")
            
            # Initialize MT5 first
            if not self.initialize_mt5():
                raise Exception("Failed to initialize MT5")
            logger.info("MT5 initialized successfully")
            
            # Initialize dashboard only if not already running
            try:
                if not hasattr(self, '_dashboard_initialized'):
                    self.initialize_dashboard()
                    self._dashboard_initialized = True
                    logger.info("Dashboard initialized successfully")
                else:
                    logger.info("Dashboard already initialized")
            except Exception as e:
                logger.error(f"Dashboard initialization failed: {str(e)}")
                # Continue even if dashboard fails
            
            # Initialize Telegram bot with retry
            telegram_init_attempts = 3
            telegram_init_success = False
            
            for attempt in range(telegram_init_attempts):
                try:
                    if not self.telegram_bot or not self.telegram_bot.is_running:
                        logger.info(f"Attempting to initialize Telegram bot (attempt {attempt + 1}/{telegram_init_attempts})")
                        if await self.telegram_bot.initialize(self.trading_config):
                            telegram_init_success = True
                            logger.info("Telegram bot initialized successfully")
                            break
                    else:
                        logger.info("Telegram bot already running")
                        telegram_init_success = True
                        break
                    
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
            
            # Enable trading by default
            self.trading_enabled = True
            await self.enable_trading()  # This will also enable trading on the Telegram bot
            
            self.running = True
            logger.info("Trading bot started successfully")
            
            # Load existing trades into memory
            trades_file = Path(BASE_DIR) / "data" / "active_trades.json"
            if trades_file.exists():
                try:
                    with open(trades_file, 'r') as f:
                        self.trades = json.load(f)
                    logger.info(f"Loaded {len(self.trades)} existing trades")
                except Exception as e:
                    logger.error(f"Error loading existing trades: {str(e)}")
            
            # Start main trading loop
            while self.running:
                try:
                    # Check if Telegram bot is running and trading is enabled
                    if not (self.telegram_bot and self.telegram_bot.is_running):
                        logger.error("Telegram bot not running. Stopping trading bot...")
                        break
                        
                    if not self.telegram_bot.trading_enabled:
                        logger.debug("Trading is disabled. Waiting for /enable command...")
                        await asyncio.sleep(5)
                        continue
                    
                    # Check if market is open
                    if not self.is_market_open():
                        logger.info("Market is currently closed. Waiting for market open...")
                        # Check every 5 minutes when market is closed
                        await asyncio.sleep(300)
                        continue
                    
                    logger.info("Starting market analysis cycle...")
                    
                    # Get current session
                    current_session = self.analyze_session()
                    logger.info(f"Current trading session: {current_session}")
                    
                    # Process each symbol
                    for symbol in self.trading_config["symbols"]:
                        try:
                            # Process each timeframe
                            for timeframe in self.trading_config["timeframes"]:
                                # Perform market analysis
                                analysis = await self.analyze_market(symbol, timeframe)
                                if analysis:
                                    # Generate signals based on analysis
                                    signals = await self.generate_signals(analysis, symbol, timeframe)
                                    if signals:
                                        # Process the signals
                                        await self.process_signals(signals)
                            
                            # Manage open trades for this symbol
                            await self.manage_open_trades()
                            
                        except Exception as e:
                            error_trace = traceback.format_exc()
                            logger.error(f"Error processing {symbol}: {str(e)}\nTraceback:\n{error_trace}")
                            if self.telegram_bot and self.telegram_bot.is_running:
                                await self.telegram_bot.send_error_alert(
                                    f"Error analyzing {symbol}: {str(e)}\nTraceback:\n{error_trace}"
                                )
                    
                    await asyncio.sleep(60)  # Main loop interval
                    
                except Exception as e:
                    logger.error(f"Error in main loop: {str(e)}")
                    if self.telegram_bot and self.telegram_bot.is_running:
                        await self.telegram_bot.send_error_alert(f"Error in main loop: {str(e)}")
                    await asyncio.sleep(60)  # Wait longer on error
                    
        except Exception as e:
            logger.error(f"Bot error: {str(e)}")
            if self.telegram_bot and self.telegram_bot.is_running:
                await self.telegram_bot.send_error_alert(f"Bot error: {str(e)}")
            self.running = False
        finally:
            if not self.running:  # Only stop if we're actually shutting down
                await self.stop()

    async def stop(self, cleanup_only=False):
        """Stop the trading bot."""
        try:
            if not cleanup_only:
                self.running = False
                logger.info("Stopping trading bot...")
            
            # Close open positions if MT5 is available
            if self.mt5 is not None:
                try:
                    positions = self.mt5.get_open_positions()
                    if positions:
                        for position in positions:
                            if self.mt5.close_position(position["ticket"]):
                                if self.telegram_bot and self.telegram_bot.is_running:
                                    await self.telegram_bot.send_trade_update(
                                        position["ticket"],
                                        position["symbol"],
                                        "CLOSED (Bot Stop)",
                                        position["price_current"],
                                        position["profit"]
                                    )
                except Exception as e:
                    logger.warning(f"Error closing positions: {str(e)}")
            
            # Stop Telegram bot only if we're doing a full shutdown
            if self.telegram_bot is not None and not cleanup_only and not self.telegram_bot.trading_enabled:
                try:
                    await self.telegram_bot.stop()
                except Exception as e:
                    logger.warning(f"Error stopping Telegram bot: {str(e)}")
            
            # Shutdown MT5 if we're not just cleaning up
            if self.mt5 is not None and not cleanup_only:
                try:
                    mt5.shutdown()
                    self.mt5 = None
                except Exception as e:
                    logger.warning(f"Error shutting down MT5: {str(e)}")
            
            if not cleanup_only:
                logger.info("Trading bot stopped")
                
        except Exception as e:
            logger.error(f"Error stopping trading bot: {str(e)}")
            raise
    
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
                    
                    # Before placing a new order, check if a trade for the same symbol and same direction already exists
                    if any(pos["symbol"] == symbol and pos.get("type", "") == signal["signal"] for pos in self.risk_manager.open_trades):
                        logger.info(f"Trade for {symbol} with direction {signal['signal']} already active, skipping duplicate signal.")
                        return
                    
                    # Place trade
                    self.trade_counter += 1
                    trade_result = self.mt5.place_market_order(
                        symbol,
                        signal["signal_type"],
                        position_size,
                        stop_loss,
                        take_profit,
                        f"Signal: {signal['confidence']:.2f}"
                    )
                    
                    if trade_result:
                        # Send trade alert
                        await self.telegram_bot.send_trade_alert(
                            chat_id=int(TELEGRAM_CONFIG["allowed_user_ids"][0]),
                            direction=signal["signal_type"],
                            symbol=symbol,
                            entry=current_price,
                            sl=stop_loss,
                            tp=take_profit,
                            confidence=signal["confidence"],
                            reason=f"Analysis: Trend={signal['trend']}"
                        )
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
        """Manage existing trades with trailing stop loss."""
        try:
            # Get open positions
            positions = self.mt5.get_open_positions()
            if not positions:
                return
                
            self.risk_manager.update_open_trades(positions)
            
            for position in positions:
                try:
                    symbol = position["symbol"]
                    entry_price = position["price_open"]
                    initial_stop = position["sl_initial"]
                    current_stop = position["sl"]
                    position_size = position["volume"]
                    trade_type = "BUY" if position["type"] == mt5.ORDER_TYPE_BUY else "SELL"
                
                    # Skip if essential data is missing
                    if not all([entry_price, current_stop, position_size]):
                        logger.warning(f"Missing essential data for position {position['ticket']}")
                        continue
                    
                    # Get current market data
                    df = self.mt5.get_market_data(symbol, "M5", 100)  # Increased lookback for better context
                    if df is None or df.empty:
                        continue
                
                    current_price = df['close'].iloc[-1]
                    
                    # Get current tick data
                    current_tick = mt5.symbol_info_tick(symbol)
                    if not current_tick:
                        continue
                
                    # Calculate risk metrics
                    if trade_type == "BUY":
                        adverse_exc = (entry_price - df['low'].min()) / entry_price * 100
                        favorable_exc = (df['high'].max() - entry_price) / entry_price * 100
                        current_risk = (current_price - current_stop) / current_price * 100
                    else:
                        adverse_exc = (df['high'].max() - entry_price) / entry_price * 100
                        favorable_exc = (entry_price - df['low'].min()) / entry_price * 100
                        current_risk = (current_stop - current_price) / current_price * 100

                    should_adjust = False
                    new_stop = None
                
                    # Only adjust stops if we have sufficient favorable excursion
                    min_favorable_exc = 0.5  # Minimum 0.5% favorable excursion before adjusting stops
                    if favorable_exc >= min_favorable_exc:
                        should_adjust, new_stop = self.risk_manager.should_adjust_stops(
                            trade={
                                'entry_price': entry_price,
                                'initial_stop': initial_stop,
                                'stop_loss': current_stop,
                                'direction': trade_type,
                                'favorable_excursion': favorable_exc
                            },
                            current_price=current_price,
                            current_atr=df['atr'].iloc[-1] if 'atr' in df.columns else None
                        )
                
                    if should_adjust and new_stop != current_stop:
                        # Validate stop adjustment
                        min_stop_distance = self.get_min_stop_distance(symbol)
                        if min_stop_distance:
                            if (trade_type == "BUY" and new_stop < current_price - min_stop_distance) or \
                               (trade_type == "SELL" and new_stop > current_price + min_stop_distance):
                                if self.mt5.modify_position(position["ticket"], new_stop, position["tp"]):
                                    await self.telegram_bot.send_management_alert(
                                        position["ticket"],
                                        symbol,
                                        "Stop Loss Adjusted",
                                        current_stop,
                                        new_stop,
                                        "Trailing Stop Update"
                                    )
                    
                    # Check for emergency closure conditions
                    emergency_close = False
                    emergency_reason = ""
                    
                    # 1. Excessive adverse excursion
                    max_adverse_exc = 2.0  # Maximum 2% adverse excursion
                    if adverse_exc > max_adverse_exc:
                        emergency_close = True
                        emergency_reason = f"Excessive adverse excursion: {adverse_exc:.2f}%"
                    
                    # 2. Invalid stop loss
                    if current_stop == 0 or not current_stop:
                        emergency_close = True
                        emergency_reason = "Invalid stop loss detected"
                    
                    # 3. Risk too high
                    max_risk = 2.0  # Maximum 2% risk per trade
                    if current_risk > max_risk:
                        emergency_close = True
                        emergency_reason = f"Risk too high: {current_risk:.2f}%"
                    
                    if emergency_close:
                        logger.warning(f"Emergency closure for {symbol} - {emergency_reason}")
                        if self.mt5.close_position(position["ticket"]):
                            await self.telegram_bot.send_trade_update(
                                position["ticket"],
                                symbol,
                                f"Emergency Close: {emergency_reason}",
                                current_price,
                                position["profit"]
                            )
                        continue
                
                except Exception as e:
                    logger.error(f"Error managing position for {position.get('symbol', 'Unknown')}: {str(e)}")
                    continue
            
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

    def analyze_trend(self, df: pd.DataFrame) -> str:
        """Analyze trend using moving averages and multiple confirmations.
        
        Returns:
            str: 'bullish', 'bearish', or 'neutral'
        """
        try:
            # Calculate moving averages
            df['MA20'] = df['close'].rolling(window=20).mean()
            df['MA50'] = df['close'].rolling(window=50).mean()
            df['MA200'] = df['close'].rolling(window=200).mean()
            
            # Get current values
            current_close = df['close'].iloc[-1]
            current_ma20 = df['MA20'].iloc[-1]
            current_ma50 = df['MA50'].iloc[-1]
            current_ma200 = df['MA200'].iloc[-1]
            
            # Calculate short-term momentum (last 5 candles)
            short_term_change = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5] * 100
            
            # Calculate medium-term momentum (last 20 candles)
            medium_term_change = (df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20] * 100
            
            # Calculate price position relative to MAs
            above_ma20 = current_close > current_ma20
            above_ma50 = current_close > current_ma50
            above_ma200 = current_close > current_ma200
            
            # Calculate MA alignments
            bullish_alignment = current_ma20 > current_ma50 and current_ma50 > current_ma200
            bearish_alignment = current_ma20 < current_ma50 and current_ma50 < current_ma200
            
            # Define trend thresholds
            MOMENTUM_THRESHOLD = 0.1  # 0.1% change
            
            # Determine trend with multiple confirmations
            bullish_conditions = [
                above_ma20,
                above_ma50,
                short_term_change > MOMENTUM_THRESHOLD,
                medium_term_change > 0,
                bullish_alignment
            ]
            
            bearish_conditions = [
                not above_ma20,
                not above_ma50,
                short_term_change < -MOMENTUM_THRESHOLD,
                medium_term_change < 0,
                bearish_alignment
            ]
            
            # Count confirmations
            bullish_count = sum(bullish_conditions)
            bearish_count = sum(bearish_conditions)
            
            # Require at least 3 confirmations for a trend
            if bullish_count >= 3:
                return 'bullish'
            elif bearish_count >= 3:
                return 'bearish'
            else:
                # Check if price is showing strong momentum in either direction
                if abs(short_term_change) > MOMENTUM_THRESHOLD * 2:
                    return 'bullish' if short_term_change > 0 else 'bearish'
                return 'neutral'
                
        except Exception as e:
            logger.error(f"Error analyzing trend: {str(e)}")
            return 'neutral'

    def calculate_momentum(self, df: pd.DataFrame) -> float:
        """Calculate momentum score based on multiple factors with enhanced accuracy."""
        try:
            # Calculate RSI with error handling
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss.replace(0, np.inf)  # Handle division by zero
            rsi = 100 - (100 / (1 + rs))
            
            # Calculate price momentum over multiple periods
            price_momentum_5 = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5] * 100
            price_momentum_10 = (df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10] * 100
            price_momentum_20 = (df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20] * 100
            
            # Calculate volume momentum with moving average crossovers
            volume_sma_short = df['volume'].rolling(window=10).mean()
            volume_sma_long = df['volume'].rolling(window=20).mean()
            volume_momentum = ((volume_sma_short.iloc[-1] - volume_sma_long.iloc[-1]) / volume_sma_long.iloc[-1]) * 100
            
            # Calculate MACD for trend confirmation
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            macd_momentum = (macd.iloc[-1] - signal.iloc[-1]) / abs(macd).mean()
            
            # Calculate moving average trend
            ma20 = df['close'].rolling(window=20).mean()
            ma50 = df['close'].rolling(window=50).mean()
            ma_trend = (ma20.iloc[-1] - ma50.iloc[-1]) / ma50.iloc[-1] * 100
            
            # Normalize and combine all factors with weighted importance
            rsi_score = (rsi.iloc[-1] - 50) / 50  # -1 to 1 scale
            price_score = (0.5 * price_momentum_5 + 0.3 * price_momentum_10 + 0.2 * price_momentum_20) / 10
            volume_score = volume_momentum / 100
            macd_score = macd_momentum
            ma_score = ma_trend / 5
            
            # Weighted combination with emphasis on recent price action and volume
            momentum_score = (
                0.25 * rsi_score +      # RSI weight
                0.30 * price_score +    # Recent price action weight
                0.20 * volume_score +   # Volume trend weight
                0.15 * macd_score +     # MACD confirmation weight
                0.10 * ma_score         # Moving average trend weight
            )
            
            # Scale to percentage and round to 2 decimals
            final_score = round(momentum_score * 100, 2)
            
            # Log components for debugging
            logger.debug(f"Momentum Components:")
            logger.debug(f"    RSI Score: {rsi_score:.4f}")
            logger.debug(f"    Price Score: {price_score:.4f}")
            logger.debug(f"    Volume Score: {volume_score:.4f}")
            logger.debug(f"    MACD Score: {macd_score:.4f}")
            logger.debug(f"    MA Score: {ma_score:.4f}")
            logger.debug(f"    Final Score: {final_score:.2f}")
            
            return final_score
            
        except Exception as e:
            logger.error(f"Error calculating momentum: {str(e)}")
            return 0.0

    def analyze_volume_trend(self, df: pd.DataFrame, momentum_score: float) -> str:
        """Determine volume trend with enhanced accuracy."""
        try:
            # Calculate volume moving averages
            volume_sma_10 = df['volume'].rolling(window=10).mean()
            volume_sma_20 = df['volume'].rolling(window=20).mean()
            
            # Calculate price direction
            price_direction = df['close'].iloc[-1] > df['close'].iloc[-5]
            
            # Calculate volume trend metrics
            volume_trend = volume_sma_10.iloc[-1] > volume_sma_20.iloc[-1]
            volume_increase = df['volume'].iloc[-1] > volume_sma_10.iloc[-1]
            
            # Calculate volume ratio
            recent_volume_avg = df['volume'].tail(5).mean()
            baseline_volume_avg = df['volume'].tail(20).mean()
            volume_ratio = recent_volume_avg / baseline_volume_avg if baseline_volume_avg > 0 else 1.0
            
            # Determine trend based on multiple factors
            if abs(momentum_score) >= 15:  # Strong momentum threshold
                if momentum_score > 0 and volume_trend and price_direction:
                    return 'bullish'
                elif momentum_score < 0 and volume_trend and not price_direction:
                    return 'bearish'
            
            if volume_ratio > 1.2 and volume_increase:  # Significant volume increase
                if price_direction:
                    return 'bullish'
                else:
                    return 'bearish'
            
            return 'neutral'
            
        except Exception as e:
            logger.error(f"Error analyzing volume trend: {str(e)}")
            return 'neutral'

    async def analyze_market(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """Analyze market conditions with enhanced logging."""
        try:
            logger.info(f"🔍 Starting market analysis for {symbol} on {timeframe}")
            
            # Get market data
            data = await self.mt5.get_rates(symbol, timeframe)
            if data is None:
                logger.error(f"❌ No data received for {symbol}")
                return None
                
            if len(data) < 100:  # Minimum required candles
                logger.error(f"❌ Insufficient data for {symbol} analysis: {len(data)} candles")
                return None
                
            # Convert to DataFrame and validate
            try:
                # Ensure data is a DataFrame
                df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data.copy()
            except Exception as e:
                logger.error(f"Error converting data to DataFrame: {str(e)}")
                return None
                
            # Add type tracing for debugging
            logger.debug(f"DataFrame types: {df.dtypes}")
                
            # Validate required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume', 'time']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"❌ Missing required columns in DataFrame for {symbol}: {missing_columns}")
                return None
                
            # Ensure proper data types and handle NaN values
            df = df.astype({
                'open': 'float64',
                'high': 'float64',
                'low': 'float64',
                'close': 'float64',
                'volume': 'float64'
            })
            
            # Handle NaN values
            if df.isna().any().any():
                logger.warning(f"Found NaN values in {symbol} data, dropping affected rows")
            df = df.dropna()
            if len(df) < 100:
                logger.error("❌ Insufficient data after dropping NaN values")
                return None
                
            current_price = float(df['close'].iloc[-1])  # Ensure current_price is float
            logger.debug(f"Current price type: {type(current_price)}")
        
            # Log basic market stats
            daily_range = ((df['high'].max() - df['low'].min()) / df['close'].mean()) * 100
            logger.info(f"    Daily Range: {daily_range:.2f}%")
            logger.info(f"    Analyzing {len(df)} candles")
            
            # Calculate technical indicators
            indicators = self.calculate_indicators(df)
            logger.info("📈 Technical Indicators:")
            logger.info(f"    RSI: {indicators['rsi']:.2f}")
            logger.info(f"    MACD Line: {indicators['macd']['macd_line']:.5f}")
            logger.info(f"    Signal Line: {indicators['macd']['signal_line']:.5f}")
            
            # Perform trend analysis
            trend = self.analyze_trend(df)
            logger.info(f"📈 Trend Analysis: {trend}")
            
            # Perform comprehensive market analysis
            structure = self.market_analysis.analyze(df, symbol=symbol, timeframe=timeframe)
            logger.info("🏗️ Market Structure:")
            if structure and isinstance(structure, dict):
                logger.info(f"    Structure Type: {structure.get('structure_type', 'Unknown')}")
                logger.info(f"    Key Levels: {structure.get('key_levels', [])}")
            else:
                logger.info("    Structure Type: Unknown")
                logger.info("    Key Levels: []")
            
            # Perform SMC analysis
            smc_results = self.smc_analysis.analyze(df)
            logger.info("🎯 SMC Analysis:")
            logger.info(f"    Liquidity Sweeps: {len(smc_results['liquidity_sweeps'])}")
            logger.info(f"    Breaker Blocks: {len(smc_results['breaker_blocks']['bullish']) + len(smc_results['breaker_blocks']['bearish'])}")
            logger.info(f"    Manipulation Points: {len(smc_results['manipulation_points'])}")
            
            # Detect points of interest
            pois = await self.detect_pois(df, symbol, timeframe)
            logger.info(f"🎯 Points of Interest detected: {len(pois['zones']['supply']) + len(pois['zones']['demand'])}")
            
            # Ensure POI values are numeric
            if isinstance(pois, dict):
                # Helper function to safely convert to float
                def safe_float_conversion(value):
                    if isinstance(value, dict):
                        if 'price_start' in value:
                            return float(value['price_start'])
                        elif 'price_end' in value:
                            return float(value['price_end'])
                        return None
                    elif value is not None:
                        try:
                            return float(value)
                        except (ValueError, TypeError):
                            return None
                    return None

                # Convert resistance
                pois['resistance'] = safe_float_conversion(pois.get('resistance'))

                # Convert support
                pois['support'] = safe_float_conversion(pois.get('support'))

                # Convert current price
                if pois.get('current_price') is not None:
                    pois['current_price'] = safe_float_conversion(pois.get('current_price'))
                else:
                    pois['current_price'] = float(df['close'].iloc[-1])

                # Log POI conversions for debugging
                logger.debug(f"POI Conversions:")
                logger.debug(f"    Resistance: {pois['resistance']} ({type(pois['resistance'])})")
                logger.debug(f"    Support: {pois['support']} ({type(pois['support'])})")
                logger.debug(f"    Current Price: {pois['current_price']} ({type(pois['current_price'])})")
        
            # Perform multi-timeframe analysis
            higher_tf = self.get_higher_timeframe(timeframe)
            if higher_tf:
                higher_tf_data = await self.mt5.get_rates(symbol, higher_tf)
                if higher_tf_data is not None:
                    mtf_analysis = self.mtf_analysis.analyze(higher_tf_data)
                    logger.info(
                        f"📊 Higher Timeframe ({higher_tf}) Analysis:\n"
                        f"    Trend: {mtf_analysis.get('trend', 'Unknown')}\n"
                        f"    Key Levels: {mtf_analysis.get('key_levels', [])}"
                    )
                else:
                    mtf_analysis = None
            
            # Enhanced Volume analysis
            volume_data = self.volume_analysis.analyze(df)
            
            # Calculate momentum using the new method
            momentum_score = self.calculate_momentum(df)
            
            # Update volume data with calculated momentum
            volume_data['momentum'] = momentum_score
            
            # Determine volume trend based on momentum
            volume_trend = self.analyze_volume_trend(df, momentum_score)
            
            # Calculate volume profile
            volume_profile = self.volume_analysis._calculate_volume_profile(df)
            
            # Calculate cumulative delta
            cumulative_delta = self.volume_analysis._calculate_cumulative_delta(df)
            
            # Update cumulative delta trend based on momentum
            if abs(momentum_score) > 25:
                cumulative_delta['trend'] = 'bullish' if momentum_score > 0 else 'bearish'
            
            # Find volume-based support/resistance levels
            volume_levels = self.volume_analysis._find_volume_levels(df, volume_profile)
            
            logger.info(
                f"📊 Volume Analysis:\n"
                f"    Volume Trend: {volume_trend}\n"
                f"    Momentum: {momentum_score:.2f}\n"
                f"    CVD Trend: {cumulative_delta.get('trend', 'neutral')}\n"
                f"    POC Price: {volume_profile.get('poc', 0):.5f}"
            )
            
            # Combine all analyses
            analysis = {
                'symbol': symbol,
                'timeframe': timeframe,
                'trend': trend,
                'structure': structure,
                'smc': smc_results,
                'pois': pois,
                'volume': {
                    **volume_data,
                    'profile': volume_profile,
                    'cumulative_delta': cumulative_delta,
                    'levels': volume_levels
                },
                'indicators': indicators,
                'current_price': current_price,
                'candle_count': len(df),
                'mtf_analysis': mtf_analysis,
                'timestamp': datetime.now(UTC).isoformat()
            }
            
            logger.info(f"✅ Market analysis completed for {symbol} on {timeframe}")
            return analysis
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            logger.error(f"❌ Error in market analysis for {symbol} on {timeframe}: {str(e)}\nTraceback:\n{error_trace}")
            return None
            
    def get_higher_timeframe(self, timeframe: str) -> Optional[str]:
        """Get the next higher timeframe."""
        timeframes = ['M5', 'M15', 'H1', 'H4']
        try:
            current_index = timeframes.index(timeframe)
            if current_index < len(timeframes) - 1:
                return timeframes[current_index + 1]
        except ValueError:
            pass
        return None

    async def detect_pois(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Dict:
        """Detect and analyze points of interest with enhanced tracking."""
        try:
            logger.info(f"Detecting POIs for {symbol} on {timeframe}")
            
            # Get POIs from detector
            poi_zones = self.poi_detector.detect_supply_demand_zones(df, timeframe)
            
            # Get current price
            current_price = float(df['close'].iloc[-1])
            logger.debug(f"POI Detection - Current Price: {current_price} ({type(current_price)})")
            
            # Update POI status based on current price
            for zone_type in ['supply', 'demand']:
                for poi in poi_zones[zone_type]:
                    try:
                        price_start = float(poi.price_start)
                        price_end = float(poi.price_end)
                        logger.debug(f"POI {zone_type} zone - Start: {price_start} ({type(price_start)}), End: {price_end} ({type(price_end)})")
                        
                        if zone_type == 'supply':
                            if current_price > price_start:
                                poi.status = 'broken'
                            elif current_price > price_end:
                                poi.status = 'tested'
                        else:  # demand
                            if current_price < price_end:
                                poi.status = 'broken'
                            elif current_price < price_start:
                                poi.status = 'tested'
                    except (ValueError, TypeError, AttributeError) as e:
                        logger.error(f"Error processing POI zone: {str(e)}")
                        continue
            
            # Get nearest POIs for immediate use and ensure numeric values
            try:
                nearest_supply = min(
                    (p for p in poi_zones['supply'] if float(p.price_start) > current_price),
                    key=lambda x: abs(float(x.price_start) - current_price),
                    default=None
                )
                
                nearest_demand = max(
                    (p for p in poi_zones['demand'] if float(p.price_end) < current_price),
                    key=lambda x: abs(float(x.price_end) - current_price),
                    default=None
                )
                
                resistance_level = float(nearest_supply.price_start) if nearest_supply else None
                support_level = float(nearest_demand.price_end) if nearest_demand else None
                
                logger.debug(f"POI Prices - Resistance: {resistance_level} ({type(resistance_level)}), "
                             f"Support: {support_level} ({type(support_level)})")
                
                result = {
                    'current_price': current_price,
                    'resistance': resistance_level,
                    'support': support_level,
                    'zones': {
                        'supply': [vars(p) for p in poi_zones['supply']],
                        'demand': [vars(p) for p in poi_zones['demand']]
                    },
                    'active_zones': [
                        vars(p) for zone_type in ['supply', 'demand']
                        for p in poi_zones[zone_type]
                        if p.status not in ['broken', 'invalid']
                    ]
                }
                
                # Log the structure of the result for debugging
                logger.debug(f"POI Result Structure: {result.keys()}")
                logger.debug(f"POI Result Types - current_price: {type(result['current_price'])}, "
                           f"resistance: {type(result['resistance'])}, support: {type(result['support'])}")
                
                return result
                
            except Exception as e:
                logger.error(f"Error processing nearest POIs: {str(e)}")
                return {
                    'current_price': current_price,
                    'resistance': None,
                    'support': None,
                    'zones': {'supply': [], 'demand': []},
                    'active_zones': []
                }
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            logger.error(f"Error detecting POIs: {str(e)}\nTraceback:\n{error_trace}")
            return {
                'current_price': float(df['close'].iloc[-1]) if not df.empty else None,
                'resistance': None,
                'support': None,
                'zones': {'supply': [], 'demand': []},
                'active_zones': []
            }

    def analyze_session(self) -> str:
        """Determine the current trading session based on local system time."""
        current_time = datetime.now().time()
        
        # Define session time ranges adjusted for local timezone
        sessions = {
            'asia': {
                'start': time(23, 0),  # 11:00 PM
                'end': time(8, 0)      # 8:00 AM
            },
            'london': {
                'start': time(8, 0),   # 8:00 AM
                'end': time(16, 0)     # 4:00 PM
            },
            'new_york': {
                'start': time(13, 0),  # 1:00 PM
                'end': time(22, 0)     # 10:00 PM
            }
        }
        
        # Log current time for debugging
        logger.info(f"🕒 Current local time: {current_time}")
        
        # Check each session
        for session_name, session_times in sessions.items():
            # Handle sessions that cross midnight
            if session_times['start'] > session_times['end']:
                # Session spans across midnight
                if current_time >= session_times['start'] or current_time <= session_times['end']:
                    logger.info(f"✅ Current session: {session_name}")
                    return session_name
            else:
                # Normal session within same day
                if session_times['start'] <= current_time <= session_times['end']:
                    logger.info(f"✅ Current session: {session_name}")
                    return session_name
            
        logger.info("❌ No active trading session")
        return "no_session"
    async def generate_signals(self, analysis, symbol, timeframe):
        """Generate trading signals based on market analysis."""
        logger.info(f"Generating signals for {symbol} on {timeframe}")
        
        # Validate input parameters
        if not symbol or not timeframe:
            logger.error("Missing required parameters: symbol or timeframe")
            return []
        
        if not analysis or not isinstance(analysis, dict):
            logger.error("Invalid analysis data")
            return [{"symbol": symbol, "signal": "HOLD", "confidence": 0}]
        
        # NEW: Check if there is already an open position for the symbol
        open_positions = self.mt5.get_open_positions()
        if open_positions:
            for position in open_positions:
                if position["symbol"] == symbol:
                    logger.info(f"Existing position detected for {symbol}. Skipping signal generation.")
                    return [{"symbol": symbol, "signal": "HOLD", "confidence": 0}]
        
        # Validate input parameters
        if not symbol or not timeframe:
            logger.error("Missing required parameters: symbol or timeframe")
            return []
        
        # Validate analysis data
        if not analysis or not isinstance(analysis, dict):
            logger.error("Invalid analysis data")
            return [{"symbol": symbol, "signal": "HOLD", "confidence": 0}]
        
        # Apply primary filters - removed session check
        tf_alignment = await self.check_timeframe_alignment(analysis)
        
        logger.debug(f"Primary Filters: TF_Alignment={tf_alignment}")
        
        if not tf_alignment:
            logger.debug("Primary filters not met")
            return [{
                "symbol": symbol,
                "signal": "HOLD",
                "direction": "HOLD",
                "timeframe": timeframe,
                "confidence": 0,
                "current_price": analysis.get("current_price", 0),
                "entry_price": analysis.get("current_price", 0),
                "support": analysis.get("pois", {}).get("support"),
                "resistance": analysis.get("pois", {}).get("resistance"),
                "stop_loss": analysis.get("current_price", 0),  # Set to current price for HOLD
                "take_profit": analysis.get("current_price", 0),  # Set to current price for HOLD
                "trend": "neutral",
                "session": analysis.get("session", self.analyze_session())
            }]
        
        # Calculate confirmation score using weights from config
        confirmation_score = 0
        confirmations_met = 0
        
        # Get POI data and calculate POI score
        poi_data = analysis.get('pois', {})
        nearest_supply = poi_data.get('resistance')
        nearest_demand = poi_data.get('support')
        active_zones = poi_data.get('active_zones', [])
        
        # Ensure POI values are numeric
        if isinstance(nearest_supply, dict):
            nearest_supply = float(nearest_supply.get('price_start', 0))
        elif nearest_supply is not None:
            nearest_supply = float(nearest_supply)
            
        if isinstance(nearest_demand, dict):
            nearest_demand = float(nearest_demand.get('price_end', 0))
        elif nearest_demand is not None:
            nearest_demand = float(nearest_demand)
            
        current_price = float(analysis.get('current_price', 0))
        
        # Analyze POI zones for signal confirmation
        poi_score = 0
        trend_lower = analysis['trend'].lower()
        
        # Get price action direction instead of relying solely on trend
        price_direction = None
        if 'current_price' in analysis and 'indicators' in analysis:
            last_close = current_price
            sma20 = float(analysis.get('indicators', {}).get('sma20', last_close))
            if last_close > sma20:
                price_direction = "bullish"
            elif last_close < sma20:
                price_direction = "bearish"
        
        # Use price_direction if trend is neutral
        direction = trend_lower if trend_lower != "neutral" else price_direction
        
        if direction == "bullish":
            if nearest_demand is not None:
                distance_to_demand = abs(current_price - nearest_demand)
                if distance_to_demand <= analysis.get('atr', 0) * 0.5:
                    poi_score += 0.3
                    logger.debug(f"Price near demand zone: +0.3 POI score")
            
            active_demand_zones = [z for z in active_zones if z['type'] == 'demand' and z['strength'] >= 0.7]
            if active_demand_zones:
                poi_score += 0.2
                logger.debug(f"Strong active demand zones present: +0.2 POI score")
        elif direction == "bearish":
            if nearest_supply is not None:
                distance_to_supply = abs(current_price - nearest_supply)
                if distance_to_supply <= analysis.get('atr', 0) * 0.5:
                    poi_score += 0.3
                    logger.debug(f"Price near supply zone: +0.3 POI score")
            
            active_supply_zones = [z for z in active_zones if z['type'] == 'supply' and z['strength'] >= 0.7]
            if active_supply_zones:
                poi_score += 0.2
                logger.debug(f"Strong active supply zones present: +0.2 POI score")
            
        # Add POI score to confirmation score
        confirmation_score += poi_score
        if poi_score > 0:
            confirmations_met += 1
            logger.debug(f"POI confirmation met with score: {poi_score:.2f}")
            
        # Analyze SMC patterns
        smc_score = 0
        smc_data = analysis.get('smc', {})
        
        # Check for liquidity sweeps using candle_count from analysis
        candle_count = analysis.get('candle_count', 0)
        recent_sweeps = [s for s in smc_data.get('liquidity_sweeps', []) 
                         if s['index'] >= candle_count - 5]
        
        if recent_sweeps:
            sweep = recent_sweeps[-1]  # Most recent sweep
            if (trend_lower == "bullish" and sweep['type'] == 'bullish') or \
               (trend_lower == "bearish" and sweep['type'] == 'bearish'):
                smc_score += 0.3
                logger.debug(f"Recent {sweep['type']} liquidity sweep: +0.3 SMC score")
        
        # Check for breaker blocks
        breaker_blocks = smc_data.get('breaker_blocks', {})
        if trend_lower == "bullish":
            bullish_breakers = breaker_blocks.get('bullish', [])
            if bullish_breakers and abs(analysis['current_price'] - bullish_breakers[-1]['low']) <= analysis.get('atr', 0):
                smc_score += 0.2
                logger.debug("Price near bullish breaker block: +0.2 SMC score")
        elif trend_lower == "bearish":
            bearish_breakers = breaker_blocks.get('bearish', [])
            if bearish_breakers and abs(analysis['current_price'] - bearish_breakers[-1]['high']) <= analysis.get('atr', 0):
                smc_score += 0.2
                logger.debug("Price near bearish breaker block: +0.2 SMC score")
        
        # Add SMC score to confirmation score
        if smc_score > 0:
            confirmation_score += smc_score
            confirmations_met += 1
            logger.debug(f"SMC confirmation met with score: {smc_score:.2f}")
        
        # Analyze Volume patterns
        volume_score = 0
        volume_data = analysis.get('volume', {})
        
        # Check volume trend alignment
        if (trend_lower == "bullish" and volume_data.get('trend') == 'bullish') or \
           (trend_lower == "bearish" and volume_data.get('trend') == 'bearish'):
            volume_score += 0.2
            logger.debug(f"Volume trend aligned: +0.2 Volume score")
        
        # Check cumulative delta
        cvd_data = volume_data.get('cumulative_delta', {})
        if (trend_lower == "bullish" and cvd_data.get('trend') == 'bullish') or \
           (trend_lower == "bearish" and cvd_data.get('trend') == 'bearish'):
            volume_score += 0.2
            logger.debug(f"CVD trend aligned: +0.2 Volume score")
        
        # Check volume levels
        volume_levels = volume_data.get('levels', {})
        if trend_lower == "bullish":
            support_levels = volume_levels.get('support', [])
            if support_levels:
                # Extract price value from support level dictionary
                support_price = support_levels[0].get('price') if isinstance(support_levels[0], dict) else support_levels[0]
                try:
                    support_price = float(support_price)
                    if abs(analysis['current_price'] - support_price) <= analysis.get('atr', 0):
                        volume_score += 0.1
                        logger.debug("Price near volume-based support: +0.1 Volume score")
                except (TypeError, ValueError) as e:
                    logger.warning(f"Could not convert support price to float: {e}")
        elif trend_lower == "bearish":
            resistance_levels = volume_levels.get('resistance', [])
            if resistance_levels:
                # Extract price value from resistance level dictionary
                resistance_price = resistance_levels[0].get('price') if isinstance(resistance_levels[0], dict) else resistance_levels[0]
                try:
                    resistance_price = float(resistance_price)
                    if abs(analysis['current_price'] - resistance_price) <= analysis.get('atr', 0):
                        volume_score += 0.1
                        logger.debug("Price near volume-based resistance: +0.1 Volume score")
                except (TypeError, ValueError) as e:
                    logger.warning(f"Could not convert resistance price to float: {e}")
        
        # Add Volume score to confirmation score
        if volume_score > 0:
            confirmation_score += volume_score
            confirmations_met += 1
            logger.debug(f"Volume confirmation met with score: {volume_score:.2f}")
        
        # Check SMT divergence
        smt_divergence = self.check_smt_divergence(analysis)
        if smt_divergence:
            confirmation_score += CONFIRMATION_CONFIG["weights"]["smt_divergence"]
            confirmations_met += 1
        
        # Check liquidity
        liquidity = self.check_liquidity(analysis)
        if liquidity:
            confirmation_score += CONFIRMATION_CONFIG["weights"]["liquidity_sweep"]
            confirmations_met += 1
        
        # Check momentum with adjusted multiplier
        momentum = self.check_momentum(analysis)
        if momentum:
            confirmation_score += CONFIRMATION_CONFIG["weights"]["momentum"]
            confirmations_met += 1
        
        logger.debug(f"Confirmation Scores:\n    POI: {poi_score:.2f}\n    SMC: {smc_score:.2f}\n    Volume: {volume_score:.2f}\n    Total: {confirmation_score:.2f}\n    Confirmations Met: {confirmations_met}")
        
        # More flexible confirmation requirements
        min_required_score = 0.5  # Minimum total score required
        min_confirmations = 2     # Minimum number of confirmations required
        
        # Check if either the score is high enough or we have enough confirmations
        if confirmation_score < min_required_score and confirmations_met < min_confirmations:
            logger.debug(f"Insufficient confirmations/score: {confirmations_met}/{min_confirmations} confirmations, score: {confirmation_score:.2f}/{min_required_score}")
            return [{"symbol": symbol, "signal": "HOLD", "confidence": 0}]
        
        # Calculate base confidence from confirmations with weighted scoring
        base_confidence = min(95, (confirmation_score / 1.5) * 100)  # Scale to percentage, max 95%
        
        # Generate signal based on trend and confirmation score
        trend_lower = analysis['trend'].lower()
        if trend_lower == "bullish":
            signal = "BUY"
            confidence = max(60, int(base_confidence))  # Minimum 60% confidence if signal generated
        elif trend_lower == "bearish":
            signal = "SELL"
            confidence = max(60, int(base_confidence))  # Minimum 60% confidence if signal generated
        else:
            signal = "HOLD"
            confidence = 0
        
        logger.info(f"Generated {signal} signal for {symbol} with {confidence}% confidence")
        
        # Compute trade levels with a minimum risk-reward ratio of 1:2
        current_price = float(analysis.get("current_price", 0))
        entry_price = current_price
        if signal == "BUY":
            stop_loss = analysis.get("pois", {}).get("support")
            if stop_loss is None:
                stop_loss = current_price * 0.999
            raw_take_profit = analysis.get("pois", {}).get("resistance")
            if raw_take_profit is None:
                raw_take_profit = current_price * 1.002
            # Adjust BUY stop loss if too close to entry
            min_stop_distance = self.get_min_stop_distance(symbol)
            if min_stop_distance and (entry_price - stop_loss) < min_stop_distance:
                logger.debug(f"Adjusting BUY stop loss from {stop_loss} to {entry_price - min_stop_distance} based on min stop dist {min_stop_distance}")
                stop_loss = entry_price - min_stop_distance
            risk = entry_price - stop_loss
            min_take_profit = entry_price + 2 * risk
            take_profit = raw_take_profit if raw_take_profit >= min_take_profit else min_take_profit
        else:
            stop_loss = analysis.get("pois", {}).get("resistance")
            if stop_loss is None:
                stop_loss = current_price * 1.001
            raw_take_profit = analysis.get("pois", {}).get("support")
            if raw_take_profit is None:
                raw_take_profit = current_price * 0.998
            # Adjust SELL stop loss if too close to entry
            min_stop_distance = self.get_min_stop_distance(symbol)
            if min_stop_distance and (stop_loss - entry_price) < min_stop_distance:
                logger.debug(f"Adjusting SELL stop loss from {stop_loss} to {entry_price + min_stop_distance} based on min stop dist {min_stop_distance}")
                stop_loss = entry_price + min_stop_distance
            risk = stop_loss - entry_price
            min_take_profit = entry_price - 2 * risk
            take_profit = raw_take_profit if raw_take_profit <= min_take_profit else min_take_profit
        
        # Create signal with all required fields
        signal_data = {
            "symbol": symbol,
            "signal": signal,
            "direction": signal,  # Add direction to match required keys
            "timeframe": timeframe,  # Add timeframe
            "confidence": confidence,
            "current_price": current_price,
            "entry_price": entry_price,
            "support": analysis.get("pois", {}).get("support"),
            "resistance": analysis.get("pois", {}).get("resistance"),
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "trend": analysis.get("trend", "neutral"),
            "session": analysis.get("session", self.analyze_session()),
            "pois": {
                "score": poi_score,
                "nearest_supply": nearest_supply,
                "nearest_demand": nearest_demand,
                "active_zones_count": len(active_zones)
            },
            "smc_data": {
                "score": smc_score,
                "recent_sweeps": recent_sweeps,
                "breaker_blocks": breaker_blocks
            },
            "volume_data": {
                "score": volume_score,
                "trend": volume_data.get("trend"),
                "cvd_trend": cvd_data.get("trend"),
                "levels": volume_levels
            }
        }
        
        # Validate signal data before returning
        required_fields = ['symbol', 'signal', 'confidence']
        if not all(field in signal_data for field in required_fields):
            logger.error(f"Generated signal missing required fields: {[field for field in required_fields if field not in signal_data]}")
            return [{"symbol": symbol, "signal": "HOLD", "confidence": 0}]
        
        # Add signal to dashboard with enhanced information
        add_signal({
            'session': analysis.get('session', self.analyze_session()),
            'symbol': symbol,  # Ensure symbol is included in dashboard data
            'type': signal,
            'price': analysis.get('current_price', 0),
            'confidence': confidence,
            'trend': analysis.get('trend', 'neutral'),
            'confirmations': {
                'total': confirmations_met,
                'required': min_confirmations,
                'score': confirmation_score,
                'details': {
                    'poi': poi_score,
                    'smc': smc_score,
                    'volume': volume_score,
                    'smt': smt_divergence,
                    'liquidity': liquidity,
                    'momentum': momentum
                }
            }
        })
        
        # NEW: Conflict check - implement cooldown period and check for conflicting signals
        now = datetime.now()
        cooldown = self.trading_config.get("signal_cooldown", 60)  # cooldown in seconds
        if symbol in self.last_signal:
            last_signal = self.last_signal[symbol]
            elapsed = (now - last_signal["timestamp"]).total_seconds()
            if elapsed < cooldown and last_signal["direction"] != signal_data["signal"]:
                logger.info(f"Conflict detected: Last signal for {symbol} was " +
                            f"{last_signal['direction']} {elapsed:.1f} sec ago; " +
                            f"new signal {signal_data['signal']} conflicts with cooldown.")
                return [{"symbol": symbol, "signal": "HOLD", "confidence": 0}]
        if signal_data["signal"] != "HOLD":
            self.last_signal[symbol] = {"timestamp": now, "direction": signal_data["signal"]}
        
        return [signal_data]

    async def check_timeframe_alignment(self, analysis):
        """Check if current timeframe trend aligns with higher timeframe trend."""
        try:
            # Temporarily disabled - always return True
            if not analysis or 'trend' not in analysis:
                logger.error(f"❌ Invalid analysis data for timeframe alignment check: {analysis}")
                return True  # Changed from False to True
                
            symbol = analysis['symbol']
            current_tf = analysis['timeframe']
            higher_tf = self.get_higher_timeframe(current_tf)
            
            logger.info(
                f"🔄 Higher Timeframe Alignment Check Disabled\n"
                f"    Current TF: {current_tf}\n"
                f"    Higher TF: {higher_tf}"
            )
            
            # Return True regardless of alignment
            return True
            
        except Exception as e:
            logger.error(f"❌ Error checking timeframe alignment: {str(e)}")
            logger.exception("Detailed error trace:")
            return True  # Changed from False to True

    def check_session_conditions(self, analysis):
        """Check session conditions with enhanced logging."""
        try:
            if not analysis:
                logger.error("❌ Invalid analysis data for session conditions check")
                return False
                
            symbol = analysis['symbol']
            current_time = datetime.now()
            current_session = self.analyze_session()
            
            logger.info(
                f"🕒 Checking session conditions for {symbol}:\n"
                f"    Current Time (Local): {current_time}\n"
                f"    Current Session: {current_session}"
            )
            
            # Get current volatility and spread
            volatility = self.calculate_volatility(symbol)
            spread = self.get_spread(symbol)
            
            logger.info(
                f"🔄 Session Conditions for {symbol}:\n"
                f"    Trading Allowed: ✅\n"  # Always allowed now
                f"    Volatility: {volatility:.2f}%\n"
                f"    Current Spread: {spread:.1f} pips"
            )
            
            # Check if conditions are favorable - ignoring session restrictions
            conditions_met = (
                volatility >= self.trading_config['min_volatility'] and
                spread <= self.trading_config['max_spread']
            )
            
            logger.info(
                f"{'✅' if conditions_met else '❌'} Session conditions "
                f"{'met' if conditions_met else 'not met'} for {symbol}"
            )
            
            return conditions_met
            
        except Exception as e:
            logger.error(f"❌ Error checking session conditions: {str(e)}")
            logger.exception("Detailed error trace:")
            return False

    def check_smt_divergence(self, analysis):
        """Check for Smart Money Concepts divergence using full divergence analysis capabilities."""
        try:
            symbol = analysis['symbol']
            timeframe = analysis['timeframe']
            df = self.mt5.get_market_data(symbol, timeframe)
            if df is None or df.empty:
                logger.error("No data available for divergence check")
                return False
            
            # Use the full DivergenceAnalysis
            div_analysis = DivergenceAnalysis()
            div_result = div_analysis.analyze(df)
            
            trend = analysis["trend"].lower()
            confirmed = False
            if trend == "bullish":
                if (len(div_result["regular"]["bullish"]) > 0 or 
                    len(div_result["hidden"]["bullish"]) > 0 or 
                    len(div_result["structural"]["bullish"]) > 0 or 
                    len(div_result["momentum"]["bullish"]) > 0):
                    confirmed = True
            elif trend == "bearish":
                if (len(div_result["regular"]["bearish"]) > 0 or 
                    len(div_result["hidden"]["bearish"]) > 0 or 
                    len(div_result["structural"]["bearish"]) > 0 or 
                    len(div_result["momentum"]["bearish"]) > 0):
                    confirmed = True
            
            logger.debug(f"Divergence analysis result: {div_result}")
            logger.debug(f"SMT divergence confirmed: {'✅' if confirmed else '❌'}")
            return confirmed
        except Exception as e:
            logger.error(f"Error in SMT divergence check: {str(e)}")
            return False

    def check_liquidity(self, analysis):
        """Check for liquidity conditions."""
        try:
            # Get market data from MT5
            df = self.mt5.get_market_data(analysis['symbol'], analysis['timeframe'])
            if df is None or df.empty:
                logger.error("No data available for liquidity check")
                return False
            
            # Calculate average volume
            avg_volume = df['tick_volume'].rolling(window=20).mean()
            current_volume = df['tick_volume'].iloc[-1]
            
            # Determine if there's a volume spike
            volume_spike = current_volume > (avg_volume.iloc[-1] * 1.5)
            
            # Set threshold based on volume spike presence
            if volume_spike:
                threshold = 0.002  # Within 0.2%
            else:
                threshold = 0.001  # Within 0.1% if no volume spike
                logger.debug("No volume spike detected - applying stricter proximity threshold.")
            
            # Get current price and key levels
            price = analysis["pois"]["current_price"]
            # Extract numeric values from POI dictionaries if needed
            support = analysis["pois"]["support"].get('price') if isinstance(
                analysis["pois"]["support"], dict) else analysis["pois"]["support"]
            resistance = analysis["pois"]["resistance"].get('price') if isinstance(
                analysis["pois"]["resistance"], dict) else analysis["pois"]["resistance"]
            
            # Calculate distances as percentage
            support_distance = abs(price - support) / price if support else float('inf')
            resistance_distance = abs(price - resistance) / price if resistance else float('inf')
            
            # Determine which level is closer and check if price is within the threshold
            if support_distance < resistance_distance:
                near_level = support_distance < threshold
                level_type = "support"
                distance = support_distance
            else:
                near_level = resistance_distance < threshold
                level_type = "resistance"
                distance = resistance_distance
            
            # Log liquidity conditions
            logger.info(
                f"Liquidity Conditions:\n"
                f"    Volume Spike: {'✅' if volume_spike else '❌'} "
                f"({current_volume:.0f} vs {avg_volume.iloc[-1]:.0f} avg)\n"
                f"    Near {level_type.title()} with threshold {threshold * 100:.2f}%: "
                f"{'✅' if near_level else '❌'} (Distance: {distance * 100:.3f}%)"
            )
            
            # Return true if the proximity condition is met
            return near_level
            
        except Exception as e:
            logger.error(f"Error in liquidity check: {str(e)}")
            return False

    def calculate_position_size(self, symbol, signal_confidence: float = 0.5):
        """Calculate position size based on risk management rules and signal confidence."""
        try:
            # Get account info
            account_info = mt5.account_info()
            if not account_info:
                raise Exception("Failed to get account info")
            
            balance = account_info.balance
            base_risk = balance * self.trading_config["risk_per_trade"]
            
            # Check daily risk limits first
            can_trade, limit_reason = self.risk_manager.check_daily_limits(balance, base_risk)
            if not can_trade:
                logger.warning(f"Trade rejected due to risk limits: {limit_reason}")
                return 0.0
            
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                raise Exception(f"Failed to get symbol info for {symbol}")
            
            # Get current volatility state
            volatility = self.calculate_volatility(symbol)
            volatility_state = 'normal'
            if volatility > 50:
                volatility_state = 'extreme'
            elif volatility > 30:
                volatility_state = 'high'
            elif volatility < 10:
                volatility_state = 'low'
            
            # Calculate current drawdown
            current_drawdown = self.calculate_drawdown()
            
            # Get current market data for ATR calculation
            df = self.mt5.get_market_data(symbol, "M5", 50)
            if df is None or df.empty:
                raise Exception(f"Failed to get market data for {symbol}")
            
            # Calculate ATR
            high = df['high']
            low = df['low']
            close = df['close'].shift()
            tr1 = high - low
            tr2 = abs(high - close)
            tr3 = abs(low - close)
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean().iloc[-1]
            
            # Get current price
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                raise Exception(f"Failed to get tick data for {symbol}")
            current_price = tick.ask  # Using ask price for calculations
            
            # Calculate fixed stop loss distance based on ATR
            stop_distance = 2.0 * atr  # Fixed SL distance
            
            # Ensure minimum stop distance
            min_stop_distance = current_price * 0.001  # 0.1% minimum
            stop_distance = max(stop_distance, min_stop_distance)
            
            # Get current trading session
            session = self.analyze_session()
            
            # Calculate position size using risk manager
            position_size = self.risk_manager.calculate_position_size(
                account_balance=balance,
                risk_per_trade=self.trading_config["risk_per_trade"],
                stop_loss_pips=stop_distance,
                current_drawdown=current_drawdown,
                volatility_state=volatility_state,
                session=session
            )
            
            # Round to valid lot size using math.ceil to round upward
            import math
            min_lot = symbol_info.volume_min
            lot_step = symbol_info.volume_step
            position_size = math.ceil(position_size / lot_step) * lot_step
            
            # Ensure within limits
            position_size = max(min_lot, min(position_size, symbol_info.volume_max))
            
            logger.info(
                f"Position Size Calculation for {symbol}:\n"
                f"  Base Risk: {base_risk:.2f}\n"
                f"  Confidence Score: {signal_confidence:.2f}\n"
                f"  Volatility State: {volatility_state}\n"
                f"  Stop Distance: {stop_distance:.5f}\n"
                f"  ATR: {atr:.5f}\n"
                f"  Final Position Size: {position_size:.2f} lots"
            )
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0.0

    def calculate_drawdown(self) -> float:
        """Calculate current drawdown based on peak balance versus current balance."""
        try:
            account_info = mt5.account_info()
            if not account_info:
                logger.error("Failed to get account info for drawdown calculation")
                return 0.0
            
            current_balance = account_info.balance
            equity = account_info.equity
            
            # Calculate drawdown from peak balance
            peak_balance = max(current_balance, self.risk_manager.daily_stats.get('starting_balance', current_balance))
            absolute_drawdown = peak_balance - equity
            drawdown_percentage = (absolute_drawdown / peak_balance) if peak_balance > 0 else 0.0
            
            logger.debug(f"Drawdown calculation: Peak={peak_balance}, Current={equity}, DD%={drawdown_percentage*100:.2f}%")
            return drawdown_percentage
            
        except Exception as e:
            logger.error(f"Error calculating drawdown: {str(e)}")
            return 0.0

    def calculate_trade_levels(self, symbol, signal, poi_data=None):
        """Calculate dynamic entry, stop loss, and take profit levels based on market conditions."""
        try:
            # Get current price data
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                raise Exception(f"Failed to get tick data for {symbol}")

            # Get recent market data for ATR calculation
            df = self.mt5.get_market_data(symbol, "M5", 50)
            if df is None or df.empty:
                raise Exception(f"Failed to get market data for {symbol}")

            # Calculate ATR
            high = df['high']
            low = df['low']
            close = df['close']
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean().iloc[-1]

            # Get POI levels if available
            nearest_supply = None
            nearest_demand = None
            if poi_data:
                try:
                    nearest_supply = float(poi_data.get('resistance')) if poi_data.get('resistance') else None
                    nearest_demand = float(poi_data.get('support')) if poi_data.get('support') else None
                except (ValueError, TypeError):
                    logger.warning("Could not convert POI levels to float")

            # Base ATR multiplier for stop loss
            atr_multiplier = 1.0  # Reduced from 1.5 to make stops more realistic
            
            # Adjust ATR multiplier based on volatility
            volatility = self.calculate_volatility(symbol)
            if volatility > 50:  # High volatility
                atr_multiplier = 1.5
            elif volatility > 30:  # Medium volatility
                atr_multiplier = 1.25

            # Calculate base stop distance
            base_stop_distance = atr * atr_multiplier

            if signal == "BUY":
                entry = tick.ask
                
                # Calculate stop loss using ATR and POI levels
                atr_based_stop = entry - base_stop_distance
                poi_based_stop = nearest_demand if nearest_demand else atr_based_stop
                
                # Use the higher of ATR-based or POI-based stop (for buy orders, higher means closer to entry)
                stop_loss = max(atr_based_stop, poi_based_stop)
                
                # Ensure minimum stop distance
                min_stop_distance = entry * 0.001  # 0.1% minimum
                if entry - stop_loss < min_stop_distance:
                    stop_loss = entry - min_stop_distance

                # Calculate dynamic take profit based on nearest supply
                risk = entry - stop_loss
                if nearest_supply and nearest_supply > entry:
                    # Use supply level as first target if it gives at least 1:1 RR
                    if nearest_supply - entry >= risk:
                        take_profit = nearest_supply
                    else:
                        take_profit = entry + (risk * 1.5)  # Default 1:1.5 RR
                else:
                    take_profit = entry + (risk * 1.5)  # Default 1:1.5 RR
                
            else:  # SELL
                entry = tick.bid
                
                # Calculate stop loss using ATR and POI levels
                atr_based_stop = entry + base_stop_distance
                poi_based_stop = nearest_supply if nearest_supply else atr_based_stop
                
                # Use the lower of ATR-based or POI-based stop (for sell orders, lower means closer to entry)
                stop_loss = min(atr_based_stop, poi_based_stop)
                
                # Ensure minimum stop distance
                min_stop_distance = entry * 0.001  # 0.1% minimum
                if stop_loss - entry < min_stop_distance:
                    stop_loss = entry + min_stop_distance
                
                # Calculate dynamic take profit based on nearest demand
                risk = stop_loss - entry
                if nearest_demand and nearest_demand < entry:
                    # Use demand level as first target if it gives at least 1:1 RR
                    if entry - nearest_demand >= risk:
                        take_profit = nearest_demand
                    else:
                        take_profit = entry - (risk * 1.5)  # Default 1:1.5 RR
                else:
                    take_profit = entry - (risk * 1.5)  # Default 1:1.5 RR

            logger.info(
                f"Trade Levels for {symbol} {signal}:\n"
                f"  Entry: {entry:.5f}\n"
                f"  Stop Loss: {stop_loss:.5f} (Distance: {abs(entry - stop_loss):.5f})\n"
                f"  Take Profit: {take_profit:.5f} (RR: {abs(take_profit - entry) / abs(stop_loss - entry):.2f})\n"
                f"  ATR: {atr:.5f}, Multiplier: {atr_multiplier}\n"
                f"  POI Levels - Supply: {nearest_supply}, Demand: {nearest_demand}"
            )
            
            return entry, stop_loss, take_profit
            
        except Exception as e:
            logger.error(f"Error calculating trade levels: {str(e)}")
            return None, None, None

    def execute_trade(self, trade_params):
        """Execute trade on MT5 with partial take profits."""
        try:
            # Get base risk and take profit levels from risk manager
            risk = abs(trade_params['entry_price'] - trade_params['stop_loss'])
            base_volume = trade_params['position_size']
            orders = []
            
            # Use risk manager's partial take profit configuration
            for i, tp_level in enumerate(self.risk_manager.partial_tp_levels):
                # Calculate take profit price based on R-multiple
                if trade_params['signal_type'] == "BUY":
                    tp_price = trade_params['entry_price'] + (risk * tp_level['ratio'])
                else:  # SELL
                    tp_price = trade_params['entry_price'] - (risk * tp_level['ratio'])
                
                # Calculate volume for this partial
                partial_volume = base_volume * tp_level['size']
                if i == len(self.risk_manager.partial_tp_levels) - 1:
                    # Adjust last partial to account for any rounding errors
                    partial_volume = base_volume - sum(order['volume'] for order in orders)
                
                # Round volume to valid lot size
                symbol_info = mt5.symbol_info(trade_params['symbol'])
                lot_step = symbol_info.volume_step
                partial_volume = round(partial_volume / lot_step) * lot_step
                
                if partial_volume > 0:  # Only create order if volume is positive
                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": trade_params['symbol'],
                        "volume": partial_volume,
                        "type": mt5.ORDER_TYPE_BUY if trade_params['signal_type'] == 'BUY' else mt5.ORDER_TYPE_SELL,
                        "price": trade_params['entry_price'],
                        "sl": trade_params['stop_loss'],
                        "tp": tp_price,
                        "deviation": 10,
                        "magic": 234000,
                        "comment": f"Python Bot - {trade_params['signal_type']} TP{i+1} ({tp_level['ratio']:.1f}R)",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }
                    orders.append(request)
            
            # Ensure the symbol is selected before executing orders
            if not mt5.symbol_select(trade_params['symbol'], True):
                logger.error(f"Failed to select symbol: {trade_params['symbol']}")
                return None

            # Wait for fresh tick data up to 3 times
            retry_count = 0
            let_tick = mt5.symbol_info_tick(trade_params['symbol'])
            while not let_tick and retry_count < 3:
                tm.sleep(0.5)
                let_tick = mt5.symbol_info_tick(trade_params['symbol'])
                retry_count += 1
            if not let_tick:
                logger.error("No tick data available for trade execution after retrying")
                return None

            # Update orders with current tick prices before sending orders
            for order in orders:
                if order["type"] == mt5.ORDER_TYPE_BUY:
                    order["price"] = let_tick.ask
                else:
                    order["price"] = let_tick.bid
            
            # Execute all orders
            results = []
            for order in orders:
                result = mt5.order_send(order)
                if result.retcode == mt5.TRADE_RETCODE_REQUOTE:
                    logger.warning(f"Order requote detected: {result.comment}. Retrying with increased deviation.")
                    if "No prices" in result.comment:
                        tick = mt5.symbol_info_tick(order["symbol"])
                        if not tick:
                            logger.error("No tick data available to update order price")
                            raise Exception("No tick data available")
                        if order["type"] == mt5.ORDER_TYPE_BUY:
                            order["price"] = tick.ask
                        else:
                            order["price"] = tick.bid
                        logger.info(f"Updated order price to current market price: {order['price']}")
                    order["deviation"] += 10
                    result_retry = mt5.order_send(order)
                    if result_retry.retcode != mt5.TRADE_RETCODE_DONE:
                        raise Exception(f"Order failed after retry: {result_retry.comment}")
                    result = result_retry
                elif result.retcode != mt5.TRADE_RETCODE_DONE:
                    raise Exception(f"Order failed: {result.comment}")
                
                results.append(result)
            
            # Save trade details
            trades_data = []
            for i, (result, order) in enumerate(zip(results, orders)):
                trade_data = {
                    "ticket": result.order,
                    "symbol": trade_params['symbol'],
                    "type": trade_params['signal_type'],
                    "volume": order['volume'],
                    "entry_price": trade_params['entry_price'],
                    "stop_loss": trade_params['stop_loss'],
                    "take_profit": order['tp'],
                    "r_multiple": self.risk_manager.partial_tp_levels[i]['ratio'],
                    "partial_number": i + 1,
                    "open_time": datetime.now(UTC).isoformat(),
                    "profit": 0.0
                }
                trades_data.append(trade_data)
            
            self._save_trades_to_file(trades_data)
            
            logger.info(
                f"Successfully opened {len(results)} partial positions for {trade_params['symbol']} {trade_params['signal_type']}\n" +
                "\n".join([f"  Partial {i+1}: {order['volume']:.2f} lots, TP at {order['tp']:.5f} ({self.risk_manager.partial_tp_levels[i]['ratio']:.1f}R)"
                          for i, order in enumerate(orders)])
            )
            
            return [result.order for result in results]
            
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
            # Calculate pip multiplier dynamically using the symbol's point value
            sym_info = mt5.symbol_info(symbol)
            if not sym_info:
                logger.error(f"Failed to get symbol info for {symbol}")
                return float('inf')
            multiplier = 1 / sym_info.point
            
            # Get current symbol info using MT5 directly
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                logger.error(f"Failed to get tick data for {symbol}")
                return float('inf')
            
            spread = (tick.ask - tick.bid) * multiplier
            logger.debug(f"Current spread for {symbol}: {spread} pips")
            return spread
        except Exception as e:
            logger.error(f"Error getting spread: {str(e)}")
            return float('inf')

    async def process_signals(self, signals: List[Dict]) -> None:
        """Process trading signals with improved error handling and validation."""
        try:
            if not self.trading_enabled:
                logger.info("Trading is disabled, skipping signal processing")
                return

            logger.info(f"Processing {len(signals)} trading signals with trading enabled: ✅")
            
            if not signals:
                logger.warning("No signals to process")
                return
                
            # Log the full signals list for debugging
            logger.debug(f"Full signals list: {signals}")
            
            for signal in signals:
                try:
                    logger.info(f"Processing signal for {signal.get('symbol', 'UNKNOWN')}")
                    logger.debug(f"Signal details: {json.dumps(signal, default=str, indent=2)}")
                    
                    # Skip processing for HOLD signals
                    if signal.get('signal') == 'HOLD':
                        logger.info("Skipping HOLD signal")
                        continue
                    
                    # Validate signal structure
                    required_keys = [
                        'timeframe', 'direction', 'entry_price', 'stop_loss',
                        'take_profit', 'symbol', 'confidence'
                    ]
                    
                    # Check for missing keys
                    missing_keys = [k for k in required_keys if k not in signal]
                    if missing_keys:
                        logger.warning(f"Signal missing required keys: {missing_keys}")
                        logger.debug("Attempting to set default values...")
                        
                        # Set default values for missing keys
                        signal.setdefault('timeframe', 'M15')
                        signal.setdefault('direction', signal.get('signal', 'HOLD'))
                        
                        # Ensure we have a valid current price
                        current_price = signal.get('current_price')
                        if current_price is None or not isinstance(current_price, (int, float)):
                            logger.error("Invalid or missing current_price")
                        continue
                        
                        signal.setdefault('entry_price', current_price)
                        
                        if 'stop_loss' not in signal:
                            if signal.get('direction', '').lower() == 'buy':
                                signal['stop_loss'] = signal.get('support', current_price * 0.999)
                                logger.debug(f"Set default BUY stop_loss: {signal['stop_loss']}")
                            else:
                                signal['stop_loss'] = signal.get('resistance', current_price * 1.001)
                                logger.debug(f"Set default SELL stop_loss: {signal['stop_loss']}")
                                
                        if 'take_profit' not in signal:
                            if signal.get('direction', '').lower() == 'buy':
                                signal['take_profit'] = signal.get('resistance', current_price * 1.002)
                                logger.debug(f"Set default BUY take_profit: {signal['take_profit']}")
                            else:
                                signal['take_profit'] = signal.get('support', current_price * 0.998)
                                logger.debug(f"Set default SELL take_profit: {signal['take_profit']}")
                    
                    # Validate signal values
                    if not all(isinstance(signal.get(k), (int, float)) for k in ['entry_price', 'stop_loss', 'take_profit']):
                        logger.error("Invalid price values in signal")
                        logger.error(f"entry_price: {signal.get('entry_price')}, stop_loss: {signal.get('stop_loss')}, take_profit: {signal.get('take_profit')}")
                        continue
                    
                    # Validate signal direction
                    if signal['direction'].upper() not in ['BUY', 'SELL']:
                        logger.error(f"Invalid signal direction: {signal['direction']}")
                        continue
                    
                    # Calculate position size
                    position_size = self.calculate_position_size(signal['symbol'], signal['confidence'])
                    if position_size <= 0:
                        logger.warning(f"Invalid position size calculated: {position_size}")
                        continue
                    
                    logger.info(f"Executing trade for {signal['symbol']} - Direction: {signal['direction']}, Entry: {signal['entry_price']}, SL: {signal['stop_loss']}, TP: {signal['take_profit']}")
                    
                    # Execute the trade
                    trade_params = {
                        'symbol': signal['symbol'],
                        'signal_type': signal['direction'],
                        'entry_price': signal['entry_price'],
                        'stop_loss': signal['stop_loss'],
                        'take_profit': signal['take_profit'],
                        'position_size': position_size
                    }
                    
                    result = self.execute_trade(trade_params)
                    if result:
                        logger.info(f"Trade executed successfully: {result}")
                    else:
                        logger.error("Trade execution failed")
                    
                except Exception as e:
                    logger.error(f"Error processing signal: {str(e)}")
                    logger.error(f"Signal that caused error: {json.dumps(signal, default=str, indent=2)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    continue
                
        except Exception as e:
            logger.error(f"Error in process_signals: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            if self.telegram_bot:
                try:
                    await self.telegram_bot.send_error_alert(f"Error processing signals: {str(e)}")
                except Exception as telegram_error:
                    logger.error(f"Failed to send error alert: {str(telegram_error)}")

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
                            # Process each timeframe
                            for timeframe in self.trading_config["timeframes"]:
                                # Perform market analysis
                                analysis = await self.analyze_market(symbol, timeframe)
                                if analysis:
                                    # Generate signals based on analysis
                                    signals = await self.generate_signals(analysis, symbol, timeframe)
                                    if signals:
                                        # Process the signals
                                        await self.process_signals(signals)
                            
                            # Manage open trades for this symbol
                            await self.manage_open_trades()
                            
                        except Exception as e:
                            error_trace = traceback.format_exc()
                            logger.error(f"Error processing {symbol}: {str(e)}\nTraceback:\n{error_trace}")
                            if self.telegram_bot and self.telegram_bot.is_running:
                                await self.telegram_bot.send_error_alert(
                                    f"Error analyzing {symbol}: {str(e)}\nTraceback:\n{error_trace}"
                                )
                    
                    # Process signals
                    if signals:
                        await self.process_signals(signals)
                    
                    # Update dashboard
                    if self.dashboard:
                        current_session = self.analyze_session()
                        self.dashboard.update_status({
                            'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'active_symbols': [signal['symbol'] for signal in signals],
                            'session': current_session,
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

    def calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate technical indicators for momentum analysis with enhanced type safety."""
        try:
            indicators = {}
            
            # Calculate RSI with NaN handling
            delta = df['close'].diff().ffill()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean().fillna(0)
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean().fillna(0)
            rs = np.where(loss != 0, gain / loss, 1)  # Avoid division by zero
            indicators['rsi'] = float(100 - (100 / (1 + rs[-1])))
            
            # Calculate MACD with explicit type conversion
            exp1 = df['close'].ewm(span=12, adjust=False).mean().astype('float64')
            exp2 = df['close'].ewm(span=26, adjust=False).mean().astype('float64')
            macd_line = (exp1 - exp2).astype('float64')
            signal_line = macd_line.ewm(span=9, adjust=False).mean().astype('float64')
            
            indicators['macd'] = {
                'macd_line': float(macd_line.iloc[-1]),
                'signal_line': float(signal_line.iloc[-1])
            }
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            return {'rsi': 50, 'macd': {'macd_line': 0.0, 'signal_line': 0.0}}

    def check_momentum(self, analysis: Dict) -> bool:
        """Check momentum confirmation with type tracing"""
        try:
            # Add type validation trace
            logger.debug(f"Momentum check input types - Analysis: {type(analysis)}, "
                       f"Indicators: {type(analysis.get('indicators'))}")
            
            # Existing validation logic
            if not analysis or 'indicators' not in analysis:
                logger.error("Invalid analysis data structure")
                return False

            # Trace individual indicator types
            indicators = analysis['indicators']
            logger.debug(f"Indicator types - RSI: {type(indicators.get('rsi'))}, "
                       f"MACD: {type(indicators.get('macd'))}")
            
            # Safely get and validate RSI value
            rsi = indicators.get('rsi', 50)
            if not isinstance(rsi, (int, float)):
                logger.error(f"Invalid RSI type: {type(rsi)}")
                return False
            
            # Safely get and validate MACD values with type conversion
            macd = indicators.get('macd', {})
            macd_line = float(macd.get('macd_line', 0))
            signal_line = float(macd.get('signal_line', 0))

            # Log indicator values with types for debugging
            logger.debug(f"Momentum Check Types - RSI: {type(rsi)}, MACD: {type(macd_line)}, Signal: {type(signal_line)}")
            
            # Validate trend value
            if 'trend' not in analysis:
                logger.error("Trend information missing from analysis")
                return False
            
            trend = analysis['trend'].lower()
            
            if trend == "bullish":
                rsi_condition = rsi > 45
                macd_condition = macd_line > signal_line
                
                logger.debug("\nBullish Momentum Check:")
                logger.debug(f"RSI > 45: {'✅' if rsi_condition else '❌'} ({rsi:.2f})")
                logger.debug(f"MACD > Signal: {'✅' if macd_condition else '❌'} ({macd_line:.5f} vs {signal_line:.5f})")
                
                momentum_confirmed = rsi_condition and macd_condition
                logger.debug(f"Bullish Momentum Confirmed: {'✅' if momentum_confirmed else '❌'}")
                return momentum_confirmed
            
            elif trend == "bearish":
                rsi_condition = rsi < 55
                macd_condition = macd_line < signal_line
                
                logger.debug("\nBearish Momentum Check:")
                logger.debug(f"RSI < 55: {'✅' if rsi_condition else '❌'} ({rsi:.2f})")
                logger.debug(f"MACD < Signal: {'✅' if macd_condition else '❌'} ({macd_line:.5f} vs {signal_line:.5f})")
                
                momentum_confirmed = rsi_condition and macd_condition
                logger.debug(f"Bearish Momentum Confirmed: {'✅' if momentum_confirmed else '❌'}")
                return momentum_confirmed
            
            logger.debug("No clear trend for momentum check")
            return False
            
        except Exception as e:
            logger.error(f"Error checking momentum: {str(e)}")
            return False

    def close_trade(self, ticket):
        """Close a trade by its ticket number."""
        try:
            # Close the trade using MT5
            position = mt5.positions_get(ticket=ticket)
            if not position:
                logger.error(f"No position found with ticket {ticket}")
                return False
            
            position = position[0]
            close_price = mt5.symbol_info_tick(position.symbol).bid if position.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(position.symbol).ask
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": ticket,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "price": close_price,
                "deviation": 20,
                "magic": 100,
                "comment": "Close trade",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Failed to close trade {ticket}: {result.comment}")
                return False
            
            # Calculate profit
            profit = position.profit
            
            # Update profit history
            history_file = Path(BASE_DIR) / "data" / "profit_history.json"
            history_data = []
            if history_file.exists():
                with open(history_file, 'r') as f:
                    history_data = json.load(f)
            
            # Calculate cumulative profit correctly
            cumulative = sum(entry['profit'] for entry in history_data) + profit
            
            # Add trade closure entry with current timestamp
            history_data.append({
                'timestamp': datetime.now(UTC).isoformat(),
                'profit': profit,
                'trade_type': 'BUY' if position.type == mt5.ORDER_TYPE_BUY else 'SELL',
                'symbol': position.symbol,
                'cumulative': cumulative
            })
            
            # Save updated history
            with open(history_file, 'w') as f:
                json.dump(history_data, f, indent=4)
            
            # Update active trades file
            trades_file = Path(BASE_DIR) / "data" / "active_trades.json"
            if trades_file.exists():
                with open(trades_file, 'r') as f:
                    trades_data = json.load(f)
                
                # Remove closed trade
                trades_data = [t for t in trades_data if t['ticket'] != ticket]
                
                # Save updated trades
                with open(trades_file, 'w') as f:
                    json.dump(trades_data, f, indent=4)
            
            logger.info(f"Trade {ticket} closed successfully with profit: {profit}")
            return True
            
        except Exception as e:
            logger.error(f"Error closing trade {ticket}: {str(e)}")
            return False

    async def enable_trading(self):
        """Enable trading."""
        try:
            self.trading_enabled = True
            if hasattr(self, 'telegram_bot') and self.telegram_bot:
                await self.telegram_bot.enable_trading_core()
            logger.info("Trading enabled")
            return True
        except Exception as e:
            logger.error(f"Failed to enable trading: {str(e)}")
            self.trading_enabled = False
            return False

    async def disable_trading(self):
        """Disable trading."""
        try:
            self.trading_enabled = False
            if hasattr(self, 'telegram_bot') and self.telegram_bot:
                await self.telegram_bot.disable_trading_core()
            logger.info("Trading disabled")
            return True
        except Exception as e:
            logger.error(f"Failed to disable trading: {str(e)}")
            return False

    def _save_trades_to_file(self, new_trades: List[Dict]) -> None:
        """Save new trades to the active trades JSON file.

        This method appends new trades to the existing file.
        """
        try:
            trades_file = Path(BASE_DIR) / "data" / "active_trades.json"
            if trades_file.exists():
                with open(trades_file, "r") as f:
                    current_trades = json.load(f)
            else:
                current_trades = []
            current_trades.extend(new_trades)
            with open(trades_file, "w") as f:
                json.dump(current_trades, f, indent=4)
            logger.info(f"Saved {len(new_trades)} new trade(s) to active_trades.json")
        except Exception as e:
            logger.error(f"Error saving trades to file: {str(e)}")

    def get_min_stop_distance(self, symbol: str) -> Optional[float]:
        """Calculate and return the minimum stop distance for a symbol based on its current market conditions."""
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is not None:
                # If the symbol info has a stops_level, use it multiplied by point
                if hasattr(symbol_info, "stops_level") and symbol_info.stops_level > 0:
                    return symbol_info.stops_level * symbol_info.point
                # Fallback: use 0.1% of the current ask price
                tick = mt5.symbol_info_tick(symbol)
                if tick is not None:
                    return tick.ask * 0.001
            return None
        except Exception as e:
            logger.error(f"Error calculating min_stop_distance for {symbol}: {str(e)}")
            return None

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