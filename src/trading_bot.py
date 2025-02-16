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

from config.config import TRADING_CONFIG, SESSION_CONFIG, MT5_CONFIG, MARKET_STRUCTURE_CONFIG, SIGNAL_THRESHOLDS, CONFIRMATION_CONFIG, MARKET_SCHEDULE_CONFIG
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
        self.trades: List[Trade] = []
        self.signals: List[Signal] = []
        self.market_data: Dict[str, MarketData] = {}
        self.news_events: List[NewsEvent] = []
        self.trade_counter = 0
        self.ny_timezone = pytz.timezone('America/New_York')
        
        # Initialize analysis components with H4 timeframe thresholds as default
        self.market_analysis = MarketAnalysis(ob_threshold=MARKET_STRUCTURE_CONFIG['structure_levels']['H4']['ob_size'])
        self.smc_analysis = SMCAnalysis()
        self.mtf_analysis = MTFAnalysis()
        self.divergence_analysis = DivergenceAnalysis()
        self.volume_analysis = VolumeAnalysis()
    
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
            
            # First, ensure everything is cleaned up
            await self.stop(cleanup_only=True)
            
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
            
            self.running = True
            logger.info("Trading bot started successfully")
            
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
            if not positions:
                return
                
            self.risk_manager.update_open_trades(positions)
            
            for position in positions:
                symbol = position["symbol"]
                entry_price = position["price_open"]
                initial_stop = position["sl_initial"] if "sl_initial" in position else position["sl"]
                current_stop = position["sl"]
                position_size = position["volume"]
                trade_type = "BUY" if position["type"] == mt5.ORDER_TYPE_BUY else "SELL"
                
                # Calculate initial risk
                initial_risk = abs(entry_price - initial_stop)
                
                # Get current market data
                df = self.mt5.get_market_data(symbol, "M5")  # Use M5 for monitoring
                if df is None or df.empty:
                    continue
                
                current_price = df['close'].iloc[-1]
                current_tick = self.mt5.get_current_tick(symbol)
                if not current_tick:
                    continue
                
                # Calculate ATR if not present
                if 'atr' not in df.columns:
                    high = df['high']
                    low = df['low']
                    close = df['close']
                    tr1 = high - low
                    tr2 = abs(high - close.shift())
                    tr3 = abs(low - close.shift())
                    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                    df['atr'] = tr.rolling(window=14).mean()
                    df['atr'].fillna(method='bfill', inplace=True)
                
                current_atr = df['atr'].iloc[-1]
                
                # Update MAE and MFE
                if trade_type == "BUY":
                    adverse_exc = (entry_price - df['low'].min()) / entry_price * 100
                    favorable_exc = (df['high'].max() - entry_price) / entry_price * 100
                else:
                    adverse_exc = (df['high'].max() - entry_price) / entry_price * 100
                    favorable_exc = (entry_price - df['low'].min()) / entry_price * 100
                
                # Check for trailing stop adjustment
                should_adjust, new_stop = self.risk_manager.should_adjust_stops(
                    trade={
                        'entry_price': entry_price,
                        'initial_stop': initial_stop,
                        'stop_loss': current_stop,
                        'direction': trade_type,
                        'partial_take_profits': position.get('partial_take_profits', [])
                    },
                    current_price=current_price,
                    current_atr=current_atr
                )
                
                if should_adjust and new_stop != current_stop:
                    logger.info(f"Adjusting trailing stop for {symbol} from {current_stop:.5f} to {new_stop:.5f}")
                    if self.mt5.modify_position(position["ticket"], new_stop, position["tp"]):
                        await self.telegram_bot.send_management_alert(
                            position["ticket"],
                            symbol,
                            "Stop Loss Adjusted",
                            current_stop,
                            new_stop,
                            "Trailing Stop Update"
                        )
                        current_stop = new_stop
                
                # Check for stop loss hit
                if trade_type == "BUY" and current_tick.bid <= current_stop:
                    # Calculate R-multiple (negative for loss)
                    r_multiple = -(abs(entry_price - current_stop) / initial_risk)
                    pnl = (current_stop - entry_price) * position_size * -100000  # Negative for loss
                    
                    if self.mt5.close_position(position["ticket"]):
                        # Record profit in dashboard
                        if hasattr(self, 'dashboard'):
                            self.dashboard.add_profit_entry(
                                profit_amount=pnl,
                                trade_type=trade_type
                            )
                        
                        # Send telegram update
                        await self.telegram_bot.send_trade_update(
                            position["ticket"],
                            symbol,
                            "Trailing Stop" if current_stop > initial_stop else "Stop Loss",
                            current_tick.bid,
                            pnl,
                            r_multiple=r_multiple,
                            mae=adverse_exc,
                            mfe=favorable_exc
                        )
                    continue
                elif trade_type == "SELL" and current_tick.ask >= current_stop:
                    # Calculate R-multiple (negative for loss)
                    r_multiple = -(abs(current_stop - entry_price) / initial_risk)
                    pnl = (entry_price - current_stop) * position_size * -100000  # Negative for loss
                    
                    if self.mt5.close_position(position["ticket"]):
                        # Record profit in dashboard
                        if hasattr(self, 'dashboard'):
                            self.dashboard.add_profit_entry(
                                profit_amount=pnl,
                                trade_type=trade_type
                            )
                        
                        # Send telegram update
                        await self.telegram_bot.send_trade_update(
                            position["ticket"],
                            symbol,
                            "Trailing Stop" if current_stop < initial_stop else "Stop Loss",
                            current_tick.ask,
                            pnl,
                            r_multiple=r_multiple,
                            mae=adverse_exc,
                            mfe=favorable_exc
                        )
                    continue
                
                # Check for take profit hit
                if trade_type == "BUY" and current_tick.bid >= position["tp"]:
                    # Calculate R-multiple for profit
                    r_multiple = abs(position["tp"] - entry_price) / initial_risk
                    pnl = (position["tp"] - entry_price) * position_size * 100000
                    
                    if self.mt5.close_position(position["ticket"]):
                        await self.telegram_bot.send_trade_update(
                            position["ticket"],
                            symbol,
                            "Take Profit",
                            current_tick.bid,
                            pnl,
                            r_multiple=r_multiple,
                            mae=adverse_exc,
                            mfe=favorable_exc
                        )
                    continue
                elif trade_type == "SELL" and current_tick.ask <= position["tp"]:
                    # Calculate R-multiple for profit
                    r_multiple = abs(entry_price - position["tp"]) / initial_risk
                    pnl = (entry_price - position["tp"]) * position_size * 100000
                    
                    if self.mt5.close_position(position["ticket"]):
                        await self.telegram_bot.send_trade_update(
                            position["ticket"],
                            symbol,
                            "Take Profit",
                            current_tick.ask,
                            pnl,
                            r_multiple=r_multiple,
                            mae=adverse_exc,
                            mfe=favorable_exc
                        )
                    continue
                
                # Check if trade should be closed based on other conditions
                should_close, reason = self.risk_manager.should_close_trade(
                    position,
                    current_price,
                    {
                        "structure": df.get("structure", "NEUTRAL"),
                        "strength": df.get("strength", 0.0)
                    }
                )
                
                if should_close:
                    # Calculate final R-multiple and PnL
                    if trade_type == "BUY":
                        r_multiple = abs(current_price - entry_price) / initial_risk
                        pnl = (current_price - entry_price) * position_size * 100000
                    else:
                        r_multiple = abs(entry_price - current_price) / initial_risk
                        pnl = (entry_price - current_price) * position_size * 100000
                    
                    # Close position
                    if self.mt5.close_position(position["ticket"]):
                        await self.telegram_bot.send_trade_update(
                            position["ticket"],
                            symbol,
                            f"CLOSED ({reason})",
                            current_price,
                            pnl,
                            r_multiple=r_multiple,
                            mae=adverse_exc,
                            mfe=favorable_exc
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

    def analyze_trend(self, df: pd.DataFrame) -> str:
        """Analyze trend using moving averages.
        
        Returns:
            str: 'bullish', 'bearish', or 'neutral'
        """
        try:
            # Calculate moving averages
            df['MA20'] = df['close'].rolling(window=20).mean()
            df['MA50'] = df['close'].rolling(window=50).mean()
            
            current_ma20 = df['MA20'].iloc[-1]
            current_ma50 = df['MA50'].iloc[-1]
            
            # Calculate price momentum
            price_change = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]
            
            # Determine trend with momentum confirmation
            if current_ma20 > current_ma50 and price_change > 0:
                return 'bullish'
            elif current_ma20 < current_ma50 and price_change < 0:
                return 'bearish'
            else:
                return 'neutral'
                
        except Exception as e:
            logger.error(f"Error analyzing trend: {str(e)}")
            return 'neutral'

    async def analyze_market(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """Analyze market conditions with enhanced logging."""
        try:
            logger.info(f"🔍 Starting market analysis for {symbol} on {timeframe}")
            
            # Get market data
            data = await self.mt5.get_rates(symbol, timeframe)
            if data is None or len(data) < 100:  # Minimum required candles
                logger.error(f"❌ Insufficient data for {symbol} analysis")
                return None
                
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Log basic market stats
            current_price = df['close'].iloc[-1]
            daily_range = ((df['high'].max() - df['low'].min()) / df['close'].mean()) * 100
            logger.info(f"📊 Market Stats for {symbol}:")
            logger.info(f"    Current Price: {current_price:.5f}")
            logger.info(f"    Daily Range: {daily_range:.2f}%")
            logger.info(f"    Analyzing {len(df)} candles")
            
            # Perform trend analysis
            trend = self.analyze_trend(df)
            logger.info(f"📈 Trend Analysis: {trend}")
            
            # Perform comprehensive market analysis
            structure = self.market_analysis.analyze(df, symbol=symbol, timeframe=timeframe)
            logger.info("🏗️ Market Structure:")
            logger.info(f"    Structure Type: {structure.get('structure_type', 'Unknown')}")
            logger.info(f"    Key Levels: {structure.get('key_levels', [])}")
            
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
            
            # Detect points of interest
            pois = self.detect_pois(df)
            logger.info(f"🎯 Points of Interest detected: {len(pois)}")
            
            # Volume analysis
            volume_data = self.volume_analysis.analyze(df)
            logger.info(
                f"📊 Volume Analysis:\n"
                f"    Average Volume: {volume_data.get('avg_volume', 0):.2f}\n"
                f"    Volume Trend: {volume_data.get('volume_trend', 'Unknown')}"
            )
            
            # Combine all analyses
            analysis = {
                'symbol': symbol,
                'timeframe': timeframe,
                'trend': trend,
                'structure': structure,
                'pois': pois,
                'volume': volume_data,
                'mtf_analysis': mtf_analysis if higher_tf else None,
                'timestamp': datetime.now(UTC).isoformat()
            }
            
            logger.info(f"✅ Market analysis completed for {symbol} on {timeframe}")
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error in market analysis for {symbol} on {timeframe}: {str(e)}")
            logger.error("Detailed error trace:", exc_info=True)
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
        
        # Apply primary filters
        tf_alignment = await self.check_timeframe_alignment(analysis)
        session_check = self.check_session_conditions(analysis)
        
        logger.debug(f"Primary Filters: TF_Alignment={tf_alignment}, Session_Check={session_check}")
        
        if not (tf_alignment and session_check):
            logger.debug("Primary filters not met")
            return [{"signal": "HOLD", "confidence": 0}]
        
        # Calculate confirmation score using weights from config
        confirmation_score = 0
        confirmations_met = 0
        
        # Check SMT divergence
        smt_divergence = self.check_smt_divergence(analysis)
        if smt_divergence:
            confirmation_score += CONFIRMATION_CONFIG["weights"]["smt_divergence"] * 1.5
            confirmations_met += 1
        
        # Check liquidity
        liquidity = self.check_liquidity(analysis)
        if liquidity:
            confirmation_score += CONFIRMATION_CONFIG["weights"]["liquidity_sweep"] * 1.5
            confirmations_met += 1
        
        # Check momentum with adjusted multiplier
        momentum = self.check_momentum(analysis)
        if momentum:
            confirmation_score += CONFIRMATION_CONFIG["weights"]["momentum"] * 1.5
            confirmations_met += 1
            
        # Check pattern
        pattern = self.check_pattern(analysis)
        if pattern:
            confirmation_score += CONFIRMATION_CONFIG["weights"]["pattern"] * 1.5
            confirmations_met += 1
        
        logger.debug(f"Confirmation Score: {confirmation_score:.2f} (SMT={smt_divergence}, Liquidity={liquidity}, Momentum={momentum}, Pattern={pattern})")
        
        # Require at least 1 confirmation
        min_required = 1
        
        if confirmations_met < min_required:
            logger.debug(f"Insufficient confirmations: {confirmations_met}/{min_required}")
            return [{"signal": "HOLD", "confidence": 0}]
        
        # Calculate base confidence from confirmations using updated multiplier
        base_confidence = (confirmation_score / (min_required * 1.5)) * 100  # Scale to percentage
        
        # Generate signal based on trend and confirmation score
        if analysis['trend'] == "Bullish":
            if confirmation_score >= SIGNAL_THRESHOLDS["minimum"]:
                signal = "BUY"
                confidence = min(95, max(60, int(base_confidence)))  # Cap between 60-95%
            else:
                # Relaxed: if trend is bullish but confirmations insufficient, use default confidence
                signal = "BUY"
                confidence = 40
        elif analysis['trend'] == "Bearish":
            if confirmation_score >= SIGNAL_THRESHOLDS["minimum"]:
                signal = "SELL"
                confidence = min(95, max(60, int(base_confidence)))  # Cap between 60-95%
            else:
                # Relaxed: if trend is bearish but confirmations insufficient, use default confidence 
                signal = "SELL"
                confidence = 40
        else:
            signal = "HOLD"
            confidence = 0
        
        logger.info(f"Generated {signal} signal for {symbol} with {confidence}% confidence")
        
        # Add signal to dashboard
        add_signal({
            'session': analysis.get('session', self.analyze_session()),
            'symbol': symbol,
            'type': signal,
            'price': analysis['pois']['current_price'],
            'confidence': confidence,
            'trend': analysis['trend'],
            'confirmations': {
                'total': confirmations_met,
                'required': min_required,
                'score': confirmation_score,
                'details': {
                    'smt': smt_divergence,
                    'liquidity': liquidity,
                    'momentum': momentum,
                    'pattern': pattern
                }
            }
        })
        
        return [{
            "signal": signal,
            "confidence": confidence,
            "current_price": analysis["pois"]["current_price"],
            "support": analysis["pois"]["support"],
            "resistance": analysis["pois"]["resistance"],
            "trend": analysis["trend"],
            "session": analysis.get("session", self.analyze_session())
        }]

    async def check_timeframe_alignment(self, analysis):
        """Check timeframe alignment with enhanced logging."""
        try:
            if not analysis or 'trend' not in analysis:
                logger.error(f"❌ Invalid analysis data for timeframe alignment check: {analysis}")
                return False
                
            symbol = analysis['symbol']
            current_tf = analysis['timeframe']
            higher_tf = self.get_higher_timeframe(current_tf)
            
            logger.info(
                f"🔄 Checking timeframe alignment for {symbol}:\n"
                f"    Current TF: {current_tf}\n"
                f"    Higher TF: {higher_tf}"
            )
            
            if higher_tf:
                higher_tf_data = await self.mt5.get_rates(symbol, higher_tf)
                if higher_tf_data is not None:
                    higher_tf_trend = self.analyze_trend(pd.DataFrame(higher_tf_data))
                    current_trend = analysis['trend']
                    
                    alignment = higher_tf_trend == current_trend
                    logger.info(
                        f"📊 Timeframe Alignment Results:\n"
                        f"    Higher TF Trend: {higher_tf_trend}\n"
                        f"    Current TF Trend: {current_trend}\n"
                        f"    Aligned: {'✅' if alignment else '❌'}"
                    )
                    return alignment
                    
            logger.warning(f"⚠️ Could not check timeframe alignment for {symbol}")
            return False
            
        except Exception as e:
            logger.error(f"❌ Error checking timeframe alignment: {str(e)}")
            logger.exception("Detailed error trace:")
            return False

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
            
            # Check if symbol is allowed in current session
            allowed = self._is_symbol_allowed_in_session(symbol, current_session)
            
            # Get current volatility
            volatility = self.calculate_volatility(symbol)
            spread = self.get_spread(symbol)
            
            logger.info(
                f"📊 Session Conditions for {symbol}:\n"
                f"    Trading Allowed: {'✅' if allowed else '❌'}\n"
                f"    Volatility: {volatility:.2f}%\n"
                f"    Current Spread: {spread:.1f} pips"
            )
            
            # Check if conditions are favorable
            conditions_met = (
                allowed and
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
        """Check for Smart Money Concepts divergence."""
        try:
            # Get market data from MT5
            df = self.mt5.get_market_data(analysis['symbol'], analysis['timeframe'])
            if df is None or df.empty:
                logger.error(f"No data available for SMT divergence check")
                return False
            
            # Calculate RSI
            df['RSI'] = self.calculate_rsi(df['close'])
            
            # Get last few candles
            last_candles = df.tail(5)
            
            # Check for bullish divergence
            if analysis["trend"].lower() == "bullish":
                price_making_lower_low = last_candles['low'].iloc[-1] < last_candles['low'].min()
                rsi_making_higher_low = last_candles['RSI'].iloc[-1] > last_candles['RSI'].min()
                return price_making_lower_low and rsi_making_higher_low
                
            # Check for bearish divergence
            elif analysis["trend"].lower() == "bearish":
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
                threshold = 0.001  # Within 0.1%
            else:
                threshold = 0.0005  # Within 0.05% if no volume spike
                logger.debug("No volume spike detected - applying stricter proximity threshold.")
            
            # Get current price and key levels
            price = analysis["pois"]["current_price"]
            support = analysis["pois"]["support"]
            resistance = analysis["pois"]["resistance"]
            
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
                            'session': session,
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

    def check_momentum(self, analysis: dict) -> bool:
        """Check momentum confirmation using RSI and MACD."""
        try:
            # Get RSI and MACD values
            rsi = analysis.get('indicators', {}).get('rsi', 50)
            macd = analysis.get('indicators', {}).get('macd', {})
            macd_line = macd.get('macd_line', 0)
            signal_line = macd.get('signal_line', 0)
            
            # Relaxed momentum conditions
            if analysis['trend'] == "Bullish":
                return rsi > 45 and macd_line > signal_line  # Reduced from 50
            elif analysis['trend'] == "Bearish":
                return rsi < 55 and macd_line < signal_line  # Increased from 50
                
            return False
            
        except Exception as e:
            logger.error(f"Error checking momentum: {str(e)}")
            return False
            
    def check_pattern(self, analysis: dict) -> bool:
        """Check for confirming price patterns."""
        try:
            patterns = analysis.get('patterns', [])
            if not patterns:
                return False
                
            # Get the most recent patterns (check last 2 patterns)
            recent_patterns = patterns[-2:]
            
            # Check if any recent pattern confirms the trend
            if analysis['trend'] == "Bullish":
                bullish_patterns = ['bullish_engulfing', 'morning_star', 'hammer', 'piercing_line', 'three_white_soldiers']
                return any(p['type'] in bullish_patterns for p in recent_patterns)
            elif analysis['trend'] == "Bearish":
                bearish_patterns = ['bearish_engulfing', 'evening_star', 'shooting_star', 'dark_cloud_cover', 'three_black_crows']
                return any(p['type'] in bearish_patterns for p in recent_patterns)
                
            return False
            
        except Exception as e:
            logger.error(f"Error checking pattern: {str(e)}")
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