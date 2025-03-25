import asyncio
import json
import traceback
import pytz
import time
import sys
import MetaTrader5 as mt5  # Add MetaTrader5 import
from datetime import datetime, timedelta, UTC
from pathlib import Path
from typing import Dict, List, Any, Type, Optional
import os
import pandas as pd

from loguru import logger

# Import custom modules
from src.mt5_handler import MT5Handler
from src.signal_generators.signal_generator import SignalGenerator
from src.signal_generators.signal_generator1 import SignalGenerator1  # Import the second signal generator
from src.signal_generators.signal_generator2 import SignalGeneratorBankTrading as SignalGenerator2  # Add the new signal generator
from src.signal_generators.signal_generator3 import SignalGenerator3  # Add the new signal generator
from src.risk_manager import RiskManager
from src.telegram.telegram_bot import TelegramBot
from src.analysis.mtf_analysis import MTFAnalysis
from src.telegram.telegram_command_handler import TelegramCommandHandler
from src.utils.position_manager import PositionManager
from src.utils.signal_processor import SignalProcessor
from src.utils.performance_tracker import PerformanceTracker
from src.technical_indicators import TechnicalIndicators
from src.pattern_detector import PatternDetector


# Import configuration
from config.config import TRADING_CONFIG, SESSION_CONFIG, MARKET_SCHEDULE_CONFIG

BASE_DIR = Path(__file__).resolve().parent.parent

class TradingBot:
    def __init__(self, config=None, signal_generator_class: Type[SignalGenerator] = SignalGenerator):
        """
        Initialize the trading bot with configurable signal generator.
        
        Args:
            config: Optional configuration override
            signal_generator_class: Class to use for signal generation (defaults to SignalGenerator)
        """
        # Load configuration from file if not provided
        if config is None:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.json')
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = config
        
        # Set up logging
        self.setup_logging()
        
        # Set configuration attributes early to avoid AttributeError
        self.session_config = getattr(self.config, 'SESSION_CONFIG', SESSION_CONFIG)
        self.market_schedule = getattr(self.config, 'MARKET_SCHEDULE_CONFIG', MARKET_SCHEDULE_CONFIG)
        self.trading_config = getattr(self.config, 'TRADING_CONFIG', TRADING_CONFIG)
        
        # Initialize with a fresh MT5 connection
        # First make sure any existing connections are closed
        try:
            if hasattr(mt5, 'shutdown'):
                mt5.shutdown()  # type: ignore
                logger.debug("Cleaned up any existing MT5 connections before initialization")
        except Exception as e:
            # Ignore errors here, just being cautious
            logger.debug(f"Could not shut down MT5 connections: {str(e)}")
            pass
            
        # Create MT5 handler with fresh connection
        self.mt5_handler = MT5Handler()
        self.mt5 = self.mt5_handler  # Alias for backward compatibility
        
        # Verify MT5 connection is working
        if not self.mt5_handler.connected:
            if self.mt5_handler.initialize():
                logger.info("MT5 connection established during initialization")
        
        # Track connection status
        self.mt5_connected = self.mt5_handler.connected
        
        # Initialize risk manager with MetaTrader5 handler
        self.risk_manager = RiskManager(self.mt5_handler)
        # Use singleton instance
        self.telegram_bot = TelegramBot.get_instance()
        
        # Lazy-load market analysis to avoid circular imports
        self._market_analysis = None
        
        # Initialize MTF analysis
        self.mtf_analysis = MTFAnalysis()
        
        # Initialize telegram command handler
        self.telegram_command_handler = TelegramCommandHandler(self)
        
        # Initialize position manager
        self.position_manager = PositionManager(
            mt5_handler=self.mt5_handler,
            risk_manager=self.risk_manager,
            telegram_bot=self.telegram_bot,
            config=self.config
        )
        
        # Initialize signal processor
        self.signal_processor = SignalProcessor(
            mt5_handler=self.mt5_handler,
            risk_manager=self.risk_manager,
            telegram_bot=self.telegram_bot,
            config=self.config
        )
        
        # Initialize performance tracker
        self.performance_tracker = PerformanceTracker(
            mt5_handler=self.mt5_handler,
            config=self.config
        )
        
        # Initialize signal generators
        self.signal_generator_class = signal_generator_class
        self.signal_generator = None
        self.available_signal_generators = {}
        self._init_signal_generators(signal_generator_class)
        
        # Set state tracking variables
        self.trading_enabled = self.config.get('trading_enabled', False)
        self.close_positions_on_shutdown = self.config.get('close_positions_on_shutdown', True)
        self.allow_position_additions = self.config.get('allow_position_additions', False)
        self.use_trailing_stop = self.config.get('use_trailing_stop', True)
        self.stop_requested = False
        self.trading_symbols = self.config.get('trading_symbols', ['EURUSD', 'GBPUSD', 'USDJPY'])
        self.start_time = datetime.now()
        self.active_trades = {}
        self.pending_trades = {}
        
        # State management
        self.running = False
        self.trading_enabled = True  # Enabled by default as requested
        self.shutdown_requested = False  # Flag to gracefully exit the main loop
        self.signals: List[Dict] = []
        self.trade_counter = 0
        self.last_signal = {}  # Dictionary to track last signal timestamp and direction per symbol
        self.check_interval = 60  # Set check interval to exactly 60 seconds
        
        # Timezone handling
        self.ny_timezone = pytz.timezone('America/New_York')
        
        # Signal thresholds
        self.min_confidence = self.trading_config.get("min_confidence", 0.5)  # Default to 50% confidence
        
        # Trade management
        self.trailing_stop_enabled = True
        self.trailing_stop_data = {}  # Store trailing stop data for open positions
        
        # Shutdown behavior
        self.close_positions_on_shutdown = self.trading_config.get("close_positions_on_shutdown", False)  # Default to False
        
        # Central market data storage
        self.market_data_cache = {}
        self.analyzed_data_cache = {}
        self.last_market_data_update = datetime.min
        
        # Flag to track if startup notification has been sent
        self.startup_notification_sent = False

    @property
    def market_analysis(self):
        """Lazy load the MarketAnalysis to avoid circular imports."""
        if self._market_analysis is None:
            # Import MarketAnalysis here to avoid circular imports
            from src.analysis.market_analysis import MarketAnalysis
            self._market_analysis = MarketAnalysis()
        return self._market_analysis

    def _init_signal_generators(self, default_generator_class: Optional[Type[SignalGenerator]] = None):
        """Initialize signal generators with configuration."""
        # Use provided signal generator class with fallback to default
        generator_class = default_generator_class or SignalGenerator
        
        # Create shared component instances to reduce initialization overhead
        shared_indicators = TechnicalIndicators()
        shared_pattern_detector = PatternDetector(technical_indicators=shared_indicators)
        
        # Initialize market analysis with shared components
        if hasattr(self.market_analysis, 'initialize'):
            self.market_analysis.initialize(
                indicators=shared_indicators,
                pattern_detector=shared_pattern_detector
            )
        
        # Load each signal generator with MT5 handler and risk manager
        self.primary_signal_generator = generator_class(
            mt5_handler=self.mt5_handler,
            risk_manager=self.risk_manager,
        )
        
        # Create a dictionary of available signal generators
        self.available_signal_generators = {
            "default": SignalGenerator,
            "signal_generator": SignalGenerator,  # Add direct mapping for config.py
            "signal_generator1": SignalGenerator1,
            "signal_generator2": SignalGenerator2,
            "signal_generator3": SignalGenerator3
        }
        
        # Create signal generator instances
        signal_generator_instances = {
            "default": self.primary_signal_generator,
            "sg1": SignalGenerator1(
                mt5_handler=self.mt5_handler,
                risk_manager=self.risk_manager,
                indicators=shared_indicators,
                pattern_detector=shared_pattern_detector
            ),
            "sg2": SignalGenerator2(
                mt5_handler=self.mt5_handler,
                risk_manager=self.risk_manager,
                indicators=shared_indicators,
                pattern_detector=shared_pattern_detector
            ),
            "sg3": SignalGenerator3(
                mt5_handler=self.mt5_handler,
                risk_manager=self.risk_manager,
                indicators=shared_indicators,
                pattern_detector=shared_pattern_detector
            )
        }
        
        # Initialize signal generators list from configuration
        configured_generators = self.trading_config.get("signal_generators", ["default"])
        
        # Convert the list to a list of actual generator instances
        self.signal_generators = []
        for generator_name in configured_generators:
            if isinstance(generator_name, str):
                # Clean the name (remove whitespace, etc.)
                generator_name = generator_name.strip().lower()
                
                # Map configuration names to our generator instances
                if generator_name == "signal_generator1" or generator_name == "sg1":
                    self.signal_generators.append(signal_generator_instances["sg1"])
                    logger.info(f"Added SignalGenerator1 to active generators")
                elif generator_name == "signal_generator2" or generator_name == "sg2":
                    self.signal_generators.append(signal_generator_instances["sg2"])
                    logger.info(f"Added SignalGenerator2 to active generators")
                elif generator_name == "signal_generator3" or generator_name == "sg3":
                    self.signal_generators.append(signal_generator_instances["sg3"])
                    logger.info(f"Added SignalGenerator3 to active generators")
                elif generator_name == "signal_generator" or generator_name == "default":
                    self.signal_generators.append(signal_generator_instances["default"])
                    logger.info(f"Added default SignalGenerator to active generators")
                else:
                    logger.warning(f"Unknown signal generator '{generator_name}' in configuration")
            elif isinstance(generator_name, object) and hasattr(generator_name, "generate_signals"):
                # If it's already an instance with generate_signals method, add it directly
                self.signal_generators.append(generator_name)
                logger.info(f"Added {generator_name.__class__.__name__} instance to active generators")
        
        # Make sure we have at least one signal generator
        if not self.signal_generators:
            logger.warning("No valid signal generators found in configuration, using default")
            self.signal_generators.append(self.primary_signal_generator)
            
        logger.info(f"Initialized {len(self.signal_generators)} signal generators")
            
        # Set the active signal generator for legacy code
        self.active_signal_generator_name = "default"
        self.active_signal_generator = self.primary_signal_generator
        
        # Initialize other components with config
        if hasattr(self.risk_manager, 'initialize'):
            self.risk_manager.initialize(self.config)
        
        # Initialize market_analysis with lazy loading - avoid circular dependency
        if hasattr(self.market_analysis, 'initialize'):
            # Create a minimal config without MTF sections to avoid circular dependency
            market_analysis_config = {
                'MARKET_ANALYSIS_CONFIG': self.config.get('MARKET_ANALYSIS_CONFIG', {})
            }
            self.market_analysis.initialize(market_analysis_config)
            
        if hasattr(self.mtf_analysis, 'initialize'):
            self.mtf_analysis.initialize(self.config)

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
        logger.add(sys.stderr, format=fmt, level="DEBUG", colorize=True)

    def change_signal_generator(self, signal_generator_class: Type[SignalGenerator]):
        """
        Change the signal generator used by the trading bot.
        
        Args:
            signal_generator_class: New signal generator class to use
        """
        logger.info(f"Changing signal generator to {signal_generator_class.__name__}")
        
        # Store current MT5 connection state
        mt5_was_connected = False
        if hasattr(self, 'mt5_handler') and self.mt5_handler and self.mt5_handler.connected:
            mt5_was_connected = True
        
        # Create new signal generator instance
        self.signal_generator_class = signal_generator_class
        self.signal_generator = signal_generator_class(mt5_handler=self.mt5_handler, risk_manager=self.risk_manager)
        
        # Ensure MT5 connection is maintained or reestablished if it was connected before
        if mt5_was_connected:
            if not hasattr(self, 'mt5_handler') or not self.mt5_handler or not self.mt5_handler.connected:
                logger.warning("MT5 connection was lost during signal generator change. Attempting to reconnect...")
                self.mt5_handler = MT5Handler()
                if not self.initialize_mt5():
                    logger.error("Failed to reestablish MT5 connection after signal generator change")
                    # Attempt direct initialization as a fallback
                    try:
                        self.mt5_handler.initialize()
                        logger.info("MT5 connection reestablished through direct initialization")
                    except Exception as e:
                        logger.error(f"Failed to reestablish MT5 connection: {str(e)}")
        
        # Send notification via Telegram
        if self.telegram_bot and hasattr(self.telegram_bot, 'is_running') and self.telegram_bot.is_running:
            try:
                # Create a task to send notification asynchronously
                async def send_notification_task():
                    try:
                        # Check for null safety once more inside the task
                        if self.telegram_bot and hasattr(self.telegram_bot, 'send_notification'):
                            await self.telegram_bot.send_notification(
                                f"Signal generator changed to {signal_generator_class.__name__}"
                            )
                        else:
                            logger.warning("Cannot send notification: telegram_bot missing or send_notification not available")
                    except Exception as e:
                        logger.error(f"Error sending notification: {str(e)}")
                
                # Create and run the task in the background
                asyncio.create_task(send_notification_task())
            except Exception as e:
                logger.warning(f"Could not create notification task: {str(e)}")

    def initialize_mt5(self):
        """Initialize MT5 connection with robust error handling and recovery."""
        try:
            # Check if already connected
            if hasattr(self, 'mt5_handler') and self.mt5_handler and getattr(self.mt5_handler, 'connected', False):
                logger.debug("MT5 already connected")
                return True
            
            # Ensure we have a valid MT5Handler instance
            if not hasattr(self, 'mt5_handler') or self.mt5_handler is None:
                self.mt5_handler = MT5Handler()
            
            # Attempt standard initialization
            if self.mt5_handler.initialize():
                logger.info("MT5 connection initialized successfully")
                return True
            else:
                logger.error("Failed to initialize MT5 connection, attempting recovery")
                return self.recover_mt5_connection()
            
        except Exception as e:
            logger.error(f"Error initializing MT5: {str(e)}")
            logger.info("Attempting connection recovery due to initialization error")
            return self.recover_mt5_connection()

    async def start(self):
        """Start the trading bot and all its components."""
        logger.info("Starting trading bot...")
        
        # Create the shutdown future (used to signal when we should stop)
        shutdown_future = asyncio.Future()
        
        # Check MetaTrader5 connection
        if not self.mt5_handler.connected:
            logger.warning("MT5 not connected, attempting to initialize...")
            if not self.initialize_mt5():
                logger.error("Failed to initialize MT5 connection")
                shutdown_future.set_result(False)
                return shutdown_future
        
        # Start telegram bot
        try:
            logger.info("Starting Telegram bot...")
            if self.telegram_bot and hasattr(self.telegram_bot, 'initialize'):
                # Initialize the TelegramBot with our configuration
                await self.telegram_bot.initialize(self.config)
                logger.info("Telegram bot started")
            else:
                logger.error("Failed to initialize Telegram bot - missing initialize method")
        except Exception as e:
            logger.error(f"Error initializing Telegram bot: {str(e)}")
            logger.error(traceback.format_exc())
        
        # Register telegram commands
        try:
            await self.register_telegram_commands()
        except Exception as e:
            logger.error(f"Error registering Telegram commands: {str(e)}")
            logger.error(traceback.format_exc())
        
        try:
            # Initialize signal processor with current config
            await self.signal_processor.initialize(self.config)
            
            # Set references between components
            self.signal_processor.set_mt5_handler(self.mt5_handler)
            self.signal_processor.set_risk_manager(self.risk_manager)
            self.signal_processor.set_telegram_bot(self.telegram_bot)
            
            # Set the shutdown handler
            self.shutdown_future = shutdown_future
            
            # Check market schedule
            logger.info("Checking market schedule and timezone data")
            market_open = self.is_market_open()
            current_session = self.analyze_session()
            logger.info(f"Market open: {market_open}, Current session: {current_session}")
            
            # Start main loop as a background task
            self.running = True
            self.main_loop_task = asyncio.create_task(self.main_loop())
            
            # Start the trades monitoring loop
            self.monitor_task = asyncio.create_task(self._monitor_trades_loop())
            
            # Start the shutdown monitor if we have a shutdown future
            self.shutdown_monitor_task = asyncio.create_task(self._monitor_shutdown())
            
            logger.info("Trading bot started successfully")
            
            # Update Telegram with bot status
            if self.telegram_bot and hasattr(self.telegram_bot, 'is_running') and self.telegram_bot.is_running and not self.startup_notification_sent:
                try:
                    # Send startup notification
                    await self.telegram_bot.send_notification(
                        f"""🤖 <b>Trading Bot Started</b>

The trading bot has been successfully started and is now {'' if self.trading_enabled else 'NOT '}ready to trade.
- Active signal generators: {', '.join([sg.__class__.__name__ for sg in self.signal_generators])}
- Trading is currently {'ENABLED' if self.trading_enabled else 'DISABLED'}
- Markets {'OPEN' if market_open else 'CLOSED'}
- Current session: {current_session}

Use /status to check bot status
Use /help to see all commands"""
                    )
                    # Set flag to prevent sending this notification again
                    self.startup_notification_sent = True
                except Exception as e:
                    logger.error(f"Error sending startup notification: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error starting trading bot: {str(e)}")
            logger.error(traceback.format_exc())
            shutdown_future.set_result(False)
        
        return shutdown_future

    async def _monitor_shutdown(self):
        """Monitor for shutdown condition and complete the shutdown future when needed."""
        try:
            while self.running and not self.shutdown_requested:
                # Check if main loop has exited
                if hasattr(self, 'main_loop_task') and self.main_loop_task.done():
                    # If main loop exited with an error, log it
                    if self.main_loop_task.exception():
                        logger.error(f"Main loop exited with an error: {self.main_loop_task.exception()}")
                        logger.error(traceback.format_exc())
                    else:
                        logger.info("Main loop completed normally")
                    
                    # Set shutdown_requested flag
                    self.shutdown_requested = True
                    break
                
                # Check if monitor task has exited
                if hasattr(self, 'monitor_task') and self.monitor_task.done():
                    # If monitor task exited with an error, log it
                    if self.monitor_task.exception():
                        logger.error(f"Monitor task exited with an error: {self.monitor_task.exception()}")
                        logger.error(traceback.format_exc())
                    else:
                        logger.info("Monitor task completed normally")
                    
                    # Set shutdown_requested flag
                    self.shutdown_requested = True
                    break
                
                # Check every second for shutdown conditions
                await asyncio.sleep(1)
            
            # Set the future's result to indicate completion
            if not self.shutdown_future.done():
                self.shutdown_future.set_result(True)
                
        except Exception as e:
            logger.error(f"Error in shutdown monitor: {str(e)}")
            logger.error(traceback.format_exc())
            if not self.shutdown_future.done():
                self.shutdown_future.set_exception(e)

    async def register_telegram_commands(self):
        """Register custom command handlers with the Telegram bot."""
        if not self.telegram_bot:
            logger.warning("Telegram bot not available, skipping command registration")
            return
            
        if not hasattr(self.telegram_bot, 'is_running') or not self.telegram_bot.is_running:
            logger.warning("Telegram bot not running, skipping command registration")
            return
            
        # Use the telegram command handler to register all commands
        try:
            await self.telegram_command_handler.register_all_commands(self.telegram_bot)
        except Exception as e:
            logger.error(f"Error registering telegram commands: {str(e)}")
            # Continue despite errors

    async def handle_list_signal_generators_command(self, args):
        """
        Handle command to list available signal generators.
        Format: /listsignalgenerators
        """
        generators = list(self.available_signal_generators.keys())
        current_generator = self.signal_generator_class.__name__
        
        message = f"Current signal generator: {current_generator}\n\nAvailable signal generators:\n"
        for gen in generators:
            message += f"- {gen}\n"
        
        return message

    async def handle_set_signal_generator_command(self, args):
        """
        Handle command to change signal generator.
        Format: /setsignalgenerator <generator_name>
        """
        if not args:
            return "Please specify a signal generator name. Use /listsignalgenerators to see available options."
            
        generator_name = args[0].lower()
        
        if generator_name in self.available_signal_generators:
            generator_class = self.available_signal_generators[generator_name]
            self.change_signal_generator(generator_class)
            return f"Signal generator set to {generator_name}"
        else:
            available_generators = ", ".join(self.available_signal_generators.keys())
            return f"Unknown signal generator: {generator_name}\nAvailable options: {available_generators}"

    async def handle_enable_trailing_stop_command(self, args):
        """Handle command to enable trailing stop loss."""
        self.trailing_stop_enabled = True
        return "Trailing stop loss enabled"
        
    async def handle_disable_trailing_stop_command(self, args):
        """Handle command to disable trailing stop loss."""
        self.trailing_stop_enabled = False
        return "Trailing stop loss disabled"

    async def handle_status_command(self, args):
        """
        Handle command to show current trading bot status.
        Format: /status
        """
        # Get account info
        account_info = self.mt5_handler.get_account_info()
        
        # Get open positions
        positions = self.mt5_handler.get_open_positions()
        
        # Determine current session
        current_session = self.analyze_session()
        
        # Build status message
        status = f"Trading Bot Status\n{'='*20}\n"
        status += f"Trading Enabled: {'✅' if self.trading_enabled else '❌'}\n"
        status += f"Trailing Stop: {'✅' if self.trailing_stop_enabled else '❌'}\n"
        status += f"Position Additions: {'✅' if self.trading_config.get('allow_position_additions', False) else '❌'}\n"
        status += f"Close Positions on Shutdown: {'✅' if self.close_positions_on_shutdown else '❌'}\n"
        status += f"Current Session: {current_session}\n"
        status += f"Signal Generator: {self.signal_generator_class.__name__}\n\n"

            
        # Account info
        if account_info:
            status += f"Account Balance: {account_info.get('balance', 'N/A')}\n"
            status += f"Account Equity: {account_info.get('equity', 'N/A')}\n"
            status += f"Free Margin: {account_info.get('free_margin', 'N/A')}\n\n"
        
        # Position summary
        status += f"Open Positions: {len(positions)}\n"
        if positions:
            total_profit = sum(pos["profit"] for pos in positions)
            status += f"Total Floating P/L: {total_profit}\n\n"
            
            # List first 5 positions
            status += "Recent Positions:\n"
            for pos in positions[:5]:
                pos_type = "BUY" if pos["type"] == 0 else "SELL"
                status += f"- {pos['symbol']} {pos_type}: {pos['profit']}\n"
            
            if len(positions) > 5:
                status += f"...and {len(positions) - 5} more\n"
        
        return status

    async def main_loop(self):
        """Main trading loop that handles signal generation and processing"""
        try:
            logger.info("Starting main trading loop...")
            # Set interval to exactly 60 seconds (1 minute)
            interval = 60  # Force cycle to run every minute regardless of config
            
            # Make sure startup_notification_sent is set to True to prevent re-sending
            if not self.startup_notification_sent:
                self.startup_notification_sent = True
            
            while self.running and not self.shutdown_requested:
                start_time = time.time()
                
                try:
                    # Check if market is open
                    market_open = self.is_market_open()
                    
                    if not market_open:
                        logger.info("Markets are closed, skipping signal generation")
                        await asyncio.sleep(interval)
                        continue
                    
                    # Only proceed if trading is enabled
                    if not self.trading_enabled:
                        logger.info("Trading is disabled, skipping signal generation")
                        await asyncio.sleep(interval)
                        continue
                    
                    # Check MT5 connection and reconnect if needed
                    if not self.mt5_handler.connected:
                        logger.warning("MT5 connection lost, attempting to reconnect...")
                        if not self.recover_mt5_connection():
                            logger.error("Failed to recover MT5 connection")
                            await asyncio.sleep(interval)
                            continue
                    
                    # Fetch latest market data
                    market_data = await self.fetch_market_data()
                    
                    # Analyze market data
                    analyzed_data = await self.analyze_market_data(market_data)
                    
                    # Generate trading signals
                    all_signals = await self.generate_trading_signals(analyzed_data)
                    
                    # Process signals if any were generated
                    if all_signals:
                        logger.info(f"Processing {len(all_signals)} trading signals")
                        await self.process_signals(all_signals)
                    else:
                        logger.info("No trading signals generated")
                    
                    # Update performance metrics
                    await self.update_performance_metrics()
                    
                    # Manage open trades
                    await self.manage_open_trades()
                
                except Exception as e:
                    logger.error(f"Error in main trading loop: {str(e)}")
                    logger.error(traceback.format_exc())
                
                # Calculate how long to sleep
                elapsed = time.time() - start_time
                sleep_time = max(0, interval - elapsed)
                
                if sleep_time > 0:
                    logger.debug(f"Sleeping for {sleep_time:.2f} seconds until next cycle")
                    await asyncio.sleep(sleep_time)
                else:
                    logger.warning(f"Processing took longer than interval: {elapsed:.2f}s > {interval}s")
                    # Sleep a minimum amount to prevent CPU overload
                    await asyncio.sleep(1)
            
            logger.info("Main trading loop exiting due to stop request")
        
        except asyncio.CancelledError:
            logger.info("Main trading loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Fatal error in main trading loop: {str(e)}")
            logger.error(traceback.format_exc())
            # Signal to shut down
            self.shutdown_requested = True
            # Signal the shutdown future if it exists
            if hasattr(self, 'shutdown_future') and not self.shutdown_future.done():
                self.shutdown_future.set_result(False)

    async def _monitor_trades_loop(self):
        """Separate loop for monitoring trades more frequently."""
        logger.info("Starting trade monitoring loop")
        
        while self.running and not self.shutdown_requested:
            try:
                # Check for active positions first
                active_positions = self.mt5_handler.get_open_positions()
                
                if not active_positions:
                    # No open positions, no need to check market status or manage trades
                    logger.debug("No active positions to monitor, sleeping for 5 minutes")
                    await asyncio.sleep(300)  # Sleep for 5 minutes when no positions
                    continue
                
                # Get unique symbols from active positions
                active_symbols = set(pos.get("symbol") for pos in active_positions if pos.get("symbol"))
                
                # Check if any of the markets for the active positions are open
                markets_open = False
                for symbol in active_symbols:
                    if self.is_market_open(symbol):
                        markets_open = True
                        break
                
                if not markets_open:
                    logger.debug("Markets are closed for all active positions. Monitoring paused.")
                    await asyncio.sleep(300)  # Check every 5 minutes during closed markets
                    continue
                
                # Markets are open for at least one active position, manage trades
                await self.position_manager.manage_open_trades()
                
                # Sleep for a shorter interval (5 seconds) to monitor trades more frequently
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Error in trade monitoring loop: {str(e)}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(1)  # Brief sleep on error before retrying

    def is_market_open(self, symbol: Optional[str] = None) -> bool:
        """
        Check if the market is currently open based on schedule.
        
        Args:
            symbol: Optional symbol to check. If provided, used to determine if it's a crypto symbol.
        """
        # Ensure symbol is a valid string when passed to market_analysis
        if symbol is None and hasattr(self, 'trading_symbols') and self.trading_symbols:
            # Use the first trading symbol as default if none specified
            symbol = self.trading_symbols[0]
            
        # Use the implementation from MarketAnalysis, with null safety
        if hasattr(self, 'market_analysis') and self.market_analysis is not None:
            return self.market_analysis.is_market_open(symbol)
        return False  # Default to closed if market_analysis is not available

    def analyze_session(self) -> str:
        """Determine the current trading session (Asian, London, NY)."""
        # Use the implementation from MarketAnalysis
        return self.market_analysis.analyze_session()

    async def process_signals(self, signals: List[Dict]) -> None:
        """Process trading signals and execute trades as needed."""
        try:
            if not self.trading_enabled:
                logger.info("Trading is disabled, skipping signal processing")
                return

            logger.info(f"Processing {len(signals)} trading signals with trading enabled: ✅")
            
            # Ensure signal_processor has the correct telegram_bot instance
            if hasattr(self, 'signal_processor') and self.signal_processor:
                # Make sure telegram_bot is set in signal_processor without reinitializing it
                if self.telegram_bot and hasattr(self.telegram_bot, 'is_running'):
                    # Only set the Telegram bot, don't try to initialize it again
                    self.signal_processor.set_telegram_bot(self.telegram_bot)
                    logger.debug("Updated SignalProcessor with current TelegramBot instance")
                
                # Process the signals
                await self.signal_processor.process_signals(signals)
            else:
                logger.error("No signal processor available to process signals")
                
        except Exception as e:
            logger.error(f"Error in process_signals: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            if self.telegram_bot and hasattr(self.telegram_bot, 'send_error_alert'):
                try:
                    await self.telegram_bot.send_error_alert(f"Error processing signals: {str(e)}")
                except Exception as telegram_error:
                    logger.error(f"Failed to send error alert: {str(telegram_error)}")

    async def execute_trade_from_signal(self, signal: Dict, is_addition: bool = False) -> None:
        """Execute a trade based on signal, using existing mt5_handler functionality and storing results in database."""
        # Delegate to signal processor
        await self.signal_processor.execute_trade_from_signal(signal, is_addition)

    async def handle_signal_with_existing_positions(self, signal: Dict, existing_positions: List[Dict]) -> None:
        """Process a signal for a symbol that already has open positions."""
        # Delegate to signal processor
        await self.signal_processor.handle_signal_with_existing_positions(signal, existing_positions)

    async def manage_open_trades(self) -> None:
        """
        Manage all currently open trading positions.
        
        Delegates to position_manager for actual implementation.
        """
        await self.position_manager.manage_open_trades()

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

    async def close_pending_trades(self):
        """
        Close all pending trades.
        
        Delegates to position_manager for actual implementation.
        
        Returns:
            Tuple of (success_count, failed_count)
        """
        return await self.position_manager.close_pending_trades()

    async def handle_enable_close_on_shutdown_command(self, args):
        """
        Handle command to enable closing positions on shutdown.
        Format: /enablecloseonshutdown
        """
        self.close_positions_on_shutdown = True
        logger.info("Enabled automatic closing of positions on shutdown")
        return "✅ Automatic closing of positions on shutdown is now ENABLED"
        
    async def handle_disable_close_on_shutdown_command(self, args):
        """
        Handle command to disable closing positions on shutdown.
        Format: /disablecloseonshutdown
        """
        self.close_positions_on_shutdown = False
        logger.info("Disabled automatic closing of positions on shutdown")
        return "✅ Automatic closing of positions on shutdown is now DISABLED"

    

    def _get_position_type(self, position: Dict[str, Any]) -> str:
        """Get the position type ('BUY' or 'SELL') from a position object. Delegate to signal processor."""
        return self.signal_processor._get_position_type(position)
  
            
    def _calculate_activation_price(self, direction: str, entry_price: float, stop_loss: float) -> float:
        """Calculate trailing stop activation price."""
        risk = abs(entry_price - stop_loss)
        activation_factor = self.trading_config.get("trailing_activation_factor", 1.0)
        return entry_price + (risk * activation_factor if direction == 'BUY' else -risk * activation_factor)
        
    async def _notify_trade_action(self, message: str) -> None:
        """Send notification about trade action to the telegram chat. Delegate to signal processor."""
        await self.signal_processor._notify_trade_action(message)

    async def request_shutdown(self):
        """Request a graceful shutdown of the trading bot."""
        logger.info("Shutdown requested - will exit after current cycle completes")
        self.shutdown_requested = True
        
        # Send notification if Telegram is available
        if self.telegram_bot and hasattr(self.telegram_bot, 'is_running') and self.telegram_bot.is_running:
            await self.telegram_bot.send_notification("⚠️ Trading bot shutdown requested. Will exit soon.")
        
        # Set up a task to force shutdown after timeout if main loop doesn't exit gracefully
        async def force_shutdown():
            try:
                # Wait for 2 minutes for graceful shutdown
                await asyncio.sleep(120)
                # If we're still running after timeout, force shutdown
                if self.running:
                    logger.warning("Graceful shutdown timeout - forcing shutdown")
                    self.running = False
                    if self.telegram_bot and hasattr(self.telegram_bot, 'is_running') and self.telegram_bot.is_running:
                        await self.telegram_bot.send_notification("⚠️ Forcing trading bot shutdown after timeout")
            except Exception as e:
                logger.error(f"Error in force shutdown task: {str(e)}")
                
        # Start force shutdown task
        asyncio.create_task(force_shutdown())
        
        return True

    async def handle_shutdown_command(self, args):
        """
        Handle command to gracefully shutdown the trading bot.
        Format: /shutdown
        """
        await self.request_shutdown()
        return "⚠️ Trading bot shutdown initiated. The bot will exit after completing the current cycle."
        
    async def handle_enable_position_additions_command(self, args):
        """
        Handle command to enable adding to positions.
        Format: /enablepositionadditions
        """
        self.allow_position_additions = True
        logger.info("Enabled adding to positions")
        return "✅ Adding to positions is now ENABLED"
        
    async def handle_disable_position_additions_command(self, args):
        """
        Handle command to disable adding to positions.
        Format: /disablepositionadditions
        """
        self.allow_position_additions = False
        logger.info("Disabled adding to positions")
        return "✅ Adding to positions is now DISABLED"
        
    async def handle_start_dashboard_command(self, args):
        """
        Handle command to start the trading dashboard.
        Format: /startdashboard
        
        Note: This is a placeholder implementation. In a real implementation,
        you would start a web dashboard or other UI.
        """
        logger.info("Dashboard start requested (not implemented)")
        return "⚠️ Dashboard functionality is not yet implemented"

    def recover_mt5_connection(self, max_attempts=3):
        """
        Attempt to recover the MT5 connection after a failure.
        
        Args:
            max_attempts: Maximum number of reconnection attempts
            
        Returns:
            bool: True if connection was recovered, False otherwise
        """
        logger.info("Attempting to recover MT5 connection")
        
        # Create a new MT5Handler if needed
        if not hasattr(self, 'mt5_handler') or self.mt5_handler is None:
            self.mt5_handler = MT5Handler()
        
        # Track if we've made progress
        connection_established = False
        
        for attempt in range(1, max_attempts + 1):
            logger.info(f"MT5 reconnection attempt {attempt}/{max_attempts}")
            
            try:
                # Try to shut down any existing connections first
                try:
                    if hasattr(mt5, 'shutdown'):
                        mt5.shutdown()  # type: ignore
                        time.sleep(1)  # Give it time to clean up
                except Exception as ex:
                    logger.debug(f"Error during MT5 shutdown in recovery: {str(ex)}")
                    logger.debug(traceback.format_exc())
                    # Continue despite shutdown errors
                
                # Create a new MT5 handler on the second attempt or if first attempt fails
                if attempt > 1 or not connection_established:
                    logger.info("Creating a fresh MT5Handler instance for clean reconnection")
                    self.mt5_handler = MT5Handler()
                
                # Attempt to initialize
                if self.mt5_handler and hasattr(self.mt5_handler, 'initialize'):
                    if self.mt5_handler.initialize():
                        logger.info("MT5 connection recovered successfully")
                        
                        # Verify connection by doing a simple query
                        try:
                            # Try a simple account query to confirm the connection
                            if hasattr(self.mt5_handler, 'get_account_info'):
                                account_info = self.mt5_handler.get_account_info()
                                if account_info:
                                    logger.info(f"MT5 connection verified with account: {account_info.get('login', 'unknown')}")
                                    # Update mt5_connected status
                                    self.mt5_connected = True
                                    return True
                                else:
                                    logger.warning("MT5 initialized but could not verify connection with account query")
                                    if attempt < max_attempts:
                                        connection_established = True  # We made progress
                                        continue
                        except Exception as e:
                            logger.warning(f"MT5 initialized but verification failed: {str(e)}")
                            logger.warning(traceback.format_exc())
                            if attempt < max_attempts:
                                connection_established = True  # We made progress
                                continue
                
                # Wait before next attempt with increasing backoff
                wait_time = 2 * attempt
                logger.info(f"Waiting {wait_time} seconds before next reconnection attempt")
                time.sleep(wait_time)
                
            except Exception as e:
                logger.error(f"Error during MT5 reconnection attempt {attempt}: {str(e)}")
                logger.error(traceback.format_exc())
                time.sleep(2 * attempt)  # Increasing backoff
        
        logger.error(f"Failed to recover MT5 connection after {max_attempts} attempts")
        self.mt5_connected = False
        return False

    async def reconcile_trades(self) -> None:
        """
        Reconcile local trade records with MT5 platform data.
        
        Delegates to position_manager for actual implementation.
        """
        await self.position_manager.reconcile_trades()

    async def update_performance_metrics(self) -> Dict[str, Any]:
        """Update performance metrics based on trading history."""
        # Delegate to PerformanceTracker
        metrics = await self.performance_tracker.update_performance_metrics()
        
        # Also update the TelegramBot metrics for backward compatibility
        if self.telegram_bot:
            self.telegram_bot.performance_metrics = metrics.copy()
            
        return metrics
        
    async def initialize_performance_tracker(self) -> None:
        """
        Initialize the performance tracker and update metrics.
        This should be called during startup to ensure metrics are ready.
        """
        try:
            logger.info("Initializing performance tracker...")
            
            # Ensure MT5 handler is set correctly (in case it was reset)
            if hasattr(self, "mt5_handler") and self.mt5_handler:
                self.performance_tracker.set_mt5_handler(self.mt5_handler)
                
                # Fetch and update metrics
                metrics = await self.performance_tracker.update_performance_metrics()
                
                # Update TelegramBot metrics for backward compatibility
                if self.telegram_bot:
                    # Initialize with zeros if no metrics yet
                    if not metrics or metrics.get("total_trades", 0) == 0:
                        self.telegram_bot.performance_metrics = {
                            'total_trades': 0,
                            'winning_trades': 0,
                            'losing_trades': 0,
                            'total_profit': 0.0,
                            'max_drawdown': 0.0,
                            'win_rate': 0.0
                        }
                    else:
                        # Convert from performance tracker format to telegram bot format
                        self.telegram_bot.performance_metrics = {
                            'total_trades': metrics.get("total_trades", 0),
                            'winning_trades': metrics.get("winning_trades", 0),
                            'losing_trades': metrics.get("losing_trades", 0),
                            'total_profit': metrics.get("total_profit", 0.0),
                            'max_drawdown': metrics.get("max_drawdown", 0.0),
                            'win_rate': metrics.get("win_rate", 0.0) * 100  # Convert to percentage
                        }
                        
                    logger.info("Telegram bot performance metrics initialized")
                
                logger.info("Performance tracker initialization completed successfully")
            else:
                logger.warning("MT5 handler not available, performance metrics initialization skipped")
                
        except Exception as e:
            logger.error(f"Error initializing performance tracker: {str(e)}")
            logger.error(traceback.format_exc())

    async def fetch_market_data(self):
        """
        Centralized method to fetch market data for all configured symbols.
        
        Returns:
            dict: All market data organized by symbol and timeframe
        """
        logger.info("Starting centralized market data fetching")
        
        # Initialize market data storage
        market_data = {}
        
        # Track which symbols were successfully processed
        processed_symbols = []
        
        # Get current session for analysis
        current_session = self.analyze_session()
        logger.info(f"Current trading session: {current_session}")
        
        # Calculate number of candles needed for 3 days of history
        # Higher resolution timeframes need more candles to cover 3 days
        candle_counts = {
            "M1": 4320,  # 3 days × 24 hours × 60 minutes
            "M5": 864,   # 3 days × 24 hours × 12 candles per hour
            "M15": 288,  # 3 days × 24 hours × 4 candles per hour
            "M30": 144,  # 3 days × 24 hours × 2 candles per hour
            "H1": 72,    # 3 days × 24 hours
            "H4": 18,    # 3 days × 6 candles per day
            "D1": 10,    # A few more than 3 for safety
        }
        
        # Process symbols from configuration
        for symbol_config in self.trading_config["symbols"]:
            # Parse symbol configuration
            if isinstance(symbol_config, dict):
                symbol = symbol_config.get("symbol", "").strip()
                timeframe = symbol_config.get("timeframe", "H1")
                additional_timeframes = symbol_config.get("additional_timeframes", [])
            else:
                symbol = symbol_config.strip()
                timeframe = "H1"  # Default timeframe
                additional_timeframes = []
            
            # Skip if symbol is not defined
            if not symbol:
                continue
                
            # Check if market is open for this symbol
            if not self.is_market_open(symbol):
                logger.debug(f"Market is closed for {symbol}. Skipping data fetch.")
                continue
            
            logger.info(f"Fetching market data for {symbol}")
            
            # Initialize symbol data
            symbol_data = {}
            fetch_success = False
            
            # Determine candle count for primary timeframe
            num_candles = candle_counts.get(timeframe, 1000)
            
            # Fetch primary timeframe data
            logger.debug(f"Fetching primary timeframe data for {symbol} on {timeframe} ({num_candles} candles)")
            primary_data = await self.mt5_handler.get_rates(symbol, timeframe, num_candles)
            
            if primary_data is None or len(primary_data) < 100:
                logger.warning(f"Failed to get sufficient data for {symbol} on {timeframe}")
                continue
            
            logger.debug(f"Successfully fetched {len(primary_data)} candles for {symbol} on {timeframe}")
            symbol_data[timeframe] = primary_data
            fetch_success = True
            
            # Fetch additional timeframes
            for add_tf in additional_timeframes:
                # Determine candle count for additional timeframe
                add_tf_candles = candle_counts.get(add_tf, 1000)
                
                logger.debug(f"Fetching additional timeframe data for {symbol} on {add_tf} ({add_tf_candles} candles)")
                
                # Log the symbol's availability status before fetching
                if hasattr(self.mt5_handler, 'is_symbol_available'):
                    is_available = self.mt5_handler.is_symbol_available(symbol)
                    logger.debug(f"Symbol {symbol} availability status: {is_available}")
                
                # Record the start time to measure fetch duration
                start_time = datetime.now()
                add_data = await self.mt5_handler.get_rates(symbol, add_tf, add_tf_candles)
                fetch_duration = (datetime.now() - start_time).total_seconds()
                
                if add_data is not None:
                    if len(add_data) >= 100:
                        logger.debug(f"Successfully fetched {len(add_data)} candles for {symbol} on {add_tf} (took {fetch_duration:.2f}s)")
                        symbol_data[add_tf] = add_data
                    else:
                        logger.warning(f"Could not fetch sufficient data for {symbol} on {add_tf}")
                        logger.warning(f"Only received {len(add_data)}/{add_tf_candles} candles for {symbol} on {add_tf}")
                        # Log the date range of available data to help diagnose
                        if len(add_data) > 0:
                            oldest_date = add_data.index[0]
                            newest_date = add_data.index[-1]
                            logger.debug(f"Available data range for {symbol} on {add_tf}: {oldest_date} to {newest_date}")
                        
                        # Check if this might be a new symbol with limited history
                        if hasattr(self.mt5_handler, 'get_symbol_info'):
                            symbol_info = self.mt5_handler.get_symbol_info(symbol)
                            if symbol_info and hasattr(symbol_info, 'time'):
                                symbol_addition_time = datetime.fromtimestamp(symbol_info.time)
                                logger.debug(f"Symbol {symbol} was added to the terminal on {symbol_addition_time}")
                else:
                    logger.warning(f"Could not fetch sufficient data for {symbol} on {add_tf}")
                    logger.warning(f"No data returned after {fetch_duration:.2f}s for {symbol} on {add_tf}")
                    
                    # Log the MT5 error if available
                    if hasattr(self.mt5_handler, 'get_last_error'):
                        error = self.mt5_handler.get_last_error()
                        if error:
                            logger.warning(f"MT5 error for {symbol} on {add_tf}: {error}")
            
            # Add to market data if we got at least the primary timeframe
            if fetch_success:
                market_data[symbol] = symbol_data
                processed_symbols.append(symbol)
        
        # Add metadata about the current session
        market_data['current_session'] = current_session
        market_data['fetch_time'] = datetime.now()
        market_data['processed_symbols'] = processed_symbols
        
        # Update the cache
        self.market_data_cache = market_data
        self.last_market_data_update = datetime.now()
        
        logger.info(f"Completed centralized market data fetching for {len(processed_symbols)} symbols")
        return market_data

    async def analyze_market_data(self, market_data):
        """
        Perform centralized analysis on fetched market data.
        
        Args:
            market_data: The market data to analyze
            
        Returns:
            dict: Analyzed market data with indicators and patterns
        """
        logger.info("Starting centralized market data analysis")
        
        # Initialize storage for analyzed data with correct structure
        analyzed_data = {
            'symbols': {},
            'fetch_time': None,
            'analysis_time': None,
            'current_session': None
        }
        
        # Add metadata from market data - with explicit type safety
        if market_data.get('current_session') is not None:
            analyzed_data['current_session'] = market_data.get('current_session')
        
        # Convert datetime objects to ISO format strings - with explicit type safety
        if 'fetch_time' in market_data:
            if isinstance(market_data['fetch_time'], datetime):
                analyzed_data['fetch_time'] = market_data['fetch_time'].isoformat()
            else:
                analyzed_data['fetch_time'] = str(market_data['fetch_time'])
        
        # Always use string for analysis_time
        analyzed_data['analysis_time'] = datetime.now().isoformat()
        
        # Get list of symbols to process (excluding metadata keys)
        symbols_to_process = [
            symbol for symbol in market_data.keys() 
            if symbol not in ['current_session', 'fetch_time', 'processed_symbols', 'analysis_time']
        ]
        
        # Process each symbol
        for symbol in symbols_to_process:
            symbol_data = market_data[symbol]
            analyzed_symbol_data = {}
            
            # Process each timeframe
            for timeframe, candles in symbol_data.items():
                # Skip if not enough data
                if len(candles) < 100:
                    logger.warning(f"Not enough data to analyze {symbol} on {timeframe}")
                    continue
                
                # Perform analysis using MTF Analysis
                try:
                    # Use mtf_analysis for comprehensive analysis
                    analysis_result = self.mtf_analysis.analyze(
                        data=candles,
                        timeframe=timeframe
                    )
                    
                    # Use market_analysis for additional analysis if available
                    if hasattr(self.market_analysis, 'analyze_market_data'):
                        market_analysis_result = await self.market_analysis.analyze_market_data(
                            symbol=symbol,
                            timeframe=timeframe,
                            candles=candles
                        )
                        
                        # Merge results if market analysis was successful
                        if market_analysis_result:
                            if not analysis_result:
                                analysis_result = market_analysis_result
                            else:
                                # Combine the two analysis results
                                for key, value in market_analysis_result.items():
                                    if key not in analysis_result:
                                        analysis_result[key] = value
                    
                    # Store the analysis result
                    analyzed_symbol_data[timeframe] = analysis_result
                    
                except Exception as e:
                    logger.error(f"Error analyzing {symbol} on {timeframe}: {str(e)}")
                    logger.error(traceback.format_exc())
            
            # Add analyzed data for this symbol
            if analyzed_symbol_data:
                analyzed_data['symbols'][symbol] = analyzed_symbol_data
        
        # Update the analysis cache
        self.analyzed_data_cache = analyzed_data
        
        logger.info(f"Completed market data analysis for {len(analyzed_data['symbols'])} symbols")
        return analyzed_data

    async def generate_trading_signals(self, analyzed_data):
        """
        Generate trading signals using all configured signal generators with the analyzed data.
        
        Args:
            analyzed_data: Pre-analyzed market data
            
        Returns:
            list: Trading signals from all signal generators
        """
        logger.info("Starting signal generation with analyzed data")
        
        all_signals = []
        
        # Get account info for risk management
        account_info = self.mt5_handler.get_account_info()
        
        # Skip if no symbols were analyzed
        if not analyzed_data or 'symbols' not in analyzed_data or not analyzed_data['symbols']:
            logger.warning("No analyzed data available for signal generation")
            return all_signals
        
        # Get current session from analyzed data
        current_session = analyzed_data.get('current_session')
        
        # Process each symbol
        for symbol, symbol_data in analyzed_data['symbols'].items():
            # Skip if no timeframes were analyzed
            if not symbol_data:
                continue
            
            # Get the primary timeframe (first one in the symbol data)
            timeframes = list(symbol_data.keys())
            if not timeframes:
                continue
                
            primary_timeframe = timeframes[0]
            
            # Format data for signal generators
            # Create a structure that resembles the original market data format
            formatted_market_data = {
                symbol: {}
            }
            
            # Add original candle data from market_data_cache
            if symbol in self.market_data_cache:
                # Copy raw price data for each timeframe
                for tf, candles in self.market_data_cache[symbol].items():
                    formatted_market_data[symbol][tf] = candles
            
            # Make sure the analyzed data is properly formatted as a dictionary structure
            formatted_analyzed_data = {}
            for timeframe, analysis in symbol_data.items():
                # First, handle if analysis is a list
                if isinstance(analysis, list):
                    logger.warning(f"Analysis for {symbol} {timeframe} is a list, converting to dictionary")
                    formatted_analyzed_data[timeframe] = {'data': analysis}
                    
                    # Try to extract key_levels from the list items to make a dictionary
                    key_levels = {}
                    for item in analysis:
                        if isinstance(item, dict) and 'key_levels' in item:
                            if isinstance(item['key_levels'], dict):
                                key_levels.update(item['key_levels'])
                            # Handle case where key_levels itself might be a list
                            elif isinstance(item['key_levels'], list):
                                for level in item['key_levels']:
                                    if isinstance(level, dict):
                                        key_levels.update(level)
                    
                    # If we couldn't extract key_levels, create default ones
                    if not key_levels:
                        # Try to get market data for this symbol/timeframe
                        candles = None
                        if symbol in self.market_data_cache and timeframe in self.market_data_cache[symbol]:
                            candles = self.market_data_cache[symbol][timeframe]
                            
                        # Create default key levels from market data if available
                        if isinstance(candles, pd.DataFrame) and not candles.empty:
                            key_levels = {
                                'bsl': float(candles['high'].max()),
                                'ssl': float(candles['low'].min())
                            }
                        else:
                            # Fallback if no market data
                            key_levels = {'bsl': 0, 'ssl': 0}
                    
                    formatted_analyzed_data[timeframe]['key_levels'] = key_levels
                else:
                    # Regular dictionary analysis
                    formatted_analyzed_data[timeframe] = analysis
                    
                    # Make sure key_levels exists and is a dictionary
                    if isinstance(analysis, dict):
                        # Ensure the timeframe entry is itself a dictionary
                        if not isinstance(formatted_analyzed_data.get(timeframe), dict):
                            formatted_analyzed_data[timeframe] = {}
                            
                        if 'key_levels' not in analysis:
                            candles = None
                            if symbol in self.market_data_cache and timeframe in self.market_data_cache[symbol]:
                                candles = self.market_data_cache[symbol][timeframe]
                                
                            # Create default key levels
                            if isinstance(candles, pd.DataFrame) and not candles.empty:
                                try:
                                    formatted_analyzed_data[timeframe]['key_levels'] = {
                                        'bsl': float(candles['high'].max()),
                                        'ssl': float(candles['low'].min())
                                    }
                                except (KeyError, TypeError) as e:
                                    logger.warning(f"Error creating key levels for {symbol} {timeframe}: {str(e)}")
                                    formatted_analyzed_data[timeframe]['key_levels'] = {'bsl': 0, 'ssl': 0}
                        elif not isinstance(analysis['key_levels'], dict):
                            # Convert non-dict key_levels to a dictionary
                            logger.warning(f"key_levels is not a dictionary for {symbol} {timeframe}")
                            try:
                                formatted_analyzed_data[timeframe]['key_levels'] = {
                                    'bsl': 0, 'ssl': 0
                                }
                            except (KeyError, TypeError) as e:
                                logger.warning(f"Error assigning key_levels for {symbol} {timeframe}: {str(e)}")
                                # If assignment fails, create a new dictionary with key_levels
                                formatted_analyzed_data[timeframe] = {'key_levels': {'bsl': 0, 'ssl': 0}}
            
            # Run each signal generator
            for i, generator in enumerate(self.signal_generators):
                if not hasattr(generator, 'generate_signals'):
                    logger.error(f"Invalid signal generator #{i+1}: missing generate_signals method")
                    continue
                
                logger.debug(f"Running signal generator #{i+1} ({generator.__class__.__name__}) for {symbol}")
                
                try:
                    # Call the generator with the formatted data
                    if asyncio.iscoroutinefunction(generator.generate_signals):
                        result = await generator.generate_signals(
                            market_data=formatted_market_data,
                            symbol=symbol,
                            timeframe=primary_timeframe,
                            account_info=account_info,
                            analyzed_data=formatted_analyzed_data  # Pass the properly formatted analyzed data
                        )
                    else:
                        result = generator.generate_signals(
                            market_data=formatted_market_data,
                            symbol=symbol,
                            timeframe=primary_timeframe,
                            account_info=account_info,
                            analyzed_data=formatted_analyzed_data  # Pass the properly formatted analyzed data
                        )
                    
                    # Handle both old and new result formats
                    if isinstance(result, dict) and "signals" in result:
                        signals = result["signals"]
                    else:
                        signals = result
                    
                    # Process results
                    gen_name = generator.__class__.__name__
                    if signals and len(signals) > 0:
                        logger.info(f"Generator {gen_name} produced {len(signals)} signals for {symbol}")
                        
                        # Tag signals with generator name
                        for signal in signals:
                            signal['generator'] = gen_name
                            
                        all_signals.extend(signals)
                    else:
                        logger.debug(f"Generator {gen_name} produced no signals for {symbol}")
                        
                except Exception as e:
                    logger.error(f"Error in signal generator {generator.__class__.__name__}: {str(e)}")
                    logger.error(traceback.format_exc())
        
        logger.info(f"Generated a total of {len(all_signals)} signals from all generators")
        return all_signals

    async def stop(self, cleanup_only=False):
        """Stop the trading bot and clean up resources"""
        logger.info("Bot stop method called")
        logger.debug(f"Stop called with cleanup_only={cleanup_only}, call stack: {traceback.format_stack()}")
        self.running = False
        
        # Cancel our monitoring tasks if they're still running
        if hasattr(self, 'main_loop_task') and not self.main_loop_task.done():
            self.main_loop_task.cancel()
            
        if hasattr(self, 'monitor_task') and not self.monitor_task.done():
            self.monitor_task.cancel()
            
        if hasattr(self, 'shutdown_monitor_task') and not self.shutdown_monitor_task.done():
            self.shutdown_monitor_task.cancel()
        
        # Stop the telegram bot if it's running
        if self.telegram_bot is not None and not cleanup_only:
            try:
                # Make sure telegram_bot has stop method
                if hasattr(self.telegram_bot, 'stop'):
                    await self.telegram_bot.stop()
                self.telegram_bot = None
            except Exception as e:
                logger.warning(f"Error stopping Telegram bot: {str(e)}")
                logger.warning(traceback.format_exc())
        
        # Stop other subsystems
        try:
            # Only close pending trades if configured to do so
            if self.close_positions_on_shutdown:
                logger.info("Closing positions on shutdown (enabled in config)")
                if hasattr(self, 'position_manager') and self.position_manager:
                    await self.position_manager.close_pending_trades()
            else:
                logger.info("Keeping positions open on shutdown (as configured)")
            
            # Get account balance BEFORE closing MT5 connection
            try:
                if hasattr(self, 'mt5_handler') and self.mt5_handler:
                    # First verify connection is active
                    if self.mt5_handler.is_connected():
                        account_info = self.mt5_handler.get_account_info()
                        if account_info and 'balance' in account_info:
                            final_balance = account_info.get('balance', 0)
                            logger.info(f"Final account balance: ${final_balance:.2f}")
                        else:
                            logger.warning("Account info is None or missing balance information")
                            logger.warning(f"Account info: {account_info}")
                    else:
                        logger.warning("MT5 not connected when trying to get final account balance")
                        # Try to reconnect
                        if self.recover_mt5_connection(max_attempts=1):
                            account_info = self.mt5_handler.get_account_info()
                            if account_info and 'balance' in account_info:
                                final_balance = account_info.get('balance', 0)
                                logger.info(f"Final account balance (after reconnect): ${final_balance:.2f}")
            except Exception as e:
                logger.warning(f"Unable to retrieve final account balance: {str(e)}")
                logger.warning(traceback.format_exc())
            
            # Shutdown MT5 connection if doing a full shutdown
            if not cleanup_only and hasattr(self, 'mt5_handler') and self.mt5_handler is not None:
                try:
                    logger.info("Shutting down MT5 connection...")
                    if hasattr(self.mt5_handler, 'shutdown'):
                        self.mt5_handler.shutdown()
                    logger.info("MT5 connection closed")
                except Exception as e:
                    logger.warning(f"Error shutting down MT5: {str(e)}")
                    logger.warning(traceback.format_exc())
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            logger.error(traceback.format_exc())
        
        # If shutdown future exists and is not done, complete it
        if hasattr(self, 'shutdown_future') and not self.shutdown_future.done():
            self.shutdown_future.set_result(True)
            
        logger.info("Trading bot stopped")

    async def initialize(self):
        """Initialize trading bot components."""
        try:
            logger.info("Starting Trading Bot initialization...")
            
            # Initialize bot components
            if not hasattr(self, 'mt5_handler') or not self.mt5_handler:
                logger.info("Initializing MT5 handler...")
                self.mt5_handler = MT5Handler()
                self.mt5_handler.initialize()
                self.mt5_connected = self.mt5_handler.connected
                
                # Set handlers in other components
                if hasattr(self, 'risk_manager'):
                    self.risk_manager.set_mt5_handler(self.mt5_handler)
                if hasattr(self, 'position_manager'):
                    self.position_manager.set_mt5_handler(self.mt5_handler)
                if hasattr(self, 'signal_processor'):
                    self.signal_processor.set_mt5_handler(self.mt5_handler)
                    
            # Initialize the performance tracker
            await self.initialize_performance_tracker()
                    
            # Initialize signal generator
            try:
                logger.info("Initializing signal generator...")
                
                # Close the signal generator if it exists
                if self.signal_generator:
                    try:
                        await self.signal_generator.close()
                    except Exception as sg_e:
                        logger.error(f"Error closing existing signal generator: {str(sg_e)}")
                
                # Create a new signal generator
                self.signal_generator = self.signal_generator_class(
                    mt5_handler=self.mt5_handler,
                    tbot=self.telegram_bot,
                    trade_processor=self.signal_processor,
                    telegram_bot=self.telegram_bot
                )
                
                # Initialize the signal generator
                await self.signal_generator.initialize()
                logger.info(f"Signal generator initialized: {self.signal_generator.__class__.__name__}")
            except Exception as e:
                logger.error(f"Error initializing signal generator: {str(e)}")
                logger.error(traceback.format_exc())
                
            # Initialize Telegram bot if not already done
            if self.telegram_bot and not self.telegram_bot.is_running:
                logger.info("Initializing Telegram bot...")
                await self.telegram_bot.initialize(self.config)
                
                # Register command handlers
                await self.register_telegram_commands()
                
            logger.info("Trading Bot initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during trading bot initialization: {str(e)}")
            logger.error(traceback.format_exc())
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
        retention="1 day",
        compression="zip",
        level="INFO"
    )
    
    # Start the bot
    bot = TradingBot()
    asyncio.run(bot.start()) 