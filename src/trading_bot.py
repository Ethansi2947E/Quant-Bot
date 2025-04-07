import asyncio
import traceback
import pytz
import time
import MetaTrader5 as mt5  # Add MetaTrader5 import
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Type, Optional
import pandas as pd
import numpy as np

from loguru import logger

# Import custom modules
from src.mt5_handler import MT5Handler
from src.risk_manager import RiskManager
from src.telegram.telegram_bot import TelegramBot
from src.telegram.telegram_command_handler import TelegramCommandHandler
from src.utils.position_manager import PositionManager
from src.utils.signal_processor import SignalProcessor
from src.utils.performance_tracker import PerformanceTracker
from src.utils.data_manager import DataManager


# Define a base SignalGenerator class if it doesn't exist elsewhere
class SignalGenerator:
    """Base SignalGenerator class that all signal generators should extend."""
    
    def __init__(self, mt5_handler=None, risk_manager=None, **kwargs):
        """
        Initialize the signal generator.
        
        Args:
            mt5_handler: MT5Handler instance
            risk_manager: RiskManager instance
            **kwargs: Additional keyword arguments
        """
        self.mt5_handler = mt5_handler
        self.risk_manager = risk_manager
        self.name = self.__class__.__name__
        
        # Timeframe configuration
        self.required_timeframes = []  # Subclasses must override
        self.primary_timeframe = None  # Subclasses must override
        
    async def initialize(self):
        """
        Initialize the signal generator with any necessary setup.
        Override in subclasses for specific initialization.
        """
        logger.debug(f"Base initialization for {self.name}")
        return True
        
    async def generate_signals(self, market_data=None, symbol=None, timeframe=None):
        """
        Generate trading signals based on market data.
        Override in subclasses with actual signal generation logic.
        
        Args:
            market_data: Dictionary of market data by symbol and timeframe
            symbol: Symbol to generate signals for
            timeframe: Timeframe to generate signals for
            
        Returns:
            list: List of signal dictionaries
        """
        logger.warning(f"generate_signals called on base SignalGenerator class for {symbol}/{timeframe}")
        return []
        
    async def close(self):
        """
        Clean up resources when the signal generator is no longer needed.
        Override in subclasses if specific cleanup is required.
        """
        logger.debug(f"Closing signal generator: {self.name}")
        return True

BASE_DIR = Path(__file__).resolve().parent.parent

class TradingBot:
    def __init__(self, config: Optional[Dict] = None, signal_generator_class: Optional[Type] = None):
        """
        Initialize the trading bot with configuration.
        
        Args:
            config: Configuration dictionary (optional)
            signal_generator_class: Signal generator class to use (optional)
        """
        # Store configuration dictionary
        self.config = config or {}
        
        # Load and merge with default config if not provided
        if not self.config:
            from config.config import TRADING_CONFIG, TELEGRAM_CONFIG, MT5_CONFIG, SESSION_CONFIG
            
            self.config = {
                "trading": TRADING_CONFIG,
                "telegram": TELEGRAM_CONFIG,
                "mt5": MT5_CONFIG,
                "session": SESSION_CONFIG,
            }
        
        # Extract commonly used config sections
        self.trading_config = self.config.get("trading", {})
        self.telegram_config = self.config.get("telegram", {})
        self.mt5_config = self.config.get("mt5", {})
        self.market_schedule = self.config.get("market_schedule", {})
        
        # Initialize Market Hours checker
        try:
            from src.utils.market_hours import MarketHours
            from src.config.market_schedule import MARKET_SCHEDULE_CONFIG
            self.market_hours = MarketHours(config=MARKET_SCHEDULE_CONFIG)
            logger.info("Market hours checker initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize market hours checker: {str(e)}")
            self.market_hours = None
        
        # Initialize market status tracking
        self.market_status = {}  # Track market open/closed status for each symbol
        
        # Initialize MT5 handler first (needed by other components)
        self.mt5_handler = MT5Handler()
        
        # Verify MT5 connection is working
        if not self.mt5_handler.connected:
            if self.mt5_handler.initialize():
                logger.info("MT5 connection established during initialization")
        
        # Track connection status
        self.mt5_connected = self.mt5_handler.connected
        
        # Initialize symbols list and state tracking variables
        self.symbols = []
        self.trading_symbols = []
        
        # Load symbols from configuration
        self._load_symbols_from_config()
        
        # Initialize risk manager
        self.risk_manager = RiskManager(
            mt5_handler=self.mt5_handler
        )
        
        # Initialize data manager
        self.data_manager = DataManager(
            mt5_handler=self.mt5_handler,
            config=self.config
        )
        
        # Apply enhanced data management settings if available
        data_management_config = self.trading_config.get("data_management", {})
        if data_management_config:
            # Set data manager direct fetch settings
            if hasattr(self.data_manager, "use_direct_fetch"):
                self.data_manager.use_direct_fetch = data_management_config.get("use_direct_fetch", True)
            
            # Set real-time bars count
            if hasattr(self.data_manager, "real_time_bars_count"):
                self.data_manager.real_time_bars_count = data_management_config.get("real_time_bars_count", 10)
                
        # Use singleton instance
        self.telegram_bot = TelegramBot.get_instance()
        
        # Lazy-load market analysis to avoid circular imports
        self._market_analysis = None
        
        # Initialize MTF analysis
        
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
        
        # Apply enhanced signal validation settings if available
        if data_management_config:
            # Configure signal processor validation settings
            if hasattr(self.signal_processor, "validate_before_execution"):
                self.signal_processor.validate_before_execution = data_management_config.get("validate_trades", True)
            
            if hasattr(self.signal_processor, "price_validation_tolerance"):
                self.signal_processor.price_validation_tolerance = data_management_config.get("price_tolerance", 0.0003)
                
            if hasattr(self.signal_processor, "tick_delay_tolerance"):
                self.signal_processor.tick_delay_tolerance = data_management_config.get("tick_delay_tolerance", 2.0)
        
        # Initialize performance tracker
        self.performance_tracker = PerformanceTracker(
            mt5_handler=self.mt5_handler,
            config=self.config
        )
        
        # Initialize signal generators
        self.signal_generator_class = signal_generator_class
        self.signal_generator = None
        self.available_signal_generators = {}
        self.signal_generators = []  # Initialize the missing signal_generators list
        self.active_signal_generators = []  # Initialize the missing active_signal_generators list
        self.latest_prices = {}  # Initialize the missing latest_prices dictionary
        self._init_signal_generators(signal_generator_class)
        
        # Set state tracking variables
        self.close_positions_on_shutdown = self.config.get('close_positions_on_shutdown', False)
        self.allow_position_additions = self.config.get('allow_position_additions', False)
        self.use_trailing_stop = self.config.get('use_trailing_stop', True)
        self.trading_enabled = self.config.get('trading_enabled', True)  # Enabled by default
        self.real_time_monitoring_enabled = self.config.get('real_time_monitoring_enabled', True)  # Enable real-time monitoring by default
        self.startup_notification_sent = False  # Flag to track startup notification
        self.stop_requested = False
        
        # Extract trading symbols from config (this is already done in _load_symbols_from_config)
        # Don't override the symbols that were already loaded
        if not self.trading_symbols:
            self.trading_symbols = self.config.get('trading_symbols', ['EURUSD', 'GBPUSD', 'USDJPY'])
        
        self.start_time = datetime.now()
        self.active_trades = {}
        self.pending_trades = {}
        
        # State management
        self.running = False
        self.shutdown_requested = False  # Flag to gracefully exit the main loop
        self.should_stop = False  # Flag to gracefully exit the analysis cycle
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
        
        # Central market data storage
        self.market_data_cache = {}
        
        # Tick data storage and tracking
        self.tick_cache = {}
        self.last_tick_times = {}
        
        # Enhanced state tracking for multi-timeframe analysis
        self.warmup_complete = False
        self.last_analysis_time = {}  # Track last analysis time per timeframe
        self.analysis_debounce_intervals = {
            "M1": 5,    # Check M1 signals every 5 seconds
            "M5": 30,   # Run M5 analysis every 30 seconds
            "M15": 60,  # Run M15 analysis every 60 seconds 
            "H1": 300,  # Run H1 analysis every 5 minutes
            "H4": 1200, # Run H4 analysis every 20 minutes
            "D1": 3600  # Run D1 analysis once per hour
        }
        
        # Track active timeframes requiring analysis
        self.active_timeframes = set()
        
        logger.info("TradingBot initialized with enhanced multi-timeframe analysis capabilities")

    def _init_signal_generators(self, default_generator_class: Optional[Type[SignalGenerator]] = None):
        """Initialize signal generators with configuration."""
        # Use provided signal generator class with fallback to default
        generator_class = default_generator_class or SignalGenerator
        
        # Load each signal generator with MT5 handler and risk manager
        self.primary_signal_generator = generator_class(
            mt5_handler=self.mt5_handler,
            risk_manager=self.risk_manager,
        )
        
        # Import the strategy classes
        try:
            # Try importing from strategy module first
            try:
                from src.strategy import BreakoutReversalStrategy  # type: ignore # pyright: ignore[reportAttributeAccessIssue]
                logger.info("Successfully imported BreakoutReversalStrategy from strategy module")
            except ImportError:
                # Fallback to direct import
                from src.strategy.breakout_reversal_strategy import BreakoutReversalStrategy  # type: ignore # pyright: ignore[reportAttributeAccessIssue]
                logger.info("Imported BreakoutReversalStrategy directly from file")
            
            # Create a dictionary of available signal generators
            self.available_signal_generators = {
                "breakout_reversal": BreakoutReversalStrategy,
                # Add other strategies as they become available
            }
            
            # Initialize the signal generators list based on config
            self.signal_generators = []
            signal_generator_names = self.trading_config.get("signal_generators", ["breakout_reversal"])
            
            for generator_name in signal_generator_names:
                if generator_name in self.available_signal_generators:
                    generator_class = self.available_signal_generators[generator_name]
                    generator = generator_class(
                        mt5_handler=self.mt5_handler,
                        risk_manager=self.risk_manager
                    )
                    self.signal_generators.append(generator)
                    logger.info(f"Loaded signal generator: {generator_name} ({generator.__class__.__name__})")
                else:
                    logger.warning(f"Unknown signal generator: {generator_name}")
            
            # If no signal generators were loaded, add the BreakoutReversalStrategy as default
            if not self.signal_generators:
                default_generator = BreakoutReversalStrategy(
                    mt5_handler=self.mt5_handler,
                    risk_manager=self.risk_manager
                )
                self.signal_generators.append(default_generator)
                logger.info(f"Using BreakoutReversalStrategy as default signal generator")
                
        except ImportError as e:
            logger.error(f"Error importing signal generators: {str(e)}")
            # Add the primary generator as fallback
            if self.primary_signal_generator:
                self.signal_generators = [self.primary_signal_generator]
                logger.warning("Using primary signal generator as fallback due to import error")
            
        logger.info(f"Initialized {len(self.signal_generators)} signal generators")
            
        # Set the active signal generator for legacy code
        self.active_signal_generator_name = "default"
        self.active_signal_generator = self.primary_signal_generator
        
        # Initialize other components with config
        if hasattr(self.risk_manager, 'initialize'):
            self.risk_manager.initialize(self.config)

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
        """
        Start the trading bot main processes.
        
        Returns:
            asyncio.Future: A future that completes when the bot should shut down
        """
        try:
            logger.info("Starting trading bot...")
            
            # Set running flag
            self.running = True
            
            # Initialize risk manager if needed
            if hasattr(self.risk_manager, 'initialize'):
                self.risk_manager.initialize()
            
            # Initialize telegram bot
            if self.telegram_bot:
                try:
                    if not self.telegram_bot.is_running:
                        await self.telegram_bot.start()
                        logger.info("Telegram bot started")
                    else:
                        logger.info("Telegram bot already running")
                except Exception as e:
                    logger.error(f"Failed to start Telegram bot: {str(e)}")
            
            # Register Telegram commands
            await self.register_telegram_commands()
            
            # Start data monitor (either real-time or periodic)
            if self.real_time_monitoring_enabled:
                await self.start_real_time_monitoring()
                
                # Initiate the market status checking mechanism
                logger.info("Starting market status monitoring...")
                asyncio.create_task(self.check_market_status_and_update_subscriptions())
            else:
                # Start the main trading loop (periodic)
                self.main_loop_task = asyncio.create_task(self.main_loop())
                
            # Start trade monitoring task
            self._monitor_trades_task = asyncio.create_task(self._monitor_trades_loop())
            
            # Start shutdown monitor
            self.shutdown_future = asyncio.Future()
            self.shutdown_monitor_task = asyncio.create_task(self._monitor_shutdown())
            
            if not self.startup_notification_sent:
                # Prepare a message with key configuration information
                config_summary = f"Trading Bot Started\n\n"
                config_summary += f"Trading Enabled: {'✅' if self.trading_enabled else '❌'}\n"
                config_summary += f"Signal Generator: {self.signal_generator.__class__.__name__}\n"
                config_summary += f"Monitoring Mode: {'Real-time' if self.real_time_monitoring_enabled else 'Periodic'}\n"
                config_summary += f"Symbols: {', '.join(self.trading_symbols)}\n"
                config_summary += f"Position Additions: {'Enabled' if self.allow_position_additions else 'Disabled'}\n"
                config_summary += f"Trailing Stop: {'Enabled' if self.use_trailing_stop else 'Disabled'}\n"
                config_summary += f"Market Hours Monitoring: {'Enabled' if hasattr(self, 'market_hours') and self.market_hours else 'Disabled'}\n"
                
                if self.telegram_bot:
                    try:
                        # Send startup notification in background task
                        asyncio.create_task(self.telegram_bot.send_message(config_summary))
                    except Exception as e:
                        logger.error(f"Failed to send startup notification: {str(e)}")
                
                # Mark as sent
                self.startup_notification_sent = True
                
            logger.info("Trading bot started successfully")
            
            # Start analysis cycle
            logger.info("Starting analysis cycle...")
            asyncio.create_task(self.analysis_cycle())
            
            # Return the shutdown future so caller can wait for it
            return self.shutdown_future
            
        except Exception as e:
            logger.error(f"Failed to start trading bot: {str(e)}")
            logger.error(traceback.format_exc())
            raise

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
                if hasattr(self, '_monitor_trades_task') and self._monitor_trades_task.done():
                    # If monitor task exited with an error, log it
                    if self._monitor_trades_task.exception():
                        logger.error(f"Monitor task exited with an error: {self._monitor_trades_task.exception()}")
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
            
        # Register commands with the command handler
        try:
            # Main commands
            self.telegram_command_handler.register_command("status", self.handle_status_command)
            self.telegram_command_handler.register_command("shutdown", self.handle_shutdown_command)
            self.telegram_command_handler.register_command("enable_trading", self.handle_enable_trading_command)
            self.telegram_command_handler.register_command("disable_trading", self.handle_disable_trading_command)
            
            # Risk management commands
            self.telegram_command_handler.register_command("enable_close_on_shutdown", self.handle_enable_close_on_shutdown_command)
            self.telegram_command_handler.register_command("disable_close_on_shutdown", self.handle_disable_close_on_shutdown_command)
            self.telegram_command_handler.register_command("enable_position_additions", self.handle_enable_position_additions_command)
            self.telegram_command_handler.register_command("disable_position_additions", self.handle_disable_position_additions_command)
            self.telegram_command_handler.register_command("enable_trailing_stop", self.handle_enable_trailing_stop_command)
            self.telegram_command_handler.register_command("disable_trailing_stop", self.handle_disable_trailing_stop_command)
            
            # Signal generator commands
            self.telegram_command_handler.register_command("list_signal_generators", self.handle_list_signal_generators_command)
            self.telegram_command_handler.register_command("set_signal_generator", self.handle_set_signal_generator_command)
                        
            # Use the telegram command handler to register all commands with the bot
            await self.telegram_command_handler.register_all_commands(self.telegram_bot)
            logger.info("Successfully registered all command handlers")
        except Exception as e:
            logger.error(f"Error registering telegram commands: {str(e)}")
            logger.error(traceback.format_exc())
            # Continue despite errors

    async def handle_list_signal_generators_command(self, args):
        """
        Handle command to list available signal generators.
        Format: /listsignalgenerators
        """
        generators = list(self.available_signal_generators.keys())
        current_generator = self.signal_generator_class.__name__ if self.signal_generator_class is not None else "None"
        
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
        status = f"🤖 Trading Bot Status\n{'='*20}\n"
        
        # Show trading state - use self.trading_enabled directly as it's the source of truth
        status += f"Trading Enabled: {'✅' if self.trading_enabled else '❌'}\n"
        status += f"Trailing Stop: {'✅' if self.trailing_stop_enabled else '❌'}\n"
        status += f"Position Additions: {'✅' if self.allow_position_additions else '❌'}\n"
        status += f"Close on Shutdown: {'✅' if self.close_positions_on_shutdown else '❌'}\n"
        status += f"Current Session: {current_session}\n"
        status += f"Signal Generator: {self.signal_generator_class.__name__ if self.signal_generator_class is not None else 'None'}\n\n"

            
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
        """
        Simplified main trading loop that handles periodic tasks.
        In real-time monitoring mode, this function focuses only on supporting tasks 
        rather than data fetching and signal generation.
        """
        try:
            logger.info("Starting simplified main trading loop for periodic tasks...")
            # Set interval to exactly 60 seconds (1 minute)
            interval = 60
            
            # Make sure startup_notification_sent is set to True to prevent re-sending
            if not self.startup_notification_sent:
                self.startup_notification_sent = True
            
            while self.running and not self.shutdown_requested:
                start_time = time.time()
                
                try:
                    # Check if market is open - skip most operations if closed
                    market_open = self.is_market_open()
                    
                    if not market_open:
                        logger.info("Markets are closed, skipping periodic tasks")
                        await asyncio.sleep(interval)
                        continue
                    
                    # Only proceed if trading is enabled
                    if not self.trading_enabled:
                        logger.info("Trading is disabled, skipping periodic tasks")
                        await asyncio.sleep(interval)
                        continue
                    
                    # Check MT5 connection and reconnect if needed
                    if not self.mt5_handler.connected:
                        logger.warning("MT5 connection lost, attempting to reconnect...")
                        if not self.recover_mt5_connection():
                            logger.error("Failed to recover MT5 connection")
                            await asyncio.sleep(interval)
                            continue
                    
                    # Update performance metrics (daily/weekly stats)
                    await self.update_performance_metrics()
                    
                    # Manage open trades (adjust stop losses, take profits, etc.)
                    await self.manage_open_trades()
                    
                    # Check real-time monitoring status
                    if self.real_time_monitoring_enabled:
                        # Get real-time monitoring status
                        rt_status = self.mt5_handler.get_real_time_status()
                        
                        # If real-time monitoring has stopped unexpectedly, restart it
                        if not rt_status.get("task_running", False):
                            logger.warning("Real-time monitoring task has stopped, restarting...")
                            await self.start_real_time_monitoring()
                
                except Exception as e:
                    logger.error(f"Error in main trading loop: {str(e)}")
                    logger.error(traceback.format_exc())
                
                # Calculate how long to sleep
                elapsed = time.time() - start_time
                sleep_time = max(0, interval - elapsed)
                
                if sleep_time > 0:
                    logger.debug(f"Sleeping for {sleep_time:.2f} seconds until next periodic tasks cycle")
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

    async def process_signals(self, signals: List[Dict]) -> None:
        """Process trading signals and execute trades as needed."""
        try:
            if not self.trading_enabled:
                logger.info("Trading is disabled, skipping signal processing")
                return

            logger.info(f"Processing {len(signals)} trading signals with trading enabled: ✅")
            logger.info(f"Signal generators in use: {[gen.__class__.__name__ for gen in self.signal_generators]}")
            
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

    async def disable_trading(self):
        """Disable trading."""
        try:
            # First update our own state
            self.trading_enabled = False
            
            # Update trading_config to persist the state
            if hasattr(self, 'trading_config'):
                self.trading_config['trading_enabled'] = False
                
            # Update signal processor if available
            if hasattr(self, 'signal_processor') and self.signal_processor:
                self.signal_processor.trading_enabled = False
                
            # Update telegram bot if available
            if hasattr(self, 'telegram_bot') and self.telegram_bot:
                if hasattr(self.telegram_bot, 'disable_trading_core'):
                    await self.telegram_bot.disable_trading_core()
                # Also update telegram bot's internal state
                self.telegram_bot.trading_enabled = False
                
            # Log the state change
            logger.info("Trading disabled across all components")
            
            # Return success message
            return "✅ Trading has been DISABLED"
        except Exception as e:
            logger.error(f"Failed to disable trading: {str(e)}")
            return "❌ Failed to disable trading"

    async def enable_trading(self):
        """Enable trading."""
        try:
            # First update our own state
            self.trading_enabled = True
            
            # Update trading_config to persist the state
            if hasattr(self, 'trading_config'):
                self.trading_config['trading_enabled'] = True
                
            # Update signal processor if available
            if hasattr(self, 'signal_processor') and self.signal_processor:
                self.signal_processor.trading_enabled = True
                
            # Update telegram bot if available
            if hasattr(self, 'telegram_bot') and self.telegram_bot:
                if hasattr(self.telegram_bot, 'enable_trading_core'):
                    await self.telegram_bot.enable_trading_core()
                # Also update telegram bot's internal state
                self.telegram_bot.trading_enabled = True
                
            # Log the state change
            logger.info("Trading enabled across all components")
            
            # Return success message
            return "✅ Trading has been ENABLED"
        except Exception as e:
            logger.error(f"Failed to enable trading: {str(e)}")
            self.trading_enabled = False  # Safety: disable on error
            return "❌ Failed to enable trading"

    async def handle_enable_trading_command(self, args):
        """Handle enable trading command from Telegram."""
        result = await self.enable_trading()
        return result
        
    async def handle_disable_trading_command(self, args):
        """Handle disable trading command from Telegram."""
        result = await self.disable_trading()
        return result

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

    def is_market_open(self, symbol=None) -> bool:
        """
        Check if the market is currently open for trading based on tick activity.
        
        Args:
            symbol: Optional trading symbol to check for specific instrument
                   If None, checks if any symbol is active
                   
        Returns:
            bool: True if the market is open, False otherwise
        """
        try:
            # If no specific symbol is provided, check all configured symbols
            if symbol is None:
                symbols_to_check = self._get_symbol_list()
                
                # Return True if any symbol is open
                for sym in symbols_to_check:
                    if self.is_market_open(sym):
                        return True
                return False
            
            # For a specific symbol, check for recent tick activity
            if not hasattr(self, 'mt5_handler') or not self.mt5_handler:
                logger.warning("MT5 handler not available for tick-based market detection")
                return False
                
            # Get the latest tick
            latest_tick = self.mt5_handler.get_last_tick(symbol)
            if latest_tick is None:
                logger.debug(f"No tick data available for {symbol}, market likely closed")
                return False
                
            # Check tick freshness to determine if market is open
            now = time.time()
            tick_time = latest_tick.time if hasattr(latest_tick, 'time') else 0
            time_diff = now - tick_time
            
            # Consider market open if tick is recent (within last 5 minutes)
            # Adjust this threshold as needed for your specific requirements
            MAX_TICK_AGE = 300  # 5 minutes in seconds
            
            if time_diff <= MAX_TICK_AGE:
                logger.debug(f"Market for {symbol} is open - latest tick {time_diff:.1f} seconds ago")
                return True
            else:
                logger.debug(f"Market for {symbol} appears closed - latest tick is {time_diff:.1f} seconds old")
                return False
            
        except Exception as e:
            logger.error(f"Error in tick-based market detection for {symbol}: {str(e)}")
            logger.error(traceback.format_exc())
            # Default to closed on error as a safety measure
            return False

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
        
        # Update signal processor's setting to ensure it's synced
        if hasattr(self, 'signal_processor') and self.signal_processor:
            self.signal_processor.allow_position_additions = True
            
        # Also update the trading_config to make this change persistent
        if hasattr(self, 'trading_config'):
            self.trading_config["allow_position_additions"] = True
            
        logger.info("Enabled adding to positions")
        return "✅ Adding to positions is now ENABLED"
        
    async def handle_disable_position_additions_command(self, args):
        """
        Handle command to disable adding to positions.
        Format: /disablepositionadditions
        """
        self.allow_position_additions = False
        
        # Update signal processor's setting to ensure it's synced
        if hasattr(self, 'signal_processor') and self.signal_processor:
            self.signal_processor.allow_position_additions = False
            
        # Also update the trading_config to make this change persistent
        if hasattr(self, 'trading_config'):
            self.trading_config["allow_position_additions"] = False
            
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

    async def stop(self, cleanup_only=False):
        """Stop the trading bot and clean up resources"""
        logger.info("Bot stop method called")
        logger.debug(f"Stop called with cleanup_only={cleanup_only}, call stack: {traceback.format_stack()}")
        self.running = False
        self.should_stop = True  # Set should_stop flag to exit analysis cycle gracefully
        
        # Cancel our monitoring tasks if they're still running
        if hasattr(self, 'main_loop_task') and not self.main_loop_task.done():
            self.main_loop_task.cancel()
            
        if hasattr(self, '_monitor_trades_task') and not self._monitor_trades_task.done():
            self._monitor_trades_task.cancel()
            
        if hasattr(self, 'shutdown_monitor_task') and not self.shutdown_monitor_task.done():
            self.shutdown_monitor_task.cancel()
        
        # Stop real-time monitoring if enabled
        if self.real_time_monitoring_enabled:
            self.mt5_handler.stop_real_time_monitoring()
            
        # Close any pending trades
        if self.close_positions_on_shutdown and not cleanup_only:
            logger.info("Closing all open positions before shutdown")
            try:
                positions = self.mt5_handler.get_open_positions()
                if positions:
                    for position in positions:
                        ticket = position.get("ticket", 0)
                        if ticket:
                            self.mt5_handler.close_position(ticket)
                            logger.info(f"Closed position {ticket} during shutdown")
            except Exception as e:
                logger.error(f"Error closing positions during shutdown: {str(e)}")
        
        # Close telegram bot
        if self.telegram_bot and hasattr(self.telegram_bot, 'stop'):
            try:
                await self.telegram_bot.stop()
                logger.info("Telegram bot stopped")
            except Exception as e:
                logger.error(f"Error stopping Telegram bot: {str(e)}")
                
        # Close signal generators
        for generator in self.signal_generators:
            if hasattr(generator, 'close'):
                try:
                    if asyncio.iscoroutinefunction(generator.close):
                        await generator.close()
                    else:
                        generator.close()
                except Exception as e:
                    logger.error(f"Error closing signal generator: {str(e)}")
        
        # Shutdown MT5
        if self.mt5_handler:
            try:
                self.mt5_handler.shutdown()
                logger.info("MT5 connection closed")
            except Exception as e:
                logger.error(f"Error shutting down MT5: {str(e)}")
        
        logger.info("Trading bot stopped")

    async def initialize(self):
        """Initialize the trading bot, configure signal generators, and prepare for trading."""
        try:
            logger.info("Initializing Trading Bot")
            
            # Verify MT5 connection
            if not self.mt5_handler.connected:
                logger.warning("MT5 not connected, trying to reconnect...")
                if not self.mt5_handler.initialize():
                    logger.error("Failed to connect to MT5. Check MT5 installation and credentials.")
                    return False
            
            # Initialize signal generators
            await self._initialize_signal_generators()
            
            # Register required timeframes with data manager
            required_timeframes = set()
            for generator in self.signal_generators:
                if hasattr(generator, 'required_timeframes'):
                    for tf in generator.required_timeframes:
                        required_timeframes.add(tf)
                        self.active_timeframes.add(tf)
            
            # Always include M1 as the base timeframe
            required_timeframes.add("M1")
            self.active_timeframes.add("M1")
            
            # Register timeframes with data manager
            self.data_manager.register_timeframes(list(required_timeframes))
            
            # Get trading symbols
            try:
                if 'symbols' in self.trading_config and self.trading_config["symbols"]:
                    # Extract symbols from dictionary format
                    symbols = [s["symbol"] for s in self.trading_config["symbols"]]
                    self.symbols = symbols
                    logger.info(f"Trading symbols loaded from trading_config: {symbols}")
                elif hasattr(self, 'trading_symbols') and self.trading_symbols:
                    # Use default symbols from initialization
                    self.symbols = self.trading_symbols
                    logger.info(f"Using default trading symbols: {self.symbols}")
                else:
                    # Last resort fallback
                    self.symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
                    logger.warning(f"No symbols found in configuration. Using default symbols: {self.symbols}")
            except Exception as e:
                logger.error(f"Error loading symbols from configuration: {str(e)}")
                # Fallback to default symbols
                self.symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
                logger.warning(f"Using default symbols after error: {self.symbols}")
            
            # Perform warmup data loading
            logger.info("🔄 Starting warmup phase to collect initial historical data...")
            warmup_success = await self.data_manager.perform_warmup(self.symbols)
            
            if warmup_success:
                logger.info("✅ Warmup phase completed successfully")
                self.warmup_complete = True
            else:
                logger.warning("⚠️ Warmup phase completed with some issues, but will attempt to continue")
                self.warmup_complete = True  # Continue anyway, but might have limited data
            
            # Initialize Telegram bot
            if self.telegram_bot:
                await self.telegram_bot.initialize(self.config)
                if hasattr(self.telegram_bot, 'set_trading_bot'):
                    self.telegram_bot.set_trading_bot(self)  # pyright: ignore[reportAttributeAccessIssue]
                else:
                    logger.warning("Telegram bot does not have set_trading_bot method, skipping")
                # Register command handlers
                if hasattr(self.telegram_command_handler, 'register_handlers'):
                    await self.telegram_command_handler.register_handlers()  # pyright: ignore[reportAttributeAccessIssue]
                else:
                    logger.warning("TelegramCommandHandler does not have register_handlers method, skipping")
                logger.info("Telegram bot initialized")
            
            # Initialize market data states
            await self.start_real_time_monitoring()
            
            # Mark as initialized
            self.running = True
            self.should_stop = False
            self.shutdown_requested = False
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing trading bot: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    async def _initialize_signal_generators(self):
        """Initialize all active signal generators."""
        try:
            # Set up signal generators
            active_generators = []
            
            # Get active generator names from config
            active_generator_names = self.trading_config.get("signal_generators", ["breakout_reversal"])
            
            # If we have a specific signal generator class passed, only use that one
            if self.signal_generator_class:
                sg = self.signal_generator_class(
                    mt5_handler=self.mt5_handler,
                    risk_manager=self.risk_manager
                )
                await sg.initialize()
                self.signal_generators = [sg]
                logger.info(f"Initialized signal generator: {sg.__class__.__name__}")
                return
            
            # Load generators from the strategies directory
            self._load_available_signal_generators()
            
            # Initialize active generators from config
            for generator_name in active_generator_names:
                if generator_name in self.available_signal_generators:
                    generator_class = self.available_signal_generators[generator_name]
                    generator = generator_class(
                        mt5_handler=self.mt5_handler, 
                        risk_manager=self.risk_manager
                    )
                    await generator.initialize()
                    active_generators.append(generator)
                    logger.info(f"Initialized signal generator: {generator.__class__.__name__}")
                else:
                    logger.warning(f"Signal generator {generator_name} not found")
            
            # If no generators were found, use BreakoutReversalStrategy as default
            if not active_generators:
                try:
                    # Import BreakoutReversalStrategy
                    try:
                        from src.strategy import BreakoutReversalStrategy  # type: ignore # pyright: ignore[reportAttributeAccessIssue]
                    except ImportError:
                        from src.strategy.breakout_reversal_strategy import BreakoutReversalStrategy  # type: ignore # pyright: ignore[reportAttributeAccessIssue]
                    
                    default_generator = BreakoutReversalStrategy(
                        mt5_handler=self.mt5_handler,
                        risk_manager=self.risk_manager
                    )
                    await default_generator.initialize()
                    active_generators.append(default_generator)
                    logger.info(f"Using BreakoutReversalStrategy as default signal generator")
                except ImportError as e:
                    logger.error(f"Could not import BreakoutReversalStrategy: {str(e)}")
            
            self.signal_generators = active_generators
            logger.info(f"Active signal generators: {[sg.__class__.__name__ for sg in self.signal_generators]}")
            
        except Exception as e:
            logger.error(f"Error initializing signal generators: {str(e)}")
            logger.error(traceback.format_exc())

    async def start_real_time_monitoring(self):
        """
        Initialize real-time market data monitoring and integrate with the analysis cycle.
        
        This method:
        1. Collects all required timeframes from signal generators
        2. Fetches initial historical data for all symbols and timeframes
        3. Starts real-time monitoring with a callback function
        """
        try:
            # Process symbols using our utility method
            symbols = self._get_symbol_list()
            logger.info(f"Starting real-time monitoring with symbols: {symbols}")
            
            # Check which symbols have open markets
            open_symbols = []  # Initialize to empty list
            if hasattr(self, 'market_hours') and self.market_hours:
                for symbol in symbols:
                    if self.market_hours.is_market_open(symbol):
                        open_symbols.append(symbol)
                    else:
                        next_open = self.market_hours.get_next_market_open(symbol)
                        if next_open:
                            next_open_str = next_open.strftime("%Y-%m-%d %H:%M:%S")
                            logger.info(f"Market for {symbol} is currently closed, next open at {next_open_str}")
            
            # If some markets are open, only monitor those
            if open_symbols:
                logger.info(f"Found {len(open_symbols)} symbols with open markets out of {len(symbols)} total symbols")
                symbols = open_symbols
            else:
                logger.warning("No symbols have open markets currently. Monitoring all configured symbols anyway.")
                
            # Log all symbols being monitored
            logger.info(f"Starting real-time monitoring for symbols: {', '.join(symbols)}")
            
            # Collect all required timeframes from signal generators
            timeframes = set()
            for generator in self.signal_generators:
                timeframes.update(generator.required_timeframes)
            
            # Always include M1 as a base timeframe
            timeframes.add("M1")
            timeframes = list(timeframes)
            
            # Register timeframes with data manager
            self.data_manager.register_timeframes(timeframes)
            
            # Warmup phase - collect initial historical data if not done already
            if not self.warmup_complete:
                logger.info("Starting data warmup phase...")
                try:
                    successful_warmup = await self.data_manager.perform_warmup(symbols)
                    if successful_warmup:
                        logger.info("✅ Warmup phase completed successfully")
                        self.warmup_complete = True
                    else:
                        logger.warning("⚠️ Warmup phase had issues, proceeding with available data")
                        self.warmup_complete = True  # Continue anyway with what we have
                except Exception as e:
                    logger.error(f"❌ Error during warmup phase: {str(e)}")
                    logger.error(traceback.format_exc())
                    self.warmup_complete = False
            else:
                logger.info("Warmup phase already completed, skipping")
            
            # Start real-time monitoring with callback
            success = self.mt5_handler.start_real_time_monitoring(
                symbols=symbols,
                timeframes=timeframes,
                callback_function=self.handle_real_time_data
            )
            
            if success:
                logger.info(f"✅ Real-time monitoring started for {len(symbols)} symbols and {len(timeframes)} timeframes")
                self.real_time_monitoring_enabled = True
                
                # Start analysis cycle if not already running
                if not hasattr(self, 'analysis_cycle_task') or self.analysis_cycle_task.done():
                    logger.info("Starting analysis cycle...")
                    self.analysis_cycle_task = asyncio.create_task(self.analysis_cycle())
            else:
                logger.error("❌ Failed to start real-time monitoring, falling back to periodic mode")
                self.real_time_monitoring_enabled = False
                self.main_loop_task = asyncio.create_task(self.main_loop())
                
        except Exception as e:
            logger.error(f"❌ Error starting real-time monitoring: {str(e)}")
            logger.error(traceback.format_exc())
            self.real_time_monitoring_enabled = False
            # Fall back to traditional approach
            self.main_loop_task = asyncio.create_task(self.main_loop())
    
    async def analysis_cycle(self):
        """
        Simplified analysis cycle that just waits for new candles.
        
        The actual analysis is triggered in handle_real_time_data when 
        new candles arrive on timeframes required by the strategies.
        """
        logger.info("Starting simplified analysis cycle - waiting for new candles")
        
        # Log important state information
        logger.info(f"Signal generators: {[gen.__class__.__name__ for gen in self.signal_generators]}")
        
        # Set warmup_complete to True if it's not already set
        if not self.warmup_complete:
            logger.info("Setting warmup_complete=True to enable analysis")
            self.warmup_complete = True
        
        # Get required timeframes from signal generators
        self.active_timeframes = set()
        for gen in self.signal_generators:
            if hasattr(gen, 'required_timeframes'):
                self.active_timeframes.update(gen.required_timeframes)
        
        logger.info(f"Monitoring these timeframes for new candles: {self.active_timeframes}")
        
        # Just keep the cycle running to receive events
        while not self.should_stop:
            try:
                # Just sleep and wait for new candles
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error in analysis cycle: {str(e)}")
                logger.exception("Analysis cycle exception")
                await asyncio.sleep(5)  # Sleep longer on error
                
    async def _analyze_timeframe(self, timeframe):
        """
        Run analysis for a specific timeframe by passing data to signal generators.
        
        This is triggered when a new candle forms on a timeframe that's required
        by at least one signal generator.
        """
        try:
            now = time.time()
            
            # For each symbol
            for symbol in self.symbols:
                # Skip if we don't have candle data for this symbol/timeframe
                if (symbol not in self.market_data_cache or 
                    timeframe not in self.market_data_cache[symbol]):
                    logger.debug(f"No {timeframe} candle data for {symbol}, skipping")
                    continue
                
                # Get the data and check if it's empty
                candle_data = self.market_data_cache[symbol][timeframe]
                
                # Check if the data is empty based on its type
                is_empty = False
                if isinstance(candle_data, pd.DataFrame):
                    # It's a pandas DataFrame
                    is_empty = candle_data.empty if hasattr(candle_data, 'empty') else len(candle_data) == 0
                elif isinstance(candle_data, dict):
                    # It's a dictionary - more careful check to avoid false empty results
                    if len(candle_data) == 0:
                        is_empty = True
                    # Special case for nested data structures
                    elif 'rates' in candle_data and isinstance(candle_data['rates'], list):
                        is_empty = len(candle_data['rates']) == 0
                    elif any(isinstance(v, (list, np.ndarray)) for v in candle_data.values()):
                        # Check if any array values have content
                        is_empty = all(len(v) == 0 for v in candle_data.values() 
                                      if isinstance(v, (list, np.ndarray)))
                else:
                    # Other data type - try to safely determine emptiness
                    try:
                        is_empty = len(candle_data) == 0 if hasattr(candle_data, '__len__') else not bool(candle_data)
                    except:
                        # If we can't determine, assume it has data
                        is_empty = False
                
                if is_empty:
                    logger.debug(f"Empty data for {symbol}/{timeframe}, skipping")
                    continue
                
                # Only pass data to generators that require this timeframe
                for signal_gen in self.signal_generators:
                    # Skip generators that don't need this timeframe
                    if (not hasattr(signal_gen, 'required_timeframes') or 
                        timeframe not in signal_gen.required_timeframes):
                        continue
                    
                    gen_name = signal_gen.__class__.__name__
                    logger.info(f"Passing new {timeframe} candle data to {gen_name} for {symbol}")
                    
                    try:
                        # Get all the timeframe data this generator needs
                        market_data = {symbol: {}}
                        
                        # First add the current timeframe that just updated
                        market_data[symbol][timeframe] = candle_data
                        
                        # Then add any other required timeframes
                        for tf in signal_gen.required_timeframes:
                            if tf != timeframe and tf in self.market_data_cache.get(symbol, {}):
                                # Get data for this timeframe
                                tf_data = self.market_data_cache[symbol][tf]
                                # Check if it's empty
                                tf_is_empty = False
                                if isinstance(tf_data, pd.DataFrame):
                                    tf_is_empty = tf_data.empty if hasattr(tf_data, 'empty') else len(tf_data) == 0
                                elif not tf_data:
                                    tf_is_empty = True
                                
                                # Only include non-empty data
                                if not tf_is_empty:
                                    market_data[symbol][tf] = tf_data
                                else:
                                    logger.debug(f"Skipping empty data for {symbol}/{tf}")
                        
                        # Log the market data being sent
                        timeframes_included = list(market_data[symbol].keys())
                        logger.debug(f"Sending market data for {symbol} with timeframes: {timeframes_included}")
                        
                        # Call generate_signals with the prepared market data
                        signals = await signal_gen.generate_signals(
                            market_data=market_data,
                            symbol=symbol,
                            timeframe=timeframe
                        )
                        
                        # Process signals if any were generated
                        if signals:
                            logger.info(f"Processing {len(signals)} signals from {gen_name}")
                            
                            # Skip trade execution if trading is disabled
                            if not self.trading_enabled:
                                logger.info("Trading is disabled, skipping trade execution")
                                continue
                                
                            # Process the signals
                            await self.process_signals(signals)
                        else:
                            logger.debug(f"No signals generated by {gen_name}")
                            
                    except Exception as e:
                        logger.error(f"Error in signal generation: {str(e)}")
                        logger.exception("Signal generator exception")
            
            # Update the last analysis time
            self.last_analysis_time[timeframe] = now
                
        except Exception as e:
            logger.error(f"Error analyzing timeframe {timeframe}: {str(e)}")
            logger.exception("Analysis exception")
    
    async def check_market_status_and_update_subscriptions(self):
        """
        Periodically check the market status for all symbols using tick-based detection.
        
        This method:
        1. Uses tick activity to determine which markets are currently open/closed
        2. Removes symbols with closed markets from active monitoring
        3. Adds symbols with newly opened markets to active monitoring
        4. Schedules more frequent checks for crypto pairs
        
        Returns:
            None
        """
        try:
            # Get the current list of all configured symbols
            all_symbols = self._get_symbol_list()
                
            logger.info(f"Checking market status for {len(all_symbols)} symbols using tick activity")
            
            # Check which markets are open based on tick activity
            open_symbols = []
            closed_symbols = []
            
            for symbol in all_symbols:
                # Check if market is currently open using tick-based detection
                is_open = self.is_market_open(symbol)
                
                if is_open:
                    open_symbols.append(symbol)
                else:
                    closed_symbols.append(symbol)
            
            # Categorize symbols as crypto or forex based on common naming patterns
            crypto_symbols = [s for s in all_symbols if any(c in s.upper() for c in ['BTC', 'ETH', 'XRP', 'LTC', 'XBT'])]
            forex_symbols = [s for s in all_symbols if s not in crypto_symbols]
            
            crypto_open = [s for s in crypto_symbols if s in open_symbols]
            forex_open = [s for s in forex_symbols if s in open_symbols]
            
            logger.info(f"Market status: {len(open_symbols)} open markets ({len(crypto_open)} crypto, {len(forex_open)} forex)")
            logger.info(f"Closed markets: {len(closed_symbols)} symbols")
            
            # If we have active MT5 monitoring, update the subscription list
            if self.real_time_monitoring_enabled and hasattr(self, 'mt5_handler') and self.mt5_handler:
                # Get the currently monitored symbols
                current_symbols = self.mt5_handler.get_monitored_symbols()
                
                # Determine which symbols to add and remove
                symbols_to_add = [s for s in open_symbols if s not in current_symbols]
                symbols_to_remove = [s for s in current_symbols if s not in open_symbols]
                
                if symbols_to_add:
                    logger.info(f"Adding {len(symbols_to_add)} symbols to real-time monitoring: {', '.join(symbols_to_add)}")
                    # Get the timeframes we need for these symbols
                    required_timeframes = set()
                    for generator in self.signal_generators:
                        if hasattr(generator, 'required_timeframes'):
                            required_timeframes.update(generator.required_timeframes)
                    
                    # Ensure M1 is included
                    required_timeframes.add("M1")
                    timeframes = list(required_timeframes)
                    
                    # Subscribe to new symbols
                    success = self.mt5_handler.subscribe_symbols(
                        symbols=symbols_to_add,
                        timeframes=timeframes,
                        callback_function=self.handle_real_time_data
                    )
                    
                    if success:
                        logger.info(f"Successfully subscribed to {len(symbols_to_add)} new symbols")
                    else:
                        logger.error(f"Failed to subscribe to new symbols")
                        
                if symbols_to_remove:
                    logger.info(f"Removing {len(symbols_to_remove)} symbols from real-time monitoring: {', '.join(symbols_to_remove)}")
                    # Unsubscribe from symbols with closed markets
                    success = self.mt5_handler.unsubscribe_symbols(symbols_to_remove)
                    
                    if success:
                        logger.info(f"Successfully unsubscribed from {len(symbols_to_remove)} symbols with closed markets")
                    else:
                        logger.error(f"Failed to unsubscribe from symbols with closed markets")
            
            # Schedule next market check with adaptive timing
            
            # Schedule more frequent checks if we have crypto pairs
            if crypto_symbols:
                # Check every 5 minutes for crypto symbols as they can trade 24/7
                # but may have liquidity gaps that appear as closed markets
                logger.info("Crypto pairs detected, scheduling next market check in 5 minutes")
                check_delay = 300  # 5 minutes in seconds
            elif len(open_symbols) == 0 and datetime.now().weekday() >= 5:
                # It's weekend and no markets open, check hourly
                logger.info("Weekend with no open markets, scheduling next check in 1 hour")
                check_delay = 3600  # 1 hour in seconds
            elif len(open_symbols) == 0:
                # No markets open but it's a weekday, check every 15 minutes
                logger.info("Weekday with no open markets, scheduling next check in 15 minutes")
                check_delay = 900  # 15 minutes in seconds
            else:
                # Regular market hours, check every 10 minutes
                logger.info("Active markets, scheduling next check in 10 minutes")
                check_delay = 600  # 10 minutes in seconds
                
            # Schedule the next check
            asyncio.create_task(self._schedule_next_market_check(check_delay))
                
        except Exception as e:
            logger.error(f"Error checking market status: {str(e)}")
            logger.error(traceback.format_exc())
            # Schedule another check in 5 minutes on error
            asyncio.create_task(self._schedule_next_market_check(300))
    
    async def _schedule_next_market_check(self, delay_seconds):
        """
        Schedule the next market status check after the specified delay.
        
        Args:
            delay_seconds: Seconds to wait before next check
        """
        try:
            await asyncio.sleep(delay_seconds)
            # Only perform the check if the bot is still running
            if self.running and not self.shutdown_requested:
                await self.check_market_status_and_update_subscriptions()
        except asyncio.CancelledError:
            logger.debug("Market status check task cancelled")
        except Exception as e:
            logger.error(f"Error in scheduled market check: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _get_symbol_list(self):
        """
        Get a complete list of trading symbols from all possible sources.
        
        Returns:
            list: List of trading symbols
        """
        all_symbols = []
        
        # First try to get symbols from self.symbols (already processed list)
        if hasattr(self, 'symbols') and self.symbols:
            all_symbols.extend(self.symbols)
            logger.debug(f"Returning {len(self.symbols)} symbols from self.symbols")
            return all_symbols
            
        # Then check trading_config
        if hasattr(self, 'trading_config') and 'symbols' in self.trading_config:
            if isinstance(self.trading_config['symbols'], list):
                if self.trading_config['symbols'] and isinstance(self.trading_config['symbols'][0], dict):
                    # Extract symbols from dictionary format
                    for symbol_config in self.trading_config.get('symbols', []):
                        symbol = symbol_config.get('symbol')
                        if symbol and symbol not in all_symbols:
                            all_symbols.append(symbol)
                    logger.debug(f"Added {len(all_symbols)} symbols from trading_config")
                else:
                    # Direct list format
                    for symbol in self.trading_config['symbols']:
                        if symbol and symbol not in all_symbols:
                            all_symbols.append(symbol)
        
        # Also check TRADING_CONFIGS (for legacy configurations)
        for config_name in self.config.get('TRADING_CONFIGS', []):
            config = self.config.get(config_name, {})
            symbol = config.get('symbol')
            if symbol and symbol not in all_symbols:
                all_symbols.append(symbol)
        
        # Add manually defined trading symbols
        if hasattr(self, 'trading_symbols') and self.trading_symbols:
            for symbol in self.trading_symbols:
                if symbol not in all_symbols:
                    all_symbols.append(symbol)
        
        # If still no symbols, use defaults
        if not all_symbols:
            default_symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
            logger.warning(f"No symbols found in any configuration, using defaults: {default_symbols}")
            all_symbols = default_symbols
            
        return all_symbols
    
    def _load_symbols_from_config(self):
        """Load trading symbols from configuration."""
        try:
            # Debug the available config
            logger.debug(f"Trading config keys: {list(self.trading_config.keys())}")
            
            # Try to load symbols from trading_config
            if 'symbols' in self.trading_config and isinstance(self.trading_config['symbols'], list):
                logger.debug(f"Found symbols list in trading_config with {len(self.trading_config['symbols'])} items")
                
                if self.trading_config['symbols'] and isinstance(self.trading_config['symbols'][0], dict):
                    # Extract symbols from dictionary format (with "symbol" key)
                    self.symbols = [s["symbol"] for s in self.trading_config['symbols']]
                    logger.info(f"Loaded symbols from trading_config dictionary format: {self.symbols}")
                elif self.trading_config['symbols']:
                    # Handle case where symbols are direct strings
                    self.symbols = self.trading_config['symbols']
                    logger.info(f"Loaded symbols from trading_config direct list: {self.symbols}")
            else:
                logger.warning(f"No 'symbols' key found in trading_config or not a list. Available keys: {list(self.trading_config.keys())}")
                
                # Try to directly import from config.py as fallback
                try:
                    from config.config import TRADING_CONFIG
                    if 'symbols' in TRADING_CONFIG and isinstance(TRADING_CONFIG['symbols'], list):
                        if TRADING_CONFIG['symbols'] and isinstance(TRADING_CONFIG['symbols'][0], dict):
                            # Extract symbols from dictionary format
                            self.symbols = [s["symbol"] for s in TRADING_CONFIG['symbols']]
                            logger.info(f"Loaded symbols directly from config.py: {self.symbols}")
                            
                            # Also update trading_config for consistency
                            self.trading_config['symbols'] = TRADING_CONFIG['symbols']
                except Exception as import_err:
                    logger.error(f"Failed to import TRADING_CONFIG directly: {str(import_err)}")

            # If symbols not found in trading_config, check trading_symbols attribute
            if not self.symbols and hasattr(self, 'trading_symbols') and self.trading_symbols:
                self.symbols = self.trading_symbols
                logger.info(f"Loaded symbols from trading_symbols attribute: {self.symbols}")
                
            # Fall back to default symbols with 'm' suffix if none found
            if not self.symbols:
                self.symbols = ['EURUSDm', 'GBPUSDm', 'USDJPYm', 'AUDUSDm', 'USDCADm']
                logger.warning(f"No symbols found in configuration, using defaults with 'm' suffix: {self.symbols}")
            
            # Override self.trading_symbols to match the loaded symbols
            self.trading_symbols = self.symbols
                
        except Exception as e:
            logger.error(f"Error loading symbols from configuration: {str(e)}")
            logger.error(traceback.format_exc())
            # Set default symbols with 'm' suffix as fallback
            self.symbols = ['EURUSDm', 'GBPUSDm', 'USDJPYm', 'AUDUSDm', 'USDCADm']
            self.trading_symbols = self.symbols
            logger.warning(f"Using default symbols with 'm' suffix after error: {self.symbols}")
        
        # Ensure no duplicates
        self.symbols = list(dict.fromkeys(self.symbols))
    
    def get_analysis_interval(self, timeframe):
        """
        Get the minimum time between analyses for a timeframe.
        
        Args:
            timeframe (str): Timeframe identifier (e.g., "M1", "M5", "H1")
            
        Returns:
            float: Minimum time in seconds between analyses
        """
        # Default debounce intervals by timeframe
        default_intervals = {
            "M1": 10,   # Every 10 seconds for M1
            "M5": 30,   # Every 30 seconds for M5
            "M15": 60,  # Every minute for M15
            "M30": 120, # Every 2 minutes for M30
            "H1": 300,  # Every 5 minutes for H1
            "H4": 900,  # Every 15 minutes for H4
            "D1": 3600  # Every hour for D1
        }
        
        # Use custom intervals if defined, otherwise use defaults
        return self.analysis_debounce_intervals.get(timeframe, default_intervals.get(timeframe, 60))
        
    def get_candle_completion_timing(self, timeframe):
        """
        Get the threshold time (in seconds) before candle completion when analysis should occur.
        Returns 0 for M1 (analyze anytime) and increasingly higher values for higher timeframes.
        
        Args:
            timeframe (str): Timeframe identifier (e.g., "M1", "M5", "H1")
            
        Returns:
            float: Threshold time in seconds before candle completion
        """
        if timeframe == "M1":
            return 0  # Analyze anytime for M1
        
        # Calculate based on percentage of candle duration
        seconds = self.get_seconds_per_candle(timeframe)
        
        if timeframe.startswith("M"):
            # For minute timeframes, analyze in the last 20% of the candle
            return 0.2 * seconds
        elif timeframe.startswith("H"):
            # For hour timeframes, analyze in the last 10% of the candle
            return 0.1 * seconds
        else:
            # For daily+ timeframes, analyze in the last 5% of the candle
            return 0.05 * seconds
    
    def get_seconds_per_candle(self, timeframe):
        """
        Convert a timeframe string to seconds.
        
        Args:
            timeframe (str): Timeframe identifier (e.g., "M1", "M5", "H1")
            
        Returns:
            int: Number of seconds per candle for the given timeframe
        """
        if timeframe.startswith("M"):
            return int(timeframe[1:]) * 60
        elif timeframe.startswith("H"):
            return int(timeframe[1:]) * 3600
        elif timeframe.startswith("D"):
            return int(timeframe[1:]) * 86400
        else:
            logger.warning(f"Unknown timeframe format: {timeframe}")
            return 60  # Default to 1 minute
    
    async def handle_real_time_data(self, symbol, timeframe, data, data_type):
        """
        Process real-time market data (ticks or candles).
        
        Args:
            symbol: The market symbol (e.g., "EURUSD")
            timeframe: The timeframe of the data (e.g., "M5")
            data: The market data (tick or candle)
            data_type: Type of data ('tick' or 'candle')
        """
        try:
            # Skip if market is closed for this symbol
            if not self.market_status.get(symbol, True):
                return
                
            if data_type == 'tick':
                # Update latest price
                if symbol in self.latest_prices:
                    self.latest_prices[symbol] = data['bid']
                logger.debug(f"Tick: {symbol} Bid: {data['bid']} Ask: {data['ask']}")
                
            elif data_type == 'candle':
                if symbol not in self.market_data_cache:
                    self.market_data_cache[symbol] = {}
                
                # Log the data structure for debugging
                logger.debug(f"Candle data type for {symbol}/{timeframe}: {type(data)}")
                
                # Check if data is empty
                is_empty = False
                if isinstance(data, pd.DataFrame):
                    # It's a pandas DataFrame
                    is_empty = data.empty if hasattr(data, 'empty') else len(data) == 0
                elif isinstance(data, dict):
                    # It's a dictionary
                    is_empty = len(data) == 0
                else:
                    # Other type - try to determine if it's empty
                    try:
                        is_empty = len(data) == 0 if hasattr(data, '__len__') else not bool(data)
                    except:
                        is_empty = False  # Assume it has data if we can't determine
                
                if is_empty:
                    logger.debug(f"Empty candle data received for {symbol}/{timeframe}")
                    return
                
                # Store candle data in cache regardless of format
                self.market_data_cache[symbol][timeframe] = data
                
                # Log new candle information
                try:
                    o = h = l = c = pct_change = None
                    
                    # Try to extract candle data with different strategies
                    if isinstance(data, pd.DataFrame) and hasattr(data, 'iloc') and len(data) > 0:
                        # It's a pandas DataFrame
                        logger.debug(f"Processing DataFrame with {len(data)} rows and columns: {list(data.columns)}")
                        latest_candle = data.iloc[-1]
                        # Convert column names to lowercase for consistency
                        columns_lower = {col.lower(): col for col in data.columns}
                        
                        # Try to get OHLC data using various column name formats
                        for open_col in ['open', 'Open', 'OPEN', 'o', 'O']:
                            if open_col in data.columns:
                                o = latest_candle[open_col]
                                break
                            elif open_col.lower() in columns_lower:
                                o = latest_candle[columns_lower[open_col.lower()]]
                                break
                                
                        for high_col in ['high', 'High', 'HIGH', 'h', 'H']:
                            if high_col in data.columns:
                                h = latest_candle[high_col]
                                break
                            elif high_col.lower() in columns_lower:
                                h = latest_candle[columns_lower[high_col.lower()]]
                                break
                                
                        for low_col in ['low', 'Low', 'LOW', 'l', 'L']:
                            if low_col in data.columns:
                                l = latest_candle[low_col]
                                break
                            elif low_col.lower() in columns_lower:
                                l = latest_candle[columns_lower[low_col.lower()]]
                                break
                                
                        for close_col in ['close', 'Close', 'CLOSE', 'c', 'C']:
                            if close_col in data.columns:
                                c = latest_candle[close_col]
                                break
                            elif close_col.lower() in columns_lower:
                                c = latest_candle[columns_lower[close_col.lower()]]
                                break
                        
                    elif isinstance(data, dict):
                        logger.debug(f"Processing dictionary with keys: {list(data.keys())}")
                        # Convert keys to lowercase for case-insensitive matching
                        keys_lower = {k.lower() if isinstance(k, str) else k: k for k in data.keys()}
                        
                        # MT5-specific format detection (common patterns)
                        mt5_container_keys = ['time_series', 'rates', 'bars', 'candles', 'ticks', 'ohlc', 'tick_data']
                        
                        # Case 0: MT5 common container patterns
                        mt5_container = None
                        for container_key in mt5_container_keys:
                            if container_key in keys_lower:
                                mt5_container = data[keys_lower[container_key]]
                                break
                                
                        if mt5_container is not None:
                            # Found a container, try to get the last item
                            if isinstance(mt5_container, list) and mt5_container:
                                last_item = mt5_container[-1]
                                
                                if hasattr(last_item, 'time'):
                                    # Likely a named tuple from MT5
                                    o = getattr(last_item, 'open', None)
                                    h = getattr(last_item, 'high', None)
                                    l = getattr(last_item, 'low', None)
                                    c = getattr(last_item, 'close', None)
                                    
                                    # If we couldn't find OHLC attributes, try different naming patterns
                                    if o is None and hasattr(last_item, 'bid'):
                                        # It might be a tick
                                        c = getattr(last_item, 'bid', None)
                                        o = c  # For ticks, use bid for open too
                                        h = getattr(last_item, 'ask', c)  # Use ask for high, or bid if not available
                                        l = c  # Use bid for low too
                                elif isinstance(last_item, dict):
                                    # Dictionary inside the container
                                    item_keys_lower = {k.lower() if isinstance(k, str) else k: k 
                                                     for k in last_item.keys()}
                                    
                                    o = last_item.get(item_keys_lower.get('open', None))
                                    h = last_item.get(item_keys_lower.get('high', None))
                                    l = last_item.get(item_keys_lower.get('low', None))
                                    c = last_item.get(item_keys_lower.get('close', None))
                        
                        # Case 1: Dictionary with direct OHLC keys
                        elif any(key.lower() in keys_lower if isinstance(key, str) else False 
                               for key in ['open', 'high', 'low', 'close']):
                            
                            # Direct OHLC keys
                            o_key = keys_lower.get('open')
                            h_key = keys_lower.get('high')
                            l_key = keys_lower.get('low')
                            c_key = keys_lower.get('close')
                            
                            o = data[o_key] if o_key in data else None
                            h = data[h_key] if h_key in data else None
                            l = data[l_key] if l_key in data else None
                            c = data[c_key] if c_key in data else None
                        
                        # Case for nested DataFrames with timeframe keys (based on debug logs)
                        elif any(isinstance(v, pd.DataFrame) for v in data.values()):
                            try:
                                # First check if this specific timeframe is directly in the keys
                                if timeframe in data and isinstance(data[timeframe], pd.DataFrame) and len(data[timeframe]) > 0:
                                    df = data[timeframe]
                                    latest_candle = df.iloc[-1]
                                    
                                    # Convert column names to lowercase for case-insensitive matching
                                    cols_lower = {col.lower(): col for col in df.columns}
                                    
                                    # Extract OHLC values
                                    if 'open' in cols_lower: o = latest_candle[cols_lower['open']]
                                    if 'high' in cols_lower: h = latest_candle[cols_lower['high']]
                                    if 'low' in cols_lower: l = latest_candle[cols_lower['low']]
                                    if 'close' in cols_lower: c = latest_candle[cols_lower['close']]
                                
                                # Map standard MT5 timeframes to alternate formats seen in logs
                                timeframe_mapping = {
                                    'M1': ['M1', '1m'],
                                    'M5': ['M5', '5m'],
                                    'M15': ['M15', '15m'],
                                    'H1': ['H1', '1h'],
                                    'H4': ['H4', '4h'],
                                    'D1': ['D1', '1d']
                                }
                                
                                # If direct key didn't work, try alternate formats
                                if o is None and timeframe in timeframe_mapping:
                                    for tf_key in timeframe_mapping[timeframe]:
                                        if tf_key in data and isinstance(data[tf_key], pd.DataFrame) and len(data[tf_key]) > 0:
                                            df = data[tf_key]
                                            latest_candle = df.iloc[-1]
                                            
                                            # Convert column names to lowercase for case-insensitive matching
                                            cols_lower = {col.lower(): col for col in df.columns}
                                            
                                            # Extract OHLC values
                                            if 'open' in cols_lower: o = latest_candle[cols_lower['open']]
                                            if 'high' in cols_lower: h = latest_candle[cols_lower['high']]
                                            if 'low' in cols_lower: l = latest_candle[cols_lower['low']]
                                            if 'close' in cols_lower: c = latest_candle[cols_lower['close']]
                                            
                                            if o is not None:
                                                logger.debug(f"Found data in alternate timeframe format: {tf_key}")
                                                break
                            except Exception as e:
                                logger.debug(f"Error extracting from nested DataFrames: {str(e)}")
                        
                        # Case 2: Dictionary with arrays/lists of OHLC values
                        elif any(key.lower() in keys_lower if isinstance(key, str) else False 
                                for key in ['open', 'high', 'low', 'close']) and \
                             isinstance(next(iter(data.values())), (list, tuple, np.ndarray)):
                            
                            # Get the last value from each array
                            try:
                                o_key = keys_lower.get('open')
                                h_key = keys_lower.get('high')
                                l_key = keys_lower.get('low')
                                c_key = keys_lower.get('close')
                                
                                o = data[o_key][-1] if o_key and o_key in data and data[o_key] else None
                                h = data[h_key][-1] if h_key and h_key in data and data[h_key] else None
                                l = data[l_key][-1] if l_key and l_key in data and data[l_key] else None
                                c = data[c_key][-1] if c_key and c_key in data and data[c_key] else None
                            except (IndexError, KeyError) as e:
                                logger.debug(f"Error extracting from arrays: {str(e)}")
                                
                        # Case 3: Dictionary of dictionaries (each key is a timestamp or index)
                        elif any(isinstance(v, dict) for v in data.values()):
                            try:
                                # Find the most recent key (assuming keys are sortable)
                                try:
                                    sorted_keys = sorted(data.keys())
                                    last_key = sorted_keys[-1]
                                    last_candle = data[last_key]
                                except (TypeError, ValueError):
                                    # If keys are not sortable, take any dictionary
                                    last_candle = next(v for v in data.values() if isinstance(v, dict))
                                
                                # Extract values from the last candle
                                last_keys_lower = {k.lower() if isinstance(k, str) else k: k 
                                                 for k in last_candle.keys()}
                                
                                o = last_candle[last_keys_lower.get('open')] if 'open' in last_keys_lower else None
                                h = last_candle[last_keys_lower.get('high')] if 'high' in last_keys_lower else None
                                l = last_candle[last_keys_lower.get('low')] if 'low' in last_keys_lower else None
                                c = last_candle[last_keys_lower.get('close')] if 'close' in last_keys_lower else None
                            except (IndexError, KeyError, StopIteration) as e:
                                logger.debug(f"Error extracting from nested dict: {str(e)}")
                        
                        # Case 4: MT5-specific format (time-indexed dictionary with rates)
                        elif 'rates' in keys_lower and isinstance(data[keys_lower.get('rates')], list):
                            try:
                                rates = data[keys_lower.get('rates')]
                                if rates and len(rates) > 0:
                                    last_rate = rates[-1]
                                    # Check if it's a dictionary or a named tuple
                                    if isinstance(last_rate, dict):
                                        rate_keys_lower = {k.lower() if isinstance(k, str) else k: k 
                                                          for k in last_rate.keys()}
                                        o = last_rate[rate_keys_lower.get('open')] if 'open' in rate_keys_lower else None
                                        h = last_rate[rate_keys_lower.get('high')] if 'high' in rate_keys_lower else None
                                        l = last_rate[rate_keys_lower.get('low')] if 'low' in rate_keys_lower else None
                                        c = last_rate[rate_keys_lower.get('close')] if 'close' in rate_keys_lower else None
                                    else:
                                        # Try to access as attributes
                                        o = getattr(last_rate, 'open', None)
                                        h = getattr(last_rate, 'high', None)
                                        l = getattr(last_rate, 'low', None)
                                        c = getattr(last_rate, 'close', None)
                            except (IndexError, KeyError, AttributeError) as e:
                                logger.debug(f"Error extracting from rates: {str(e)}")
                    
                    # Handle objects with direct attributes
                    elif hasattr(data, 'open') and hasattr(data, 'high') and hasattr(data, 'low') and hasattr(data, 'close'):
                        o = getattr(data, 'open')
                        h = getattr(data, 'high')
                        l = getattr(data, 'low')
                        c = getattr(data, 'close')
                    
                    # Case for MQL5/MT5 native type objects (array of structures)
                    elif hasattr(data, '__iter__') and hasattr(data, '__len__') and len(data) > 0:
                        try:
                            # Get the last item
                            last_item = data[-1]
                            
                            # Try to extract OHLC from it
                            if hasattr(last_item, 'open') and hasattr(last_item, 'high') and hasattr(last_item, 'low') and hasattr(last_item, 'close'):
                                o = getattr(last_item, 'open')
                                h = getattr(last_item, 'high')
                                l = getattr(last_item, 'low')
                                c = getattr(last_item, 'close')
                            elif hasattr(last_item, 'time') and hasattr(last_item, 'price'):
                                # It's likely a tick
                                c = getattr(last_item, 'price')
                                o = h = l = c  # Use the same value for all
                        except (IndexError, AttributeError) as e:
                            logger.debug(f"Error extracting from iterable: {str(e)}")
                    
                    # If we found at least open and close, calculate percentage change
                    if o is not None and c is not None and o != 0:
                        pct_change = ((c - o) / o) * 100
                    
                    # Log the candle information
                    if o is not None and h is not None and l is not None and c is not None:
                        logger.info(f"New candle: {symbol}/{timeframe} - O: {o:.5f}, H: {h:.5f}, L: {l:.5f}, C: {c:.5f}, Chg: {pct_change:.2f}%")
                    else:
                        # If we couldn't extract OHLC, log detailed structure for debugging
                        debug_structure = self._debug_data_structure(data)
                        logger.warning(f"Unrecognized candle data format for {symbol}/{timeframe}")
                        logger.debug(f"Data structure: {debug_structure}")
                        logger.info(f"New candle: {symbol}/{timeframe} - Could not extract OHLC values")
                        
                except Exception as e:
                    logger.warning(f"Error logging candle info: {str(e)}")
                    debug_structure = self._debug_data_structure(data)
                    logger.debug(f"Problematic data structure: {debug_structure}")
                
                # Check if this timeframe is active for analysis
                if timeframe in self.active_timeframes:
                    # Check if enough time has passed since last analysis
                    now = time.time()
                    if (timeframe not in self.last_analysis_time or 
                        now - self.last_analysis_time.get(timeframe, 0) > 5):  # 5 second debounce
                        
                        logger.info(f"Triggering analysis for {timeframe} candle update")
                        asyncio.create_task(self._analyze_timeframe(timeframe))
                    else:
                        logger.debug(f"Skipping analysis, last one was {now - self.last_analysis_time.get(timeframe, 0):.1f}s ago")
        
        except Exception as e:
            logger.error(f"Error in handle_real_time_data: {str(e)}")
            logger.exception("Exception details:")
    
    def analyze_session(self):
        """
        Determine the current trading session based on time.
        
        Returns:
            str: The current trading session (e.g. "Asian", "European", "US", "Closed")
        """
        try:
            # Get current UTC time
            now_utc = datetime.now(pytz.UTC)
            
            # Convert to New York time for market sessions
            ny_time = now_utc.astimezone(self.ny_timezone)
            current_hour = ny_time.hour
            weekday = ny_time.weekday()  # 0-6, Monday is 0
            
            # Weekend check
            if weekday >= 5:  # Saturday or Sunday
                return "Weekend (Markets Closed)"
                
            # Session times based on New York time
            if 0 <= current_hour < 3:
                return "Late US/Early Asian Session"
            elif 3 <= current_hour < 8:
                return "Asian Session"
            elif 8 <= current_hour < 12:
                return "European Session"
            elif 12 <= current_hour < 16:
                return "European/US Overlap Session"
            elif 16 <= current_hour < 20:
                return "US Session"
            else:  # 20-24
                return "Late US Session"
        except Exception as e:
            logger.error(f"Error analyzing session: {str(e)}")
            return "Unknown Session"
    
    def _load_available_signal_generators(self):
        """
        Load and register all available signal generators from the strategy module.
        """
        try:
            # Store all found signal generators
            self.available_signal_generators = {}
            
            # Try to import the main breakout reversal strategy
            try:
                # Try importing from strategy module first
                try:
                    from src.strategy import BreakoutReversalStrategy  # type: ignore # pyright: ignore[reportAttributeAccessIssue]
                    self.available_signal_generators["breakout_reversal"] = BreakoutReversalStrategy
                    logger.info("Successfully imported BreakoutReversalStrategy from strategy module")
                except ImportError:
                    # Fallback to direct import
                    from src.strategy.breakout_reversal_strategy import BreakoutReversalStrategy  # type: ignore # pyright: ignore[reportAttributeAccessIssue]
                    self.available_signal_generators["breakout_reversal"] = BreakoutReversalStrategy
                    logger.info("Imported BreakoutReversalStrategy directly from file")
            except ImportError as e:
                logger.warning(f"Could not import BreakoutReversalStrategy: {str(e)}")
            
            # Try to import other available strategies
            try:
                # Import strategy directory
                import src.strategy as strategy_module
                
                # Check if it has a __all__ attribute listing available strategies
                # pyright: ignore[reportAttributeAccessIssue]
                strategy_names = []
                try:
                    all_attr = getattr(strategy_module, '__all__', [])
                    if all_attr:
                        strategy_names = all_attr
                except AttributeError:
                    # Strategy module doesn't have __all__, try other methods to find strategies
                    pass
                
                # If no strategy names found, look for classes ending with 'Strategy'
                if not strategy_names:
                    for attr_name in dir(strategy_module):
                        if attr_name.endswith('Strategy') and attr_name != 'SignalGenerator':
                            strategy_names.append(attr_name)
                
                # Process the found strategy names
                for strategy_name in strategy_names:
                    if strategy_name == 'BreakoutReversalStrategy':
                        # Already imported
                        continue
                        
                    try:
                        # Get the strategy class dynamically
                        strategy_class = getattr(strategy_module, strategy_name)
                        
                        # Convert CamelCase to snake_case
                        snake_case_name = ''.join(['_'+c.lower() if c.isupper() else c for c in strategy_name]).lstrip('_')
                        
                        # Add it to the available strategies
                        self.available_signal_generators[snake_case_name] = strategy_class
                        logger.info(f"Loaded signal generator: {strategy_name}")
                    except Exception as strat_error:
                        logger.warning(f"Error loading strategy {strategy_name}: {str(strat_error)}")
            
            except Exception as dir_error:
                logger.warning(f"Could not scan strategy directory: {str(dir_error)}")
                
            # Log successful loading
            if self.available_signal_generators:
                logger.info(f"Loaded {len(self.available_signal_generators)} signal generators: {list(self.available_signal_generators.keys())}")
            else:
                logger.warning("No signal generators were loaded")
                
        except Exception as e:
            logger.error(f"Error loading signal generators: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _debug_data_structure(self, data, max_depth=2, current_depth=0):
        """
        Helper method to debug complex nested data structures.
        
        Args:
            data: The data structure to analyze
            max_depth: Maximum recursion depth
            current_depth: Current recursion depth
            
        Returns:
            str: A string representation of the data structure
        """
        if current_depth >= max_depth:
            return f"{type(data).__name__}(...)"
            
        if isinstance(data, dict):
            if not data:
                return "{}"
                
            result = "{\n"
            for k, v in list(data.items())[:5]:  # Limit to first 5 items
                v_repr = self._debug_data_structure(v, max_depth, current_depth + 1)
                result += f"  {'  ' * current_depth}{k!r}: {v_repr},\n"
            if len(data) > 5:
                result += f"  {'  ' * current_depth}... ({len(data) - 5} more items)\n"
            result += f"{'  ' * current_depth}}}"
            return result
            
        elif isinstance(data, (list, tuple)):
            if not data:
                return f"{type(data).__name__}[]"
                
            result = f"{type(data).__name__}[\n"
            for i, item in enumerate(data[:5]):  # Limit to first 5 items
                item_repr = self._debug_data_structure(item, max_depth, current_depth + 1)
                result += f"  {'  ' * current_depth}{item_repr},\n"
            if len(data) > 5:
                result += f"  {'  ' * current_depth}... ({len(data) - 5} more items)\n"
            result += f"{'  ' * current_depth}]"
            return result
            
        elif isinstance(data, pd.DataFrame):
            if data.empty:
                return "DataFrame(empty)"
                
            columns = ", ".join(str(col) for col in data.columns)
            return f"DataFrame(shape={data.shape}, columns=[{columns}])"
            
        elif hasattr(data, "__dict__"):
            # For objects with __dict__ attribute
            attrs = vars(data)
            if not attrs:
                return f"{type(data).__name__}()"
                
            result = f"{type(data).__name__}(\n"
            for k, v in list(attrs.items())[:5]:
                if not k.startswith("_"):  # Skip private attributes
                    v_repr = self._debug_data_structure(v, max_depth, current_depth + 1)
                    result += f"  {'  ' * current_depth}{k}={v_repr},\n"
            if len(attrs) > 5:
                result += f"  {'  ' * current_depth}... ({len(attrs) - 5} more attributes)\n"
            result += f"{'  ' * current_depth})"
            return result
            
        else:
            # For primitive types
            return repr(data)[:100] + "..." if len(repr(data)) > 100 else repr(data)
    
    