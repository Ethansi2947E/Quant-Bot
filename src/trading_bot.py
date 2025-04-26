import asyncio
import traceback
import pytz
import time
import MetaTrader5 as mt5  # Add MetaTrader5 import
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Type, Optional, Set
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
        
        # Log which MT5Handler instance we're using
        if self.mt5_handler:
            logger.info(f"SignalGenerator {self.name} using MT5Handler instance: {id(self.mt5_handler)}")
        else:
            logger.warning(f"No MT5Handler passed to {self.name} - this might cause connection issues")
        
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
        
        Args:
            market_data: Dictionary of market data by symbol and timeframe
            symbol: Symbol to generate signals for
            timeframe: Timeframe to use
            
        Returns:
            List of signal dictionaries
        """
        logger.warning(f"Base generate_signals method called for {self.name}. This should be overridden.")
        return []
        
    async def close(self):
        """
        Perform cleanup operations when the signal generator is no longer needed.
        """
        logger.debug(f"Base close method called for {self.name}")
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
        
        # Load and merge with default config if not provided or empty
        if not self.config:
            logger.info("No config provided, loading default configuration from config.py")
            from config.config import TRADING_CONFIG, TELEGRAM_CONFIG, MT5_CONFIG, SESSION_CONFIG
            
            self.config = {
                "trading": TRADING_CONFIG,
                "telegram": TELEGRAM_CONFIG,
                "mt5": MT5_CONFIG,
                "session": SESSION_CONFIG,
            }
        
        # Extract commonly used config sections
        self.trading_config = self.config.get("trading", {})
        
        # DEBUG: Check if trading_config is empty and try to fix it
        if not self.trading_config:
            logger.warning("trading_config is empty after extraction from self.config. Attempting direct import...")
            try:
                from config.config import TRADING_CONFIG
                # Don't try to modify the config object, just use it
                self.trading_config = TRADING_CONFIG
                logger.info(f"Directly loaded TRADING_CONFIG with {len(TRADING_CONFIG.keys()) if isinstance(TRADING_CONFIG, dict) else 0} keys")
                # Don't try to modify self.config if it's read-only
                # Create a new dictionary instead
                try:
                    self.config = {
                        "trading": TRADING_CONFIG,
                        "telegram": self.config.get("telegram", {}),
                        "mt5": self.config.get("mt5", {}),
                        "session": self.config.get("session", {})
                    }
                    logger.info("Successfully created new config dictionary")
                except Exception as config_err:
                    logger.warning(f"Could not create new config dictionary: {str(config_err)}")
                    # Continue with self.trading_config set properly
            except Exception as e:
                logger.error(f"Failed to directly load TRADING_CONFIG: {str(e)}")
        else:
            logger.info(f"Loaded trading_config with {len(self.trading_config.keys())} keys")
        
        self.telegram_config = self.config.get("telegram", {})
        self.mt5_config = self.config.get("mt5", {})
        
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
        
        # Set the risk_manager on the MT5Handler
        self.mt5_handler.risk_manager = self.risk_manager
        logger.info(f"Set RiskManager on MT5Handler to ensure proper position sizing")
        
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
        # Explicitly use the trading_config value, with a default of False for position additions
        self.allow_position_additions = self.trading_config.get('allow_position_additions', False)
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
        
        logger.info("TradingBot initialized with enhanced multi-timeframe analysis capabilities")

        # Create or use provided MT5Handler
        if self.config.get('mt5_handler'):
            self.mt5_handler = self.config.get('mt5_handler')
            logger.info(f"Using provided MT5Handler instance")
        else:
            self.mt5_handler = MT5Handler()
            logger.info(f"Created new MT5Handler instance")
            
        # Set the risk_manager on the MT5Handler
        self.mt5_handler.risk_manager = self.risk_manager
        logger.info(f"Set RiskManager on MT5Handler")

        # Initialize MT5 connection (if not already connected)
        if not self.mt5_handler.is_connected():
            self.initialize_mt5()

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
            # Force direct import of both strategies for reliability
            try:
                from src.strategy.breakout_reversal_strategy import BreakoutReversalStrategy
                logger.info("Force imported BreakoutReversalStrategy directly from file")
                
                from src.strategy.confluence_price_action_strategy import ConfluencePriceActionStrategy
                logger.info("Force imported ConfluencePriceActionStrategy directly from file")
            except ImportError as import_e:
                logger.error(f"Error force importing strategy classes: {str(import_e)}")
                
            # Try importing from strategy module first
            try:
                from src.strategy import BreakoutReversalStrategy  # type: ignore # pyright: ignore[reportAttributeAccessIssue]
                logger.info("Successfully imported BreakoutReversalStrategy from strategy module")
            except ImportError:
                # Fallback to direct import
                from src.strategy.breakout_reversal_strategy import BreakoutReversalStrategy  # type: ignore # pyright: ignore[reportAttributeAccessIssue]
                logger.info("Imported BreakoutReversalStrategy directly from file")
            
            # Create a dictionary of available signal generators
            self.available_signal_generators = {"breakout_reversal": BreakoutReversalStrategy}
            # Import and register ConfluencePriceActionStrategy if available
            try:
                from src.strategy.confluence_price_action_strategy import ConfluencePriceActionStrategy  # type: ignore
                self.available_signal_generators["confluence_price_action"] = ConfluencePriceActionStrategy
            except ImportError:
                logger.debug("ConfluencePriceActionStrategy not found or failed import")
            
            # Initialize the signal generators list based on config
            self.signal_generators = []
            signal_generator_names = self.trading_config.get("signal_generators", ["breakout_reversal"])
            logger.info(f"Loaded signal_generator_names from trading_config: {signal_generator_names}")
            
            # Debug: See what's in the config
            if not self.trading_config:
                logger.warning("trading_config is STILL empty, attempting fresh load directly from config.py")
                try:
                    # Import directly from config.py 
                    from config.config import TRADING_CONFIG
                    signal_generator_names = TRADING_CONFIG.get("signal_generators", ["breakout_reversal"])
                    logger.info(f"Loaded signal_generators directly from config.py: {signal_generator_names}")
                    
                    # Fix the trading_config
                    self.trading_config = TRADING_CONFIG
                    self.config["trading"] = TRADING_CONFIG
                except Exception as e:
                    logger.error(f"Failed to load TRADING_CONFIG directly: {str(e)}")
            
            # Always load from config - our fix for empty config above ensures this works
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
                # No matching strategy for configured names; warn and do not fallback to breakout
                logger.error(
                    f"No signal generators matched config {signal_generator_names}. "
                    "Ensure 'signal_generators' lists valid keys: "
                    f"{list(self.available_signal_generators.keys())}"
                )
                # Optionally default to first available if desired:
                # name, cls = next(iter(self.available_signal_generators.items()))
                # self.signal_generators.append(cls(mt5_handler=self.mt5_handler, risk_manager=self.risk_manager))
                # logger.warning(f"Defaulting to {name} strategy")
                
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

    async def start(self):
        """Start the trading bot."""
        # Create a future object that will be returned
        self.shutdown_future = asyncio.Future()
        
        try:
            self.running = True
            logger.info("Starting trading bot...")
            
            # Initialize MT5 connection if needed
            if not self.mt5_handler.connected:
                logger.info("Initializing MT5 connection...")
                if not self.initialize_mt5():
                    logger.error("Failed to initialize MT5 connection")
                    self.running = False
                    self.shutdown_future.set_result(False)
                    return self.shutdown_future
            
            # Make sure we have our symbols list
            if not self.symbols:
                self._load_symbols_from_config()
            
            # Initialize the bot components
            logger.info("Initializing bot components...")
            await self.initialize()
            
            # Set up Telegram commands
            if self.telegram_bot:
                await self.register_telegram_commands()
                logger.info("Telegram commands registered")
            
            # Start the notification task
            self.notification_task = asyncio.create_task(self.send_notification_task())
            
            # Set up the main trading tasks
            self.main_task = asyncio.create_task(self.main_loop())
            self.trade_monitor_task = asyncio.create_task(self._monitor_trades_loop())
            self.shutdown_monitor_task = asyncio.create_task(self._monitor_shutdown())
            
            # Send startup notification
            startup_message = f"🚀 Trading Bot started\n\n"
            startup_message += f"📊 Symbols: {', '.join(self.symbols)}\n"
            startup_message += f"🧠 Signal Generator: {self.signal_generators[0].__class__.__name__ if self.signal_generators else 'None'}\n"
            startup_message += f"⚙️ Trading Enabled: {'✅' if self.trading_enabled else '❌'}"
            
            # Send notification directly through signal processor instead of using the wrapper method
            if self.signal_processor:
                await self.signal_processor._notify_trade_action(startup_message)
            self.startup_notification_sent = True
            
            logger.info("Trading bot started successfully")
            # Don't set the future result now; it will be set when the bot shuts down
            return self.shutdown_future
        
        except Exception as e:
            logger.error(f"Error starting trading bot: {str(e)}")
            logger.error(traceback.format_exc())
            self.running = False
            # Set the future with an exception in case of error
            self.shutdown_future.set_exception(e)
            return self.shutdown_future

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
            
        # Start the Telegram bot if it's not running
        if not hasattr(self.telegram_bot, 'is_running') or not self.telegram_bot.is_running:
            logger.info("Telegram bot not running, starting it now...")
            try:
                await self.telegram_bot.start()
                logger.info("Telegram bot started successfully")
            except Exception as e:
                logger.error(f"Failed to start Telegram bot: {str(e)}")
                logger.error(traceback.format_exc())
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
            # Instead of using change_signal_generator, directly use _init_signal_generators
            generator_class = self.available_signal_generators[generator_name]
            self.signal_generator_class = generator_class
            self._init_signal_generators(generator_class)
            
            # Send notification via Telegram
            if self.telegram_bot and hasattr(self.telegram_bot, 'is_running') and self.telegram_bot.is_running:
                try:
                    # Create a task to send notification asynchronously
                    async def send_notification_task():
                        try:
                            # Check for null safety once more inside the task
                            if self.telegram_bot and hasattr(self.telegram_bot, 'send_notification'):
                                await self.telegram_bot.send_notification(
                                    f"Signal generator changed to {generator_class.__name__}"
                                )
                            else:
                                logger.warning("Cannot send notification: telegram_bot missing or send_notification not available")
                        except Exception as e:
                            logger.error(f"Error sending notification: {str(e)}")
                    
                    # Create and run the task in the background
                    asyncio.create_task(send_notification_task())
                except Exception as e:
                    logger.warning(f"Could not create notification task: {str(e)}")
                
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
        """Main operational loop for the trading bot."""
        
        # Log the start of the main loop
        logger.info("Starting main trading loop")
        
        while not self.shutdown_requested:
            try:
                if self.trading_enabled:
                    # Simple fetch and generate cycle
                    await self.simplified_analysis_cycle()
                
                    # Manage existing trades (trailing stops, etc)
                    await self.manage_open_trades()
                
                # Update performance metrics
                await self.update_performance_metrics()
                
                # Fixed interval sleep between cycles - more predictable behavior
                await asyncio.sleep(120)  # 2 minute interval between analyses
                
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(5)  # Error recovery delay

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
        process_start = time.time()
        logger.info(f"⭐ SIGNAL PROCESSING START: Received {len(signals)} signals")
        
        try:
            if not self.trading_enabled:
                logger.info("🚫 Trading is disabled, skipping signal processing")
                return

            logger.info(f"💼 Processing {len(signals)} trading signals with trading enabled: ✅")
            logger.info(f"🔍 Signal generators in use: {[gen.__class__.__name__ for gen in self.signal_generators]}")
            
            # Log signal details
            if signals:
                for i, signal in enumerate(signals):
                    symbol = signal.get('symbol', 'Unknown')
                    direction = signal.get('direction', 'Unknown')
                    confidence = signal.get('confidence', 0)
                    logger.info(f"📝 Signal #{i+1}: {symbol} {direction} with confidence {confidence:.2f}")
            
            # Ensure signal_processor has the correct telegram_bot instance
            if hasattr(self, 'signal_processor') and self.signal_processor:
                # Make sure telegram_bot is set in signal_processor without reinitializing it
                if self.telegram_bot and hasattr(self.telegram_bot, 'is_running'):
                    # Only set the Telegram bot, don't try to initialize it again
                    self.signal_processor.set_telegram_bot(self.telegram_bot)
                    logger.debug("Updated SignalProcessor with current TelegramBot instance")
                
                # Process the signals
                logger.info("🚀 Delegating signal processing to SignalProcessor")
                processor_start = time.time()
                processed_signals = await self.signal_processor.process_signals(signals)
                processor_time = time.time() - processor_start
                logger.info(f"✅ SignalProcessor completed processing in {processor_time:.2f}s")
                
                # Log processing results
                if processed_signals:
                    executed = sum(1 for s in processed_signals if s.get('status') == 'executed')
                    skipped = sum(1 for s in processed_signals if s.get('status') == 'skipped')
                    failed = sum(1 for s in processed_signals if s.get('status') in ['failed', 'error', 'invalid'])
                    logger.info(f"📊 Processing summary: {executed} executed, {skipped} skipped, {failed} failed")
            else:
                logger.error("❌ No signal processor available to process signals")
                
        except Exception as e:
            logger.error(f"❌ Error in process_signals: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            if self.telegram_bot and hasattr(self.telegram_bot, 'send_error_alert'):
                try:
                    await self.telegram_bot.send_error_alert(f"Error processing signals: {str(e)}")
                except Exception as telegram_error:
                    logger.error(f"Failed to send error alert: {str(telegram_error)}")
        
        process_total_time = time.time() - process_start
        logger.info(f"🏁 Signal processing completed in {process_total_time:.2f}s")

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
            tick_time = latest_tick.get('time', 0)
            time_diff = now - tick_time
            
            # Consider market open if tick is recent (within last 1 minute)
            # Reduced from 5 minutes to be more responsive to market conditions
            MAX_TICK_AGE = 60  # 60 seconds
            
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
                await asyncio.sleep(60)
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

    async def stop(self):
        """Stop the trading bot and clean up resources."""
        logger.info("Stopping TradingBot...")
        
        # Set shutdown flags
        self.shutdown_requested = True
        self.should_stop = True
        
        try:
            # Stop real-time monitoring
            self.real_time_monitoring_enabled = False
            
            # Cancel the data fetch task if it's running
            if hasattr(self, 'data_fetch_task') and self.data_fetch_task:
                logger.info("Cancelling data fetch task...")
                self.data_fetch_task.cancel()
                try:
                    await self.data_fetch_task
                except asyncio.CancelledError:
                    logger.info("Data fetch task cancelled")
                    
            # Cancel the queue processor task if it's running
            if hasattr(self, 'queue_processor_task') and self.queue_processor_task:
                logger.info("Cancelling analysis queue processor...")
                self.queue_processor_task.cancel()
                try:
                    await self.queue_processor_task
                except asyncio.CancelledError:
                    logger.info("Analysis queue processor cancelled")
            
            # Rest of shutdown code...
            
            # Close positions if configured to do so on shutdown
            if self.close_positions_on_shutdown:
                logger.info("Closing all positions before shutdown")
                try:
                    # Get a list of positions to close
                    positions = self.mt5_handler.get_open_positions()
                    
                    if positions:
                        logger.info(f"Found {len(positions)} positions to close")
                        for pos in positions:
                            result = self.mt5_handler.close_position(pos["ticket"])
                            logger.info(f"Position {pos['ticket']} close result: {result}")
                    else:
                        logger.info("No open positions to close")
                except Exception as e:
                    logger.error(f"Error closing positions: {str(e)}")
            
            # Don't shutdown MT5 handler automatically - it can be reused
            # and shutting it down can cause problems with other operations
            
            # Notify on telegram if enabled
            if hasattr(self, 'telegram_bot') and self.telegram_bot:
                await self.telegram_bot.send_message("Trading bot shutting down.")
                
            logger.info("Trading bot stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping trading bot: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    async def initialize(self):
        """Initialize the TradingBot and all components."""
        try:
            # Initialize the terminal first
            logger.info("Initializing trading bot")
            self.stop_requested = False
            
            # Create MT5Handler if not already created
            if not hasattr(self, 'mt5_handler') or not self.mt5_handler:
                logger.info("Creating MT5Handler instance")
                self.mt5_handler = MT5Handler()
                # Initialize connection to MT5
                mt5_initialized = self.mt5_handler.initialize()
                if not mt5_initialized:
                    logger.error("Failed to initialize MT5 connection")
                    return False
                logger.info("MT5 Handler initialized and connected")
            else:
                logger.info("Using existing MT5Handler instance")
                # Ensure MT5 is connected
                if not self.mt5_handler.is_connected():
                    logger.info("Reconnecting existing MT5Handler")
                    mt5_initialized = self.mt5_handler.initialize()
                    if not mt5_initialized:
                        logger.error("Failed to reconnect existing MT5Handler")
                        return False
                    logger.info("Existing MT5Handler reconnected successfully")
            
            # Initialize and start Telegram bot if available
            if self.telegram_bot:
                logger.info("Starting Telegram bot...")
                try:
                    await self.telegram_bot.start()
                    logger.info("Telegram bot started successfully")
                except Exception as e:
                    logger.error(f"Failed to start Telegram bot: {str(e)}")
                    logger.error(traceback.format_exc())
                    # Continue with initialization even if Telegram fails
            
            # Create other components that depend on MT5Handler
            # Initialize risk manager
            self.risk_manager = RiskManager()
            
            # Initialize Signal Processor with the shared MT5Handler
            self.signal_processor = SignalProcessor(
                mt5_handler=self.mt5_handler,
                risk_manager=self.risk_manager,
                telegram_bot=self.telegram_bot,
                config=self.config
            )
            logger.info("Signal processor initialized with shared MT5Handler")
            
            # Initialize Position Manager with the shared MT5Handler
            self.position_manager = PositionManager(
                mt5_handler=self.mt5_handler,
                telegram_bot=self.telegram_bot
            )
            logger.info("Position manager initialized with shared MT5Handler")
            
            # Initialize signal generators with the shared MT5Handler
            await self._initialize_signal_generators()
            logger.info("Signal generators initialized with shared MT5Handler")
            
            # The rest of the initialization
            # ... existing code ...
        
        except Exception as e:
            logger.error(f"Error during TradingBot initialization: {str(e)}")
            logger.error(traceback.format_exc())
            return False
            
        return True

    async def _initialize_signal_generators(self):
        """Initialize all active signal generators."""
        try:
            # Set up signal generators
            active_generators = []
            
            # Get active generator names from config
            active_generator_names = self.trading_config.get("signal_generators", ["breakout_reversal"])
            
            # If we have a specific signal generator class passed, only use that one
            if self.signal_generator_class:
                # Ensure we're passing the SAME MT5Handler instance
                sg = self.signal_generator_class(
                    mt5_handler=self.mt5_handler,  # Use the shared instance
                    risk_manager=self.risk_manager
                )
                await sg.initialize()
                self.signal_generators = [sg]
                logger.info(f"Initialized signal generator: {sg.__class__.__name__} with shared MT5Handler")
                return
            
            # Load generators from the strategies directory
            self._load_available_signal_generators()
            
            # Initialize active generators from config
            for generator_name in active_generator_names:
                if generator_name in self.available_signal_generators:
                    generator_class = self.available_signal_generators[generator_name]
                    # Ensure we're passing the SAME MT5Handler instance
                    generator = generator_class(
                        mt5_handler=self.mt5_handler,  # Use the shared instance
                        risk_manager=self.risk_manager
                    )
                    self.signal_generators.append(generator)
                    logger.info(f"Initialized signal generator: {generator_name} ({generator.__class__.__name__})")
                else:
                    logger.warning(f"Unknown signal generator: {generator_name}")
            
            # Set the active generators
            self.signal_generators = active_generators
            
            # Note: We no longer need to pass the MT5Handler to the generators here,
            # as they should be created with the shared instance
            
        except Exception as e:
            logger.error(f"Error initializing signal generators: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _load_symbols_from_config(self):
        """Load trading symbols from configuration."""
        try:
            # Debug the available config
            logger.debug(f"Trading config keys: {list(self.trading_config.keys())}")
            
            # Try to load symbols from trading_config
            if 'symbols' in self.trading_config and isinstance(self.trading_config['symbols'], list):
                logger.debug(f"Found symbols list in trading_config with {len(self.trading_config['symbols'])} items")
                
                if len(self.trading_config['symbols']) > 0:
                    # In the new format, symbols are direct strings in a list
                    self.symbols = self.trading_config['symbols']
                    logger.info(f"Loaded symbols from trading_config: {self.symbols}")
            else:
                logger.warning(f"No 'symbols' key found in trading_config or not a list. Available keys: {list(self.trading_config.keys())}")
                
                # Try to directly import from config.py as fallback
                try:
                    from config.config import TRADING_CONFIG
                    if 'symbols' in TRADING_CONFIG and isinstance(TRADING_CONFIG['symbols'], list):
                        self.symbols = TRADING_CONFIG['symbols']
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
                # Try importing the new confluence strategy
                try:
                    from src.strategy.confluence_price_action_strategy import ConfluencePriceActionStrategy  # type: ignore
                    self.available_signal_generators["confluence_price_action"] = ConfluencePriceActionStrategy
                except ImportError:
                    logger.warning("ConfluencePriceActionStrategy not found in strategy module")
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
    
    async def simplified_analysis_cycle(self):
        """
        Directly fetch data for strategy timeframes and generate signals.
        This is the core trading functionality that processes each signal generator.
        """
        
        analysis_start = time.time()
        logger.info("🔄 Starting simplified analysis cycle")
        
        # Check if we have signal generators
        if not self.signal_generators:
            # This is unexpected - try to re-initialize them if needed
            logger.warning("No signal generators found in simplified_analysis_cycle! Attempting to recover...")
            
            try:
                # Check if we can load them from the module
                from config.config import TRADING_CONFIG
                signal_generator_names = TRADING_CONFIG.get("signal_generators", ["confluence_price_action"])
                logger.info(f"Re-loading signal generators from config: {signal_generator_names}")
                
                # Load available signal generators
                if not hasattr(self, 'available_signal_generators') or not self.available_signal_generators:
                    self._load_available_signal_generators()
                
                # Create new instances as needed
                for generator_name in signal_generator_names:
                    if generator_name in self.available_signal_generators:
                        generator_class = self.available_signal_generators[generator_name]
                        generator = generator_class(
                            mt5_handler=self.mt5_handler,
                            risk_manager=self.risk_manager
                        )
                        if not self.signal_generators:
                            self.signal_generators = []
                        self.signal_generators.append(generator)
                        logger.info(f"Recovered signal generator: {generator_name} ({generator.__class__.__name__})")
                    else:
                        logger.warning(f"Unknown signal generator during recovery: {generator_name}")
            except Exception as e:
                logger.error(f"Failed to recover signal generators: {str(e)}")
                
            # If still no generators, abort
            if not self.signal_generators:
                logger.error("No signal generators available even after recovery attempt. Cannot continue analysis.")
                return
        
        # Log how many generators we're using
        logger.info(f"Starting analysis with {len(self.signal_generators)} signal generators")
        
        # Fetch and process for each signal generator one by one
        for signal_generator in self.signal_generators:
            generator_start = time.time()
            generator_name = signal_generator.__class__.__name__
            logger.info(f"⏳ Starting analysis with {generator_name}")
            
            # Get the specific timeframes required by this strategy
            # Handle missing attributes gracefully with defaults
            primary_tf = getattr(signal_generator, 'primary_timeframe', 'M15')
            higher_tf = getattr(signal_generator, 'higher_timeframe', 'H1')
            
            # Get all required timeframes from the strategy if available
            required_timeframes = getattr(signal_generator, 'required_timeframes', [primary_tf, higher_tf])
            
            logger.debug(f"Fetching data for strategy with timeframes: {required_timeframes}")
            
            # Process each symbol one by one for immediate trade execution
            for symbol in self.symbols:
                symbol_start = time.time()
                logger.info(f"Analyzing {symbol} with {generator_name}")
                
                # Fetch market data just for this symbol
                market_data = {symbol: {}}
                data_fetch_start = time.time()
                
                for tf in required_timeframes:
                    # Use the _ensure_data_available method which honors strategy lookback requirements
                    logger.info(f"Directly fetching {tf} data for {symbol}")
                    success = await self._ensure_data_available(symbol, tf)
                    
                    if success and symbol in self.market_data_cache and tf in self.market_data_cache[symbol]:
                        market_data[symbol][tf] = self.market_data_cache[symbol][tf]
                        data_length = len(market_data[symbol][tf]) if hasattr(market_data[symbol][tf], '__len__') else 0
                        logger.debug(f"Successfully fetched {data_length} bars for {symbol}/{tf}")
                    else:
                        logger.warning(f"Failed to fetch data for {symbol}/{tf}")
                
                data_fetch_time = time.time() - data_fetch_start
                logger.debug(f"📊 Data fetching for {symbol} completed in {data_fetch_time:.2f}s")
                
                # Generate and process signals for this symbol immediately
                try:
                    signal_gen_start = time.time()
                    # Check if this is the BreakoutReversalStrategy to force trendline detection
                    is_breakout_strategy = signal_generator.__class__.__name__ == 'BreakoutReversalStrategy'
                    
                    if is_breakout_strategy:
                        logger.info(f"🔍 Analyzing {symbol} with BreakoutReversalStrategy (immediate processing)")
                        new_signals = await signal_generator.generate_signals(
                            market_data=market_data, 
                            debug_visualize=True, 
                            skip_plots=True,
                            force_trendlines=True,
                            process_immediately=True  # Enable immediate processing
                        )
                    else:
                        new_signals = await signal_generator.generate_signals(
                            market_data=market_data,
                            process_immediately=True  # Enable immediate processing
                        )
                    
                    signal_gen_time = time.time() - signal_gen_start
                    
                    if new_signals:
                        logger.info(f"✅ Generated {len(new_signals)} signals for {symbol} in {signal_gen_time:.2f}s")
                        # Process signals immediately instead of adding to self.signals list
                        logger.info(f"⚡ Processing {len(new_signals)} signals immediately for {symbol}")
                        await self.process_signals(new_signals)
                    else:
                        logger.info(f"📭 No signals generated for {symbol} in {signal_gen_time:.2f}s")
                        
                    symbol_total_time = time.time() - symbol_start
                    logger.info(f"Symbol {symbol} analysis completed in {symbol_total_time:.2f}s")
                    
                except Exception as e:
                    logger.error(f"Error generating signals for {symbol}: {str(e)}")
                    logger.error(traceback.format_exc())
            
            generator_time = time.time() - generator_start
            logger.info(f"⏱️ Analysis with {generator_name} completed in {generator_time:.2f}s")
        
        analysis_time = time.time() - analysis_start
        logger.info(f"🏁 Simplified analysis cycle completed in {analysis_time:.2f}s")

    async def send_notification_task(self):
        """Periodically send status notifications to the Telegram bot."""
        try:
            while self.running and not self.shutdown_requested:
                # Only send periodic notifications if telegram bot is available
                if self.telegram_bot and hasattr(self.telegram_bot, 'is_running') and self.telegram_bot.is_running:
                    # Get current status information
                    positions = self.mt5_handler.get_open_positions()
                    position_count = len(positions)
                    
                    # Only send notification if there are open positions
                    if position_count > 0:
                        try:
                            # Calculate floating profit
                            total_profit = sum(pos["profit"] for pos in positions)
                            
                            # Format message
                            message = f"📊 Status Update\n"
                            message += f"Open Positions: {position_count}\n"
                            message += f"Total P/L: {total_profit:.2f}\n"
                            
                            # Send notification
                            if hasattr(self.telegram_bot, 'send_notification'):
                                await self.telegram_bot.send_notification(message)
                        except Exception as e:
                            logger.error(f"Error sending status notification: {str(e)}")
                
                # Wait for next notification interval (4 hours)
                await asyncio.sleep(4 * 60 * 60)
        except asyncio.CancelledError:
            logger.info("Notification task cancelled")
        except Exception as e:
            logger.error(f"Error in notification task: {str(e)}")
            logger.error(traceback.format_exc())
    
    async def _ensure_data_available(self, symbol: str, timeframe: str) -> bool:
        """
        Make sure data is available for the given symbol and timeframe.
        Directly fetches from MT5 if needed.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe (e.g., 'M15', 'H1')
            
        Returns:
            True if data is available, False otherwise
        """
        try:
            # Check if data is already available in cache
            if (symbol in self.market_data_cache and 
                timeframe in self.market_data_cache.get(symbol, {}) and
                self.market_data_cache[symbol][timeframe] is not None and
                not (hasattr(self.market_data_cache[symbol][timeframe], 'empty') and 
                     self.market_data_cache[symbol][timeframe].empty)):
                logger.debug(f"Data already available for {symbol}/{timeframe}")
                return True
            
            # Default lookback periods based on timeframe - significantly reduced from previous values
            default_bars = {
                "M1": 120,
                "M5": 120,
                "M15": 120,
                "H1": 120,
                "H4": 60,
                "D1": 30
            }.get(timeframe, 100)  # Default to 100 bars if not specified
            
            # Check for specific lookback requirements from signal generators
            max_lookback = default_bars
            
            # Look for strategy-specific lookback periods
            for gen in self.signal_generators:
                logger.debug(f"Checking lookback settings for {gen.__class__.__name__}")
                
                # For debugging, log the attributes of the generator
                if hasattr(gen, 'timeframe_settings'):
                    logger.info(f"Found timeframe_settings attribute on {gen.__class__.__name__}")
                    tf_keys = list(gen.timeframe_settings.keys()) if isinstance(gen.timeframe_settings, dict) else "Not a dict"
                    logger.info(f"Available timeframes in settings: {tf_keys}")
                    
                    if isinstance(gen.timeframe_settings, dict) and timeframe in gen.timeframe_settings:
                        logger.info(f"Found settings for {timeframe} in {gen.__class__.__name__}")
                        tf_setting_keys = list(gen.timeframe_settings[timeframe].keys())
                        logger.info(f"Keys in {timeframe} settings: {tf_setting_keys}")
                
                # Check for lookback_periods dictionary attribute
                if hasattr(gen, 'lookback_periods') and isinstance(gen.lookback_periods, dict) and timeframe in gen.lookback_periods:
                    gen_lookback = gen.lookback_periods[timeframe]
                    max_lookback = max(max_lookback, gen_lookback)
                    logger.info(f"✅ Using lookback_periods dict for {timeframe}: {gen_lookback} bars from {gen.__class__.__name__}")
                
                # Check for timeframe-specific settings in the strategy (for BreakoutReversalStrategy)
                elif hasattr(gen, 'timeframe_settings') and isinstance(gen.timeframe_settings, dict) and timeframe in gen.timeframe_settings:
                    tf_settings = gen.timeframe_settings[timeframe]
                    if 'lookback_period' in tf_settings:
                        tf_lookback = tf_settings['lookback_period']
                        max_lookback = max(max_lookback, tf_lookback)
                        logger.info(f"✅ Using timeframe_settings for {timeframe}: lookback_period={tf_lookback} bars from {gen.__class__.__name__}")
                
                # Check for plain lookback_period attribute on the strategy
                elif hasattr(gen, 'lookback_period'):
                    lookback = gen.lookback_period
                    max_lookback = max(max_lookback, lookback)
                    logger.info(f"✅ Using direct lookback_period: {lookback} bars from {gen.__class__.__name__}")
            
            # Add a buffer (20%) to ensure we have enough data for calculations
            bars_required = int(max_lookback * 1.2)
            
            # Enforce minimum bars for reliable analysis
            bars_required = max(bars_required, 30)
            
            # Fetch the data directly from MT5
            logger.info(f"Direct fetch: Getting {bars_required} {timeframe} bars for {symbol}")
            data = self.mt5_handler.get_market_data(symbol, timeframe, bars_required)
            
            if data is None or (hasattr(data, 'empty') and data.empty):
                logger.warning(f"Failed to fetch data for {symbol}/{timeframe}")
                return False
            
            # If data is a DataFrame, check if it has enough bars
            data_length = len(data) if hasattr(data, '__len__') else 0
            if data_length < 10:  # Require at least 10 bars minimum
                logger.warning(f"Not enough data for {symbol}/{timeframe}, got only {data_length} bars")
                return False
                
            # Ensure we have a place to store the data
            if symbol not in self.market_data_cache:
                self.market_data_cache[symbol] = {}
            
            # Store the data in cache
            self.market_data_cache[symbol][timeframe] = data
            logger.info(f"Successfully fetched and cached {data_length} bars for {symbol}/{timeframe}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error ensuring data availability for {symbol}/{timeframe}: {str(e)}")
            return False
    
    