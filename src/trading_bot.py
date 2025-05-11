import asyncio
import traceback
import pytz
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Type, Optional, Set

from loguru import logger
import copy

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
            from config.config import TRADING_CONFIG, TELEGRAM_CONFIG, MT5_CONFIG
            self.config = {
                "trading": TRADING_CONFIG,
                "telegram": TELEGRAM_CONFIG,
                "mt5": MT5_CONFIG,
            }
        self.trading_config = self.config.get("trading", {})
        self.telegram_config = self.config.get("telegram", {})
        self.mt5_config = self.config.get("mt5", {})
        
        # Initialize market status tracking
        self.market_status = {}  # Track market open/closed status for each symbol
        
        # Initialize MT5 handler first (needed by other components)
        mt5_handler_candidate = self.config.get('mt5_handler')
        if isinstance(mt5_handler_candidate, MT5Handler):
            self.mt5_handler = mt5_handler_candidate
            logger.info(f"Using provided MT5Handler instance")
        else:
            self.mt5_handler = MT5Handler()
            logger.info(f"Created new MT5Handler instance")
        # Verify MT5 connection is working
        if self.mt5_handler is not None and not getattr(self.mt5_handler, 'connected', False):
            self.mt5_handler.initialize()
        self.mt5_connected = self.mt5_handler.connected
        # Initialize symbols list and state tracking variables
        self.symbols = []
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
        self.signal_generators = []
        self.latest_prices = {}  # Initialize the missing latest_prices dictionary
        
        # Set state tracking variables
        self.close_positions_on_shutdown = self.config.get('close_positions_on_shutdown', False)
        # Explicitly use the trading_config value, with a default of False for position additions
        self.allow_position_additions = self.trading_config.get('allow_position_additions', False)
        self.use_trailing_stop = self.config.get('use_trailing_stop', True)
        self.trading_enabled = self.config.get('trading_enabled', True)  # Enabled by default
        self.real_time_monitoring_enabled = self.config.get('real_time_monitoring_enabled', True)  # Enable real-time monitoring by default
        self.startup_notification_sent = False  # Flag to track startup notification
        self.stop_requested = False
        
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
        self.min_confidence = self.trading_config.get("min_confidence", 0.6)  # Default to 60% confidence
        
        # Trade management
        self.trailing_stop_enabled = True
        self.trailing_stop_data = {}  # Store trailing stop data for open positions
        
        logger.info("TradingBot initialized with enhanced multi-timeframe analysis capabilities")

        self.main_loop_task = None
        self._monitor_trades_task = None
        self.data_fetch_task = None
        self.queue_processor_task = None

        # Log config order at the very start
        logger.info(f"[TRACE INIT] Initial config signal_generators order: {self.config.get('trading', {}).get('signal_generators', 'NOT FOUND')}")
        # Log config order after extracting trading_config
        logger.info(f"[TRACE INIT] trading_config signal_generators order: {self.trading_config.get('signal_generators', 'NOT FOUND')}")

    # --- NEW: Tick listener/dispatcher ---
    async def tick_event_loop(self):
        """
        Background task: For each registered symbol, call update_on_tick every 2 seconds.
        Trigger ALL strategies in self.signal_generators for each symbol/timeframe.
        Adds detailed diagnostic logging for debugging.
        """
        import traceback as tb
        if not self.signal_generators:
            logger.error("No signal generators loaded! Cannot start tick_event_loop.")
            return
        # Log which strategies will be used
        logger.info(f"[TICK LOOP] Using {len(self.signal_generators)} strategies: {[s.__class__.__name__ for s in self.signal_generators]}")
        last_seen_candle = {}  # {(symbol, timeframe, strategy_name): last_datetime}
        while not getattr(self, 'shutdown_requested', False):
            try:
                logger.debug(f"tick_event_loop: symbols={self.symbols}, time={datetime.now()}")
                symbols = set(sym for (sym, tf) in getattr(self.data_manager, 'requirements', {}).keys())
                for symbol in symbols:
                    await self.data_manager.update_on_tick(symbol)
                
                # Iterate over all loaded strategies
                for strategy in self.signal_generators:
                    strategy_name = strategy.__class__.__name__
                    required_timeframes = getattr(strategy, 'required_timeframes', [])
                    primary_tf = getattr(strategy, 'primary_timeframe', required_timeframes[0] if required_timeframes else None)
                    
                    logger.info(f"[TICK LOOP] Processing strategy: {strategy_name}")
                    
                    if not required_timeframes:
                        logger.debug(f"Skipping {strategy_name}: required_timeframes is empty")
                        continue
                    if not primary_tf:
                        logger.debug(f"Skipping {strategy_name}: primary_timeframe is not set")
                        continue
                        
                    lookback_periods = getattr(strategy, 'lookback_periods', {})
                    default_lookback = getattr(strategy, 'lookback_period', 100)
                    
                    for symbol in self.symbols:
                        if symbol not in symbols:
                            logger.debug(f"Symbol {symbol} not in requirements, skipping for {strategy_name}.")
                            continue
                            
                        # Use a unique key for last seen candle per strategy
                        last_seen_key = (symbol, primary_tf, strategy_name)
                        last_time = self.data_manager.last_candle_time.get((symbol, primary_tf))
                        prev_time = last_seen_candle.get(last_seen_key)
                        
                        logger.debug(f"{strategy_name} {symbol}/{primary_tf}: last_time={last_time}, prev_time={prev_time}")
                        
                        if last_time is None:
                            logger.debug(f"{strategy_name} {symbol}/{primary_tf}: last_time is None, skipping.")
                            continue
                        if last_time == prev_time:
                            logger.debug(f"{strategy_name} {symbol}/{primary_tf}: last_time == prev_time ({last_time}), skipping.")
                            continue
                            
                        logger.info(f"Triggering {strategy_name} for {symbol}/{primary_tf} at {last_time}")
                        
                        # Gather all required data windows for this symbol/strategy
                        market_data = {symbol: {}}
                        missing_data = False
                        for tf in required_timeframes:
                            lb = lookback_periods.get(tf, default_lookback)
                            df = self.data_manager.get_data_window(symbol, tf, lb)
                            if df is None or df.empty:
                                logger.warning(f"Missing data for {symbol}/{tf} (needed for {strategy_name})")
                                missing_data = True
                                break  # Break if any required timeframe data is missing
                            market_data[symbol][tf] = df
                        
                        # Skip signal generation if data is missing
                        if missing_data:
                            logger.warning(f"Skipping signal generation for {strategy_name} on {symbol}/{primary_tf} due to missing data")
                            continue
                        
                        # Call generate_signals
                        try:
                            signals = await strategy.generate_signals(
                                market_data=copy.deepcopy(market_data),
                                symbol=symbol,
                                timeframe=primary_tf,
                                skip_plots=True  # Ensure plots are skipped during live trading
                            )
                            if signals:
                                logger.info(f"Strategy {strategy_name} generated {len(signals)} signals for {symbol}/{primary_tf}")
                            else:
                                logger.info(f"Strategy {strategy_name} returned no signals for {symbol}/{primary_tf}")
                            await self.process_signals(signals)
                        except Exception as e:
                            logger.error(f"Error in strategy {strategy_name} generate_signals: {str(e)}\n{tb.format_exc()}")
                        
                        # Update last seen for this specific strategy
                        last_seen_candle[last_seen_key] = last_time
                        
                # End of strategy loop
                await asyncio.sleep(2)
            
            except Exception as e:
                logger.error(f"Error in tick_event_loop: {str(e)}\n{tb.format_exc()}")
                await asyncio.sleep(2)

    def _normalize_strategy_key(self, key: str) -> str:
        """Normalize strategy key to handle different naming conventions."""
        # Convert to lowercase and remove any 'strategy' suffix
        key = key.lower()
        # Handle both underscore and camel case formats
        if '_strategy' in key:
            key = key.replace('_strategy', '')
        elif 'strategy' in key:
            key = key.replace('strategy', '')
        # Remove any remaining underscores
        key = key.replace('_', '')
        return key

    async def _initialize_signal_generators(self):
        logger.info(f"[TRACE] _initialize_signal_generators called. trading_config signal_generators: {self.trading_config.get('signal_generators', 'NOT FOUND')}")
        import traceback as tb
        logger.debug(f"[TRACE] Call stack for _initialize_signal_generators:\n{''.join(tb.format_stack(limit=5))}")
        try:
            # Get active generator names from config
            # Check if config is missing, if so load directly from config.py
            active_generator_names = self.trading_config.get("signal_generators")
            if active_generator_names is None:
                # Try to get directly from config.py
                try:
                    from config.config import TRADING_CONFIG
                    active_generator_names = TRADING_CONFIG.get("signal_generators", ["breakout_reversal"])
                    logger.info(f"Loaded signal generators directly from config.py: {active_generator_names}")
                    # Update trading_config for future use
                    self.trading_config["signal_generators"] = active_generator_names
                except Exception as e:
                    logger.error(f"Failed to import from config.py: {str(e)}")
                    active_generator_names = ["breakout_reversal"]
            
            # Fallback to default if still None
            if not active_generator_names:
                active_generator_names = ["breakout_reversal", "confluence_price_action"]
                logger.warning(f"Using default signal generators: {active_generator_names}")

            # Load generators from the strategies directory
            self._load_available_signal_generators()

            # Build a normalized lookup for available strategies
            normalized_available = {self._normalize_strategy_key(k): v for k, v in self.available_signal_generators.items()}
            # Initialize active generators from config
            for generator_name in active_generator_names:
                norm_key = self._normalize_strategy_key(generator_name)
                if norm_key in normalized_available:
                    generator_class = normalized_available[norm_key]
                    # Ensure we're passing the SAME MT5Handler instance
                    generator = generator_class(
                        mt5_handler=self.mt5_handler,  # Use the shared instance
                        risk_manager=self.risk_manager
                    )
                    self.signal_generators.append(generator)
                    logger.info(f"Initialized signal generator: {generator_name} ({generator.__class__.__name__})")
                    # --- NEW: Register data requirements with DataManager ---
                    required_timeframes = getattr(generator, 'required_timeframes', [])
                    lookback_periods = getattr(generator, 'lookback_periods', {})
                    default_lookback = getattr(generator, 'lookback_period', 100)
                    for symbol in self.symbols:
                        for tf in required_timeframes:
                            lookback = lookback_periods.get(tf, default_lookback)
                            self.data_manager.register_timeframe(symbol, tf, lookback)
                else:
                    logger.warning(f"Unknown signal generator: {generator_name} (normalized: {norm_key})")
            logger.info(f"[TRACE] active_generator_names used: {active_generator_names}")
            logger.info(f"[TRACE] Final signal_generators order: {[gen.__class__.__name__ for gen in self.signal_generators]}")
        except Exception as e:
            logger.error(f"Error initializing signal generators: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    async def start(self):
        """Start the trading bot."""
        # Create a future object that will be returned
        self.shutdown_future = asyncio.Future()
        
        try:
            self.running = True
            logger.info("Starting trading bot...")
            
            # Initialize MT5 connection if needed
            if self.mt5_handler is not None and not getattr(self.mt5_handler, 'connected', False):
                logger.info("Initializing MT5 connection...")
                if not self.mt5_handler.initialize():
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

            # Confirm signal generators are loaded
            if not self.signal_generators:
                logger.error("No signal generators loaded after initialization! Cannot start tick_event_loop.")
                raise RuntimeError("No signal generators loaded.")
            
            # Set up Telegram commands
            if self.telegram_bot:
                await self.register_telegram_commands()
                logger.info("Telegram commands registered")
            
            # Start the notification task
            self.notification_task = asyncio.create_task(self.send_notification_task())
            
            # --- REPLACE OLD MAIN LOOP WITH TICK EVENT LOOP ---
            self.main_task = asyncio.create_task(self.tick_event_loop())
            self.trade_monitor_task = asyncio.create_task(self._monitor_trades_loop())
            self.shutdown_monitor_task = asyncio.create_task(self._monitor_shutdown())
            
            # Send startup notification
            startup_message = (
                "🚀 <b>Trading Bot Started</b>\n"
                "\n"
                "<b>📊 Symbols:</b> <code>{symbols}</code>\n"
                "<b>🧠 Strategy:</b> <code>{strategy}</code>\n"
                "<b>⚙️ Trading Enabled:</b> {enabled}\n"
                "\n"
                "<b>ℹ️ Tip:</b> Use /start in this chat to view the Telegram command keyboard and available bot commands.\n"
                "\n"
                "<i>Happy trading! If you need help, type /help or use the keyboard.</i>"
            ).format(
                symbols=', '.join(self.symbols),
                strategy=(self.signal_generators[0].__class__.__name__ if self.signal_generators else 'None'),
                enabled='✅' if self.trading_enabled else '❌',
            )
            
            # Send notification directly through signal processor instead of using the wrapper method
            if self.signal_processor:
                await self.signal_processor._notify_trade_action(startup_message)
            self.startup_notification_sent = True
            
            # Add log to confirm config order and which strategy will be used as primary
            logger.info(f"[STARTUP] Config signal_generators order: {self.trading_config.get('signal_generators', [])}")
            logger.info(f"[STARTUP] Will use {self.signal_generators[0].__class__.__name__ if self.signal_generators else 'None'} as the default/primary strategy.")
            
            logger.info("Trading bot started successfully")
            # Don't set the future result now; it will be set when the bot shuts down
            logger.info(f"[TRACE START] signal_generators order at start: {[gen.__class__.__name__ for gen in self.signal_generators]}")
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
                if self.main_loop_task is not None and self.main_loop_task.done():
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
                if self._monitor_trades_task is not None and self._monitor_trades_task.done():
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
            # Instead of using change_signal_generator, directly use _initialize_signal_generators
            generator_class = self.available_signal_generators[generator_name]
            self.signal_generator_class = generator_class
            await self._initialize_signal_generators()
            
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
        account_info = self.mt5_handler.get_account_info() if self.mt5_handler is not None else {}
        
        # Get open positions
        positions = self.mt5_handler.get_open_positions() if self.mt5_handler is not None else []
        
        # Build status message
        status = f"🤖 Trading Bot Status\n{'='*20}\n"
        
        # Show trading state - use self.trading_enabled directly as it's the source of truth
        status += f"Trading Enabled: {'✅' if self.trading_enabled else '❌'}\n"
        status += f"Trailing Stop: {'✅' if self.trailing_stop_enabled else '❌'}\n"
        status += f"Position Additions: {'✅' if self.allow_position_additions else '❌'}\n"
        status += f"Close on Shutdown: {'✅' if self.close_positions_on_shutdown else '❌'}\n"
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
                if self.mt5_handler is not None:
                    result = self.mt5_handler.close_position(pos["ticket"])
                    logger.info(f"Position {pos['ticket']} close result: {result}")
                else:
                    logger.error("MT5 handler is not initialized, cannot close position.")
                status += f"- {pos['symbol']} {pos_type}: {pos['profit']}\n"
            
            if len(positions) > 5:
                status += f"...and {len(positions) - 5} more\n"
        
        return status

    async def _monitor_trades_loop(self):
        """Separate loop for monitoring trades more frequently."""
        logger.info("Starting trade monitoring loop")
        
        while self.running and not self.shutdown_requested:
            try:
                # Check for active positions first
                active_positions = self.mt5_handler.get_open_positions() if self.mt5_handler is not None else []
                
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
                symbols_to_check = self.symbols
                
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
            if self.data_fetch_task is not None and not self.data_fetch_task.done():
                logger.info("Cancelling data fetch task...")
                try:
                    await asyncio.wait([self.data_fetch_task])
                    logger.info("data_fetch_task cancelled successfully")
                except asyncio.CancelledError:
                    logger.info("data_fetch_task cancellation raised CancelledError")
            else:
                logger.warning("data_fetch_task not running")
            
            # Cancel the queue processor task if it's running
            if self.queue_processor_task is not None and not self.queue_processor_task.done():
                logger.info("Cancelling analysis queue processor...")
                try:
                    await asyncio.wait([self.queue_processor_task])
                    logger.info("queue_processor_task cancelled successfully")
                except asyncio.CancelledError:
                    logger.info("queue_processor_task cancellation raised CancelledError")
            
            # Rest of shutdown code...
            
            # Close positions if configured to do so on shutdown
            if self.close_positions_on_shutdown:
                logger.info("Closing all positions before shutdown")
                try:
                    # Get a list of positions to close
                    positions = self.mt5_handler.get_open_positions() if self.mt5_handler is not None else []
                    
                    if positions:
                        logger.info(f"Found {len(positions)} positions to close")
                        for pos in positions:
                            if self.mt5_handler is not None:
                                result = self.mt5_handler.close_position(pos["ticket"])
                                logger.info(f"Position {pos['ticket']} close result: {result}")
                            else:
                                logger.error("MT5 handler is not initialized, cannot close position.")
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
                if not getattr(self.mt5_handler, 'connected', False):
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
                
                # Try importing the new price action SR strategy
                try:
                    from src.strategy.price_action_sr_strategy import PriceActionSRStrategy
                    self.available_signal_generators["price_action_sr"] = PriceActionSRStrategy
                    logger.info("Successfully imported PriceActionSRStrategy")
                except ImportError as e:
                    logger.warning(f"PriceActionSRStrategy not found in strategy module: {str(e)}")
                
                # Try importing the confluence strategy
                try:
                    from src.strategy.confluence_price_action_strategy import ConfluencePriceActionStrategy  # type: ignore
                    self.available_signal_generators["confluence_price_action"] = ConfluencePriceActionStrategy
                    logger.info("Successfully imported ConfluencePriceActionStrategy")
                except ImportError as e:
                    logger.warning(f"ConfluencePriceActionStrategy not found in strategy module: {str(e)}")
            except ImportError as e:
                logger.warning(f"Could not import BreakoutReversalStrategy: {str(e)}")
            
            # Try to import other available strategies
            try:
                # Import strategy directory
                import src.strategy as strategy_module
                
                # Check if it has a __all__ attribute listing available strategies
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
                    if strategy_name in ['BreakoutReversalStrategy', 'ConfluencePriceActionStrategy']:
                        # Already imported
                        continue
                        
                    try:
                        # Get the strategy class dynamically
                        strategy_class = getattr(strategy_module, strategy_name)
                        
                        # Convert CamelCase to snake_case
                        snake_case_name = ''.join(['_'+c.lower() if c.isupper() else c for c in strategy_name]).lstrip('_')
                        # Remove 'strategy' suffix if present
                        snake_case_name = snake_case_name.replace('_strategy', '')
                        
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
    
    async def send_notification_task(self):
        """Periodically send status notifications to the Telegram bot."""
        try:
            while self.running and not self.shutdown_requested:
                # Only send periodic notifications if telegram bot is available
                if self.telegram_bot and hasattr(self.telegram_bot, 'is_running') and self.telegram_bot.is_running:
                    # Get current status information
                    positions = self.mt5_handler.get_open_positions() if self.mt5_handler is not None else []
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
    
    