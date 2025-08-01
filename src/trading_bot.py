import asyncio
import traceback
import pytz
import time
import importlib.util
import inspect
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Type, Optional

from loguru import logger
import copy
import pandas as pd

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
        # Load default configurations
        from config.config import TRADING_CONFIG as DEFAULT_TRADING_CONFIG
        from config.config import TELEGRAM_CONFIG as DEFAULT_TELEGRAM_CONFIG
        from config.config import MT5_CONFIG as DEFAULT_MT5_CONFIG

        passed_config = config or {}

        # Start with deep copies of defaults
        self.trading_config = copy.deepcopy(DEFAULT_TRADING_CONFIG)
        self.telegram_config = copy.deepcopy(DEFAULT_TELEGRAM_CONFIG)
        self.mt5_config = copy.deepcopy(DEFAULT_MT5_CONFIG)

        
        if passed_config:
            logger.info("External config provided, merging with defaults.")
            if "trading" in passed_config and isinstance(passed_config["trading"], dict):
                self.trading_config.update(passed_config["trading"])
            elif "trading" in passed_config:
                 logger.warning("External config 'trading' key is not a dictionary or is None. Using default trading_config values, not merging.")
            
            if "telegram" in passed_config and isinstance(passed_config["telegram"], dict):
                self.telegram_config.update(passed_config["telegram"])
            elif "telegram" in passed_config:
                 logger.warning("External config 'telegram' key is not a dictionary or is None. Using default telegram_config values, not merging.")

            if "mt5" in passed_config and isinstance(passed_config["mt5"], dict):
                self.mt5_config.update(passed_config["mt5"])
            elif "mt5" in passed_config:
                 logger.warning("External config 'mt5' key is not a dictionary or is None. Using default mt5_config values, not merging.")
        else:
            logger.info("No external config provided, using default configurations from config.py.")

        # self.config stores the original passed_config for direct access to other potential top-level keys like 'mt5_handler'
        self.config = passed_config
        
        # Initialize market status tracking
        self.market_status = {}  # Track market open/closed status for each symbol
        
        # Initialize MT5 handler first (needed by other components)
        mt5_handler_candidate = self.config.get('mt5_handler') # Check from original passed_config
        if isinstance(mt5_handler_candidate, MT5Handler):
            self.mt5_handler = mt5_handler_candidate
            logger.info(f"Using provided MT5Handler instance from passed_config.")
        else:
            # MT5Handler() likely uses its own config loading or defaults if config arg is not supported/used.
            # If MT5Handler could take self.mt5_config, it would be MT5Handler(config=self.mt5_config)
            self.mt5_handler = MT5Handler() 
            logger.info(f"Created new MT5Handler instance (default initialization).")
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
            mt5_handler=self.mt5_handler
        )
        
        # --- NEW: Perform historical data sync on startup ---
        self.data_manager.synchronize_historical_trades()

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
        self.telegram_command_handler = TelegramCommandHandler(self, self.mt5_handler)
        
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
        self.last_candle_timestamps = {} # Tracks the last seen candle timestamp for each symbol/timeframe
        self.check_interval = 0.2  # Set check interval to 200ms for high-frequency polling
        self.last_tick_times = {} # Tracks the last seen tick timestamp to avoid redundant analysis
        self.last_analysis_time = {} # Tracks the last analysis time per symbol
        
        # Timezone handling
        self.ny_timezone = pytz.timezone('America/New_York')
        
        # Signal thresholds
        self.min_confidence = self.trading_config.get("min_confidence", 0.6)  # Default to 60% confidence
        
        # Trade management
        self.trailing_stop_enabled = True
        self.trailing_stop_data = {}  # Store trailing stop data for open positions
        
        logger.info("TradingBot initialized with enhanced multi-timeframe analysis capabilities")

        self.main_loop_task = None
        self.main_task = None # Renamed from main_loop_task
        self._monitor_trades_task = None
        self.data_fetch_task = None
        self.queue_processor_task = None

        # Log config order at the very start
        logger.info(f"[TRACE INIT] Initial trading_config signal_generators order: {self.trading_config.get('signal_generators', 'NOT FOUND')}")

    def get_default_lookback_for_timeframe(self, timeframe: str) -> int:
        """
        Provides an intelligent default lookback period based on the timeframe.
        These values are chosen to accommodate common long-period indicators.
        """
        timeframe_defaults = {
            'M1': 300,   # Adjusted for efficiency
            'M5': 250,   # A common default
            'M15': 200,
            'M30': 150,
            'H1': 100,
            'H4': 100,
            'D1': 100
        }
        # Default to 250 if timeframe is not in the map
        return timeframe_defaults.get(timeframe.upper(), 250)
        
    async def live_tick_event_loop(self):
        """
        High-frequency loop that processes live ticks and handles new candle events.
        """
        logger.info("🚀 Starting live tick and new candle event loop...")

        while not self.shutdown_requested:
            try:
                if not self.mt5_handler.is_connected():
                    logger.warning("MT5 disconnected. Pausing event loop.")
                    await asyncio.sleep(10)
                    self.mt5_handler.initialize()
                    continue

                # Step 1: Check for and load new closed candles first.
                unique_pairs_to_monitor = set()
                for sg in self.signal_generators:
                    if hasattr(sg, 'required_timeframes'):
                        for symbol in self.symbols:
                            for timeframe in sg.required_timeframes:
                                unique_pairs_to_monitor.add((symbol, timeframe))
                
                if unique_pairs_to_monitor:
                    check_candle_tasks = [self._check_for_new_candle(s, t) for s, t in unique_pairs_to_monitor]
                    await asyncio.gather(*check_candle_tasks)

                # Step 2: Process the latest ticks if they are new.
                tick_processing_tasks = []
                for symbol in self.symbols:
                    tick_processing_tasks.append(self._process_live_tick_for_symbol(symbol))
                
                if tick_processing_tasks:
                    await asyncio.gather(*tick_processing_tasks)

                await asyncio.sleep(self.check_interval)

            except asyncio.CancelledError:
                logger.info("Live event loop cancelled.")
                break
            except Exception as e:
                logger.error(f"An error occurred in the live event loop: {e}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(5)

    async def _process_live_tick_for_symbol(self, symbol: str):
        """ Processes a single live tick for a symbol if it's new. """
        latest_tick = self.mt5_handler.get_last_tick(symbol)
        if not latest_tick:
            return

        last_known_tick_time = self.last_tick_times.get(symbol, 0)
        
        # MT5 tick time is in milliseconds
        if latest_tick['time_msc'] > last_known_tick_time:
            self.last_tick_times[symbol] = latest_tick['time_msc']

            # Debounce analysis to avoid over-processing
            now = time.time()
            if (now - self.last_analysis_time.get(symbol, 0)) < self.check_interval:
                return
            self.last_analysis_time[symbol] = now

            # Update all relevant timeframes with the new tick
            for sg in self.signal_generators:
                if hasattr(sg, 'required_timeframes'):
                    for timeframe in sg.required_timeframes:
                        self.data_manager.update_real_time_data(symbol, timeframe, latest_tick)
            
            # After updating data, run analysis
            asyncio.create_task(self.run_realtime_analysis_for_symbol(symbol))

    async def tick_event_loop(self):
        """
        High-frequency loop that checks for new candle events.
        """
        logger.info("🚀 Starting tick event loop for new candle detection...")
        
        # Determine the unique set of symbol/timeframe pairs to monitor
        unique_pairs_to_monitor = set()
        for sg in self.signal_generators:
            if hasattr(sg, 'required_timeframes'):
                for symbol in self.symbols:
                    for timeframe in sg.required_timeframes:
                        unique_pairs_to_monitor.add((symbol, timeframe))
        
        if not unique_pairs_to_monitor:
            logger.warning("No symbol/timeframe pairs to monitor. The tick event loop will do nothing.")
            return

        logger.info(f"Monitoring {len(unique_pairs_to_monitor)} symbol/timeframe pairs.")

        while not self.shutdown_requested:
            try:
                if not self.mt5_handler.is_connected():
                    logger.warning("MT5 disconnected. Pausing candle checks.")
                    await asyncio.sleep(10)
                    self.mt5_handler.initialize()
                    continue

                check_tasks = [
                    self._check_for_new_candle(symbol, timeframe)
                    for symbol, timeframe in unique_pairs_to_monitor
                ]
                await asyncio.gather(*check_tasks)
                
                await asyncio.sleep(self.check_interval)

            except asyncio.CancelledError:
                logger.info("Tick event loop cancelled.")
                break
            except Exception as e:
                logger.error(f"An error occurred in the tick event loop: {e}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(5)

    async def _check_for_new_candle(self, symbol: str, timeframe: str):
        """
        Checks if a new candle has closed for a given symbol and timeframe.
        This is intended for the live monitoring loop, not for startup.
        """
        key = (symbol, timeframe)
        last_known_timestamp = self.last_candle_timestamps.get(key)

        if last_known_timestamp is None:
            logger.warning(f"[{symbol}/{timeframe}] Missing initial timestamp in live loop. Re-initializing.")
            latest_ts = self.mt5_handler.get_latest_candle_time(symbol, timeframe)
            if latest_ts:
                self.last_candle_timestamps[key] = latest_ts
            return

        latest_timestamp = self.mt5_handler.get_latest_candle_time(symbol, timeframe)
        if latest_timestamp is None:
            return

        if latest_timestamp > last_known_timestamp:
            logger.info(f"🕯️ New candle detected for {symbol}/{timeframe}. Timestamp: {datetime.fromtimestamp(latest_timestamp)}")
            self.last_candle_timestamps[key] = latest_timestamp
            
            asyncio.create_task(self.run_analysis_cycle_for_symbol(symbol))

    async def _execute_analysis_for_symbol(self, symbol: str, market_data_for_symbol: Dict[str, pd.DataFrame]):
        """Executes the signal generation and processing part of the analysis."""
        try:
            if not market_data_for_symbol:
                logger.warning(f"No market data provided for {symbol} analysis.")
                return

            all_signals: list[dict] = []
            for sg in self.signal_generators:
                logger.debug(f"Executing signal generator '{sg.name}' for {symbol}...")
                try:
                    signals = await sg.generate_signals(
                        market_data={symbol: market_data_for_symbol},
                        balance=self.risk_manager.get_account_balance()
                    )
                    if signals:
                        all_signals.extend(signals)
                except Exception as e:
                    logger.error(f"Error executing signal generator '{sg.name}' for {symbol}: {e}")
                    logger.error(traceback.format_exc())
            
            if all_signals:
                await self.process_signals(all_signals)
        except Exception as e:
            logger.error(f"Error during analysis execution for {symbol}: {e}")
            logger.error(traceback.format_exc())

    async def run_analysis_cycle_for_symbol(self, symbol: str):
        """
        Runs the full analysis pipeline for a single symbol, including fetching fresh data.
        """
        logger.debug(f"Running full analysis cycle for {symbol}...")
        
        required_timeframes: set[str] = set()
        for sg in self.signal_generators:
            if hasattr(sg, 'required_timeframes'):
                required_timeframes.update(getattr(sg, 'required_timeframes'))
        
        if not required_timeframes:
            return

        try:
            for timeframe in required_timeframes:
                lookback = self.data_manager.requirements.get((symbol, timeframe), 100)
                self.data_manager.update_data(symbol, timeframe, force=True, num_candles=lookback)

            market_data_for_symbol = {}
            for timeframe in required_timeframes:
                cached_data = self.data_manager.get_market_data(symbol, timeframe)
                if cached_data is not None and not cached_data.empty:
                    market_data_for_symbol[timeframe] = cached_data

            await self._execute_analysis_for_symbol(symbol, market_data_for_symbol)

        except Exception as e:
            logger.error(f"Error during full analysis cycle for {symbol}: {e}")
            logger.error(traceback.format_exc())

    async def run_realtime_analysis_for_symbol(self, symbol: str):
        """
        Runs a lightweight analysis cycle for a symbol using cached, real-time data.
        """
        logger.debug(f"Running real-time analysis for {symbol} on tick update...")

        required_timeframes: set[str] = set()
        for sg in self.signal_generators:
            if hasattr(sg, 'required_timeframes'):
                required_timeframes.update(getattr(sg, 'required_timeframes'))
        
        if not required_timeframes:
            return

        try:
            market_data_for_symbol = {}
            for timeframe in required_timeframes:
                cached_data = self.data_manager.get_market_data(symbol, timeframe)
                if cached_data is not None and not cached_data.empty:
                    market_data_for_symbol[timeframe] = cached_data

            await self._execute_analysis_for_symbol(symbol, market_data_for_symbol)

        except Exception as e:
            logger.error(f"Error during real-time analysis for {symbol}: {e}")
            logger.error(traceback.format_exc())

    async def _initialize_signal_generators(self):
        """
        Load and initialize signal generators based on the configuration.
        This method dynamically loads signal generator classes from the 'strategy' directory.
        """
        logger.info(f"[TRACE] _initialize_signal_generators called. trading_config signal_generators: {self.trading_config.get('signal_generators', 'NOT FOUND')}")
        try:
            active_generator_names = self.trading_config.get("signal_generators")
            if not active_generator_names:
                # Fallback logic
                try:
                    from config.config import TRADING_CONFIG
                    active_generator_names = TRADING_CONFIG.get("signal_generators", [])
                    logger.info(f"Loaded signal generators directly from config.py: {active_generator_names}")
                    self.trading_config["signal_generators"] = active_generator_names
                except Exception as e:
                    logger.error(f"Failed to import from config.py: {str(e)}")
                
                if not active_generator_names:
                    active_generator_names = ["breakout_reversal"] # Minimal default
                    logger.warning(f"Using default signal generators: {active_generator_names}")

            # Dynamically load all available strategies from the directory
            self._load_available_signal_generators()

            logger.info(f"Available signal generators: {list(self.available_signal_generators.keys())}")
            logger.info(f"Attempting to activate generators from config: {active_generator_names}")
            
            # Initialize active generators from config
            for generator_name in active_generator_names:
                # The keys should now match directly
                if generator_name in self.available_signal_generators:
                    generator_class = self.available_signal_generators[generator_name]
                    logger.info(f"Found generator class for '{generator_name}': {generator_class.__name__}")
                    
                    generator = generator_class(
                        mt5_handler=self.mt5_handler,
                        risk_manager=self.risk_manager
                    )
                    self.signal_generators.append(generator)
                    logger.info(f"Initialized signal generator: {generator_name} ({generator.__class__.__name__})")
                    
                    # Register data requirements with DataManager
                    required_timeframes = getattr(generator, 'required_timeframes', [])
                    lookback_periods = getattr(generator, 'lookback_periods', {})
                    
                    for symbol in self.symbols:
                        for tf in required_timeframes:
                            # Use strategy's specific lookback, its general lookback, or a smart default
                            lookback = lookback_periods.get(tf, getattr(generator, 'lookback', self.get_default_lookback_for_timeframe(tf)))
                            self.data_manager.register_timeframe(symbol, tf, lookback)
                else:
                    logger.warning(f"Unknown signal generator specified in config: '{generator_name}'")

            if not self.signal_generators:
                raise RuntimeError("No signal generators were successfully initialized. Check config and strategy files.")
                
            logger.info(f"[TRACE] Final signal_generators order: {[gen.__class__.__name__ for gen in self.signal_generators]}")
            
            # Log requirements for all loaded strategies
            for gen in self.signal_generators:
                logger.info(f"[StrategyLoad] {gen.__class__.__name__}: required_timeframes={getattr(gen, 'required_timeframes', None)}, lookback_periods={getattr(gen, 'lookback_periods', None)}")
                if not getattr(gen, 'required_timeframes', []):
                    logger.warning(f"[StrategyLoad] {gen.__class__.__name__} has empty required_timeframes and will be skipped in tick_event_loop!")
        except Exception as e:
            logger.error(f"Error initializing signal generators: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    async def _perform_startup_analysis(self):
        """
        Performs an initial data fetch and analysis for all required symbols.
        This "warms up" the bot with historical data before live monitoring begins.
        """
        logger.info("�� Performing startup data warmup and initial analysis...")

        unique_symbols = set(self.symbols)
        if not unique_symbols:
            logger.warning("No symbols configured for startup analysis.")
            return

        analysis_tasks = [
            self.run_analysis_cycle_for_symbol(symbol) for symbol in unique_symbols
        ]
        await asyncio.gather(*analysis_tasks)
        
        logger.info("✅ Startup analysis complete. Bot is now live and monitoring for new candles.")

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
            
            # Perform initial data fetch and analysis before starting live monitoring
            await self._perform_startup_analysis()
            
            # Set up Telegram commands
            if self.telegram_bot:
                await self.register_telegram_commands()
                logger.info("Telegram commands registered")
            
            # --- REPLACE OLD MAIN LOOP WITH TICK EVENT LOOP ---
            await self._prime_last_candle_timestamps()
            self.main_task = asyncio.create_task(self.live_tick_event_loop())
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
                await self.signal_processor.notify_trade_action(startup_message)
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
                if self.main_task is not None and self.main_task.done():
                    # If main loop exited with an error, log it
                    if self.main_task.exception():
                        logger.error(f"Main loop exited with an error: {self.main_task.exception()}")
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
            
            # --- NEW: Log signals to the database ---
            if signals:
                for signal in signals:
                    try:
                        # Construct a dictionary that matches the Signal model
                        signal_to_log = {
                            "symbol": signal.get('symbol'),
                            "timeframe": signal.get('timeframe'),
                            "strategy": signal.get('strategy_name', 'Unknown'),
                            "direction": signal.get('direction'),
                            "price": signal.get('price'),
                            "details": signal.get('details') 
                        }
                        self.data_manager.log_signal(signal_to_log)
                    except Exception as e:
                        logger.error(f"Failed to log signal: {signal}. Error: {e}")
            # --- END NEW ---

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
        Dynamically load and register all available signal generators from the src/strategy directory.
        The key for the strategy is the class name itself.
        """
        self.available_signal_generators = {}
        strategy_dir = BASE_DIR / "src" / "strategy"
        logger.info(f"[LOAD_STRAT] Scanning for strategies in: {strategy_dir}")

        for file_path in strategy_dir.glob("*.py"):
            if file_path.name == "__init__.py":
                continue

            module_name = file_path.stem
            try:
                # Dynamically import the module
                module_spec = importlib.util.spec_from_file_location(f"src.strategy.{module_name}", file_path)
                if module_spec and module_spec.loader:
                    module = importlib.util.module_from_spec(module_spec)
                    module_spec.loader.exec_module(module)

                    # Find all classes that inherit from SignalGenerator
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if issubclass(obj, SignalGenerator) and obj is not SignalGenerator:
                            # Use the class name as the key
                            strategy_key = name
                            self.available_signal_generators[strategy_key] = obj
                            logger.info(f"[LOAD_STRAT] Successfully loaded strategy '{strategy_key}' -> class '{name}'")
                else:
                    logger.warning(f"[LOAD_STRAT] Could not create module spec for {file_path}")

            except Exception as e:
                logger.error(f"[LOAD_STRAT] Error loading strategy from {file_path}: {e}")
                logger.error(traceback.format_exc())

        if self.available_signal_generators:
            logger.info(f"[LOAD_STRAT] Finished loading. Found {len(self.available_signal_generators)} generators: {list(self.available_signal_generators.keys())}")
        else:
            logger.warning("[LOAD_STRAT] No signal generators were loaded dynamically.")

    async def _prime_last_candle_timestamps(self) -> None:
        """Fill last_candle_timestamps so the live loop starts without warnings."""
        for sg in self.signal_generators:
            if not hasattr(sg, "required_timeframes"):
                continue
            for symbol in self.symbols:
                for tf in sg.required_timeframes:
                    ts = self.mt5_handler.get_latest_candle_time(symbol, tf)
                    if ts:
                        self.last_candle_timestamps[(symbol, tf)] = ts