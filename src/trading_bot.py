import asyncio
import json
import traceback
import pytz
import time
import sys
import socket  # Add socket module import
import MetaTrader5 as mt5  # Add MetaTrader5 import
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Type, Union
import string  # Add string module import

from loguru import logger

# Import custom modules
from src.mt5_handler import MT5Handler
from src.signal_generator import SignalGenerator
from src.signal_generator1 import SignalGenerator1  # Import the second signal generator
from src.signal_generator2 import SignalGeneratorBankTrading as SignalGenerator2  # Add the new signal generator
from src.signal_generator3 import SignalGenerator3  # Add the new signal generator
from src.risk_manager import RiskManager
from src.telegram_bot import TelegramBot
from src.database import db
from src.market_analysis import MarketAnalysis
from src.smc_analysis import SMCAnalysis
from src.mtf_analysis import MTFAnalysis


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
        self.config = config or TRADING_CONFIG
        self.trading_config = self.config
        
        # Initialize components
        self.risk_manager = RiskManager()
        self.signal_generators = []
        
        # Available signal generators mapping
        self.available_signal_generators = {
            "default": SignalGenerator,
            "signal_generator": SignalGenerator,  # Add direct mapping for config.py
            "signal_generator1": SignalGenerator1,
            "signal_generator2": SignalGenerator2,
            "signal_generator3": SignalGenerator3
        }
        
        # Initialize with a fresh MT5 connection
        # First make sure any existing connections are closed
        try:
            mt5.shutdown()
            logger.debug("Cleaned up any existing MT5 connections before initialization")
        except Exception as e:
            # Ignore errors here, just being cautious
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
        
        # Initialize other components
        self.risk_manager = RiskManager(self.mt5_handler)
        self.telegram_bot = TelegramBot()
        self.market_analysis = MarketAnalysis()
        
        # Initialize dashboard API
        self.dashboard_api = None
        self.dashboard_thread = None
        self.dashboard_enabled = TRADING_CONFIG.get("enable_dashboard", True)
        
       
        
        # Configuration - set this BEFORE initializing signal generators
        self.session_config = SESSION_CONFIG
        self.market_schedule = MARKET_SCHEDULE_CONFIG
        self.trading_config = TRADING_CONFIG if config is None else config.TRADING_CONFIG
        
        # Initialize multiple signal generators
        self.signal_generators = []
        self._init_signal_generators(signal_generator_class)
        
        # Keep the original signal generator for backward compatibility
        self.signal_generator_class = signal_generator_class
        self.signal_generator = signal_generator_class(risk_manager=self.risk_manager)
        
        # State management
        self.running = False
        self.trading_enabled = True  # Enabled by default as requested
        self.shutdown_requested = False  # Flag to gracefully exit the main loop
        self.signals: List[Dict] = []
        self.trade_counter = 0
        self.last_signal = {}  # Dictionary to track last signal timestamp and direction per symbol
        self.check_interval = 60  # Default check interval in seconds
        
        # Timezone handling
        self.ny_timezone = pytz.timezone('America/New_York')
        
        # Signal thresholds
        self.min_confidence = self.trading_config.get("min_confidence", 0.5)  # Default to 50% confidence
        
        # Trade management
        self.trailing_stop_enabled = True
        self.trailing_stop_data = {}  # Store trailing stop data for open positions
        
        # Shutdown behavior
        self.close_positions_on_shutdown = self.trading_config.get("close_positions_on_shutdown", False)  # Default to False
        
    def _init_signal_generators(self, default_generator_class: Type[SignalGenerator] = None):
        """Initialize signal generators from configuration.
        
        Args:
            default_generator_class: Default signal generator class if not specified in config
        """
        # Get configured signal generators from trading config
        configured_generators = self.trading_config.get("signal_generators", [])
        
        # If no generators configured, use the default
        if not configured_generators and default_generator_class:
            generator = default_generator_class(mt5_handler=self.mt5_handler, risk_manager=self.risk_manager)
            self.signal_generators.append(generator)
            logger.info(f"Using default signal generator: {default_generator_class.__name__}")
            return
            
        # Initialize each configured generator
        for generator_name in configured_generators:
            try:
                # Get the generator class from the available generators dictionary
                generator_class = self.available_signal_generators.get(generator_name)
                
                if generator_class:
                    # Initialize the generator with the MT5 handler and risk manager
                    generator = generator_class(mt5_handler=self.mt5_handler, risk_manager=self.risk_manager)
                    self.signal_generators.append(generator)
                    logger.info(f"Initialized signal generator: {generator_name}")
                else:
                    logger.error(f"Signal generator {generator_name} not found in available generators")
                
            except Exception as e:
                logger.error(f"Failed to initialize signal generator {generator_name}: {str(e)}")
                logger.error(traceback.format_exc())
                
        # If no generators could be initialized, add the default
        if not self.signal_generators and default_generator_class:
            generator = default_generator_class(mt5_handler=self.mt5_handler, risk_manager=self.risk_manager)
            self.signal_generators.append(generator)
            logger.info(f"Falling back to default signal generator: {default_generator_class.__name__}")

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
        if self.telegram_bot and self.telegram_bot.is_running:
            # Create a task to send notification asynchronously
            async def send_notification_task():
                await self.telegram_bot.send_notification(
                    f"Signal generator changed to {signal_generator_class.__name__}"
                )
            
            # Create and run the task in the background
            asyncio.create_task(send_notification_task())

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
        """Start the trading bot with proper initialization of all components."""
        try:
            # If already running, just enable trading
            if self.running:
                if self.telegram_bot and self.telegram_bot.is_running:
                    self.telegram_bot.trading_enabled = True
                    logger.info("Trading enabled on already running bot")
                return
            
            logger.info("Starting trading bot...")
            
            # Setup logging
            self.setup_logging()
            
            # Initialize MT5 first
            if not self.initialize_mt5():
                raise Exception("Failed to initialize MT5")
            logger.info("MT5 initialized successfully")
            
            # Reconcile trades that may have closed while offline
            await self.reconcile_trades()
            
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
                            
                            # Register custom commands
                            await self.register_telegram_commands()
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
            
            # Initialize Dashboard if enabled
            if self.dashboard_enabled:
                dashboard_init_success = await self.initialize_dashboard()
                if dashboard_init_success:
                    logger.info("Dashboard initialized successfully")
                else:
                    logger.warning("Failed to initialize dashboard, continuing without it")
            
            # Enable trading by default
            self.trading_enabled = True
            await self.enable_trading()  # This will also enable trading on the Telegram bot
            
            self.running = True
            logger.info("Trading bot started successfully")
            
            
            # Send startup notification
            if self.telegram_bot and self.telegram_bot.is_running:
                account_info = self.mt5_handler.get_account_info()
                
                startup_message = (
                    "📊 Trading Bot Started 📊\n\n"
                    f"Using signal generator: {self.signal_generator_class.__name__}\n"
                    f"Account Balance: {account_info.get('balance', 'N/A')}\n"
                    f"Symbols: {', '.join([s['symbol'] if isinstance(s, dict) else s for s in self.trading_config['symbols']])}\n"
                    f"Trailing Stop: {'Enabled' if self.trailing_stop_enabled else 'Disabled'}\n"
                    f"Dashboard: {'Enabled' if self.dashboard_enabled else 'Disabled'}\n\n"
                    "Use /status for more information"
                )
                
                await self.telegram_bot.send_notification(startup_message)
            
            # Start main trading loop
            await self.main_loop()
                    
        except Exception as e:
            logger.error(f"Bot error: {str(e)}")
            if self.telegram_bot and self.telegram_bot.is_running:
                await self.telegram_bot.send_error_alert(f"Bot error: {str(e)}")
            self.running = False
        finally:
            if not self.running:  # Only stop if we're actually shutting down
                await self.stop()

    async def initialize_dashboard(self) -> bool:
        """Initialize and start the dashboard API server."""
        try:
            logger.info("Initializing trading dashboard...")
            
            # Import DashboardAPI here to avoid circular imports
            from src.dashboard_api import DashboardAPI
            import uvicorn
            import asyncio
            import threading
            
            # Get dashboard configuration
            dashboard_api_port = self.trading_config.get("dashboard_api_port", 8000)
            auto_start_frontend = self.trading_config.get("auto_start_frontend", False)
            
            # Check if the port is already in use
            def is_port_in_use(port):
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    return s.connect_ex(('localhost', port)) == 0
                    
            if is_port_in_use(dashboard_api_port):
                logger.warning(f"Port {dashboard_api_port} is already in use. Dashboard API may already be running.")
                # Don't attempt to start a new server, but continue with the rest of initialization
                self.dashboard_api = DashboardAPI(trading_bot=self)
                return True
            
            # Initialize the dashboard API with the current trading bot instance
            self.dashboard_api = DashboardAPI(trading_bot=self)
            
            # Define a simple function to run the dashboard API server directly
            def run_dashboard():
                try:
                    logger.info(f"Starting dashboard API server directly on port {dashboard_api_port}")
                    # Call uvicorn directly to run the dashboard API
                    uvicorn.run(
                        self.dashboard_api.app,
                        host="0.0.0.0",
                        port=dashboard_api_port,
                        log_level="info"
                    )
                except Exception as e:
                    logger.error(f"Error running dashboard API: {str(e)}")
            
            # Start a thread to run the dashboard
            self.dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
            self.dashboard_thread.start()
            
            # Wait a moment to ensure the server has time to start
            await asyncio.sleep(3)
            
            # Auto-start frontend if configured
            frontend_started = False
            if auto_start_frontend:
                logger.info("Auto-starting dashboard frontend...")
                frontend_started = self.dashboard_api.start_frontend()
                if frontend_started:
                    logger.info("Dashboard frontend started successfully")
                else:
                    logger.warning("Failed to auto-start dashboard frontend")
            
            # Send notification if Telegram is running
            if self.telegram_bot and self.telegram_bot.is_running:
                dashboard_url = f"http://localhost:{dashboard_api_port}"
                frontend_url = "http://localhost:3000" if frontend_started else None
                
                notification = f"Trading Dashboard API is now available at {dashboard_url}"
                if frontend_url:
                    notification += f"\nDashboard frontend is available at {frontend_url}"
                else:
                    notification += "\nUse /startdashboard to start the frontend"
                    
                await self.telegram_bot.send_notification(notification)
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing dashboard: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    async def stop(self, cleanup_only=False):
        """Stop the trading bot and clean up resources"""
        self.running = False
        
        # Stop the telegram bot if it's running
        if self.telegram_bot is not None and not cleanup_only:
            try:
                await self.telegram_bot.stop()
                self.telegram_bot = None
            except Exception as e:
                logger.warning(f"Error stopping Telegram bot: {str(e)}")
        
        # Stop Dashboard API server if it's running
        if self.dashboard_api is not None and not cleanup_only:
            try:
                logger.info("Stopping dashboard API server...")
                # If we're using the direct uvicorn method, we can't easily stop it
                # The thread will automatically terminate when the process exits
                # But we can still clean up any frontend processes
                if hasattr(self.dashboard_api, 'frontend_process') and self.dashboard_api.frontend_process:
                    try:
                        logger.info("Stopping dashboard frontend...")
                        self.dashboard_api.frontend_process.terminate()
                        self.dashboard_api.frontend_process = None
                        logger.info("Dashboard frontend stopped")
                    except Exception as e:
                        logger.warning(f"Error stopping dashboard frontend: {str(e)}")
                
                # Set to None to allow garbage collection
                self.dashboard_api = None
                self.dashboard_thread = None
                logger.info("Dashboard API server references cleared")
            except Exception as e:
                logger.warning(f"Error stopping dashboard API: {str(e)}")
        
        # Stop other subsystems
        try:
            # Only close pending trades if configured to do so
            if self.close_positions_on_shutdown:
                logger.info("Closing positions on shutdown (enabled in config)")
                await self.close_pending_trades()
            else:
                logger.info("Keeping positions open on shutdown (as configured)")
            
            # Shutdown MT5 connection if doing a full shutdown
            if not cleanup_only and self.mt5_handler is not None:
                try:
                    logger.info("Shutting down MT5 connection...")
                    self.mt5_handler.shutdown()
                    logger.info("MT5 connection closed")
                except Exception as e:
                    logger.warning(f"Error shutting down MT5: {str(e)}")
            
            # Log last known balance
            try:
                account_info = self.mt5_handler.get_account_info()
                final_balance = account_info.get('balance', 0)
                logger.info(f"Final account balance: ${final_balance:.2f}")
            except:
                logger.info("Unable to retrieve final account balance")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
        
        logger.info("Trading bot stopped")

    async def register_telegram_commands(self):
        """Register custom command handlers with the Telegram bot."""
        if not self.telegram_bot or not self.telegram_bot.is_running:
            logger.warning("Telegram bot not running, skipping command registration")
            return
            
        # Register handler for listing available signal generators
        await self.telegram_bot.register_command_handler(
            "listsignalgenerators", 
            self.handle_list_signal_generators_command
        )
        
        # Register handler for switching signal generators
        await self.telegram_bot.register_command_handler(
            "setsignalgenerator", 
            self.handle_set_signal_generator_command
        )
        
        # Register trailing stop commands
        await self.telegram_bot.register_command_handler(
            "enabletrailing",
            self.handle_enable_trailing_stop_command
        )
        
        await self.telegram_bot.register_command_handler(
            "disabletrailing",
            self.handle_disable_trailing_stop_command
        )
        
        # Register command to show current trading status
        await self.telegram_bot.register_command_handler(
            "status",
            self.handle_status_command
        )
        
        # Register dashboard commands
        await self.telegram_bot.register_command_handler(
            "startdashboard",
            self.handle_start_dashboard_command
        )
        
        # Register position closing on shutdown commands (new functionality)
        await self.telegram_bot.register_command_handler(
            'enablecloseonshutdown', 
            self.handle_enable_close_on_shutdown_command
        )
        
        await self.telegram_bot.register_command_handler(
            'disablecloseonshutdown', 
            self.handle_disable_close_on_shutdown_command
        )
        
        # Register shutdown command
        await self.telegram_bot.register_command_handler(
            'shutdown',
            self.handle_shutdown_command
        )
        
        # Register new commands
        await self.telegram_bot.register_command_handler(
            'enabletrading',
            self.handle_enable_trading_command
        )
        
        await self.telegram_bot.register_command_handler(
            'disabletrading',
            self.handle_disable_trading_command
        )
        
        await self.telegram_bot.register_command_handler(
            'enablepositionadditions',
            self.handle_enable_position_additions_command
        )
        
        await self.telegram_bot.register_command_handler(
            'disablepositionadditions',
            self.handle_disable_position_additions_command
        )
        
        logger.info("Successfully registered telegram commands")

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
        status += f"Dashboard: {'✅' if self.dashboard_enabled and self.dashboard_api is not None else '❌'}\n"
        status += f"Current Session: {current_session}\n"
        status += f"Signal Generator: {self.signal_generator_class.__name__}\n\n"
        
        # Dashboard info
        if self.dashboard_enabled and self.dashboard_api is not None:
            status += f"Dashboard URL: http://localhost:8000\n"
            status += f"Dashboard Frontend: http://localhost:3000\n\n"
            
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

    async def handle_start_dashboard_command(self, args):
        """
        Handle command to start the dashboard frontend.
        Format: /startdashboard
        """
        if not self.dashboard_enabled or self.dashboard_api is None:
            return "Dashboard is not enabled. Please enable it in the configuration."
            
        # Start the frontend
        success = self.dashboard_api.start_frontend()
        
        if success:
            return (
                "Dashboard frontend started!\n\n"
                "Access it at: http://localhost:3000\n\n"
                "API URL: http://localhost:8000"
            )
        else:
            return "Failed to start dashboard frontend. Check the logs for more information."

    async def main_loop(self):
        """Main trading loop."""
        logger.info("Starting main trading loop")
        
        # Initialize MT5 connection
        if not self.initialize_mt5():
            logger.error("Failed to initialize MT5. Cannot start trading loop.")
            return
        
        # Initialize dashboard if enabled
        if self.dashboard_enabled:
            dashboard_initialized = await self.initialize_dashboard()
            if not dashboard_initialized:
                logger.warning("Failed to initialize dashboard. Continuing without dashboard.")
        
        self.running = True
        last_performance_update = datetime.now() - timedelta(hours=2)  # Ensure immediate first update
        
        # Track consecutive connection failures
        consecutive_connection_failures = 0
        max_consecutive_failures = 5
        
        # Create trade monitoring task
        trade_monitor_task = asyncio.create_task(self._monitor_trades_loop())
        
        try:
            while self.running:
                try:
                    # Check MT5 connection at the start of each iteration
                    if not hasattr(self, 'mt5_handler') or not self.mt5_handler or not getattr(self.mt5_handler, 'connected', False):
                        logger.warning("MT5 connection lost. Attempting to recover...")
                        connection_recovered = self.recover_mt5_connection()
                        if not connection_recovered:
                            consecutive_connection_failures += 1
                            logger.warning(f"Connection recovery failed (attempt {consecutive_connection_failures}/{max_consecutive_failures})")
                            
                            if consecutive_connection_failures >= max_consecutive_failures:
                                logger.error(f"Too many consecutive MT5 connection failures ({consecutive_connection_failures}). Pausing trading for safety.")
                                self.trading_enabled = False
                                # Try to notify via Telegram
                                if self.telegram_bot and self.telegram_bot.is_running:
                                    await self.telegram_bot.send_notification("Trading disabled due to persistent MT5 connection failures")
                                await asyncio.sleep(60)  # Wait before retrying
                                continue
                            
                            # Shorter wait time for non-critical failures
                            await asyncio.sleep(5 * consecutive_connection_failures)  # Increasing backoff
                            continue
                        else:
                            logger.info("MT5 connection successfully recovered")
                            consecutive_connection_failures = 0  # Reset on successful recovery
                    else:
                        # Check if connection is responsive by doing a simple query
                        try:
                            account_info = self.mt5_handler.get_account_info()
                            if not account_info:
                                logger.warning("MT5 connection appears unresponsive. Attempting recovery...")
                                if self.recover_mt5_connection():
                                    logger.info("MT5 connection successfully recovered")
                                    consecutive_connection_failures = 0
                                else:
                                    consecutive_connection_failures += 1
                                    await asyncio.sleep(5)  # Brief pause before continuing
                                    continue
                            else:
                                consecutive_connection_failures = 0  # Connection is fine
                        except Exception as e:
                            logger.warning(f"Error checking MT5 connection status: {str(e)}")
                            if "IPC" in str(e) or "connection" in str(e).lower():
                                if self.recover_mt5_connection():
                                    logger.info("MT5 connection successfully recovered")
                                    consecutive_connection_failures = 0
                                else:
                                    consecutive_connection_failures += 1
                                    await asyncio.sleep(5)  # Brief pause
                                    continue
                    
                    # Check which markets are open
                    any_market_open = False
                    crypto_markets_open = False
                    forex_markets_open = False
                    
                    # First, check market status without processing any symbols yet
                    symbols_to_process = []
                    
                    for symbol_config in self.trading_config["symbols"]:
                        if isinstance(symbol_config, dict):
                            symbol = symbol_config.get("symbol", "").strip()
                        else:
                            symbol = symbol_config.strip()
                            
                        # Skip if symbol is not defined
                        if not symbol:
                            continue
                            
                        # Check if the symbol is a crypto symbol using the same logic as market_analysis
                        is_crypto = False
                        if symbol is not None:
                            # True cryptocurrency symbols typically contain BTC, ETH, etc.
                            crypto_identifiers = ["BTC", "ETH", "XBT", "LTC", "DOT", "SOL", "ADA", "DOGE", "CRYPTO"]
                            is_crypto = any(identifier in symbol for identifier in crypto_identifiers)
                            
                            # Check for explicit cryptocurrency pairs
                            crypto_pairs = ["BTCUSD", "ETHUSD"]
                            if any(pair in symbol for pair in crypto_pairs):
                                is_crypto = True
                        
                        # Check if this symbol's market is open
                        market_open = self.is_market_open(symbol)
                        
                        if is_crypto and market_open:
                            crypto_markets_open = True
                            any_market_open = True
                            symbols_to_process.append((symbol_config, "crypto"))
                        elif not is_crypto and market_open:
                            forex_markets_open = True
                            any_market_open = True
                            symbols_to_process.append((symbol_config, "forex"))
                    
                    if not any_market_open:
                        logger.info("All markets are closed. Waiting for markets to open...")
                        await asyncio.sleep(300)  # Check every 5 minutes to reduce resource usage
                        continue
                    
                    # Continue with trading as at least one market is open
                    if forex_markets_open:
                        logger.info("Forex markets are open")
                    if crypto_markets_open:
                        logger.info("Crypto markets are open")
                    
                    # Analyze current session
                    current_session = self.analyze_session()
                    logger.info(f"Current session: {current_session}")
                    
                    # Get account info for risk management
                    account_info = self.mt5_handler.get_account_info()
                    
                    # Periodically update performance metrics (every hour)
                    current_time = datetime.now()
                    if (current_time - last_performance_update).total_seconds() > 3600:
                        await self.update_performance_metrics()
                        last_performance_update = current_time
                    
                    # Process only symbols that have open markets
                    for symbol_config, market_type in symbols_to_process:
                        # Check if symbol_config is a string or dictionary
                        if isinstance(symbol_config, str):
                            # If it's a string, it's just the symbol name
                            symbol = symbol_config
                            timeframe = "M15"  # Default timeframe
                            additional_timeframes = []
                        else:
                            # If it's a dictionary, extract values using get()
                            symbol = symbol_config.get("symbol")
                            timeframe = symbol_config.get("timeframe", "H1")
                            additional_timeframes = symbol_config.get("additional_timeframes", [])
                        
                        # Skip if symbol or timeframe is not defined (redundant check)
                        if not symbol:
                            continue
                        
                        # Double check if this symbol's market is still open
                        if not self.is_market_open(symbol):
                            logger.debug(f"Market is now closed for {symbol}. Skipping.")
                            continue
                            
                        logger.info(f"Processing {market_type} symbol: {symbol}")
                        
                        # Get market data
                        market_data = {}
                        
                        # Fetch primary timeframe data
                        logger.debug(f"Fetching primary timeframe data for {symbol} on {timeframe}")
                        primary_data = await self.mt5_handler.get_rates(symbol, timeframe, 500)
                        if primary_data is None:
                            # Log as warning instead of error for missing symbols
                            logger.warning(f"Failed to get primary data for {symbol} on {timeframe}")
                            # Skip this symbol but continue with others
                            continue
                        elif len(primary_data) < 100:
                            logger.warning(f"Insufficient data for {symbol} on {timeframe} (only {len(primary_data)} candles available)")
                            # Skip this symbol but continue with others
                            continue
                            
                        logger.debug(f"Successfully fetched {len(primary_data)} candles for {symbol} on {timeframe}")
                        market_data[timeframe] = primary_data
                        
                        # Fetch additional timeframes if needed
                        for add_tf in additional_timeframes:
                            logger.debug(f"Fetching additional timeframe data for {symbol} on {add_tf}")
                            add_data = await self.mt5_handler.get_rates(symbol, add_tf, 500)
                            if add_data is not None and len(add_data) >= 100:
                                logger.debug(f"Successfully fetched {len(add_data)} candles for {symbol} on {add_tf}")
                                market_data[add_tf] = add_data
                            else:
                                logger.warning(f"Could not fetch sufficient data for {symbol} on {add_tf}")
                        
                        # Skip if not enough data
                        if not market_data:
                            logger.warning(f"No valid market data for {symbol}")
                            continue
                        
                        # Add current session information to market data
                        market_data['current_session'] = current_session
                        
                        # Add market structure, SMC analysis and technical indicators to market data
                        try:
                            # Run market structure analysis
                            market_structure = self.market_analysis.analyze_market_structure(
                                market_data[timeframe], symbol, timeframe
                            )
                            market_data['market_structure'] = market_structure
                            logger.debug(f"Market structure analysis completed for {symbol}")
                            
                            # Run SMC analysis
                            smc_analysis = SMCAnalysis()
                            smc_signals = smc_analysis.analyze(market_data[timeframe])
                            market_data['smc_signals'] = smc_signals
                            logger.debug(f"SMC analysis completed for {symbol} with {len(smc_signals.get('liquidity_sweeps', []))} liquidity sweeps")
                            
                            # Run MTF analysis
                            mtf_analysis = MTFAnalysis()
                            mtf_data = mtf_analysis.analyze_multiple_timeframes(market_data)
                            market_data['mtf_analysis'] = mtf_data
                            logger.debug(f"MTF analysis completed for {symbol}")
                            
                            # Add volatility state
                            atr_series = self.market_analysis.calculate_atr(market_data[timeframe], 14)
                            volatility_state = self.market_analysis.classify_volatility(atr_series)
                            market_data['volatility_state'] = volatility_state
                            logger.debug(f"Volatility analysis completed for {symbol}: {volatility_state}")
                            
                        except Exception as e:
                            logger.error(f"Error during market analysis for {symbol}: {str(e)}")
                            logger.exception("Detailed error information:")
                            market_data['market_structure'] = {}
                            market_data['smc_signals'] = {}
                            market_data['mtf_analysis'] = {}
                            market_data['volatility_state'] = 'normal'
                        
                        # Generate signals from all configured signal generators
                        all_signals = []
                        
                        # Log how many signal generators we're using
                        logger.debug(f"Using {len(self.signal_generators)} signal generators for {symbol}")
                        
                        # Skip using the legacy signal generator if we have multiple generators configured
                        if len(self.signal_generators) > 0:
                            # Use each configured signal generator
                            for i, generator in enumerate(self.signal_generators):
                                logger.debug(f"Running signal generator #{i+1} ({generator.__class__.__name__}) for {symbol}")
                                try:
                                    # Call the generator with the appropriate market data
                                    if asyncio.iscoroutinefunction(generator.generate_signals):
                                        result = await generator.generate_signals(market_data=market_data, symbol=symbol, timeframe=timeframe, account_info=account_info)
                                    else:
                                        result = generator.generate_signals(market_data=market_data, symbol=symbol, timeframe=timeframe, account_info=account_info)
                                    
                                    # Handle both old (direct signals list) and new (dict with signals key) formats
                                    if isinstance(result, dict) and "signals" in result:
                                        signals = result["signals"]
                                    else:
                                        signals = result
                                    
                                    # Enhanced signal logging for debugging
                                    gen_name = generator.__class__.__name__
                                    if signals and len(signals) > 0:
                                        logger.info(f"Generator {gen_name} produced {len(signals)} signals for {symbol}")
                                        logger.debug(f"Signal details from {gen_name}: {json.dumps(signals[0], default=str)[:200]}...")
                                        # Tag signals with generator name for tracking
                                        for signal in signals:
                                            signal['generator'] = gen_name
                                        all_signals.extend(signals)
                                    else:
                                        logger.debug(f"Generator {gen_name} produced no signals for {symbol}")
                                        if gen_name == "SignalGenerator3":
                                            logger.debug(f"SignalGenerator3 returned: {type(signals)} - {signals}")
                                except Exception as e:
                                    logger.error(f"Error in signal generator {generator.__class__.__name__}: {str(e)}")
                                    logger.error(traceback.format_exc())
                        else:
                            # Fall back to the legacy single generator for backward compatibility
                            logger.debug(f"Using legacy signal generator for {symbol}")
                            all_signals = await self.signal_generator.generate_signals(
                                market_data=market_data,
                                symbol=symbol,
                                timeframe=timeframe,
                                account_info=account_info
                            )
                        
                        # Process signals
                        if all_signals:
                            logger.info(f"Generated a total of {len(all_signals)} signals for {symbol} from all generators")
                            await self.process_signals(all_signals)
                    
                    # Manage existing positions
                    await self.manage_open_trades()
                    
                    # Sleep between iterations
                    await asyncio.sleep(self.trading_config.get("loop_interval", 60))
                    
                except Exception as e:
                    logger.error(f"Error in main loop: {str(e)}")
                    logger.exception("Exception details:")
                    await asyncio.sleep(10)  # Sleep a bit before retrying
                
            logger.info("Main loop exited cleanly")
            
        except asyncio.CancelledError:
            logger.info("Main loop cancelled")
            
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {str(e)}")
            logger.exception("Exception details:")
            
        finally:
            self.running = False
            # Cancel trade monitoring task
            if not trade_monitor_task.done():
                trade_monitor_task.cancel()
                try:
                    await trade_monitor_task
                except asyncio.CancelledError:
                    pass
            # Close MT5 connection if we own it
            self.mt5_handler.shutdown()

    async def _monitor_trades_loop(self):
        """Separate loop for monitoring trades more frequently."""
        logger.info("Starting trade monitoring loop")
        
        while self.running:
            try:
                # Check for active positions first
                active_positions = self.mt5_handler.get_open_positions()
                
                if not active_positions:
                    # No open positions, no need to check market status or manage trades
                    await asyncio.sleep(30)  # Longer sleep when no positions
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
                await self.manage_open_trades()
                
                # Sleep for a shorter interval (5 seconds) to monitor trades more frequently
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Error in trade monitoring loop: {str(e)}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(1)  # Brief sleep on error before retrying

    def is_market_open(self, symbol: str = None) -> bool:
        """
        Check if the market is currently open based on schedule.
        
        Args:
            symbol: Optional symbol to check. If provided, used to determine if it's a crypto symbol.
        """
        # Use the implementation from MarketAnalysis
        return self.market_analysis.is_market_open(symbol)

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
            
            if not signals:
                logger.debug("No signals to process")
                return
                
            # Log the full signals list for debugging
            logger.debug(f"Full signals list: {json.dumps(signals, default=str, indent=2)}")
            
            # Validate signals before processing
            valid_signals = []
            for i, signal in enumerate(signals):
                # Check required fields
                required_fields = ['symbol', 'direction', 'entry_price', 'stop_loss', 'take_profit', 'position_size']
                missing_fields = [field for field in required_fields if signal.get(field) is None]
                
                if missing_fields:
                    logger.warning(f"Signal #{i+1} missing required fields: {', '.join(missing_fields)}")
                    continue
                
                # Check if market is open for this symbol
                symbol = signal.get('symbol')
                if not self.is_market_open(symbol):
                    logger.warning(f"Market is closed for {symbol}. Skipping signal.")
                    continue
                
                # Validate numeric fields
                numeric_fields = ['entry_price', 'stop_loss', 'position_size']
                invalid_fields = []
                
                for field in numeric_fields:
                    value = signal.get(field)
                    try:
                        if value is None:
                            invalid_fields.append(f"{field} (None)")
                            continue
                            
                        # Try converting to float
                        float_value = float(value)
                        if float_value <= 0:
                            invalid_fields.append(f"{field} (not positive)")
                    except (ValueError, TypeError):
                        invalid_fields.append(f"{field} (not numeric)")
                
                # Special handling for take_profit which can be either a numeric value or a dictionary
                take_profit = signal.get('take_profit')
                if take_profit is None:
                    invalid_fields.append("take_profit (None)")
                elif isinstance(take_profit, dict):
                    # For dictionary format, verify it has a 'price' field that's numeric and positive
                    if 'price' not in take_profit:
                        invalid_fields.append("take_profit (missing 'price' field)")
                    else:
                        try:
                            tp_price = float(take_profit['price'])
                            if tp_price <= 0:
                                invalid_fields.append("take_profit.price (not positive)")
                            # If valid, normalize signal to store price in the main signal object
                            signal['tp_price'] = tp_price
                        except (ValueError, TypeError):
                            invalid_fields.append("take_profit.price (not numeric)")
                else:
                    # For direct numeric format
                    try:
                        tp_value = float(take_profit)
                        if tp_value <= 0:
                            invalid_fields.append("take_profit (not positive)")
                        # Store the value in a consistent way
                        signal['tp_price'] = tp_value
                    except (ValueError, TypeError):
                        invalid_fields.append("take_profit (not numeric)")
                
                if invalid_fields:
                    logger.warning(f"Signal #{i+1} has invalid fields: {', '.join(invalid_fields)}")
                    continue
                
                # Add validated signal
                valid_signals.append(signal)
            
            if len(valid_signals) < len(signals):
                logger.warning(f"Filtered out {len(signals) - len(valid_signals)} invalid signals, proceeding with {len(valid_signals)} valid signals")
            
            if not valid_signals:
                logger.warning("No valid signals after validation, stopping processing")
                return
                
            # Continue with the valid signals
            signals = valid_signals
            
            # Group signals by symbol for better management
            signals_by_symbol = {}
            for signal in signals:
                symbol = signal.get('symbol', 'UNKNOWN')
                if symbol not in signals_by_symbol:
                    signals_by_symbol[symbol] = []
                signals_by_symbol[symbol].append(signal)
            
            # Process signals by symbol
            for symbol, symbol_signals in signals_by_symbol.items():
                logger.info(f"Processing {len(symbol_signals)} signals for {symbol}")
                
                # Log signals by generator
                generators = set(signal.get('generator', 'unknown') for signal in symbol_signals)
                for generator in generators:
                    gen_signals = [s for s in symbol_signals if s.get('generator', 'unknown') == generator]
                    logger.info(f"  - {len(gen_signals)} signals from {generator}")
                
                # Process each signal
                for signal in symbol_signals:
                    try:
                        # Get signal details
                        generator = signal.get('generator', 'unknown')
                        logger.info(f"Processing {symbol} signal from {generator}")
                        
                        # Skip processing for HOLD signals
                        if signal.get('signal') == 'HOLD':
                            logger.info("Skipping HOLD signal")
                            continue
                        
                        # Only check confidence if it's included in the signal
                        if 'confidence' in signal:
                            confidence = signal.get('confidence', 0)
                            if confidence < self.min_confidence:
                                logger.info(f"Signal confidence {confidence} below minimum threshold {self.min_confidence}")
                                continue
                        
                        # Store the signal in the database
                        signal_id = db.insert_signal(signal)
                        if signal_id > 0:
                            logger.info(f"Signal stored in database with ID: {signal_id}")
                            # Add the database ID to the signal object
                            signal['id'] = signal_id
                        else:
                            logger.error("Failed to store signal in database")
                        
                        # Check for existing positions on this symbol
                        open_positions = self.mt5_handler.get_open_positions()
                        existing_positions = [p for p in open_positions if p["symbol"] == symbol]
                        
                        if existing_positions:
                            # Handle existing positions
                            await self.handle_signal_with_existing_positions(signal, existing_positions)
                        else:
                            # Place new trade using existing functionality
                            await self.execute_trade_from_signal(signal)
                        
                    except Exception as e:
                        logger.error(f"Error processing signal for {symbol}: {str(e)}")
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

    async def execute_trade_from_signal(self, signal: Dict, is_addition: bool = False) -> None:
        """Execute a trade based on signal, using existing mt5_handler functionality and storing results in database."""
        try:
            # Extract signal information
            symbol = signal.get('symbol')
            direction = signal.get('direction', '').upper()
            entry_price = signal.get('entry_price')
            stop_loss = signal.get('stop_loss')
            take_profit = signal.get('take_profit')
            position_size = signal.get('position_size')
            signal_id = signal.get('id')  # Database ID
            
            # Process structured take profit format if present
            tp_price = signal.get('tp_price')  # This should be set during validation
            if tp_price is None:
                # If tp_price wasn't set during validation, extract it now
                if isinstance(take_profit, dict) and 'price' in take_profit:
                    tp_price = take_profit['price']
                else:
                    tp_price = take_profit
            
            # Log full signal data for debugging
            logger.debug(f"Attempting to execute trade with signal: {json.dumps(signal, default=str, indent=2)}")
            
            # Validate required parameters
            required_params = ['symbol', 'direction', 'entry_price', 'stop_loss', 'position_size']
            missing_params = [param for param in required_params if signal.get(param) is None]
            
            # Check for take_profit separately since it can be in different formats
            if tp_price is None:
                missing_params.append('take_profit')
            
            if missing_params:
                error_msg = f"Missing required trade parameters: {', '.join(missing_params)}"
                logger.error(error_msg)
                if signal_id:
                    db.update_signal_status(signal_id, "invalid", False)
                await self._notify_trade_action(f"❌ Trade Validation Error: {error_msg}")
                return
            
            # Validate numeric values
            try:
                # Check for None values or other invalid types before conversion
                numeric_params = {
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit_price': tp_price,
                    'position_size': position_size
                }
                
                for param_name, param_value in numeric_params.items():
                    if param_value is None:
                        raise TypeError(f"{param_name} is None")
                    if not isinstance(param_value, (int, float, str)):
                        raise TypeError(f"{param_name} is not a number or string: {type(param_value)}")
                
                # Convert to float after validation
                entry_price = float(entry_price)
                stop_loss = float(stop_loss)
                take_profit_price = float(tp_price)
                position_size = float(position_size)
                
                # Additional validation for zero or negative values
                if entry_price <= 0:
                    raise ValueError("Entry price must be positive")
                if position_size <= 0:
                    raise ValueError("Position size must be positive")
                if stop_loss <= 0:
                    raise ValueError("Stop loss must be positive")
                if take_profit_price <= 0:
                    raise ValueError("Take profit must be positive") 
                    
                # Additional validation for correct SL/TP placement
                if direction == 'BUY':
                    # For BUY, SL should be below entry, TP should be above entry
                    if stop_loss >= entry_price:
                        logger.warning(f"Invalid stop loss for BUY trade: {stop_loss} >= {entry_price}, adjusting")
                        stop_loss = entry_price * 0.995  # Set to 0.5% below entry
                    if take_profit_price <= entry_price:
                        logger.warning(f"Invalid take profit for BUY trade: {take_profit_price} <= {entry_price}, adjusting")
                        take_profit_price = entry_price * 1.01  # Set to 1% above entry
                else:  # SELL
                    # For SELL, SL should be above entry, TP should be below entry
                    if stop_loss <= entry_price:
                        logger.warning(f"Invalid stop loss for SELL trade: {stop_loss} <= {entry_price}, adjusting")
                        stop_loss = entry_price * 1.005  # Set to 0.5% above entry
                    if take_profit_price >= entry_price:
                        logger.warning(f"Invalid take profit for SELL trade: {take_profit_price} >= {entry_price}, adjusting")
                        take_profit_price = entry_price * 0.99  # Set to 1% below entry
            except (ValueError, TypeError) as e:
                error_msg = f"Invalid numeric values in trade parameters: {str(e)}"
                logger.error(error_msg)
                logger.error(f"Signal with invalid parameters: {json.dumps(signal, default=str, indent=2)}")
                if signal_id:
                    db.update_signal_status(signal_id, "invalid", False)
                await self._notify_trade_action(f"❌ Trade Validation Error: {error_msg}")
                return
            
            # Log trade attempt
            logger.info(f"Executing {'additional' if is_addition else 'new'} {direction} trade for {symbol}")
            logger.info(f"Parameters: Entry={entry_price}, SL={stop_loss}, TP={take_profit_price}, Size={position_size}")
            
            # Check if MT5 is connected
            if not self.mt5_handler or not getattr(self.mt5_handler, 'connected', False):
                error_msg = "MT5 connection not available"
                logger.error(error_msg)
                await self._notify_trade_action(f"❌ Trade Execution Error: {error_msg}")
                return
            
            # Check if symbol is available for trading
            symbol_info = self.mt5_handler.get_symbol_info(symbol)
            if not symbol_info:
                error_msg = f"Symbol {symbol} not found or not available for trading"
                logger.error(error_msg)
                await self._notify_trade_action(f"❌ Trade Execution Error: {error_msg}")
                return

            # Get current market prices to handle slippage
            current_ask = symbol_info.ask  # price for buying
            current_bid = symbol_info.bid  # price for selling
            logger.info(f"Current market prices for {symbol} - Ask: {current_ask}, Bid: {current_bid}")
            
            # Check current spread and reject trade if it's too high
            current_spread = current_ask - current_bid
            # Convert to pips for easier interpretation
            pip_size = 0.0001  # Default for most forex pairs
            if symbol.endswith('JPY') or 'JPY' in symbol:
                pip_size = 0.01
            elif symbol.startswith('XAU'):
                pip_size = 0.1
            elif symbol.startswith('XAG'):
                pip_size = 0.01
            # Special handling for cryptocurrency pairs
            elif symbol.endswith('USDm') or symbol.endswith('USDT') or symbol.endswith('USD') and any(crypto in symbol for crypto in ['BTC', 'ETH', 'LTC', 'XRP', 'DOGE']):
                # For cryptocurrencies, use 1 point as 1 pip for high-priced assets like BTC
                pip_size = 1.0
            
            # For indices, stocks, etc. - try to get point value from MT5
            if hasattr(symbol_info, 'point'):
                point_value = symbol_info.point
                if symbol.endswith('JPY') or 'JPY' in symbol or symbol.startswith('XAU') or symbol.startswith('XAG') or symbol.endswith('USDm') or symbol.endswith('USDT'):
                    # These instruments already have correct pip size
                    pass
                else:
                    # For standard forex, a pip is typically 10 points
                    pip_size = point_value * 10
            
            # Calculate spread in pips
            spread_in_pips = current_spread / pip_size
            
            # Get max allowed spread from config or use default
            max_allowed_spread = 20  # Default max spread in pips
            try:
                # Safely check if config exists and has max_spread attribute
                if hasattr(self, 'config') and isinstance(self.config, dict) and 'max_spread' in self.config:
                    max_allowed_spread = self.config['max_spread']
                # Also check if config is a class with max_spread attribute
                elif hasattr(self, 'config') and hasattr(self.config, 'max_spread'):
                    max_allowed_spread = self.config.max_spread
            except Exception as e:
                logger.warning(f"Error accessing max_spread from config: {str(e)}, using default value: {max_allowed_spread}")
            
            logger.info(f"Current spread for {symbol}: {spread_in_pips:.1f} pips (Max allowed: {max_allowed_spread} pips)")
            
            # Reject trade if spread is too high
            if spread_in_pips > max_allowed_spread:
                error_msg = f"Spread too high for {symbol}: {spread_in_pips:.1f} pips (Max allowed: {max_allowed_spread} pips)"
                logger.warning(error_msg)
                if signal_id:
                    # Remove the reason parameter which is not supported
                    db.update_signal_status(signal_id, "rejected", False)
                await self._notify_trade_action(f"⚠️ Trade Rejected: {error_msg}")
                return
            
            # Check for significant price slippage and adjust if necessary
            adjusted_params = False
            original_entry = entry_price
            original_sl = stop_loss
            original_tp = take_profit_price
            
            # Calculate pip size for this instrument
            pip_size = 0.0001  # Default for most forex pairs
            if symbol.endswith('JPY') or 'JPY' in symbol:
                pip_size = 0.01
            elif symbol.startswith('XAU'):
                pip_size = 0.1
            elif symbol.startswith('XAG'):
                pip_size = 0.01
            # Special handling for cryptocurrency pairs
            elif symbol.endswith('USDm') or symbol.endswith('USDT') or symbol.endswith('USD') and any(crypto in symbol for crypto in ['BTC', 'ETH', 'LTC', 'XRP', 'DOGE']):
                # For cryptocurrencies, use 1 point as 1 pip for high-priced assets like BTC
                pip_size = 1.0
            
            # For indices, stocks, etc. - try to get point value from MT5
            if hasattr(symbol_info, 'point'):
                point_value = symbol_info.point
                if symbol.endswith('JPY') or 'JPY' in symbol or symbol.startswith('XAU') or symbol.startswith('XAG') or symbol.endswith('USDm') or symbol.endswith('USDT'):
                    # These instruments already have correct pip size
                    pass
                else:
                    # For standard forex, a pip is typically 10 points
                    pip_size = point_value * 10
            
            # Calculate acceptable slippage in price units (20 pips default)
            slippage_threshold = 20 * pip_size
            
            # Define the risk-reward ratio from the original signal
            if direction == 'BUY':
                original_risk = entry_price - stop_loss
                original_reward = take_profit_price - entry_price
            else:  # SELL
                original_risk = stop_loss - entry_price
                original_reward = entry_price - take_profit_price
                
            original_rr_ratio = original_reward / original_risk if original_risk > 0 else 1.5
            logger.info(f"Original risk-reward ratio: {original_rr_ratio:.2f}")
            
            # Adjust parameters based on current prices
            if direction == 'BUY':
                # For BUY orders, we use Ask price
                if abs(current_ask - entry_price) > slippage_threshold:
                    logger.warning(f"Significant price slippage detected for {symbol}: Signal entry: {entry_price}, Current ask: {current_ask}")
                    
                    # Adjust entry to current ask
                    entry_price = current_ask
                    
                    # Maintain the same risk-reward ratio and distance relationships
                    risk_distance = original_risk  # Keep similar risk distance
                    stop_loss = entry_price - risk_distance
                    take_profit_price = entry_price + (risk_distance * original_rr_ratio)
                    
                    adjusted_params = True
                    logger.info(f"Adjusted trade parameters - Entry: {entry_price}, SL: {stop_loss}, TP: {take_profit_price}")
            else:  # SELL
                # For SELL orders, we use Bid price
                if abs(current_bid - entry_price) > slippage_threshold:
                    logger.warning(f"Significant price slippage detected for {symbol}: Signal entry: {entry_price}, Current bid: {current_bid}")
                    
                    # Adjust entry to current bid
                    entry_price = current_bid
                    
                    # Maintain the same risk-reward ratio and distance relationships
                    risk_distance = original_risk  # Keep similar risk distance
                    stop_loss = entry_price + risk_distance
                    take_profit_price = entry_price - (risk_distance * original_rr_ratio)
                    
                    adjusted_params = True
                    logger.info(f"Adjusted trade parameters - Entry: {entry_price}, SL: {stop_loss}, TP: {take_profit_price}")
            
            # Get minimum stop distance from MT5
            min_stop_distance = 0
            if hasattr(symbol_info, 'trade_stops_level') and hasattr(symbol_info, 'point'):
                min_stop_distance = symbol_info.point * symbol_info.trade_stops_level
                logger.info(f"Minimum stop distance for {symbol}: {min_stop_distance}")
            
            # Final validation and adjustment for stop loss
            if direction == 'BUY':
                # For BUY orders, SL must be below entry
                if stop_loss >= entry_price:
                    stop_loss = entry_price - (10 * pip_size)  # At least 10 pips below
                    logger.warning(f"Stop loss above entry for BUY, adjusting to: {stop_loss}")
                    adjusted_params = True
                    
                # Ensure minimum distance
                if min_stop_distance > 0 and (entry_price - stop_loss) < min_stop_distance:
                    stop_loss = entry_price - min_stop_distance - (2 * pip_size)  # Add extra 2 pips for safety
                    logger.warning(f"Stop loss too close to entry, adjusting to: {stop_loss}")
                    adjusted_params = True
            else:  # SELL
                # For SELL orders, SL must be above entry
                if stop_loss <= entry_price:
                    stop_loss = entry_price + (10 * pip_size)  # At least 10 pips above
                    logger.warning(f"Stop loss below entry for SELL, adjusting to: {stop_loss}")
                    adjusted_params = True
                    
                # Ensure minimum distance
                if min_stop_distance > 0 and (stop_loss - entry_price) < min_stop_distance:
                    stop_loss = entry_price + min_stop_distance + (2 * pip_size)  # Add extra 2 pips for safety
                    logger.warning(f"Stop loss too close to entry, adjusting to: {stop_loss}")
                    adjusted_params = True
            
            # Re-adjust take profit to maintain risk-reward if stop loss was adjusted
            if adjusted_params:
                if direction == 'BUY':
                    risk = entry_price - stop_loss
                    take_profit_price = entry_price + (risk * original_rr_ratio)
                else:  # SELL
                    risk = stop_loss - entry_price
                    take_profit_price = entry_price - (risk * original_rr_ratio)
                
                logger.info(f"Final adjusted parameters - Entry: {entry_price}, SL: {stop_loss}, TP: {take_profit_price}")
                
                # Notify about the adjustment
                adjustment_message = (
                    f"⚠️ Trade parameters adjusted due to price slippage:\n"
                    f"Original: Entry={original_entry}, SL={original_sl}, TP={original_tp}\n"
                    f"Adjusted: Entry={entry_price}, SL={stop_loss}, TP={take_profit_price}"
                )
                await self._notify_trade_action(adjustment_message)
            
            # Use existing MT5Handler to execute the trade
            ticket = None
            try:
                # Check if we have partial take profit settings in the take_profit field
                has_partial_tp = False
                
                # Create a simple comment for the trade
                try:
                    raw_comment = f"{signal.get('generator', 'Bot')}_{signal.get('strategy', 'Signal')}"
                    
                    # Truncate if too long (max 30 chars)
                    if len(raw_comment) > 30:
                        raw_comment = raw_comment[:27] + "..."
                    
                    # Remove any invalid characters (only allow alphanumeric, underscore, dash)
                    valid_chars = string.ascii_letters + string.digits + '_-'
                    comment = ''.join(c for c in raw_comment if c in valid_chars)
                    
                    if not comment:
                        # Fallback to a default comment if everything was invalid
                        comment = "TradingBot"
                    
                except Exception as e:
                    logger.warning(f"Failed to generate trade comment: {str(e)}, using default comment")
                    comment = "TradingBot"
                
                # Check if take_profit is a dictionary with partial TP information
                if isinstance(take_profit, dict) and 'price' in take_profit and hasattr(self.mt5_handler, 'execute_trade'):
                    # It could be a partial take profit configuration
                    if 'size' in take_profit and 'ratio' in take_profit:
                        has_partial_tp = True
                        logger.info(f"Using partial take profit settings from signal: {take_profit}")
                        
                        # Calculate risk in price terms
                        if direction == 'BUY':
                            risk = entry_price - stop_loss
                        else:  # SELL
                            risk = stop_loss - entry_price
                            
                        # Define the partial take profit levels
                        partial_tp_levels = [{
                            'ratio': float(take_profit['ratio']),
                            'size': float(take_profit['size'])
                        }]
                        
                        # If size is less than 1.0, add a second level
                        if float(take_profit['size']) < 1.0:
                            remaining_size = 1.0 - float(take_profit['size'])
                            # Second level has higher R multiple 
                            second_ratio = float(take_profit['ratio']) * 1.5
                            partial_tp_levels.append({
                                'ratio': second_ratio,
                                'size': remaining_size
                            })
                            
                        # Prepare trade parameters
                        trade_params = {
                            'symbol': symbol,
                            'signal_type': direction,
                            'entry_price': entry_price,
                            'stop_loss': stop_loss,
                            'position_size': position_size,
                            'partial_tp_levels': partial_tp_levels
                        }
                        
                        # Execute trade with partial take profits
                        tickets = self.mt5_handler.execute_trade(trade_params)
                        
                        if tickets:
                            logger.info(f"Successfully executed trade with {len(tickets)} partial take profit levels")
                            ticket = tickets[0]  # Use first ticket for tracking
                        else:
                            logger.warning("Failed to execute trade with partial take profits, falling back to standard method")
                            has_partial_tp = False
                
                # If not using partial take profits or it failed, use standard execution
                if not has_partial_tp:
                    if direction == 'BUY':
                        ticket = self.mt5_handler.open_buy(
                            symbol, position_size, stop_loss, take_profit_price, 
                            comment=comment
                        )
                    else:  # SELL
                        ticket = self.mt5_handler.open_sell(
                            symbol, position_size, stop_loss, take_profit_price,
                            comment=comment
                        )
            except Exception as e:
                error_msg = f"MT5 trade execution error: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                await self._notify_trade_action(f"❌ Trade Execution Error: {error_msg}")
                if signal_id:
                    db.update_signal_status(signal_id, "failed", False)
                return
            
            # Process the result
            if ticket:
                try:
                    # Get position details
                    position = self.mt5_handler.get_position_by_ticket(ticket)
                    if not position:
                        error_msg = f"Position not found after execution for ticket {ticket}"
                        logger.error(error_msg)
                        await self._notify_trade_action(f"⚠️ Trade Warning: {error_msg}")
                        return
                    
                    logger.info(f"Position opened successfully: {json.dumps(position, default=str, indent=2)}")
                    
                    # Adjust take profit to maintain the original risk-reward ratio if execution price is different
                    actual_entry_price = position.get('price_open')
                    
                    if actual_entry_price != entry_price:
                        logger.info(f"Actual execution price {actual_entry_price} differs from intended entry {entry_price}")
                        
                        # Recalculate take profit based on actual entry while keeping original stop loss
                        if direction == 'BUY':
                            # For BUY: 
                            # - If entry is higher than intended, we need to move TP higher to maintain RR
                            actual_risk = actual_entry_price - stop_loss
                            adjusted_take_profit = actual_entry_price + (actual_risk * original_rr_ratio)
                            
                            logger.info(f"Adjusting BUY take profit: Original {take_profit_price:.2f} → New {adjusted_take_profit:.2f}")
                            logger.info(f"New risk: {actual_risk:.2f} pips, New reward: {(adjusted_take_profit - actual_entry_price):.2f} pips")
                            
                        else:  # SELL
                            # For SELL:
                            # - If entry is lower than intended, we need to move TP lower to maintain RR
                            actual_risk = stop_loss - actual_entry_price
                            adjusted_take_profit = actual_entry_price - (actual_risk * original_rr_ratio)
                            
                            logger.info(f"Adjusting SELL take profit: Original {take_profit_price:.2f} → New {adjusted_take_profit:.2f}")
                            logger.info(f"New risk: {actual_risk:.2f} pips, New reward: {(actual_entry_price - adjusted_take_profit):.2f} pips")
                        
                        # Update position with the adjusted take profit
                        if self.mt5_handler.modify_position(ticket, stop_loss, adjusted_take_profit):
                            logger.info(f"Successfully adjusted take profit to maintain {original_rr_ratio:.2f} risk-reward ratio")
                            take_profit_price = adjusted_take_profit  # Update for database record
                        else:
                            logger.warning(f"Failed to adjust take profit, using original value")
                    
                    # Save position for trailing stop management
                    if self.trailing_stop_enabled:
                        activation_price = self._calculate_activation_price(direction, entry_price, stop_loss)
                        self.trailing_stop_data[ticket] = {
                            'symbol': symbol,
                            'direction': direction,
                            'entry_price': entry_price,
                            'initial_stop': stop_loss,
                            'current_stop': stop_loss,
                            'activated': False,
                            'activation_price': activation_price
                        }
                    
                    # Store in database if position was successfully opened
                    if position:
                        # Calculate pip value for the symbol
                        symbol_info = self.mt5_handler.get_symbol_info(symbol)
                        pip_value = 0.0001  # Default for forex
                        if symbol_info and hasattr(symbol_info, 'point'):
                            pip_value = symbol_info.point * 10
                        
                        # Calculate initial profit/loss in pips
                        current_price = position.get('price_current', entry_price)
                        profit_loss_pips = abs(current_price - entry_price) / pip_value
                        if (direction == 'BUY' and current_price < entry_price) or \
                           (direction == 'SELL' and current_price > entry_price):
                            profit_loss_pips = -profit_loss_pips
                        
                        # Create comprehensive trade data
                        trade_data = {
                            'ticket': str(ticket),
                            'symbol': symbol,
                            'direction': direction,
                            'entry_price': position.get('price_open', entry_price),
                            'current_price': current_price,
                            'stop_loss': position.get('sl', stop_loss),
                            'take_profit': position.get('tp', take_profit_price),
                            'position_size': position.get('volume', position_size),
                            'open_time': datetime.now().isoformat(),
                            'status': 'open',
                            'profit_loss': position.get('profit', 0),
                            'profit_loss_pips': profit_loss_pips,
                            'pip_value': pip_value,
                            'strategy': signal.get('strategy', 'unknown'),
                            'confidence': signal.get('confidence', 0),
                            'market_condition': signal.get('market_condition', 'normal'),
                            'volatility_state': signal.get('volatility_state', 'normal'),
                            'trailing_stop_enabled': self.trailing_stop_enabled,
                            'trailing_stop_activation_price': activation_price if self.trailing_stop_enabled else None,
                            'is_addition': is_addition,
                            'signal_generator': signal.get('generator', 'unknown'),
                            'comment': position.get('comment', '')
                        }
                        
                        # Store the trade in the database
                        trade_id = db.insert_trade(trade_data, signal_id)
                        if trade_id > 0:
                            logger.info(f"Trade stored in database with ID: {trade_id}")
                        else:
                            logger.error("Failed to store trade in database")
                            await self._notify_trade_action("⚠️ Warning: Trade executed but failed to store in database")
                    
                    # Update signal status
                    if signal_id:
                        db.update_signal_status(signal_id, "executed", True)
                    
                    # Send success notification
                    notification_message = (
                        f"✅ {'Additional' if is_addition else 'New'} {direction} position opened\n"
                        f"Symbol: {symbol}\n"
                        f"Entry: {position.get('price_open', entry_price)}\n"
                        f"Stop Loss: {position.get('sl', stop_loss)}\n"
                        f"Take Profit: {position.get('tp', take_profit_price)}\n"
                        f"Size: {position_size} lots\n"
                        f"Strategy: {signal.get('strategy', 'Signal')}\n"
                        f"Confidence: {signal.get('confidence', 'N/A')}\n"
                        f"Generator: {signal.get('generator', 'unknown')}"
                    )
                    await self._notify_trade_action(notification_message)
                    
                    logger.info(f"Trade executed successfully: Ticket {ticket}")
                    
                except Exception as e:
                    error_msg = f"Error processing successful trade: {str(e)}"
                    logger.error(error_msg)
                    logger.error(traceback.format_exc())
                    await self._notify_trade_action(f"⚠️ Trade Warning: {error_msg}")
            else:
                # Trade execution failed
                error_msg = f"Failed to execute {direction} trade for {symbol}"
                logger.error(error_msg)
                
                # Get MT5 error code if available
                mt5_error = self.mt5_handler.get_last_error() if hasattr(self.mt5_handler, 'get_last_error') else None
                
                # Check for margin-related errors specifically
                if "No margin available" in str(error_msg).lower() or (mt5_error and "margin" in str(mt5_error).lower()):
                    # Get account info for detailed error message
                    account_info = self.mt5_handler.get_account_info()
                    
                    # Create a detailed margin error message
                    margin_error = f"❌ Margin Error: Insufficient margin to execute {direction} trade for {symbol}.\n"
                    
                    if account_info:
                        free_margin = account_info.get('free_margin', 'Unknown')
                        balance = account_info.get('balance', 'Unknown')
                        margin_error += f"Free Margin: {free_margin}, Balance: {balance}\n"
                    
                    margin_error += f"Position size: {position_size} lots\n"
                    margin_error += "Consider adding funds or reducing position size."
                    
                    logger.error(margin_error)
                    await self._notify_trade_action(margin_error)
                else:
                    # General error handling
                    if mt5_error:
                        error_msg += f"\nError details: {mt5_error}"
                        logger.error(f"MT5 Error: {mt5_error}")
                    
                    # Send detailed error notification
                    await self._notify_trade_action(f"❌ Trade Execution Failed:\n{error_msg}")
                
                # Update signal status in both cases
                if signal_id:
                    db.update_signal_status(signal_id, "failed", False)
        
        except Exception as e:
            error_msg = f"Unexpected error executing trade: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            await self._notify_trade_action(f"❌ Critical Trade Error: {error_msg}")

    async def handle_signal_with_existing_positions(self, signal: Dict, existing_positions: List[Dict]) -> None:
        """Handle signal when there are existing positions for the symbol."""
        symbol = signal.get('symbol')
        signal_direction = signal.get('direction', '').upper()
        
        # Check if we have positions in the opposite direction
        opposite_positions = []
        same_direction_positions = []
        
        for position in existing_positions:
            position_type = "BUY" if position["type"] == 0 else "SELL"
            if position_type != signal_direction:
                opposite_positions.append(position)
            else:
                same_direction_positions.append(position)
        
        # Handle opposite direction positions based on configuration
        if opposite_positions:
            if self.trading_config.get("close_on_reverse_signal", True):
                logger.info(f"Signal ({signal_direction}) is opposite to existing positions. Closing them.")
                
                for position in opposite_positions:
                    ticket = position["ticket"]
                    if self.mt5_handler.close_position(ticket):
                        logger.info(f"Closed opposite position {ticket}")
                        
                        # Send notification
                        if self.telegram_bot and self.telegram_bot.is_running:
                            await self.telegram_bot.send_trade_update(
                                order_id=ticket,
                                symbol=symbol,
                                action="CLOSED",
                                price=position["price_current"],
                                profit=position["profit"],
                                reason="Reverse signal detected"
                            )
                
                # After closing opposite positions, place the new trade
                await self.execute_trade_from_signal(signal)
                
            elif not self.trading_config.get("allow_hedging", False):
                logger.info(f"Signal ({signal_direction}) is opposite to existing positions, but hedging is disabled.")
            else:
                # Hedging is allowed, place the new trade
                logger.info(f"Signal ({signal_direction}) is opposite to existing positions. Hedging allowed.")
                await self.execute_trade_from_signal(signal)
        
        # Handle same direction positions
        elif same_direction_positions:
            if self.trading_config.get("allow_position_additions", False):
                current_volume = sum(pos["volume"] for pos in same_direction_positions)
                max_volume = self.trading_config.get("max_position_size", 1.0)
                
                if current_volume < max_volume:
                    logger.info(f"Adding to existing {signal_direction} position for {symbol}")
                    await self.execute_trade_from_signal(signal, is_addition=True)
                else:
                    logger.info(f"Max position size reached for {symbol}, not adding")
            else:
                logger.info(f"Signal matches existing position direction ({signal_direction}), but position additions are disabled.")

    async def manage_open_trades(self) -> None:
        """Manage open trades using existing functionality from MT5Handler and RiskManager."""
        try:
            # Get all open positions
            positions = self.mt5_handler.get_open_positions()
            if not positions:
                return
                
            logger.debug(f"Managing {len(positions)} open positions")
            
            for position in positions:
                ticket = position["ticket"]
                symbol = position["symbol"]
                position_type = self._get_position_type(position)
                current_price = position["price_current"]
                current_sl = position["sl"]
                current_tp = position["tp"]
                
                # Update trade in database
                self._update_trade_in_database(position)
                
                # Skip trailing stop management if not enabled or not applicable
                if not self.trailing_stop_enabled or ticket not in self.trailing_stop_data:
                    continue
                    
                # Get trailing stop data
                ts_data = self.trailing_stop_data[ticket]
                
                # Check if trailing stop is already activated
                if ts_data['activated']:
                    # Create trade object for risk manager
                    trade = {
                        'direction': position_type,
                        'entry_price': ts_data['entry_price'],
                        'initial_stop': ts_data['initial_stop'],
                        'stop_loss': ts_data['current_stop']
                    }
                    
                    # Use risk manager to calculate new stop loss
                    should_adjust, new_sl = self.risk_manager.calculate_trailing_stop(
                        trade=trade,
                        current_price=current_price,
                        market_condition='normal'
                    )
                    
                    # If we should adjust and it's better than current stop
                    if should_adjust:
                        if (position_type == "BUY" and new_sl > ts_data['current_stop']) or \
                           (position_type == "SELL" and new_sl < ts_data['current_stop']):
                            logger.info(f"Updating trailing stop for ticket {ticket} from {ts_data['current_stop']} to {new_sl}")
                            
                            # Use MT5Handler to modify the position
                            if self.mt5_handler.modify_position(ticket, new_sl, current_tp):
                                ts_data['current_stop'] = new_sl
                                
                                # Send notification
                                notification_message = (
                                    f"Trailing stop updated for {symbol} {position_type}\n"
                                    f"New stop loss: {new_sl}"
                                )
                                await self._notify_trade_action(notification_message)
                
                # Check if trailing stop should be activated
                else:
                    activation_price = ts_data['activation_price']
                    
                    if (position_type == "BUY" and current_price >= activation_price) or \
                       (position_type == "SELL" and current_price <= activation_price):
                        ts_data['activated'] = True
                        logger.info(f"Trailing stop activated for ticket {ticket}")
                        
                        # Send notification
                        notification_message = (
                            f"Trailing stop activated for {symbol} {position_type}\n"
                            f"Current price: {current_price}\n"
                            f"Activation price: {activation_price}"
                        )
                        await self._notify_trade_action(notification_message)
        
        except Exception as e:
            logger.error(f"Error managing open trades: {str(e)}")
            logger.error(traceback.format_exc())

    def _update_trade_in_database(self, position: Dict[str, Any]) -> None:
        """Update trade information in the database."""
        try:
            # Find trade in database by ticket
            trade_id = self._find_trade_id_by_ticket(position["ticket"])
            
            if trade_id:
                # Get current trade info from database
                current_trades = db.get_active_trades()
                current_trade = next((t for t in current_trades if t.get('id') == trade_id), None)
                
                if not current_trade:
                    logger.warning(f"Trade {trade_id} not found in active trades")
                    return
                    
                # Calculate pip value for the symbol
                symbol = position.get('symbol', '')
                symbol_info = self.mt5_handler.get_symbol_info(symbol)
                pip_value = 0.0001  # Default for forex
                if symbol_info and hasattr(symbol_info, 'point'):
                    pip_value = symbol_info.point * 10
                
                # Calculate profit/loss in pips
                entry_price = current_trade.get('entry_price', 0)
                current_price = position.get('price_current', 0)
                direction = self._get_position_type(position)
                
                profit_loss_pips = abs(current_price - entry_price) / pip_value
                if (direction == 'BUY' and current_price < entry_price) or \
                   (direction == 'SELL' and current_price > entry_price):
                    profit_loss_pips = -profit_loss_pips
                
                # Check if SL/TP was hit
                stop_loss = position.get('sl', 0)
                take_profit = position.get('tp', 0)
                status = 'open'
                
                if direction == 'BUY':
                    if current_price <= stop_loss:
                        status = 'closed'
                        logger.info(f"Trade {trade_id} hit stop loss")
                    elif current_price >= take_profit:
                        status = 'closed'
                        logger.info(f"Trade {trade_id} hit take profit")
                else:  # SELL
                    if current_price >= stop_loss:
                        status = 'closed'
                        logger.info(f"Trade {trade_id} hit stop loss")
                    elif current_price <= take_profit:
                        status = 'closed'
                        logger.info(f"Trade {trade_id} hit take profit")
                
                # Update trade with current information
                trade_data = {
                    'current_price': current_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'profit_loss': position.get('profit', 0),
                    'profit_loss_pips': profit_loss_pips,
                    'status': status
                }
                
                # Update in database
                db.update_trade(trade_id, trade_data)
                
                # If trade closed, update final stats
                if status == 'closed':
                    db.close_trade(
                        trade_id,
                        current_price,
                        position.get('profit', 0),
                        profit_loss_pips,
                        datetime.now().isoformat()
                    )
                    
                    # Send notification
                    close_reason = "stop loss" if (
                        (direction == 'BUY' and current_price <= stop_loss) or 
                        (direction == 'SELL' and current_price >= stop_loss)
                    ) else "take profit"
                    
                    notification_message = (
                        f"Trade closed - Hit {close_reason}\n"
                        f"Symbol: {symbol}\n"
                        f"Direction: {direction}\n"
                        f"Profit/Loss: {position.get('profit', 0)}\n"
                        f"Pips: {profit_loss_pips:.1f}"
                    )
                    asyncio.create_task(self._notify_trade_action(notification_message))
                
        except Exception as e:
            logger.error(f"Error updating trade in database: {str(e)}")
            logger.error(traceback.format_exc())

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
        """Close all pending trades when the bot is stopping."""
        try:
            # Get all open positions
            open_positions = self.mt5_handler.get_open_positions()
            if not open_positions:
                logger.info("No open positions to close")
                return
            
            logger.info(f"Closing {len(open_positions)} open positions...")
            
            # Track success/failure counts
            success_count = 0
            failed_count = 0
            
            # Try to close each position
            for position in open_positions:
                ticket = position["ticket"]
                symbol = position["symbol"]
                profit = position["profit"]
                
                # Use MT5Handler to close the position
                if self.mt5_handler.close_position(ticket):
                    success_count += 1
                    logger.info(f"Closed position {ticket} on {symbol} with P/L: {profit}")
                    
                    # Update the database record
                    trade_id = self._find_trade_id_by_ticket(ticket)
                    
                    if trade_id:
                        # Calculate basic pip values for database record
                        entry_price = position.get('price_open', 0)
                        close_price = position.get('price_current', 0)
                        direction = self._get_position_type(position)
                        
                        # Calculate pips using pip_value
                        symbol_info = self.mt5_handler.get_symbol_info(symbol)
                        pip_value = 0.0001  # Default for most forex pairs
                        if symbol_info and hasattr(symbol_info, 'point'):
                            pip_value = symbol_info.point * 10
                            
                        profit_loss_pips = abs(close_price - entry_price) / pip_value
                        if (direction == 'BUY' and close_price < entry_price) or \
                           (direction == 'SELL' and close_price > entry_price):
                            profit_loss_pips = -profit_loss_pips
                        
                        # Create comprehensive trade close data
                        close_data = {
                            'close_price': close_price,
                            'profit': profit,
                            'profit_loss_pips': profit_loss_pips,
                            'close_time': datetime.now().isoformat(),
                            'close_reason': 'Bot Shutdown',
                            'final_status': 'closed',
                            'final_stop_loss': position.get('sl', 0),
                            'final_take_profit': position.get('tp', 0),
                            'close_comment': f"Closed by bot shutdown - P/L: {profit}"
                        }
                        
                        # Update database with complete trade data
                        db.close_trade(
                            trade_id,
                            close_data['close_price'],
                            close_data['profit'],
                            close_data['profit_loss_pips'],
                            close_data['close_time']
                        )
                        
                        # Update trade with additional closing details
                        db.update_trade(trade_id, close_data)
                        
                        # Send notification
                        if self.telegram_bot and self.telegram_bot.is_running:
                            try:
                                position_type = self._get_position_type(position)
                                await self.telegram_bot.send_trade_update(
                                    ticket,
                                    symbol,
                                    f"{position_type} CLOSED (Bot Shutdown)",
                                    position["price_current"],
                                    profit
                                )
                            except Exception as e:
                                logger.warning(f"Failed to send trade update: {str(e)}")
                    else:
                        failed_count += 1
                        logger.warning(f"Failed to close position {ticket} on {symbol}")
            
            logger.info(f"Closed {success_count} positions, {failed_count} failed")
            
        except Exception as e:
            logger.error(f"Error closing pending trades: {str(e)}")
            logger.error(traceback.format_exc())

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

    async def update_performance_metrics(self):
        """Update performance metrics in the database."""
        try:
            # Calculate metrics for different timeframes
            for timeframe in ['1D', '1W', '1M']:
                metrics = db.calculate_performance_metrics(timeframe)
                db.update_or_insert_performance_metrics(metrics)
                
            logger.info("Performance metrics updated successfully")
        except Exception as e:
            logger.error(f"Error updating performance metrics: {str(e)}")

    def _get_position_type(self, position: Dict[str, Any]) -> str:
        """Get the direction (BUY/SELL) from a position."""
        return "BUY" if position["type"] == 0 else "SELL"
        
    def _find_trade_id_by_ticket(self, ticket: int) -> Optional[int]:
        """Find a trade ID in the database by its ticket number."""
        try:
            trades = db.get_active_trades()
            for trade in trades:
                if trade.get('ticket') == str(ticket):
                    return trade.get('id')
            return None
        except Exception as e:
            logger.error(f"Error finding trade by ticket: {str(e)}")
            return None
            
    def _calculate_activation_price(self, direction: str, entry_price: float, stop_loss: float) -> float:
        """Calculate trailing stop activation price."""
        risk = abs(entry_price - stop_loss)
        activation_factor = self.trading_config.get("trailing_activation_factor", 1.0)
        return entry_price + (risk * activation_factor if direction == 'BUY' else -risk * activation_factor)
        
    async def _notify_trade_action(self, message: str) -> None:
        """Send notification if telegram bot is available."""
        if self.telegram_bot and self.telegram_bot.is_running:
            try:
                await self.telegram_bot.send_notification(message)
            except Exception as e:
                logger.warning(f"Failed to send notification: {str(e)}")

    async def request_shutdown(self):
        """Request a graceful shutdown of the trading bot."""
        logger.info("Shutdown requested - will exit after current cycle completes")
        self.shutdown_requested = True
        
        # Send notification if Telegram is available
        if self.telegram_bot and self.telegram_bot.is_running:
            await self.telegram_bot.send_notification("⚠️ Trading bot shutdown requested. Will exit soon.")
        
        return True

    async def handle_shutdown_command(self, args):
        """
        Handle command to gracefully shutdown the trading bot.
        Format: /shutdown
        """
        await self.request_shutdown()
        return "⚠️ Trading bot shutdown initiated. The bot will exit after completing the current cycle."

    async def reconcile_trades(self):
        """Reconcile trades that may have closed while the bot was offline."""
        try:
            # Get active trades from database
            active_db_trades = db.get_active_trades()
            
            if not active_db_trades:
                logger.info("No active trades in database to reconcile")
                return
            
            # Get current open positions from MT5
            mt5_positions = self.mt5_handler.get_open_positions()
            mt5_tickets = {str(pos['ticket']) for pos in mt5_positions}
            
            # Find trades that are active in DB but not in MT5 (likely closed while offline)
            closed_while_offline = []
            for trade in active_db_trades:
                if trade['ticket'] not in mt5_tickets:
                    closed_while_offline.append(trade)
            
            if not closed_while_offline:
                logger.info("All database trades match MT5 positions, no reconciliation needed")
                return
                
            logger.info(f"Found {len(closed_while_offline)} trades that closed while bot was offline")
            
            # Process each trade that closed while offline
            for trade in closed_while_offline:
                try:
                    trade_id = trade['id']
                    ticket = trade['ticket']
                    symbol = trade['symbol']
                    
                    # Get historical trade data from MT5
                    history = self.mt5_handler.get_order_history(ticket=int(ticket) if ticket.isdigit() else 0)
                    
                    if history:
                        # Found historical data
                        closed_order = history[0]  # Use the first matching order
                        
                        # Calculate profit/loss in pips
                        entry_price = trade['entry_price']
                        # Use price_current if available, otherwise use just price
                        close_price = closed_order.get('price_current', closed_order.get('price', entry_price))
                        direction = trade['direction']
                        
                        # Calculate pips
                        symbol_info = self.mt5_handler.get_symbol_info(symbol)
                        pip_value = 0.0001  # Default for forex
                        if symbol_info and hasattr(symbol_info, 'point'):
                            pip_value = symbol_info.point * 10
                            
                        profit_loss_pips = abs(close_price - entry_price) / pip_value
                        if (direction.upper() == 'BUY' and close_price < entry_price) or \
                           (direction.upper() == 'SELL' and close_price > entry_price):
                            profit_loss_pips = -profit_loss_pips
                        
                        # Get profit from the history data
                        profit = closed_order.get('profit', 0)
                        close_time = closed_order.get('time_close', datetime.now().isoformat())
                        
                        # Update database record
                        db.close_trade(
                            trade_id,
                            close_price,
                            profit,
                            profit_loss_pips,
                            close_time
                        )
                        
                        logger.info(f"Reconciled trade {ticket} with P/L: {profit}")
                    else:
                        # No history found, mark as closed with unknown outcome
                        logger.warning(f"No history found for ticket {ticket}, marking as closed with unknown outcome")
                        db.close_trade(
                            trade_id,
                            trade['entry_price'],  # Use entry as close if no data
                            0,  # Assume zero profit/loss
                            0,  # Assume zero pips
                            datetime.now().isoformat()
                        )
                except Exception as e:
                    logger.error(f"Error reconciling trade {trade.get('ticket')}: {str(e)}")
            
            logger.info(f"Trade reconciliation completed for {len(closed_while_offline)} trades")
            
        except Exception as e:
            logger.error(f"Error during trade reconciliation: {str(e)}")
            logger.error(traceback.format_exc())

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
                    mt5.shutdown()  # Use the globally imported mt5 module
                    time.sleep(1)  # Give it time to clean up
                except Exception as ex:
                    logger.debug(f"Error during MT5 shutdown in recovery: {str(ex)}")
                    # Continue despite shutdown errors
                
                # Create a new MT5 handler on the second attempt or if first attempt fails
                if attempt > 1 or not connection_established:
                    logger.info("Creating a fresh MT5Handler instance for clean reconnection")
                    self.mt5_handler = MT5Handler()
                
                # Attempt to initialize
                if self.mt5_handler.initialize():
                    logger.info("MT5 connection recovered successfully")
                    
                    # Verify connection by doing a simple query
                    try:
                        # Try a simple account query to confirm the connection
                        account_info = self.mt5_handler.get_account_info()
                        if account_info:
                            logger.info(f"MT5 connection verified with account: {account_info.get('login', 'unknown')}")
                            return True
                        else:
                            logger.warning("MT5 initialized but could not verify connection with account query")
                            if attempt < max_attempts:
                                connection_established = True  # We made progress
                                continue
                    except Exception as e:
                        logger.warning(f"MT5 initialized but verification failed: {str(e)}")
                        if attempt < max_attempts:
                            connection_established = True  # We made progress
                            continue
                
                # Wait before next attempt with increasing backoff
                wait_time = 2 * attempt
                logger.info(f"Waiting {wait_time} seconds before next reconnection attempt")
                time.sleep(wait_time)
                
            except Exception as e:
                logger.error(f"Error during MT5 reconnection attempt {attempt}: {str(e)}")
                time.sleep(2 * attempt)  # Increasing backoff
        
        logger.error(f"Failed to recover MT5 connection after {max_attempts} attempts")
        return False

    async def handle_enable_trading_command(self, args):
        """
        Handle command to enable trading.
        Format: /enabletrading
        """
        await self.enable_trading()
        logger.info("Trading enabled via Telegram command")
        return "✅ Trading has been enabled"
        
    async def handle_disable_trading_command(self, args):
        """
        Handle command to disable trading.
        Format: /disabletrading
        """
        await self.disable_trading()
        logger.info("Trading disabled via Telegram command")
        return "✅ Trading has been disabled"
        
    async def handle_enable_position_additions_command(self, args):
        """
        Handle command to enable position additions.
        Format: /enablepositionadditions
        """
        self.trading_config["allow_position_additions"] = True
        logger.info("Position additions enabled via Telegram command")
        return "✅ Position additions have been enabled"
        
    async def handle_disable_position_additions_command(self, args):
        """
        Handle command to disable position additions.
        Format: /disablepositionadditions
        """
        self.trading_config["allow_position_additions"] = False
        logger.info("Position additions disabled via Telegram command")
        return "✅ Position additions have been disabled"

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